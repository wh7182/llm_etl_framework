"""
Pipeline orchestration for the LLM ETL framework.

Coordinates source ingestion, step execution, error handling, and sink persistence.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from ..core.exceptions import StepExecutionError
from ..core.state import GlobalState
from ..llm.client import LLMClient, LLMClientWithRetry
from ..sinks.base import AbstractSink
from ..sources.base import AbstractSource
from ..steps.base import AbstractBaseStep


# Type alias for error handling modes
ErrorHandling = Literal["fail", "skip", "dead_letter"]


@dataclass
class PipelineResult:
    """
    Result of a pipeline run with execution metrics.

    Attributes:
        success_count: Number of rows processed successfully
        failure_count: Number of rows that failed (skipped or written to dead letter)
        total_count: Total number of rows attempted
        duration_seconds: Total execution time in seconds
        dead_letter_path: Path to dead letter file if any rows failed, None otherwise

    Example:
        >>> result = pipeline.run()
        >>> print(f"Success: {result.success_count}/{result.total_count}")
        >>> print(f"Duration: {result.duration_seconds:.2f}s")
        >>> if result.failure_count > 0:
        ...     print(f"Failed rows in: {result.dead_letter_path}")
    """

    success_count: int
    failure_count: int
    total_count: int
    duration_seconds: float
    dead_letter_path: Optional[str] = None


class Pipeline:
    """
    Orchestrates the complete ETL pipeline flow.

    Manages the flow from source → steps → sink with error handling,
    progress logging, and dead letter tracking.

    Example:
        >>> pipeline = Pipeline(
        ...     name="patient_enrichment",
        ...     source=CSVSource("data/patients.csv", "patient_id"),
        ...     steps=[
        ...         ClassifierStep(...),
        ...         SummarizerStep(...),
        ...     ],
        ...     sink=CSVSink("output/enriched.csv", {...}),
        ...     llm_provider=AzureOpenAIProvider(),
        ...     on_row_error="dead_letter",
        ...     dead_letter_path="output/failed_rows.jsonl",
        ... )
        >>>
        >>> # Validate configuration without processing
        >>> result = pipeline.run(dry_run=True)
        >>>
        >>> # Run the full pipeline
        >>> result = pipeline.run()
        >>> print(f"Processed {result.success_count} rows in {result.duration_seconds:.1f}s")
    """

    def __init__(
        self,
        name: str,
        source: AbstractSource,
        steps: list[AbstractBaseStep],
        sink: AbstractSink,
        llm_provider: LLMClient,
        on_row_error: ErrorHandling = "dead_letter",
        dead_letter_path: str = "output/dead_letter.jsonl",
        max_retries: int = 3,
    ):
        """
        Initialize the pipeline with all components.

        Args:
            name: Unique identifier for this pipeline (used in logging)
            source: Data source to read rows from
            steps: List of processing steps to execute in order
            sink: Destination to write processed rows to
            llm_provider: LLM client implementation (will be wrapped with retry logic)
            on_row_error: How to handle row processing errors:
                         - "fail": Stop on first error and re-raise
                         - "skip": Log error and continue to next row
                         - "dead_letter": Write failed row to dead letter file and continue
            dead_letter_path: Path for dead letter file (if on_row_error="dead_letter")
            max_retries: Maximum LLM validation retry attempts (default: 3)

        Raises:
            ValueError: If on_row_error is not a valid option

        Example:
            >>> pipeline = Pipeline(
            ...     name="visit_classification",
            ...     source=SQLServerSource("sql/get_visits.sql", "visit_id"),
            ...     steps=[ClassifierStep(...)],
            ...     sink=SQLServerSink("dbo.enriched_visits", ...),
            ...     llm_provider=AzureOpenAIProvider(),
            ...     on_row_error="dead_letter",
            ... )
        """
        self.name = name
        self.source = source
        self.steps = steps
        self.sink = sink
        self.on_row_error = on_row_error
        self.dead_letter_path = Path(dead_letter_path)
        self.max_retries = max_retries

        # Validate error handling mode
        valid_modes: tuple[ErrorHandling, ...] = ("fail", "skip", "dead_letter")
        if on_row_error not in valid_modes:
            raise ValueError(
                f"Invalid on_row_error: {on_row_error!r}. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

        # Wrap LLM provider with retry logic
        self.llm_client = LLMClientWithRetry(llm_provider, max_retries=max_retries)

        # Setup logging
        self.logger = logging.getLogger(f"llm_etl.pipeline.{name}")

    def run(self, dry_run: bool = False) -> PipelineResult:
        """
        Execute the pipeline on all rows from the source.

        Args:
            dry_run: If True, validate configuration without processing data

        Returns:
            PipelineResult with execution metrics

        Raises:
            Exception: If on_row_error="fail" and a row fails processing

        Example:
            >>> # Validate configuration
            >>> result = pipeline.run(dry_run=True)
            >>> assert result.total_count == 0
            >>>
            >>> # Run for real
            >>> result = pipeline.run()
            >>> print(f"Success rate: {result.success_count / result.total_count:.1%}")
        """
        if dry_run:
            return self._run_dry_run()

        start_time = time.time()
        success_count = 0
        failure_count = 0
        total_count = 0

        # Get total count for progress tracking (may be None)
        total_rows = self.source.count()
        if total_rows is not None:
            self.logger.info(f"Pipeline '{self.name}' starting: {total_rows} rows to process")
        else:
            self.logger.info(f"Pipeline '{self.name}' starting (row count unknown)")

        try:
            for state in self.source:
                total_count += 1

                try:
                    # Process the row through all steps and write to sink
                    self._process_row(state)
                    success_count += 1

                except KeyboardInterrupt:
                    self.logger.warning("Keyboard interrupt received, stopping pipeline...")
                    raise

                except Exception as e:
                    # Handle error according to configured strategy
                    self._handle_error(state, e)
                    failure_count += 1

                # Progress logging every 100 rows or 10% milestones
                if total_rows is not None:
                    milestone_interval = max(1, total_rows // 10)  # 10% milestones
                    if total_count % min(100, milestone_interval) == 0:
                        progress_pct = (total_count / total_rows) * 100
                        self.logger.info(
                            f"Progress: {total_count}/{total_rows} ({progress_pct:.1f}%) - "
                            f"Success: {success_count}, Failed: {failure_count}"
                        )
                elif total_count % 100 == 0:
                    self.logger.info(
                        f"Progress: {total_count} rows - "
                        f"Success: {success_count}, Failed: {failure_count}"
                    )

        except KeyboardInterrupt:
            self.logger.warning(
                f"Pipeline interrupted after {total_count} rows. "
                f"Success: {success_count}, Failed: {failure_count}"
            )
            # Don't re-raise, return partial results

        duration_seconds = time.time() - start_time

        # Log final summary
        self.logger.info(
            f"Pipeline '{self.name}' completed: {success_count} succeeded, "
            f"{failure_count} failed out of {total_count} total "
            f"in {duration_seconds:.2f}s"
        )

        # Return results
        return PipelineResult(
            success_count=success_count,
            failure_count=failure_count,
            total_count=total_count,
            duration_seconds=duration_seconds,
            dead_letter_path=str(self.dead_letter_path) if failure_count > 0 else None,
        )

    def _run_dry_run(self) -> PipelineResult:
        """
        Validate pipeline configuration without processing data.

        Returns:
            PipelineResult with zero counts

        Raises:
            Exception: If configuration validation fails
        """
        self.logger.info(f"Dry run: validating pipeline '{self.name}' configuration...")

        # Validate source is accessible
        try:
            row_count = self.source.count()
            self.logger.info(f"✓ Source accessible: {row_count} rows available")
        except Exception as e:
            self.logger.error(f"✗ Source validation failed: {e}")
            raise

        # Validate steps are configured
        if not self.steps:
            self.logger.warning("⚠ No steps configured - pipeline will only copy source to sink")
        else:
            self.logger.info(f"✓ {len(self.steps)} steps configured:")
            for step in self.steps:
                self.logger.info(f"  - {step.name} → {step.output_key}")

        # Validate sink configuration
        # (Most sinks create resources lazily, so we just log the config)
        self.logger.info(f"✓ Sink configured: {type(self.sink).__name__}")

        # Validate error handling
        self.logger.info(f"✓ Error handling: {self.on_row_error}")
        if self.on_row_error == "dead_letter":
            self.logger.info(f"  Dead letter path: {self.dead_letter_path}")

        # Validate LLM client
        self.logger.info(
            f"✓ LLM client configured with {self.max_retries} max retries"
        )

        self.logger.info("Dry run validation completed successfully")

        return PipelineResult(
            success_count=0,
            failure_count=0,
            total_count=0,
            duration_seconds=0.0,
            dead_letter_path=None,
        )

    def _process_row(self, state: GlobalState) -> GlobalState:
        """
        Process a single row through all steps and write to sink.

        Args:
            state: GlobalState to process

        Returns:
            The processed GlobalState with completed_at timestamp

        Raises:
            StepExecutionError: If any step fails
            SinkError: If writing to sink fails
        """
        try:
            # Run through all steps
            for step in self.steps:
                try:
                    state = step.run(state, self.llm_client)
                except Exception as e:
                    # Wrap in StepExecutionError with context
                    raise StepExecutionError(
                        step_name=step.name,
                        pk=state.pk,
                        original_error=e,
                    )

            # Mark completion timestamp
            state.completed_at = datetime.now()

            # Write to sink
            self.sink.write(state)

            return state

        except StepExecutionError:
            # Re-raise StepExecutionError as-is (already has context)
            raise
        except Exception as e:
            # Wrap other errors (like SinkError) with step context
            # Use "sink" as the step name for sink errors
            raise StepExecutionError(
                step_name="sink",
                pk=state.pk,
                original_error=e,
            )

    def _handle_error(self, state: GlobalState, error: Exception) -> None:
        """
        Handle a row processing error according to the configured strategy.

        Args:
            state: The GlobalState that failed
            error: The exception that occurred

        Raises:
            Exception: If on_row_error="fail", re-raises the original error
        """
        # Extract step name from StepExecutionError if available
        if isinstance(error, StepExecutionError):
            step_name = error.step_name
            error_msg = str(error.original_error)
        else:
            step_name = "unknown"
            error_msg = str(error)

        if self.on_row_error == "fail":
            # Re-raise immediately
            self.logger.error(
                f"Pipeline halted due to error in step '{step_name}' "
                f"for pk={state.pk}: {error_msg}"
            )
            raise

        elif self.on_row_error == "skip":
            # Log and continue
            self.logger.warning(
                f"Skipping pk={state.pk} due to error in step '{step_name}': {error_msg}"
            )

        elif self.on_row_error == "dead_letter":
            # Write to dead letter and continue
            self.logger.warning(
                f"Writing pk={state.pk} to dead letter due to error in step '{step_name}': {error_msg}"
            )
            self._write_dead_letter(state, step_name, error)

    def _write_dead_letter(
        self, state: GlobalState, step_name: str, error: Exception
    ) -> None:
        """
        Write a failed row to the dead letter file.

        Creates a JSONL record with full context for debugging.
        Creates parent directories if needed.

        Args:
            state: The GlobalState that failed
            step_name: Name of the step that failed
            error: The exception that occurred

        Note:
            This method is synchronous and writes immediately to ensure
            failed rows are never lost, even if the pipeline crashes.
        """
        # Extract retry count if this was an LLMValidationError
        from ..core.exceptions import LLMValidationError, StepExecutionError

        retry_attempts = 0
        # Check if error is LLMValidationError (direct or wrapped in StepExecutionError)
        actual_error = error
        if isinstance(error, StepExecutionError):
            actual_error = error.original_error

        if isinstance(actual_error, LLMValidationError):
            retry_attempts = actual_error.retry_count

        # Create dead letter record
        record = {
            "pk": state.pk,
            "step_name": step_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "raw_data": dict(state.raw),  # Convert MappingProxyType to dict
            "processed_state": state.processed,
            "steps_completed": state.log,
            "timestamp": datetime.now().isoformat(),
            "retry_attempts": retry_attempts,
        }

        # Ensure parent directory exists
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)

        # Write synchronously (atomic append on most filesystems)
        with open(self.dead_letter_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
