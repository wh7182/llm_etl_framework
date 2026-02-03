"""
CSV file sink for the LLM ETL framework.

Writes processed GlobalState objects to a CSV file using a configurable column mapping.
Useful for testing pipelines without SQL Server dependencies.
"""

import csv
import threading
from pathlib import Path
from typing import Any, Callable

from ..core.exceptions import SinkError
from ..core.state import GlobalState
from .base import AbstractSink


class CSVSink(AbstractSink):
    """
    Write data to a CSV file.

    Uses a column_map to extract values from GlobalState and write them to CSV.
    Automatically creates parent directories and handles headers.

    Thread-safety note:
        This implementation uses a lock to ensure thread-safe writes when
        multiple threads might write to the same file. For single-threaded
        pipelines, the lock adds minimal overhead.

    Example:
        >>> sink = CSVSink(
        ...     file_path="output/enriched_encounters.csv",
        ...     column_map={
        ...         "encounter_id": lambda s: s.pk,
        ...         "visit_category": lambda s: s.processed["visit_type"]["category"],
        ...         "confidence": lambda s: s.processed["visit_type"]["confidence"],
        ...         "summary": lambda s: s.processed["summary"]["text"],
        ...     }
        ... )
        >>>
        >>> # Write a single state
        >>> sink.write(state)
        >>>
        >>> # Or write a batch (more efficient)
        >>> sink.write_batch([state1, state2, state3])
    """

    def __init__(
        self,
        file_path: str,
        column_map: dict[str, Callable[[GlobalState], Any]],
    ):
        """
        Initialize the CSV sink.

        Args:
            file_path: Path to the output CSV file (will be created if needed)
            column_map: Dictionary mapping column names to functions that extract
                       values from GlobalState. Column names determine CSV header order.

        Raises:
            SinkError: If parent directory creation fails

        Example:
            >>> sink = CSVSink(
            ...     file_path="data/output/results.csv",
            ...     column_map={
            ...         "id": lambda s: s.pk,
            ...         "category": lambda s: s.processed["class"]["category"],
            ...         "processed_at": lambda s: s.completed_at.isoformat(),
            ...     }
            ... )
        """
        self.file_path = Path(file_path)
        self.column_map = column_map
        self._lock = threading.Lock()  # Thread-safe file writes

        # Create parent directories if they don't exist
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Use base exception since we don't have a pk yet
            from ..core.exceptions import LLMETLError
            raise LLMETLError(
                f"Failed to create parent directory for {self.file_path}: {e}"
            )

    def write(self, state: GlobalState) -> None:
        """
        Write a single GlobalState to the CSV file.

        If the file doesn't exist, creates it with a header row.
        Otherwise, appends a data row.

        Args:
            state: GlobalState object to write

        Raises:
            SinkError: If extracting values from column_map fails or if
                      writing to the file fails
        """
        with self._lock:  # Thread-safe access
            try:
                # Extract values using column_map
                row = self._extract_row(state)

                # Determine if we need to write a header
                file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0

                # Open in append mode with UTF-8 encoding
                with self.file_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(self.column_map.keys()),
                        quoting=csv.QUOTE_MINIMAL,
                        extrasaction="ignore",
                    )

                    # Write header if file is new or empty
                    if not file_exists:
                        writer.writeheader()

                    # Write the data row
                    writer.writerow(row)

            except SinkError:
                raise  # Re-raise SinkErrors from _extract_row
            except OSError as e:
                raise SinkError(pk=state.pk, original_error=e)

    def write_batch(self, states: list[GlobalState]) -> None:
        """
        Write multiple GlobalState objects to the CSV file efficiently.

        Opens the file once and writes all rows in a single operation.
        If the file doesn't exist, creates it with a header row.

        This is more efficient than calling write() multiple times because
        it minimizes file I/O operations.

        Args:
            states: List of GlobalState objects to write

        Raises:
            SinkError: If extracting values from column_map fails for any state,
                      or if writing to the file fails. Includes the pk of the
                      failed state in the error message.
        """
        if not states:
            return  # Nothing to write

        with self._lock:  # Thread-safe access
            try:
                # Extract all rows first (fail fast if column_map has issues)
                rows = []
                for state in states:
                    # _extract_row already raises SinkError with proper pk context
                    rows.append(self._extract_row(state))

                # Determine if we need to write a header
                file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0

                # Open in append mode with UTF-8 encoding
                with self.file_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(self.column_map.keys()),
                        quoting=csv.QUOTE_MINIMAL,
                        extrasaction="ignore",
                    )

                    # Write header if file is new or empty
                    if not file_exists:
                        writer.writeheader()

                    # Write all data rows
                    writer.writerows(rows)

            except SinkError:
                raise  # Re-raise our own errors
            except OSError as e:
                # For batch operations, we don't have a single pk, so use the first one
                first_pk = states[0].pk if states else "unknown"
                raise SinkError(pk=first_pk, original_error=e)

    def _extract_row(self, state: GlobalState) -> dict[str, Any]:
        """
        Extract a row dictionary from a GlobalState using column_map.

        Args:
            state: GlobalState to extract values from

        Returns:
            Dictionary mapping column names to extracted values

        Raises:
            SinkError: If any extractor function raises an exception
        """
        row = {}
        for col_name, extractor_fn in self.column_map.items():
            try:
                value = extractor_fn(state)

                # Convert None to empty string for CSV compatibility
                if value is None:
                    value = ""

                row[col_name] = value

            except Exception as e:
                # Wrap the extraction error with proper context
                error_msg = f"Column '{col_name}' extractor failed: {e}"
                raise SinkError(pk=state.pk, original_error=Exception(error_msg))

        return row
