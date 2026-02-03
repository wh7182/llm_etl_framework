"""
Custom exceptions for the LLM ETL framework.

Provides a hierarchy of exceptions with rich context for debugging pipeline failures.
"""

from typing import Optional


class LLMETLError(Exception):
    """
    Base exception for all LLM ETL framework errors.

    All custom exceptions in the framework inherit from this class,
    allowing users to catch framework-specific errors.

    Example:
        >>> try:
        ...     # pipeline operations
        ...     pass
        ... except LLMETLError as e:
        ...     print(f"Framework error: {e}")
    """

    pass


class StepExecutionError(LLMETLError):
    """
    Raised when a step fails during execution.

    Captures the step name, primary key, and the original error that caused the failure.
    This allows the pipeline to log detailed error information and potentially retry.

    Attributes:
        step_name: Name of the step that failed
        pk: Primary key of the row being processed
        original_error: The underlying exception that caused the failure

    Example:
        >>> try:
        ...     # step logic
        ...     raise ValueError("Invalid input")
        ... except ValueError as e:
        ...     raise StepExecutionError(
        ...         step_name="visit_classifier",
        ...         pk="enc_12345",
        ...         original_error=e
        ...     )
    """

    def __init__(self, step_name: str, pk: str, original_error: Exception):
        """
        Initialize a StepExecutionError.

        Args:
            step_name: Name of the step that failed
            pk: Primary key of the row being processed
            original_error: The underlying exception that caused the failure
        """
        self.step_name = step_name
        self.pk = pk
        self.original_error = original_error

        message = (
            f"Step '{step_name}' failed for pk='{pk}': "
            f"{type(original_error).__name__}: {original_error}"
        )
        super().__init__(message)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"StepExecutionError(step_name={self.step_name!r}, "
            f"pk={self.pk!r}, "
            f"original_error={self.original_error!r})"
        )


class LLMValidationError(LLMETLError):
    """
    Raised when LLM output fails Pydantic validation.

    Captures validation errors and retry attempts to help diagnose why
    the LLM is producing invalid outputs even after error feedback.

    Attributes:
        step_name: Name of the step that failed validation
        pk: Primary key of the row being processed
        validation_errors: List of validation error messages from Pydantic
        retry_count: Number of retry attempts made before giving up

    Example:
        >>> raise LLMValidationError(
        ...     step_name="visit_classifier",
        ...     pk="enc_12345",
        ...     validation_errors=["confidence must be between 0 and 1, got 1.5"],
        ...     retry_count=3
        ... )
    """

    def __init__(
        self,
        step_name: str,
        pk: str,
        validation_errors: list[str],
        retry_count: int,
    ):
        """
        Initialize an LLMValidationError.

        Args:
            step_name: Name of the step that failed validation
            pk: Primary key of the row being processed
            validation_errors: List of validation error messages
            retry_count: Number of retry attempts made
        """
        self.step_name = step_name
        self.pk = pk
        self.validation_errors = validation_errors
        self.retry_count = retry_count

        errors_str = "; ".join(validation_errors)
        message = (
            f"Step '{step_name}' validation failed for pk='{pk}' "
            f"after {retry_count} retries. Errors: {errors_str}"
        )
        super().__init__(message)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"LLMValidationError(step_name={self.step_name!r}, "
            f"pk={self.pk!r}, "
            f"validation_errors={self.validation_errors!r}, "
            f"retry_count={self.retry_count})"
        )


class SourceError(LLMETLError):
    """
    Raised when data ingestion from a source fails.

    Used for failures in SQLServerSource, CSVSource, or other data sources.

    Example:
        >>> raise SourceError("Failed to connect to SQL Server: timeout")
    """

    pass


class SinkError(LLMETLError):
    """
    Raised when writing processed data to a sink fails.

    Captures the primary key and original error to help identify
    which rows failed to persist.

    Attributes:
        pk: Primary key of the row that failed to write
        original_error: The underlying exception that caused the failure

    Example:
        >>> try:
        ...     # sink write operation
        ...     raise ConnectionError("Database unavailable")
        ... except ConnectionError as e:
        ...     raise SinkError(pk="enc_12345", original_error=e)
    """

    def __init__(self, pk: str, original_error: Exception):
        """
        Initialize a SinkError.

        Args:
            pk: Primary key of the row that failed to write
            original_error: The underlying exception that caused the failure
        """
        self.pk = pk
        self.original_error = original_error

        message = (
            f"Failed to write pk='{pk}' to sink: "
            f"{type(original_error).__name__}: {original_error}"
        )
        super().__init__(message)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"SinkError(pk={self.pk!r}, "
            f"original_error={self.original_error!r})"
        )
