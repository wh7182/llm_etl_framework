"""LLM client abstraction with retry logic and validation."""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TypeVar, Generic, Type
from pydantic import BaseModel, ValidationError

from llm_etl.core.exceptions import LLMValidationError


T = TypeVar('T', bound=BaseModel)


class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    def complete(self, messages: list[dict], response_model: Type[T]) -> T:
        """
        Send messages to LLM and return validated response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_model: Pydantic model class to validate response against

        Returns:
            Validated instance of response_model

        Raises:
            ValidationError: If response doesn't match schema
            Exception: For LLM API errors
        """
        pass


class LLMClientWithRetry:
    """
    Wrapper that adds retry logic with error feedback to any LLMClient.

    Validates LLM responses against Pydantic schemas. On validation failure,
    appends the error to the conversation and retries, giving the LLM a
    chance to fix its output.
    """

    def __init__(
        self,
        client: LLMClient,
        max_retries: int = 3,
        log_dir: str = "logs"
    ):
        """
        Initialize retry wrapper.

        Args:
            client: Underlying LLMClient implementation
            max_retries: Maximum number of retry attempts after initial failure
            log_dir: Directory for logging LLM payloads
        """
        self.client = client
        self.max_retries = max_retries
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def complete_with_validation(
        self,
        messages: list[dict],
        response_model: Type[T],
        step_name: str,
        pk: str
    ) -> T:
        """
        Complete LLM request with validation and retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_model: Pydantic model class to validate response against
            step_name: Name of the pipeline step (for logging)
            pk: Primary key of the record being processed (for logging)

        Returns:
            Validated instance of response_model

        Raises:
            LLMValidationError: If validation fails after all retries
        """
        # Work with a copy to avoid mutating original
        current_messages = messages.copy()
        accumulated_errors = []

        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            error = None
            response = None

            try:
                # Attempt completion
                response = self.client.complete(current_messages, response_model)
                latency_ms = int((time.time() - start_time) * 1000)

                # Log successful attempt
                self._log_payload(
                    step_name=step_name,
                    pk=pk,
                    messages=current_messages,
                    response=response.model_dump() if response else None,
                    retry_count=attempt,
                    latency_ms=latency_ms,
                    error=None
                )

                # Success!
                return response

            except ValidationError as e:
                latency_ms = int((time.time() - start_time) * 1000)
                error_msg = str(e)
                accumulated_errors.append(error_msg)

                # Log failed attempt
                self._log_payload(
                    step_name=step_name,
                    pk=pk,
                    messages=current_messages,
                    response=None,
                    retry_count=attempt,
                    latency_ms=latency_ms,
                    error=error_msg
                )

                # If we have retries left, append error and try again
                if attempt < self.max_retries:
                    retry_message = {
                        "role": "user",
                        "content": f"Your previous response failed validation: {error_msg}. Please fix and try again."
                    }
                    current_messages.append(retry_message)
                else:
                    # Out of retries
                    raise LLMValidationError(
                        step_name=step_name,
                        pk=pk,
                        validation_errors=accumulated_errors,
                        retry_count=self.max_retries + 1
                    )

            except Exception as e:
                # Non-validation errors (API errors, etc.)
                latency_ms = int((time.time() - start_time) * 1000)
                error_msg = f"{type(e).__name__}: {str(e)}"

                self._log_payload(
                    step_name=step_name,
                    pk=pk,
                    messages=current_messages,
                    response=None,
                    retry_count=attempt,
                    latency_ms=latency_ms,
                    error=error_msg
                )

                # Re-raise non-validation errors immediately
                raise

    def _log_payload(
        self,
        step_name: str,
        pk: str,
        messages: list[dict],
        response: dict | None,
        retry_count: int,
        latency_ms: int,
        error: str | None = None
    ) -> None:
        """
        Log LLM request/response to JSONL file.

        Args:
            step_name: Name of the pipeline step
            pk: Primary key of the record
            messages: List of message dicts sent to LLM
            response: Response dict from LLM (or None if failed)
            retry_count: Which attempt this was (0-indexed)
            latency_ms: Time taken in milliseconds
            error: Error message if request failed
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step_name,
            "pk": pk,
            "request": {"messages": messages},
            "response": response,
            "retry_count": retry_count,
            "latency_ms": latency_ms,
            "error": error
        }

        log_file = self.log_dir / "llm_payloads.jsonl"

        # Use append mode for concurrent writes
        # Each write is a single line, which is atomic on most filesystems
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
