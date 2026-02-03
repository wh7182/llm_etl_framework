"""Mock LLM provider for testing without API calls."""

from typing import Type, TypeVar
from pydantic import BaseModel

from llm_etl.llm.client import LLMClient


T = TypeVar('T', bound=BaseModel)


class MockProvider(LLMClient):
    """
    Mock LLM client that returns predefined responses.

    Useful for testing pipeline logic without making actual API calls.
    Supports simulating validation failures to test retry logic.
    """

    def __init__(
        self,
        responses: dict[str, dict] | None = None,
        default_response: dict | None = None,
        fail_validation_times: int = 0
    ):
        """
        Initialize mock provider.

        Args:
            responses: Map of step_name -> response dict matching expected schema
            default_response: Fallback response if step_name not found
            fail_validation_times: Return invalid data this many times before
                                    returning valid data (tests retry logic)
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.fail_validation_times = fail_validation_times
        self._failure_count: dict[str, int] = {}

    def set_response(self, step_name: str, response: dict) -> None:
        """
        Add or update response for a specific step.

        Args:
            step_name: Name of the pipeline step
            response: Dict that should validate against step's schema
        """
        self.responses[step_name] = response

    def complete(self, messages: list[dict], response_model: Type[T]) -> T:
        """
        Return mock response validated against schema.

        Args:
            messages: List of message dicts (inspected to extract step name)
            response_model: Pydantic model to validate response against

        Returns:
            Validated instance of response_model

        Raises:
            ValueError: If no response configured for this step
            ValidationError: If configured response doesn't match schema
                            (or if simulating failure)
        """
        # Extract step name from messages
        step_name = self._extract_step_name(messages)

        # Initialize failure counter for this step
        if step_name not in self._failure_count:
            self._failure_count[step_name] = 0

        # Simulate validation failures if requested
        if self._failure_count[step_name] < self.fail_validation_times:
            self._failure_count[step_name] += 1
            # Return intentionally invalid data
            invalid_data = {"invalid_field": "this should fail validation"}
            return response_model.model_validate(invalid_data)

        # Look up response data
        response_data = None
        if step_name in self.responses:
            response_data = self.responses[step_name]
        elif self.default_response is not None:
            response_data = self.default_response
        else:
            raise ValueError(
                f"No mock response configured for step '{step_name}'. "
                f"Available steps: {list(self.responses.keys())}. "
                f"Use set_response() or provide default_response."
            )

        # Validate and return
        return response_model.model_validate(response_data)

    def _extract_step_name(self, messages: list[dict]) -> str:
        """
        Extract step name from message history.

        Looks for step name in:
        1. System message content (searches for common patterns)
        2. Last user message (fallback)

        Args:
            messages: List of message dicts

        Returns:
            Extracted step name or 'unknown_step'
        """
        # Strategy 1: Look in system message for step name
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Common patterns: "Step: classifier", "step_name: classifier"
                if "step:" in content.lower() or "step_name:" in content.lower():
                    for line in content.split("\n"):
                        line_lower = line.lower()
                        if "step:" in line_lower or "step_name:" in line_lower:
                            # Extract value after colon
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                return parts[1].strip().strip('"\'')

        # Strategy 2: Look for step in user messages (might contain step context)
        for msg in reversed(messages):  # Check most recent first
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "step:" in content.lower() or "step_name:" in content.lower():
                    for line in content.split("\n"):
                        line_lower = line.lower()
                        if "step:" in line_lower or "step_name:" in line_lower:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                return parts[1].strip().strip('"\'')

        # Fallback: return generic name
        return "unknown_step"
