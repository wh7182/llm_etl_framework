"""
SummarizerStep: Text summarization with automatic validation.

Calls the LLM to summarize text with optional focus context.
"""

from pydantic import Field, field_validator

from ..llm.base_schemas import LLMOutputBase
from .base import AbstractBaseStep


# ============================================================================
# Output Schema
# ============================================================================


class SummaryOutput(LLMOutputBase):
    """Validated output from SummarizerStep."""

    text: str = Field(..., description="The summarized text")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip whitespace and ensure non-empty."""
        v = v.strip()
        if not v:
            raise ValueError("Summary text cannot be empty")
        return v


# ============================================================================
# SummarizerStep
# ============================================================================


class SummarizerStep(AbstractBaseStep):
    """
    Text summarization step.

    Calls the LLM to summarize text with optional focus context.

    Example:
        step = SummarizerStep(
            name="note_summarizer",
            input_map={
                "text": lambda s: s.raw["clinical_notes"],
                "focus": lambda s: s.processed["visit_type"]["category"],
            },
            output_key="summary",
        )
    """

    def __init__(
        self,
        name: str,
        input_map: dict,
        output_key: str,
    ):
        """
        Initialize the summarizer step.

        Args:
            name: Step identifier for logging
            input_map: Dict of lambdas to extract inputs from GlobalState
            output_key: Key to store result in state.processed
        """
        super().__init__(name=name, input_map=input_map, output_key=output_key)


    def _build_prompt(self, mapped_input: dict) -> list[dict]:
        """
        Construct the system and user messages for the LLM.

        Args:
            mapped_input: Dictionary of inputs extracted via input_map

        Returns:
            List of message dicts for the LLM
        """
        system_message = """You are a precise text summarization system. Summarize the provided text concisely.

Respond with a JSON object:
- text: your summarized text
- reasoning: explain your summarization approach"""

        # Build user message
        user_parts = [f"Text to summarize:\n{mapped_input['text']}"]

        if "focus" in mapped_input:
            user_parts.append(f"\nFocus area: {mapped_input['focus']}")

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]


    def execute(self, mapped_input: dict, llm_client, pk: str) -> SummaryOutput:
        """
        Execute the summarization step.

        Args:
            mapped_input: Dictionary of inputs extracted via input_map
            llm_client: LLMClientWithRetry instance
            pk: Primary key for logging

        Returns:
            Validated SummaryOutput

        Raises:
            ValidationError: If Pydantic validation fails after max_retries
        """
        # Handle edge case: empty input
        if not mapped_input.get("text", "").strip():
            raise ValueError("Cannot summarize empty text")

        # Build prompt
        messages = self._build_prompt(mapped_input)

        # Call LLM with validation
        result = llm_client.complete_with_validation(
            messages=messages,
            response_model=SummaryOutput,
            step_name=self.name,
            pk=pk,
        )

        return result
