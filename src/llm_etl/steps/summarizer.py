"""
SummarizerStep: Rule-based text reduction with automatic validation.

Parses human-readable rules (e.g., "Maximum 50 words", "Must include: diagnosis")
and enforces them programmatically after LLM generation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import Field, field_validator

from ..llm.base_schemas import LLMOutputBase
from .base import AbstractBaseStep


# ============================================================================
# Output Schema
# ============================================================================


class SummaryOutput(LLMOutputBase):
    """Validated output from SummarizerStep."""

    text: str = Field(..., description="The summarized text")
    word_count: int = Field(ge=0, description="Actual word count of the summary")
    rules_applied: list[str] = Field(
        default_factory=list, description="Which rules were followed"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip whitespace and ensure non-empty."""
        v = v.strip()
        if not v:
            raise ValueError("Summary text cannot be empty")
        return v


# ============================================================================
# Rule Parsing
# ============================================================================


@dataclass
class ParsedRules:
    """Structured representation of summarization constraints."""

    max_words: Optional[int] = None
    max_chars: Optional[int] = None
    required_terms: list[str] = field(default_factory=list)
    original_rules: list[str] = field(default_factory=list)


# ============================================================================
# SummarizerStep
# ============================================================================


class SummarizerStep(AbstractBaseStep):
    """
    Rule-based text summarization step.

    Parses natural language rules into programmatic constraints and validates
    LLM output against them. Validation errors trigger automatic retry.

    Example:
        rules = [
            "Maximum 50 words",
            "Focus on chief complaint and diagnosis",
            "Must include: medication, dosage",
        ]

        step = SummarizerStep(
            name="note_summarizer",
            rules=rules,
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
        rules: list[str],
        input_map: dict,
        output_key: str,
    ):
        """
        Initialize the summarizer step.

        Args:
            name: Step identifier for logging
            rules: List of human-readable rules (e.g., "Maximum 50 words")
            input_map: Dict of lambdas to extract inputs from GlobalState
            output_key: Key to store result in state.processed
        """
        super().__init__(name=name, input_map=input_map, output_key=output_key)
        self.rules = rules
        self.parsed_rules = self._parse_rules(rules)

    def _parse_rules(self, rules: list[str]) -> ParsedRules:
        """
        Extract programmatic constraints from natural language rules.

        Supported patterns (case-insensitive):
            - "Maximum N words" / "Max N words" -> max_words = N
            - "Maximum N characters" / "Max N chars" -> max_chars = N
            - "Must include: X, Y, Z" / "Preserve: X, Y" -> required_terms = [X, Y, Z]

        Args:
            rules: List of natural language rules

        Returns:
            ParsedRules object with extracted constraints
        """
        parsed = ParsedRules(original_rules=rules)

        for rule in rules:
            rule_lower = rule.lower().strip()

            # Match "Maximum N words" or "Max N words"
            word_match = re.search(r"(?:maximum|max)\s+(\d+)\s+words?", rule_lower)
            if word_match:
                parsed.max_words = int(word_match.group(1))
                continue

            # Match "Maximum N characters" or "Max N chars"
            char_match = re.search(
                r"(?:maximum|max)\s+(\d+)\s+(?:characters?|chars?)", rule_lower
            )
            if char_match:
                parsed.max_chars = int(char_match.group(1))
                continue

            # Match "Must include: X, Y, Z" or "Preserve: X, Y"
            include_match = re.search(
                r"(?:must include|preserve|include):\s*(.+)", rule_lower
            )
            if include_match:
                terms_str = include_match.group(1)
                # Split on commas and clean up whitespace
                terms = [t.strip() for t in terms_str.split(",") if t.strip()]
                parsed.required_terms.extend(terms)

        return parsed

    def _build_prompt(self, mapped_input: dict) -> list[dict]:
        """
        Construct the system and user messages for the LLM.

        Args:
            mapped_input: Dictionary of inputs extracted via input_map

        Returns:
            List of message dicts for the LLM
        """
        # Build numbered rules list
        rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(self.rules))

        system_message = f"""You are a precise text summarization system. Summarize the provided text while strictly following ALL rules below.

RULES (you MUST follow every rule):
{rules_text}

Respond with a JSON object:
- text: your summarized text
- word_count: count the words in your summary
- rules_applied: list which rules you followed (by number or description)
- reasoning: explain your summarization approach

IMPORTANT: Count your words carefully. If a rule says "Maximum 50 words", your summary must have 50 or fewer words."""

        # Build user message
        user_parts = [f"Text to summarize:\n{mapped_input['text']}"]

        if "focus" in mapped_input:
            user_parts.append(f"\nFocus area: {mapped_input['focus']}")

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _validate_rules(self, result: SummaryOutput) -> SummaryOutput:
        """
        Programmatically validate the LLM's summary against parsed rules.

        Updates word_count with actual count (don't trust LLM's count).
        Raises ValueError on constraint violations, which triggers retry.

        Args:
            result: The LLM's output

        Returns:
            Validated result with corrected word_count

        Raises:
            ValueError: If any rule is violated (triggers retry)
        """
        # Count actual words using split (whitespace tokenization)
        actual_count = len(result.text.split())
        result.word_count = actual_count

        # Validate max words
        if self.parsed_rules.max_words is not None:
            if actual_count > self.parsed_rules.max_words:
                raise ValueError(
                    f"Summary has {actual_count} words but maximum is {self.parsed_rules.max_words}. "
                    f"Please shorten your summary to meet the word limit."
                )

        # Validate max characters
        if self.parsed_rules.max_chars is not None:
            actual_chars = len(result.text)
            if actual_chars > self.parsed_rules.max_chars:
                raise ValueError(
                    f"Summary has {actual_chars} characters but maximum is {self.parsed_rules.max_chars}. "
                    f"Please shorten your summary to meet the character limit."
                )

        # Validate required terms
        if self.parsed_rules.required_terms:
            text_lower = result.text.lower()
            missing = [
                term
                for term in self.parsed_rules.required_terms
                if term.lower() not in text_lower
            ]
            if missing:
                raise ValueError(
                    f"Summary must include these terms: {', '.join(missing)}. "
                    f"Please revise your summary to include all required terms."
                )

        return result

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
            ValueError: If rule validation fails after max_retries
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

        # Additional rule validation (may raise ValueError -> retry)
        result = self._validate_rules(result)

        return result
