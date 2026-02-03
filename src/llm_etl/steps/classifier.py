"""
ClassifierStep for taxonomy-constrained text classification.

Classifies text into predefined categories with confidence scores and reasoning.
Enforces strict taxonomy validation to ensure only valid categories are returned.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from llm_etl.llm.base_schemas import LLMOutputBase
from llm_etl.llm.client import LLMClientWithRetry
from llm_etl.steps.base import AbstractBaseStep


class TaxonomyCategory(BaseModel):
    """
    A single category in a classification taxonomy.

    Attributes:
        name: Unique identifier for the category (e.g., "Emergency")
        definition: Clear description of what this category represents
    """

    name: str = Field(..., description="Category identifier")
    definition: str = Field(..., description="What this category means")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Emergency",
                "definition": "Urgent, unplanned visit for acute symptoms or trauma",
            }
        }
    }


class ClassificationOutput(LLMOutputBase):
    """
    Output schema for classification results.

    Extends LLMOutputBase to include category, confidence, and reasoning.
    Category must exactly match one of the taxonomy names.
    """

    category: str = Field(..., description="The assigned category (must match taxonomy)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence score")
    reasoning: str  # Inherited from LLMOutputBase

    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "Emergency",
                "confidence": 0.92,
                "reasoning": "Patient presents with acute chest pain requiring immediate evaluation",
            }
        }
    }


class ClassifierStep(AbstractBaseStep):
    """
    Step that classifies text into predefined taxonomy categories.

    Uses LLM to classify text with strict validation that the returned category
    exactly matches one of the valid taxonomy names. Includes confidence scoring
    and reasoning for all classifications.

    Example:
        >>> taxonomy = [
        ...     {"name": "Emergency", "definition": "Urgent care needed"},
        ...     {"name": "Routine", "definition": "Scheduled visit"}
        ... ]
        >>> step = ClassifierStep(
        ...     name="visit_classifier",
        ...     taxonomy=taxonomy,
        ...     input_map={"text": lambda s: s.raw["note"]},
        ...     output_key="visit_type"
        ... )
    """

    def __init__(
        self,
        name: str,
        taxonomy: list[TaxonomyCategory | dict],
        input_map: dict,
        output_key: str,
    ):
        """
        Initialize the classifier step.

        Args:
            name: Unique identifier for this step
            taxonomy: List of TaxonomyCategory objects or dicts defining valid categories
            input_map: Mapping of parameter names to lambdas extracting from GlobalState
            output_key: Key under which to store classification results

        Raises:
            ValueError: If taxonomy is empty or contains duplicate names
        """
        super().__init__(name, input_map, output_key)

        # Normalize taxonomy: convert dicts to TaxonomyCategory objects
        self.taxonomy: list[TaxonomyCategory] = []
        for item in taxonomy:
            if isinstance(item, dict):
                self.taxonomy.append(TaxonomyCategory(**item))
            else:
                self.taxonomy.append(item)

        if not self.taxonomy:
            raise ValueError("Taxonomy cannot be empty")

        # Build set of valid category names for fast validation
        self.valid_categories: set[str] = {cat.name for cat in self.taxonomy}

        # Check for duplicates
        if len(self.valid_categories) != len(self.taxonomy):
            raise ValueError("Taxonomy contains duplicate category names")

    @property
    def output_schema(self) -> type[ClassificationOutput]:
        """
        Return the output schema for this classifier.

        Allows subclasses to override with custom classification schemas.
        """
        return ClassificationOutput

    def _build_prompt(self, mapped_input: dict[str, Any]) -> list[dict]:
        """
        Build LLM prompt messages for classification.

        Args:
            mapped_input: Dictionary with "text" (required) and "context" (optional)

        Returns:
            List of message dicts with role and content

        Raises:
            KeyError: If "text" not in mapped_input
        """
        if "text" not in mapped_input:
            raise KeyError("mapped_input must contain 'text' key")

        # Build taxonomy description
        taxonomy_lines = []
        for cat in self.taxonomy:
            taxonomy_lines.append(f"- {cat.name}: {cat.definition}")
        taxonomy_str = "\n".join(taxonomy_lines)

        # Valid category names as JSON array
        valid_names_json = json.dumps(sorted(self.valid_categories))

        # System message with taxonomy and instructions
        system_message = {
            "role": "system",
            "content": f"""You are a precise classification system. Classify the provided text into exactly one category from the taxonomy below.

TAXONOMY:
{taxonomy_str}

VALID CATEGORY NAMES (you MUST use one of these exactly):
{valid_names_json}

Respond with a JSON object matching this schema:
- category: string (must be one of the valid names above)
- confidence: number between 0.0 and 1.0
- reasoning: string explaining your classification decision

Step: {self.name}""",
        }

        # User message with text to classify
        user_content_parts = [f"Text to classify:\n{mapped_input['text']}"]

        # Add optional context if provided
        if "context" in mapped_input and mapped_input["context"]:
            user_content_parts.append(f"\nAdditional context:\n{mapped_input['context']}")

        user_message = {"role": "user", "content": "\n".join(user_content_parts)}

        return [system_message, user_message]

    def _validate_category(self, result: ClassificationOutput) -> ClassificationOutput:
        """
        Validate that the returned category is in the taxonomy.

        This is a business rule validation that happens after Pydantic validation.
        If the LLM returns an invalid category, this raises an error that triggers retry.

        Args:
            result: The classification output from the LLM

        Returns:
            The validated result (unchanged if valid)

        Raises:
            ValueError: If category is not in valid_categories
        """
        if result.category not in self.valid_categories:
            raise ValueError(
                f"Invalid category '{result.category}'. "
                f"Must be one of: {sorted(self.valid_categories)}"
            )
        return result

    def execute(
        self,
        mapped_input: dict[str, Any],
        llm_client: LLMClientWithRetry,
        pk: str,
    ) -> ClassificationOutput:
        """
        Execute classification on the mapped input.

        Args:
            mapped_input: Dictionary with "text" and optional "context"
            llm_client: LLM client with retry logic
            pk: Primary key of the record being processed

        Returns:
            Validated ClassificationOutput with category, confidence, and reasoning

        Raises:
            LLMValidationError: If validation fails after all retries
            ValueError: If category not in taxonomy (triggers retry)
        """
        # 1. Build prompt messages
        messages = self._build_prompt(mapped_input)

        # 2. Call LLM with validation and retry
        result = llm_client.complete_with_validation(
            messages=messages,
            response_model=self.output_schema,
            step_name=self.name,
            pk=pk,
        )

        # 3. Validate category is in taxonomy
        result = self._validate_category(result)

        # 4. Return validated result
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ClassifierStep("
            f"name={self.name!r}, "
            f"output_key={self.output_key!r}, "
            f"categories={sorted(self.valid_categories)})"
        )
