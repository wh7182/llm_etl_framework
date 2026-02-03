"""
Base Pydantic schemas for LLM outputs.

All LLM responses inherit from LLMOutputBase to ensure consistent
reasoning and debugging capabilities.
"""

from pydantic import BaseModel, Field


class LLMOutputBase(BaseModel):
    """
    Base class for all LLM output schemas.

    Ensures every LLM response includes reasoning for its decision,
    which is invaluable for debugging and auditing pipeline behavior.

    All step-specific output schemas should inherit from this class.

    Example:
        >>> class ClassificationOutput(LLMOutputBase):
        ...     category: str = Field(..., description="The assigned category")
        ...     confidence: float = Field(..., ge=0.0, le=1.0)
        ...
        >>> output = ClassificationOutput(
        ...     category="Emergency",
        ...     confidence=0.92,
        ...     reasoning="Patient presents with acute symptoms requiring immediate care"
        ... )
        >>> output.reasoning
        'Patient presents with acute symptoms requiring immediate care'
    """

    reasoning: str = Field(
        ...,
        description="Step-by-step explanation of the model's decision process",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "reasoning": "The model analyzed the input and determined..."
            }
        }
    }
