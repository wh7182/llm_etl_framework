"""
Tests for ClassifierStep.

Validates:
- Invalid category triggers validation error
- Confidence bounds are enforced
- Prompt includes all taxonomy entries
- Taxonomy validation
"""

import pytest

from llm_etl.core.state import GlobalState
from llm_etl.llm.client import LLMClientWithRetry
from llm_etl.llm.providers.mock import MockProvider
from llm_etl.steps.classifier import ClassificationOutput, ClassifierStep, TaxonomyCategory


@pytest.fixture
def simple_taxonomy():
    """Basic taxonomy for testing."""
    return [
        {"name": "CategoryA", "definition": "First category"},
        {"name": "CategoryB", "definition": "Second category"},
        {"name": "CategoryC", "definition": "Third category"},
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_provider = MockProvider(
        responses={
            "test_classifier": {
                "category": "CategoryA",
                "confidence": 0.9,
                "reasoning": "Test reasoning"
            }
        }
    )
    return LLMClientWithRetry(mock_provider, max_retries=2)


def test_invalid_category_triggers_validation_error(simple_taxonomy, mock_llm_client):
    """Test that returning an invalid category raises ValueError."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Create a state
    state = GlobalState(pk="test_001", raw={"note": "Some text"})

    # Create a classification output with invalid category
    invalid_output = ClassificationOutput(
        category="InvalidCategory",  # Not in taxonomy
        confidence=0.9,
        reasoning="Test"
    )

    # Validation should fail
    with pytest.raises(ValueError) as exc_info:
        step._validate_category(invalid_output)

    # Error message should list valid categories
    error_msg = str(exc_info.value)
    assert "InvalidCategory" in error_msg
    assert "CategoryA" in error_msg or "CategoryB" in error_msg or "CategoryC" in error_msg


def test_valid_category_passes_validation(simple_taxonomy):
    """Test that valid categories pass validation."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Test each valid category
    for category_name in ["CategoryA", "CategoryB", "CategoryC"]:
        output = ClassificationOutput(
            category=category_name,
            confidence=0.8,
            reasoning="Test"
        )

        # Should not raise
        validated = step._validate_category(output)
        assert validated.category == category_name


def test_confidence_bounds_enforced():
    """Test that confidence must be between 0.0 and 1.0."""
    from pydantic import ValidationError

    # Valid confidence values should work
    valid_output = ClassificationOutput(
        category="Test",
        confidence=0.5,
        reasoning="Test"
    )
    assert valid_output.confidence == 0.5

    # Test boundary values
    min_output = ClassificationOutput(category="Test", confidence=0.0, reasoning="Test")
    assert min_output.confidence == 0.0

    max_output = ClassificationOutput(category="Test", confidence=1.0, reasoning="Test")
    assert max_output.confidence == 1.0

    # Confidence > 1.0 should fail
    with pytest.raises(ValidationError) as exc_info:
        ClassificationOutput(category="Test", confidence=1.5, reasoning="Test")
    assert "less than or equal to 1" in str(exc_info.value).lower()

    # Confidence < 0.0 should fail
    with pytest.raises(ValidationError) as exc_info:
        ClassificationOutput(category="Test", confidence=-0.5, reasoning="Test")
    assert "greater than or equal to 0" in str(exc_info.value).lower()


def test_prompt_includes_all_taxonomy_entries(simple_taxonomy):
    """Test that the generated prompt includes all taxonomy categories."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Build a prompt
    mapped_input = {"text": "Sample text to classify"}
    messages = step._build_prompt(mapped_input)

    # Find the system message
    system_message = next(msg for msg in messages if msg["role"] == "system")
    system_content = system_message["content"]

    # All taxonomy entries should be in the prompt
    assert "CategoryA" in system_content
    assert "First category" in system_content
    assert "CategoryB" in system_content
    assert "Second category" in system_content
    assert "CategoryC" in system_content
    assert "Third category" in system_content

    # Step name should be in the prompt
    assert "test_classifier" in system_content


def test_prompt_includes_input_text(simple_taxonomy):
    """Test that the user message includes the input text."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Build a prompt with specific text
    test_text = "This is the text to classify"
    mapped_input = {"text": test_text}
    messages = step._build_prompt(mapped_input)

    # Find the user message
    user_message = next(msg for msg in messages if msg["role"] == "user")
    user_content = user_message["content"]

    # Input text should be in the user message
    assert test_text in user_content


def test_prompt_includes_optional_context(simple_taxonomy):
    """Test that optional context is included in the prompt when provided."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={
            "text": lambda s: s.raw["note"],
            "context": lambda s: s.raw["context"]
        },
        output_key="classification"
    )

    # Build a prompt with context
    mapped_input = {
        "text": "Text to classify",
        "context": "Additional context information"
    }
    messages = step._build_prompt(mapped_input)

    # Find the user message
    user_message = next(msg for msg in messages if msg["role"] == "user")
    user_content = user_message["content"]

    # Both text and context should be present
    assert "Text to classify" in user_content
    assert "Additional context information" in user_content


def test_empty_taxonomy_raises_error():
    """Test that creating a classifier with empty taxonomy raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        ClassifierStep(
            name="test_classifier",
            taxonomy=[],  # Empty taxonomy
            input_map={"text": lambda s: s.raw["note"]},
            output_key="classification"
        )
    assert "cannot be empty" in str(exc_info.value).lower()


def test_duplicate_taxonomy_names_raises_error():
    """Test that duplicate category names in taxonomy raise ValueError."""
    duplicate_taxonomy = [
        {"name": "Category", "definition": "First definition"},
        {"name": "Category", "definition": "Second definition"},  # Duplicate name
    ]

    with pytest.raises(ValueError) as exc_info:
        ClassifierStep(
            name="test_classifier",
            taxonomy=duplicate_taxonomy,
            input_map={"text": lambda s: s.raw["note"]},
            output_key="classification"
        )
    assert "duplicate" in str(exc_info.value).lower()


def test_taxonomy_from_dicts(simple_taxonomy):
    """Test that taxonomy can be created from dicts or TaxonomyCategory objects."""
    # Create from dicts
    step_from_dicts = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Create from TaxonomyCategory objects
    taxonomy_objects = [TaxonomyCategory(**cat) for cat in simple_taxonomy]
    step_from_objects = ClassifierStep(
        name="test_classifier",
        taxonomy=taxonomy_objects,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Both should have the same valid categories
    assert step_from_dicts.valid_categories == step_from_objects.valid_categories
    assert step_from_dicts.valid_categories == {"CategoryA", "CategoryB", "CategoryC"}


def test_execute_with_mock_provider(simple_taxonomy, mock_llm_client):
    """Test that execute() works end-to-end with mock provider."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    state = GlobalState(pk="test_001", raw={"note": "Sample clinical note"})

    # Map the input
    mapped_input = {key: fn(state) for key, fn in step.input_map.items()}

    # Execute
    result = step.execute(mapped_input, mock_llm_client, state.pk)

    # Verify result
    assert isinstance(result, ClassificationOutput)
    assert result.category == "CategoryA"
    assert result.confidence == 0.9
    assert result.reasoning == "Test reasoning"


def test_step_run_updates_state(simple_taxonomy, mock_llm_client):
    """Test that step.run() properly updates GlobalState."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    state = GlobalState(pk="test_001", raw={"note": "Sample note"})

    # Initially empty
    assert "classification" not in state.processed
    assert len(state.log) == 0

    # Run the step
    updated_state = step.run(state, mock_llm_client)

    # State should be updated
    assert "classification" in updated_state.processed
    assert updated_state.processed["classification"]["category"] == "CategoryA"
    assert "test_classifier" in updated_state.log


def test_repr_includes_key_info(simple_taxonomy):
    """Test that __repr__ provides useful debugging information."""
    step = ClassifierStep(
        name="my_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="my_classification"
    )

    repr_str = repr(step)

    # Should include step name, output key, and categories
    assert "my_classifier" in repr_str
    assert "my_classification" in repr_str
    # At least one category should be mentioned
    assert "CategoryA" in repr_str or "CategoryB" in repr_str or "CategoryC" in repr_str


def test_missing_text_in_input_map_raises_error(simple_taxonomy, mock_llm_client):
    """Test that missing 'text' key in mapped_input raises KeyError."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"context": lambda s: s.raw["context"]},  # Missing "text"
        output_key="classification"
    )

    # This should raise KeyError when building prompt
    with pytest.raises(KeyError) as exc_info:
        step._build_prompt({"context": "some context"})
    assert "text" in str(exc_info.value).lower()


def test_valid_category_names_json_in_prompt(simple_taxonomy):
    """Test that prompt includes valid category names in JSON format."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    messages = step._build_prompt({"text": "Test"})
    system_message = next(msg for msg in messages if msg["role"] == "system")
    system_content = system_message["content"]

    # Should contain JSON array of valid names
    import json
    # Extract the JSON array (look for pattern like ["CategoryA", "CategoryB", "CategoryC"])
    assert "[" in system_content and "]" in system_content

    # The valid names should appear in sorted order in the prompt
    valid_names_sorted = '["CategoryA", "CategoryB", "CategoryC"]'
    assert valid_names_sorted in system_content


def test_case_sensitive_category_validation(simple_taxonomy):
    """Test that category validation is case-sensitive."""
    step = ClassifierStep(
        name="test_classifier",
        taxonomy=simple_taxonomy,
        input_map={"text": lambda s: s.raw["note"]},
        output_key="classification"
    )

    # Exact match should work
    valid = ClassificationOutput(category="CategoryA", confidence=0.8, reasoning="Test")
    assert step._validate_category(valid).category == "CategoryA"

    # Wrong case should fail
    invalid = ClassificationOutput(category="categorya", confidence=0.8, reasoning="Test")
    with pytest.raises(ValueError):
        step._validate_category(invalid)
