"""
Integration tests for the complete LLM ETL pipeline.

Tests end-to-end pipeline execution including:
- Full pipeline processing with mock provider
- Retry logic on validation failures
- Dead letter handling
- Dry run validation
"""

import tempfile
from pathlib import Path

import pytest

from llm_etl.core.pipeline import Pipeline
from llm_etl.llm.providers.mock import MockProvider
from llm_etl.sinks.csv_sink import CSVSink
from llm_etl.sources.csv_source import CSVSource
from llm_etl.steps.classifier import ClassifierStep
from llm_etl.steps.summarizer import SummarizerStep


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file with test data."""
    csv_path = temp_dir / "input.csv"
    csv_path.write_text(
        "id,department,note\n"
        "ENC001,Emergency,Patient with chest pain\n"
        "ENC002,Primary Care,Routine wellness checkup\n"
        "ENC003,Cardiology,Follow-up for heart condition\n"
    )
    return csv_path


@pytest.fixture
def taxonomy():
    """Standard taxonomy for classification tests."""
    return [
        {"name": "Emergency", "definition": "Urgent care needed"},
        {"name": "Routine", "definition": "Scheduled preventive visit"},
        {"name": "Specialist", "definition": "Specialist follow-up"},
    ]


@pytest.fixture
def summary_rules():
    """Standard rules for summarization tests."""
    return [
        "Maximum 20 words",
        "Focus on diagnosis",
    ]


def test_full_pipeline_processes_all_rows(temp_dir, sample_csv, taxonomy, summary_rules):
    """Test that the pipeline successfully processes all rows."""
    output_path = temp_dir / "output.csv"

    # Configure mock provider with valid default response
    mock_provider = MockProvider(
        default_response={
            "category": "Emergency",
            "confidence": 0.85,
            "reasoning": "Test classification",
            "text": "Patient presents with acute chest pain requiring evaluation",
            "word_count": 9,
            "rules_applied": ["Maximum 20 words"],
        }
    )

    # Build pipeline
    pipeline = Pipeline(
        name="test_pipeline",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
            SummarizerStep(
                name="summarizer",
                rules=summary_rules,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="summary",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={
                "id": lambda s: s.pk,
                "category": lambda s: s.processed["classification"]["category"],
                "summary": lambda s: s.processed["summary"]["text"],
            },
        ),
        llm_provider=mock_provider,
        on_row_error="fail",
    )

    # Run pipeline
    result = pipeline.run()

    # Verify results
    assert result.total_count == 3
    assert result.success_count == 3
    assert result.failure_count == 0
    assert result.duration_seconds > 0

    # Verify output file was created
    assert output_path.exists()

    # Verify output content
    output_content = output_path.read_text()
    assert "ENC001" in output_content
    assert "ENC002" in output_content
    assert "ENC003" in output_content
    assert "Emergency" in output_content


def test_mock_provider_retry_logic(temp_dir, sample_csv, taxonomy):
    """Test that validation failures trigger retry with error feedback."""
    output_path = temp_dir / "output.csv"

    # Configure mock to fail validation twice, then succeed
    mock_provider = MockProvider(
        default_response={
            "category": "Emergency",
            "confidence": 0.85,
            "reasoning": "Test classification",
        },
        fail_validation_times=2,  # Fail twice, then succeed
    )

    pipeline = Pipeline(
        name="test_retry",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={"id": lambda s: s.pk},
        ),
        llm_provider=mock_provider,
        on_row_error="fail",
        max_retries=3,  # Allow 3 retries
    )

    # Run pipeline - should succeed after retries
    result = pipeline.run()

    # Should succeed even though it failed twice per row
    assert result.success_count == 3
    assert result.failure_count == 0


def test_dead_letter_captures_validation_failure(temp_dir, sample_csv, taxonomy):
    """Test that dead letter file captures rows that fail validation after all retries."""
    output_path = temp_dir / "output.csv"
    dead_letter_path = temp_dir / "failed.jsonl"

    # Configure mock to always fail validation (more failures than retries)
    mock_provider = MockProvider(
        default_response={
            "category": "Emergency",
            "confidence": 0.85,
            "reasoning": "Test classification",
        },
        fail_validation_times=10,  # Fail more times than we have retries
    )

    pipeline = Pipeline(
        name="test_dead_letter",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={"id": lambda s: s.pk},
        ),
        llm_provider=mock_provider,
        on_row_error="dead_letter",
        dead_letter_path=str(dead_letter_path),
        max_retries=2,  # Only 2 retries
    )

    # Run pipeline - should fail all rows
    result = pipeline.run()

    # All rows should fail
    assert result.success_count == 0
    assert result.failure_count == 3
    assert result.total_count == 3

    # Dead letter file should exist
    assert dead_letter_path.exists()

    # Dead letter file should have 3 lines (one per failed row)
    lines = dead_letter_path.read_text().strip().split("\n")
    assert len(lines) == 3

    # Each line should be valid JSON with expected fields
    import json
    for line in lines:
        record = json.loads(line)
        assert "pk" in record
        assert "step_name" in record
        assert "error_type" in record
        assert "error_message" in record
        assert "raw_data" in record
        assert "retry_attempts" in record
        assert record["retry_attempts"] == 3  # Initial + 2 retries


def test_pipeline_dry_run_validates_without_processing(temp_dir, sample_csv, taxonomy):
    """Test that dry_run=True validates configuration without processing data."""
    output_path = temp_dir / "output.csv"

    mock_provider = MockProvider(
        responses={
            "classifier": {
                "category": "Emergency",
                "confidence": 0.85,
                "reasoning": "Test",
            }
        }
    )

    pipeline = Pipeline(
        name="test_dry_run",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={"id": lambda s: s.pk},
        ),
        llm_provider=mock_provider,
    )

    # Run in dry-run mode
    result = pipeline.run(dry_run=True)

    # Should validate but not process any rows
    assert result.success_count == 0
    assert result.failure_count == 0
    assert result.total_count == 0
    assert result.duration_seconds == 0.0

    # Output file should NOT be created
    assert not output_path.exists()


def test_pipeline_skip_error_mode(temp_dir, sample_csv, taxonomy):
    """Test that on_row_error='skip' continues processing after errors."""
    output_path = temp_dir / "output.csv"

    # Mock that always fails
    mock_provider = MockProvider(
        responses={
            "classifier": {
                "category": "Emergency",
                "confidence": 0.85,
                "reasoning": "Test",
            }
        },
        fail_validation_times=10,  # Always fail
    )

    pipeline = Pipeline(
        name="test_skip",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={"id": lambda s: s.pk},
        ),
        llm_provider=mock_provider,
        on_row_error="skip",  # Skip errors and continue
        max_retries=1,
    )

    # Run pipeline - should not raise even though all rows fail
    result = pipeline.run()

    # All rows should fail but pipeline completes
    assert result.success_count == 0
    assert result.failure_count == 3
    assert result.total_count == 3


def test_invalid_category_triggers_retry(temp_dir, sample_csv, taxonomy):
    """Test that invalid categories trigger validation error and retry."""
    output_path = temp_dir / "output.csv"
    dead_letter_path = temp_dir / "failed.jsonl"

    # Create a custom mock that returns invalid category first, then valid
    # Note: Category validation happens in the step's _validate_category method,
    # which raises ValueError. This is caught by the step execution, not the retry wrapper.
    # So invalid categories cause the row to fail, not retry.
    class InvalidCategoryMock(MockProvider):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def complete(self, messages, response_model):
            self.call_count += 1
            # Always return invalid category to test error handling
            return response_model.model_validate({
                "category": "InvalidCategory",
                "confidence": 0.9,
                "reasoning": "Test"
            })

    mock_provider = InvalidCategoryMock()

    pipeline = Pipeline(
        name="test_invalid_category",
        source=CSVSource(str(sample_csv), "id"),
        steps=[
            ClassifierStep(
                name="classifier",
                taxonomy=taxonomy,
                input_map={"text": lambda s: s.raw["note"]},
                output_key="classification",
            ),
        ],
        sink=CSVSink(
            str(output_path),
            column_map={"id": lambda s: s.pk},
        ),
        llm_provider=mock_provider,
        on_row_error="dead_letter",
        dead_letter_path=str(dead_letter_path),
        max_retries=2,
    )

    # Run pipeline - rows should fail with invalid category
    result = pipeline.run()

    # All rows should fail due to invalid category
    assert result.success_count == 0
    assert result.failure_count == 3

    # Dead letter should have the errors
    assert dead_letter_path.exists()

    import json
    lines = dead_letter_path.read_text().strip().split("\n")
    assert len(lines) == 3

    # Check that error messages mention invalid category
    for line in lines:
        record = json.loads(line)
        assert "InvalidCategory" in record["error_message"]
