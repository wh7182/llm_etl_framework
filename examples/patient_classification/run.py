"""
Example pipeline: Patient visit classification and summarization.

Demonstrates the complete LLM ETL framework with:
- CSV source and sink (no SQL Server required)
- ClassifierStep for visit type classification
- SummarizerStep for clinical note reduction
- MockProvider for testing without API calls
"""

import logging
from pathlib import Path

from llm_etl.core.pipeline import Pipeline
from llm_etl.llm.providers.mock import MockProvider
from llm_etl.llm.providers.azure_openai import AzureOpenaiProvider
from llm_etl.sinks.csv_sink import CSVSink
from llm_etl.sources.csv_source import CSVSource
from llm_etl.steps.classifier import ClassifierStep
from llm_etl.steps.summarizer import SummarizerStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get the directory containing this script
EXAMPLE_DIR = Path(__file__).parent


def create_pipeline() -> Pipeline:
    """
    Create a patient visit classification pipeline.

    Returns:
        Configured Pipeline instance ready to run
    """

    # Define visit type taxonomy
    visit_taxonomy = [
        {
            "name": "Emergency",
            "definition": "Urgent, unplanned visit for acute symptoms or trauma"
        },
        {
            "name": "Routine Checkup",
            "definition": "Scheduled preventive care or routine follow-up appointment"
        },
        {
            "name": "Specialist Follow-up",
            "definition": "Visit to a specialist based on referral or ongoing treatment"
        },
    ]

    # Define summarization rules
    summary_rules = [
        "Maximum 30 words",
        "Focus on chief complaint and diagnosis",
        "Preserve medication names",
    ]

    # Configure LLM provider
    # Option 1: Use MockProvider for testing without API calls
    llm_provider = MockProvider(
        default_response={
            "category": "Emergency",
            "confidence": 0.92,
            "reasoning": "Mock classification for testing",
            "text": "Patient with acute condition requiring medical attention and evaluation.",
            "word_count": 11,
            "rules_applied": ["Maximum 30 words", "Focus on chief complaint"],
        }
    )

    # Option 2: Use AzureOpenAIProvider (requires .env file with credentials)
    # Copy .env.example to .env and fill in your Azure OpenAI credentials
    # llm_provider = AzureOpenaiProvider()

    # Build the pipeline
    pipeline = Pipeline(
        name="patient_visit_enrichment",

        source=CSVSource(
            file_path=str(EXAMPLE_DIR / "sample_data.csv"),
            primary_key_column="encounter_id",
        ),

        steps=[
            ClassifierStep(
                name="visit_classifier",
                taxonomy=visit_taxonomy,
                input_map={
                    "text": lambda s: s.raw["clinical_notes"],
                    "context": lambda s: f"Department: {s.raw['department']}, Age: {s.raw['age']}, Gender: {s.raw['gender']}",
                },
                output_key="visit_type",
            ),
            SummarizerStep(
                name="note_summarizer",
                rules=summary_rules,
                input_map={
                    "text": lambda s: s.raw["clinical_notes"],
                    "focus": lambda s: s.processed["visit_type"]["category"],
                },
                output_key="summary",
            ),
        ],

        sink=CSVSink(
            file_path=str(EXAMPLE_DIR / "output" / "enriched_encounters.csv"),
            column_map={
                "encounter_id": lambda s: s.pk,
                "patient_id": lambda s: s.raw["patient_id"],
                "department": lambda s: s.raw["department"],
                "visit_category": lambda s: s.processed["visit_type"]["category"],
                "confidence": lambda s: s.processed["visit_type"]["confidence"],
                "reasoning": lambda s: s.processed["visit_type"]["reasoning"],
                "clinical_summary": lambda s: s.processed["summary"]["text"],
                "summary_word_count": lambda s: s.processed["summary"]["word_count"],
                "processed_at": lambda s: s.completed_at.isoformat() if s.completed_at else "",
            },
        ),

        # Use mock provider for testing
        llm_provider=llm_provider,

        # Error handling: write failed rows to dead letter file
        on_row_error="dead_letter",
        dead_letter_path=str(EXAMPLE_DIR / "output" / "failed_rows.jsonl"),
    )

    return pipeline


def main():
    """Run the example pipeline."""
    print("=" * 80)
    print("Patient Visit Classification Pipeline Example")
    print("=" * 80)
    print()

    # Create the pipeline
    pipeline = create_pipeline()

    # First, validate the configuration
    print("Step 1: Validating pipeline configuration...")
    print("-" * 80)
    dry_run_result = pipeline.run(dry_run=True)
    print()

    # Run the pipeline
    print("Step 2: Running pipeline on sample data...")
    print("-" * 80)
    result = pipeline.run()
    print()

    # Print results summary
    print("=" * 80)
    print("Pipeline Results")
    print("=" * 80)
    print(f"Total rows processed: {result.total_count}")
    print(f"Successful: {result.success_count}")
    print(f"Failed: {result.failure_count}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")

    if result.success_count > 0:
        print()
        print(f"✓ Output written to: {EXAMPLE_DIR / 'output' / 'enriched_encounters.csv'}")

    if result.failure_count > 0:
        print()
        print(f"⚠ Failed rows written to: {result.dead_letter_path}")

    print()
    print("Done!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()
