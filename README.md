A modular Python framework for Cognitive ETL pipelines. Extract data from SQL Server, process unstructured text with Azure OpenAI, and sink structured results back to SQL Server—all with type-safe schemas and automatic retry logic. The project should use uv

---

## Architecture Overview

```
llm_etl_framework/
├── src/
│   └── llm_etl/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py          # Orchestration & error handling
│       │   ├── state.py             # GlobalState container
│       │   └── exceptions.py        # Custom exceptions
│       ├── steps/
│       │   ├── __init__.py
│       │   ├── base.py              # AbstractBaseStep with input_map pattern
│       │   ├── classifier.py        # Taxonomy-constrained classification
│       │   └── summarizer.py        # Rule-based text reduction
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py            # LLM client with retry logic
│       │   ├── base_schemas.py      # LLMOutputBase & shared patterns
│       │   └── providers/
│       │       ├── __init__.py
│       │       ├── azure_openai.py  # Azure OpenAI implementation
│       │       └── mock.py          # Mock provider for testing
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── base.py              # AbstractSource
│       │   ├── sql_server.py        # SQL Server ingestion from .sql files
│       │   └── csv_source.py        # CSV ingestion
│       ├── sinks/
│       │   ├── __init__.py
│       │   ├── base.py              # AbstractSink
│       │   ├── sql_server.py        # SQL Server MERGE/upsert
│       │   └── csv_sink.py          # CSV export
│       └── validation/
│           ├── __init__.py
│           └── retry.py             # Retry-with-error-feedback logic
├── examples/
│   └── patient_classification/
│       ├── sql/
│       │   └── get_encounters.sql
│       ├── schemas.py
│       └── run.py
├── tests/
│   ├── test_state.py
│   ├── test_classifier.py
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

---

## Core Concepts

### GlobalState

Each row from the source becomes a `GlobalState` object that travels through the pipeline:

```python
state.pk                           # Primary key value (e.g., "patient_123")
state.raw["ClinicalNote"]          # Original data from source
state.processed["visit_type"]      # Output from ClassifierStep
state.processed["summary"]         # Output from SummarizerStep
state.log                          # Audit trail: ["classifier", "summarizer"]
```

### Input Mapping (Decoupled Steps)

Steps don't know about each other. They receive data through an `input_map`—a dictionary of lambdas that pull from GlobalState:

```python
ClassifierStep(
    input_map={
        "text": lambda s: s.raw["ClinicalNote"],
        "context": lambda s: f"Patient age: {s.raw['Age']}"
    },
    output_key="visit_type"
)

# Later step can reference earlier step's output:
SummarizerStep(
    input_map={
        "text": lambda s: s.raw["ClinicalNote"],
        "focus": lambda s: s.processed["visit_type"]["category"]  # Uses classifier output
    },
    output_key="summary"
)
```

### Validation & Retry

All LLM outputs are validated against Pydantic schemas. If validation fails:
1. The error message is appended to the prompt
2. The LLM is re-called to fix its response
3. This repeats up to `max_retries` times (default: 3)

---

## Quick Start

### Installation

```bash
cd llm_etl_framework
pip install -e .
```

### Environment Variables

```bash
# For Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# For SQL Server
export SQL_SERVER_CONN="Driver={ODBC Driver 18 for SQL Server};Server=...;Database=...;UID=...;PWD=..."
```

### Example Pipeline

```python
# examples/patient_classification/run.py

from llm_etl.core.pipeline import Pipeline
from llm_etl.sources.sql_server import SQLServerSource
from llm_etl.steps.classifier import ClassifierStep
from llm_etl.steps.summarizer import SummarizerStep
from llm_etl.sinks.sql_server import SQLServerSink
from llm_etl.llm.providers.mock import MockProvider  # Start with mock!

# Define taxonomy for visit classification
visit_taxonomy = [
    {
        "name": "Emergency",
        "definition": "Urgent, unplanned visit for acute symptoms or trauma"
    },
    {
        "name": "Routine Checkup", 
        "definition": "Scheduled preventive care or follow-up appointment"
    },
    {
        "name": "Specialist Referral",
        "definition": "Visit to a specialist based on primary care referral"
    },
]

# Build the pipeline
pipeline = Pipeline(
    name="patient_visit_enrichment",
    
    source=SQLServerSource(
        sql_file="sql/get_encounters.sql",
        primary_key_column="encounter_id",
    ),
    
    steps=[
        ClassifierStep(
            name="visit_classifier",
            taxonomy=visit_taxonomy,
            input_map={
                "text": lambda s: s.raw["clinical_notes"],
                "context": lambda s: f"Department: {s.raw['department']}",
            },
            output_key="visit_type",
        ),
        SummarizerStep(
            name="note_summarizer",
            input_map={
                "text": lambda s: s.raw["clinical_notes"],
                "focus": lambda s: s.processed["visit_type"]["category"],
            },
            output_key="summary",
        ),
    ],
    
    sink=SQLServerSink(
        target_table="dbo.enriched_encounters",
        merge_keys=["encounter_id"],
        column_map={
            "encounter_id": lambda s: s.pk,
            "visit_category": lambda s: s.processed["visit_type"]["category"],
            "confidence": lambda s: s.processed["visit_type"]["confidence"],
            "clinical_summary": lambda s: s.processed["summary"]["text"],
            "processed_at": lambda s: s.completed_at,
        },
    ),
    
    # Use mock provider during development
    llm_provider=MockProvider(),
    
    # Error handling
    on_row_error="dead_letter",  # Options: "fail", "skip", "dead_letter"
    dead_letter_path="output/failed_rows.jsonl",
)

if __name__ == "__main__":
    result = pipeline.run()
    print(f"Processed: {result.success_count}, Failed: {result.failure_count}")
```

### Switching to Azure OpenAI

When ready for real LLM calls, swap the provider:

```python
from llm_etl.llm.providers.azure_openai import AzureOpenAIProvider

pipeline = Pipeline(
    # ... same config ...
    llm_provider=AzureOpenAIProvider(),  # Reads from environment variables
)
```

---

## SQL File Examples

### Source Query

```sql
-- sql/get_encounters.sql
SELECT
    e.encounter_id,
    e.patient_id,
    e.encounter_date,
    e.department,
    e.clinical_notes,
    p.age,
    p.gender
FROM dbo.encounters e
INNER JOIN dbo.patients p ON e.patient_id = p.patient_id
WHERE e.encounter_date >= DATEADD(day, -7, GETDATE())
  AND e.status = 'COMPLETED'
ORDER BY e.encounter_date;
```

### Sink Table

```sql
CREATE TABLE dbo.enriched_encounters (
    encounter_id        VARCHAR(50) PRIMARY KEY,
    visit_category      VARCHAR(100) NOT NULL,
    confidence          DECIMAL(3,2) NOT NULL,
    clinical_summary    NVARCHAR(500) NOT NULL,
    processed_at        DATETIME2 NOT NULL,
    
    INDEX IX_processed_at (processed_at)
);
```

---

## Debugging Features

### LLM Payload Logging

All LLM requests and responses are logged to `logs/llm_payloads.jsonl`:

```json
{
  "timestamp": "2025-02-02T10:30:00Z",
  "step": "visit_classifier",
  "pk": "enc_12345",
  "request": {"messages": [...], "model": "gpt-4o"},
  "response": {"category": "Emergency", "confidence": 0.92, "reasoning": "..."},
  "retry_count": 0,
  "latency_ms": 450
}
```

### Dead Letter Office

Failed rows are written to `dead_letter_path` with full context:

```json
{
  "pk": "enc_67890",
  "step_name": "visit_classifier",
  "error_type": "ValidationError",
  "error_message": "confidence must be between 0 and 1, got 1.5",
  "raw_data": {"clinical_notes": "...", "department": "..."},
  "partial_state": {"visit_type": null},
  "timestamp": "2025-02-02T10:31:00Z",
  "retry_attempts": 3
}
```

### GlobalState Serialization

Any `GlobalState` can be serialized for debugging:

```python
state.to_dict()  # Full state as JSON-serializable dict
state.to_json()  # JSON string for logging
```

---

## API Reference

### Steps

| Step | Purpose | Required Input | Output Schema |
|------|---------|----------------|---------------|
| `ClassifierStep` | Classify text into predefined taxonomy | `text`, optional `context` | `{category, confidence, reasoning}` |
| `SummarizerStep` | Summarize text with optional focus context | `text`, optional `focus` | `{text, reasoning}` |

### Sources

| Source | Configuration |
|--------|---------------|
| `SQLServerSource` | `sql_file`, `primary_key_column`, `connection_string` (env default) |
| `CSVSource` | `file_path`, `primary_key_column` |

### Sinks

| Sink | Configuration |
|------|---------------|
| `SQLServerSink` | `target_table`, `merge_keys`, `column_map`, `connection_string` (env default) |
| `CSVSink` | `file_path`, `column_map` |

---

## License

MIT