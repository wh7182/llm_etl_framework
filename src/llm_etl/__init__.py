"""
LLM ETL Framework - A modular Python framework for Cognitive ETL pipelines.

Extract data from SQL Server, process unstructured text with Azure OpenAI,
and sink structured results back to SQL Server—all with type-safe schemas
and automatic retry logic.
"""

__version__ = "0.1.0"

# Core components
from llm_etl.core.pipeline import Pipeline
from llm_etl.core.state import GlobalState
from llm_etl.core.exceptions import (
    LLMETLException,
    ValidationError,
    PipelineError,
    StepError,
)

# Steps
from llm_etl.steps.classifier import ClassifierStep
from llm_etl.steps.summarizer import SummarizerStep

# Sources
from llm_etl.sources.sql_server import SQLServerSource
from llm_etl.sources.csv_source import CSVSource

# Sinks
from llm_etl.sinks.sql_server import SQLServerSink
from llm_etl.sinks.csv_sink import CSVSink

# LLM Providers
from llm_etl.llm.providers.mock import MockProvider
from llm_etl.llm.providers.azure_openai import AzureOpenAIProvider

__all__ = [
    # Version
    "__version__",
    # Core
    "Pipeline",
    "GlobalState",
    "LLMETLException",
    "ValidationError",
    "PipelineError",
    "StepError",
    # Steps
    "ClassifierStep",
    "SummarizerStep",
    # Sources
    "SQLServerSource",
    "CSVSource",
    # Sinks
    "SQLServerSink",
    "CSVSink",
    # LLM Providers
    "MockProvider",
    "AzureOpenAIProvider",
]
