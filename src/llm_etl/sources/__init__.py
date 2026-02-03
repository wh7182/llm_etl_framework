"""
Data source implementations.

Sources are responsible for ingesting data into the pipeline.
"""

from llm_etl.sources.base import AbstractSource
from llm_etl.sources.csv_source import CSVSource

# Lazy import for SQL Server to avoid pyodbc dependency when not needed
def __getattr__(name):
    if name == "SQLServerSource":
        from llm_etl.sources.sql_server import SQLServerSource
        return SQLServerSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AbstractSource",
    "SQLServerSource",
    "CSVSource",
]
