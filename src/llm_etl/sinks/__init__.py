"""
Data sink implementations.

Sinks are responsible for persisting processed data from the pipeline.
"""

from llm_etl.sinks.base import AbstractSink
from llm_etl.sinks.csv_sink import CSVSink

# Lazy import for SQL Server to avoid pyodbc dependency when not needed
def __getattr__(name):
    if name == "SQLServerSink":
        from llm_etl.sinks.sql_server import SQLServerSink
        return SQLServerSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AbstractSink",
    "SQLServerSink",
    "CSVSink",
]
