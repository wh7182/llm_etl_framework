"""
Data sink implementations.

Sinks are responsible for persisting processed data from the pipeline.
"""

from llm_etl.sinks.base import AbstractSink
from llm_etl.sinks.sql_server import SQLServerSink
from llm_etl.sinks.csv_sink import CSVSink

__all__ = [
    "AbstractSink",
    "SQLServerSink",
    "CSVSink",
]
