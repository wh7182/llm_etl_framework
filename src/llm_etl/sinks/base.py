"""
Abstract base class for data sinks.

Sinks are responsible for writing processed GlobalState objects to external systems
(SQL Server, CSV files, APIs, etc.) using a configurable column mapping.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..core.state import GlobalState


class AbstractSink(ABC):
    """
    Base class for all data sinks in the pipeline.

    Subclasses must implement write() for single-row operations and
    write_batch() for transactional batch operations.

    Attributes:
        column_map: Dictionary mapping output column names to extractor functions
                   that pull values from GlobalState

    Example:
        >>> class MySink(AbstractSink):
        ...     def __init__(self, column_map: dict):
        ...         self.column_map = column_map
        ...
        ...     def write(self, state: GlobalState) -> None:
        ...         # Extract values using column_map
        ...         row = {col: fn(state) for col, fn in self.column_map.items()}
        ...         # Write row to destination
        ...         print(f"Writing: {row}")
        ...
        ...     def write_batch(self, states: list[GlobalState]) -> None:
        ...         # Write all rows in a transaction
        ...         for state in states:
        ...             self.write(state)
        >>>
        >>> sink = MySink(column_map={
        ...     "encounter_id": lambda s: s.pk,
        ...     "visit_category": lambda s: s.processed["visit_type"]["category"],
        ...     "confidence": lambda s: s.processed["visit_type"]["confidence"],
        ... })
    """

    column_map: dict[str, Callable[[GlobalState], Any]]

    @abstractmethod
    def write(self, state: GlobalState) -> None:
        """
        Write a single GlobalState to the sink.

        Applies the column_map to extract values and writes them to the destination.

        Args:
            state: GlobalState object to write

        Raises:
            SinkError: If writing fails
        """
        pass

    @abstractmethod
    def write_batch(self, states: list[GlobalState]) -> None:
        """
        Write multiple GlobalState objects in a transaction.

        Ensures atomicity: either all rows are written successfully,
        or none are written (rollback on error).

        Args:
            states: List of GlobalState objects to write

        Raises:
            SinkError: If writing fails, includes pk of the failed row
        """
        pass
