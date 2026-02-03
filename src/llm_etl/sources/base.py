"""
Abstract base class for data sources.

Sources are responsible for reading data from external systems
(SQL Server, CSV files, APIs, etc.) and converting each row into a GlobalState object.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from ..core.state import GlobalState


class AbstractSource(ABC):
    """
    Base class for all data sources in the pipeline.

    Subclasses must implement __iter__() to yield GlobalState objects
    and count() to optionally provide total row count for progress tracking.

    Attributes:
        primary_key_column: Name of the column to use as GlobalState.pk

    Example:
        >>> class MySource(AbstractSource):
        ...     def __init__(self, primary_key_column: str):
        ...         self.primary_key_column = primary_key_column
        ...
        ...     def __iter__(self) -> Iterator[GlobalState]:
        ...         # Read data from somewhere
        ...         for row in data:
        ...             yield GlobalState(
        ...                 pk=row[self.primary_key_column],
        ...                 raw=row
        ...             )
        ...
        ...     def count(self) -> Optional[int]:
        ...         return len(data)  # or None if unknown
    """

    primary_key_column: str

    @abstractmethod
    def __iter__(self) -> Iterator[GlobalState]:
        """
        Iterate over all rows from the source, yielding GlobalState objects.

        Each row is converted to a GlobalState with:
        - pk: Value from the primary_key_column
        - raw: Dictionary of all column values

        Yields:
            GlobalState objects, one per row

        Raises:
            SourceError: If reading from the source fails
        """
        pass

    @abstractmethod
    def count(self) -> Optional[int]:
        """
        Return the total number of rows if known upfront.

        This is used for progress tracking. If the count cannot be determined
        efficiently (e.g., for streaming sources), return None.

        Returns:
            Total row count, or None if unknown
        """
        pass
