"""
CSV file source for the LLM ETL framework.

Reads rows from a CSV file and converts each to a GlobalState object.
Useful for testing pipelines without SQL Server dependencies.
"""

import csv
from pathlib import Path
from typing import Iterator, Optional

from ..core.exceptions import SourceError
from ..core.state import GlobalState
from .base import AbstractSource


class CSVSource(AbstractSource):
    """
    Read data from a CSV file.

    Each row in the CSV becomes a GlobalState object with all columns
    available in the `raw` dictionary.

    The file is read with UTF-8 encoding and handles BOM if present.

    Example:
        >>> # Given a CSV file at data/encounters.csv with columns:
        >>> # encounter_id,patient_id,clinical_notes,department
        >>> # enc_001,p_123,"Patient presents with...",Emergency
        >>> # enc_002,p_456,"Routine checkup...",Primary Care
        >>>
        >>> source = CSVSource(
        ...     file_path="data/encounters.csv",
        ...     primary_key_column="encounter_id"
        ... )
        >>>
        >>> for state in source:
        ...     print(state.pk)  # "enc_001", "enc_002", ...
        ...     print(state.raw["clinical_notes"])
        ...     print(state.raw["department"])
    """

    def __init__(
        self,
        file_path: str,
        primary_key_column: str,
    ):
        """
        Initialize the CSV source.

        Args:
            file_path: Path to the CSV file (relative or absolute)
            primary_key_column: Name of the column to use as GlobalState.pk

        Raises:
            SourceError: If the file does not exist or is not readable

        Note:
            The file is validated at initialization to fail fast if the path
            is invalid, rather than waiting until iteration begins.
        """
        self.file_path = Path(file_path)
        self.primary_key_column = primary_key_column

        # Fail fast: verify file exists
        if not self.file_path.exists():
            raise SourceError(f"CSV file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise SourceError(f"Path is not a file: {self.file_path}")

    def __iter__(self) -> Iterator[GlobalState]:
        """
        Iterate over all rows in the CSV file.

        Each row is converted to a GlobalState with:
        - pk: Value from the primary_key_column
        - raw: Dictionary of all column values from the CSV row

        Yields:
            GlobalState objects, one per CSV row

        Raises:
            SourceError: If the file cannot be read, is missing the primary key
                        column, or contains malformed CSV data
        """
        try:
            # Open with UTF-8 encoding and handle BOM
            with self.file_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                # Verify primary key column exists in CSV headers
                if reader.fieldnames is None:
                    raise SourceError(f"CSV file has no headers: {self.file_path}")

                if self.primary_key_column not in reader.fieldnames:
                    raise SourceError(
                        f"Primary key column '{self.primary_key_column}' not found in CSV. "
                        f"Available columns: {', '.join(reader.fieldnames)}"
                    )

                # Yield each row as a GlobalState
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
                    # Get primary key value
                    pk_value = row.get(self.primary_key_column)

                    if not pk_value:
                        raise SourceError(
                            f"Row {row_num} has empty primary key "
                            f"in column '{self.primary_key_column}'"
                        )

                    # Convert OrderedDict to regular dict for cleaner state
                    raw_dict = dict(row)

                    yield GlobalState(pk=str(pk_value), raw=raw_dict)

        except csv.Error as e:
            raise SourceError(f"CSV parsing error in {self.file_path}: {e}")
        except UnicodeDecodeError as e:
            raise SourceError(
                f"CSV file encoding error in {self.file_path}. "
                f"Expected UTF-8: {e}"
            )
        except OSError as e:
            raise SourceError(f"Failed to read CSV file {self.file_path}: {e}")

    def count(self) -> Optional[int]:
        """
        Return the total number of data rows in the CSV file.

        This counts all rows except the header row.

        Returns:
            Number of data rows in the file

        Note:
            This method reads the entire file to count rows, so it may be
            slow for very large CSV files. For progress tracking, consider
            using file size estimation instead if performance is critical.
        """
        try:
            with self.file_path.open("r", encoding="utf-8-sig") as f:
                # Count all lines except the header
                # We use sum() with a generator for memory efficiency
                return sum(1 for _ in f) - 1  # Subtract 1 for header
        except OSError:
            # If we can't read the file for counting, return None
            # The error will be caught when __iter__ is called
            return None
