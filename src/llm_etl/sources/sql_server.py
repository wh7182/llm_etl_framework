"""
SQL Server data source implementation.

Reads data from SQL Server using a .sql file and converts each row to GlobalState.
Supports environment variable fallback for connection strings.
"""

import os
from pathlib import Path
from typing import Iterator, Optional

import pyodbc

from ..core.exceptions import SourceError
from ..core.state import GlobalState
from .base import AbstractSource


class SQLServerSource(AbstractSource):
    """
    SQL Server data source.

    Executes a SQL query from a file and yields GlobalState objects for each row.
    The SQL file is read at initialization for fail-fast error detection.

    Attributes:
        sql_file: Path to .sql file containing the query
        primary_key_column: Column name to use as GlobalState.pk
        connection_string: ODBC connection string (reads from SQL_SERVER_CONN env var if None)

    Example:
        >>> source = SQLServerSource(
        ...     sql_file="sql/get_encounters.sql",
        ...     primary_key_column="encounter_id",
        ... )
        >>>
        >>> for state in source:
        ...     print(f"Processing {state.pk}")
        ...     print(f"Raw data: {state.raw}")

    SQL File Example:
        ```sql
        -- sql/get_encounters.sql
        SELECT
            encounter_id,
            patient_id,
            clinical_notes,
            department
        FROM dbo.encounters
        WHERE encounter_date >= DATEADD(day, -7, GETDATE())
        ORDER BY encounter_date;
        ```

    Environment Variables:
        SQL_SERVER_CONN: Default ODBC connection string
            Example: "Driver={ODBC Driver 18 for SQL Server};Server=...;Database=...;UID=...;PWD=..."
    """

    def __init__(
        self,
        sql_file: str,
        primary_key_column: str,
        connection_string: Optional[str] = None,
    ):
        """
        Initialize the SQL Server source.

        Reads the SQL file content at initialization for fail-fast error detection.
        If connection_string is not provided, reads from SQL_SERVER_CONN environment variable.

        Args:
            sql_file: Path to .sql file containing the query
            primary_key_column: Column name to use as GlobalState.pk
            connection_string: ODBC connection string (optional, defaults to env var)

        Raises:
            FileNotFoundError: If the SQL file does not exist
            ValueError: If connection_string is None and SQL_SERVER_CONN env var is not set
        """
        self.primary_key_column = primary_key_column
        self.sql_file = sql_file

        # Read SQL file content (fail fast if file doesn't exist)
        sql_path = Path(sql_file)
        if not sql_path.exists():
            raise FileNotFoundError(
                f"SQL file not found: {sql_file}. "
                f"Provide an absolute path or a path relative to the current working directory."
            )

        self.sql_content = sql_path.read_text(encoding="utf-8")

        # Get connection string from parameter or environment
        if connection_string is None:
            connection_string = os.getenv("SQL_SERVER_CONN")
            if connection_string is None:
                raise ValueError(
                    "No connection string provided. Either pass connection_string parameter "
                    "or set SQL_SERVER_CONN environment variable."
                )

        self.connection_string = connection_string

    def __iter__(self) -> Iterator[GlobalState]:
        """
        Execute the SQL query and yield GlobalState objects for each row.

        Connects to SQL Server, executes the query, and converts each row
        to a GlobalState object using the configured primary_key_column.

        Yields:
            GlobalState objects with pk and raw data from each row

        Raises:
            SourceError: If connection fails, query execution fails, or primary key column is missing
        """
        try:
            # Connect to SQL Server
            conn = pyodbc.connect(self.connection_string)
        except pyodbc.Error as e:
            raise SourceError(
                f"Failed to connect to SQL Server: {e}. "
                f"Check your connection string and network connectivity."
            ) from e

        try:
            cursor = conn.cursor()

            # Execute the query
            try:
                cursor.execute(self.sql_content)
            except pyodbc.Error as e:
                raise SourceError(
                    f"Failed to execute SQL query from {self.sql_file}: {e}. "
                    f"Check your SQL syntax and table permissions."
                ) from e

            # Get column names from cursor description
            if cursor.description is None:
                raise SourceError(
                    f"Query in {self.sql_file} did not return any columns. "
                    f"Ensure the query is a SELECT statement."
                )

            column_names = [col[0] for col in cursor.description]

            # Validate primary key column exists
            if self.primary_key_column not in column_names:
                raise SourceError(
                    f"Primary key column '{self.primary_key_column}' not found in query results. "
                    f"Available columns: {column_names}"
                )

            # Yield GlobalState for each row
            for row in cursor:
                # Convert row to dictionary
                row_dict = dict(zip(column_names, row))

                # Extract primary key value
                pk_value = row_dict[self.primary_key_column]
                if pk_value is None:
                    raise SourceError(
                        f"Primary key column '{self.primary_key_column}' has NULL value. "
                        f"All primary keys must be non-NULL."
                    )

                # Convert pk to string for consistency
                pk_str = str(pk_value)

                # Create and yield GlobalState
                yield GlobalState(pk=pk_str, raw=row_dict)

        except SourceError:
            # Re-raise SourceError as-is
            raise
        except Exception as e:
            # Wrap any other errors
            raise SourceError(f"Unexpected error while reading from SQL Server: {e}") from e
        finally:
            # Always close the connection
            conn.close()

    def count(self) -> Optional[int]:
        """
        Return the total number of rows (not implemented).

        Implementing COUNT(*) wrapper is an optional enhancement.
        For now, returns None to indicate unknown row count.

        Returns:
            None (row count unknown)
        """
        return None
