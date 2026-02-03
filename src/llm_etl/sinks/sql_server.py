"""
SQL Server data sink implementation.

Writes GlobalState objects to SQL Server using MERGE (upsert) operations.
Supports environment variable fallback for connection strings.
"""

import os
from typing import Any, Callable, Optional

import pyodbc

from ..core.exceptions import SinkError
from ..core.state import GlobalState
from .base import AbstractSink


class SQLServerSink(AbstractSink):
    """
    SQL Server data sink using MERGE for upsert operations.

    Writes processed GlobalState objects to a SQL Server table. Uses MERGE
    to update existing rows or insert new ones based on merge_keys.

    Attributes:
        target_table: Fully qualified table name (e.g., "dbo.enriched_encounters")
        merge_keys: List of column names that uniquely identify a row
        column_map: Dictionary mapping column names to extractors from GlobalState
        connection_string: ODBC connection string (reads from SQL_SERVER_CONN env var if None)

    Example:
        >>> sink = SQLServerSink(
        ...     target_table="dbo.enriched_encounters",
        ...     merge_keys=["encounter_id"],
        ...     column_map={
        ...         "encounter_id": lambda s: s.pk,
        ...         "visit_category": lambda s: s.processed["visit_type"]["category"],
        ...         "confidence": lambda s: s.processed["visit_type"]["confidence"],
        ...         "clinical_summary": lambda s: s.processed["summary"]["text"],
        ...         "processed_at": lambda s: s.completed_at,
        ...     },
        ... )
        >>>
        >>> sink.write(state)  # Single row upsert

    Generated MERGE SQL:
        ```sql
        MERGE dbo.enriched_encounters AS target
        USING (SELECT ?, ?, ?, ?, ?) AS source (encounter_id, visit_category, confidence, clinical_summary, processed_at)
        ON target.encounter_id = source.encounter_id
        WHEN MATCHED THEN
            UPDATE SET
                visit_category = source.visit_category,
                confidence = source.confidence,
                clinical_summary = source.clinical_summary,
                processed_at = source.processed_at
        WHEN NOT MATCHED THEN
            INSERT (encounter_id, visit_category, confidence, clinical_summary, processed_at)
            VALUES (source.encounter_id, source.visit_category, source.confidence, source.clinical_summary, source.processed_at);
        ```

    Environment Variables:
        SQL_SERVER_CONN: Default ODBC connection string
    """

    def __init__(
        self,
        target_table: str,
        merge_keys: list[str],
        column_map: dict[str, Callable[[GlobalState], Any]],
        connection_string: Optional[str] = None,
    ):
        """
        Initialize the SQL Server sink.

        Args:
            target_table: Fully qualified table name (e.g., "dbo.enriched_encounters")
            merge_keys: List of column names that uniquely identify a row
            column_map: Dictionary mapping column names to GlobalState extractors
            connection_string: ODBC connection string (optional, defaults to env var)

        Raises:
            ValueError: If connection_string is None and SQL_SERVER_CONN env var is not set,
                       or if merge_keys is empty or contains columns not in column_map
        """
        self.target_table = target_table
        self.merge_keys = merge_keys
        self.column_map = column_map

        # Validate merge_keys
        if not merge_keys:
            raise ValueError("merge_keys cannot be empty. Provide at least one key column.")

        for key in merge_keys:
            if key not in column_map:
                raise ValueError(
                    f"Merge key '{key}' not found in column_map. "
                    f"All merge keys must have extractors in column_map."
                )

        # Get connection string from parameter or environment
        if connection_string is None:
            connection_string = os.getenv("SQL_SERVER_CONN")
            if connection_string is None:
                raise ValueError(
                    "No connection string provided. Either pass connection_string parameter "
                    "or set SQL_SERVER_CONN environment variable."
                )

        self.connection_string = connection_string

    def _apply_column_map(self, state: GlobalState) -> dict[str, Any]:
        """
        Apply the column_map to extract values from GlobalState.

        Args:
            state: GlobalState object to extract values from

        Returns:
            Dictionary mapping column names to extracted values

        Raises:
            Exception: If any extractor function fails (caller wraps in SinkError)
        """
        result = {}
        for column_name, extractor in self.column_map.items():
            result[column_name] = extractor(state)
        return result

    def _build_merge_sql(self, columns: list[str]) -> str:
        """
        Generate a T-SQL MERGE statement for upserting a row.

        Creates a MERGE statement that:
        1. Matches rows based on merge_keys
        2. Updates non-key columns if matched
        3. Inserts new row if not matched

        Args:
            columns: List of column names in the order they'll be passed as parameters

        Returns:
            T-SQL MERGE statement with ? placeholders for pyodbc parameters

        Example:
            Given:
                target_table = "dbo.enriched_encounters"
                merge_keys = ["encounter_id"]
                columns = ["encounter_id", "visit_category", "confidence"]

            Returns:
                ```sql
                MERGE dbo.enriched_encounters AS target
                USING (SELECT ?, ?, ?) AS source (encounter_id, visit_category, confidence)
                ON target.encounter_id = source.encounter_id
                WHEN MATCHED THEN
                    UPDATE SET
                        visit_category = source.visit_category,
                        confidence = source.confidence
                WHEN NOT MATCHED THEN
                    INSERT (encounter_id, visit_category, confidence)
                    VALUES (source.encounter_id, source.visit_category, source.confidence);
                ```
        """
        # Build SELECT clause with ? placeholders
        placeholders = ", ".join("?" for _ in columns)

        # Build column list
        column_list = ", ".join(columns)

        # Build ON clause (join conditions for merge keys)
        on_conditions = " AND ".join(
            f"target.{key} = source.{key}" for key in self.merge_keys
        )

        # Build UPDATE SET clause (only non-key columns)
        non_key_columns = [col for col in columns if col not in self.merge_keys]
        if non_key_columns:
            update_set = ", ".join(
                f"{col} = source.{col}" for col in non_key_columns
            )
            update_clause = f"""WHEN MATCHED THEN
            UPDATE SET
                {update_set}
        """
        else:
            # If all columns are keys, no UPDATE needed (just INSERT for new rows)
            update_clause = ""

        # Build INSERT clause
        insert_values = ", ".join(f"source.{col}" for col in columns)

        # Assemble MERGE statement
        merge_sql = f"""MERGE {self.target_table} AS target
        USING (SELECT {placeholders}) AS source ({column_list})
        ON {on_conditions}
        {update_clause}WHEN NOT MATCHED THEN
            INSERT ({column_list})
            VALUES ({insert_values});"""

        return merge_sql

    def write(self, state: GlobalState) -> None:
        """
        Write a single GlobalState to SQL Server using MERGE.

        Extracts values using column_map and executes a MERGE statement
        to upsert the row.

        Args:
            state: GlobalState object to write

        Raises:
            SinkError: If extraction fails, connection fails, or MERGE fails
        """
        # Extract values from state
        try:
            row_data = self._apply_column_map(state)
        except Exception as e:
            raise SinkError(
                pk=state.pk,
                original_error=Exception(
                    f"Failed to apply column_map: {type(e).__name__}: {e}"
                ),
            ) from e

        # Ensure columns are in consistent order
        columns = list(row_data.keys())
        values = [row_data[col] for col in columns]

        # Build MERGE SQL
        merge_sql = self._build_merge_sql(columns)

        # Execute MERGE
        try:
            conn = pyodbc.connect(self.connection_string)
            try:
                cursor = conn.cursor()
                cursor.execute(merge_sql, values)
                conn.commit()
            finally:
                conn.close()
        except pyodbc.Error as e:
            raise SinkError(
                pk=state.pk,
                original_error=Exception(
                    f"Failed to execute MERGE for table {self.target_table}: {e}"
                ),
            ) from e
        except Exception as e:
            raise SinkError(pk=state.pk, original_error=e) from e

    def write_batch(self, states: list[GlobalState]) -> None:
        """
        Write multiple GlobalState objects in a single transaction.

        Ensures atomicity: either all rows are written successfully,
        or the transaction is rolled back on any failure.

        Args:
            states: List of GlobalState objects to write

        Raises:
            SinkError: If any write fails, includes the pk of the failed row
        """
        if not states:
            return  # Nothing to write

        conn = None
        failed_pk = None

        try:
            conn = pyodbc.connect(self.connection_string)
            conn.autocommit = False  # Begin transaction

            for state in states:
                failed_pk = state.pk  # Track which row we're processing

                # Extract values from state
                try:
                    row_data = self._apply_column_map(state)
                except Exception as e:
                    raise SinkError(
                        pk=state.pk,
                        original_error=Exception(
                            f"Failed to apply column_map: {type(e).__name__}: {e}"
                        ),
                    ) from e

                # Ensure columns are in consistent order
                columns = list(row_data.keys())
                values = [row_data[col] for col in columns]

                # Build and execute MERGE
                merge_sql = self._build_merge_sql(columns)

                cursor = conn.cursor()
                cursor.execute(merge_sql, values)

            # Commit transaction if all writes succeeded
            conn.commit()

        except SinkError:
            # Re-raise SinkError as-is
            if conn:
                conn.rollback()
            raise
        except pyodbc.Error as e:
            if conn:
                conn.rollback()
            raise SinkError(
                pk=failed_pk or "unknown",
                original_error=Exception(
                    f"Failed to execute batch MERGE for table {self.target_table}: {e}"
                ),
            ) from e
        except Exception as e:
            if conn:
                conn.rollback()
            raise SinkError(
                pk=failed_pk or "unknown",
                original_error=e,
            ) from e
        finally:
            if conn:
                conn.close()
