"""
GlobalState - The core data container for pipeline execution.

Each row from the source becomes a GlobalState object that travels through
the pipeline, accumulating processed results while preserving the original data.
"""

import json
from datetime import datetime
from types import MappingProxyType
from typing import Any, Optional


class GlobalState:
    """
    Immutable container for a single row's journey through the pipeline.

    The raw data is frozen after initialization to prevent accidental mutations.
    Each step adds its output to the `processed` dict using its configured output_key.

    Example:
        >>> state = GlobalState(pk="patient_123", raw={"name": "John", "age": 45})
        >>> state.processed["visit_type"] = {"category": "Emergency", "confidence": 0.92}
        >>> state.log.append("visit_classifier")
        >>> state.raw["age"] = 50  # Raises TypeError - raw is immutable!
    """

    def __init__(
        self,
        pk: str,
        raw: dict[str, Any],
        processed: Optional[dict[str, Any]] = None,
        log: Optional[list[str]] = None,
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ):
        """
        Initialize a new GlobalState instance.

        Args:
            pk: Primary key value for this row (e.g., "encounter_12345")
            raw: Original data from the source (will be made immutable)
            processed: Outputs from steps (defaults to empty dict)
            log: Ordered list of step names that have run (defaults to empty list)
            created_at: Timestamp of creation (defaults to now)
            completed_at: Timestamp when processing finished (defaults to None)
        """
        self.pk = pk
        self._raw = MappingProxyType(raw)  # Immutable view of raw data
        self.processed = processed if processed is not None else {}
        self.log = log if log is not None else []
        self.created_at = created_at if created_at is not None else datetime.now()
        self.completed_at = completed_at

    @property
    def raw(self) -> MappingProxyType[str, Any]:
        """
        Read-only access to original source data.

        Returns:
            Immutable mapping of the raw data

        Raises:
            TypeError: If attempting to modify the returned mapping
        """
        return self._raw

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the complete state to a JSON-serializable dictionary.

        Converts the immutable raw data back to a regular dict for serialization.
        Datetime objects are converted to ISO format strings.

        Returns:
            Dictionary representation of the state

        Example:
            >>> state = GlobalState(pk="enc_123", raw={"note": "Patient presents..."})
            >>> state.processed["summary"] = {"text": "Brief summary"}
            >>> state_dict = state.to_dict()
            >>> state_dict["pk"]
            'enc_123'
            >>> state_dict["raw"]["note"]
            'Patient presents...'
        """
        return {
            "pk": self.pk,
            "raw": dict(self._raw),  # Convert MappingProxyType back to dict
            "processed": self.processed,
            "log": self.log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_json(self) -> str:
        """
        Serialize the complete state to a JSON string.

        Returns:
            JSON string representation of the state

        Example:
            >>> state = GlobalState(pk="enc_123", raw={"note": "text"})
            >>> json_str = state.to_json()
            >>> "enc_123" in json_str
            True
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalState":
        """
        Deserialize a GlobalState from a dictionary.

        Reconstructs a GlobalState instance from the output of to_dict().
        Datetime strings are parsed back to datetime objects.

        Args:
            data: Dictionary containing state data (from to_dict())

        Returns:
            Reconstructed GlobalState instance

        Example:
            >>> original = GlobalState(pk="enc_123", raw={"note": "text"})
            >>> state_dict = original.to_dict()
            >>> restored = GlobalState.from_dict(state_dict)
            >>> restored.pk == original.pk
            True
            >>> restored.raw["note"] == original.raw["note"]
            True
        """
        # Parse datetime strings back to datetime objects
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            pk=data["pk"],
            raw=data["raw"],
            processed=data.get("processed", {}),
            log=data.get("log", []),
            created_at=created_at,
            completed_at=completed_at,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"GlobalState(pk={self.pk!r}, "
            f"raw_keys={list(self._raw.keys())}, "
            f"processed_keys={list(self.processed.keys())}, "
            f"log={self.log})"
        )
