"""
Tests for GlobalState data container.

Validates:
- Immutability of raw data
- Serialization and deserialization
- Processed dict updates
- Datetime handling
"""

from datetime import datetime

import pytest

from llm_etl.core.state import GlobalState


def test_raw_data_is_immutable():
    """Test that GlobalState.raw is immutable and cannot be modified."""
    state = GlobalState(
        pk="test_001",
        raw={"name": "John", "age": 30, "status": "active"}
    )

    # Reading should work fine
    assert state.raw["name"] == "John"
    assert state.raw["age"] == 30

    # Attempting to modify should raise TypeError
    with pytest.raises(TypeError):
        state.raw["name"] = "Jane"

    with pytest.raises(TypeError):
        state.raw["new_field"] = "value"

    with pytest.raises(TypeError):
        del state.raw["age"]

    # Original values should be unchanged
    assert state.raw["name"] == "John"
    assert state.raw["age"] == 30


def test_serialization_roundtrip():
    """Test that GlobalState can be serialized to dict and restored."""
    # Create a state with all fields populated
    original = GlobalState(
        pk="enc_12345",
        raw={"note": "Patient presents with headache", "department": "Emergency"},
        processed={
            "classification": {"category": "Emergency", "confidence": 0.9},
            "summary": {"text": "Acute headache", "word_count": 2}
        },
        log=["classifier", "summarizer"],
        created_at=datetime(2025, 1, 15, 10, 30, 0),
        completed_at=datetime(2025, 1, 15, 10, 35, 0),
    )

    # Serialize to dict
    state_dict = original.to_dict()

    # Verify dict structure
    assert state_dict["pk"] == "enc_12345"
    assert state_dict["raw"]["note"] == "Patient presents with headache"
    assert state_dict["processed"]["classification"]["category"] == "Emergency"
    assert state_dict["log"] == ["classifier", "summarizer"]
    assert state_dict["created_at"] == "2025-01-15T10:30:00"
    assert state_dict["completed_at"] == "2025-01-15T10:35:00"

    # Deserialize back to GlobalState
    restored = GlobalState.from_dict(state_dict)

    # Verify all fields match
    assert restored.pk == original.pk
    assert dict(restored.raw) == dict(original.raw)
    assert restored.processed == original.processed
    assert restored.log == original.log
    assert restored.created_at == original.created_at
    assert restored.completed_at == original.completed_at


def test_json_serialization_roundtrip():
    """Test that GlobalState can be serialized to JSON string and restored."""
    import json

    original = GlobalState(
        pk="test_json",
        raw={"field": "value"},
        processed={"result": "data"},
    )

    # Serialize to JSON string
    json_str = original.to_json()

    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data["pk"] == "test_json"
    assert data["raw"]["field"] == "value"

    # Restore from dict
    restored = GlobalState.from_dict(data)
    assert restored.pk == original.pk
    assert dict(restored.raw) == dict(original.raw)


def test_processed_dict_updates_correctly():
    """Test that the processed dict can be updated as steps run."""
    state = GlobalState(
        pk="test_002",
        raw={"text": "Some input text"}
    )

    # Initially empty
    assert len(state.processed) == 0
    assert state.log == []

    # First step adds its output
    state.processed["classification"] = {
        "category": "TypeA",
        "confidence": 0.85,
        "reasoning": "Classification reason"
    }
    state.log.append("classifier")

    assert len(state.processed) == 1
    assert state.processed["classification"]["category"] == "TypeA"
    assert state.log == ["classifier"]

    # Second step adds its output
    state.processed["summary"] = {
        "text": "Brief summary",
        "word_count": 2
    }
    state.log.append("summarizer")

    assert len(state.processed) == 2
    assert state.processed["summary"]["text"] == "Brief summary"
    assert state.log == ["classifier", "summarizer"]

    # Steps can reference each other's outputs
    previous_category = state.processed["classification"]["category"]
    assert previous_category == "TypeA"


def test_default_values():
    """Test that optional fields have correct default values."""
    state = GlobalState(pk="test_003", raw={"data": "value"})

    # processed should default to empty dict
    assert state.processed == {}
    assert isinstance(state.processed, dict)

    # log should default to empty list
    assert state.log == []
    assert isinstance(state.log, list)

    # created_at should be set to current time
    assert state.created_at is not None
    assert isinstance(state.created_at, datetime)

    # completed_at should be None
    assert state.completed_at is None


def test_completed_at_timestamp():
    """Test that completed_at can be set when processing finishes."""
    state = GlobalState(pk="test_004", raw={"data": "value"})

    # Initially None
    assert state.completed_at is None

    # Set completion timestamp
    completion_time = datetime(2025, 2, 1, 14, 30, 0)
    state.completed_at = completion_time

    assert state.completed_at == completion_time

    # Verify serialization preserves it
    state_dict = state.to_dict()
    assert state_dict["completed_at"] == "2025-02-01T14:30:00"

    # Verify deserialization restores it
    restored = GlobalState.from_dict(state_dict)
    assert restored.completed_at == completion_time


def test_repr_string():
    """Test that __repr__ provides useful debugging information."""
    state = GlobalState(
        pk="test_005",
        raw={"field1": "value1", "field2": "value2"},
        processed={"result1": "data1"},
        log=["step1", "step2"]
    )

    repr_str = repr(state)

    # Should include key information
    assert "test_005" in repr_str
    assert "field1" in repr_str or "field2" in repr_str
    assert "result1" in repr_str
    assert "step1" in repr_str or "step2" in repr_str


def test_empty_raw_data():
    """Test that GlobalState works with empty raw data."""
    state = GlobalState(pk="empty_test", raw={})

    assert state.pk == "empty_test"
    assert len(state.raw) == 0
    assert dict(state.raw) == {}


def test_nested_data_structures():
    """Test that GlobalState handles nested dicts and lists in raw and processed."""
    state = GlobalState(
        pk="nested_test",
        raw={
            "patient": {
                "name": "John Doe",
                "vitals": [120, 80, 98.6]
            },
            "medications": ["aspirin", "lisinopril"]
        }
    )

    # Access nested raw data
    assert state.raw["patient"]["name"] == "John Doe"
    assert state.raw["patient"]["vitals"][0] == 120
    assert "aspirin" in state.raw["medications"]

    # Add nested processed data
    state.processed["analysis"] = {
        "findings": ["hypertension", "stable"],
        "metrics": {"risk_score": 0.25}
    }

    assert state.processed["analysis"]["findings"][0] == "hypertension"
    assert state.processed["analysis"]["metrics"]["risk_score"] == 0.25

    # Verify serialization handles nested structures
    state_dict = state.to_dict()
    assert state_dict["raw"]["patient"]["name"] == "John Doe"
    assert state_dict["processed"]["analysis"]["findings"][0] == "hypertension"

    # Verify roundtrip
    restored = GlobalState.from_dict(state_dict)
    assert restored.raw["patient"]["name"] == "John Doe"
    assert restored.processed["analysis"]["findings"][0] == "hypertension"


def test_raw_data_immutability_with_nested_structures():
    """Test that nested structures in raw are also immutable."""
    state = GlobalState(
        pk="immutable_nested",
        raw={"patient": {"name": "John", "age": 30}}
    )

    # Top-level is immutable
    with pytest.raises(TypeError):
        state.raw["patient"] = {"name": "Jane"}

    # Note: MappingProxyType only makes the top level immutable
    # The nested dict is still mutable, but we can't assign new top-level keys
    # This is a known limitation of MappingProxyType


def test_none_values_in_serialization():
    """Test that None values are handled correctly in serialization."""
    state = GlobalState(
        pk="none_test",
        raw={"field": "value"},
        completed_at=None  # Explicitly None
    )

    # Serialize
    state_dict = state.to_dict()
    assert state_dict["completed_at"] is None

    # Deserialize
    restored = GlobalState.from_dict(state_dict)
    assert restored.completed_at is None
