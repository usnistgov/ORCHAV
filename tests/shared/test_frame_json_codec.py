"""Tests for NumPy-safe frame metadata JSON."""

from __future__ import annotations

import base64
import json
from types import MappingProxyType

import numpy as np
import pytest

from shared.frames.json_codec import dumps_frame_json, loads_frame_json


def test_frame_json_round_trip_preserves_nested_numpy_values() -> None:
    value = MappingProxyType(
        {
            "scalar": np.float32(1.25),
            "complex": np.complex64(2.0 + 3.0j),
            "array": np.arange(6, dtype=np.int16).reshape(2, 3),
            "nested": [MappingProxyType({"array": np.array([1.0 + 2.0j], dtype=np.complex64)})],
        }
    )

    restored = loads_frame_json(dumps_frame_json(value))

    assert restored["scalar"] == pytest.approx(1.25)
    assert restored["complex"] == pytest.approx(2.0 + 3.0j)
    np.testing.assert_array_equal(restored["array"], value["array"])
    np.testing.assert_array_equal(restored["nested"][0]["array"], value["nested"][0]["array"])


def test_frame_json_rejects_mismatched_array_payload_size() -> None:
    payload = {
        "__orchav_ndarray__": True,
        "dtype": np.dtype(np.float32).str,
        "shape": [2],
        "data": base64.b64encode(np.array([1.0], dtype=np.float32).tobytes()).decode("ascii"),
    }

    with pytest.raises(ValueError, match="byte count"):
        loads_frame_json(json.dumps(payload))


def test_frame_json_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps_frame_json(object())
