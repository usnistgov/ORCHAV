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


def test_plain_metadata_keeps_canonical_bytes_and_nonfinite_policy() -> None:
    value = {
        "unicode": "é🚀",
        "zeros": [-0.0, 0.0],
        "optional": (None, True, False),
        "numbers": [2**65, float("nan"), float("inf"), -float("inf")],
        7: {False: "boolean key", "1": "string key", 1: "last key wins"},
    }
    expected = (
        '{"7":{"1":"last key wins","False":"boolean key"},'
        '"numbers":[36893488147419103232,NaN,Infinity,-Infinity],'
        '"optional":[null,true,false],"unicode":"\\u00e9\\ud83d\\ude80",'
        '"zeros":[-0.0,0.0]}'
    )
    assert dumps_frame_json(value) == expected
    assert dumps_frame_json(dict(reversed(list(value.items())))) == expected


def test_numpy_and_complex_metadata_keep_exact_encoded_bytes() -> None:
    value = {
        "array": np.array([1, 9, -2, 9], dtype="<i2")[::2],
        "complex": [complex(-0.0, 2.0), np.complex64(3 - 4j)],
        "scalar": [np.bool_(True), np.int64(7), np.float32(-0.0), np.str_("é")],
        "nonfinite": [np.float64(np.nan), np.float32(np.inf)],
    }
    assert dumps_frame_json(value) == (
        '{"array":{"__orchav_ndarray__":true,"data":"AQD+/w==",'
        '"dtype":"<i2","shape":[2]},'
        '"complex":[{"__orchav_complex__":true,"imag":2.0,"real":-0.0},'
        '{"__orchav_complex__":true,"imag":-4.0,"real":3.0}],'
        '"nonfinite":[NaN,Infinity],"scalar":[true,7,-0.0,"\\u00e9"]}'
    )


def test_builtin_subclasses_and_mapping_views_keep_conversion_behavior() -> None:
    class EqualToEveryType(type):
        def __eq__(cls, _other):
            return True

        __hash__ = type.__hash__

    class DeceptiveSequence(list, metaclass=EqualToEveryType):
        pass

    class Metadata(dict):
        def items(self):
            return [("declared", np.int64(3))]

    class Sequence(list):
        def __iter__(self):
            return iter((np.float32(1.5),))

    class Number(int):
        pass

    class Label(str):
        pass

    class Coordinates(tuple):
        pass

    value = MappingProxyType(
        {
            "mapping": Metadata(ignored=4),
            "metaclass": DeceptiveSequence([np.int64(7)]),
            "sequence": Sequence([99]),
            "values": Coordinates((Number(7), Label("name"))),
        }
    )
    assert dumps_frame_json(value) == (
        '{"mapping":{"declared":3},"metaclass":[7],"sequence":[1.5],"values":[7,"name"]}'
    )


@pytest.mark.parametrize("value", [{"nested": {1, 2}}, [object()], (b"unsupported",)])
def test_fast_containers_still_reject_unsupported_nested_values(value) -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps_frame_json(value)
