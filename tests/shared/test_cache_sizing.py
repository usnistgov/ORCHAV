"""Tests for conservative retained-payload cache sizing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.cache_sizing import estimate_retained_bytes


def test_shared_array_backing_is_counted_once() -> None:
    backing = np.arange(4096, dtype=np.float32)

    retained = estimate_retained_bytes(backing[:2048], backing[2048:])

    assert retained == backing.nbytes


def test_nested_payloads_and_cycles_are_bounded() -> None:
    payload: dict[str, object] = {
        "array": np.zeros((32, 3), dtype=np.float64),
        "bytes": b"retained-payload",
    }
    payload["cycle"] = payload

    retained = estimate_retained_bytes(payload)

    assert retained >= payload["array"].nbytes + len(payload["bytes"])


def test_opaque_object_graph_is_not_traversed() -> None:
    class EngineObject:
        def __init__(self) -> None:
            self.hidden = np.zeros(1_000_000, dtype=np.float64)

    retained = estimate_retained_bytes(EngineObject())

    assert retained < 1024


def test_slotted_dataclass_arrays_are_counted() -> None:
    @dataclass(slots=True)
    class Payload:
        first: np.ndarray
        second: np.ndarray

    backing = np.zeros(1024, dtype=np.float32)
    payload = Payload(first=backing, second=backing[::2])

    retained = estimate_retained_bytes(payload)

    assert backing.nbytes <= retained < 2 * backing.nbytes
