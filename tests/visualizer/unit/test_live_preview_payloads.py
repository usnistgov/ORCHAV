from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.services.live_preview_payloads import (
    build_live_overrides,
    decode_orientation_array,
    decode_position_array,
    encode_live_preview_arrays,
)


def test_decode_position_array_accepts_flat_and_nested_payloads() -> None:
    flat = decode_position_array([1.0, 2.0, 3.0], key="tx_positions")
    nested = decode_position_array([[1.0, 2.0, 3.0, 4.0]], key="tx_positions")

    np.testing.assert_allclose(flat, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(nested, [[1.0, 2.0, 3.0]])


def test_decode_position_array_rejects_malformed_payloads() -> None:
    with pytest.raises(RuntimeError, match="Invalid rx_positions payload"):
        decode_position_array([[1.0, 2.0]], key="rx_positions")


def test_decode_orientation_array_accepts_flat_and_nested_payloads() -> None:
    flat = decode_orientation_array([0.1, 0.2, 0.3], key="target_orientations")
    nested = decode_orientation_array([[0.4, 0.5, 0.6, 0.7]], key="target_orientations")

    np.testing.assert_allclose(flat, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(nested, [[0.4, 0.5, 0.6]])


def test_encode_position_arrays_normalizes_numpy_payloads() -> None:
    payload = encode_live_preview_arrays(
        np.asarray([[1.0, 2.0, 3.0, 9.0]], dtype=float),
        np.asarray([4.0, 5.0, 6.0], dtype=float),
        np.asarray([[7.0, 8.0, 9.0]], dtype=float),
        np.asarray([[0.1, 0.2, 0.3]], dtype=float),
    )

    assert payload == {
        "tx_positions": [[1.0, 2.0, 3.0]],
        "rx_positions": [[4.0, 5.0, 6.0]],
        "target_positions": [[7.0, 8.0, 9.0]],
        "target_orientations": [[0.1, 0.2, 0.3]],
    }


def test_build_live_overrides_uses_config_names_and_limits_to_positions() -> None:
    simulation = SimpleNamespace(
        tx_configs=[SimpleNamespace(name="tx-main"), SimpleNamespace(name="tx-spare")],
        rx_configs=[SimpleNamespace(name="rx-main")],
        target_configs=[
            SimpleNamespace(name="walker"),
            SimpleNamespace(name="car"),
            SimpleNamespace(name="unused"),
        ],
    )

    overrides = build_live_overrides(
        simulation,
        np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=float),
        np.asarray([[10.0, 11.0, 12.0]], dtype=float),
        np.asarray(
            [[np.pi / 2.0, -np.pi / 4.0, np.pi / 6.0], [np.pi, 0.0, -np.pi / 2.0]],
            dtype=float,
        ),
    )

    assert overrides[:2] == [
        {"name": "tx-main", "category": "tx", "position": (1.0, 2.0, 3.0)},
        {"name": "rx-main", "category": "rx", "position": (4.0, 5.0, 6.0)},
    ]
    assert overrides[2]["name"] == "walker"
    assert overrides[2]["category"] == "target"
    assert overrides[2]["position"] == (10.0, 11.0, 12.0)
    np.testing.assert_allclose(overrides[2]["orientation"], (90.0, -45.0, 30.0))
    assert overrides[3]["name"] == "car"
    assert overrides[3]["category"] == "target"
    np.testing.assert_allclose(overrides[3]["orientation"], (180.0, 0.0, -90.0))
