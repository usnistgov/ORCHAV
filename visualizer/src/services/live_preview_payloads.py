"""Shared payload helpers for interactive live-preview worker requests."""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_position_array(values: Any, *, key: str) -> np.ndarray:
    """Decode a request position field into a copied ``(N, 3)`` array."""
    arr = np.asarray([] if values is None else values, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1 and arr.size >= 3:
        arr = arr.reshape((1, arr.size))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Invalid {key} payload")
    return np.array(arr[:, :3], dtype=np.float64, copy=True)


def decode_orientation_array(values: Any, *, key: str) -> np.ndarray:
    """Decode a request orientation field into a copied ``(N, 3)`` array."""
    arr = np.asarray([] if values is None else values, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1 and arr.size >= 3:
        arr = arr.reshape((1, arr.size))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Invalid {key} payload")
    return np.array(arr[:, :3], dtype=np.float64, copy=True)


def encode_live_preview_arrays(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    target_positions: np.ndarray | None = None,
    target_orientations: np.ndarray | None = None,
) -> dict[str, list[list[float]]]:
    """Encode editable actor arrays for the JSON-line worker protocol."""
    tx = decode_position_array(tx_positions, key="tx_positions")
    rx = decode_position_array(rx_positions, key="rx_positions")
    payload = {
        "tx_positions": tx.tolist(),
        "rx_positions": rx.tolist(),
    }
    if target_positions is not None:
        payload["target_positions"] = decode_position_array(
            target_positions,
            key="target_positions",
        ).tolist()
    if target_orientations is not None:
        payload["target_orientations"] = decode_orientation_array(
            target_orientations,
            key="target_orientations",
        ).tolist()
    return payload


def build_live_overrides(
    simulation: Any,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    target_positions: np.ndarray | None = None,
    target_orientations: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Translate edited actor poses into generator live overrides."""
    overrides: list[dict[str, Any]] = []
    for kind, configs, positions in (
        ("tx", getattr(simulation, "tx_configs", []), tx_positions),
        ("rx", getattr(simulation, "rx_configs", []), rx_positions),
    ):
        valid_positions = decode_position_array(positions, key=f"{kind}_positions")
        limit = min(len(configs), len(valid_positions))
        for index in range(limit):
            cfg = configs[index]
            overrides.append(
                {
                    "name": str(getattr(cfg, "name", f"{kind}_{index + 1}")),
                    "category": kind,
                    "position": tuple(float(x) for x in valid_positions[index, :3]),
                }
            )

    target_configs = getattr(simulation, "target_configs", [])
    valid_target_positions = decode_position_array(
        target_positions,
        key="target_positions",
    )
    valid_target_orientations = decode_orientation_array(
        target_orientations,
        key="target_orientations",
    )
    target_limit = min(
        len(target_configs),
        max(len(valid_target_positions), len(valid_target_orientations)),
    )
    for index in range(target_limit):
        cfg = target_configs[index]
        override: dict[str, Any] = {
            "name": str(getattr(cfg, "name", f"target_{index + 1}")),
            "category": "target",
        }
        if index < len(valid_target_positions):
            override["position"] = tuple(float(x) for x in valid_target_positions[index, :3])
        if index < len(valid_target_orientations):
            # Visualizer target state stores Sionna radians; generator live overrides use degrees.
            orientation_degrees = np.degrees(valid_target_orientations[index, :3])
            override["orientation"] = tuple(float(x) for x in orientation_degrees)
        if "position" in override or "orientation" in override:
            overrides.append(override)
    return overrides
