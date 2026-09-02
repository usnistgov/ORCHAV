"""Generate antenna element positions and infer rectangular array dimensions.

Both standalone beamforming and the frame pipeline use these helpers so their
array-shape and coordinate conventions remain identical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_C = 299_792_458.0
_DEFAULT_CARRIER_HZ = 28e9
_VISUAL_PATTERN_ALIASES = {
    "iso": "isotropic",
    "isotropic": "isotropic",
    "dipole": "dipole",
    "hw_dipole": "dipole",
    "tr38901": "tr38901",
}


def wavelength_m_from_frequency_ghz(frequency_ghz: Any, default_ghz: float = 28.0) -> float:
    """Return carrier wavelength in metres for a GHz frequency."""
    try:
        freq_hz = float(frequency_ghz) * 1e9
    except (TypeError, ValueError):
        freq_hz = float(default_ghz) * 1e9
    if freq_hz <= 0.0:
        freq_hz = float(default_ghz) * 1e9
    return _C / freq_hz


def spacing_m_to_wavelengths(spacing_m: Any, frequency_ghz: Any) -> float:
    """Convert element spacing in metres to spacing in wavelengths."""
    wavelength_m = wavelength_m_from_frequency_ghz(frequency_ghz)
    try:
        spacing = float(spacing_m)
    except (TypeError, ValueError):
        spacing = 0.5 * wavelength_m
    if spacing <= 0.0:
        spacing = 0.5 * wavelength_m
    return spacing / wavelength_m


def spacing_wavelengths_to_m(spacing_lambda: Any, frequency_ghz: Any) -> float:
    """Convert element spacing in wavelengths to metres."""
    try:
        spacing = float(spacing_lambda)
    except (TypeError, ValueError):
        spacing = 0.5
    if spacing <= 0.0:
        spacing = 0.5
    return spacing * wavelength_m_from_frequency_ghz(frequency_ghz)


def _mapping_get(mapping_or_obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a dict-like or attribute-style object."""
    if isinstance(mapping_or_obj, dict):
        return mapping_or_obj.get(key, default)
    return getattr(mapping_or_obj, key, default)


def normalize_visual_element_pattern(pattern: Any) -> tuple[str, str | None]:
    """Return a supported visual element pattern plus an optional status note."""
    raw = str(pattern or "isotropic").strip()
    key = raw.lower()
    mapped = _VISUAL_PATTERN_ALIASES.get(key)
    if mapped is None:
        return "isotropic", f"Pattern '{raw}' is shown as isotropic"
    if key == "hw_dipole":
        return mapped, "Pattern 'hw_dipole' is shown with the dipole visual model"
    if key == "iso":
        return mapped, None
    return mapped, None


def beamforming_defaults_from_scenario_config(scenario_config: Any) -> dict[str, Any]:
    """Derive Antennas panel defaults from a loaded scenario configuration.

    YAML antenna spacing is expressed in wavelengths.  The visualizer runtime
    state uses metres while the panel displays wavelength multiples, so this
    helper applies the carrier-frequency conversion while leaving scenario YAML
    unchanged.
    """
    raytracing = _mapping_get(scenario_config, "raytracing", {}) or {}
    carrier_hz = _mapping_get(raytracing, "carrier_frequency_hz", None) or _DEFAULT_CARRIER_HZ
    try:
        carrier_hz = float(carrier_hz)
    except (TypeError, ValueError):
        carrier_hz = _DEFAULT_CARRIER_HZ
    if carrier_hz <= 0.0:
        carrier_hz = _DEFAULT_CARRIER_HZ
    wavelength_m = _C / carrier_hz

    antenna = _mapping_get(raytracing, "antenna", None) or {}
    tx_cfg = _mapping_get(antenna, "tx", None) or {}
    rx_cfg = _mapping_get(antenna, "rx", None) or {}
    array_cfg = tx_cfg or rx_cfg or {}

    def _positive_int(key: str, default: int) -> int:
        """Read a positive antenna-array dimension from scenario config."""
        try:
            return max(1, int(_mapping_get(array_cfg, key, default)))
        except (TypeError, ValueError):
            return default

    def _spacing_m(key: str) -> float:
        """Read wavelength-based YAML spacing and convert it to metres."""
        try:
            spacing_lambda = float(_mapping_get(array_cfg, key, 0.5))
        except (TypeError, ValueError):
            spacing_lambda = 0.5
        if spacing_lambda <= 0.0:
            spacing_lambda = 0.5
        return spacing_lambda * wavelength_m

    tx_pattern, tx_note = normalize_visual_element_pattern(_mapping_get(tx_cfg, "pattern", "iso"))
    rx_pattern, rx_note = normalize_visual_element_pattern(_mapping_get(rx_cfg, "pattern", "iso"))
    notes = [note for note in (tx_note, rx_note) if note]

    return {
        "standalone_antenna_rows": _positive_int("num_rows", 1),
        "standalone_antenna_cols": _positive_int("num_cols", 1),
        "standalone_horizontal_spacing_m": _spacing_m("horizontal_spacing"),
        "standalone_vertical_spacing_m": _spacing_m("vertical_spacing"),
        "standalone_carrier_frequency_ghz": carrier_hz / 1e9,
        "beamforming_element_pattern": tx_pattern,
        "beamforming_tx_element_pattern": tx_pattern,
        "beamforming_rx_element_pattern": rx_pattern,
        "beamforming_pattern_status": "; ".join(notes),
    }


def get_element_positions(
    num_rows: int,
    num_cols: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
) -> np.ndarray:
    """Return element positions for a planar array in the YZ plane.

    The array boresight is along ``+X``.  Columns run along ``Y``
    (horizontal) and rows run along ``Z`` (vertical), centred at the origin.

    Args:
        num_rows: Number of rows (vertical direction).
        num_cols: Number of columns (horizontal direction).
        horizontal_spacing_m: Element spacing along columns in metres.
        vertical_spacing_m: Element spacing along rows in metres.

    Returns:
        ``(num_rows * num_cols, 3)`` float64 array of ``[x, y, z]`` positions.
    """
    cols = np.arange(num_cols, dtype=np.float64) - 0.5 * (num_cols - 1)
    rows = np.arange(num_rows, dtype=np.float64) - 0.5 * (num_rows - 1)
    yy, zz = np.meshgrid(cols, rows)
    positions = np.stack(
        (
            np.zeros_like(yy),
            yy * float(horizontal_spacing_m),
            zz * float(vertical_spacing_m),
        ),
        axis=-1,
    )
    return positions.reshape(-1, 3)


def infer_array_dimensions(num_elements: int) -> tuple[int, int]:
    """Infer ``(rows, cols)`` from element count.

    Known configurations are returned directly; otherwise the function
    searches for the factor pair closest to a square layout.

    Args:
        num_elements: Total number of antenna elements.

    Returns:
        ``(rows, cols)`` tuple with ``rows * cols == num_elements``.
    """
    known: dict[int, tuple[int, int]] = {
        32: (2, 16),
        64: (8, 8),
        128: (8, 16),
        256: (16, 16),
    }
    if num_elements in known:
        return known[num_elements]

    sqrt_n = int(math.sqrt(num_elements))
    for rows in range(sqrt_n, 0, -1):
        if num_elements % rows == 0:
            return (rows, num_elements // rows)
    # Fallback for primes or 0
    return (1, max(num_elements, 1))
