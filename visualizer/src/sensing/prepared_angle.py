"""Compact, exact angular projections prepared before Qt presents a result.

The source cube remains authoritative. These arrays retain its displayed maxima
and coordinate grids without retaining the elevation/azimuth/Doppler/range cube.
Display units, color levels, zoom and detection selection remain GUI choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.sensing.angle import select_range_doppler_angle_product

from .angular_spectrum import angular_spectrum_mesh


@dataclass(frozen=True)
class PreparedAngle:
    """Linear display power and exact coordinates for one saved angular product."""

    product_id: str
    variant: str
    shape: tuple[int, int, int, int]
    range_coordinate_semantics: str
    range_axis_m: np.ndarray
    doppler_axis_hz: np.ndarray
    azimuth_axis_deg: np.ndarray
    range_azimuth_power: np.ndarray
    doppler_azimuth_power: np.ndarray
    spectrum_azimuth_deg: np.ndarray | None = None
    spectrum_zenith_deg: np.ndarray | None = None
    spectrum_power: np.ndarray | None = None


def _owned(values: np.ndarray) -> np.ndarray:
    """Retain compact immutable arrays, with no view into a source cube."""
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def prepare_angular(sensing: Mapping[str, Any], variant: str = "primary") -> PreparedAngle | None:
    """Validate once and project the saved product without changing its values.

    Broadside maps mask nonphysical directions before taking maxima, while the
    spherical map retains the existing per-cell NaN mask and warped corners.
    No beamforming, interpolation or new sensing computation is performed.
    """
    angular = select_range_doppler_angle_product(sensing, variant=variant)
    if angular is None:
        return None
    az_visible = np.isfinite(angular.azimuth_broadside_angle_deg)
    power = angular.forward_power_linear[:, az_visible]
    shape = tuple(int(size) for size in angular.power_linear.shape)
    spectrum = (None, None, None)
    if min(shape[:2]) >= 2 and sensing.get("config_resolved", {}).get("antenna_rows") != 1:
        spectrum = tuple(_owned(values) for values in angular_spectrum_mesh(angular))
    return PreparedAngle(
        product_id=angular.product_id,
        variant=angular.variant,
        shape=shape,
        range_coordinate_semantics=angular.range_coordinate_semantics,
        range_axis_m=_owned(angular.range_axis_m),
        doppler_axis_hz=_owned(angular.doppler_axis_hz),
        azimuth_axis_deg=_owned(angular.azimuth_broadside_angle_deg[az_visible]),
        range_azimuth_power=_owned(np.max(power, axis=(0, 2))),
        doppler_azimuth_power=_owned(np.max(power, axis=(0, 3))),
        spectrum_azimuth_deg=spectrum[0],
        spectrum_zenith_deg=spectrum[1],
        spectrum_power=spectrum[2],
    )
