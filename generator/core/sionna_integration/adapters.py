"""Sionna/Mitsuba assignment-boundary conversion helpers.

Inside the generator, orientation policies and metadata use yaw/pitch/roll in
degrees.  Sionna object assignment expects Mitsuba ``Point3f`` values in
radians.  These helpers keep that conversion at the edge where values are
written to Sionna objects, rather than leaking engine units through mobility,
orientation, or metadata code.
"""

from __future__ import annotations

import math
from typing import Any, TypeAlias

import mitsuba as mi

from ..orientation.base import Orientation3, orientation_to_tuple
from ..utils import point_to_tuple

EngineOrientation3: TypeAlias = tuple[float, float, float]


def orientation_deg_to_engine_radians(orientation: Orientation3) -> EngineOrientation3:
    """Convert project-facing ``(yaw, pitch, roll)`` degrees to engine radians."""
    yaw, pitch, roll = orientation_to_tuple(orientation)
    return (
        math.radians(yaw),
        math.radians(pitch),
        math.radians(roll),
    )


def orientation_to_point3f(orientation: Orientation3) -> mi.Point3f:
    """Convert yaw/pitch/roll degrees to the Sionna assignment value."""
    return mi.Point3f(*orientation_deg_to_engine_radians(orientation))


def orientation_to_point3f_with_engine_radians(
    orientation: Orientation3,
) -> tuple[mi.Point3f, EngineOrientation3]:
    """Return both the Sionna assignment value and radians for diagnostics."""
    engine_orientation = orientation_deg_to_engine_radians(orientation)
    return mi.Point3f(*engine_orientation), engine_orientation


def point3f(value: Any) -> mi.Point3f:
    """Convert a point-like value to a Sionna ``Point3f`` at assignment boundaries."""
    point3f_type = mi.Point3f
    try:
        if isinstance(value, point3f_type):
            return value
    except TypeError:
        # Unit tests and light-weight import checks may replace Point3f with a
        # simple callable.  In that case, skip isinstance and use construction.
        pass
    return point3f_type(*point_to_tuple(value, error_type=ValueError))
