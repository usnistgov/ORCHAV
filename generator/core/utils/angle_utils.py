"""Degree-based angle normalization used by orientation and comparison tools.

The generator stores yaw/azimuth-facing angles in degrees at the package
boundary. These helpers keep the shared convention in one place: normalized
angles live in the half-open range ``[-180, 180)``, so the branch cut has a
single representation and ``180`` normalizes to ``-180``.
"""

from typing import cast

import numpy as np


def wrap_angle_deg(angle: float) -> float:
    """Wrap one degree value to the canonical ``[-180, 180)`` range.

    Args:
        angle: Angle in degrees.

    Returns:
        Wrapped angle in [-180, 180) degrees.

    Examples:
        >>> wrap_angle_deg(270.0)
        -90.0
        >>> wrap_angle_deg(-270.0)
        90.0
    """
    return (angle + 180.0) % 360.0 - 180.0


def wrap_angles_deg(angles: np.ndarray) -> np.ndarray:
    """Vectorize :func:`wrap_angle_deg` for NumPy angle arrays.

    Empty arrays are returned unchanged so callers that use an empty array as a
    no-data sentinel do not lose object identity.

    Args:
        angles: NumPy array of angles in degrees.

    Returns:
        NumPy array with angles wrapped to [-180, 180) degrees.

    Examples:
        >>> wrap_angles_deg(np.array([270.0, -270.0, 0.0]))
        array([-90.,  90.,   0.])
    """
    if angles.size == 0:
        return angles
    return cast(np.ndarray, (angles + 180.0) % 360.0 - 180.0)


def angle_difference_deg(a_from: float, a_to: float) -> float:
    """Compute the shortest signed rotation from ``a_from`` to ``a_to``.

    The result is wrapped to [-180, 180) degrees and represents the smallest
    rotation needed to go from ``a_from`` to ``a_to``. At the 180-degree branch
    cut, the shared half-open convention returns ``-180`` rather than ``180``.

    Args:
        a_from: Starting angle in degrees.
        a_to: Target angle in degrees.

    Returns:
        Signed angular difference in [-180, 180) degrees.

    Examples:
        >>> angle_difference_deg(10.0, 350.0)
        -20.0
        >>> angle_difference_deg(350.0, 10.0)
        20.0
    """
    diff = (a_to - a_from + 180.0) % 360.0 - 180.0
    return diff
