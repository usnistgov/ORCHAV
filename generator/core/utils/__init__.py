"""Small normalization helpers shared by generator core packages.

This package is the narrow place where runtime values from scenario YAML,
Sionna/Mitsuba assignments, NumPy arrays, tensor-like objects, and internal
point structs are converted into plain Python values.

Start with ``point_to_tuple`` when code needs a strict ``(x, y, z)`` contract,
``to_numpy``/``to_float`` when code needs tensor output in Python/NumPy form,
and the angle helpers when orientation or comparison code needs the shared
degree convention.
"""

from .angle_utils import angle_difference_deg, wrap_angle_deg, wrap_angles_deg
from .geometry_utils import point_to_tuple
from .tensor_utils import to_float, to_numpy, to_scalar

__all__ = [
    "point_to_tuple",
    "to_numpy",
    "to_scalar",
    "to_float",
    "wrap_angle_deg",
    "wrap_angles_deg",
    "angle_difference_deg",
]
