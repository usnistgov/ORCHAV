"""Utility helpers shared across visualizer modules.

The package re-exports common color, transform, and validation helpers. More
specialized antenna, aperture, and trajectory-color helpers are imported from
their modules directly so call sites keep the domain boundary visible.
"""

from .colors import (
    clear_categorical_palette_cache,
    clear_continuous_lut_cache,
    ensure_continuous_lut,
    ensure_good_bad_lut,
    ensure_viridis_lut,
    get_categorical_order_palette,
    get_categorical_type_palette,
    map_values_to_lut,
)
from .geometry import (
    build_rotation_matrix_from_angles,
    build_transform_matrix,
    compute_position_after_scale_rotation,
)
from .validation import (
    AVAILABLE_COLOR_MODES,
    DEFAULT_COLOR_MODE,
    MIN_STEP,
    validate_color_mode,
    validate_step,
)

__all__ = [
    "build_rotation_matrix_from_angles",
    "build_transform_matrix",
    "compute_position_after_scale_rotation",
    "clear_categorical_palette_cache",
    "clear_continuous_lut_cache",
    "ensure_continuous_lut",
    "ensure_good_bad_lut",
    "ensure_viridis_lut",
    "get_categorical_order_palette",
    "get_categorical_type_palette",
    "map_values_to_lut",
    "AVAILABLE_COLOR_MODES",
    "DEFAULT_COLOR_MODE",
    "MIN_STEP",
    "validate_color_mode",
    "validate_step",
]
