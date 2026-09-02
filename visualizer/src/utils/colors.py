"""Color utilities shared across visualization modules."""

from __future__ import annotations

from typing import Optional

import numpy as np

_VIRIDIS256: Optional[np.ndarray] = None
_GOOD_BAD256: Optional[np.ndarray] = None
_CONTINUOUS_LUT: Optional[np.ndarray] = None
_CONTINUOUS_LUT_NAME: Optional[str] = None


def ensure_viridis_lut() -> np.ndarray:
    """Return (and cache) the 256x3 Viridis color map."""
    global _VIRIDIS256
    if _VIRIDIS256 is not None:
        return _VIRIDIS256
    try:
        import matplotlib as mpl

        # Use newer API to avoid deprecation warning (matplotlib 3.7+)
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap("viridis")
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap("viridis")

        _VIRIDIS256 = cmap(np.linspace(0, 1, 256)).astype(np.float32)[:, :3]
    except (ImportError, ValueError, AttributeError):
        # Preserve viridis semantics even in lightweight installations that do
        # not include matplotlib. These stops are sampled from viridis and
        # linearly interpolated into the same canonical 256-entry LUT.
        stops = np.asarray(
            [
                (0.000, 68, 1, 84),
                (0.143, 72, 35, 116),
                (0.286, 64, 67, 135),
                (0.429, 52, 94, 141),
                (0.571, 33, 145, 140),
                (0.714, 53, 183, 121),
                (0.857, 143, 215, 68),
                (1.000, 253, 231, 37),
            ],
            dtype=np.float32,
        )
        sample_positions = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        _VIRIDIS256 = np.column_stack(
            [np.interp(sample_positions, stops[:, 0], stops[:, channel]) for channel in range(1, 4)]
        ).astype(np.float32)
        _VIRIDIS256 /= 255.0
    return _VIRIDIS256


def ensure_continuous_lut() -> np.ndarray:
    """Return (and cache) a 256x3 LUT using the theme's continuous colormap.

    The LUT is regenerated if the theme's colormap changes.
    """
    global _CONTINUOUS_LUT, _CONTINUOUS_LUT_NAME

    # Get current colormap from theme
    try:
        from shared.statistics.themes import theme_manager

        current_cmap = theme_manager.current.continuous_colormap
    except (ImportError, AttributeError):
        current_cmap = "viridis"  # Fallback

    # Regenerate if colormap changed
    if _CONTINUOUS_LUT is not None and _CONTINUOUS_LUT_NAME == current_cmap:
        return _CONTINUOUS_LUT

    try:
        import matplotlib as mpl

        # Use newer API to avoid deprecation warning (matplotlib 3.7+)
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(current_cmap)
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap(current_cmap)

        _CONTINUOUS_LUT = cmap(np.linspace(0, 1, 256)).astype(np.float32)[:, :3]
        _CONTINUOUS_LUT_NAME = current_cmap
    except (ValueError, AttributeError):
        # Fallback to grayscale
        g = np.linspace(0, 1, 256, dtype=np.float32)
        _CONTINUOUS_LUT = np.stack([g, g, g], axis=1)
        _CONTINUOUS_LUT_NAME = "grayscale"

    return _CONTINUOUS_LUT


def clear_continuous_lut_cache() -> None:
    """Clear the continuous LUT cache to force regeneration on next use."""
    global _CONTINUOUS_LUT, _CONTINUOUS_LUT_NAME
    _CONTINUOUS_LUT = None
    _CONTINUOUS_LUT_NAME = None


def ensure_good_bad_lut() -> np.ndarray:
    """Return a 256x3 LUT from good (green) to bad (red)."""
    global _GOOD_BAD256
    if _GOOD_BAD256 is not None:
        return _GOOD_BAD256
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.float32)
    mid = 0.5
    h1 = x <= mid
    t1 = np.zeros_like(x)
    t1[h1] = x[h1] / mid
    lut[h1, 0] = t1[h1]
    lut[h1, 1] = 1.0
    lut[h1, 2] = 0.0
    h2 = x > mid
    t2 = np.zeros_like(x)
    t2[h2] = (x[h2] - mid) / (1.0 - mid)
    lut[h2, 0] = 1.0
    lut[h2, 1] = 1.0 - t2[h2]
    lut[h2, 2] = 0.0
    _GOOD_BAD256 = lut
    return _GOOD_BAD256


def map_values_to_lut(
    values: np.ndarray, vmin: float, vmax: float, lut256: np.ndarray
) -> np.ndarray:
    """Map scalar values to RGB colors using the given LUT."""
    denom = vmax - vmin
    if denom <= 1e-12:
        idx = np.zeros(values.shape[0], dtype=np.uint8)
    else:
        norm = (values - vmin) / denom
        # Replace NaN with 0.0 before uint8 cast to avoid RuntimeWarning
        np.nan_to_num(norm, copy=False, nan=0.0)
        idx = np.clip((norm * 255.0).astype(np.uint8), 0, 255)
    return lut256[idx]


# Cache for categorical palettes
_CATEGORICAL_ORDER_PALETTE: Optional[np.ndarray] = None
_CATEGORICAL_ORDER_CMAP: Optional[str] = None
_CATEGORICAL_TYPE_PALETTE: Optional[np.ndarray] = None
_CATEGORICAL_TYPE_CMAP: Optional[str] = None


def get_categorical_order_palette(n_colors: int = 7) -> np.ndarray:
    """Get a palette for reflection orders from the theme's categorical colormap.

    Returns (n_colors, 3) float32 array of RGB colors.
    If categorical_colormap is "theme_default", returns None to signal
    that the caller should use explicit theme colors instead.
    """
    global _CATEGORICAL_ORDER_PALETTE, _CATEGORICAL_ORDER_CMAP

    try:
        from shared.statistics.themes import theme_manager

        current_cmap = theme_manager.current.categorical_colormap
    except (ImportError, AttributeError):
        current_cmap = "tab10"

    # Special value meaning "use theme's explicit colors"
    if current_cmap == "theme_default":
        return None

    # Return cached if unchanged
    if (
        _CATEGORICAL_ORDER_PALETTE is not None
        and _CATEGORICAL_ORDER_CMAP == current_cmap
        and len(_CATEGORICAL_ORDER_PALETTE) == n_colors
    ):
        return _CATEGORICAL_ORDER_PALETTE

    try:
        import matplotlib as mpl

        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(current_cmap)
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap(current_cmap)

        # Generate n_colors evenly spaced from the colormap
        _CATEGORICAL_ORDER_PALETTE = cmap(np.linspace(0, 1, n_colors)).astype(np.float32)[:, :3]
        _CATEGORICAL_ORDER_CMAP = current_cmap
    except (ValueError, AttributeError):
        # Fallback to grayscale
        g = np.linspace(0.2, 0.8, n_colors, dtype=np.float32)
        _CATEGORICAL_ORDER_PALETTE = np.stack([g, g, g], axis=1)
        _CATEGORICAL_ORDER_CMAP = "grayscale"

    return _CATEGORICAL_ORDER_PALETTE


def get_categorical_type_palette(n_colors: int = 9) -> np.ndarray:
    """Get a palette for MPC types from the theme's categorical colormap.

    Returns (n_colors, 3) float32 array of RGB colors.
    If categorical_colormap is "theme_default", returns None to signal
    that the caller should use explicit theme colors instead.
    """
    global _CATEGORICAL_TYPE_PALETTE, _CATEGORICAL_TYPE_CMAP

    try:
        from shared.statistics.themes import theme_manager

        current_cmap = theme_manager.current.categorical_colormap
    except (ImportError, AttributeError):
        current_cmap = "tab10"

    # Special value meaning "use theme's explicit colors"
    if current_cmap == "theme_default":
        return None

    # Return cached if unchanged
    if (
        _CATEGORICAL_TYPE_PALETTE is not None
        and _CATEGORICAL_TYPE_CMAP == current_cmap
        and len(_CATEGORICAL_TYPE_PALETTE) == n_colors
    ):
        return _CATEGORICAL_TYPE_PALETTE

    try:
        import matplotlib as mpl

        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(current_cmap)
        else:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap(current_cmap)

        # Generate n_colors evenly spaced from the colormap
        _CATEGORICAL_TYPE_PALETTE = cmap(np.linspace(0, 1, n_colors)).astype(np.float32)[:, :3]
        _CATEGORICAL_TYPE_CMAP = current_cmap
    except (ValueError, AttributeError):
        # Fallback to grayscale
        g = np.linspace(0.2, 0.8, n_colors, dtype=np.float32)
        _CATEGORICAL_TYPE_PALETTE = np.stack([g, g, g], axis=1)
        _CATEGORICAL_TYPE_CMAP = "grayscale"

    return _CATEGORICAL_TYPE_PALETTE


def clear_categorical_palette_cache() -> None:
    """Clear the categorical palette caches to force regeneration."""
    global _CATEGORICAL_ORDER_PALETTE, _CATEGORICAL_ORDER_CMAP
    global _CATEGORICAL_TYPE_PALETTE, _CATEGORICAL_TYPE_CMAP
    _CATEGORICAL_ORDER_PALETTE = None
    _CATEGORICAL_ORDER_CMAP = None
    _CATEGORICAL_TYPE_PALETTE = None
    _CATEGORICAL_TYPE_CMAP = None


# Cache for distinct material palette
_DISTINCT_MATERIAL_PALETTE: Optional[np.ndarray] = None
_DISTINCT_MATERIAL_N: int = 0


def get_distinct_material_palette(n_materials: int) -> np.ndarray:
    """Generate N maximally distinct colors for material identification.

    Uses tab20 (<=20), tab20+tab20b (<=40), or HSV spacing (>40).

    Args:
        n_materials: Number of distinct colors to generate.

    Returns:
        (n_materials, 3) float32 array of RGB colors.
    """
    global _DISTINCT_MATERIAL_PALETTE, _DISTINCT_MATERIAL_N
    if _DISTINCT_MATERIAL_PALETTE is not None and _DISTINCT_MATERIAL_N == n_materials:
        return _DISTINCT_MATERIAL_PALETTE

    try:
        import matplotlib as mpl

        if n_materials <= 20:
            if hasattr(mpl, "colormaps"):
                cmap = mpl.colormaps.get_cmap("tab20")
            else:
                import matplotlib.pyplot as plt

                cmap = plt.cm.get_cmap("tab20")
            palette = cmap(np.linspace(0, 1, max(n_materials, 2))).astype(np.float32)[
                :n_materials, :3
            ]
        elif n_materials <= 40:
            if hasattr(mpl, "colormaps"):
                t20 = mpl.colormaps.get_cmap("tab20")
                t20b = mpl.colormaps.get_cmap("tab20b")
            else:
                import matplotlib.pyplot as plt

                t20 = plt.cm.get_cmap("tab20")
                t20b = plt.cm.get_cmap("tab20b")
            p1 = t20(np.linspace(0, 1, 20)).astype(np.float32)[:, :3]
            p2 = t20b(np.linspace(0, 1, 20)).astype(np.float32)[:, :3]
            palette = np.vstack([p1, p2])[:n_materials]
        else:
            # HSV spacing for large counts
            hues = np.linspace(0, 1, n_materials, endpoint=False, dtype=np.float32)
            palette = np.zeros((n_materials, 3), dtype=np.float32)
            for i, h in enumerate(hues):
                # Convert HSV (h, 0.9, 0.9) to RGB
                import colorsys

                r, g, b = colorsys.hsv_to_rgb(float(h), 0.9, 0.9)
                palette[i] = [r, g, b]
    except (ValueError, AttributeError, ImportError):
        # Fallback: evenly spaced hues via simple math
        hues = np.linspace(0, 1, n_materials, endpoint=False, dtype=np.float32)
        palette = np.zeros((n_materials, 3), dtype=np.float32)
        for i, h in enumerate(hues):
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(float(h), 0.9, 0.9)
            palette[i] = [r, g, b]

    _DISTINCT_MATERIAL_PALETTE = palette
    _DISTINCT_MATERIAL_N = n_materials
    return _DISTINCT_MATERIAL_PALETTE


def clear_distinct_material_palette_cache() -> None:
    """Clear the distinct material palette cache."""
    global _DISTINCT_MATERIAL_PALETTE, _DISTINCT_MATERIAL_N
    _DISTINCT_MATERIAL_PALETTE = None
    _DISTINCT_MATERIAL_N = 0
