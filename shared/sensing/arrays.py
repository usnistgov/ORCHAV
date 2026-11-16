"""Explicit planar-array ordering and spatial-FFT helpers for sensing.

Sionna ``PlanarArray`` elements are ordered column first, with rows cycling
from top to bottom inside each column.  The helpers in this module keep that
ordering, the FFT sign, and the physical-direction mapping explicit so sensing
code never relies on an anonymous reshape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

PLANAR_ELEMENT_ORDER = "sionna_column_major_top_to_bottom"
SPATIAL_FFT_NORMALIZATION = "unitary_ortho"
SPATIAL_PHASE_CONVENTION = "positive_phase_toward_positive_y_and_z"


@dataclass(frozen=True)
class PlanarSpatialSpectrum:
    """Unitary spatial-FFT result with physical direction-cosine axes.

    ``spectrum`` has axes ``(elevation_direction, azimuth_direction, ...)``.
    Horizontal direction cosine is positive toward Sionna's local ``+y``;
    vertical direction cosine is positive toward local ``+z``.  Invalid DFT
    bins outside the visible region are represented by NaN in the optional
    broadside angle axes rather than being clipped to repeated +/-90 degree
    values.
    """

    spectrum: np.ndarray
    azimuth_direction_cosine: np.ndarray
    elevation_direction_cosine: np.ndarray
    broadside_azimuth_deg: np.ndarray
    broadside_elevation_deg: np.ndarray
    grating_lobes_possible: bool


def planar_element_coordinates(num_rows: int, num_cols: int) -> tuple[tuple[int, int], ...]:
    """Return ``(row, column)`` coordinates in Sionna element-axis order."""

    rows = int(num_rows)
    cols = int(num_cols)
    if rows < 1 or cols < 1:
        raise ValueError("Planar array row and column counts must be positive")
    return tuple((row, col) for col in range(cols) for row in range(rows))


def direction_cosine_axis(size: int, spacing_wavelengths: float) -> np.ndarray:
    """Return the shifted spatial-FFT direction-cosine coordinate.

    A steering sequence ``exp(+j 2 pi d n u)`` peaks at coordinate ``u`` under
    NumPy's negative-exponent FFT convention.  ``d`` is element spacing in
    wavelengths.
    """

    count = int(size)
    spacing = float(spacing_wavelengths)
    if count < 1:
        raise ValueError("Spatial axis size must be positive")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Element spacing must be finite and positive")
    return np.fft.fftshift(np.fft.fftfreq(count, d=spacing)).astype(np.float64)


def broadside_angle_axis_deg(direction_cosine: np.ndarray) -> np.ndarray:
    """Map visible direction-cosine bins to broadside angle in degrees.

    Values outside ``[-1, 1]`` are non-physical DFT bins and remain NaN.
    """

    values = np.asarray(direction_cosine, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    visible = np.abs(values) <= 1.0
    result[visible] = np.degrees(np.arcsin(values[visible]))
    return result


def forward_hemisphere_angles(
    elevation_direction_cosine: np.ndarray,
    azimuth_direction_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a physical-bin mask and spherical angles on the joint array grid.

    Spatial FFT coordinates are ``uy`` and ``uz``, not independent spherical
    angles. Use ``ux = sqrt(1 - uy**2 - uz**2)`` on the forward hemisphere.
    Compact arrays can have FFT bins outside the unit disk; those bins and
    singular exact-endfire directions are masked, with NaN angles.
    """
    el = np.asarray(elevation_direction_cosine, dtype=np.float64)
    az = np.asarray(azimuth_direction_cosine, dtype=np.float64)
    if any(v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) for v in (el, az)):
        raise ValueError("Array direction-cosine axes must be nonempty finite vectors")
    z, y = np.meshgrid(el, az, indexing="ij")
    with np.errstate(over="ignore"):
        x_squared = 1.0 - y**2 - z**2
    visible = x_squared > 0.0
    if not np.any(visible):
        raise ValueError("The angular tensor has no identifiable forward-hemisphere beam")
    azimuth = np.full(y.shape, np.nan)
    elevation = np.full(z.shape, np.nan)
    azimuth[visible] = np.degrees(np.arctan2(y[visible], np.sqrt(x_squared[visible])))
    elevation[visible] = np.degrees(np.arcsin(z[visible]))
    return visible, azimuth, elevation


def broadside_azimuth_projection(azimuth_deg: ArrayLike, elevation_deg: ArrayLike) -> np.ndarray:
    """Map spherical measurement angles onto a horizontal broadside plot axis."""
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    return np.asarray(np.rad2deg(np.arcsin(np.sin(azimuth) * np.cos(elevation))))


def strongest_array_directions(
    power: np.ndarray,
    elevation_direction_cosine: np.ndarray,
    azimuth_direction_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spherical angles of one joint physical peak per Doppler/range cell.

    Ties use the first visible bin in elevation/azimuth order. Empty cells have
    NaN angles. Selecting both angles from the same bin avoids inventing a
    direction from independent marginal peaks.
    """
    values = np.asarray(power)
    visible, azimuth, elevation = forward_hemisphere_angles(
        elevation_direction_cosine, azimuth_direction_cosine
    )
    if (
        values.ndim != 4
        or values.shape[:2] != visible.shape
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
    ):
        raise ValueError("Angular power must be finite nonnegative [E,A,D,R] on the array axes")
    beams = values[visible]
    peak = np.argmax(beams, axis=0)
    az_map = azimuth[visible][peak].astype(np.float32)
    el_map = elevation[visible][peak].astype(np.float32)
    empty = np.max(beams, axis=0) <= 0
    az_map[empty] = np.nan
    el_map[empty] = np.nan
    return az_map, el_map


def planar_spatial_fft(
    element_data: np.ndarray,
    *,
    num_rows: int,
    num_cols: int,
    vertical_spacing_wavelengths: float,
    horizontal_spacing_wavelengths: float,
) -> PlanarSpatialSpectrum:
    """Apply a unitary ULA/UPA FFT to a Sionna-ordered element axis.

    Args:
        element_data: Complex values with the antenna-element dimension first.
        num_rows: Number of top-to-bottom planar-array rows.
        num_cols: Number of left-to-right planar-array columns.
        vertical_spacing_wavelengths: Row spacing in wavelengths.
        horizontal_spacing_wavelengths: Column spacing in wavelengths.

    Returns:
        A spectrum whose first two dimensions are elevation and azimuth spatial
        bins.  A one-row ULA therefore has a singleton elevation dimension.
    """

    values = np.asarray(element_data)
    rows = int(num_rows)
    cols = int(num_cols)
    if values.ndim < 1:
        raise ValueError("element_data must have an antenna-element axis")
    if rows < 1 or cols < 1 or rows * cols != values.shape[0]:
        raise ValueError(
            "Planar array dimensions must match the leading element axis: "
            f"rows={rows}, cols={cols}, shape={values.shape}"
        )

    # Sionna cycles rows inside columns. Reshape to (column, row, ...), then
    # transpose to the physical (row, column, ...) grid used by the 2-D FFT.
    grid = values.reshape((cols, rows, *values.shape[1:])).swapaxes(0, 1)
    spectrum = np.fft.fftshift(
        np.fft.fft2(grid, s=(rows, cols), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )

    azimuth_u = direction_cosine_axis(cols, horizontal_spacing_wavelengths)
    # Row index increases downward while physical local z decreases. Negating
    # the FFT coordinate restores positive elevation toward local +z.
    elevation_u = -direction_cosine_axis(rows, vertical_spacing_wavelengths)
    return PlanarSpatialSpectrum(
        spectrum=spectrum,
        azimuth_direction_cosine=azimuth_u,
        elevation_direction_cosine=elevation_u,
        broadside_azimuth_deg=broadside_angle_axis_deg(azimuth_u),
        broadside_elevation_deg=broadside_angle_axis_deg(elevation_u),
        grating_lobes_possible=(
            float(horizontal_spacing_wavelengths) > 0.5 or float(vertical_spacing_wavelengths) > 0.5
        ),
    )


__all__ = [
    "PLANAR_ELEMENT_ORDER",
    "SPATIAL_FFT_NORMALIZATION",
    "SPATIAL_PHASE_CONVENTION",
    "PlanarSpatialSpectrum",
    "broadside_angle_axis_deg",
    "broadside_azimuth_projection",
    "direction_cosine_axis",
    "forward_hemisphere_angles",
    "planar_element_coordinates",
    "planar_spatial_fft",
    "strongest_array_directions",
]
