"""Independent references for the sensing array-axis contract."""

from __future__ import annotations

import numpy as np
import pytest

from shared.sensing.arrays import (
    PLANAR_ELEMENT_ORDER,
    broadside_angle_axis_deg,
    direction_cosine_axis,
    planar_element_coordinates,
    planar_spatial_fft,
    strongest_array_directions,
)


def test_sionna_planar_element_order_is_column_major() -> None:
    assert PLANAR_ELEMENT_ORDER == "sionna_column_major_top_to_bottom"
    assert planar_element_coordinates(3, 4) == (
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    )


@pytest.mark.parametrize("direction_cosine", [-0.5, 0.25])
def test_ula_peak_sign_and_unitary_normalization(direction_cosine: float) -> None:
    element_count = 8
    spacing = 0.5
    # Independent steering reference: the horizontal coordinate of element m
    # is m*d wavelengths.  Do not use a production coordinate/axis helper.
    steering = np.exp(
        1j * 2.0 * np.pi * spacing * np.arange(element_count, dtype=np.float64) * direction_cosine
    )

    result = planar_spatial_fft(
        steering,
        num_rows=1,
        num_cols=element_count,
        vertical_spacing_wavelengths=spacing,
        horizontal_spacing_wavelengths=spacing,
    )

    expected_axis = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(result.azimuth_direction_cosine, expected_axis)
    peak = np.unravel_index(np.argmax(np.abs(result.spectrum)), result.spectrum.shape)
    assert peak == (0, int(np.flatnonzero(expected_axis == direction_cosine)[0]))
    assert np.abs(result.spectrum[peak]) == pytest.approx(np.sqrt(element_count))
    np.testing.assert_allclose(
        np.sum(np.abs(result.spectrum) ** 2),
        np.sum(np.abs(steering) ** 2),
    )


def test_non_square_upa_peak_preserves_row_column_signs() -> None:
    rows = 3
    cols = 4
    spacing = 0.5
    azimuth_u = 0.5
    elevation_u = 2.0 / 3.0

    # Independent Sionna ordering reference.  Row index increases downward,
    # hence its physical +z coordinate has the opposite sign.
    steering = np.asarray(
        [
            np.exp(1j * 2.0 * np.pi * spacing * (column * azimuth_u - row * elevation_u))
            for column in range(cols)
            for row in range(rows)
        ]
    )

    result = planar_spatial_fft(
        steering,
        num_rows=rows,
        num_cols=cols,
        vertical_spacing_wavelengths=spacing,
        horizontal_spacing_wavelengths=spacing,
    )

    assert result.spectrum.shape == (rows, cols)
    expected_azimuth = np.array([-1.0, -0.5, 0.0, 0.5])
    expected_elevation = np.array([2.0 / 3.0, -0.0, -2.0 / 3.0])
    np.testing.assert_allclose(result.azimuth_direction_cosine, expected_azimuth)
    np.testing.assert_allclose(result.elevation_direction_cosine, expected_elevation)
    peak = np.unravel_index(np.argmax(np.abs(result.spectrum)), result.spectrum.shape)
    assert peak == (0, 3)
    assert np.abs(result.spectrum[peak]) == pytest.approx(np.sqrt(rows * cols))


def test_non_visible_bins_are_nan_and_spacing_declares_alias_risk() -> None:
    axis = direction_cosine_axis(8, 0.25)
    angles = broadside_angle_axis_deg(axis)
    assert np.all(np.isnan(angles[np.abs(axis) > 1.0]))
    assert np.all(np.isfinite(angles[np.abs(axis) <= 1.0]))

    result = planar_spatial_fft(
        np.ones(4, dtype=np.complex128),
        num_rows=1,
        num_cols=4,
        vertical_spacing_wavelengths=0.5,
        horizontal_spacing_wavelengths=0.75,
    )
    assert result.grating_lobes_possible is True


@pytest.mark.parametrize(
    ("values", "rows", "cols"),
    [
        (np.ones(3), 1, 4),
        (np.ones(4), 0, 4),
    ],
)
def test_planar_fft_rejects_shape_mismatch(values: np.ndarray, rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        planar_spatial_fft(
            values,
            num_rows=rows,
            num_cols=cols,
            vertical_spacing_wavelengths=0.5,
            horizontal_spacing_wavelengths=0.5,
        )


@pytest.mark.parametrize("rows,uy,uz", [(1, -0.5, 0.0), (4, 0.5, 0.5), (4, -0.5, -0.5)])
def test_measurement_angles_match_independent_steering_geometry(rows, uy, uz):
    steering = np.array(
        [np.exp(1j * np.pi * (col * uy - row * uz)) for col in range(4) for row in range(rows)]
    )
    spatial = planar_spatial_fft(
        steering[:, None, None],
        num_rows=rows,
        num_cols=4,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
    )
    az, el = strongest_array_directions(
        abs(spatial.spectrum) ** 2,
        spatial.elevation_direction_cosine,
        spatial.azimuth_direction_cosine,
    )
    # Reconstruct a unit Cartesian vector independently of FFT angle helpers.
    az, el = np.deg2rad([az.item(), el.item()])
    np.testing.assert_allclose(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        [np.sqrt(1 - uy**2 - uz**2), uy, uz],
        atol=1e-7,
    )


def test_joint_peak_tie_does_not_invent_a_direction_and_nonphysical_bins_are_ignored():
    # Tied peaks at (uz=.5,uy=0) and (uz=0,uy=.5). Independent marginal
    # argmaxes would incorrectly combine them into the unoccupied (.5,.5).
    power = np.zeros((3, 3, 1, 2))
    power[0, 1, 0, 0] = power[1, 0, 0, 0] = 4
    power[2, 2, 0, :] = 1000  # outside the physical disk, not a candidate
    az, el = strongest_array_directions(power, np.array([0.5, 0, 2]), np.array([0.5, 0, 2]))
    assert az[0, 0] == pytest.approx(0)
    assert el[0, 0] == pytest.approx(30)
    assert np.isnan(az[0, 1]) and np.isnan(el[0, 1])
