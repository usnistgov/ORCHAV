"""Focused tests for strict typed angular-product selection."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.angle import select_range_doppler_angle_product
from shared.sensing.products import (
    attach_product_descriptors,
    build_product_descriptor,
    catalog_axis,
    dense_axis,
)

PAIR_ID = "a" * 64


def _typed_angular_payload() -> dict[str, object]:
    elevation_u = np.array([-0.5, 0.5], dtype=np.float64)
    azimuth_u = np.array([-0.8, 0.0, 0.8], dtype=np.float64)
    doppler_hz = np.array([-40.0, 20.0], dtype=np.float64)
    range_m = np.array([0.0, 1.25, 4.5, 9.0], dtype=np.float64)
    sensing: dict[str, object] = {
        "pair_axis": [
            {
                "pair_index": 0,
                "frame_pair_index": 0,
                "pair_id": PAIR_ID,
                "tx_sensor_id": "tx",
                "rx_sensor_id": "rx",
            }
        ],
        "elevation_direction_cosine": elevation_u,
        "azimuth_direction_cosine": azimuth_u,
        # Deliberately not reconstructed arcsines: the selector must preserve
        # these exact persisted alternate coordinates.
        "elevation_broadside_angle_deg": np.array([-44.0, 33.0]),
        "azimuth_broadside_angle_deg": np.array([-70.0, -5.0, 44.0]),
        "doppler_axis_hz": doppler_hz,
        "range_axis_m": range_m,
        "angular_storage_key": np.arange(48, dtype=np.float32).reshape(1, 2, 3, 2, 4),
        "temporal": {
            "clock": {"clock_id": "scenario-clock"},
            "acquisition": {
                "acquisition_id": "0bbf0697-ed8a-4056-b298-82a7ee4d53cf",
                "pair_id": PAIR_ID,
                "tx_sensor_id": "tx",
                "rx_sensor_id": "rx",
                "start_time_s": 1.0,
                "end_time_s": 1.1,
                "default_reference_time_s": 1.05,
                "default_reference_policy": "cpi_center",
                "samples": [],
            },
            "result": {
                "processing_run_id": "0d882796-0068-49e1-acd6-30aeaa90129b",
                "processing_recipe_id": "b" * 64,
                "reference_time_s": 1.05,
                "reference_time_policy": "cpi_center",
            },
        },
    }
    descriptor = build_product_descriptor(
        sensing,
        data_key="angular_storage_key",
        kind="range_doppler_angle",
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "elevation_direction_cosine",
                size=2,
                unit="1",
                coordinate_key="elevation_direction_cosine",
                alternate_coordinate_keys=("elevation_broadside_angle_deg",),
                coordinate_semantics="rx_array_local_z_direction_cosine",
                reference_frame="rx_array_local",
            ),
            dense_axis(
                "azimuth_direction_cosine",
                size=3,
                unit="1",
                coordinate_key="azimuth_direction_cosine",
                alternate_coordinate_keys=("azimuth_broadside_angle_deg",),
                coordinate_semantics="rx_array_local_y_direction_cosine",
                reference_frame="rx_array_local",
            ),
            dense_axis(
                "doppler_frequency",
                size=2,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
                coordinate_semantics="fftshifted_doppler_frequency",
            ),
            dense_axis(
                "range",
                size=4,
                unit="m",
                coordinate_key="range_axis_m",
                coordinate_semantics="bistatic_path_length",
                reference_frame="tx_target_rx_path",
            ),
        ),
        representation="power_linear",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
    )
    attach_product_descriptors(sensing, [descriptor])
    return sensing


def test_typed_angular_selection_preserves_exact_alternate_axes() -> None:
    sensing = _typed_angular_payload()

    product = select_range_doppler_angle_product(sensing)

    assert product is not None
    assert product.data_key == "angular_storage_key"
    assert product.pair_id == PAIR_ID
    assert product.power_linear.shape == (2, 3, 2, 4)
    np.testing.assert_array_equal(
        product.elevation_broadside_angle_deg,
        np.array([-44.0, 33.0]),
    )
    np.testing.assert_array_equal(
        product.azimuth_broadside_angle_deg,
        np.array([-70.0, -5.0, 44.0]),
    )
    np.testing.assert_array_equal(product.doppler_axis_hz, sensing["doppler_axis_hz"])
    np.testing.assert_array_equal(product.range_axis_m, sensing["range_axis_m"])


def test_typed_angular_selection_rejects_pair_mismatch() -> None:
    with pytest.raises(ValueError, match="requested pair identity"):
        select_range_doppler_angle_product(
            _typed_angular_payload(),
            pair_id="different-pair",
        )


def test_typed_angular_selection_returns_none_when_variant_is_absent() -> None:
    sensing = _typed_angular_payload()
    sensing["products"] = {}

    assert select_range_doppler_angle_product(sensing) is None


def test_typed_angular_selection_rejects_malformed_dimensions() -> None:
    sensing = deepcopy(_typed_angular_payload())
    sensing["angular_storage_key"] = np.ones((2, 3, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must have shape"):
        select_range_doppler_angle_product(sensing)


def test_typed_angular_selection_rejects_missing_alternate_coordinates() -> None:
    sensing = _typed_angular_payload()
    sensing.pop("azimuth_broadside_angle_deg")

    with pytest.raises(ValueError, match="azimuth broadside-angle"):
        select_range_doppler_angle_product(sensing)


def test_native_planar_descending_elevation_is_preserved():
    sensing = _typed_angular_payload()
    sensing["elevation_direction_cosine"] = np.array([0.5, -0.5])
    sensing["elevation_broadside_angle_deg"] = np.array([33.0, -44.0])
    product = select_range_doppler_angle_product(sensing)
    np.testing.assert_array_equal(product.elevation_direction_cosine, [0.5, -0.5])
    np.testing.assert_array_equal(product.power_linear, sensing["angular_storage_key"][0])


def test_repeated_elevation_coordinates_are_rejected():
    sensing = _typed_angular_payload()
    sensing["elevation_direction_cosine"] = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="monotonic"):
        select_range_doppler_angle_product(sensing)


def test_compact_array_keeps_full_fft_coordinates_and_nonphysical_degree_mask():
    sensing = _typed_angular_payload()
    sensing["azimuth_direction_cosine"] = np.array([-2.0, 0.0, 0.5])
    sensing["azimuth_broadside_angle_deg"] = np.array([np.nan, 0.0, 30.0])
    product = select_range_doppler_angle_product(sensing)
    np.testing.assert_array_equal(product.azimuth_direction_cosine, [-2.0, 0.0, 0.5])
    np.testing.assert_array_equal(product.azimuth_broadside_angle_deg, [np.nan, 0.0, 30.0])
    np.testing.assert_array_equal(product.power_linear, sensing["angular_storage_key"][0])


@pytest.mark.parametrize(
    "degrees", [[-90.0, 0.0, 30.0], [np.inf, 0.0, 30.0], [np.nan, np.nan, 30.0]]
)
def test_degree_mask_must_match_the_physical_fft_bins(degrees):
    sensing = _typed_angular_payload()
    sensing["azimuth_direction_cosine"] = np.array([-2.0, 0.0, 0.5])
    sensing["azimuth_broadside_angle_deg"] = np.array(degrees)
    with pytest.raises(ValueError, match="broadside-angle"):
        select_range_doppler_angle_product(sensing)
