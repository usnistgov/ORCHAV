"""Typed sensing-v4 round trips over the unchanged packed-HDF5 v2 container."""

from __future__ import annotations

import h5py
import numpy as np

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    assert_sensing_contract,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
)
from shared.frames.contracts import MPC_HDF5_LAYOUT, MPC_HDF5_SCHEMA_VERSION, FrameReadRequest
from shared.frames.json_codec import loads_frame_json
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5 import PackedHDF5Reader
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth

GENERATION_ID = "c3312b2e-b906-4a7f-a8e7-e456ad3c5e23"
PAIR_ID = canonical_sensing_hash({"tx_sensor_id": "tx", "rx_sensor_id": "rx"})
DETECTION_STAGE_PRODUCT_KEYS = (
    "range_doppler_detection_input_power_by_pair",
    "range_doppler_cfar_noise_power_by_pair",
    "range_doppler_cfar_threshold_power_by_pair",
    "range_doppler_cfar_training_cell_count_by_pair",
    "range_doppler_cfar_eligible_mask_by_pair",
    "range_doppler_cfar_mask_by_pair",
    "range_doppler_nms_mask_by_pair",
    "range_doppler_snr_db_by_pair",
    "range_doppler_snr_valid_mask_by_pair",
    "range_doppler_cluster_labels_by_pair",
)


def _typed_sensing() -> dict[str, object]:
    sample_times = [1.0, 1.1, 1.2]
    processing_recipe = {"window_range": "hann", "window_doppler": "hann"}
    temporal = build_sensing_temporal_contract(
        clock_id=GENERATION_ID,
        acquisition_id="06575322-dc98-426a-a62e-2c9518366b1d",
        acquisition_config_id=canonical_sensing_hash({"samples": len(sample_times)}),
        pair_id=PAIR_ID,
        tx_sensor_id="tx",
        rx_sensor_id="rx",
        samples=[
            {
                "sample_index": index,
                "sample_time_s": time_s,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": GENERATION_ID,
                    "scene_frame_id": 11,
                    "sample_offset": index,
                },
            }
            for index, time_s in enumerate(sample_times)
        ],
        processing_run_id=GENERATION_ID,
        processing_recipe_id=canonical_processing_recipe_id(processing_recipe),
        reference_time_s=1.1,
        reference_time_policy="cpi_center",
        display_scene_frame_id=11,
        display_policy="exact",
        display_reason="Typed HDF5 fixture.",
        truth={
            "reference_time_s": 1.1,
            "scene_frame_id": 11,
            "policy": "fixture",
            "source": "authored_test",
        },
    )
    catalog = build_target_catalog(
        [
            {
                "target_id": "B",
                "display_name": "Target B",
                "identity_source": "authored_test",
            },
            {
                "target_id": "A",
                "display_name": "Target A",
                "identity_source": "authored_test",
            },
        ]
    )
    config_resolved = {
        "contract_version": 1,
        "fft_size_range": 4,
        "fft_size_doppler": 3,
        "sampling_frequency": 1.0e6,
        "prf_hz": 10.0,
        "carrier_freq_hz": 3.5e9,
        "range_mode": "monostatic",
        "processing_mode": "coherent_cpi",
        "persist_detection_stage_artifacts": True,
        "processing_recipe": processing_recipe,
        "acquisition": {
            "sample_source": "coherent_one_solve",
            "rt_solve_stride_scene_steps": 3,
            "sample_period_s": 0.1,
            "sample_rate_hz": 10.0,
            "cpi_length_samples": 3,
            "cpi_hop_samples": 3,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "sionna_intra_solve_phase_coherent",
        },
        "processing": {
            "mode": "coherent_cpi",
            "strategy": "coherent_cpi_fft",
            "fft_size_doppler": 3,
            "result_stride_samples": 3,
            "product_semantics": "cir_point_bin_slow_time_fft",
            "evidence_status": "typed_hdf_fixture",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 2,
            "rx_array_processing": "element_preserving",
        },
    }
    sensing: dict[str, object] = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "config_resolved": config_resolved,
        "config": dict(config_resolved),
        "processing_strategy": "coherent_cpi_fft",
        "target_catalog": catalog,
        "target_state": build_target_state(
            catalog,
            scene_frame_id=11,
            reference_time_s=1.1,
            targets=[
                {
                    "target_id": "B",
                    "lifecycle": "active",
                    "valid": True,
                    "position_m": [2.0, 0.0, 0.0],
                    "orientation_rad": None,
                    "velocity_m_s": [0.0, 1.0, 0.0],
                },
                {
                    "target_id": "A",
                    "lifecycle": "active",
                    "valid": True,
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_rad": None,
                    "velocity_m_s": [1.0, 0.0, 0.0],
                },
            ],
        ),
        "truth": build_target_truth(
            catalog,
            targets=[
                {
                    "target_id": "B",
                    "lifecycle": "active",
                    "valid": True,
                    "position_m": [2.0, 0.0, 0.0],
                    "orientation_rad": None,
                    "velocity_m_s": [0.0, 1.0, 0.0],
                    "invalid_reason": None,
                    "pair_observables": [
                        {
                            "pair_id": temporal["acquisition"]["pair_id"],
                            "bistatic_path_length_m": 8.0,
                            "monostatic_equivalent_range_m": 4.0,
                            "path_closing_rate_m_s": -2.0,
                            "monostatic_equivalent_velocity_m_s": -1.0,
                            "bistatic_bisector_velocity_m_s": -1.0,
                            "velocity_valid": True,
                            "velocity_derivation_source": "analytic_velocity_projection",
                            "velocity_reference_dt_s": None,
                            "undefined_reason": None,
                        }
                    ],
                },
                {
                    "target_id": "A",
                    "lifecycle": "active",
                    "valid": True,
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_rad": None,
                    "velocity_m_s": [1.0, 0.0, 0.0],
                    "invalid_reason": None,
                    "pair_observables": [
                        {
                            "pair_id": temporal["acquisition"]["pair_id"],
                            "bistatic_path_length_m": 6.0,
                            "monostatic_equivalent_range_m": 3.0,
                            "path_closing_rate_m_s": 2.0,
                            "monostatic_equivalent_velocity_m_s": 1.0,
                            "bistatic_bisector_velocity_m_s": 1.0,
                            "velocity_valid": True,
                            "velocity_derivation_source": "analytic_velocity_projection",
                            "velocity_reference_dt_s": None,
                            "undefined_reason": None,
                        }
                    ],
                },
            ],
        ),
        "rx_element_axis": [
            {"element_index": 0, "row": 0, "column": 0},
            {"element_index": 1, "row": 0, "column": 1},
        ],
        "range_axis_m": np.array([0.0, 1.0, 2.0, 3.0]),
        "doppler_axis_hz": np.array([-5.0, 0.0, 5.0]),
        "elevation_direction_cosine": np.array([0.0]),
        "azimuth_direction_cosine": np.array([-0.5, 0.5]),
        "elevation_broadside_angle_deg": np.array([0.0]),
        "azimuth_broadside_angle_deg": np.array([-30.0, 30.0]),
        "slow_time_by_pair_rx_element": np.arange(24, dtype=np.float32)
        .reshape(1, 2, 3, 4)
        .astype(np.complex64),
        "range_doppler_by_pair": np.ones((1, 3, 4), dtype=np.complex64),
        "range_doppler_angle_by_pair": np.arange(24, dtype=np.float32).reshape(1, 1, 2, 3, 4),
    }
    detection_stage_values = {
        "range_doppler_detection_input_power_by_pair": np.ones((1, 3, 4), dtype=np.float64),
        "range_doppler_cfar_noise_power_by_pair": np.full((1, 3, 4), 0.5, dtype=np.float64),
        "range_doppler_cfar_threshold_power_by_pair": np.full((1, 3, 4), 1.5, dtype=np.float64),
        "range_doppler_cfar_training_cell_count_by_pair": np.full((1, 3, 4), 8, dtype=np.int32),
        "range_doppler_cfar_eligible_mask_by_pair": np.ones((1, 3, 4), dtype=np.bool_),
        "range_doppler_cfar_mask_by_pair": np.zeros((1, 3, 4), dtype=np.bool_),
        "range_doppler_nms_mask_by_pair": np.zeros((1, 3, 4), dtype=np.bool_),
        "range_doppler_snr_db_by_pair": np.full((1, 3, 4), 3.0102999566, dtype=np.float64),
        "range_doppler_snr_valid_mask_by_pair": np.ones((1, 3, 4), dtype=np.bool_),
        "range_doppler_cluster_labels_by_pair": np.full((1, 3, 4), -1, dtype=np.int32),
    }
    detection_stage_values["range_doppler_cfar_mask_by_pair"][0, 1, 2] = True
    detection_stage_values["range_doppler_nms_mask_by_pair"][0, 1, 2] = True
    detection_stage_values["range_doppler_cluster_labels_by_pair"][0, 1, 2] = 0
    sensing.update(detection_stage_values)
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    range_axis = dense_axis("range", size=4, unit="m", coordinate_key="range_axis_m")
    range_doppler_axes = (
        pair_axis,
        dense_axis(
            "doppler_frequency",
            size=3,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
        ),
        range_axis,
    )
    add_product_spec(
        sensing,
        data_key="slow_time_by_pair_rx_element",
        kind="delay_domain_slow_time",
        axes=(
            pair_axis,
            catalog_axis("rx_element", size=2, catalog_key="rx_element_axis"),
            dense_axis("slow_time", size=3, unit="s", coordinate_key="slow_time_axis_s"),
            range_axis,
        ),
        representation="complex_voltage",
        unit="1",
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cluster_labels_by_pair",
        kind="range_doppler_cluster_labels",
        axes=range_doppler_axes,
        representation="integer_label",
        unit="1",
        source_data_keys=(
            "range_doppler_nms_mask_by_pair",
            "range_doppler_snr_db_by_pair",
            "range_doppler_snr_valid_mask_by_pair",
        ),
        processing={"algorithm": "fixture_dbscan"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_nms_mask_by_pair",
        kind="range_doppler_nms_mask",
        axes=range_doppler_axes,
        representation="boolean_mask",
        unit="1",
        source_data_keys=(
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_mask_by_pair",
            "range_doppler_snr_valid_mask_by_pair",
        ),
        processing={"algorithm": "fixture_nms"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_snr_db_by_pair",
        kind="range_doppler_snr_db",
        axes=range_doppler_axes,
        representation="log_power_ratio_db",
        unit="dB",
        source_data_keys=(
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
        processing={"formula": "10 * log10(input_power / noise_power)"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_snr_valid_mask_by_pair",
        kind="range_doppler_snr_valid_mask",
        axes=range_doppler_axes,
        representation="boolean_mask",
        unit="1",
        source_data_keys=(
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
        processing={"formula": "eligible and noise_power > 0"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cfar_mask_by_pair",
        kind="range_doppler_cfar_mask",
        axes=range_doppler_axes,
        representation="boolean_mask",
        unit="1",
        source_data_keys=(
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_threshold_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
        processing={"comparison": "strict_greater_than_threshold"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cfar_threshold_power_by_pair",
        kind="range_doppler_cfar_threshold_power",
        axes=range_doppler_axes,
        representation="power_linear",
        unit="1",
        source_data_keys=(
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
        processing={"formula": "noise_power * threshold_scale"},
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cfar_eligible_mask_by_pair",
        kind="range_doppler_cfar_eligible_mask",
        axes=range_doppler_axes,
        representation="boolean_mask",
        unit="1",
        source_data_keys=(
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_training_cell_count_by_pair",
        ),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cfar_training_cell_count_by_pair",
        kind="range_doppler_cfar_training_cell_count",
        axes=range_doppler_axes,
        representation="integer_count",
        unit="1",
        source_data_keys=("range_doppler_detection_input_power_by_pair",),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_cfar_noise_power_by_pair",
        kind="range_doppler_cfar_noise_power",
        axes=range_doppler_axes,
        representation="power_linear",
        unit="1",
        source_data_keys=("range_doppler_detection_input_power_by_pair",),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_detection_input_power_by_pair",
        kind="range_doppler_detection_input_power",
        axes=range_doppler_axes,
        representation="power_linear",
        unit="1",
        source_data_keys=("range_doppler_by_pair",),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler",
        axes=range_doppler_axes,
        representation="complex_voltage",
        unit="1",
        source_data_keys=("slow_time_by_pair_rx_element",),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_angle_by_pair",
        kind="range_doppler_angle",
        axes=(
            pair_axis,
            dense_axis(
                "elevation_direction_cosine",
                size=1,
                unit="1",
                coordinate_key="elevation_direction_cosine",
                alternate_coordinate_keys=("elevation_broadside_angle_deg",),
            ),
            dense_axis(
                "azimuth_direction_cosine",
                size=2,
                unit="1",
                coordinate_key="azimuth_direction_cosine",
                alternate_coordinate_keys=("azimuth_broadside_angle_deg",),
            ),
            dense_axis("doppler_frequency", size=3, unit="Hz", coordinate_key="doppler_axis_hz"),
            range_axis,
        ),
        representation="power_linear",
        unit="1",
        source_data_keys=("slow_time_by_pair_rx_element",),
    )
    materialize_product_specs(sensing)
    assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
    return sensing


def test_typed_products_catalog_and_reordered_state_survive_hdf5_v2(tmp_path) -> None:
    sensing = _typed_sensing()
    frame = standard_mpc_frame_from_pair_data(
        frame_index=11,
        timestamp_s=1.1,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 0.0]]),
        rx_positions=np.asarray([[1.0, 0.0, 0.0]]),
        vertices_by_pair=[np.zeros((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.zeros((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.zeros((1,), dtype=np.int64)],
        target_positions_m=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        targets_metadata=({"name": "A"}, {"name": "B"}),
        sensing=sensing,
    )
    writer = PackedMPCChunkWriter(tmp_path, generation_id=GENERATION_ID, compression=None)
    writer.append(frame)
    chunk = writer.finalize_to_range_name()
    write_frame_manifest_atomic(
        tmp_path,
        manifest_from_chunks(
            generation_id=GENERATION_ID,
            frame_set_id="typed-sensing-v4-fixture",
            chunks=[chunk],
            compression={"algorithm": "none"},
            segmentation={"policy": "test"},
            provenance={"fixture": "typed-sensing-v4"},
            created_utc="2026-08-27T00:00:00+00:00",
        ),
    )

    with h5py.File(tmp_path / chunk.file, "r") as handle:
        assert int(handle.attrs["schema_version"]) == MPC_HDF5_SCHEMA_VERSION == 2
        assert handle.attrs["storage_layout"] == MPC_HDF5_LAYOUT
        static_bundle = loads_frame_json(handle["static/sensing_config_json"][()])
        assert static_bundle["target_catalog"] == sensing["target_catalog"]
        per_frame = loads_frame_json(handle["frames/sensing_metadata_json"][0])
        assert "target_catalog" not in per_frame
        assert per_frame["schema_version"] == 4
        assert handle["sensing/fixed/slow_time_by_pair_rx_element/data"].shape == (
            1,
            1,
            2,
            3,
            4,
        )
        assert handle["sensing/fixed/range_doppler_angle_by_pair/data"].shape == (
            1,
            1,
            1,
            2,
            3,
            4,
        )
        assert handle["sensing/fixed/range_doppler_cluster_labels_by_pair/data"].shape == (
            1,
            1,
            3,
            4,
        )

    reader = PackedHDF5Reader(tmp_path)
    try:
        loaded = reader.load_standard_frame(11)
        projected = reader.load_projection(
            11,
            FrameReadRequest(
                sensing_products=frozenset(
                    {
                        "range_doppler_angle_by_pair",
                        "range_doppler_cluster_labels_by_pair",
                    }
                )
            ),
        )
    finally:
        reader.close()
    assert loaded.sensing is not None
    assert loaded.sensing["target_catalog"] == sensing["target_catalog"]
    assert [row["target_id"] for row in loaded.sensing["target_state"]["targets"]] == [
        "B",
        "A",
    ]
    assert [row["target_id"] for row in loaded.sensing["truth"]["targets"]] == ["B", "A"]
    np.testing.assert_array_equal(
        loaded.sensing["slow_time_by_pair_rx_element"],
        sensing["slow_time_by_pair_rx_element"],
    )
    np.testing.assert_array_equal(
        loaded.sensing["range_doppler_angle_by_pair"],
        sensing["range_doppler_angle_by_pair"],
    )
    for data_key in DETECTION_STAGE_PRODUCT_KEYS:
        np.testing.assert_array_equal(loaded.sensing[data_key], sensing[data_key])
    assert loaded.sensing["products"] == sensing["products"]
    projected_sensing = projected.frame.sensing
    assert projected_sensing is not None
    assert "slow_time_by_pair_rx_element" not in projected_sensing
    assert "range_doppler_detection_input_power_by_pair" not in projected_sensing
    np.testing.assert_array_equal(
        projected_sensing["range_doppler_cluster_labels_by_pair"],
        sensing["range_doppler_cluster_labels_by_pair"],
    )
    assert projected_sensing["products"] == sensing["products"]
    assert (
        projected_sensing["products"]["range_doppler_cluster_labels_by_pair"]["lineage"]
        == sensing["products"]["range_doppler_cluster_labels_by_pair"]["lineage"]
    )
    for coordinate_key in (
        "range_axis_m",
        "doppler_axis_hz",
        "elevation_direction_cosine",
        "azimuth_direction_cosine",
        "elevation_broadside_angle_deg",
        "azimuth_broadside_angle_deg",
    ):
        assert coordinate_key in projected_sensing
