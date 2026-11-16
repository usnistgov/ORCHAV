"""Temporal-identity invariants for generated sensing products."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import (
    ANALYTIC_ORACLE_MODE,
    COHERENT_MODE,
    SEQUENTIAL_MODE,
    resolve_sensing_execution_values,
)
from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS,
    assert_sensing_contract,
    canonical_sensing_pair_id,
)
from shared.frames.contracts import MPC_HDF5_SCHEMA_VERSION
from shared.frames.json_codec import loads_frame_json
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5 import PackedHDF5Reader
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.sensing.products import (
    ACQUISITION_SAMPLE_REFERENCE_POLICY,
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    catalog_axis,
    dense_axis,
)

GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def _assert_native_v4_has_no_compatibility_aliases(sensing: dict) -> None:
    assert SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS.isdisjoint(sensing)


def _attach_coherent_typed_products(
    sensing: dict,
    slow_time: np.ndarray,
    *,
    include_slow_time: bool = True,
    include_angle: bool = False,
    include_raw: bool = False,
) -> None:
    """Add the canonical typed products emitted beside internal hook aliases."""

    matrix = np.asarray(slow_time, dtype=np.complex64)
    sensing["range_axis_m"] = np.arange(matrix.shape[1], dtype=np.float64)
    sensing["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
    sensing["cir_by_pair_rx_element"] = matrix[-1][np.newaxis, np.newaxis, :]
    if include_slow_time:
        sensing["slow_time_by_pair_rx_element"] = matrix[np.newaxis, np.newaxis, :, :]
    sensing["range_profile_by_pair"] = (np.abs(matrix[-1]) ** 2)[np.newaxis, :]
    if include_raw:
        sensing["range_profile_raw_by_pair"] = sensing["range_profile_by_pair"].copy()
        sensing["range_doppler_raw_by_pair"] = matrix[np.newaxis, ...].copy()
        sensing["doppler_axis_hz"] = np.arange(matrix.shape[0], dtype=np.float64)
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    element_axis = catalog_axis(
        "rx_element",
        size=1,
        catalog_key="rx_element_axis",
        coordinate_semantics="test_rx_element_order",
        reference_frame="rx_array_local",
    )
    range_axis = dense_axis(
        "range",
        size=matrix.shape[1],
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics="monostatic_range",
        reference_frame="tx_target_rx_path",
    )
    combine = {"policy": "siso_identity", "normalization": "identity"}
    add_product_spec(
        sensing,
        data_key="cir_by_pair_rx_element",
        kind="delay_domain_cir",
        axes=(pair_axis, element_axis, range_axis),
        representation="complex_voltage",
        unit="1",
        temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        combine=combine,
    )
    if include_slow_time:
        add_product_spec(
            sensing,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(
                pair_axis,
                element_axis,
                dense_axis(
                    "slow_time",
                    size=matrix.shape[0],
                    unit="s",
                    coordinate_key="slow_time_axis_s",
                    coordinate_semantics="scenario_clock_sample_time",
                    reference_frame="scenario_clock",
                ),
                range_axis,
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=("cir_by_pair_rx_element",),
            combine=combine,
        )
    add_product_spec(
        sensing,
        data_key="range_profile_by_pair",
        kind="range_profile",
        axes=(pair_axis, range_axis),
        representation="power_linear",
        unit="1",
        temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        source_data_keys=("cir_by_pair_rx_element",),
        combine=combine,
    )
    if include_raw:
        add_product_spec(
            sensing,
            data_key="range_profile_raw_by_pair",
            kind="range_profile",
            variant="raw",
            axes=(pair_axis, range_axis),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            source_data_keys=("cir_by_pair_rx_element",),
            combine=combine,
        )
        add_product_spec(
            sensing,
            data_key="range_doppler_raw_by_pair",
            kind="range_doppler",
            variant="raw",
            axes=(
                pair_axis,
                dense_axis(
                    "doppler_frequency",
                    size=matrix.shape[0],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    coordinate_semantics="fftshifted_doppler_frequency",
                ),
                range_axis,
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=("slow_time_by_pair_rx_element",),
            combine=combine,
        )
    if include_angle:
        elevation_size = 2
        azimuth_size = 3
        sensing.setdefault(
            "doppler_axis_hz",
            np.arange(matrix.shape[0], dtype=np.float64),
        )
        sensing["elevation_direction_cosine"] = np.linspace(
            -1.0, 1.0, elevation_size, dtype=np.float64
        )
        sensing["azimuth_direction_cosine"] = np.linspace(-1.0, 1.0, azimuth_size, dtype=np.float64)
        sensing["elevation_broadside_angle_deg"] = np.linspace(
            -90.0, 90.0, elevation_size, dtype=np.float64
        )
        sensing["azimuth_broadside_angle_deg"] = np.linspace(
            -90.0, 90.0, azimuth_size, dtype=np.float64
        )
        sensing["range_doppler_angle_by_pair"] = np.ones(
            (1, elevation_size, azimuth_size, matrix.shape[0], matrix.shape[1]),
            dtype=np.float32,
        )
        add_product_spec(
            sensing,
            data_key="range_doppler_angle_by_pair",
            kind="range_doppler_angle",
            axes=(
                pair_axis,
                dense_axis(
                    "elevation_direction_cosine",
                    size=elevation_size,
                    unit="1",
                    coordinate_key="elevation_direction_cosine",
                    alternate_coordinate_keys=("elevation_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_z_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "azimuth_direction_cosine",
                    size=azimuth_size,
                    unit="1",
                    coordinate_key="azimuth_direction_cosine",
                    alternate_coordinate_keys=("azimuth_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_y_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "doppler_frequency",
                    size=matrix.shape[0],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    coordinate_semantics="fftshifted_doppler_frequency",
                ),
                range_axis,
            ),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=("slow_time_by_pair_rx_element",),
            combine=combine,
        )


def _execution(
    mode: str,
    *,
    frame_dt_s: float,
    cpi_length: int = 4,
    coherent_sample_rate_hz: float | None = None,
):
    prf_hz = (
        float(coherent_sample_rate_hz) if coherent_sample_rate_hz is not None else 1.0 / frame_dt_s
    )
    fft_d = cpi_length if mode != ANALYTIC_ORACLE_MODE else 4
    config = SensingConfig(
        enabled=True,
        processing_mode=mode,
        prf_hz=prf_hz,
        fft_size_doppler=fft_d,
    )
    cir_steps = cpi_length if mode == COHERENT_MODE else 1
    return resolve_sensing_execution_values(
        config,
        cir_time_steps=cir_steps,
        cir_sampling_frequency_hz=(prf_hz if mode == COHERENT_MODE else None),
        frame_dt_s=frame_dt_s,
    )


def _source_resolved_scalars(*, fft_size_doppler: int, prf_hz: float) -> dict:
    """Return the exact scalar/processing fields a real processor publishes."""

    return {
        "fft_size_range": 4,
        "fft_size_doppler": fft_size_doppler,
        "sampling_frequency": 20.0e6,
        "prf_hz": prf_hz,
        "carrier_freq_hz": 3.5e9,
        "range_mode": "monostatic",
        "window_range": "hann",
        "window_doppler": "hann",
    }


def _frame(
    step: int,
    target_x_m: float,
    *,
    scene_time_s: float,
    sensing: dict | None = None,
    tx_sensor_ids: tuple[str, ...] = ("tx-0",),
    rx_sensor_ids: tuple[str, ...] = ("rx-0",),
) -> dict:
    return {
        "frame_idx": step,
        "scene_time_s": scene_time_s,
        "tx_list": [SimpleNamespace(name=name) for name in tx_sensor_ids],
        "rx_list": [SimpleNamespace(name=name) for name in rx_sensor_ids],
        "tx_positions_snapshot": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions_snapshot": np.array([[0.0, 0.0, 0.0]]),
        "tx_velocities_snapshot": np.zeros((len(tx_sensor_ids), 3)),
        "rx_velocities_snapshot": np.zeros((len(rx_sensor_ids), 3)),
        "target_positions_snapshot": np.array([[target_x_m, 0.0, 0.0]]),
        "target_velocities_snapshot": np.array([[1.0, 0.0, 0.0]]),
        "targets_metadata": [{"name": "moving_target"}],
        "sensing": sensing,
    }


def _multi_target_frame(
    step: int,
    targets: tuple[tuple[str, float, float], ...],
    *,
    scene_time_s: float,
    sensing: dict,
) -> dict:
    """Build a source frame whose runtime row order is explicit in metadata."""

    frame = _frame(
        step,
        targets[0][1] if targets else 0.0,
        scene_time_s=scene_time_s,
        sensing=sensing,
    )
    frame["target_positions_snapshot"] = np.asarray(
        [[x_m, 0.0, 0.0] for _target_id, x_m, _velocity_m_s in targets],
        dtype=np.float64,
    ).reshape((-1, 3))
    frame["target_velocities_snapshot"] = np.asarray(
        [[velocity_m_s, 0.0, 0.0] for _target_id, _x_m, velocity_m_s in targets],
        dtype=np.float64,
    ).reshape((-1, 3))
    frame["targets_metadata"] = [{"name": target_id} for target_id, _x, _v in targets]
    return frame


def _truth_observable(sensing: dict, target_id: str) -> dict:
    target = next(row for row in sensing["truth"]["targets"] if row["target_id"] == target_id)
    assert target["lifecycle"] == "active"
    assert target["valid"] is True
    assert len(target["pair_observables"]) == 1
    return target["pair_observables"][0]


def test_tim_001_coherent_cpi_preserves_one_solve_membership_and_truth_epoch(monkeypatch) -> None:
    """One Sionna solve owns every coherent offset and its start-state truth."""

    first_step = 100
    end_step = 101
    frame_dt_s = 0.5
    cir_sampling_frequency_hz = 4.0
    scheduler = FixedIntervalScheduler(
        first_step=first_step,
        last_step=end_step,
        interval_length=2,
    )
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=frame_dt_s,
        gt_dt_s=frame_dt_s * 2,
        first_step=first_step,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(
            COHERENT_MODE,
            frame_dt_s=frame_dt_s,
            cpi_length=2,
            coherent_sample_rate_hz=cir_sampling_frequency_hz,
        ),
    )
    from shared.sensing.plotting import cartesian as sensing_cartesian_figures

    collected_scenes = []
    original_cartesian_event = sensing_cartesian_figures.cartesian_tracking_event

    def collect_cartesian(sensing, *, scene_context):
        collected_scenes.append(scene_context)
        return original_cartesian_event(sensing, scene_context=scene_context)

    state.collect_summary_cartesian = True
    monkeypatch.setattr(sensing_cartesian_figures, "cartesian_tracking_event", collect_cartesian)

    slow_time = np.zeros((2, 4), dtype=np.complex64)
    start_sensing = {
        "source_frame_step": first_step,
        "processing_strategy": "coherent_cpi_fft",
        "rd_source": "fft_cpi",
        "rd_model": "point_binned_cir_doppler_fft",
        "pair_index": 0,
        "raw_payload_policy": "always",
        "slow_time_matrix": slow_time.copy(),
        "compatibility_views": {"slow_time_matrix": "typed product projection"},
        "range_doppler_cfar_mask": np.ones((2, 4), dtype=np.uint8),
        "range_doppler_aoa_az_map": np.ones((2, 4), dtype=np.float32),
        "analytical_paths": {
            "amplitudes": np.ones((2,), dtype=np.complex64),
            "delays_s": np.ones((2,), dtype=np.float64),
            "doppler_hz": np.ones((2,), dtype=np.float64),
        },
        "azimuth_axis_deg": [-45.0, 0.0, 45.0],
        "elevation_axis_deg": [-30.0, 30.0],
        "prf_hz": cir_sampling_frequency_hz,
        "config": {"prf_hz": cir_sampling_frequency_hz},
        "config_resolved": _source_resolved_scalars(
            fft_size_doppler=2,
            prf_hz=cir_sampling_frequency_hz,
        ),
    }
    _attach_coherent_typed_products(
        start_sensing,
        slow_time,
        include_angle=True,
        include_raw=True,
    )
    start_frame = _frame(
        first_step,
        10.0,
        scene_time_s=50.0,
        sensing=start_sensing,
    )
    state.process_frame(
        start_frame,
        step_idx=first_step,
        is_rt_step=True,
        schedule_event=scheduler.event_for_step(first_step),
    )

    end_frame = _frame(end_step, 99.0, scene_time_s=50.5)
    end_frame["_cached_rt_source_frame_idx"] = first_step
    state.process_frame(
        end_frame,
        step_idx=end_step,
        is_rt_step=False,
        schedule_event=scheduler.event_for_step(end_step),
    )

    sensing = end_frame["sensing"]
    assert collected_scenes == [start_frame]
    assert collected_scenes[0]["scene_time_s"] == 50.0
    assert end_frame["scene_time_s"] == 50.5
    assert state.summary_cartesian_scene is None
    assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
    assert sensing["schema_version"] == SENSING_SCHEMA_VERSION == 4
    _assert_native_v4_has_no_compatibility_aliases(sensing)
    np.testing.assert_array_equal(
        sensing["slow_time_by_pair_rx_element"],
        slow_time[np.newaxis, np.newaxis, :, :],
    )
    assert set(sensing["products"]) == {
        "cir_by_pair_rx_element",
        "slow_time_by_pair_rx_element",
        "range_profile_by_pair",
        "range_profile_raw_by_pair",
        "range_doppler_raw_by_pair",
        "range_doppler_angle_by_pair",
    }
    for internal_key in (
        "range_doppler_cfar_mask",
        "range_doppler_aoa_az_map",
        "analytical_paths",
        "azimuth_axis_deg",
        "elevation_axis_deg",
    ):
        assert internal_key not in sensing
    for described_key in (
        "range_doppler_angle_by_pair",
        "doppler_axis_hz",
        "elevation_direction_cosine",
        "azimuth_direction_cosine",
        "elevation_broadside_angle_deg",
        "azimuth_broadside_angle_deg",
    ):
        assert described_key in sensing
    assert end_frame["frame_idx"] == end_step

    temporal = sensing["temporal"]
    acquisition = temporal["acquisition"]
    samples = acquisition["samples"]
    assert [sample["sample_time_s"] for sample in samples] == [50.0, 50.25]
    assert [sample["source"]["scene_frame_id"] for sample in samples] == [100, 100]
    assert [sample["source"]["sample_offset"] for sample in samples] == [0, 1]
    assert len({sample["source"]["solve_id"] for sample in samples}) == 1
    assert all(sample["source"]["source_generation_id"] == GENERATION_ID for sample in samples)
    assert acquisition["default_reference_time_s"] == 50.125
    assert acquisition["default_reference_policy"] == "cpi_center"
    assert acquisition["tx_sensor_id"] == "tx-0"
    assert acquisition["rx_sensor_id"] == "rx-0"
    assert acquisition["pair_id"] == canonical_sensing_pair_id("tx-0", "rx-0")

    assert temporal["result"]["reference_time_s"] == 50.0
    assert temporal["result"]["reference_time_policy"] == "solve_start"
    assert temporal["display_binding"]["scene_frame_id"] == first_step
    assert temporal["truth"]["reference_time_s"] == 50.0
    assert temporal["truth"]["scene_frame_id"] == first_step
    cir_descriptor = sensing["products"]["cir_by_pair_rx_element"]
    assert cir_descriptor["temporal_support"] == {
        "type": "acquisition_sample",
        "selection_policy": TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        "sample_index": 1,
    }
    assert cir_descriptor["time_span"] == {
        "clock_id": GENERATION_ID,
        "start_time_s": 50.25,
        "end_time_s": 50.25,
        "reference_time_s": 50.25,
        "reference_time_policy": ACQUISITION_SAMPLE_REFERENCE_POLICY,
    }
    slow_time_descriptor = sensing["products"]["slow_time_by_pair_rx_element"]
    assert slow_time_descriptor["temporal_support"] == {"type": TEMPORAL_SUPPORT_FULL_ACQUISITION}
    assert slow_time_descriptor["time_span"]["start_time_s"] == 50.0
    assert slow_time_descriptor["time_span"]["end_time_s"] == 50.25
    assert slow_time_descriptor["time_span"]["reference_time_s"] == 50.0
    assert slow_time_descriptor["time_span"]["reference_time_policy"] == "solve_start"
    assert sensing["products"]["range_profile_raw_by_pair"]["variant"] == "raw"
    assert sensing["products"]["range_doppler_raw_by_pair"]["variant"] == "raw"
    assert sensing["target_state"]["scene_frame_id"] == first_step
    assert sensing["target_state"]["targets"][0]["target_id"] == "moving_target"
    assert sensing["truth"]["targets"][0]["target_id"] == "moving_target"
    assert _truth_observable(sensing, "moving_target")["monostatic_equivalent_range_m"] == 10.0


def test_truncated_final_coherent_cpi_persists_only_observed_acquisition(
    tmp_path,
) -> None:
    """A final partial CPI is valid metadata, not a normal RD result."""

    first_step = 100
    end_step = 101
    sample_rate_hz = 4.0
    scheduler = FixedIntervalScheduler(
        first_step=first_step,
        last_step=end_step,
        interval_length=4,
    )
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.5,
        gt_dt_s=2.0,
        first_step=first_step,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(
            COHERENT_MODE,
            frame_dt_s=0.5,
            cpi_length=4,
            coherent_sample_rate_hz=sample_rate_hz,
        ),
    )
    slow_time = np.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
            [2.0 + 1.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j],
            [30.0 + 0.0j, 31.0 + 0.0j, 32.0 + 0.0j, 33.0 + 0.0j],
            [40.0 + 0.0j, 41.0 + 0.0j, 42.0 + 0.0j, 43.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    start_frame = _frame(
        first_step,
        10.0,
        scene_time_s=50.0,
        sensing={
            "source_frame_step": first_step,
            "processing_strategy": "coherent_cpi_fft",
            "rd_source": "fft_cpi",
            "rd_model": "point_binned_cir_doppler_fft",
            "frame_role": "cpi_end",
            "pair_index": 0,
            "raw_payload_policy": "always",
            "slow_time_matrix": slow_time.copy(),
            "cir": slow_time[-1].copy(),
            "range_profile": np.abs(slow_time[-1]) ** 2,
            "range_profile_raw": np.abs(slow_time[-1]) ** 2,
            "range_profile_kind": "power_linear",
            "compatibility_views": {"cir": "typed product projection"},
            "range_profile_cfar_mask": np.ones(4, dtype=np.uint8),
            "range_doppler_map": np.ones((4, 4), dtype=np.complex64),
            "range_doppler_cfar_mask": np.ones((4, 4), dtype=np.uint8),
            "detections": [{"range_m": 10.0, "velocity_m_s": 1.0}],
            "detection_filter_metadata": {"raw": 1, "kept": 1},
            "cfar_metadata": {"range_doppler": {"method": "ca"}},
            "rd_history_size": 4,
            "rd_required_history": 4,
            "rd_map_frames_used": 4,
            "rd_ready": True,
            "rd_fresh": True,
            "rp_fresh": True,
            "cir_time_steps": 4,
            "prf_hz": sample_rate_hz,
            "config": {"prf_hz": sample_rate_hz, "cir_time_steps": 4},
            "config_resolved": {
                **_source_resolved_scalars(
                    fft_size_doppler=4,
                    prf_hz=sample_rate_hz,
                ),
                "cir_time_steps": 4,
            },
        },
    )
    _attach_coherent_typed_products(
        start_frame["sensing"],
        slow_time,
        include_raw=True,
    )
    state.process_frame(
        start_frame,
        step_idx=first_step,
        is_rt_step=True,
        schedule_event=scheduler.event_for_step(first_step),
    )

    end_frame = _frame(end_step, 11.0, scene_time_s=50.5)
    state.process_frame(
        end_frame,
        step_idx=end_step,
        is_rt_step=False,
        schedule_event=scheduler.event_for_step(end_step),
    )

    sensing = end_frame["sensing"]
    assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
    assert end_frame["frame_idx"] == end_step
    assert state.last_sensing_data is None
    assert state.gt_history == []

    temporal = sensing["temporal"]
    acquisition = temporal["acquisition"]
    assert acquisition["completeness"] == "incomplete_unprocessed"
    assert acquisition["expected_sample_count"] == 4
    assert acquisition["observed_sample_count"] == 2
    assert acquisition["valid_sample_count"] == 2
    assert [sample["sample_time_s"] for sample in acquisition["samples"]] == [
        50.0,
        50.25,
        50.5,
        50.75,
    ]
    assert [sample["valid"] for sample in acquisition["samples"]] == [
        True,
        True,
        False,
        False,
    ]
    for offset, sample in enumerate(acquisition["samples"][:2]):
        assert sample["source"]["scene_frame_id"] == first_step
        assert sample["source"]["sample_offset"] == offset
        assert sample["source"]["source_generation_id"] == GENERATION_ID
    for sample in acquisition["samples"][2:]:
        assert "source" not in sample
        assert sample["phase_coherent"] is False
        assert sample["invalid_reason"] == "not_observed_before_end_of_run"
    assert "result" not in temporal
    assert "display_binding" not in temporal
    assert "truth" not in temporal

    for key in (
        "range_doppler_map",
        "range_doppler_cfar_mask",
        "range_profile_cfar_mask",
        "detections",
        "detection_filter_metadata",
        "cfar_metadata",
        "gt",
        "gt_sensing",
        "gt_state",
    ):
        assert key not in sensing
    assert sensing["frame_role"] == "cpi_incomplete"
    assert sensing["rd_ready"] is False
    assert sensing["rd_fresh"] is False
    assert sensing["rd_history_size"] == 2
    assert sensing["rd_map_frames_used"] == 2
    _assert_native_v4_has_no_compatibility_aliases(sensing)
    assert "slow_time_by_pair_rx_element" not in sensing
    np.testing.assert_array_equal(
        sensing["cir_by_pair_rx_element"],
        slow_time[1][np.newaxis, np.newaxis, :],
    )
    np.testing.assert_array_equal(
        sensing["range_profile_by_pair"],
        (np.abs(slow_time[1]) ** 2)[np.newaxis, :],
    )
    np.testing.assert_array_equal(
        sensing["range_profile_raw_by_pair"],
        (np.abs(slow_time[1]) ** 2)[np.newaxis, :],
    )
    assert set(sensing["products"]) == {
        "cir_by_pair_rx_element",
        "range_profile_by_pair",
        "range_profile_raw_by_pair",
    }
    for data_key in (
        "cir_by_pair_rx_element",
        "range_profile_by_pair",
        "range_profile_raw_by_pair",
    ):
        descriptor = sensing["products"][data_key]
        assert descriptor["temporal_support"]["sample_index"] == 1
        assert descriptor["time_span"]["start_time_s"] == 50.25
        assert descriptor["time_span"]["end_time_s"] == 50.25
        assert descriptor["time_span"]["reference_time_s"] == 50.25

    packed_frame = standard_mpc_frame_from_pair_data(
        frame_index=end_step,
        timestamp_s=50.5,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        vertices_by_pair=[np.zeros((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.zeros((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.zeros((1,), dtype=np.int64)],
        sensing=sensing,
    )
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id=GENERATION_ID,
        compression=None,
    )
    writer.append(packed_frame)
    chunk = writer.finalize_to_range_name()
    write_frame_manifest_atomic(
        tmp_path,
        manifest_from_chunks(
            generation_id=GENERATION_ID,
            frame_set_id="tim-001-incomplete-fixture",
            chunks=[chunk],
            compression={"algorithm": "none"},
            segmentation={"policy": "test"},
            provenance={"fixture": "tim-001"},
            created_utc="2026-08-26T00:00:00+00:00",
        ),
    )

    with h5py.File(tmp_path / chunk.file, "r") as handle:
        assert handle.attrs["schema_version"] == MPC_HDF5_SCHEMA_VERSION == 2
        assert float(handle["frames/timestamp_s"][0]) == 50.5
        stored_metadata = loads_frame_json(handle["frames/sensing_metadata_json"][0])
        assert stored_metadata["schema_version"] == SENSING_SCHEMA_VERSION == 4
        assert (
            stored_metadata["temporal"]["acquisition"]["completeness"] == "incomplete_unprocessed"
        )
        assert "result" not in stored_metadata["temporal"]
        assert "range_doppler_map" not in handle["sensing/fixed"]
        for alias_key in SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS:
            assert alias_key not in stored_metadata
            assert alias_key not in handle["sensing/fixed"]
        np.testing.assert_array_equal(
            handle["sensing/fixed/cir_by_pair_rx_element/data"][0],
            slow_time[1][np.newaxis, np.newaxis, :],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_profile_raw_by_pair/data"][0],
            (np.abs(slow_time[1]) ** 2)[np.newaxis, :],
        )

    reader = PackedHDF5Reader(tmp_path)
    try:
        loaded = reader.load_standard_frame(end_step)
    finally:
        reader.close()
    assert loaded.frame_index == end_step
    assert loaded.timestamp_s == 50.5
    assert loaded.sensing["temporal"]["acquisition"]["start_time_s"] == 50.0
    assert loaded.sensing["config_resolved"]["acquisition"] == (
        state.execution_config.persisted()["acquisition"]
    )
    assert loaded.sensing["config_resolved"]["processing"]["mode"] == COHERENT_MODE
    assert loaded.sensing["config_resolved"]["processing_mode"] == COHERENT_MODE
    assert loaded.sensing["config"] == loaded.sensing["config_resolved"]
    _assert_native_v4_has_no_compatibility_aliases(loaded.sensing)
    np.testing.assert_array_equal(
        loaded.sensing["range_profile_by_pair"],
        (np.abs(slow_time[1]) ** 2)[np.newaxis, :],
    )
    np.testing.assert_array_equal(
        loaded.sensing["range_profile_raw_by_pair"],
        (np.abs(slow_time[1]) ** 2)[np.newaxis, :],
    )


def test_sequential_history_is_incomplete_until_expected_membership_is_observed() -> None:
    """Rolling STFT metadata names missing slots and emits no result prematurely."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.5,
        gt_dt_s=0.5,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(
            SEQUENTIAL_MODE,
            frame_dt_s=0.5,
            cpi_length=4,
        ),
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=0, interval_length=1)
    frame = _frame(
        0,
        10.0,
        scene_time_s=0.0,
        sensing={
            "processing_strategy": "sequential_stft",
            "rd_source": "fft_cpi",
            "rd_model": "point_binned_cir_doppler_fft",
            "pair_index": 0,
            "raw_payload_policy": "diagnostic",
            "rd_required_history": 4,
            "rd_map_frames_used": 1,
            "rd_ready": False,
            "rd_fresh": False,
            "rp_fresh": True,
            "prf_hz": 2.0,
            "cir": np.zeros(4, dtype=np.complex64),
            "range_profile": np.ones(4),
            "detections": [],
            "detection_filter_metadata": {},
            "cfar_metadata": {},
            "config_resolved": _source_resolved_scalars(
                fft_size_doppler=4,
                prf_hz=2.0,
            ),
        },
    )
    frame["sensing"]["compatibility_views"] = {"cir": "typed product projection"}
    frame["sensing"]["range_profile_kind"] = "power_linear"
    _attach_coherent_typed_products(
        frame["sensing"],
        np.zeros((1, 4), dtype=np.complex64),
        include_slow_time=False,
    )

    state.process_frame(
        frame,
        step_idx=0,
        is_rt_step=True,
        schedule_event=scheduler.event_for_step(0),
    )

    sensing = frame["sensing"]
    assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
    temporal = sensing["temporal"]
    acquisition = temporal["acquisition"]
    assert acquisition["completeness"] == "incomplete_unprocessed"
    assert acquisition["expected_sample_count"] == 4
    assert acquisition["observed_sample_count"] == 1
    assert [sample["sample_time_s"] for sample in acquisition["samples"]] == [
        0.0,
        0.5,
        1.0,
        1.5,
    ]
    assert "result" not in temporal
    assert "display_binding" not in temporal
    assert "truth" not in temporal
    assert "range_doppler_map" not in sensing
    assert "detections" not in sensing
    assert "detection_filter_metadata" not in sensing
    assert "cfar_metadata" not in sensing
    _assert_native_v4_has_no_compatibility_aliases(sensing)
    assert set(sensing["products"]) == {
        "cir_by_pair_rx_element",
        "range_profile_by_pair",
    }


def test_sequential_complete_window_uses_center_truth_and_current_result_detections() -> None:
    """A first rolling result pairs center-epoch truth with its own detections."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.5,
        gt_dt_s=0.5,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(
            SEQUENTIAL_MODE,
            frame_dt_s=0.5,
            cpi_length=4,
        ),
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=3, interval_length=1)
    frames = []
    for step in range(4):
        sensing = {
            "processing_strategy": "sequential_stft",
            "rd_source": "fft_cpi",
            "rd_model": "point_binned_cir_doppler_fft",
            "pair_index": 0,
            "raw_payload_policy": "none",
            "rd_required_history": 4,
            "rd_map_frames_used": step + 1,
            "rd_ready": step == 3,
            "rd_fresh": step == 3,
            "rp_fresh": True,
            "prf_hz": 2.0,
            "range_profile": np.ones(4),
            "config_resolved": _source_resolved_scalars(
                fft_size_doppler=4,
                prf_hz=2.0,
            ),
        }
        if step == 3:
            sensing["range_doppler_map"] = np.ones((4, 4), dtype=np.complex64)
            # The CPI-center truth is 11.5 m, while the current source scene is
            # 13 m.  This detection belongs to the current result and must not
            # be looked up on either historical midpoint source frame.
            sensing["detections"] = [{"range_m": 11.5, "velocity_m_s": -2.0}]
        _attach_coherent_typed_products(
            sensing,
            np.zeros((step + 1, 4), dtype=np.complex64),
            include_slow_time=step == 3,
        )
        frame = _frame(step, 10.0 + step, scene_time_s=step * 0.5, sensing=sensing)
        frame["target_velocities_snapshot"] = np.array([[2.0, 0.0, 0.0]])
        state.process_frame(
            frame,
            step_idx=step,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(step),
        )
        frames.append(frame)

    final_sensing = frames[-1]["sensing"]
    assert_sensing_contract({"sensing": final_sensing}, require_schema_version=True)
    _assert_native_v4_has_no_compatibility_aliases(final_sensing)
    temporal = final_sensing["temporal"]
    samples = temporal["acquisition"]["samples"]
    assert [sample["source"]["scene_frame_id"] for sample in samples] == [0, 1, 2, 3]
    assert [sample["sample_time_s"] for sample in samples] == [0.0, 0.5, 1.0, 1.5]
    assert temporal["result"]["reference_time_s"] == 0.75
    assert temporal["result"]["reference_time_policy"] == "cpi_center"
    cir_descriptor = final_sensing["products"]["cir_by_pair_rx_element"]
    assert cir_descriptor["time_span"]["reference_time_s"] == 1.5
    assert cir_descriptor["time_span"]["start_time_s"] == 1.5
    assert cir_descriptor["time_span"]["end_time_s"] == 1.5
    slow_time_descriptor = final_sensing["products"]["slow_time_by_pair_rx_element"]
    assert slow_time_descriptor["time_span"]["start_time_s"] == 0.0
    assert slow_time_descriptor["time_span"]["end_time_s"] == 1.5
    assert slow_time_descriptor["time_span"]["reference_time_s"] == 0.75
    assert temporal["display_binding"]["scene_frame_id"] == 1
    assert temporal["display_binding"]["policy"] == "nearest_at_or_before_reference"
    assert temporal["truth"]["reference_time_s"] == 0.75
    assert temporal["truth"]["scene_frame_id"] is None
    assert temporal["truth"]["policy"] == "linear_interpolation_between_source_scenes"
    assert final_sensing["target_state"]["reference_time_s"] == 0.75
    assert final_sensing["target_state"]["scene_frame_id"] == (
        "interpolated:1:2:0x1.8000000000000p-1"
    )
    assert final_sensing["target_state"]["scene_frame_id"] not in {1, 2}
    assert final_sensing["truth"]["targets"][0]["target_id"] == "moving_target"
    assert (
        _truth_observable(final_sensing, "moving_target")["monostatic_equivalent_range_m"] == 11.5
    )
    assert state.gt_history[-1]["target_ranges"] == [11.5]
    assert state.gt_history[-1]["det_ranges"] == [11.5]
    assert state.gt_history[-1]["target_speeds"] == [-2.0]
    assert state.gt_history[-1]["det_velocities"] == [-2.0]
    assert state.association_history == [
        {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "range_rmse": 0.0,
            "velocity_rmse": 0.0,
            "velocity_matched_pair_count": 1,
            "velocity_excluded_matched_pair_count": 0,
            "velocity_unavailable_opportunity_count": 0,
            "evaluation_reference_protocol": "instantaneous_velocity_support_v1",
        }
    ]


def test_sequential_truth_interpolation_aligns_reordered_target_rows_by_id() -> None:
    """Runtime row order cannot exchange target identity during interpolation."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=1.0,
        gt_dt_s=1.0,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(SEQUENTIAL_MODE, frame_dt_s=1.0, cpi_length=2),
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=2, interval_length=1)
    frames = []
    ordered_targets = (
        (("alpha", 10.0, 1.0), ("bravo", 20.0, 2.0)),
        (("bravo", 24.0, 4.0), ("alpha", 14.0, 3.0)),
        (("bravo", 28.0, 6.0), ("alpha", 18.0, 5.0)),
    )
    for step, targets in enumerate(ordered_targets):
        sensing = {
            "processing_strategy": "sequential_stft",
            "rd_source": "fft_cpi",
            "rd_model": "point_binned_cir_doppler_fft",
            "pair_index": 0,
            "raw_payload_policy": "none",
            "rd_required_history": 2,
            "rd_map_frames_used": min(step + 1, 2),
            "rd_ready": step >= 1,
            "rd_fresh": step >= 1,
            "rp_fresh": True,
            "prf_hz": 1.0,
            "range_profile": np.ones(4),
            "config_resolved": _source_resolved_scalars(
                fft_size_doppler=2,
                prf_hz=1.0,
            ),
        }
        if step >= 1:
            sensing["range_doppler_map"] = np.ones((2, 4), dtype=np.complex64)
        frame = _multi_target_frame(
            step,
            targets,
            scene_time_s=float(step),
            sensing=sensing,
        )
        state.process_frame(
            frame,
            step_idx=step,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(step),
        )
        frames.append(frame)

    result = frames[1]["sensing"]
    assert result["temporal"]["truth"]["reference_time_s"] == 0.5
    assert result["temporal"]["truth"]["scene_frame_id"] is None
    assert result["target_state"]["reference_time_s"] == 0.5
    assert result["target_state"]["scene_frame_id"] == ("interpolated:0:1:0x1.0000000000000p-1")
    state_by_id = {row["target_id"]: row for row in result["target_state"]["targets"]}
    truth_by_id = {
        row["target_id"]: row["pair_observables"][0] for row in result["truth"]["targets"]
    }
    assert set(state_by_id) == set(truth_by_id) == {"alpha", "bravo"}
    np.testing.assert_array_equal(state_by_id["alpha"]["position_m"], [12.0, 0.0, 0.0])
    np.testing.assert_array_equal(state_by_id["bravo"]["position_m"], [22.0, 0.0, 0.0])
    np.testing.assert_array_equal(state_by_id["alpha"]["velocity_m_s"], [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(state_by_id["bravo"]["velocity_m_s"], [3.0, 0.0, 0.0])
    assert truth_by_id["alpha"]["monostatic_equivalent_range_m"] == 12.0
    assert truth_by_id["bravo"]["monostatic_equivalent_range_m"] == 22.0

    second_result = frames[2]["sensing"]
    assert second_result["temporal"]["truth"]["reference_time_s"] == 1.5
    history = state.gt_history[-1]
    assert history["target_ids"] == ["bravo", "alpha"]
    assert history["target_ranges"] == [26.0, 16.0]
    assert history["target_speeds"] == [-5.0, -4.0]
    assert history["recorded_target_speeds"] == [-5.0, -4.0]
    assert [row["target_id"] for row in history["target_velocity_reference_evidence"]] == [
        "bravo",
        "alpha",
    ]
    assert {
        row["target_id"]: row["pair_observables"][0]["monostatic_equivalent_velocity_m_s"]
        for row in second_result["truth"]["targets"]
    } == {"alpha": -4.0, "bravo": -5.0}


def test_selected_evaluation_fields_align_by_target_id_without_reordering_native_rows(
    monkeypatch,
) -> None:
    """All reference diagnostics retain the identity of their numeric row."""

    selected = {
        "target_ids": ["alpha", "bravo"],
        "target_speeds": [-1.0, -2.0],
        "target_velocity_valid": [True, False],
        "target_velocity_sources": ["analytic_velocity_projection", "unavailable"],
        "target_velocity_reference_dt_s": [None, 0.5],
        "target_velocity_unavailable_reasons": [None, "missing sensor velocity"],
        "target_velocity_references": ["instantaneous", "recorded interval"],
        "recorded_target_speeds": [-3.0, -4.0],
        "recorded_target_velocity_sources": ["source-alpha", "source-bravo"],
        "recorded_target_velocity_valid": [False, True],
        "recorded_target_velocity_reference_dt_s": [0.25, None],
        "evaluation_reference_protocol": "instantaneous_velocity_support_v1",
    }
    monkeypatch.setattr(
        "shared.sensing.evaluation_truth.extract_evaluation_truth",
        lambda *_args: selected,
    )
    entry = {
        "evaluation_reference_valid": True,
        "target_ids": ["bravo", "alpha"],
        "target_ranges": [20.0, 10.0],
        "target_path_closing_rates_m_s": [40.0, 20.0],
        "target_monostatic_equivalent_velocities_m_s": [20.0, 10.0],
        "target_bistatic_bisector_velocities_m_s": [20.0, 10.0],
    }
    SensingPipelineState._select_evaluation_truth(entry, {}, {})
    assert entry["target_ids"] == ["bravo", "alpha"]
    assert entry["target_ranges"] == [20.0, 10.0]
    for key, value in selected.items():
        if key == "target_ids":
            continue
        assert entry[key] == (list(reversed(value)) if isinstance(value, list) else value)
    selected["target_velocity_references"][0] = "changed source object"
    assert entry["target_velocity_references"] == ["recorded interval", "instantaneous"]


@pytest.mark.parametrize(
    ("selected_ids", "native_ids"),
    [
        (["alpha", "bravo"], ["alpha"]),
        (["alpha", "charlie"], ["alpha", "bravo"]),
        (["alpha", "alpha", "bravo"], ["alpha", "bravo"]),
        (["alpha", "bravo"], ["alpha", "alpha", "bravo"]),
    ],
)
def test_selected_evaluation_rejects_changed_or_duplicate_target_ids(
    monkeypatch, selected_ids, native_ids
) -> None:
    monkeypatch.setattr(
        "shared.sensing.evaluation_truth.extract_evaluation_truth",
        lambda *_args: {"target_ids": selected_ids},
    )
    entry = {"evaluation_reference_valid": True, "target_ids": native_ids}
    with pytest.raises(ValueError, match="exact native target identities"):
        SensingPipelineState._select_evaluation_truth(entry, {}, {})


def test_target_catalog_keeps_temporarily_absent_target_inactive_without_origin_teleport() -> None:
    """An absent catalog member has null state, never a fabricated origin row."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=1.0,
        gt_dt_s=1.0,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=1.0),
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=1, interval_length=1)
    first = _multi_target_frame(
        0,
        (("target_car", 10.0, 1.0), ("target_person", 20.0, 2.0)),
        scene_time_s=0.0,
        sensing=_oracle_sensing(pair_index=0),
    )
    second = _multi_target_frame(
        1,
        (("target_car", 11.0, 1.0),),
        scene_time_s=1.0,
        sensing=_oracle_sensing(pair_index=0),
    )
    for step, frame in enumerate((first, second)):
        state.process_frame(
            frame,
            step_idx=step,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(step),
        )

    result = second["sensing"]
    state_by_id = {row["target_id"]: row for row in result["target_state"]["targets"]}
    assert state_by_id["target_car"]["lifecycle"] == "active"
    assert state_by_id["target_person"] == {
        "target_id": "target_person",
        "lifecycle": "inactive",
        "valid": False,
        "position_m": None,
        "orientation_rad": None,
        "velocity_m_s": None,
        "invalid_reason": None,
    }
    truth_by_id = {row["target_id"]: row for row in result["truth"]["targets"]}
    assert set(truth_by_id) == {"target_car", "target_person"}
    assert truth_by_id["target_car"]["valid"] is True
    assert truth_by_id["target_person"]["lifecycle"] == "inactive"
    assert truth_by_id["target_person"]["pair_observables"] == []


def test_analytic_oracle_binds_single_source_scene_at_scene_time() -> None:
    """An analytic oracle has one sample and exact result/display/truth epochs."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.25,
        gt_dt_s=0.25,
        first_step=4,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=0.25),
    )
    scheduler = FixedIntervalScheduler(first_step=4, last_step=4, interval_length=1)
    frame = _frame(
        4,
        10.0,
        scene_time_s=1.0,
        sensing={
            "processing_strategy": "analytic_path_doppler_oracle",
            "rd_source": "path_doppler",
            "rd_model": "analytic_path_power_oracle_nonwaveform",
            "pair_index": 0,
            "raw_payload_policy": "none",
            "prf_hz": 4.0,
            "config_resolved": _source_resolved_scalars(
                fft_size_doppler=4,
                prf_hz=4.0,
            ),
        },
    )

    state.process_frame(
        frame,
        step_idx=4,
        is_rt_step=True,
        schedule_event=scheduler.event_for_step(4),
    )

    sensing = frame["sensing"]
    assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
    _assert_native_v4_has_no_compatibility_aliases(sensing)
    temporal = sensing["temporal"]
    sample = temporal["acquisition"]["samples"][0]
    assert sample["sample_time_s"] == 1.0
    assert sample["source"]["scene_frame_id"] == 4
    assert sample["source"]["sample_offset"] == 0
    assert temporal["result"]["reference_time_s"] == 1.0
    assert temporal["display_binding"]["scene_frame_id"] == 4
    assert temporal["truth"]["scene_frame_id"] == 4


def _oracle_sensing(*, pair_index: int) -> dict:
    return {
        "processing_strategy": "analytic_path_doppler_oracle",
        "rd_source": "path_doppler",
        "rd_model": "analytic_path_power_oracle_nonwaveform",
        "pair_index": pair_index,
        "raw_payload_policy": "none",
        "prf_hz": 4.0,
        "config_resolved": _source_resolved_scalars(
            fft_size_doppler=4,
            prf_hz=4.0,
        ),
    }


def test_multiple_pair_topology_is_rejected_by_phase2_contract() -> None:
    """Phase 2 rejects multi-pair frames instead of flattening pair order."""

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True, "range_mode": "monostatic"}),
        frame_dt_s=0.25,
        gt_dt_s=0.25,
        first_step=77,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=0.25),
    )
    scheduler = FixedIntervalScheduler(first_step=77, last_step=77, interval_length=1)
    frame = _frame(
        77,
        10.0,
        scene_time_s=12.5,
        sensing=_oracle_sensing(pair_index=2),
        tx_sensor_ids=("tx-a", "tx-b"),
        rx_sensor_ids=("rx-a", "rx-b"),
    )

    with pytest.raises(ValueError, match="topology disagrees"):
        state.process_frame(
            frame,
            step_idx=77,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(77),
        )


@pytest.mark.parametrize(
    ("tx_sensor_ids", "rx_sensor_ids", "pair_index", "match"),
    [
        (("duplicate", "duplicate"), ("rx",), 0, "TX sensor names must be unique"),
        (("tx",), ("",), 0, "stable non-empty name"),
        (("tx",), ("rx",), 1, "pair_index disagrees"),
    ],
)
def test_pair_identity_rejects_ambiguous_or_invalid_raw_devices(
    tx_sensor_ids: tuple[str, ...],
    rx_sensor_ids: tuple[str, ...],
    pair_index: int,
    match: str,
) -> None:
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True}),
        frame_dt_s=1.0,
        gt_dt_s=1.0,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=1.0),
    )
    frame = _frame(
        0,
        1.0,
        scene_time_s=8.0,
        sensing=_oracle_sensing(pair_index=pair_index),
        tx_sensor_ids=tx_sensor_ids,
        rx_sensor_ids=rx_sensor_ids,
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=0, interval_length=1)

    with pytest.raises(ValueError, match=match):
        state.process_frame(
            frame,
            step_idx=0,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(0),
        )


def test_sensing_source_requires_explicit_finite_scene_time() -> None:
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True}),
        frame_dt_s=1.0,
        gt_dt_s=1.0,
        first_step=900,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=1.0),
    )
    frame = _frame(
        900,
        1.0,
        scene_time_s=3.25,
        sensing=_oracle_sensing(pair_index=0),
    )
    frame.pop("scene_time_s")
    scheduler = FixedIntervalScheduler(first_step=900, last_step=900, interval_length=1)

    with pytest.raises(ValueError, match="explicit finite scene_time_s"):
        state.process_frame(
            frame,
            step_idx=900,
            is_rt_step=True,
            schedule_event=scheduler.event_for_step(900),
        )


def test_disabled_pipeline_does_not_rewrite_schema_less_sensing_payload() -> None:
    state = SensingPipelineState(
        enabled=False,
        frame_dt_s=1.0,
        gt_dt_s=1.0,
        first_step=0,
    )
    legacy_cir = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    frame = {"sensing": {"cir": legacy_cir}}
    event = FixedIntervalScheduler(first_step=0, last_step=0, interval_length=1).event_for_step(0)

    state.process_frame(frame, step_idx=0, is_rt_step=True, schedule_event=event)

    assert set(frame["sensing"]) == {"cir"}
    assert frame["sensing"]["cir"] is legacy_cir


@pytest.mark.parametrize(
    ("sensing", "match"),
    [
        (
            {"config": {"prf_hz": 4.0}, "prf_hz": 4.0},
            "config_resolved.*not fallback",
        ),
        (
            {
                "config_resolved": {},
                "config": {"prf_hz": 4.0},
                "prf_hz": 4.0,
            },
            "config_resolved.prf_hz",
        ),
    ],
)
def test_native_timing_never_falls_back_to_projected_or_outer_prf(
    sensing: dict,
    match: str,
) -> None:
    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True}),
        frame_dt_s=0.25,
        gt_dt_s=0.25,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=0.25),
    )
    event = FixedIntervalScheduler(first_step=0, last_step=0, interval_length=1).event_for_step(0)

    with pytest.raises(ValueError, match=match):
        state._attach_timing_metadata(
            {"sensing": sensing},
            step_idx=0,
            schedule_event=event,
        )


def test_gt_finite_difference_uses_exact_irregular_reference_epochs() -> None:
    state = SensingPipelineState(
        enabled=False,
        frame_dt_s=0.25,
        gt_dt_s=99.0,
        first_step=0,
    )

    # No physical elapsed interval exists before the first retained epoch;
    # the configured cadence must not be presented as an exact truth delta.
    assert state._gt_elapsed_time_s(4.0) is None
    state._record_gt(
        {
            "reference_time_s": 4.0,
            "target_ids": ["target-a"],
            "target_ranges": [12.0],
        }
    )
    assert state._gt_elapsed_time_s(4.35) == pytest.approx(0.35)

    state._record_gt(
        {
            "reference_time_s": 4.35,
            "target_ids": ["target-a"],
            "target_ranges": [11.3],
        }
    )
    assert state.prev_target_ranges == {"target-a": 11.3}
    with pytest.raises(ValueError, match="increase strictly"):
        state._gt_elapsed_time_s(4.35)
