"""Unit tests for the sensing post-processor."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pytest

import shared.frames.packed_hdf5 as packed_hdf5_module
from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
)
from shared.frames.manifest import (
    load_frame_manifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.frames.types import StandardMPCFrame
from shared.sensing.detection.pipeline import DetectionPipelineResult
from shared.sensing.evaluation import evaluate_scenario
from shared.sensing.loader import load_saved_results
from shared.sensing.plotting.figures import load_cir_sequence, load_gt_entries_for_tracking
from shared.sensing.postprocessor import (
    PostProcessingConfig,
    SensingPostProcessor,
    _analysis_output_root,
    _build_effective_processing_recipe,
    _build_reprocessed_temporal,
    _claim_analysis_root,
    _combine_typed_slow_time,
    _effective_processing_windows,
    _evaluation_join_metadata,
    _group_per_pulse_frames,
    _join_evaluation_events,
    _load_frames_sensing,
    _native_v4_processing_windows,
    _resolve_source_frames_dir,
    _save_stage_artifacts,
    _select_typed_detection_input,
    _source_sensing_payload,
    _typed_range_doppler_arrays,
    _validate_postprocessed_result,
    sweep_thresholds,
)
from shared.sensing.postprocessor import main as sensing_postprocessor_main
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.replay import (
    ReplaySceneContext,
    build_replay_event,
    compare_replay_events,
)
from shared.sensing.signal_processing import apply_pre_cfar_doppler_guards
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth
from shared.sensing.writer import (
    ColumnarResultWriter,
    attach_sensing_to_standard_frame,
    build_incomplete_sensing_block,
    build_reprocessed_sensing_block,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SENSING_META = {
    "config": {
        "fft_size_range": 64,
        "fft_size_doppler": 32,
        "sampling_frequency": 100e6,
        "prf_hz": 1000.0,
        "range_mode": "monostatic",
        "carrier_freq_hz": 28e9,
    }
}


@pytest.mark.parametrize("mode", ["none", "legacy", "windowed_mean"])
def test_clutter_override_requires_complex_samples(mode):
    processor = SensingPostProcessor(PostProcessingConfig(coherent_clutter_mode=mode))
    with pytest.raises(ValueError, match="Clutter overrides require retained complex"):
        processor.process_from_rd_map(np.ones((32, 64)), SENSING_META)


@pytest.mark.parametrize("mode", ["none", "legacy", "windowed_mean", None])
def test_clutter_recipe_survives_derived_slow_time_replay(tmp_path, mode):
    samples = _make_slow_time_matrix()
    source_frame = _make_standard_frame(
        frame_idx=0,
        sensing={
            "config": {**SENSING_META["config"], "coherent_clutter_mode": "windowed_mean"},
            "slow_time_matrix": samples,
        },
    )
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, source_frame)
    frame = _load_frames_sensing(directory)[0]
    source = frame["_typed_sensing"]
    config = PostProcessingConfig(
        mti_enabled=mode is not None, coherent_clutter_mode=mode, skip_detection=True
    )
    result = SensingPostProcessor(config).process_from_slow_time_matrix(samples, source)
    key = "slow_time_by_pair_rx_element"
    result["processing_recipe_resolved"] = _build_effective_processing_recipe(
        frame, postprocessing_config=config, input_kind=key
    )
    result["temporal"] = _build_reprocessed_temporal(
        frame,
        processing_run_id="af345435-dfb6-46f2-87ad-7253d8f5b1bd",
        postprocessing_config=config,
        input_kind=key,
    )
    derived = build_reprocessed_sensing_block(
        source_sensing=source, result=result, postprocessing_config=config, input_kind=key
    )
    assert derived["config_resolved"]["coherent_clutter_mode"] == (mode or "none")
    _write_packed_frame_set(
        tmp_path / "derived", _make_standard_frame(frame_idx=0, sensing=derived)
    )
    loaded = _load_frames_sensing(tmp_path / "derived")[0]["_typed_sensing"]
    replay = SensingPostProcessor(
        PostProcessingConfig(skip_detection=True)
    ).process_from_slow_time_matrix(samples, loaded)
    np.testing.assert_array_equal(replay["rd_cube"], result["rd_cube"])


@pytest.mark.parametrize("radius,expected", [(2, [20, 23]), (4, [20])])
def test_explicit_nms_radius_controls_two_neighboring_returns(radius, expected):
    power = np.ones((32, 64))
    power[22, 20], power[22, 23] = 100.0, 80.0
    config = PostProcessingConfig(
        cfar_guard_cells_range=0,
        cfar_ref_cells_range=1,
        cfar_guard_cells_doppler=0,
        cfar_ref_cells_doppler=1,
        nms_radius_range=radius,
        cluster_range_gate_m=0.5,
    )
    result = SensingPostProcessor(config).process_from_rd_map(power, SENSING_META)
    assert sorted(d["range_bin"] for d in result["detections"]) == expected


_TEST_GENERATION_ID = "00000000-0000-4000-8000-000000000001"
_TEST_PROCESSING_RUN_ID = "00000000-0000-4000-8000-000000000002"
_TEST_SECOND_PROCESSING_RUN_ID = "00000000-0000-4000-8000-000000000003"


def _resolved_v4_config(
    authored: dict,
    *,
    per_pulse: bool,
    requested_strategy: str | None = None,
) -> dict:
    """Build one complete native-v4 execution envelope for focused tests."""

    cfg = deepcopy(authored)
    fft_d = int(cfg.get("fft_size_doppler", 32))
    prf_hz = float(cfg.get("prf_hz", 1000.0))
    sequential = per_pulse or requested_strategy == "sequential_stft"
    mode = "sequential_stft" if sequential else "coherent_cpi"
    strategy = "sequential_stft" if sequential else "coherent_cpi_fft"
    cpi_hop = 1 if sequential else fft_d
    acquisition = {
        "sample_source": "independent_solves" if sequential else "coherent_one_solve",
        "rt_solve_stride_scene_steps": 1 if sequential else fft_d,
        "sample_period_s": 1.0 / prf_hz,
        "sample_rate_hz": prf_hz,
        "cpi_length_samples": fft_d,
        "cpi_hop_samples": cpi_hop,
        "overlap_samples": fft_d - cpi_hop,
        "partial_window_policy": "retain_unprocessed",
        "phase_contract": (
            "independent_solve_phase_continuity_controlled_specular_only"
            if sequential
            else "sionna_intra_solve_phase_coherent"
        ),
    }
    processing = {
        "mode": mode,
        "strategy": strategy,
        "fft_size_doppler": fft_d,
        "result_stride_samples": cpi_hop,
        "product_semantics": (
            "cross_solve_cir_doppler_fft" if sequential else "cir_point_bin_slow_time_fft"
        ),
        "evidence_status": (
            "phase2_controlled_specular_validated_experimental"
            if sequential
            else "phase2_controlled_specular_validated"
        ),
    }
    cfg.update(
        {
            "fft_size_range": int(cfg.get("fft_size_range", 64)),
            "fft_size_doppler": fft_d,
            "sampling_frequency": float(cfg.get("sampling_frequency", 100e6)),
            "prf_hz": prf_hz,
            "range_mode": str(cfg.get("range_mode", "monostatic")),
            "carrier_freq_hz": float(cfg.get("carrier_freq_hz", 28e9)),
            "processing_mode": mode,
            "pre_cfar_doppler_edge_guard_bins": int(cfg.get("pre_cfar_doppler_edge_guard_bins", 2)),
            "pre_cfar_doppler_dc_guard_bins": int(cfg.get("pre_cfar_doppler_dc_guard_bins", 0)),
            "processing_recipe": {
                "window_range": str(cfg.get("window_range", "hann")),
                "window_doppler": str(cfg.get("window_doppler", "hann")),
            },
            "contract_version": 1,
            "acquisition": acquisition,
            "processing": processing,
            "topology": {
                "pair_count": 1,
                "pair_index": 0,
                "tx_device_count": 1,
                "rx_device_count": 1,
                "tx_elements": 1,
                "rx_elements": 1,
                "rx_array_processing": "siso",
            },
        }
    )
    return cfg


def _v4_sensing(*, frame_idx: int, sensing: dict) -> dict:
    """Upgrade a focused test payload to the mandatory v4 contracts."""

    upgraded = deepcopy(sensing)
    existing_temporal = upgraded.get("temporal")
    existing_truth = existing_temporal.get("truth") if isinstance(existing_temporal, dict) else None
    sample_time_s = float(upgraded.get("source_sample_time_s", frame_idx))
    slow_time = upgraded.pop("slow_time_matrix", None)
    sample_count = int(np.asarray(slow_time).shape[0]) if slow_time is not None else 1
    config = upgraded.get("config")
    authored_resolved = upgraded.get("config_resolved")
    config_inputs = dict(config) if isinstance(config, dict) else {}
    if isinstance(authored_resolved, dict):
        config_inputs.update(authored_resolved)
    if isinstance(authored_resolved, dict) and authored_resolved.get("contract_version") == 1:
        resolved_config = deepcopy(authored_resolved)
    else:
        resolved_config = _resolved_v4_config(
            config_inputs,
            per_pulse=slow_time is not None and sample_count == 1,
            requested_strategy=upgraded.get("processing_strategy"),
        )
    upgraded["config_resolved"] = resolved_config
    upgraded["config"] = deepcopy(resolved_config)
    upgraded["processing_strategy"] = resolved_config["processing"]["strategy"]
    prf_hz = float(resolved_config["prf_hz"])
    samples = [
        {
            "sample_index": sample_index,
            "sample_time_s": sample_time_s + sample_index / prf_hz,
            "valid": True,
            "phase_coherent": True,
            "source": {
                "source_generation_id": _TEST_GENERATION_ID,
                "scene_frame_id": frame_idx,
                "sample_offset": sample_index,
            },
        }
        for sample_index in range(sample_count)
    ]
    acquisition_id = str(
        uuid5(
            UUID(_TEST_GENERATION_ID),
            f"test-acquisition:{frame_idx}:{sample_time_s}:{sample_count}",
        )
    )
    upgraded["schema_version"] = SENSING_SCHEMA_VERSION
    upgraded.setdefault(
        "product_contract_version",
        SENSING_PRODUCT_CONTRACT_VERSION,
    )
    upgraded.setdefault("products", {})
    upgraded["temporal"] = build_sensing_temporal_contract(
        clock_id=_TEST_GENERATION_ID,
        acquisition_id=acquisition_id,
        acquisition_config_id=canonical_sensing_hash(
            {
                "config": upgraded.get("config", {}),
                "frame_idx": frame_idx,
                "sample_count": sample_count,
            }
        ),
        pair_id=canonical_sensing_hash(
            {"tx_sensor_id": "postprocessor-tx", "rx_sensor_id": "postprocessor-rx"}
        ),
        tx_sensor_id="postprocessor-tx",
        rx_sensor_id="postprocessor-rx",
        samples=samples,
        processing_run_id=_TEST_PROCESSING_RUN_ID,
        processing_recipe_id=canonical_processing_recipe_id(resolved_config["processing_recipe"]),
        reference_time_s=sample_time_s,
        reference_time_policy="sample_time",
        display_scene_frame_id=frame_idx,
        display_policy="exact_reference",
        display_reason="single-sample test fixture",
        truth=existing_truth,
    )
    queued_product = False
    cir = upgraded.pop("cir", None)
    if cir is not None and "cir_by_pair_rx_element" not in upgraded:
        cir_values = np.asarray(cir)
        if cir_values.ndim != 1:
            raise ValueError("Focused CIR fixtures must be one-dimensional")
        upgraded["cir_by_pair_rx_element"] = cir_values[np.newaxis, np.newaxis, ...]
        upgraded.setdefault(
            "range_axis_m",
            np.arange(cir_values.shape[0], dtype=np.float64),
        )
        upgraded["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
        add_product_spec(
            upgraded,
            data_key="cir_by_pair_rx_element",
            kind="delay_domain_cir",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                catalog_axis(
                    "rx_element",
                    size=1,
                    catalog_key="rx_element_axis",
                    coordinate_semantics="sionna_rx_element_index",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "range",
                    size=cir_values.shape[0],
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
        queued_product = True
    if slow_time is not None and "slow_time_by_pair_rx_element" not in upgraded:
        slow_values = np.asarray(slow_time)
        upgraded["slow_time_by_pair_rx_element"] = slow_values[np.newaxis, np.newaxis, ...]
        upgraded.setdefault(
            "range_axis_m",
            np.arange(slow_values.shape[1], dtype=np.float64),
        )
        upgraded["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
        add_product_spec(
            upgraded,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                catalog_axis(
                    "rx_element",
                    size=1,
                    catalog_key="rx_element_axis",
                    coordinate_semantics="sionna_rx_element_index",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "slow_time",
                    size=slow_values.shape[0],
                    unit="s",
                    coordinate_key="slow_time_axis_s",
                    coordinate_semantics="scenario_clock_sample_time",
                    reference_frame="scenario_clock",
                ),
                dense_axis(
                    "range",
                    size=slow_values.shape[1],
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
        queued_product = True

    legacy_rd = upgraded.pop("range_doppler_map", None)
    if (
        legacy_rd is not None
        and "range_doppler_by_pair" not in upgraded
        and "range_doppler_power_by_pair" not in upgraded
    ):
        rd_values = np.asarray(legacy_rd)
        upgraded.setdefault(
            "range_axis_m",
            np.arange(rd_values.shape[1], dtype=np.float64),
        )
        upgraded.setdefault(
            "doppler_axis_hz",
            np.arange(rd_values.shape[0], dtype=np.float64),
        )
        if np.iscomplexobj(rd_values):
            data_key = "range_doppler_by_pair"
            representation = "complex_voltage"
        else:
            data_key = "range_doppler_power_by_pair"
            representation = "power_linear"
        upgraded[data_key] = rd_values[np.newaxis, ...]
        add_product_spec(
            upgraded,
            data_key=data_key,
            kind="range_doppler",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                dense_axis(
                    "doppler_frequency",
                    size=rd_values.shape[0],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                ),
                dense_axis(
                    "range",
                    size=rd_values.shape[1],
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ),
            representation=representation,
            unit="1",
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
        queued_product = True
    if queued_product:
        materialize_product_specs(upgraded)
        if slow_time is None:
            upgraded.pop("slow_time_axis_s", None)
    return upgraded


def _rematerialize_range_doppler_descriptor(sensing: dict) -> None:
    """Rebind a test product after deliberately changing temporal identity."""

    data_key = (
        "range_doppler_by_pair"
        if "range_doppler_by_pair" in sensing
        else "range_doppler_power_by_pair"
    )
    values = np.asarray(sensing[data_key])
    sensing["products"] = {}
    add_product_spec(
        sensing,
        data_key=data_key,
        kind="range_doppler",
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=values.shape[1],
                unit="Hz",
                coordinate_key="doppler_axis_hz",
            ),
            dense_axis(
                "range",
                size=values.shape[2],
                unit="m",
                coordinate_key="range_axis_m",
            ),
        ),
        representation=(
            "complex_voltage" if data_key == "range_doppler_by_pair" else "power_linear"
        ),
        unit="1",
        combine={"policy": "siso_identity", "normalization": "identity"},
    )
    materialize_product_specs(sensing)
    sensing.pop("slow_time_axis_s", None)


def _with_single_target_truth(
    *,
    frame_idx: int,
    sensing: dict,
    target_range_m: float,
    target_speed_m_s: float,
) -> dict:
    """Attach one complete canonical target/state/truth fixture."""

    upgraded = _v4_sensing(frame_idx=frame_idx, sensing=sensing)
    catalog = build_target_catalog(
        [{"target_id": "fixture_target", "display_name": "Fixture target"}]
    )
    pair_id = upgraded["temporal"]["acquisition"]["pair_id"]
    reference_time_s = float(upgraded["temporal"]["result"]["reference_time_s"])
    upgraded["temporal"]["truth"] = {
        "reference_time_s": reference_time_s,
        "scene_frame_id": frame_idx,
        "policy": "authored_fixture",
        "source": "authored_test",
    }
    state_row = {
        "target_id": "fixture_target",
        "lifecycle": "active",
        "valid": True,
        "position_m": [target_range_m, 0.0, 0.0],
        "orientation_rad": None,
        "velocity_m_s": [target_speed_m_s, 0.0, 0.0],
        "invalid_reason": None,
    }
    upgraded["target_catalog"] = catalog
    upgraded["target_state"] = build_target_state(
        catalog,
        scene_frame_id=frame_idx,
        reference_time_s=reference_time_s,
        targets=[state_row],
    )
    upgraded["truth"] = build_target_truth(
        catalog,
        targets=[
            {
                **state_row,
                "pair_observables": [
                    {
                        "pair_id": pair_id,
                        "bistatic_path_length_m": 2.0 * target_range_m,
                        "monostatic_equivalent_range_m": target_range_m,
                        "path_closing_rate_m_s": 2.0 * target_speed_m_s,
                        "monostatic_equivalent_velocity_m_s": target_speed_m_s,
                        "bistatic_bisector_velocity_m_s": target_speed_m_s,
                        "velocity_valid": True,
                        "velocity_derivation_source": "analytic_velocity_projection",
                        "velocity_reference_dt_s": None,
                        "undefined_reason": None,
                    }
                ],
            }
        ],
    )
    return upgraded


def _make_slow_time_matrix(
    fft_d: int = 32,
    fft_r: int = 64,
    *,
    target_range_bin: int = 20,
    target_doppler_bin: int = 5,
    snr_linear: float = 100.0,
) -> np.ndarray:
    """Create a synthetic slow-time matrix with a single point target.

    The target is placed at a specific range bin with a Doppler shift
    corresponding to target_doppler_bin offset from DC.
    """
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal((fft_d, fft_r)) + 1j * rng.standard_normal((fft_d, fft_r))) * 0.01

    # Add a sinusoidal target in slow-time at the target range bin
    doppler_freq = target_doppler_bin / fft_d  # normalized frequency
    slow_time = np.arange(fft_d)
    signal = np.sqrt(snr_linear) * 0.01 * np.exp(2j * np.pi * doppler_freq * slow_time)
    noise[:, target_range_bin] += signal

    return noise


def _make_cir_sequence(
    n_pulses: int = 32,
    fft_r: int = 64,
    *,
    target_delay_bin: int = 20,
    sampling_frequency: float = 100e6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create a synthetic CIR sequence with a single target."""
    rng = np.random.default_rng(42)
    cir_list = []
    for i in range(n_pulses):
        delays = np.array([target_delay_bin / sampling_frequency])
        phase = np.exp(2j * np.pi * 0.15 * i)  # some Doppler shift
        amplitudes = np.array([0.5 * phase + 0.01 * rng.standard_normal()])
        cir_list.append((amplitudes, delays))
    return cir_list


def _make_standard_frame(*, frame_idx: int, sensing: dict | None = None) -> StandardMPCFrame:
    normalized_sensing = (
        deepcopy(sensing)
        if sensing and sensing.get("schema_version") == SENSING_SCHEMA_VERSION
        else (_v4_sensing(frame_idx=frame_idx, sensing=sensing) if sensing else {})
    )
    return StandardMPCFrame(
        frame_index=frame_idx,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.array([0, 1], dtype=np.int64),
        bounce_offsets=np.array([0, 0], dtype=np.int64),
        tx_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.array([[10.0, 0.0, 0.0]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        tx_names=("tx_0",),
        rx_names=("rx_0",),
        bounce_xyz_m=np.empty((0, 3), dtype=np.float32),
        interactions=np.empty((0,), dtype=np.uint8),
        material_ids=np.empty((0,), dtype=np.uint16),
        material_names=("",),
        material_itu_types=("",),
        delays_ns=np.array([1.0], dtype=np.float32),
        path_loss_db=np.array([2.0], dtype=np.float32),
        aoa_az_deg=np.array([3.0], dtype=np.float32),
        aoa_el_deg=np.array([4.0], dtype=np.float32),
        aod_az_deg=np.array([5.0], dtype=np.float32),
        aod_el_deg=np.array([6.0], dtype=np.float32),
        metric_valid_bits=np.array([0x3F], dtype=np.uint8),
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        sensing=normalized_sensing,
        provenance={"provider": "test", "frame_idx": frame_idx},
        timestamp_s=(
            float(normalized_sensing.get("source_sample_time_s", frame_idx))
            if normalized_sensing
            else None
        ),
    )


def _write_packed_frame_set(frames_dir: Path, *frames: StandardMPCFrame) -> None:
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id=_TEST_GENERATION_ID,
        compression="lzf",
    )
    for frame in frames:
        writer.append(frame)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id=_TEST_GENERATION_ID,
        frame_set_id="sensing-test-frame-set",
        chunks=[chunk],
        compression={"codec": "lzf", "shuffle": True},
        segmentation={"policy": "test"},
        provenance={"fixture": "sensing-postprocessor"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)


def _postprocessed_result(
    doppler_bins: int = 32,
    range_bins: int = 64,
    *,
    frame_idx: int = 0,
) -> dict:
    rd_cube = np.ones((doppler_bins, range_bins), dtype=np.complex64) * (1.0 + 2.0j)
    processing_recipe = {
        "tool": "shared.sensing_postprocessor",
        "input_kind": "test_fixture",
        "inherited_numeric_model": dict(SENSING_META["config"]),
        "source_processing": None,
        "source_recipe": None,
        "effective_overrides": {},
    }
    temporal = _v4_sensing(
        frame_idx=frame_idx,
        sensing={"config": dict(SENSING_META["config"])},
    )["temporal"]
    temporal["result"]["processing_recipe_id"] = canonical_sensing_hash(processing_recipe)
    return {
        "rd_map": np.abs(rd_cube) ** 2,
        "rd_cube": rd_cube,
        "range_axis": np.arange(range_bins, dtype=np.float64),
        "doppler_axis": np.arange(doppler_bins, dtype=np.float64),
        "detections": [],
        "config": dict(SENSING_META["config"]),
        "temporal": temporal,
        "processing_recipe_resolved": processing_recipe,
    }


def _complete_stage_artifacts(rd_power: np.ndarray) -> dict[str, np.ndarray]:
    """Return one representation-valid detector DAG aligned to ``rd_power``."""

    power = np.asarray(rd_power, dtype=np.float64)
    shape = power.shape
    return {
        "cfar_input_power": np.array(power, copy=True),
        "cfar_noise_floor": np.full(shape, 0.5, dtype=np.float64),
        "cfar_threshold": np.full(shape, 1.5, dtype=np.float64),
        "cfar_training_cell_count": np.full(shape, 8, dtype=np.int32),
        "cfar_eligible_mask": np.ones(shape, dtype=np.bool_),
        "cfar_mask": np.zeros(shape, dtype=np.bool_),
        "nms_mask": np.zeros(shape, dtype=np.bool_),
        "snr_map": np.full(shape, 3.0102999566, dtype=np.float64),
        "snr_valid_mask": np.ones(shape, dtype=np.bool_),
        "cluster_labels": np.full(shape, -1, dtype=np.int32),
    }


def test_attach_sensing_reuses_unchanged_compact_arrays() -> None:
    source = _make_standard_frame(frame_idx=2, sensing={"config": {}})
    sensing = {"schema_version": 1, "range_doppler_map": np.ones((2, 2))}

    derived = attach_sensing_to_standard_frame(
        source,
        sensing_block=sensing,
        frame_index=9,
    )

    assert derived.frame_index == 9
    assert derived.sensing is sensing
    assert derived.bounce_xyz_m is source.bounce_xyz_m
    assert derived.interactions is source.interactions
    assert derived.material_ids is source.material_ids
    assert derived.delays_ns is source.delays_ns


def test_reprocessed_block_drops_old_run_products_and_descriptors() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
            "range_profile": np.arange(64, dtype=np.float32),
        },
    )
    source_sensing.update(
        {
            "range_doppler_map_raw": np.full((32, 64), 91.0, dtype=np.float32),
            "range_doppler_aoa_az_map": np.full((32, 64), 92.0, dtype=np.float32),
            "range_doppler_cfar_mask": np.ones((32, 64), dtype=np.uint8),
            "cfar_metadata": {"sentinel": "old-cfar"},
            "detections": [{"sentinel": "old-detection"}],
            "detection_filter_metadata": {"sentinel": "old-filter"},
            "azimuth_axis_deg": [91.0, 92.0],
            "beamforming_mode": "old-mode",
            "postprocessing": {"sentinel": "old-run"},
        }
    )
    source_binding = {
        "scene_frame_id": 3,
        "policy": "source-policy",
        "reason": "source reason",
    }
    source_product_id = source_sensing["products"]["slow_time_by_pair_rx_element"]["product_id"]

    result = _postprocessed_result(frame_idx=4)
    reprocessed = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result,
        postprocessing_config=PostProcessingConfig(),
        input_kind="slow_time_by_pair_rx_element",
        source_display_binding=source_binding,
    )

    assert "slow_time_matrix" not in reprocessed
    assert "slow_time_by_pair_rx_element" not in reprocessed
    assert "range_profile" not in reprocessed
    assert "range_doppler_map" not in reprocessed
    assert "compatibility_views" not in reprocessed
    for stale_key in (
        "range_doppler_map_raw",
        "range_doppler_aoa_az_map",
        "range_doppler_cfar_mask",
        "cfar_metadata",
        "azimuth_axis_deg",
        "beamforming_mode",
    ):
        assert stale_key not in reprocessed
    assert reprocessed["detections"] == []
    assert reprocessed["detection_filter_metadata"] == {}
    assert reprocessed["postprocessing"]["source_display_binding"] == source_binding
    assert reprocessed["postprocessing"]["source_product_id"] == source_product_id
    assert set(reprocessed["products"]) == {"range_doppler_power_by_pair"}
    assert reprocessed["products"]["range_doppler_power_by_pair"]["lineage"][
        "source_product_ids"
    ] == [source_product_id]
    assert "sentinel" not in reprocessed["postprocessing"]
    assert reprocessed["config"] == reprocessed["config_resolved"]
    assert reprocessed["config_resolved"]["processing_mode"] == "offline_reprocess"
    assert (
        reprocessed["config_resolved"]["processing_recipe"] == result["processing_recipe_resolved"]
    )
    assert reprocessed["config_resolved"]["processing"]["strategy"] == (
        "shared_sensing_postprocessor"
    )
    assert reprocessed["config_resolved"]["window_range"] == "hann"
    assert reprocessed["config_resolved"]["window_doppler"] == "hann"
    assert reprocessed["processing_strategy"] == "shared_sensing_postprocessor"
    assert reprocessed["rd_source"] == "slow_time_by_pair_rx_element"


def test_reprocessed_block_rejects_partial_stage_set_and_never_publishes_legacy_angular() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    result = _postprocessed_result(frame_idx=4)
    result.update(
        {
            "range_doppler_aoa_az_map": np.zeros((32, 64), dtype=np.float32),
            "range_doppler_aoa_el_map": np.zeros((32, 64), dtype=np.float32),
            "range_doppler_azimuth_cube": np.zeros((8, 32, 64), dtype=np.float32),
            "azimuth_axis_deg": np.linspace(-45.0, 45.0, 8),
            "stage_artifacts": {
                "cfar_mask": np.ones((32, 64), dtype=np.uint8),
                "snr_map": np.ones((32, 64), dtype=np.float32),
            },
        }
    )

    with pytest.raises(ValueError, match="one complete stage set"):
        build_reprocessed_sensing_block(
            source_sensing=source_sensing,
            result=result,
            postprocessing_config=PostProcessingConfig(include_stage_artifacts=True),
            input_kind="slow_time_by_pair_rx_element",
        )


def test_reprocessed_block_persists_complete_typed_detector_stage_dag() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    result = _postprocessed_result(frame_idx=4)
    shape = np.asarray(result["rd_map"]).shape
    result["stage_artifacts"] = _complete_stage_artifacts(result["rd_map"])
    result["stage_artifacts"]["cfar_input_power"] = apply_pre_cfar_doppler_guards(
        result["rd_map"],
        edge_guard_bins=2,
        dc_guard_bins=0,
    )

    reprocessed = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result,
        postprocessing_config=PostProcessingConfig(include_stage_artifacts=True),
        input_kind="slow_time_by_pair_rx_element",
    )

    assert len(reprocessed["products"]) == 11
    for descriptor in reprocessed["products"].values():
        assert descriptor["lineage"]["processing_recipe_id"] == (
            result["temporal"]["result"]["processing_recipe_id"]
        )
    assert reprocessed["range_doppler_cfar_mask_by_pair"].shape == (1, *shape)
    assert reprocessed["range_doppler_cfar_mask_by_pair"].dtype == np.bool_
    assert reprocessed["range_doppler_cluster_labels_by_pair"].dtype == np.int32
    assert reprocessed["range_doppler_detection_input_power_by_pair"].dtype == np.float64
    input_processing = reprocessed["products"]["range_doppler_detection_input_power_by_pair"][
        "processing"
    ]
    assert input_processing["input_domain"] == "power_linear"
    assert input_processing["pre_cfar_doppler_guard"] == {
        "operation": "zero_selected_doppler_rows_before_detection",
        "edge_guard_bins_each_side": 2,
        "dc_guard_bins_each_side": 0,
        "fill_value": 0.0,
    }
    cfar_processing = reprocessed["products"]["range_doppler_cfar_threshold_power_by_pair"][
        "processing"
    ]
    assert cfar_processing["os_rank_fraction"] == 0.75
    assert cfar_processing["os_rank_rounding"] == (
        "nearest_integer_ties_to_even_of_fraction_times_k_minus_one"
    )
    assert cfar_processing["formula"] == "noise_power * threshold_scale"
    nms_processing = reprocessed["products"]["range_doppler_nms_mask_by_pair"]["processing"]
    assert nms_processing["suppression_radius_bins"] == {"doppler": 2, "range": 4}
    assert nms_processing["boundary_policy"] == "clip_without_wrap"
    snr_processing = reprocessed["products"]["range_doppler_snr_db_by_pair"]["processing"]
    assert snr_processing["validity"] == (
        "eligible_training_window_and_strictly_positive_noise_power"
    )
    assert snr_processing["finite_censor_floor_db"] == -300.0
    assert snr_processing["finite_censor_ceiling_db"] == 300.0
    cluster_processing = reprocessed["products"]["range_doppler_cluster_labels_by_pair"][
        "processing"
    ]
    assert cluster_processing["cluster_range_gate_m"] == 1.0
    assert cluster_processing["cluster_velocity_gate_m_s"] == 5.0
    assert cluster_processing["background_label"] == -1
    assert cluster_processing["missing_sklearn_fallback"] == (
        "connected_components_of_unit_radius_graph"
    )
    assert reprocessed["products"]["range_doppler_nms_mask_by_pair"]["lineage"][
        "source_product_ids"
    ]


def test_reprocessed_block_requires_exact_typed_source_descriptor() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    source_sensing["products"] = {}

    with pytest.raises(ValueError, match="has no exact source descriptor"):
        build_reprocessed_sensing_block(
            source_sensing=source_sensing,
            result=_postprocessed_result(frame_idx=4),
            postprocessing_config=PostProcessingConfig(),
            input_kind="slow_time_by_pair_rx_element",
        )


def test_reprocessed_detector_input_has_exact_float64_primary_parent() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    result = _postprocessed_result(frame_idx=4)
    precise_power = np.full(
        np.asarray(result["rd_map"]).shape,
        1.0 + 2.0**-30,
        dtype=np.float64,
    )
    assert not np.array_equal(
        precise_power,
        precise_power.astype(np.float32).astype(np.float64),
    )
    result["rd_map"] = precise_power
    result["stage_artifacts"] = _complete_stage_artifacts(precise_power)

    reprocessed = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result,
        postprocessing_config=PostProcessingConfig(
            include_stage_artifacts=True,
            pre_cfar_doppler_edge_guard_bins=0,
            pre_cfar_doppler_dc_guard_bins=0,
        ),
        input_kind="slow_time_by_pair_rx_element",
    )

    primary = reprocessed["range_doppler_power_by_pair"]
    detector_input = reprocessed["range_doppler_detection_input_power_by_pair"]
    assert primary.dtype == np.float64
    assert detector_input.dtype == np.float64
    np.testing.assert_array_equal(primary, detector_input)
    primary_id = reprocessed["products"]["range_doppler_power_by_pair"]["product_id"]
    assert reprocessed["products"]["range_doppler_detection_input_power_by_pair"]["lineage"][
        "source_product_ids"
    ] == [primary_id]


def test_reprocessed_recipes_preserve_same_typed_source_root_for_ab() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    source_product_id = source_sensing["products"]["slow_time_by_pair_rx_element"]["product_id"]

    def result_for_run(*, processing_run_id: str, threshold_scale: float) -> dict:
        result = _postprocessed_result(frame_idx=4)
        result["temporal"] = deepcopy(source_sensing["temporal"])
        recipe = deepcopy(result["processing_recipe_resolved"])
        recipe["effective_overrides"] = {"cfar_threshold_scale": threshold_scale}
        result["processing_recipe_resolved"] = recipe
        result["temporal"]["result"]["processing_run_id"] = processing_run_id
        result["temporal"]["result"]["processing_recipe_id"] = canonical_processing_recipe_id(
            recipe
        )
        return result

    left_sensing = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result_for_run(
            processing_run_id=_TEST_PROCESSING_RUN_ID,
            threshold_scale=3.0,
        ),
        postprocessing_config=PostProcessingConfig(cfar_threshold_scale=3.0),
        input_kind="slow_time_by_pair_rx_element",
    )
    right_sensing = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result_for_run(
            processing_run_id=_TEST_SECOND_PROCESSING_RUN_ID,
            threshold_scale=8.0,
        ),
        postprocessing_config=PostProcessingConfig(cfar_threshold_scale=8.0),
        input_kind="slow_time_by_pair_rx_element",
    )
    reference_time_s = float(left_sensing["temporal"]["result"]["reference_time_s"])
    context = ReplaySceneContext(
        storage_frame_id=4,
        bound_scene_frame_id=4,
        current_scene_frame_id=4,
        bound_scene_time_s=reference_time_s,
        current_scene_time_s=reference_time_s,
    )
    left = build_replay_event(
        _make_standard_frame(frame_idx=4, sensing=left_sensing),
        context,
    )
    right = build_replay_event(
        _make_standard_frame(frame_idx=4, sensing=right_sensing),
        context,
    )

    comparison = compare_replay_events(left, right)

    assert left.root_source_product_ids == (source_product_id,)
    assert right.root_source_product_ids == (source_product_id,)
    assert comparison.root_source_product_ids == (source_product_id,)


def test_reprocessed_typed_product_preserves_exact_axes_and_parent_lineage() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "config_resolved": dict(SENSING_META["config"]),
            "range_doppler_by_pair": np.full((1, 2, 3), 1.0 + 2.0j, dtype=np.complex64),
            "doppler_axis_hz": np.array([-125.5, 250.25], dtype=np.float64),
            "range_axis_m": np.array([0.0, 0.375, 1.25], dtype=np.float64),
        },
    )
    add_product_spec(
        source_sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler",
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=2,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
            ),
            dense_axis("range", size=3, unit="m", coordinate_key="range_axis_m"),
        ),
        representation="complex_voltage",
        unit="1",
    )
    materialize_product_specs(source_sensing)
    source_product_id = source_sensing["products"]["range_doppler_by_pair"]["product_id"]

    result = _postprocessed_result(doppler_bins=2, range_bins=3, frame_idx=4)
    result["doppler_axis"] = np.array([-9.25, 17.5], dtype=np.float64)
    result["range_axis"] = np.array([0.0, 0.4, 1.7], dtype=np.float64)
    reprocessed = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result,
        postprocessing_config=PostProcessingConfig(),
        input_kind="range_doppler_by_pair",
    )

    descriptor = reprocessed["products"]["range_doppler_power_by_pair"]
    assert descriptor["representation"] == "power_linear"
    assert descriptor["shape"] == [1, 2, 3]
    assert descriptor["lineage"]["source_product_ids"] == [source_product_id]
    assert [axis["name"] for axis in descriptor["axes"]] == [
        "pair",
        "doppler_frequency",
        "range",
    ]
    np.testing.assert_array_equal(
        reprocessed["range_doppler_power_by_pair"],
        np.full((1, 2, 3), 5.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(reprocessed["doppler_axis_hz"], [-9.25, 17.5])
    np.testing.assert_array_equal(reprocessed["range_axis_m"], [0.0, 0.4, 1.7])


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "sampling_frequency"),
        (None, "pre_cfar_doppler_edge_guard_bins"),
        ("processing_recipe", "window_range"),
        ("acquisition", "cpi_hop_samples"),
        ("processing", "result_stride_samples"),
        ("topology", "rx_array_processing"),
    ],
)
def test_reprocessed_writer_rejects_missing_exact_v4_execution_fields(
    section: str | None,
    field: str,
) -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    resolved = source_sensing["config_resolved"]
    if section is None:
        resolved.pop(field)
    else:
        resolved[section].pop(field)
    assert field in (
        source_sensing["config"] if section is None else source_sensing["config"][section]
    )

    with pytest.raises(ValueError, match=field):
        build_reprocessed_sensing_block(
            source_sensing=source_sensing,
            result=_postprocessed_result(frame_idx=4),
            postprocessing_config=PostProcessingConfig(),
            input_kind="slow_time_by_pair_rx_element",
        )


def test_incomplete_writer_rejects_missing_exact_v4_acquisition_field() -> None:
    source_sensing = _v4_sensing(
        frame_idx=0,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((1, 64), dtype=np.complex64),
        },
    )
    source_sensing["config_resolved"]["acquisition"].pop("cpi_hop_samples")
    assert "cpi_hop_samples" in source_sensing["config"]["acquisition"]

    with pytest.raises(ValueError, match="cpi_hop_samples"):
        build_incomplete_sensing_block(
            source_sensing=source_sensing,
            temporal=source_sensing["temporal"],
            postprocessing_config=PostProcessingConfig(),
        )


def test_reprocessed_writer_records_explicit_window_and_guard_overrides() -> None:
    source_sensing = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    source_sensing["config_resolved"].pop("pre_cfar_doppler_edge_guard_bins")
    source_sensing["config_resolved"].pop("pre_cfar_doppler_dc_guard_bins")
    result = _postprocessed_result(frame_idx=4)
    overrides = result["processing_recipe_resolved"]["effective_overrides"]
    overrides.update(
        {
            "window_range": "blackman",
            "window_doppler": "rect",
            "pre_cfar_doppler_edge_guard_bins": 7,
            "pre_cfar_doppler_dc_guard_bins": 3,
        }
    )
    result["temporal"]["result"]["processing_recipe_id"] = canonical_sensing_hash(
        result["processing_recipe_resolved"]
    )

    reprocessed = build_reprocessed_sensing_block(
        source_sensing=source_sensing,
        result=result,
        postprocessing_config=PostProcessingConfig(
            window_range="blackman",
            window_doppler="rect",
            pre_cfar_doppler_edge_guard_bins=7,
            pre_cfar_doppler_dc_guard_bins=3,
        ),
        input_kind="slow_time_by_pair_rx_element",
    )

    resolved = reprocessed["config_resolved"]
    assert resolved["window_range"] == "blackman"
    assert resolved["window_doppler"] == "rect"
    assert resolved["pre_cfar_doppler_edge_guard_bins"] == 7
    assert resolved["pre_cfar_doppler_dc_guard_bins"] == 3


# ---------------------------------------------------------------------------
# Tests: PostProcessingConfig
# ---------------------------------------------------------------------------


class TestPostProcessingConfig:
    def test_cli_help_names_relative_awgn_semantics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("sys.argv", ["shared.sensing.postprocessor", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 0
        help_text = " ".join(capsys.readouterr().out.split())
        assert "dimensionless target-relative complex AWGN" in help_text
        assert "not physical thermal or link-budget noise" in help_text
        assert "--threshold-sweep" in help_text
        assert "--roc" not in help_text

    def test_defaults(self):
        cfg = PostProcessingConfig()
        assert cfg.mti_enabled is True
        assert cfg.cfar_threshold_scale == 3.0
        assert cfg.cfar_edge_policy == "full_window"
        assert cfg.window_range is None
        assert cfg.window_doppler is None
        assert cfg.min_detection_range_m is None
        assert cfg.pre_cfar_doppler_edge_guard_bins is None
        assert cfg.pre_cfar_doppler_dc_guard_bins is None
        assert cfg.angle_peak_detection_enabled is False
        assert cfg.angle_max_peaks_per_rd_cell == 2
        assert cfg.angle_edge_exclusion_bins == 0
        assert cfg.angle_secondary_peak_max_gap_db == 3.0
        assert cfg.angle_secondary_peak_min_separation_bins == 2
        assert cfg.angle_secondary_peak_min_prominence_db == 3.0
        assert cfg.vote_fusion_enabled is False
        assert cfg.vote_min_votes == 2
        assert cfg.tracker_azimuth_gate_deg is None
        assert cfg.tracker_azimuth_cost_scale_deg is None
        assert cfg.include_stage_artifacts is False

    def test_custom_values(self):
        cfg = PostProcessingConfig(
            cfar_threshold_scale=5.0,
            mti_enabled=False,
            min_detection_range_m=1.0,
            max_detection_range_m=50.0,
        )
        assert cfg.cfar_threshold_scale == 5.0
        assert cfg.mti_enabled is False
        assert cfg.min_detection_range_m == 1.0
        assert cfg.max_detection_range_m == 50.0


# ---------------------------------------------------------------------------
# Tests: process_from_slow_time_matrix
# ---------------------------------------------------------------------------


class TestProcessFromSlowTimeMatrix:
    def test_basic_output_structure(self):
        pp = SensingPostProcessor()
        stm = _make_slow_time_matrix()
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        assert "rd_map" in result
        assert "range_axis" in result
        assert "doppler_axis" in result
        assert "detections" in result
        assert "config" in result
        assert result["detection_velocity_axis_label"] == "Radial Closing Velocity (m/s)"
        assert isinstance(result["detections"], list)

    def test_rd_map_shape(self):
        pp = SensingPostProcessor()
        stm = _make_slow_time_matrix(fft_d=32, fft_r=64)
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        assert result["rd_map"].shape == (32, 64)
        assert result["range_axis"].shape == (64,)
        assert result["doppler_axis"].shape == (32,)

    def test_mti_affects_output(self):
        stm = _make_slow_time_matrix()

        pp_mti = SensingPostProcessor(PostProcessingConfig(mti_enabled=True))
        pp_no_mti = SensingPostProcessor(PostProcessingConfig(mti_enabled=False))

        result_mti = pp_mti.process_from_slow_time_matrix(stm, SENSING_META)
        result_no_mti = pp_no_mti.process_from_slow_time_matrix(stm, SENSING_META)

        # MTI should change the RD map (removes zero-Doppler clutter)
        assert not np.allclose(result_mti["rd_map"], result_no_mti["rd_map"])

    def test_cfar_threshold_affects_detections(self):
        stm = _make_slow_time_matrix(snr_linear=100.0)

        pp_low = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=2.0))
        pp_high = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=50.0))

        result_low = pp_low.process_from_slow_time_matrix(stm, SENSING_META)
        result_high = pp_high.process_from_slow_time_matrix(stm, SENSING_META)

        # Higher threshold should yield fewer or equal detections
        assert len(result_high["detections"]) <= len(result_low["detections"])

    def test_invalid_shape_raises(self):
        pp = SensingPostProcessor()
        with pytest.raises(ValueError, match="2-D"):
            pp.process_from_slow_time_matrix(np.zeros(100), SENSING_META)

    def test_optional_stage_artifacts_are_exposed_on_demand(self, tmp_path: Path):
        pp = SensingPostProcessor(PostProcessingConfig(include_stage_artifacts=True))
        stm = _make_slow_time_matrix(snr_linear=400.0)
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        assert "detection_filter_metadata" in result
        assert "stage_artifacts" in result
        artifacts = result["stage_artifacts"]
        assert set(artifacts) == {
            "cfar_input_power",
            "cfar_input_power_db",
            "cfar_mask",
            "cfar_noise_floor",
            "cfar_threshold",
            "cfar_training_cell_count",
            "cfar_eligible_mask",
            "nms_mask",
            "snr_map",
            "snr_valid_mask",
            "cluster_labels",
        }
        assert artifacts["cluster_labels"].shape == result["rd_map"].shape

        saved = _save_stage_artifacts(tmp_path, result, frame_index=0)
        assert saved is not None
        with np.load(saved) as data:
            assert "cluster_labels" in data
            assert data["cluster_labels"].shape == result["rd_map"].shape

    def test_cfar_edge_policy_is_applied_and_exposed_exactly(self):
        pp = SensingPostProcessor(
            PostProcessingConfig(
                cfar_edge_policy="truncate",
                cfar_min_snr_db=0.0,
            )
        )

        result = pp.process_from_rd_map(
            np.ones((32, 64), dtype=np.float64),
            SENSING_META,
        )

        assert result["config"]["cfar_edge_policy"] == "truncate"
        assert result["detection_filter_metadata"]["cfar"]["edge_policy"] == "truncate"

    def test_skip_detection_returns_rd_only_without_stage_artifacts(self):
        pp = SensingPostProcessor(
            PostProcessingConfig(skip_detection=True, include_stage_artifacts=True)
        )
        stm = _make_slow_time_matrix(snr_linear=400.0)

        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        assert result["detections"] == []
        assert result["detection_filter_metadata"] == {}
        assert "stage_artifacts" not in result


# ---------------------------------------------------------------------------
# Tests: process_from_cir_frames
# ---------------------------------------------------------------------------


class TestProcessFromCirFrames:
    def test_basic_output(self):
        pp = SensingPostProcessor()
        cir_list = _make_cir_sequence()
        result = pp.process_from_cir_frames(cir_list, SENSING_META)

        assert "rd_map" in result
        assert result["rd_map"].shape[0] == 32  # fft_size_doppler
        assert result["rd_map"].shape[1] == 64  # fft_size_range

    def test_different_window_changes_output(self):
        cir_list = _make_cir_sequence()

        pp_hann = SensingPostProcessor(PostProcessingConfig(window_doppler="hann"))
        pp_rect = SensingPostProcessor(PostProcessingConfig(window_doppler="rect"))

        result_hann = pp_hann.process_from_cir_frames(cir_list, SENSING_META)
        result_rect = pp_rect.process_from_cir_frames(cir_list, SENSING_META)

        assert not np.allclose(result_hann["rd_map"], result_rect["rd_map"])


# ---------------------------------------------------------------------------
# Tests: process_from_rd_map
# ---------------------------------------------------------------------------


class TestProcessFromRdMap:
    def test_redetect_from_power_map(self):
        # First generate an RD map from STM
        pp = SensingPostProcessor()
        stm = _make_slow_time_matrix(snr_linear=200.0)
        original = pp.process_from_slow_time_matrix(stm, SENSING_META)

        # Now re-detect with different threshold
        pp2 = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=50.0))
        result = pp2.process_from_rd_map(original["rd_map"], SENSING_META)

        assert "detections" in result
        # Same RD map should produce fewer or equal detections with higher threshold
        assert len(result["detections"]) <= len(original["detections"])

    def test_rd_map_passthrough(self):
        pp = SensingPostProcessor()
        rd_map = np.random.default_rng(0).random((32, 64))
        result = pp.process_from_rd_map(rd_map, SENSING_META)

        # The power map should be passed through as-is
        np.testing.assert_array_equal(result["rd_map"], rd_map)

    def test_skip_detection_returns_rd_only_without_stage_artifacts(self):
        pp = SensingPostProcessor(
            PostProcessingConfig(skip_detection=True, include_stage_artifacts=True)
        )
        rd_map = np.random.default_rng(0).random((32, 64))

        result = pp.process_from_rd_map(rd_map, SENSING_META)

        assert result["detections"] == []
        assert result["detection_filter_metadata"] == {}
        assert "stage_artifacts" not in result

    def test_preserves_azimuth_artifacts_and_detection_angles(self):
        pp = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=2.0))
        stm = _make_slow_time_matrix(snr_linear=200.0)
        original = pp.process_from_slow_time_matrix(stm, SENSING_META)
        aoa_az_map = np.full_like(original["rd_map"], 18.0, dtype=np.float32)
        azimuth_cube = np.zeros((8, *original["rd_map"].shape), dtype=np.float32)
        azimuth_cube[4] = original["rd_map"].astype(np.float32)
        azimuth_axis_deg = np.linspace(-45.0, 45.0, 8, dtype=np.float32)

        result = pp.process_from_rd_map(
            original["rd_map"],
            {"config": dict(SENSING_META["config"]), "beamforming_mode": "rx_azimuth_fft"},
            aoa_az_map=aoa_az_map,
            azimuth_cube=azimuth_cube,
            azimuth_axis_deg=azimuth_axis_deg,
        )

        np.testing.assert_array_equal(result["range_doppler_aoa_az_map"], aoa_az_map)
        np.testing.assert_array_equal(result["range_doppler_azimuth_cube"], azimuth_cube)
        np.testing.assert_array_equal(result["azimuth_axis_deg"], azimuth_axis_deg)
        if result["detections"]:
            assert result["detections"][0]["aoa_az_deg"] == pytest.approx(18.0)

    @pytest.mark.parametrize(
        "config",
        [
            PostProcessingConfig(ghost_filter_azimuth_gate_deg=10.0),
            PostProcessingConfig(vote_fusion_enabled=True),
            PostProcessingConfig(angle_peak_detection_enabled=True),
        ],
    )
    def test_native_v4_angle_aware_requests_fail_closed(
        self,
        config: PostProcessingConfig,
    ) -> None:
        rd_map = np.ones((32, 64), dtype=np.float64)
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": rd_map,
            },
        )

        with pytest.raises(
            ValueError,
            match="typed P,E,A,D,R postprocessing not implemented; use rd or beam_scan detection",
        ):
            SensingPostProcessor(config).process_from_rd_map(
                rd_map,
                sensing,
                range_axis_m=sensing["range_axis_m"],
                doppler_axis_hz=sensing["doppler_axis_hz"],
            )

    def test_native_v4_normal_2d_request_ignores_legacy_angular_arguments(self) -> None:
        rd_map = np.ones((32, 64), dtype=np.float64)
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": rd_map,
            },
        )

        result = SensingPostProcessor().process_from_rd_map(
            rd_map,
            sensing,
            range_axis_m=sensing["range_axis_m"],
            doppler_axis_hz=sensing["doppler_axis_hz"],
            aoa_az_map=np.zeros_like(rd_map),
            azimuth_cube=np.zeros((8, *rd_map.shape)),
            azimuth_axis_deg=np.linspace(-45.0, 45.0, 8),
        )

        assert "range_doppler_aoa_az_map" not in result
        assert "range_doppler_azimuth_cube" not in result

    def test_vote_fusion_uses_azimuth_cube_and_exposes_vote_artifacts(self):
        rd_map = np.full((32, 64), 8.0, dtype=np.float64)
        azimuth_cube = np.ones((32, 32, 64), dtype=np.float32)
        azimuth_cube[13, 10, 20] += 100.0
        rd_map = azimuth_cube.sum(axis=0)
        azimuth_axis_deg = np.linspace(-45.0, 45.0, 32, dtype=np.float32)

        pp = SensingPostProcessor(
            PostProcessingConfig(
                cfar_threshold_scale=1.75,
                cfar_min_snr_db=0.0,
                vote_fusion_enabled=True,
                vote_min_votes=2,
                vote_pre_threshold_db=30.0,
                include_stage_artifacts=True,
            )
        )
        result = pp.process_from_rd_map(
            rd_map,
            SENSING_META,
            azimuth_cube=azimuth_cube,
            azimuth_axis_deg=azimuth_axis_deg,
        )

        assert result["detection_filter_metadata"]["detection_mode"] == "vote_fusion_3d"
        assert result["detection_filter_metadata"]["cfar"]["edge_policy"] == "full_window"
        assert result["detection_filter_metadata"]["projection_edge_policies"] == {
            "rd": "full_window",
            "raz": "full_window",
            "daz": "full_window",
        }
        assert result["detections"]
        det = result["detections"][0]
        assert det["azimuth_bin"] == 13
        assert det["vote_count"] == 3
        assert det["aoa_az_deg"] == pytest.approx(float(azimuth_axis_deg[13]))
        assert "stage_artifacts" in result
        assert result["stage_artifacts"]["vote_counts"].shape == azimuth_cube.shape
        assert result["stage_artifacts"]["vote_fused_mask"].shape == azimuth_cube.shape

    def test_vote_fusion_projects_configured_edge_policy_exactly(self):
        azimuth_cube = np.ones((8, 32, 64), dtype=np.float32)
        result = SensingPostProcessor(
            PostProcessingConfig(
                cfar_edge_policy="truncate",
                cfar_min_snr_db=0.0,
                vote_fusion_enabled=True,
            )
        ).process_from_rd_map(
            azimuth_cube.sum(axis=0),
            SENSING_META,
            azimuth_cube=azimuth_cube,
            azimuth_axis_deg=np.linspace(-45.0, 45.0, 8, dtype=np.float32),
        )

        assert result["detection_filter_metadata"]["cfar"]["edge_policy"] == "truncate"
        assert result["detection_filter_metadata"]["projection_edge_policies"] == {
            "rd": "truncate",
            "raz": "truncate",
            "daz": "truncate",
        }

    def test_angle_peak_detection_uses_azimuth_cube_and_exposes_angle_artifacts(self):
        # A positive background gives CFAR a real RD noise denominator while
        # keeping the target-cell azimuth-profile median below both peaks.
        azimuth_cube = np.ones((8, 32, 64), dtype=np.float32)
        azimuth_cube[1, 10, 20] = 100.0
        azimuth_cube[6, 10, 20] = 95.0
        rd_map = azimuth_cube.sum(axis=0)
        azimuth_axis_deg = np.linspace(-45.0, 45.0, 8, dtype=np.float32)

        pp = SensingPostProcessor(
            PostProcessingConfig(
                cfar_threshold_scale=1.5,
                cfar_min_snr_db=0.0,
                angle_peak_detection_enabled=True,
                angle_max_peaks_per_rd_cell=2,
                include_stage_artifacts=True,
            )
        )
        result = pp.process_from_rd_map(
            rd_map,
            SENSING_META,
            azimuth_cube=azimuth_cube,
            azimuth_axis_deg=azimuth_axis_deg,
        )

        assert result["detection_filter_metadata"]["detection_mode"] == "rd_angle_peaks_3d"
        assert len(result["detections"]) == 2
        assert sorted(det["azimuth_bin"] for det in result["detections"]) == [1, 6]
        assert "stage_artifacts" in result
        assert result["stage_artifacts"]["angle_candidate_mask"].shape == azimuth_cube.shape
        assert result["stage_artifacts"]["angle_peak_count_map"].shape == rd_map.shape

    def test_threshold_sweep_accepts_saved_rd_map(self):
        rd_map = np.zeros((32, 64), dtype=np.float64)
        rd_map[20, 12] = 100.0
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "range_doppler_map": rd_map,
                "config": dict(SENSING_META["config"]),
            },
        )
        frames = [{"_typed_sensing": sensing, "config": dict(SENSING_META["config"])}]

        sweep = sweep_thresholds(
            frames,
            threshold_range=(1.0, 2.0),
            threshold_steps=2,
            base_config=PostProcessingConfig(cfar_min_snr_db=0.0),
        )

        assert len(sweep) == 2
        assert all(point.evaluated_event_count == 1 for point in sweep)
        assert all(point.target_opportunity_count == 0 for point in sweep)
        assert all(point.pd is None for point in sweep)

    def test_threshold_sweep_rejects_deferred_typed_angular_request(self):
        rd_map = np.ones((32, 64), dtype=np.float64)
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "range_doppler_map": rd_map,
                "config": dict(SENSING_META["config"]),
            },
        )
        frames = [{"_typed_sensing": sensing}]

        with pytest.raises(
            ValueError,
            match="typed P,E,A,D,R postprocessing not implemented; use rd or beam_scan detection",
        ):
            sweep_thresholds(
                frames,
                threshold_steps=1,
                base_config=PostProcessingConfig(vote_fusion_enabled=True),
            )


def _evaluation_join_frame(frame_idx: int, target_range_m: float) -> dict:
    """Build one exact-identity source frame for evaluation-join tests."""

    sensing = _v4_sensing(
        frame_idx=frame_idx,
        sensing={"config": dict(SENSING_META["config"])},
    )
    return {
        "_typed_sensing": sensing,
        "_gt": {
            "target_ids": [f"walker.{frame_idx}"],
            "target_ranges": [target_range_m],
            "target_speeds": [0.0],
            "target_velocity_valid": [True],
        },
    }


def _evaluation_join_result(
    frame: dict,
    detection_range_m: float,
    *,
    processing_run_id: str = _TEST_PROCESSING_RUN_ID,
) -> dict:
    """Build one result occurrence keyed to a source frame."""

    temporal = deepcopy(frame["_typed_sensing"]["temporal"])
    temporal["result"]["processing_run_id"] = processing_run_id
    return {
        "temporal": temporal,
        "detections": [{"range_m": detection_range_m, "velocity_m_s": 0.0}],
    }


def _evaluate_join(join):
    """Evaluate an already identity-aligned test join."""

    per_event_gt = [(event.gt_ranges, event.gt_velocities) for event in join.events]
    per_event_detections = [
        (
            np.asarray([detection["range_m"] for detection in event.detections]),
            np.asarray([detection["velocity_m_s"] for detection in event.detections]),
        )
        for event in join.events
    ]
    return evaluate_scenario(per_event_gt, per_event_detections)


class TestEvaluationTemporalJoin:
    """Tests for full outer joins on exact sensing temporal identities."""

    def test_reordered_results_join_to_their_exact_truth_events(self):
        frame_a = _evaluation_join_frame(0, 10.0)
        frame_b = _evaluation_join_frame(1, 20.0)
        result_a = _evaluation_join_result(frame_a, 10.0)
        result_b = _evaluation_join_result(frame_b, 20.0)

        joined = _join_evaluation_events(
            [frame_a, frame_b],
            [result_b, result_a],
        )
        evaluation = _evaluate_join(joined)

        assert joined.missing_result_keys == ()
        assert joined.extra_result_keys == ()
        assert evaluation.total_tp == 2
        assert evaluation.total_fp == 0
        assert evaluation.total_fn == 0

    def test_missing_and_extra_results_are_explicit_and_do_not_shift(self):
        frame_a = _evaluation_join_frame(0, 10.0)
        frame_b = _evaluation_join_frame(1, 20.0)
        extra_frame = _evaluation_join_frame(2, 30.0)
        result_b = _evaluation_join_result(frame_b, 20.0)
        extra_result = _evaluation_join_result(extra_frame, 30.0)

        joined = _join_evaluation_events(
            [frame_a, frame_b],
            [extra_result, result_b],
        )
        evaluation = _evaluate_join(joined)

        assert len(joined.missing_result_keys) == 1
        assert len(joined.extra_result_keys) == 1
        assert evaluation.evaluated_event_count == 3
        assert evaluation.total_tp == 1
        assert evaluation.total_fp == 1
        assert evaluation.total_fn == 1

    def test_duplicate_frame_key_is_rejected(self):
        frame = _evaluation_join_frame(0, 10.0)

        with pytest.raises(ValueError, match="Duplicate sensing evaluation frame key"):
            _join_evaluation_events([frame, deepcopy(frame)], [])

    def test_duplicate_result_occurrence_is_rejected(self):
        frame = _evaluation_join_frame(0, 10.0)
        result = _evaluation_join_result(frame, 10.0)

        with pytest.raises(ValueError, match="Duplicate sensing evaluation result occurrence"):
            _join_evaluation_events([frame], [result, deepcopy(result)])

    def test_multiple_processing_runs_require_explicit_selection(self):
        frame = _evaluation_join_frame(0, 10.0)
        first = _evaluation_join_result(frame, 10.0)
        second = _evaluation_join_result(
            frame,
            10.0,
            processing_run_id="00000000-0000-4000-8000-000000000099",
        )

        with pytest.raises(ValueError, match="select one processing_run_id"):
            _join_evaluation_events([frame], [first, second])

    def test_incomplete_unprocessed_source_is_not_an_evaluated_event(self):
        frame = _evaluation_join_frame(0, 10.0)
        temporal = frame["_typed_sensing"]["temporal"]
        temporal["acquisition"]["completeness"] = "incomplete_unprocessed"

        joined = _join_evaluation_events([frame], [])

        assert joined.events == ()
        assert joined.missing_result_keys == ()

    def test_invalid_truth_velocity_skips_event_and_its_detections(self):
        frame = _evaluation_join_frame(0, 10.0)
        frame["_gt"]["target_speeds"] = [None]
        frame["_gt"]["target_velocity_valid"] = [False]
        result = _evaluation_join_result(frame, 10.0)

        joined = _join_evaluation_events([frame], [result])

        assert joined.events == ()
        assert len(joined.skipped_invalid_truth_keys) == 1
        assert joined.missing_result_keys == ()
        assert joined.extra_result_keys == ()
        metadata = _evaluation_join_metadata(joined)
        assert metadata["skipped_invalid_truth_event_count"] == 1
        assert metadata["skipped_invalid_truth_event_keys"] == [
            asdict(joined.skipped_invalid_truth_keys[0])
        ]

    def test_duplicate_key_is_rejected_across_valid_and_invalid_truth(self):
        valid = _evaluation_join_frame(0, 10.0)
        invalid = deepcopy(valid)
        invalid["_gt"]["target_speeds"] = [None]
        invalid["_gt"]["target_velocity_valid"] = [False]

        with pytest.raises(ValueError, match="Duplicate sensing evaluation frame key"):
            _join_evaluation_events([valid, invalid], [])


# ---------------------------------------------------------------------------
# Tests: schema-v4 persisted product boundary
# ---------------------------------------------------------------------------


class TestTypedPersistedProducts:
    def test_legacy_compatibility_array_is_not_a_persisted_input(self):
        with pytest.raises(ValueError, match="require schema-v4 typed products"):
            _select_typed_detection_input(
                {
                    "_typed_sensing": {
                        "products": {},
                        "slow_time_matrix": np.ones((2, 3), dtype=np.complex64),
                    }
                },
                rebuild_rd=False,
            )

    def test_power_linear_is_not_squared_and_exact_nonuniform_axes_survive(self):
        power = np.full((3, 4), 3.0, dtype=np.float32)
        range_axis = np.array([0.0, 0.4, 1.7, 4.25], dtype=np.float64)
        doppler_axis = np.array([-17.5, -1.25, 23.0], dtype=np.float64)
        config = {
            **SENSING_META["config"],
            "fft_size_range": 4,
            "fft_size_doppler": 3,
        }
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": config,
                "range_doppler_map": power,
                "range_axis_m": range_axis,
                "doppler_axis_hz": doppler_axis,
            },
        )

        selected = _select_typed_detection_input(
            {"_typed_sensing": sensing},
            rebuild_rd=False,
        )
        assert selected is not None
        assert selected.data_key == "range_doppler_power_by_pair"
        selected_power, selected_cube = _typed_range_doppler_arrays(selected)
        result = SensingPostProcessor(
            PostProcessingConfig(skip_detection=True)
        ).process_from_rd_map(
            selected_power,
            sensing,
            range_axis_m=selected.range_axis_m,
            doppler_axis_hz=selected.doppler_axis_hz,
            rd_cube=selected_cube,
        )

        assert selected_cube is None
        np.testing.assert_array_equal(result["rd_map"], power)
        np.testing.assert_array_equal(result["range_axis"], range_axis)
        np.testing.assert_array_equal(result["doppler_axis"], doppler_axis)

    def test_named_broadside_combiner_is_unitary_voltage_sum(self):
        config = {
            **SENSING_META["config"],
            "fft_size_range": 3,
            "fft_size_doppler": 2,
        }
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": config,
                "slow_time_matrix": np.ones((2, 3), dtype=np.complex64),
            },
        )
        sensing["products"] = {}
        sensing["slow_time_by_pair_rx_element"] = np.stack(
            (
                np.full((2, 3), 1.0 + 0.0j, dtype=np.complex64),
                np.full((2, 3), 3.0 + 0.0j, dtype=np.complex64),
            ),
            axis=0,
        )[np.newaxis, ...]
        sensing["rx_element_axis"] = [
            {"element_index": 0, "row": 0, "column": 0},
            {"element_index": 1, "row": 0, "column": 1},
        ]
        add_product_spec(
            sensing,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                catalog_axis(
                    "rx_element",
                    size=2,
                    catalog_key="rx_element_axis",
                    coordinate_semantics="sionna_rx_element_index",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "slow_time",
                    size=2,
                    unit="s",
                    coordinate_key="slow_time_axis_s",
                    reference_frame="scenario_clock",
                ),
                dense_axis(
                    "range",
                    size=3,
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            combine={
                "input_axis": "rx_element",
                "policy": "rx_broadside_unitary_voltage_sum",
                "normalization": "one_over_sqrt_element_count",
            },
        )
        materialize_product_specs(sensing)

        selected = _select_typed_detection_input(
            {"_typed_sensing": sensing},
            rebuild_rd=True,
        )
        assert selected is not None
        np.testing.assert_allclose(
            _combine_typed_slow_time(selected),
            np.full((2, 3), 4.0 / np.sqrt(2.0), dtype=np.complex64),
        )

    def test_unsupported_slow_time_combine_policy_fails_explicitly(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": {
                    **SENSING_META["config"],
                    "fft_size_range": 3,
                    "fft_size_doppler": 2,
                },
                "slow_time_matrix": np.ones((2, 3), dtype=np.complex64),
            },
        )
        sensing["products"] = {}
        add_product_spec(
            sensing,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                catalog_axis("rx_element", size=1, catalog_key="rx_element_axis"),
                dense_axis(
                    "slow_time",
                    size=2,
                    unit="s",
                    coordinate_key="slow_time_axis_s",
                ),
                dense_axis(
                    "range",
                    size=3,
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            combine={"policy": "selected_element", "normalization": "identity"},
        )
        materialize_product_specs(sensing)

        with pytest.raises(ValueError, match="unsupported combine policy"):
            _select_typed_detection_input(
                {"_typed_sensing": sensing},
                rebuild_rd=True,
            )

    def test_more_than_one_pair_fails_before_pair_selection(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": {
                    **SENSING_META["config"],
                    "fft_size_range": 4,
                    "fft_size_doppler": 3,
                },
                "range_doppler_map": np.ones((3, 4), dtype=np.float32),
            },
        )
        sensing["range_doppler_power_by_pair"] = np.ones((2, 3, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="P=1"):
            _select_typed_detection_input(
                {"_typed_sensing": sensing},
                rebuild_rd=False,
            )


# ---------------------------------------------------------------------------
# Tests: range gating
# ---------------------------------------------------------------------------


class TestRangeGating:
    def test_min_range_filters(self):
        pp = SensingPostProcessor(PostProcessingConfig(min_detection_range_m=100.0))
        stm = _make_slow_time_matrix(snr_linear=200.0)
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        # All detections should be >= 100 m
        for det in result["detections"]:
            assert det["range_m"] >= 100.0

    def test_max_range_filters(self):
        pp = SensingPostProcessor(PostProcessingConfig(max_detection_range_m=1.0))
        stm = _make_slow_time_matrix(snr_linear=200.0)
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        # All detections should be <= 1 m
        for det in result["detections"]:
            assert det["range_m"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: _resolve_meta
# ---------------------------------------------------------------------------


class TestResolveMeta:
    def test_from_config_key(self):
        meta = {"config": {"fft_size_range": 128, "prf_hz": 500.0}}
        cfg = SensingPostProcessor._resolve_meta(meta)
        assert cfg["fft_size_range"] == 128
        assert cfg["prf_hz"] == 500.0

    def test_from_config_resolved_key(self):
        meta = {
            "config_resolved": {"fft_size_range": 256, "sampling_frequency": 200e6},
            "config": {"fft_size_range": 128},
        }
        # config_resolved takes precedence
        cfg = SensingPostProcessor._resolve_meta(meta)
        assert cfg["fft_size_range"] == 256
        assert cfg["sampling_frequency"] == 200e6

    def test_preserves_source_processing_lineage(self):
        source_processing = {
            "mode": "coherent_cpi",
            "strategy": "coherent_cpi_fft",
        }
        cfg = SensingPostProcessor._resolve_meta(
            {
                "config_resolved": {
                    "processing_mode": "offline_reprocess",
                    "source_processing": source_processing,
                }
            }
        )

        assert cfg["source_processing"] == source_processing

    def test_fallback_defaults(self):
        cfg = SensingPostProcessor._resolve_meta({})
        assert cfg["fft_size_range"] == 256
        assert cfg["fft_size_doppler"] == 128
        assert cfg["range_mode"] == "bistatic"

    def test_native_v4_rejects_config_and_outer_alias_fallback(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        sensing.pop("config_resolved")
        sensing["sampling_frequency"] = 999e6

        with pytest.raises(ValueError, match="complete config_resolved"):
            SensingPostProcessor._resolve_meta(sensing)

    def test_native_v4_rejects_missing_exact_scalar_even_when_config_has_it(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        sensing["config_resolved"].pop("sampling_frequency")
        assert "sampling_frequency" in sensing["config"]

        with pytest.raises(ValueError, match="config_resolved.sampling_frequency"):
            SensingPostProcessor._resolve_meta(sensing)

    @pytest.mark.parametrize("section", ["acquisition", "processing", "topology"])
    def test_native_v4_requires_complete_execution_sections(self, section: str):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        sensing["config_resolved"].pop(section)

        with pytest.raises(ValueError, match=rf"config_resolved\.{section}"):
            SensingPostProcessor._resolve_meta(sensing)

    def test_native_v4_windows_have_no_hann_fallback(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        resolved = sensing["config_resolved"]
        resolved["processing_recipe"].pop("window_range")

        with pytest.raises(ValueError, match="window_range"):
            _native_v4_processing_windows(resolved)

    def test_native_v4_inherits_exact_windows_without_implicit_override(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": {
                    **SENSING_META["config"],
                    "window_range": "blackman",
                    "window_doppler": "rect",
                },
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        resolved = SensingPostProcessor._resolve_meta(sensing)

        assert _effective_processing_windows(resolved, PostProcessingConfig()) == (
            "blackman",
            "rect",
        )
        assert _effective_processing_windows(
            resolved,
            PostProcessingConfig(window_doppler="hann"),
        ) == ("blackman", "hann")

    def test_schema_less_direct_array_api_keeps_hann_window_defaults(self):
        resolved = SensingPostProcessor._resolve_meta(SENSING_META)

        assert _effective_processing_windows(resolved, PostProcessingConfig()) == (
            "hann",
            "hann",
        )

    def test_effective_recipe_records_explicit_postprocessing_overrides(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        postprocessing = PostProcessingConfig(
            window_range="blackman",
            window_doppler="rect",
            pre_cfar_doppler_edge_guard_bins=7,
            pre_cfar_doppler_dc_guard_bins=3,
        )

        recipe = _build_effective_processing_recipe(
            {"_sensing_json": sensing},
            postprocessing_config=postprocessing,
            input_kind="range_doppler_power_by_pair",
        )

        assert recipe["effective_overrides"]["window_range"] == "blackman"
        assert recipe["effective_overrides"]["window_doppler"] == "rect"
        assert recipe["effective_overrides"]["pre_cfar_doppler_edge_guard_bins"] == 7
        assert recipe["effective_overrides"]["pre_cfar_doppler_dc_guard_bins"] == 3


class TestPreCfarDopplerGuards:
    def test_guard_helper_zeros_edges_without_touching_dc(self):
        rd_power = np.ones((8, 4), dtype=np.float64)
        guarded = apply_pre_cfar_doppler_guards(
            rd_power,
            edge_guard_bins=2,
            dc_guard_bins=0,
        )

        assert np.allclose(guarded[:2], 0.0)
        assert np.allclose(guarded[-2:], 0.0)
        assert np.allclose(guarded[4], 1.0)

    def test_postprocessor_defaults_to_edge_guard_only_for_coherent_meta(self):
        pp = SensingPostProcessor()

        coherent_cfg = SensingPostProcessor._resolve_meta(
            {"config": {"processing_mode": "coherent_cpi"}}
        )
        snapshot_cfg = SensingPostProcessor._resolve_meta(
            {"config": {"processing_mode": "analytic_path_doppler_oracle"}}
        )

        assert pp._resolve_pre_cfar_guard_bins(coherent_cfg) == (2, 0)
        assert pp._resolve_pre_cfar_guard_bins(snapshot_cfg) == (0, 0)

    def test_repeat_reprocessing_inherits_coherent_guards(self):
        pp = SensingPostProcessor()
        repeated_cfg = SensingPostProcessor._resolve_meta(
            {
                "config_resolved": {
                    "processing_mode": "offline_reprocess",
                    "source_processing": {
                        "mode": "coherent_cpi",
                        "strategy": "coherent_cpi_fft",
                    },
                }
            }
        )

        assert pp._resolve_pre_cfar_guard_bins(repeated_cfg) == (2, 0)

    def test_native_v4_coherent_guards_require_storage_or_explicit_overrides(self):
        sensing = _v4_sensing(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "range_doppler_map": np.ones((32, 64), dtype=np.float32),
            },
        )
        for key in (
            "pre_cfar_doppler_edge_guard_bins",
            "pre_cfar_doppler_dc_guard_bins",
        ):
            sensing["config_resolved"].pop(key)
        cfg = SensingPostProcessor._resolve_meta(sensing)

        with pytest.raises(ValueError, match="pre_cfar_doppler_edge_guard_bins"):
            SensingPostProcessor()._resolve_pre_cfar_guard_bins(cfg)

        overridden = SensingPostProcessor(
            PostProcessingConfig(
                pre_cfar_doppler_edge_guard_bins=5,
                pre_cfar_doppler_dc_guard_bins=2,
            )
        )
        assert overridden._resolve_pre_cfar_guard_bins(cfg) == (5, 2)

    def test_guard_helper_dc_only(self):
        rd_power = np.ones((8, 4), dtype=np.float64)
        guarded = apply_pre_cfar_doppler_guards(rd_power, edge_guard_bins=0, dc_guard_bins=3)
        # DC bin = 4, guard covers bins 3..5 (3 bins centered on bin 4)
        dc_bin = 8 // 2  # = 4
        dc_lo = max(0, dc_bin - 3 // 2)  # = 3
        dc_hi = dc_lo + 3  # = 6
        assert np.allclose(guarded[dc_lo:dc_hi], 0.0)
        # Edges should be untouched
        assert np.allclose(guarded[0], 1.0)
        assert np.allclose(guarded[-1], 1.0)

    def test_guard_helper_edge_and_dc_combined(self):
        rd_power = np.ones((16, 4), dtype=np.float64)
        guarded = apply_pre_cfar_doppler_guards(rd_power, edge_guard_bins=2, dc_guard_bins=4)
        # Edge: first 2 and last 2 rows zeroed
        assert np.allclose(guarded[:2], 0.0)
        assert np.allclose(guarded[-2:], 0.0)
        # DC: center 4 bins zeroed (bins 6..9)
        dc_bin = 16 // 2  # = 8
        dc_lo = max(0, dc_bin - 4 // 2)  # = 6
        dc_hi = dc_lo + 4  # = 10
        assert np.allclose(guarded[dc_lo:dc_hi], 0.0)
        # A bin between edge and DC should be untouched
        assert np.allclose(guarded[3], 1.0)

    def test_guard_helper_edge_covers_entire_doppler(self):
        rd_power = np.ones((6, 4), dtype=np.float64)
        guarded = apply_pre_cfar_doppler_guards(rd_power, edge_guard_bins=3, dc_guard_bins=0)
        # 2 * 3 == 6 == fft_size_doppler → entire array zeroed
        assert np.allclose(guarded, 0.0)


# ---------------------------------------------------------------------------
# Tests: JSON serialization (for CLI output)
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_config_is_json_serializable(self):
        pp = SensingPostProcessor()
        stm = _make_slow_time_matrix()
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        # Config should be JSON-serializable
        json_str = json.dumps(result["config"])
        assert json_str

    def test_detections_are_json_serializable(self):
        pp = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=1.5))
        stm = _make_slow_time_matrix(snr_linear=500.0)
        result = pp.process_from_slow_time_matrix(stm, SENSING_META)

        # Detections should serialize without numpy type errors
        from shared.sensing.postprocessor import _json_default

        json_str = json.dumps(result["detections"], default=_json_default)
        assert json_str


class TestOutputValidation:
    def test_validate_postprocessed_result_rejects_axis_mismatch(self):
        bad_result = {
            "rd_map": np.zeros((4, 8), dtype=np.float32),
            "range_axis": np.arange(7, dtype=np.float32),
            "doppler_axis": np.arange(4, dtype=np.float32),
            "detections": [],
            "config": {},
        }

        with pytest.raises(ValueError, match="range_axis"):
            _validate_postprocessed_result(bad_result)

    def test_validate_postprocessed_result_ignores_internal_angular_arrays(self):
        result = {
            "rd_map": np.zeros((4, 8), dtype=np.float32),
            "range_axis": np.arange(8, dtype=np.float32),
            "doppler_axis": np.arange(4, dtype=np.float32),
            "detections": [],
            "config": {},
            "range_doppler_azimuth_cube": np.zeros((6, 5, 8), dtype=np.float32),
        }

        _validate_postprocessed_result(result)


class TestHdf5RoundTrip:
    @pytest.mark.parametrize(
        ("incompatible_args", "expected_label"),
        [
            (("--evaluate",), "--evaluate"),
            (("--report",), "--report"),
            (("--threshold-sweep",), "--threshold-sweep"),
            (("--figures",), "--figures/--figure"),
            (("--gif",), "GIF generation"),
            (("--micro-doppler",), "micro-Doppler/vertex overlay"),
            (
                ("--ghost-track-correlation", "0.85"),
                "batch track-correlation ghost pruning",
            ),
            (("--dry-run",), "--dry-run"),
        ],
    )
    def test_cli_streaming_rejects_cross_frame_lanes_before_source_or_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        incompatible_args: tuple[str, ...],
        expected_label: str,
    ) -> None:
        output_root = tmp_path / "must-not-exist"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(tmp_path / "missing-source"),
                "--streaming",
                "--output-dir",
                str(output_root),
                *incompatible_args,
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        error = capsys.readouterr().err
        assert "--streaming is a result-generation-only lane" in error
        assert expected_label in error
        assert not output_root.exists()

    def test_cli_streaming_rejects_per_pulse_grouping_before_output_claim(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        frames_dir, _source_ids, _source_times = self._write_tim_002_source(
            tmp_path,
            count=4,
        )
        output_root = tmp_path / "streaming-per-pulse"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--streaming",
                "--output-dir",
                str(output_root),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        assert "does not support per-pulse slow-time CPI grouping" in capsys.readouterr().err
        assert not output_root.exists()

    def test_cli_streaming_result_generation_matches_default_scientific_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir = tmp_path / "streaming-parity-source" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
            _make_standard_frame(
                frame_idx=1,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.full((32, 64), 2.0, dtype=np.float32),
                },
            ),
        )
        eager_root = tmp_path / "derived-eager"
        streaming_root = tmp_path / "derived-streaming"

        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(eager_root),
            ],
        )
        sensing_postprocessor_main()
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--streaming",
                "--output-dir",
                str(streaming_root),
            ],
        )
        sensing_postprocessor_main()

        eager = load_saved_results(eager_root)
        streaming = load_saved_results(streaming_root)
        assert len(eager) == len(streaming) == 2
        for eager_result, streaming_result in zip(eager, streaming):
            assert eager_result["_frame_index"] == streaming_result["_frame_index"]
            np.testing.assert_array_equal(eager_result["rd_map"], streaming_result["rd_map"])
            np.testing.assert_array_equal(
                eager_result["range_axis"], streaming_result["range_axis"]
            )
            np.testing.assert_array_equal(
                eager_result["doppler_axis"], streaming_result["doppler_axis"]
            )
            assert json.dumps(
                eager_result["detections"], sort_keys=True, allow_nan=True
            ) == json.dumps(streaming_result["detections"], sort_keys=True, allow_nan=True)
            assert eager_result["config"] == streaming_result["config"]
            assert eager_result["_range_mode"] == streaming_result["_range_mode"]
        assert load_frame_manifest(eager_root / "frames").frame_ids == (0, 1)
        assert load_frame_manifest(streaming_root / "frames").frame_ids == (0, 1)
        assert json.loads((eager_root / "postprocessing_config.json").read_text()) == json.loads(
            (streaming_root / "postprocessing_config.json").read_text()
        )

    def test_cli_persists_exact_cfar_edge_policy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir = tmp_path / "source-edge-policy" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
        )
        output_root = tmp_path / "derived-edge-policy"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--skip-detection",
                "--cfar-edge-policy",
                "truncate",
            ],
        )

        sensing_postprocessor_main()

        with (output_root / "postprocessing_config.json").open(encoding="utf-8") as handle:
            persisted_config = json.load(handle)
        assert persisted_config["cfar_edge_policy"] == "truncate"
        derived = _load_frames_sensing(output_root)
        recipe = derived[0]["_sensing_json"]["config_resolved"]["processing_recipe"]
        assert recipe["effective_overrides"]["cfar_edge_policy"] == "truncate"

    def test_reprocessing_complete_metadata_only_acquisition_creates_first_result(self) -> None:
        source_sensing = _v4_sensing(
            frame_idx=41,
            sensing={"config": dict(SENSING_META["config"])},
        )
        source_temporal = source_sensing["temporal"]
        source_temporal.pop("result")
        source_temporal.pop("display_binding")

        reprocessed = _build_reprocessed_temporal(
            {"_frame_index": 41, "_sensing_json": source_sensing},
            processing_run_id=_TEST_PROCESSING_RUN_ID,
            postprocessing_config=PostProcessingConfig(),
            input_kind="slow_time_matrix",
        )

        assert reprocessed["acquisition"] == source_temporal["acquisition"]
        assert reprocessed["result"]["reference_time_s"] == 41.0
        assert reprocessed["result"]["reference_time_policy"] == "cpi_center"
        assert reprocessed["display_binding"]["scene_frame_id"] == 41
        assert reprocessed["display_binding"]["policy"] == "exact_derived_scene"

    @staticmethod
    def _write_tim_002_source(
        tmp_path: Path,
        *,
        count: int = 8,
    ) -> tuple[Path, tuple[int, ...], tuple[float, ...]]:
        source_ids = (100, 101, 103, 104, 108, 109, 111, 112)[:count]
        source_times_s = tuple(index * 0.001 for index in range(len(source_ids)))
        cpi_size = 4
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        sensing_config = {
            **SENSING_META["config"],
            "fft_size_range": 8,
            "fft_size_doppler": cpi_size,
            "prf_hz": 1000.0,
        }
        source_frames = [
            replace(
                _make_standard_frame(
                    frame_idx=source_id,
                    sensing={
                        "config": dict(sensing_config),
                        "config_resolved": dict(sensing_config),
                        "fft_size_doppler": cpi_size,
                        "processing_strategy": "sequential_stft",
                        "source_sample_time_s": source_time_s,
                        "slow_time_matrix": np.full(
                            (1, sensing_config["fft_size_range"]),
                            complex(source_id, 0.0),
                            dtype=np.complex64,
                        ),
                    },
                ),
                tx_positions=np.array([[float(source_id), 0.0, 0.0]], dtype=np.float64),
            )
            for source_id, source_time_s in zip(source_ids, source_times_s)
        ]
        _write_packed_frame_set(frames_dir, *source_frames)
        return frames_dir, source_ids, source_times_s

    def test_tim_002_grouping_uses_explicit_time_and_preserves_opaque_ids(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, source_ids, source_times_s = self._write_tim_002_source(tmp_path)
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = frames[0]["_sensing_json"]["config_resolved"]["acquisition"]
        groups = _group_per_pulse_frames(
            frames,
            acquisition_plan=acquisition_plan,
        )
        complete_groups = groups[3:]

        memberships = tuple(
            tuple(
                sample["source"]["scene_frame_id"]
                for sample in group["_grouped_temporal"]["acquisition"]["samples"]
            )
            for group in complete_groups
        )
        sample_times = tuple(
            tuple(
                sample["sample_time_s"]
                for sample in group["_grouped_temporal"]["acquisition"]["samples"]
            )
            for group in complete_groups
        )
        reference_times = tuple(
            group["_grouped_temporal"]["reference_time_s"] for group in complete_groups
        )

        assert memberships == tuple(source_ids[start : start + 4] for start in range(5))
        assert sample_times == tuple(source_times_s[start : start + 4] for start in range(5))
        assert reference_times == pytest.approx((0.0015, 0.0025, 0.0035, 0.0045, 0.0055))
        assert np.diff(reference_times) == pytest.approx(np.full(4, 0.001))
        assert tuple(group["_frame_index"] for group in groups) == tuple(range(8))
        assert tuple(group["_source_frame_step"] for group in complete_groups) == (
            source_ids[1],
            source_ids[2],
            source_ids[3],
            source_ids[4],
            source_ids[5],
        )
        for group in groups:
            acquisition = group["_grouped_temporal"]["acquisition"]
            assert acquisition["tx_sensor_id"] == "postprocessor-tx"
            assert acquisition["rx_sensor_id"] == "postprocessor-rx"
            assert acquisition["pair_id"] == canonical_sensing_hash(
                {
                    "tx_sensor_id": "postprocessor-tx",
                    "rx_sensor_id": "postprocessor-rx",
                }
            )

    def test_v02_sequential_grouping_matches_online_hop_one_windows(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, source_ids, source_times_s = self._write_tim_002_source(tmp_path)
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = frames[0]["_sensing_json"]["config_resolved"]["acquisition"]

        groups = _group_per_pulse_frames(
            frames,
            acquisition_plan=acquisition_plan,
        )

        assert len(groups) == len(source_ids)
        acquisitions = [group["_grouped_temporal"]["acquisition"] for group in groups]
        assert [entry["completeness"] for entry in acquisitions[:4]] == [
            "incomplete_unprocessed",
            "incomplete_unprocessed",
            "incomplete_unprocessed",
            "complete",
        ]
        assert [
            sample["source"]["scene_frame_id"] for sample in acquisitions[3]["samples"]
        ] == list(source_ids[:4])
        assert [
            sample["source"]["scene_frame_id"] for sample in acquisitions[4]["samples"]
        ] == list(source_ids[1:5])
        assert [sample["sample_time_s"] for sample in acquisitions[4]["samples"]] == list(
            source_times_s[1:5]
        )
        assert all(
            sample["phase_coherent"] is False
            for acquisition in acquisitions
            for sample in acquisition["samples"]
        )
        assert len({entry["acquisition_config_id"] for entry in acquisitions}) == 1

    def test_v4_grouping_rejects_missing_cpi_hop_instead_of_defaulting(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, _source_ids, _source_times_s = self._write_tim_002_source(
            tmp_path,
            count=4,
        )
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = deepcopy(frames[0]["_sensing_json"]["config_resolved"]["acquisition"])
        acquisition_plan.pop("cpi_hop_samples")

        with pytest.raises(ValueError, match="missing fields: cpi_hop_samples"):
            _group_per_pulse_frames(frames, acquisition_plan=acquisition_plan)

    def test_grouping_rejects_samples_from_different_sensor_pairs(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, _source_ids, _source_times_s = self._write_tim_002_source(
            tmp_path,
            count=4,
        )
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = frames[0]["_sensing_json"]["config_resolved"]["acquisition"]
        acquisition = frames[2]["_sensing_json"]["temporal"]["acquisition"]
        acquisition["tx_sensor_id"] = "different-tx"
        acquisition["pair_id"] = canonical_sensing_hash(
            {"tx_sensor_id": "different-tx", "rx_sensor_id": acquisition["rx_sensor_id"]}
        )

        with pytest.raises(ValueError, match="same TX/RX pair identity"):
            _group_per_pulse_frames(frames, acquisition_plan=acquisition_plan)

    def test_grouped_cpi_omits_unbound_last_pulse_ground_truth(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, _source_ids, _source_times_s = self._write_tim_002_source(
            tmp_path,
            count=4,
        )
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = frames[0]["_sensing_json"]["config_resolved"]["acquisition"]
        frames[-1]["_gt"] = {"target_ranges": [99.0], "target_speeds": [9.0]}
        frames[-1]["_sensing_json"]["truth"] = {"targets": []}

        grouped = _group_per_pulse_frames(
            frames,
            acquisition_plan=acquisition_plan,
        )[-1]

        assert "_gt" not in grouped
        assert "truth" not in _source_sensing_payload(grouped)

    def test_grouping_uses_local_ids_for_nonmonotonic_source_scene_ids(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir, _source_ids, _source_times_s = self._write_tim_002_source(
            tmp_path,
            count=4,
        )
        frames = _load_frames_sensing(frames_dir)
        acquisition_plan = frames[0]["_sensing_json"]["config_resolved"]["acquisition"]
        for frame, opaque_scene_id in zip(frames, (10, 9, 8, 7)):
            sample = frame["_sensing_json"]["temporal"]["acquisition"]["samples"][0]
            sample["source"]["scene_frame_id"] = opaque_scene_id

        groups = _group_per_pulse_frames(frames, acquisition_plan=acquisition_plan)

        assert [group["_frame_index"] for group in groups] == [0, 1, 2, 3]
        assert [group["_source_frame_step"] for group in groups] == [100, 101, 101, 101]
        assert [group["_grouped_temporal"]["display_scene_frame_id"] for group in groups] == [
            10,
            9,
            9,
            9,
        ]

    def test_tim_002_reprocessing_round_trip_and_tracker_use_result_time(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir, source_ids, source_times_s = self._write_tim_002_source(tmp_path)
        loaded_source_frames = _load_frames_sensing(frames_dir)
        acquisition_plan = loaded_source_frames[0]["_sensing_json"]["config_resolved"][
            "acquisition"
        ]
        expected_grouped_inputs = _group_per_pulse_frames(
            loaded_source_frames,
            acquisition_plan=acquisition_plan,
        )
        expected_source_product_ids = [
            group["_typed_sensing"]["products"]["slow_time_by_pair_rx_element"]["product_id"]
            for group in expected_grouped_inputs
            if group["_grouped_temporal"]["acquisition"]["completeness"] == "complete"
        ]
        source_hashes_before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frames_dir.glob("*.h5")
        }
        tracker_update_times_s: list[float] = []

        def _timestamp_only_detection(self, _rd_result, _cfg, *, update_time_s=None, **_kwargs):
            self._ensure_tracker()
            assert self._tracker is not None
            detections, metadata = self._tracker.filter_detections(
                [],
                update_time_s=update_time_s,
            )
            tracker_update_times_s.append(float(metadata["update_time_s"]))
            return DetectionPipelineResult(detections=detections, metadata=metadata)

        monkeypatch.setattr(SensingPostProcessor, "_detect", _timestamp_only_detection)
        output_root = tmp_path / "derived"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--tracker",
                "kalman",
            ],
        )

        sensing_postprocessor_main()

        derived_frames = _load_frames_sensing(output_root)
        temporals = [frame["_sensing_json"]["temporal"] for frame in derived_frames]
        complete_frames = [
            frame for frame in derived_frames if "result" in frame["_sensing_json"]["temporal"]
        ]
        complete_temporals = [frame["_sensing_json"]["temporal"] for frame in complete_frames]
        round_trip_membership = tuple(
            tuple(
                sample["source"]["scene_frame_id"] for sample in temporal["acquisition"]["samples"]
            )
            for temporal in complete_temporals
        )
        round_trip_times = tuple(
            tuple(sample["sample_time_s"] for sample in temporal["acquisition"]["samples"])
            for temporal in complete_temporals
        )
        source_hashes_after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frames_dir.glob("*.h5")
        }

        assert tuple(frame["_frame_index"] for frame in derived_frames) == tuple(range(8))
        assert len(complete_frames) == 5
        assert all("result" not in temporal for temporal in temporals[:3])
        assert all(
            frame["_frame_index"] == temporal["display_binding"]["scene_frame_id"]
            for frame, temporal in zip(complete_frames, complete_temporals)
        )
        assert (
            tuple(
                frame["_sensing_json"]["postprocessing"]["source_display_binding"]["scene_frame_id"]
                for frame in complete_frames
            )
            == source_ids[1:6]
        )
        assert tuple(float(frame["tx_positions"][0, 0]) for frame in complete_frames) == tuple(
            float(source_id) for source_id in source_ids[1:6]
        )
        assert tuple(frame["_frame_timestamp_s"] for frame in complete_frames) == pytest.approx(
            source_times_s[1:6]
        )
        assert round_trip_membership == tuple(source_ids[start : start + 4] for start in range(5))
        assert round_trip_times == tuple(source_times_s[start : start + 4] for start in range(5))
        assert tracker_update_times_s == pytest.approx((0.0015, 0.0025, 0.0035, 0.0045, 0.0055))
        assert {temporal["result"]["processing_run_id"] for temporal in complete_temporals} == {
            complete_temporals[0]["result"]["processing_run_id"]
        }
        assert all(
            temporal["acquisition"]["pair_id"]
            == canonical_sensing_hash(
                {
                    "tx_sensor_id": "postprocessor-tx",
                    "rx_sensor_id": "postprocessor-rx",
                }
            )
            for temporal in temporals
        )
        for derived_frame, source_product_id in zip(
            complete_frames,
            expected_source_product_ids,
        ):
            sensing = derived_frame["_sensing_json"]
            assert set(sensing["products"]) == {"range_doppler_power_by_pair"}
            assert "slow_time_by_pair_rx_element" not in derived_frame
            assert "slow_time_matrix" not in derived_frame
            assert "range_doppler_map" not in derived_frame
            assert sensing["postprocessing"]["source_product_id"] == source_product_id
            assert sensing["products"]["range_doppler_power_by_pair"]["lineage"][
                "source_product_ids"
            ] == [source_product_id]
        assert source_hashes_after == source_hashes_before

    def test_reprocessing_copies_earlier_bound_scene_into_local_output_frame(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir = tmp_path / "source-later-result" / "frames"
        frames_dir.mkdir(parents=True)
        sensing_config = dict(SENSING_META["config"])
        bound_frame = replace(
            _make_standard_frame(
                frame_idx=10,
                sensing={
                    "config": sensing_config,
                    "range_doppler_map": np.ones((32, 64), dtype=np.complex64),
                },
            ),
            tx_positions=np.array([[10.0, 1.0, 0.0]], dtype=np.float64),
            timestamp_s=10.25,
        )
        later_frame = _make_standard_frame(
            frame_idx=20,
            sensing={
                "config": sensing_config,
                "range_doppler_map": np.ones((32, 64), dtype=np.complex64),
            },
        )
        later_sensing = deepcopy(later_frame.sensing)
        later_sensing["temporal"]["display_binding"]["scene_frame_id"] = 10
        later_frame = replace(
            later_frame,
            sensing=later_sensing,
            tx_positions=np.array([[20.0, 2.0, 0.0]], dtype=np.float64),
            timestamp_s=20.75,
        )
        _write_packed_frame_set(frames_dir, bound_frame, later_frame)
        output_root = tmp_path / "derived-later-result"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--skip-detection",
            ],
        )

        sensing_postprocessor_main()

        derived = _load_frames_sensing(output_root)
        assert [frame["_frame_index"] for frame in derived] == [10, 20]
        later_output = derived[1]
        np.testing.assert_array_equal(
            later_output["tx_positions"],
            bound_frame.tx_positions,
        )
        assert later_output["_frame_timestamp_s"] == pytest.approx(bound_frame.timestamp_s)
        temporal = later_output["_sensing_json"]["temporal"]
        assert temporal["display_binding"]["scene_frame_id"] == 20
        assert temporal["result"]["reference_time_s"] == 20.0
        assert (
            later_output["_sensing_json"]["postprocessing"]["source_display_binding"][
                "scene_frame_id"
            ]
            == 10
        )
        assert "truth" not in later_output["_sensing_json"]

    @pytest.mark.parametrize("changed_identity", ["clock", "pair"])
    def test_tracking_rejects_clock_or_pair_change_between_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        changed_identity: str,
    ) -> None:
        frames_dir = tmp_path / f"source-tracker-{changed_identity}" / "frames"
        frames_dir.mkdir(parents=True)
        frames = [
            _make_standard_frame(
                frame_idx=frame_idx,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.complex64),
                },
            )
            for frame_idx in (0, 1)
        ]
        second_sensing = deepcopy(frames[1].sensing)
        if changed_identity == "clock":
            changed_clock_id = "00000000-0000-4000-8000-000000000099"
            second_sensing["temporal"]["clock"]["clock_id"] = changed_clock_id
            for sample in second_sensing["temporal"]["acquisition"]["samples"]:
                sample["source"]["source_generation_id"] = changed_clock_id
        else:
            acquisition = second_sensing["temporal"]["acquisition"]
            acquisition["tx_sensor_id"] = "postprocessor-other-tx"
            acquisition["pair_id"] = canonical_sensing_hash(
                {
                    "tx_sensor_id": acquisition["tx_sensor_id"],
                    "rx_sensor_id": acquisition["rx_sensor_id"],
                }
            )
        _rematerialize_range_doppler_descriptor(second_sensing)
        frames[1] = replace(frames[1], sensing=second_sensing)
        _write_packed_frame_set(frames_dir, *frames)
        output_root = tmp_path / f"derived-tracker-{changed_identity}"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--tracker",
                "kalman",
                "--skip-detection",
            ],
        )

        with pytest.raises(ValueError, match="clock or sensor-pair changes"):
            sensing_postprocessor_main()

    def test_sequential_warmup_cpi_is_retained_without_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir, source_ids, _source_times_s = self._write_tim_002_source(
            tmp_path,
            count=6,
        )
        output_root = tmp_path / "derived-partial"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--skip-detection",
            ],
        )

        sensing_postprocessor_main()

        derived = _load_frames_sensing(output_root)
        assert len(derived) == 6
        partial_temporal = derived[2]["_sensing_json"]["temporal"]
        partial_acquisition = partial_temporal["acquisition"]
        assert partial_acquisition["completeness"] == "incomplete_unprocessed"
        assert [sample["valid"] for sample in partial_acquisition["samples"]] == [
            True,
            True,
            True,
            False,
        ]
        assert [
            sample["source"]["scene_frame_id"]
            for sample in partial_acquisition["samples"]
            if sample["valid"]
        ] == list(source_ids[:3])
        assert "result" not in partial_temporal
        assert len(load_saved_results(output_root)) == 3

    def test_ungrouped_incomplete_acquisition_is_retained_without_processing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frames_dir = tmp_path / "source-direct-partial" / "frames"
        frames_dir.mkdir(parents=True)
        sample_rate_hz = 1000.0
        samples = [
            {
                "sample_index": sample_index,
                "sample_time_s": 31.0 + sample_index / sample_rate_hz,
                "valid": sample_index < 2,
                "phase_coherent": sample_index < 2,
                **(
                    {
                        "source": {
                            "source_generation_id": _TEST_GENERATION_ID,
                            "scene_frame_id": 31,
                            "sample_offset": sample_index,
                        }
                    }
                    if sample_index < 2
                    else {"invalid_reason": "not_observed_before_end_of_run"}
                ),
            }
            for sample_index in range(4)
        ]
        temporal = build_sensing_temporal_contract(
            clock_id=_TEST_GENERATION_ID,
            acquisition_id=str(uuid5(UUID(_TEST_GENERATION_ID), "direct-partial:31")),
            acquisition_config_id=canonical_sensing_hash({"kind": "direct-partial", "samples": 4}),
            pair_id=canonical_sensing_hash(
                {"tx_sensor_id": "postprocessor-tx", "rx_sensor_id": "postprocessor-rx"}
            ),
            tx_sensor_id="postprocessor-tx",
            rx_sensor_id="postprocessor-rx",
            samples=samples,
            completeness="incomplete_unprocessed",
        )
        resolved_config = _resolved_v4_config(
            {**SENSING_META["config"], "fft_size_doppler": 4},
            per_pulse=False,
        )
        source_sensing = {
            "schema_version": SENSING_SCHEMA_VERSION,
            "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
            "products": {},
            "temporal": temporal,
            "config": deepcopy(resolved_config),
            "config_resolved": resolved_config,
            "processing_strategy": "coherent_cpi_fft",
        }
        source_frame = replace(
            _make_standard_frame(frame_idx=31),
            sensing=source_sensing,
            timestamp_s=31.0,
        )
        _write_packed_frame_set(frames_dir, source_frame)
        output_root = tmp_path / "derived-direct-partial"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--skip-detection",
            ],
        )

        sensing_postprocessor_main()

        derived = _load_frames_sensing(output_root)
        assert [frame["_frame_index"] for frame in derived] == [31]
        persisted = derived[0]["_sensing_json"]
        persisted_acquisition = deepcopy(persisted["temporal"]["acquisition"])
        for sample in persisted_acquisition["samples"]:
            source = sample.get("source")
            if source is not None:
                assert source.pop("source_frame_set_id") == "sensing-test-frame-set"
        assert persisted_acquisition == temporal["acquisition"]
        assert "result" not in persisted["temporal"]
        assert "display_binding" not in persisted["temporal"]
        assert "slow_time_matrix" not in derived[0]
        assert persisted["postprocessing"]["status"] == "not_processed"
        assert load_saved_results(output_root) == []

    def test_cli_dry_run_creates_no_analysis_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "slow_time_matrix": _make_slow_time_matrix().astype(np.complex64),
                },
            ),
        )
        output_root = tmp_path / "scenario" / "analysis" / "dry-run"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--dry-run",
            ],
        )

        sensing_postprocessor_main()

        assert not output_root.exists()

    @pytest.mark.parametrize("flag", ["--vote-fusion", "--angle-peak-detection"])
    @pytest.mark.parametrize("cfr", [False, True])
    def test_cli_rejects_deferred_typed_angular_processing_before_output_claim(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        flag: str,
        cfr: bool,
    ) -> None:
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
        )
        output_root = tmp_path / "scenario" / "analysis" / "typed-angle"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                flag,
                *(["--frequency-window", "hann"] if cfr else []),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        assert "typed P,E,A,D,R postprocessing not implemented; use rd or beam_scan detection" in (
            capsys.readouterr().err
        )
        assert not output_root.exists()

    def test_cli_does_not_claim_output_without_detection_inputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "cir": np.ones(64, dtype=np.complex64),
                },
            ),
        )
        output_root = tmp_path / "analysis" / "no-detection-input"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        assert not output_root.exists()

    def test_cli_validates_full_source_provider_before_claiming_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
        )
        output_root = tmp_path / "analysis" / "unreadable-source"
        provider_events: list[str] = []

        class _UnreadableSourceProvider:
            def __init__(self, *_args, **_kwargs):
                provider_events.append("open")

            def load_frame(self, step: int):
                provider_events.append(f"load:{step}")
                raise RuntimeError("full source frame is unreadable")

            def close(self):
                provider_events.append("close")

        monkeypatch.setattr(
            "shared.sensing.postprocessor.Hdf5Provider",
            _UnreadableSourceProvider,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
            ],
        )

        with pytest.raises(RuntimeError, match="full source frame is unreadable"):
            sensing_postprocessor_main()

        assert provider_events == ["open", "load:0", "close"]
        assert not output_root.exists()

    def test_micro_doppler_only_does_not_claim_output_without_samples(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
        )
        output_root = tmp_path / "analysis" / "no-micro-doppler-samples"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
                "--skip-detection",
                "--micro-doppler",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        assert not output_root.exists()

    def test_cli_publishes_after_source_preflight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float32),
                },
            ),
        )
        output_root = tmp_path / "analysis" / "valid-run"
        monkeypatch.setattr(
            "sys.argv",
            [
                "shared.sensing.postprocessor",
                str(frames_dir),
                "--output-dir",
                str(output_root),
            ],
        )

        sensing_postprocessor_main()

        manifest = load_frame_manifest(output_root / "frames")
        assert manifest.frame_ids == (0,)

    def test_cli_requires_output_for_frame_producing_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(frame_idx=0, sensing=SENSING_META),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["shared.sensing.postprocessor", str(frames_dir)],
        )

        with pytest.raises(SystemExit) as exc_info:
            sensing_postprocessor_main()

        assert exc_info.value.code == 2
        assert not (frames_dir / "reprocessed").exists()

    def test_analysis_paths_require_explicit_absent_nonoverlapping_root(
        self,
        tmp_path: Path,
    ):
        frames_dir = tmp_path / "scenario" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(frame_idx=0, sensing=SENSING_META),
        )

        assert _resolve_source_frames_dir(frames_dir) == frames_dir.resolve()
        assert _resolve_source_frames_dir(frames_dir.parent) == frames_dir.resolve()

        output_root = tmp_path / "scenario" / "analysis" / "cfar_baseline"
        assert (
            _analysis_output_root(frames_dir, output_root, existing=False) == output_root.resolve()
        )
        assert not output_root.exists()
        _claim_analysis_root(output_root)
        assert output_root.is_dir()

        with pytest.raises(ValueError, match="explicit --output-dir"):
            _analysis_output_root(frames_dir, None, existing=False)
        with pytest.raises(FileExistsError, match="new absent analysis root"):
            _analysis_output_root(frames_dir, output_root, existing=False)
        with pytest.raises(ValueError, match="must not equal, contain, or be inside"):
            _analysis_output_root(frames_dir, frames_dir / "result", existing=False)
        with pytest.raises(ValueError, match="must not equal, contain, or be inside"):
            _analysis_output_root(frames_dir, frames_dir.parent, existing=False)

    def test_gif_only_requires_completed_existing_analysis_root(self, tmp_path: Path):
        source_frames = tmp_path / "scenario" / "frames"
        source_frames.mkdir(parents=True)
        _write_packed_frame_set(
            source_frames,
            _make_standard_frame(frame_idx=0, sensing=SENSING_META),
        )
        analysis_root = tmp_path / "scenario" / "analysis" / "run"
        derived_frames = analysis_root / "frames"
        derived_frames.mkdir(parents=True)
        _write_packed_frame_set(
            derived_frames,
            _make_standard_frame(frame_idx=0, sensing=SENSING_META),
        )

        assert (
            _analysis_output_root(source_frames, analysis_root, existing=True)
            == analysis_root.resolve()
        )

    def test_analysis_output_refuses_exact_symlink_entries(self, tmp_path: Path):
        source_frames = tmp_path / "scenario" / "frames"
        source_frames.mkdir(parents=True)
        _write_packed_frame_set(
            source_frames,
            _make_standard_frame(frame_idx=0, sensing=SENSING_META),
        )
        real_root = tmp_path / "real-analysis"
        real_root.mkdir()
        linked_root = tmp_path / "linked-analysis"
        try:
            linked_root.symlink_to(real_root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks are unavailable: {exc}")

        with pytest.raises(FileExistsError, match="new absent analysis root"):
            _analysis_output_root(source_frames, linked_root, existing=False)
        with pytest.raises(ValueError, match="must be a real directory"):
            _analysis_output_root(source_frames, linked_root, existing=True)

    def test_load_frames_sensing_reads_gt_from_packed_metadata(self, tmp_path: Path):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        sensing = _with_single_target_truth(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "config_resolved": {**SENSING_META["config"], "rng_seed": 7},
                "slow_time_matrix": _make_slow_time_matrix().astype(np.complex64),
            },
            target_range_m=12.5,
            target_speed_m_s=1.25,
        )
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(frame_idx=0, sensing=sensing),
        )

        frames = _load_frames_sensing(frames_dir)

        assert len(frames) == 1
        assert frames[0]["_frame_index"] == 0
        assert frames[0]["_gt"]["target_ids"] == ["fixture_target"]
        assert frames[0]["_gt"]["target_ranges"] == [12.5]
        assert frames[0]["_gt"]["target_speeds"] == [1.25]

    def test_packed_writer_rejects_undeclared_v4_azimuth_artifacts(self, tmp_path: Path):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        sensing = {
            "config": dict(SENSING_META["config"]),
            "config_resolved": {**SENSING_META["config"], "rng_seed": 11},
            "range_doppler_map": np.ones((32, 64), dtype=np.complex64),
            "range_doppler_aoa_az_map": np.full((32, 64), 7.5, dtype=np.float32),
            "range_doppler_azimuth_cube": np.zeros((8, 32, 64), dtype=np.float32),
            "beamforming_mode": "rx_azimuth_fft",
        }
        with pytest.raises(ValueError, match="forbids undeclared numeric payloads"):
            _write_packed_frame_set(
                frames_dir,
                _make_standard_frame(frame_idx=0, sensing=sensing),
            )

    def test_micro_doppler_loaders_read_coherent_packed_v2_data(self, tmp_path: Path):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        slow_time_matrix = np.arange(24, dtype=np.float32).reshape(3, 8).astype(np.complex64)
        sensing = _with_single_target_truth(
            frame_idx=0,
            sensing={
                "config": {
                    **SENSING_META["config"],
                    "fft_size_range": 8,
                    "fft_size_doppler": 3,
                },
                "processing_strategy": "coherent_cpi_fft",
                "slow_time_matrix": slow_time_matrix,
            },
            target_range_m=4.5,
            target_speed_m_s=1.25,
        )
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(frame_idx=0, sensing=sensing),
        )

        samples, config = load_cir_sequence(frames_dir)

        assert len(samples) == 3
        np.testing.assert_array_equal(np.stack(samples), slow_time_matrix)
        assert config["fft_size_range"] == 8
        np.testing.assert_array_equal(config["range_axis_m"], np.arange(8, dtype=np.float64))
        with pytest.raises(ValueError, match="no exact truth snapshot for physical sample"):
            load_gt_entries_for_tracking(frames_dir)

    def test_micro_doppler_loader_reads_sequential_packed_v2_cir(self, tmp_path: Path):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        cir = np.arange(8, dtype=np.float32).astype(np.complex64)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": {**SENSING_META["config"], "fft_size_range": 8},
                    "processing_strategy": "sequential_stft",
                    "cir": cir,
                },
            ),
        )

        samples, config = load_cir_sequence(frames_dir)

        assert len(samples) == 1
        np.testing.assert_array_equal(samples[0], cir)
        assert config["fft_size_range"] == 8

    def test_load_frames_sensing_reads_frame_positions_for_xy_figures(self, tmp_path: Path):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        frame_input = _make_standard_frame(
            frame_idx=0,
            sensing={
                "config": dict(SENSING_META["config"]),
                "slow_time_matrix": _make_slow_time_matrix().astype(np.complex64),
            },
        )
        frame_input = replace(
            frame_input,
            target_positions_m=np.array([[1.0, 2.0, 0.5]], dtype=np.float64),
            targets_metadata=(
                {
                    "name": "TargetA",
                    "current_position": [1.0, 2.0, 0.5],
                    "use_ply_position": False,
                },
            ),
        )
        _write_packed_frame_set(
            frames_dir,
            frame_input,
        )

        frames = _load_frames_sensing(frames_dir)

        assert len(frames) == 1
        np.testing.assert_allclose(
            frames[0]["tx_positions"],
            frame_input.tx_positions,
        )
        np.testing.assert_allclose(
            frames[0]["rx_positions"],
            frame_input.rx_positions,
        )
        np.testing.assert_allclose(
            frames[0]["target_pos"],
            frame_input.target_positions_m,
        )
        assert frames[0]["targets_metadata"][0]["name"] == "TargetA"

    def test_load_frames_sensing_does_not_read_mpc_path_columns(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)
        _write_packed_frame_set(
            frames_dir,
            _make_standard_frame(
                frame_idx=0,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "slow_time_matrix": _make_slow_time_matrix().astype(np.complex64),
                },
            ),
        )
        touched_datasets = set()
        original_dataset = packed_hdf5_module._dataset

        def _record_dataset(root, path):
            parent = str(getattr(root, "name", "")).strip("/")
            touched_datasets.add(f"{parent}/{path}".strip("/"))
            return original_dataset(root, path)

        monkeypatch.setattr(packed_hdf5_module, "_dataset", _record_dataset)

        assert len(_load_frames_sensing(frames_dir)) == 1

        assert not any(path.startswith("paths/") for path in touched_datasets)
        assert not any(path.startswith("bounces/") for path in touched_datasets)
        assert "sensing/fixed/slow_time_by_pair_rx_element/data" in touched_datasets

    def test_columnar_result_writer_publishes_packed_manifest_and_round_trips(
        self,
        tmp_path: Path,
    ):
        output_dir = tmp_path / "reprocessed"
        output_dir.mkdir()
        config = PostProcessingConfig()
        writer = ColumnarResultWriter(output_dir, chunk_size=1)
        for frame_index in (3, 7):
            source_sensing = _v4_sensing(
                frame_idx=frame_index,
                sensing={
                    "config": dict(SENSING_META["config"]),
                    "range_doppler_map": np.ones((32, 64), dtype=np.float64),
                },
            )
            source_frame = _make_standard_frame(
                frame_idx=frame_index,
                sensing=source_sensing,
            )
            writer.add_frame(
                _postprocessed_result(frame_idx=frame_index),
                frame_index=frame_index,
                source_frame=source_frame,
                source_sensing=source_sensing,
                postprocessing_config=config,
                input_kind="range_doppler_power_by_pair",
            )

        chunks = writer.finalize()

        frames_dir = output_dir / "frames"
        manifest = load_frame_manifest(frames_dir)
        assert manifest.frame_ids == (3, 7)
        assert len(chunks) == len(manifest.chunks) == 2
        assert manifest.provenance["output_kind"] == "reprocessed_sensing"
        assert not (frames_dir / "frames_index.json").exists()

        reloaded_inputs = _load_frames_sensing(output_dir)
        assert [item["_frame_index"] for item in reloaded_inputs] == [3, 7]
        saved_results = load_saved_results(output_dir)
        assert [item["_frame_index"] for item in saved_results] == [3, 7]
        np.testing.assert_allclose(saved_results[0]["rd_map"], 5.0)
        np.testing.assert_array_equal(saved_results[0]["range_axis"], np.arange(64))
        np.testing.assert_array_equal(saved_results[0]["doppler_axis"], np.arange(32))
        first_reloaded = reloaded_inputs[0]
        np.testing.assert_array_equal(
            first_reloaded["range_doppler_power_by_pair"],
            np.full((1, 32, 64), 5.0, dtype=np.float64),
        )
        descriptor = first_reloaded["_sensing_json"]["products"]["range_doppler_power_by_pair"]
        assert descriptor["representation"] == "power_linear"
        assert [axis["coordinate_key"] for axis in descriptor["axes"]] == [
            "pair_axis",
            "doppler_axis_hz",
            "range_axis_m",
        ]

    def test_load_frames_sensing_requires_packed_manifest(self, tmp_path: Path):
        frames_dir = tmp_path / "legacy" / "frames"
        frames_dir.mkdir(parents=True)
        (frames_dir / "mpc_frames_00000-00000.h5").write_bytes(b"legacy")

        with pytest.raises(FileNotFoundError, match="frames_manifest.json"):
            _load_frames_sensing(frames_dir)

    def test_columnar_result_writer_rotates_on_sensing_layout_boundary(
        self,
        tmp_path: Path,
    ):
        output_dir = tmp_path / "shape-change"
        output_dir.mkdir()
        writer = ColumnarResultWriter(output_dir, chunk_size=10)
        config = PostProcessingConfig()
        for frame_index, shape in ((0, (32, 64)), (1, (16, 32))):
            source_sensing = _v4_sensing(
                frame_idx=frame_index,
                sensing={
                    "config": {
                        **SENSING_META["config"],
                        "fft_size_doppler": shape[0],
                        "fft_size_range": shape[1],
                    },
                    "range_doppler_map": np.ones(shape, dtype=np.float64),
                },
            )
            writer.add_frame(
                _postprocessed_result(*shape, frame_idx=frame_index),
                frame_index=frame_index,
                source_frame=_make_standard_frame(
                    frame_idx=frame_index,
                    sensing=source_sensing,
                ),
                source_sensing=source_sensing,
                postprocessing_config=config,
                input_kind="range_doppler_power_by_pair",
            )

        writer.finalize()

        manifest = load_frame_manifest(output_dir / "frames")
        assert [chunk.frame_ids for chunk in manifest.chunks] == [(0,), (1,)]


def _beam_scan_source():
    source = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=complex),
            "range_doppler_map": np.ones((32, 64), dtype=complex),
        },
    )
    power = np.ones((1, 1, 3, 32, 64))
    power[0, 0, 0, 14, 25] = 1000.0
    power[0, 0, 2, 15, 25] = 800.0
    source.update(
        angular_power=power,
        elevation_direction_cosine=np.array([0.0]),
        azimuth_direction_cosine=np.array([-0.5, 0.0, 0.5]),
        elevation_broadside_angle_deg=np.array([0.0]),
        azimuth_broadside_angle_deg=np.array([-30.0, 0.0, 30.0]),
    )
    axes = [catalog_axis("pair", size=1, catalog_key="pair_axis")]
    for name, size, direction in [("elevation", 1, "z"), ("azimuth", 3, "y")]:
        axes.append(
            dense_axis(
                f"{name}_direction_cosine",
                size=size,
                unit="1",
                coordinate_key=f"{name}_direction_cosine",
                alternate_coordinate_keys=(f"{name}_broadside_angle_deg",),
                coordinate_semantics=f"rx_array_local_{direction}_direction_cosine",
                reference_frame="rx_array_local",
            )
        )
    axes.extend(
        [
            dense_axis(
                "doppler_frequency",
                size=32,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
                coordinate_semantics="fftshifted_doppler_frequency",
            ),
            dense_axis(
                "range",
                size=64,
                unit="m",
                coordinate_key="range_axis_m",
                coordinate_semantics="monostatic_range",
                reference_frame="tx_target_rx_path",
            ),
        ]
    )
    add_product_spec(
        source,
        data_key="angular_power",
        kind="range_doppler_angle",
        axes=axes,
        representation="power_linear",
        unit="1",
    )
    materialize_product_specs(source)
    return source


def test_beam_scan_typed_source_lineage_and_published_stage_replay():
    source = _beam_scan_source()
    cfg = PostProcessingConfig(array_detection_mode="beam_scan", include_stage_artifacts=True)
    pp = SensingPostProcessor(cfg)
    result = pp.process_from_rd_map(
        np.ones((32, 64)),
        source,
        range_axis_m=source["range_axis_m"],
        doppler_axis_hz=source["doppler_axis_hz"],
    )
    assert len(result["detections"]) == 2
    assert sorted(d["aoa_az_deg"] for d in result["detections"]) == pytest.approx([-30.0, 30.0])
    parent = source["products"]["angular_power"]["product_id"]
    assert result["angular_input_product_id"] == parent
    envelope = _postprocessed_result(frame_idx=4)
    for key in ("temporal", "processing_recipe_resolved"):
        result[key] = envelope[key]
    published = build_reprocessed_sensing_block(
        source_sensing=source,
        result=result,
        postprocessing_config=cfg,
        input_kind="range_doppler_by_pair",
    )
    stages = published["products"]
    assert stages["range_doppler_detection_input_power_by_pair"]["lineage"][
        "source_product_ids"
    ] == [parent]
    assert parent in stages["range_doppler_nms_mask_by_pair"]["lineage"]["source_product_ids"]
    assert (
        stages["range_doppler_nms_mask_by_pair"]["processing"]["stage_projection"]
        == "strongest_visible_beam_per_rd_cell"
    )
    assert published["detection_filter_metadata"]["angular_input_product_id"] == parent


def test_beam_scan_rejects_stale_angular_products_when_window_or_grid_changes():
    source = _beam_scan_source()
    with pytest.raises(ValueError, match="window changes"):
        SensingPostProcessor(
            PostProcessingConfig(array_detection_mode="beam_scan", window_doppler="rect")
        ).process_from_rd_map(np.ones((32, 64)), source)
    pp = SensingPostProcessor(PostProcessingConfig(array_detection_mode="beam_scan"))
    with pytest.raises(ValueError, match="coordinate grids"):
        pp.process_from_rd_map(
            np.ones((32, 64)),
            source,
            range_axis_m=source["range_axis_m"] + 1.0,
            doppler_axis_hz=source["doppler_axis_hz"],
        )
    with pytest.raises(ValueError, match="combined slow-time"):
        pp.process_from_slow_time_matrix(np.ones((32, 64), complex), source)


@pytest.mark.parametrize("value", [0, 17, True, 2.5])
def test_beam_refinement_configuration_rejects_invalid_factors(value):
    with pytest.raises(ValueError, match="beam_angle_oversampling"):
        PostProcessingConfig(array_detection_mode="beam_scan", beam_angle_oversampling=value)


def test_beam_refinement_requires_native_complex_element_input_and_is_opt_in():
    assert PostProcessingConfig().beam_angle_oversampling == 1
    with pytest.raises(ValueError, match="requires array_detection_mode"):
        PostProcessingConfig(beam_angle_oversampling=8)
    pp = SensingPostProcessor(
        PostProcessingConfig(array_detection_mode="beam_scan", beam_angle_oversampling=8)
    )
    with pytest.raises(ValueError, match="native analytical-Doppler"):
        pp._beam_angle_refiner({})
    source = _beam_scan_source()
    source["processing_strategy"] = "coherent_cpi_fft"
    source.pop("slow_time_by_pair_rx_element", None)
    with pytest.raises(ValueError, match="retained slow_time_by_pair_rx_element"):
        pp._beam_angle_refiner(source)


@pytest.mark.parametrize("mode", ["latched", "continuous"])
@pytest.mark.parametrize("enabled", [False, True])
def test_reprocessed_confirmation_policy_replaces_source_policy(mode, enabled):
    source = _v4_sensing(
        frame_idx=4,
        sensing={
            "config": dict(SENSING_META["config"]),
            "slow_time_matrix": np.ones((32, 64), dtype=np.complex64),
        },
    )
    source["config_resolved"]["track_confirmation_mode"] = (
        "continuous" if mode == "latched" else "latched"
    )
    result = _postprocessed_result(frame_idx=4)
    result["detection_filter_metadata"] = {
        "tracker": {"enabled": enabled, "confirmation_mode": mode}
    }
    derived = build_reprocessed_sensing_block(
        source_sensing=source,
        result=result,
        postprocessing_config=PostProcessingConfig(tracker_confirmation_mode=mode),
        input_kind="slow_time_by_pair_rx_element",
    )
    if enabled:
        assert derived["config_resolved"]["track_confirmation_mode"] == mode
    else:
        assert "track_confirmation_mode" not in derived["config_resolved"]
