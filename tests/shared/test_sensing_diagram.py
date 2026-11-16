"""Strict schema-v4 tests for sensing acquisition/processing diagrams."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    assert_sensing_contract,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    canonical_sensing_pair_id,
    finalize_sensing_v4_publication,
)
from shared.sensing.plotting.diagram import (
    ANALYTIC_ORACLE_MODE,
    COHERENT_MODE,
    SEQUENTIAL_MODE,
    build_sensing_diagram_model,
    create_sensing_acquisition_processing_figure,
    create_sensing_mode_comparison_figure,
    sensing_diagram_manifest,
    sensing_mode_comparison_manifest,
    write_sensing_diagram_manifest,
)
from shared.sensing.products import (
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.signal_processing import (
    DELAY_PROJECTION_POINT_BIN_FLOOR,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF,
    DOPPLER_FFT_NORMALIZATION_UNITARY,
    RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
    WINDOW_NORMALIZATION_ENERGY,
)

_GENERATION_ID = "00000000-0000-4000-8000-000000000351"
_PROCESSING_RUN_ID = "00000000-0000-4000-8000-000000000352"
_PROCESSING_RUN_ID_SECOND = "00000000-0000-4000-8000-000000000353"
_TX_ID = "diagram-tx"
_RX_ID = "diagram-rx"
_PAIR_ID = canonical_sensing_pair_id(_TX_ID, _RX_ID)


def _execution_contract(
    mode: str,
    *,
    cpi_length: int,
    sample_period_s: float | None,
) -> tuple[dict, str]:
    if mode == COHERENT_MODE:
        strategy = "coherent_cpi_fft"
        sample_source = "coherent_one_solve"
        hop = cpi_length
        phase_contract = "sionna_intra_solve_phase_coherent"
        product_semantics = "cir_point_bin_slow_time_fft"
        evidence_status = "phase2_controlled_specular_validated"
    elif mode == SEQUENTIAL_MODE:
        strategy = "sequential_stft"
        sample_source = "independent_solves"
        hop = 1
        phase_contract = "independent_solve_phase_continuity_controlled_specular_only"
        product_semantics = "cross_solve_cir_doppler_fft"
        evidence_status = "phase2_controlled_specular_validated_experimental"
    elif mode == ANALYTIC_ORACLE_MODE:
        strategy = "analytic_path_doppler_oracle"
        sample_source = "analytic_path_snapshot"
        hop = 1
        phase_contract = "analytic_nonwaveform_path_doppler"
        product_semantics = "analytic_oracle_nonwaveform"
        evidence_status = "diagnostic_quarantined"
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(f"Unsupported fixture mode: {mode}")

    fft_size_doppler = 8 if mode == ANALYTIC_ORACLE_MODE else cpi_length
    sample_rate_hz = None if sample_period_s is None else 1.0 / sample_period_s
    recipe = {
        "window_range": "hann",
        "window_doppler": "blackman",
        "output_detections": True,
        "clutter_removal_enabled": mode == COHERENT_MODE,
        "clutter_removal_window": 8,
        "cfar_type": "ca",
    }
    acquisition = {
        "sample_source": sample_source,
        "rt_solve_stride_scene_steps": cpi_length if mode == COHERENT_MODE else 1,
        "sample_period_s": sample_period_s,
        "sample_rate_hz": sample_rate_hz,
        "cpi_length_samples": cpi_length,
        "cpi_hop_samples": hop,
        "overlap_samples": cpi_length - hop,
        "partial_window_policy": "retain_unprocessed",
        "phase_contract": phase_contract,
    }
    processing = {
        "mode": mode,
        "strategy": strategy,
        "fft_size_doppler": fft_size_doppler,
        "result_stride_samples": hop,
        "product_semantics": product_semantics,
        "evidence_status": evidence_status,
    }
    topology = {
        "pair_count": 1,
        "pair_index": 0,
        "tx_device_count": 1,
        "rx_device_count": 1,
        "tx_elements": 1,
        "rx_elements": 1,
        "rx_array_processing": "siso",
    }
    return (
        {
            "contract_version": 1,
            "fft_size_range": 16,
            "fft_size_doppler": fft_size_doppler,
            "sampling_frequency": 100.0e6,
            "bandwidth": 100.0e6,
            "bandwidth_hz": 100.0e6,
            "prf_hz": 1000.0,
            "carrier_freq_hz": 28.0e9,
            "carrier_frequency": 28.0e9,
            "range_mode": "monostatic",
            "processing_mode": mode,
            "processing_recipe": recipe,
            "acquisition": acquisition,
            "processing": processing,
            "topology": topology,
        },
        strategy,
    )


def _event(
    mode: str,
    *,
    label: str,
    sample_times_s: list[float],
    scene_frame_ids: list[int],
    solve_ids: list[str],
    sample_offsets: list[int],
    config: dict | None = None,
    slow_time_mean_removal_applied: bool = True,
    processing_run_id: str = _PROCESSING_RUN_ID,
) -> dict:
    resolved, strategy = (
        _execution_contract(
            mode,
            cpi_length=len(sample_times_s),
            sample_period_s=None if mode == ANALYTIC_ORACLE_MODE else 0.001,
        )
        if config is None
        else (deepcopy(config), str(config["processing"]["strategy"]))
    )
    if not (len(sample_times_s) == len(scene_frame_ids) == len(solve_ids) == len(sample_offsets)):
        raise AssertionError("Fixture source lists must align")

    samples = [
        {
            "sample_index": index,
            "sample_time_s": sample_time_s,
            "valid": True,
            "phase_coherent": mode == COHERENT_MODE,
            "source": {
                "source_generation_id": _GENERATION_ID,
                "scene_frame_id": scene_frame_ids[index],
                "sample_offset": sample_offsets[index],
                "solve_id": solve_ids[index],
            },
        }
        for index, sample_time_s in enumerate(sample_times_s)
    ]
    if mode == COHERENT_MODE:
        reference_time_s = sample_times_s[0]
        reference_policy = "solve_start"
    elif mode == SEQUENTIAL_MODE:
        reference_time_s = 0.5 * (sample_times_s[0] + sample_times_s[-1])
        reference_policy = "cpi_center"
    else:
        reference_time_s = sample_times_s[0]
        reference_policy = "solve_start"

    acquisition_config_id = canonical_sensing_hash(
        {
            "acquisition": resolved["acquisition"],
            "topology": resolved["topology"],
            "pair_id": _PAIR_ID,
        }
    )
    processing_recipe_id = canonical_sensing_hash(resolved["processing_recipe"])
    acquisition_id = str(uuid5(UUID(_GENERATION_ID), f"{mode}:{label}"))
    temporal = build_sensing_temporal_contract(
        clock_id=_GENERATION_ID,
        acquisition_id=acquisition_id,
        acquisition_config_id=acquisition_config_id,
        pair_id=_PAIR_ID,
        tx_sensor_id=_TX_ID,
        rx_sensor_id=_RX_ID,
        samples=samples,
        processing_run_id=processing_run_id,
        processing_recipe_id=processing_recipe_id,
        reference_time_s=reference_time_s,
        reference_time_policy=reference_policy,
        display_scene_frame_id=scene_frame_ids[0],
        display_policy="source_scene_exact",
        display_reason="Strict sensing-diagram fixture binding.",
        completeness="complete",
    )

    fft_d = int(resolved["fft_size_doppler"])
    fft_r = int(resolved["fft_size_range"])
    data_key = (
        "range_doppler_power_by_pair" if mode == ANALYTIC_ORACLE_MODE else "range_doppler_by_pair"
    )
    values = (
        np.ones((1, fft_d, fft_r), dtype=np.float64)
        if mode == ANALYTIC_ORACLE_MODE
        else np.ones((1, fft_d, fft_r), dtype=np.complex128)
    )
    sensing = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "processing_strategy": strategy,
        "config_resolved": resolved,
        "config": deepcopy(resolved),
        "temporal": temporal,
        data_key: values,
        "doppler_axis_hz": np.linspace(-500.0, 375.0, fft_d, dtype=np.float64),
        "range_axis_m": np.arange(fft_r, dtype=np.float64),
    }
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    doppler_axis = dense_axis(
        "doppler_frequency",
        size=fft_d,
        unit="Hz",
        coordinate_key="doppler_axis_hz",
        coordinate_semantics="fftshifted_doppler_frequency",
    )
    range_axis = dense_axis(
        "range",
        size=fft_r,
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics="monostatic_range",
        reference_frame="tx_target_rx_path",
    )
    if mode == ANALYTIC_ORACLE_MODE:
        direct_filter_enabled = bool(resolved["processing_recipe"]["clutter_removal_enabled"])
        processing_metadata = {
            "model": "analytic_path_power_oracle_nonwaveform",
            "doppler_assignment": "nearest_bin",
            "direct_doppler_clutter_filter_policy": (
                DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
                if direct_filter_enabled
                else DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
            ),
            "direct_doppler_clutter_cutoff_hz": (
                resolved["prf_hz"] / (2.0 * resolved["processing_recipe"]["clutter_removal_window"])
                if direct_filter_enabled
                else 0.0
            ),
        }
        representation = "power_linear"
        evidence_role = "oracle_assisted"
    else:
        processing_metadata = {
            "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
            "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
            "range_fft_applied": False,
            "delay_taper": resolved["processing_recipe"]["window_range"],
            "doppler_taper": resolved["processing_recipe"]["window_doppler"],
            "fft_axis": "slow_time",
            "fft_normalization": DOPPLER_FFT_NORMALIZATION_UNITARY,
            "window_normalization": WINDOW_NORMALIZATION_ENERGY,
            "slow_time_mean_removal_policy": "when_row_variation_nonzero",
            "slow_time_mean_removal_applied": slow_time_mean_removal_applied,
        }
        representation = "complex_voltage"
        evidence_role = "observation_derived"
    add_product_spec(
        sensing,
        data_key=data_key,
        kind="range_doppler",
        axes=(pair_axis, doppler_axis, range_axis),
        representation=representation,
        unit="1",
        evidence_role=evidence_role,
        temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
        combine={"policy": "siso_identity", "normalization": "identity"},
        processing=processing_metadata,
    )
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    assert_sensing_contract(
        {"sensing": sensing},
        require_schema_version=True,
        require_described_products=True,
    )
    return sensing


def _coherent_events() -> tuple[dict, dict]:
    config, _strategy = _execution_contract(
        COHERENT_MODE,
        cpi_length=4,
        sample_period_s=0.001,
    )
    first = _event(
        COHERENT_MODE,
        label="first",
        sample_times_s=[10.000, 10.001, 10.002, 10.003],
        scene_frame_ids=[100, 100, 100, 100],
        solve_ids=["coherent-solve-100"] * 4,
        sample_offsets=[0, 1, 2, 3],
        config=config,
        slow_time_mean_removal_applied=False,
    )
    second = _event(
        COHERENT_MODE,
        label="second",
        sample_times_s=[10.008, 10.009, 10.010, 10.011],
        scene_frame_ids=[104, 104, 104, 104],
        solve_ids=["coherent-solve-104"] * 4,
        sample_offsets=[0, 1, 2, 3],
        config=config,
        slow_time_mean_removal_applied=True,
        processing_run_id=_PROCESSING_RUN_ID_SECOND,
    )
    return first, second


def _sequential_events() -> tuple[dict, dict]:
    config, _strategy = _execution_contract(
        SEQUENTIAL_MODE,
        cpi_length=4,
        sample_period_s=0.001,
    )
    first = _event(
        SEQUENTIAL_MODE,
        label="first",
        sample_times_s=[20.000, 20.001, 20.002, 20.003],
        scene_frame_ids=[200, 201, 202, 203],
        solve_ids=[f"sequential-solve-{index}" for index in range(200, 204)],
        sample_offsets=[0, 0, 0, 0],
        config=config,
        slow_time_mean_removal_applied=True,
    )
    second = _event(
        SEQUENTIAL_MODE,
        label="second",
        sample_times_s=[20.001, 20.002, 20.003, 20.004],
        scene_frame_ids=[201, 202, 203, 204],
        solve_ids=[f"sequential-solve-{index}" for index in range(201, 205)],
        sample_offsets=[0, 0, 0, 0],
        config=config,
        slow_time_mean_removal_applied=False,
        processing_run_id=_PROCESSING_RUN_ID_SECOND,
    )
    return first, second


def test_coherent_model_uses_exact_cpi_timing_rt_epochs_and_processing() -> None:
    first, second = _coherent_events()

    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])

    assert model.mode == COHERENT_MODE
    assert model.evidence_status == "PRIMARY"
    assert "phase2_controlled_specular_validated" in model.evidence_detail
    assert model.sample_source == "coherent_one_solve"
    assert model.sample_period_s == pytest.approx(0.001)
    assert model.sample_rate_hz == pytest.approx(1000.0)
    assert model.cpi_length_samples == 4
    assert model.cpi_hop_samples == 4
    assert model.overlap_samples == 0
    assert model.overlap_fraction == 0.0
    assert model.cpi_duration_s == pytest.approx(0.004)
    assert model.sample_center_span_s == pytest.approx(0.003)
    assert model.hop_duration_s == pytest.approx(0.004)
    assert model.observed_revisit_interval_s == pytest.approx(0.008)
    assert model.rt_solve_samples == (0, 4)
    assert [(window.start_sample, window.stop_sample) for window in model.windows] == [
        (0, 4),
        (4, 8),
    ]
    assert [window.result_emitted_with_sample for window in model.windows] == [3, 7]

    assert len(model.acquisitions) == 2
    assert [len(acquisition.rt_epochs) for acquisition in model.acquisitions] == [1, 1]
    assert model.acquisitions[0].rt_epochs[0].member_indices == (0, 1, 2, 3)
    assert model.acquisitions[0].rt_epochs[0].source_epoch_s == pytest.approx(10.0)
    assert model.acquisitions[1].rt_epochs[0].source_epoch_s == pytest.approx(10.008)
    assert [acquisition.slow_time_mean_removal_applied for acquisition in model.acquisitions] == [
        False,
        True,
    ]
    assert model.acquisitions[0].start_time_s == pytest.approx(10.0)
    assert model.acquisitions[0].end_time_s == pytest.approx(10.003)
    assert model.acquisitions[0].default_reference_policy == "cpi_center"
    assert model.acquisitions[0].default_reference_time_s == pytest.approx(10.0015)
    assert all(slot.phase_coherent for slot in model.acquisitions[0].slots)
    assert model.result_reference_policy == "solve_start"
    assert model.processing_run_id == _PROCESSING_RUN_ID_SECOND

    assert model.delay_projection == DELAY_PROJECTION_POINT_BIN_FLOOR
    assert model.range_processing == RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS
    assert model.range_fft_applied is False
    assert model.delay_taper == "hann"
    assert model.doppler_taper == "blackman"
    assert model.fft_axis == "slow_time"
    assert model.fft_normalization == DOPPLER_FFT_NORMALIZATION_UNITARY
    assert model.window_normalization == WINDOW_NORMALIZATION_ENERGY
    assert model.slow_time_mean_removal_policy == "when_row_variation_nonzero"
    assert model.slow_time_mean_removal_applied is True
    assert [stage.title for stage in model.processing_stages] == [
        "RT → CIR paths",
        "Delay projection",
        "CPI tapers",
        "Slow-time DC removal",
        "CPI Doppler DFT",
        "Range–Doppler power",
        "Detection output",
    ]
    assert model.delay_sampling_frequency_hz == pytest.approx(100.0e6)
    assert "floor(τ·F_delay)" in model.processing_stages[1].detail
    assert "F_delay=100 MHz" in model.processing_stages[1].detail
    assert "configured B=100 MHz" in model.processing_stages[1].detail
    assert "delay-bin weighting=hann" in model.processing_stages[2].detail
    assert "CPI Doppler window=blackman" in model.processing_stages[2].detail
    assert "does not consume" in model.processing_stages[2].detail
    assert "displayed result: applied" in model.processing_stages[3].detail


def test_processing_run_identity_varies_per_result_and_latest_display_wins() -> None:
    first, second = _coherent_events()

    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])

    assert [acquisition.processing_run_id for acquisition in model.acquisitions] == [
        _PROCESSING_RUN_ID,
        _PROCESSING_RUN_ID_SECOND,
    ]
    assert model.processing_run_id == _PROCESSING_RUN_ID_SECOND
    assert len({acquisition.processing_recipe_id for acquisition in model.acquisitions}) == 1


def test_sequential_model_preserves_sliding_window_hop_and_exact_overlap() -> None:
    first, second = _sequential_events()

    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])

    assert model.mode == SEQUENTIAL_MODE
    assert model.evidence_status == "EXPERIMENTAL"
    assert "phase2_controlled_specular_validated_experimental" in model.evidence_detail
    assert model.sample_source == "independent_solves"
    assert model.cpi_length_samples == 4
    assert model.cpi_hop_samples == 1
    assert model.overlap_samples == 3
    assert model.overlap_fraction == pytest.approx(0.75)
    assert model.hop_duration_s == pytest.approx(0.001)
    assert model.observed_revisit_interval_s == pytest.approx(0.001)
    assert model.rt_solve_samples == (0, 1, 2, 3, 4)
    assert [(window.start_sample, window.stop_sample) for window in model.windows] == [
        (0, 4),
        (1, 5),
    ]
    assert [window.result_emitted_with_sample for window in model.windows] == [3, 4]

    first_acquisition, second_acquisition = model.acquisitions
    assert len(first_acquisition.rt_epochs) == len(second_acquisition.rt_epochs) == 4
    assert all(not slot.phase_coherent for slot in first_acquisition.slots)
    first_sources = tuple(slot.composite_source_identity for slot in first_acquisition.slots)
    second_sources = tuple(slot.composite_source_identity for slot in second_acquisition.slots)
    assert first_sources[1:] == second_sources[:3]
    assert first_acquisition.result_reference_time_s == pytest.approx(20.0015)
    assert second_acquisition.result_reference_time_s == pytest.approx(20.0025)
    assert [acquisition.slow_time_mean_removal_applied for acquisition in model.acquisitions] == [
        True,
        False,
    ]
    assert model.result_reference_policy == "cpi_center"
    assert model.range_fft_applied is False
    assert model.clutter_removal_enabled is False
    assert model.slow_time_mean_removal_policy == "when_row_variation_nonzero"
    assert model.slow_time_mean_removal_applied is False
    assert "N=4 independent solves" in model.processing_stages[0].detail
    assert model.processing_stages[2].title == "STFT tapers / clutter"
    assert "rolling STFT window=blackman" in model.processing_stages[2].detail
    assert "sliding-baseline removal disabled" in model.processing_stages[2].detail
    assert "displayed result: not applied" in model.processing_stages[3].detail
    assert model.processing_stages[4].title == "Rolling STFT DFT"


def test_analytic_oracle_is_quarantined_without_sampled_pri_or_fft() -> None:
    oracle = _event(
        ANALYTIC_ORACLE_MODE,
        label="snapshot",
        sample_times_s=[30.0],
        scene_frame_ids=[300],
        solve_ids=["oracle-solve-300"],
        sample_offsets=[0],
    )

    model = build_sensing_diagram_model(oracle)

    assert model.mode == ANALYTIC_ORACLE_MODE
    assert model.evidence_status == "DIAGNOSTIC"
    assert "diagnostic_quarantined" in model.evidence_detail
    assert model.sample_source == "analytic_path_snapshot"
    assert model.sample_period_s is None
    assert model.sample_rate_hz is None
    assert model.cpi_duration_s is None
    assert model.sample_center_span_s is None
    assert model.hop_duration_s is None
    assert model.cpi_length_samples == 1
    assert model.cpi_hop_samples == 1
    assert model.overlap_samples == 0
    assert model.delay_projection is None
    assert model.range_processing is None
    assert model.range_fft_applied is None
    assert model.delay_taper is None
    assert model.doppler_taper is None
    assert model.fft_axis is None
    assert model.fft_normalization is None
    assert model.slow_time_mean_removal_policy is None
    assert model.slow_time_mean_removal_applied is None
    assert (
        model.direct_doppler_clutter_filter_policy == DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
    )
    assert model.direct_doppler_clutter_cutoff_hz == 0.0
    assert model.rt_solve_samples == (0,)
    assert [stage.title for stage in model.processing_stages] == [
        "RT paths",
        "Direct path binning",
        "Oracle RD power",
        "Detection output",
    ]
    assert all("FFT" not in stage.title for stage in model.processing_stages)
    assert "complete per-path" in model.processing_stages[0].detail
    assert "direct path-Doppler clutter filter disabled" in model.processing_stages[1].detail


def test_analytic_oracle_reports_exact_enabled_direct_doppler_filter() -> None:
    config, _strategy = _execution_contract(
        ANALYTIC_ORACLE_MODE,
        cpi_length=1,
        sample_period_s=None,
    )
    config["processing_recipe"]["clutter_removal_enabled"] = True
    oracle = _event(
        ANALYTIC_ORACLE_MODE,
        label="snapshot-filtered",
        sample_times_s=[31.0],
        scene_frame_ids=[301],
        solve_ids=["oracle-solve-filtered"],
        sample_offsets=[0],
        config=config,
    )

    model = build_sensing_diagram_model(oracle)

    assert (
        model.direct_doppler_clutter_filter_policy
        == DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
    )
    assert model.direct_doppler_clutter_cutoff_hz == pytest.approx(62.5)
    assert "|f_D| < 62.5 Hz" in model.processing_stages[1].detail
    assert "exactly at the cutoff are retained" in model.processing_stages[1].detail


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "direct_doppler_clutter_filter_policy",
            DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF,
            "direct_doppler_clutter_filter_policy must equal",
        ),
        (
            "direct_doppler_clutter_cutoff_hz",
            1.0,
            "direct_doppler_clutter_cutoff_hz must equal",
        ),
    ],
)
def test_analytic_oracle_rejects_direct_doppler_filter_descriptor_mismatch(
    field: str,
    value: object,
    match: str,
) -> None:
    oracle = _event(
        ANALYTIC_ORACLE_MODE,
        label=f"snapshot-mismatch-{field}",
        sample_times_s=[32.0],
        scene_frame_ids=[302],
        solve_ids=["oracle-solve-mismatch"],
        sample_offsets=[0],
    )
    malformed = deepcopy(oracle)
    next(iter(malformed["products"].values()))["processing"][field] = value

    with pytest.raises(ValueError, match=match):
        build_sensing_diagram_model(malformed)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda sensing: sensing.__setitem__("processing_strategy", "sequential_stft"),
            "strategy disagrees",
        ),
        (
            lambda sensing: next(iter(sensing["products"].values()))["processing"].__setitem__(
                "range_fft_applied", True
            ),
            "range_fft_applied must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"]["samples"][2].__setitem__(
                "sample_time_s", 10.0025
            ),
            "sample cadence disagrees",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].__setitem__(
                "expected_sample_count", 3
            ),
            "expected_sample_count must match",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].__setitem__("start_time_s", 9.999),
            "start_time_s must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].__setitem__("end_time_s", 10.012),
            "end_time_s must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].__setitem__(
                "default_reference_policy", "solve_start"
            ),
            "default_reference_policy must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].__setitem__(
                "default_reference_time_s", 10.0
            ),
            "default_reference_time_s must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["result"].__setitem__(
                "reference_time_policy", "cpi_center"
            ),
            "result reference policy must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["result"].__setitem__("reference_time_s", 10.001),
            "reference_time_s must equal",
        ),
        (
            lambda sensing: sensing["temporal"]["result"].__setitem__("processing_run_id", ""),
            "processing_run_id must be a non-empty string",
        ),
        (
            lambda sensing: sensing["config_resolved"]["processing"].__setitem__(
                "evidence_status", "diagnostic_quarantined"
            ),
            "evidence_status must equal",
        ),
        (
            lambda sensing: (
                sensing["config_resolved"]["acquisition"].update(
                    {"cpi_hop_samples": 3, "overlap_samples": 1}
                ),
                sensing["config_resolved"]["processing"].__setitem__("result_stride_samples", 3),
            ),
            "Coherent CPI requires H=N and O=0",
        ),
        (
            lambda sensing: next(iter(sensing["products"].values()))["processing"].__setitem__(
                "slow_time_mean_removal_policy", "always"
            ),
            "slow_time_mean_removal_policy must equal",
        ),
        (
            lambda sensing: next(iter(sensing["products"].values()))["processing"].__setitem__(
                "slow_time_mean_removal_applied", "yes"
            ),
            "slow_time_mean_removal_applied must be a Boolean",
        ),
        (
            lambda sensing: next(iter(sensing["products"].values()))["lineage"].__setitem__(
                "processing_run_id", _PROCESSING_RUN_ID
            ),
            "descriptor processing identity disagrees",
        ),
    ],
)
def test_model_fails_closed_on_contract_or_processing_mismatch(mutation, match: str) -> None:
    _first, coherent = _coherent_events()
    malformed = deepcopy(coherent)
    mutation(malformed)

    with pytest.raises(ValueError, match=match):
        build_sensing_diagram_model(malformed)


def test_sequential_history_rejects_membership_that_disagrees_with_overlap() -> None:
    first, second = _sequential_events()
    malformed = deepcopy(second)
    sample = malformed["temporal"]["acquisition"]["samples"][0]
    sample["source"]["scene_frame_id"] = 999
    sample["source"]["solve_id"] = "sequential-solve-999"

    with pytest.raises(ValueError, match="overlap/hop"):
        build_sensing_diagram_model(malformed, sensing_event_history=[first, malformed])


def test_bounded_history_keeps_insertion_order_and_omits_unneeded_latest() -> None:
    first, second = _sequential_events()
    third = _event(
        SEQUENTIAL_MODE,
        label="third-not-displayed",
        sample_times_s=[20.002, 20.003, 20.004, 20.005],
        scene_frame_ids=[202, 203, 204, 205],
        solve_ids=[f"sequential-solve-{index}" for index in range(202, 206)],
        sample_offsets=[0, 0, 0, 0],
        config=second["config_resolved"],
    )

    model = build_sensing_diagram_model(third, sensing_event_history=[first, second])

    assert [acquisition.acquisition_id for acquisition in model.acquisitions] == [
        first["temporal"]["acquisition"]["acquisition_id"],
        second["temporal"]["acquisition"]["acquisition_id"],
    ]
    assert all(
        acquisition.acquisition_id != third["temporal"]["acquisition"]["acquisition_id"]
        for acquisition in model.acquisitions
    )


def test_latest_payload_semantics_are_validated_when_history_controls_display() -> None:
    first, second = _coherent_events()
    malformed_latest = deepcopy(second)
    next(iter(malformed_latest["products"].values()))["processing"]["delay_taper"] = "hamming"

    with pytest.raises(ValueError, match="processing.delay_taper must equal"):
        build_sensing_diagram_model(
            malformed_latest,
            sensing_event_history=[first, second],
        )


def test_sequential_nonunit_hop_is_exact_and_renders_generic_second_window(
    tmp_path: Path,
) -> None:
    config, _strategy = _execution_contract(
        SEQUENTIAL_MODE,
        cpi_length=4,
        sample_period_s=0.001,
    )
    config["acquisition"].update({"cpi_hop_samples": 2, "overlap_samples": 2})
    config["processing"]["result_stride_samples"] = 2
    first = _event(
        SEQUENTIAL_MODE,
        label="hop-two-first",
        sample_times_s=[40.000, 40.001, 40.002, 40.003],
        scene_frame_ids=[400, 401, 402, 403],
        solve_ids=[f"sequential-solve-{index}" for index in range(400, 404)],
        sample_offsets=[0, 0, 0, 0],
        config=config,
    )
    second = _event(
        SEQUENTIAL_MODE,
        label="hop-two-second",
        sample_times_s=[40.002, 40.003, 40.004, 40.005],
        scene_frame_ids=[402, 403, 404, 405],
        solve_ids=[f"sequential-solve-{index}" for index in range(402, 406)],
        sample_offsets=[0, 0, 0, 0],
        config=config,
    )

    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])
    output = create_sensing_acquisition_processing_figure(
        second,
        sensing_event_history=[first, second],
        output_path=tmp_path / "hop_two.png",
    )

    assert model.cpi_hop_samples == 2
    assert model.overlap_samples == 2
    assert model.observed_revisit_interval_s == pytest.approx(0.002)
    assert [window.start_sample for window in model.windows] == [0, 2]
    assert [window.result_emitted_with_sample for window in model.windows] == [3, 5]
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_and_json_manifest_smoke_use_the_exact_model(tmp_path: Path) -> None:
    first, second = _sequential_events()
    png_path = tmp_path / "sensing_acquisition_processing.png"
    json_path = tmp_path / "sensing_acquisition_processing.json"

    rendered = create_sensing_acquisition_processing_figure(
        second,
        output_path=png_path,
        sensing_event_history=[first, second],
    )
    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])
    manifest_path = write_sensing_diagram_manifest(model, json_path)

    assert rendered == png_path
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000
    assert manifest_path == json_path
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SENSING_SCHEMA_VERSION
    assert manifest["mode"] == SEQUENTIAL_MODE
    assert manifest["cpi_length_samples"] == 4
    assert manifest["cpi_hop_samples"] == 1
    assert manifest["overlap_samples"] == 3
    assert manifest["observed_revisit_interval_s"] == pytest.approx(0.001)
    assert manifest["cpi_duration_s"] == pytest.approx(0.004)
    assert manifest["sample_center_span_s"] == pytest.approx(0.003)
    assert manifest["hop_duration_s"] == pytest.approx(0.001)
    assert len(manifest["acquisitions"]) == 2
    assert "does not synthesize an RF waveform" in manifest["note"]
    assert "N·PRI" in manifest["note"]


def test_render_coherent_and_oracle_individual_diagrams(tmp_path: Path) -> None:
    coherent_first, coherent_second = _coherent_events()
    oracle = _event(
        ANALYTIC_ORACLE_MODE,
        label="snapshot-individual",
        sample_times_s=[30.0],
        scene_frame_ids=[300],
        solve_ids=["oracle-solve-individual"],
        sample_offsets=[0],
    )

    coherent_path = create_sensing_acquisition_processing_figure(
        coherent_second,
        sensing_event_history=[coherent_first, coherent_second],
        output_path=tmp_path / "coherent.png",
    )
    oracle_path = create_sensing_acquisition_processing_figure(
        oracle,
        output_path=tmp_path / "oracle.png",
    )

    for output in (coherent_path, oracle_path):
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert output.stat().st_size > 10_000


def test_render_exact_three_mode_comparison(tmp_path: Path) -> None:
    coherent_first, coherent_second = _coherent_events()
    sequential_first, sequential_second = _sequential_events()
    oracle = _event(
        ANALYTIC_ORACLE_MODE,
        label="snapshot-comparison",
        sample_times_s=[30.0],
        scene_frame_ids=[300],
        solve_ids=["oracle-solve-comparison"],
        sample_offsets=[0],
    )
    manifests = [
        sensing_diagram_manifest(
            build_sensing_diagram_model(
                coherent_second,
                sensing_event_history=[coherent_first, coherent_second],
            )
        ),
        sensing_diagram_manifest(
            build_sensing_diagram_model(
                sequential_second,
                sensing_event_history=[sequential_first, sequential_second],
            )
        ),
        sensing_diagram_manifest(build_sensing_diagram_model(oracle)),
    ]

    output = create_sensing_mode_comparison_figure(
        manifests,
        output_path=tmp_path / "mode_comparison.png",
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000
    comparison_json = output.with_suffix(".json")
    comparison_manifest = json.loads(comparison_json.read_text(encoding="utf-8"))
    assert comparison_manifest["schema_version"] == SENSING_SCHEMA_VERSION
    assert comparison_manifest["kind"] == "sensing_mode_comparison"
    assert [mode["mode"] for mode in comparison_manifest["modes"]] == [
        COHERENT_MODE,
        SEQUENTIAL_MODE,
        ANALYTIC_ORACLE_MODE,
    ]
    assert sensing_mode_comparison_manifest(manifests)["modes"] == manifests
    with pytest.raises(ValueError, match="Duplicate sensing comparison mode"):
        create_sensing_mode_comparison_figure(
            [manifests[0], manifests[0], manifests[2]],
            output_path=tmp_path / "invalid.png",
        )

    malformed = deepcopy(manifests)
    malformed[1]["observed_revisit_interval_s"] = 0.123
    with pytest.raises(ValueError, match="observed revisit interval is inconsistent"):
        sensing_mode_comparison_manifest(malformed)

    malformed = deepcopy(manifests)
    malformed[0]["cpi_hop_samples"] = 3
    malformed[0]["overlap_samples"] = 1
    malformed[0]["result_stride_samples"] = 3
    with pytest.raises(ValueError, match="Coherent comparison requires H=N and O=0"):
        sensing_mode_comparison_manifest(malformed)

    malformed = deepcopy(manifests)
    malformed[1]["acquisitions"][0]["slots"][0]["valid"] = False
    with pytest.raises(
        ValueError,
        match="Incomplete sensing comparison acquisition must not carry a result",
    ):
        sensing_mode_comparison_manifest(malformed)


def _coherent_cfr_events():
    events = _coherent_events()
    for event in events:
        config = event["config_resolved"]
        response = {"model": "cfr_ifft", "frequency_window": "hann", "num_frequency_bins": 16}
        config["range_response"] = response
        config["processing_recipe"]["range_response"] = deepcopy(response)
        config["processing_recipe"]["window_range"] = "boxcar"
        config["processing"]["evidence_status"] = "finite_bandwidth_bounded_coherent"
        config["processing"]["product_semantics"] = "cfr_ifft_slow_time_fft"
        descriptor = event["products"]["range_doppler_by_pair"]["processing"]
        descriptor.update(
            model="cfr_ifft",
            model_version=1,
            delay_projection="cfr_ifft",
            range_processing="finite_bandwidth_delay_response",
            range_fft_applied=True,
            range_transform="ifft",
            range_transform_axis="frequency",
            range_transform_normalization="unit_coherent_peak",
            frequency_window="hann",
            num_frequency_bins=16,
            fft_size_range=16,
            delay_taper="boxcar",
        )
    return events


def test_coherent_cfr_diagram_uses_retained_frequency_transform(tmp_path):
    first, second = _coherent_cfr_events()
    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])
    assert model.evidence_status == "BOUNDED"
    assert model.range_fft_applied is True
    assert model.range_processing == "finite_bandwidth_delay_response"
    assert model.delay_projection == "cfr_ifft"
    assert model.processing_stages[1].title == "CFR → delay IFFT"
    assert model.processing_stages[2].title == "Slow-time DC removal"
    assert model.processing_stages[3].title == "CPI Doppler taper"
    assert "frequency taper=hann" in model.processing_stages[1].detail
    assert "unit coherent peak" in model.processing_stages[1].detail
    assert "floor(" not in str(model.processing_stages)
    assert "frequency-windowed IFFT" in sensing_diagram_manifest(model)["note"]
    output = create_sensing_acquisition_processing_figure(second, output_path=tmp_path / "cfr.svg")
    assert output.is_file()
    assert "no range FFT" not in output.read_text()


@pytest.mark.parametrize(
    "key,value",
    [
        ("range_transform", "fft"),
        ("range_fft_applied", False),
        ("range_transform_axis", "slow_time"),
        ("frequency_window", "boxcar"),
        ("num_frequency_bins", 32),
        ("range_transform_normalization", "ortho"),
        ("model_version", 2),
    ],
)
def test_coherent_cfr_diagram_rejects_conflicting_product_evidence(key, value):
    _, event = _coherent_cfr_events()
    event["products"]["range_doppler_by_pair"]["processing"][key] = value
    with pytest.raises(ValueError, match="must equal the executed value"):
        build_sensing_diagram_model(event)


def test_coherent_cfr_diagram_rejects_mismatched_recipe_and_evidence():
    _, event = _coherent_cfr_events()
    event["config_resolved"]["processing_recipe"]["range_response"]["frequency_window"] = "boxcar"
    with pytest.raises(ValueError, match="disagrees with processing recipe"):
        build_sensing_diagram_model(event)
    _, event = _coherent_cfr_events()
    event["config_resolved"]["processing"][
        "evidence_status"
    ] = "phase2_controlled_specular_validated"
    with pytest.raises(ValueError, match="evidence_status must equal"):
        build_sensing_diagram_model(event)


@pytest.mark.parametrize("policy,applied", [("disabled", False), ("window_weighted_mean", True)])
def test_explicit_coherent_clutter_policy_in_summary(policy, applied):
    first, second = _coherent_cfr_events()
    for event in (first, second):
        event["products"]["range_doppler_by_pair"]["processing"].update(
            slow_time_mean_removal_policy=policy, slow_time_mean_removal_applied=applied
        )
    model = build_sensing_diagram_model(second, sensing_event_history=[first, second])
    assert model.slow_time_mean_removal_policy == policy
    assert model.slow_time_mean_removal_applied == applied
    if not applied:
        assert any("no mean removal" in stage.detail for stage in model.processing_stages)
