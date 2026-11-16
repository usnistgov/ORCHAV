"""Phase-2 sensing acquisition/mode contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import (
    ANALYTIC_ORACLE_MODE,
    COHERENT_MODE,
    SEQUENTIAL_MODE,
    resolve_sensing_execution_values,
)
from generator.core.sensing.processor import (
    AnalyticPathDopplerOracleProcessor,
    CoherentCpiFftProcessor,
    SequentialStftProcessor,
    create_sensing_processor,
)


def _resolve(
    mode: str,
    *,
    cir_steps: int,
    fft_d: int = 8,
    prf_hz: float = 1000.0,
    cir_rate_hz: float | None = None,
    frame_dt_s: float | None = None,
    tx_devices: int = 1,
    rx_devices: int = 1,
    tx_elements: int = 1,
    rx_elements: int = 1,
):
    config = SensingConfig(
        enabled=True,
        processing_mode=mode,
        fft_size_doppler=fft_d,
        prf_hz=prf_hz,
    )
    return config, resolve_sensing_execution_values(
        config,
        cir_time_steps=cir_steps,
        cir_sampling_frequency_hz=cir_rate_hz,
        frame_dt_s=1.0 / prf_hz if frame_dt_s is None else frame_dt_s,
        tx_device_count=tx_devices,
        rx_device_count=rx_devices,
        tx_elements=tx_elements,
        rx_elements=rx_elements,
    )


def test_coherent_contract_is_explicit_frozen_and_persistable() -> None:
    config, resolved = _resolve(
        COHERENT_MODE,
        cir_steps=8,
        fft_d=8,
        cir_rate_hz=1000.0,
        rx_elements=4,
    )

    assert resolved.processing.mode == COHERENT_MODE
    assert resolved.processing.strategy == "coherent_cpi_fft"
    assert resolved.acquisition.rt_solve_stride_scene_steps == 8
    assert resolved.acquisition.cpi_length_samples == 8
    assert resolved.acquisition.cpi_hop_samples == 8
    assert resolved.acquisition.overlap_samples == 0
    assert resolved.processing.result_stride_samples == 8
    assert resolved.processing.product_semantics == "cir_point_bin_slow_time_fft"
    assert resolved.acquisition.partial_window_policy == "retain_unprocessed"
    assert resolved.topology.rx_array_processing == "explicit_planar_rx_fft"
    assert resolved.persisted()["processing"]["mode"] == COHERENT_MODE
    assert "requested_mode" not in resolved.persisted()
    with pytest.raises(FrozenInstanceError):
        resolved.acquisition.cpi_length_samples = 4  # type: ignore[misc]

    processor = create_sensing_processor(config, 28e9, resolved)
    assert isinstance(processor, CoherentCpiFftProcessor)
    assert processor.execution_config is resolved

    state = SensingPipelineState.from_config(
        SimpleNamespace(sensing={"enabled": True}),
        frame_dt_s=0.001,
        gt_dt_s=0.008,
        first_step=0,
        source_generation_id="12345678-1234-5678-9234-567812345678",
        execution_config=resolved,
    )
    assert state.eval_range_gate_m == 10.0
    assert state.eval_velocity_gate_m_s == 5.0
    scheduler = FixedIntervalScheduler(
        first_step=0,
        last_step=15,
        interval_length=resolved.acquisition.rt_solve_stride_scene_steps,
    )
    assert state.execution_config is processor.execution_config is resolved
    assert [scheduler.event_for_step(step).is_acquisition_step for step in range(9)] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_pipeline_state_honors_explicit_evaluation_association_gates() -> None:
    _config, resolved = _resolve(
        COHERENT_MODE,
        cir_steps=8,
        fft_d=8,
        cir_rate_hz=1000.0,
    )
    state = SensingPipelineState.from_config(
        SimpleNamespace(
            sensing={
                "enabled": True,
                "eval_association_range_gate_m": 2.5,
                "eval_association_velocity_gate_m_s": 0.75,
            }
        ),
        frame_dt_s=0.001,
        gt_dt_s=0.008,
        first_step=0,
        source_generation_id="12345678-1234-5678-9234-567812345678",
        execution_config=resolved,
    )

    assert state.eval_range_gate_m == 2.5
    assert state.eval_velocity_gate_m_s == 0.75


def test_auto_is_input_only_and_persists_explicit_coherent_result() -> None:
    _config, resolved = _resolve(
        "auto",
        cir_steps=8,
        fft_d=8,
        cir_rate_hz=1000.0,
    )

    assert resolved.requested_mode == "auto"
    assert resolved.processing.mode == COHERENT_MODE
    assert "auto" not in str(resolved.persisted())


def test_sequential_contract_declares_sliding_independent_solve_window() -> None:
    config, resolved = _resolve(SEQUENTIAL_MODE, cir_steps=1, fft_d=8)

    assert resolved.acquisition.sample_source == "independent_solves"
    assert resolved.acquisition.cpi_length_samples == 8
    assert resolved.acquisition.cpi_hop_samples == 1
    assert resolved.acquisition.overlap_samples == 7
    assert resolved.processing.result_stride_samples == 1
    assert resolved.acquisition.phase_contract.endswith("controlled_specular_only")
    assert resolved.processing.evidence_status.endswith("experimental")
    processor = create_sensing_processor(config, 28e9, resolved)
    assert isinstance(processor, SequentialStftProcessor)
    assert processor.execution_config is resolved


def test_analytic_oracle_is_explicitly_quarantined() -> None:
    config, resolved = _resolve(ANALYTIC_ORACLE_MODE, cir_steps=1, fft_d=16)

    assert resolved.acquisition.sample_source == "analytic_path_snapshot"
    assert resolved.acquisition.cpi_length_samples == 1
    assert resolved.processing.product_semantics == "analytic_oracle_nonwaveform"
    assert resolved.processing.evidence_status == "diagnostic_quarantined"
    processor = create_sensing_processor(config, 28e9, resolved)
    assert isinstance(processor, AnalyticPathDopplerOracleProcessor)
    assert processor.execution_config is resolved


@pytest.mark.parametrize(
    ("mode", "cir_steps", "fft_d", "cir_rate", "frame_dt", "match"),
    [
        ("auto", 1, 8, None, 0.001, "ambiguous"),
        (COHERENT_MODE, 1, 8, None, 0.001, "cir_time_steps >= 2"),
        (COHERENT_MODE, 8, 8, None, 0.001, "cir_sampling_frequency_hz"),
        (COHERENT_MODE, 8, 8, 999.0, 0.001, "requires sensing.prf_hz to equal"),
        (COHERENT_MODE, 8, 16, 1000.0, 0.001, "fft_size_doppler to equal"),
        (SEQUENTIAL_MODE, 8, 8, 1000.0, 0.001, "cir_time_steps == 1"),
        (SEQUENTIAL_MODE, 1, 8, 1000.0, 0.001, "be omitted"),
        (SEQUENTIAL_MODE, 1, 8, None, 0.002, "frame interval"),
        (ANALYTIC_ORACLE_MODE, 8, 8, 1000.0, 0.001, "cir_time_steps == 1"),
    ],
)
def test_invalid_mode_cadence_matrix_fails(
    mode: str,
    cir_steps: int,
    fft_d: int,
    cir_rate: float | None,
    frame_dt: float,
    match: str,
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        _resolve(
            mode,
            cir_steps=cir_steps,
            fft_d=fft_d,
            cir_rate_hz=cir_rate,
            frame_dt_s=frame_dt,
        )


@pytest.mark.parametrize(
    ("mode", "tx_devices", "rx_devices", "tx_elements", "rx_elements", "match"),
    [
        (COHERENT_MODE, 2, 1, 1, 1, "exactly one transmitter device"),
        (COHERENT_MODE, 1, 2, 1, 1, "exactly one transmitter device"),
        (COHERENT_MODE, 1, 1, 2, 1, "requires one TX antenna element"),
        (SEQUENTIAL_MODE, 1, 1, 1, 2, "Phase-2 SISO only"),
        (ANALYTIC_ORACLE_MODE, 1, 1, 1, 2, "single-pair SISO diagnostic"),
    ],
)
def test_unsupported_topology_fails_before_processor_creation(
    mode: str,
    tx_devices: int,
    rx_devices: int,
    tx_elements: int,
    rx_elements: int,
    match: str,
) -> None:
    cir_steps = 8 if mode == COHERENT_MODE else 1
    cir_rate = 1000.0 if mode == COHERENT_MODE else None
    with pytest.raises(ConfigurationError, match=match):
        _resolve(
            mode,
            cir_steps=cir_steps,
            fft_d=8,
            cir_rate_hz=cir_rate,
            tx_devices=tx_devices,
            rx_devices=rx_devices,
            tx_elements=tx_elements,
            rx_elements=rx_elements,
        )


def test_unknown_and_removed_modes_do_not_fall_back() -> None:
    with pytest.raises(ConfigurationError, match="Invalid sensing.processing_mode"):
        SensingConfig(enabled=True, processing_mode="typo_mode")
    with pytest.raises(ConfigurationError, match="renamed"):
        SensingConfig(enabled=True, processing_mode="snapshot_direct_rd")


def test_loader_style_post_init_mutation_is_still_rejected_by_resolver() -> None:
    config = SensingConfig(enabled=True, processing_mode=COHERENT_MODE)
    config.processing_mode = "typo_after_construction"
    with pytest.raises(ConfigurationError, match="Invalid sensing.processing_mode"):
        resolve_sensing_execution_values(
            config,
            cir_time_steps=8,
            cir_sampling_frequency_hz=1000.0,
            frame_dt_s=0.001,
        )


@pytest.mark.parametrize("mode,rx_elements", [("coherent_cpi", 1), ("sequential_stft", 4)])
def test_beam_scan_rejects_unsupported_acquisition_before_solver(mode, rx_elements):
    cfg = SensingConfig(
        enabled=True,
        array_detection_mode="beam_scan",
        range_mode="monostatic",
        processing_mode=mode,
        fft_size_doppler=8,
    )
    with pytest.raises(ConfigurationError, match="beam_scan requires"):
        resolve_sensing_execution_values(
            cfg,
            cir_time_steps=8 if mode == "coherent_cpi" else 1,
            cir_sampling_frequency_hz=cfg.prf_hz,
            frame_dt_s=1 / cfg.prf_hz,
            rx_elements=rx_elements,
        )
