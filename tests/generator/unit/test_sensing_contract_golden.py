from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from generator.core.configuration import SensingConfig
from generator.core.sensing.acquisition import (
    ANALYTIC_ORACLE_MODE,
    resolve_sensing_execution_values,
)
from generator.core.sensing.processor import create_sensing_processor


class _FakePaths:
    def __init__(self, amplitudes: list[complex], delay_s: float = 1e-9):
        self.amplitudes = amplitudes
        self.delay_s = delay_s
        self._idx = 0

    def cir(self, normalize_delays: bool = False):
        idx = min(self._idx, len(self.amplitudes) - 1)
        self._idx += 1
        amp = np.array([self.amplitudes[idx]], dtype=np.complex128)
        tau = np.array([self.delay_s], dtype=np.float64)
        return amp, tau

    @property
    def doppler(self) -> np.ndarray:
        return np.array([100.0], dtype=np.float64)


def _processor(config: SensingConfig, *, cir_steps: int):
    resolved = resolve_sensing_execution_values(
        config,
        cir_time_steps=cir_steps,
        cir_sampling_frequency_hz=(config.prf_hz if cir_steps > 1 else None),
        frame_dt_s=1.0 / config.prf_hz,
    )
    return create_sensing_processor(config, 28e9, resolved)


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sensing" / name
    with open(fixture_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _extract_contract_metadata(sensing: dict) -> dict:
    cfg = sensing.get("config_resolved") or {}
    return {
        "schema_version": sensing.get("schema_version"),
        "processing_strategy": sensing.get("processing_strategy"),
        "rd_source": sensing.get("rd_source"),
        "rd_model": sensing.get("rd_model"),
        "frame_role": sensing.get("frame_role"),
        "pair_index": sensing.get("pair_index"),
        "raw_payload_policy": sensing.get("raw_payload_policy"),
        "rd_required_history": sensing.get("rd_required_history"),
        "rd_ready": sensing.get("rd_ready"),
        "rd_map_frames_used": sensing.get("rd_map_frames_used"),
        "rd_fresh": sensing.get("rd_fresh"),
        "rp_fresh": sensing.get("rp_fresh"),
        "source_frame_step": sensing.get("source_frame_step"),
        "range_mode": sensing.get("range_mode"),
        "config_resolved": {
            "raw_payload_policy": cfg.get("raw_payload_policy"),
            "rd_required_history": cfg.get("rd_required_history"),
        },
    }


def test_analytic_path_doppler_oracle_contract_matches_golden_fixture() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=False,
        fft_size_range=32,
        fft_size_doppler=4,
        processing_mode=ANALYTIC_ORACLE_MODE,
    )
    processor = _processor(cfg, cir_steps=1)
    result = processor.process(
        None,
        7,
        _FakePaths([1.0 + 0j]),
        source_time_s=3.5,
    )

    assert result is not None
    assert "schema_version" not in result
    assert "temporal" not in result
    assert _extract_contract_metadata(result) == _load_fixture(
        "analytic_path_doppler_oracle_contract.yaml"
    )


def test_coherent_cpi_fft_contract_matches_golden_fixture() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="none",
        processing_mode="coherent_cpi",
    )
    processor = _processor(cfg, cir_steps=4)
    a = np.array([[1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j]], dtype=np.complex128)
    tau = np.array([1e-9], dtype=np.float64)
    result = processor.process(
        None,
        frame_idx=11,
        paths=None,
        cir_stack=(a, tau),
        source_time_s=5.5,
    )

    assert result is not None
    assert "schema_version" not in result
    assert "temporal" not in result
    assert _extract_contract_metadata(result) == _load_fixture("coherent_cpi_fft_contract.yaml")
