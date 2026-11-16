"""Physical-time regressions for generator sensing summaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.hooks import collect_fast_time_samples
from generator.figures.summary import sensing as sensing_summary
from shared.sensing import signal_processing as sensing_algorithms


def _typed_delay_product(
    values: np.ndarray,
    *,
    kind: str,
    sample_times_s: list[float],
) -> dict:
    pair_id = "summary-pair"
    if kind == "delay_domain_cir":
        stored = values[np.newaxis, np.newaxis, :]
        data_key = "typed_cir"
        axes = (
            {"name": "pair", "unit": "1", "coordinate_kind": "catalog"},
            {"name": "rx_element", "unit": "1", "coordinate_kind": "catalog"},
            {"name": "range", "unit": "m", "coordinate_kind": "dense"},
        )
    else:
        stored = values[np.newaxis, np.newaxis, :, :]
        data_key = "typed_slow_time"
        axes = (
            {"name": "pair", "unit": "1", "coordinate_kind": "catalog"},
            {"name": "rx_element", "unit": "1", "coordinate_kind": "catalog"},
            {"name": "slow_time", "unit": "s", "coordinate_kind": "dense"},
            {"name": "range", "unit": "m", "coordinate_kind": "dense"},
        )
    for axis, coordinate_key in zip(
        axes,
        (
            ("pair_axis", "rx_element_axis", "slow_time_axis_s", "range_axis_m")
            if kind == "delay_domain_slow_time"
            else ("pair_axis", "rx_element_axis", "range_axis_m")
        ),
        strict=True,
    ):
        axis["coordinate_key"] = coordinate_key
    return {
        data_key: stored,
        "pair_axis": [
            {
                "pair_index": 0,
                "frame_pair_index": 0,
                "pair_id": pair_id,
                "tx_sensor_id": "summary-tx",
                "rx_sensor_id": "summary-rx",
            }
        ],
        "rx_element_axis": [{"element_index": 0, "row": 0, "column": 0}],
        "slow_time_axis_s": np.asarray(sample_times_s, dtype=np.float64),
        "range_axis_m": np.arange(values.shape[-1], dtype=np.float64),
        "products": {
            data_key: {
                "data_key": data_key,
                "kind": kind,
                "variant": "primary",
                "representation": "complex_voltage",
                "axes": list(axes),
                "pair_ids": [pair_id],
                "pair": {
                    "pair_id": pair_id,
                    "tx_sensor_id": "summary-tx",
                    "rx_sensor_id": "summary-rx",
                },
                "time_span": {"reference_time_s": sample_times_s[-1]},
                "combine": {"policy": "siso_identity", "normalization": "identity"},
            }
        },
        "temporal": {
            "acquisition": {
                "pair_id": pair_id,
                "tx_sensor_id": "summary-tx",
                "rx_sensor_id": "summary-rx",
                "samples": [
                    {
                        "sample_index": index,
                        "sample_time_s": sample_time_s,
                        "valid": True,
                    }
                    for index, sample_time_s in enumerate(sample_times_s)
                ],
            },
            "result": {"reference_time_s": sample_times_s[-1]},
        },
    }


def test_summary_sample_collection_uses_typed_products_and_deduplicates_windows() -> None:
    rows = np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64)
    samples: list[np.ndarray] = []
    sample_times_s: list[float] = []

    collect_fast_time_samples(
        _typed_delay_product(rows[0], kind="delay_domain_cir", sample_times_s=[0.0]),
        samples,
        sample_times_s,
    )
    collect_fast_time_samples(
        _typed_delay_product(rows[1], kind="delay_domain_cir", sample_times_s=[0.25]),
        samples,
        sample_times_s,
    )
    overlapping = _typed_delay_product(
        rows[1:],
        kind="delay_domain_slow_time",
        sample_times_s=[0.25, 0.5],
    )
    overlapping["cir"] = np.full(4, 99.0, dtype=np.complex64)
    overlapping["slow_time_matrix"] = np.full((2, 4), 123.0, dtype=np.complex64)
    collect_fast_time_samples(overlapping, samples, sample_times_s)

    assert len(samples) == 3
    np.testing.assert_array_equal(np.stack(samples), rows)
    assert sample_times_s == [0.0, 0.25, 0.5]


def test_pipeline_summary_passes_aligned_physical_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _capture_summary(*_args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        sensing_summary,
        "maybe_generate_sensing_summary",
        _capture_summary,
    )
    config = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        prf_hz=4.0,
        fft_size_doppler=2,
    )
    execution_config = resolve_sensing_execution_values(
        config,
        cir_time_steps=2,
        cir_sampling_frequency_hz=4.0,
        frame_dt_s=0.5,
    )
    state = SensingPipelineState(
        enabled=True,
        frame_dt_s=0.5,
        gt_dt_s=0.5,
        first_step=0,
        execution_config=execution_config,
        last_sensing_data={"config": {}},
        gt_history=[{"target_ranges": [1.0], "target_speeds": [0.0]}],
        fast_time_samples=[np.ones(4), np.ones(4)],
        fast_time_sample_times_s=[0.0, 0.25],
        analytical_paths=[{"delays": [1e-9]}, {"delays": [1e-9]}],
        analytical_path_sample_times_s=[0.0, 0.25],
    )
    scenario = SimpleNamespace(
        generator_summary={"enabled": True, "create": ["sensing"]},
    )

    state.generate_summary(scenario, output_root=tmp_path, strict=True)

    assert captured["frame_dt_s"] == 0.5
    assert captured["fast_time_sample_times_s"] == [0.0, 0.25]
    assert captured["analytical_path_sample_times_s"] == [0.0, 0.25]


def test_summary_uses_physical_sample_cadence_for_all_slow_time_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numerical_calls: dict[str, dict] = {}
    micro_doppler_plot_calls: list[dict] = []

    def _record_figure(*_args, output_path: Path, **_kwargs):
        return output_path

    def _record_algorithm(name: str, result=None):
        def _capture(*_args, **kwargs):
            numerical_calls[name] = kwargs
            return result

        return _capture

    monkeypatch.setattr(sensing_summary, "create_range_profile_figure", _record_figure)
    monkeypatch.setattr(sensing_summary, "create_range_doppler_figure", _record_figure)
    monkeypatch.setattr(sensing_summary, "create_gt_evolution_figure", _record_figure)
    monkeypatch.setattr(
        sensing_summary,
        "build_sensing_diagram_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        sensing_summary,
        "create_sensing_acquisition_processing_figure",
        _record_figure,
    )
    monkeypatch.setattr(
        sensing_summary,
        "write_sensing_diagram_manifest",
        lambda _model, output_path: output_path,
    )
    monkeypatch.setattr(
        sensing_summary,
        "create_micro_doppler_figure",
        lambda *_args, output_path, **kwargs: (
            micro_doppler_plot_calls.append(kwargs) or output_path
        ),
    )
    monkeypatch.setattr(
        sensing_algorithms,
        "auto_select_range_bin",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        sensing_algorithms,
        "compute_range_tracked_range_doppler",
        _record_algorithm("tracked_rd"),
    )
    monkeypatch.setattr(
        sensing_algorithms,
        "compute_micro_doppler",
        _record_algorithm("micro_doppler"),
    )
    monkeypatch.setattr(
        sensing_algorithms,
        "compute_range_tracked_micro_doppler",
        _record_algorithm("tracked_micro_doppler"),
    )
    monkeypatch.setattr(
        sensing_algorithms,
        "compute_range_collapsed_micro_doppler",
        _record_algorithm("collapsed_micro_doppler"),
    )

    scenario = SimpleNamespace(
        generator_summary={
            "enabled": True,
            "create": ["sensing"],
            "output": {"format": "png"},
        }
    )
    last_sensing_data = _typed_delay_product(
        np.ones((2, 4), dtype=np.complex64),
        kind="delay_domain_slow_time",
        sample_times_s=[0.0, 0.25],
    )
    last_sensing_data["config_resolved"] = {
        "fft_size_range": 4,
        "fft_size_doppler": 2,
        "range_mode": "monostatic",
        "carrier_freq_hz": 28e9,
        "processing_recipe": {
            "window_range": "hann",
            "window_doppler": "hann",
        },
    }
    last_sensing_data["detections"] = [{"range_m": 1.0, "power_linear": 1.0}]
    gt_history = [
        {
            "reference_time_s": time_s,
            "target_ids": ["walker"],
            "target_ranges": [1.0],
            "target_speeds": [0.0],
        }
        for time_s in (0.0, 0.25)
    ]
    sensing_summary.maybe_generate_sensing_summary(
        last_sensing_data,
        gt_history,
        signal_diagnostics=True,
        gt_state=gt_history[-1],
        scenario_context=scenario,
        fast_time_samples=[np.ones(4), np.ones(4)],
        fast_time_sample_times_s=[0.0, 0.25],
        target_id="walker",
        # The scenario frame spacing deliberately disagrees with the samples.
        frame_dt_s=0.5,
        output_root=tmp_path,
        strict=True,
    )

    assert set(numerical_calls) == {"collapsed_micro_doppler"}
    call = numerical_calls["collapsed_micro_doppler"]
    assert call["prf_hz"] == 4.0
    assert call["frame_duration_s"] == 0.25
    assert call["mti_enabled"] is False
    np.testing.assert_array_equal(call["range_axis_m"], [0.0, 1.0, 2.0, 3.0])


def test_summary_rejects_nonuniform_or_missing_physical_cadence() -> None:
    with pytest.raises(ValueError, match="resample onto a uniform physical-time grid"):
        sensing_summary._uniform_physical_sample_interval_s(
            [0.0, 0.25, 0.6],
            sample_count=3,
            label="Sensing summary",
        )

    with pytest.raises(ValueError, match="physical sample timestamps"):
        sensing_summary._uniform_physical_sample_interval_s(
            None,
            sample_count=2,
            label="Analytical sensing path sequence",
        )
