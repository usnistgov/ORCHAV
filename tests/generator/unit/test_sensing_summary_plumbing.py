"""Phase-3.5 plumbing tests for bounded sensing summary metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from generator.core.pipeline.sensing import SensingPipelineState
from generator.figures.summary import sensing as sensing_summary


def _summary_event(
    acquisition_id: str,
    *,
    result_bearing: bool,
    marker: str,
) -> dict[str, Any]:
    temporal: dict[str, Any] = {
        "acquisition": {
            "acquisition_id": acquisition_id,
            "samples": [
                {
                    "sample_index": 0,
                    "sample_time_s": 0.25,
                    "valid": True,
                }
            ],
        }
    }
    if result_bearing:
        temporal["result"] = {
            "processing_run_id": f"run-{marker}",
            "processing_recipe_id": f"recipe-{marker}",
        }
    return {
        "schema_version": 4,
        "processing_strategy": "sequential_stft",
        "config_resolved": {
            "processing_recipe": {
                "window_range": marker,
                "window_doppler": "hann",
            }
        },
        "temporal": temporal,
        "products": {
            "range_doppler_by_pair": {
                "data_key": "range_doppler_by_pair",
                "kind": "range_doppler",
                "processing": {"delay_taper": marker},
            }
        },
        "range_doppler_by_pair": np.ones((1, 2, 3), dtype=np.complex64),
        "doppler_axis_hz": np.asarray([-1.0, 1.0], dtype=np.float64),
        "range_axis_m": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    }


def _acquisition_ids(state: SensingPipelineState) -> list[str]:
    return [
        event["temporal"]["acquisition"]["acquisition_id"] for event in state.summary_sensing_events
    ]


def test_summary_history_keeps_one_warmup_and_first_two_results_without_arrays() -> None:
    state = SensingPipelineState(
        enabled=True,
        frame_dt_s=0.25,
        gt_dt_s=0.25,
        first_step=0,
    )
    first_warmup = _summary_event("warmup-1", result_bearing=False, marker="hann")

    state._record_summary_sensing_event(first_warmup)
    retained_warmup = state.summary_sensing_events[0]

    assert set(retained_warmup) == {
        "schema_version",
        "processing_strategy",
        "config_resolved",
        "temporal",
        "products",
    }
    assert not any(isinstance(value, np.ndarray) for value in retained_warmup.values())
    assert retained_warmup["products"] is not first_warmup["products"]
    assert (
        retained_warmup["products"]["range_doppler_by_pair"]["processing"]
        is not first_warmup["products"]["range_doppler_by_pair"]["processing"]
    )

    first_warmup["config_resolved"]["processing_recipe"]["window_range"] = "rect"
    first_warmup["temporal"]["acquisition"]["samples"][0]["sample_time_s"] = 99.0
    first_warmup["products"]["range_doppler_by_pair"]["processing"]["delay_taper"] = "rect"
    first_warmup["range_doppler_by_pair"].fill(0.0)

    assert retained_warmup["config_resolved"]["processing_recipe"]["window_range"] == "hann"
    assert retained_warmup["temporal"]["acquisition"]["samples"][0]["sample_time_s"] == 0.25
    assert (
        retained_warmup["products"]["range_doppler_by_pair"]["processing"]["delay_taper"] == "hann"
    )

    events = (
        _summary_event("warmup-2", result_bearing=False, marker="warmup-2"),
        _summary_event("result-1", result_bearing=True, marker="result-1"),
        _summary_event("result-1", result_bearing=True, marker="duplicate"),
        _summary_event("result-2", result_bearing=True, marker="result-2"),
        _summary_event("result-3", result_bearing=True, marker="result-3"),
    )
    for event in events:
        state._record_summary_sensing_event(event)

    assert _acquisition_ids(state) == ["warmup-1", "result-1", "result-2"]
    assert sum("result" not in event["temporal"] for event in state.summary_sensing_events) == 1
    assert sum("result" in event["temporal"] for event in state.summary_sensing_events) == 2


def test_pipeline_dispatches_summary_history_without_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture_summary(
        last_sensing_data: dict[str, Any],
        gt_history: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[Path]:
        captured["last_sensing_data"] = last_sensing_data
        captured["gt_history"] = gt_history
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        sensing_summary,
        "maybe_generate_sensing_summary",
        capture_summary,
    )
    last_sensing_data = {"schema_version": 4, "frame_role": "stft_ready"}
    event_history = [
        _summary_event("warmup", result_bearing=False, marker="warmup"),
        _summary_event("result", result_bearing=True, marker="result"),
    ]
    state = SensingPipelineState(
        enabled=True,
        frame_dt_s=0.25,
        gt_dt_s=0.25,
        first_step=0,
        last_sensing_data=last_sensing_data,
        summary_sensing_events=event_history,
        gt_history=[],
    )
    scenario = SimpleNamespace(
        generator_summary={"enabled": True, "create": ["sensing"]},
    )

    state.generate_summary(scenario, output_root=tmp_path, strict=True)

    assert captured["last_sensing_data"] is last_sensing_data
    assert captured["gt_history"] == []
    assert captured["gt_state"] is None
    assert captured["sensing_event_history"] is event_history
    assert captured["output_root"] == tmp_path
    assert captured["strict"] is True
