"""Native summaries use recorded radial track states at their source-scene steps."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from generator.core.pipeline.sensing import SensingPipelineState
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload


@pytest.mark.parametrize("interpolated_midpoints", [False, True])
def test_native_tracking_summary_receives_saved_states_and_source_epoch(
    tmp_path, monkeypatch, interpolated_midpoints
):
    state = SensingPipelineState(
        enabled=True,
        frame_dt_s=0.001,
        gt_dt_s=0.001,
        first_step=0,
        collect_summary_tracking=True,
        helpers=SimpleNamespace(finalize_sensing_v4_publication=lambda _: None),
    )
    steps = (31, 32) if interpolated_midpoints else (0, 128)
    times = (0.0315, 0.0325) if interpolated_midpoints else (0, 0.128)
    for index, (step, reference_time) in enumerate(zip(steps, times, strict=True)):
        detection = {
            "range_m": 10 - index,
            "velocity_m_s": 1.0,
            "track_id": 4,
            "track_range_m": 10.2 - index,
            "track_velocity_m_s": 0.9,
            "track_confirmed": step > 0,
        }
        payload = build_visualizer_sensing_payload(
            {"detections": [detection]}, scene_frame_id=step, sample_time_s=reference_time
        )
        original = deepcopy(payload)
        state._finalize_published_sensing({"sensing": payload, "frame_idx": step + 127})
        assert payload == original
        state.gt_history.append(
            {
                **({} if interpolated_midpoints else {"step": step}),
                "reference_time_s": reference_time,
                "target_ids": [],
                "target_ranges": [],
                "target_speeds": [],
                "target_velocity_valid": [],
            }
        )
    assert [d["frame"] for d in state.summary_tracking_detections] == list(steps)
    assert [d["track_range_m"] for d in state.summary_tracking_detections] == [10.2, 9.2]
    state.last_sensing_data = payload
    scenario = SimpleNamespace(
        generator_summary={
            "enabled": True,
            "create": ["sensing"],
            "sensing": {"figures": ["tracking"], "history_time_axis": "time"},
        }
    )
    from generator.figures.summary import sensing as summary

    original_history = deepcopy(state.gt_history)
    render = summary.create_tracking_figure
    rendered_histories = []

    def record_history(history, *args, **kwargs):
        rendered_histories.append(deepcopy(history))
        return render(history, *args, **kwargs)

    monkeypatch.setattr(summary, "create_tracking_figure", record_history)
    state.generate_summary(scenario, output_root=tmp_path, strict=True)
    assert (tmp_path / "sensing/tracking.png").is_file()
    assert state.gt_history == original_history
    assert [entry["step"] for entry in rendered_histories[0]] == list(steps)
    assert [entry["reference_time_s"] for entry in rendered_histories[0]] == list(times)
    assert SensingPipelineState(True, 0.001, 0.001, 0).collect_summary_tracking is False
