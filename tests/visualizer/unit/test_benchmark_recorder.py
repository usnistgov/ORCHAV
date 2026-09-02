from __future__ import annotations

import json

from visualizer.src.benchmarking.recorder import BenchmarkRecorder


def test_benchmark_recorder_summary_and_metadata(tmp_path):
    output = tmp_path / "bench.json"
    recorder = BenchmarkRecorder(n_frames=3, n_warmup=1, output_path=output)
    recorder.set_metadata("renderer", "pygfx")
    recorder.set_metadata(
        "startup_timing_profile",
        {
            "scenario_open_stages_ms": {"cleanup_previous_scene_ms": 12.3},
            "first_frame_pipeline_ms": {"apply_ms": 7.9},
        },
    )
    recorder.set_runtime_stats(
        {
            "startup_to_first_frame_ms": 42.5,
            "present_failures": 3,
            "event_pump_calls": 0,
            "redraw_requests": 12,
        }
    )

    for step, total_ms in enumerate((10.0, 20.0, 30.0)):
        recorder.record_prepare_step(0.5 + step)
        recorder.begin_frame(step)
        recorder.record_load(1.0 + step)
        recorder.record_viewmodel(2.0 + step)
        recorder.record_render(3.0 + step)
        recorder.record_total_before_end(4.0 + step)
        recorder.record_breakdowns(
            {
                "canonical_lookup_ms": 0.5 + step,
                "filter_ms": 1.5 + step,
                "canonical_cache_hit": 1.0 if step else 0.0,
            }
        )
        recorder.record_breakdown_bytes_many(
            {
                "pygfx_push_buffer_bytes": 1024 * (step + 1),
                "pygfx_push_buffer_line_positions_bytes": 256 * (step + 1),
            }
        )
        recorder.record_geometry(100 + step, 200 + step)
        recorder.end_frame(total_ms)

    path = recorder.finalize()
    assert path == output
    assert output.exists()

    data = json.loads(output.read_text())
    assert data["metadata"]["renderer"] == "pygfx"
    assert data["metadata"]["startup_timing_profile"]["first_frame_pipeline_ms"]["apply_ms"] == 7.9
    assert "startup_to_first_frame_ms" not in data["metadata"]
    assert data["runtime_stats"]["startup_to_first_frame_ms"] == 42.5
    assert data["runtime_stats"]["present_failures"] == 3
    assert data["runtime_stats"]["redraw_requests"] == 12
    assert data["metadata"]["n_frames"] == 3
    assert data["metadata"]["n_warmup"] == 1
    assert data["metadata"]["n_timed"] == 2
    assert data["metadata"]["wall_update_rate_hz"] > 0.0

    summary = data["summary"]
    assert "avg_total_ms" in summary
    assert "p95_total_ms" in summary
    assert summary["avg_total_ms"] == 25.0
    assert summary["avg_total_fps_equiv"] == 40.0
    assert summary["p95_total_fps_equiv"] == 33.898
    assert summary["avg_render_ms"] == 4.5
    assert summary["avg_render_fps_equiv"] == 222.222
    assert summary["avg_prepare_step_ms"] == 2.0
    assert summary["avg_total_before_end_ms"] == 5.5
    assert summary["avg_mpc_points"] == 101.5
    assert summary["avg_mpc_lines"] == 201.5
    assert summary["avg_breakdown_ms"]["canonical_lookup_ms"] == 2.0
    assert summary["avg_breakdown_ms"]["filter_ms"] == 3.0
    assert summary["avg_breakdown_ms"]["canonical_cache_hit"] == 1.0
    assert summary["p95_breakdown_ms"]["filter_ms"] == 3.45
    assert summary["avg_breakdown_bytes"]["pygfx_push_buffer_bytes"] == 2560.0
    assert summary["p95_breakdown_bytes"]["pygfx_push_buffer_line_positions_bytes"] == 755.2
    assert data["timed"][0]["breakdown_ms"]["filter_ms"] == 2.5
    assert data["timed"][0]["breakdown_bytes"]["pygfx_push_buffer_bytes"] == 2048
    assert data["timed"][0]["prepare_step_ms"] == 1.5
