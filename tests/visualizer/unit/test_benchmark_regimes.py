from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from shared.frames.contracts import MPC_HDF5_LAYOUT, MPC_HDF5_SCHEMA_VERSION
from shared.frames.manifest import (
    FrameChunkManifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from visualizer.src.benchmarking.regimes import build_standard_regimes, summarize_benchmark_json
from visualizer.src.benchmarking.scenario_profile import profile_generated_scenario


def test_build_standard_regimes():
    regimes = build_standard_regimes(frames=120, warmup=5, play_frames=20)
    assert [regime.name for regime in regimes] == ["cold", "warm", "play"]
    assert regimes[0].previsit_all_frames is False
    assert regimes[0].present_mode == "request"
    assert regimes[1].previsit_all_frames is True
    assert regimes[1].present_mode == "request"
    assert regimes[2].warmup == 0
    assert regimes[2].frames == 20
    assert regimes[2].present_mode == "blocking"
    assert "may coalesce" in regimes[0].description
    assert "may coalesce" in regimes[1].description
    assert "backend draw" in regimes[2].description
    assert "physical present" in regimes[2].description


def test_summarize_benchmark_json(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "n_frames": 12,
                    "n_timed": 10,
                    "wall_update_rate_hz": 41.25,
                    "benchmark_previsit_all_frames": True,
                    "benchmark_present_mode": "request",
                    "previsit_wall_ms": 123.4,
                },
                "runtime_stats": {
                    "startup_to_first_frame_ms": 456.7,
                    "avg_update_to_present_ms": 78.9,
                    "benchmark_draw_callbacks": 15,
                    "renderer_content_size": [1920.0, 1080.0],
                    "renderer_window_size": [1280.0, 720.0],
                    "renderer_internal_size": [1920.0, 1080.0],
                    "renderer_pixel_scale": 1.0,
                    "renderer_pixel_ratio": 1.5,
                },
                "summary": {
                    "avg_prepare_step_ms": 1.2,
                    "avg_total_before_end_ms": 7.8,
                    "avg_total_ms": 12.3,
                    "p95_total_ms": 20.1,
                    "p95_total_before_end_ms": 13.2,
                    "avg_viewmodel_ms": 3.4,
                    "avg_render_ms": 5.6,
                    "avg_breakdown_ms": {
                        "end_frame_update_ms": 4.5,
                        "geometry_update_ms": 6.7,
                        "canonical_lookup_ms": 0.9,
                        "targets_ms": 0.1,
                        "request_draw_ms": 0.2,
                        "draw_callback_total_ms": 3.0,
                        "renderer_submit_ms": 1.5,
                        "canvas_present_residual_ms": 7.5,
                    },
                },
                "timed": [{"step": 63, "total_ms": 10.5}],
            }
        )
    )

    summary = summarize_benchmark_json(path)

    assert summary["n_timed"] == 10
    assert summary["wall_update_rate_hz"] == 41.25
    assert summary["benchmark_previsit_all_frames"] is True
    assert summary["benchmark_present_mode"] == "request"
    assert summary["previsit_wall_ms"] == 123.4
    assert summary["startup_to_first_frame_ms"] == 456.7
    assert summary["avg_prepare_step_ms"] == 1.2
    assert summary["avg_total_before_end_ms"] == 7.8
    assert summary["avg_end_frame_update_ms"] == 4.5
    assert summary["avg_geometry_update_ms"] == 6.7
    assert summary["avg_canonical_lookup_ms"] == 0.9
    assert summary["avg_targets_ms"] == 0.1
    assert summary["avg_request_draw_ms"] == 0.2
    assert summary["avg_draw_callback_total_ms"] == 3.0
    assert summary["avg_renderer_submit_ms"] == 1.5
    assert summary["avg_canvas_present_residual_ms"] == 7.5
    assert summary["renderer_content_size"] == [1920.0, 1080.0]
    assert summary["renderer_window_size"] == [1280.0, 720.0]
    assert summary["renderer_internal_size"] == [1920.0, 1080.0]
    assert summary["renderer_pixel_scale"] == 1.0
    assert summary["renderer_pixel_ratio"] == 1.5
    assert summary["benchmark_draw_callbacks"] == 15
    assert summary["draw_callbacks_per_frame"] == 1.25
    assert summary["avg_total_ms"] == 12.3
    assert summary["first_timed_step"] == 63
    assert summary["first_timed_total_ms"] == 10.5


def test_summarize_blocking_benchmark_uses_force_draw_callback_count(tmp_path):
    path = tmp_path / "blocking.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "n_frames": 60,
                    "benchmark_present_mode": "blocking",
                },
                "runtime_stats": {
                    # One unrelated deferred callback occurred during the
                    # benchmark interval, but each blocking frame itself drew
                    # exactly once.
                    "benchmark_draw_callbacks": 61,
                    "blocking_frame_count": 60,
                    "blocking_force_draw_callbacks": 60,
                },
            }
        )
    )

    summary = summarize_benchmark_json(path)

    assert summary["benchmark_draw_callbacks"] == 61
    assert summary["blocking_frame_count"] == 60
    assert summary["blocking_force_draw_callbacks"] == 60
    assert summary["draw_callbacks_per_frame"] == 1.0


def test_summarize_open3d_blocking_benchmark_reports_redraw_pump_parity(tmp_path):
    path = tmp_path / "open3d_blocking.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "n_frames": 50,
                    "benchmark_present_mode": "blocking",
                },
                "runtime_stats": {
                    "presentation_observable": False,
                    "benchmark_frame_submissions": 50,
                    "benchmark_redraw_pump_attempts": 50,
                    "benchmark_redraw_pump_alive": 50,
                },
            }
        )
    )

    summary = summarize_benchmark_json(path)

    assert summary["benchmark_frame_submissions"] == 50
    assert summary["benchmark_redraw_pump_attempts"] == 50
    assert summary["benchmark_redraw_pump_alive"] == 50
    assert summary["redraw_pumps_per_submission"] == 1.0


def _write_profile_fixture(
    tmp_path: Path,
    *,
    num_mpcs: list[int],
    num_targets: list[int],
    expected_targets: int,
) -> Path:
    scenario_dir = tmp_path / "scenario_case"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {"mode": "files", "files": {"format": "hdf5"}},
                "actors": {
                    "targets": [{"name": f"T{i}"} for i in range(expected_targets)],
                },
            },
            sort_keys=False,
        )
    )
    frame_ids = tuple(range(len(num_mpcs)))
    chunk_name = f"mpc_frames_{frame_ids[0]:05d}-{frame_ids[-1]:05d}.h5"
    chunk_path = frames_dir / chunk_name
    path_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(num_mpcs, dtype=np.int64))
    )
    target_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(num_targets, dtype=np.int64))
    )
    with h5py.File(chunk_path, "w") as handle:
        handle.attrs["file_kind"] = "mpc_frames"
        handle.attrs["schema_version"] = MPC_HDF5_SCHEMA_VERSION
        handle.attrs["storage_layout"] = MPC_HDF5_LAYOUT
        handle.create_dataset("frames/id", data=frame_ids)
        handle.create_dataset(
            "index/frame_pair_path_offsets",
            data=np.column_stack((path_offsets[:-1], path_offsets[1:])),
        )
        handle.create_dataset(
            "index/frame_target_offsets",
            data=target_offsets,
        )
    chunk = FrameChunkManifest(
        file=chunk_name,
        frame_ids=frame_ids,
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=0,
        topology_id="profile-fixture",
        sensing_layout_id="",
    )
    manifest = manifest_from_chunks(
        generation_id="profile-fixture-generation",
        frame_set_id="profile-fixture-frame-set",
        chunks=[chunk],
        compression={"filter": "none", "shuffle": False},
        segmentation={"frame_limit": len(frame_ids)},
        provenance={"test": "_write_profile_fixture"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return scenario_dir / "scenario.yaml"


def test_profile_generated_scenario_ok(tmp_path):
    scenario_yaml = _write_profile_fixture(
        tmp_path,
        num_mpcs=[3, 4, 2, 1],
        num_targets=[1, 1, 1, 1],
        expected_targets=1,
    )
    profile = profile_generated_scenario(scenario_yaml)
    assert profile["status"] == "ok"
    assert profile["num_mpcs"]["zero_frames"] == 0
    assert profile["num_targets"]["max"] == 1


def test_profile_generated_scenario_flags_obvious_failures(tmp_path):
    scenario_yaml = _write_profile_fixture(
        tmp_path,
        num_mpcs=[0, 0, 0, 0],
        num_targets=[0, 0, 0, 0],
        expected_targets=1,
    )
    profile = profile_generated_scenario(scenario_yaml)
    assert profile["status"] == "fail"
    assert "all_zero_mpcs" in profile["issues"]
    assert "targets_missing_from_export" in profile["issues"]
