from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from visualizer.src.benchmarking import sensing_replay
from visualizer.src.benchmarking.sensing_replay import (
    SENSING_REPLAY_BENCHMARK_SCHEMA,
    SensingReplayBenchmarkEnvironment,
    SensingReplayBenchmarkSample,
    SensingReplayCacheObservation,
    aggregate_sensing_replay_samples,
    telemetry_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _environment() -> SensingReplayBenchmarkEnvironment:
    return SensingReplayBenchmarkEnvironment(
        python_version="3.12.12",
        python_implementation="CPython",
        operating_system="Linux",
        operating_system_release="test-kernel",
        machine="x86_64",
        numpy_version="2.2.6",
        h5py_version="3.14.0",
        git_commit="a" * 40,
        git_dirty=True,
        git_status_sha256="b" * 64,
        artifact_path="/tmp/threshold-sweep-a",
        artifact_manifest_id="event-a",
    )


def _sample(repetition: int, duration_ms: float) -> SensingReplayBenchmarkSample:
    rss_before = 1_000 + repetition * 10
    return SensingReplayBenchmarkSample(
        operation="indexed_seek",
        repetition=repetition,
        duration_ms=duration_ms,
        rss_before_bytes=rss_before,
        rss_peak_bytes=rss_before + 100 + repetition,
        rss_after_bytes=rss_before + 50,
        load_counts={
            "metadata_probes": 2 if repetition == 0 else 0,
            "selected_dense_loads": 1,
        },
    )


def test_aggregate_preserves_raw_samples_and_builds_deterministic_distributions() -> None:
    samples = (_sample(3, 4.0), _sample(1, 2.0), _sample(0, 1.0), _sample(2, 3.0))

    report = aggregate_sensing_replay_samples(_environment(), samples)
    reversed_report = aggregate_sensing_replay_samples(_environment(), reversed(samples))

    assert [sample.repetition for sample in report.samples] == [0, 1, 2, 3]
    summary = report.to_manifest()["operations"]["indexed_seek"]
    assert summary["duration_ms"] == pytest.approx(
        {
            "count": 4,
            "max": 4.0,
            "mean": 2.5,
            "p50": 2.5,
            "p95": 3.85,
            "stddev": 1.118033988749895,
        }
    )
    assert summary["rss_bytes"]["peak_delta"]["max"] == 103.0
    assert summary["load_counts"]["metadata_probes"]["mean"] == 0.5
    assert report.to_json() == reversed_report.to_json()
    assert json.loads(report.to_json())["schema"] == SENSING_REPLAY_BENCHMARK_SCHEMA
    assert "timestamp" not in report.to_json()


def test_benchmark_schema_import_is_qt_and_renderer_neutral() -> None:
    probe = """
import sys
import visualizer.src.benchmarking.sensing_replay

forbidden = sorted(
    name for name in sys.modules
    if name.startswith(('PySide6', 'pygfx', 'open3d', 'visualizer.src.renderers'))
)
if forbidden:
    raise SystemExit('renderer/runtime imports: ' + ', '.join(forbidden))
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_capture_environment_fingerprints_git_and_optional_real_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = " M visualizer/src/services/sensing_service.py\n"

    def fake_git(_root: Path, *arguments: str) -> str:
        return "c" * 40 + "\n" if arguments[0] == "rev-parse" else status

    monkeypatch.setattr(sensing_replay, "_git_output", fake_git)
    monkeypatch.setattr(sensing_replay.platform, "python_version", lambda: "3.12.12")
    monkeypatch.setattr(sensing_replay.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(sensing_replay.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sensing_replay.platform, "release", lambda: "test-kernel")
    monkeypatch.setattr(sensing_replay.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        sensing_replay,
        "_package_version",
        lambda name: {"numpy": "2.2.6", "h5py": "3.14.0"}[name],
    )

    environment = sensing_replay.capture_sensing_replay_environment(
        "/repo",
        artifact_path="/tmp/threshold-sweep-a",
        artifact_manifest_id="event-a",
    )

    assert environment.git_commit == "c" * 40
    assert environment.git_dirty is True
    assert environment.git_status_sha256 == hashlib.sha256(status.encode()).hexdigest()
    assert environment.artifact_path == "/tmp/threshold-sweep-a"
    assert environment.artifact_manifest_id == "event-a"
    assert environment.numpy_version == "2.2.6"
    assert environment.h5py_version == "3.14.0"


def test_samples_and_counter_deltas_fail_closed_without_timing_thresholds() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _sample(0, float("nan"))
    with pytest.raises(ValueError, match="RSS boundary"):
        SensingReplayBenchmarkSample(
            operation="bad_peak",
            repetition=0,
            duration_ms=1.0,
            rss_before_bytes=100,
            rss_peak_bytes=90,
            rss_after_bytes=95,
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        aggregate_sensing_replay_samples(_environment(), (_sample(0, 1.0), _sample(0, 2.0)))
    with pytest.raises(ValueError, match="decreased"):
        telemetry_delta({"metadata_probes": 2}, {"metadata_probes": 1})

    assert telemetry_delta(
        {"metadata_probes": 2, "selected_dense_loads": 1},
        {"metadata_probes": 2, "selected_dense_loads": 3, "stage_loads": 1},
    ) == {
        "metadata_probes": 0,
        "selected_dense_loads": 2,
        "stage_loads": 1,
    }


def test_cache_observation_proves_all_reported_occupancies_are_bounded() -> None:
    observation = SensingReplayCacheObservation(
        mode="indexed",
        worker_repetition=2,
        selection_index=11,
        result_id=191,
        metadata_entries=192,
        selected_product_entries=2,
        selected_product_capacity=2,
        detection_stage_entries=2,
        detection_stage_capacity=2,
        raw_cache_entries=3,
        raw_cache_capacity=32,
        raw_cache_evictions=160,
    )

    assert observation.to_manifest() == {
        "detection_stage_capacity": 2,
        "detection_stage_entries": 2,
        "metadata_entries": 192,
        "mode": "indexed",
        "raw_cache_capacity": 32,
        "raw_cache_entries": 3,
        "raw_cache_evictions": 160,
        "result_id": 191,
        "selected_product_capacity": 2,
        "selected_product_entries": 2,
        "selection_index": 11,
        "worker_repetition": 2,
    }
    with pytest.raises(ValueError, match="selected product cache entries 3 exceed capacity 2"):
        SensingReplayCacheObservation(
            mode="indexed",
            worker_repetition=0,
            selection_index=0,
            result_id=63,
            metadata_entries=192,
            selected_product_entries=3,
            selected_product_capacity=2,
            detection_stage_entries=0,
            detection_stage_capacity=2,
            raw_cache_entries=0,
            raw_cache_capacity=32,
            raw_cache_evictions=0,
        )
