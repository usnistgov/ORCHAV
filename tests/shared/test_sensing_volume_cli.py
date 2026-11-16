"""Volume CLI progress, resume accounting, and non-destructive preparation."""

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from generator.core.pipeline.progress import ProgressInfo
from scenarios.sensing import volume_tools
from scenarios.sensing.applications.wall_detection_volume import generate, parallel
from scenarios.sensing.volume_cli import prepare_volume
from scenarios.sensing.volume_progress import VolumeProgress


def campaign(root, count=5):
    root.mkdir(exist_ok=True)
    trials = []
    for i in range(count):
        native = root / f"trials/t{i}/native"
        native.mkdir(parents=True)
        (native / "scenario.yaml").write_text("{}")
        trials.append(
            dict(
                id=f"t{i}",
                status="pending",
                native=str(native.relative_to(root)),
                configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
                position_m=[3.0, 0.0, 2.0],
                velocity_m_s=[1.0, 0.0, 0.0],
            )
        )
    spec = dict(
        trials=[*trials, {"id": "excluded", "status": "excluded_geometry"}],
        scene_scenario="scenario.yaml",
        grid_preset="tutorial",
        cpi_samples=2,
        sensor_position_m=[-6.0, 0.0, 2.0],
        position_gate_m=0.75,
    )
    (root / "scenario.yaml").write_text("{}")
    (root / "volume_recipe.json").write_text(json.dumps(spec))
    return trials


def test_resume_counts_saved_selected_trials_and_preserves_partition_limit(
    tmp_path, monkeypatch, capsys
):
    trials = campaign(tmp_path)
    saved_path = tmp_path / "trials/t1/outcome.json"
    saved_path.write_text('{"saved": true}')
    generated = []

    def generate_frames(native, *, on_step_complete):
        generated.append(native.parent.name)
        frames = native / "frames"
        frames.mkdir()
        for step in range(2):
            on_step_complete(ProgressInfo(step, 2, step + 1, 1))
        (frames / "frames_manifest.json").write_text('{"frame_ids": [0, 1]}')

    monkeypatch.setitem(
        sys.modules, "generator.__main__", SimpleNamespace(_run_from_yaml=generate_frames)
    )
    monkeypatch.setattr(
        volume_tools,
        "load_sensor",
        lambda *_: {
            "epochs": [{"observations": []}],
            "truth": [{"available": True, "targets": [trials[3]]}],
            "source": {"hashes": {}},
        },
    )
    monkeypatch.setattr(volume_tools, "saved_reports", lambda _: [])
    volume_tools.run_trials(tmp_path, worker=1, workers=2, limit=2)
    output = capsys.readouterr()
    assert generated == ["t3"]
    assert saved_path.read_text() == '{"saved": true}'
    outcome = json.loads((tmp_path / "trials/t3/outcome.json").read_text())
    assert (outcome["status"], outcome["tp"], outcome["fp"], outcome["fn"]) == ("measured", 0, 0, 1)
    assert "2 selected trials | 1 already saved | 1 remaining | 1 geometry exclusions" in output.out
    assert "2/2 trials complete (1 new, 1 already saved)" in output.err
    assert "frames 2/2" in output.err
    assert "Verifying frames / scoring detections" in output.err
    assert '"event"' not in output.err + output.out
    assert "\r" not in output.err


def test_failure_is_not_counted_as_completion(tmp_path, monkeypatch, capsys):
    campaign(tmp_path, count=1)

    def fail(*args, **kwargs):
        raise RuntimeError("ray tracer failed")

    monkeypatch.setitem(sys.modules, "generator.__main__", SimpleNamespace(_run_from_yaml=fail))
    with pytest.raises(RuntimeError, match="ray tracer failed"):
        volume_tools.run_trials(tmp_path, 0, 1, None)
    output = capsys.readouterr().err
    assert "Failed" in output and "0/1" in output
    assert "Finished" not in output
    assert not (tmp_path / "trials/t0/outcome.json").exists()


def test_empty_partition_and_all_saved_need_no_generator(tmp_path, monkeypatch, capsys):
    campaign(tmp_path, count=1)
    (tmp_path / "trials/t0/outcome.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "generator.__main__", None)
    volume_tools.run_trials(tmp_path, 0, 1, None)
    volume_tools.run_trials(tmp_path, 1, 2, None)
    output = capsys.readouterr().err
    assert "1/1 trials complete (0 new, 1 already saved)" in output
    assert "0/0 trials complete (0 new, 0 already saved)" in output


@pytest.mark.parametrize("terminal", [False, True])
def test_progress_throttles_frames_and_eta_excludes_saved_trials(monkeypatch, terminal):
    now = [0.0]
    monkeypatch.setattr("scenarios.sensing.volume_progress.time.monotonic", lambda: now[0])
    stream = io.StringIO()
    stream.isatty = lambda: terminal
    progress = VolumeProgress("wall", 12, completed=10, stream=stream)
    progress.start_trial({"id": "a", "position_m": [1, 2, 3]}, 64)
    for step in range(64):
        progress.step_completed(ProgressInfo(step, 64, 0.0, 0.0))
    # Frame callbacks do not flood either stream; first/last frames stay visible.
    assert stream.getvalue().count("| frames ") <= 3
    assert "ETA estimating" in stream.getvalue()
    now[0] = 20.0
    progress.complete_trial({"hit": True, "fp": 0, "wall_seconds": 20})
    assert "ETA 00:20" in stream.getvalue()
    assert ("\r" in stream.getvalue()) is terminal
    stream.seek(0)
    stream.truncate()
    now[0] = 40.0
    progress.render()  # The heartbeat also renders without a frame callback.
    assert "00:40" in stream.getvalue()


def test_repeated_wall_prepare_exits_cleanly_and_keeps_all_files(tmp_path, monkeypatch, capsys):
    root = tmp_path / "wall tutorial"
    argv = [str(Path(generate.__file__)), str(root), "--prepare", "--grid", "tutorial"]
    monkeypatch.setattr(sys, "argv", argv)
    generate.main()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    capsys.readouterr()
    generate.main()
    output = capsys.readouterr()
    assert "Already prepared:" in output.out
    assert "grid=tutorial" in output.out
    assert "python -m scenarios.sensing.applications.wall_detection_volume.generate" in output.out
    assert "--yes" in output.out
    assert "Traceback" not in output.out + output.err
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


def test_existing_partial_directory_is_not_claimed_to_be_prepared(tmp_path):
    (tmp_path / "my_notes.txt").write_text("keep me")
    with pytest.raises(SystemExit, match="incomplete or unrecognized"):
        prepare_volume(tmp_path, lambda *_: pytest.fail("must not prepare over existing files"))
    assert (tmp_path / "my_notes.txt").read_text() == "keep me"


def test_parallel_progress_counts_only_selected_trials(tmp_path, capsys):
    campaign(tmp_path)
    (tmp_path / "trials/t0/outcome.json").write_text("{}")
    worker = tmp_path / "worker.py"
    worker.write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--worker", type=int)
p.add_argument("--workers", type=int)
p.add_argument("--limit", type=int)
p.add_argument("--yes", action="store_true")
a = p.parse_args()
spec = json.loads((a.root / "volume_recipe.json").read_text())
trials = [t for t in spec["trials"] if t["status"] == "pending"]
for t in trials[a.worker::a.workers][:a.limit]:
    (a.root / t["native"]).parent.joinpath("outcome.json").write_text("{}")
""")
    parallel.run_parallel(tmp_path, [0, 1], 1, limit=1, script=worker)
    output = capsys.readouterr().err
    assert "All workers" in output
    assert "2/2 trials complete (1 new, 1 already saved)" in output
    assert "logs:" in output
    assert not (tmp_path / "trials/t2/outcome.json").exists()
    records = list(tmp_path.glob("worker_logs/*/workers.json"))
    assert len(records) == 1
    assert all(w["returncode"] == 0 for w in json.loads(records[0].read_text())["workers"])


def test_parallel_failure_remains_visible_and_keeps_worker_logs(tmp_path, capsys):
    campaign(tmp_path, count=1)
    worker = tmp_path / "failed_worker.py"
    worker.write_text('raise RuntimeError("worker failure detail")')
    with pytest.raises(RuntimeError, match="Native worker failed"):
        parallel.run_parallel(tmp_path, [0], 1, script=worker)
    output = capsys.readouterr().err
    assert "Failed" in output and "Finished" not in output
    assert "worker failure detail" in next(tmp_path.glob("worker_logs/*/*.log")).read_text()
