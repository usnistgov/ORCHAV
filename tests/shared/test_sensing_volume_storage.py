"""Storage consent and authenticated, restart-safe release of compact evidence."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scenarios.sensing.volume_storage import (
    confirm_storage,
    release_native_frames,
    retention_spec,
    storage_plan,
)


def prepared(tmp_path, mode="volume"):
    trials = []
    for i in range(2):
        native = tmp_path / f"trials/t{i}/native"
        native.mkdir(parents=True)
        (native / "scenario.yaml").write_text("{}")
        trials.append(
            dict(
                id=f"t{i}",
                native=f"trials/t{i}/native",
                status="pending",
                configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
            )
        )
    spec = dict(
        trials=trials, retention=retention_spec(trials, mode), estimated_native_mb_per_trial=1.5
    )
    (tmp_path / "volume_recipe.json").write_text(json.dumps(spec))
    return spec, trials[1]


def evidence(root, trial, *, marked=True):
    native = root / trial["native"]
    frames = native / "frames"
    frames.mkdir()
    hashes = {}
    for name in ("chunk.h5", "frames_manifest.json"):
        (frames / name).write_bytes(b"saved evidence")
        hashes[name] = hashlib.sha256(b"saved evidence").hexdigest()
    (frames / "user_notes.txt").write_text("leave this file alone")
    outcome = dict(trial, reports=[], source_hashes=hashes)
    if marked:
        outcome["native_evidence"] = "not_included"
    (native.parent / "outcome.json").write_text(json.dumps(outcome))
    return frames


def test_full_retention_never_releases_recordings(tmp_path):
    spec, t = prepared(tmp_path, "full")
    frames = evidence(tmp_path, t)
    release_native_frames(tmp_path, spec, t)
    assert (frames / "chunk.h5").exists()


def test_compact_keeps_reference_and_releases_only_authenticated_files(tmp_path):
    spec, t = prepared(tmp_path)
    reference = spec["trials"][0]
    reference_frames = evidence(tmp_path, reference)
    frames = evidence(tmp_path, t)
    release_native_frames(tmp_path, spec, reference)
    assert (reference_frames / "chunk.h5").exists()
    release_native_frames(tmp_path, spec, t)
    assert sorted(p.name for p in frames.iterdir()) == ["user_notes.txt"]
    assert (tmp_path / t["native"] / "scenario.yaml").exists()
    release_native_frames(tmp_path, spec, t)  # Restart after completed cleanup.


def test_cleanup_requires_persisted_reports_and_checks_all_hashes_first(tmp_path):
    spec, t = prepared(tmp_path)
    frames = evidence(tmp_path, t, marked=False)
    with pytest.raises(ValueError, match="Persist"):
        release_native_frames(tmp_path, spec, t)
    assert (frames / "chunk.h5").exists()
    outcome = frames.parent.parent / "outcome.json"
    row = json.loads(outcome.read_text())
    row["native_evidence"] = "not_included"
    outcome.write_text(json.dumps(row))
    (frames / "frames_manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        release_native_frames(tmp_path, spec, t)
    assert (frames / "chunk.h5").exists()


def test_cleanup_refuses_symlinks(tmp_path):
    spec, t = prepared(tmp_path)
    frames = evidence(tmp_path, t)
    (frames / "chunk.h5").unlink()
    destination = tmp_path / "unrelated.h5"
    destination.write_bytes(b"saved evidence")
    (frames / "chunk.h5").symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        release_native_frames(tmp_path, spec, t)
    assert destination.exists()


def test_confirmation_is_default_no_and_batch_requires_explicit_yes(tmp_path, monkeypatch, capsys):
    prepared(tmp_path, "full")
    monkeypatch.setattr(
        "scenarios.sensing.volume_storage.sys.stdin", SimpleNamespace(isatty=lambda: False)
    )
    with pytest.raises(SystemExit, match="--yes"):
        confirm_storage([tmp_path])
    assert "retention=full" in capsys.readouterr().out
    confirm_storage([tmp_path], yes=True)
    monkeypatch.setattr(
        "scenarios.sensing.volume_storage.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(SystemExit, match="cancelled"):
        confirm_storage([tmp_path])
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    confirm_storage([tmp_path])
    assert not list(tmp_path.glob("trials/*/native/frames"))


def test_estimates_include_partition_limit_and_compact_reference(tmp_path):
    spec, t = prepared(tmp_path)
    plan = storage_plan([tmp_path], worker=None, workers=2, limit=1)[0]
    assert plan["trials"] == 2 and plan["remaining"] == 2
    assert plan["native_recordings"] == 1
    assert plan["estimated_final_gb"] == pytest.approx((1.5 + 0.064) / 1000)
    evidence(tmp_path, t)
    plan = storage_plan([tmp_path], worker=None, workers=2, limit=1)[0]
    assert plan["remaining"] == 1
