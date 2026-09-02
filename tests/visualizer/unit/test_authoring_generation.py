"""Focused tests for direct Scenario Builder generation publication."""

from __future__ import annotations

import sys
import time
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from shared.frames.manifest import load_frame_manifest
from tests.visualizer.fixtures.packed_v2 import write_real_frame_set
from visualizer.src.authoring.generation import (
    GenerationJob,
    GenerationPaths,
    GenerationState,
    JsonlEventDecoder,
)

_HELPER_SOURCE = r"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

expected_source_identity = json.loads(os.environ["ORCHAV_EXPECTED_SOURCE_IDENTITY"])
sys.path.insert(0, expected_source_identity["source_root"])
from tests.visualizer.fixtures.packed_v2 import write_real_frame_set

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--progress-format", required=True)
parser.add_argument("scenario_root")
parser.add_argument("--_authoring-snapshot-yaml", required=True)
args = parser.parse_args()

scenario_root = Path(args.scenario_root)
snapshot = Path(args._authoring_snapshot_yaml)
assert scenario_root.is_dir()
assert snapshot.parent == scenario_root
assert snapshot.name.startswith(".orchav-aq-")

identity = expected_source_identity
if args.mode == "wrong-identity":
    identity = {
        "schema_version": 1,
        "source_root": str(scenario_root / "wrong-source"),
        "version": "0.1.0",
        "git_sha": None,
    }
elif args.mode == "invalid-identity":
    identity = None

print(json.dumps({
    "schema_version": 1,
    "event": "run_started",
    "first_step": 0,
    "total_steps": 1,
    "source_identity": identity,
}), flush=True)
sys.stderr.write("generator diagnostic\n")
sys.stderr.flush()

if args.mode == "cancel":
    while True:
        time.sleep(1)

if args.mode == "failure":
    print(json.dumps({
        "schema_version": 1,
        "event": "run_failed",
        "completed_steps": 0,
        "total_steps": 1,
        "message": "deliberate failure",
        "error_type": "RuntimeError",
    }), flush=True)
    raise SystemExit(3)

if args.mode == "no-completion":
    raise SystemExit(0)

if args.mode not in {"unchanged", "wrong-identity", "invalid-identity"}:
    write_real_frame_set(scenario_root / "frames")

print(json.dumps({
    "schema_version": 1,
    "event": "step_completed",
    "step": 0,
    "completed_steps": 1,
    "total_steps": 1,
    "elapsed_s": 0.1,
    "step_duration_s": 0.1,
}), flush=True)
print(json.dumps({
    "schema_version": 1,
    "event": "run_completed",
    "completed_steps": 1,
    "total_steps": 1,
    "output": "direct",
}), flush=True)
"""


@pytest.fixture
def scenario(tmp_path: Path) -> tuple[Path, str]:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    source_text = """\
schema_version: 2
timeline:
  steps: 1
  duration_s: 0.0
scene:
  source: sionna
  id: floor_wall
raytracing:
  enabled: true
"""
    scenario_yaml = scenario_root / "scenario.yaml"
    scenario_yaml.write_text(source_text, encoding="utf-8")
    return scenario_yaml, source_text


@pytest.fixture
def helper(tmp_path: Path) -> Path:
    path = tmp_path / "fake_generator.py"
    path.write_text(_HELPER_SOURCE, encoding="utf-8")
    return path


def _command(helper: Path, mode: str) -> tuple[str, ...]:
    return sys.executable, "-u", str(helper), mode


def test_snapshot_is_exact_and_builder_owns_no_output_transaction(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    source_bytes = scenario_yaml.read_bytes()
    job = GenerationJob(scenario_yaml, launched_revision=0, job_id="snapshot")

    job._write_snapshot()
    try:
        assert job.paths.snapshot_yaml.read_bytes() == source_bytes
        assert {field.name for field in fields(GenerationPaths)} == {
            "scenario_yaml",
            "snapshot_yaml",
            "final_frames",
        }
        assert job.paths.final_frames == scenario_yaml.parent / "frames"
        assert not list(scenario_yaml.parent.glob(".orchav-ar-*"))
        assert not list(scenario_yaml.parent.glob(".orchav-afb-*"))
        assert not list(scenario_yaml.parent.glob(".orchav-asb-*"))
    finally:
        job._cleanup_working_paths()


def test_snapshot_rejects_a_saved_scenario_change_after_job_creation(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, source_text = scenario
    job = GenerationJob(scenario_yaml, launched_revision=0, job_id="immutable")
    scenario_yaml.write_text(source_text + "debug_level: DEBUG\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="scenario.yaml changed"):
        job._write_snapshot()

    assert not job.paths.snapshot_yaml.exists()


def test_builder_allows_coverage_owned_by_the_generator_child(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    mapping = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8"))
    mapping["raytracing"]["enabled"] = False
    mapping["coverage"] = {"enabled": True}
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    job = GenerationJob(scenario_yaml, launched_revision=0, job_id="coverage")

    assert job.paths.final_frames == scenario_yaml.parent / "frames"


def test_builder_rejects_summary_only_generation_before_launch(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    mapping = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8"))
    mapping["raytracing"]["enabled"] = False
    mapping["generator_summary"] = {
        "enabled": True,
        "create": ["scene2d"],
    }
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a frame-producing"):
        GenerationJob(scenario_yaml, launched_revision=0)


def test_builder_rejects_coverage_without_persisted_data_before_launch(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    mapping = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8"))
    mapping["raytracing"]["enabled"] = False
    mapping["coverage"] = {
        "enabled": True,
        "save": {"data": {"enabled": False}},
    }
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a frame-producing"):
        GenerationJob(scenario_yaml, launched_revision=0)


def test_builder_rejects_reader_selected_generation_destination(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    mapping = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8"))
    mapping["data"] = {"mode": "files", "files": {"directory": "imported_frames"}}
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="publishes only to <scenario>/frames"):
        GenerationJob(scenario_yaml, launched_revision=0)


def test_jsonl_decoder_handles_partial_and_final_records() -> None:
    decoder = JsonlEventDecoder()
    first = '{"schema_version":1,"event":"run_started","first_step":0'
    second = ',"total_steps":1}\n{"schema_version":1,"event":"run_completed"}'

    assert decoder.feed(first) == []
    assert decoder.feed(second) == [
        {
            "schema_version": 1,
            "event": "run_started",
            "first_step": 0,
            "total_steps": 1,
        }
    ]
    assert decoder.finish() == [{"schema_version": 1, "event": "run_completed"}]
    assert decoder.errors == []


def test_success_is_direct_publication_with_changed_identity_and_stale_reporting(
    scenario: tuple[Path, str], helper: Path
) -> None:
    scenario_yaml, source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")
    revisions = [6]
    progress = []

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=5,
            command=_command(helper, "success"),
            revision_provider=lambda: revisions[0],
            on_progress=progress.append,
            job_id="success",
        )
        .start()
        .wait(timeout=10)
    )

    current = load_frame_manifest(scenario_yaml.parent / "frames")
    assert result.state is GenerationState.SUCCEEDED
    assert result.succeeded
    assert current.frame_set_id != previous.frame_set_id
    assert current.generation_id != previous.generation_id
    assert result.stale
    assert result.current_revision == 6
    assert result.progress is not None
    assert progress == [result.progress]
    assert [event["event"] for event in result.events] == [
        "run_started",
        "step_completed",
        "run_completed",
    ]
    assert any("generator diagnostic" in line for line in result.stderr_log)
    assert scenario_yaml.read_text(encoding="utf-8") == source_text
    assert not result.paths.snapshot_yaml.exists()
    assert not list(scenario_yaml.parent.glob(".orchav-ar-*"))


def test_success_without_changed_frame_set_identity_is_rejected(
    scenario: tuple[Path, str], helper: Path
) -> None:
    scenario_yaml, _source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=0,
            command=_command(helper, "unchanged"),
            job_id="unchanged",
        )
        .start()
        .wait(timeout=10)
    )

    assert result.state is GenerationState.FAILED
    assert result.error_message is not None
    assert "without publishing a new frame-set identity" in result.error_message
    assert (
        load_frame_manifest(scenario_yaml.parent / "frames").frame_set_id == previous.frame_set_id
    )


def test_child_failure_preserves_previous_frames(scenario: tuple[Path, str], helper: Path) -> None:
    scenario_yaml, _source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=2,
            command=_command(helper, "failure"),
            job_id="failure",
        )
        .start()
        .wait(timeout=10)
    )

    assert result.state is GenerationState.FAILED
    assert result.returncode == 3
    assert result.error_message == "deliberate failure"
    assert (
        load_frame_manifest(scenario_yaml.parent / "frames").frame_set_id == previous.frame_set_id
    )
    assert not result.paths.snapshot_yaml.exists()


def test_cancellation_preserves_previous_frames(scenario: tuple[Path, str], helper: Path) -> None:
    scenario_yaml, _source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")
    job = GenerationJob(
        scenario_yaml,
        launched_revision=3,
        command=_command(helper, "cancel"),
        cancel_grace_s=0.1,
        job_id="cancel",
    ).start()

    deadline = time.monotonic() + 5
    while not job.stdout_log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.cancel()
    result = job.wait(timeout=10)

    assert result.state is GenerationState.CANCELLED
    assert (
        load_frame_manifest(scenario_yaml.parent / "frames").frame_set_id == previous.frame_set_id
    )
    assert not result.paths.snapshot_yaml.exists()


def test_zero_exit_without_completion_preserves_previous_frames(
    scenario: tuple[Path, str], helper: Path
) -> None:
    scenario_yaml, _source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=1,
            command=_command(helper, "no-completion"),
            job_id="no-completion",
        )
        .start()
        .wait(timeout=10)
    )

    assert result.state is GenerationState.FAILED
    assert result.error_message == "generator exited without a run_completed progress event"
    assert (
        load_frame_manifest(scenario_yaml.parent / "frames").frame_set_id == previous.frame_set_id
    )


@pytest.mark.parametrize("mode", ["wrong-identity", "invalid-identity"])
def test_source_identity_failure_preserves_previous_frames(
    scenario: tuple[Path, str], helper: Path, mode: str
) -> None:
    scenario_yaml, _source_text = scenario
    previous = write_real_frame_set(scenario_yaml.parent / "frames")

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=1,
            command=_command(helper, mode),
            job_id=mode,
        )
        .start()
        .wait(timeout=10)
    )

    assert result.state is GenerationState.FAILED
    assert result.error_message is not None
    assert "source identity" in result.error_message
    assert (
        load_frame_manifest(scenario_yaml.parent / "frames").frame_set_id == previous.frame_set_id
    )


def test_launch_failure_is_terminal_and_cleans_snapshot(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=9,
            command=(str(scenario_yaml.parent / "missing-generator-command"),),
            job_id="launch-failure",
        )
        .start()
        .wait(timeout=1)
    )

    assert result.state is GenerationState.FAILED
    assert result.returncode is None
    assert result.error_message is not None
    assert "could not launch generator" in result.error_message
    assert not result.paths.snapshot_yaml.exists()


def test_preexisting_private_snapshot_is_never_cleaned_as_job_owned(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario
    job = GenerationJob(
        scenario_yaml,
        launched_revision=1,
        command=(str(scenario_yaml.parent / "unused-generator"),),
        job_id="preexisting",
    )
    job.paths.snapshot_yaml.write_text("foreign snapshot", encoding="utf-8")

    result = job.start().wait(timeout=1)

    assert result.state is GenerationState.FAILED
    assert result.error_message is not None
    assert "snapshot path already exists" in result.error_message
    assert job.paths.snapshot_yaml.read_text(encoding="utf-8") == "foreign snapshot"


def test_generation_requires_canonical_scenario_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario.yaml"):
        GenerationJob(tmp_path / "draft.yaml", launched_revision=0)


def test_default_command_uses_current_interpreter_and_loaded_source(
    scenario: tuple[Path, str],
) -> None:
    scenario_yaml, _source_text = scenario

    job = GenerationJob(scenario_yaml, launched_revision=0, job_id="source-bound")

    assert Path(job.command[0]).samefile(sys.executable)
    assert job.command[1:3] == ("-I", "-u")
    assert "orchav-generator" not in job.command
    assert str(job.source_identity.source_root) in job.command
    assert "generator" in job.command
