"""Tests for the generator's machine-readable progress protocol."""

import io
import json

from generator.core.pipeline.progress import JsonlProgressReporter, ProgressInfo

_SOURCE_IDENTITY = {
    "schema_version": 1,
    "source_root": "/source/orchav",
    "version": "0.1.0",
    "git_sha": "abc123",
}


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_jsonl_reporter_emits_versioned_lifecycle_records() -> None:
    stream = io.StringIO()
    reporter = JsonlProgressReporter(
        first_step=2,
        total_steps=2,
        source_identity=_SOURCE_IDENTITY,
        stream=stream,
    )

    reporter.run_started()
    reporter.step_completed(
        ProgressInfo(step=2, total_steps=2, elapsed_s=1.23456789, step_duration_s=0.25)
    )
    reporter.step_completed(
        ProgressInfo(step=3, total_steps=2, elapsed_s=2.0, step_duration_s=0.75)
    )
    reporter.run_completed("HDF5 Output")

    assert _records(stream) == [
        {
            "schema_version": 1,
            "event": "run_started",
            "first_step": 2,
            "total_steps": 2,
            "frame_set_mode": "fresh_partial",
            "source_identity": _SOURCE_IDENTITY,
        },
        {
            "schema_version": 1,
            "event": "step_completed",
            "step": 2,
            "completed_steps": 1,
            "total_steps": 2,
            "elapsed_s": 1.234568,
            "step_duration_s": 0.25,
        },
        {
            "schema_version": 1,
            "event": "step_completed",
            "step": 3,
            "completed_steps": 2,
            "total_steps": 2,
            "elapsed_s": 2.0,
            "step_duration_s": 0.75,
        },
        {
            "schema_version": 1,
            "event": "run_completed",
            "completed_steps": 2,
            "total_steps": 2,
            "output": "HDF5 Output",
        },
    ]


def test_jsonl_reporter_emits_failure_details() -> None:
    stream = io.StringIO()
    reporter = JsonlProgressReporter(
        first_step=0,
        total_steps=4,
        source_identity=_SOURCE_IDENTITY,
        stream=stream,
    )

    reporter.run_failed("scene failed", error_type="RuntimeError")

    assert _records(stream) == [
        {
            "schema_version": 1,
            "event": "run_failed",
            "completed_steps": 0,
            "total_steps": 4,
            "message": "scene failed",
            "error_type": "RuntimeError",
        }
    ]
