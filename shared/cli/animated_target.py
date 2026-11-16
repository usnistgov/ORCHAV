"""Installed CLI for source-neutral animated-target binding evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.animated_target import (
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    AnimatedTargetError,
)
from shared.animated_target_workflow import (
    AnimatedTargetWorkflowError,
    publish_animated_target_workflow,
    publish_synthetic_animated_target_workflow,
)

_REQUEST_CONTRACT = "orchav.animated_target_binding_request"
_REQUEST_VERSION = 1


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _exact_mapping(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise ValueError(f"{path} fields differ: missing={missing}, unexpected={unexpected}")
    return value


def _load_binding_request(
    path: str | Path,
) -> tuple[AnimatedTargetBindingContext, tuple[AnimatedTargetBindingTime, ...]]:
    request_path = Path(path)
    try:
        raw = json.loads(
            request_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot load binding request {request_path}: {exc}") from exc
    request = _exact_mapping(
        raw,
        {"contract", "contract_version", "context", "samples"},
        "binding request",
    )
    if request["contract"] != _REQUEST_CONTRACT:
        raise ValueError(f"binding request contract must be {_REQUEST_CONTRACT!r}")
    contract_version = request["contract_version"]
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version != _REQUEST_VERSION
    ):
        raise ValueError(f"binding request contract_version must be the integer {_REQUEST_VERSION}")

    context_raw = _exact_mapping(
        request["context"],
        {
            "target_id",
            "scenario_clock_id",
            "motion_clock_id",
            "pulse_clock_id",
            "playback_rate",
            "playback_offset_s",
            "retiming_policy",
        },
        "binding request context",
    )
    context = AnimatedTargetBindingContext(
        target_id=context_raw["target_id"],
        scenario_clock_id=context_raw["scenario_clock_id"],
        motion_clock_id=context_raw["motion_clock_id"],
        pulse_clock_id=context_raw["pulse_clock_id"],
        playback_rate=context_raw["playback_rate"],
        playback_offset_s=context_raw["playback_offset_s"],
        retiming_policy=context_raw["retiming_policy"],
    )

    samples_raw = request["samples"]
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError("binding request samples must be a non-empty JSON list")
    samples: list[AnimatedTargetBindingTime] = []
    for index, raw_sample in enumerate(samples_raw):
        sample = _exact_mapping(
            raw_sample,
            {"sample_id", "scenario_time_s", "motion_time_s", "pulse_time_s"},
            f"binding request samples[{index}]",
        )
        samples.append(
            AnimatedTargetBindingTime(
                sample_id=sample["sample_id"],
                scenario_time_s=sample["scenario_time_s"],
                motion_time_s=sample["motion_time_s"],
                pulse_time_s=sample["pulse_time_s"],
            )
        )
    return context, tuple(samples)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchav-animated-target",
        description=(
            "Validate and bind source-neutral animated-target assets, then publish "
            "JSON/CSV/PNG timing and geometry evidence without ray tracing."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="publish the ORCHAV-authored asymmetric synthetic reference",
    )
    demo.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish atomically; it must not already exist",
    )

    bind = subparsers.add_parser(
        "bind",
        help="bind an existing V1 manifest using an explicit clock request",
    )
    bind.add_argument("--manifest", type=Path, required=True)
    bind.add_argument("--request", type=Path, required=True)
    bind.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish atomically; it must not already exist",
    )
    return parser


def _summary(result: Any) -> str:
    return json.dumps(
        {
            "output_dir": str(result.output_dir),
            "asset_id": result.asset_id,
            "asset_revision_sha256": result.asset_revision_sha256,
            "binding_id": result.binding_id,
            "workflow_id": result.workflow_id,
            "artifact_manifest_sha256": result.artifact_manifest_sha256,
        },
        allow_nan=False,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed animated-target evidence workflow."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            result = publish_synthetic_animated_target_workflow(args.output_dir)
        else:
            context, samples = _load_binding_request(args.request)
            result = publish_animated_target_workflow(
                args.manifest,
                context,
                samples,
                args.output_dir,
            )
    except (AnimatedTargetError, AnimatedTargetWorkflowError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(_summary(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - installed entry point owns dispatch
    raise SystemExit(main())
