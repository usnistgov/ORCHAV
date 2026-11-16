"""Installed-command coverage for animated-target evidence workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.animated_target_workflow import (
    default_synthetic_binding_context,
    default_synthetic_binding_times,
)
from shared.cli.animated_target import main


def test_demo_command_publishes_machine_readable_summary(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "demo"

    assert main(["demo", "--output-dir", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert Path(summary["output_dir"]) == output.resolve()
    assert summary["asset_id"] == "orchav:synthetic:asymmetric_motion_workflow_v1"
    assert (
        summary["binding_id"]
        == json.loads((output / "binding.json").read_text(encoding="utf-8"))["binding_id"]
    )
    assert (output / "alignment_and_geometry.png").is_file()


def test_bind_command_round_trips_the_demo_request_without_copying_asset(
    tmp_path: Path,
    capsys,
) -> None:
    demo = tmp_path / "demo"
    rebound = tmp_path / "rebound"
    assert main(["demo", "--output-dir", str(demo)]) == 0
    demo_summary = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "bind",
                "--manifest",
                str(demo / "asset" / "animated_target.json"),
                "--request",
                str(demo / "binding_request.json"),
                "--output-dir",
                str(rebound),
            ]
        )
        == 0
    )

    rebound_summary = json.loads(capsys.readouterr().out)
    assert rebound_summary["binding_id"] == demo_summary["binding_id"]
    assert not (rebound / "asset").exists()
    assert json.loads((demo / "binding_request.json").read_text())["contract_version"] == 1


def test_command_failure_returns_two_without_overwriting_output(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "owned"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("caller-owned", encoding="utf-8")

    assert main(["demo", "--output-dir", str(output)]) == 2

    assert "already exists" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "caller-owned"


@pytest.mark.parametrize("contract_version", [True, 1.0])
def test_bind_command_rejects_non_integer_contract_version(
    tmp_path: Path,
    capsys,
    contract_version: object,
) -> None:
    request = tmp_path / "binding_request.json"
    request.write_text(
        json.dumps(
            {
                "contract": "orchav.animated_target_binding_request",
                "contract_version": contract_version,
                "context": default_synthetic_binding_context().to_dict(),
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "scenario_time_s": sample.scenario_time_s,
                        "motion_time_s": sample.motion_time_s,
                        "pulse_time_s": sample.pulse_time_s,
                    }
                    for sample in default_synthetic_binding_times()
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "must_not_publish"

    assert (
        main(
            [
                "bind",
                "--manifest",
                str(tmp_path / "unused_manifest.json"),
                "--request",
                str(request),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )

    assert "contract_version must be the integer 1" in capsys.readouterr().err
    assert not output.exists()
