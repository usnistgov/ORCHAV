"""Installed-command coverage for already-calibrated comparison workflows."""

from __future__ import annotations

import json
from pathlib import Path

from shared.cli.sensing_compare import main


def test_demo_command_publishes_machine_readable_summary(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "demo"

    assert main(["demo", "--output-dir", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert Path(summary["output_dir"]) == output.resolve()
    assert Path(summary["comparison_json"]).is_file()
    assert Path(summary["residual_csv"]).is_file()
    assert Path(summary["synchronized_png"]).is_file()
    assert len(summary["inventory_sha256"]) == 64


def test_compare_command_round_trips_demo_inputs_and_recipe(
    tmp_path: Path,
    capsys,
) -> None:
    demo = tmp_path / "demo"
    comparison = tmp_path / "comparison"
    assert main(["demo", "--output-dir", str(demo)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "compare",
                "--measurement-manifest",
                str(demo / "inputs" / "measurement" / "manifest.json"),
                "--simulation-manifest",
                str(demo / "inputs" / "simulation" / "manifest.json"),
                "--alignment-recipe",
                str(demo / "comparison" / "alignment_recipe.json"),
                "--output-dir",
                str(comparison),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    expected = json.loads((demo / "comparison" / "comparison.json").read_text())
    actual = json.loads(Path(summary["comparison_json"]).read_text())
    assert actual["comparison_id"] == expected["comparison_id"]
    assert actual["alignment_id"] == expected["alignment_id"]
    assert actual["alignment_sha256"] == expected["alignment_sha256"]
    assert not (comparison / "inputs").exists()


def test_command_failure_returns_two_without_overwriting_output(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "owned"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("caller-owned", encoding="utf-8")

    assert main(["demo", "--output-dir", str(output)]) == 2

    assert "overwrite" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "caller-owned"
