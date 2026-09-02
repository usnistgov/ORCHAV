"""Unit tests for the committed-tree scenario archive gate."""

import subprocess
import sys
from pathlib import Path

from scripts.ci.check_git_archive import (
    COVERAGE_REQUIRED_PATHS,
    missing_scenario_archive_paths,
    required_scenario_archive_paths,
)


def test_archive_requirements_include_all_coverage_inputs() -> None:
    required = required_scenario_archive_paths()

    assert COVERAGE_REQUIRED_PATHS.issubset(required)


def test_archive_inventory_reports_missing_scenario_files() -> None:
    required = required_scenario_archive_paths()
    omitted = "scenarios/generator/coverage/single_tx/README.md"

    assert missing_scenario_archive_paths(required - {omitted}) == [omitted]
    assert missing_scenario_archive_paths(required) == []


def test_archive_gate_runs_without_private_release_audit(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "scripts/ci/check_git_archive.py"
    destination = tmp_path / "scripts/ci/check_git_archive.py"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())

    result = subprocess.run(
        [sys.executable, str(destination), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
