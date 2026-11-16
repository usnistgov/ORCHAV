"""Unit tests for the committed-tree scenario archive gate."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.ci.check_git_archive import (
    COVERAGE_REQUIRED_PATHS,
    missing_scenario_archive_paths,
    release_payload_issues,
    required_scenario_archive_paths,
)


def _release_archive(*, mutation: str = "none") -> bytes:
    asset = "scenarios/example/scene/mesh/building.ply"
    content = b"retained authored mesh\n"
    provenance = {
        "payload": {
            "files": [
                {
                    "path": asset,
                    "kind": "file",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ]
        }
    }
    files = {"RELEASE_PROVENANCE.json": json.dumps(provenance).encode(), asset: content}
    if mutation == "missing":
        del files[asset]
    elif mutation == "changed":
        files[asset] = b"different geometry\n"
    elif mutation == "extra":
        files["scenarios/example/frames/unreviewed.h5"] = b"unreviewed data"
    elif mutation == "no_provenance":
        del files["RELEASE_PROVENANCE.json"]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for path, data in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def test_archive_checks_every_declared_release_asset() -> None:
    assert release_payload_issues(_release_archive()) == []


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing", "missing release payload"),
        ("changed", "release payload content mismatch"),
        ("extra", "undeclared release payload"),
        ("no_provenance", "release provenance is absent"),
    ],
)
def test_archive_rejects_incomplete_or_altered_release_payload(mutation: str, reason: str) -> None:
    issues = release_payload_issues(_release_archive(mutation=mutation), require_provenance=True)
    assert any(reason in issue for issue in issues)


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
