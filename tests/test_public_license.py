"""Keep project licensing explicit while preserving separately licensed inputs."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_and_license_text_agree_on_gpl3_only():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "GPL-3.0-only"
    assert 'license: "GPL-3.0-only"' in (ROOT / "CITATION.cff").read_text()
    assert "version 3 only (GPL-3.0-only)" in (ROOT / "NOTICE").read_text()
    text = (ROOT / "LICENSE").read_text()
    assert "Version 3, 29 June 2007" in text
    assert "END OF TERMS AND CONDITIONS" in text
    assert "Version 2, June 1991" not in text


def test_procedural_vehicle_is_first_party_gpl3():
    asset = ROOT / "libraries/targets/procedural_car"
    metadata = json.loads((asset / "target_metadata.json").read_text())
    assert metadata["provenance"]["license"] == "GPL-3.0-only"
    assert metadata["provenance"]["third_party_mesh_inputs"] == []
