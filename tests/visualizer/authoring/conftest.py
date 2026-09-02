"""Shared fixtures for Scenario Builder authoring tests."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def authoring_project_root(tmp_path: Path) -> Path:
    """Create the minimal ORCHAV library surface used by Builder save tests."""

    root = tmp_path / "orchav-project"
    scene = root / "libraries" / "scenes" / "empty" / "empty.xml"
    target = root / "libraries" / "targets" / "cube" / "cube.ply"
    scene.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    scene.write_bytes((PROJECT_ROOT / "libraries" / "scenes" / "empty" / "empty.xml").read_bytes())
    target.write_bytes((PROJECT_ROOT / "libraries" / "targets" / "cube" / "cube.ply").read_bytes())
    (root / "README.md").write_text("ORCHAV test project\n", encoding="utf-8")
    return root
