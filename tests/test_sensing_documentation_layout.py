"""Sensing guide layout and release-boundary checks."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sensing_guides_have_one_current_entry_per_topic():
    folder = ROOT / "docs/sensing"
    expected = {
        "README.md",
        "concepts.md",
        "detection_and_tracking.md",
        "evaluation.md",
        "reprocessing.md",
        "parameters.md",
        "target_models.md",
        "custom_tracker.md",
        "timings.md",
    }
    assert {path.name for path in folder.glob("*.md")} == expected
    # These source-only studies are excluded by the curated release selector.
    private = {
        "detection-tracking/tracker_lab.md",
        "targets-motion/rotor_feasibility.md",
        "targets-motion/rotor_short_acquisition.md",
        "targets-motion/human_motion_example.md",
        "targets-motion/uav_evaluation.md",
    }
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*.md")}
    assert actual - private == expected
    for relative in expected:
        text = (folder / relative).read_text(encoding="utf-8")
        assert "This guide has moved" not in text
        assert "docs/_internal" not in text
        assert "path_map.md" not in text


def test_diagrams_are_accessible_and_linked_from_their_guides():
    folder = ROOT / "docs/assets/sensing"
    ns = {"svg": "http://www.w3.org/2000/svg"}
    concepts = (ROOT / "docs/sensing/concepts.md").read_text(encoding="utf-8")
    for image in ("sionna_relationship.svg", "acquisition_timing.svg"):
        svg = ET.parse(folder / image).getroot()
        assert svg.find("svg:title", ns).text
        assert svg.find("svg:desc", ns).text
        assert svg.attrib["aria-labelledby"] == "title desc"
        assert f"](../assets/sensing/{image})" in concepts
    assert "accepted at IEEE ISAC 2026" in " ".join(concepts.split())
    assert "RCSSolver" in concepts and "cfr_ifft" in concepts
    assert "/commit/" not in concepts


def test_concepts_include_the_range_model_comparison():
    concepts = (ROOT / "docs/sensing/concepts.md").read_text(encoding="utf-8")
    assert "Monostatic" in concepts and "Bistatic" in concepts
    assert "discrete range samples" in concepts
    rows = [line for line in concepts.splitlines() if line.startswith("|")]
    for terms in (
        ("Delay-bin accumulation", "point_bin_floor", "sensing.processing.range.fft_size"),
        ("Frequency-domain channel synthesis", "cfr_ifft", "frequency_samples", "frequency_window"),
    ):
        assert any(all(term in row for term in terms) for row in rows), terms


def test_motion_conversion_help_does_not_load_optional_model_packages():
    code = """
import importlib.abc
import runpy
import sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'smplx', 'trimesh', 'matplotlib', 'PIL'}:
            raise AssertionError('Optional dependency imported during help: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
from generator.tools import smpl_to_ply, render_mesh_previews
for module in (smpl_to_ply, render_mesh_previews):
    try:
        module.main(['--help'])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError('Expected argparse help exit')
sys.argv = ['prepare.py', '--help']
try:
    runpy.run_path('scenarios/sensing/targets_motion/marching_person/prepare.py', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('Expected preparation helper help exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
