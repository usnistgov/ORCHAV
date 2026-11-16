"""Public tutorial commands and inspector excerpts."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

from generator.core.configuration.loader import resolve_sensing_document
from shared.cli.inspect import _format_sensing_lines

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "scenarios/sensing"
CORE = tuple(
    SCENARIOS / "cube_radar" / name / "README.md"
    for name in (
        "stationary",
        "stationary_filtering",
        "moving",
        "oscillating_comparison",
        "moving_tracking",
        "azimuth",
        "moving_tracking_3d",
        "two_moving",
    )
)
INSPECTOR_PAGES = (
    "detection_tracking/detection_threshold_sweep",
    "detection_tracking/two_target_resolution",
    "detection_tracking/multipath_clutter_box",
    "targets_motion/micro_motion_coherent_cube",
    "targets_motion/micro_motion_stft_cube",
)


def test_tutorial_inspector_excerpts_use_typed_contract():
    sample = {
        "processing_mode": "coherent_cpi",
        "frame_role": "cpi_end",
        "products": [],
        "targets": {},
    }
    assert _format_sensing_lines(sample)[0] == "Frame extension: sensing"
    assert "  Products: none" in _format_sensing_lines(sample)
    assert "  Products: 0 typed descriptor(s)" in _format_sensing_lines(sample, verbose=True)
    for name in INSPECTOR_PAGES:
        page = SCENARIOS / name / "README.md"
        if not page.is_file():
            continue  # Historical inspector pages are private in the curated export.
        text = page.read_text()
        # Routine inspection steps were deliberately removed from these
        # tutorials; any remaining excerpts must still use the current format.
        assert "Frame extension: sensing\n  Mode:" not in text
        assert "typed descriptor(s)" not in text
        assert "\nSensing:\n" not in text


def test_core_tutorial_navigation_follows_learning_path():
    for index, page in enumerate(CORE):
        navigation = re.findall(
            r"^## (?:Explore Next|Browse Scenarios)\n(.*?)(?=^## |\Z)",
            page.read_text(),
            re.M | re.S,
        )
        assert navigation, f"Missing tutorial navigation: {page}"
        if index + 1 < len(CORE):
            assert os.path.relpath(CORE[index + 1], page.parent) in "\n".join(navigation)


def test_array_tutorial_commands_use_supported_native_detector():
    text = (SCENARIOS / "cube_radar/azimuth/README.md").read_text()
    commands = "\n".join(re.findall(r"```bash\n(.*?)```", text, re.S))
    # Angular replay belongs to the reference; the lesson uses generation.
    reference = (ROOT / "docs/sensing/reprocessing.md").read_text()
    assert "--array-detection-mode beam_scan" in reference
    assert "--angle-peak-detection" not in commands
    config = resolve_sensing_document(
        yaml.safe_load((SCENARIOS / "cube_radar/azimuth/scenario.yaml").read_text())
    )
    assert config["sensing"]["array_detection_mode"] == "beam_scan"
    assert config["sensing"]["recording_profile"] == "maps"
    assert "raw_payload_policy" not in config["sensing"]


def test_documented_dashboard_labels_exist_in_current_ui():
    code = (ROOT / "visualizer/src/sensing/sensing_window.py").read_text()
    shell = (ROOT / "visualizer/src/sensing/workspace_shell.py").read_text()
    summary = (ROOT / "visualizer/src/sensing/current_result_summary.py").read_text()
    guide = " ".join((ROOT / "docs/visualizer/sensing_dashboard.md").read_text().split())
    for label in (
        "Run summary",
        "Current result",
        "Motion",
        "Tracking",
        "Detector",
        "Inputs",
    ):
        assert label in shell and label in guide
    for label in ("Detection", "Filtering", "Tracking", "Evaluation"):
        assert label in summary and label in guide
    for label in ("Before clutter suppression", "After clutter suppression"):
        assert label in code and label in guide
    assert "**Persisted stage**" not in guide


def test_all_public_sensing_postprocessor_commands_parse(monkeypatch):
    """Parse the exact authored flags, stopping before input I/O or writes."""
    import argparse
    import shlex

    from shared.sensing.postprocessor import main

    class Parsed(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args
    commands = []
    pages = [
        *SCENARIOS.rglob("*.md"),
        *(ROOT / "docs/sensing").rglob("*.md"),
        ROOT / "scenarios/visualizer/sensing_gui/README.md",
    ]
    for page in pages:
        if not page.is_file():
            continue
        for block in re.findall(r"```(?:bash|shell)\n(.*?)```", page.read_text(), re.S):
            for command in block.replace("\\\n", " ").splitlines():
                if command.startswith("python -m shared.sensing.postprocessor "):
                    commands.append(shlex.split(command)[3:])
                elif command.startswith("orchav-sensing "):
                    commands.append(shlex.split(command)[1:])
    assert commands, "Public saved-acquisition instructions must remain parseable"
    assert any("--recipes" in command for command in commands)
    assert any("--frequency-window" in command for command in commands)
    assert any("--coherent-clutter-mode" in command for command in commands)
    for command in commands:

        def stop_after_parse(parser, *args, **kwargs):
            parse_args(parser, command)
            raise Parsed

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parse)
        with pytest.raises(Parsed):
            main()
