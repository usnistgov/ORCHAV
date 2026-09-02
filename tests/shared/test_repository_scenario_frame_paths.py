"""Repository census for the split frame-directory/pattern contract."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from shared.scenarios.frame_paths import (
    validate_frames_directory,
    validate_frames_pattern,
)

try:
    from tests.shared._scenario_frame_path_waivers import KNOWN_UNMIGRATED_PATTERNS
except ModuleNotFoundError as exc:
    if exc.name != "tests.shared._scenario_frame_path_waivers":
        raise
    # The curated public export excludes private-repository exceptions. Every
    # retained public scenario must satisfy the current contract without one.
    KNOWN_UNMIGRATED_PATTERNS = {}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tracked_yaml_paths() -> list[Path]:
    if not (PROJECT_ROOT / ".git").exists():
        pytest.skip("tracked-file scenario census requires a Git checkout")
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.yaml", "*.yml"],
        cwd=PROJECT_ROOT,
    )
    return [PROJECT_ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def _files_section(document: Any) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    data = document.get("data")
    if not isinstance(data, dict):
        return None
    files = data.get("files")
    return files if isinstance(files, dict) else None


def _known_unmigrated_patterns_for(tracked_paths: list[Path]) -> dict[str, str]:
    """Return private exceptions that are actually tracked in this checkout.

    Public release exports omit ``scenarios/_internal`` while retaining this
    census test.  Scoping the exception to the tracked input set keeps the
    public census strict without expecting a private-only file to exist.
    """

    tracked_relative_paths = {path.relative_to(PROJECT_ROOT).as_posix() for path in tracked_paths}
    return {
        relative_path: pattern
        for relative_path, pattern in KNOWN_UNMIGRATED_PATTERNS.items()
        if relative_path in tracked_relative_paths
    }


def test_private_exception_is_scoped_to_the_tracked_input_set() -> None:
    if not KNOWN_UNMIGRATED_PATTERNS:
        assert _known_unmigrated_patterns_for([]) == {}
        return

    relative_path, pattern = next(iter(KNOWN_UNMIGRATED_PATTERNS.items()))

    assert _known_unmigrated_patterns_for([]) == {}
    assert _known_unmigrated_patterns_for([PROJECT_ROOT / relative_path]) == {
        relative_path: pattern
    }


def test_tracked_scenario_frame_paths_follow_split_contract() -> None:
    tracked_paths = _tracked_yaml_paths()
    path_bearing_patterns: dict[str, str] = {}

    for path in tracked_paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        files = _files_section(document)
        if files is None:
            continue

        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        pattern = files.get("pattern")
        if isinstance(pattern, str) and any(separator in pattern for separator in "/\\"):
            path_bearing_patterns[relative_path] = pattern
        elif pattern is not None:
            validate_frames_pattern(pattern)

        directory = files.get("directory")
        if directory is not None:
            validate_frames_directory(directory)

    assert path_bearing_patterns == _known_unmigrated_patterns_for(tracked_paths)
