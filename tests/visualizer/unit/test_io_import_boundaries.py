import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_importing_scenario_config_does_not_load_frame_sources():
    code = (
        "import importlib, sys; "
        "module = importlib.import_module('visualizer.src.io.scenario_config'); "
        "assert module.AppConfig is not None; "
        "assert 'visualizer.src.io.frame_sources' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_importing_frame_sources_does_not_load_scene_io():
    code = (
        "import importlib, sys; "
        "module = importlib.import_module('visualizer.src.io.frame_sources'); "
        "assert module.FileSource is not None; "
        "assert 'visualizer.src.scene.io' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
