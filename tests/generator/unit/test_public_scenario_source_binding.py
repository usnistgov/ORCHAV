"""Process-boundary checks for public Python scenario drivers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DIRECT_DRIVERS = (
    PROJECT_ROOT / "scenarios/getting_started/hello_world_scripted/generate.py",
    PROJECT_ROOT / "scenarios/visualizer/multi_device_trajectory/generate.py",
    PROJECT_ROOT / "scenarios/visualizer/synthetic_mpc_benchmark/generate.py",
)

_SOURCE_PROBE = """
import pathlib
import runpy
import sys

namespace = runpy.run_path(sys.argv[1], run_name="orchav_source_binding_probe")
project_root = pathlib.Path(namespace["PROJECT_ROOT"]).resolve()

import generator
import shared

assert pathlib.Path(sys.path[0]).resolve() == project_root
assert pathlib.Path(generator.__file__).resolve().is_relative_to(project_root)
assert pathlib.Path(shared.__file__).resolve().is_relative_to(project_root)
print("ORCHAV_SOURCE_BINDING_OK")
"""


@pytest.mark.parametrize("driver", PUBLIC_DIRECT_DRIVERS, ids=lambda path: path.parent.name)
def test_public_driver_prefers_its_checkout_over_conflicting_pythonpath(
    driver: Path,
    tmp_path: Path,
) -> None:
    """A stale package before this checkout must not win driver imports."""
    decoy_root = tmp_path / "decoy"
    for package in ("generator", "shared"):
        package_dir = decoy_root / package
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text(
            f"raise RuntimeError('decoy {package} package was imported')\n",
            encoding="utf-8",
        )

    environment = os.environ.copy()
    # Keep PROJECT_ROOT later in sys.path to catch a conditional "already
    # present" check that fails to move it ahead of the decoy.
    environment["PYTHONPATH"] = os.pathsep.join((str(decoy_root), str(PROJECT_ROOT)))
    environment["ORCHAV_HEADLESS"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", _SOURCE_PROBE, str(driver)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORCHAV_SOURCE_BINDING_OK" in result.stdout
