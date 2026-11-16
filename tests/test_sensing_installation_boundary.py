"""Characterize the supported sensing installation boundary.

These checks build the project without dependency resolution, then inspect the
wheel as an installation artifact.  They intentionally avoid importing Sionna,
opening a display, or selecting a GPU.
"""

from __future__ import annotations

import ast
import configparser
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Iterator
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CONSOLE_SCRIPTS = {
    "orchav-smpl-import": "generator.tools.smpl_to_animated_target:main",
    "orchav-kimodo-import": "generator.tools.kimodo_to_animated_target:main",
    "orchav-animated-target": "shared.cli.animated_target:main",
    "orchav-generator": "generator.__main__:main",
    "orchav-inspect": "shared.cli.inspect:main",
    "orchav-sensing": "shared.sensing.postprocessor:main",
    "orchav-sensing-compare": "shared.cli.sensing_compare:main",
    "orchav-sensing-replay": "visualizer.src.sensing.replay_cli:main",
    "orchav-validate": "shared.cli.validate:main",
    "orchav-visualizer": "visualizer.__main__:main",
}
EXPECTED_SENSING_REQUIRES_DIST = {
    "Pillow>=13.0.0",
    "matplotlib>=3.8",
    "scikit-learn>=1.4",
    "scipy>=1.9",
}


@pytest.fixture(scope="module")
def sensing_wheel(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build one network-free wheel from a disposable copy of the worktree."""

    build_root = tmp_path_factory.mktemp("sensing-wheel-source")
    source_root = build_root / "source"
    source_root.mkdir()

    for relative_path in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "NOTICE",
        "NIST_SOFTWARE_DISCLAIMER.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        source = PROJECT_ROOT / relative_path
        if source.is_file():
            shutil.copy2(source, source_root / relative_path)

    ignored = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        "build",
        "dist",
    )
    for package_name in ("generator", "shared", "tools", "visualizer"):
        source = PROJECT_ROOT / package_name
        if source.is_dir():
            shutil.copytree(source, source_root / package_name, ignore=ignored)

    wheel_dir = build_root / "wheelhouse"
    wheel_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            ".",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = sorted(wheel_dir.glob("orchav-*.whl"))
    assert len(wheels) == 1, wheels
    yield wheels[0]


def _wheel_metadata(archive: zipfile.ZipFile):
    metadata_members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata_members) == 1, metadata_members
    return BytesParser(policy=policy.default).parsebytes(archive.read(metadata_members[0]))


def _console_scripts(archive: zipfile.ZipFile) -> dict[str, str]:
    entry_point_members = [
        name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
    ]
    assert len(entry_point_members) == 1, entry_point_members
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(archive.read(entry_point_members[0]).decode("utf-8"))
    return dict(parser["console_scripts"])


def test_wheel_contains_complete_generator_sensing_implementation(sensing_wheel: Path) -> None:
    """Every processor and strategy source module must survive wheel discovery."""

    sensing_root = PROJECT_ROOT / "generator" / "core" / "sensing"
    expected_members = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in sensing_root.rglob("*.py")
    }
    assert "generator/core/sensing/processor.py" in expected_members
    assert any(
        member.startswith("generator/core/sensing/strategies/") for member in expected_members
    )

    with zipfile.ZipFile(sensing_wheel) as archive:
        wheel_members = set(archive.namelist())

    missing = sorted(expected_members - wheel_members)
    assert not missing, f"sensing implementation missing from wheel: {missing}"


def test_wheel_material_extension_noops_without_excluded_calibration(
    sensing_wheel: Path,
    tmp_path: Path,
) -> None:
    """The wheel must omit calibration; a retained optional hook must remain inert."""

    extension_member = "generator/core/materials/_private_scattering_pattern_extensions.py"
    with zipfile.ZipFile(sensing_wheel) as archive:
        wheel_members = set(archive.namelist())
    assert not any(member.startswith("generator/core/calibration/") for member in wheel_members)
    if extension_member not in wheel_members:
        # Curated public exports remove the private adapter and its package.
        return

    script = f"""
import sys

sys.path.insert(0, {str(sensing_wheel)!r})
from generator.core.materials import scattering_patterns

scattering_patterns._register_private_scattering_pattern_extensions()
assert scattering_patterns._PRIVATE_EXTENSIONS_LOADED is True
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_requires_sensing_dependencies_by_default(sensing_wheel: Path) -> None:
    """A standard install supplies sensing; its empty extra adds no requirements."""

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert EXPECTED_SENSING_REQUIRES_DIST <= set(project["project"]["dependencies"])
    assert project["project"]["optional-dependencies"]["sensing"] == []

    with zipfile.ZipFile(sensing_wheel) as archive:
        metadata = _wheel_metadata(archive)
        provided_extras = set(metadata.get_all("Provides-Extra", []))
        requires_dist = set(metadata.get_all("Requires-Dist", []))
    assert "sensing" in provided_extras
    assert EXPECTED_SENSING_REQUIRES_DIST <= requires_dist
    sensing_requires_dist = {
        requirement for requirement in requires_dist if requirement.endswith('; extra == "sensing"')
    }
    assert not sensing_requires_dist


def test_wheel_exposes_exact_supported_console_scripts(sensing_wheel: Path) -> None:
    """The installed command surface is an exact, reviewable compatibility contract."""

    with zipfile.ZipFile(sensing_wheel) as archive:
        assert _console_scripts(archive) == EXPECTED_CONSOLE_SCRIPTS


@pytest.mark.parametrize(("script_name", "target"), EXPECTED_CONSOLE_SCRIPTS.items())
def test_wheel_console_script_targets_concrete_main_function(
    sensing_wheel: Path,
    script_name: str,
    target: str,
) -> None:
    """Installed commands must resolve to concrete ``main`` functions in the wheel."""

    module_name, function_name = target.split(":", maxsplit=1)
    module_member = module_name.replace(".", "/") + ".py"

    with zipfile.ZipFile(sensing_wheel) as archive:
        scripts = _console_scripts(archive)
        assert scripts[script_name] == target
        assert module_member in archive.namelist()
        module_tree = ast.parse(archive.read(module_member), filename=module_member)

    top_level_bindings = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    top_level_bindings.update(
        alias.asname or alias.name
        for node in module_tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    assert function_name in top_level_bindings


def test_wheel_contains_source_neutral_phase8_contracts_and_workflows(
    sensing_wheel: Path,
) -> None:
    """Motion and measurement workflows must be installed without private adapters."""

    expected_members = {
        "shared/sensing/__init__.py",
        "shared/sensing/detection/__init__.py",
        "shared/sensing/fusion/__init__.py",
        "shared/sensing/plotting/__init__.py",
        "shared/sensing/volume/__init__.py",
        "shared/sensing/postprocessor.py",
        "shared/sensing_postprocessor.py",
        "shared/animated_target_export.py",
        "generator/tools/smpl_to_animated_target.py",
        "generator/tools/kimodo_to_animated_target.py",
        "shared/animated_target.py",
        "shared/animated_target_workflow.py",
        "shared/cli/animated_target.py",
        "shared/cli/sensing_compare.py",
        "shared/sensing/measurement.py",
        "shared/sensing/measurement_workflow.py",
    }
    with zipfile.ZipFile(sensing_wheel) as archive:
        members = set(archive.namelist())
    assert expected_members <= members


def test_wheel_phase8_import_boundary_avoids_restricted_source_runtimes(
    sensing_wheel: Path,
    tmp_path: Path,
) -> None:
    """Importing contract and CLI surfaces must not activate source/calibration stacks."""

    script = f"""
import builtins
import sys

forbidden = (
    'chumpy',
    'generator.core.calibration',
    'kimodo',
    'smplx',
    'tensorflow',
    'torch',
)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in forbidden):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, {str(sensing_wheel)!r})

import generator.tools.smpl_to_animated_target
import generator.tools.kimodo_to_animated_target
import shared.animated_target_export
import shared.animated_target
import shared.animated_target_workflow
import shared.cli.animated_target
import shared.cli.sensing_compare
import shared.sensing.measurement
import shared.sensing.measurement_workflow

assert not any(
    any(name == item or name.startswith(item + '.') for item in forbidden)
    for name in sys.modules
)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_wheel_registers_sensing_without_development_bootstrap(sensing_wheel, tmp_path):
    """File presence alone is insufficient: an installed GUI must activate its service."""
    script = f"""
import sys
sys.path.insert(0, {str(sensing_wheel)!r})
# A test runner can itself have an editable development bootstrap installed.
# Deny that optional package to exercise the wheel-only registration route.
class NoDevelopmentBootstrap:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'visualizer_extensions':
            raise ModuleNotFoundError('No development bootstrap', name=fullname)
        return None
sys.meta_path.insert(0, NoDevelopmentBootstrap())
from visualizer.src.extensions import registered_runtime_extensions
features = registered_runtime_extensions()
assert [feature.key for feature in features] == ['experiments', 'sensing']
assert 'visualizer_extensions' not in sys.modules
assert 'visualizer.src.io.private_frame_source_extensions' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_runs_saved_input_recipe_comparison(sensing_wheel, tmp_path):
    """The tutorial utility must run from the wheel outside the source checkout."""
    from tests.shared.test_sensing_experiment import _synthetic_recording

    source = _synthetic_recording(tmp_path / "source")
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(sensing_wheel)!r})
import shared.sensing.experiment as experiment
from shared.sensing.postprocessor import PostProcessingConfig
assert experiment.__file__.startswith({str(sensing_wheel)!r})
result = experiment.compare_recipes(
    Path({str(source)!r}), {{'example': PostProcessingConfig(
        cfar_threshold_scale=3., cfar_min_snr_db=6.,
        min_detection_range_m=1., max_detection_range_m=8.)}},
    Path({str(tmp_path / 'comparison')!r}), figures=False, range_gate_m=1.0,
)
assert result['source_unchanged']
assert result['result_count'] == 1
assert result['recipes']['example']['metrics']['total_tp'] == 1
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_runs_cpu_fusion_without_generator_or_gui(sensing_wheel, tmp_path):
    """Fusion consumer and estimator must be usable from a standard installation."""
    script = f"""
import sys
sys.path.insert(0, {str(sensing_wheel)!r})
class NoRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'sionna', 'mitsuba', 'drjit', 'tensorflow', 'torch', 'PySide6'}}:
            raise AssertionError('Fusion imported runtime: ' + fullname)
        return None
sys.meta_path.insert(0, NoRuntime())
import numpy as np
import shared.sensing.fusion.core as fusion
import shared.sensing.fusion.experiment as experiment
import shared.sensing.fusion.figures as figures
for module in (fusion, experiment, figures):
    assert module.__file__.startswith({str(sensing_wheel)!r})
sensors = [fusion.Sensor('a', (0.,0.,1.), (0.,0.,0.)),
           fusion.Sensor('b', (5.,-5.,1.), (np.pi/2,0.,0.))]
state = np.array([5.,0.,1.,.7,.4,0.])
reports = [fusion.Observation(s.sensor_id, '0', 0., *fusion.measurement_model(state,s)[0]) for s in sensors]
estimate = fusion.MonostaticFusion(sensors).step(0., reports)['tracks']
assert len(estimate) == 1
np.testing.assert_allclose(estimate[0]['velocity_m_s'], state[3:], atol=.002)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_imports_complete_volume_ui_outside_source(
    sensing_wheel: Path, tmp_path: Path
) -> None:
    """A build alone must not hide missing feature modules behind lazy imports."""
    expected = {
        "visualizer/src/sensing/volume_window.py",
        "visualizer/src/sensing/volume_workflow.py",
        "visualizer/src/sensing/volume_antenna.py",
        "visualizer/src/sensing/volume_antenna_controls.py",
        "visualizer/src/sensing/volume_cell_edges.py",
    }
    with zipfile.ZipFile(sensing_wheel) as archive:
        assert expected <= set(archive.namelist())
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(sensing_wheel)!r})
import visualizer.src.sensing.volume_window as window
import visualizer.src.sensing.volume_workflow as workflow
from PySide6.QtWidgets import QApplication
from visualizer.src.sensing.volume_antenna_controls import ReceivePatternControls
assert {str(sensing_wheel)!r} in window.__file__, window.__file__
assert {str(sensing_wheel)!r} in workflow.__file__, workflow.__file__
app = QApplication([])
controls = ReceivePatternControls()
assert controls.state()["mode"] == "off"
controls.close()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_imports_visual_experiments_outside_source(
    sensing_wheel: Path, tmp_path: Path
) -> None:
    """Pygfx experiments are installed as a complete, explicitly gated feature."""
    expected = {
        "visualizer/src/experiments/runtime.py",
        "visualizer/src/experiments/service.py",
        "visualizer/src/experiments/window.py",
        "visualizer/src/experiments/path_analysis.py",
        "visualizer/src/experiments/range_geometry.py",
        "visualizer/src/experiments/range_shell.py",
        "visualizer/src/renderers/pygfx/experiments.py",
    }
    with zipfile.ZipFile(sensing_wheel) as archive:
        assert expected <= set(archive.namelist())
    script = f"""
import sys
from types import SimpleNamespace
sys.path.insert(0, {str(sensing_wheel)!r})
from visualizer.src.experiments import runtime, service, window, range_geometry
from visualizer.src.renderers.pygfx import experiments
for module in (runtime, service, window, range_geometry, experiments):
    assert {str(sensing_wheel)!r} in module.__file__, module.__file__
assert runtime.enabled(SimpleNamespace(renderer=SimpleNamespace(renderer_id="pygfx")))
assert not runtime.enabled(SimpleNamespace(renderer=SimpleNamespace(renderer_id="open3d")))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_preserves_project_and_upstream_license_texts(sensing_wheel: Path) -> None:
    """The project SPDX change must not discard the adapted producer's terms."""
    with zipfile.ZipFile(sensing_wheel) as archive:
        metadata = _wheel_metadata(archive)
        assert metadata["License-Expression"] == "GPL-3.0-only"
        license_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/licenses/LICENSE")
        ]
        assert len(license_members) == 1
        assert b"Version 3, 29 June 2007" in archive.read(license_members[0])
        assert b"Apache License" in archive.read("generator/osm/LICENSE")
        notice = archive.read("generator/osm/NOTICE.md").decode()
        assert "NVIDIA" in notice and "Geo2SigMap" in notice


def test_wheel_registers_osm_and_constructs_builder_without_private_hooks(
    sensing_wheel: Path, tmp_path: Path
) -> None:
    """A public install must support recipe validation and Builder without downloading."""
    script = f"""
import sys
sys.path.insert(0, {str(sensing_wheel)!r})
class NoPrivateScenarioHooks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'shared.scenarios._private_extensions':
            raise ModuleNotFoundError('No private scenario hooks', name=fullname)
        return None
sys.meta_path.insert(0, NoPrivateScenarioHooks())
from shared.scenarios.extensions import registered_extension_keys, registered_scene_source_keys
assert 'osm' in registered_scene_source_keys()
assert 'calibration' not in registered_extension_keys()
assert 'shared.scenarios._private_extensions' not in sys.modules
from shared.scenarios.yaml import validate_scenario_data
validate_scenario_data({{'schema_version': 2, 'timeline': {{'steps': 1, 'duration_s': 0.}},
    'scene': {{'source': 'osm', 'osm': {{'bbox': [48.8675, 2.3265, 48.876, 2.3385],
        'terrain': False, 'height_enrichment': False}}}}}})
from PySide6.QtWidgets import QApplication
from visualizer.src.authoring import osm_panel, osm_worker, osm_jobs
for module in (osm_panel, osm_worker, osm_jobs):
    assert {str(sensing_wheel)!r} in module.__file__, module.__file__
app = QApplication([])
panel = osm_panel.OsmScenePanel()
assert panel.bbox is None and panel._build_job is None and panel._map_job is None
panel.close()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
