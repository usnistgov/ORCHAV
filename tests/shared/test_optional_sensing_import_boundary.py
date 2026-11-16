"""Regression tests for lazy sensing imports and missing-dependency diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENSING_INSTALL_GUIDANCE = "Install ORCHAV with: python -m pip install orchav"


def _run_with_blocked_imports(
    code: str,
    *blocked_modules: str,
) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter that rejects selected imports."""

    bootstrap = f"""
import builtins

blocked_modules = {json.dumps(blocked_modules)}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in blocked_modules):
        exc = ModuleNotFoundError(f"No module named '{{name}}'")
        exc.name = name
        raise exc
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
{textwrap.dedent(code)}
"""
    return subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _dependency_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "angle_peak_detection": False,
        "dry_run": False,
        "evaluate": False,
        "figure": [],
        "figures": False,
        "gif": False,
        "gif_3d": False,
        "gif_only": False,
        "micro_doppler": False,
        "report": False,
        "threshold_sweep": False,
        "tracker": "none",
        "vertex_overlay": None,
        "vote_fusion": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_cli_with_missing_dependencies(
    tmp_path: Path,
    *,
    cli_args: list[str],
    blocked_modules: list[str],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output_root = tmp_path / "analysis-output-must-remain-absent"
    argv = [
        "orchav-sensing",
        str(tmp_path / "source-not-needed-for-dependency-preflight"),
        "--output-dir",
        str(output_root),
        *cli_args,
    ]
    result = _run_with_blocked_imports(f"""
import sys
from pathlib import Path

import shared.sensing.postprocessor as sensing_postprocessor

blocked_modules = {json.dumps(blocked_modules)}
original_import_module = sensing_postprocessor.importlib.import_module

def guarded_import_module(name, package=None):
    if any(name == module or name.startswith(module + ".") for module in blocked_modules):
        exc = ModuleNotFoundError(f"No module named '{{name}}'")
        exc.name = name
        raise exc
    return original_import_module(name, package)

sensing_postprocessor.importlib.import_module = guarded_import_module
sys.argv = {json.dumps(argv)}
try:
    sensing_postprocessor.main()
except SystemExit as exc:
    assert exc.code == 2
else:
    raise AssertionError("dependency preflight did not stop the CLI")
assert not Path({str(output_root)!r}).exists()
""")
    return result, output_root


def test_sensing_cli_help_does_not_load_analysis_dependencies() -> None:
    result = _run_with_blocked_imports(
        """
import sys

from shared.sensing.postprocessor import main

assert "shared.sensing.plotting.viz" not in sys.modules
sys.argv = ["orchav-sensing", "--help"]
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("--help did not terminate through argparse")
""",
        "scipy",
        "sklearn",
        "matplotlib",
        "PIL",
    )

    _assert_success(result)
    assert "usage:" in result.stdout


def test_scipy_is_required_only_for_nonempty_association() -> None:
    result = _run_with_blocked_imports(
        """
import numpy as np

from shared.sensing.evaluation import AssociationResult, associate_detections_to_gt

empty = associate_detections_to_gt(
    gt_ranges=np.array([]),
    gt_velocities=np.array([]),
    det_ranges=np.array([]),
    det_velocities=np.array([]),
)
assert isinstance(empty, AssociationResult)

try:
    associate_detections_to_gt(
        gt_ranges=np.array([10.0]),
        gt_velocities=np.array([1.0]),
        det_ranges=np.array([10.0]),
        det_velocities=np.array([1.0]),
    )
except RuntimeError as exc:
    assert str(exc) == (
        "Sensing association requires SciPy. "
        'Install ORCHAV with: python -m pip install orchav'
    )
else:
    raise AssertionError("nonempty association unexpectedly ran without SciPy")
""",
        "scipy",
    )

    _assert_success(result)


@pytest.mark.parametrize(
    ("tracker_type", "requires_scipy"),
    [
        ("none", False),
        ("m_of_n", False),
        ("kalman", True),
        ("jpdaf", True),
        ("cartesian", True),
        ("cartesian3d", True),
    ],
)
def test_dependency_map_requires_scipy_for_hungarian_trackers(
    tracker_type: str,
    requires_scipy: bool,
) -> None:
    from shared.sensing.postprocessor import _requested_cli_dependencies

    requirements = _requested_cli_dependencies(_dependency_args(tracker=tracker_type))

    assert ("scipy" in requirements) is requires_scipy


@pytest.mark.parametrize(
    ("overrides", "expected_modules"),
    [
        ({"evaluate": True}, {"scipy"}),
        ({"report": True}, {"scipy"}),
        ({"threshold_sweep": True}, {"scipy"}),
        ({"figures": True}, {"scipy", "matplotlib"}),
        ({"vote_fusion": True}, {"sklearn"}),
        ({"angle_peak_detection": True}, {"sklearn"}),
        ({"gif": True}, {"matplotlib", "PIL"}),
        ({"gif_only": True}, {"matplotlib", "PIL"}),
        ({"gif_3d": True}, {"matplotlib", "PIL"}),
        ({"micro_doppler": True}, {"matplotlib"}),
        (
            {"micro_doppler": True, "vertex_overlay": "meshes"},
            {"scipy", "matplotlib", "trimesh"},
        ),
        ({"dry_run": True, "figures": True, "tracker": "kalman"}, set()),
    ],
)
def test_dependency_map_matches_requested_operations(
    overrides: dict[str, object],
    expected_modules: set[str],
) -> None:
    from shared.sensing.postprocessor import _requested_cli_dependencies

    requirements = _requested_cli_dependencies(_dependency_args(**overrides))

    assert set(requirements) == expected_modules


def test_dependency_preflight_preserves_present_but_broken_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.sensing.postprocessor as sensing_postprocessor

    broken_runtime = ImportError("SciPy runtime is present but broken", name="numpy._core")

    def import_module(name: str, package: str | None = None) -> object:
        del package
        if name == "scipy.optimize":
            raise broken_runtime
        return object()

    monkeypatch.setattr(sensing_postprocessor.importlib, "import_module", import_module)

    with pytest.raises(ImportError) as exc_info:
        sensing_postprocessor._preflight_cli_dependencies(_dependency_args(evaluate=True))

    assert exc_info.value is broken_runtime


def test_dependency_preflight_selects_agg_before_importing_pyplot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.sensing.postprocessor as sensing_postprocessor

    calls: list[tuple[str, str]] = []

    class MatplotlibProbe:
        def use(self, backend: str) -> None:
            calls.append(("backend", backend))

    def import_module(name: str, package: str | None = None) -> object:
        del package
        calls.append(("import", name))
        if name == "matplotlib":
            return MatplotlibProbe()
        if name == "matplotlib.pyplot":
            assert ("backend", "Agg") in calls
            return object()
        raise AssertionError(f"unexpected dependency probe: {name}")

    monkeypatch.setattr(sensing_postprocessor.importlib, "import_module", import_module)

    sensing_postprocessor._preflight_cli_dependencies(_dependency_args(micro_doppler=True))

    assert calls == [
        ("import", "matplotlib"),
        ("backend", "Agg"),
        ("import", "matplotlib.pyplot"),
    ]


def test_cli_preflight_aggregates_missing_sensing_dependencies_before_output_claim(
    tmp_path: Path,
) -> None:
    result, output_root = _run_cli_with_missing_dependencies(
        tmp_path,
        cli_args=["--evaluate", "--tracker", "kalman", "--figures", "--gif"],
        blocked_modules=["scipy", "matplotlib", "PIL"],
    )

    _assert_success(result)
    assert not output_root.exists()
    assert "SciPy" in result.stderr
    assert "Matplotlib" in result.stderr
    assert "Pillow" in result.stderr
    assert result.stderr.count(SENSING_INSTALL_GUIDANCE) == 1


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--vote-fusion"],
        ["--angle-peak-detection"],
        ["--ghost-filter-azimuth-gate", "3"],
    ],
)
def test_cli_rejects_deferred_angular_processing_before_dependency_guidance(
    tmp_path: Path,
    cli_args: list[str],
) -> None:
    result, output_root = _run_cli_with_missing_dependencies(
        tmp_path,
        cli_args=cli_args,
        blocked_modules=["sklearn"],
    )

    _assert_success(result)
    assert not output_root.exists()
    assert "typed P,E,A,D,R postprocessing not implemented" in result.stderr
    assert "scikit-learn" not in result.stderr
    assert SENSING_INSTALL_GUIDANCE not in result.stderr


def test_cli_vertex_overlay_gives_separate_trimesh_guidance_before_output_claim(
    tmp_path: Path,
) -> None:
    result, output_root = _run_cli_with_missing_dependencies(
        tmp_path,
        cli_args=["--micro-doppler", "--vertex-overlay", str(tmp_path / "meshes")],
        blocked_modules=["trimesh"],
    )

    _assert_success(result)
    assert not output_root.exists()
    assert "not included in the standard ORCHAV installation" in result.stderr
    assert 'python -m pip install "trimesh>=3.20"' in result.stderr
    assert SENSING_INSTALL_GUIDANCE not in result.stderr


def test_cli_vertex_overlay_requires_micro_doppler_before_output_claim(tmp_path: Path) -> None:
    output_root = tmp_path / "analysis-output-must-remain-absent"
    result = _run_with_blocked_imports(f"""
import sys
from pathlib import Path

from shared.sensing.postprocessor import main

sys.argv = [
    "orchav-sensing",
    {str(tmp_path / "source-not-needed")!r},
    "--output-dir",
    {str(output_root)!r},
    "--vertex-overlay",
    {str(tmp_path / "meshes")!r},
]
try:
    main()
except SystemExit as exc:
    assert exc.code == 2
else:
    raise AssertionError("invalid vertex-overlay request unexpectedly continued")
assert not Path({str(output_root)!r}).exists()
""")

    _assert_success(result)
    assert "--vertex-overlay requires --micro-doppler" in result.stderr
    assert not output_root.exists()


def test_cli_m_of_n_does_not_require_scipy(tmp_path: Path) -> None:
    result, output_root = _run_cli_with_missing_dependencies(
        tmp_path,
        cli_args=["--tracker", "m_of_n"],
        blocked_modules=["scipy"],
    )

    _assert_success(result)
    assert not output_root.exists()
    assert SENSING_INSTALL_GUIDANCE not in result.stderr
    assert "No packed HDF5" in result.stderr


def test_cli_preserves_unexpected_errors_from_a_present_dependency(monkeypatch, tmp_path):
    import shared.sensing.postprocessor as postprocessor

    original_import = postprocessor.importlib.import_module

    def broken_dependency(name, package=None):
        if name == "scipy":
            raise ValueError("injected invalid installed dependency")
        return original_import(name, package)

    monkeypatch.setattr(postprocessor.importlib, "import_module", broken_dependency)
    monkeypatch.setattr(
        "sys.argv", ["orchav-sensing", str(tmp_path / "unused-source"), "--evaluate"]
    )
    with pytest.raises(ValueError, match="injected invalid installed dependency"):
        postprocessor.main()
