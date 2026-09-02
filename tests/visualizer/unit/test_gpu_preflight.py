"""Regression tests for the Open3D/Filament GPU preflight."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from visualizer import gpu_preflight

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _clear_preflight_env(monkeypatch) -> None:
    for key in (
        "DISPLAY",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "ORCHAV_GPU_PREFLIGHT_APPLIED",
        "ORCHAV_NO_GPU_PREFLIGHT",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("display", [None, ":1"])
def test_apply_sets_nvidia_icd_only_for_open3d_headless(monkeypatch, display: str | None) -> None:
    _clear_preflight_env(monkeypatch)
    if display is not None:
        monkeypatch.setenv("DISPLAY", display)
    monkeypatch.setattr(gpu_preflight, "_find_nvidia_icd", lambda: "/tmp/10_nvidia.json")

    gpu_preflight.apply(wants_gui=False, argv=["prog", "--renderer", "open3d"])

    assert os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/tmp/10_nvidia.json"
    assert os.environ["ORCHAV_GPU_PREFLIGHT_APPLIED"] == "1"


@pytest.mark.parametrize("display", [None, ":1", "localhost:11.0"])
def test_apply_leaves_open3d_gui_gl_selection_untouched(monkeypatch, display: str | None) -> None:
    _clear_preflight_env(monkeypatch)
    if display is not None:
        monkeypatch.setenv("DISPLAY", display)
    icd_lookups: list[bool] = []
    monkeypatch.setattr(
        gpu_preflight,
        "_find_nvidia_icd",
        lambda: icd_lookups.append(True) or "/tmp/10_nvidia.json",
    )

    gpu_preflight.apply(wants_gui=True, argv=["prog", "--renderer", "open3d"])

    assert os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") is None
    assert os.environ.get("ORCHAV_GPU_PREFLIGHT_APPLIED") is None
    assert icd_lookups == []


def test_apply_skips_pygfx_even_with_display(monkeypatch) -> None:
    _clear_preflight_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(gpu_preflight, "_find_nvidia_icd", lambda: "/tmp/10_nvidia.json")

    gpu_preflight.apply(wants_gui=True, argv=["prog", "--renderer", "pygfx"])

    assert os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") is None
    assert os.environ.get("ORCHAV_GPU_PREFLIGHT_APPLIED") is None


def test_apply_honors_opt_out(monkeypatch) -> None:
    _clear_preflight_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("ORCHAV_NO_GPU_PREFLIGHT", "1")
    monkeypatch.setattr(gpu_preflight, "_find_nvidia_icd", lambda: "/tmp/10_nvidia.json")

    gpu_preflight.apply(wants_gui=True, argv=["prog", "--renderer", "open3d"])

    assert os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") is None
    assert os.environ.get("ORCHAV_GPU_PREFLIGHT_APPLIED") is None


def test_apply_skips_help_invocations(monkeypatch) -> None:
    _clear_preflight_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(gpu_preflight, "_find_nvidia_icd", lambda: "/tmp/10_nvidia.json")

    gpu_preflight.apply(wants_gui=True, argv=["prog", "--renderer", "open3d", "--help"])

    assert os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") is None
    assert os.environ.get("ORCHAV_GPU_PREFLIGHT_APPLIED") is None


@pytest.mark.parametrize(
    "argv",
    [
        ["orchav-visualizer", "--renderer", "open3d"],
        ["orchav-visualizer", "--renderer=open3d"],
    ],
)
def test_macos_rejects_open3d_before_startup(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="not supported on macOS.*--renderer pygfx"):
        gpu_preflight.reject_unsupported_macos_open3d(
            argv,
            platform_name="darwin",
        )


@pytest.mark.parametrize(
    ("argv", "platform_name"),
    [
        (["orchav-visualizer", "--renderer", "pygfx"], "darwin"),
        (["orchav-visualizer"], "darwin"),
        (["orchav-visualizer", "--renderer", "open3d"], "win32"),
        (["orchav-visualizer", "--renderer=open3d", "--help"], "darwin"),
    ],
)
def test_macos_renderer_gate_allows_supported_invocations(
    argv: list[str], platform_name: str
) -> None:
    gpu_preflight.reject_unsupported_macos_open3d(argv, platform_name=platform_name)


def test_console_entry_point_runs_platform_gate_before_visualizer_import() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["orchav-visualizer"] == "visualizer.__main__:main"

    script = textwrap.dedent("""
        import importlib
        import sys
        import types

        from visualizer import gpu_preflight

        events = []
        gpu_preflight.reject_unsupported_macos_open3d = lambda: events.append("platform_gate")

        visualizer_module = types.ModuleType("visualizer.visualizer")

        def _getattr(name):
            if name == "main":
                assert events == ["platform_gate"], events
                events.append("visualizer_import")
                return lambda: None
            raise AttributeError(name)

        visualizer_module.__getattr__ = _getattr
        sys.modules["visualizer.visualizer"] = visualizer_module
        sys.modules.pop("visualizer.__main__", None)

        imported = importlib.import_module("visualizer.__main__")
        assert callable(imported.main)
        assert events == ["platform_gate", "visualizer_import"]
        """)
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)
