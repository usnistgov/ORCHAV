"""Tests for generator headless preview compatibility stubs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUB_MODULES = (
    "ipywidgets",
    "ipywidgets.widgets",
    "ipywidgets.embed",
    "pythreejs",
    "IPython",
    "IPython.display",
)


def _load_headless_module():
    module_path = ROOT / "generator" / "_headless_compat.py"
    spec = importlib.util.spec_from_file_location("headless_compat_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clear_headless_environment(monkeypatch):
    for name in STUB_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (
        "ORCHAV_HEADLESS",
        "JPY_PARENT_PID",
        "COLAB_RELEASE_TAG",
        "KAGGLE_KERNEL_RUN_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_headless_mode_installs_preview_stubs(monkeypatch):
    module = _load_headless_module()
    _clear_headless_environment(monkeypatch)

    module.enable_headless_preview_compat()

    assert "ipywidgets" in sys.modules
    assert "ipywidgets.widgets" in sys.modules
    assert "ipywidgets.embed" in sys.modules
    assert "pythreejs" in sys.modules
    assert sys.modules["ipywidgets"].widgets is sys.modules["ipywidgets.widgets"]

    with pytest.raises(RuntimeError, match="preview support is disabled"):
        sys.modules["ipywidgets.widgets"].Button()
    with pytest.raises(RuntimeError, match="preview support is disabled"):
        sys.modules["ipywidgets.embed"].embed_snippet()
    with pytest.raises(RuntimeError, match="preview support is disabled"):
        sys.modules["pythreejs"].Mesh()


def test_headless_opt_out_does_not_install_preview_stubs(monkeypatch):
    module = _load_headless_module()
    _clear_headless_environment(monkeypatch)
    monkeypatch.setenv("ORCHAV_HEADLESS", "0")

    module.enable_headless_preview_compat()

    for name in STUB_MODULES:
        assert name not in sys.modules
