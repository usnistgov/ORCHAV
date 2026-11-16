"""Validate GPU selection in the optional Paris capture script without a GUI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _capture_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scenarios/showcases/paris_opera_multitarget/capture.py"
    )
    spec = importlib.util.spec_from_file_location("paris_capture_adapter_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def capture_module():
    return _capture_module()


def _adapter(vendor, backend):
    return SimpleNamespace(info={"vendor": vendor, "backend_type": backend})


def test_select_capture_adapter_filters_backend_and_uses_zero_based_index(capture_module):
    intel = _adapter("Intel", "Vulkan")
    first = _adapter("NVIDIA", "Vulkan")
    other_backend = _adapter("NVIDIA", "D3D12")
    second = _adapter("nvidia", "vulkan")
    adapters = [intel, first, other_backend, second]

    assert capture_module.select_capture_adapter(adapters, 0) is first
    assert capture_module.select_capture_adapter(adapters, 1) is second


def test_select_capture_adapter_reports_missing_nvidia_vulkan(capture_module):
    with pytest.raises(RuntimeError, match="No NVIDIA Vulkan adapter"):
        capture_module.select_capture_adapter([_adapter("Intel", "Vulkan")], 0)


@pytest.mark.parametrize("index", [-1, 1])
def test_select_capture_adapter_rejects_unavailable_index(capture_module, index):
    with pytest.raises(ValueError, match="Select an index from 0 to 0"):
        capture_module.select_capture_adapter([_adapter("NVIDIA", "Vulkan")], index)
