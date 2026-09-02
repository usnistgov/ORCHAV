"""Unit coverage for the deterministic MPC Explorer capture helper."""

from __future__ import annotations

import pytest

from scripts.capture_mpc_explorer_visual import _logical_capture_resolution_scale


@pytest.mark.parametrize("pixel_ratio", (1.0, 1.25, 1.5))
def test_logical_capture_scale_normalizes_native_dpr(pixel_ratio: float) -> None:
    """Map a DPR-native framebuffer back to the requested logical dimensions."""
    width, height = 720, 480
    physical_width = round(width * pixel_ratio)
    physical_height = round(height * pixel_ratio)

    scale = _logical_capture_resolution_scale({"canvas_pixel_ratio": pixel_ratio})

    assert round(physical_width * scale) == width
    assert round(physical_height * scale) == height


def test_logical_capture_scale_uses_renderer_fallback() -> None:
    """Use renderer DPR metadata when the canvas does not expose a ratio."""
    scale = _logical_capture_resolution_scale(
        {
            "canvas_pixel_ratio": None,
            "renderer_pixel_ratio": 1.25,
        }
    )

    assert scale == pytest.approx(0.8)


@pytest.mark.parametrize("pixel_ratio", (None, 0.0, -1.0, float("nan")))
def test_logical_capture_scale_defaults_for_invalid_metadata(pixel_ratio: object) -> None:
    """Keep the native dimensions when no usable DPR metadata is available."""
    assert _logical_capture_resolution_scale({"canvas_pixel_ratio": pixel_ratio}) == 1.0
