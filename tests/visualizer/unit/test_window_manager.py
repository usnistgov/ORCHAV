"""Tests for the window layout manager."""

from types import SimpleNamespace

from visualizer.src.app.renderer_lifecycle import boot_visualizer_empty
from visualizer.src.app.window_manager import (
    CAPTURE_FRAME_HEIGHT_PX,
    CAPTURE_FRAME_WIDTH_PX,
    CAPTURE_WORKSPACE_QT_WIDTH,
    GAP_PX,
    MARGIN_PX,
    O3D_MIN_WIDTH,
    PRESENTATION_RENDERER_HEIGHT,
    PRESENTATION_RENDERER_WIDTH,
    QT_MIN_WIDTH,
    WindowLayout,
    compute_layout,
    qt_min_width_for_profile,
)
from visualizer.src.renderers.protocol import RendererCapabilities


def test_standard_1080p_layout():
    layout = compute_layout(1920, 1080)
    assert layout.qt_width >= QT_MIN_WIDTH
    assert layout.renderer_logical_width >= O3D_MIN_WIDTH
    # Both windows fit within the screen
    assert layout.qt_x + layout.qt_width + GAP_PX + layout.renderer_logical_width <= 1920
    assert layout.qt_height > 500
    assert layout.renderer_logical_height == layout.qt_height


def test_small_screen_1366():
    layout = compute_layout(1366, 768)
    assert layout.qt_width >= QT_MIN_WIDTH
    assert layout.renderer_logical_width >= O3D_MIN_WIDTH
    assert layout.qt_x + layout.qt_width + GAP_PX + layout.renderer_logical_width <= 1366


def test_4k_screen():
    layout = compute_layout(3840, 2160)
    assert layout.qt_width >= QT_MIN_WIDTH
    assert layout.renderer_logical_width == PRESENTATION_RENDERER_WIDTH
    assert layout.renderer_logical_height == PRESENTATION_RENDERER_HEIGHT
    assert layout.qt_height > 1000


def test_presentation_layout_for_scaled_qhd_display():
    layout = compute_layout(2731, 1392)
    assert layout.qt_width == QT_MIN_WIDTH
    assert layout.renderer_logical_width == PRESENTATION_RENDERER_WIDTH
    assert layout.renderer_logical_height == PRESENTATION_RENDERER_HEIGHT
    assert layout.qt_height == layout.renderer_logical_height
    assert layout.qt_x + layout.qt_width + GAP_PX + layout.renderer_logical_width <= 2731


def test_capture_workspace_layout_targets_1080p_capture_group():
    layout = compute_layout(2731, 1392, layout_profile="capture-workspace")
    total_w = layout.qt_width + GAP_PX + layout.renderer_logical_width + 2 * MARGIN_PX
    assert layout.qt_width == CAPTURE_WORKSPACE_QT_WIDTH
    assert total_w == CAPTURE_FRAME_WIDTH_PX
    assert layout.qt_height == CAPTURE_FRAME_HEIGHT_PX
    assert layout.renderer_logical_height == CAPTURE_FRAME_HEIGHT_PX


def test_capture_renderer_layout_targets_1080p_renderer_when_screen_fits():
    layout = compute_layout(2731, 1392, layout_profile="capture-renderer")
    assert layout.qt_width == QT_MIN_WIDTH
    assert layout.renderer_logical_width == CAPTURE_FRAME_WIDTH_PX
    assert layout.renderer_logical_height == CAPTURE_FRAME_HEIGHT_PX
    assert layout.qt_x + layout.qt_width + GAP_PX + layout.renderer_logical_width <= 2731


def test_capture_renderer_layout_is_dpi_aware_for_150_percent_scaling():
    layout = compute_layout(
        2731,
        1392,
        layout_profile="capture-renderer",
        device_pixel_ratio=1.5,
    )
    assert layout.qt_width == QT_MIN_WIDTH
    assert layout.renderer_logical_width == 1280
    assert layout.renderer_logical_height == 720
    assert layout.renderer_physical_width == CAPTURE_FRAME_WIDTH_PX
    assert layout.renderer_physical_height == CAPTURE_FRAME_HEIGHT_PX


def test_capture_workspace_layout_is_dpi_aware_for_150_percent_scaling():
    layout = compute_layout(
        2731,
        1392,
        layout_profile="capture-workspace",
        device_pixel_ratio=1.5,
    )
    total_w = layout.qt_width + GAP_PX + layout.renderer_logical_width + 2 * MARGIN_PX
    assert layout.qt_width == CAPTURE_WORKSPACE_QT_WIDTH
    assert total_w == 1280
    assert layout.renderer_logical_height == 720
    assert round(total_w * 1.5) == CAPTURE_FRAME_WIDTH_PX
    assert layout.renderer_physical_height == CAPTURE_FRAME_HEIGHT_PX


def test_auto_layout_is_dpi_aware_for_150_percent_scaling():
    layout = compute_layout(2731, 1392, device_pixel_ratio=1.5)

    assert layout.renderer_logical_width == 960
    assert layout.renderer_logical_height == 720
    assert layout.renderer_physical_width == PRESENTATION_RENDERER_WIDTH
    assert layout.renderer_physical_height == PRESENTATION_RENDERER_HEIGHT


def test_unknown_layout_profile_is_rejected():
    try:
        compute_layout(1920, 1080, layout_profile="unknown")
    except ValueError as exc:
        assert "Unknown window layout profile" in str(exc)
    else:
        raise AssertionError("Unknown layout profile should raise ValueError")


def test_retired_layout_profiles_are_rejected():
    # Construct the retired spellings so release hygiene scans can forbid them
    # everywhere else without flagging this negative-compatibility test.
    retired_profiles = ("obs" + "-split", "obs" + "-full")
    for retired_profile in retired_profiles:
        try:
            compute_layout(1920, 1080, layout_profile=retired_profile)
        except ValueError:
            continue
        raise AssertionError(f"Retired layout profile {retired_profile!r} should be rejected")


def test_qt_min_width_for_profile():
    assert qt_min_width_for_profile("auto") == QT_MIN_WIDTH
    assert qt_min_width_for_profile("capture-renderer") == QT_MIN_WIDTH
    assert qt_min_width_for_profile("capture-workspace") == CAPTURE_WORKSPACE_QT_WIDTH


def test_screen_offset():
    layout = compute_layout(1920, 1080, screen_x=100, screen_y=50)
    assert layout.qt_x >= 100 + MARGIN_PX
    assert layout.qt_y >= 50 + MARGIN_PX
    assert layout.renderer_x > layout.qt_x + layout.qt_width


def test_window_layout_dataclass():
    layout = WindowLayout(0, 0, 500, 800, 508, 0, 1000, 800, 1500, 1200, 1.5)
    assert layout.qt_width == 500
    assert layout.renderer_logical_width == 1000
    assert layout.renderer_physical_width == 1500


def test_renderer_lifecycle_selects_backend_window_size_units():
    layout = compute_layout(2731, 1392, device_pixel_ratio=1.5)

    def _boot(uses_physical_window_size: bool) -> tuple[int, int, int, int]:
        calls: list[tuple[int, int, int, int]] = []
        renderer = SimpleNamespace(
            capabilities=RendererCapabilities(physical_window_size=uses_physical_window_size),
            initialize_visualizer=lambda _title, width, height, left, top, **_kwargs: (
                calls.append((width, height, left, top)) or object()
            ),
            set_background_color=lambda _color: None,
        )
        viz = SimpleNamespace(
            _window_layout=layout,
            _pending_camera=None,
            renderer=renderer,
        )
        boot_visualizer_empty(viz, window_title="test")
        return calls[0]

    logical = _boot(False)
    physical = _boot(True)

    assert logical[:2] == (960, 720)
    assert physical[:2] == (1440, 1080)
    assert logical[2:] == physical[2:] == (layout.renderer_x, layout.renderer_y)
