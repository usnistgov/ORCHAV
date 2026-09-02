"""Window geometry policy for the Qt shell and renderer viewport.

The Qt control window and the renderer surface may be separate native windows.
``compute_layout()`` keeps Qt placement in device-independent logical pixels
while also recording the intended physical renderer size. Renderer lifecycle
code selects the representation required by each backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.logging import get_logger

logger = get_logger("orchav.window_manager")

GAP_PX = 8
MARGIN_PX = 8
QT_MIN_WIDTH = 680
CAPTURE_WORKSPACE_QT_WIDTH = round(QT_MIN_WIDTH * 2 / 3)
O3D_MIN_WIDTH = 640
PRESENTATION_RENDERER_WIDTH = 1440
PRESENTATION_RENDERER_HEIGHT = 1080
CAPTURE_FRAME_WIDTH_PX = 1920
CAPTURE_FRAME_HEIGHT_PX = 1080
LAYOUT_PROFILE_CHOICES = ("auto", "capture-workspace", "capture-renderer")


@dataclass
class WindowLayout:
    """Computed logical placement and physical renderer output dimensions."""

    qt_x: int
    qt_y: int
    qt_width: int
    qt_height: int
    renderer_x: int
    renderer_y: int
    renderer_logical_width: int
    renderer_logical_height: int
    renderer_physical_width: int
    renderer_physical_height: int
    device_pixel_ratio: float


def compute_layout(
    screen_w: int,
    screen_h: int,
    screen_x: int = 0,
    screen_y: int = 0,
    qt_preferred_width: int = QT_MIN_WIDTH,
    layout_profile: str = "auto",
    device_pixel_ratio: float = 1.0,
) -> WindowLayout:
    """Compute a side-by-side layout for Qt controls and renderer output.

    The heuristic favors a presentation-friendly renderer viewport when the
    screen is large enough, while clamping both windows to fit smaller
    displays. Explicit capture profiles target a 1920x1080 physical frame.
    """
    if layout_profile not in LAYOUT_PROFILE_CHOICES:
        choices = ", ".join(LAYOUT_PROFILE_CHOICES)
        raise ValueError(f"Unknown window layout profile {layout_profile!r}; expected {choices}")

    logical_capture_w = _physical_to_logical(CAPTURE_FRAME_WIDTH_PX, device_pixel_ratio)
    logical_capture_h = _physical_to_logical(CAPTURE_FRAME_HEIGHT_PX, device_pixel_ratio)
    usable_w = max(O3D_MIN_WIDTH, screen_w - 2 * MARGIN_PX - GAP_PX)
    usable_h = max(500, screen_h - 2 * MARGIN_PX)

    if layout_profile == "capture-workspace":
        h = min(logical_capture_h, usable_h)
        target_usable_w = min(logical_capture_w - 2 * MARGIN_PX - GAP_PX, usable_w)
        preferred_qt_w = (
            CAPTURE_WORKSPACE_QT_WIDTH if qt_preferred_width == QT_MIN_WIDTH else qt_preferred_width
        )
        qt_w = min(max(preferred_qt_w, 1), target_usable_w - 1)
        qt_w = max(1, qt_w)
        o3d_w = max(1, target_usable_w - qt_w)
        return _make_layout(
            screen_w,
            screen_h,
            screen_x,
            screen_y,
            qt_w,
            h,
            o3d_w,
            device_pixel_ratio,
        )

    if layout_profile == "capture-renderer":
        h = min(logical_capture_h, usable_h)
        renderer_aspect_w = int(round(h * CAPTURE_FRAME_WIDTH_PX / CAPTURE_FRAME_HEIGHT_PX))
        preferred_o3d_w = min(logical_capture_w, max(O3D_MIN_WIDTH, renderer_aspect_w))
        qt_w = min(max(qt_preferred_width, QT_MIN_WIDTH), max(QT_MIN_WIDTH, usable_w))
        o3d_w = max(O3D_MIN_WIDTH, min(preferred_o3d_w, usable_w - qt_w))
        return _make_layout(
            screen_w,
            screen_h,
            screen_x,
            screen_y,
            qt_w,
            h,
            o3d_w,
            device_pixel_ratio,
        )

    logical_presentation_w = _physical_to_logical(
        PRESENTATION_RENDERER_WIDTH,
        device_pixel_ratio,
    )
    logical_presentation_h = _physical_to_logical(
        PRESENTATION_RENDERER_HEIGHT,
        device_pixel_ratio,
    )
    h = min(logical_presentation_h, usable_h)
    qt_w = min(max(qt_preferred_width, QT_MIN_WIDTH), max(QT_MIN_WIDTH, usable_w - O3D_MIN_WIDTH))
    if usable_w - qt_w < O3D_MIN_WIDTH:
        qt_w = max(QT_MIN_WIDTH, usable_w - O3D_MIN_WIDTH)

    o3d_w = max(O3D_MIN_WIDTH, min(logical_presentation_w, usable_w - qt_w))

    return _make_layout(
        screen_w,
        screen_h,
        screen_x,
        screen_y,
        qt_w,
        h,
        o3d_w,
        device_pixel_ratio,
    )


def _physical_to_logical(physical_px: int, device_pixel_ratio: float) -> int:
    """Convert desired captured pixels to Qt logical pixels."""
    try:
        ratio = float(device_pixel_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    if ratio <= 0:
        ratio = 1.0
    return max(1, int(round(physical_px / ratio)))


def qt_min_width_for_profile(layout_profile: str) -> int:
    """Return the shell minimum width for a launch layout profile."""
    if layout_profile == "capture-workspace":
        return CAPTURE_WORKSPACE_QT_WIDTH
    return QT_MIN_WIDTH


def _make_layout(
    screen_w: int,
    screen_h: int,
    screen_x: int,
    screen_y: int,
    qt_w: int,
    h: int,
    o3d_w: int,
    device_pixel_ratio: float,
) -> WindowLayout:
    """Build centered window coordinates from computed Qt/renderer sizes."""
    total_w = qt_w + GAP_PX + o3d_w + 2 * MARGIN_PX
    qt_x = screen_x + (screen_w - total_w) // 2 + MARGIN_PX
    qt_y = screen_y + (screen_h - h) // 2
    renderer_x = qt_x + qt_w + GAP_PX
    renderer_y = qt_y
    try:
        ratio = float(device_pixel_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    if ratio <= 0.0:
        ratio = 1.0

    return WindowLayout(
        qt_x=qt_x,
        qt_y=qt_y,
        qt_width=qt_w,
        qt_height=h,
        renderer_x=renderer_x,
        renderer_y=renderer_y,
        renderer_logical_width=o3d_w,
        renderer_logical_height=h,
        renderer_physical_width=max(1, int(round(o3d_w * ratio))),
        renderer_physical_height=max(1, int(round(h * ratio))),
        device_pixel_ratio=ratio,
    )


def apply_qt_layout(window, layout: WindowLayout) -> None:
    """Position and resize a QMainWindow according to *layout*."""
    window.setGeometry(layout.qt_x, layout.qt_y, layout.qt_width, layout.qt_height)
