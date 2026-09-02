"""Renderer lifecycle orchestration for the visualizer composition root.

Renderer creation happens in ``services.construct_services``; this module owns
the first visualizer window/canvas initialization once a scene path actually
needs rendering. That split keeps Qt startup responsive and lets startup code
pre-read camera state before the backend chooses its initial camera.
"""

from __future__ import annotations

from typing import Any

from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR
from .viewport_workspace import ViewportState


def boot_visualizer_empty(viz: Any, *, window_title: str = "ORCHAV") -> None:
    """Create an empty renderer window so the unified update pipeline can start.

    Window geometry comes from the Qt layout calculation when available. A
    pending session camera suppresses backend default-camera placement so the
    restored camera can be applied without a visible jump.
    """
    o3d_w = 1024
    o3d_h = 768
    o3d_left = -1
    o3d_top = -1
    capabilities = renderer_capabilities(viz.renderer)
    embedded = getattr(viz, "_viewport_mode", "detached") == "embedded"
    host_parent = None
    if embedded:
        if not capabilities.embedded_viewport:
            raise RuntimeError(
                f"Renderer {getattr(viz.renderer, 'renderer_id', '<unknown>')} "
                "does not support an embedded viewport"
            )
        host = getattr(viz, "_viewport_host", None)
        if host is None:
            raise RuntimeError("Embedded viewport host is not available")
        # The canvas page must be the active stack page before its geometry is
        # authoritative. Hidden QStackedWidget pages retain a default size.
        # Page visibility is application-owned; renderers only parent a canvas
        # into the final page and must never show that page themselves.
        host.set_state(ViewportState.ACTIVE)
        host_parent = host.canvas_parent
        host_size = host_parent.size()
        if host_size.width() > 0 and host_size.height() > 0:
            o3d_w = host_size.width()
            o3d_h = host_size.height()
    elif viz._window_layout is not None:
        if renderer_capabilities(viz.renderer).physical_window_size:
            o3d_w = viz._window_layout.renderer_physical_width
            o3d_h = viz._window_layout.renderer_physical_height
        else:
            o3d_w = viz._window_layout.renderer_logical_width
            o3d_h = viz._window_layout.renderer_logical_height
        o3d_left = viz._window_layout.renderer_x
        o3d_top = viz._window_layout.renderer_y
    try:
        initialize_kwargs = {
            "left": o3d_left,
            "top": o3d_top,
            "suppress_default_camera": bool(viz._pending_camera),
        }
        if embedded:
            initialize_kwargs["host_parent"] = host_parent
        viz.vis = viz.renderer.initialize_visualizer(
            window_title,
            o3d_w,
            o3d_h,
            **initialize_kwargs,
        )
        viz.renderer.set_background_color(DEFAULT_SCENE_BACKGROUND_COLOR)
        if not hasattr(viz, "current_background_color"):
            viz.current_background_color = list(DEFAULT_SCENE_BACKGROUND_COLOR)

        viz.vis_initialized = True
        viz.force_update_next_frame = True
        if embedded:
            final_size = viz._viewport_host.canvas_parent.size()
            viz._on_viewport_resized(final_size.width(), final_size.height())
    except Exception:
        stop_renderer_session(viz, host_state=ViewportState.ERROR)
        raise


def stop_renderer_session(
    viz: Any,
    *,
    host_state: ViewportState | str = ViewportState.EMPTY,
) -> None:
    """Close one renderer session and reset application ownership atomically."""

    renderer = getattr(viz, "renderer", None)
    close = getattr(renderer, "close", None)
    try:
        if callable(close):
            close()
    finally:
        viz.vis_initialized = False
        viz.vis = None
        if getattr(viz, "_viewport_mode", "detached") == "embedded":
            host = getattr(viz, "_viewport_host", None)
            if host is not None:
                host.set_state(host_state)
