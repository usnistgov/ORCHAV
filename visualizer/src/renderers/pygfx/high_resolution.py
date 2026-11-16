"""Native-resolution, current-camera figure capture without resizing the viewport.

Tiles use pygfx's virtual logical view, so screen-sized glyphs, markers, and
lines keep their figure-relative sizes. Layout is resolved by the caller once,
never separately for a tile. Export renderers/cameras are short-lived and have
no interactive callbacks or animation clock.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np

_MAX_OUTPUT_PIXELS = 100_000_000
_MAX_TILE_EDGE = 4096


class ScreenshotCancelled(RuntimeError):
    """The caller cancelled a figure before it could be saved."""


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ScreenshotCancelled("Screenshot export cancelled")


def _clone_camera(camera: Any, logical_size: tuple[float, float]) -> Any:
    """Clone projection and world pose, leaving live controller state alone."""
    if type(camera).__name__ == "ScreenCoordsCamera":
        # This camera's base projection uses _view_size rather than the
        # virtual full size. Keep its coordinates in full-image logical pixels.
        class FixedLogicalScreenCamera(type(camera)):
            def set_view_size(self, width, height):
                super().set_view_size(*logical_size)

        clone = FixedLogicalScreenCamera(invert_y=camera._invert_y)
    else:
        clone = type(camera)()
    clone.set_state(camera.get_state())
    clone.local.matrix = np.array(camera.world.matrix, copy=True)
    return clone


def _copy_effects(renderer: Any) -> tuple[Any, ...]:
    """Create fresh built-in effects, never share their GPU texture caches."""
    result = []
    options = {
        "FXAAPass": (),
        "DDAAPass": ("max_edge_iters",),
        "DepthPass": (),
        "PhysicalBasedBloomPass": (
            "bloom_strength",
            "max_mip_levels",
            "filter_radius",
            "use_karis_average",
        ),
        "EDLPass": ("strength", "radius", "depth_edge_threshold"),
    }
    for effect in getattr(renderer, "effect_passes", ()):
        if not effect.enabled:
            continue
        name = type(effect).__name__
        if name not in options or not type(effect).__module__.startswith("pygfx."):
            raise ValueError(f"High-resolution export does not support {name}; disable it first")
        result.append(type(effect)(**{key: getattr(effect, key) for key in options[name]}))
    return tuple(result)


def _tile_margin(effects: tuple[Any, ...], tiled: bool) -> int:
    """Bound the sampling footprint; reject effects with nonlocal support."""
    if not tiled:
        return 0
    margin = 2
    for effect in effects:
        name = type(effect).__name__
        if name == "DDAAPass":
            margin += 3 * int(effect.max_edge_iters) + 4
        elif name == "FXAAPass":
            margin += 32
        elif name == "DepthPass":
            continue
        else:
            raise ValueError(
                f"Tiled export cannot preserve {name} across seams; reduce the "
                "resolution or disable that effect before exporting"
            )
    return margin


def _render_view(
    owner: Any,
    scene: Any,
    camera: Any,
    *,
    output_size: tuple[int, int],
    logical_size: tuple[float, float],
    max_tile_size: int | None,
    cancel_check: Callable[[], bool] | None,
    overlay_scene: Any = None,
    overlay_camera: Any = None,
    progress_callback: Callable[[float], None] | None = None,
) -> np.ndarray:
    from rendercanvas.offscreen import RenderCanvas

    width, height = output_size
    if min(width, height) < 1 or width * height > _MAX_OUTPUT_PIXELS:
        raise ValueError(f"Screenshot must contain 1 to {_MAX_OUTPUT_PIXELS:,} pixels")
    device_limit = int(owner._renderer.device.limits["max-texture-dimension-2d"])
    limit = min(device_limit, _MAX_TILE_EDGE if max_tile_size is None else int(max_tile_size))
    if limit < 16:
        raise ValueError("Screenshot tile size must be at least 16 pixels")
    effects = _copy_effects(owner._renderer)
    margin = _tile_margin(effects, max(width, height) > limit)
    step = limit - 2 * margin
    if step < 1:
        raise ValueError("Screenshot tile size is too small for the active anti-aliasing")
    clone = _clone_camera(camera, logical_size)
    overlay_clone = (
        _clone_camera(overlay_camera, logical_size) if overlay_camera is not None else None
    )
    # A pre-existing subview is composed with each export tile instead of lost.
    original_offset = getattr(camera, "_view_offset", None)
    view = (
        dict(original_offset)
        if original_offset
        else dict(
            full_width=logical_size[0],
            full_height=logical_size[1],
            x=0.0,
            y=0.0,
            width=logical_size[0],
            height=logical_size[1],
        )
    )
    canvas = RenderCanvas(size=(min(width, limit), min(height, limit)), pixel_ratio=1)
    try:
        renderer = owner._gfx.WgpuRenderer(canvas, pixel_scale=1, enable_events=False)
        renderer.effect_passes = effects
        renderer.gamma_correction = owner._renderer.gamma_correction
        result = np.empty((height, width, 3), dtype=np.uint8)
        tile_count = ((width + step - 1) // step) * ((height + step - 1) // step)
        completed = 0
        for top in range(0, height, step):
            bottom = min(top + step, height)
            for left in range(0, width, step):
                _check_cancelled(cancel_check)
                right = min(left + step, width)
                x0, y0 = max(0, left - margin), max(0, top - margin)
                x1, y1 = min(width, right + margin), min(height, bottom + margin)
                canvas.set_physical_size(x1 - x0, y1 - y0)
                clone.set_view_offset(
                    view["full_width"],
                    view["full_height"],
                    view["x"] + x0 * view["width"] / width,
                    view["y"] + y0 * view["height"] / height,
                    (x1 - x0) * view["width"] / width,
                    (y1 - y0) * view["height"] / height,
                )
                renderer.render(scene, clone, flush=overlay_scene is None)
                if overlay_scene is not None:
                    overlay_clone.set_view_offset(
                        logical_size[0],
                        logical_size[1],
                        x0 * logical_size[0] / width,
                        y0 * logical_size[1] / height,
                        (x1 - x0) * logical_size[0] / width,
                        (y1 - y0) * logical_size[1] / height,
                    )
                    renderer.render(overlay_scene, overlay_clone, flush=True)
                tile = owner._coerce_rgb8_image(renderer.snapshot())
                if tile is None or tile.shape[:2] != (y1 - y0, x1 - x0):
                    raise RuntimeError("High-resolution renderer returned an unexpected image size")
                result[top:bottom, left:right] = tile[
                    top - y0 : bottom - y0, left - x0 : right - x0
                ]
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed / tile_count)
        _check_cancelled(cancel_check)
        return result
    finally:
        canvas.close()


@contextmanager
def _preserve_camera_state(owner: Any):
    """Restore preparation mutations too, including on cancellation/failure."""
    saved = []
    for name in ("_camera", "_minimap_camera", "_minimap_overlay_camera"):
        camera = getattr(owner, name, None)
        if camera is not None:
            offset = getattr(camera, "_view_offset", None)
            saved.append(
                (camera, camera.get_state(), camera._view_size, dict(offset) if offset else None)
            )
    try:
        yield
    finally:
        for camera, state, size, offset in saved:
            camera.set_state(state)
            camera.set_view_size(*size)
            if offset is None:
                camera.clear_view_offset()
            else:
                camera.set_view_offset(
                    *(
                        offset[key]
                        for key in ("full_width", "full_height", "x", "y", "width", "height")
                    )
                )


def render_high_resolution_frame(
    owner: Any,
    *,
    resolution_scale: float,
    include_hud: bool,
    max_tile_size: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Render more pixels without changing the live camera or viewport."""
    with _preserve_camera_state(owner):
        return _render_prepared_frame(
            owner,
            resolution_scale=resolution_scale,
            include_hud=include_hud,
            max_tile_size=max_tile_size,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )


def _render_prepared_frame(
    owner: Any,
    *,
    resolution_scale: float,
    include_hud: bool,
    max_tile_size: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Render the same frame at more pixels, keeping its logical layout fixed.

    No GUI events are processed here: simulation, controllers, and animated
    overlays stay at the selected frame. Cancellation is checked between tiles.
    No low-resolution fallback is allowed to masquerade as a successful export.
    """
    _check_cancelled(cancel_check)
    if progress_callback is not None:
        progress_callback(0.0)
    # This resolves callouts once in full-image coordinates and prepares the
    # same headlight, depth range, and minimap as an ordinary screenshot.
    baseline = owner._coerce_rgb8_image(owner._capture_renderer_snapshot(include_hud=include_hud))
    if baseline is None:
        raise RuntimeError("Cannot prepare the current frame for high-resolution export")
    height, width = baseline.shape[:2]
    output_size = (
        max(1, round(width * resolution_scale)),
        max(1, round(height * resolution_scale)),
    )
    logical_size = owner._canvas.get_logical_size()
    result = _render_view(
        owner,
        owner._scene,
        owner._camera,
        output_size=output_size,
        logical_size=logical_size,
        max_tile_size=max_tile_size,
        cancel_check=cancel_check,
        progress_callback=(
            (lambda fraction: progress_callback(fraction * 0.9))
            if progress_callback is not None
            else None
        ),
    )
    if include_hud:
        if (
            bool(getattr(owner, "_minimap_enabled", False))
            and owner._viewport_hud_policy().enabled
            and owner.compute_scene_bounds(scope="whole") is not None
            and getattr(owner, "_minimap_overlay_scene", None) is not None
        ):
            x, y, w, h = owner._compute_minimap_rect()
            sx, sy = output_size[0] / logical_size[0], output_size[1] / logical_size[1]
            left, top = round(x * sx), round(y * sy)
            right, bottom = round((x + w) * sx), round((y + h) * sy)
            from .target_views import without_main_view_annotations

            with without_main_view_annotations(owner, guides_only=True):
                minimap = _render_view(
                    owner,
                    owner._scene,
                    owner._minimap_camera,
                    output_size=(right - left, bottom - top),
                    logical_size=(w, h),
                    max_tile_size=max_tile_size,
                    cancel_check=cancel_check,
                    overlay_scene=owner._minimap_overlay_scene,
                    overlay_camera=owner._minimap_overlay_camera,
                )
            # The minimap scene includes the same opaque scene background as
            # the live viewport; Qt HUD widgets are composited above it below.
            rgba = np.concatenate(
                (minimap, np.full((*minimap.shape[:2], 1), 255, np.uint8)), axis=2
            )
            owner._alpha_composite_rgba(result, rgba, left=left, top=top)
        result = owner._composite_visible_qt_overlays(result)
    elif getattr(owner, "_hud_overlay_specs", {}).get("map_attribution", {}).get("visible"):
        result = owner._composite_visible_qt_overlays(result, attribution_only=True)
    _check_cancelled(cancel_check)
    if progress_callback is not None:
        progress_callback(1.0)
    return result
