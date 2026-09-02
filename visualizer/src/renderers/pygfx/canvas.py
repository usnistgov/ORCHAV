"""pygfx canvas, WGPU renderer, and optional effect-pass setup helpers.

This module isolates renderer creation from ``PygfxRenderer`` so environment
flags for anti-aliasing, bloom, depth, clipping, and export readback stay in one
backend-specific place.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "build_pygfx_effect_passes",
    "create_wgpu_renderer",
    "display_renderer_kwargs",
    "export_renderer_kwargs",
    "load_physical_bloom_pass",
]


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean backend env flag with common false spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    """Read a float backend env setting and fall back on invalid input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float for %s=%r", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer backend env setting and fall back on invalid input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer for %s=%r", name, raw)
        return default


def _parse_env_clipping_planes() -> tuple[tuple[float, float, float, float], ...]:
    """Parse ``ORCHAV_PYGFX_CLIP_PLANES`` into world-space plane tuples."""
    raw = os.environ.get("ORCHAV_PYGFX_CLIP_PLANES", "").strip()
    if not raw:
        return ()
    planes: list[tuple[float, float, float, float]] = []
    for entry in raw.split(";"):
        parts = entry.strip().split(",")
        if len(parts) != 4:
            logger.warning("Ignoring bad clipping plane entry %r", entry)
            continue
        try:
            planes.append(tuple(float(p) for p in parts))  # type: ignore[arg-type]
        except ValueError:
            logger.warning("Ignoring non-numeric clipping plane entry %r", entry)
    return tuple(planes)


def load_physical_bloom_pass() -> Any | None:
    """Return the pygfx bloom pass class across supported pygfx layouts."""
    try:
        from pygfx.renderers.wgpu import PhysicalBasedBloomPass

        return PhysicalBasedBloomPass
    except ImportError:
        try:
            from pygfx.renderers.wgpu.engine.bloom import PhysicalBasedBloomPass

            return PhysicalBasedBloomPass
        except ImportError:
            return None


def _load_edl_pass() -> Any | None:
    """Return the optional Eye-Dome-Lighting pass when this pygfx exposes it."""
    try:
        from pygfx.renderers.wgpu.engine.edl import EDLPass

        return EDLPass
    except ImportError:
        return None


def _load_capture_passes() -> dict[str, Any]:
    """Return optional AA/depth pass classes exposed by the installed pygfx."""
    found: dict[str, Any] = {}
    try:
        from pygfx.renderers.wgpu.engine import effectpasses as _ep

        for name in ("FXAAPass", "DDAAPass", "PPAAPass", "DepthPass"):
            cls = getattr(_ep, name, None)
            if cls is not None:
                found[name] = cls
    except ImportError:
        pass
    return found


_AA_MODES = ("off", "fxaa", "ddaa", "ppaa")


def build_pygfx_effect_passes(
    gfx: Any,
    *,
    clear_color: tuple[float, float, float, float],
    aa_mode: str | None = None,
    depth_pass_enabled: bool | None = None,
) -> tuple[Any, ...]:
    """Build optional post-processing passes requested by pygfx env flags."""
    passes: list[Any] = []

    if aa_mode is None:
        env_aa = os.environ.get("ORCHAV_PYGFX_AA_MODE", "").strip().lower()
        aa_mode = env_aa if env_aa in _AA_MODES else "off"
    else:
        aa_mode = str(aa_mode).strip().lower()
        if aa_mode not in _AA_MODES:
            aa_mode = "off"
    if depth_pass_enabled is None:
        depth_pass_enabled = _env_flag("ORCHAV_PYGFX_ENABLE_DEPTH_PASS", False)
    else:
        depth_pass_enabled = bool(depth_pass_enabled)

    if _env_flag("ORCHAV_PYGFX_ENABLE_FOG", False):
        logger.warning(
            "Ignoring ORCHAV_PYGFX_ENABLE_FOG: pygfx FogPass currently whites out dense "
            "scenes on the tested stack"
        )

    if _env_flag("ORCHAV_PYGFX_ENABLE_BLOOM", False):
        bloom_cls = load_physical_bloom_pass()
        if bloom_cls is None:
            logger.warning("Bloom requested but unavailable in this pygfx install")
        else:
            passes.append(
                bloom_cls(
                    bloom_strength=_env_float("ORCHAV_PYGFX_BLOOM_STRENGTH", 0.035),
                    max_mip_levels=max(
                        1,
                        int(_env_float("ORCHAV_PYGFX_BLOOM_MAX_MIP_LEVELS", 6.0)),
                    ),
                    filter_radius=_env_float("ORCHAV_PYGFX_BLOOM_FILTER_RADIUS", 0.005),
                    use_karis_average=_env_flag("ORCHAV_PYGFX_BLOOM_USE_KARIS_AVERAGE", False),
                )
            )

    if _env_flag("ORCHAV_PYGFX_ENABLE_EDL", False):
        edl_cls = _load_edl_pass()
        if edl_cls is None:
            logger.warning("EDL requested but unavailable in this pygfx install")
        else:
            edl_depth_edge_threshold = _env_float(
                "ORCHAV_PYGFX_EDL_DEPTH_EDGE_THRESHOLD",
                0.0,
            )
            passes.append(
                edl_cls(
                    strength=_env_float("ORCHAV_PYGFX_EDL_STRENGTH", 1.0),
                    radius=_env_float("ORCHAV_PYGFX_EDL_RADIUS", 1.5),
                    depth_edge_threshold=edl_depth_edge_threshold,
                )
            )

    capture_classes = _load_capture_passes()
    aa_cls_map = {
        "fxaa": capture_classes.get("FXAAPass"),
        "ddaa": capture_classes.get("DDAAPass"),
        "ppaa": capture_classes.get("PPAAPass"),
    }
    if aa_mode != "off":
        cls = aa_cls_map.get(aa_mode)
        if cls is None:
            logger.warning("AA mode %r requested but pass class unavailable", aa_mode)
        else:
            passes.append(cls())

    if depth_pass_enabled:
        depth_cls = capture_classes.get("DepthPass")
        if depth_cls is None:
            logger.warning("Depth pass requested but unavailable")
        else:
            logger.warning(
                "Depth pass enabled: pygfx DepthPass uses raw non-linear "
                "depth and typically renders white on perspective scenes. "
                "Treat as experimental."
            )
            passes.append(depth_cls())

    return tuple(passes)


def export_renderer_kwargs() -> dict[str, Any]:
    """Return WgpuRenderer kwargs used only for offscreen/export readback."""
    kwargs: dict[str, Any] = {}

    pixel_scale = _env_float("ORCHAV_PYGFX_EXPORT_PIXEL_SCALE", 1.0)
    if pixel_scale > 0.0 and abs(pixel_scale - 1.0) > 1e-6:
        kwargs["pixel_scale"] = pixel_scale

    pixel_filter = os.environ.get("ORCHAV_PYGFX_EXPORT_PIXEL_FILTER")
    if pixel_filter:
        kwargs["pixel_filter"] = pixel_filter

    ppaa = os.environ.get("ORCHAV_PYGFX_EXPORT_PPAA")
    if ppaa:
        kwargs["ppaa"] = False if ppaa.strip().lower() in {"0", "false", "no", "off"} else ppaa

    return kwargs


def display_renderer_kwargs() -> dict[str, Any]:
    """Return renderer kwargs for a native-resolution interactive canvas.

    pygfx defaults to 2x internal supersampling on displays whose device-pixel
    ratio is below 2. ORCHAV already renders to the canvas' physical size and
    exposes explicit post-process AA controls, so that implicit supersampling
    multiplies raster work without representing an application quality choice.
    """
    pixel_scale = _env_float("ORCHAV_PYGFX_PIXEL_SCALE", 1.0)
    if pixel_scale <= 0.0:
        logger.warning(
            "Ignoring non-positive ORCHAV_PYGFX_PIXEL_SCALE=%r; using native scale 1.0",
            pixel_scale,
        )
        pixel_scale = 1.0
    return {"pixel_scale": pixel_scale}


def _refresh_renderer_effect_passes(
    renderer: Any,
    gfx: Any,
    *,
    clear_color: tuple[float, float, float, float],
    aa_mode: str | None = None,
    depth_pass_enabled: bool | None = None,
) -> None:
    """Refresh renderer effect passes after renderer or GUI settings change."""
    if renderer is None or not hasattr(renderer, "effect_passes"):
        return
    renderer.effect_passes = build_pygfx_effect_passes(
        gfx,
        clear_color=clear_color,
        aa_mode=aa_mode,
        depth_pass_enabled=depth_pass_enabled,
    )


def create_wgpu_renderer(
    gfx: Any,
    target: Any,
    *,
    clear_color: tuple[float, float, float, float],
    offscreen: bool = False,
    configure_effects: bool = True,
) -> Any:
    """Create a pygfx ``WgpuRenderer`` with version-tolerant constructor args."""
    renderer_cls = getattr(getattr(gfx, "renderers", None), "WgpuRenderer", None)
    if renderer_cls is None:
        renderer_cls = gfx.WgpuRenderer

    kwargs = export_renderer_kwargs() if offscreen else display_renderer_kwargs()
    try:
        renderer = renderer_cls(target, **kwargs)
    except TypeError:
        renderer = renderer_cls(target)

    if configure_effects:
        _refresh_renderer_effect_passes(renderer, gfx, clear_color=clear_color)
    return renderer
