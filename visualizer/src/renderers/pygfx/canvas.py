"""pygfx canvas, WGPU renderer, and optional effect-pass setup helpers.

This module isolates renderer creation from ``PygfxRenderer`` so environment
flags for anti-aliasing, bloom, depth, clipping, and export readback stay in one
backend-specific place.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..effect_settings import DEFAULT_BLOOM, validate_bloom

logger = logging.getLogger(__name__)

__all__ = [
    "build_pygfx_effect_passes",
    "create_wgpu_renderer",
    "display_renderer_kwargs",
    "export_renderer_kwargs",
    "load_physical_bloom_pass",
    "retire_canvas_readback",
]


def retire_canvas_readback(canvas: Any) -> None:
    """Cancel and finish a pending bitmap transfer before Qt is destroyed.

    rendercanvas 2.6 closes the presentation texture but leaves its downloader's
    pending map callback alive. Its native completion can then call a deleted
    Qt CallerHelper. Use the pinned backend's cancellation/drain operation while
    Qt still exists. Screen presentation and uninitialized canvases have no
    downloader. This is a teardown-only compatibility boundary, not a draw cost.
    """
    context = getattr(canvas, "_canvas_context", None)
    downloader = getattr(context, "_downloader", None)
    if downloader is not None:
        downloader._clear_pending_download()


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


def bloom_from_environment() -> dict:
    """Read launch defaults once; live renderer controls override them afterward."""
    settings = dict(
        enabled=_env_flag("ORCHAV_PYGFX_ENABLE_BLOOM", False),
        bloom_strength=_env_float("ORCHAV_PYGFX_BLOOM_STRENGTH", 0.035),
        max_mip_levels=_env_int("ORCHAV_PYGFX_BLOOM_MAX_MIP_LEVELS", 6),
        filter_radius=_env_float("ORCHAV_PYGFX_BLOOM_FILTER_RADIUS", 0.005),
        use_karis_average=_env_flag("ORCHAV_PYGFX_BLOOM_USE_KARIS_AVERAGE", False),
    )
    try:
        return validate_bloom(settings)
    except ValueError:
        logger.warning("Invalid bloom launch settings; using the default effect settings")
        return dict(DEFAULT_BLOOM)


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

        for name in ("FXAAPass", "DepthPass"):
            cls = getattr(_ep, name, None)
            if cls is not None:
                found[name] = cls
    except ImportError:
        pass
    return found


_AA_MODES = ("off", "fxaa")


def _normalize_aa_mode(mode: str) -> str:
    """Keep older workspace/environment AA selections on a concrete FXAA pass."""
    normalized = str(mode).strip().lower()
    return "fxaa" if normalized in {"ddaa", "ppaa"} else normalized


def build_pygfx_effect_passes(
    gfx: Any,
    *,
    clear_color: tuple[float, float, float, float],
    aa_mode: str | None = None,
    depth_pass_enabled: bool | None = None,
    bloom: dict | None = None,
) -> tuple[Any, ...]:
    """Build optional post-processing passes requested by pygfx env flags."""
    passes: list[Any] = []

    if aa_mode is None:
        env_aa = _normalize_aa_mode(os.environ.get("ORCHAV_PYGFX_AA_MODE", ""))
        aa_mode = env_aa if env_aa in _AA_MODES else "off"
    else:
        aa_mode = _normalize_aa_mode(aa_mode)
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

    bloom = bloom_from_environment() if bloom is None else validate_bloom(bloom)
    if bloom["enabled"]:
        bloom_cls = load_physical_bloom_pass()
        if bloom_cls is None:
            logger.warning("Bloom requested but unavailable in this pygfx install")
        else:
            passes.append(
                bloom_cls(**{key: value for key, value in bloom.items() if key != "enabled"})
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
    if aa_mode != "off":
        # PPAAPass is pygfx's abstract marker class, not another AA algorithm.
        cls = capture_classes.get("FXAAPass")
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
        normalized = _normalize_aa_mode(ppaa)
        if normalized in {"0", "false", "no", "off", "none"}:
            kwargs["ppaa"] = "none"
        elif normalized in {"default", "fxaa"}:
            kwargs["ppaa"] = "fxaa"
        else:
            logger.warning("Ignoring unknown export AA mode %r; using Off", ppaa)
            kwargs["ppaa"] = "none"

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
    bloom: dict | None = None,
) -> None:
    """Refresh renderer effect passes after renderer or GUI settings change."""
    if renderer is None or not hasattr(renderer, "effect_passes"):
        return
    renderer.effect_passes = build_pygfx_effect_passes(
        gfx,
        clear_color=clear_color,
        aa_mode=aa_mode,
        depth_pass_enabled=depth_pass_enabled,
        bloom=bloom,
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
