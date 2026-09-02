"""Frame loader setup and teardown helpers for visualizer startup/load paths.

Frame sources that expose a shared ``DataProvider`` can use
``FrameLoaderService`` and the visualizer frame cache. Scene-only sources and
providers without that adapter fall back to direct frame-source access.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.frames.loader import FrameLoaderService
from shared.frames.provider_base import DataProvider
from shared.logging import get_logger

logger = get_logger(__name__)


def teardown_frame_loader(viz: Any) -> None:
    """Invalidate and detach the current frame loader before source changes.

    AnimationService is detached even if cache invalidation fails so a stale
    provider cannot serve frames after scenario cleanup switches sources.
    """
    if getattr(viz, "frame_loader", None):
        try:
            viz.frame_loader.invalidate()
        except (OSError, RuntimeError):
            logger.debug("Failed to invalidate frame loader cache", exc_info=True)
    viz.frame_loader = None
    if hasattr(viz, "animation_service"):
        viz.animation_service.set_frame_loader(None)


def configure_frame_loader(viz: Any, frame_source: Any) -> None:
    """Initialize ``FrameLoaderService`` from the active frame source/provider.

    The loader is attached to ``AnimationService`` only when the source exposes
    a shared data-provider contract; otherwise the animation path keeps using
    the frame source directly.
    """
    teardown_frame_loader(viz)
    provider: Optional[DataProvider] = None
    if isinstance(frame_source, DataProvider):
        provider = frame_source
    elif hasattr(frame_source, "provider"):
        candidate = getattr(frame_source, "provider", None)
        if isinstance(candidate, DataProvider):
            provider = candidate
    if provider is None:
        logger.debug("Frame loader not configured: frame source has no data provider")
        if hasattr(viz, "animation_service"):
            viz.animation_service.set_frame_loader(None)
        return

    cache = viz.cache_service.frame_cache
    cache.set_max_size(viz._frame_loader_cache_size)
    viz.frame_loader = FrameLoaderService(provider, cache=cache)
    logger.debug(
        "Frame loader configured with provider %s (cache size=%d)",
        provider.__class__.__name__,
        viz._frame_loader_cache_size,
    )
    if hasattr(viz, "animation_service"):
        viz.animation_service.set_frame_loader(viz.frame_loader)
