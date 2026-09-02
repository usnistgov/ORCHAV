"""Renderer factory for the ORCHAV visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.logging import get_logger

from .protocol import RendererProtocol
from .registry import (
    DEFAULT_RENDERER_ID,
    RendererId,
    canonicalize_renderer_id,
)

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.renderers.factory")


def create_renderer(
    visualizer: OrchavVisualizer,
    renderer_type: RendererId = DEFAULT_RENDERER_ID,
) -> RendererProtocol:
    """Create the configured renderer backend.

    Supported canonical renderers are:
    - ``pygfx``: default pygfx + wgpu backend.
    - ``open3d``: Open3D O3DVisualizer + Filament backend.

    """
    canonical_renderer = canonicalize_renderer_id(renderer_type)

    if canonical_renderer == "pygfx":
        try:
            from .pygfx.renderer import PygfxRenderer
        except ImportError as exc:
            raise ImportError(
                "Pygfx renderer dependencies are missing. Install the default runtime "
                "from a cloned repository with: python -m pip install -e ."
            ) from exc

        logger.info("Creating PygfxRenderer (pygfx + wgpu backend)")
        return PygfxRenderer(visualizer)

    if canonical_renderer == "open3d":
        from .open3d.renderer import Open3DRenderer

        logger.info("Creating Open3DRenderer (Open3D O3DVisualizer backend)")
        return Open3DRenderer(visualizer)

    raise AssertionError(f"Unhandled renderer backend: {canonical_renderer}")
