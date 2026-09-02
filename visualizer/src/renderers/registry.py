"""Renderer backend identifiers for supported visualizer backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

CanonicalRendererId = Literal["pygfx", "open3d"]
RendererId = CanonicalRendererId

DEFAULT_RENDERER_ID: CanonicalRendererId = "pygfx"
CANONICAL_RENDERER_IDS: tuple[CanonicalRendererId, ...] = ("pygfx", "open3d")


@dataclass(frozen=True)
class RendererBackend:
    """User-facing metadata for one supported renderer backend."""

    renderer_id: CanonicalRendererId
    label: str
    description: str


RENDERER_BACKENDS: dict[CanonicalRendererId, RendererBackend] = {
    "pygfx": RendererBackend(
        renderer_id="pygfx",
        label="pygfx",
        description="pygfx + wgpu backend",
    ),
    "open3d": RendererBackend(
        renderer_id="open3d",
        label="Open3D/Filament",
        description="Open3D O3DVisualizer + Filament backend",
    ),
}


def canonicalize_renderer_id(renderer_id: str) -> CanonicalRendererId:
    """Return the canonical renderer identifier for a supported CLI value."""
    normalized = str(renderer_id).strip().lower()
    if normalized in RENDERER_BACKENDS:
        return cast(CanonicalRendererId, normalized)
    choices = ", ".join(renderer_choices())
    raise ValueError(f"Unsupported renderer {renderer_id!r}; expected one of: {choices}")


def renderer_choices() -> tuple[str, ...]:
    """Return supported CLI renderer choices."""
    return tuple(CANONICAL_RENDERER_IDS)
