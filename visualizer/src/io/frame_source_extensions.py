"""Optional frame-source extension registry for visualizer data modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..extension_loader import ensure_external_extensions_loaded

FrameSourceFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class FrameSourceExtension:
    """Factory metadata for a non-built-in visualizer frame source."""

    mode: str
    factory: FrameSourceFactory
    label: str
    badge_color: str = "#95a5a6"


_REGISTRY: dict[str, FrameSourceExtension] = {}


def register_frame_source_extension(
    mode: str,
    factory: FrameSourceFactory,
    *,
    label: Optional[str] = None,
    badge_color: str = "#95a5a6",
) -> None:
    """Register a frame-source factory for an optional data mode."""
    normalized = mode.strip()
    if not normalized:
        raise ValueError("Frame-source extension mode cannot be empty")
    _REGISTRY[normalized] = FrameSourceExtension(
        mode=normalized,
        factory=factory,
        label=label or normalized.replace("_", " ").title(),
        badge_color=badge_color,
    )


def create_registered_frame_source(scenario: Any) -> Any | None:
    """Create an optional frame source for ``scenario.data_mode`` if registered."""
    ensure_external_extensions_loaded()
    mode = getattr(scenario, "data_mode", "")
    extension = _REGISTRY.get(mode)
    if extension is None:
        return None

    source = extension.factory(scenario)
    if not hasattr(source, "frame_source_label"):
        setattr(source, "frame_source_label", extension.label)
    if not hasattr(source, "frame_source_badge_color"):
        setattr(source, "frame_source_badge_color", extension.badge_color)
    return source


def registered_frame_source_modes() -> list[str]:
    """Return registered optional frame-source mode names."""
    ensure_external_extensions_loaded()
    return sorted(_REGISTRY)
