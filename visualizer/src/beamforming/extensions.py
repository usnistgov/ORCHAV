"""Optional beamforming mode registry for visualizer extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..extension_loader import ensure_external_extensions_loaded

BeamformingModeBuilder = Callable[..., Optional[dict[str, Any]]]


@dataclass(frozen=True)
class BeamformingModeExtension:
    """Metadata and builder for an optional beamforming mode."""

    key: str
    label: str
    tooltip: str
    builder: BeamformingModeBuilder


_REGISTRY: dict[str, BeamformingModeExtension] = {}


def register_beamforming_mode(
    key: str,
    *,
    label: str,
    tooltip: str,
    builder: BeamformingModeBuilder,
) -> None:
    """Register an optional beamforming mode."""
    normalized = key.strip()
    if not normalized:
        raise ValueError("Beamforming mode key cannot be empty")
    _REGISTRY[normalized] = BeamformingModeExtension(
        key=normalized,
        label=label,
        tooltip=tooltip,
        builder=builder,
    )


def get_beamforming_mode(key: str) -> BeamformingModeExtension | None:
    """Return the optional beamforming mode registered as ``key``."""
    ensure_external_extensions_loaded()
    return _REGISTRY.get(key)


def registered_beamforming_modes() -> list[BeamformingModeExtension]:
    """Return optional beamforming modes available in this installation."""
    ensure_external_extensions_loaded()
    return [extension for _, extension in sorted(_REGISTRY.items())]
