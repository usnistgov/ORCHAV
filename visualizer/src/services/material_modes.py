"""Transient material-overlay registry for persistent object appearance."""

from __future__ import annotations

from collections.abc import Iterable

from ..materials.appearance import MaterialDisplayMode
from .base import BaseService


class MaterialModeService(BaseService):
    """Own material-key overlays without mutating object visibility/highlight."""

    def __init__(self) -> None:
        """Initialize the mutable mode registry shared with material panels."""
        super().__init__()
        self._modes: dict[str, MaterialDisplayMode] = {}

    @property
    def modes(self) -> dict[str, MaterialDisplayMode]:
        """Return the mutable material-id to display-mode mapping."""
        return self._modes

    def register_materials(self, materials: Iterable[str]) -> None:
        """Ensure every material has a registered mode."""
        for material_id in materials:
            self._modes.setdefault(material_id, MaterialDisplayMode.NORMAL)

    def set_mode(
        self,
        material_id: str,
        mode: MaterialDisplayMode | str,
    ) -> None:
        """Set the transient display mode for one material id."""
        self._modes[material_id] = MaterialDisplayMode.coerce(mode)

    def resolve_toggled_mode(
        self,
        material_id: str,
        requested_mode: MaterialDisplayMode | str,
    ) -> MaterialDisplayMode:
        """Resolve a checkable-button click into the next material mode."""
        requested = MaterialDisplayMode.coerce(requested_mode)
        if requested is MaterialDisplayMode.NORMAL:
            return requested
        current = self.get_mode(material_id)
        return MaterialDisplayMode.NORMAL if current is requested else requested

    def get_mode(self, material_id: str) -> MaterialDisplayMode:
        """Return the registered display mode, defaulting to normal."""
        return self._modes.get(material_id, MaterialDisplayMode.NORMAL)

    def clear(self) -> None:
        """Clear all registered material display modes."""
        self._modes.clear()
