"""Batch transient material-mode changes through resolved appearance."""

from __future__ import annotations

from typing import Any

from .base import BaseService
from .material_modes import MaterialModeService


class MaterialModeCommandService(BaseService):
    """Republish scene and target entries after a material overlay changes."""

    def __init__(self, mode_service: MaterialModeService) -> None:
        """Bind command application to the shared material-mode registry."""
        super().__init__()
        self.mode_service = mode_service

    def apply_material_modes(
        self,
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
        refresh_appearance_batch: Any,
        *,
        material_key: str | None = None,
        visual_material_key: Any = None,
        update_renderer: bool = True,
    ) -> bool:
        """Republish matching persistent objects without changing manual state."""
        combined_entries = mesh_entries + target_entries
        resolve_key = visual_material_key if callable(visual_material_key) else None

        def _entry_visual_key(entry: dict[str, Any]) -> str:
            if resolve_key is not None:
                return str(resolve_key(entry))
            return str(entry.get("material_type") or "default")

        material_ids = {entry.get("material_id", "Unknown") for entry in combined_entries}
        material_ids.update(_entry_visual_key(entry) for entry in combined_entries)
        self.mode_service.register_materials(material_ids)
        affected_entries = (
            [
                entry
                for entry in combined_entries
                if material_key is None
                or str(entry.get("material_id") or "Unknown") == material_key
                or _entry_visual_key(entry) == material_key
            ]
            if combined_entries
            else []
        )
        if not callable(refresh_appearance_batch):
            return False
        return bool(
            refresh_appearance_batch(
                affected_entries,
                materials_changed=False,
                update_renderer=update_renderer,
            )
        )
