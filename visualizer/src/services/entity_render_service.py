"""Renderer-neutral visual entity synchronization service.

Visualizer services use :class:`VisualEntity` for any scene thing with a stable
identity: targets, TX markers, RX markers, and their optional labels. This
service is the handoff point from renderer-neutral payloads to renderer-owned
objects; native Open3D/pygfx geometry should stay behind renderer APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ..model import VisualEntity
from .base import BaseService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer


class EntityRenderService(BaseService):
    """Sync visual entities through the common renderer object surface."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind the service to the visualizer-owned renderer."""
        super().__init__()
        self.visualizer = visualizer

    @property
    def renderer(self):
        """Return the current renderer from the visualizer."""
        return getattr(self.visualizer, "renderer", None)

    def sync_entities(self, entities: Iterable[VisualEntity]) -> dict[str, bool]:
        """Ensure all visual entities exist with current transform/material/visibility."""
        results: dict[str, bool] = {}
        renderer = self.renderer
        if renderer is None:
            return results
        with renderer.batch_updates():
            for entity in entities:
                results[entity.entity_id] = self.sync_entity(entity)
        return results

    def sync_entity(self, entity: VisualEntity) -> bool:
        """Ensure one entity's primary render object and optional label are synced."""
        renderer = self.renderer
        if renderer is None:
            return False

        object_ok = bool(renderer.ensure_object(entity.render_object))
        label_ok = self._sync_label(entity)
        return object_ok and label_ok

    def remove_entity(self, entity: VisualEntity | str, *, label_id: str | None = None) -> bool:
        """Remove an entity primary object and optional label."""
        render_id = entity.render_id if isinstance(entity, VisualEntity) else str(entity)
        if isinstance(entity, VisualEntity) and label_id is None:
            label_id = entity.label_id
        renderer = self.renderer
        if renderer is None:
            return False
        object_removed = bool(renderer.remove_object(render_id))
        label_removed = not label_id or bool(renderer.remove_object(label_id))
        return object_removed and label_removed

    def _sync_label(self, entity: VisualEntity) -> bool:
        """Sync an optional text label through the same object contract."""
        label = entity.label_render_object
        renderer = self.renderer
        if label is None:
            return True
        if renderer is None:
            return False
        return bool(renderer.ensure_object(label))
