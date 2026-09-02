"""Renderer handoff helpers for TX/RX node visual entities."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from ..model import RenderObjectState, Transform, VisualEntity
from ..services.object_identity import make_node_geometry_name
from .entity_render_service import EntityRenderService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer


class NodeRenderSync:
    """Sync TX/RX node render state through renderer-neutral APIs."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        entity_render_service: EntityRenderService,
    ) -> None:
        """Bind node renderer synchronization to the active visualizer renderer."""
        self.visualizer = visualizer
        self._entity_render_service = entity_render_service

    @property
    def renderer(self) -> Any:
        """Return the active renderer, if one is installed."""
        return getattr(self.visualizer, "renderer", None)

    def entity_renderer_available(self) -> bool:
        """Return True when the renderer supports declarative entity sync."""
        return callable(getattr(self.renderer, "ensure_object", None))

    def entity_renderer_ready(self) -> bool:
        """Return True when the active renderer can accept entity sync calls."""
        if not self.entity_renderer_available():
            return False
        renderer = self.renderer
        initialized = getattr(renderer, "_initialized", None)
        if isinstance(initialized, bool):
            return initialized
        return bool(
            getattr(self.visualizer, "vis_initialized", False)
            or getattr(self.visualizer, "vis", None) is not None
            or initialized is not None
        )

    def sync_entities(self, entities: Iterable[VisualEntity]) -> dict[str, bool]:
        """Sync node visual entities through the shared entity renderer."""
        return self._entity_render_service.sync_entities(entities)

    def remove_marker_entity(self, kind: str, index: int) -> bool:
        """Remove one TX/RX marker entity by stable render IDs."""
        kind_norm = str(kind).lower()
        label_id = make_node_geometry_name(kind_norm, int(index), "label")
        return bool(
            self._entity_render_service.remove_entity(
                make_node_geometry_name(kind_norm, int(index), "marker"),
                label_id=label_id,
            )
        )

    def set_render_handle_visibility(self, handle: Any, visible: bool) -> bool:
        """Apply effective visibility without changing semantic handle state."""
        if not isinstance(handle, RenderObjectState):
            return False
        effective_visible = bool(visible)
        set_visible = getattr(self.renderer, "set_visible", None)
        if callable(set_visible) and set_visible(handle.id, effective_visible):
            return True

        # A direct visibility update can fail when the object has not been
        # registered yet. Ensure one complete effective-state snapshot in
        # that case; never follow a successful ensure with a duplicate setter.
        ensure_object = getattr(self.renderer, "ensure_object", None)
        if callable(ensure_object):
            return bool(ensure_object(handle.to_render_object(effective_visible=effective_visible)))
        return False

    def sync_label(
        self,
        *,
        label_id: str,
        label: Any,
        visible: bool,
        anchor_position: Any | None = None,
        offset: Any | None = None,
    ) -> bool:
        """Synchronize a neutral label through the common object contract."""
        if not isinstance(label, RenderObjectState) or label.id != label_id:
            return False
        if anchor_position is not None:
            anchor = np.asarray(anchor_position, dtype=float).reshape(-1)
            delta = (
                np.zeros(3, dtype=float)
                if offset is None
                else np.asarray(offset, dtype=float).reshape(-1)
            )
            if anchor.size < 3 or delta.size < 3:
                return False
            label.metadata["layout_anchor"] = tuple(float(value) for value in anchor[:3])
            label.metadata["layout_offset"] = tuple(float(value) for value in delta[:3])
            label.world_transform = Transform.from_translation(anchor[:3] + delta[:3])
        effective_visible = bool(label.visible and visible)
        ensure_object = getattr(self.renderer, "ensure_object", None)
        return bool(
            callable(ensure_object)
            and ensure_object(label.to_render_object(effective_visible=effective_visible))
        )

    def remove_label(self, label_id: str) -> bool:
        """Remove a persistent label render object."""
        remover = getattr(self.renderer, "remove_object", None)
        return bool(callable(remover) and remover(label_id))

    @staticmethod
    def set_label_world_position(label: Any, position: Any) -> bool:
        """Move a neutral label state to an absolute world position."""
        if not isinstance(label, RenderObjectState):
            return False
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return False
        label.world_transform = Transform.from_translation(pos[:3])
        return True

    @staticmethod
    def set_label_color(label: Any, color: Any) -> bool:
        """Apply a uniform color to a neutral label material."""
        if not isinstance(label, RenderObjectState):
            return False
        rgb = np.asarray(color, dtype=float).reshape(-1)
        if rgb.size < 3 or not np.all(np.isfinite(rgb[:3])):
            return False
        alpha = float(label.material.base_color[3])
        label.material = replace(
            label.material,
            base_color=(float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha),
        )
        return True
