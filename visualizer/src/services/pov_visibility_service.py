"""POV-mode entity visibility ownership."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from shared.logging import get_logger

logger = get_logger("orchav.pov_visibility_service")


def is_hidden_for_pov(app_state: Any, node_type: str, index: int) -> bool:
    """Return whether an entity should be hidden by the active POV camera."""
    if app_state is None:
        return False
    if getattr(app_state, "camera_mode", "overview") != "pov":
        return False

    pov_hidden = getattr(app_state, "pov_hidden_node", None)
    if pov_hidden is None:
        return False

    try:
        hidden_type, hidden_index = pov_hidden
    except (TypeError, ValueError):
        return False
    try:
        return str(hidden_type) == str(node_type) and int(hidden_index) == int(index)
    except (TypeError, ValueError):
        return False


class PovVisibilityService:
    """Publish POV visibility intent through the semantic entity owners."""

    def __init__(self, visualizer: Any) -> None:
        """Bind POV visibility transitions to the app-level visualizer object."""
        self.visualizer = visualizer

    def hide(self, entity_info: Optional[dict]) -> bool:
        """Set the POV-hidden entity and republish complete semantic snapshots."""
        viz = self.visualizer
        if not entity_info or not hasattr(viz, "vis") or viz.vis is None:
            return False

        try:
            entity_type = str(entity_info.get("type", "")).lower()
            if entity_type not in {"tx", "rx", "target"}:
                return False
            index = int(entity_info.get("index"))
            if index < 0:
                return False
            return self.set_hidden_entity((entity_type, index))
        except (RuntimeError, TypeError, ValueError, KeyError, AttributeError):
            logger.exception("Error hiding POV entity")
            return False

    def current_hidden_entity(self) -> tuple[str, int] | None:
        """Return the normalized entity currently hidden by POV policy."""
        return self._current_hidden_ref()

    def set_hidden_entity(self, entity_ref: tuple[str, int] | None) -> bool:
        """Replace POV visibility intent and republish both affected entities.

        Camera changes use this as a small semantic transaction: stage the new
        hidden entity, then restore the prior reference if the renderer rejects
        the camera update.
        """
        viz = self.visualizer
        if not hasattr(viz, "vis") or viz.vis is None:
            return False

        try:
            normalized_ref: tuple[str, int] | None = None
            if entity_ref is not None:
                entity_type, raw_index = entity_ref
                entity_type = str(entity_type).lower()
                index = int(raw_index)
                if entity_type not in {"tx", "rx", "target"} or index < 0:
                    return False
                normalized_ref = (entity_type, index)

            previous_ref = self._current_hidden_ref()
            if previous_ref == normalized_ref:
                return True

            refs = tuple(ref for ref in (previous_ref, normalized_ref) if ref is not None)
            self._set_hidden_ref(normalized_ref)
            if refs and not self._sync_semantic_snapshots(refs):
                logger.warning(
                    "POV visibility synchronization failed; restoring %s",
                    previous_ref,
                )
                self._set_hidden_ref(previous_ref)
                if not self._sync_semantic_snapshots(refs):
                    logger.error("POV visibility rollback synchronization remains incomplete")
                return False
            if normalized_ref is None:
                logger.debug("Cleared POV-hidden entity")
            else:
                logger.debug(
                    "Updated POV-hidden entity to %s%d",
                    normalized_ref[0],
                    normalized_ref[1] + 1,
                )
            return True
        except (RuntimeError, TypeError, ValueError, AttributeError):
            logger.exception("Error updating POV-hidden entity")
            return False

    def restore(self, *, update_renderer: bool = True) -> None:
        """Clear POV hiding and republish the entity's current semantic state."""
        viz = self.visualizer
        if not hasattr(viz, "vis") or viz.vis is None:
            return

        try:
            self.set_hidden_entity(None)

            if update_renderer:
                renderer = getattr(viz, "renderer", None)
                poll = getattr(renderer, "poll_events", None)
                if callable(poll):
                    poll()
                update = getattr(renderer, "update_renderer", None)
                if callable(update):
                    update()
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            logger.error("Error restoring entity visibility: %s", exc)

    def _current_hidden_ref(self) -> tuple[str, int] | None:
        """Return the normalized POV-hidden entity reference from app state."""
        raw = getattr(getattr(self.visualizer, "app_state", None), "pov_hidden_node", None)
        if raw is None:
            return None
        try:
            entity_type, index = raw
            entity_type = str(entity_type).lower()
            index = int(index)
        except (TypeError, ValueError):
            return None
        if entity_type not in {"tx", "rx", "target"} or index < 0:
            return None
        return entity_type, index

    def _set_hidden_ref(self, entity_ref: tuple[str, int] | None) -> None:
        """Update the sole application-owned POV visibility state."""
        viz = self.visualizer
        set_state = getattr(viz, "set_state", None)
        if callable(set_state):
            set_state(pov_hidden_node=entity_ref)
            return
        app_state = getattr(viz, "app_state", None)
        if app_state is not None:
            app_state.pov_hidden_node = entity_ref

    def _sync_semantic_snapshots(self, entity_refs: Iterable[tuple[str, int]]) -> bool:
        """Ask the node/target owner to republish complete effective snapshots."""
        node_service = getattr(self.visualizer, "node_service", None)
        refresh = getattr(node_service, "sync_pov_entity_visibility", None)
        if not callable(refresh):
            logger.warning("NodeService POV visibility synchronization is unavailable")
            return False
        try:
            return bool(refresh(tuple(entity_refs)))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            logger.exception("POV semantic snapshot synchronization failed")
            return False
