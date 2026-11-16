"""Failure-safe synchronization for renderer-neutral sensing overlays."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from shared.logging import get_logger

from ..model import RenderObject

if TYPE_CHECKING:
    from ..renderers.protocol import RendererProtocol

logger = get_logger("orchav.services.sensing_overlay_sync")


class SensingOverlaySync:
    """Own the desired renderer IDs used by sensing scene overlays.

    The service tracks only which semantic overlay IDs may need removal.
    Applied snapshots and no-op detection belong to each renderer's
    idempotent ``ensure_object()`` implementation.
    """

    def __init__(self) -> None:
        """Initialize empty semantic overlay ownership."""
        self._owned_ids: set[str] = set()

    @property
    def owned_ids(self) -> frozenset[str]:
        """Return every overlay ID that may still exist in the renderer."""
        return frozenset(self._owned_ids)

    def sync(
        self,
        renderer: RendererProtocol,
        snapshots: Iterable[RenderObject],
    ) -> bool:
        """Converge renderer ownership to ``snapshots`` and report success."""
        desired: dict[str, RenderObject] = {}
        for snapshot in snapshots:
            if not isinstance(snapshot, RenderObject):
                raise TypeError("Sensing overlays must be complete RenderObject snapshots")
            if snapshot.id in desired:
                raise ValueError(f"Duplicate sensing overlay ID: {snapshot.id}")
            desired[snapshot.id] = snapshot

        # Remove obsolete owners first. A failed removal blocks new owners so
        # one semantic overlay transition cannot temporarily display both the
        # stale and replacement layer sets.
        removals_succeeded = True
        for object_id in sorted(self._owned_ids - desired.keys()):
            if self._remove_owned_object(renderer, object_id):
                self._owned_ids.discard(object_id)
            else:
                removals_succeeded = False
        if not removals_succeeded:
            return False

        all_succeeded = True
        for object_id, snapshot in desired.items():
            # An ensure may create native geometry before a later material or
            # transform operation fails. Record ownership before calling it so
            # that object can still be removed if the desired set changes.
            self._owned_ids.add(object_id)
            try:
                ensured = bool(renderer.ensure_object(snapshot))
            except (RuntimeError, TypeError, ValueError):
                logger.warning(
                    "Failed to synchronize sensing overlay '%s'",
                    object_id,
                    exc_info=True,
                )
                ensured = False
            if not ensured:
                all_succeeded = False
        return all_succeeded

    def clear(self, renderer: RendererProtocol) -> bool:
        """Remove every owned sensing overlay without forgetting failures."""
        return self.sync(renderer, ())

    @staticmethod
    def _remove_owned_object(renderer: RendererProtocol, object_id: str) -> bool:
        """Remove one owned ID through the idempotent object contract."""
        try:
            return bool(renderer.remove_object(object_id))
        except (RuntimeError, TypeError, ValueError):
            logger.warning(
                "Failed to remove sensing overlay '%s'",
                object_id,
                exc_info=True,
            )
            return False
