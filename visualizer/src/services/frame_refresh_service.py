"""Current-frame refresh policy shared by panels and controllers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer

from shared.logging import get_logger

from .base import BaseService
from .cache_service import CacheInvalidationScope

logger = get_logger("orchav.frame_refresh_service")


class FrameRefreshService(BaseService):
    """Invalidate frame data and schedule a public current-frame refresh."""

    def __init__(
        self,
        visualizer: Any,
        *,
        timer_single_shot: Optional[Callable[[int, Callable[[], None]], None]] = None,
    ) -> None:
        """Store the visualizer owner and timer hook used for delayed refreshes."""
        super().__init__()
        self.visualizer = visualizer
        self._timer_single_shot = timer_single_shot or QTimer.singleShot

    def refresh_current_frame_after_data_change(
        self,
        *,
        reason: str,
        delay_ms: int = 300,
    ) -> bool:
        """Invalidate data caches and refresh the current frame after a mutation."""
        viz = self.visualizer
        current_frame = getattr(viz, "animation_step", None)
        if current_frame is None:
            logger.debug("No current frame to refresh after %s", reason)
            return False

        self._invalidate_current_frame(int(current_frame), reason=reason)

        if hasattr(viz, "last_app_state"):
            viz.last_app_state = None
        if hasattr(viz, "force_update_next_frame"):
            viz.force_update_next_frame = True

        self._timer_single_shot(
            max(0, int(delay_ms)),
            lambda frame=int(current_frame): self._trigger_frame_update(frame),
        )
        logger.info(
            "Scheduled frame refresh for frame %d after %s",
            int(current_frame),
            reason,
        )
        return True

    def _invalidate_current_frame(self, frame: int, *, reason: str) -> None:
        """Invalidate broad frame-data caches and the specific raw frame."""
        cache_service = getattr(self.visualizer, "cache_service", None)
        if cache_service is None:
            return

        invalidate = getattr(cache_service, "invalidate", None)
        if callable(invalidate):
            invalidate(CacheInvalidationScope.FRAME_DATA, reason=reason)

        invalidate_frame = getattr(cache_service, "invalidate_frame", None)
        if callable(invalidate_frame) and invalidate_frame(frame):
            logger.debug("Cleared frame cache entry for frame %d after %s", frame, reason)

    def _trigger_frame_update(self, frame: int) -> None:
        """Refresh through public visualizer APIs only."""
        viz = self.visualizer
        try:
            update_frame = getattr(viz, "update_frame", None)
            if callable(update_frame):
                update_frame(frame)
                return

            schedule_update = getattr(viz, "schedule_update", None)
            if callable(schedule_update):
                schedule_update()
                return

            logger.warning("No public frame refresh API available for frame %d", frame)
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("Error refreshing frame %d after data change: %s", frame, exc)
