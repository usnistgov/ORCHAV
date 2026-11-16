"""Temporary geometric pulses for a Demo tour of one accepted frame."""

from __future__ import annotations

import numpy as np

from ..experiments.cinematic_pulses import CinematicPulseGeometry, cinematic_subset
from ..metrics.mpc_path_catalog import MpcPathCatalog, MpcPathScope
from ..renderers.pygfx.experiments import ExperimentOverlay


class DemoMpcTimeline:
    """Borrow recorded, visible paths without opening or altering MPC analysis."""

    def __init__(self, viz):
        service, catalog = self._visible_catalog(viz)
        self.viz = viz
        self.service = service
        self.identity = self._identity()
        ids, self.selection_note = cinematic_subset(
            catalog, catalog.scope_path_ids(MpcPathScope.RENDERED), 50
        )
        self.geometry = CinematicPulseGeometry(catalog, ids)
        if not len(self.geometry.ids):
            raise ValueError("Show at least one MPC path before starting the showcase.")
        self.overlay = ExperimentOverlay(viz.renderer)
        try:
            self.overlay.cinematic_routes(catalog, self.geometry.ids)
            self.overlay.set_cinematic_halo_opacity(0.06)
        except Exception:
            self.close()
            raise
        self.started = 0.0
        self.duration = 1.0

    @staticmethod
    def _visible_catalog(viz):
        """Share the exact pulse eligibility check with the capability-aware picker."""
        service = getattr(viz, "extension_services", {}).get("experiments")
        packet = getattr(service, "packet", None)
        canonical = getattr(packet, "canonical_data", None)
        if canonical is None:
            raise ValueError("Showcase needs an accepted frame with visible MPC paths.")
        clock = getattr(service, "_clock", None)
        if clock is not None and clock.isActive():
            raise ValueError("Pause the MPC analysis timeline before starting the showcase.")
        visibility = getattr(packet, "mpc_visibility", None)
        catalog = MpcPathCatalog(
            canonical,
            filtered_path_mask=packet.path_mask,
            rendered_segment_mask=packet.segment_mask,
            rendered_paths_enabled=bool(getattr(visibility, "effective_paths", True)),
        )
        return service, catalog

    @classmethod
    def available(cls, viz) -> bool:
        """Check visible recorded paths without creating overlays or changing playback."""
        try:
            _, catalog = cls._visible_catalog(viz)
            return bool(catalog.scope_path_ids(MpcPathScope.RENDERED).size)
        except (AttributeError, TypeError, ValueError):
            return False

    def _identity(self):
        return (getattr(self.viz, "_mpc_presented_source_epoch", 0), self.service.step)

    def restart(self, now: float, duration: float) -> None:
        """Give each look one complete geometric trip, including its fading tail."""
        self.started, self.duration = now, duration
        self.sample(now)

    def sample(self, now: float) -> None:
        if self.identity != self._identity() or self.viz.animation_running:
            raise ValueError("The recorded frame changed during the MPC tour.")
        fraction = float(np.clip((now - self.started) / self.duration, 0, 1))
        sample = self.geometry.sample(
            fraction * self.geometry.extent * 1.13,
            interaction_mode="flash",
            trail_mode="long",
        )
        self.overlay.cinematic_pulses(sample)

    def close(self) -> None:
        self.overlay.clear()
