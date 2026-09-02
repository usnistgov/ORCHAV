"""Coverage and beamforming application helpers for the pygfx renderer.

Frame packets and scene helper modules provide renderer-neutral mesh/line payloads.
This mixin owns the pygfx named-geometry lifecycle for those payloads, including
backend-specific material tweaks and stale overlay removal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...scene.surface_payloads import (
    BeamformingSurface,
    build_coverage_isolines_payload,
    build_coverage_payloads,
)
from ...types.render_payloads import LineSetPayload, MaterialPayload, MeshPayload
from ..shared.overlay_reconciliation import (
    CoverageReconcilePlan,
    CoverageSnapshot,
    capture_owned_snapshot_state,
    complete_owned_snapshot_reconciliation,
    coverage_snapshot_from_packet,
    owned_snapshot_plan_succeeded,
    plan_coverage_reconciliation,
    plan_owned_snapshot_reconciliation,
)

if TYPE_CHECKING:
    from ...pipeline.core import FrameRenderPacket

logger = logging.getLogger(__name__)

_BEAMFORMING_MATERIAL = MaterialPayload(
    base_color=(1.0, 1.0, 1.0, 1.0),
    roughness=0.5,
    metallic=0.0,
    shader="unlit",
)


@dataclass(frozen=True, slots=True)
class _AppliedBeamformingSurface:
    """Beamforming payload known to have reached pygfx."""

    payload: MeshPayload


class PygfxSurfaceOverlayMixin:
    """Apply coverage and beamforming overlay payloads to pygfx."""

    def _pygfx_beamforming_surfaces(self) -> dict[str, _AppliedBeamformingSurface]:
        """Return the backend-private successfully applied surface snapshot."""
        applied = getattr(self, "_applied_beamforming_surfaces", None)
        if applied is None:
            applied = {}
            self._applied_beamforming_surfaces = applied
        return applied

    def _pygfx_beamforming_owned_names(self) -> set[str]:
        """Return names whose native lifecycle remains owned by this adapter."""
        owned = getattr(self, "_beamforming_owned_names", None)
        if owned is None:
            owned = set()
            self._beamforming_owned_names = owned
        return owned

    def _apply_beamforming_material(self, surface_id: str) -> bool:
        """Apply required beamforming material state with failure propagation."""
        if not self.set_named_material(surface_id, _BEAMFORMING_MATERIAL):
            return False

        obj = self._objects.get(surface_id)
        material = None if obj is None else getattr(obj, "material", None)
        if material is None:
            return False
        try:
            if hasattr(material, "color_mode"):
                material.color_mode = "vertex"
            if hasattr(material, "side"):
                material.side = "both"
            self.request_redraw()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "PygfxRenderer: beamforming material failed for '%s': %s",
                surface_id,
                exc,
            )
            return False
        return True

    def _ensure_beamforming_surface(
        self,
        surface: BeamformingSurface,
    ) -> tuple[_AppliedBeamformingSurface | None, bool]:
        """Apply one surface and return its successful snapshot and ownership."""
        try:
            geometry_applied = self.ensure_named_geometry(
                surface.id,
                surface.payload,
                visible=True,
            )
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            logger.debug(
                "PygfxRenderer: beamforming geometry failed for '%s': %s",
                surface.id,
                exc,
            )
            return None, self.has_named_geometry(surface.id)
        is_realized = self.has_named_geometry(surface.id)
        if not geometry_applied or not is_realized:
            return None, is_realized
        if not self._apply_beamforming_material(surface.id):
            return None, self.has_named_geometry(surface.id)

        return _AppliedBeamformingSurface(payload=surface.payload), True

    def _remove_stale_beamforming_surface(self, surface_id: str) -> bool:
        """Remove one stale surface without forgetting a failed removal."""
        if self.has_named_geometry(surface_id) and not self.remove_named_geometry(surface_id):
            return False
        return not self.has_named_geometry(surface_id)

    def _apply_beamforming(self, packet: "FrameRenderPacket") -> bool:
        """Idempotently synchronize final service-owned beamforming surfaces."""
        desired: dict[str, BeamformingSurface] = {
            surface.id: surface for surface in packet.beamforming_meshes
        }
        state = capture_owned_snapshot_state(
            self._pygfx_beamforming_surfaces(),
            self._pygfx_beamforming_owned_names(),
        )
        candidate_ids = set(desired) | set(state.owned) | set(state.applied)
        native_ids = {
            surface_id for surface_id in candidate_ids if self.has_named_geometry(surface_id)
        }
        plan = plan_owned_snapshot_reconciliation(
            {surface_id: surface.payload for surface_id, surface in desired.items()},
            state,
            native_ids=native_ids,
            matches=lambda payload, applied: applied.payload is payload,
        )

        successful_snapshots: dict[str, _AppliedBeamformingSurface] = {}
        realized_ids: set[str] = set()
        removed_ids: set[str] = set()
        for surface_id in plan.ensure_ids:
            snapshot, realized = self._ensure_beamforming_surface(desired[surface_id])
            if realized:
                realized_ids.add(surface_id)
            if snapshot is not None:
                successful_snapshots[surface_id] = snapshot
        for surface_id in plan.remove_ids:
            if self._remove_stale_beamforming_surface(surface_id):
                removed_ids.add(surface_id)

        next_state = complete_owned_snapshot_reconciliation(
            state,
            plan,
            successful_snapshots=successful_snapshots,
            realized_ids=realized_ids,
            removed_ids=removed_ids,
        )
        self._applied_beamforming_surfaces = dict(next_state.applied)
        self._beamforming_owned_names = set(next_state.owned)
        return owned_snapshot_plan_succeeded(
            plan,
            successful_ids=set(successful_snapshots),
            removed_ids=removed_ids,
        )

    def _coverage_native_presence_matches(
        self,
        *,
        has_mesh: bool,
        has_isolines: bool,
    ) -> bool:
        """Return whether pygfx currently owns the expected coverage objects."""
        return (
            self.has_named_geometry(self.COVERAGE_MESH_NAME) is has_mesh
            and self.has_named_geometry(self.COVERAGE_ISOLINES_NAME) is has_isolines
        )

    def _complete_coverage_plan(
        self,
        plan: CoverageReconcilePlan,
        *,
        succeeded: bool,
    ) -> bool:
        """Commit one plan only when native pygfx ownership is truthful."""
        next_applied = plan.complete(
            succeeded=succeeded,
            native_has_mesh=self.has_named_geometry(self.COVERAGE_MESH_NAME),
            native_has_isolines=self.has_named_geometry(self.COVERAGE_ISOLINES_NAME),
        )
        self._applied_coverage_state = next_applied
        self._last_coverage_signature = (
            next_applied.signature if next_applied is not None and next_applied.has_mesh else None
        )
        return bool(succeeded and next_applied == plan.desired)

    def _clear_coverage_geometry(self) -> bool:
        """Remove coverage objects, returning success only when both are absent."""
        for name in (self.COVERAGE_MESH_NAME, self.COVERAGE_ISOLINES_NAME):
            if self.has_named_geometry(name) and not self.remove_named_geometry(name):
                return False
        return self._coverage_native_presence_matches(has_mesh=False, has_isolines=False)

    def _apply_coverage_data(self, packet: "FrameRenderPacket") -> bool:
        """Synchronize coverage and commit state only after native success."""
        desired = coverage_snapshot_from_packet(packet)
        plan = plan_coverage_reconciliation(
            desired,
            getattr(self, "_applied_coverage_state", None),
            native_has_mesh=self.has_named_geometry(self.COVERAGE_MESH_NAME),
            native_has_isolines=self.has_named_geometry(self.COVERAGE_ISOLINES_NAME),
        )

        if plan.clear:
            return self._complete_coverage_plan(
                plan,
                succeeded=self._clear_coverage_geometry(),
            )
        if not desired.has_mesh:
            return self._complete_coverage_plan(plan, succeeded=True)

        if not plan.replace_mesh:
            if plan.update_opacity and not self.set_coverage_transparency(desired.opacity):
                return self._complete_coverage_plan(plan, succeeded=False)
            if plan.update_isolines:
                isoline_payload = build_coverage_isolines_payload(packet)
                if not self._apply_coverage_isolines_payload(isoline_payload):
                    return self._complete_coverage_plan(plan, succeeded=False)
            return self._complete_coverage_plan(plan, succeeded=True)

        coverage_payloads = build_coverage_payloads(packet)
        if coverage_payloads.mesh is None:
            hidden = CoverageSnapshot(
                signature=None,
                isoline_signature=None,
                opacity=desired.opacity,
                has_mesh=False,
                has_isolines=False,
            )
            hidden_plan = plan_coverage_reconciliation(
                hidden,
                getattr(self, "_applied_coverage_state", None),
                native_has_mesh=self.has_named_geometry(self.COVERAGE_MESH_NAME),
                native_has_isolines=self.has_named_geometry(self.COVERAGE_ISOLINES_NAME),
            )
            return self._complete_coverage_plan(
                hidden_plan,
                succeeded=self._clear_coverage_geometry(),
            )

        if not self.ensure_named_geometry(
            self.COVERAGE_MESH_NAME,
            coverage_payloads.mesh,
            visible=True,
        ):
            return self._complete_coverage_plan(plan, succeeded=False)
        if not self.has_named_geometry(self.COVERAGE_MESH_NAME):
            return self._complete_coverage_plan(plan, succeeded=False)

        if not self._apply_coverage_material_state(
            desired.opacity,
            request_redraw=False,
        ):
            return self._complete_coverage_plan(plan, succeeded=False)
        if not self._apply_coverage_isolines_payload(coverage_payloads.isolines):
            return self._complete_coverage_plan(plan, succeeded=False)
        return self._complete_coverage_plan(plan, succeeded=True)

    def _apply_coverage_isolines_payload(self, payload: LineSetPayload | None) -> bool:
        """Synchronize optional isolines and report native success."""
        if payload is None:
            if self.has_named_geometry(self.COVERAGE_ISOLINES_NAME):
                if not self.remove_named_geometry(self.COVERAGE_ISOLINES_NAME):
                    return False
            return not self.has_named_geometry(self.COVERAGE_ISOLINES_NAME)

        material = MaterialPayload(
            base_color=(0.05, 0.05, 0.05, 1.0),
            shader="unlit",
            line_width=max(3.0, float(getattr(self, "_line_width", 2.0))),
        )
        if not self.ensure_named_geometry(
            self.COVERAGE_ISOLINES_NAME,
            payload,
            material=material,
            visible=True,
        ):
            return False
        return self.has_named_geometry(self.COVERAGE_ISOLINES_NAME)

    def _apply_coverage_data_diff(
        self,
        old_packet: "FrameRenderPacket",
        new_packet: "FrameRenderPacket",
    ) -> bool:
        """Synchronize against backend-applied state, not the prior desired packet."""
        del old_packet
        return self._apply_coverage_data(new_packet)
