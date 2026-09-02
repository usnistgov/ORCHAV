"""Coverage and beamforming overlay helpers for the Open3D backend.

Coverage maps arrive through the frame packet as mesh/isoline buffers. Beamforming
meshes are uploaded directly as backend-owned named Open3D geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...backends.open3d_payload_codec import lines_payload_to_o3d, mesh_payload_to_o3d
from ...scene.surface_payloads import (
    BeamformingSurface,
    build_coverage_isolines_payload,
    build_coverage_payloads,
)
from ...types.render_payloads import MeshPayload
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

logger = get_logger("orchav.renderer_open3d")


@dataclass(frozen=True, slots=True)
class _AppliedBeamformingSurface:
    """Beamforming payload known to have reached Open3D."""

    payload: MeshPayload
    geometry: o3d.geometry.TriangleMesh


class Open3DSurfaceOverlayMixin:
    """Apply coverage and beamforming payloads to Open3D geometry."""

    def _coverage_native_presence_matches(
        self,
        *,
        has_mesh: bool,
        has_isolines: bool,
    ) -> bool:
        """Return whether Open3D currently owns the expected coverage objects."""
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
        """Commit one plan only when native Open3D ownership is truthful."""
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
            if self.has_named_geometry(name) and not self._remove_geometry(name):
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

        coverage_mesh = mesh_payload_to_o3d(coverage_payloads.mesh)

        mat = rendering.MaterialRecord()
        if desired.opacity < 1.0:
            mat.shader = "defaultLitTransparency"
        else:
            mat.shader = "defaultUnlit"
        # White base color preserves the per-vertex heatmap colors.
        mat.base_color = [1.0, 1.0, 1.0, desired.opacity]

        if not self._add_or_update_geometry(self.COVERAGE_MESH_NAME, coverage_mesh, mat):
            return self._complete_coverage_plan(plan, succeeded=False)
        if not self.has_named_geometry(self.COVERAGE_MESH_NAME):
            return self._complete_coverage_plan(plan, succeeded=False)
        if not self._apply_coverage_isolines_payload(coverage_payloads.isolines):
            return self._complete_coverage_plan(plan, succeeded=False)
        self.visualizer.coverage_mesh = coverage_mesh
        return self._complete_coverage_plan(plan, succeeded=True)

    def _apply_coverage_isolines_payload(self, payload) -> bool:
        """Synchronize coverage isolines and report native success."""
        if payload is None:
            if self.has_named_geometry(self.COVERAGE_ISOLINES_NAME):
                if not self._remove_geometry(self.COVERAGE_ISOLINES_NAME):
                    return False
            return not self.has_named_geometry(self.COVERAGE_ISOLINES_NAME)

        isolines = lines_payload_to_o3d(payload)
        if payload.colors is None:
            isolines.colors = o3d.utility.Vector3dVector(
                np.tile(
                    np.array([[0.05, 0.05, 0.05]], dtype=np.float64),
                    (payload.lines.shape[0], 1),
                )
            )

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = max(3.0, float(getattr(self, "_line_width", 2.0)))
        if not self._add_or_update_geometry(self.COVERAGE_ISOLINES_NAME, isolines, mat):
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

    def _open3d_beamforming_surfaces(self) -> dict[str, _AppliedBeamformingSurface]:
        """Return the backend-private successfully applied surface snapshot."""
        applied = getattr(self, "_applied_beamforming_surfaces", None)
        if applied is None:
            applied = {}
            self._applied_beamforming_surfaces = applied
        return applied

    def _open3d_beamforming_owned_names(self) -> set[str]:
        """Return names whose native lifecycle remains owned by this adapter."""
        owned = getattr(self, "_beamforming_owned_names", None)
        if owned is None:
            owned = set()
            self._beamforming_owned_names = owned
        return owned

    @staticmethod
    def _beamforming_material() -> rendering.MaterialRecord:
        """Return the unlit white material that preserves vertex colors."""
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        return material

    def _ensure_beamforming_surface(
        self,
        surface: BeamformingSurface,
    ) -> tuple[_AppliedBeamformingSurface | None, bool]:
        """Apply one surface and return its successful snapshot and ownership."""
        try:
            geometry = mesh_payload_to_o3d(surface.payload)
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            logger.debug(
                "Open3DRenderer: failed to convert beamforming surface '%s': %s",
                surface.id,
                exc,
            )
            return None, self.has_named_geometry(surface.id)

        geometry_applied = self._add_or_update_geometry(
            surface.id,
            geometry,
            self._beamforming_material(),
        )
        is_realized = self.has_named_geometry(surface.id)
        if not geometry_applied or not is_realized:
            return None, is_realized

        return (
            _AppliedBeamformingSurface(
                payload=surface.payload,
                geometry=geometry,
            ),
            True,
        )

    def _remove_stale_beamforming_surface(self, surface_id: str) -> bool:
        """Remove one stale surface without forgetting a failed removal."""
        if self.has_named_geometry(surface_id) and not self._remove_geometry(surface_id):
            return False
        return not self.has_named_geometry(surface_id)

    def _apply_beamforming(self, packet: "FrameRenderPacket") -> bool:
        """Idempotently synchronize final service-owned beamforming surfaces."""
        desired: dict[str, BeamformingSurface] = {
            surface.id: surface for surface in packet.beamforming_meshes
        }
        state = capture_owned_snapshot_state(
            self._open3d_beamforming_surfaces(),
            self._open3d_beamforming_owned_names(),
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
