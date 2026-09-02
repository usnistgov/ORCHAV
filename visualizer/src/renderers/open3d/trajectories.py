"""Trajectory payload application helpers for the Open3D backend.

Trajectory construction is renderer-neutral and lives in
``scene.trajectory_payloads``. This mixin only maps the named line/point payloads
to Open3D geometry and removes stale target-specific geometry names.
"""

from __future__ import annotations

from typing import Optional, Tuple

import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...backends.open3d_payload_codec import lines_payload_to_o3d, points_payload_to_o3d
from ...scene.trajectory_payloads import (
    NamedTrajectoryPayload,
    TrajectoryNaming,
    build_trajectory_payloads,
)

logger = get_logger("orchav.renderer_open3d")


class Open3DTrajectoryMixin:
    """Apply trajectory line and point payloads to Open3D geometries."""

    def apply_trajectory(
        self,
        kind: str,
        trajectory_data: dict,
        color_mode: str = "node_color",
        scalar_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Build and display trajectory geometry for TX, RX, or target nodes."""
        if kind not in {"tx", "rx", "target"}:
            logger.debug("Open3DRenderer: unknown trajectory kind '%s'", kind)
            return

        batch = build_trajectory_payloads(
            kind=kind,
            trajectory_data=trajectory_data,
            visualizer=self.visualizer,
            naming=self._trajectory_naming(),
            color_mode=color_mode,
            scalar_range=scalar_range,
            existing_names=tuple(self._geometry_names),
        )
        with self.batch_updates():
            for name in batch.stale_names:
                self._remove_geometry(name)
            for payload in batch.payloads:
                self._apply_trajectory_payload(payload)
            self._post_redraw()
        logger.debug(
            "Applied %s trajectory (%s): %d payloads",
            kind,
            color_mode,
            len(batch.payloads),
        )

    def _trajectory_naming(self) -> TrajectoryNaming:
        """Return Open3D geometry names used by the shared payload builder."""
        return TrajectoryNaming(
            tx_lines=self.TRAJECTORY_TX_LINES_NAME,
            tx_points=self.TRAJECTORY_TX_POINTS_NAME,
            rx_lines=self.TRAJECTORY_RX_LINES_NAME,
            rx_points=self.TRAJECTORY_RX_POINTS_NAME,
            target_lines_prefix=self.TRAJECTORY_TARGET_LINES_PREFIX,
            target_points_prefix=self.TRAJECTORY_TARGET_POINTS_PREFIX,
        )

    def _apply_trajectory_payload(self, payload: NamedTrajectoryPayload) -> None:
        """Upload one trajectory's optional line set and required point cloud."""
        if payload.line_payload is not None:
            line_mat = rendering.MaterialRecord()
            line_mat.shader = "unlitLine"
            line_mat.line_width = self.trajectory_line_width
            self._add_or_update_geometry(
                payload.lines_name,
                lines_payload_to_o3d(payload.line_payload),
                line_mat,
            )
        else:
            self._remove_geometry(payload.lines_name)

        point_mat = rendering.MaterialRecord()
        point_mat.shader = "defaultUnlit"
        point_mat.point_size = self.trajectory_point_size
        self._add_or_update_geometry(
            payload.points_name,
            points_payload_to_o3d(payload.point_payload),
            point_mat,
        )

    def remove_trajectory(self, kind: str) -> None:
        """Remove 3D trajectory geometry for TX, RX, or target nodes."""
        if kind == "target":
            self._remove_target_trajectories()
            return

        if kind == "tx":
            names = (self.TRAJECTORY_TX_LINES_NAME, self.TRAJECTORY_TX_POINTS_NAME)
        elif kind == "rx":
            names = (self.TRAJECTORY_RX_LINES_NAME, self.TRAJECTORY_RX_POINTS_NAME)
        else:
            return

        with self.batch_updates():
            for name in names:
                self._remove_geometry(name)
            self._post_redraw()

        logger.debug("Removed %s trajectory geometry", kind)

    def _remove_target_trajectories(self) -> None:
        """Remove all target trajectory geometries by target-name prefixes."""
        names_to_remove = [
            name
            for name in self._geometry_names
            if name.startswith(self.TRAJECTORY_TARGET_LINES_PREFIX)
            or name.startswith(self.TRAJECTORY_TARGET_POINTS_PREFIX)
        ]
        if names_to_remove:
            with self.batch_updates():
                for name in names_to_remove:
                    self._remove_geometry(name)
                self._post_redraw()
            logger.debug("Removed %d target trajectory geometries", len(names_to_remove))
