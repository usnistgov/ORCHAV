"""Trajectory application helpers for the pygfx renderer.

The shared trajectory payload builder decides which line and point payloads are
visible for TX, RX, and target trajectories. This mixin applies those payloads
through stable pygfx names and keeps backend-only size controls on the native
materials.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...scene.trajectory_payloads import (
    NamedTrajectoryPayload,
    TrajectoryNaming,
    build_trajectory_payloads,
    sanitize_trajectory_name,
)

logger = logging.getLogger(__name__)


class PygfxTrajectoryMixin:
    """Apply backend-neutral trajectory payloads to pygfx named geometry."""

    def apply_trajectory(
        self,
        kind: str,
        trajectory_data: dict,
        color_mode: str = "node_color",
        scalar_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """Apply one trajectory family and remove names made stale by the batch."""
        if kind not in {"tx", "rx", "target"}:
            logger.debug("PygfxRenderer: unknown trajectory kind '%s'", kind)
            return

        batch = build_trajectory_payloads(
            kind=kind,
            trajectory_data=trajectory_data,
            visualizer=self.visualizer,
            naming=self._trajectory_naming(),
            color_mode=color_mode,
            scalar_range=scalar_range,
            existing_names=tuple(self._name_to_handle),
        )
        for name in batch.stale_names:
            self.remove_named_geometry(name)
        for payload in batch.payloads:
            self._apply_trajectory_payload(payload)
        self._trajectory_hud_color_mode = str(color_mode)
        self._trajectory_hud_scalar_range = scalar_range
        if batch.payloads:
            self._visible_trajectory_kinds.add(kind)
        else:
            self._visible_trajectory_kinds.discard(kind)
        update_hud = getattr(self, "_update_trajectory_hud_overlay", None)
        if callable(update_hud):
            update_hud()
        self.request_redraw()

    def remove_trajectory(self, kind: str) -> None:
        """Remove all named trajectory geometry for one family."""
        if kind == "target":
            names = [
                name
                for name in list(self._name_to_handle)
                if name.startswith(self.TRAJECTORY_TARGET_LINES_PREFIX)
                or name.startswith(self.TRAJECTORY_TARGET_POINTS_PREFIX)
            ]
            for name in names:
                self.remove_named_geometry(name)
            self._visible_trajectory_kinds.discard(kind)
            update_hud = getattr(self, "_update_trajectory_hud_overlay", None)
            if callable(update_hud):
                update_hud()
            self.request_redraw()
            return

        if kind == "tx":
            names = (self.TRAJECTORY_TX_LINES_NAME, self.TRAJECTORY_TX_POINTS_NAME)
        elif kind == "rx":
            names = (self.TRAJECTORY_RX_LINES_NAME, self.TRAJECTORY_RX_POINTS_NAME)
        else:
            return
        for name in names:
            self.remove_named_geometry(name)
        self._visible_trajectory_kinds.discard(kind)
        update_hud = getattr(self, "_update_trajectory_hud_overlay", None)
        if callable(update_hud):
            update_hud()
        self.request_redraw()

    def _trajectory_naming(self) -> TrajectoryNaming:
        """Return the stable names/prefixes used by shared trajectory builders."""
        return TrajectoryNaming(
            tx_lines=self.TRAJECTORY_TX_LINES_NAME,
            tx_points=self.TRAJECTORY_TX_POINTS_NAME,
            rx_lines=self.TRAJECTORY_RX_LINES_NAME,
            rx_points=self.TRAJECTORY_RX_POINTS_NAME,
            target_lines_prefix=self.TRAJECTORY_TARGET_LINES_PREFIX,
            target_points_prefix=self.TRAJECTORY_TARGET_POINTS_PREFIX,
        )

    def _apply_trajectory_payload(self, payload: NamedTrajectoryPayload) -> None:
        """Upload one named trajectory payload and apply backend size controls."""
        if payload.line_payload is not None:
            self.ensure_named_geometry(payload.lines_name, payload.line_payload)
            line_obj = self._objects.get(payload.lines_name)
            if line_obj is not None:
                line_mat = getattr(line_obj, "material", None)
                if line_mat is not None and hasattr(line_mat, "thickness"):
                    try:
                        line_mat.thickness = float(self.trajectory_line_width)
                    except Exception:
                        pass
        else:
            self.remove_named_geometry(payload.lines_name)

        self.ensure_named_geometry(payload.points_name, payload.point_payload)
        point_obj = self._objects.get(payload.points_name)
        if point_obj is not None:
            point_mat = getattr(point_obj, "material", None)
            if point_mat is not None and hasattr(point_mat, "size"):
                try:
                    point_mat.size = float(self.trajectory_point_size)
                except Exception:
                    pass

    def _trajectory_line_names(self) -> list[str]:
        """Return currently registered named geometries that encode trajectory lines."""
        return [
            name
            for name in self._name_to_handle
            if name in (self.TRAJECTORY_TX_LINES_NAME, self.TRAJECTORY_RX_LINES_NAME)
            or name.startswith(self.TRAJECTORY_TARGET_LINES_PREFIX)
        ]

    def _trajectory_point_names(self) -> list[str]:
        """Return currently registered named geometries that encode trajectory points."""
        return [
            name
            for name in self._name_to_handle
            if name in (self.TRAJECTORY_TX_POINTS_NAME, self.TRAJECTORY_RX_POINTS_NAME)
            or name.startswith(self.TRAJECTORY_TARGET_POINTS_PREFIX)
        ]

    def _is_trajectory_line_name(self, name: str) -> bool:
        """Report whether a named geometry belongs to the trajectory line namespace."""
        return name in (
            self.TRAJECTORY_TX_LINES_NAME,
            self.TRAJECTORY_RX_LINES_NAME,
        ) or name.startswith(self.TRAJECTORY_TARGET_LINES_PREFIX)

    def _is_trajectory_point_name(self, name: str) -> bool:
        """Report whether a named geometry belongs to the trajectory point namespace."""
        return name in (
            self.TRAJECTORY_TX_POINTS_NAME,
            self.TRAJECTORY_RX_POINTS_NAME,
        ) or name.startswith(self.TRAJECTORY_TARGET_POINTS_PREFIX)

    @staticmethod
    def _sanitize_trajectory_name(name: str) -> str:
        """Normalize a scenario trajectory label for use in pygfx object names."""
        return sanitize_trajectory_name(name)
