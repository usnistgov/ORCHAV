"""MPC line and bounce-point application for the Open3D backend.

Frame packet MPC arrays are renderer-neutral NumPy buffers. This mixin owns the
Open3D ``LineSet`` and ``PointCloud`` mutation policy, including the cached
bounce-point arrays used when the user toggles bounce visibility without a full
MPC data rebuild.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

from shared.logging import get_logger

if TYPE_CHECKING:
    from ...pipeline.core import FrameRenderPacket

logger = get_logger("orchav.renderer_open3d")

# Empty Open3D-compatible buffers shared with renderer clear/reset paths.
EMPTY_POINTS_3D = np.empty((0, 3), dtype=np.float64)
EMPTY_LINES_2D = np.empty((0, 2), dtype=np.int32)
EMPTY_COLORS_3D = np.empty((0, 3), dtype=np.float64)


def _open3d_writeable_array(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return a C-contiguous writeable array for Open3D vector constructors."""
    return np.require(array, dtype=dtype, requirements=("C", "W"))


def _vector3d(array: np.ndarray) -> o3d.utility.Vector3dVector:
    """Build an Open3D 3D vector from immutable frame-packet arrays."""
    return o3d.utility.Vector3dVector(_open3d_writeable_array(array, np.float64))


def _vector2i(array: np.ndarray) -> o3d.utility.Vector2iVector:
    """Build an Open3D 2D index vector from immutable frame-packet arrays."""
    return o3d.utility.Vector2iVector(_open3d_writeable_array(array, np.int32))


class Open3DMpcMixin:
    """Apply frame-packet MPC arrays to Open3D line and point geometries."""

    @staticmethod
    def _mpc_line_source(packet: "FrameRenderPacket") -> object:
        """Return the immutable source token for MPC path geometry."""
        return getattr(
            packet,
            "mpc_line_revision",
            (
                id(packet.mpc_points),
                id(packet.mpc_lines),
                id(packet.mpc_colors),
            ),
        )

    @staticmethod
    def _mpc_bounce_source(packet: "FrameRenderPacket") -> object:
        """Return the immutable source token for physical bounce geometry."""
        return getattr(
            packet,
            "mpc_point_revision",
            (
                id(packet.mpc_bounce_points),
                id(packet.mpc_bounce_colors),
            ),
        )

    def _apply_mpc_lines(self, packet: "FrameRenderPacket") -> bool:
        """Apply or remove MPC path segments and report native acceptance."""
        visibility = packet.mpc_visibility
        if (
            not visibility.effective_paths
            or len(packet.mpc_points) == 0
            or len(packet.mpc_lines) == 0
        ):
            if self.MPC_LINES_NAME not in self._geometry_names:
                return True
            return self._remove_geometry(self.MPC_LINES_NAME)

        self.mpc_lineset.points = _vector3d(packet.mpc_points)
        self.mpc_lineset.lines = _vector2i(packet.mpc_lines)
        self.mpc_lineset.colors = _vector3d(packet.mpc_colors)
        return self._add_or_update_geometry(
            self.MPC_LINES_NAME,
            self.mpc_lineset,
            self._line_material,
        )

    def _apply_mpc_bounce_points(self, packet: "FrameRenderPacket") -> bool:
        """Apply the physical-bounce payload and report native acceptance."""
        bounce_points = packet.mpc_bounce_points
        visibility = packet.mpc_visibility
        if (
            not visibility.effective_bounce_points
            or bounce_points is None
            or len(bounce_points) == 0
        ):
            if self.MPC_POINTS_NAME not in self._geometry_names:
                return True
            return self._remove_geometry(self.MPC_POINTS_NAME)

        self.mpc_pcd.points = _vector3d(bounce_points)
        bounce_colors = packet.mpc_bounce_colors
        if bounce_colors is not None and len(bounce_colors) == len(bounce_points):
            self.mpc_pcd.colors = _vector3d(bounce_colors)
        else:
            self.mpc_pcd.colors = _vector3d(EMPTY_COLORS_3D)
        return self._add_or_update_geometry(
            self.MPC_POINTS_NAME,
            self.mpc_pcd,
            self._point_material,
        )

    def _apply_mpc_data(self, packet: "FrameRenderPacket") -> bool:
        """Apply independent MPC domains and report complete acceptance."""
        logger.debug(
            "Open3DRenderer: Applying MPC data with %d points, %d lines",
            len(packet.mpc_points),
            len(packet.mpc_lines),
        )

        lines_succeeded = self._apply_mpc_lines(packet)
        bounce_points_succeeded = self._apply_mpc_bounce_points(packet)
        return bool(lines_succeeded and bounce_points_succeeded)

    def _apply_mpc_data_diff(
        self,
        old_packet: "FrameRenderPacket",
        new_packet: "FrameRenderPacket",
    ) -> bool:
        """Update changed MPC domains and report complete native acceptance."""
        old_visibility = old_packet.mpc_visibility
        new_visibility = new_packet.mpc_visibility

        lines_changed = (
            old_visibility.effective_paths != new_visibility.effective_paths
            or self._mpc_line_source(old_packet) != self._mpc_line_source(new_packet)
            or (new_visibility.effective_paths and self.MPC_LINES_NAME not in self._geometry_names)
        )
        lines_succeeded = True
        if lines_changed:
            lines_succeeded = self._apply_mpc_lines(new_packet)

        bounce_points_changed = (
            old_visibility.effective_bounce_points != new_visibility.effective_bounce_points
            or self._mpc_bounce_source(old_packet) != self._mpc_bounce_source(new_packet)
            or (
                new_visibility.effective_bounce_points
                and self.MPC_POINTS_NAME not in self._geometry_names
            )
        )
        bounce_points_succeeded = True
        if bounce_points_changed:
            bounce_points_succeeded = self._apply_mpc_bounce_points(new_packet)

        return bool(lines_succeeded and bounce_points_succeeded)
