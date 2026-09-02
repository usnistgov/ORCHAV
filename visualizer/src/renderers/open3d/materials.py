"""Material selection and reapplication helpers for the Open3D backend.

Open3D exposes global line-width and point-size knobs that can reset
per-geometry material state. This mixin centralizes the material choices and
the repair steps needed after those global settings change.
"""

from __future__ import annotations

from typing import Any

import open3d as o3d
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...types.render_payloads import MaterialPayload


def _multiplied_rgba(material: MaterialPayload) -> list[float]:
    """Return the native RGBA factor after applying transient tint."""
    return [
        float(material.base_color[index]) * float(material.color_multiplier[index])
        for index in range(3)
    ] + [float(material.base_color[3])]


logger = get_logger("orchav.renderer_open3d")


class Open3DMaterialMixin:
    """Own Open3D material defaults, named updates, and restore policies."""

    def _default_material_for_geometry(
        self, geometry: o3d.geometry.Geometry, *, is_edge: bool = False
    ) -> rendering.MaterialRecord:
        """Return the renderer's default material for a geometry type."""
        if isinstance(geometry, o3d.geometry.LineSet):
            return self._edge_material if is_edge else self._line_material
        if isinstance(geometry, o3d.geometry.PointCloud):
            return self._point_material
        return self._mesh_material

    def _material_payload_for_geometry(
        self,
        geometry: o3d.geometry.Geometry,
        material: MaterialPayload | dict[str, Any] | None,
        *,
        is_edge: bool = False,
    ) -> rendering.MaterialRecord | None:
        """Map neutral line/point materials to Open3D unlit MaterialRecords.

        Mesh PBR payloads are handled by ``set_named_material()`` and the PBR
        methods in ``lighting.py``; this helper is only for geometry types where
        Open3D uses line-width or point-size fields instead of mesh PBR factors.
        """
        if material is None or not isinstance(material, MaterialPayload):
            return None
        if isinstance(geometry, o3d.geometry.LineSet):
            line_material = rendering.MaterialRecord()
            line_material.shader = "unlitLine"
            fallback_width = self._edge_line_width if is_edge else self._line_width
            line_material.line_width = float(material.line_width or fallback_width)
            try:
                line_material.base_color = _multiplied_rgba(material)
            except (AttributeError, TypeError, ValueError):
                pass
            return line_material
        if isinstance(geometry, o3d.geometry.PointCloud):
            point_material = rendering.MaterialRecord()
            point_material.shader = "defaultUnlit"
            point_material.point_size = float(material.point_size or self._point_size)
            try:
                point_material.base_color = _multiplied_rgba(material)
            except (AttributeError, TypeError, ValueError):
                pass
            return point_material
        return None

    def set_named_material(self, name: str, pbr_state: MaterialPayload | dict[str, Any]) -> bool:
        """Apply material properties for a stable named geometry.

        Backend-local object synchronization calls this after state changes.
        Mesh reapply must forward texture/detail-map fields so a textured
        material is not degraded to flat color.
        """
        if name not in self._geometry_names:
            return False
        kind = getattr(self, "_geometry_types", {}).get(name)
        if isinstance(pbr_state, MaterialPayload) and kind in {"lines", "points"}:
            material = rendering.MaterialRecord()
            if kind == "lines":
                material.shader = "unlitLine"
                fallback_width = (
                    self._edge_line_width if name in self._edge_geometry_names else self._line_width
                )
                material.line_width = float(pbr_state.line_width or fallback_width)
            else:
                material.shader = "defaultUnlit"
                material.point_size = float(pbr_state.point_size or self._point_size)
            try:
                material.base_color = _multiplied_rgba(pbr_state)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                self._o3d_vis.modify_geometry_material(name, material)
                self._post_redraw()
                if not self._frame_update_in_progress:
                    self._request_visibility_settle_redraw(f"line or point material '{name}'")
                return True
            except (RuntimeError, AttributeError) as exc:
                logger.debug("Open3DRenderer: set_named_material failed for '%s': %s", name, exc)
                return False
        if isinstance(pbr_state, MaterialPayload):
            from ...materials.catalog import material_payload_to_pbr_kwargs

            material_kwargs = material_payload_to_pbr_kwargs(pbr_state)
        else:
            color = pbr_state.get("color", [1.0, 1.0, 1.0])
            props = dict(pbr_state)
            from ...materials.catalog import pbr_props_to_kwargs

            material_kwargs = pbr_props_to_kwargs(color, props)

        return self.modify_geometry_material_pbr(
            name=name,
            **material_kwargs,
        )

    def set_line_width(self, width: float) -> bool:
        """Set MPC path line width in pixels and restore affected materials."""
        self._line_width = width
        self._line_material.line_width = width

        if self._o3d_vis is not None:
            self._o3d_vis.line_width = int(width)
            if self.MPC_LINES_NAME in self._geometry_names:
                self._add_or_update_geometry(
                    self.MPC_LINES_NAME, self.mpc_lineset, self._line_material
                )
            # Global property setters can reset per-geometry materials;
            # restore backend-owned material snapshots and line widths.
            self._restore_cached_native_materials()
            self._restore_per_geometry_line_widths()
            # Restore both global properties last — O3D operations above
            # can reset them as a side effect.
            self._o3d_vis.line_width = int(self._line_width)
            self._o3d_vis.point_size = int(self._point_size)
            self._post_redraw()
            logger.info("Open3DRenderer: Set MPC line width to %s", width)
            return True
        return False

    def set_edge_line_width(self, width: float) -> bool:
        """Set scene-edge wireframe line width in pixels."""
        self._edge_line_width = width
        self._edge_material.line_width = width

        if self._o3d_vis is not None:
            self._reapply_edge_materials()
            self._post_redraw()
            logger.info("Open3DRenderer: Set edge line width to %s", width)
            return True
        return False

    def set_trajectory_line_width(self, width: float) -> bool:
        """Set trajectory line width in pixels for active trajectory geometry."""
        self.trajectory_line_width = width

        if self._o3d_vis is not None:
            self._reapply_trajectory_line_materials()
            self._post_redraw()
            logger.info("Open3DRenderer: Set trajectory line width to %s", width)
            return True
        return False

    def set_trajectory_point_size(self, size: float) -> bool:
        """Set point size for trajectory points and reapply active materials."""
        self.trajectory_point_size = float(size)
        if self._o3d_vis is None:
            return False
        self._reapply_trajectory_point_materials()
        self._post_redraw()
        logger.info("Open3DRenderer: Set trajectory point size to %s", size)
        return True

    def set_point_size(self, size: float) -> bool:
        """Set bounce-point size in pixels and restore affected materials."""
        self._point_size = size
        self._point_material.point_size = size

        if self._o3d_vis is not None:
            self._o3d_vis.point_size = int(size)
            if self.MPC_POINTS_NAME in self._geometry_names:
                self._add_or_update_geometry(
                    self.MPC_POINTS_NAME, self.mpc_pcd, self._point_material
                )
            # Global property setters can reset per-geometry materials;
            # restore backend-owned material snapshots and line widths.
            self._restore_cached_native_materials()
            self._restore_per_geometry_line_widths()
            # Restore both global properties last — O3D operations above
            # can reset them as a side effect.
            self._o3d_vis.point_size = int(self._point_size)
            self._o3d_vis.line_width = int(self._line_width)
            self._post_redraw()
            logger.info(f"Open3DRenderer: Set point size to {size}")
            return True
        return False

    def _restore_cached_native_materials(self) -> None:
        """Reapply successful native material snapshots after a global change.

        ``O3DVisualizer`` may reset per-geometry materials when its global
        point-size or line-width property changes. The renderer already owns
        the last native material that successfully reached each geometry, so
        restoration stays inside the backend and does not replay application
        appearance commands.
        """
        if self._o3d_vis is None:
            return
        for name, material in list(self._pbr_materials.items()):
            if name not in self._geometry_names:
                continue
            try:
                self._o3d_vis.modify_geometry_material(name, material)
            except (RuntimeError, AttributeError) as exc:
                logger.debug("Failed to restore cached material for '%s': %s", name, exc)

    def _restore_per_geometry_line_widths(self) -> None:
        """Re-apply per-geometry line width materials after a global property change.

        When ``O3DVisualizer.line_width`` is set globally, it clobbers
        per-geometry ``MaterialRecord.line_width`` on edge and trajectory
        LineSets. This method re-applies the correct materials.
        """
        self._reapply_edge_materials()
        self._reapply_trajectory_line_materials()

    def _reapply_edge_materials(self) -> None:
        """Re-apply the edge material to all tracked edge geometries."""
        if self._o3d_vis is None:
            return
        scene = self._o3d_vis.scene
        if scene is None:
            return
        for name in list(self._edge_geometry_names):
            if name in self._geometry_names:
                try:
                    scene.modify_geometry_material(name, self._edge_material)
                except (RuntimeError, AttributeError) as exc:
                    logger.debug("Failed to reapply edge material for '%s': %s", name, exc)

    def _reapply_trajectory_line_materials(self) -> None:
        """Re-apply trajectory line width to all trajectory line geometries."""
        if self._o3d_vis is None:
            return
        scene = self._o3d_vis.scene
        if scene is None:
            return
        traj_mat = rendering.MaterialRecord()
        traj_mat.shader = "unlitLine"
        traj_mat.line_width = self.trajectory_line_width
        traj_prefixes = (
            self.TRAJECTORY_TX_LINES_NAME,
            self.TRAJECTORY_RX_LINES_NAME,
            self.TRAJECTORY_TARGET_LINES_PREFIX,
        )
        for name in list(self._geometry_names):
            if any(name == prefix or name.startswith(prefix) for prefix in traj_prefixes):
                try:
                    scene.modify_geometry_material(name, traj_mat)
                except (RuntimeError, AttributeError) as exc:
                    logger.debug("Failed to reapply trajectory material for '%s': %s", name, exc)

    def _reapply_trajectory_point_materials(self) -> None:
        """Re-apply trajectory point-size material to all trajectory point geometries."""
        if self._o3d_vis is None:
            return
        scene = self._o3d_vis.scene
        if scene is None:
            return
        pt_mat = rendering.MaterialRecord()
        pt_mat.shader = "defaultUnlit"
        pt_mat.point_size = self.trajectory_point_size
        traj_prefixes = (
            self.TRAJECTORY_TX_POINTS_NAME,
            self.TRAJECTORY_RX_POINTS_NAME,
            self.TRAJECTORY_TARGET_POINTS_PREFIX,
        )
        for name in list(self._geometry_names):
            if any(name == prefix or name.startswith(prefix) for prefix in traj_prefixes):
                try:
                    scene.modify_geometry_material(name, pt_mat)
                except (RuntimeError, AttributeError) as exc:
                    logger.debug(
                        "Failed to reapply trajectory point material for '%s': %s", name, exc
                    )
