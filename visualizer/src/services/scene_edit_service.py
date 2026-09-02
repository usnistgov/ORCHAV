"""Scene edit operations for XML-backed meshes and live node updates."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QTimer

from shared.logging import get_logger

from ..model import RenderObjectState
from ..scene.geometry_helpers import (
    ensure_scene_mesh_render_state,
    require_target_mesh_render_state,
)
from ..services.base import BaseService
from ..services.cache_service import CacheInvalidationScope
from ..types.render_payloads import MeshPayload
from ..utils import geometry as viz_geometry

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.scene_edit_service")


class SceneEditService(BaseService):
    """Owns mesh editing, XML synchronization, and node edit RPC calls."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind scene editing to the visualizer scene, cache, and frame source."""
        super().__init__()
        self.visualizer = visualizer

    # Scene mesh editing helpers
    def set_object_position(self, entry: Dict[str, Any], new_position: List[float]) -> None:
        """Adjust the mesh position and corresponding XML transform."""
        target_entry = self._resolve_scene_entry(entry)
        if target_entry is None:
            logger.warning("Cannot update position: entry not found")
            return

        mesh = target_entry.get("mesh")
        xml_shape = target_entry.get("xml_shape")
        entry_name = target_entry.get("name", "")
        if mesh is None:
            logger.warning("Cannot update position: no mesh for '%s'", entry_name)
            return

        try:
            original_center = self._ensure_original_center(target_entry, mesh)
            state = self._ensure_transform_state(target_entry)
            desired_center = np.asarray(new_position, dtype=float)

            rotation_matrix = viz_geometry.build_rotation_matrix_from_angles(
                state.get("rotation", [0.0, 0.0, 0.0])
            )
            position_after_scale_rotation = rotation_matrix @ (
                state.get("scale", 1.0) * original_center
            )
            translate_vec = desired_center - position_after_scale_rotation
            state["translate"] = translate_vec.tolist()

            original_vertices = self._ensure_original_vertices(target_entry)
            mesh, transformed_center, final_center = self._apply_transform_to_entry_mesh(
                target_entry, mesh, original_vertices, original_center, state
            )

            if transformed_center is not None:
                target_entry["position_after_scale_rotation"] = transformed_center
            if final_center is not None:
                target_entry["current_center"] = final_center

            self._refresh_mesh_in_window(target_entry, mesh)

            if entry is not target_entry:
                entry["transform_state"] = state
                if transformed_center is not None:
                    entry["position_after_scale_rotation"] = transformed_center

            if xml_shape is not None:
                self._update_xml_transform_from_state(xml_shape, state)
                logger.debug("Updated position for '%s' -> %s", entry_name, new_position)
                self.push_scene_updates_online(f"Position update '{entry_name}'")
            else:
                logger.warning(
                    "No XML shape for '%s' - position updated visually but not serialized",
                    entry_name,
                )
        except (ValueError, TypeError, RuntimeError):
            logger.exception("Failed to update object position")

    def set_object_rotation(self, entry: Dict[str, Any], new_rotation: List[float]) -> None:
        """Update mesh rotation (yaw, pitch, roll) and sync XML."""
        target_entry = self._resolve_scene_entry(entry)
        if target_entry is None:
            logger.warning("Cannot update rotation: entry not found")
            return

        mesh = target_entry.get("mesh")
        xml_shape = target_entry.get("xml_shape")
        entry_name = target_entry.get("name", "")
        if mesh is None:
            logger.warning("Cannot update rotation: no mesh for '%s'", entry_name)
            return

        try:
            original_center = self._ensure_original_center(target_entry, mesh)
            state = self._ensure_transform_state(target_entry)
            current_center = viz_geometry.mesh_center(mesh)
            state["rotation"] = [
                float(new_rotation[0]),
                float(new_rotation[1]),
                float(new_rotation[2]),
            ]

            rotation_matrix = viz_geometry.build_rotation_matrix_from_angles(state["rotation"])
            position_after_scale_rotation = rotation_matrix @ (
                state.get("scale", 1.0) * original_center
            )
            translate_vec = current_center - position_after_scale_rotation
            state["translate"] = translate_vec.tolist()

            original_vertices = self._ensure_original_vertices(target_entry)
            mesh, transformed_center, final_center = self._apply_transform_to_entry_mesh(
                target_entry, mesh, original_vertices, original_center, state
            )

            if transformed_center is not None:
                target_entry["position_after_scale_rotation"] = transformed_center
            if final_center is not None:
                target_entry["current_center"] = final_center

            self._refresh_mesh_in_window(target_entry, mesh)

            if entry is not target_entry:
                entry["transform_state"] = state
                if transformed_center is not None:
                    entry["position_after_scale_rotation"] = transformed_center

            if xml_shape is not None:
                self._update_xml_transform_from_state(xml_shape, state)
                logger.debug(
                    "Updated rotation for '%s' to Yaw=%.2f, Pitch=%.2f, Roll=%.2f",
                    entry_name,
                    new_rotation[0],
                    new_rotation[1],
                    new_rotation[2],
                )
                self.push_scene_updates_online(f"Rotation update '{entry_name}'")
            else:
                logger.warning(
                    "No XML shape for '%s' - rotation updated visually but not serialized",
                    entry_name,
                )
        except (ValueError, TypeError, RuntimeError):
            logger.exception("Failed to update object rotation")

    def set_object_scale(self, entry: Dict[str, Any], new_scale: float) -> None:
        """Update mesh scale and sync XML transforms."""
        target_entry = self._resolve_scene_entry(entry)
        if target_entry is None:
            logger.warning("Cannot update scale: entry not found")
            return

        mesh = target_entry.get("mesh")
        xml_shape = target_entry.get("xml_shape")
        entry_name = target_entry.get("name", "")
        if mesh is None:
            logger.warning("Cannot update scale: no mesh for '%s'", entry_name)
            return

        try:
            original_center = self._ensure_original_center(target_entry, mesh)
            state = self._ensure_transform_state(target_entry)
            current_center = viz_geometry.mesh_center(mesh)
            state["scale"] = float(new_scale)

            rotation_matrix = viz_geometry.build_rotation_matrix_from_angles(
                state.get("rotation", [0.0, 0.0, 0.0])
            )
            position_after_scale_rotation = rotation_matrix @ (state["scale"] * original_center)
            translate_vec = current_center - position_after_scale_rotation
            state["translate"] = translate_vec.tolist()

            original_vertices = self._ensure_original_vertices(target_entry)
            mesh, transformed_center, final_center = self._apply_transform_to_entry_mesh(
                target_entry, mesh, original_vertices, original_center, state
            )

            if transformed_center is not None:
                target_entry["position_after_scale_rotation"] = transformed_center
            if final_center is not None:
                target_entry["current_center"] = final_center

            self._refresh_mesh_in_window(target_entry, mesh)

            if entry is not target_entry:
                entry["transform_state"] = state
                if transformed_center is not None:
                    entry["position_after_scale_rotation"] = transformed_center

            if xml_shape is not None:
                self._update_xml_transform_from_state(xml_shape, state)
                logger.debug("Updated scale for '%s' to %.3f", entry_name, new_scale)
                self.push_scene_updates_online(f"Scale update '{entry_name}'")
            else:
                logger.warning(
                    "No XML shape for '%s' - scale updated visually but not serialized",
                    entry_name,
                )
        except (ValueError, TypeError, RuntimeError):
            logger.exception("Failed to update object scale")

    # Scene live gRPC update helpers
    def push_scene_updates_online(self, reason: str) -> bool:
        """Send updated XML to the generator when running in live gRPC mode."""
        viz = self.visualizer
        try:
            from ..io.frame_sources import LiveGrpcSource  # type: ignore
        except ImportError:
            return False

        frame_source = getattr(viz, "frame_source", None)
        if not isinstance(frame_source, LiveGrpcSource):
            return False
        if viz.xml_root is None:
            logger.warning("Cannot push live gRPC scene updates: xml_root is None")
            return False

        has_transforms = any(
            shape.find("transform[@name='to_world']/translate") is not None
            for shape in viz.xml_root.findall("shape")
        )
        if not has_transforms:
            logger.warning("XML does not contain translate elements - transforms may be missing")

        scene_name = "scene_modified.xml"
        if getattr(viz, "xml_path", None):
            scene_name = os.path.basename(str(viz.xml_path))

        logger.info("Pushing scene changes (%s) to generator", reason)
        success = frame_source.update_object_via_xml(
            viz.xml_root,
            scene_name=scene_name,
            reason=reason,
        )
        if not success:
            logger.error("Live gRPC scene update failed; generator will keep old geometry")
            return False

        if hasattr(frame_source, "clear_buffer"):
            frame_source.clear_buffer()
        cache_service = getattr(viz, "cache_service", None)
        if cache_service is not None:
            cache_service.invalidate(
                (
                    CacheInvalidationScope.FRAME_DATA,
                    CacheInvalidationScope.STATIC_SCENE_GEOMETRY,
                ),
                reason="OBJECT_UPDATE",
            )

        current_step = getattr(viz, "animation_step", 0)
        total_steps = getattr(viz, "total_animation_steps", 1)
        if current_step < 0 or current_step >= total_steps:
            current_step = 0
        viz.force_update_next_frame = True

        try:
            viz.update_frame(current_step)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to refresh frame after live gRPC update: %s", exc)
        return True

    # Node edit RPC helper
    def edit_node_properties(self, entry: Dict[str, Any], values: Dict[str, Any]) -> bool:
        """Send target/TX/RX property updates to the generator."""
        viz = self.visualizer
        loader = getattr(viz, "scenario_loader_service", None)
        if loader is None:
            logger.warning("ScenarioLoaderService not available; cannot edit nodes")
            return False

        entry_ref = self._resolve_ui_entry_reference(entry)
        if entry_ref is None:
            entry_ref = entry
        if entry_ref is not entry:
            entry.clear()
            entry.update(entry_ref)

        frame_source = getattr(viz, "frame_source", None)
        if not loader.validate_node_editing_context(frame_source):
            return False

        node_type = entry_ref.get("entry_type")
        if node_type not in ("target", "tx", "rx"):
            loader.warn_unsupported_node()
            return False
        node_name = (
            entry_ref.get("node_name") or entry_ref.get("target_name") or entry_ref.get("name")
        )
        if not node_name:
            loader.warn_missing_identifier()
            return False

        position = values.get("position") if entry_ref.get("supports_position", True) else None
        if position is not None:
            position = [float(position[0]), float(position[1]), float(position[2])]

        orientation = (
            values.get("orientation") if entry_ref.get("supports_orientation", True) else None
        )
        if orientation is not None:
            orientation = [float(orientation[0]), float(orientation[1]), float(orientation[2])]

        scale = values.get("scale") if entry_ref.get("supports_scale", False) else None
        if scale is not None:
            scale = float(scale)

        if position is None and orientation is None and scale is None:
            loader.inform_no_changes()
            return False

        detail_parts = []
        if position is not None:
            detail_parts.append(f"pos={tuple(round(val, 3) for val in position)}")
        if orientation is not None:
            detail_parts.append(f"orient={tuple(round(val, 2) for val in orientation)}")
        if scale is not None:
            detail_parts.append(f"scale={scale:.3f}")
        detail_str = ", ".join(detail_parts)

        try:
            success = frame_source.update_node_properties(
                node_type=node_type,
                node_name=node_name,
                position=position,
                orientation=orientation,
                scale=scale,
                flush_cache=True,
                origin="node_edit",
                details=f"{node_type}:{node_name} ({detail_str})" if detail_str else None,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to send node update for %s: %s", node_name, exc)
            loader.show_node_update_error(
                "Node Update Failed",
                f"Failed to send update for {node_name}: {exc}",
            )
            return False

        if not success:
            loader.show_node_update_error(
                "Node Update Rejected",
                f"The generator rejected the update for {node_name}. Check logs for details.",
            )
            return False

        if position is not None:
            entry_ref["position"] = position
        if orientation is not None:
            entry_ref["orientation_degrees"] = orientation
        if scale is not None:
            entry_ref["scale"] = scale
            if node_type == "target":
                target_name = entry_ref.get("target_name") or entry_ref.get("name")
                if target_name:
                    viz.target_scale_overrides[target_name] = scale
                viz.target_service.apply_target_scale_from_metadata(entry_ref, scale)

        viz._set_status_message(
            f"Node update sent for {node_name}; waiting for refreshed MPC data…",
            5000,
        )

        if hasattr(frame_source, "clear_buffer"):
            try:
                frame_source.clear_buffer(preserve_position=False)
            except (OSError, RuntimeError) as exc:
                logger.debug("Failed to clear frame buffer after node update: %s", exc)

        cache_service = getattr(viz, "cache_service", None)
        if cache_service is not None:
            # TargetService has already atomically updated the authoritative
            # scaled asset record above. Invalidating TARGET_GEOMETRY here
            # would immediately discard that successful update and force the
            # next animation switch to rediscover the same source files.
            cache_service.invalidate(
                CacheInvalidationScope.FRAME_DATA,
                reason=f"NODE_UPDATE {node_name}",
            )
        viz.force_update_next_frame = True

        current_step = getattr(viz, "animation_step", 0)
        total_steps = getattr(viz, "total_animation_steps", 1)
        if current_step < 0 or current_step >= total_steps:
            current_step = 0

        QTimer.singleShot(150, lambda: viz.update_frame(current_step))

        try:
            viz.ui_controller.populate_controls()
        except (RuntimeError, ValueError, AttributeError):
            logger.debug("Unable to refresh control panels after node update", exc_info=True)

        if entry is not entry_ref:
            entry.clear()
            entry.update(entry_ref)

        return True

    # Internal helpers
    def _resolve_scene_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Locate the canonical entry stored on the visualizer."""
        if entry is None:
            return None

        viz = self.visualizer
        entry_name = entry.get("name", "")
        entry_type = entry.get("entry_type", "mesh")

        def _match(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            """Find a canonical entry by object identity or display name."""
            for candidate in entries:
                if candidate is entry:
                    return entry
                if not candidate:
                    continue
                if entry_name and candidate.get("name") == entry_name:
                    return candidate
            return None

        resolved = None
        if entry_type == "mesh":
            resolved = _match(getattr(viz, "mesh_entries", []))
        else:
            resolved = _match(getattr(viz, "target_entries", []))
        return resolved or entry

    def _ensure_original_center(self, entry: Dict[str, Any], mesh) -> np.ndarray:
        """Cache and return the original mesh center."""
        original_center = entry.get("original_center")
        if original_center is None:
            original_center = viz_geometry.mesh_center(mesh)
            entry["original_center"] = original_center
        else:
            original_center = np.asarray(original_center)
        return original_center

    def _apply_transform_to_entry_mesh(
        self,
        entry: Dict[str, Any],
        mesh: Any,
        original_vertices: Optional[np.ndarray],
        original_center: np.ndarray,
        state: Dict[str, Any],
    ) -> tuple[Any, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply a transform to an entry mesh, replacing neutral payloads in-place."""
        if original_vertices is None:
            return mesh, None, None
        if entry.get("entry_type", "mesh") == "mesh":
            try:
                index = list(getattr(self.visualizer, "mesh_entries", [])).index(entry)
            except ValueError:
                index = None
            mesh = ensure_scene_mesh_render_state(entry, index)
        elif entry.get("entry_type") == "target":
            try:
                index = list(getattr(self.visualizer, "target_entries", [])).index(entry)
            except ValueError:
                index = None
            mesh = require_target_mesh_render_state(entry, index)
        if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload):
            updated, transformed_center, final_center = viz_geometry.apply_transform_to_payload(
                mesh.payload, original_vertices, original_center, state
            )
            mesh.replace_payload(updated)
            return mesh, transformed_center, final_center
        logger.debug(
            "Skipping transform for unsupported native mesh type %s; scene editing "
            "expects MeshPayload or RenderObjectState",
            type(mesh).__name__,
        )
        return mesh, None, None

    def _refresh_mesh_in_window(self, entry: Dict[str, Any], mesh: Any) -> bool:
        """Synchronize edited geometry when the renderer is initialized."""
        viz = self.visualizer
        if not getattr(viz, "vis_initialized", False) or viz.vis is None:
            return True
        try:
            if not isinstance(mesh, RenderObjectState):
                raise TypeError("Persistent edits require RenderObjectState")
            if entry.get("entry_type") == "target":
                target_service = getattr(viz, "target_service", None)
                sync_snapshot = getattr(target_service, "sync_target_entry_snapshot", None)
                if not callable(sync_snapshot):
                    raise RuntimeError("TargetService snapshot synchronization is unavailable")
                if not sync_snapshot(entry):
                    entry["_renderer_sync_pending"] = True
                    return False
                viz.renderer.update_renderer()
                return True
            scene_service = getattr(viz, "scene_service", None)
            sync_snapshot = getattr(scene_service, "sync_scene_entry_snapshot", None)
            if not callable(sync_snapshot):
                raise RuntimeError("SceneService snapshot synchronization is unavailable")
            if not sync_snapshot(entry, geometry_changed=True):
                entry["_renderer_sync_pending"] = True
                return False
            entry["_renderer_sync_pending"] = False
            viz.renderer.update_renderer()
            return True
        except (RuntimeError, ValueError) as exc:
            logger.debug("Failed to update geometry during edit: %s", exc)
            entry["_renderer_sync_pending"] = True
            return False

    def _ensure_transform_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize transform dictionary stored on the entry."""
        state = entry.get("transform_state")
        if state is None:
            state = {"scale": 1.0, "rotation": [0.0, 0.0, 0.0], "translate": [0.0, 0.0, 0.0]}
            entry["transform_state"] = state
        state["scale"] = float(state.get("scale", 1.0))
        rotation = state.get("rotation", [0.0, 0.0, 0.0])
        if len(rotation) != 3:
            rotation = (list(rotation) + [0.0, 0.0, 0.0])[:3]
        state["rotation"] = [float(angle) for angle in rotation]
        translate = state.get("translate", [0.0, 0.0, 0.0])
        if len(translate) != 3:
            translate = (list(translate) + [0.0, 0.0, 0.0])[:3]
        state["translate"] = [float(val) for val in translate]
        return state

    def _ensure_original_vertices(self, entry: Dict[str, Any]):
        """Cache original mesh vertices for stable transforms."""
        vertices = entry.get("original_vertices")
        mesh = entry.get("mesh")
        if vertices is None and mesh is not None:
            raw_vertices = viz_geometry.mesh_vertices(mesh)
            if raw_vertices is not None:
                vertices = np.asarray(raw_vertices, dtype=float).copy()
            entry["original_vertices"] = vertices
        return vertices

    def _resolve_ui_entry_reference(self, ui_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map a UI entry dict back to the canonical entry stored on the visualizer."""
        if not isinstance(ui_entry, dict):
            return None

        viz = self.visualizer
        entry_type = ui_entry.get("entry_type")
        name = ui_entry.get("name")
        node_name = ui_entry.get("node_name") or ui_entry.get("target_name")

        def _match(entry_list):
            """Find a canonical UI entry by object identity, node name, or display name."""
            for candidate in entry_list:
                if candidate is ui_entry:
                    return ui_entry
                if node_name and candidate.get("node_name") == node_name:
                    return candidate
                if candidate.get("name") == name:
                    return candidate
            return None

        if entry_type == "target":
            return _match(getattr(viz, "target_entries", []))
        if entry_type == "tx":
            return _match(getattr(viz, "tx_entries", []))
        if entry_type == "rx":
            return _match(getattr(viz, "rx_entries", []))
        if entry_type == "mesh":
            return _match(getattr(viz, "mesh_entries", []))
        return None

    def _update_xml_transform_from_state(self, xml_shape, transform_state):
        """Persist one canonical lightweight transform in Mitsuba degrees."""
        if xml_shape is None:
            return

        transform = xml_shape.find("transform[@name='to_world']")
        if transform is None:
            transform = ET.SubElement(xml_shape, "transform", {"name": "to_world"})

        for operation in list(transform):
            if operation.tag in {"scale", "rotate", "translate"}:
                transform.remove(operation)

        scale_elem = ET.SubElement(transform, "scale")
        scale_elem.set("value", f"{transform_state.get('scale', 1.0):.6f}")

        rotation = transform_state.get("rotation", [0.0, 0.0, 0.0])
        yaw, pitch, roll = rotation
        # The in-memory state is yaw(Z), pitch(Y), roll(X). Mitsuba operations
        # are left-multiplied in written order, so the lightweight XML contract
        # writes the corresponding X, Y, Z rotations in that order.
        for axis, angle in (("x", roll), ("y", pitch), ("z", yaw)):
            if abs(angle) > 1e-6:
                rotate = ET.SubElement(transform, "rotate", {axis: "1"})
                rotate.set("angle", f"{angle:.6f}")

        translate_elem = ET.SubElement(transform, "translate")

        translate_vec = np.asarray(transform_state.get("translate", [0.0, 0.0, 0.0]), dtype=float)
        translate_elem.set("x", f"{translate_vec[0]:.6f}")
        translate_elem.set("y", f"{translate_vec[1]:.6f}")
        translate_elem.set("z", f"{translate_vec[2]:.6f}")
