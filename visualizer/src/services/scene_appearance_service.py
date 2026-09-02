"""Scene-level appearance controls for backgrounds, outlines, and transparency."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from PySide6.QtGui import QColor

from shared.logging import get_logger

from ..model import RenderObjectState
from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR, DEFAULT_SCENE_BACKGROUND_PRESET
from ..scene.geometry_payload_factory import extract_wireframe_payload
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
)
from .base import BaseService
from .scene_batches import SceneBatch

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.scene_appearance_service")
_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\-]+")
_OUTLINE_REBUILD_NEEDED = "_outline_rebuild_needed"
_OUTLINE_SYNC_PENDING = "_outline_sync_pending"


class SceneAppearanceService(BaseService):
    """Handles background, lighting, and outline controls for the scene."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind scene appearance updates to the visualizer composition root."""
        super().__init__()
        self.visualizer = visualizer

    @staticmethod
    def _sanitize_name(value: Any) -> str:
        """Normalize scene labels into renderer-safe outline identifiers."""
        text = str(value).strip().lower().replace(" ", "_")
        text = _NAME_SANITIZE_RE.sub("_", text)
        return text or "unknown"

    def _sync_outline_handle(
        self,
        owner: MutableMapping[str, Any],
        handle: RenderObjectState,
        *,
        visible: bool,
    ) -> bool:
        """Apply desired outline state and report whether it converged.

        Native idempotence and applied snapshots belong to the renderer.  The
        service retains only a failed desired transition so a visibility
        change can be retried even after the mutable handle was updated.
        Visible handles are always handed to ``ensure_object``; a renderer
        reset can therefore rematerialize them without an application-side
        mirror of backend state.
        """
        renderer = self.visualizer.renderer
        visibility_changed = handle.visible != bool(visible)
        pending = bool(owner.get(_OUTLINE_SYNC_PENDING, False))
        handle.visible = bool(visible)

        should_sync = bool(visible or visibility_changed or pending)
        if not should_sync:
            return True

        owner[_OUTLINE_SYNC_PENDING] = True
        if not renderer.ensure_object(handle.to_render_object()):
            return False

        owner.pop(_OUTLINE_SYNC_PENDING, None)
        return True

    def _remove_outline_object(
        self,
        owner: MutableMapping[str, Any],
        object_id: str,
    ) -> bool:
        """Ensure one outline is absent through the idempotent renderer API."""
        if not self.visualizer.renderer.remove_object(object_id):
            return False
        owner.pop(_OUTLINE_SYNC_PENDING, None)
        return True

    def _mesh_payload(self, mesh: Any) -> MeshPayload | None:
        """Coerce a mesh or render handle into a renderer-neutral mesh payload."""
        if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload):
            return mesh.payload
        if isinstance(mesh, MeshPayload):
            return mesh
        vertices = getattr(mesh, "vertices", None)
        triangles = getattr(mesh, "triangles", None)
        if vertices is None or triangles is None:
            return None
        try:
            return MeshPayload(
                vertices=np.asarray(vertices, dtype=np.float64),
                triangles=np.asarray(triangles, dtype=np.int32),
            )
        except (TypeError, ValueError):
            return None

    def _outline_handle(self, name: str, mesh: Any, color: np.ndarray) -> RenderObjectState | None:
        """Build a hidden wireframe render handle for one mesh payload."""
        payload = self._mesh_payload(mesh)
        if payload is None:
            return None
        outline = extract_wireframe_payload(payload)
        if len(outline.lines) == 0:
            return None
        line_payload = LineSetPayload(
            points=outline.points,
            lines=outline.lines,
            colors=np.tile(color[:3], (len(outline.lines), 1)),
        )
        return RenderObjectState(
            id=name,
            payload=line_payload,
            material=MaterialPayload(
                base_color=(float(color[0]), float(color[1]), float(color[2]), 1.0),
                shader="unlit",
            ),
            visible=False,
            is_edge=True,
            metadata={"type": "scene_edge"},
        )

    def _scene_outline_name(self, entry: Dict[str, Any]) -> str:
        """Return the canonical renderer ID for one scene-entry outline."""
        mesh = entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            raise TypeError("Scene outline ownership requires RenderObjectState")
        # Backend edge classification recognizes this prefix; identity still
        # comes from the canonical mesh ID rather than a display name.
        outline_name = f"scene_outline_{mesh.id}"
        entry["_outline_geometry_name"] = outline_name
        return outline_name

    def _merged_outline_name(self, group_name: str) -> str:
        """Return the renderer name for one merged scene-outline group."""
        return f"scene_merged_outline_{self._sanitize_name(group_name)}"

    def _sync_single_scene_outline(
        self,
        entry: Dict[str, Any],
        *,
        enabled: bool,
    ) -> bool:
        """Synchronize one scene-entry outline and report success."""
        return self._sync_scene_entry_outline_snapshot(
            entry,
            visible=bool(enabled),
            rebuild=False,
        )

    def _sync_scene_entry_outline_snapshot(
        self,
        entry: Dict[str, Any],
        *,
        visible: bool,
        rebuild: bool,
    ) -> bool:
        """Converge one individual outline and report success."""
        outline = entry.get("outline_geometry")

        # Hidden outlines do not need a new CPU wireframe or GPU upload. Keep
        # an existing native outline hidden and rebuild from the current mesh
        # only when the user enables it again.
        if rebuild and not visible:
            if isinstance(outline, RenderObjectState):
                entry[_OUTLINE_REBUILD_NEEDED] = True
            entry["outline_visible"] = False
            if not isinstance(outline, RenderObjectState):
                return True
            return self._sync_outline_handle(entry, outline, visible=False)

        rebuild_now = bool(rebuild or (visible and entry.get(_OUTLINE_REBUILD_NEEDED, False)))
        if not visible and not isinstance(outline, RenderObjectState):
            entry["outline_visible"] = False
            return outline is None

        if rebuild_now or not isinstance(outline, RenderObjectState):
            replacement = self._outline_handle(
                self._scene_outline_name(entry),
                entry.get("mesh"),
                np.asarray(self.visualizer.outline_color, dtype=np.float64),
            )
            if replacement is None:
                if isinstance(outline, RenderObjectState) and not self._remove_outline_object(
                    entry,
                    outline.id,
                ):
                    return False
                entry.pop("outline_geometry", None)
                entry["outline_visible"] = False
                return not bool(visible)
            if isinstance(outline, RenderObjectState) and outline.id == replacement.id:
                outline.replace_payload(replacement.payload)
                outline.material = replacement.material
                outline.world_transform = replacement.world_transform
                outline.is_edge = replacement.is_edge
                outline.metadata = dict(replacement.metadata)
            else:
                if isinstance(outline, RenderObjectState) and not self._remove_outline_object(
                    entry,
                    outline.id,
                ):
                    return False
                outline = replacement
                entry["outline_geometry"] = outline

        if not isinstance(outline, RenderObjectState):
            return not bool(visible)
        entry["outline_visible"] = bool(visible)
        success = self._sync_outline_handle(
            entry,
            outline,
            visible=bool(visible),
        )
        if success and visible:
            entry.pop(_OUTLINE_REBUILD_NEEDED, None)
        return success

    def sync_scene_entry_outline_snapshot(
        self,
        entry: Dict[str, Any],
        *,
        visible: bool,
        rebuild: bool = False,
    ) -> bool:
        """Ensure one individual scene outline matches its desired snapshot.

        SceneService uses the success result while moving an entry between
        individual and merged ownership: it must not publish the new owner
        until the old mesh and outline are gone.
        """
        return self._sync_scene_entry_outline_snapshot(
            entry,
            visible=bool(visible),
            rebuild=bool(rebuild),
        )

    def remove_scene_entry_outline(self, entry: Dict[str, Any]) -> bool:
        """Ensure an individual outline is absent before aggregate ownership."""
        outline = entry.get("outline_geometry")
        if not isinstance(outline, RenderObjectState):
            entry["outline_visible"] = False
            return True
        removed = self._remove_outline_object(entry, outline.id)
        if removed:
            outline.visible = False
            entry["outline_visible"] = False
        return removed

    def sync_entry_outline_visibility(self, entry: Dict[str, Any]) -> None:
        """Sync one scene-entry outline after a mesh visibility change."""
        if entry.get("entry_type") == "target":
            return
        if entry.get("mesh") is None:
            return
        viz = self.visualizer
        enabled = bool(viz.outlines_enabled and entry.get("visible", True))
        success = self._sync_single_scene_outline(
            entry,
            enabled=enabled,
        )
        if not success:
            logger.debug("Failed to synchronize outline for %s", entry.get("name"))

    # Background helpers
    def set_background_color(self, color: List[float]) -> None:
        """Apply a background RGB color to renderer and app mirror state."""
        viz = self.visualizer
        try:
            if hasattr(viz, "renderer") and viz.renderer is not None:
                viz.renderer.set_background_color(color)
                viz.current_background_color = list(color)
                if hasattr(viz, "current_background_image"):
                    viz.current_background_image = None
                logger.info(
                    "Background color changed to RGB(%.2f, %.2f, %.2f)",
                    color[0],
                    color[1],
                    color[2],
                )
            else:
                logger.warning("Renderer not available for background updates")

            self._update_background_color_display(color)
        except (AttributeError, ValueError, RuntimeError) as exc:
            logger.error("Failed to change background color: %s", exc)

    def _update_background_color_display(self, color: List[float]) -> None:
        """Update menus/status bar with the current color."""
        viz = self.visualizer
        try:
            color_name = self._get_color_name(color)
            if hasattr(viz, "current_bg_action") and viz.current_bg_action is not None:
                viz.current_bg_action.setText(f"&Current: {color_name}")

            rgb_text = f"RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})"
            if hasattr(viz, "_set_status_message"):
                viz._set_status_message(
                    f"Background: {color_name} {rgb_text}",
                    5000,
                )
        except (AttributeError, ValueError) as exc:
            logger.debug("Failed to update background color display: %s", exc)

    def _get_color_name(self, color: List[float]) -> str:
        """Return a coarse display label for an RGB background color."""
        r, g, b = color
        if r < 0.1 and g < 0.1 and b < 0.1:
            return "Black"
        if r > 0.9 and g > 0.9 and b > 0.9:
            return "White"
        if abs(r - g) < 0.1 and abs(g - b) < 0.1:
            if r < 0.3:
                return "Dark Gray"
            if r < 0.6:
                return "Gray"
            return "Light Gray"
        return "Custom"

    def ensure_light_gray_background(self) -> None:
        """Restore the default scene background unless an image is active."""
        viz = self.visualizer
        try:
            if (
                hasattr(viz, "current_background_image")
                and viz.current_background_image
                and hasattr(viz, "renderer")
                and viz.renderer is not None
                and hasattr(viz.renderer, "set_background_image")
            ):
                viz.renderer.set_background_image(viz.current_background_image)
                return
            if hasattr(viz, "renderer") and viz.renderer is not None:
                viz.renderer.set_background_color(DEFAULT_SCENE_BACKGROUND_COLOR)
                viz.current_background_color = list(DEFAULT_SCENE_BACKGROUND_COLOR)
                logger.info("Forced background color to default dark gray")
        except (AttributeError, RuntimeError) as exc:
            logger.debug("Could not force default background: %s", exc)

    def reset_to_default_background(self) -> None:
        """Reset background preset and color to the default scene value."""
        viz = self.visualizer
        viz.current_background_preset = DEFAULT_SCENE_BACKGROUND_PRESET
        self.set_background_color(DEFAULT_SCENE_BACKGROUND_COLOR)
        logger.info("Reset background color to default dark gray")

    def set_background_preset(self, preset_name: str, color: List[float]) -> None:
        """Store the selected background preset and apply its RGB color."""
        viz = self.visualizer
        viz.current_background_preset = preset_name
        self.set_background_color(color)

    def pick_custom_background_color(self) -> None:
        """Open the app color picker and apply a valid custom color."""
        viz = self.visualizer
        try:
            current_color = QColor(255, 255, 255)
            if not hasattr(viz, "dialog_manager") or viz.dialog_manager is None:
                logger.warning("Dialog manager not available for color picking")
                return
            color = viz.dialog_manager.pick_color(current_color, "Pick Background Color")
            if color and color.isValid():
                r = color.red() / 255.0
                g = color.green() / 255.0
                b = color.blue() / 255.0
                viz.current_background_preset = "Custom"
                self.set_background_color([r, g, b])
                logger.info("Custom background color selected: RGB(%.2f, %.2f, %.2f)", r, g, b)
        except (RuntimeError, AttributeError) as exc:
            logger.error("Failed to open color picker: %s", exc)

    # Outlines
    def set_edge_visibility(self, enabled: bool) -> None:
        """Show or hide wireframe edges on all visible scene meshes.

        When mesh merging is active, edges are computed from the merged
        geometries (one LineSet per merged group) instead of per-building.
        This avoids adding thousands of individual LineSet objects to
        Filament.
        """
        viz = self.visualizer
        viz.outlines_enabled = bool(enabled)
        scene_svc = getattr(viz, "scene_service", None)
        has_merged_groups = bool(
            scene_svc and scene_svc._merge_enabled and scene_svc._merged_groups
        )
        if (
            (not viz.mesh_entries and not has_merged_groups)
            or viz.vis is None
            or not viz.vis_initialized
        ):
            logger.debug("set_edge_visibility(%s): early return (no entries or vis)", enabled)
            return

        if has_merged_groups:
            self._set_merged_edge_visibility(enabled, scene_svc)
            return

        use_batch = hasattr(viz.renderer, "batch_updates")
        ctx = viz.renderer.batch_updates() if use_batch else nullcontext()
        all_succeeded = True
        with ctx:
            for entry in viz.mesh_entries:
                enabled_for_entry = bool(viz.outlines_enabled and entry.get("visible", True))
                try:
                    success = self._sync_single_scene_outline(
                        entry,
                        enabled=enabled_for_entry,
                    )
                    all_succeeded = bool(success and all_succeeded)
                except (RuntimeError, ValueError) as exc:
                    all_succeeded = False
                    logger.debug("Outline toggle failed for %s: %s", entry.get("name"), exc)
        if not all_succeeded:
            logger.warning("One or more scene outlines did not accept the visibility change")

    def _set_merged_edge_visibility(self, enabled: bool, scene_svc: Any) -> None:
        """Toggle edges using one merged outline per material group."""

        viz = self.visualizer
        use_batch = hasattr(viz.renderer, "batch_updates")
        ctx = viz.renderer.batch_updates() if use_batch else nullcontext()
        all_succeeded = True
        split_member_ids: set[int] = set()
        batched_member_ids: set[int] = set()

        with ctx:
            for group_name, group_info in scene_svc._merged_meshes.items():
                member_ids = set(scene_svc._merged_groups.get(group_name, ()))
                batched_member_ids.update(member_ids)
                partition = group_info.current_partition
                resolved_by_mesh_id = {}
                resolver = getattr(scene_svc, "_resolved_scene_appearance", None)
                if callable(resolver):
                    for mesh_id in member_ids:
                        entry = scene_svc._mesh_id_to_scene_entry.get(mesh_id)
                        if isinstance(entry, dict):
                            resolved_by_mesh_id[mesh_id] = resolver(entry)
                    if set(resolved_by_mesh_id) == member_ids:
                        partition = group_info.resolve_partition(resolved_by_mesh_id)
                split_member_ids.update(partition.individual_member_ids)
                success = self._sync_merged_outline_snapshot(
                    group_name,
                    group_info,
                    visible=bool(enabled and partition.aggregate_member_ids),
                    rebuild=False,
                )
                all_succeeded = bool(success and all_succeeded)

            # Regular members use the aggregate outline. Highlighted members
            # are split into individual render owners and therefore need their
            # own outline synchronization alongside ordinary unmerged meshes.
            for entry in viz.mesh_entries:
                mesh = entry.get("mesh")
                if mesh is None:
                    continue
                mesh_id = id(mesh)
                if mesh_id in batched_member_ids and mesh_id not in split_member_ids:
                    success = self.remove_scene_entry_outline(entry)
                    all_succeeded = bool(success and all_succeeded)
                    continue
                if mesh_id in split_member_ids:
                    enabled_for_entry = bool(enabled)
                else:
                    resolver = getattr(scene_svc, "_resolved_scene_appearance", None)
                    resolved = resolver(entry) if callable(resolver) else None
                    enabled_for_entry = bool(
                        enabled
                        and (
                            resolved.visible if resolved is not None else entry.get("visible", True)
                        )
                    )
                try:
                    success = self._sync_single_scene_outline(
                        entry,
                        enabled=enabled_for_entry,
                    )
                    all_succeeded = bool(success and all_succeeded)
                except (RuntimeError, ValueError) as exc:
                    all_succeeded = False
                    logger.debug("Outline toggle failed for %s: %s", entry.get("name"), exc)

        if not all_succeeded:
            logger.warning("One or more merged scene outlines did not accept the visibility change")

    def _sync_merged_outline_snapshot(
        self,
        group_name: str,
        group_info: SceneBatch,
        *,
        visible: bool,
        rebuild: bool,
    ) -> bool:
        """Converge one merged outline and report success."""
        outline_key = "_merged_outline"
        outline_name = self._merged_outline_name(group_name)
        group_info["_merged_outline_name"] = outline_name
        outline = group_info.get(outline_key)

        if rebuild and not visible:
            if isinstance(outline, RenderObjectState):
                group_info[_OUTLINE_REBUILD_NEEDED] = True
            if not isinstance(outline, RenderObjectState):
                return True
            return self._sync_outline_handle(group_info, outline, visible=False)

        rebuild_now = bool(rebuild or (visible and group_info.get(_OUTLINE_REBUILD_NEEDED, False)))
        if not visible and not isinstance(outline, RenderObjectState):
            return outline is None

        if rebuild_now or not isinstance(outline, RenderObjectState):
            merged_mesh = group_info.get("geometry")
            if merged_mesh is None:
                return not bool(visible)
            replacement = self._build_outline_from_mesh(merged_mesh, name=outline_name)
            if replacement is None:
                if isinstance(outline, RenderObjectState) and not self._remove_outline_object(
                    group_info,
                    outline.id,
                ):
                    return False
                group_info.pop(outline_key, None)
                return True
            if isinstance(outline, RenderObjectState) and outline.id == replacement.id:
                outline.replace_payload(replacement.payload)
                outline.material = replacement.material
                outline.world_transform = replacement.world_transform
                outline.is_edge = replacement.is_edge
                outline.metadata = dict(replacement.metadata)
            else:
                if isinstance(outline, RenderObjectState) and not self._remove_outline_object(
                    group_info,
                    outline.id,
                ):
                    return False
                outline = replacement
                group_info[outline_key] = outline

        if not isinstance(outline, RenderObjectState):
            return False

        success = self._sync_outline_handle(
            group_info,
            outline,
            visible=bool(visible),
        )
        if success and visible:
            group_info.pop(_OUTLINE_REBUILD_NEEDED, None)
        return success

    def sync_merged_outline_geometry(
        self,
        group_name: str,
        group_info: SceneBatch,
        *,
        visible: bool,
        rebuild: bool,
    ) -> bool:
        """Ensure one merged outline matches its current aggregate geometry."""
        return self._sync_merged_outline_snapshot(
            group_name,
            group_info,
            visible=bool(visible),
            rebuild=bool(rebuild),
        )

    def remove_merged_outline_geometry(self, group_info: SceneBatch) -> bool:
        """Ensure a merged outline is absent without dropping failed cleanup state."""
        outline = group_info.get("_merged_outline")
        if not isinstance(outline, RenderObjectState):
            return True
        if not self._remove_outline_object(group_info, outline.id):
            return False
        group_info.pop("_merged_outline", None)
        group_info.pop("_merged_outline_name", None)
        return True

    def _build_outline_from_mesh(self, mesh: Any, *, name: str) -> Any:
        """Compute a stable neutral wireframe handle from an aggregate mesh."""
        viz = self.visualizer
        color = np.asarray(viz.outline_color, dtype=np.float64)
        return self._outline_handle(name, mesh, color)

    # Transparency
    def _set_entry_transparency(self, entry: Dict[str, Any], alpha: float) -> bool:
        """Delegate transparency to the entry's complete semantic owner."""
        del alpha  # The global alpha is already stored on the visualizer.
        if entry.get("mesh") is None:
            return False
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh = getattr(appearance, "refresh_entry_material", None)
        return bool(callable(refresh) and refresh(entry, update_renderer=False))

    def _renderer_supports_transparency(self) -> bool:
        """Return whether the active renderer advertises transparency support."""
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None:
            return False
        return renderer_capabilities(renderer).transparency

    def set_building_transparency(self, alpha: float) -> None:
        """Set transparency for scene/building meshes when supported."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not self._renderer_supports_transparency():
            return

        viz.current_building_alpha = alpha

        batch_updates = getattr(renderer, "batch_updates", None)
        ctx = batch_updates() if callable(batch_updates) else nullcontext()
        with ctx:
            scene_service = getattr(viz, "scene_service", None)
            refresh_all = getattr(scene_service, "refresh_all_scene_materials", None)
            if callable(refresh_all):
                refresh_all()

        if hasattr(renderer, "update_renderer"):
            renderer.update_renderer()

    def set_target_transparency(self, alpha: float) -> None:
        """Set transparency for target meshes when supported."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not self._renderer_supports_transparency():
            return

        viz.current_target_alpha = alpha
        batch_updates = getattr(renderer, "batch_updates", None)
        ctx = batch_updates() if callable(batch_updates) else nullcontext()
        with ctx:
            for entry in getattr(viz, "target_entries", []) or []:
                self._set_entry_transparency(entry, alpha)

        if hasattr(renderer, "update_renderer"):
            renderer.update_renderer()

    # Accessors used by wrappers/tests
    def update_background_color_display(self, color: List[float]) -> None:
        """Public wrapper for updating background color UI mirrors."""
        self._update_background_color_display(color)

    def get_color_name(self, color: List[float]) -> str:
        """Public wrapper returning a coarse display label for an RGB color."""
        return self._get_color_name(color)
