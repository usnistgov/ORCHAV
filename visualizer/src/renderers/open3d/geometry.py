"""Declarative-object synchronization for the Open3D backend.

The adapter keeps stable IDs and applied snapshots internally so shared callers
never hold native Open3D objects or participate in Open3D's destructive
remove/re-add replacement lifecycle. Object-identity helpers remain local to
backend compatibility paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...backends.open3d_payload_codec import geometry_payload_to_o3d
from ...model import RenderObject, Transform
from ...types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    OrientationFramePayload,
    SurfaceColorSource,
    TextLabelPayload,
    material_payload_from_mapping,
)

logger = get_logger("orchav.renderer_open3d")


@dataclass(frozen=True, slots=True)
class _Open3DAppliedObject:
    """Last successfully applied declarative state for one Open3D object."""

    payload: Any
    material: MaterialPayload | None
    transform: np.ndarray
    visible: bool
    is_edge: bool
    color_source: SurfaceColorSource


class Open3DGeometryMixin:
    """Own Open3D geometry identity, visibility, and transform bookkeeping.

    Declarative object methods form the renderer-neutral contract. Named and
    object-identity helpers below are backend-local implementation surfaces.
    """

    @staticmethod
    def _is_label_geometry_name(name: str) -> bool:
        """Return whether a stable geometry name identifies a text label."""
        text = str(name)
        return text.endswith("::label") or "_label_" in text

    @staticmethod
    def _payload_color_source(payload: Any) -> SurfaceColorSource:
        """Return intrinsic color ownership for one neutral payload."""
        if isinstance(payload, MeshPayload):
            return payload.color_source
        if isinstance(payload, OrientationFramePayload):
            return SurfaceColorSource.VERTEX
        return SurfaceColorSource.MATERIAL

    def _pending_hidden_intents(self) -> set[str]:
        """Return hidden intents awaiting successful native application."""
        pending = getattr(self, "_pending_hidden_geometry_names", None)
        if pending is None:
            pending = set()
            self._pending_hidden_geometry_names = pending
        return pending

    def _pending_visualizer_visibility_updates(self) -> dict[str, bool]:
        """Return O3DVisualizer bookkeeping updates deferred by a render scope."""
        pending = getattr(self, "_pending_o3d_visualizer_visibility", None)
        if pending is None:
            pending = {}
            self._pending_o3d_visualizer_visibility = pending
        return pending

    def _apply_scene_visibility(self, name: str, visible: bool) -> bool:
        """Apply actual scene visibility without presenting an intermediate frame.

        ``O3DVisualizer.show_geometry()`` posts a native window redraw for every
        object on Windows.  Calling it repeatedly inside ORCHAV's logical batch
        can therefore display TX/RX labels or orientation frames one at a time.
        The public ``Open3DScene.show_geometry()`` API changes the Filament scene
        without that per-object window post, so all objects can reach their final
        state before O3DVisualizer bookkeeping is synchronized at the boundary.
        """
        if self._o3d_vis is None:
            return False
        scene = getattr(self._o3d_vis, "scene", None)
        show_geometry = getattr(scene, "show_geometry", None)
        if not callable(show_geometry):
            logger.warning("Open3D scene does not expose show_geometry() for '%s'", name)
            return False
        try:
            show_geometry(name, bool(visible))
            geometry_is_visible = getattr(scene, "geometry_is_visible", None)
            if callable(geometry_is_visible) and bool(geometry_is_visible(name)) != bool(visible):
                logger.warning(
                    "Open3D scene visibility verification failed for '%s' (wanted %s)",
                    name,
                    bool(visible),
                )
                return False
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3D scene visibility failed for '%s': %s", name, exc)
            return False
        return True

    def _stage_visualizer_visibility(self, name: str, visible: bool) -> bool:
        """Queue high-level Open3D bookkeeping after low-level scene success."""
        self._pending_visualizer_visibility_updates()[name] = bool(visible)
        if self._batch_mode or self._frame_update_in_progress:
            return True
        # The low-level scene is the rendered source of truth. Failure of the
        # high-level geometry-tree bookkeeping must not roll back or misreport
        # a scene change that Open3D already verified; retain it for retry.
        self._flush_visualizer_visibility_updates()
        return True

    def _flush_visualizer_visibility_updates(self) -> bool:
        """Synchronize O3DVisualizer bookkeeping after all scene values are final.

        The high-level calls keep Open3D's built-in geometry tree, time/group
        policy, and selection state coherent.  They may each post a native draw,
        but every low-level scene object already has its final visibility by this
        point, so none of those draws can expose a partially applied batch.
        """
        pending = self._pending_visualizer_visibility_updates()
        if not pending:
            return True
        if self._o3d_vis is None:
            return False

        all_synced = True
        for name, visible in tuple(pending.items()):
            if name not in self._geometry_names:
                pending.pop(name, None)
                continue
            try:
                self._o3d_vis.show_geometry(name, bool(visible))
            except (RuntimeError, AttributeError) as exc:
                all_synced = False
                logger.warning(
                    "Open3D visualizer visibility bookkeeping failed for '%s': %s",
                    name,
                    exc,
                )
                continue
            pending.pop(name, None)
        return all_synced

    @staticmethod
    def _average_vertex_color(geometry: o3d.geometry.Geometry) -> Optional[list[float]]:
        """Return the average RGB vertex color for uniformly painted label meshes."""
        try:
            colors = np.asarray(getattr(geometry, "vertex_colors"), dtype=float)
        except (AttributeError, TypeError, ValueError):
            return None
        if colors.ndim != 2 or colors.shape[0] == 0 or colors.shape[1] < 3:
            return None
        color = np.nanmean(colors[:, :3], axis=0)
        if color.size < 3 or not np.all(np.isfinite(color[:3])):
            return None
        return [float(x) for x in np.clip(color[:3], 0.0, 1.0)]

    @staticmethod
    def _effective_label_material(material: MaterialPayload | None) -> MaterialPayload:
        """Return the renderer-owned material semantics for a neutral label."""
        if material is None:
            return MaterialPayload(shader="unlit")
        if material.shader == "unlit":
            return material
        return replace(material, shader="unlit")

    def _material_for_label_geometry(
        self,
        geometry: o3d.geometry.Geometry | None,
        *,
        requested: MaterialPayload | dict[str, Any] | rendering.MaterialRecord | None = None,
        fallback: rendering.MaterialRecord | None = None,
    ) -> rendering.MaterialRecord:
        """Build the unlit native material for a label geometry.

        Declarative labels carry an explicit neutral color, which takes
        precedence over baked vertex colors. Raw native labels without an
        explicit material retain their uniformly painted mesh color.
        """
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"

        requested_color: list[float] | None = None
        if isinstance(requested, MaterialPayload):
            requested_color = [float(value) for value in requested.base_color]
        elif isinstance(requested, dict):
            values = requested.get("base_color", requested.get("color"))
            if values is not None:
                try:
                    requested_color = [float(value) for value in values]
                except (TypeError, ValueError):
                    requested_color = None
                if requested_color is not None and len(requested_color) == 3:
                    requested_color.append(float(requested.get("alpha", 1.0)))
        elif requested is not None:
            try:
                requested_color = [float(value) for value in requested.base_color]
            except (AttributeError, TypeError, ValueError):
                requested_color = None

        if requested_color is not None and len(requested_color) >= 3:
            alpha = requested_color[3] if len(requested_color) >= 4 else 1.0
            material.base_color = [*requested_color[:3], alpha]
            return material

        color = None if geometry is None else self._average_vertex_color(geometry)
        if color is not None:
            material.base_color = [color[0], color[1], color[2], 1.0]
            return material

        if fallback is not None:
            try:
                material.base_color = list(fallback.base_color)
            except (AttributeError, TypeError, ValueError):
                pass
        return material

    def _add_or_update_geometry(
        self, name: str, geometry: o3d.geometry.Geometry, material: rendering.MaterialRecord
    ) -> bool:
        """Upload geometry under a stable name and refresh backend metadata.

        Empty or collapsed geometry is skipped because Open3D's Filament backend
        can crash or reject degenerate buffers during upload.
        """
        if self._o3d_vis is None:
            return False

        pending_hidden = self._pending_hidden_intents()
        was_hidden = name in self._hidden_geometry_names or name in pending_hidden
        if was_hidden:
            pending_hidden.add(name)
        native_added = False
        try:
            is_empty = self._is_geometry_empty(geometry)
            if is_empty:
                if name in self._geometry_names:
                    self._o3d_vis.remove_geometry(name)
                    self._discard_geometry_tracking(
                        name,
                        clear_object_state=False,
                        preserve_hidden_intent=was_hidden,
                    )
                logger.debug(f"Open3DRenderer: Skipping empty geometry '{name}'")
                return False

            upload_material = material
            if self._is_label_geometry_name(name):
                # O3DVisualizer does not consistently display TriangleMesh
                # vertex colors for text meshes, so labels carry their color
                # explicitly in the unlit material used for upload.
                if getattr(material, "shader", "") == "defaultUnlit":
                    upload_material = material
                else:
                    upload_material = self._material_for_label_geometry(
                        geometry,
                        fallback=material,
                    )

            # Lit Filament shaders expect normals; add them for generated meshes.
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                shader = getattr(upload_material, "shader", "")
                if "Lit" in shader and not geometry.has_vertex_normals():
                    geometry.compute_vertex_normals()

            if name in self._geometry_names:
                self._o3d_vis.remove_geometry(name)
                self._discard_geometry_tracking(
                    name,
                    clear_object_state=False,
                    preserve_hidden_intent=was_hidden,
                )

            self._o3d_vis.add_geometry(name, geometry, upload_material)
            native_added = True
            self._geometry_names.add(name)
            if isinstance(geometry, o3d.geometry.LineSet):
                self._geometry_types[name] = "lines"
            elif isinstance(geometry, o3d.geometry.PointCloud):
                self._geometry_types[name] = "points"
            else:
                self._geometry_types[name] = "mesh"
            self._geometry_id_to_name[id(geometry)] = name
            if self._is_label_geometry_name(name):
                self._pbr_materials[name] = upload_material
            # Snapshot the vertex center at upload time so that future
            # transform calculations are not affected by CPU-side vertex
            # mutations (for example direct object-geometry offset calls).
            try:
                self._geometry_upload_center[name] = np.asarray(
                    geometry.get_center(), dtype=float
                ).copy()
            except (ValueError, AttributeError):
                self._geometry_upload_center.pop(name, None)
            try:
                self._o3d_vis.scene.scene.set_geometry_culling(name, bool(self._culling_enabled))
            except (RuntimeError, AttributeError) as exc:
                # Culling is part of the backend-owned object policy. Reporting
                # success here would let ensure_object() cache a state that did
                # not fully reach Filament and would suppress all later retries.
                logger.warning(
                    "Open3DRenderer: failed to apply culling for '%s': %s",
                    name,
                    exc,
                )
                return False
            if name in self._hidden_geometry_names:
                pending_hidden.add(name)
            if name in pending_hidden:
                if not self._apply_scene_visibility(name, False):
                    raise RuntimeError(f"failed to restore hidden visibility for '{name}'")
                if not self._stage_visualizer_visibility(name, False):
                    raise RuntimeError(f"failed to synchronize hidden visibility for '{name}'")
                self._hidden_geometry_names.add(name)
                pending_hidden.discard(name)
                try:
                    self._post_redraw()
                except (RuntimeError, AttributeError) as exc:
                    logger.warning(
                        "Open3DRenderer: Failed to request redraw for hidden replacement %s: %s",
                        name,
                        exc,
                    )
                self._request_visibility_settle_redraw(f"restore hidden replacement '{name}'")
            logger.debug(f"Open3DRenderer: Added/updated geometry '{name}'")
            return True

        except (RuntimeError, AttributeError) as exc:
            logger.debug(f"Open3DRenderer: Failed to add geometry '{name}': {exc}")
            if not native_added:
                # Open3D may have allocated a partial native object before
                # raising. Best-effort removal prevents an untracked object
                # from colliding with the next retry.
                try:
                    self._o3d_vis.remove_geometry(name)
                except (RuntimeError, AttributeError):
                    pass
                self._discard_geometry_tracking(
                    name,
                    clear_object_state=False,
                    preserve_hidden_intent=was_hidden,
                )
            elif was_hidden:
                # The replacement is still tracked so ensure_object() can
                # remove it before rollback. Keep the requested hidden state
                # even when show_geometry(False) was the failing operation.
                self._hidden_geometry_names.discard(name)
                pending_hidden.add(name)
            return False

    def _discard_geometry_tracking(
        self,
        name: str,
        *,
        clear_object_state: bool = True,
        preserve_hidden_intent: bool = False,
    ) -> None:
        """Clear backend metadata after geometry is known to be absent."""
        pending_hidden = self._pending_hidden_intents()
        had_hidden_intent = name in self._hidden_geometry_names or name in pending_hidden
        self._geometry_names.discard(name)
        self._hidden_geometry_names.discard(name)
        pending_hidden.discard(name)
        self._geometry_upload_center.pop(name, None)
        self._geometry_position_cache.pop(name, None)
        self._pbr_materials.pop(name, None)
        self._geometry_types.pop(name, None)
        self._edge_geometry_names.discard(name)
        self._pending_visualizer_visibility_updates().pop(name, None)
        stale_ids = [
            gid for gid, mapped_name in self._geometry_id_to_name.items() if mapped_name == name
        ]
        for gid in stale_ids:
            self._geometry_id_to_name.pop(gid, None)
        if preserve_hidden_intent and had_hidden_intent:
            pending_hidden.add(name)
        if clear_object_state:
            self._open3d_applied_objects().pop(name, None)

    def _open3d_applied_objects(self) -> dict[str, _Open3DAppliedObject]:
        """Return the renderer-local declarative state cache."""
        cache = getattr(self, "_applied_render_objects", None)
        if cache is None:
            cache = {}
            self._applied_render_objects = cache
        return cache

    def _is_geometry_empty(self, geometry: o3d.geometry.Geometry) -> bool:
        """Return True for geometry Open3D/Filament cannot safely upload."""
        try:
            if isinstance(geometry, o3d.geometry.PointCloud):
                return len(geometry.points) == 0
            elif isinstance(geometry, o3d.geometry.LineSet):
                if len(geometry.points) == 0 or len(geometry.lines) == 0:
                    return True
                # Filament rejects geometry whose AABB collapses to a single
                # point (e.g. a stationary TX trajectory where every position
                # is identical).  Detect this early to avoid the build error.
                bbox = geometry.get_axis_aligned_bounding_box()
                extent = bbox.get_max_bound() - bbox.get_min_bound()
                if np.max(extent) < 1e-8:
                    return True
                return False
            elif isinstance(geometry, o3d.geometry.TriangleMesh):
                return len(geometry.vertices) == 0 or len(geometry.triangles) == 0
            return False
        except (RuntimeError, AttributeError):
            return False

    def _remove_geometry(self, name: str) -> bool:
        """Remove a named geometry and all related tracking metadata."""
        if self._o3d_vis is None:
            return False

        if name not in self._geometry_names:
            self._discard_geometry_tracking(name)
            return False

        try:
            self._o3d_vis.remove_geometry(name)
        except (RuntimeError, AttributeError) as exc:
            logger.debug(f"Open3DRenderer: Failed to remove geometry '{name}': {exc}")
            return False

        self._discard_geometry_tracking(name)
        logger.debug("Open3DRenderer: Removed geometry '%s'", name)
        return True

    def has_named_geometry(self, name: str) -> bool:
        """Return True if a named geometry exists in the renderer scene."""
        return name in self._geometry_names

    def get_named_geometry_names(self) -> tuple[str, ...]:
        """Return a stable snapshot of Open3D-owned geometry names."""
        return tuple(sorted(self._geometry_names))

    def is_named_visible(self, name: str) -> Optional[bool]:
        """Return named visibility, or None when the geometry is unknown."""
        if name not in self._geometry_names:
            return None
        return name not in self._hidden_geometry_names

    def set_named_visibility(self, name: str, visible: bool) -> bool:
        """Show/hide geometry and keep declarative state coherent on success."""
        if self._o3d_vis is None or name not in self._geometry_names:
            return False
        visible = bool(visible)
        pending_hidden = self._pending_hidden_intents()
        currently_visible = name not in self._hidden_geometry_names
        retry_pending_hide = not visible and name in pending_hidden
        if currently_visible == visible and not retry_pending_hide:
            if visible:
                pending_hidden.discard(name)
            # Visibility tracking is backend-owned and authoritative. Keep a
            # declarative object snapshot coherent without waking Filament or
            # scheduling a redraw for a same-value runtime update.
            cache = self._open3d_applied_objects()
            if name in cache and cache[name].visible != visible:
                cache[name] = replace(cache[name], visible=visible)
            pending_visualizer = self._pending_visualizer_visibility_updates()
            if (
                pending_visualizer.get(name) == visible
                and not self._batch_mode
                and not self._frame_update_in_progress
            ):
                self._flush_visualizer_visibility_updates()
            return True
        if not self._apply_scene_visibility(name, visible):
            if not visible:
                pending_hidden.add(name)
            return False
        if not self._stage_visualizer_visibility(name, visible):
            if not visible:
                pending_hidden.add(name)
            return False

        if visible:
            self._hidden_geometry_names.discard(name)
            pending_hidden.discard(name)
        else:
            self._hidden_geometry_names.add(name)
            pending_hidden.discard(name)
        cache = self._open3d_applied_objects()
        if name in cache:
            cache[name] = replace(cache[name], visible=visible)
        try:
            self._post_redraw()
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to request redraw for %s: %s", name, exc)
        self._request_visibility_settle_redraw(f"named visibility '{name}'")
        return True

    def add_or_update_named_geometry(
        self,
        name: str,
        geometry: o3d.geometry.Geometry,
        material: Optional[rendering.MaterialRecord] = None,
        *,
        is_edge: bool = False,
    ) -> bool:
        """Add or replace a native Open3D geometry under a renderer-stable name."""
        chosen_material = material or self._default_material_for_geometry(geometry, is_edge=is_edge)
        if self._is_label_geometry_name(name) and material is not None:
            chosen_material = self._material_for_label_geometry(
                geometry,
                requested=material,
            )
        applied = self._add_or_update_geometry(name, geometry, chosen_material)
        if not applied:
            return False
        if is_edge:
            self._edge_geometry_names.add(name)
        else:
            self._edge_geometry_names.discard(name)
        self._geometry_id_to_name[id(geometry)] = name
        return True

    def ensure_named_geometry(
        self,
        name: str,
        geometry: Any,
        material: Optional[MaterialPayload | dict[str, Any]] = None,
        transform: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        is_edge: bool = False,
    ) -> bool:
        """Ensure a named geometry exists from a native object or neutral payload."""
        native_geometry = geometry_payload_to_o3d(geometry)
        is_label = self._is_label_geometry_name(name)
        native_material = self._material_payload_for_geometry(
            native_geometry,
            material,
            is_edge=is_edge,
        )
        if is_label:
            native_material = self._material_for_label_geometry(
                native_geometry,
                requested=material,
            )
        if isinstance(native_geometry, o3d.geometry.TriangleMesh):
            color_source = self._payload_color_source(geometry)
            if is_label:
                color_source = SurfaceColorSource.VERTEX
            if color_source is SurfaceColorSource.MATERIAL and native_geometry.has_vertex_colors():
                native_geometry.paint_uniform_color([1.0, 1.0, 1.0])
        if not self.add_or_update_named_geometry(
            name=name,
            geometry=native_geometry,
            material=native_material,
            is_edge=is_edge,
        ):
            return False
        if material is not None and native_material is None:
            if not self.set_named_material(name, material):
                return False
        if transform is not None:
            if not self.set_named_transform(name, np.asarray(transform, dtype=float)):
                return False
        if visible is not None:
            if not self.set_named_visibility(name, bool(visible)):
                return False
        return True

    @staticmethod
    def _same_object_transform(previous: np.ndarray, current: np.ndarray) -> bool:
        """Return whether two declarative transforms are equivalent."""
        try:
            return bool(np.allclose(previous, current, atol=1e-9, rtol=0.0))
        except (TypeError, ValueError):
            return False

    def _effective_object_material(self, obj: RenderObject) -> MaterialPayload | None:
        """Return backend material semantics used for cache comparisons."""
        if isinstance(obj.payload, TextLabelPayload):
            return self._effective_label_material(obj.material_payload)
        return obj.material_payload

    def _applied_from_render_object(self, obj: RenderObject) -> _Open3DAppliedObject:
        """Build a strong, immutable record for a successful object ensure."""
        return _Open3DAppliedObject(
            payload=obj.payload,
            material=self._effective_object_material(obj),
            transform=np.array(obj.transform_matrix, dtype=float, copy=True),
            visible=bool(obj.visible),
            is_edge=bool(obj.is_edge),
            color_source=self._payload_color_source(obj.payload),
        )

    def _apply_full_render_object(self, obj: RenderObject) -> bool:
        """Create or replace one object and reapply all renderer-owned state."""
        geometry = obj.payload
        material = self._effective_object_material(obj)
        if isinstance(geometry, TextLabelPayload):
            color = [1.0, 1.0, 1.0] if material is None else list(material.base_color[:3])
            geometry = self._create_native_text_label(
                geometry.text,
                color=color,
                font_size=geometry.font_size,
            )
            try:
                center = np.asarray(geometry.get_center(), dtype=float).reshape(-1)
                if center.size >= 3 and np.all(np.isfinite(center[:3])):
                    geometry.translate(-center[:3])
            except (RuntimeError, AttributeError, TypeError, ValueError):
                logger.debug("Could not center neutral text label '%s'", obj.id)
        applied = bool(
            self.ensure_named_geometry(
                name=obj.id,
                geometry=geometry,
                material=material,
                transform=obj.transform_matrix,
                visible=obj.visible,
                is_edge=obj.is_edge,
            )
        )
        if applied:
            try:
                self._post_redraw()
            except (RuntimeError, AttributeError) as exc:
                logger.warning(
                    "Open3DRenderer: Failed to request redraw for %s: %s",
                    obj.id,
                    exc,
                )
            if not self._frame_update_in_progress:
                self._request_visibility_settle_redraw(
                    f"create or replace persistent geometry '{obj.id}'"
                )
        return applied

    def _restore_applied_render_object(
        self,
        object_id: str,
        state: _Open3DAppliedObject,
    ) -> bool:
        """Best-effort rollback to the last fully applied object snapshot."""
        return self._apply_full_render_object(
            RenderObject(
                id=object_id,
                payload=state.payload,
                material=state.material,
                transform=state.transform,
                visibility=state.visible,
                is_edge=state.is_edge,
            )
        )

    def ensure_object(self, obj: RenderObject) -> bool:
        """Idempotently synchronize a declarative object inside Open3D."""
        cache = self._open3d_applied_objects()
        previous = cache.get(obj.id)
        desired_material = self._effective_object_material(obj)
        geometry_changed = previous is None or (
            previous.payload is not obj.payload
            or previous.is_edge != bool(obj.is_edge)
            or previous.color_source != self._payload_color_source(obj.payload)
        )

        if geometry_changed:
            if self._apply_full_render_object(obj):
                cache[obj.id] = self._applied_from_render_object(obj)
                return True

            # A replacement may have removed the old native object before a
            # later state application failed. Remove any partial replacement
            # and restore the last complete snapshot when one exists.
            cleanup_succeeded = True
            if obj.id in self._geometry_names:
                cleanup_succeeded = self._remove_geometry(obj.id)

            # Rollback targets the previous visibility, while a failed first
            # creation retains the requested visibility for its next retry.
            hidden_intent = not bool(previous.visible if previous is not None else obj.visible)
            if hidden_intent:
                self._pending_hidden_intents().add(obj.id)

            if not cleanup_succeeded:
                # The tracked native object may be only partially applied, so
                # neither the old nor requested snapshot is truthful.
                cache.pop(obj.id, None)
                return False
            if previous is not None and self._restore_applied_render_object(obj.id, previous):
                cache[obj.id] = previous
            else:
                cache.pop(obj.id, None)
            return False

        assert previous is not None
        applied = previous
        if applied.material != desired_material:
            if desired_material is None or not self._set_object_material(
                obj.id,
                desired_material,
            ):
                return False
            applied = replace(applied, material=desired_material)
            cache[obj.id] = applied

        if not self._same_object_transform(applied.transform, obj.transform_matrix):
            if not self.set_named_transform(obj.id, obj.transform_matrix):
                return False
            applied = replace(
                applied,
                transform=np.array(obj.transform_matrix, dtype=float, copy=True),
            )
            cache[obj.id] = applied

        if applied.visible != bool(obj.visible):
            if not self.set_named_visibility(obj.id, bool(obj.visible)):
                return False
            applied = replace(applied, visible=bool(obj.visible))
            cache[obj.id] = applied

        return True

    def remove_object(self, object_id: str) -> bool:
        """Ensure a declarative object is absent, treating absence as success."""
        object_id = str(object_id)
        if object_id not in self._geometry_names:
            self._discard_geometry_tracking(object_id)
            return True
        if not self._remove_geometry(object_id):
            return False
        try:
            self._post_redraw()
        except (RuntimeError, AttributeError) as exc:
            logger.warning(
                "Open3DRenderer: Failed to request redraw for %s: %s",
                object_id,
                exc,
            )
        if not self._frame_update_in_progress:
            self._request_visibility_settle_redraw(f"remove persistent geometry '{object_id}'")
        return True

    def set_visible(self, object_id: str, visible: bool) -> bool:
        """Apply visibility through the cache-aware named adapter."""
        return self.set_named_visibility(object_id, bool(visible))

    def set_material(
        self,
        object_id: str,
        material: MaterialPayload | dict[str, Any],
    ) -> bool:
        """Apply material and keep declarative state coherent on success."""
        cache = self._open3d_applied_objects()
        applied = cache.get(object_id)
        effective_material = material_payload_from_mapping(material)
        if isinstance(
            getattr(applied, "payload", None), TextLabelPayload
        ) or self._is_label_geometry_name(object_id):
            effective_material = self._effective_label_material(effective_material)
        if not self._set_object_material(object_id, effective_material):
            return False
        if object_id in cache:
            cache[object_id] = replace(cache[object_id], material=effective_material)
        return True

    def _set_object_material(
        self,
        object_id: str,
        material: MaterialPayload | dict[str, Any],
    ) -> bool:
        """Apply a material while preserving label-specific native semantics."""
        if not self._is_label_geometry_name(object_id):
            return self.set_named_material(object_id, material)
        if self._o3d_vis is None or object_id not in self._geometry_names:
            return False

        native_material = self._material_for_label_geometry(
            None,
            requested=material,
        )
        try:
            self._o3d_vis.modify_geometry_material(object_id, native_material)
        except (RuntimeError, AttributeError) as exc:
            logger.debug(
                "Open3DRenderer: label material update failed for '%s': %s",
                object_id,
                exc,
            )
            return False

        self._pbr_materials[object_id] = native_material
        try:
            self._post_redraw()
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to request redraw for %s: %s", object_id, exc)
        if not self._frame_update_in_progress:
            self._request_visibility_settle_redraw(f"label material '{object_id}'")
        return True

    def set_transform(self, object_id: str, transform: Transform | np.ndarray) -> bool:
        """Apply transform and keep declarative state coherent on success."""
        matrix = transform.matrix if isinstance(transform, Transform) else np.asarray(transform)
        if not self.set_named_transform(object_id, matrix):
            return False
        cache = self._open3d_applied_objects()
        if object_id in cache:
            cache[object_id] = replace(
                cache[object_id],
                transform=np.array(matrix, dtype=float, copy=True),
            )
        return True

    def remove_named_geometry(self, name: str) -> bool:
        """Remove geometry by stable name."""
        return self._remove_geometry(name)

    def set_named_transform(self, name: str, transform: np.ndarray) -> bool:
        """Apply a full 4x4 scene transform to named geometry."""
        if self._o3d_vis is None or name not in self._geometry_names:
            return False
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            return False
        try:
            o3d_scene = self._o3d_vis.scene
            o3d_scene.set_geometry_transform(name, matrix)
            self._geometry_position_cache[name] = tuple(float(value) for value in matrix[:3, 3])
            self._post_redraw()
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Open3DRenderer: set_named_transform failed for '%s': %s", name, exc)
            return False

    def get_named_position(self, name: str) -> Optional[np.ndarray]:
        """Return cached world position for a named geometry if available."""
        pos = self._geometry_position_cache.get(name)
        if pos is None:
            return None
        return np.asarray(pos, dtype=float)

    def add_geometry_to_visualizer(
        self,
        geometry: o3d.geometry.Geometry,
        *,
        is_edge: bool = False,
    ) -> None:
        """Add geometry through the backend-local object-identity API.

        Shared callers use declarative objects. This compatibility helper is
        retained for backend-local tests and native integration paths.
        """
        name = f"geometry_{id(geometry)}"

        if isinstance(geometry, o3d.geometry.LineSet):
            if is_edge:
                material = self._edge_material
                self._edge_geometry_names.add(name)
            else:
                material = self._line_material
        elif isinstance(geometry, o3d.geometry.PointCloud):
            material = self._point_material
        else:
            material = self._mesh_material

        self._add_or_update_geometry(name, geometry, material)

        self._geometry_id_to_name[id(geometry)] = name

    def remove_geometry_from_visualizer(self, geometry: o3d.geometry.Geometry) -> None:
        """Remove geometry added through the object-identity compatibility API."""
        mapped_name = self._geometry_id_to_name.get(id(geometry))
        fallback_name = f"geometry_{id(geometry)}"
        if mapped_name in self._geometry_names:
            name = mapped_name
        elif fallback_name in self._geometry_names:
            name = fallback_name
        else:
            name = mapped_name or fallback_name
        self._remove_geometry(name)

        self._edge_geometry_names.discard(name)

        self._geometry_id_to_name.pop(id(geometry), None)

        self._pbr_materials.pop(name, None)

    def set_geometry_visible(self, geometry: o3d.geometry.Geometry, visible: bool) -> bool:
        """Toggle object-identity geometry visibility without re-upload.

        ``O3DVisualizer.show_geometry()`` changes visibility in place and avoids
        the GPU work required by remove/re-add.
        """
        if self._o3d_vis is None:
            return False

        name = self._geometry_id_to_name.get(id(geometry), f"geometry_{id(geometry)}")
        if name not in self._geometry_names:
            logger.debug(f"Open3DRenderer: Cannot toggle visibility - '{name}' not in scene")
            return False

        if self.set_named_visibility(name, bool(visible)):
            logger.debug(f"Open3DRenderer: Set '{name}' visible={visible}")
            return True
        return False

    def set_culling(self, enabled: bool) -> None:
        """Enable or disable view frustum culling for all geometry.

        When disabled, geometry remains visible even when its bounding box
        falls outside the camera frustum. This prevents buildings from
        popping in/out on large scenes.
        """
        self._culling_enabled = enabled
        if self._o3d_vis is None:
            return
        try:
            scene = self._o3d_vis.scene.scene
            for name in self._geometry_names:
                scene.set_geometry_culling(name, enabled)
            self._post_redraw()
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Failed to set culling: %s", exc)

    def is_geometry_in_scene(self, geometry: o3d.geometry.Geometry) -> bool:
        """Return whether object-identity geometry is in the scene."""
        name = self._geometry_id_to_name.get(id(geometry), f"geometry_{id(geometry)}")
        return name in self._geometry_names

    def set_geometry_transform_fast(
        self,
        geometry: o3d.geometry.Geometry,
        position: np.ndarray,
        rotation_matrix: Optional[np.ndarray] = None,
        mesh_center: Optional[np.ndarray] = None,
    ) -> bool:
        """Set geometry position or transform without a GPU re-upload.

        Supports two compatibility invocation styles:
        - ``set_geometry_transform_fast(geometry, transform_4x4)``
        - ``set_geometry_transform_fast(geometry, position_xyz, rotation_matrix=..., mesh_center=...)``
        """
        if self._o3d_vis is None:
            return False

        mapped_name = self._geometry_id_to_name.get(id(geometry))
        fallback_name = f"geometry_{id(geometry)}"
        if mapped_name in self._geometry_names:
            name = mapped_name
        elif fallback_name in self._geometry_names:
            name = fallback_name
        else:
            name = mapped_name or fallback_name
            logger.debug(f"Open3DRenderer: Cannot transform '{name}' - not in scene")
            return False
        if name not in self._geometry_names:
            logger.debug(f"Open3DRenderer: Cannot transform '{name}' - not in scene")
            return False

        try:
            pos_arr = np.asarray(position, dtype=np.float64)

            if pos_arr.shape == (4, 4):
                transform = pos_arr
            else:
                transform = np.eye(4)

                if rotation_matrix is not None:
                    transform[:3, :3] = rotation_matrix

                if mesh_center is not None:
                    if rotation_matrix is not None:
                        transform[:3, 3] = pos_arr - rotation_matrix @ mesh_center
                    else:
                        transform[:3, 3] = pos_arr - mesh_center
                else:
                    transform[:3, 3] = pos_arr

            o3d_scene = self._o3d_vis.scene
            if o3d_scene is not None:
                o3d_scene.set_geometry_transform(name, transform)

                logger.debug(
                    f"Open3DRenderer: Applied fast transform to '{name}', "
                    f"translation={transform[:3, 3]}"
                )
                return True
            else:
                logger.debug("Open3DRenderer: Scene not available for transform")
                return False

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug(f"Open3DRenderer: Failed to set transform for '{name}': {exc}")
            return False

    def _clear_geometry_transform(self, name: str) -> bool:
        """Reset scene transform for geometry already expressed in world space."""
        try:
            o3d_scene = self._o3d_vis.scene
            if o3d_scene is None:
                return False
            o3d_scene.set_geometry_transform(name, np.eye(4))
            return True
        except (RuntimeError, AttributeError, ValueError):
            logger.debug(
                "Open3DRenderer: Failed to clear initial node transform for '%s'",
                name,
                exc_info=True,
            )
            return False

    def update_geometry_in_visualizer(self, geometry: o3d.geometry.Geometry) -> None:
        """Re-upload object-identity geometry with its existing name.

        This is intentionally slower than ``set_geometry_transform_fast()``; use
        it only when vertex data or material ownership changed.
        """
        mapped_name = self._geometry_id_to_name.get(id(geometry))
        fallback_name = f"geometry_{id(geometry)}"
        if mapped_name in self._geometry_names:
            name = mapped_name
        elif fallback_name in self._geometry_names:
            name = fallback_name
        else:
            name = mapped_name or fallback_name

        # Prefer cached PBR material so user material changes survive
        # geometry re-uploads (e.g. after scale or position updates).
        pbr_mat = self._pbr_materials.get(name)
        if pbr_mat is not None:
            material = pbr_mat
        elif isinstance(geometry, o3d.geometry.LineSet):
            material = (
                self._edge_material if name in self._edge_geometry_names else self._line_material
            )
        elif isinstance(geometry, o3d.geometry.PointCloud):
            material = self._point_material
        else:
            material = self._mesh_material

        self._add_or_update_geometry(name, geometry, material)

    def create_sphere(
        self, radius: float, color: list[float] | None = None, resolution: int = 10
    ) -> o3d.geometry.TriangleMesh:
        """Create an Open3D sphere geometry."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        if color:
            sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere

    def create_coordinate_frame(self, size: float = 0.9) -> o3d.geometry.TriangleMesh:
        """Create an Open3D coordinate frame geometry."""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    def create_box(
        self,
        width: float,
        height: float,
        depth: float,
        color: list[float] | None = None,
    ) -> o3d.geometry.TriangleMesh:
        """Create an Open3D box geometry."""
        box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        if color:
            box.paint_uniform_color(color)
        return box

    def load_triangle_mesh(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """Load an Open3D triangle mesh from file."""
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        return mesh

    def set_geometry_vertices(
        self, geometry: o3d.geometry.TriangleMesh, vertices: np.ndarray
    ) -> None:
        """Set vertices for an Open3D geometry."""
        geometry.vertices = o3d.utility.Vector3dVector(vertices)

    def set_geometry_triangles(
        self, geometry: o3d.geometry.TriangleMesh, triangles: np.ndarray
    ) -> None:
        """Set triangles for an Open3D geometry."""
        geometry.triangles = o3d.utility.Vector3iVector(triangles)

    def translate_geometry(
        self,
        geometry: o3d.geometry.TriangleMesh,
        position: np.ndarray,
        relative: bool = False,
    ) -> None:
        """Translate an Open3D geometry."""
        geometry.translate(position, relative=relative)

    def transform_geometry(
        self, geometry: o3d.geometry.TriangleMesh, transform_matrix: np.ndarray
    ) -> None:
        """Apply a transformation matrix to an Open3D geometry."""
        geometry.transform(transform_matrix)

    def paint_geometry_uniform_color(
        self, geometry: o3d.geometry.TriangleMesh, color: list[float]
    ) -> None:
        """Paint an Open3D geometry with a uniform color."""
        geometry.paint_uniform_color(color)

    def compute_vertex_normals(self, geometry: o3d.geometry.TriangleMesh) -> None:
        """Compute Open3D vertex normals."""
        geometry.compute_vertex_normals()

    def has_vertex_normals(self, geometry: o3d.geometry.TriangleMesh) -> bool:
        """Return whether an Open3D geometry has vertex normals."""
        return geometry.has_vertex_normals()

    def create_orientation_frames(
        self, count: int, size: float = 0.9
    ) -> list[o3d.geometry.TriangleMesh]:
        """Create coordinate frame geometries for orientation visualization."""
        return [self.create_coordinate_frame(size=size) for _ in range(count)]

    def reset_coordinate_frame(self, frame: o3d.geometry.TriangleMesh, size: float = 0.9) -> None:
        """Reset a coordinate frame to its original Open3D geometry."""
        new_frame = self.create_coordinate_frame(size=size)
        self.set_geometry_vertices(frame, np.asarray(new_frame.vertices))
        self.set_geometry_triangles(frame, np.asarray(new_frame.triangles))

    def create_placeholder_label(
        self, color: list[float] | None = None
    ) -> o3d.geometry.TriangleMesh:
        """Create a placeholder label mesh."""
        return self.create_sphere(radius=0.1, color=color or [1.0, 1.0, 1.0])

    def _create_native_text_label(
        self,
        text: str,
        color: list[float] | None = None,
        font_size: float = 0.3,
    ) -> o3d.geometry.TriangleMesh:
        """Create backend-native text geometry for a neutral label payload."""
        try:
            font_size_value = float(font_size)
        except (TypeError, ValueError):
            font_size_value = 0.3
        if not np.isfinite(font_size_value) or font_size_value <= 0:
            font_size_value = 0.3

        try:
            text_mesh = self._create_text_mesh(text, depth=0.05, color=color)
            if text_mesh is None:
                raise RuntimeError("Open3D text creation returned no mesh")

            scale = font_size_value
            try:
                extent = np.asarray(
                    text_mesh.get_axis_aligned_bounding_box().get_extent(),
                    dtype=float,
                )
                if extent.size >= 2 and np.isfinite(extent[1]) and extent[1] > 1e-6:
                    target_height = max(0.05, (font_size_value / 0.3) * 0.5)
                    scale = float(target_height / extent[1])
            except (RuntimeError, AttributeError, TypeError, ValueError):
                scale = font_size_value
            scale_matrix = np.array(
                [[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]
            )
            text_mesh.transform(scale_matrix)
            text_mesh.translate([0, 0, 2.0])
            return text_mesh
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            logger.warning("Open3DRenderer: text label creation failed for '%s': %s", text, exc)

        box_scale = font_size_value / 0.3
        fallback = self.create_box(
            width=0.5 * box_scale,
            height=0.5 * box_scale,
            depth=0.1 * box_scale,
            color=color,
        )
        fallback.translate([0, 0, 2.0])
        return fallback

    def _create_text_mesh(
        self,
        text: str,
        depth: float = 0.05,
        color: list[float] | None = None,
    ) -> o3d.geometry.TriangleMesh:
        """Create text with Open3D tensor geometry and fall back to a small box."""
        try:
            text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=depth).to_legacy()
            if color:
                text_mesh.paint_uniform_color(color)
            return text_mesh
        except (RuntimeError, ValueError) as exc:
            logger.warning("Renderer: Failed to create text mesh '%s': %s", text, exc)
            fallback = self.create_box(width=0.5, height=0.5, depth=0.1, color=color)
            fallback.translate([0, 0, 2.0])
            return fallback

    def create_line_set(
        self,
        points: np.ndarray,
        lines: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> o3d.geometry.LineSet:
        """Create an Open3D LineSet from numpy arrays."""
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        if colors is not None:
            line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        return line_set

    def create_wireframe_from_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        color: Optional[np.ndarray] = None,
    ) -> Optional[o3d.geometry.LineSet]:
        """Extract a wireframe LineSet from an Open3D TriangleMesh."""
        triangles = np.asarray(mesh.triangles)
        if triangles.size == 0:
            return None
        edges = np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
            axis=0,
        )
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        if edges.size == 0:
            return None
        line_set = o3d.geometry.LineSet()
        line_set.points = mesh.vertices
        line_set.lines = o3d.utility.Vector2iVector(edges)
        if color is not None:
            edge_color = np.asarray(color, dtype=np.float64)
            line_set.colors = o3d.utility.Vector3dVector(np.tile(edge_color, (edges.shape[0], 1)))
        return line_set

    def merge_meshes(
        self,
        meshes: list[o3d.geometry.TriangleMesh],
    ) -> o3d.geometry.TriangleMesh:
        """Merge multiple Open3D meshes."""
        merged = o3d.geometry.TriangleMesh()
        for mesh in meshes:
            merged += mesh
        merged.compute_vertex_normals()
        return merged

    def set_triangle_uvs(
        self,
        mesh: Any,
        uvs: np.ndarray,
    ) -> None:
        """Assign triangle UVs to a native mesh."""
        uv_values = np.asarray(uvs, dtype=np.float64)
        if isinstance(mesh, MeshPayload):
            # Frozen payloads must be replaced by the caller before renderer upload.
            return
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_values)
