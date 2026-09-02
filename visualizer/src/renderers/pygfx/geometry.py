"""pygfx geometry payload, object, and external-adapter helpers.

This mixin is the backend-local bridge from renderer-neutral
``GeometryPayload`` objects to native pygfx world objects. It owns stable
renderer names, in-place buffer updates, backend-local compatibility/diagnostic
adapters, and small scene-control geometries such as ground grids and axes.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ...backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers
from ...model import RenderObject, Transform
from ...types.render_payloads import (
    GeometryPayload,
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    OrientationFramePayload,
    PointCloudPayload,
    SurfaceColorSource,
    TextLabelPayload,
    mesh_payload_for_pbr_material,
)
from .canvas import _env_flag, _env_float

__all__ = ["PygfxGeometryMixin"]

logger = logging.getLogger(__name__)

_BufferLayoutSignature = tuple[str, tuple[tuple[str, tuple[int, ...], str], ...]]

_VERTEX_STREAM_ARRAY_TOKEN_CACHE_LIMIT = 4096
_VERTEX_STREAM_INCOMPATIBLE_OBJECT_CACHE_LIMIT = 256
_VERTEX_STREAM_INCOMPATIBLE_TRANSITION_LIMIT = 256


class PygfxGeometryMixin:
    """Named-geometry and external-object behavior for ``PygfxRenderer``.

    Shared callers enter through the declarative object methods. Named and
    external-object helpers are backend-local implementation surfaces used to
    manage pygfx objects and compatibility inputs.
    """

    @staticmethod
    def _payload_has_texcoords(payload: MeshPayload) -> bool:
        """Return whether a mesh payload carries usable UV coordinates."""
        uvs = payload.triangle_uvs
        if uvs is None:
            return False
        uv_arr = np.asarray(uvs)
        if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
            return False
        return len(uv_arr) in {len(payload.vertices), len(payload.triangles) * 3}

    def ensure_object(self, obj: RenderObject) -> bool:
        """Idempotently synchronize a declarative object inside pygfx."""
        snapshots = getattr(self, "_render_object_snapshots", None)
        if snapshots is None:
            snapshots = {}
            self._render_object_snapshots = snapshots
        dirty_geometry = getattr(self, "_dirty_render_object_geometry", None)
        if dirty_geometry is None:
            dirty_geometry = set()
            self._dirty_render_object_geometry = dirty_geometry
        color_source = (
            obj.payload.color_source
            if isinstance(obj.payload, MeshPayload)
            else SurfaceColorSource.MATERIAL
        )
        previous = snapshots.get(obj.id)
        geometry_changed = (
            obj.id in dirty_geometry
            or previous is None
            or obj.id not in self._name_to_handle
            or previous[0] is not obj.payload
            or previous[1] != bool(obj.is_edge)
            or previous[2] != color_source
        )
        if geometry_changed:
            if isinstance(obj.payload, TextLabelPayload):
                applied = self._ensure_text_label_payload(obj.id, obj.payload)
                if applied and obj.material_payload is not None:
                    applied = self.set_named_material(obj.id, obj.material_payload)
                if applied:
                    applied = self._apply_render_object_transform(obj)
                if applied:
                    applied = self.set_named_visibility(obj.id, obj.visible)
            else:
                applied = self.ensure_named_geometry(
                    name=obj.id,
                    geometry=obj.payload,
                    material=obj.material_payload,
                    transform=obj.transform_matrix,
                    visible=obj.visible,
                    is_edge=obj.is_edge,
                )
            if applied:
                snapshots[obj.id] = (
                    obj.payload,
                    bool(obj.is_edge),
                    color_source,
                )
                dirty_geometry.discard(obj.id)
                PygfxGeometryMixin._sync_render_object_metadata(self, obj)
            else:
                # Same-layout updates mutate native buffers in place and can
                # fail after only part of the payload or its component state
                # reached pygfx. Force the next ensure through geometry sync
                # even when the caller returns to the cached payload.
                dirty_geometry.add(obj.id)
            return bool(applied)

        material_signatures = getattr(self, "_material_apply_signatures", None)
        if obj.material_payload is not None and (
            self._materials.get(obj.id) != obj.material_payload
            or (material_signatures is not None and obj.id not in material_signatures)
        ):
            if not self.set_named_material(obj.id, obj.material_payload):
                return False
        if not self._apply_render_object_transform(obj):
            return False
        if not self.set_named_visibility(obj.id, obj.visible):
            return False
        # Metadata can carry current-step semantic pose information even when
        # the mesh payload and native buffer layout stay unchanged.
        PygfxGeometryMixin._sync_render_object_metadata(self, obj)
        return True

    def update_mesh_vertex_stream(self, obj: RenderObject) -> bool:
        """Stream dynamic mesh attributes when installed topology is identical.

        This optional fast path deliberately proves all non-dynamic prepared
        buffers against pygfx's installed CPU mirrors before changing native
        state.  A rejection is not an error: target synchronization immediately
        falls back to :meth:`ensure_object`, which remains authoritative for
        topology, color/UV changes, creation, and repair.
        """
        payload = obj.payload
        if not isinstance(payload, MeshPayload):
            return False

        snapshots = getattr(self, "_render_object_snapshots", None)
        dirty_geometry = getattr(self, "_dirty_render_object_geometry", None)
        if snapshots is None or dirty_geometry is None or obj.id in dirty_geometry:
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_reject_state_count",
                1.0,
            )
            return False
        previous = snapshots.get(obj.id)
        if (
            previous is None
            or not isinstance(previous[0], MeshPayload)
            or previous[1] != bool(obj.is_edge)
            or previous[2] != payload.color_source
            or obj.id not in getattr(self, "_name_to_handle", {})
        ):
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_reject_snapshot_count",
                1.0,
            )
            return False

        previous_payload = previous[0]
        if previous_payload is payload:
            # No vertex data changed. Let the authoritative complete-object
            # path synchronize only material, transform, visibility, and
            # metadata without preparing or uploading mesh buffers.
            return False
        identity_started = time.perf_counter()
        previous_topology = self._mesh_immutable_topology_identity(previous_payload)
        desired_topology = self._mesh_immutable_topology_identity(payload)
        topology_transition = (
            (previous_topology, desired_topology)
            if previous_topology is not None and desired_topology is not None
            else None
        )
        self._record_profile_metric(
            "pygfx_mesh_vertex_stream_topology_identity_ms",
            (time.perf_counter() - identity_started) * 1000.0,
        )
        if topology_transition is not None and self._is_known_incompatible_mesh_transition(
            obj.id,
            topology_transition,
        ):
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_reject_cached_incompatible_count",
                1.0,
            )
            return False

        native_object = getattr(self, "_objects", {}).get(obj.id)
        native_geometry = getattr(native_object, "geometry", None)
        if native_geometry is None:
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_reject_native_count",
                1.0,
            )
            return False

        prepared_payload = payload
        if (
            payload.color_source is not SurfaceColorSource.VERTEX
            and payload.vertex_colors is not None
        ):
            # Match ensure_named_geometry(): PBR material-color meshes ignore
            # loader-provided vertex colors, so those buffers are neither
            # installed nor part of the fixed-topology stream identity.
            prepared_payload = mesh_payload_for_pbr_material(payload)

        prepare_started = time.perf_counter()
        try:
            prepared = self._prepare_geometry_buffers(prepared_payload)
            if prepared is None or "positions" not in prepared:
                return False
            layout = self._get_buffer_layout_signature(
                prepared_payload,
                buffers=prepared,
            )
        except Exception as exc:
            logger.debug(
                "PygfxRenderer: vertex-stream preparation failed for '%s': %s",
                obj.id,
                exc,
            )
            return False
        self._record_profile_metric(
            "pygfx_mesh_vertex_stream_prepare_ms",
            (time.perf_counter() - prepare_started) * 1000.0,
        )

        if getattr(self, "_topology", {}).get(obj.id) != layout:
            self._remember_incompatible_mesh_transition(
                obj.id,
                topology_transition,
            )
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_reject_layout_count",
                1.0,
            )
            return False

        verify_started = time.perf_counter()
        dynamic_buffer_names = {"positions", "normals", "colors"}
        for buffer_name, values in prepared.items():
            native_buffer = getattr(native_geometry, buffer_name, None)
            if native_buffer is None:
                dirty_geometry.add(obj.id)
                self._record_profile_metric(
                    f"pygfx_mesh_vertex_stream_reject_missing_{buffer_name}_count",
                    1.0,
                )
                return False
            if buffer_name not in dynamic_buffer_names:
                comparison = self._native_buffer_comparison(native_buffer, values)
                if comparison is not True:
                    if comparison is False:
                        self._remember_incompatible_mesh_transition(
                            obj.id,
                            topology_transition,
                        )
                    dirty_geometry.add(obj.id)
                    self._record_profile_metric(
                        f"pygfx_mesh_vertex_stream_reject_changed_{buffer_name}_count",
                        1.0,
                    )
                    return False
                continue
            try:
                installed = np.asarray(getattr(native_buffer, "data", None))
                desired = np.asarray(values)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                dirty_geometry.add(obj.id)
                return False
            if installed.shape != desired.shape or installed.dtype != desired.dtype:
                dirty_geometry.add(obj.id)
                self._record_profile_metric(
                    f"pygfx_mesh_vertex_stream_reject_dynamic_{buffer_name}_count",
                    1.0,
                )
                return False
        self._record_profile_metric(
            "pygfx_mesh_vertex_stream_verify_ms",
            (time.perf_counter() - verify_started) * 1000.0,
        )

        # From this point onward a partial failure may leave native vertex data
        # changed. Keep the object dirty until every component and cache commit
        # succeeds so the full path can repair it immediately or on retry.
        dirty_geometry.add(obj.id)
        push_started = time.perf_counter()
        try:
            for buffer_name in ("positions", "normals", "colors"):
                values = prepared.get(buffer_name)
                if values is None:
                    continue
                self._push_buffer(
                    getattr(native_geometry, buffer_name),
                    values,
                    label=f"mesh_vertex_stream_{buffer_name}",
                )

            material_signatures = getattr(self, "_material_apply_signatures", None)
            if obj.material_payload is not None and (
                getattr(self, "_materials", {}).get(obj.id) != obj.material_payload
                or (material_signatures is not None and obj.id not in material_signatures)
            ):
                if not self.set_named_material(obj.id, obj.material_payload):
                    return False
            if not self._apply_render_object_transform(obj):
                return False
            if not self.set_named_visibility(obj.id, obj.visible):
                return False
        except Exception as exc:
            logger.debug(
                "PygfxRenderer: vertex-stream update failed for '%s': %s",
                obj.id,
                exc,
            )
            return False
        self._record_profile_metric(
            "pygfx_mesh_vertex_stream_push_and_state_ms",
            (time.perf_counter() - push_started) * 1000.0,
        )

        snapshots[obj.id] = (
            payload,
            bool(obj.is_edge),
            payload.color_source,
        )
        # The complete declarative transform above already owns target
        # placement. Keep the installed compatibility center instead of
        # rescanning every deforming vertex solely for the legacy external
        # ``set_geometry_transform_fast(position)`` adapter; no production
        # target path uses that adapter.
        dirty_geometry.discard(obj.id)
        PygfxGeometryMixin._sync_render_object_metadata(self, obj)
        self._record_profile_metric("pygfx_mesh_vertex_stream_success_count", 1.0)
        return True

    def _immutable_array_token(self, values: np.ndarray) -> Optional[int]:
        """Return a collision-free renderer-local token for one immutable array.

        ``MeshPayload`` snapshots use bytes-backed read-only arrays. Target
        vertex replacement preserves their triangle/UV array objects, so an
        object token identifies one exact asset revision without hashing a
        large index buffer in the frame-hot path. A reloaded asset receives a
        new token and is verified normally. Weak references prevent this cache
        from retaining evicted target assets.
        """
        array = np.asarray(values)
        cache = getattr(self, "_vertex_stream_array_tokens", None)
        if cache is None:
            cache = OrderedDict()
            self._vertex_stream_array_tokens = cache
        array_id = id(array)
        cached = cache.get(array_id)
        if cached is not None and cached[0]() is array:
            cache.move_to_end(array_id)
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_topology_token_cache_hit_count",
                1.0,
            )
            return int(cached[1])

        try:
            array_ref = weakref.ref(array)
        except TypeError:
            return None
        next_token = int(getattr(self, "_vertex_stream_next_array_token", 0)) + 1
        self._vertex_stream_next_array_token = next_token
        cache[array_id] = (array_ref, next_token)
        cache.move_to_end(array_id)
        while len(cache) > _VERTEX_STREAM_ARRAY_TOKEN_CACHE_LIMIT:
            cache.popitem(last=False)
        self._record_profile_metric(
            "pygfx_mesh_vertex_stream_topology_token_cache_miss_count",
            1.0,
        )
        return next_token

    @staticmethod
    def _array_layout(values: Optional[np.ndarray]) -> Optional[tuple[int, ...]]:
        """Return the source shape relevant to prepared native buffer layout."""
        if values is None:
            return None
        return tuple(int(size) for size in np.asarray(values).shape)

    def _mesh_immutable_topology_identity(
        self,
        payload: MeshPayload,
    ) -> Optional[tuple[Any, ...]]:
        """Return an exact immutable-asset identity for topology rejection.

        This identity is used only to remember transitions already rejected by
        exact prepared/native verification. It can never approve streaming.
        Consequently cache eviction or a newly loaded equivalent asset merely
        causes another full verification, while token reuse cannot occur
        because the weak-reference entry must still point to the same array.
        """
        triangle_token = self._immutable_array_token(payload.triangles)
        if triangle_token is None:
            return None
        texcoord_token: Optional[int] = None
        texcoord_layout: Optional[tuple[int, ...]] = None
        if self._payload_has_texcoords(payload):
            if payload.triangle_uvs is None:
                return None
            texcoord_token = self._immutable_array_token(payload.triangle_uvs)
            if texcoord_token is None:
                return None
            texcoord_layout = self._array_layout(payload.triangle_uvs)

        effective_colors = (
            payload.vertex_colors if payload.color_source is SurfaceColorSource.VERTEX else None
        )
        return (
            "mesh_immutable_topology_v1",
            triangle_token,
            self._array_layout(payload.vertices),
            self._array_layout(payload.triangles),
            self._array_layout(payload.normals),
            self._array_layout(effective_colors),
            texcoord_token,
            texcoord_layout,
        )

    def _incompatible_mesh_transition_cache(
        self,
    ) -> OrderedDict[str, OrderedDict[tuple[Any, ...], None]]:
        """Return the bounded per-object incompatible-transition cache."""
        cache = getattr(self, "_vertex_stream_incompatible_transitions", None)
        if cache is None:
            cache = OrderedDict()
            self._vertex_stream_incompatible_transitions = cache
        return cache

    def _is_known_incompatible_mesh_transition(
        self,
        object_id: str,
        transition: tuple[Any, ...],
    ) -> bool:
        """Return whether exact verification rejected this asset transition."""
        cache = self._incompatible_mesh_transition_cache()
        transitions = cache.get(object_id)
        if transitions is None or transition not in transitions:
            return False
        transitions.move_to_end(transition)
        cache.move_to_end(object_id)
        return True

    def _remember_incompatible_mesh_transition(
        self,
        object_id: str,
        transition: Optional[tuple[Any, ...]],
    ) -> None:
        """Remember one exactly rejected immutable topology transition."""
        if transition is None or transition[0] == transition[1]:
            return
        cache = self._incompatible_mesh_transition_cache()
        transitions = cache.get(object_id)
        if transitions is None:
            transitions = OrderedDict()
            cache[object_id] = transitions
        learned = transition not in transitions
        transitions[transition] = None
        transitions.move_to_end(transition)
        while len(transitions) > _VERTEX_STREAM_INCOMPATIBLE_TRANSITION_LIMIT:
            transitions.popitem(last=False)
        cache.move_to_end(object_id)
        while len(cache) > _VERTEX_STREAM_INCOMPATIBLE_OBJECT_CACHE_LIMIT:
            cache.popitem(last=False)
        if learned:
            self._record_profile_metric(
                "pygfx_mesh_vertex_stream_incompatible_transition_learn_count",
                1.0,
            )

    def _forget_incompatible_mesh_transitions(self, object_id: str) -> None:
        """Drop negative topology history when a stable object is removed."""
        cache = getattr(self, "_vertex_stream_incompatible_transitions", None)
        if cache is not None:
            cache.pop(object_id, None)

    def _remove_named_geometry_for_rebuild(self, name: str) -> bool:
        """Remove native geometry while retaining learned transition history."""
        preserved = getattr(self, "_vertex_stream_rebuild_names", None)
        if preserved is None:
            preserved = set()
            self._vertex_stream_rebuild_names = preserved
        preserved.add(name)
        try:
            return self.remove_named_geometry(name)
        finally:
            preserved.discard(name)

    def _sync_render_object_metadata(self, obj: RenderObject) -> None:
        """Refresh pick metadata, native ordering, and explicit depth/pick policy.

        Render-object metadata is the narrow boundary where feature-specific
        overlays can distinguish semantic interaction proxies from purely
        decorative geometry.  Pygfx materials write pick IDs by default, so
        authoring labels and guides would otherwise obscure the handles they
        describe.  Absence of ``pickable`` preserves the backend default for
        every existing caller; an explicit value is reapplied after both
        creation and in-place reconciliation. Interaction priority also becomes
        native render order so a semantic handle wins overlapping picks.
        """

        metadata = getattr(self, "_pick_metadata", None)
        if metadata is None:
            metadata = {}
            self._pick_metadata = metadata
        if obj.metadata:
            metadata[obj.id] = dict(obj.metadata)
        else:
            metadata.pop(obj.id, None)
        native = getattr(self, "_objects", {}).get(obj.id)
        render_order = obj.metadata.get(
            "render_order",
            obj.metadata.get("interaction_priority"),
        )
        if native is not None and render_order is not None and hasattr(native, "render_order"):
            try:
                native.render_order = int(render_order)
            except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
                pass
        material = getattr(native, "material", None)
        pickable = obj.metadata.get("pickable")
        if pickable is not None and material is not None and hasattr(material, "pick_write"):
            try:
                material.pick_write = bool(pickable)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        depth_write = obj.metadata.get("depth_write")
        if depth_write is not None and material is not None and hasattr(material, "depth_write"):
            try:
                material.depth_write = bool(depth_write)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        depth_compare = obj.metadata.get("depth_compare")
        if (
            depth_compare is not None
            and material is not None
            and hasattr(material, "depth_compare")
        ):
            try:
                material.depth_compare = str(depth_compare)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    def _apply_render_object_transform(self, obj: RenderObject) -> bool:
        """Apply a render-object transform, including optional label layout."""
        if isinstance(obj.payload, TextLabelPayload):
            anchor = obj.metadata.get("layout_anchor")
            offset = obj.metadata.get("layout_offset")
            if anchor is not None and offset is not None:
                return self._register_and_layout_label(
                    obj.id,
                    np.asarray(anchor, dtype=np.float32),
                    np.asarray(offset, dtype=np.float32),
                )
        return self.set_named_transform(obj.id, obj.transform_matrix)

    def _ensure_text_label_payload(self, name: str, payload: TextLabelPayload) -> bool:
        """Create native pygfx text for one neutral label payload."""
        if not self._initialized or self._scene is None:
            return False
        if name in self._name_to_handle or name in self._objects:
            if not self.remove_named_geometry(name):
                return False

        pixel_size = max(10.0, min(48.0, float(payload.font_size) * 60.0))
        world_size = max(0.05, (float(payload.font_size) / 0.3) * 0.5)
        try:
            native = self._gfx.Text(
                text=payload.text,
                font_size=pixel_size if payload.screen_space else world_size,
                screen_space=payload.screen_space,
                anchor="middle-center",
                material=self._gfx.TextMaterial(
                    color=(1.0, 1.0, 1.0),
                    outline_color=payload.outline_color,
                    outline_thickness=payload.outline_thickness,
                    aa=True,
                ),
            )
            self._scene.add(native)
        except Exception as exc:
            logger.warning("PygfxRenderer: failed to create text label '%s': %s", name, exc)
            return False

        handle = self._allocate_handle()
        self._name_to_handle[name] = handle
        self._handle_to_name[handle] = name
        self._objects[name] = native
        self._kinds[name] = "text"
        self._reverse_objects[id(native)] = name
        self._geometry_upload_center[name] = np.zeros(3, dtype=np.float32)
        return True

    def remove_object(self, object_id: str) -> bool:
        """Ensure a declarative render object is absent by stable ID."""
        handles = getattr(self, "_name_to_handle", None)
        if handles is not None and object_id not in handles:
            getattr(self, "_render_object_snapshots", {}).pop(object_id, None)
            getattr(self, "_dirty_render_object_geometry", set()).discard(object_id)
            getattr(self, "_uncertain_mesh_index_buffers", set()).discard(object_id)
            PygfxGeometryMixin._forget_incompatible_mesh_transitions(self, object_id)
            return True
        removed = self.remove_named_geometry(object_id)
        if removed:
            PygfxGeometryMixin._forget_incompatible_mesh_transitions(self, object_id)
        return removed

    def set_visible(self, object_id: str, visible: bool) -> bool:
        """Set visibility by stable render-object ID."""
        return self.set_named_visibility(object_id, visible)

    def set_material(self, object_id: str, material: MaterialPayload | dict[str, Any]) -> bool:
        """Set material by stable render-object ID."""
        return self.set_named_material(object_id, material)

    def set_transform(self, object_id: str, transform: Transform | np.ndarray) -> bool:
        """Set transform by stable render-object ID."""
        matrix = transform.matrix if isinstance(transform, Transform) else np.asarray(transform)
        return self.set_named_transform(object_id, matrix)

    def ensure_named_geometry(
        self,
        name: str,
        geometry: GeometryPayload | Any,
        material: Optional[MaterialPayload | dict[str, Any]] = None,
        transform: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        is_edge: bool = False,
    ) -> bool:
        """Create or update one renderer-named geometry object.

        The payload's topology decides whether pygfx buffers can be updated in
        place. When topology changes, the world object is rebuilt but the stable
        name, material, transform, visibility, and external-object mapping are
        preserved where possible.
        """
        if not self._initialized:
            return False

        t_coerce_start = time.perf_counter()
        payload = self._coerce_geometry_payload(geometry)
        self._record_frame_update_metric(
            "ensure_named_geometry_payload_coerce_ms",
            (time.perf_counter() - t_coerce_start) * 1000.0,
        )
        if payload is None:
            logger.warning(
                "PygfxRenderer: unsupported geometry for '%s' (%s), skipping",
                name,
                type(geometry).__name__,
            )
            return False
        mesh_uses_vertex_colors = False
        if isinstance(payload, MeshPayload):
            if (
                payload.color_source is SurfaceColorSource.VERTEX
                and payload.vertex_colors is not None
            ):
                mesh_uses_vertex_colors = True
            else:
                if payload.vertex_colors is not None:
                    payload = mesh_payload_for_pbr_material(payload)
        texcoords_available = bool(
            isinstance(payload, MeshPayload) and self._payload_has_texcoords(payload)
        )
        upload_center = self._compute_payload_center(payload)

        line_strip = isinstance(payload, LineSetPayload) and payload.line_strip
        try:
            prepared_buffers = self._prepare_geometry_buffers(payload, line_strip=line_strip)
            new_layout = self._get_buffer_layout_signature(
                payload,
                line_strip=line_strip,
                buffers=prepared_buffers,
            )
        except Exception as exc:
            logger.warning(
                "PygfxRenderer: failed to prepare geometry buffers for '%s': %s",
                name,
                exc,
            )
            return False
        prev_material = self._materials.get(name)
        prev_transform = self._transforms.get(name)
        prev_visible = self.is_named_visible(name)

        if name in self._name_to_handle:
            old_layout = self._topology.get(name)
            if old_layout == new_layout:
                if not self._update_in_place(
                    name,
                    payload,
                    buffers=prepared_buffers,
                ):
                    return False
            else:
                # Buffer layout mismatch: must destroy and recreate the entity.
                # Preserve external-object mappings that point to this name so
                # that callers with a stale Python geometry reference can
                # still resolve it to the stable name after the rebuild.
                preserved_external = [
                    gid for gid, gname in self._external_geometry_names.items() if gname == name
                ]
                t_remove_start = time.perf_counter()
                removed = self._remove_named_geometry_for_rebuild(name)
                self._record_frame_update_metric(
                    "ensure_named_geometry_recreate_remove_ms",
                    (time.perf_counter() - t_remove_start) * 1000.0,
                )
                if not removed:
                    return False
                t_create_start = time.perf_counter()
                handle = self._create_entity(
                    name,
                    payload,
                    buffers=prepared_buffers,
                    layout_signature=new_layout,
                )
                self._record_frame_update_metric(
                    "ensure_named_geometry_create_entity_ms",
                    (time.perf_counter() - t_create_start) * 1000.0,
                )
                if handle is None:
                    return False
                for gid in preserved_external:
                    self._external_geometry_names[gid] = name
        else:
            t_create_start = time.perf_counter()
            handle = self._create_entity(
                name,
                payload,
                buffers=prepared_buffers,
                layout_signature=new_layout,
            )
            self._record_frame_update_metric(
                "ensure_named_geometry_create_entity_ms",
                (time.perf_counter() - t_create_start) * 1000.0,
            )
            if handle is None:
                return False

        # These caches describe installed native geometry. Commit them only
        # after the create/update path succeeds so failures remain retryable.
        self._geometry_texcoords_available[name] = texcoords_available
        self._geometry_upload_center[name] = upload_center
        self._geometry_color_sources[name] = (
            SurfaceColorSource.VERTEX if mesh_uses_vertex_colors else SurfaceColorSource.MATERIAL
        )

        material_to_apply = material if material is not None else prev_material
        transform_to_apply = transform if transform is not None else prev_transform
        visible_to_apply = visible if visible is not None else prev_visible

        if material_to_apply is not None:
            if not self.set_named_material(name, material_to_apply):
                return False
        if transform_to_apply is not None:
            if not self.set_named_transform(name, transform_to_apply):
                return False
        if visible_to_apply is not None:
            if not self.set_named_visibility(name, bool(visible_to_apply)):
                return False
        if is_edge:
            self._edge_geometry_names.add(name)
        self._apply_named_visual_overrides(name, is_edge=is_edge)

        return True

    def remove_named_geometry(self, name: str) -> bool:
        """Remove a named world object and all renderer-side bookkeeping."""
        handle = self._name_to_handle.get(name)
        obj = self._objects.get(name)
        if handle is None and obj is None:
            return False

        parent = None
        if obj is not None:
            parent = (
                self._static_group
                if self._is_scene_mesh_name(name) and self._static_group is not None
                else self._scene
            )
            if parent is not None:
                try:
                    children = getattr(parent, "children", None)
                    is_attached = children is None or any(child is obj for child in children)
                    if is_attached:
                        parent.remove(obj)
                except Exception as exc:
                    logger.debug(
                        "PygfxRenderer: failed to remove native geometry '%s': %s",
                        name,
                        exc,
                    )
                    return False

        # Commit bookkeeping only after native removal succeeds so a failed
        # call remains observable and retryable by backend-local synchronizers.
        getattr(self, "_render_object_snapshots", {}).pop(name, None)
        getattr(self, "_dirty_render_object_geometry", set()).discard(name)
        getattr(self, "_uncertain_mesh_index_buffers", set()).discard(name)
        if name not in getattr(self, "_vertex_stream_rebuild_names", set()):
            PygfxGeometryMixin._forget_incompatible_mesh_transitions(self, name)
        self._unregister_label_layout(name)
        self._edge_geometry_names.discard(name)
        self._name_to_handle.pop(name, None)
        self._objects.pop(name, None)
        if handle is not None:
            self._handle_to_name.pop(handle, None)

        self._kinds.pop(name, None)
        self._topology.pop(name, None)
        self._hidden.discard(name)
        self._geometry_color_sources.pop(name, None)
        self._materials.pop(name, None)
        self._material_apply_signatures.pop(name, None)
        self._transforms.pop(name, None)
        self._positions.pop(name, None)
        self._geometry_upload_center.pop(name, None)
        self._geometry_texcoords_available.pop(name, None)
        self._pick_metadata.pop(name, None)
        if obj is not None:
            self._reverse_objects.pop(id(obj), None)
        self._external_remove_name(name)
        if name == "mpc_lines":
            self._clear_mpc_segment_capacity()
        elif name == "mpc_points":
            self._clear_mpc_point_capacity()

        overlay = self._normal_line_overlays.pop(name, None)
        if overlay is not None:
            try:
                parent_overlay = (
                    self._static_group
                    if self._is_scene_mesh_name(name) and self._static_group is not None
                    else self._scene
                )
                if parent_overlay is not None:
                    parent_overlay.remove(overlay)
            except Exception:
                pass

        return True

    def has_named_geometry(self, name: str) -> bool:
        """Return whether a stable named pygfx object exists."""
        return name in self._name_to_handle

    def get_named_geometry_names(self) -> tuple[str, ...]:
        """Return a stable snapshot of pygfx-owned geometry names."""
        return tuple(sorted(self._name_to_handle))

    def is_named_visible(self, name: str) -> Optional[bool]:
        """Return tracked visibility for a stable named pygfx object."""
        if name not in self._name_to_handle:
            return None
        return name not in self._hidden

    def set_named_visibility(self, name: str, visible: bool) -> bool:
        """Synchronize tracked visibility and native pygfx object visibility."""
        obj = self._objects.get(name)
        if obj is None:
            return False

        desired_visible = bool(visible)
        tracked_visible = self.is_named_visible(name)
        actual_visible = getattr(obj, "visible", tracked_visible)
        if actual_visible is not None:
            actual_visible = bool(actual_visible)
        if tracked_visible is desired_visible and actual_visible is desired_visible:
            return True

        try:
            obj.visible = desired_visible
        except Exception as exc:
            logger.warning(
                "PygfxRenderer: failed to set visibility for '%s' to %s: %s",
                name,
                desired_visible,
                exc,
            )
            return False

        if desired_visible:
            self._hidden.discard(name)
        else:
            self._hidden.add(name)
        self.request_redraw()
        return True

    def set_named_transform(self, name: str, transform: np.ndarray) -> bool:
        """Apply a 4x4 world transform while retaining cached logical position."""
        obj = self._objects.get(name)
        if obj is None:
            return False

        mat = np.asarray(transform, dtype=np.float32)
        if mat.shape != (4, 4):
            logger.warning(
                "PygfxRenderer: transform for '%s' has shape %s, expected (4,4)",
                name,
                mat.shape,
            )
            return False

        current = self._transforms.get(name)
        if (
            current is not None
            and current.shape == (4, 4)
            and np.allclose(current, mat, atol=1e-6, rtol=0.0)
        ):
            return True

        applied = False
        failures: list[str] = []
        try:
            local = getattr(obj, "local", None)
            if local is not None and hasattr(local, "matrix"):
                local.matrix = mat
                applied = True
        except Exception as exc:
            failures.append(f"matrix assignment: {exc}")

        translation_only = bool(
            np.allclose(mat[:3, :3], np.eye(3, dtype=np.float32), atol=1e-6)
            and np.allclose(mat[3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
        )
        if not applied and translation_only:
            try:
                local = getattr(obj, "local", None)
                if local is not None and hasattr(local, "position"):
                    local.position = tuple(float(x) for x in mat[:3, 3])
                    applied = True
            except Exception as exc:
                failures.append(f"position assignment: {exc}")

        if not applied:
            detail = "; ".join(failures) or "no writable native transform"
            if not translation_only:
                detail = f"rotation requires matrix assignment; {detail}"
            logger.warning("PygfxRenderer: could not apply transform for '%s': %s", name, detail)
            return False

        self._transforms[name] = mat
        self._positions[name] = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
        return True

    def get_named_position(self, name: str) -> Optional[np.ndarray]:
        """Return cached world position for a stable named pygfx object."""
        pos = self._positions.get(name)
        if pos is None:
            return None
        return np.asarray(pos, dtype=np.float64)

    def get_ground_grid_visible(self) -> bool:
        """Return whether the optional pygfx ground grid is visible."""
        return self._ground_grid_visible

    def set_ground_grid_visible(self, visible: bool) -> None:
        """Toggle an axis-aligned ground-plane grid at z = scene floor."""
        new = bool(visible)
        if (
            new == self._ground_grid_visible
            and self._ground_grid_obj is not None
            and not self._ground_grid_needs_rebuild
        ):
            return
        if new and (self._ground_grid_obj is None or self._ground_grid_needs_rebuild):
            self._remove_ground_grid()
            self._build_ground_grid()
        if self._ground_grid_obj is not None:
            try:
                self._ground_grid_obj.visible = new
            except Exception as exc:
                logger.debug("Failed to toggle ground grid visibility: %s", exc)
        self._ground_grid_visible = new
        self._update_status_chip_overlay()
        self.request_redraw()

    def _remove_ground_grid(self) -> None:
        """Remove the renderer-owned ground grid object from the scene."""
        grid = self._ground_grid_obj
        if grid is None:
            return
        try:
            if self._scene is not None:
                self._scene.remove(grid)
        except Exception:
            pass
        self._ground_grid_obj = None

    def _build_ground_grid(self) -> None:
        """Create a GridHelper sized to the current scene bounds."""
        gfx = self._gfx
        if gfx is None or self._scene is None:
            return
        bounds = self.compute_scene_bounds()
        if bounds is None:
            size = max(self._scene_extent * 2.0, 20.0)
            z_floor = 0.0
        else:
            extent = np.max(bounds.get_extent())
            size = float(max(extent * 1.2, 20.0))
            z_floor = float(bounds.min_bound[2])
        divisions = 20
        try:
            grid = gfx.GridHelper(
                size=size,
                divisions=divisions,
                color1=(0.45, 0.45, 0.45, 1.0),
                color2=(0.25, 0.25, 0.25, 1.0),
                thickness=1.0,
            )
        except Exception as exc:
            logger.debug("Failed to build GridHelper: %s", exc)
            return
        # GridHelper is XZ-oriented in pygfx (three.js convention), so it must
        # be rotated about X and translated into the XY plane at z = floor.
        try:
            transform = np.eye(4, dtype=np.float64)
            # Rotate +90° about X: y-axis -> z, z-axis -> -y
            transform[1, 1] = 0.0
            transform[2, 2] = 0.0
            transform[1, 2] = -1.0
            transform[2, 1] = 1.0
            transform[2, 3] = z_floor
            grid.local.matrix = transform
        except Exception as exc:
            logger.debug("Failed to orient ground grid: %s", exc)
        try:
            self._scene.add(grid)
        except Exception as exc:
            logger.debug("Failed to add ground grid to scene: %s", exc)
            return
        self._ground_grid_obj = grid
        self._ground_grid_needs_rebuild = False

    def _ensure_ground_grid_current(self) -> None:
        """Rebuild the ground grid when visible and marked stale."""
        if not self._ground_grid_visible:
            return
        if self._ground_grid_obj is not None and not self._ground_grid_needs_rebuild:
            return
        self._remove_ground_grid()
        self._build_ground_grid()
        if self._ground_grid_obj is not None:
            try:
                self._ground_grid_obj.visible = True
            except Exception:
                pass

    def _get_entity_handle(self, name: str) -> Optional[int]:
        """Return the internal handle assigned to a stable name."""
        return self._name_to_handle.get(name)

    def _allocate_handle(self) -> int:
        """Allocate the next monotonic handle for renderer bookkeeping."""
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def _get_buffer_sizes(
        self, geometry: GeometryPayload, *, line_strip: bool = False
    ) -> tuple[int, int]:
        """Return topology sizes used to choose update-in-place vs rebuild."""
        if isinstance(geometry, MeshPayload):
            return (len(geometry.vertices), len(geometry.triangles) * 3)
        if isinstance(geometry, LineSetPayload):
            if line_strip or len(geometry.lines) == 0:
                return (len(geometry.points), len(geometry.points))
            return (len(geometry.lines) * 2, len(geometry.lines) * 2)
        if isinstance(geometry, PointCloudPayload):
            return (len(geometry.points), len(geometry.points))
        if isinstance(geometry, OrientationFramePayload):
            size_key = int(round(max(float(geometry.size), 0.0) * 1000.0))
            thickness_key = int(round(max(float(geometry.thickness), 0.0) * 1000.0))
            return (size_key, thickness_key)
        return (0, 0)

    def _prepare_geometry_buffers(
        self,
        geometry: GeometryPayload,
        *,
        line_strip: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        """Return the exact native buffer arrays for a geometry payload.

        Mesh UV seams can change the pygfx position/index layout, so renderer
        synchronization must compare the prepared buffers rather than only the
        source vertex and triangle counts. The same arrays are then reused by
        either the in-place update or reconstruction path.
        """
        if isinstance(geometry, MeshPayload):
            if self._payload_has_texcoords(geometry):
                return mesh_payload_to_pygfx_buffers(geometry)
            mesh_buffers = {
                "positions": self._ensure_contiguous(geometry.vertices, np.float32),
                "indices": self._ensure_contiguous(geometry.triangles, np.int32),
            }
            if geometry.normals is not None:
                mesh_buffers["normals"] = self._ensure_contiguous(
                    geometry.normals,
                    np.float32,
                )
            if geometry.vertex_colors is not None:
                mesh_buffers["colors"] = self._ensure_contiguous(
                    geometry.vertex_colors,
                    np.float32,
                )
            return mesh_buffers
        if isinstance(geometry, LineSetPayload):
            return self._line_geometry_kwargs(geometry, line_strip=line_strip)
        if isinstance(geometry, PointCloudPayload):
            buffers: dict[str, np.ndarray] = {
                "positions": self._ensure_contiguous(geometry.points, np.float32),
            }
            if geometry.colors is not None:
                buffers["colors"] = self._ensure_contiguous(
                    geometry.colors,
                    np.float32,
                )
            return buffers
        return None

    def _get_buffer_layout_signature(
        self,
        geometry: GeometryPayload,
        *,
        line_strip: bool = False,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> _BufferLayoutSignature:
        """Return native buffer presence, shape, and dtype for synchronization."""
        prepared = (
            buffers
            if buffers is not None
            else self._prepare_geometry_buffers(geometry, line_strip=line_strip)
        )
        if prepared is None:
            if isinstance(geometry, OrientationFramePayload):
                topology = self._get_buffer_sizes(geometry, line_strip=line_strip)
                return (
                    "orientation_frame",
                    (("parameters", tuple(topology), "float"),),
                )
            return (type(geometry).__name__, ())

        layout = tuple(
            (
                name,
                tuple(int(size) for size in np.asarray(values).shape),
                np.asarray(values).dtype.str,
            )
            for name, values in sorted(prepared.items())
        )
        return (type(geometry).__name__, layout)

    @staticmethod
    def _compute_payload_center(geometry: GeometryPayload) -> np.ndarray:
        """Return a finite local-space center for transform helpers.

        Non-finite vertices can appear in partially populated diagnostic
        payloads; they are ignored so transform math never propagates NaNs into
        pygfx object matrices.
        """
        if isinstance(geometry, MeshPayload):
            points = np.asarray(geometry.vertices, dtype=np.float32)
        elif isinstance(geometry, LineSetPayload):
            points = np.asarray(geometry.points, dtype=np.float32)
        elif isinstance(geometry, PointCloudPayload):
            points = np.asarray(geometry.points, dtype=np.float32)
        elif isinstance(geometry, OrientationFramePayload):
            return np.zeros(3, dtype=np.float32)
        else:
            return np.zeros(3, dtype=np.float32)

        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return np.zeros(3, dtype=np.float32)

        pts3 = points[:, :3]
        finite_rows = np.all(np.isfinite(pts3), axis=1)
        if not np.any(finite_rows):
            return np.zeros(3, dtype=np.float32)
        pts3 = pts3[finite_rows]
        mins = np.min(pts3, axis=0)
        maxs = np.max(pts3, axis=0)
        center = (mins + maxs) * 0.5
        if not np.all(np.isfinite(center)):
            return np.zeros(3, dtype=np.float32)
        return np.asarray(center, dtype=np.float32)

    def _get_parent_for(self, name: str) -> Any:
        """Return the scene sub-graph parent for *name*.

        Static scene meshes are placed in a shared ``gfx.Group`` so the
        renderer can batch their draw calls.
        """
        if self._is_scene_mesh_name(name) and self._scene is not None:
            if self._static_group is None:
                self._static_group = self._gfx.Group()
                self._scene.add(self._static_group)
            return self._static_group
        return self._scene

    def _create_entity(
        self,
        name: str,
        geometry: GeometryPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
        layout_signature: Optional[_BufferLayoutSignature] = None,
    ) -> Optional[int]:
        """Build, attach, and register one pygfx world object."""
        t_build_start = time.perf_counter()
        obj, kind = self._build_world_object(
            geometry,
            name=name,
            buffers=buffers,
        )
        self._record_frame_update_metric(
            "create_entity_build_world_object_ms",
            (time.perf_counter() - t_build_start) * 1000.0,
        )
        if obj is None:
            return None

        parent = self._get_parent_for(name)
        try:
            t_add_start = time.perf_counter()
            if parent is not None:
                parent.add(obj)
            self._record_frame_update_metric(
                "create_entity_scene_add_ms",
                (time.perf_counter() - t_add_start) * 1000.0,
            )
        except Exception as exc:
            logger.warning("PygfxRenderer: failed to add object '%s': %s", name, exc)
            return None

        handle = self._allocate_handle()
        self._name_to_handle[name] = handle
        self._handle_to_name[handle] = name
        self._objects[name] = obj
        self._kinds[name] = kind
        line_strip = isinstance(geometry, LineSetPayload) and geometry.line_strip
        self._topology[name] = layout_signature or self._get_buffer_layout_signature(
            geometry,
            line_strip=line_strip,
            buffers=buffers,
        )
        self._reverse_objects[id(obj)] = name
        if isinstance(geometry, MeshPayload):
            getattr(self, "_uncertain_mesh_index_buffers", set()).discard(name)

        self._apply_shadow_flags(name, obj)

        if (
            kind == "mesh"
            and parent is not None
            and _env_flag("ORCHAV_PYGFX_SHOW_NORMAL_LINES", False)
            and self._is_scene_mesh_name(name)
        ):
            self._attach_normal_lines_overlay(name, obj, parent)

        return handle

    def _attach_normal_lines_overlay(self, name: str, obj: Any, parent: Any) -> None:
        """Attach a sibling MeshNormalLinesMaterial overlay to a scene mesh.

        Gated by ``ORCHAV_PYGFX_SHOW_NORMAL_LINES=1`` and applied only to
        scene meshes (not targets/TX/RX/overlays). Useful when a specular
        bounce goes somewhere unexpected and you want to confirm whether
        the mesh normal is flipped. No UI — this is a debug-only flag.
        """
        gfx = self._gfx
        cls = getattr(gfx, "MeshNormalLinesMaterial", None)
        if cls is None:
            return
        try:
            length = _env_float("ORCHAV_PYGFX_NORMAL_LINES_LENGTH", 1.0)
            debug_mat = cls(line_length=length)
        except Exception as exc:
            logger.debug("Failed to build MeshNormalLinesMaterial: %s", exc)
            return
        try:
            debug_mesh = gfx.Mesh(obj.geometry, debug_mat)
            parent.add(debug_mesh)
            self._normal_line_overlays[name] = debug_mesh
        except Exception as exc:
            logger.debug("Failed to attach normal lines overlay for '%s': %s", name, exc)

    def _build_world_object(
        self,
        geometry: GeometryPayload,
        *,
        name: Optional[str] = None,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[Optional[Any], str]:
        """Build the native pygfx object and return its renderer kind."""
        gfx = self._gfx

        try:
            if isinstance(geometry, MeshPayload):
                geom = self._build_mesh_geometry(geometry, buffers=buffers)
                has_colors = geometry.vertex_colors is not None
                material = self._build_mesh_material(has_vertex_colors=has_colors)
                return gfx.Mesh(geom, material), "mesh"

            if isinstance(geometry, LineSetPayload):
                line_strip = geometry.line_strip
                geom = self._build_lines_geometry(
                    geometry,
                    line_strip=line_strip,
                    buffers=buffers,
                )
                has_colors = geometry.colors is not None
                material = self._build_line_material(
                    has_vertex_colors=has_colors,
                    line_strip=line_strip or len(geometry.lines) == 0,
                    pick_write=not self._is_orientation_frame_name(name),
                )
                return gfx.Line(geom, material), "lines"

            if isinstance(geometry, PointCloudPayload):
                geom = self._build_points_geometry(geometry, buffers=buffers)
                has_colors = geometry.colors is not None
                material = self._build_points_material(has_vertex_colors=has_colors)
                return gfx.Points(geom, material), "points"

            if isinstance(geometry, OrientationFramePayload):
                axes_helper = getattr(gfx, "AxesHelper", None)
                if axes_helper is None:
                    logger.warning("PygfxRenderer: AxesHelper unavailable for orientation frame")
                    return None, ""
                return (
                    axes_helper(
                        size=max(float(geometry.size), 0.0),
                        thickness=max(float(geometry.thickness), 0.0),
                    ),
                    "orientation_frame",
                )
        except Exception as exc:
            logger.warning("PygfxRenderer: failed to build object: %s", exc)

        return None, ""

    @staticmethod
    def _is_orientation_frame_name(name: Optional[str]) -> bool:
        """Return whether the name identifies a non-pickable orientation frame."""
        return bool(name and name.endswith("::orientation_frame"))

    @staticmethod
    def _ensure_contiguous(arr: Any, dtype: np.dtype) -> np.ndarray:
        """Return *arr* as a C-contiguous ndarray of *dtype*, avoiding a copy when possible."""
        a = np.asarray(arr)
        if a.dtype == dtype and a.flags["C_CONTIGUOUS"]:
            return a
        return np.ascontiguousarray(a, dtype=dtype)

    @staticmethod
    def _writable_backend_array(arr: Any) -> np.ndarray:
        """Detach immutable payload memory only when a backend buffer needs it."""
        values = np.asarray(arr)
        if values.flags.c_contiguous and values.flags.writeable and values.flags.owndata:
            return values
        return np.array(values, copy=True, order="C")

    def _build_mesh_geometry(
        self,
        geometry: MeshPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> Any:
        """Convert a neutral mesh payload into a pygfx Geometry object."""
        gfx = self._gfx
        t_buffers_start = time.perf_counter()
        kwargs = buffers if buffers is not None else mesh_payload_to_pygfx_buffers(geometry)
        self._record_frame_update_metric(
            "build_mesh_geometry_buffers_ms",
            (time.perf_counter() - t_buffers_start) * 1000.0,
        )
        t_geometry_start = time.perf_counter()
        geom = gfx.Geometry(
            **{name: self._writable_backend_array(values) for name, values in kwargs.items()}
        )
        self._record_frame_update_metric(
            "build_mesh_geometry_gfx_geometry_ms",
            (time.perf_counter() - t_geometry_start) * 1000.0,
        )
        return geom

    def _line_geometry_kwargs(
        self, geometry: LineSetPayload, *, line_strip: bool = False
    ) -> dict[str, Any]:
        """Convert neutral line payloads into pygfx geometry kwargs.

        Indexed line sets are expanded to endpoint pairs because pygfx segment
        materials render independent segments from position order, not from an
        index buffer.
        """
        positions = self._ensure_contiguous(geometry.points, np.float32)
        lines = self._ensure_contiguous(geometry.lines, np.int32)
        colors = (
            self._ensure_contiguous(geometry.colors, np.float32)
            if geometry.colors is not None
            else None
        )

        if len(lines) > 0 and not line_strip:
            # Pygfx LineSegmentMaterial expects disjoint endpoint pairs. Expand
            # indexed line sets so independent mesh edges cannot be connected
            # by the order of the shared vertex buffer.
            n_segs = len(lines)
            seg_points = np.empty((n_segs * 2, 3), dtype=np.float32)
            seg_points[0::2] = positions[lines[:, 0]]
            seg_points[1::2] = positions[lines[:, 1]]

            kwargs: dict[str, Any] = {"positions": seg_points}
            if colors is not None:
                if len(colors) == len(positions):
                    seg_colors = np.empty((n_segs * 2, colors.shape[-1]), dtype=np.float32)
                    seg_colors[0::2] = colors[lines[:, 0]]
                    seg_colors[1::2] = colors[lines[:, 1]]
                else:
                    seg_colors = np.empty((n_segs * 2, colors.shape[-1]), dtype=np.float32)
                    seg_colors[0::2] = colors[:n_segs]
                    seg_colors[1::2] = colors[:n_segs]
                kwargs["colors"] = seg_colors
            return kwargs

        kwargs: dict[str, Any] = {"positions": positions}
        if colors is not None:
            kwargs["colors"] = colors
        return kwargs

    def _build_lines_geometry(
        self,
        geometry: LineSetPayload,
        *,
        line_strip: bool = False,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> Any:
        """Convert a neutral line payload into a pygfx Geometry object."""
        gfx = self._gfx
        kwargs = {
            name: self._writable_backend_array(values)
            for name, values in (
                buffers
                if buffers is not None
                else self._line_geometry_kwargs(
                    geometry,
                    line_strip=line_strip,
                )
            ).items()
        }
        return gfx.Geometry(**kwargs)

    def _build_points_geometry(
        self,
        geometry: PointCloudPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> Any:
        """Convert a neutral point-cloud payload into a pygfx Geometry object."""
        gfx = self._gfx
        prepared = buffers if buffers is not None else self._prepare_geometry_buffers(geometry)
        if prepared is None:
            raise TypeError("point payload did not produce native buffers")
        kwargs: dict[str, Any] = {
            name: self._writable_backend_array(values) for name, values in prepared.items()
        }
        return gfx.Geometry(**kwargs)

    def _update_in_place(
        self,
        name: str,
        geometry: GeometryPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> bool:
        """Update existing pygfx geometry buffers directly from numpy arrays.

        Avoids creating an intermediate ``gfx.Geometry`` object — writes
        straight into the existing GPU-side buffers, halving data movement.
        """
        obj = self._objects.get(name)
        if obj is None:
            return False

        old_geom = getattr(obj, "geometry", None)
        if old_geom is None:
            return isinstance(geometry, OrientationFramePayload)
        prepared = (
            buffers
            if buffers is not None
            else self._prepare_geometry_buffers(
                geometry,
                line_strip=(isinstance(geometry, LineSetPayload) and geometry.line_strip),
            )
        )

        try:
            if isinstance(geometry, MeshPayload):
                self._update_buffers_mesh(
                    old_geom,
                    geometry,
                    name=name,
                    buffers=prepared,
                )
                self._sync_color_mode(obj, bool(prepared and "colors" in prepared))
            elif isinstance(geometry, LineSetPayload):
                self._update_buffers_lines(
                    old_geom,
                    geometry,
                    line_strip=geometry.line_strip,
                    buffers=prepared,
                )
                self._sync_color_mode(obj, bool(prepared and "colors" in prepared))
            elif isinstance(geometry, PointCloudPayload):
                self._update_buffers_points(old_geom, geometry, buffers=prepared)
                self._sync_color_mode(obj, bool(prepared and "colors" in prepared))
            elif isinstance(geometry, OrientationFramePayload):
                return True
            else:
                return False
            return True
        except Exception as exc:
            logger.debug(
                "PygfxRenderer: in-place buffer update failed for '%s': %s — rebuilding",
                name,
                exc,
            )
            return self._update_in_place_fallback(
                name,
                obj,
                geometry,
                buffers=prepared,
            )

    def _push_buffer(self, buf: Any, data: np.ndarray, *, label: str = "buffer") -> None:
        """Write *data* into an existing pygfx Buffer and mark it dirty."""
        t_start = time.perf_counter()
        buf_data = getattr(buf, "data", None)
        if buf_data is not None and buf_data.shape == data.shape and buf_data.dtype == data.dtype:
            buf_data[:] = data
            if hasattr(buf, "update_full"):
                buf.update_full()
        elif hasattr(buf, "set_data"):
            buf.set_data(data)
            if hasattr(buf, "update_full"):
                buf.update_full()
        else:
            raise AttributeError("buffer has neither writable .data nor .set_data()")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        nbytes = float(getattr(data, "nbytes", 0))
        safe_label = label.replace("::", "_").replace(" ", "_")
        self._record_profile_metric("pygfx_push_buffer_ms", elapsed_ms)
        self._record_profile_metric(f"pygfx_push_buffer_{safe_label}_ms", elapsed_ms)
        self._record_profile_bytes("pygfx_push_buffer_bytes", nbytes)
        self._record_profile_bytes(f"pygfx_push_buffer_{safe_label}_bytes", nbytes)

    def _set_buffer_draw_range(self, buf: Any, count: int) -> None:
        """Limit a reusable pygfx buffer to the active element prefix."""
        if buf is None or not hasattr(buf, "draw_range"):
            return
        try:
            buf.draw_range = (0, max(0, int(count)))
        except Exception:
            pass

    def _push_buffer_prefix(
        self,
        buf: Any,
        data: np.ndarray,
        *,
        label: str = "buffer",
    ) -> None:
        """Write the active prefix of *data* into a larger pygfx Buffer."""
        t_start = time.perf_counter()
        data = np.asarray(data)
        buf_data = getattr(buf, "data", None)
        if (
            buf_data is None
            or buf_data.dtype != data.dtype
            or buf_data.ndim != data.ndim
            or buf_data.shape[0] < data.shape[0]
            or buf_data.shape[1:] != data.shape[1:]
        ):
            self._push_buffer(buf, data, label=label)
            return

        active = int(data.shape[0])
        if active:
            buf_data[:active] = data
        self._set_buffer_draw_range(buf, active)
        if hasattr(buf, "update_range"):
            try:
                buf.update_range(0, active)
            except Exception:
                if hasattr(buf, "update_full"):
                    buf.update_full()
        elif hasattr(buf, "update_full"):
            buf.update_full()

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        nbytes = float(getattr(data, "nbytes", 0))
        safe_label = label.replace("::", "_").replace(" ", "_")
        self._record_profile_metric("pygfx_push_buffer_prefix_ms", elapsed_ms)
        self._record_profile_metric(f"pygfx_push_buffer_prefix_{safe_label}_ms", elapsed_ms)
        self._record_profile_bytes("pygfx_push_buffer_prefix_bytes", nbytes)
        self._record_profile_bytes(f"pygfx_push_buffer_prefix_{safe_label}_bytes", nbytes)

    def _push_geometry_buffers(
        self,
        geom: Any,
        buffers: dict[str, np.ndarray],
        *,
        label_prefix: str,
    ) -> None:
        """Push every prepared buffer, failing if native layout is inconsistent."""
        for buffer_name, values in buffers.items():
            native_buffer = getattr(geom, buffer_name, None)
            if native_buffer is None:
                raise AttributeError(f"native geometry has no '{buffer_name}' buffer")
            self._push_buffer(
                native_buffer,
                values,
                label=f"{label_prefix}_{buffer_name}",
            )

    @staticmethod
    def _native_buffer_comparison(
        native_buffer: Any,
        desired: np.ndarray,
    ) -> Optional[bool]:
        """Compare native CPU data exactly, returning ``None`` when unreadable.

        Shape alone is not a topology identity: two equally sized index arrays
        can describe different connectivity.  Comparing the installed native
        CPU mirror to the already-prepared pygfx array also covers UV seam
        expansion, whose indices can differ from the source mesh triangles.
        """
        try:
            installed = getattr(native_buffer, "data", None)
            if installed is None:
                return None
            installed_array = np.asarray(installed)
            desired_array = np.asarray(desired)
            if (
                installed_array.shape != desired_array.shape
                or installed_array.dtype != desired_array.dtype
            ):
                return False
            return bool(np.array_equal(installed_array, desired_array))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _native_buffer_matches(native_buffer: Any, desired: np.ndarray) -> bool:
        """Return whether native CPU data exactly matches one prepared buffer."""
        return PygfxGeometryMixin._native_buffer_comparison(native_buffer, desired) is True

    def _update_buffers_mesh(
        self,
        geom: Any,
        payload: MeshPayload,
        *,
        name: Optional[str] = None,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Update mesh buffers while retaining an identical native index buffer."""
        prepared = buffers if buffers is not None else self._prepare_geometry_buffers(payload)
        if prepared is None:
            raise TypeError("mesh payload did not produce native buffers")
        uncertain_names = getattr(self, "_uncertain_mesh_index_buffers", None)
        if uncertain_names is None:
            uncertain_names = set()
            self._uncertain_mesh_index_buffers = uncertain_names

        for buffer_name, values in prepared.items():
            native_buffer = getattr(geom, buffer_name, None)
            if native_buffer is None:
                raise AttributeError(f"native geometry has no '{buffer_name}' buffer")
            if (
                buffer_name == "indices"
                and name is not None
                and name not in uncertain_names
                and self._native_buffer_matches(native_buffer, values)
            ):
                continue

            if buffer_name == "indices" and name is not None:
                # Mark the topology uncertain before mutating native CPU data.
                # If update_full() raises after the copy, the next retry must
                # upload the index buffer even when the CPU mirrors compare
                # equal, because GPU synchronization was not confirmed.
                uncertain_names.add(name)
            self._push_buffer(
                native_buffer,
                values,
                label=f"mesh_{buffer_name}",
            )
            if buffer_name == "indices" and name is not None:
                uncertain_names.discard(name)

    def _update_buffers_lines(
        self,
        geom: Any,
        payload: LineSetPayload,
        *,
        line_strip: bool = False,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Update existing pygfx line buffers from a same-topology payload."""
        t_kwargs = time.perf_counter()
        kwargs = (
            buffers
            if buffers is not None
            else self._line_geometry_kwargs(payload, line_strip=line_strip)
        )
        self._record_profile_metric(
            "pygfx_update_lines_geometry_kwargs_ms",
            (time.perf_counter() - t_kwargs) * 1000.0,
        )
        self._push_geometry_buffers(geom, kwargs, label_prefix="line")

    def _update_buffers_points(
        self,
        geom: Any,
        payload: PointCloudPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Update existing pygfx point buffers from a same-topology payload."""
        prepared = buffers if buffers is not None else self._prepare_geometry_buffers(payload)
        if prepared is None:
            raise TypeError("point payload did not produce native buffers")
        self._push_geometry_buffers(geom, prepared, label_prefix="point")

    def _update_in_place_fallback(
        self,
        name: str,
        obj: Any,
        geometry: GeometryPayload,
        *,
        buffers: Optional[dict[str, np.ndarray]] = None,
    ) -> bool:
        """Fallback: rebuild geometry object while preserving world-object identity."""
        try:
            if isinstance(geometry, MeshPayload):
                obj.geometry = self._build_mesh_geometry(geometry, buffers=buffers)
                getattr(self, "_uncertain_mesh_index_buffers", set()).discard(name)
            elif isinstance(geometry, LineSetPayload):
                obj.geometry = self._build_lines_geometry(
                    geometry,
                    line_strip=geometry.line_strip,
                    buffers=buffers,
                )
            elif isinstance(geometry, PointCloudPayload):
                obj.geometry = self._build_points_geometry(geometry, buffers=buffers)
            else:
                return False
            return True
        except Exception as exc:
            logger.warning("PygfxRenderer: fallback rebuild failed for '%s': %s", name, exc)
            return False

    def add_geometry_to_visualizer(
        self, geometry: Any, reset_bounding_box: bool = True, is_edge: bool = False
    ) -> None:
        """Compatibility entry point for Open3D-shaped add calls."""
        payload = self._coerce_geometry_payload(geometry)
        if payload is None:
            return
        name = self._external_name_for_geometry(geometry)
        if name is None:
            name = self._register_external_geometry_name(geometry)
        self.ensure_named_geometry(name, payload, is_edge=is_edge)

    def remove_geometry_from_visualizer(
        self, geometry: Any, reset_bounding_box: bool = True
    ) -> None:
        """Compatibility removal for externally owned geometry objects."""
        name = self._external_name_for_geometry(geometry)
        if name is None:
            return
        self.remove_named_geometry(name)

    def invalidate_geometry_payload_cache(self, geometry: Any) -> None:
        """Drop cached payload for mutable Open3D geometry objects."""
        self._payload_cache.pop(id(geometry), None)

    def register_geometry_payload_cache_key(self, geometry: Any, cache_key: str) -> None:
        """Associate one Open3D geometry object with a stable MeshPayload cache key."""
        if geometry is None or not cache_key:
            return
        self._geometry_payload_cache_keys[id(geometry)] = str(cache_key)

    def prime_geometry_buffer_cache(self, geometry: Any) -> None:
        """Populate the prepared pygfx buffer cache for one geometry when possible."""
        payload = self._coerce_geometry_payload(geometry)
        if isinstance(payload, MeshPayload) and payload.cache_key is not None:
            mesh_payload_to_pygfx_buffers(payload)

    def update_geometry_in_visualizer(self, geometry: Any, force_refresh: bool = False) -> None:
        """Compatibility entry point for mutable external geometry updates."""
        if force_refresh:
            self.invalidate_geometry_payload_cache(geometry)
        name = self._external_name_for_geometry(geometry)
        if name is None:
            name = self._register_external_geometry_name(geometry)
        payload = self._coerce_geometry_payload(geometry)
        if payload is not None:
            self.ensure_named_geometry(name, payload)

    def add_or_update_named_geometry(
        self, name: str, geometry: Any, material: Any = None, **kwargs: Any
    ) -> bool:
        """Compatibility wrapper for services that still pass native objects."""
        payload = self._coerce_geometry_payload(geometry)
        if payload is None:
            return False
        ok = self.ensure_named_geometry(name, payload, material=material)
        if ok and not isinstance(geometry, (MeshPayload, LineSetPayload, PointCloudPayload)):
            self._external_geometry_names[id(geometry)] = name
        return ok

    def create_sphere(
        self,
        radius: float = 1.0,
        resolution: int = 20,
        color: Any = None,
        center: Any = None,
    ) -> Any:
        """Create a renderer-neutral sphere payload for Open3D-shaped callers."""
        from ...scene.geometry_payload_factory import make_sphere_payload

        payload = make_sphere_payload(
            radius=radius,
            color=[0.5, 0.5, 0.5] if color is None else list(color),
            resolution=resolution,
        )
        if center is None:
            return payload
        vertices = np.asarray(payload.vertices, dtype=np.float64) + np.asarray(
            center, dtype=np.float64
        ).reshape(1, 3)
        return replace(payload, vertices=vertices)

    def create_wireframe_from_mesh(self, mesh: Any, color: Any = None) -> Any:
        """Create a line payload representing mesh edges when feasible."""
        if isinstance(mesh, MeshPayload):
            if len(mesh.triangles) > 200_000:
                logger.warning(
                    "Skipping wireframe for mesh with %d triangles (threshold 200k)",
                    len(mesh.triangles),
                )
                return None
            from ...scene.geometry_payload_factory import extract_wireframe_payload

            payload = extract_wireframe_payload(mesh)
            if color is not None:
                n = len(payload.points)
                payload = LineSetPayload(
                    points=payload.points,
                    lines=payload.lines,
                    colors=np.tile(np.array(color, dtype=np.float32), (n, 1)),
                )
            return payload

        payload = self._coerce_geometry_payload(mesh)
        if isinstance(payload, MeshPayload):
            return self.create_wireframe_from_mesh(payload, color=color)

        return None

    def merge_meshes(self, meshes: Any) -> Any:
        """Merge mesh-like payloads into one neutral mesh payload."""
        if not meshes:
            return None
        payloads: list[MeshPayload] = []
        for mesh in meshes:
            payload = self._coerce_geometry_payload(mesh)
            if isinstance(payload, MeshPayload):
                payloads.append(payload)
        if not payloads:
            return None
        from ...scene.geometry_payload_factory import merge_mesh_payloads

        return merge_mesh_payloads(payloads)

    def load_triangle_mesh(self, file_path: str) -> Any:
        """Load a mesh file into the renderer-neutral payload model."""
        from ...scene.geometry_payload_factory import load_mesh_payload

        return load_mesh_payload(file_path)

    def set_triangle_uvs(self, mesh: Any, uvs: Any) -> None:
        """Attach UV coordinates to mutable mesh-like objects when possible."""
        uv_values = np.asarray(uvs, dtype=np.float64)
        if isinstance(mesh, MeshPayload):
            return
        payload = getattr(mesh, "payload", None)
        if isinstance(payload, MeshPayload):
            mesh.replace_payload(replace(payload, triangle_uvs=uv_values))
            return
        if hasattr(mesh, "triangle_uvs"):
            current_uvs = getattr(mesh, "triangle_uvs", None)
            try:
                mesh.triangle_uvs = type(current_uvs)(uv_values)
            except Exception:
                try:
                    mesh.triangle_uvs = uv_values
                except Exception:
                    return
            self.invalidate_geometry_payload_cache(mesh)

    def set_geometry_vertices(self, geometry: Any, vertices: Any) -> None:
        """Replace external geometry vertices and refresh named pygfx state."""
        if hasattr(geometry, "vertices"):
            try:
                geometry.vertices = np.asarray(vertices, dtype=np.float64)
            except (AttributeError, TypeError, ValueError):
                return
            if self._external_name_for_geometry(geometry) is not None:
                self.invalidate_geometry_payload_cache(geometry)
                self.update_geometry_in_visualizer(geometry)

    def set_geometry_visible(self, geometry: Any, visible: bool) -> None:
        """Apply visibility to an externally mapped geometry object."""
        name = self._external_name_for_geometry(geometry)
        if name is None:
            return
        self.set_named_visibility(name, bool(visible))

    def set_geometry_transform_fast(
        self,
        geometry: Any,
        position: Any,
        rotation_matrix: Optional[np.ndarray] = None,
        mesh_center: Optional[np.ndarray] = None,
    ) -> bool:
        """Apply fast transform for external geometry callsites.

        Supports two invocation styles:
        - set_geometry_transform_fast(geometry, transform_4x4)
        - set_geometry_transform_fast(geometry, position_xyz, rotation_matrix=..., mesh_center=...)
        """
        name = self._external_name_for_geometry(geometry)
        if name is None:
            return False

        pos_arr = np.asarray(position, dtype=np.float32)
        if pos_arr.shape == (4, 4):
            return self.set_named_transform(name, pos_arr)

        pos_vec = pos_arr.reshape(-1)
        if pos_vec.size != 3:
            return False

        transform = np.eye(4, dtype=np.float32)
        rot = None
        if rotation_matrix is not None:
            rot = np.asarray(rotation_matrix, dtype=np.float32)
            if rot.shape != (3, 3):
                return False
            transform[:3, :3] = rot

        center_source = mesh_center
        if center_source is None:
            center_source = self._geometry_upload_center.get(name)
        if center_source is not None:
            center = np.asarray(center_source, dtype=np.float32).reshape(-1)
            if center.size != 3:
                return False
            if rot is not None:
                transform[:3, 3] = pos_vec - rot @ center
            else:
                transform[:3, 3] = pos_vec - center
        else:
            transform[:3, 3] = pos_vec

        return self.set_named_transform(name, transform)

    def translate_geometry(self, geometry: Any, translation: Any) -> None:
        """Translate an externally mapped geometry object by world delta."""
        name = self._external_name_for_geometry(geometry)
        if name is None:
            return
        t = np.asarray(translation, dtype=np.float32).reshape(-1)
        if t.size != 3:
            return
        current = self._transforms.get(name, np.eye(4, dtype=np.float32))
        next_t = np.array(current, copy=True)
        next_t[:3, 3] += t[:3]
        self.set_named_transform(name, next_t)

    def has_vertex_normals(self, mesh: Any) -> bool:
        """Return whether a neutral mesh payload already carries normals."""
        if isinstance(mesh, MeshPayload):
            return mesh.normals is not None and len(mesh.normals) > 0
        return False

    def compute_vertex_normals(self, mesh: Any) -> None:
        """Compatibility no-op; pygfx path does not mutate normals in place."""
        pass

    def create_box(
        self,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        color: Any = None,
    ) -> Any:
        """Create a renderer-neutral box payload for Open3D-shaped callers."""
        from ...scene.geometry_payload_factory import make_box_payload

        return make_box_payload(
            width=width,
            height=height,
            depth=depth,
            color=color[:3] if color is not None and len(color) >= 3 else color,
        )

    def create_coordinate_frame(self, size: float = 0.9) -> Any:
        """Create an RGB axis line payload for orientation-frame callers."""
        points = np.asarray(
            [[0.0, 0.0, 0.0], [size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, size]],
            dtype=np.float64,
        )
        lines = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
        colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return LineSetPayload(points=points, lines=lines, colors=colors)

    def reset_coordinate_frame(self, frame: Any, size: float = 0.9) -> None:
        """Replace a mutable coordinate-frame handle payload when supported."""
        replacement = self.create_coordinate_frame(size=size)
        if hasattr(frame, "payload") and isinstance(replacement, LineSetPayload):
            frame.replace_payload(replacement)

    def transform_geometry(self, geometry: Any, transform_matrix: Any) -> None:
        """Apply a transform to mapped geometry or delegate to native object."""
        name = self._external_name_for_geometry(geometry)
        if name is not None:
            mat = np.asarray(transform_matrix, dtype=np.float32)
            if mat.shape == (4, 4):
                self.set_named_transform(name, mat)
            return
        if hasattr(geometry, "transform"):
            geometry.transform(transform_matrix)

    def _coerce_geometry_payload(self, geometry: Any) -> Optional[GeometryPayload]:
        """Convert supported payload-like objects into renderer-neutral payloads."""
        if isinstance(
            geometry,
            (MeshPayload, LineSetPayload, PointCloudPayload, OrientationFramePayload),
        ):
            return geometry
        payload_attr = getattr(geometry, "payload", None)
        if isinstance(
            payload_attr,
            (MeshPayload, LineSetPayload, PointCloudPayload, OrientationFramePayload),
        ):
            return payload_attr

        sig = self._geometry_signature(geometry)
        payload_cache_key = self._geometry_payload_cache_keys.get(id(geometry))
        if sig is not None:
            cached = self._payload_cache.get(id(geometry))
            if cached is not None and cached[0] == sig:
                cached_payload = cached[1]
                if (
                    payload_cache_key is not None
                    and isinstance(cached_payload, MeshPayload)
                    and cached_payload.cache_key != payload_cache_key
                ):
                    cached_payload = replace(cached_payload, cache_key=payload_cache_key)
                    self._payload_cache[id(geometry)] = (sig, cached_payload)
                return cached_payload

        payload = self._structural_geometry_payload(geometry)

        if (
            payload_cache_key is not None
            and isinstance(payload, MeshPayload)
            and payload.cache_key != payload_cache_key
        ):
            payload = replace(payload, cache_key=payload_cache_key)

        if payload is not None and sig is not None:
            self._payload_cache[id(geometry)] = (sig, payload)
        return payload

    @staticmethod
    def _geometry_signature(geometry: Any) -> Optional[tuple[Any, ...]]:
        """Return a structural signature for payload-cache invalidation."""
        # Use element counts and presence flags only. Some native geometry
        # libraries create ephemeral Python wrapper objects for vertex buffers
        # on every property access, so id()-based signatures are not stable.
        if hasattr(geometry, "vertices") and hasattr(geometry, "triangles"):
            return (
                "mesh",
                len(geometry.vertices),
                len(geometry.triangles),
                PygfxGeometryMixin._optional_len(geometry, "vertex_normals"),
                PygfxGeometryMixin._optional_len(geometry, "vertex_colors"),
                PygfxGeometryMixin._optional_len(geometry, "triangle_uvs"),
            )
        if hasattr(geometry, "points") and hasattr(geometry, "lines"):
            return (
                "lines",
                len(geometry.points),
                len(geometry.lines),
                PygfxGeometryMixin._optional_len(geometry, "colors"),
            )
        if hasattr(geometry, "points"):
            return (
                "pcd",
                len(geometry.points),
                PygfxGeometryMixin._optional_len(geometry, "colors"),
            )
        return None

    @staticmethod
    def _optional_len(geometry: Any, attr: str) -> int:
        """Return length of an optional geometry attribute, or zero."""
        value = getattr(geometry, attr, None)
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    @staticmethod
    def _optional_array(geometry: Any, attr: str, dtype: Any) -> Optional[np.ndarray]:
        """Return an optional geometry attribute as an array when nonempty."""
        value = getattr(geometry, attr, None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _structural_geometry_payload(geometry: Any) -> Optional[GeometryPayload]:
        """Convert Open3D-shaped geometry objects by structural attributes."""
        try:
            if hasattr(geometry, "vertices") and hasattr(geometry, "triangles"):
                return MeshPayload(
                    vertices=np.asarray(geometry.vertices, dtype=np.float64),
                    triangles=np.asarray(geometry.triangles, dtype=np.int32),
                    normals=PygfxGeometryMixin._optional_array(
                        geometry, "vertex_normals", np.float64
                    ),
                    vertex_colors=PygfxGeometryMixin._optional_array(
                        geometry, "vertex_colors", np.float64
                    ),
                    triangle_uvs=PygfxGeometryMixin._optional_array(
                        geometry, "triangle_uvs", np.float64
                    ),
                )
            if hasattr(geometry, "points") and hasattr(geometry, "lines"):
                return LineSetPayload(
                    points=np.asarray(geometry.points, dtype=np.float64),
                    lines=np.asarray(geometry.lines, dtype=np.int32),
                    colors=PygfxGeometryMixin._optional_array(geometry, "colors", np.float64),
                )
            if hasattr(geometry, "points"):
                return PointCloudPayload(
                    points=np.asarray(geometry.points, dtype=np.float64),
                    colors=PygfxGeometryMixin._optional_array(geometry, "colors", np.float64),
                )
        except (TypeError, ValueError):
            return None
        return None

    def _register_external_geometry_name(self, geometry: Any) -> str:
        """Assign a stable renderer name to an external geometry object."""
        name = f"external_geom_{id(geometry)}"
        self._external_geometry_names[id(geometry)] = name
        return name

    def remap_external_geometry_name(
        self,
        *,
        old_geometry: Any,
        new_geometry: Any,
        name: str,
    ) -> bool:
        """Move an external geometry-object mapping to a new object instance."""
        old_id = id(old_geometry)
        new_id = id(new_geometry)
        self._external_geometry_names.pop(old_id, None)
        self._external_geometry_names[new_id] = str(name)
        self._payload_cache.pop(old_id, None)
        self._payload_cache.pop(new_id, None)
        self._geometry_payload_cache_keys.pop(old_id, None)
        return True

    @staticmethod
    def _split_named_geometry(name: str) -> tuple[str, Optional[str]]:
        """Split ``entity::component`` renderer names."""
        if "::" not in name:
            return name, None
        entity, component = name.split("::", 1)
        return entity, component

    def _name_domain(self, name: str) -> str:
        """Classify stable names into high-level renderer domains."""
        entity, _ = self._split_named_geometry(name)
        if ":" in entity:
            return entity.split(":", 1)[0].strip().lower()
        if name.startswith(("scene_", "mesh_", "merged_group_", "geometry_")):
            return "scene"
        if name.startswith("target_"):
            return "target"
        if name.startswith(("tx_", "rx_", "node_")):
            return "node"
        if name.startswith("coverage_"):
            return "coverage"
        return ""

    def _name_component(self, name: str) -> str:
        """Classify the component part of renderer names used by visual policy."""
        _, component = self._split_named_geometry(name)
        if component is not None:
            return component.strip().lower()
        if "_label_" in name or name.endswith("_label"):
            return "label"
        if "_sphere_" in name or name.endswith("_sphere"):
            return "sphere"
        if "_edge_" in name or "_outline_" in name or name.endswith("_edge"):
            return "edge"
        if name.startswith("target_"):
            return "mesh"
        if name.startswith(("coverage_", "geometry_", "merged_group_", "scene_", "mesh_")):
            return "mesh"
        return ""

    def _is_scene_mesh_name(self, name: str) -> bool:
        """Return whether *name* denotes a scene mesh payload."""
        if name.startswith(("scene_outline_", "scene_merged_outline_")):
            return False
        domain = self._name_domain(name)
        component = self._name_component(name)
        if domain == "scene" and component == "mesh":
            return True
        return name.startswith(("scene_", "mesh_", "merged_group_", "geometry_"))

    def _is_target_mesh_name(self, name: str) -> bool:
        """Return whether *name* denotes a target mesh payload."""
        domain = self._name_domain(name)
        component = self._name_component(name)
        if domain == "target" and component == "mesh":
            return True
        if name.startswith("target_") and component == "mesh":
            return True
        return False

    def _is_wireframe_candidate(self, name: str) -> bool:
        """Return whether wireframe mode should apply to a named object."""
        component = self._name_component(name)
        if component != "mesh":
            return False
        return self._is_scene_mesh_name(name) or self._is_target_mesh_name(name)

    def _external_name_for_geometry(self, geometry: Any) -> Optional[str]:
        """Return the stable renderer name mapped to an external object."""
        return self._external_geometry_names.get(id(geometry))

    def _external_remove_name(self, name: str) -> None:
        """Drop external-object mappings that point to a removed name."""
        stale: list[int] = []
        for gid, gname in self._external_geometry_names.items():
            if gname == name:
                stale.append(gid)
        for gid in stale:
            self._external_geometry_names.pop(gid, None)

    def set_wireframe(self, enabled: bool) -> None:
        """Toggle wireframe rendering on scene + target mesh geometries."""
        self._wireframe_enabled = enabled
        for name, obj in self._objects.items():
            if not self._is_wireframe_candidate(name):
                continue
            mat = getattr(obj, "material", None)
            if mat is not None and hasattr(mat, "wireframe"):
                try:
                    mat.wireframe = enabled
                except (AttributeError, RuntimeError):
                    pass
        self.request_redraw()

    def show_axes(self, show: bool, size: float = 1.0) -> bool:
        """Show or hide a simple RGB world-axis overlay at the origin."""
        if not self._initialized:
            return False
        if not show:
            removed = self.remove_named_geometry(self.AXES_NAME)
            if removed:
                self.request_redraw()
            return True

        axis_size = self._default_axes_size(size)
        payload = LineSetPayload(
            points=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [axis_size, 0.0, 0.0],
                    [0.0, axis_size, 0.0],
                    [0.0, 0.0, axis_size],
                ],
                dtype=np.float32,
            ),
            lines=np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32),
            colors=np.asarray(
                [
                    [1.0, 0.08, 0.06, 1.0],
                    [0.1, 0.8, 0.1, 1.0],
                    [0.1, 0.35, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        material = MaterialPayload(
            base_color=(1.0, 1.0, 1.0, 1.0),
            shader="unlit",
            line_width=max(3.0, float(getattr(self, "_line_width", 2.0))),
        )
        ok = self.ensure_named_geometry(
            self.AXES_NAME,
            payload,
            material=material,
            visible=True,
        )
        if ok:
            self.request_redraw()
        return bool(ok)

    def _default_axes_size(self, requested_size: float) -> float:
        """Scale default axes to large scenes unless caller supplied a size."""
        try:
            requested = float(requested_size)
        except (TypeError, ValueError):
            requested = 1.0
        if requested > 1.0:
            return requested

        try:
            bbox = self.compute_scene_bounds(scope="whole")
            extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        except Exception:
            return 1.0
        if extent.size < 3 or not np.all(np.isfinite(extent)):
            return 1.0
        max_extent = float(np.max(extent[:3]))
        if max_extent <= 0.0:
            return 1.0
        return max(2.0, min(75.0, max_extent * 0.12))
