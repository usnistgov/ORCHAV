"""Load, transform, and synchronize target meshes for visualizer frames.

Targets are dynamic scene entities whose geometry may come from cached PLY
assets or frame metadata. This service preserves the target-entry contract used
by panels while publishing complete renderer-neutral object snapshots.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from shared.frames import StandardMPCFrame
from shared.logging import get_logger

from ..io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    visual_frame_read_request,
)
from ..materials.appearance import (
    ResolvedAppearance,
    VisualMaterialBinding,
    VisualMaterialSource,
)
from ..materials.catalog import ResolvedMaterial, resolve_pbr_material
from ..model import (
    RenderObjectState,
    Transform,
)
from ..scene.geometry_payload_factory import extract_wireframe_payload, load_mesh_payload
from ..scene.target_materials import (
    MINIMAL_TARGET_PBR_PROPS,
    resolve_target_pbr_props,
    target_entry_pbr_fields,
)
from ..scene.target_runtime import target_label_visible, target_runtime_visible
from ..scene.target_transforms import (
    TargetGeometryMeta,
    build_sionna_rotation_matrix,
    build_sionna_rotation_transform,
    mesh_aabb_center,
    orientation_metadata,
    scale_target_mesh,
    set_target_mesh_vertices,
    sionna_ypr_to_xyz_rotation,
    target_mesh_payload,
    target_mesh_vertex_colors,
    target_mesh_vertices,
    target_transform_matrix,
    transform_target_mesh_payload,
    translate_target_mesh_by,
)
from ..scene.target_transforms import (
    rotated_aabb_center as compute_rotated_aabb_center,
)
from ..services.base import BaseService
from ..services.cache_service import CacheInvalidationScope
from ..services.entity_render_service import EntityRenderService
from ..services.object_identity import (
    ensure_target_entry_identity,
    make_target_entry_geometry_name,
)
from ..services.pov_visibility_service import is_hidden_for_pov
from ..services.target_asset_cache import (
    DEFAULT_TARGET_ASSET_CACHE_ENTRIES,
    ResolvedTargetAssetSource,
    TargetAsset,
    TargetAssetCache,
    TargetRuntimeState,
)
from ..services.target_render_sync import TargetRenderSync
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
    material_payload_from_mapping,
)

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.target_service")

TARGET_ASSET_LOOKAHEAD = 2
TARGET_VERTEX_COLOR_RICH_THRESHOLD = 16
TARGET_VERTEX_COLOR_PROBE_COUNT = 256


@dataclass(frozen=True, slots=True)
class _TargetAssetBuildSpec:
    """Immutable data required to build target frames off the UI thread."""

    scale: float
    orientation: tuple[float, float, float]
    use_ply_position: bool
    uniform_material: MaterialPayload
    vertex_material: MaterialPayload


def _aabb_center(mesh: RenderObjectState) -> np.ndarray:
    """Return the axis-aligned bounding-box center of a target mesh.

    Sionna RT defines ``SceneObject.position`` as the AABB center
    ``(bbox_min + bbox_max) / 2``, so the visualizer must use the same
    reference point when positioning target meshes. Vertex means/centroids
    differ from the AABB center for asymmetric meshes such as human bodies.
    """
    return mesh_aabb_center(mesh)


class TargetService(BaseService):
    """Service for loading, positioning, and rendering 3D target meshes.

    Handles target model loading from PLY files, mesh switching between
    animation frames, and scale/orientation transforms.

    Args:
        visualizer: The main visualizer application instance.
    """

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        *,
        target_asset_cache: TargetAssetCache | None = None,
    ) -> None:
        """Create target rendering helpers bound to application state.

        ``target_asset_cache`` lets application composition provide the one
        typed owner shared by target loading, animation, and invalidation.
        """
        super().__init__()
        self.visualizer = visualizer
        self._entity_render_service = EntityRenderService(visualizer)
        self._target_render_sync = TargetRenderSync(visualizer, self._entity_render_service)
        self._last_runtime_breakdown: dict[str, float] = {}
        if isinstance(target_asset_cache, TargetAssetCache):
            cache = target_asset_cache
        else:
            existing = getattr(visualizer, "target_asset_cache", None)
            cache = existing if isinstance(existing, TargetAssetCache) else TargetAssetCache()
            if cache is not existing:
                visualizer.target_asset_cache = cache
        self._target_asset_cache = cache

    def _is_hidden_for_pov(self, node_type: str, index: int) -> bool:
        """Resolve target visibility from the shared application POV policy."""
        return is_hidden_for_pov(
            getattr(self.visualizer, "app_state", None),
            node_type,
            index,
        )

    def get_last_runtime_breakdown(self) -> dict[str, float]:
        """Return the latest aggregated target-runtime timing breakdown."""
        return dict(self._last_runtime_breakdown)

    def _should_bake_target_orientation(self, *, use_ply_position: bool) -> bool:
        """Return whether load-time target meshes should bake orientation.

        All supported renderers own target rotation through ``ensure_object``.
        Only PLY-position meshes carry their placement in payload coordinates.
        """
        return bool(use_ply_position)

    def _target_mesh_handle(
        self,
        *,
        render_id: str,
        mesh_path: str,
        visible: bool = True,
        material: Optional[MaterialPayload] = None,
    ) -> RenderObjectState:
        """Load a target mesh file into a neutral render handle."""
        payload = load_mesh_payload(mesh_path)
        return RenderObjectState(
            id=render_id,
            payload=payload,
            material=material or MaterialPayload(base_color=(0.7, 0.7, 0.7, 1.0)),
            visible=bool(visible),
            metadata={"type": "target_mesh", "mesh_path": str(mesh_path)},
        )

    def _mesh_payload(self, mesh: RenderObjectState) -> Optional[MeshPayload]:
        """Return the mesh payload from application-owned target state."""
        return target_mesh_payload(mesh)

    def _mesh_vertices_array(self, mesh: RenderObjectState) -> np.ndarray:
        """Return target mesh vertices from renderer-neutral geometry."""
        return target_mesh_vertices(mesh)

    def _scale_mesh(self, mesh: RenderObjectState, scale: float, center: Any) -> None:
        """Scale a mutable renderer-neutral target mesh around ``center``."""
        if not scale_target_mesh(mesh, scale, center):
            raise TypeError(f"{type(mesh).__name__} is not a mutable target mesh")

    def _transform_mesh_payload(self, mesh: RenderObjectState, transform: Any) -> None:
        """Apply a geometry-space transform to a mutable neutral target mesh."""
        if not transform_target_mesh_payload(mesh, transform):
            raise TypeError(f"{type(mesh).__name__} is not a mutable target mesh")

    def _translate_mesh_by(self, mesh: RenderObjectState, delta: Any) -> None:
        """Translate a mutable neutral target mesh by a relative delta."""
        if not translate_target_mesh_by(mesh, delta):
            raise TypeError(f"{type(mesh).__name__} is not a mutable target mesh")

    def _sync_target_mesh_geometry(
        self,
        mesh: RenderObjectState,
        mesh_name: str,
        *,
        visible: bool = True,
        snapshot_material: MaterialPayload | None = None,
    ) -> bool:
        """Sync target mesh state or neutral payload through the renderer API."""
        return self._target_render_sync.sync_mesh_geometry(
            mesh,
            mesh_name,
            visible=visible,
            snapshot_material=snapshot_material,
        )

    def _sync_target_outline_geometry(
        self,
        target_entry: Dict[str, Any],
        outline: RenderObjectState,
        *,
        visible: bool = True,
    ) -> bool:
        """Sync one target outline through its stable object identity."""
        outline_name = make_target_entry_geometry_name(target_entry, "outline")
        return self._target_render_sync.sync_outline_geometry(
            outline,
            outline_name,
            visible=visible,
        )

    def _mesh_vertex_colors_array(self, mesh: RenderObjectState) -> Optional[np.ndarray]:
        """Return vertex colors from renderer-neutral target geometry."""
        return target_mesh_vertex_colors(mesh)

    def _has_rich_vertex_colors(self, mesh: RenderObjectState) -> bool:
        """Return whether vertex colors look like texture data, not one tint."""
        colors = self._mesh_vertex_colors_array(mesh)
        if colors is None or colors.size == 0:
            return False
        if colors.ndim != 2 or colors.shape[0] == 0:
            return False

        # Uniform colors are common on animated PLY sequences. Prove that case
        # with one vectorized comparison instead of sorting every vertex row.
        # A small deterministic probe can likewise prove a rich-color result.
        # If neither proof succeeds, retain the full exact check so sparse
        # authored color variation cannot be misclassified.
        if np.all(colors == colors[0]):
            return False
        if len(colors) > TARGET_VERTEX_COLOR_PROBE_COUNT:
            probe_indices = np.linspace(
                0,
                len(colors) - 1,
                num=TARGET_VERTEX_COLOR_PROBE_COUNT,
                dtype=np.intp,
            )
            if len(np.unique(colors[probe_indices], axis=0)) > TARGET_VERTEX_COLOR_RICH_THRESHOLD:
                return True
        return len(np.unique(colors, axis=0)) > TARGET_VERTEX_COLOR_RICH_THRESHOLD

    @staticmethod
    def _set_handle_material(
        geometry: RenderObjectState,
        material: MaterialPayload | dict[str, Any],
    ) -> None:
        """Update a neutral render handle material without touching the renderer."""
        if isinstance(geometry, RenderObjectState):
            geometry.material = material_payload_from_mapping(material)

    @staticmethod
    def _set_handle_vertex_color_source(geometry: RenderObjectState) -> None:
        """Declare authored target vertex colors as intrinsic surface detail."""
        if isinstance(geometry, RenderObjectState) and isinstance(geometry.payload, MeshPayload):
            geometry.replace_payload(
                replace(geometry.payload, color_source=SurfaceColorSource.VERTEX)
            )

    @staticmethod
    def _target_asset_build_spec(
        *,
        scale: Any,
        orientation: Any,
        use_ply_position: bool,
        pbr_props: dict[str, Any],
        resolved_material: ResolvedMaterial | None = None,
    ) -> _TargetAssetBuildSpec:
        """Resolve immutable materials and transform inputs once per target."""
        try:
            orientation_values = tuple(float(value) for value in orientation)
        except (TypeError, ValueError):
            orientation_values = (0.0, 0.0, 0.0)
        if len(orientation_values) != 3:
            orientation_values = (0.0, 0.0, 0.0)
        resolved = resolved_material or resolve_pbr_material(
            pbr_props.get("color", [0.8, 0.6, 0.5]),
            pbr_props,
            context="target asset",
        )
        uniform_material = resolved.payload
        vertex_material = replace(
            uniform_material,
            base_color=(1.0, 1.0, 1.0, uniform_material.base_color[3]),
            texture_path=None,
        )
        return _TargetAssetBuildSpec(
            scale=float(scale),
            orientation=orientation_values,
            use_ply_position=bool(use_ply_position),
            uniform_material=uniform_material,
            vertex_material=vertex_material,
        )

    @staticmethod
    def _target_asset_base_material(
        asset: TargetAsset,
        spec: _TargetAssetBuildSpec,
    ) -> MaterialPayload:
        """Return the build-time material compatible with one target asset."""
        selected = spec.vertex_material if asset.has_vertex_texture else spec.uniform_material
        payload = asset.mesh.payload
        if (
            asset.has_vertex_texture
            and isinstance(payload, MeshPayload)
            and payload.triangle_uvs is None
        ):
            selected = replace(
                selected,
                normal_map_path=None,
                roughness_map_path=None,
                ao_map_path=None,
                metallic_map_path=None,
            )
        return selected

    def _build_target_asset(
        self,
        source: ResolvedTargetAssetSource,
        spec: _TargetAssetBuildSpec,
    ) -> TargetAsset:
        """Parse and prepare one neutral target frame without renderer access."""
        mesh = self._target_mesh_handle(
            render_id=make_target_entry_geometry_name(
                {"target_name": source.target_name},
                "mesh",
            ),
            mesh_path=source.canonical_path,
        )
        original_vertices = self._mesh_vertices_array(mesh)
        if abs(spec.scale - 1.0) > 1e-12:
            self._scale_mesh(mesh, spec.scale, _aabb_center(mesh))
        scaled_vertices = self._mesh_vertices_array(mesh)
        geometry_meta = TargetGeometryMeta(
            scaled_aabb_center=np.asarray(_aabb_center(mesh), dtype=np.float64)
        )

        if self._should_bake_target_orientation(use_ply_position=spec.use_ply_position) and any(
            abs(angle) > 1e-6 for angle in spec.orientation
        ):
            self._transform_mesh_payload(
                mesh,
                build_sionna_rotation_transform(*spec.orientation),
            )

        has_vertex_texture = self._has_rich_vertex_colors(mesh)
        if has_vertex_texture:
            self._set_handle_vertex_color_source(mesh)
        provisional_asset = TargetAsset(
            source=source,
            mesh=mesh,
            original_vertices=original_vertices,
            scaled_vertices=scaled_vertices,
            geometry_meta=geometry_meta,
            has_vertex_texture=has_vertex_texture,
        )
        self._set_handle_material(
            mesh,
            self._target_asset_base_material(provisional_asset, spec),
        )
        return provisional_asset

    def _load_registered_target_asset(
        self,
        target_name: str,
        mesh_file: str,
        spec: _TargetAssetBuildSpec,
    ) -> TargetAsset:
        """Return a registered frame, reusing any in-flight prefetch."""
        return self._target_asset_cache.get_or_load(
            target_name,
            mesh_file,
            lambda source: self._build_target_asset(source, spec),
        )

    def _asset_for_entry(self, entry: Dict[str, Any]) -> Optional[TargetAsset]:
        """Return the active asset retained by an entry or the bounded LRU."""
        target_name = self._target_name(entry)
        mesh_file = str(entry.get("mesh_file") or "")
        asset = entry.get("_target_asset")
        if isinstance(asset, TargetAsset) and asset.logical_key == (target_name, mesh_file):
            self._target_asset_cache.pin(asset)
            return asset
        asset = self._target_asset_cache.asset_for_logical_key((target_name, mesh_file))
        if asset is not None:
            entry["_target_asset"] = asset
            self._target_asset_cache.pin(asset)
        return asset

    def _schedule_target_lookahead(
        self,
        entry: Dict[str, Any],
        *,
        count: int = TARGET_ASSET_LOOKAHEAD,
    ) -> None:
        """Prefetch the next small animation window without renderer work."""
        spec = entry.get("_target_asset_build_spec")
        mesh_file = entry.get("mesh_file")
        requested_count = min(TARGET_ASSET_LOOKAHEAD, max(0, int(count)))
        if not isinstance(spec, _TargetAssetBuildSpec) or not mesh_file or requested_count <= 0:
            return
        target_name = self._target_name(entry)
        try:
            play_direction = float(getattr(self.visualizer, "play_direction", 1))
        except (TypeError, ValueError):
            play_direction = 1.0
        direction = -1 if np.isfinite(play_direction) and play_direction < 0.0 else 1
        schedule_key = (str(mesh_file), direction)
        prior_schedule = entry.get("_target_lookahead_schedule")
        if isinstance(prior_schedule, tuple) and prior_schedule[:2] == schedule_key:
            # Historical two-item tokens represent a complete lookahead. New
            # tokens record the deepest wave already requested.
            prior_count = (
                TARGET_ASSET_LOOKAHEAD
                if len(prior_schedule) < 3
                else max(0, int(prior_schedule[2]))
            )
            if prior_count >= requested_count:
                return
        self._target_asset_cache.prefetch_after(
            target_name,
            str(mesh_file),
            count=requested_count,
            loader=lambda source: self._build_target_asset(source, spec),
            direction=direction,
        )
        entry["_target_lookahead_schedule"] = (*schedule_key, requested_count)

    def _schedule_target_lookahead_waves(
        self,
        entries: List[Dict[str, Any]],
    ) -> None:
        """Queue immediate frames for every target before deeper lookahead.

        Thread-pool submission is FIFO. Scheduling one depth across all targets
        first prevents a later target's immediately required frame from sitting
        behind speculative second-frame work for earlier targets.
        """
        for depth in range(1, TARGET_ASSET_LOOKAHEAD + 1):
            for entry in entries:
                self._schedule_target_lookahead(entry, count=depth)

    def _set_mesh_vertices(self, mesh: RenderObjectState, vertices: Any) -> None:
        """Replace mesh vertices without synchronizing an intermediate payload.

        Target edits are assembled in application-owned ``RenderObjectState``
        first.  The caller synchronizes once after scaling, rotation, and color
        changes have all been applied.
        """
        if set_target_mesh_vertices(mesh, vertices):
            return
        raise TypeError(f"{type(mesh).__name__} is not a mutable target mesh")

    @staticmethod
    def _rotation_cache_key(rotation_matrix: np.ndarray) -> bytes:
        """Return a stable cache key for one concrete 3x3 rotation matrix."""
        return np.asarray(rotation_matrix, dtype=np.float64).tobytes()

    def _store_target_geometry_meta(
        self,
        cache_key: tuple[str, str],
        *,
        scaled_vertices: Optional[np.ndarray] = None,
        mesh: Optional[RenderObjectState] = None,
    ) -> TargetGeometryMeta:
        """Store or refresh cached local-space geometry metadata."""
        if scaled_vertices is not None:
            verts = np.asarray(scaled_vertices, dtype=np.float64)
            if verts.size:
                scaled_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
            elif mesh is not None:
                scaled_center = _aabb_center(mesh)
            else:
                scaled_center = np.zeros(3, dtype=np.float64)
        elif mesh is not None:
            scaled_center = _aabb_center(mesh)
        else:
            scaled_center = np.zeros(3, dtype=np.float64)
        meta = TargetGeometryMeta(scaled_aabb_center=np.asarray(scaled_center, dtype=np.float64))
        asset = self._target_asset_cache.asset_for_logical_key(cache_key)
        if asset is not None:
            asset.geometry_meta = meta
            if scaled_vertices is not None:
                asset.scaled_vertices = np.asarray(scaled_vertices, dtype=np.float64)
            self._target_asset_cache.put(asset)
        return meta

    def _get_target_geometry_meta(
        self,
        target_name: str,
        mesh_file: Optional[str],
        mesh: Optional[RenderObjectState] = None,
    ) -> TargetGeometryMeta:
        """Return cached local-space metadata, building it lazily when needed."""
        cache_key = (target_name, mesh_file or "")
        asset = self._target_asset_cache.asset_for_logical_key(cache_key)
        if asset is not None:
            return asset.geometry_meta
        scaled_vertices = None
        return self._store_target_geometry_meta(
            cache_key, scaled_vertices=scaled_vertices, mesh=mesh
        )

    def _resolve_rotated_aabb_center(
        self,
        target_name: str,
        mesh_file: Optional[str],
        rotation_matrix: np.ndarray,
        mesh: Optional[RenderObjectState] = None,
    ) -> Optional[np.ndarray]:
        """Return cached rotated AABB center for one target mesh + rotation."""
        meta = self._get_target_geometry_meta(target_name, mesh_file, mesh=mesh)
        rotation_key = self._rotation_cache_key(rotation_matrix)
        if meta.rotation_key == rotation_key and meta.rotated_aabb_center is not None:
            return np.asarray(meta.rotated_aabb_center, dtype=np.float64)
        cache_key = (target_name, mesh_file or "")
        asset = self._target_asset_cache.asset_for_logical_key(cache_key)
        scaled_vertices = asset.scaled_vertices if asset is not None else None
        if scaled_vertices is None:
            return None
        rotated_center = compute_rotated_aabb_center(scaled_vertices, rotation_matrix)
        if rotated_center is None:
            return None
        meta.rotated_aabb_center = np.asarray(rotated_center, dtype=np.float64)
        meta.rotation_key = rotation_key
        if asset is not None:
            asset.geometry_meta = meta
            self._target_asset_cache.touch(asset)
        return np.asarray(rotated_center, dtype=np.float64)

    @staticmethod
    def _target_name(target_entry: Dict[str, Any], fallback: str = "target") -> str:
        """Resolve target logical name from entry metadata."""
        return (
            target_entry.get("target_name")
            or target_entry.get("name")
            or target_entry.get("node_name")
            or target_entry.get("stable_target_id")
            or fallback
        )

    @staticmethod
    def _coerce_vec3(values: Any) -> tuple[list[float], bool]:
        """Return (vec3, valid) for arbitrary position-like input."""
        if values is None:
            return [0.0, 0.0, 0.0], False
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return [0.0, 0.0, 0.0], False
        return [float(arr[0]), float(arr[1]), float(arr[2])], True

    @staticmethod
    def _normalize_runtime_scale(value: Any) -> float | None:
        """Return one finite scale value for state comparison and application."""
        if value is None:
            return None
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return None
        return scale if np.isfinite(scale) else None

    def _target_label_anchor(self, target_entry: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return the final world-space anchor for one target label."""
        if not bool(target_entry.get("_use_ply_position", False)):
            for key in ("_target_position", "position"):
                values, valid = self._coerce_vec3(target_entry.get(key))
                if valid:
                    return np.asarray(values, dtype=np.float64)

        mesh = target_entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            return None
        local_center = _aabb_center(mesh)
        world_center = mesh.world_transform.matrix @ np.append(local_center, 1.0)
        return np.asarray(world_center[:3], dtype=np.float64)

    def sync_target_entry_snapshot(
        self,
        target_entry: Dict[str, Any],
        *,
        effective_visible: bool | None = None,
        resolved_appearance: Any = None,
    ) -> bool:
        """Publish one complete target, label, and outline state snapshot."""
        # Mesh, label, and outline are one semantic object. The backend may
        # nest this inside a frame-wide batch; only the outer boundary presents.
        with self.visualizer.renderer.batch_updates():
            return self._sync_target_entry_snapshot(
                target_entry,
                effective_visible=effective_visible,
                resolved_appearance=resolved_appearance,
            )

    def _sync_target_entry_snapshot(
        self,
        target_entry: Dict[str, Any],
        *,
        effective_visible: bool | None = None,
        resolved_appearance: Any = None,
    ) -> bool:
        """Synchronize target members while the semantic batch is active."""
        viz = self.visualizer
        mesh = target_entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            logger.error(
                "Target '%s' does not own a RenderObjectState mesh",
                self._target_name(target_entry),
            )
            target_entry["_renderer_sync_pending"] = True
            return False

        target_index = target_entry.get("node_index")
        if target_index is None:
            try:
                target_index = viz.target_entries.index(target_entry)
            except ValueError:
                target_index = None
        ensure_target_entry_identity(target_entry, target_index)
        mesh_name = make_target_entry_geometry_name(target_entry, "mesh")
        if resolved_appearance is None:
            appearance = getattr(viz, "object_appearance_service", None)
            resolve_runtime = getattr(appearance, "resolve_entry_runtime_appearance", None)
            if callable(resolve_runtime):
                resolved_appearance = resolve_runtime(target_entry)
        if isinstance(resolved_appearance, ResolvedAppearance):
            desired_visible = bool(resolved_appearance.visible)
            if effective_visible is not None:
                desired_visible = bool(desired_visible and effective_visible)
            snapshot_material = resolved_appearance.material
        else:
            desired_visible = (
                target_runtime_visible(
                    target_entry,
                    target_index,
                    self._is_hidden_for_pov,
                )
                if effective_visible is None
                else bool(effective_visible)
            )
            snapshot_material = None
        self._target_render_sync.record_benchmark_metric(
            "target_runtime_visible_count" if desired_visible else "target_runtime_hidden_count"
        )
        self._target_render_sync.record_benchmark_metric("target_runtime_visibility_call_count")
        self._target_render_sync.record_benchmark_metric(
            "target_batch_visible_update_count"
            if desired_visible
            else "target_batch_hidden_skip_count"
        )

        transform = self._target_transform_matrix(target_entry)
        if transform is not None:
            mesh.world_transform = Transform(transform)
            self._target_render_sync.record_benchmark_metric("target_transform_snapshot_count")
        else:
            # PLY-position snapshots bake placement into payload coordinates.
            # Clear any transform retained from a previous frame-coordinate
            # snapshot when the positioning mode changes.
            mesh.world_transform = Transform.identity()

        mesh_synced = self._sync_target_mesh_geometry(
            mesh,
            mesh_name,
            visible=desired_visible,
            snapshot_material=snapshot_material,
        )

        label_synced = True
        if target_index is not None:
            label_name = make_target_entry_geometry_name(target_entry, "label")
            label = next(
                (
                    candidate
                    for candidate in getattr(viz, "target_labels", [])
                    if isinstance(candidate, RenderObjectState) and candidate.id == label_name
                ),
                None,
            )
        else:
            label = None
            label_name = ""
        if label is not None:
            label_visible = target_label_visible(
                target_entry,
                target_index,
                self._is_hidden_for_pov,
                show_target_labels=getattr(
                    getattr(viz, "app_state", None), "show_target_labels", True
                ),
                runtime_visible=desired_visible,
            )
            label_synced = self._target_render_sync.sync_target_label(
                geometry_name=label_name,
                label=label,
                visible=label_visible,
                anchor_position=self._target_label_anchor(target_entry),
                offset=np.asarray(
                    [
                        float(getattr(viz, "label_offset_x", 1.5)),
                        float(getattr(viz, "label_offset_y", 0.0)),
                        float(getattr(viz, "label_offset_z", 1.0)),
                    ],
                    dtype=np.float64,
                ),
            )

        outline_synced = self.sync_target_entry_edge_visibility(
            target_entry,
            mesh_visible=desired_visible,
            update_renderer=False,
        )
        synced = bool(mesh_synced and label_synced and outline_synced)
        target_entry["_renderer_sync_pending"] = not synced
        return synced

    def refresh_target_entry_material(
        self,
        target_entry: Dict[str, Any],
        *,
        effective_visible: bool | None = None,
        resolved_appearance: Any = None,
    ) -> bool:
        """Resolve target appearance and publish one complete target snapshot."""
        appearance_service = getattr(self.visualizer, "object_appearance_service", None)
        resolve = getattr(appearance_service, "resolve_entry_appearance", None)
        if resolved_appearance is None and callable(resolve):
            resolved_appearance = resolve(target_entry)
        if isinstance(resolved_appearance, ResolvedAppearance):
            if effective_visible is None:
                effective_visible = resolved_appearance.visible
        else:
            resolved_appearance = None
        return self.sync_target_entry_snapshot(
            target_entry,
            effective_visible=effective_visible,
            resolved_appearance=resolved_appearance,
        )

    # Outline (wireframe edge) helpers

    def _ensure_target_outline(self, entry: Dict[str, Any]) -> Any:
        """Lazily create and cache a wireframe LineSet for a target mesh.

        Uses the same renderer-neutral edge-extraction policy as scene
        outlines while retaining target-specific visibility ownership.

        Args:
            entry: The mutable target entry dict.

        Returns:
            A renderer-neutral wireframe payload, or None if the mesh has no triangles.
        """
        mesh = entry.get("mesh")
        if mesh is None:
            return None
        ensure_target_entry_identity(entry, entry.get("node_index"))
        outline_name = make_target_entry_geometry_name(entry, "outline")
        outline = entry.get("outline_geometry")
        outline_dirty = bool(entry.get("_outline_payload_dirty", False))
        if isinstance(outline, RenderObjectState) and not outline_dirty:
            return outline
        viz = self.visualizer
        color = np.asarray(viz.outline_color, dtype=np.float64)
        mesh_payload = self._mesh_payload(mesh)
        if mesh_payload is None:
            return None
        outline_payload = extract_wireframe_payload(mesh_payload)
        if len(outline_payload.lines) == 0:
            return None
        outline_payload = LineSetPayload(
            points=outline_payload.points,
            lines=outline_payload.lines,
            colors=np.tile(color[:3], (len(outline_payload.lines), 1)),
        )
        material = MaterialPayload(
            base_color=(float(color[0]), float(color[1]), float(color[2]), 1.0),
            shader="unlit",
        )
        if isinstance(outline, RenderObjectState) and outline.id == outline_name:
            outline.replace_payload(outline_payload)
            outline.material = material
            outline.is_edge = True
            outline.metadata = {"type": "target_outline"}
        else:
            outline = RenderObjectState(
                id=outline_name,
                payload=outline_payload,
                material=material,
                visible=False,
                is_edge=True,
                metadata={"type": "target_outline"},
            )
        entry["outline_geometry"] = outline
        entry["_outline_payload_dirty"] = False
        return outline

    def sync_target_entry_edge_visibility(
        self,
        entry: Dict[str, Any],
        *,
        mesh_visible: bool | None = None,
        update_renderer: bool = True,
    ) -> bool:
        """Synchronize one target outline using effective mesh visibility."""
        viz = self.visualizer
        mesh = entry.get("mesh")
        if mesh is None or viz.vis is None or not viz.vis_initialized:
            return mesh is not None

        target_index = entry.get("node_index")
        if target_index is None:
            try:
                target_index = viz.target_entries.index(entry)
            except ValueError:
                target_index = None
        target_mesh_visible = bool(
            target_runtime_visible(
                entry,
                target_index,
                self._is_hidden_for_pov,
            )
            if mesh_visible is None
            else mesh_visible
        )
        outline_enabled = bool(getattr(viz, "target_outlines_enabled", False))
        outline_should_show = bool(outline_enabled and target_mesh_visible)
        outline = (
            self._ensure_target_outline(entry)
            if outline_should_show
            else entry.get("outline_geometry")
        )
        if not isinstance(outline, RenderObjectState):
            stale_outline = entry.get("outline_geometry")
            if isinstance(stale_outline, RenderObjectState) and bool(
                entry.get("_outline_payload_dirty", False)
            ):
                stale_outline.visible = False
                synced = self._sync_target_outline_geometry(
                    entry,
                    stale_outline,
                    visible=False,
                )
                entry["outline_visible"] = False
                return synced
            entry["outline_visible"] = False
            return True

        # Preserve the global outline toggle as semantic state. Frame and POV
        # constraints are renderer-only and belong on the immutable snapshot.
        outline.visible = outline_enabled
        # Mesh and outline use the same local coordinate space. Mirroring the
        # mesh snapshot prevents duplicate transform policy and avoids a stale
        # transform when switching to PLY-embedded positioning.
        if isinstance(mesh, RenderObjectState):
            outline.world_transform = mesh.world_transform
        synced = self._sync_target_outline_geometry(
            entry,
            outline,
            visible=outline_should_show,
        )
        entry["outline_visible"] = outline_should_show

        if update_renderer:
            request_redraw = getattr(viz.renderer, "request_redraw", None)
            if callable(request_redraw):
                request_redraw()
                return synced
            try:
                viz.renderer.update_renderer()
            except RuntimeError:
                logger.debug("Failed to update renderer after target outline sync")
        return synced

    def set_target_edge_visibility(self, enabled: bool) -> None:
        """Show or hide wireframe edges on all visible target meshes.

        Outline geometry is created lazily — zero overhead when disabled.

        Args:
            enabled: Whether target outlines should be visible.
        """
        from contextlib import nullcontext

        viz = self.visualizer
        viz.target_outlines_enabled = bool(enabled)
        if not viz.target_entries or viz.vis is None or not viz.vis_initialized:
            logger.debug(
                "set_target_edge_visibility(%s): early return " "(entries=%d, vis=%s, vis_init=%s)",
                enabled,
                len(viz.target_entries),
                viz.vis is not None,
                viz.vis_initialized,
            )
            return

        use_batch = hasattr(viz.renderer, "batch_updates")
        ctx = viz.renderer.batch_updates() if use_batch else nullcontext()
        updated = False

        with ctx:
            for entry in viz.target_entries:
                mesh = entry.get("mesh")
                if mesh is None:
                    continue
                synced = self.sync_target_entry_edge_visibility(
                    entry,
                    update_renderer=False,
                )
                if not synced:
                    entry["_renderer_sync_pending"] = True
                updated = True

        if updated:
            if use_batch:
                if hasattr(viz.renderer, "request_redraw"):
                    viz.renderer.request_redraw()
            else:
                try:
                    viz.renderer.update_renderer()
                except RuntimeError:
                    logger.debug("Failed to update renderer after target outline toggle")

    def _invalidate_target_outline(self, entry: Dict[str, Any]) -> None:
        """Mark a target's outline payload stale without losing its native identity.

        Called after mesh switches or scale changes that invalidate
        the cached LineSet vertices. Retaining the handle lets the next
        declarative snapshot hide an existing native outline immediately;
        the payload is rebuilt lazily when the outline becomes visible.
        """
        outline = entry.get("outline_geometry")
        if isinstance(outline, RenderObjectState):
            entry["_outline_payload_dirty"] = True
            entry["outline_visible"] = False

    def _target_transform_matrix(self, entry: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return the current target mesh transform matrix for an entry."""
        if bool(entry.get("_use_ply_position", False)):
            return None
        pos = entry.get("_target_position")
        if pos is None:
            return None
        mesh_center = entry.get("_mesh_center")
        rotation_matrix = entry.get("_rotation_matrix")
        rotated_center = None
        if rotation_matrix is not None:
            try:
                R = np.asarray(rotation_matrix, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if R.shape != (3, 3):
                return None
            rotated_center = self._resolve_rotated_aabb_center(
                self._target_name(entry),
                entry.get("mesh_file"),
                R,
                mesh=entry.get("mesh"),
            )
            rotation_matrix = R

        return target_transform_matrix(
            position=pos,
            mesh_center=mesh_center,
            rotation_matrix=rotation_matrix,
            rotated_center=rotated_center,
        )

    def _restore_scaled_target_payload(
        self,
        entry: Dict[str, Any],
        mesh: RenderObjectState,
    ) -> bool:
        """Restore one target mesh to its scaled, unrotated payload baseline."""
        asset = self._asset_for_entry(entry)
        vertices = asset.scaled_vertices if asset is not None else None
        if vertices is None and asset is not None:
            vertices = asset.original_vertices
        if vertices is None:
            logger.warning(
                "Could not restore scaled target payload for %s",
                (self._target_name(entry), entry.get("mesh_file")),
            )
            return False
        self._set_mesh_vertices(mesh, vertices)
        return True

    # Coordinate conversion

    @staticmethod
    def sionna_ypr_to_xyz_rotation(
        yaw: float, pitch: float, roll: float
    ) -> Tuple[float, float, float]:
        """Map Sionna RT rotation angles to X/Y/Z rotation components.

        Sionna RT frame metadata stores yaw/pitch/roll. Target and marker
        transforms consume those values as X/Y/Z matrix components.

        Args:
            yaw: Yaw angle in radians.
            pitch: Pitch angle in radians.
            roll: Roll angle in radians.

        Returns:
            Tuple of (rx, ry, rz) angles in radians.
        """
        return sionna_ypr_to_xyz_rotation(yaw, pitch, roll)

    # Scale helper

    def apply_target_scale_from_metadata(
        self,
        target_entry: Dict[str, Any],
        scale_value: float,
        *,
        sync_renderer: bool = True,
    ) -> bool:
        """Apply a scale update coming from generator metadata.

        Resets the mesh vertices to their cached original state, then
        applies the requested uniform scale. Handles mesh-switching by
        tracking which mesh file the scale was last applied to.

        Args:
            target_entry: The mutable target entry dict.
            scale_value: The desired uniform scale factor.
            sync_renderer: Whether to publish the completed state immediately.
                Frame assembly passes ``False`` and performs one final object
                synchronization after position, orientation, and visibility
                have also been resolved.

        Returns:
            True if the scale was actually applied, False otherwise.
        """
        viz = self.visualizer
        try:
            new_scale = float(scale_value)
        except (TypeError, ValueError):
            logger.debug(
                "Invalid scale value '%s' for target '%s'",
                scale_value,
                target_entry.get("name"),
            )
            return False

        target_name = target_entry.get("target_name") or target_entry.get("name", "unknown")

        old_scale = float(target_entry.get("scale", 1.0))
        scale_unchanged = abs(new_scale - old_scale) < 1e-6

        current_mesh_file = target_entry.get("mesh_file")
        last_mesh_file = target_entry.get("_last_applied_scale_mesh_file")

        mesh_switched = last_mesh_file is None or (
            last_mesh_file is not None and current_mesh_file != last_mesh_file
        )

        if scale_unchanged and not mesh_switched:
            last_applied_scale = target_entry.get("_last_applied_scale")
            if last_applied_scale is not None and abs(float(last_applied_scale) - new_scale) < 1e-6:
                logger.debug(
                    "Scale unchanged for '%s': %s and already applied " "to mesh '%s' (skipping)",
                    target_name,
                    new_scale,
                    current_mesh_file,
                )
                target_entry["scale"] = new_scale
                return False
        elif mesh_switched:
            logger.debug(
                "Reapplying scale %s to '%s' (mesh switched)",
                new_scale,
                current_mesh_file,
            )

        mesh = target_entry.get("mesh")
        if mesh is None:
            target_entry["scale"] = new_scale
            return False

        mesh_file = target_entry.get("mesh_file")
        if not mesh_file:
            logger.warning(
                "No mesh_file in target_entry for '%s', cannot apply scale",
                target_name,
            )
            target_entry["scale"] = new_scale
            return False

        logger.debug("Applying scale %s to '%s'", new_scale, target_name)
        cache_key = (target_name, mesh_file)
        asset = self._asset_for_entry(target_entry)
        original_vertices = asset.original_vertices if asset is not None else None

        if original_vertices is None:
            logger.warning(
                "No original vertices cache for '%s' mesh '%s' " "(cache_key=%s)",
                target_name,
                mesh_file,
                cache_key,
            )
            logger.debug(
                "Available target asset keys: %s", self._target_asset_cache.logical_keys()[:10]
            )

        try:
            if original_vertices is not None and original_vertices.size:
                self._set_mesh_vertices(mesh, original_vertices.copy())
                center = _aabb_center(mesh)
                self._scale_mesh(mesh, new_scale, center)
                logger.debug(
                    "Applied scale %s to '%s' mesh '%s' using original vertices",
                    new_scale,
                    target_name,
                    mesh_file,
                )
            else:
                scale_ratio = new_scale / old_scale if abs(old_scale) > 1e-9 else new_scale
                self._scale_mesh(mesh, scale_ratio, _aabb_center(mesh))
                logger.debug(
                    "Applied relative scale %s to '%s' mesh '%s' " "(no original vertices cache)",
                    scale_ratio,
                    target_name,
                    mesh_file,
                )

            scaled_vertices = self._mesh_vertices_array(mesh)
            if asset is not None:
                asset.scaled_vertices = scaled_vertices
                asset.geometry_meta = TargetGeometryMeta(
                    scaled_aabb_center=np.asarray(_aabb_center(mesh), dtype=np.float64)
                )
                self._target_asset_cache.put(asset)
            else:
                self._store_target_geometry_meta(
                    cache_key,
                    scaled_vertices=scaled_vertices,
                )
            self._invalidate_target_outline(target_entry)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning(
                "Failed to apply scale %s to target '%s' mesh '%s': %s",
                new_scale,
                target_name,
                mesh_file,
                exc,
            )
            return False

        target_entry["scale"] = new_scale
        target_entry["_last_applied_scale"] = new_scale
        target_entry["_last_applied_scale_mesh_file"] = mesh_file
        state = target_entry.get("transform_state")
        if isinstance(state, dict):
            state["scale"] = new_scale

        target_name_for_override = target_entry.get("target_name") or target_entry.get("name")
        if target_name_for_override:
            override_scale = viz.target_scale_overrides.get(target_name_for_override)
            if override_scale is not None and abs(override_scale - new_scale) < 1e-6:
                viz.target_scale_overrides.pop(target_name_for_override, None)

        target_entry["_renderer_sync_pending"] = True

        if sync_renderer and viz.vis_initialized:
            try:
                geometry_meta = self._get_target_geometry_meta(
                    target_name,
                    mesh_file,
                    mesh,
                )
                if target_entry.get("_target_position") is not None:
                    target_entry["_mesh_center"] = list(geometry_meta.scaled_aabb_center)
                self.sync_target_entry_snapshot(target_entry)
            except (RuntimeError, ValueError):
                logger.debug("Unable to update geometry after scale change", exc_info=True)
                target_entry["_renderer_sync_pending"] = True

        logger.debug("Applied scale %.3f to target '%s'", new_scale, target_name)
        return True

    # Target model loading

    def load_target_models(self) -> None:
        """Load all target 3D models from the first frame's metadata.

        Reads ``targets_metadata`` from the first frame, resolves mesh sources,
        loads only each target's initial neutral payload, and registers the
        remaining animation files for bounded background look-ahead.
        """
        viz = self.visualizer

        def visual_payload(frame_data: Any) -> Any:
            """Return the mapping consumed by target-model loading."""
            if not isinstance(frame_data, StandardMPCFrame):
                return frame_data
            return standard_frame_to_visual_frame(
                frame_data,
                request=visual_frame_read_request(),
                points_dtype=getattr(viz.mpc_core, "canon_points_dtype", np.float32),
            )

        logger.debug("Starting target model loading...")

        if not viz.frame_source:
            logger.info("No frame source yet; skipping target model loading")
            return

        logger.debug("Frame source type: %s", type(viz.frame_source))

        first_frame = 0
        if hasattr(viz.frame_source, "list_frames"):
            try:
                available_frames = viz.frame_source.list_frames()
                if available_frames:
                    first_frame = min(available_frames)
            except (OSError, TypeError, ValueError):
                pass

        if not viz.frame_source.has_frame(first_frame):
            logger.info("Frame %d not available yet; trying to load it...", first_frame)
            try:
                frame_data = viz.frame_source.load_frame(first_frame)
                if frame_data is None:
                    logger.warning(
                        "Failed to load frame %d; skipping target model loading", first_frame
                    )
                    return

            except (OSError, KeyError, ValueError) as e:
                logger.warning(
                    "Error loading frame %d: %s; skipping target model loading", first_frame, e
                )
                return

        viz._set_status_message("Loading target models...")

        viz.target_entries.clear()
        viz.target_meshes.clear()
        viz.cache_service.invalidate(
            CacheInvalidationScope.TARGET_GEOMETRY,
            reason="target_models_reload",
        )

        try:
            data = viz.cache_service.get_frame(first_frame)
            if data is None:
                load_frame = getattr(
                    getattr(viz, "animation_service", None), "ensure_step_cached", None
                )
                if callable(load_frame):
                    data = load_frame(first_frame)
                elif viz.frame_source:
                    data = viz.frame_source.load_frame(first_frame)
                    data = visual_payload(data)
                    if data is not None and not viz.cache_service.has_frame(first_frame):
                        viz.cache_service.store_frame(first_frame, data)
                else:
                    logger.warning("No frame source available for target loading")
                    return

            data = visual_payload(data)

            if data is None:
                logger.warning("No frame data available for target loading")
                return

            if "targets_metadata" not in data:
                logger.debug("No targets_metadata in frame data (keys: %s)", list(data.keys()))
                viz._set_status_message("No target metadata found in frame data")
                return

            targets_metadata = data["targets_metadata"]
            viz.num_targets = len(targets_metadata)

            if viz.num_targets == 0:
                return

            viz._set_status_message(f"Loading {viz.num_targets} target models...")
            target_assets = self._target_asset_cache
            target_assets.configure(
                max_entries=max(
                    DEFAULT_TARGET_ASSET_CACHE_ENTRIES,
                    viz.num_targets * (TARGET_ASSET_LOOKAHEAD + 2),
                )
            )

            for i, target_meta in enumerate(targets_metadata):
                target_name = target_meta["name"]
                mesh_file = target_meta["mesh_file"]
                mesh_directory = target_meta["mesh_directory"]
                logger.debug("Mesh directory from metadata: %s", mesh_directory)

                if hasattr(viz, "current_scenario_policy") and viz.current_scenario_policy:
                    try:
                        if mesh_directory.startswith("libraries/"):
                            resolved_mesh_dir = viz.current_scenario_policy.resolve_path(
                                mesh_directory,
                                base=viz.current_scenario_policy.project_root,
                            )
                        else:
                            resolved_mesh_dir = viz.current_scenario_policy.resolve_path(
                                mesh_directory
                            )

                        mesh_directory = str(resolved_mesh_dir)
                    except (OSError, ValueError) as e:
                        logger.warning("Failed to resolve mesh path with policy: %s", e)

                if not os.path.exists(mesh_directory):
                    logger.warning("Mesh directory does not exist: %s", mesh_directory)

                scale = target_meta["scale"]
                orientation = target_meta["orientation"]
                use_ply_position = bool(target_meta.get("use_ply_position", False))
                material_type = target_meta["material_type"]

                mesh_file = target_meta["mesh_file"]
                full_mesh_path = os.path.join(mesh_directory, mesh_file)

                if not os.path.exists(full_mesh_path):
                    logger.warning("Mesh file does not exist: %s", full_mesh_path)

                ext = os.path.splitext(mesh_file)[1] or ".ply"
                mesh_pattern = f"*{ext}"

                if mesh_directory == ".":
                    full_pattern = mesh_pattern
                else:
                    full_pattern = os.path.join(mesh_directory, mesh_pattern)

                logger.debug("Original mesh_file: '%s'", mesh_file)
                logger.debug("Extracted pattern: '%s'", mesh_pattern)
                logger.debug("Full search pattern: '%s'", full_pattern)
                logger.debug("Current directory: %s", os.getcwd())

                matching_mesh_files = glob.glob(full_pattern)

                matching_mesh_files.sort()

                if not matching_mesh_files:
                    logger.warning("No mesh files found matching pattern: %s", mesh_pattern)
                    continue

                logger.debug(
                    "Found %s mesh files for target '%s'",
                    len(matching_mesh_files),
                    target_name,
                )

                try:
                    target_assets.register_sequence(target_name, matching_mesh_files)
                except OSError as exc:
                    logger.warning(
                        "Could not register target animation sources for '%s': %s",
                        target_name,
                        exc,
                    )
                    continue
                matching_by_name = {os.path.basename(path): path for path in matching_mesh_files}
                initial_mesh_path = matching_by_name.get(
                    os.path.basename(mesh_file),
                    matching_mesh_files[0],
                )
                initial_mesh_filename = os.path.basename(initial_mesh_path)

                # Build authored/catalog defaults without evaluating profile
                # rules here. MaterialPBRService is the one owner that applies
                # profile binding, overrides, alpha, texture policy, and the
                # final immutable renderer payload.
                material_defaults = resolve_target_pbr_props(
                    target_name=target_name,
                    material_type=material_type,
                    visual_profile_service=None,
                )
                prototype_entry: dict[str, Any] = {
                    "name": target_name,
                    "target_name": target_name,
                    "entry_type": "target",
                    "material_type": material_type,
                    "pbr_properties": material_defaults.props,
                    "color": material_defaults.props.get("color", [0.8, 0.6, 0.5]),
                }
                target_texture_path = target_meta.get("texture_path")
                if target_texture_path:
                    prototype_entry["texture_path"] = target_texture_path
                material_service = getattr(viz, "material_pbr_service", None)
                resolve_entry_material = getattr(
                    material_service,
                    "resolve_entry_material",
                    None,
                )
                resolved_target_material = None
                if callable(resolve_entry_material):
                    candidate = resolve_entry_material(prototype_entry)
                    if isinstance(candidate, ResolvedMaterial):
                        resolved_target_material = candidate
                if resolved_target_material is not None:
                    pbr_props = resolved_target_material.properties_copy()
                else:
                    fallback_resolution = resolve_target_pbr_props(
                        target_name=target_name,
                        material_type=material_type,
                        visual_profile_service=getattr(viz, "visual_profile_service", None),
                    )
                    pbr_props = fallback_resolution.props
                    resolved_target_material = resolve_pbr_material(
                        pbr_props.get("color", [0.8, 0.6, 0.5]),
                        pbr_props,
                        context=f"target:{target_name}",
                    )
                build_spec = self._target_asset_build_spec(
                    scale=scale,
                    orientation=orientation,
                    use_ply_position=use_ply_position,
                    pbr_props=pbr_props,
                    resolved_material=resolved_target_material,
                )
                try:
                    first_asset = self._load_registered_target_asset(
                        target_name,
                        initial_mesh_filename,
                        build_spec,
                    )
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    logger.warning(
                        "Initial target mesh load failed for %s: %s",
                        initial_mesh_path,
                        exc,
                    )
                    continue

                first_mesh = first_asset.mesh
                target_assets.pin(first_asset)
                has_vertex_texture = first_asset.has_vertex_texture
                if has_vertex_texture:
                    logger.info(
                        "Target '%s' has rich vertex colors — preserving PLY texture",
                        target_name,
                    )
                color = pbr_props["color"]
                target_material_id = target_meta.get(
                    "material_id", f"mat-itu_{material_type}_{target_name}"
                )
                orientation_rad, orientation_deg = orientation_metadata(orientation)

                target_entry = {
                    "name": (f"{target_name} " f"({os.path.splitext(initial_mesh_filename)[0]})"),
                    "mesh": first_mesh,
                    "material_id": target_material_id,
                    "color": color,
                    "visible": True,
                    "show_label": True,
                    "highlighted": False,
                    "id_edit": None,
                    "entry_type": "target",
                    "xml_bsdf": None,
                    "xml_shape": None,
                    "rel_path": initial_mesh_path,
                    "target_name": target_name,
                    "node_name": target_name,
                    "node_index": len(viz.target_entries),
                    "mesh_file": initial_mesh_filename,
                    "scale": scale,
                    "orientation": orientation,
                    "texture_path": target_texture_path,
                    "orientation_radians": orientation_rad,
                    "orientation_degrees": orientation_deg,
                    "material_type": material_type,
                    "position": list(target_meta.get("current_position", [0.0, 0.0, 0.0])),
                    "_last_applied_scale": scale,
                    "_last_applied_scale_mesh_file": initial_mesh_filename,
                    "_target_asset": first_asset,
                    "_target_asset_build_spec": build_spec,
                    "supports_position": True,
                    "supports_orientation": True,
                    "supports_scale": True,
                    "supports_label_toggle": True,
                    "supports_highlight_toggle": True,
                    "has_vertex_texture": has_vertex_texture,
                    **target_entry_pbr_fields(pbr_props),
                    "_visual_profile_matched": bool(
                        isinstance(
                            prototype_entry.get("_visual_material_binding"), VisualMaterialBinding
                        )
                        and prototype_entry["_visual_material_binding"].source
                        is VisualMaterialSource.PROFILE
                    ),
                }
                binding = prototype_entry.get("_visual_material_binding")
                if isinstance(binding, VisualMaterialBinding):
                    target_entry["_visual_material_binding"] = binding
                ensure_target_entry_identity(target_entry, len(viz.target_entries))

                viz.target_entries.append(target_entry)

                viz.target_meshes[target_name] = first_mesh

                logger.info(
                    "Loaded target '%s' initial mesh; registered %s animation frames",
                    target_name,
                    len(matching_mesh_files),
                )

            # Prime animation only after every target has its initial frame;
            # background I/O must not contend with first-visible startup work.
            self._schedule_target_lookahead_waves(viz.target_entries)

            loaded_count = len(viz.target_entries)
            if loaded_count > 0:
                viz._set_status_message(
                    f"Loaded {len(viz.mesh_entries)} objects + {loaded_count} targets."
                )
                if not (viz.frame_source and viz.frame_source.has_frame(0)):
                    logger.info("No initial frame data found, targets will use PLY positions")

                logger.debug("Refreshing object list with %s targets", loaded_count)
                if (
                    hasattr(viz, "ui_manager")
                    and hasattr(viz.ui_manager, "panels")
                    and "objects" in viz.ui_manager.panels
                ):
                    try:
                        total_objects = (
                            len(viz.mesh_entries)
                            + len(viz.target_entries)
                            + len(viz.tx_entries)
                            + len(viz.rx_entries)
                        )
                        target_count = len(viz.target_entries)
                        viz.ui_manager.panels["objects"].update_object_count(
                            total_objects,
                            target_count,
                            len(viz.tx_entries),
                            len(viz.rx_entries),
                        )

                        search_text = getattr(viz, "object_search_filter", None)
                        search_text = search_text.text().lower() if search_text else ""
                        group_by = getattr(viz, "group_by_combo", None)
                        group_by = group_by.currentText() if group_by else "Material"

                        viz.ui_manager.panels["objects"].populate_object_list(
                            viz.mesh_entries,
                            viz.target_entries,
                            viz.tx_entries,
                            viz.rx_entries,
                            search_text,
                            group_by,
                        )
                        logger.debug("Object list refreshed with %s targets", target_count)
                    except (RuntimeError, ValueError, AttributeError) as e:
                        logger.warning("Could not refresh object list after target loading: %s", e)
                else:
                    logger.warning("UI manager not available for object list refresh")

            else:
                if viz.frame_source is not None and viz.total_animation_steps > 0:
                    viz._set_status_message(
                        f"Loaded {len(viz.mesh_entries)} objects + "
                        "MPC data available for animation"
                    )
                else:
                    viz._set_status_message(
                        f"Loaded {len(viz.mesh_entries)} objects " "(no targets or MPC data)"
                    )

        except (OSError, KeyError, ValueError, RuntimeError) as e:
            logger.error("Error loading target models: %s", e)
            viz._set_status_message(f"Error loading target models: {e}")

    def _load_mesh_on_demand(
        self,
        target_name: str,
        mesh_file: str,
        target_meta: dict,
        viz: "OrchavVisualizer",
    ) -> TargetAsset | None:
        """Load a single target asset into the cache on demand.

        This is the normal slow path for a lazily loaded or LRU-evicted frame,
        and also covers files that were absent from the initial sequence.

        Returns:
            The loaded target asset, or ``None`` on failure.
        """
        target_assets = self._target_asset_cache
        source = target_assets.source_for(target_name, mesh_file)
        if source is None:
            mesh_directory = target_meta.get("mesh_directory", ".")
            if hasattr(viz, "current_scenario_policy") and viz.current_scenario_policy:
                try:
                    if mesh_directory.startswith("libraries/"):
                        resolved = viz.current_scenario_policy.resolve_path(
                            mesh_directory,
                            base=viz.current_scenario_policy.project_root,
                        )
                    else:
                        resolved = viz.current_scenario_policy.resolve_path(mesh_directory)
                    mesh_directory = str(resolved)
                except (OSError, ValueError):
                    pass
            full_path = os.path.join(mesh_directory, mesh_file)
            if not os.path.isfile(full_path):
                logger.warning("On-demand mesh file not found: %s", full_path)
                return None
            try:
                target_assets.register_source(target_name, full_path)
            except OSError as exc:
                logger.warning("Could not register on-demand target mesh %s: %s", full_path, exc)
                return None

        target_entry = next(
            (
                entry
                for entry in getattr(viz, "target_entries", [])
                if self._target_name(entry) == target_name
            ),
            None,
        )
        spec = (
            target_entry.get("_target_asset_build_spec") if isinstance(target_entry, dict) else None
        )
        if not isinstance(spec, _TargetAssetBuildSpec):
            material_type = target_meta.get("material_type", "custom")
            pbr_props = resolve_target_pbr_props(
                target_name=target_name,
                material_type=material_type,
                visual_profile_service=getattr(viz, "visual_profile_service", None),
                default_props=MINIMAL_TARGET_PBR_PROPS,
                use_material_fallbacks=False,
            ).props
            spec = self._target_asset_build_spec(
                scale=target_meta.get("scale", 1.0),
                orientation=target_meta.get("orientation", [0.0, 0.0, 0.0]),
                use_ply_position=bool(target_meta.get("use_ply_position", False)),
                pbr_props=pbr_props,
            )
            if isinstance(target_entry, dict):
                target_entry["_target_asset_build_spec"] = spec

        try:
            asset = self._load_registered_target_asset(target_name, mesh_file, spec)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            logger.warning(
                "On-demand mesh load failed for %s/%s: %s",
                target_name,
                mesh_file,
                exc,
            )
            return None

        logger.info("On-demand loaded mesh: %s/%s", target_name, mesh_file)
        return asset

    # ViewModel-based target processing

    def process_targets_from_view_model(self, step: int, view_model: Any) -> bool:
        """Process target updates for a single animation frame.

        Uses data from the unified ViewModel to switch meshes, apply
        position/orientation/scale transforms, and update the active
        renderer. Change-detection caching avoids redundant work when
        a target's state is unchanged between frames.

        Args:
            step: The 0-based animation step index.
            view_model: The ViewModel containing target data arrays.
        """
        viz = self.visualizer
        if not hasattr(view_model, "target_positions") or not hasattr(
            view_model, "target_metadata"
        ):
            return True

        target_assets = self._target_asset_cache
        process_targets_start = time.perf_counter()
        self._last_runtime_breakdown = {}
        self._target_render_sync.reset_benchmark_metrics()

        logger.debug(
            "Processing targets from ViewModel for step %s",
            step,
        )
        logger.debug(
            "Found %s targets",
            len(view_model.target_positions),
        )

        updated_entries: List[Dict[str, Any]] = []
        processed_target_names: set[str] = set()

        target_entries_by_name: dict[str, tuple[int, Dict[str, Any]]] = {}
        for stable_index, entry in enumerate(viz.target_entries):
            entry_name = entry.get("target_name") or entry.get("name")
            if entry_name is not None:
                target_entries_by_name[str(entry_name)] = (stable_index, entry)

        state_cache = target_assets.runtime_states
        for i, target_meta in enumerate(view_model.target_metadata):
            target_name = str(target_meta.get("name") or f"target_{i}")
            if i >= len(view_model.target_positions):
                logger.debug("Skipping target %s: missing target_positions[%d]", target_name, i)
                continue
            if i >= len(view_model.target_orientations):
                logger.debug("Skipping target %s: missing target_orientations[%d]", target_name, i)
                continue
            if i >= len(view_model.target_mesh_files):
                logger.debug("Skipping target %s: missing target_mesh_files[%d]", target_name, i)
                continue
            if i >= len(view_model.target_use_ply_positions):
                logger.debug(
                    "Skipping target %s: missing target_use_ply_positions[%d]", target_name, i
                )
                continue

            current_position = view_model.target_positions[i]
            orientation = view_model.target_orientations[i]
            mesh_file = view_model.target_mesh_files[i]
            use_ply_position = view_model.target_use_ply_positions[i]
            position_valid_meta = bool(target_meta.get("position_valid", True))
            pos_list, position_valid_data = self._coerce_vec3(current_position)
            position_valid = bool(position_valid_meta and position_valid_data)
            frame_visible = bool(position_valid or use_ply_position)
            processed_target_names.add(target_name)

            pos_tuple = tuple(pos_list)
            orient_tuple = (
                tuple(float(x) for x in orientation) if orientation is not None else (0.0, 0.0, 0.0)
            )

            target_match = target_entries_by_name.get(target_name)
            if target_match is None:
                logger.warning(
                    "No target entry found for %s",
                    target_name,
                )
                logger.warning("Target entries: %s", viz.target_entries)
                continue

            stable_index, target_entry = target_match
            target_entry["node_index"] = stable_index
            ensure_target_entry_identity(target_entry, stable_index)

            scale_to_apply = self._normalize_runtime_scale(target_meta.get("scale"))
            # Missing frame metadata preserves the currently assembled target
            # scale for state comparison, but should not reapply that scale on
            # every pose-only frame.
            runtime_scale = (
                scale_to_apply
                if scale_to_apply is not None
                else self._normalize_runtime_scale(target_entry.get("scale"))
            )
            override_scale = self._normalize_runtime_scale(
                viz.target_scale_overrides.get(target_name)
            )
            if override_scale is not None:
                if scale_to_apply is None or abs(scale_to_apply - override_scale) > 1e-6:
                    scale_to_apply = override_scale
                else:
                    viz.target_scale_overrides.pop(target_name, None)
                runtime_scale = override_scale

            previous_entry_use_ply_position = bool(target_entry.get("_use_ply_position", False))

            desired_runtime_visible = target_runtime_visible(
                target_entry,
                stable_index,
                self._is_hidden_for_pov,
                frame_visible=frame_visible,
            )

            cached_state = state_cache.get(target_name)
            orientation_changed = True
            mesh_changed = True
            use_ply_position_changed = previous_entry_use_ply_position != bool(use_ply_position)
            if isinstance(cached_state, TargetRuntimeState):
                pos_unchanged = np.allclose(pos_tuple, cached_state.position, atol=1e-6)
                orient_unchanged = np.allclose(
                    orient_tuple,
                    cached_state.orientation,
                    atol=1e-6,
                )
                mesh_unchanged = cached_state.mesh_filename == mesh_file
                position_valid_unchanged = cached_state.position_valid == bool(position_valid)
                use_ply_position_unchanged = cached_state.use_ply_position == bool(use_ply_position)
                orientation_changed = not orient_unchanged
                mesh_changed = not mesh_unchanged
                use_ply_position_changed = not use_ply_position_unchanged
                runtime_visibility_unchanged = (
                    cached_state.runtime_visible == desired_runtime_visible
                )
                scale_unchanged = (
                    cached_state.scale is None
                    if runtime_scale is None
                    else cached_state.scale is not None
                    and abs(cached_state.scale - runtime_scale) < 1e-6
                )

                if (
                    pos_unchanged
                    and orient_unchanged
                    and mesh_unchanged
                    and position_valid_unchanged
                    and use_ply_position_unchanged
                    and runtime_visibility_unchanged
                    and scale_unchanged
                    and not bool(target_entry.get("_renderer_sync_pending", False))
                ):
                    self._target_render_sync.record_benchmark_metric(
                        "target_state_unchanged_skip_count"
                    )
                    logger.debug(
                        "Skipping %s - no changes detected",
                        target_name,
                    )
                    # Direction can change without changing target geometry.
                    # The schedule token makes the ordinary unchanged path a
                    # no-op while still priming the new playback direction.
                    self._schedule_target_lookahead(target_entry, count=1)
                    continue

            logger.debug(
                "Processing target %s for step %s: pos=%s, orient=%s",
                target_name,
                step,
                current_position,
                orientation,
            )
            self._target_render_sync.record_benchmark_metric("target_state_changed_count")

            target_entry["_frame_visible"] = frame_visible
            target_entry["_use_ply_position"] = bool(use_ply_position)
            if frame_visible:
                target_entry["position"] = [float(p) for p in pos_list]
            target_entry["node_name"] = target_entry.get("node_name") or target_name
            if orientation is None:
                orientation_rad, orientation_deg = orientation_metadata(None)
            else:
                orientation_rad, orientation_deg = orientation_metadata(orientation)
            target_entry["orientation_radians"] = orientation_rad
            target_entry["orientation_degrees"] = orientation_deg
            target_entry["orientation"] = orientation_rad
            target_entry["supports_position"] = True
            target_entry["supports_orientation"] = True
            target_entry["supports_scale"] = True

            current_mesh_file = target_entry.get("mesh_file")
            if current_mesh_file != mesh_file:
                mesh_changed = True
                mesh_switch_start = time.perf_counter()
                self._target_render_sync.record_benchmark_metric("target_mesh_switch_count")
                logger.debug(
                    "Switching mesh for %s: %s -> %s",
                    target_name,
                    current_mesh_file,
                    mesh_file,
                )

                new_asset = target_assets.get(target_name, mesh_file)
                if new_asset is not None:
                    self._target_render_sync.record_benchmark_metric(
                        "target_mesh_switch_cache_hit_count"
                    )
                else:
                    self._target_render_sync.record_benchmark_metric(
                        "target_mesh_switch_cache_miss_count"
                    )
                    logger.debug(
                        "Mesh not found in cache: %s — attempting on-demand load",
                        (target_name, mesh_file),
                    )
                    new_asset = self._load_mesh_on_demand(
                        target_name,
                        mesh_file,
                        target_meta,
                        viz,
                    )
                    if new_asset is None:
                        logger.warning(
                            "On-demand load failed for %s; keeping current mesh '%s'",
                            mesh_file,
                            target_entry.get("mesh_file"),
                        )

                if new_asset is not None:
                    new_mesh = new_asset.mesh
                    old_mesh = target_entry["mesh"]
                    old_asset = target_entry.get("_target_asset")
                    old_color_source = (
                        old_mesh.payload.color_source
                        if isinstance(old_mesh, RenderObjectState)
                        and isinstance(old_mesh.payload, MeshPayload)
                        else SurfaceColorSource.MATERIAL
                    )
                    color_source_changed = (old_color_source is SurfaceColorSource.VERTEX) != bool(
                        new_asset.has_vertex_texture
                    )
                    target_assets.pin_handoff(
                        new_asset,
                        old_asset if isinstance(old_asset, TargetAsset) else None,
                    )
                    if not color_source_changed:
                        # Most animation sequences retain one color owner. Reuse
                        # the resolved material without entering material
                        # resolution in the frame-hot path.
                        new_mesh.material = old_mesh.material
                    else:
                        spec = target_entry.get("_target_asset_build_spec")
                        if isinstance(spec, _TargetAssetBuildSpec):
                            new_mesh.material = self._target_asset_base_material(new_asset, spec)
                    # A cached frame may have been active earlier and therefore
                    # mutated by baked orientation. Restore its authoritative
                    # scaled baseline before assembling the new snapshot.
                    self._set_mesh_vertices(new_mesh, new_asset.scaled_vertices)
                    new_mesh.world_transform = Transform.identity()

                    target_entry["mesh"] = new_mesh
                    target_entry["mesh_file"] = mesh_file
                    target_entry["rel_path"] = new_asset.source.canonical_path
                    target_entry["_target_asset"] = new_asset
                    target_entry["has_vertex_texture"] = new_asset.has_vertex_texture
                    if color_source_changed:
                        material_service = getattr(viz, "material_pbr_service", None)
                        resolve_entry_material = getattr(
                            material_service,
                            "resolve_entry_material",
                            None,
                        )
                        if callable(resolve_entry_material):
                            resolved = resolve_entry_material(target_entry)
                            if isinstance(resolved, ResolvedMaterial):
                                new_mesh.material = resolved.payload
                    target_entry["_renderer_sync_pending"] = True
                    target_entry["_mesh_center"] = list(new_asset.geometry_meta.scaled_aabb_center)
                    target_entry["_last_applied_scale_mesh_file"] = mesh_file
                    spec = target_entry.get("_target_asset_build_spec")
                    if isinstance(spec, _TargetAssetBuildSpec):
                        target_entry["_last_applied_scale"] = spec.scale
                    self._invalidate_target_outline(target_entry)
                self._last_runtime_breakdown["mesh_switch_ms"] = (
                    self._last_runtime_breakdown.get("mesh_switch_ms", 0.0)
                    + (time.perf_counter() - mesh_switch_start) * 1000.0
                )

            updated_entries.append(target_entry)

            if scale_to_apply is not None:
                skip_identity_scale_apply = (
                    not use_ply_position
                    and abs(float(scale_to_apply) - 1.0) < 1e-6
                    and self._asset_for_entry(target_entry) is not None
                )
                if skip_identity_scale_apply:
                    target_entry["scale"] = float(scale_to_apply)
                    target_entry["_last_applied_scale"] = float(scale_to_apply)
                    target_entry["_last_applied_scale_mesh_file"] = target_entry.get("mesh_file")
                else:
                    self.apply_target_scale_from_metadata(
                        target_entry,
                        scale_to_apply,
                        sync_renderer=False,
                    )

            # Record applied state, not merely desired metadata. A failed mesh
            # load or scale mutation must remain different on the next frame
            # so the normal change detector retries it.
            actual_mesh_file = target_entry.get("mesh_file", mesh_file)
            state_cache[target_name] = TargetRuntimeState(
                position=pos_tuple,
                orientation=orient_tuple,
                mesh_filename=str(actual_mesh_file),
                position_valid=bool(position_valid),
                use_ply_position=bool(use_ply_position),
                runtime_visible=bool(desired_runtime_visible),
                scale=self._normalize_runtime_scale(target_entry.get("scale")),
            )

            if not use_ply_position and position_valid:
                mesh = target_entry["mesh"]
                logger.debug(
                    "Updating %s position: %s",
                    target_name,
                    current_position,
                )

                # Cached payloads stay in scaled, unrotated local space.
                # Frame playback changes only the declarative transform.
                geometry_meta = self._get_target_geometry_meta(
                    target_name, target_entry.get("mesh_file"), mesh
                )
                center = np.asarray(geometry_meta.scaled_aabb_center, dtype=np.float64)

                target_entry["_use_transform_position"] = True
                target_entry["_target_position"] = pos_list
                target_entry["_mesh_center"] = list(center)
                logger.debug(
                    "%s using transform-based positioning, center=%s",
                    target_name,
                    center,
                )

            orient_list = (
                orientation.tolist() if hasattr(orientation, "tolist") else list(orientation)
            )
            if position_valid or use_ply_position:
                mesh = target_entry["mesh"]
                yaw, pitch, roll = orient_list
                has_rotation = any(abs(angle) > 1e-6 for angle in orient_list)

                if use_ply_position:
                    # PLY placement owns payload coordinates. Rebuild from the
                    # scaled baseline whenever the baked orientation or mode
                    # changes, including a transition back to zero rotation.
                    target_entry.pop("_rotation_matrix", None)
                    if orientation_changed or mesh_changed or use_ply_position_changed:
                        payload_center = _aabb_center(mesh)
                        if self._restore_scaled_target_payload(target_entry, mesh):
                            if has_rotation:
                                rotation_transform = np.eye(4, dtype=np.float64)
                                rotation_transform[:3, :3] = build_sionna_rotation_matrix(
                                    yaw,
                                    pitch,
                                    roll,
                                )
                                self._transform_mesh_payload(mesh, rotation_transform)
                            updated_center = _aabb_center(mesh)
                            self._translate_mesh_by(mesh, payload_center - updated_center)
                else:
                    # Transform-managed targets keep an unrotated local payload.
                    # A prior PLY-baked frame must not leak into this mode.
                    if use_ply_position_changed:
                        self._restore_scaled_target_payload(target_entry, mesh)
                    if has_rotation:
                        target_entry["_rotation_matrix"] = build_sionna_rotation_matrix(
                            yaw,
                            pitch,
                            roll,
                        )
                    else:
                        target_entry.pop("_rotation_matrix", None)

            # Queue the immediately required successor as soon as this target
            # is assembled. Deeper speculative work is added only after every
            # target has received this first lookahead slot.
            self._schedule_target_lookahead(target_entry, count=1)

            # The renderer update pass below synchronizes outlines after all
            # mesh mutations and applies the final frame/POV visibility.

        for target_entry in viz.target_entries:
            name = target_entry.get("target_name") or target_entry.get("name")
            if name in processed_target_names:
                continue
            was_frame_visible = bool(target_entry.get("_frame_visible", True))
            target_entry["_frame_visible"] = False
            # A returning target with the same pose must not hit the previous
            # present-frame cache entry and remain hidden forever.
            state_cache.pop(name, None)
            if (
                was_frame_visible or bool(target_entry.get("_renderer_sync_pending", False))
            ) and target_entry not in updated_entries:
                updated_entries.append(target_entry)

        if TARGET_ASSET_LOOKAHEAD > 1:
            for target_entry in viz.target_entries:
                self._schedule_target_lookahead(
                    target_entry,
                    count=TARGET_ASSET_LOOKAHEAD,
                )

        logger.debug("Target processing complete for step %s", step)

        viz.current_view_model = view_model
        logger.debug("Stored current ViewModel for camera focus system")

        camera = getattr(viz, "camera_controller", None)
        update_focus_dropdown = getattr(camera, "update_target_focus_dropdown", None)
        if hasattr(viz, "target_focus_dropdown") and callable(update_focus_dropdown):
            update_focus_dropdown()
            logger.debug("Updated target/TX/RX focus dropdown " "after ViewModel storage")

        all_synced = True
        if hasattr(viz, "vis") and viz.vis is not None and updated_entries:
            try:

                def _do_target_updates() -> bool:
                    """Publish one complete declarative snapshot per target."""
                    updates_start = time.perf_counter()
                    self._target_render_sync.record_benchmark_metric("target_batch_update_count")
                    updates_synced = True
                    for target_entry in updated_entries:
                        if "mesh" not in target_entry:
                            target_entry["_renderer_sync_pending"] = True
                            updates_synced = False
                            self._target_render_sync.record_benchmark_metric(
                                "target_batch_missing_mesh_skip_count"
                            )
                            continue
                        updates_synced = (
                            self.sync_target_entry_snapshot(target_entry) and updates_synced
                        )

                    self._last_runtime_breakdown["do_target_updates_ms"] = (
                        time.perf_counter() - updates_start
                    ) * 1000.0
                    return updates_synced

                if hasattr(viz.renderer, "batch_updates"):
                    self._target_render_sync.record_benchmark_metric(
                        "target_renderer_batch_context_count"
                    )
                    with viz.renderer.batch_updates():
                        all_synced = _do_target_updates()
                else:
                    all_synced = _do_target_updates()

                renderer_update_start = time.perf_counter()
                viz.renderer.update_renderer()
                self._last_runtime_breakdown["target_renderer_update_ms"] = (
                    self._last_runtime_breakdown.get("target_renderer_update_ms", 0.0)
                    + (time.perf_counter() - renderer_update_start) * 1000.0
                )
                self._target_render_sync.record_benchmark_metric("target_renderer_update_count")
                logger.debug(
                    "Forced renderer update after %s target changes",
                    len(updated_entries),
                )
            except (RuntimeError, ValueError) as e:
                all_synced = False
                for target_entry in updated_entries:
                    target_entry["_renderer_sync_pending"] = True
                logger.warning("Error updating active renderer: %s", e)
        elif updated_entries:
            all_synced = False
            for target_entry in updated_entries:
                target_entry["_renderer_sync_pending"] = True
            logger.debug("Renderer not available for target update")

        self._last_runtime_breakdown["process_targets_total_ms"] = (
            time.perf_counter() - process_targets_start
        ) * 1000.0
        self._target_render_sync.record_benchmark_metric(
            "target_entries_updated_count",
            float(len(updated_entries)),
        )
        self._last_runtime_breakdown.update(self._target_render_sync.get_benchmark_metrics())
        return all_synced
