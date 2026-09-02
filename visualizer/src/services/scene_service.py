"""Scene loading, material preparation, and renderer handoff service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QTimer

from ..app.renderer_lifecycle import boot_visualizer_empty
from ..materials.appearance import AppearanceIntent, resolve_appearance
from ..materials.catalog import ResolvedMaterial, resolve_pbr_material
from ..materials.texture_policy import TEXTURE_MAP_KEYS, textures_globally_enabled
from ..model import RenderObjectState
from ..renderers.protocol import renderer_capabilities
from ..scene.geometry_helpers import (
    create_building_labels,
    create_target_labels,
    ensure_scene_mesh_render_state,
    require_target_mesh_render_state,
)
from ..scene.geometry_payload_factory import make_sphere_payload
from ..scene.io import XMLSceneHandler, build_scene_from_root, finalize_uv_cache_stores
from ..scene.visibility_policy import (
    effective_entry_label_visibility,
    effective_entry_visibility,
)
from ..state import get_beamforming_state_defaults
from ..types.camera_state import CameraState
from ..types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
    material_payload_from_mapping,
    mesh_payload_for_pbr_material,
)
from .base import BaseService
from .cache_service import CacheInvalidationScope
from .object_identity import (
    ensure_target_entry_identity,
    make_component_geometry_name,
    make_target_entry_geometry_name,
)
from .pov_visibility_service import is_hidden_for_pov
from .scene_batches import PreparedSceneMember, SceneBatch, SceneBatchRegistry
from .scene_render_sync import (
    SceneRenderSync,
    merge_scene_mesh_payloads,
    scene_mesh_has_triangle_uvs,
    scene_mesh_payload,
)

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = logging.getLogger("orchav.scene")

# Path to texture library
TEXTURE_LIBRARY_PATH = Path(__file__).parent.parent.parent.parent / "libraries" / "textures"

_PYGFX_MESH_BUFFER_CACHE_VERSION = "20260605_pygfx_mesh_material_color_v2"
TEXTURE_MAP_PROP_KEYS = TEXTURE_MAP_KEYS


class SceneService(BaseService):
    """Service responsible for loading and managing XML scenes."""

    def __init__(self, visualizer: OrchavVisualizer):
        """Initialize scene path, merge caches, and texture lookup state."""
        super().__init__()
        self.visualizer = visualizer
        self.scene_path: str | None = None

        # One typed registry owns canonical batch membership, reverse indexes,
        # individual exceptions, and retry state.  Compatibility properties
        # below expose read-oriented views for adjacent services while keeping
        # a single source of truth.
        self._scene_batches = SceneBatchRegistry()
        self._merge_enabled: bool = False
        self._render_sync = SceneRenderSync(self.visualizer)

        # Cached texture library scan (populated once on first use)
        self._texture_cache: dict[str, str | None] | None = None
        self._announced_texture_cache_notice_keys: set[str] = set()

    @property
    def _merged_groups(self) -> dict[str, list[int]]:
        """Compatibility view of canonical batch membership."""
        return {
            batch_name: batch.member_mesh_ids
            for batch_name, batch in self._scene_batches.batches.items()
        }

    @property
    def _mesh_id_to_group(self) -> dict[int, str]:
        """Return the registry's canonical reverse batch index."""
        return self._scene_batches.mesh_to_batch

    @property
    def _merged_meshes(self) -> dict[str, SceneBatch]:
        """Return typed scene batches keyed by stable aggregate ID."""
        return self._scene_batches.batches

    @property
    def _mesh_id_to_scene_entry(self) -> dict[int, dict[str, Any]]:
        """Return the registry's canonical mesh-to-entry index."""
        return self._scene_batches.entries_by_mesh_id

    @property
    def _pending_merged_group_syncs(self) -> set[str]:
        """Return batches whose desired renderer snapshot still needs retry."""
        return self._scene_batches.pending_batch_ids

    @property
    def _pending_scene_entry_sources(self) -> dict[int, str]:
        """Return source batches retained across failed ownership transitions."""
        return self._scene_batches.pending_entry_sources

    @property
    def _individual_scene_owner_ids(self) -> set[str]:
        """Return IDs that may currently own individual native scene objects."""
        return self._scene_batches.individual_owner_ids

    @staticmethod
    def _textures_disabled() -> bool:
        """Return True when the CLI requested texture-free rendering."""
        return not textures_globally_enabled()

    def _uv_cache_source_path(self, scene_entry: dict[str, Any] | None) -> str | None:
        """Resolve the absolute source-mesh path used for persisted UV caches."""
        if scene_entry is None:
            return None
        rel_path = scene_entry.get("rel_path")
        if not rel_path:
            return None
        rel_path_str = str(rel_path)
        if os.path.isabs(rel_path_str):
            return rel_path_str
        if not self.scene_path:
            return None
        return str(Path(self.scene_path).resolve(strict=False).parent / rel_path_str)

    def _texture_cache_notice_key(self) -> str:
        """Return a stable per-scene key for one-time texture cache notices."""
        scenario_root = getattr(getattr(self.visualizer, "scenario", None), "root", None)
        if scenario_root is None and self.scene_path:
            scenario_root = Path(self.scene_path).resolve(strict=False).parent
        if scenario_root is None:
            return "scene.unknown"
        return str(Path(scenario_root).resolve(strict=False))

    def _maybe_announce_texture_cache_build(self, *, reason: str) -> None:
        """Emit a one-time status/log notice when this scene is building texture caches."""
        notice_key = self._texture_cache_notice_key()
        if notice_key in self._announced_texture_cache_notice_keys:
            return
        self._announced_texture_cache_notice_keys.add(notice_key)

        from ..scene.io import get_uv_cache_root

        cache_locations = [f"UV cache: {get_uv_cache_root()}"]
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer_capabilities(renderer).mesh_buffer_cache:
            from ..backends.pygfx_scene_helpers import get_pygfx_mesh_buffer_cache_root

            mesh_root = get_pygfx_mesh_buffer_cache_root()
            if mesh_root is not None:
                cache_locations.append(f"pygfx mesh cache: {mesh_root}")

        status_message = (
            "Building scene texture caches. First textured launch can take longer; "
            "later launches will be faster."
        )
        if hasattr(self.visualizer, "_set_status_message"):
            self.visualizer._set_status_message(status_message, 12000)

        logger.info(
            "%s Trigger: cold %s cache. Stored in %s",
            status_message,
            reason,
            "; ".join(cache_locations),
        )

    @staticmethod
    def _sanitize_cache_segment(value: str) -> str:
        """Return a filesystem-friendly cache directory segment."""
        sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
        sanitized = sanitized.strip("._")
        return sanitized or "cache"

    def _pygfx_mesh_cache_namespace(self) -> str:
        """Return a readable per-scenario namespace for pygfx mesh buffer caches."""
        scenario_root = getattr(getattr(self.visualizer, "scenario", None), "root", None)
        if scenario_root is None and self.scene_path:
            scenario_root = Path(self.scene_path).resolve(strict=False).parent
        if scenario_root is None:
            return "scene.unknown"

        scenario_root = Path(scenario_root).resolve(strict=False)
        label = self._sanitize_cache_segment(scenario_root.name)
        digest = hashlib.sha1(str(scenario_root).encode("utf-8")).hexdigest()[:10]
        return f"{label}.{digest}"

    def _pygfx_mesh_buffer_cache_key(
        self,
        scene_entry: dict[str, Any] | None,
        *,
        effective_props: dict[str, Any],
        color_source: SurfaceColorSource = SurfaceColorSource.MATERIAL,
    ) -> str | None:
        """Return a stable cache key for persistent pygfx mesh-buffer reuse."""
        if scene_entry is None:
            return None

        source_path = self._uv_cache_source_path(scene_entry)
        if not source_path:
            return None

        source = Path(source_path).expanduser().resolve(strict=False)
        source_signature = scene_entry.get("_source_signature")
        supplied_path = str(
            source_signature.get("path") if isinstance(source_signature, dict) else ""
        )
        signature_matches = bool(
            supplied_path
            and Path(supplied_path).resolve(strict=False) == source
            and isinstance(source_signature, dict)
        )
        if signature_matches:
            source_size = source_signature.get("size")
            source_mtime_ns = source_signature.get("mtime_ns")
            source_ctime_ns = source_signature.get("ctime_ns")
        else:
            try:
                stat = source.stat()
                source_size = int(stat.st_size)
                source_mtime_ns = int(stat.st_mtime_ns)
                source_ctime_ns = int(stat.st_ctime_ns)
            except OSError:
                source_size = None
                source_mtime_ns = None
                source_ctime_ns = None

        payload = {
            "version": _PYGFX_MESH_BUFFER_CACHE_VERSION,
            "source_path": str(source),
            "source_size": source_size,
            "source_mtime_ns": source_mtime_ns,
            "source_ctime_ns": source_ctime_ns,
            "transform_state": scene_entry.get("transform_state") or {},
            "uv_scale_meters": round(float(effective_props.get("uv_scale_meters", 2.0)), 9),
            "color_source": SurfaceColorSource(color_source).value,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"{self._pygfx_mesh_cache_namespace()}/{digest}"

    def _wrap_pygfx_scene_mesh_payload(
        self,
        mesh: Any,
        *,
        scene_entry: dict[str, Any] | None,
        effective_props: dict[str, Any],
        color_source: SurfaceColorSource = SurfaceColorSource.MATERIAL,
    ) -> Any:
        """Convert a scene mesh to a cacheable payload for the pygfx renderer."""
        renderer = getattr(self.visualizer, "renderer", None)
        payload = self._mesh_payload(mesh)
        if payload is None:
            return mesh
        if not renderer_capabilities(renderer).mesh_buffer_cache:
            prepared = mesh_payload_for_pbr_material(
                payload,
                color_source=color_source,
            )
            return self._replace_scene_mesh_payload(mesh, prepared)
        if scene_entry is None or not self._mesh_has_triangle_uvs(payload):
            prepared = mesh_payload_for_pbr_material(
                payload,
                color_source=color_source,
            )
            return self._replace_scene_mesh_payload(mesh, prepared)

        cache_key = self._pygfx_mesh_buffer_cache_key(
            scene_entry,
            effective_props=effective_props,
            color_source=color_source,
        )

        prepared = mesh_payload_for_pbr_material(
            payload,
            color_source=color_source,
            cache_key=cache_key,
        )
        return self._replace_scene_mesh_payload(mesh, prepared)

    @staticmethod
    def _replace_scene_mesh_payload(mesh: Any, payload: MeshPayload) -> Any:
        """Keep prepared scene payloads on their application-owned render handle."""
        if not isinstance(mesh, RenderObjectState):
            raise TypeError("Prepared persistent scene geometry requires RenderObjectState")
        mesh.replace_payload(payload)
        return mesh

    @staticmethod
    def _mesh_payload(mesh: Any) -> MeshPayload | None:
        """Return a mesh payload for scene-owned neutral geometry."""
        return scene_mesh_payload(mesh)

    def _merge_scene_mesh_payloads(
        self,
        meshes: list[Any],
        *,
        cache_baseline: bool = False,
    ) -> MeshPayload | None:
        """Merge scene payloads and optionally retain a stable batch cache key."""
        merged = merge_scene_mesh_payloads(meshes)
        if merged is None or not cache_baseline:
            return merged
        payloads = [self._mesh_payload(mesh) for mesh in meshes]
        if not payloads or any(payload is None or not payload.cache_key for payload in payloads):
            return merged
        identity = {
            "version": _PYGFX_MESH_BUFFER_CACHE_VERSION,
            "kind": "scene_batch",
            "members": [
                {
                    "cache_key": payload.cache_key,
                    "vertices": tuple(payload.vertices.shape),
                    "triangles": tuple(payload.triangles.shape),
                }
                for payload in payloads
                if payload is not None
            ],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return replace(
            merged,
            cache_key=f"{self._pygfx_mesh_cache_namespace()}/batch/{digest}",
        )

    @staticmethod
    def _mesh_has_triangle_uvs(mesh: Any) -> bool:
        """Return whether neutral scene mesh geometry has triangle UVs."""
        return scene_mesh_has_triangle_uvs(mesh)

    @staticmethod
    def _effective_initial_scene_visibility(
        geometry: RenderObjectState,
        entry: dict[str, Any] | None,
    ) -> bool:
        """Resolve final first-snapshot visibility for one scene member."""
        if entry is None:
            return bool(geometry.visible)
        return effective_entry_visibility(
            entry,
            state_visible=bool(geometry.visible),
            entry_type="mesh",
        )

    def _ensure_persistent_material_geometry(
        self,
        *,
        geometry: RenderObjectState,
        material: dict[str, Any] | MaterialPayload,
        visible: bool = True,
    ) -> bool:
        """Ensure a persistent mesh through the common object contract."""
        return self._render_sync.ensure_object(
            geometry,
            material=material,
            effective_visible=bool(visible),
        )

    @staticmethod
    def _merged_group_render_state(
        group_name: str,
        group_info: SceneBatch,
        *,
        geometry: MeshPayload | None = None,
        material: dict[str, Any] | MaterialPayload | None = None,
        visible: bool | None = None,
    ) -> RenderObjectState | None:
        """Return the stable desired render state for one merged group.

        The group owns this state for its full lifetime. Rebuilds replace only
        its immutable payload, allowing renderer adapters to distinguish real
        geometry changes from material and visibility updates by identity.
        """
        payload = geometry if geometry is not None else group_info.get("geometry")
        state = group_info.get("render_state")
        if not isinstance(state, RenderObjectState):
            if not isinstance(payload, MeshPayload):
                return None
            state = RenderObjectState(
                id=group_name,
                payload=payload,
                metadata={"type": "scene_aggregate"},
            )
            group_info["render_state"] = state
        elif state.id != group_name:
            raise ValueError(
                f"Merged render state id {state.id!r} does not match group {group_name!r}"
            )
        elif isinstance(payload, MeshPayload) and state.payload is not payload:
            state.replace_payload(payload)

        if material is not None:
            state.material = (
                material
                if isinstance(material, MaterialPayload)
                else material_payload_from_mapping(material)
            )
        if visible is not None:
            state.visible = bool(visible)
        return state

    def _ensure_merged_material_geometry(
        self,
        *,
        group_name: str,
        group_info: SceneBatch,
        geometry: MeshPayload,
        material: dict[str, Any] | MaterialPayload,
        visible: bool = True,
    ) -> bool:
        """Ensure a merged aggregate through its stable render state."""
        state = self._merged_group_render_state(
            group_name,
            group_info,
            geometry=geometry,
            material=material,
            visible=visible,
        )
        if state is None:
            return False
        return self._render_sync.ensure_object(state, effective_visible=bool(visible))

    @staticmethod
    def _remember_resolved_texture_maps(
        entry: dict[str, Any] | None,
        effective_props: dict[str, Any],
    ) -> None:
        """Remember runtime texture fallbacks for later material-panel rewrites.

        Some scenes bind visual textures through material IDs instead of the
        ITU material type itself. Material-panel updates start from the entry's
        base PBR properties, so cache those resolved paths on the entry to avoid
        clearing an explicitly resolved texture when a scalar like clearcoat
        changes.
        """
        if not isinstance(entry, dict):
            return
        resolved = {
            key: effective_props.get(key)
            for key in TEXTURE_MAP_PROP_KEYS
            if effective_props.get(key) is not None
        }
        if resolved:
            entry["_resolved_texture_maps"] = resolved

    def remove_empty_scene_anchor(self) -> None:
        """Remove the empty-scene camera anchor once real MPC content is visible."""
        viz = self.visualizer
        anchor = getattr(viz, "_empty_scene_anchor", None)
        if anchor is None:
            return
        try:
            if not isinstance(anchor, RenderObjectState):
                raise TypeError("Empty-scene anchor requires RenderObjectState")
            if self._render_sync.remove_object(anchor):
                viz._empty_scene_anchor = None
                logger.debug("Removed empty-scene camera anchor; real content is available")
        except (RuntimeError, ValueError):
            pass

    def load_scene(
        self,
        xml_path: str,
        *,
        render_immediately: bool = True,
        cleanup_first: bool = True,
    ) -> None:
        """Load a complete XML scene or propagate the load failure."""
        viz = self.visualizer
        if cleanup_first:
            self.cleanup_previous_scene()
        viz.xml_path = xml_path
        try:
            with viz.progress.task(f"Parsing XML scene {xml_path}"):
                viz.xml_root = XMLSceneHandler.load_xml_scene(xml_path)
            with viz.progress.task("Loading environment meshes"):
                self.load_from_xml()
        except (OSError, ValueError, RuntimeError):
            logger.exception("Failed to load XML scene %s", xml_path)
            viz.xml_root = None
            viz.xml_path = None
            viz.mesh_entries = []
            self.scene_path = None
            raise

        self._finalize_loaded_scene(xml_path, render_immediately=render_immediately)

    def load_prepared_scene(
        self,
        xml_path: str,
        xml_root: Any,
        mesh_entries: list[dict[str, Any]],
        *,
        render_immediately: bool = True,
        cleanup_first: bool = True,
    ) -> None:
        """Install a renderer-neutral scene payload staged during preflight."""

        viz = self.visualizer
        if cleanup_first:
            self.cleanup_previous_scene()
        viz.xml_path = xml_path
        viz.xml_root = xml_root
        try:
            with viz.progress.task("Installing preflighted environment meshes"):
                self.load_from_xml(prepared_entries=mesh_entries)
        except (OSError, ValueError, RuntimeError):
            logger.exception("Failed to install preflighted XML scene %s", xml_path)
            viz.xml_root = None
            viz.xml_path = None
            viz.mesh_entries = []
            self.scene_path = None
            raise

        self._finalize_loaded_scene(xml_path, render_immediately=render_immediately)

    def _finalize_loaded_scene(self, xml_path: str, *, render_immediately: bool) -> None:
        """Publish one parsed scene to UI and optional renderer consumers."""

        viz = self.visualizer

        ui_controller = getattr(viz, "ui_controller", None)
        if ui_controller is not None:
            ui_controller.populate_controls()

        self.scene_path = xml_path

        if render_immediately:
            if not getattr(viz, "vis_initialized", False):
                boot_visualizer_empty(viz)
            self.render_scene()
            ui_controller = getattr(viz, "ui_controller", None)
            if ui_controller is not None:
                ui_controller.populate_material_filters()

    def load_from_xml(self, *, prepared_entries: list[dict[str, Any]] | None = None) -> None:
        """Load scene from XML using the centralized I/O module."""
        viz = self.visualizer
        # Build mesh entries from the same XML tree used for serialization.
        viz.mesh_entries = (
            build_scene_from_root(viz.xml_root, viz.xml_path)
            if prepared_entries is None
            else list(prepared_entries)
        )
        for i, entry in enumerate(viz.mesh_entries):
            if isinstance(entry, dict):
                ensure_scene_mesh_render_state(entry, i)
        logger.debug(
            "load_from_xml: mesh_entries assigned (%d entries)",
            len(viz.mesh_entries),
        )

        # Refresh Materials panel now that mesh_entries is populated
        # This also sets renderer-appropriate visibility for PBR controls
        if hasattr(viz, "ui_manager") and viz.ui_manager:
            materials_panel = viz.ui_manager.panels.get("materials")
            if materials_panel is not None and hasattr(materials_panel, "update_ui_from_viz"):
                materials_panel.update_ui_from_viz()
                logger.debug("Updated Materials panel UI")

        if hasattr(viz, "outlines_enabled") and viz.outlines_enabled:
            scene_appearance = getattr(viz, "scene_appearance_service", None)
            if scene_appearance is not None:
                scene_appearance.set_edge_visibility(True)

        # Mesh entries retain references into the editable XML tree. Verify the
        # first material points into the tree that will later be saved.
        if viz.mesh_entries and viz.xml_root is not None:
            first_entry = viz.mesh_entries[0]
            if "xml_bsdf" in first_entry and first_entry["xml_bsdf"] is not None:
                bsdf_id = first_entry["material_id"]
                root_bsdf = viz.xml_root.find(f".//bsdf[@id='{bsdf_id}']")
                if root_bsdf is first_entry["xml_bsdf"]:
                    logger.debug("Scene entries reference the active XML tree")
                elif root_bsdf is not None:
                    logger.warning("Scene entry material references a different XML tree")
                else:
                    logger.warning("Could not find BSDF %s in the active XML tree", bsdf_id)

    def cleanup_previous_scene(self) -> None:
        """Clean up previous scene data and UI elements."""
        viz = self.visualizer
        logger.info("Starting scene cleanup")

        viz.xml_root = None
        viz.xml_path = None
        self.scene_path = None

        # The anchor is a persistent scene owner too; retire it before the
        # hard renderer reset so empty-to-nonempty transitions cannot retain
        # stale application ownership.
        self.remove_empty_scene_anchor()

        # SceneService owns aggregate and individual scene objects; retire
        # them through the same failure-aware path used by a repeated render.
        if getattr(viz, "vis_initialized", False) and viz.vis is not None:
            try:
                if not self._retire_current_scene_render_owners():
                    logger.warning(
                        "Scene cleanup will rely on the renderer reset for remaining owners"
                    )
            except (RuntimeError, ValueError) as exc:
                logger.warning("Scene-owner retirement failed during cleanup: %s", exc)
        self._scene_batches.clear()
        self._merge_enabled = False

        # TargetService owns target meshes separately from scene aggregation.
        # Remove those persistent owners before clearing target entries.
        if getattr(viz, "vis_initialized", False) and viz.vis is not None:
            try:
                removed_render_ids: set[str] = set()

                def _remove_entry_geometry(
                    entry: dict[str, Any],
                ) -> None:
                    """Remove one persistent entry through the common object API."""
                    mesh = entry.get("mesh")
                    if not isinstance(mesh, RenderObjectState):
                        raise TypeError("Persistent cleanup requires RenderObjectState")
                    if mesh.id in removed_render_ids:
                        return
                    self._render_sync.remove_object(mesh)
                    removed_render_ids.add(mesh.id)

                for index, entry in enumerate(viz.target_entries):
                    mesh = entry.get("mesh")
                    if mesh is not None:
                        ensure_target_entry_identity(entry, index)
                        _remove_entry_geometry(entry)

                # Remove any cached target handle not already owned by an entry.
                target_meshes = getattr(viz, "target_meshes", {})
                target_mesh_values = (
                    target_meshes.values() if isinstance(target_meshes, dict) else target_meshes
                )
                for mesh in target_mesh_values:
                    if not isinstance(mesh, RenderObjectState):
                        raise TypeError("Persistent target cache requires RenderObjectState")
                    if mesh.id not in removed_render_ids:
                        self._render_sync.remove_object(mesh)
                        removed_render_ids.add(mesh.id)

                logger.debug("Removed old target meshes from visualizer")
            except (RuntimeError, ValueError) as e:
                logger.error(f"Error removing old target meshes: {e}")

        viz.mesh_entries.clear()

        viz.target_entries.clear()
        if hasattr(viz, "target_meshes"):
            viz.target_meshes.clear()

        viz.selected_objects.clear()
        material_modes = getattr(viz, "material_mode_service", None)
        clear_modes = getattr(material_modes, "clear", None)
        if callable(clear_modes):
            clear_modes()
        pbr_service = getattr(viz, "material_pbr_service", None)
        overrides = getattr(pbr_service, "overrides", None)
        if hasattr(overrides, "clear"):
            overrides.clear()

        # Clear derived and raw frame caches through the cache owner.
        viz.cache_service.invalidate(CacheInvalidationScope.ALL, reason="scene_cleanup")

        viz.available_tx = []
        viz.available_rx = []
        viz.tx_rx_data_loaded = False

        node_service = getattr(viz, "node_service", None)
        remove_node_marker = getattr(node_service, "_remove_node_marker_entity", None)
        if callable(remove_node_marker):
            for i in range(max(len(viz.tx_markers), len(viz.tx_labels))):
                remove_node_marker("tx", i)
            for i in range(max(len(viz.rx_markers), len(viz.rx_labels))):
                remove_node_marker("rx", i)

        viz.tx_markers.clear()
        viz.rx_markers.clear()
        viz.tx_labels.clear()
        viz.rx_labels.clear()

        viz.num_tx = None
        viz.num_rx = None

        viz.current_tx_positions = []
        viz.current_rx_positions = []
        viz.positions_loaded_step = None

        viz.building_labels.clear()
        viz.target_labels.clear()

        # Detach from current frame source/provider if present
        if hasattr(viz, "frame_source") and viz.frame_source is not None:
            if hasattr(viz.frame_source, "close"):
                try:
                    viz.frame_source.close()
                    logger.info("Frame source closed during cleanup")
                except (OSError, RuntimeError) as exc:
                    logger.warning("Failed to close frame source during cleanup: %s", exc)
            viz.frame_source = None

        viz.animation_step = 0

        # Sync AppState step with animation step and clear runtime-only beamforming state
        if hasattr(viz, "app_state"):
            viz.set_state(step=0, **get_beamforming_state_defaults())

        if hasattr(viz, "renderer") and viz.renderer is not None:
            viz.renderer.reset_state()
        aperture_service = getattr(viz, "aperture_service", None)
        clear_apertures = getattr(aperture_service, "clear_all", None)
        if callable(clear_apertures) and not clear_apertures():
            logger.warning("Failed to clear angular preview geometry after renderer reset")
        self._scene_batches.individual_owner_ids.clear()

        if hasattr(viz, "pipeline") and viz.pipeline is not None:
            viz.pipeline.clear_coverage_cache()

        viz.frame_source = None
        viz.ready = False
        viz.force_update_next_frame = False
        viz.last_app_state = None

        if hasattr(viz, "_cached_bounce_points"):
            viz._cached_bounce_points = np.empty((0, 3), dtype=np.float64)
        if hasattr(viz, "_cached_bounce_colors"):
            viz._cached_bounce_colors = np.empty((0, 3), dtype=np.float64)

        if hasattr(viz, "coverage_mesh"):
            viz.coverage_mesh = None

        # Clear beamforming state, including any transient computation/error status.
        beamforming_ui = getattr(viz, "beamforming_ui_controller", None)
        clear_beamforming = getattr(beamforming_ui, "clear_result_metadata", None)
        if callable(clear_beamforming):
            clear_beamforming()
        else:
            if hasattr(viz, "_beamforming_tx_nodes"):
                viz._beamforming_tx_nodes.clear()
            if hasattr(viz, "_beamforming_rx_nodes"):
                viz._beamforming_rx_nodes.clear()
            if hasattr(viz, "_latest_beamforming_info"):
                viz._latest_beamforming_info = None
            if hasattr(viz, "_latest_beamforming_pairs"):
                viz._latest_beamforming_pairs.clear()
            if hasattr(viz, "_beamforming_computing"):
                viz._beamforming_computing = False
            if hasattr(viz, "_beamforming_completed_without_result"):
                viz._beamforming_completed_without_result = False
            if hasattr(viz, "_beamforming_error_message"):
                viz._beamforming_error_message = None

        if hasattr(viz, "_set_status_message"):
            viz._set_status_message("Loading new scene...")

    def _building_labels_requested(self) -> bool:
        """Return True when startup should build building labels."""
        viz = self.visualizer
        if bool(getattr(viz, "show_building_labels", False)):
            return True
        return any(
            bool(entry.get("show_label", False)) and entry.get("mesh") is not None
            for entry in getattr(viz, "mesh_entries", [])
            if isinstance(entry, dict)
        )

    def ensure_building_labels_created(self, *, force_rebuild: bool = False) -> None:
        """Build building labels on demand for the current scene."""
        viz = self.visualizer
        expected_count = sum(
            1
            for entry in getattr(viz, "mesh_entries", [])
            if isinstance(entry, dict) and entry.get("mesh") is not None and "name" in entry
        )
        if expected_count == 0:
            viz.building_labels.clear()
            return
        if not force_rebuild and len(getattr(viz, "building_labels", [])) == expected_count:
            return
        create_building_labels(viz)

    def _retire_current_scene_render_owners(self) -> bool:
        """Remove current scene owners before rebuilding ownership maps.

        ``render_scene()`` can be invoked again without a scenario cleanup.
        Retiring the old aggregate and individual owners first prevents a
        changed material grouping from orphaning native geometry under IDs no
        longer present in the rebuilt maps. Tracking is retained on failure so
        the same idempotent cleanup can be retried.
        """
        if not self._merged_meshes and not self._individual_scene_owner_ids:
            return True

        appearance = getattr(self.visualizer, "scene_appearance_service", None)
        remove_merged_outline = getattr(appearance, "remove_merged_outline_geometry", None)
        all_removed = True
        with self.visualizer.renderer.batch_updates():
            for group_info in self._merged_meshes.values():
                state = group_info.get("render_state")
                if isinstance(state, RenderObjectState) and not self._render_sync.remove_object(
                    state
                ):
                    all_removed = False
                if callable(remove_merged_outline):
                    if not remove_merged_outline(group_info):
                        all_removed = False
                elif isinstance(group_info.get("_merged_outline"), RenderObjectState):
                    all_removed = False

            entries_by_id: dict[str, dict[str, Any]] = {}
            for entry in getattr(self.visualizer, "mesh_entries", []):
                mesh = entry.get("mesh") if isinstance(entry, dict) else None
                if isinstance(mesh, RenderObjectState):
                    entries_by_id[mesh.id] = entry
            for object_id in tuple(self._individual_scene_owner_ids):
                entry = entries_by_id.get(object_id)
                if entry is None:
                    logger.error(
                        "Cannot retire scene owner '%s': semantic entry is missing",
                        object_id,
                    )
                    all_removed = False
                    continue
                if not self._retire_individual_scene_owner(entry):
                    all_removed = False
        return all_removed

    def render_scene(self) -> None:
        """OPTIMIZED: Initial render with reduced operations."""
        viz = self.visualizer
        capture_startup_breakdown = bool(
            getattr(viz, "_scene_boot_start", None) is not None
            and not getattr(viz, "_scene_boot_logged", False)
            and hasattr(viz, "set_startup_detail_timing")
        )
        breakdown: OrderedDict[str, float] = OrderedDict()

        def _record_breakdown(name: str, start: float) -> float:
            """Record a scene-render startup stage and return the current timestamp."""
            end = time.perf_counter()
            if capture_startup_breakdown:
                breakdown[str(name)] = (end - start) * 1000.0
            return end

        # Preserve renderer-neutral camera state across scene reconstruction.
        t_stage = time.perf_counter()
        if viz.vis is not None and hasattr(viz.renderer, "get_camera_state"):
            camera_state = viz.renderer.get_camera_state()
            viz.last_cam = camera_state if isinstance(camera_state, CameraState) else None

        scene_member_sources: list[tuple[RenderObjectState, dict[str, Any] | None]] = []
        target_entries_to_sync: list[dict[str, Any]] = []
        has_visible_geometry = False

        # Scene I/O caches raw neutral payloads.  Establish persistent object
        # ownership once per entry before any render lookup or merge tracking.
        for index, entry in enumerate(viz.mesh_entries):
            if entry.get("mesh") is not None:
                mesh = ensure_scene_mesh_render_state(entry, index)
                scene_member_sources.append((mesh, entry))
                has_visible_geometry = bool(has_visible_geometry or entry.get("visible", True))

        # TargetService already owns target states; validate their persistent
        # identity before adding them to the common scene handoff.
        for index, target_entry in enumerate(viz.target_entries):
            mesh = target_entry.get("mesh")
            if mesh is not None:
                mesh = require_target_mesh_render_state(target_entry, index)
            if mesh is not None:
                target_entries_to_sync.append(target_entry)
                has_visible_geometry = bool(
                    has_visible_geometry or target_entry.get("visible", True)
                )

        if not self._retire_current_scene_render_owners():
            raise RuntimeError(
                "Scene synchronization failed because previous renderer owners remain"
            )

        # Empty-scene camera anchor: renderers need initial geometry for camera bounds.
        existing_anchor = getattr(viz, "_empty_scene_anchor", None)
        if has_visible_geometry:
            if isinstance(existing_anchor, RenderObjectState):
                if not self._render_sync.remove_object(existing_anchor):
                    raise RuntimeError(
                        "Scene synchronization failed because the empty anchor remains"
                    )
                viz._empty_scene_anchor = None
        else:
            # Create a small sphere at the origin so the window has geometry for
            # camera initialization.  It stays visible until the first frame
            # update adds real MPC content, then is removed automatically.
            anchor = existing_anchor
            if not isinstance(anchor, RenderObjectState):
                anchor = RenderObjectState(
                    id="scene:empty_anchor::sphere",
                    payload=make_sphere_payload(radius=0.5, color=[0.7, 0.7, 0.7]),
                    metadata={"type": "empty_scene_anchor"},
                )
            scene_member_sources.append((anchor, None))
            viz._empty_scene_anchor = anchor
            logger.debug("Created empty-scene camera anchor")
        t_stage = _record_breakdown("collect_visible_geometry_ms", t_stage)

        # Reuse existing window if available, otherwise create one
        if viz.vis is None:
            boot_visualizer_empty(viz)
        t_stage = _record_breakdown("prepare_renderer_window_ms", t_stage)

        viz.selected_objects = set()

        # The typed registry is the sole scene reverse index. TargetService
        # owns target identity and presentation independently.
        self._scene_batches.entries_by_mesh_id.clear()
        for index, entry in enumerate(viz.target_entries):
            ensure_target_entry_identity(entry, index)
        t_stage = _record_breakdown("prepare_entry_identity_ms", t_stage)

        # Cache texture library contents once (avoids per-mesh filesystem stat())
        has_pbr = renderer_capabilities(viz.renderer).pbr
        auto_generate_uvs = True
        if hasattr(viz, "scenario") and viz.scenario is not None:
            auto_generate_uvs = viz.scenario.view_defaults.get("auto_generate_uvs", True)

        if self._texture_cache is None:
            self._texture_cache = {}
            if TEXTURE_LIBRARY_PATH.is_dir():
                for tex_file in TEXTURE_LIBRARY_PATH.iterdir():
                    if tex_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        self._texture_cache[tex_file.stem.lower()] = str(tex_file)
        textures_enabled = not self._textures_disabled()
        texture_cache = (
            self._texture_cache if (has_pbr and auto_generate_uvs and textures_enabled) else {}
        )

        # Decide whether to merge scene meshes by material
        merge_scene_meshes = self._should_merge_meshes(viz.mesh_entries)

        self._scene_batches.clear_presentations()
        self._merge_enabled = merge_scene_meshes
        prepared_members = self._prepare_scene_members(
            scene_member_sources,
            has_pbr=has_pbr,
            auto_generate_uvs=auto_generate_uvs,
            texture_cache=texture_cache,
        )
        finalize_uv_cache_stores()
        t_stage = _record_breakdown("prepare_render_state_ms", t_stage)

        batch_start = time.perf_counter()
        with viz.renderer.batch_updates():
            if merge_scene_meshes and has_pbr:
                self._render_merged(prepared_members)
            else:
                self._render_individual(prepared_members)
            # TargetService is the only target presentation owner.  Initial
            # publication and frame updates now follow the same complete
            # mesh/label/outline snapshot path.
            target_service = getattr(viz, "target_service", None)
            sync_target = getattr(target_service, "sync_target_entry_snapshot", None)
            if target_entries_to_sync and not callable(sync_target):
                raise RuntimeError("TargetService is unavailable for initial target rendering")
            for target_entry in target_entries_to_sync:
                if not sync_target(target_entry):
                    target_name = target_entry.get("name") or target_entry.get("target_name")
                    raise RuntimeError(
                        f"Initial target renderer synchronization failed for {target_name}"
                    )
        t_stage = _record_breakdown("upload_scene_geometry_ms", batch_start)

        # Don't add empty MPC geometries to visualizer - they will be added when they contain data
        # This prevents Open3D warnings about 0-point geometries and improves performance

        # Ensure background color is set correctly
        viz.scene_appearance_service.ensure_light_gray_background()
        t_stage = _record_breakdown("apply_background_ms", t_stage)

        overlay_start = time.perf_counter()
        with viz.renderer.batch_updates():
            # TX/RX markers and labels are synced by NodeService/EntityRenderService.
            tx_rx_start = time.perf_counter()
            tx_rx_end = _record_breakdown("upload_tx_rx_geometry_ms", tx_rx_start)

            label_build_start = tx_rx_end
            if self._building_labels_requested():
                self.ensure_building_labels_created()

            create_target_labels(viz)
            label_visibility_start = _record_breakdown("build_label_geometry_ms", label_build_start)

            # Resolve labels by stable ID because entries without geometry do
            # not allocate a label-list slot.
            building_labels = {
                label.id: label
                for label in viz.building_labels
                if isinstance(label, RenderObjectState)
            }
            for i, entry in enumerate(viz.mesh_entries):
                label_name = f"bldg_label_{i}"
                label = building_labels.get(label_name)
                if label is None:
                    continue
                if not self._render_sync.sync_label_geometry(
                    name=label_name,
                    geometry=label,
                    visible=effective_entry_label_visibility(
                        entry,
                        labels_enabled=True,
                        entry_type="mesh",
                    ),
                ):
                    raise RuntimeError(
                        f"Initial building-label synchronization failed for {label_name}"
                    )

            # Manage target labels through the same global/local/parent policy.
            show_target_labels = getattr(
                getattr(viz, "app_state", None), "show_target_labels", True
            )
            target_labels = {
                label.id: label
                for label in viz.target_labels
                if isinstance(label, RenderObjectState)
            }

            def _target_hidden_for_pov(node_type: str, index: int) -> bool:
                return is_hidden_for_pov(
                    getattr(viz, "app_state", None),
                    node_type,
                    index,
                )

            for i, target_entry in enumerate(viz.target_entries):
                ensure_target_entry_identity(target_entry, i)
                label_name = make_target_entry_geometry_name(target_entry, "label")
                label = target_labels.get(label_name)
                if label is None:
                    continue
                if not self._render_sync.sync_label_geometry(
                    name=label_name,
                    geometry=label,
                    visible=effective_entry_label_visibility(
                        target_entry,
                        labels_enabled=show_target_labels,
                        entry_type="target",
                        target_index=i,
                        is_hidden_for_pov=_target_hidden_for_pov,
                    ),
                ):
                    raise RuntimeError(
                        f"Initial target-label synchronization failed for {label_name}"
                    )
            _record_breakdown("apply_label_visibility_ms", label_visibility_start)
        t_stage = _record_breakdown("upload_overlay_geometry_ms", overlay_start)

        # Populate target dropdown in the nodes panel
        target_dropdown_start = time.perf_counter()
        nodes_panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("nodes")
        if nodes_panel and hasattr(nodes_panel, "populate_target_dropdown"):
            nodes_panel.populate_target_dropdown()
        t_stage = _record_breakdown("populate_target_dropdown_ms", target_dropdown_start)

        # NOTE: the empty-scene anchor stays visible until the first frame
        # update adds real MPC geometry. See remove_empty_scene_anchor().

        camera_restore_start = time.perf_counter()
        if viz.last_cam is not None and hasattr(viz.renderer, "set_camera_state"):
            if isinstance(viz.last_cam, CameraState):
                viz.renderer.set_camera_state(viz.last_cam)
        t_stage = _record_breakdown("restore_initial_camera_ms", camera_restore_start)

        # Reset camera bounds to fit all loaded geometry (critical for Open3D renderer)
        fit_start = time.perf_counter()
        try:
            viz.renderer.reset_camera_bounds()
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.debug(f"Could not reset camera bounds: {e}")
        t_stage = _record_breakdown("fit_camera_bounds_ms", fit_start)

        viz.vis_initialized = True

        if not (viz.frame_source and viz.frame_source.has_frame(0)):
            logger.debug("No initial frame data found, skipping TX/RX position initialization")

        # Detect MPC frames immediately if they exist
        frame_detect_start = time.perf_counter()
        viz.detect_mpc_frames()
        marker_visibility_start = _record_breakdown(
            "detect_available_frames_ms", frame_detect_start
        )

        viz.node_service.update_tx_rx_visibility()
        t_stage = _record_breakdown("update_node_visibility_ms", marker_visibility_start)

        # Populate object selection dropdown
        schedule_dropdown_start = time.perf_counter()
        QTimer.singleShot(500, viz.selection_manager.populate_dropdown)
        t_stage = _record_breakdown("schedule_object_dropdown_ms", schedule_dropdown_start)

        # Both supported backends preserve declarative camera state during
        # scene synchronization. Session/startup policy owns any later camera
        # override; rendering the scene must not issue a second overview reset.

        if capture_startup_breakdown and breakdown:
            viz.set_startup_detail_timing("render_initial_scene_breakdown_ms", dict(breakdown))

        # Don't call run() - let the timer handle updates

    # Mesh-merging helpers

    def _should_merge_meshes(self, entries: list[dict[str, Any]]) -> bool:
        """Decide whether static batching provides a meaningful owner reduction.

        Merging is enabled when:
        - The scenario explicitly sets ``merge_scene_meshes: true``, OR
        - The scenario doesn't set it and ``mesh_count`` exceeds the threshold.

        Setting ``merge_scene_meshes: false`` in the scenario YAML disables
        merging unconditionally.
        """
        viz = self.visualizer
        explicit = None
        if hasattr(viz, "scenario") and viz.scenario is not None:
            explicit = viz.scenario.view_defaults.get("merge_scene_meshes")

        if explicit is not None:
            return bool(explicit)

        capabilities = renderer_capabilities(getattr(viz, "renderer", None))
        if not capabilities.static_mesh_batching:
            return False

        mesh_entries = [
            entry for entry in entries if isinstance(entry, dict) and entry.get("mesh") is not None
        ]
        mesh_count = len(mesh_entries)
        if mesh_count <= capabilities.static_mesh_batch_object_threshold:
            return False

        # Estimate the lower bound on native owners without resolving PBR or
        # touching texture paths.  Auto-batching is worthwhile only when the
        # authored material distribution predicts a substantial reduction.
        material_groups = {
            str(entry.get("material_type") or entry.get("material_id") or "default")
            for entry in mesh_entries
        }
        predicted_owners = max(1, len(material_groups))
        owner_reduction = 1.0 - (predicted_owners / max(1, mesh_count))
        return owner_reduction >= 0.25

    @staticmethod
    def _discover_scene_texture_defaults(
        entry: dict[str, Any],
        *,
        auto_generate_uvs: bool,
        texture_cache: dict[str, str | None],
    ) -> dict[str, str]:
        """Return the material-ID-derived albedo as a low-priority default.

        This lookup performs no material resolution and never overrides an
        authored, profile, or manual map. ``MaterialPBRService`` applies it
        only to FOLLOW_EM bindings before its single policy/payload resolution.
        """
        if not auto_generate_uvs or not texture_cache:
            return {}
        material_id = str(entry.get("material_id") or "").strip().lower()
        material_type = str(entry.get("material_type") or "").strip().lower()
        for prefix in ("mat-", "mat_"):
            if material_id.startswith(prefix):
                material_id = material_id[len(prefix) :]
                break
        if not material_id:
            return {}
        generic_keys = {material_type, f"itu_{material_type}", f"itu-{material_type}"}
        if material_id in generic_keys:
            return {}
        texture_path = texture_cache.get(material_id)
        return {"texture_path": texture_path} if texture_path else {}

    def _resolve_scene_member_material(
        self,
        entry: dict[str, Any],
        *,
        discovered_texture_defaults: dict[str, str],
    ) -> ResolvedMaterial:
        """Resolve one scene material through the application material owner."""
        service = getattr(self.visualizer, "material_pbr_service", None)
        resolve_entry = getattr(service, "resolve_entry_material", None)
        if callable(resolve_entry):
            return resolve_entry(
                entry,
                discovered_texture_defaults=discovered_texture_defaults,
            )

        # Lightweight visualizer stubs used by embedders may omit the service.
        # Keep compatibility at this boundary without duplicating profile or
        # override semantics in either renderer publication path.
        props = dict(entry.get("pbr_properties", {}) or {})
        props.update({key: value for key, value in discovered_texture_defaults.items() if value})
        props.setdefault("color", entry.get("color", [0.7, 0.7, 0.7]))
        props.setdefault("roughness", 0.5)
        props.setdefault("metallic", 0.0)
        props.setdefault("reflectance", 0.5)
        props.setdefault(
            "alpha",
            getattr(self.visualizer, "current_building_alpha", 1.0),
        )
        return resolve_pbr_material(
            props["color"],
            props,
            context=str(entry.get("material_type") or "default"),
        )

    def _ensure_scene_member_uvs(
        self,
        state: RenderObjectState,
        entry: dict[str, Any],
        material: ResolvedMaterial | MaterialPayload,
    ) -> bool:
        """Attach cached or generated neutral UVs when the material needs them."""
        if isinstance(material, ResolvedMaterial):
            has_active_maps = material.texture_policy.has_active_maps
            uv_scale_meters = float(material.properties.get("uv_scale_meters", 2.0))
        else:
            has_active_maps = any(
                getattr(material, key, None) is not None for key in TEXTURE_MAP_PROP_KEYS
            )
            uv_scale_meters = float(material.uv_scale_meters)
        if not has_active_maps or self._mesh_has_triangle_uvs(state):
            return False
        from ..scene.io import load_or_generate_box_projection_uvs

        uv_scale = 1.0 / max(0.1, uv_scale_meters)
        source_path = self._uv_cache_source_path(entry)
        source_signature = entry.get("_source_signature")
        transform_state = entry.get("transform_state")
        auto_uvs = load_or_generate_box_projection_uvs(
            state.payload,
            scale=uv_scale,
            cache_source_path=source_path,
            transform_state=transform_state,
            source_signature=(source_signature if isinstance(source_signature, dict) else None),
            cache_miss_callback=(
                lambda: (
                    self._maybe_announce_texture_cache_build(reason="UV") if source_path else None
                )
            ),
        )
        if auto_uvs is not None and isinstance(state.payload, MeshPayload):
            state.replace_payload(
                replace(
                    state.payload,
                    # The UV store returns renderer-ready float32 snapshots.
                    # Preserve their immutable backing instead of widening and
                    # copying every cached array during scene preparation.
                    triangle_uvs=np.asarray(auto_uvs),
                )
            )
            return True
        return False

    def _prepare_scene_members(
        self,
        members: list[tuple[RenderObjectState, dict[str, Any] | None]],
        *,
        has_pbr: bool,
        auto_generate_uvs: bool,
        texture_cache: dict[str, str | None],
    ) -> list[PreparedSceneMember]:
        """Resolve every member once for either publication strategy."""
        prepared: list[PreparedSceneMember] = []
        for state, entry in members:
            visible = self._effective_initial_scene_visibility(state, entry)
            if entry is None or not has_pbr:
                prepared.append(
                    PreparedSceneMember(
                        state=state,
                        entry=entry,
                        visible=visible,
                        material_type="default",
                        resolved_material=None,
                    )
                )
                continue

            discovered = self._discover_scene_texture_defaults(
                entry,
                auto_generate_uvs=auto_generate_uvs,
                texture_cache=texture_cache,
            )
            resolved = self._resolve_scene_member_material(
                entry,
                discovered_texture_defaults=discovered,
            )
            self._ensure_scene_member_uvs(state, entry, resolved)
            props = resolved.properties_copy(mark_texture_policy=True)
            payload = self._mesh_payload(state)
            color_source = (
                payload.color_source
                if isinstance(payload, MeshPayload)
                else SurfaceColorSource.MATERIAL
            )
            state = self._wrap_pygfx_scene_mesh_payload(
                state,
                scene_entry=entry,
                effective_props=props,
                color_source=color_source,
            )
            if not isinstance(state, RenderObjectState):
                raise TypeError("Prepared scene member lost persistent render ownership")
            state.material = resolved.payload
            self._remember_resolved_texture_maps(entry, props)
            self._scene_batches.register_entry(id(state), entry)
            prepared.append(
                PreparedSceneMember(
                    state=state,
                    entry=entry,
                    visible=visible,
                    material_type=str(entry.get("material_type") or "default"),
                    resolved_material=resolved,
                )
            )
        return prepared

    def _render_individual(self, members: list[PreparedSceneMember]) -> None:
        """Publish prepared members without changing appearance semantics."""
        for member in members:
            if member.entry is not None:
                self._individual_scene_owner_ids.add(member.state.id)
            if not self._ensure_persistent_material_geometry(
                geometry=member.state,
                material=member.material,
                visible=member.visible,
            ):
                raise RuntimeError(
                    f"Initial scene geometry synchronization failed for {member.state.id}"
                )

    def _batch_member_chunks(
        self,
        members: list[PreparedSceneMember],
    ) -> list[list[PreparedSceneMember]]:
        """Bound aggregate rebuild and upload cost using renderer limits."""
        member_limit, triangle_limit = self._scene_batch_limits()
        chunks: list[list[PreparedSceneMember]] = []
        current: list[PreparedSceneMember] = []
        current_triangles = 0
        for member in members:
            triangle_count = self._scene_mesh_triangle_count(member.state)
            exceeds_limit = bool(
                current
                and (
                    len(current) >= member_limit
                    or current_triangles + triangle_count > triangle_limit
                )
            )
            if exceeds_limit:
                chunks.append(current)
                current = []
                current_triangles = 0
            current.append(member)
            current_triangles += triangle_count
        if current:
            chunks.append(current)
        return chunks

    def _scene_batch_limits(self) -> tuple[int, int]:
        """Return the renderer's hard member and triangle limits for one batch."""
        capabilities = renderer_capabilities(self.visualizer.renderer)
        return (
            max(1, int(capabilities.static_mesh_batch_member_limit)),
            max(1, int(capabilities.static_mesh_batch_triangle_limit)),
        )

    def _scene_mesh_triangle_count(self, mesh: Any) -> int:
        """Return the neutral triangle count used by static-batch limits."""
        payload = self._mesh_payload(mesh)
        return len(payload.triangles) if payload is not None else 0

    @staticmethod
    def _scene_batch_name(
        signature: tuple[str, MaterialPayload],
        members: list[PreparedSceneMember],
    ) -> str:
        """Return a deterministic ID unique across bounded material chunks."""
        return SceneService._scene_batch_name_for_owner_ids(
            signature,
            [member.state.id for member in members],
        )

    @staticmethod
    def _scene_batch_name_for_owner_ids(
        signature: tuple[str, MaterialPayload],
        owner_ids: list[str],
    ) -> str:
        """Return a deterministic batch ID from material and exact membership."""
        identity = (signature, tuple(owner_ids))
        token = hashlib.sha1(repr(identity).encode("utf-8")).hexdigest()[:12]
        return make_component_geometry_name(f"scene:merged_{token}", "mesh")

    def _render_merged(self, members: list[PreparedSceneMember]) -> None:
        """Publish compatible prepared members through bounded static batches."""
        groups: dict[tuple[str, MaterialPayload], list[PreparedSceneMember]] = {}
        individual: list[PreparedSceneMember] = []
        for member in members:
            if member.entry is None or member.resolved_material is None:
                individual.append(member)
                continue
            groups.setdefault(member.material_signature, []).append(member)

        self._render_individual(individual)
        aggregate_count = 0
        aggregated_member_count = 0
        for signature, compatible_members in groups.items():
            for chunk in self._batch_member_chunks(compatible_members):
                if len(chunk) == 1:
                    self._render_individual(chunk)
                    continue
                merged = self._merge_scene_mesh_payloads(
                    [member.state for member in chunk],
                    cache_baseline=True,
                )
                if merged is None:
                    self._render_individual(chunk)
                    continue

                merged_name = self._scene_batch_name(signature, chunk)
                mesh_ids = [id(member.state) for member in chunk]
                sources = tuple(
                    (member.state.id, member.state.payload)
                    for member in chunk
                    if isinstance(member.state.payload, MeshPayload)
                )
                batch = SceneBatch(
                    name=merged_name,
                    material_signature=signature,
                    member_mesh_ids=mesh_ids,
                    baseline_geometry=merged,
                    baseline_sources=sources,
                    presentation={
                        "geometry": merged,
                        "geometry_sources": sources,
                    },
                )
                self._scene_batches.add_batch(batch)
                if self._sync_merged_group_snapshot(merged_name):
                    self._pending_merged_group_syncs.discard(merged_name)
                else:
                    self._pending_merged_group_syncs.add(merged_name)
                    raise RuntimeError(
                        f"Initial merged-scene synchronization failed for {merged_name}"
                    )
                aggregate_count += 1
                aggregated_member_count += len(chunk)

        logger.info(
            "Static scene batching: %d members -> %d bounded aggregate owners",
            aggregated_member_count,
            aggregate_count,
        )

    # Merged-group visibility / transparency helpers

    def _scene_entry_base_material(
        self,
        entry: dict[str, Any],
    ) -> ResolvedMaterial:
        """Resolve the unhighlighted semantic material for one scene entry."""
        return self._resolve_scene_member_material(
            entry,
            discovered_texture_defaults={},
        )

    def refresh_scene_entry_material(
        self,
        entry: dict[str, Any],
        *,
        effective_visible: bool | None = None,
    ) -> bool:
        """Refresh one scene entry from its unhighlighted semantic material."""
        material = self._scene_entry_base_material(entry).payload
        return self.sync_scene_entry_snapshot(
            entry,
            material=material,
            effective_visible=effective_visible,
        )

    def refresh_all_scene_materials(self) -> bool:
        """Refresh every distinct scene owner once after a global material change."""
        entries = [
            entry
            for entry in getattr(self.visualizer, "mesh_entries", [])
            if isinstance(entry, dict) and entry.get("mesh") is not None
        ]
        if not entries:
            return False
        return self._refresh_scene_material_entries(entries)

    def _refresh_scene_material_entries(self, entries: list[dict[str, Any]]) -> bool:
        """Refresh a scene-entry batch without skipping material regrouping.

        A material override can temporarily merge entries whose authored
        materials differ.  Once that override is removed, refreshing the
        first member may detach it from the old aggregate; the remaining
        members must still be visited until the aggregate has converged on one
        semantic material.  Only then is it safe to coalesce later updates for
        that group.
        """
        all_synced = True
        synchronized_groups: set[str] = set()
        for entry in entries:
            mesh = ensure_scene_mesh_render_state(entry)
            group_name = self._mesh_id_to_group.get(id(mesh))
            if group_name is not None and group_name in synchronized_groups:
                continue
            if not self.refresh_scene_entry_material(entry):
                all_synced = False
                continue

            # The refresh may have detached this entry, joined it to another
            # group, or changed the old group's material.  Deduplicate only a
            # post-transition group whose complete membership now resolves to
            # its recorded material signature.
            current_group = self._mesh_id_to_group.get(id(mesh))
            if current_group is None:
                continue
            signature = self._merged_group_material_signature(current_group)
            if signature is not None and self._merged_group_members_share_material(
                current_group,
                signature,
            ):
                synchronized_groups.add(current_group)
        return all_synced

    def _try_sync_merged_material_batch(
        self,
        group_name: str,
        resolved_entries: list[tuple[dict[str, Any], Any]],
    ) -> tuple[bool, bool]:
        """Update one compatible aggregate directly for a material-wide edit.

        The appearance coordinator has already staged each member's new base
        material. When the batch covers the complete merge group and every
        member still has one compatible material, changing a PBR slider is an
        owner-level material update. Genuine regrouping changes fall back to
        the complete entry refresh path; visibility and highlight partitions
        still flow through the normal batch synchronizer.

        Returns:
            ``(eligible, synced)``. An ineligible group has not been mutated.
        """
        members = self._merged_groups.get(group_name)
        group_info = self._merged_meshes.get(group_name)
        if not members or not isinstance(group_info, SceneBatch):
            return False, False
        if group_name in self._pending_merged_group_syncs:
            return False, False

        by_mesh_id: dict[int, tuple[dict[str, Any], Any]] = {}
        for entry, resolved in resolved_entries:
            mesh = entry.get("mesh")
            if not isinstance(mesh, RenderObjectState):
                return False, False
            by_mesh_id[id(mesh)] = (entry, resolved)
        if set(by_mesh_id) != set(members):
            return False, False

        signatures: list[tuple[str, MaterialPayload]] = []
        ordered_meshes: list[RenderObjectState] = []
        for mesh_id in members:
            entry, _resolved = by_mesh_id[mesh_id]
            mesh = entry["mesh"]
            signatures.append(self._scene_material_signature(entry, mesh.material))
            ordered_meshes.append(mesh)
        signature = signatures[0]
        if any(candidate != signature for candidate in signatures[1:]):
            return False, False

        base_material = signature[1]
        group_info.material_signature = signature
        for mesh in ordered_meshes:
            mesh.material = base_material

        synced = self._sync_merged_group_snapshot(
            group_name,
            resolved_by_mesh_id={mesh_id: by_mesh_id[mesh_id][1] for mesh_id in members},
        )
        return True, synced

    def sync_scene_entry_visibility_batch(
        self,
        entries: list[tuple[dict[str, Any], bool]],
    ) -> bool:
        """Publish a semantic visibility batch with one rebuild per aggregate.

        Session restore and other bulk policy changes may touch thousands of
        scene entries while the renderer owns only a few merged aggregates.
        Stage every entry's semantic state first, then converge each aggregate
        once so renderer work scales with render owners instead of XML members.
        """
        if not entries:
            return True

        all_synced = True
        merged_groups: set[str] = set()
        individual_entries: list[tuple[dict[str, Any], RenderObjectState, bool]] = []

        for entry, effective_visible in entries:
            mesh = ensure_scene_mesh_render_state(entry)
            group_name = self._mesh_id_to_group.get(id(mesh))
            if group_name is not None:
                merged_groups.add(group_name)
            else:
                individual_entries.append((entry, mesh, bool(effective_visible)))

        with self.visualizer.renderer.batch_updates():
            for entry, mesh, effective_visible in individual_entries:
                if not self._sync_individual_scene_snapshot(
                    entry,
                    mesh,
                    effective_visible=effective_visible,
                    rebuild_outline=False,
                ):
                    all_synced = False

            for group_name in sorted(merged_groups):
                if not self.rebuild_merged_group_from_state(group_name):
                    all_synced = False

        return all_synced

    def sync_scene_entry_snapshot(
        self,
        entry: dict[str, Any],
        *,
        material: dict[str, Any] | MaterialPayload | None = None,
        effective_visible: bool | None = None,
        geometry_changed: bool = False,
    ) -> bool:
        """Converge one scene entry through its sole current render owner."""
        # Ownership transitions can remove an aggregate and create an
        # individual mesh plus outline. Present only after that full semantic
        # transition has converged; frame-wide batches may safely nest this.
        with self.visualizer.renderer.batch_updates():
            return self._sync_scene_entry_snapshot(
                entry,
                material=material,
                effective_visible=effective_visible,
                geometry_changed=geometry_changed,
            )

    def _sync_scene_entry_snapshot(
        self,
        entry: dict[str, Any],
        *,
        material: dict[str, Any] | MaterialPayload | None = None,
        effective_visible: bool | None = None,
        geometry_changed: bool = False,
    ) -> bool:
        """Synchronize one scene entry while its semantic batch is active."""
        mesh = ensure_scene_mesh_render_state(entry)
        if material is not None:
            mesh.material = material_payload_from_mapping(material)

        pending_source = self._pending_scene_entry_sources.get(id(mesh))
        if pending_source is not None:
            if not self._sync_merged_group_snapshot(pending_source):
                return False
            self._pending_scene_entry_sources.pop(id(mesh), None)

        group_name = self._mesh_id_to_group.get(id(mesh))
        if group_name is not None:
            # A previous backend failure may have left the desired group
            # published in the ownership maps without a complete native
            # representation. Retry that exact snapshot before interpreting
            # this call as a material-only update; otherwise an absent
            # aggregate could be mistaken for a successful no-op.
            retried_pending_snapshot = group_name in self._pending_merged_group_syncs
            if retried_pending_snapshot:
                if not self._sync_merged_group_snapshot(group_name):
                    return False
                # The successful retry above already applied an identical
                # material-only snapshot. Avoid publishing the same aggregate
                # twice in one command. Visibility and geometry requests must
                # still continue through their dedicated synchronization.
                if (
                    material is not None
                    and effective_visible is None
                    and not geometry_changed
                    and self._merged_group_accepts_material(
                        group_name,
                        entry,
                        mesh.material,
                    )
                ):
                    return True
            if material is not None and not self._merged_group_accepts_material(
                group_name,
                entry,
                mesh.material,
            ):
                signature = self._scene_material_signature(entry, mesh.material)
                if self._merged_group_members_share_material(group_name, signature):
                    return self._update_merged_group_material(
                        group_name,
                        entry,
                        mesh.material,
                    )
                if not self._detach_merged_member(group_name, id(mesh)):
                    self._pending_scene_entry_sources[id(mesh)] = group_name
                    return False
                return self._sync_unmerged_material_entry(
                    entry,
                    mesh,
                    mesh.material,
                    effective_visible=effective_visible,
                )
            if material is not None:
                return self._update_merged_group_material(
                    group_name,
                    entry,
                    mesh.material,
                )
            return self.rebuild_merged_group_from_state(group_name)

        if material is not None:
            return self._sync_unmerged_material_entry(
                entry,
                mesh,
                mesh.material,
                effective_visible=effective_visible,
            )
        return self._sync_individual_scene_snapshot(
            entry,
            mesh,
            effective_visible=effective_visible,
            rebuild_outline=geometry_changed,
        )

    @staticmethod
    def _scene_material_signature(
        entry: dict[str, Any],
        material: MaterialPayload,
    ) -> tuple[str, MaterialPayload]:
        """Return the material identity that permits scene meshes to share a draw."""
        return str(entry.get("material_type") or "default"), material

    def _merged_group_material_signature(
        self,
        group_name: str,
    ) -> tuple[str, MaterialPayload] | None:
        """Return the effective material identity of a merged aggregate."""
        group_info = self._merged_meshes.get(group_name)
        if not isinstance(group_info, SceneBatch):
            return None
        material_key = group_info.material_signature
        if (
            isinstance(material_key, tuple)
            and len(material_key) == 2
            and isinstance(material_key[1], MaterialPayload)
        ):
            return str(material_key[0]), material_key[1]
        return None

    def _merged_group_accepts_material(
        self,
        group_name: str,
        entry: dict[str, Any],
        material: MaterialPayload,
    ) -> bool:
        """Return whether *entry* still belongs in the aggregate's material batch."""
        return self._merged_group_material_signature(group_name) == (
            self._scene_material_signature(entry, material)
        )

    def _find_compatible_merged_group(
        self,
        entry: dict[str, Any],
        material: MaterialPayload,
    ) -> str | None:
        """Find a compatible aggregate that remains inside renderer limits."""
        mesh = entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            return None
        signature = self._scene_material_signature(entry, material)
        for group_name, batch in self._merged_meshes.items():
            if self._merged_group_material_signature(
                group_name
            ) == signature and self._scene_batch_accepts_member(batch, mesh):
                return group_name
        return None

    def _scene_batch_accepts_member(
        self,
        batch: SceneBatch,
        mesh: RenderObjectState,
    ) -> bool:
        """Return whether attaching *mesh* preserves both hard batch limits."""
        mesh_id = id(mesh)
        if mesh_id in batch.member_mesh_ids:
            return True

        member_limit, triangle_limit = self._scene_batch_limits()
        if len(batch.member_mesh_ids) >= member_limit:
            return False

        triangle_count = self._scene_mesh_triangle_count(mesh)
        if triangle_count > triangle_limit:
            return False
        for member_id in batch.member_mesh_ids:
            entry = self._mesh_id_to_scene_entry.get(member_id)
            member = entry.get("mesh") if isinstance(entry, dict) else None
            if not isinstance(member, RenderObjectState):
                return False
            triangle_count += self._scene_mesh_triangle_count(member)
            if triangle_count > triangle_limit:
                return False
        return True

    def _bounded_compatible_entries(
        self,
        entry: dict[str, Any],
        signature: tuple[str, MaterialPayload],
    ) -> list[dict[str, Any]]:
        """Collect at most one renderer-bounded group containing *entry*."""
        mesh = entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            return []

        member_limit, triangle_limit = self._scene_batch_limits()
        triangle_count = self._scene_mesh_triangle_count(mesh)
        selected = [entry]
        if member_limit < 2 or triangle_count > triangle_limit:
            return selected

        for candidate in getattr(self.visualizer, "mesh_entries", []):
            if len(selected) >= member_limit:
                break
            if candidate is entry or not isinstance(candidate, dict):
                continue
            candidate_mesh = candidate.get("mesh")
            if not isinstance(candidate_mesh, RenderObjectState):
                continue
            if id(candidate_mesh) in self._mesh_id_to_group:
                continue
            if str(candidate.get("material_type") or "default") != signature[0]:
                continue
            candidate_material = self._scene_entry_base_material(candidate).payload
            if self._scene_material_signature(candidate, candidate_material) != signature:
                continue
            candidate_triangles = self._scene_mesh_triangle_count(candidate_mesh)
            if triangle_count + candidate_triangles > triangle_limit:
                continue
            selected.append(candidate)
            triangle_count += candidate_triangles
        return selected

    def _merged_group_members_share_material(
        self,
        group_name: str,
        signature: tuple[str, MaterialPayload],
    ) -> bool:
        """Return whether every member now resolves to one material signature."""
        member_ids = self._merged_groups.get(group_name, [])
        if not member_ids:
            return False
        for mesh_id in member_ids:
            member_entry = self._mesh_id_to_scene_entry.get(mesh_id)
            if not isinstance(member_entry, dict):
                return False
            member_material = self._scene_entry_base_material(member_entry).payload
            if self._scene_material_signature(member_entry, member_material) != signature:
                return False
        return True

    def _resolved_scene_appearance(self, entry: dict[str, Any]) -> Any:
        """Resolve one member through the application appearance coordinator."""
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        resolve = getattr(appearance, "resolve_entry_runtime_appearance", None)
        if callable(resolve):
            return resolve(entry)
        material = self._scene_entry_base_material(entry).payload
        mesh = entry.get("mesh")
        color_source = (
            mesh.payload.color_source
            if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload)
            else SurfaceColorSource.MATERIAL
        )
        return resolve_appearance(
            AppearanceIntent(
                manual_visible=bool(entry.get("visible", True)),
                manual_highlight=bool(entry.get("highlighted", False)),
                material=material,
                color_source=color_source,
            )
        )

    def _update_merged_group_material(
        self,
        group_name: str,
        entry: dict[str, Any],
        material: MaterialPayload,
    ) -> bool:
        """Update one aggregate material while keeping its grouping metadata coherent."""
        group_info = self._merged_meshes.get(group_name)
        if not isinstance(group_info, SceneBatch):
            return False
        group_info.material_signature = self._scene_material_signature(entry, material)
        for mesh_id in self._merged_groups.get(group_name, []):
            member_entry = self._mesh_id_to_scene_entry.get(mesh_id)
            member = member_entry.get("mesh") if isinstance(member_entry, dict) else None
            if isinstance(member, RenderObjectState):
                member.material = material
        return self.rebuild_merged_group_from_state(group_name)

    def _sync_unmerged_material_entry(
        self,
        entry: dict[str, Any],
        mesh: RenderObjectState,
        material: MaterialPayload,
        *,
        effective_visible: bool | None,
    ) -> bool:
        """Synchronize a material-owned member without losing merge efficiency."""
        compatible_group = self._find_compatible_merged_group(entry, material)
        if compatible_group is not None:
            if compatible_group in self._pending_merged_group_syncs and not (
                self._sync_merged_group_snapshot(compatible_group)
            ):
                return False
            retired = self._retire_scene_entries_for_batch([entry])
            if retired is None:
                return False
            try:
                self._attach_merged_member(compatible_group, entry, mesh)
            except ValueError as exc:
                self._restore_individual_scene_owners(retired)
                logger.warning("Failed to attach scene member to %s: %s", compatible_group, exc)
                return False
            self._pending_merged_group_syncs.add(compatible_group)
            return self.rebuild_merged_group_from_state(compatible_group)

        if not self._merge_enabled:
            return self._sync_individual_scene_snapshot(
                entry,
                mesh,
                effective_visible=effective_visible,
            )

        signature = self._scene_material_signature(entry, material)
        compatible_entries = self._bounded_compatible_entries(entry, signature)
        if len(compatible_entries) >= 2:
            retired = self._retire_scene_entries_for_batch(compatible_entries)
            if retired is None:
                return False
            try:
                group_name = self._create_material_group(compatible_entries, signature)
            except ValueError as exc:
                self._restore_individual_scene_owners(retired)
                logger.warning("Failed to create bounded scene batch: %s", exc)
                return False
            self._pending_merged_group_syncs.add(group_name)
            return self.rebuild_merged_group_from_state(group_name)
        return self._sync_individual_scene_snapshot(
            entry,
            mesh,
            effective_visible=effective_visible,
        )

    def _sync_individual_scene_snapshot(
        self,
        entry: dict[str, Any],
        mesh: RenderObjectState,
        *,
        effective_visible: bool | None,
        rebuild_outline: bool = False,
    ) -> bool:
        """Synchronize an unmerged mesh and outline inside its caller's batch."""
        self._individual_scene_owner_ids.add(mesh.id)
        if not self._render_sync.ensure_object(
            mesh,
            effective_visible=effective_visible,
        ):
            return False
        visible = bool(entry.get("visible", True))
        if effective_visible is not None:
            visible = bool(effective_visible)
        appearance = getattr(self.visualizer, "scene_appearance_service", None)
        sync_outline = getattr(appearance, "sync_scene_entry_outline_snapshot", None)
        if callable(sync_outline):
            return bool(
                sync_outline(
                    entry,
                    visible=bool(visible and getattr(self.visualizer, "outlines_enabled", False)),
                    rebuild=rebuild_outline,
                )
            )
        return not bool(
            getattr(self.visualizer, "outlines_enabled", False)
            or isinstance(entry.get("outline_geometry"), RenderObjectState)
        )

    def _retire_individual_scene_owner(
        self,
        entry: dict[str, Any],
    ) -> bool:
        """Ensure one tracked individual mesh and outline owner is absent."""
        mesh = entry.get("mesh")
        if not isinstance(mesh, RenderObjectState):
            return False
        if mesh.id not in self._individual_scene_owner_ids:
            return True
        if not self._render_sync.remove_object(mesh):
            return False
        appearance = getattr(self.visualizer, "scene_appearance_service", None)
        remove_outline = getattr(appearance, "remove_scene_entry_outline", None)
        if callable(remove_outline):
            if not remove_outline(entry):
                return False
        elif isinstance(entry.get("outline_geometry"), RenderObjectState):
            return False
        self._individual_scene_owner_ids.discard(mesh.id)
        return True

    def _retire_scene_entries_for_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Retire candidates and return owners to restore if commit later fails."""
        retired: list[dict[str, Any]] = []
        for entry in entries:
            mesh = entry.get("mesh")
            if not isinstance(mesh, RenderObjectState):
                self._restore_individual_scene_owners(retired)
                return None
            resolved = self._resolved_scene_appearance(entry)
            should_retire = not bool(
                resolved is not None and resolved.visible and resolved.highlighted
            )
            if not should_retire:
                continue
            if mesh.id in self._individual_scene_owner_ids:
                retired.append(entry)
            if not self._retire_individual_scene_owner(entry):
                self._restore_individual_scene_owners(retired)
                return None
        return retired

    def _restore_individual_scene_owners(self, entries: list[dict[str, Any]]) -> None:
        """Best-effort rollback for owners retired before a batch transition failed."""
        for entry in entries:
            mesh = entry.get("mesh")
            if not isinstance(mesh, RenderObjectState):
                continue
            resolved = self._resolved_scene_appearance(entry)
            effective_visible = None if resolved is None else bool(resolved.visible)
            if not self._sync_individual_scene_snapshot(
                entry,
                mesh,
                effective_visible=effective_visible,
                rebuild_outline=True,
            ):
                logger.warning(
                    "Failed to restore individual scene owner %s after batch rollback",
                    mesh.id,
                )

    def _create_material_group(
        self,
        entries: list[dict[str, Any]],
        signature: tuple[str, MaterialPayload],
    ) -> str:
        """Create one renderer-owned aggregate for compatible unmerged entries."""
        meshes = [entry.get("mesh") for entry in entries]
        if len(meshes) < 2 or any(not isinstance(mesh, RenderObjectState) for mesh in meshes):
            raise ValueError("A material batch requires at least two persistent scene meshes")
        persistent_meshes = [mesh for mesh in meshes if isinstance(mesh, RenderObjectState)]
        member_limit, triangle_limit = self._scene_batch_limits()
        if len(persistent_meshes) > member_limit:
            raise ValueError("Material batch exceeds the renderer member limit")
        if (
            sum(self._scene_mesh_triangle_count(mesh) for mesh in persistent_meshes)
            > triangle_limit
        ):
            raise ValueError("Material batch exceeds the renderer triangle limit")

        group_name = self._scene_batch_name_for_owner_ids(
            signature,
            [mesh.id for mesh in persistent_meshes],
        )
        batch = SceneBatch(
            name=group_name,
            material_signature=signature,
            member_mesh_ids=[id(mesh) for mesh in persistent_meshes],
        )
        self._scene_batches.add_batch(batch)
        for candidate, candidate_mesh in zip(entries, persistent_meshes, strict=True):
            candidate_mesh.material = signature[1]
            self._scene_batches.register_entry(id(candidate_mesh), candidate)
        return group_name

    def _attach_merged_member(
        self,
        group_name: str,
        entry: dict[str, Any],
        mesh: RenderObjectState,
    ) -> None:
        """Attach one persistent scene entry to an existing compatible aggregate."""
        mesh_id = id(mesh)
        self._scene_batches.attach(group_name, mesh_id, entry)

    def _detach_merged_member(self, group_name: str, mesh_id: int) -> bool:
        """Remove one member from its old owner before publishing a new one."""
        if not self._scene_batches.detach(group_name, mesh_id):
            return True
        return self._sync_merged_group_snapshot(group_name)

    def _sync_merged_group_snapshot(
        self,
        group_name: str,
        *,
        resolved_by_mesh_id: dict[int, Any] | None = None,
    ) -> bool:
        """Converge one desired aggregate without publishing duplicate owners."""
        members = self._merged_groups.get(group_name)
        group_info = self._merged_meshes.get(group_name)
        if members is None or not isinstance(group_info, SceneBatch):
            self._pending_merged_group_syncs.discard(group_name)
            return True

        if not members:
            state = group_info.get("render_state")
            if isinstance(state, RenderObjectState) and not self._render_sync.remove_object(state):
                self._pending_merged_group_syncs.add(group_name)
                return False
            appearance = getattr(self.visualizer, "scene_appearance_service", None)
            remove_outline = getattr(appearance, "remove_merged_outline_geometry", None)
            if callable(remove_outline):
                if not remove_outline(group_info):
                    self._pending_merged_group_syncs.add(group_name)
                    return False
            elif isinstance(group_info.get("_merged_outline"), RenderObjectState):
                self._pending_merged_group_syncs.add(group_name)
                return False
            self._scene_batches.remove_batch(group_name)
            return True

        entries: list[dict[str, Any]] = []
        for mesh_id in members:
            entry = self._mesh_id_to_scene_entry.get(mesh_id)
            if not isinstance(entry, dict) or not isinstance(entry.get("mesh"), RenderObjectState):
                self._pending_merged_group_syncs.add(group_name)
                return False
            entries.append(entry)

        if resolved_by_mesh_id is None:
            resolved_by_mesh_id = {
                mesh_id: self._resolved_scene_appearance(entry)
                for mesh_id, entry in zip(members, entries, strict=True)
            }
        elif set(resolved_by_mesh_id) != set(members):
            self._pending_merged_group_syncs.add(group_name)
            return False
        partition = group_info.resolve_partition(resolved_by_mesh_id)
        entries_by_mesh_id = dict(zip(members, entries, strict=True))
        aggregate_meshes = [
            entries_by_mesh_id[mesh_id]["mesh"] for mesh_id in partition.aggregate_member_ids
        ]
        split_entries = [
            (entries_by_mesh_id[mesh_id], resolved_by_mesh_id[mesh_id])
            for mesh_id in partition.individual_member_ids
        ]

        # Meshes and outlines consume this same partition.  Anything not in
        # the individual-exception set must have its obsolete individual owner
        # retired before the aggregate (or hidden state) is published.
        split_mesh_ids = set(partition.individual_member_ids)
        for mesh_id, entry in entries_by_mesh_id.items():
            if mesh_id in split_mesh_ids:
                continue
            if not self._retire_individual_scene_owner(entry):
                self._pending_merged_group_syncs.add(group_name)
                return False

        with self.visualizer.renderer.batch_updates():
            if aggregate_meshes:
                outline_rebuild_needed = bool(
                    group_info.get("_merged_outline_needs_rebuild", False)
                )
                geometry_sources = tuple((mesh.id, mesh.payload) for mesh in aggregate_meshes)
                previous_sources = group_info.get("geometry_sources")
                sources_unchanged = (
                    isinstance(previous_sources, tuple)
                    and len(previous_sources) == len(geometry_sources)
                    and all(
                        previous_id == desired_id and previous_payload is desired_payload
                        for (previous_id, previous_payload), (
                            desired_id,
                            desired_payload,
                        ) in zip(previous_sources, geometry_sources, strict=True)
                    )
                )
                state = group_info.get("render_state")
                baseline_matches = (
                    bool(group_info.baseline_sources)
                    and len(group_info.baseline_sources) == len(geometry_sources)
                    and all(
                        baseline_id == desired_id and baseline_payload is desired_payload
                        for (baseline_id, baseline_payload), (
                            desired_id,
                            desired_payload,
                        ) in zip(group_info.baseline_sources, geometry_sources, strict=True)
                    )
                )
                if (
                    sources_unchanged
                    and isinstance(state, RenderObjectState)
                    and isinstance(state.payload, MeshPayload)
                ):
                    merged = state.payload
                elif baseline_matches and isinstance(group_info.baseline_geometry, MeshPayload):
                    merged = group_info.baseline_geometry
                    group_info["geometry_sources"] = geometry_sources
                    outline_rebuild_needed = not sources_unchanged
                else:
                    merged = self._merge_scene_mesh_payloads(aggregate_meshes)
                    if merged is None:
                        self._pending_merged_group_syncs.add(group_name)
                        return False
                    group_info["geometry"] = merged
                    group_info["geometry_sources"] = geometry_sources
                    outline_rebuild_needed = True
                    group_info["_merged_outline_needs_rebuild"] = True
                group_info["geometry"] = merged
                aggregate_material = partition.aggregate_material
                if aggregate_material is None:
                    aggregate_material = group_info.material_signature[1]
                if not self._ensure_merged_material_geometry(
                    group_name=group_name,
                    group_info=group_info,
                    geometry=merged,
                    material=aggregate_material,
                    visible=True,
                ):
                    self._pending_merged_group_syncs.add(group_name)
                    return False
                if not self._sync_merged_outline_after_geometry_change(
                    group_name,
                    group_info,
                    parent_visible=True,
                    rebuild=outline_rebuild_needed,
                ):
                    self._pending_merged_group_syncs.add(group_name)
                    return False
                group_info.pop("_merged_outline_needs_rebuild", None)
            else:
                state = group_info.get("render_state")
                if isinstance(state, RenderObjectState):
                    state.visible = False
                    if not self._render_sync.remove_object(state):
                        self._pending_merged_group_syncs.add(group_name)
                        return False
                if not self._sync_merged_outline_after_geometry_change(
                    group_name,
                    group_info,
                    parent_visible=False,
                    rebuild=bool(group_info.get("_merged_outline_needs_rebuild", False)),
                ):
                    self._pending_merged_group_syncs.add(group_name)
                    return False
                group_info.pop("_merged_outline_needs_rebuild", None)

            # A mixed aggregate has excluded only its highlighted exceptions.
            # Publish their temporary material on immutable snapshots; the
            # application-owned mesh keeps its canonical base material.
            appearance = getattr(self.visualizer, "scene_appearance_service", None)
            sync_outline = getattr(appearance, "sync_scene_entry_outline_snapshot", None)
            for entry, resolved in split_entries:
                mesh = entry["mesh"]
                self._individual_scene_owner_ids.add(mesh.id)
                if not self._render_sync.ensure_object(
                    mesh,
                    effective_visible=True,
                    snapshot_material=resolved.material,
                ):
                    self._pending_merged_group_syncs.add(group_name)
                    return False
                if callable(sync_outline):
                    if not sync_outline(
                        entry,
                        visible=bool(getattr(self.visualizer, "outlines_enabled", False)),
                        rebuild=True,
                    ):
                        self._pending_merged_group_syncs.add(group_name)
                        return False
                elif bool(getattr(self.visualizer, "outlines_enabled", False)):
                    self._pending_merged_group_syncs.add(group_name)
                    return False

        self._pending_merged_group_syncs.discard(group_name)
        return True

    def rebuild_merged_group_from_state(self, group_name: str) -> bool:
        """Rebuild one merged group after batched visibility/highlight changes."""
        self._pending_merged_group_syncs.add(group_name)
        return self._sync_merged_group_snapshot(group_name)

    def _sync_merged_outline_after_geometry_change(
        self,
        group_name: str,
        group_info: SceneBatch,
        *,
        parent_visible: bool,
        rebuild: bool,
    ) -> bool:
        """Keep a cached merged outline on the same aggregate revision."""
        appearance = getattr(self.visualizer, "scene_appearance_service", None)
        sync_outline = getattr(appearance, "sync_merged_outline_geometry", None)
        if not callable(sync_outline):
            outline = group_info.get("_merged_outline")
            return not bool(
                getattr(self.visualizer, "outlines_enabled", False)
                or isinstance(outline, RenderObjectState)
            )
        return bool(
            sync_outline(
                group_name,
                group_info,
                visible=bool(
                    parent_visible and getattr(self.visualizer, "outlines_enabled", False)
                ),
                rebuild=bool(rebuild),
            )
        )

    def sync_scene_resolved_appearance(
        self,
        entry: dict[str, Any],
        resolved: Any,
    ) -> bool:
        """Publish one resolved scene snapshot while retaining canonical base material."""
        mesh = ensure_scene_mesh_render_state(entry)
        base_material = self._scene_entry_base_material(entry).payload
        if not self._sync_scene_entry_snapshot(
            entry,
            material=base_material,
            effective_visible=bool(resolved.visible),
        ):
            return False
        group_name = self._mesh_id_to_group.get(id(mesh))
        if group_name is not None:
            return True
        if not resolved.highlighted:
            return True
        mesh.material = base_material
        self._individual_scene_owner_ids.add(mesh.id)
        return self._render_sync.ensure_object(
            mesh,
            effective_visible=bool(resolved.visible),
            snapshot_material=resolved.material,
        )

    def sync_scene_resolved_appearance_batch(
        self,
        resolved_entries: list[tuple[dict[str, Any], Any]],
        *,
        materials_changed: bool,
    ) -> bool:
        """Publish a scene batch with at most one rebuild per merged owner."""
        if not resolved_entries:
            return True
        all_synced = True
        if materials_changed:
            generated_uvs = False
            for entry, _resolved in resolved_entries:
                mesh = ensure_scene_mesh_render_state(entry)
                generated_uvs = bool(
                    self._ensure_scene_member_uvs(mesh, entry, mesh.material) or generated_uvs
                )
            if generated_uvs:
                finalize_uv_cache_stores()

            by_group: dict[str, list[tuple[dict[str, Any], Any]]] = {}
            fallback_entries: list[dict[str, Any]] = []
            for entry, resolved in resolved_entries:
                mesh = ensure_scene_mesh_render_state(entry)
                group_name = self._mesh_id_to_group.get(id(mesh))
                if group_name is None:
                    fallback_entries.append(entry)
                else:
                    by_group.setdefault(group_name, []).append((entry, resolved))

            for group_name, group_entries in by_group.items():
                eligible, synced = self._try_sync_merged_material_batch(
                    group_name,
                    group_entries,
                )
                if eligible:
                    all_synced = bool(all_synced and synced)
                else:
                    fallback_entries.extend(entry for entry, _resolved in group_entries)
            if fallback_entries and not self._refresh_scene_material_entries(fallback_entries):
                all_synced = False

        merged_groups: set[str] = set()
        individual: list[tuple[dict[str, Any], Any, RenderObjectState]] = []
        for entry, resolved in resolved_entries:
            mesh = ensure_scene_mesh_render_state(entry)
            group_name = self._mesh_id_to_group.get(id(mesh))
            if group_name is None:
                individual.append((entry, resolved, mesh))
            else:
                merged_groups.add(group_name)

        with self.visualizer.renderer.batch_updates():
            for entry, resolved, mesh in individual:
                self._individual_scene_owner_ids.add(mesh.id)
                snapshot_material = resolved.material if resolved.highlighted else None
                if not self._render_sync.ensure_object(
                    mesh,
                    effective_visible=bool(resolved.visible),
                    snapshot_material=snapshot_material,
                ):
                    all_synced = False
                    continue
                appearance = getattr(self.visualizer, "scene_appearance_service", None)
                sync_outline = getattr(
                    appearance,
                    "sync_scene_entry_outline_snapshot",
                    None,
                )
                if callable(sync_outline) and not sync_outline(
                    entry,
                    visible=bool(
                        resolved.visible and getattr(self.visualizer, "outlines_enabled", False)
                    ),
                    rebuild=False,
                ):
                    all_synced = False

            if not materials_changed:
                for group_name in sorted(merged_groups):
                    if not self.rebuild_merged_group_from_state(group_name):
                        all_synced = False
        return all_synced
