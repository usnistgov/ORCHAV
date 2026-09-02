"""Object-level visibility, label, and material appearance orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict

from shared.logging import get_logger

from ..materials.appearance import (
    AppearanceIntent,
    MaterialDisplayMode,
    ResolvedAppearance,
    resolve_appearance,
)
from ..materials.catalog import pbr_props_to_kwargs
from ..model import RenderObjectState
from ..scene.geometry_helpers import (
    ensure_scene_mesh_render_state,
    require_target_mesh_render_state,
)
from ..scene.visibility_policy import (
    effective_entry_label_visibility,
    effective_entry_visibility,
)
from ..types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
    material_payload_from_mapping,
)
from .base import BaseService
from .object_entry_index import CanonicalEntryIndex
from .pov_visibility_service import is_hidden_for_pov
from .scene_render_sync import SceneRenderSync

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.object_appearance_service")


class ObjectAppearanceService(BaseService):
    """Coordinate object visibility, labels, and highlight material state."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind object appearance operations to visualizer-owned services."""
        super().__init__()
        self.visualizer = visualizer
        self._scene_render_sync = SceneRenderSync(visualizer)
        self._entry_registry = CanonicalEntryIndex(visualizer)

    @property
    def renderer(self) -> Any:
        """Return the active renderer, if one is attached."""
        return getattr(self.visualizer, "renderer", None)

    @property
    def scene_service(self) -> Any:
        """Return the scene service used for merged-scene operations."""
        return getattr(self.visualizer, "scene_service", None)

    def resolve_canonical_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the authoritative entry dict across scene, target, and node entries."""
        return self._entry_registry.resolve(entry)

    def refresh_entry_index(
        self,
        entry: Dict[str, Any],
        *,
        entry_type: str | None = None,
    ) -> None:
        """Publish stable identity keys assigned after initial indexing."""
        self._entry_registry.refresh_entry(entry, entry_type=entry_type)

    def entry_index(
        self,
        entry: Dict[str, Any],
        *,
        entry_type: str | None = None,
    ) -> int:
        """Return a persistent entry index without rebuilding combined lists."""
        return self._entry_registry.index_for_entry(entry, entry_type=entry_type)

    def entry_kind(self, entry: Dict[str, Any]) -> str | None:
        """Return the persistent collection kind containing *entry*."""
        return self._entry_registry.entry_type(entry)

    def set_object_visibility(
        self,
        entry: Dict[str, Any],
        visible: bool,
        update_renderer: bool = True,
    ) -> None:
        """Apply one entry's visibility through the canonical batch path."""
        self.set_object_visibility_batch(
            [entry],
            visible,
            update_renderer=update_renderer,
        )

    def set_object_visibility_batch(
        self,
        entries: list[Dict[str, Any]],
        visible: bool,
        *,
        update_renderer: bool = True,
    ) -> bool:
        """Set mixed scene, target, and node visibility as one transaction.

        This service owns the cross-domain transaction because object-panel
        groups can contain entries from several semantic owners. Each owner
        still performs its own synchronization; the renderer sees only the
        final combined state and receives one redraw request.
        """
        canonical_entries = self._canonical_visibility_entries(
            entries,
            visible=bool(visible),
        )
        return self._publish_object_visibility_batch(
            canonical_entries,
            update_renderer=update_renderer,
        )

    def refresh_object_visibility_batch(
        self,
        entries: list[Dict[str, Any]],
        *,
        update_renderer: bool = True,
    ) -> bool:
        """Publish staged visibility for mixed canonical entries once.

        Callers must stage semantic ``entry['visible']`` values before this
        method. Each domain service retains its synchronization policy while
        this service owns the combined renderer transaction and redraw.
        """
        canonical_entries = self._canonical_visibility_entries(entries)
        return self._publish_object_visibility_batch(
            canonical_entries,
            update_renderer=update_renderer,
        )

    def _canonical_visibility_entries(
        self,
        entries: list[Dict[str, Any]],
        *,
        visible: bool | None = None,
    ) -> list[Dict[str, Any]]:
        """Resolve, stage, and deduplicate visibility entries by ownership."""
        canonical_entries: list[Dict[str, Any]] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            canonical = self.resolve_canonical_entry(entry)
            if visible is not None:
                canonical["visible"] = bool(visible)
                entry["visible"] = bool(visible)

            entry_type = str(canonical.get("entry_type", "mesh")).lower()
            mesh = canonical.get("mesh")
            if mesh is not None and entry_type in {"mesh", "target"}:
                mesh = self._require_entry_render_state(canonical)
                mesh.visible = bool(canonical.get("visible", True))

            canonical_id = id(canonical)
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            canonical_entries.append(canonical)
        return canonical_entries

    def _publish_object_visibility_batch(
        self,
        entries: list[Dict[str, Any]],
        *,
        update_renderer: bool,
    ) -> bool:
        """Synchronize canonical visibility intents in one renderer batch."""
        if not entries:
            return True
        if not self._renderer_ready():
            return False

        renderer = self.renderer
        batch_updates = getattr(renderer, "batch_updates", None)
        ctx = batch_updates() if callable(batch_updates) else nullcontext()
        with ctx:
            return self._refresh_object_visibility_batch_unpresented(
                entries,
                update_renderer=update_renderer,
            )

    def _refresh_object_visibility_batch_unpresented(
        self,
        entries: list[Dict[str, Any]],
        *,
        update_renderer: bool,
    ) -> bool:
        """Apply a staged visibility batch inside one renderer transaction."""

        scene_batch: list[tuple[Dict[str, Any], bool]] = []
        scene_entries: list[Dict[str, Any]] = []
        all_synced = True
        publish_attempted = False
        nodes_changed = False

        for canonical in entries:
            entry_type = str(canonical.get("entry_type", "mesh")).lower()
            if entry_type in {"tx", "rx"}:
                nodes_changed = True
                continue

            mesh = canonical.get("mesh")
            if mesh is None:
                continue
            mesh = self._require_entry_render_state(canonical)
            mesh.visible = bool(canonical.get("visible", True))
            effective_visible = self._effective_entry_render_visibility(canonical, mesh)

            if entry_type == "target":
                publish_attempted = True
                target_service = getattr(self.visualizer, "target_service", None)
                sync_snapshot = getattr(target_service, "sync_target_entry_snapshot", None)
                if not callable(sync_snapshot):
                    all_synced = False
                    continue
                synced = bool(sync_snapshot(canonical, effective_visible=bool(effective_visible)))
                all_synced = bool(all_synced and synced)
                continue

            if entry_type != "mesh":
                all_synced = False
                continue
            scene_entries.append(canonical)
            scene_batch.append((canonical, bool(effective_visible)))

        if scene_batch:
            publish_attempted = True
            scene_service = self.scene_service
            sync_batch = getattr(scene_service, "sync_scene_entry_visibility_batch", None)
            if not callable(sync_batch):
                all_synced = False
            else:
                synced = bool(sync_batch(scene_batch))
                all_synced = bool(all_synced and synced)

            # Parent visibility participates in effective label visibility.
            # Mesh synchronization is batched above; labels remain independent
            # persistent objects and are each updated at most once.
            for scene_entry in scene_entries:
                self._sync_building_label_geometry(
                    scene_entry,
                    create_if_missing=False,
                )

        if nodes_changed:
            publish_attempted = True
            node_service = getattr(self.visualizer, "node_service", None)
            sync_nodes = getattr(node_service, "sync_node_visibility_snapshot", None)
            if not callable(sync_nodes):
                all_synced = False
            else:
                all_synced = bool(sync_nodes()) and all_synced

        if publish_attempted:
            self._redraw(update_renderer)
        return all_synced

    def set_building_label_visibility(
        self,
        entry: Dict[str, Any],
        show_label: bool,
        update_renderer: bool = True,
    ) -> None:
        """Toggle label visibility for a scene, target, TX, or RX entry."""
        canonical = self.resolve_canonical_entry(entry)
        canonical["show_label"] = bool(show_label)
        entry["show_label"] = bool(show_label)
        entry_type = canonical.get("entry_type", "mesh")

        node_service = getattr(self.visualizer, "node_service", None)
        if entry_type in {"tx", "rx"}:
            if node_service is not None:
                node_service.set_node_label_visibility(canonical, bool(show_label))
            return

        if entry_type == "target":
            target_service = getattr(self.visualizer, "target_service", None)
            sync_snapshot = getattr(target_service, "sync_target_entry_snapshot", None)
            if callable(sync_snapshot) and sync_snapshot(canonical):
                self._redraw(update_renderer)
            return

        if entry_type != "mesh" or not self._renderer_ready():
            return

        try:
            updated = self._sync_building_label_geometry(
                canonical,
                create_if_missing=bool(show_label),
            )
            if updated:
                self._redraw(update_renderer)
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "Failed to update label visibility for %s: %s",
                canonical.get("name"),
                exc,
            )

    def _sync_building_label_geometry(
        self,
        entry: Dict[str, Any],
        *,
        create_if_missing: bool,
    ) -> bool:
        """Sync one scene label from stable identity and parent visibility."""
        if entry.get("entry_type", "mesh") != "mesh":
            return False
        index = self._entry_index(getattr(self.visualizer, "mesh_entries", []), entry)
        if index < 0:
            return False
        label_name = f"bldg_label_{index}"

        def _find_label() -> RenderObjectState | None:
            for label in getattr(self.visualizer, "building_labels", []) or []:
                if isinstance(label, RenderObjectState) and label.id == label_name:
                    return label
            return None

        label = _find_label()
        if label is None and create_if_missing and self.scene_service is not None:
            self.scene_service.ensure_building_labels_created()
            label = _find_label()
        if label is None:
            return False

        visible = effective_entry_label_visibility(
            entry,
            labels_enabled=True,
            entry_type="mesh",
        )
        return self._scene_render_sync.sync_label_geometry(
            name=label_name,
            geometry=label,
            visible=visible,
        )

    def set_object_highlight(
        self,
        entry: Dict[str, Any],
        highlighted: bool,
        update_renderer: bool = True,
    ) -> None:
        """Apply or remove object highlight material for a scene or target entry."""
        canonical = self.resolve_canonical_entry(entry)
        canonical["highlighted"] = bool(highlighted)
        entry["highlighted"] = bool(highlighted)
        entry_type = canonical.get("entry_type", "mesh")
        if entry_type in {"tx", "rx"}:
            return
        if canonical.get("mesh") is None:
            logger.warning(
                "No mesh found for entry %s (type=%s)", canonical.get("name"), entry_type
            )
            return
        if not self._renderer_ready():
            return

        try:
            if self.refresh_entry_material(
                canonical,
                update_renderer=False,
            ):
                self._redraw(update_renderer)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed to update highlight for %s: %s",
                canonical.get("name"),
                exc,
                exc_info=True,
            )

    def refresh_entry_material(
        self,
        entry: Dict[str, Any],
        *,
        highlighted: bool | None = None,
        update_renderer: bool = True,
    ) -> bool:
        """Refresh one entry's renderer material from effective PBR state."""
        canonical = self.resolve_canonical_entry(entry)
        mesh = canonical.get("mesh")
        if mesh is None or self.renderer is None:
            return False
        mesh = self._require_entry_render_state(canonical)
        base_material = self._entry_material_payload(canonical)
        mesh.material = base_material
        resolved = self._resolve_entry_appearance_with_material(
            canonical,
            mesh,
            base_material,
            manual_highlight_override=highlighted,
        )
        effective_visible = resolved.visible

        if canonical.get("entry_type", "mesh") == "mesh" and self.scene_service is not None:
            sync_resolved = getattr(
                self.scene_service,
                "sync_scene_resolved_appearance",
                None,
            )
            if not callable(sync_resolved):
                raise RuntimeError("SceneService material synchronization is unavailable")
            updated = bool(sync_resolved(canonical, resolved))
            if updated:
                self._redraw(update_renderer)
            return updated

        if canonical.get("entry_type") == "target":
            target_service = getattr(self.visualizer, "target_service", None)
            refresh_target = getattr(target_service, "refresh_target_entry_material", None)
            if not callable(refresh_target):
                raise RuntimeError("TargetService material synchronization is unavailable")
            updated = bool(
                refresh_target(
                    canonical,
                    resolved_appearance=resolved,
                )
            )
            if updated and update_renderer:
                self._sync_target_orientation_visibility()
            if updated:
                self._redraw(update_renderer)
            return updated

        material = resolved.material
        set_material = getattr(self.renderer, "set_material", None)
        updated = bool(callable(set_material) and set_material(mesh.id, material))
        if not updated:
            ensure_object = getattr(self.renderer, "ensure_object", None)
            if callable(ensure_object):
                updated = bool(
                    ensure_object(mesh.to_render_object(effective_visible=effective_visible))
                )
        if updated:
            self._redraw(update_renderer)
        return updated

    def refresh_entry_appearance_batch(
        self,
        entries: list[Dict[str, Any]],
        *,
        materials_changed: bool = True,
        update_renderer: bool = True,
    ) -> bool:
        """Resolve and publish mixed scene/target appearance in one presentation.

        Material-mode commands pass ``materials_changed=False`` so a visibility
        or highlight overlay never re-resolves PBR or rebuilds one merged owner
        per XML member. Durable material edits retain the default and refresh
        canonical base materials before the resolved snapshots are published.
        """
        canonical_entries: list[Dict[str, Any]] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("mesh") is None:
                continue
            canonical = self.resolve_canonical_entry(entry)
            token = id(canonical)
            if token in seen:
                continue
            seen.add(token)
            canonical_entries.append(canonical)
        if not canonical_entries:
            return True
        if not self._renderer_ready():
            return False

        resolved_entries: list[tuple[Dict[str, Any], ResolvedAppearance]] = []
        for canonical in canonical_entries:
            mesh = self._require_entry_render_state(canonical)
            if materials_changed:
                base_material = self._entry_material_payload(canonical)
                mesh.material = base_material
                resolved = self._resolve_entry_appearance_with_material(
                    canonical,
                    mesh,
                    base_material,
                )
            else:
                resolved = self._resolve_entry_appearance_with_material(
                    canonical,
                    mesh,
                    mesh.material,
                )
            resolved_entries.append((canonical, resolved))
        scene_entries = [
            (canonical, resolved)
            for canonical, resolved in resolved_entries
            if canonical.get("entry_type", "mesh") == "mesh"
        ]
        target_entries = [
            (canonical, resolved)
            for canonical, resolved in resolved_entries
            if canonical.get("entry_type") == "target"
        ]
        other_entries = [
            (canonical, resolved)
            for canonical, resolved in resolved_entries
            if canonical.get("entry_type", "mesh") not in {"mesh", "target"}
        ]

        renderer = self.renderer
        batch_updates = getattr(renderer, "batch_updates", None)
        ctx = batch_updates() if callable(batch_updates) else nullcontext()
        all_synced = True
        with ctx:
            if scene_entries:
                try:
                    sync_scene_batch = getattr(
                        self.scene_service,
                        "sync_scene_resolved_appearance_batch",
                        None,
                    )
                    if not callable(sync_scene_batch):
                        raise RuntimeError(
                            "SceneService batch appearance synchronization is unavailable"
                        )
                    all_synced = bool(
                        sync_scene_batch(
                            scene_entries,
                            materials_changed=materials_changed,
                        )
                        and all_synced
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    all_synced = False
                    logger.warning(
                        "Failed to publish scene appearance batch: %s",
                        exc,
                    )

            target_service = getattr(self.visualizer, "target_service", None)
            sync_target = getattr(target_service, "sync_target_entry_snapshot", None)
            for canonical, resolved in target_entries:
                try:
                    if not callable(sync_target):
                        raise RuntimeError(
                            "TargetService batch appearance synchronization is unavailable"
                        )
                    all_synced = bool(
                        sync_target(
                            canonical,
                            resolved_appearance=resolved,
                        )
                        and all_synced
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    all_synced = False
                    logger.warning(
                        "Failed to publish target appearance for %s: %s",
                        canonical.get("name", "Unknown"),
                        exc,
                    )
            if target_entries:
                self._sync_target_orientation_visibility()

            ensure_object = getattr(renderer, "ensure_object", None)
            for canonical, resolved in other_entries:
                try:
                    if not callable(ensure_object):
                        raise RuntimeError("Renderer object synchronization is unavailable")
                    mesh = self._require_entry_render_state(canonical)
                    mesh.material = resolved.material
                    all_synced = bool(
                        ensure_object(mesh.to_render_object(effective_visible=resolved.visible))
                        and all_synced
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    all_synced = False
                    logger.warning(
                        "Failed to publish appearance for %s: %s",
                        canonical.get("name", "Unknown"),
                        exc,
                    )
        self._redraw(update_renderer)
        return all_synced

    def _sync_target_orientation_visibility(self) -> bool:
        """Synchronize target orientation children from their resolved parents."""
        node_service = getattr(self.visualizer, "node_service", None)
        update = getattr(node_service, "update_orientation_visibility", None)
        return bool(callable(update) and update())

    def refresh_transient_selection_object(
        self,
        geometry: RenderObjectState,
        *,
        selected: bool,
        update_renderer: bool = True,
    ) -> bool:
        """Publish a node/marker selection snapshot without repainting its payload."""
        renderer = self.renderer
        ensure_object = getattr(renderer, "ensure_object", None)
        if not isinstance(geometry, RenderObjectState) or not callable(ensure_object):
            return False
        material = geometry.material
        if selected:
            material = replace(
                material,
                base_color=(1.0, 0.0, 0.0, material.base_color[3]),
                color_multiplier=(1.0, 1.0, 1.0),
            )
        snapshot = replace(geometry.to_render_object(), material=material)
        updated = bool(ensure_object(snapshot))
        if updated:
            self._redraw(update_renderer)
        return updated

    def _renderer_ready(self) -> bool:
        """Return whether renderer-side object updates can be applied."""
        viz = self.visualizer
        return bool(
            getattr(viz, "vis_initialized", False) and getattr(viz, "vis", None) is not None
        )

    def _ensure_scene_render_state(self, entry: Dict[str, Any]) -> RenderObjectState:
        """Normalize one persistent scene entry at its application boundary."""
        index = self._entry_index(getattr(self.visualizer, "mesh_entries", []), entry)
        state = ensure_scene_mesh_render_state(entry, index if index >= 0 else None)
        self.refresh_entry_index(entry, entry_type="mesh")
        return state

    def _require_entry_render_state(self, entry: Dict[str, Any]) -> RenderObjectState:
        """Return canonical state for a persistent scene or target entry."""
        if entry.get("entry_type") == "target":
            index = self._entry_index(getattr(self.visualizer, "target_entries", []), entry)
            state = require_target_mesh_render_state(entry, index if index >= 0 else None)
            self.refresh_entry_index(entry, entry_type="target")
            return state
        return self._ensure_scene_render_state(entry)

    def _entry_index(self, entries: Any, entry: Dict[str, Any]) -> int:
        """Return an entry's index in a list-like collection, or -1."""
        return self._entry_registry.index_for_collection(entries, entry)

    def _entry_material_payload(self, entry: Dict[str, Any]) -> MaterialPayload:
        """Build a renderer-neutral material payload for an entry."""
        pbr_service = getattr(self.visualizer, "material_pbr_service", None)
        resolve_material = getattr(pbr_service, "resolve_entry_material", None)
        if callable(resolve_material):
            return resolve_material(entry).payload
        props = self._entry_material_props(entry)
        return material_payload_from_mapping(self._entry_material_mapping_from_props(props))

    def _entry_material_props(self, entry: Dict[str, Any]) -> dict[str, Any]:
        """Resolve effective PBR properties for scene and target entries."""
        pbr_service = getattr(self.visualizer, "material_pbr_service", None)
        if pbr_service is not None:
            props = pbr_service.get_effective_entry_properties(entry)
        elif entry.get("entry_type") == "target":
            props = {
                "color": entry.get("color", [0.7, 0.7, 0.7]),
                "roughness": entry.get("pbr_roughness", 0.5),
                "metallic": entry.get("pbr_metallic", 0.0),
                "reflectance": entry.get("pbr_reflectance", 0.5),
                "alpha": entry.get("pbr_alpha", 1.0),
            }
        else:
            props = dict(entry.get("pbr_properties", {}) or {})
            props.setdefault("color", entry.get("color", [0.7, 0.7, 0.7]))
        return dict(props)

    def _entry_material_mode(self, entry: Dict[str, Any]) -> MaterialDisplayMode:
        """Resolve material-id/type overlays without changing object intent."""
        mode_service = getattr(self.visualizer, "material_mode_service", None)
        get_mode = getattr(mode_service, "get_mode", None)
        if not callable(get_mode):
            return MaterialDisplayMode.NORMAL
        mode = MaterialDisplayMode.coerce(get_mode(str(entry.get("material_id") or "Unknown")))
        pbr_service = getattr(self.visualizer, "material_pbr_service", None)
        get_visual_key = getattr(pbr_service, "get_visual_material_key", None)
        visual_key = (
            str(get_visual_key(entry))
            if callable(get_visual_key)
            else str(entry.get("material_type") or "default")
        )
        if mode is MaterialDisplayMode.NORMAL:
            mode = MaterialDisplayMode.coerce(get_mode(visual_key))
        return mode

    def _entry_selected(self, entry: Dict[str, Any], mesh: RenderObjectState) -> bool:
        """Return whether the stable entry ID is present in selection state."""
        selected = getattr(self.visualizer, "selected_objects", set()) or set()
        tokens = {
            str(value)
            for value in (
                entry.get("object_key"),
                entry.get("object_id"),
                entry.get("geometry_name"),
            )
            if value is not None
        } | {
            mesh.id,
        }
        return any(token in selected for token in tokens) or mesh in selected

    def _resolve_entry_appearance_with_material(
        self,
        canonical: Dict[str, Any],
        mesh: RenderObjectState,
        material: MaterialPayload,
        *,
        manual_highlight_override: bool | None = None,
    ) -> ResolvedAppearance:
        """Resolve dynamic gates against an already-resolved base material."""
        payload = mesh.payload
        color_source = (
            payload.color_source
            if isinstance(payload, MeshPayload)
            else SurfaceColorSource.MATERIAL
        )
        entry_type = str(canonical.get("entry_type", "mesh")).lower()
        target_index = canonical.get("node_index")
        if target_index is None and entry_type == "target":
            index = self._entry_index(getattr(self.visualizer, "target_entries", []), canonical)
            target_index = index if index >= 0 else None
        pov_visible = True
        if entry_type == "target" and target_index is not None:
            pov_visible = not is_hidden_for_pov(
                getattr(self.visualizer, "app_state", None),
                "target",
                int(target_index),
            )
        return resolve_appearance(
            AppearanceIntent(
                manual_visible=bool(canonical.get("visible", True)),
                runtime_visible=bool(canonical.get("_runtime_visible", True)),
                frame_visible=bool(canonical.get("_frame_visible", True)),
                pov_visible=pov_visible,
                global_visible=bool(canonical.get("_global_visible", True)),
                manual_highlight=bool(
                    canonical.get("highlighted", False)
                    if manual_highlight_override is None
                    else manual_highlight_override
                ),
                selected=self._entry_selected(canonical, mesh),
                material_mode=self._entry_material_mode(canonical),
                material=material,
                color_source=color_source,
            )
        )

    def resolve_entry_appearance(
        self,
        entry: Dict[str, Any],
        *,
        manual_highlight_override: bool | None = None,
    ) -> ResolvedAppearance:
        """Build and resolve the sole effective appearance snapshot for an entry."""
        canonical = self.resolve_canonical_entry(entry)
        mesh = self._require_entry_render_state(canonical)
        base_material = self._entry_material_payload(canonical)
        return self._resolve_entry_appearance_with_material(
            canonical,
            mesh,
            base_material,
            manual_highlight_override=manual_highlight_override,
        )

    def resolve_entry_runtime_appearance(
        self,
        entry: Dict[str, Any],
    ) -> ResolvedAppearance:
        """Resolve frame/selection/mode gates without re-resolving visual PBR."""
        canonical = self.resolve_canonical_entry(entry)
        mesh = self._require_entry_render_state(canonical)
        return self._resolve_entry_appearance_with_material(
            canonical,
            mesh,
            mesh.material,
        )

    @staticmethod
    def _entry_material_mapping_from_props(props: dict[str, Any]) -> dict[str, Any]:
        """Convert PBR properties into renderer material keyword arguments."""
        return pbr_props_to_kwargs(props.get("color", [0.7, 0.7, 0.7]), props)

    def _effective_entry_render_visibility(
        self,
        entry: Dict[str, Any],
        mesh: Any,
    ) -> bool:
        """Resolve renderer visibility without changing semantic object intent."""
        state_visible = bool(mesh.visible) if isinstance(mesh, RenderObjectState) else True
        if entry.get("entry_type") in {"mesh", "target"}:
            return bool(self.resolve_entry_appearance(entry).visible and state_visible)
        return effective_entry_visibility(entry, state_visible=state_visible)

    def _sync_outline_visibility(
        self,
        entry: Dict[str, Any],
        *,
        effective_visible: bool | None = None,
    ) -> None:
        """Keep scene or target outline visibility aligned with mesh visibility."""
        if entry.get("entry_type") == "target":
            target_service = getattr(self.visualizer, "target_service", None)
            if target_service is not None and (
                getattr(self.visualizer, "target_outlines_enabled", False)
                or entry.get("outline_visible")
            ):
                sync_entry = getattr(target_service, "sync_target_entry_edge_visibility", None)
                if callable(sync_entry):
                    sync_entry(
                        entry,
                        mesh_visible=(
                            bool(entry.get("visible", True))
                            if effective_visible is None
                            else bool(effective_visible)
                        ),
                        update_renderer=False,
                    )
                else:
                    target_service.set_target_edge_visibility(
                        getattr(self.visualizer, "target_outlines_enabled", False)
                    )
            return

        scene_appearance = getattr(self.visualizer, "scene_appearance_service", None)
        if scene_appearance is None:
            return
        if (
            getattr(self.visualizer, "outlines_enabled", False)
            or entry.get("outline_visible")
            or entry.get("outline_geometry") is not None
        ):
            scene_appearance.sync_entry_outline_visibility(entry)

    def _sync_scene_outlines(self, entry: Dict[str, Any]) -> None:
        """Delegate scene-outline refresh after merged-scene visibility changes."""
        scene_appearance = getattr(self.visualizer, "scene_appearance_service", None)
        if getattr(self.visualizer, "outlines_enabled", False) and scene_appearance is not None:
            scene_appearance.set_edge_visibility(
                getattr(self.visualizer, "outlines_enabled", False)
            )
        else:
            self._sync_outline_visibility(entry)

    def _redraw(self, update_renderer: bool) -> None:
        """Request a renderer redraw unless the caller batched the update."""
        if not update_renderer:
            return
        renderer = self.renderer
        if renderer is None:
            return
        request_redraw = getattr(renderer, "request_redraw", None)
        if callable(request_redraw):
            request_redraw()
            return
        update = getattr(renderer, "update_renderer", None)
        if callable(update):
            update()
