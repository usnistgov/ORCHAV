"""Manage visualizer TX/RX node markers, labels, and target-service delegation.

This service owns communication-node scene entries and renderer-neutral marker
payloads. Target mesh loading lives in ``TargetService``, but ``NodeService``
keeps the shared object-panel state, node selections, labels, and orientation
frame visibility synchronized with the active renderer.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QLabel

from shared.frames import StandardMPCFrame
from shared.logging import get_logger

from ..io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    visual_frame_read_request,
)
from ..materials.appearance import ResolvedAppearance
from ..model import (
    RenderObjectState,
    Transform,
    VisualEntity,
    render_state_center,
    render_state_points,
    tint_render_state_payload,
)
from ..scene.geometry_helpers import create_text_label, resolve_node_label
from ..scene.geometry_payload_factory import (
    load_mesh_payload,
    make_box_payload,
    make_sphere_payload,
)
from ..scene.orientation_helpers import (
    create_orientation_frames as helpers_create_orientation_frames,
)
from ..scene.orientation_helpers import (
    create_orientation_transform,
)
from ..scene.orientation_helpers import (
    sync_orientation_frame_visibility as helpers_sync_orientation_frame_visibility,
)
from ..scene.orientation_helpers import (
    update_rx_orientation_frames as helpers_update_rx_orientation_frames,
)
from ..scene.orientation_helpers import (
    update_target_orientation_frames as helpers_update_target_orientation_frames,
)
from ..scene.orientation_helpers import (
    update_tx_orientation_frames as helpers_update_tx_orientation_frames,
)
from ..scene.target_runtime import target_label_visible, target_runtime_visible
from ..scene.target_transforms import target_entry_anchor_position
from ..services.base import BaseService
from ..services.entity_render_service import EntityRenderService
from ..services.node_render_sync import NodeRenderSync
from ..services.object_identity import (
    ensure_node_entry_identity,
    ensure_target_entry_identity,
    make_node_entity_id,
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from ..services.pov_visibility_service import is_hidden_for_pov
from ..services.target_service import TargetService
from ..types.render_payloads import (
    MaterialPayload,
    MeshPayload,
)

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.node_service")


class NodeService(BaseService):
    """Manage TX/RX communication nodes and target discovery.

    TX/RX nodes are visual entities like targets. The node service chooses a
    marker payload (sphere by default, box or custom mesh when configured),
    applies the current node-coloring strategy, and lets EntityRenderService
    sync the marker plus label into the active renderer.
    """

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        target_service: TargetService | None = None,
    ) -> None:
        """Create node rendering helpers with the application target owner."""
        super().__init__()
        self.visualizer = visualizer
        self._entity_render_service = EntityRenderService(visualizer)
        self._node_render_sync = NodeRenderSync(visualizer, self._entity_render_service)
        composed_target_service = target_service
        if composed_target_service is None:
            composed_target_service = vars(visualizer).get("target_service")
        if not isinstance(composed_target_service, TargetService):
            raise ValueError("NodeService requires the application-composed TargetService owner")
        self.target_service = composed_target_service
        self._node_marker_payload_cache: dict[tuple[Any, ...], MeshPayload] = {}
        self._pending_node_entity_ids: set[str] = set()
        self._pending_node_removals: set[tuple[str, int]] = set()

    def _selected_tx(self):
        """Return authoritative TX selection from AppState."""
        return getattr(self.visualizer.app_state, "selected_tx", "all")

    def _selected_rx(self):
        """Return authoritative RX selection from AppState."""
        return getattr(self.visualizer.app_state, "selected_rx", "all")

    def _show_labels(self) -> bool:
        """Return authoritative node-label visibility from AppState."""
        return bool(getattr(self.visualizer.app_state, "show_labels", True))

    def _node_label_text(self, kind: str, index: int) -> str:
        """Return the current display label for a TX/RX node."""
        viz = self.visualizer
        app_state = getattr(viz, "app_state", None)
        is_tx = str(kind).lower() == "tx"
        role = "TX" if is_tx else "RX"
        if app_state is None:
            return resolve_node_label(role, index)
        custom = getattr(app_state, "tx_labels" if is_tx else "rx_labels", ())
        device_names = getattr(app_state, "tx_device_names" if is_tx else "rx_device_names", ())
        return resolve_node_label(
            role,
            index,
            custom,
            label_mode=getattr(app_state, "node_label_mode", "role"),
            device_names=device_names,
        )

    def _labels_use_screen_space(self) -> bool:
        """Return the application-owned text sizing policy."""
        app_state = getattr(self.visualizer, "app_state", None)
        return bool(getattr(app_state, "label_screen_space", True))

    def _create_node_label_state(
        self,
        kind: str,
        index: int,
        text: str,
        color: Any,
        font_size: float,
    ) -> RenderObjectState:
        """Create one stable TX/RX label render object."""
        kind_norm = str(kind).lower()
        return create_text_label(
            make_node_geometry_name(kind_norm, index, "label"),
            text,
            color,
            font_size,
            screen_space=self._labels_use_screen_space(),
        )

    def _node_label_enabled(self, kind: str, index: int) -> bool:
        """Return per-node label toggle from TX/RX entry state."""
        viz = self.visualizer
        entries = viz.tx_entries if str(kind).lower() == "tx" else viz.rx_entries
        if 0 <= int(index) < len(entries):
            return bool(entries[int(index)].get("show_label", True))
        return True

    def _node_entry_visible(self, kind: str, index: int) -> bool:
        """Return the application-owned visibility intent for one node."""
        viz = self.visualizer
        entries = viz.tx_entries if str(kind).lower() == "tx" else viz.rx_entries
        if 0 <= int(index) < len(entries):
            return bool(entries[int(index)].get("visible", True))
        return True

    def _set_tx_rx_selection(self, selected_tx, selected_rx) -> None:
        """Update the authoritative TX/RX selection snapshot."""
        self.visualizer.set_state(selected_tx=selected_tx, selected_rx=selected_rx)

    def set_node_label_visibility(self, entry: Dict[str, Any], visible: bool) -> None:
        """Update per-node TX/RX label visibility from object panel toggles."""
        viz = self.visualizer
        entry_type = entry.get("entry_type")
        if entry_type not in {"tx", "rx"}:
            return
        node_index = entry.get("node_index")
        if node_index is None:
            return
        node_index = int(node_index)
        entries = viz.tx_entries if entry_type == "tx" else viz.rx_entries
        if 0 <= node_index < len(entries):
            canonical = entries[node_index]
            canonical["show_label"] = bool(visible)
            entry["show_label"] = bool(visible)
        self._update_tx_rx_visibility()

    def _node_default_color(self, kind: str) -> list[float]:
        """Return the conventional RGB color for a TX/RX node."""
        return [1.0, 0.0, 0.0] if str(kind).lower() == "tx" else [0.0, 0.0, 1.0]

    def _node_marker_size(self, kind: str, override: Optional[float] = None) -> float:
        """Return the configured marker size for a TX/RX node."""
        if override is not None:
            try:
                candidate = float(override)
                if np.isfinite(candidate) and candidate > 0.0:
                    return candidate
            except (TypeError, ValueError):
                pass
        attr = "tx_marker_size" if str(kind).lower() == "tx" else "rx_marker_size"
        try:
            value = float(getattr(self.visualizer, attr, 0.3))
        except (TypeError, ValueError):
            return 0.3
        return value if np.isfinite(value) and value > 0.0 else 0.3

    def _node_marker_config(self, kind: str) -> dict[str, Any]:
        """Return merged node marker config for *kind* from visualizer state."""
        kind_norm = str(kind).lower()
        raw = getattr(self.visualizer, "node_marker_config", None)
        if not isinstance(raw, Mapping):
            return {}
        merged: dict[str, Any] = {}
        default_config = raw.get("default")
        if isinstance(default_config, Mapping):
            merged.update(default_config)
        kind_config = raw.get(kind_norm)
        if isinstance(kind_config, Mapping):
            merged.update(kind_config)
        return merged

    def _node_marker_setting(self, kind: str, key: str, default: Any = None) -> Any:
        """Read a marker setting from config or explicit visualizer attributes."""
        config = self._node_marker_config(kind)
        if key in config:
            return config[key]
        kind_norm = str(kind).lower()
        attrs = getattr(self.visualizer, "__dict__", {})
        for attr in (f"{kind_norm}_marker_{key}", f"node_marker_{key}"):
            if isinstance(attrs, Mapping) and attr in attrs:
                return attrs[attr]
        return default

    def _node_marker_shape(self, kind: str) -> str:
        """Return normalized marker shape: sphere, box, or mesh."""
        shape = self._node_marker_setting(kind, "shape")
        if shape is None:
            shape = self._node_marker_setting(kind, "type")
        if shape is None and self._node_marker_mesh_path(kind) is not None:
            return "mesh"
        if not isinstance(shape, (str, Path)):
            return "sphere"
        normalized = str(shape).strip().lower()
        if normalized in {"", "default"}:
            return "sphere"
        if normalized in {"cube"}:
            return "box"
        if normalized in {"custom", "file", "path"}:
            return "mesh"
        if normalized in {"sphere", "box", "mesh"}:
            return normalized
        logger.warning("Unknown %s marker shape '%s'; using sphere", kind, shape)
        return "sphere"

    def _node_marker_mesh_path(self, kind: str) -> Optional[Path]:
        """Return the configured custom marker mesh path, if any."""
        raw_path = self._node_marker_setting(kind, "mesh_path")
        if raw_path is None:
            raw_path = self._node_marker_setting(kind, "path")
        if not isinstance(raw_path, (str, Path)):
            return None
        text = str(raw_path).strip()
        if not text:
            return None
        return Path(text)

    def _node_marker_center_enabled(self, kind: str) -> bool:
        """Return whether a marker payload should be centered around the node anchor."""
        value = self._node_marker_setting(kind, "center", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _node_marker_mesh_scale(self, kind: str, size: float) -> float:
        """Return custom mesh scale from marker size and optional asset multiplier."""
        try:
            marker_size = float(size)
        except (TypeError, ValueError):
            marker_size = 0.3
        if not np.isfinite(marker_size) or marker_size <= 0.0:
            marker_size = 0.3

        value = self._node_marker_setting(kind, "scale", 1.0)
        try:
            multiplier = float(value)
        except (TypeError, ValueError):
            return marker_size
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            return marker_size
        return marker_size * multiplier

    @staticmethod
    def _with_marker_vertices(
        payload: MeshPayload,
        vertices: np.ndarray,
        *,
        cache_key: Optional[str],
    ) -> MeshPayload:
        """Return a marker payload copy with replaced vertices and identity."""
        return MeshPayload(
            vertices=np.asarray(vertices, dtype=float),
            triangles=np.asarray(payload.triangles, dtype=np.int32),
            normals=payload.normals,
            vertex_colors=None,
            triangle_uvs=payload.triangle_uvs,
            cache_key=cache_key,
        )

    def _prepare_marker_payload(
        self,
        payload: MeshPayload,
        *,
        scale: float,
        center: bool,
        cache_key: str,
    ) -> MeshPayload:
        """Return a centered/scaled marker payload without mutating the source mesh."""
        vertices = np.asarray(payload.vertices, dtype=float).reshape((-1, 3))
        prepared = np.array(vertices, dtype=float, copy=True)
        finite = prepared[np.all(np.isfinite(prepared), axis=1)]
        if center and finite.size:
            mins = finite.min(axis=0)
            maxs = finite.max(axis=0)
            prepared -= (mins + maxs) * 0.5
        if scale != 1.0:
            prepared *= float(scale)
        return self._with_marker_vertices(payload, prepared, cache_key=cache_key)

    def _node_marker_payload(self, kind: str, *, size: float) -> MeshPayload:
        """Create or reuse the renderer-neutral payload for a TX/RX marker."""
        kind_norm = str(kind).lower()
        shape = self._node_marker_shape(kind_norm)
        center = self._node_marker_center_enabled(kind_norm)
        if shape == "box":
            side = max(float(size) * 2.0, 1e-6)
            cache_key = ("box", side, center)
            cached = self._node_marker_payload_cache.get(cache_key)
            if cached is None:
                cached = self._prepare_marker_payload(
                    make_box_payload(side, side, side),
                    scale=1.0,
                    center=center,
                    cache_key=f"node_marker:box:{side:g}:{int(center)}",
                )
                self._node_marker_payload_cache[cache_key] = cached
            return cached

        if shape == "mesh":
            mesh_path = self._node_marker_mesh_path(kind_norm)
            if mesh_path is not None:
                scale = self._node_marker_mesh_scale(kind_norm, size)
                try:
                    stat = mesh_path.stat()
                    cache_key = (
                        "mesh",
                        str(mesh_path.resolve()),
                        stat.st_mtime_ns,
                        stat.st_size,
                        scale,
                        center,
                    )
                    cached = self._node_marker_payload_cache.get(cache_key)
                    if cached is None:
                        loaded = load_mesh_payload(mesh_path)
                        cached = self._prepare_marker_payload(
                            loaded,
                            scale=scale,
                            center=center,
                            cache_key=(
                                f"node_marker:mesh:{mesh_path.resolve()}:"
                                f"{stat.st_mtime_ns}:{stat.st_size}:{scale:g}:{int(center)}"
                            ),
                        )
                        self._node_marker_payload_cache[cache_key] = cached
                    return cached
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Failed to load %s marker mesh '%s'; using sphere: %s",
                        kind_norm,
                        mesh_path,
                        exc,
                    )

        radius = max(float(size), 1e-6)
        cache_key = ("sphere", radius)
        cached = self._node_marker_payload_cache.get(cache_key)
        if cached is None:
            cached = make_sphere_payload(radius=radius, color=None)
            cached = self._with_marker_vertices(
                cached,
                cached.vertices,
                cache_key=f"node_marker:sphere:{radius:g}",
            )
            self._node_marker_payload_cache[cache_key] = cached
        return cached

    def _node_marker_handle(
        self,
        kind: str,
        index: int,
        *,
        size: Optional[float] = None,
        color: Optional[list[float]] = None,
        visible: bool = True,
        position: Optional[Any] = None,
    ) -> RenderObjectState:
        """Create a neutral render handle for one TX/RX marker."""
        kind_norm = str(kind).lower()
        rgb = color if color is not None else self._node_default_color(kind_norm)
        marker_size = self._node_marker_size(kind_norm, override=size)
        payload = self._node_marker_payload(kind_norm, size=marker_size)
        transform = Transform.identity()
        if position is not None:
            transform = self._node_marker_transform(kind_norm, index, position)
        shape = self._node_marker_shape(kind_norm)
        return RenderObjectState(
            id=make_node_geometry_name(kind_norm, index, "marker"),
            payload=payload,
            material=MaterialPayload(
                base_color=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
                roughness=0.5,
            ),
            world_transform=transform,
            visible=bool(visible),
            metadata={
                "type": "node_marker",
                "kind": kind_norm,
                "index": int(index),
                "shape": shape,
                "size": marker_size,
            },
        )

    def _node_marker_entity(
        self,
        kind: str,
        index: int,
        marker: RenderObjectState,
        *,
        label: Any = None,
        anchor: Optional[np.ndarray] = None,
        visible: bool = True,
        label_visible: bool = False,
        label_offset: Optional[np.ndarray] = None,
    ) -> VisualEntity:
        """Build the renderer-neutral visual entity for one TX/RX marker."""
        kind_norm = str(kind).lower()
        if anchor is not None:
            marker.world_transform = self._node_marker_transform(kind_norm, index, anchor)
        marker_visible = bool(marker.visible and visible)
        label_render_object = None
        if isinstance(label, RenderObjectState):
            label_position = np.zeros(3, dtype=float)
            if anchor is not None:
                label_position = np.asarray(anchor, dtype=float).reshape(-1)[:3]
                label.metadata["layout_anchor"] = tuple(float(value) for value in label_position)
            if label_offset is not None:
                offset_values = np.asarray(label_offset, dtype=float).reshape(-1)[:3]
                label.metadata["layout_offset"] = tuple(float(value) for value in offset_values)
                label_position = label_position + offset_values
            label.world_transform = Transform.from_translation(label_position)
            label_render_object = label.to_render_object(
                effective_visible=bool(label.visible and label_visible)
            )
        return VisualEntity(
            entity_id=make_node_entity_id(kind_norm, int(index)),
            category=kind_norm,
            render_object=marker.to_render_object(effective_visible=marker_visible),
            display_name=self._node_label_text(kind_norm, int(index)),
            label_render_object=label_render_object,
            metadata={
                "kind": kind_norm,
                "index": int(index),
                "marker": marker.metadata.get("shape", "sphere"),
            },
        )

    def _node_entity_renderer_available(self) -> bool:
        """Return True when the renderer supports declarative entity sync."""
        return self._node_render_sync.entity_renderer_available()

    def _node_entity_renderer_ready(self) -> bool:
        """Return True when the active renderer can accept entity sync calls."""
        return self._node_render_sync.entity_renderer_ready()

    def _remove_node_marker_entity(self, kind: str, index: int) -> bool:
        """Remove one TX/RX marker entity by stable render IDs."""
        key = (str(kind).lower(), int(index))
        removed = self._node_render_sync.remove_marker_entity(*key)
        if removed:
            self._pending_node_removals.discard(key)
        else:
            self._pending_node_removals.add(key)
        return removed

    def _retry_pending_node_removals(self) -> bool:
        """Retry node removals left incomplete by an earlier backend attempt."""
        all_removed = True
        for kind, index in tuple(self._pending_node_removals):
            if self._node_render_sync.remove_marker_entity(kind, index):
                self._pending_node_removals.discard((kind, index))
            else:
                all_removed = False
        return all_removed

    def _ensure_node_marker_handle(
        self,
        kind: str,
        index: int,
        marker: Any,
        anchor: Optional[np.ndarray],
    ) -> RenderObjectState:
        """Return a neutral marker handle for app state."""
        kind_norm = str(kind).lower()
        if isinstance(marker, RenderObjectState):
            return marker

        logger.warning(
            "Replacing non-render-handle %s node marker %d with a neutral marker handle",
            kind_norm,
            int(index),
        )
        color = self._node_default_color(kind_norm)
        position = anchor
        if position is None:
            position = np.zeros(3, dtype=float)
        return self._node_marker_handle(
            kind_norm,
            index,
            size=self._node_marker_size(kind_norm),
            color=color,
            # Missing anchors, selection filters, and POV masking are runtime
            # policy. A newly normalized node remains semantically enabled.
            visible=True,
            position=position,
        )

    def _sync_tx_rx_visual_entities(self) -> bool:
        """Sync all TX/RX marker entities from current state into the renderer."""
        viz = self.visualizer
        if not self._node_entity_renderer_ready():
            return False
        removals_ok = self._retry_pending_node_removals()
        label_offset = np.array(
            [
                float(getattr(viz, "label_offset_x", 0.0)),
                float(getattr(viz, "label_offset_y", 0.0)),
                float(getattr(viz, "label_offset_z", 0.5)),
            ],
            dtype=float,
        )
        show_labels = self._show_labels()
        entities: list[VisualEntity] = []

        for kind, markers, labels in (
            ("tx", getattr(viz, "tx_markers", []), getattr(viz, "tx_labels", [])),
            ("rx", getattr(viz, "rx_markers", []), getattr(viz, "rx_labels", [])),
        ):
            for index, marker in enumerate(markers):
                anchor = self._resolve_node_anchor(kind, index, marker, allow_fallback=False)
                visible = self._node_visible_for_anchor(kind, index, anchor)
                marker = self._ensure_node_marker_handle(kind, index, marker, anchor)
                markers[index] = marker
                if anchor is None:
                    anchor = render_state_center(marker)
                label = labels[index] if index < len(labels) else None
                label_visible = visible and show_labels and self._node_label_enabled(kind, index)
                entities.append(
                    self._node_marker_entity(
                        kind,
                        index,
                        marker,
                        label=label,
                        anchor=np.asarray(anchor, dtype=float),
                        visible=visible,
                        label_visible=label_visible,
                        label_offset=label_offset,
                    )
                )

        results = self._node_render_sync.sync_entities(entities)
        attempted_ids = {entity.entity_id for entity in entities}
        self._pending_node_entity_ids.difference_update(attempted_ids)
        self._pending_node_entity_ids.update(
            entity_id for entity_id, synced in results.items() if not synced
        )
        return removals_ok and not self._pending_node_entity_ids

    def retry_pending_node_syncs(self) -> bool:
        """Retry incomplete node object operations using current semantic state."""
        if not self._pending_node_entity_ids and not self._pending_node_removals:
            return True
        return self._sync_tx_rx_visual_entities()

    def _selected_node_index(self, kind: str) -> Optional[int]:
        """Return selected TX/RX index, or ``None`` when all nodes are active."""
        selected = self._selected_tx() if str(kind).lower() == "tx" else self._selected_rx()
        if selected == "all":
            return None
        try:
            return int(selected)
        except (TypeError, ValueError):
            return None

    def _node_positions(self, kind: str) -> list[Any]:
        """Return current TX/RX positions as a tolerant list for indexing."""
        attr = "current_tx_positions" if str(kind).lower() == "tx" else "current_rx_positions"
        positions = getattr(self.visualizer, attr, None)
        if positions is None:
            return []
        try:
            return list(positions)
        except TypeError:
            return []

    def _node_orientations(self, kind: str) -> list[Any]:
        """Return current TX/RX orientations as a tolerant list for indexing."""
        attr = "current_tx_orientations" if str(kind).lower() == "tx" else "current_rx_orientations"
        orientations = getattr(self.visualizer, attr, None)
        if orientations is None:
            return []
        try:
            return list(orientations)
        except TypeError:
            return []

    def _resolve_node_orientation(self, kind: str, index: int) -> Optional[np.ndarray]:
        """Return the current yaw/pitch/roll tuple for one TX/RX marker."""
        kind_norm = str(kind).lower()
        orientations = self._node_orientations(kind_norm)
        if int(index) < len(orientations):
            raw_orientation = orientations[int(index)]
        else:
            return None

        try:
            values = np.asarray(raw_orientation, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 3 or not np.all(np.isfinite(values[:3])):
            return None
        return values[:3]

    def _node_marker_transform(self, kind: str, index: int, anchor: Any) -> Transform:
        """Return the absolute marker transform from current position and orientation."""
        try:
            position = np.asarray(anchor, dtype=float).reshape(-1)[:3]
        except (TypeError, ValueError):
            return Transform.identity()
        if position.size < 3 or not np.all(np.isfinite(position[:3])):
            return Transform.identity()

        orientation = self._resolve_node_orientation(kind, int(index))
        if orientation is None:
            return Transform.from_translation(position[:3])
        try:
            matrix = create_orientation_transform(
                position[:3],
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
            )
            return Transform(matrix)
        except (TypeError, ValueError):
            logger.debug(
                "Falling back to translation-only transform for %s marker %d",
                kind,
                int(index),
                exc_info=True,
            )
            return Transform.from_translation(position[:3])

    def _resolve_node_anchor(
        self,
        kind: str,
        index: int,
        marker: Any,
        *,
        allow_fallback: bool = True,
    ) -> Optional[np.ndarray]:
        """Return the current world anchor for a TX/RX marker."""
        kind_norm = str(kind).lower()
        positions = self._node_positions(kind_norm)
        if int(index) < len(positions):
            try:
                return np.asarray(positions[int(index)], dtype=float).reshape(-1)[:3]
            except (TypeError, ValueError):
                return None

        if not allow_fallback:
            return None

        if isinstance(marker, RenderObjectState):
            return render_state_center(marker)
        return None

    def _node_visible_for_anchor(
        self,
        kind: str,
        index: int,
        anchor: Optional[np.ndarray],
    ) -> bool:
        """Return whether a TX/RX node should be visible for its current anchor."""
        if anchor is None:
            return False
        selected_idx = self._selected_node_index(kind)
        selected_visible = selected_idx is None or selected_idx == int(index)
        return bool(
            self._node_entry_visible(kind, index)
            and selected_visible
            and not self._is_hidden_for_pov(str(kind).lower(), int(index))
        )

    def _uniform_color(self, geometry: Any, default: Optional[list[float]] = None) -> list[float]:
        """Extract a representative RGB color from renderer-neutral geometry."""
        if isinstance(geometry, RenderObjectState):
            color = np.asarray(geometry.material.base_color, dtype=float).reshape(-1)
            if color.size >= 3:
                return [float(x) for x in np.clip(color[:3], 0.0, 1.0)]
        return list(default) if default is not None else [1.0, 1.0, 1.0]

    @staticmethod
    def _paint_geometry_color(geometry: Any, color: list[float]) -> None:
        """Apply a uniform color to renderer-neutral geometry."""
        if isinstance(geometry, RenderObjectState):
            tint_render_state_payload(geometry, color)

    def _request_redraw(self, *, context: str) -> None:
        """Request presentation after the surrounding semantic batch completes."""
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None:
            return
        request_redraw = getattr(renderer, "request_redraw", None)
        if not callable(request_redraw):
            return
        try:
            request_redraw()
        except (RuntimeError, ValueError) as exc:
            logger.error("Error requesting redraw after %s: %s", context, exc)

    def discover_available_tx_rx(self) -> None:
        """Discover available TX/RX nodes by scanning cached data or files."""
        viz = self.visualizer

        def visual_payload(frame_data: Any) -> Any:
            """Return the mapping consumed by TX/RX discovery."""
            if not isinstance(frame_data, StandardMPCFrame):
                return frame_data
            return standard_frame_to_visual_frame(
                frame_data,
                request=visual_frame_read_request(),
                points_dtype=getattr(viz.mpc_core, "canon_points_dtype", np.float32),
            )

        # Guard: don't try to discover TX/RX until frame source is ready
        if not viz.frame_source:
            logger.debug("No frame source yet; skipping TX/RX discovery")
            return

        first_frame = 0
        if hasattr(viz.frame_source, "list_frames"):
            try:
                available_frames = viz.frame_source.list_frames()
                if available_frames:
                    first_frame = min(available_frames)
            except (OSError, TypeError, ValueError):
                pass

        if not viz.frame_source.has_frame(first_frame):
            logger.debug("Frame %d not available yet; skipping TX/RX discovery", first_frame)
            return

        viz.available_tx = []
        viz.available_rx = []

        # First try to use cached data if available
        data = viz.cache_service.get_frame(first_frame)
        if data is not None:
            data = visual_payload(data)
            logger.debug("Using cached data from frame %d", first_frame)

            available_tx, available_rx = viz.mpc_core.discover_tx_rx(data)

            if available_tx and available_rx:
                viz.available_tx = available_tx
                viz.available_rx = available_rx
                logger.debug(
                    "Found %d TX and %d RX from cached frame %d",
                    len(viz.available_tx),
                    len(viz.available_rx),
                    first_frame,
                )

        # If no cached data or no TX/RX found, fall back to loading from provider
        if not viz.available_tx or not viz.available_rx:
            logger.debug("No cached data available, trying provider")

            try:
                load_frame = getattr(
                    getattr(viz, "animation_service", None), "ensure_step_cached", None
                )
                if callable(load_frame):
                    data = load_frame(first_frame)
                    logger.debug("Using animation service cache prime for frame %d", first_frame)
                else:
                    if not viz.frame_source:
                        logger.warning("No frame source available")
                        return
                    data = viz.frame_source.load_frame(first_frame)
                    data = visual_payload(data)
                    if data is not None and not viz.cache_service.has_frame(first_frame):
                        viz.cache_service.store_frame(first_frame, data)
                    logger.debug("Using frame source")

                data = visual_payload(data)

                if data is None:
                    logger.warning("Failed to load frame %d for TX/RX discovery", first_frame)
                    return

                available_tx, available_rx = viz.mpc_core.discover_tx_rx(data)

                if available_tx and available_rx:
                    viz.available_tx = available_tx
                    viz.available_rx = available_rx
                    logger.debug(
                        "Found %d TX and %d RX from provider frame %d",
                        len(viz.available_tx),
                        len(viz.available_rx),
                        first_frame,
                    )

            except (OSError, KeyError, ValueError) as e:
                logger.error("Error loading frame %d: %s", first_frame, e)

        # If no data found, don't use defaults - wait for actual frame data
        if not viz.available_tx:
            logger.debug("No TX data found - waiting for frame data")
        if not viz.available_rx:
            logger.debug("No RX data found - waiting for frame data")
        viz.progress.note(f"TX nodes: {len(viz.available_tx)}, RX nodes: {len(viz.available_rx)}")

        logger.debug(f"Final - TX: {viz.available_tx}, RX: {viz.available_rx}")

        # Populate dropdowns
        self.populate_tx_rx_selections()
        viz.tx_rx_data_loaded = True

        # Also update the target focus dropdown to include TX/RX options
        if hasattr(viz, "target_focus_dropdown"):
            viz.camera_controller.update_target_focus_dropdown()
            logger.debug("Updated target focus dropdown after TX/RX discovery")

    def populate_tx_rx_selections(self, *, preserve_selection: bool = False) -> None:
        """Populate TX/RX dropdowns with available options."""
        viz = self.visualizer
        if not viz.tx_dropdown or not viz.rx_dropdown:
            logger.warning("Combo boxes not initialized")
            return

        logger.debug(f"Available TX: {viz.available_tx}, RX: {viz.available_rx}")
        prev_tx_selection = self._selected_tx() if preserve_selection else "all"
        prev_rx_selection = self._selected_rx() if preserve_selection else "all"

        with QSignalBlocker(viz.tx_dropdown), QSignalBlocker(viz.rx_dropdown):
            viz.tx_dropdown.clear()
            viz.rx_dropdown.clear()

            viz.tx_dropdown.addItem("All TX")
            viz.rx_dropdown.addItem("All RX")

            if viz.available_tx:
                for tx_idx in viz.available_tx:
                    tx_text = self._node_label_text("tx", tx_idx)
                    viz.tx_dropdown.addItem(tx_text, tx_idx)
                    logger.debug(f"Added TX option: {tx_text} (data index: {tx_idx})")
            else:
                logger.debug("No TX options available yet")

            if viz.available_rx:
                for rx_idx in viz.available_rx:
                    rx_text = self._node_label_text("rx", rx_idx)
                    viz.rx_dropdown.addItem(rx_text, rx_idx)
                    logger.debug(f"Added RX option: {rx_text} (data index: {rx_idx})")
            else:
                logger.debug("No RX options available yet")

            # Restore previous selection if requested and still valid
            def _restore_selection(combo, previous, all_label):
                """Restore a combo selection, falling back to the all-nodes row."""
                if previous == "all" or previous is None:
                    combo.setCurrentText(all_label)
                    return "all"
                idx = combo.findData(previous)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    return previous
                else:
                    combo.setCurrentText(all_label)
                    return "all"

            if preserve_selection:
                selected_tx = _restore_selection(viz.tx_dropdown, prev_tx_selection, "All TX")
                selected_rx = _restore_selection(viz.rx_dropdown, prev_rx_selection, "All RX")
            else:
                viz.tx_dropdown.setCurrentText("All TX")
                viz.rx_dropdown.setCurrentText("All RX")
                selected_tx = "all"
                selected_rx = "all"

            self._set_tx_rx_selection(selected_tx, selected_rx)

        nodes_panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("nodes")
        if nodes_panel and hasattr(nodes_panel, "update_node_rename_visibility"):
            nodes_panel.update_node_rename_visibility()

        logger.debug(
            f"Final - TX combo has {viz.tx_dropdown.count()} items, RX combo has {viz.rx_dropdown.count()} items"
        )

    def update_available_tx_rx_from_frame(
        self, num_tx: Optional[int], num_rx: Optional[int]
    ) -> bool:
        """Refresh available TX/RX lists when live data becomes available."""
        viz = self.visualizer
        try:
            tx_count = int(num_tx) if num_tx is not None else None
        except (TypeError, ValueError):
            tx_count = None
        try:
            rx_count = int(num_rx) if num_rx is not None else None
        except (TypeError, ValueError):
            rx_count = None

        # Fallback to currently cached positions if counts weren't provided
        if tx_count is None and hasattr(viz, "current_tx_positions"):
            tx_positions = getattr(viz, "current_tx_positions", None)
            try:
                tx_count = len(tx_positions) if tx_positions is not None else 0
            except TypeError:
                tx_count = 0
        if rx_count is None and hasattr(viz, "current_rx_positions"):
            rx_positions = getattr(viz, "current_rx_positions", None)
            try:
                rx_count = len(rx_positions) if rx_positions is not None else 0
            except TypeError:
                rx_count = 0

        updates_needed = False
        if tx_count and tx_count > 0:
            if len(viz.available_tx) != tx_count:
                viz.available_tx = list(range(tx_count))
                updates_needed = True
        if rx_count and rx_count > 0:
            if len(viz.available_rx) != rx_count:
                viz.available_rx = list(range(rx_count))
                updates_needed = True

        if updates_needed:
            logger.info(
                "Live data reported %s TX / %s RX devices. Refreshing selectors.",
                tx_count,
                rx_count,
            )
            self.populate_tx_rx_selections(preserve_selection=True)
            # Refresh camera/target dropdowns so TX/RX entries appear there as well
            if hasattr(viz, "camera_controller"):
                viz.camera_controller.update_target_focus_dropdown()
                logger.debug("Target dropdown refreshed after TX/RX update")
        return updates_needed

    def update_tx_rx_positions(self, tx_positions, rx_positions) -> bool:
        """Update cached TX/RX positions and report renderer synchronization."""
        viz = self.visualizer
        tx_count = len(tx_positions) if tx_positions is not None else 0
        rx_count = len(rx_positions) if rx_positions is not None else 0
        logger.debug(
            "update_tx_rx_positions called with %s TX and %s RX positions",
            tx_count,
            rx_count,
        )

        if tx_positions is not None:
            viz.current_tx_positions = tx_positions
            logger.debug(f"Stored {len(tx_positions)} TX positions")

        if rx_positions is not None:
            viz.current_rx_positions = rx_positions
            logger.debug(f"Stored {len(rx_positions)} RX positions")

        nodes_synced = self._update_tx_rx_visibility()
        self._refresh_comm_node_entries()
        return nodes_synced

    def refresh_comm_node_entries(self) -> None:
        """Public wrapper for entry refresh (used by visualizer delegates)."""
        self._refresh_comm_node_entries()

    def update_tx_rx_visibility(self) -> bool:
        """Public wrapper for TX/RX visibility updates."""
        return self._update_tx_rx_visibility()

    def sync_node_visibility_snapshot(self) -> bool:
        """Synchronize node children without presenting an intermediate state.

        Cross-domain transactions such as an object-panel group toggle call
        this once while their outer renderer batch is active. Markers, labels,
        and orientation frames then inherit the same final node visibility.
        """
        if not self._node_entity_renderer_ready():
            logger.debug("Cannot sync node visibility: renderer not initialized")
            return False
        with self._renderer_update_batch():
            nodes_synced = self._sync_tx_rx_visual_entities()
            orientations_synced = self.update_orientation_visibility()
            return bool(nodes_synced and orientations_synced)

    def update_label_visibility(self) -> None:
        """Sync label visibility with current state."""
        self._update_tx_rx_visibility()

    def sync_pov_entity_visibility(
        self,
        entity_refs: Iterable[tuple[str, int]],
    ) -> bool:
        """Republish complete entity snapshots after a POV visibility transition.

        ``pov_hidden_node`` is the sole POV visibility state. This method owns
        the semantic re-evaluation that follows a transition: TX/RX markers
        and labels are synchronized together, targets delegate their mesh,
        label, and outline snapshot to :class:`TargetService`, and orientation
        frames are refreshed from the same policy.
        """
        normalized_refs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for raw_type, raw_index in entity_refs:
            entity_type = str(raw_type).lower()
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            entity_ref = (entity_type, index)
            if entity_type not in {"tx", "rx", "target"} or index < 0 or entity_ref in seen:
                continue
            seen.add(entity_ref)
            normalized_refs.append(entity_ref)

        if not normalized_refs:
            return True

        # A POV transition can touch markers, labels, target meshes/outlines,
        # and orientation frames. Publish that complete semantic change as one
        # renderer transaction so no intermediate visibility mix is presented.
        with self._renderer_update_batch():
            all_synced = True
            if any(entity_type in {"tx", "rx"} for entity_type, _ in normalized_refs):
                all_synced = self._sync_tx_rx_visual_entities() and all_synced

            target_entries = getattr(self.visualizer, "target_entries", [])
            for entity_type, index in normalized_refs:
                if entity_type != "target":
                    continue
                if not 0 <= index < len(target_entries):
                    logger.debug("Cannot sync POV target %d: entry is unavailable", index)
                    all_synced = False
                    continue
                all_synced = (
                    self.target_service.sync_target_entry_snapshot(target_entries[index])
                    and all_synced
                )

            all_synced = self.update_orientation_visibility() and all_synced
            self._request_redraw(
                context="POV entity visibility transition",
            )
        return all_synced

    def _set_label_world_position(self, label: Any, position: Any) -> None:
        """Move a label geometry to an absolute world position.

        Recreated labels are often positioned before they are re-registered
        with the renderer. Text labels may not have a named scene object yet,
        so renderer-owned label placement is delegated to NodeRenderSync.
        """
        self._node_render_sync.set_label_world_position(label, position)

    def recreate_tx_rx_labels(self, font_size: float) -> None:
        """Recreate TX/RX labels with new font size.

        Args:
            font_size: New font size scale factor
        """
        viz = self.visualizer
        label_offset = np.array(
            [
                float(getattr(viz, "label_offset_x", 1.5)),
                float(getattr(viz, "label_offset_y", 0.0)),
                float(getattr(viz, "label_offset_z", 1.0)),
            ],
            dtype=float,
        )

        with self._renderer_update_batch():
            # Store current positions before replacing the label snapshots.
            tx_positions = [render_state_center(marker) for marker in viz.tx_markers]
            rx_positions = [render_state_center(marker) for marker in viz.rx_markers]

            for i in range(len(viz.tx_labels)):
                self._remove_node_marker_entity("tx", i)
            for i in range(len(viz.rx_labels)):
                self._remove_node_marker_entity("rx", i)

            viz.tx_labels.clear()
            viz.rx_labels.clear()

            for i in range(len(viz.tx_markers)):
                text = self._node_label_text("tx", i)
                label = self._create_node_label_state("tx", i, text, [1.0, 0.0, 0.0], font_size)
                if i < len(tx_positions):
                    self._set_label_world_position(
                        label,
                        np.asarray(tx_positions[i], dtype=float) + label_offset,
                    )
                viz.tx_labels.append(label)

            for i in range(len(viz.rx_markers)):
                text = self._node_label_text("rx", i)
                label = self._create_node_label_state("rx", i, text, [0.0, 0.0, 1.0], font_size)
                if i < len(rx_positions):
                    self._set_label_world_position(
                        label,
                        np.asarray(rx_positions[i], dtype=float) + label_offset,
                    )
                viz.rx_labels.append(label)

            # Re-run the normal visibility path so labels are re-registered
            # under stable names and inherit all visibility policies.
            self._update_tx_rx_visibility()
            self._request_redraw(context="TX/RX label recreation")

        logger.debug("Recreated TX/RX labels with font size %s", font_size)

    def recreate_target_labels(self, font_size: float) -> None:
        """Recreate target labels with new font size or custom names.

        Args:
            font_size: Font size scale factor.
        """
        viz = self.visualizer

        with self._renderer_update_batch():
            # Remove existing target labels by stable render ID. The list is
            # intentionally sparse when an entry has no renderable mesh.
            for label in viz.target_labels:
                if isinstance(label, RenderObjectState):
                    self._node_render_sync.remove_label(label.id)

            viz.target_labels.clear()

            custom_labels = getattr(getattr(viz, "app_state", None), "target_labels", ())
            offset_x = getattr(viz, "label_offset_x", 1.5)
            offset_y = getattr(viz, "label_offset_y", 0.0)
            offset_z = getattr(viz, "label_offset_z", 1.0)

            for i, target_entry in enumerate(viz.target_entries):
                if "mesh" not in target_entry or target_entry["mesh"] is None:
                    continue

                default_name = target_entry.get("name", f"Target{i + 1}")
                if custom_labels and i < len(custom_labels) and custom_labels[i]:
                    text = custom_labels[i]
                else:
                    text = default_name

                label_color = target_entry.get("color", [0.8, 0.8, 0.8])
                ensure_target_entry_identity(target_entry, i)
                label = create_text_label(
                    make_target_entry_geometry_name(target_entry, "label"),
                    text,
                    label_color,
                    font_size,
                    screen_space=self._labels_use_screen_space(),
                )

                pos = target_entry.get("position")
                if pos is not None:
                    self._set_label_world_position(
                        label,
                        [pos[0] + offset_x, pos[1] + offset_y, pos[2] + offset_z],
                    )
                else:
                    center = target_entry_anchor_position(target_entry)
                    if center is None:
                        continue
                    self._set_label_world_position(
                        label,
                        [
                            center[0] + offset_x,
                            center[1] + offset_y,
                            center[2] + offset_z,
                        ],
                    )

                viz.target_labels.append(label)

            self.update_target_label_visibility()
            self._request_redraw(context="target label recreation")
        logger.debug(
            "Recreated %d target labels with font size %s", len(viz.target_labels), font_size
        )

    def update_target_label_visibility(self) -> None:
        """Sync target label visibility with current state."""
        viz = self.visualizer
        if not getattr(viz, "vis_initialized", False):
            return

        show = getattr(getattr(viz, "app_state", None), "show_target_labels", True)
        labels_by_id = {
            label.id: label for label in viz.target_labels if isinstance(label, RenderObjectState)
        }
        with self._renderer_update_batch():
            for i, target_entry in enumerate(viz.target_entries):
                ensure_target_entry_identity(target_entry, i)
                label_name = make_target_entry_geometry_name(target_entry, "label")
                label = labels_by_id.get(label_name)
                if label is None:
                    continue
                visible = target_label_visible(
                    target_entry,
                    i,
                    self._is_hidden_for_pov,
                    show_target_labels=show,
                )
                self._node_render_sync.sync_label(
                    label_id=label_name,
                    label=label,
                    visible=visible,
                )
            self._request_redraw(
                context="target label visibility change",
            )

    def update_orientation_visibility(self) -> bool:
        """Update the visibility of TX/RX/Target orientation frames."""
        viz = self.visualizer
        if not getattr(viz, "vis_initialized", False) or viz.vis is None:
            return False

        def _do_orientation_updates() -> bool:
            """Inner function to perform orientation updates (can be batched)."""
            all_synced = True
            for i, frame in enumerate(getattr(viz, "tx_orientation_frames", [])):
                if frame is not None:
                    should_show = self.orientation_frame_visible("tx", i)
                    all_synced = (
                        helpers_sync_orientation_frame_visibility(viz, frame, should_show)
                        and all_synced
                    )
                    logger.debug(
                        "TX%s orientation frame %s",
                        i + 1,
                        "made visible" if should_show else "hidden",
                    )

            for i, frame in enumerate(getattr(viz, "rx_orientation_frames", [])):
                if frame is not None:
                    should_show = self.orientation_frame_visible("rx", i)
                    all_synced = (
                        helpers_sync_orientation_frame_visibility(viz, frame, should_show)
                        and all_synced
                    )
                    logger.debug(
                        "RX%s orientation frame %s",
                        i + 1,
                        "made visible" if should_show else "hidden",
                    )

            logger.debug(
                "show_target_orientation=%s, frames=%s, targets=%s",
                getattr(viz, "show_target_orientation", False),
                len(getattr(viz, "target_orientation_frames", [])),
                len(getattr(viz, "target_entries", [])),
            )
            for i, frame in enumerate(getattr(viz, "target_orientation_frames", [])):
                if frame is not None:
                    should_show = self.orientation_frame_visible("target", i)
                    all_synced = (
                        helpers_sync_orientation_frame_visibility(viz, frame, should_show)
                        and all_synced
                    )
                    logger.debug(
                        "Target%s orientation frame %s",
                        i + 1,
                        "made visible" if should_show else "hidden",
                    )
            return all_synced

        try:
            with self._renderer_update_batch():
                return _do_orientation_updates()
        except (RuntimeError, ValueError):
            logger.exception("Error updating orientation visibility")
            return False

    def _renderer_update_batch(self):
        """Group one semantic multi-object change into one presentation."""
        renderer = getattr(self.visualizer, "renderer", None)
        batch_updates = getattr(renderer, "batch_updates", None)
        if callable(batch_updates):
            return batch_updates()
        return nullcontext()

    def orientation_frame_visible(self, node_type: str, index: int) -> bool:
        """Return effective orientation visibility for frame and UI synchronization."""
        viz = self.visualizer
        kind = str(node_type).lower()
        if kind == "tx":
            selected = self._selected_tx()
            enabled = bool(getattr(viz, "show_tx_orientation", False))
            selected_visible = selected == "all" or index == selected
        elif kind == "rx":
            selected = self._selected_rx()
            enabled = bool(getattr(viz, "show_rx_orientation", False))
            selected_visible = selected == "all" or index == selected
        elif kind == "target":
            enabled = bool(getattr(viz, "show_target_orientation", False))
            entries = getattr(viz, "target_entries", [])
            if not 0 <= int(index) < len(entries):
                return False
            appearance = getattr(viz, "object_appearance_service", None)
            resolve = getattr(appearance, "resolve_entry_runtime_appearance", None)
            if callable(resolve):
                resolved = resolve(entries[int(index)])
                if isinstance(resolved, ResolvedAppearance):
                    return bool(enabled and resolved.visible)
            return bool(
                enabled
                and target_runtime_visible(
                    entries[int(index)],
                    int(index),
                    self._is_hidden_for_pov,
                )
            )
        else:
            return False
        return bool(
            enabled
            and selected_visible
            and self._node_entry_visible(kind, int(index))
            and not self._is_hidden_for_pov(kind, int(index))
        )

    def create_orientation_frames(self, step: int) -> bool:
        """Create or update orientation frames and report complete synchronization."""
        with self._renderer_update_batch():
            return helpers_create_orientation_frames(self.visualizer, step)

    def update_tx_orientation_frames(self, tx_orientations) -> bool:
        """Update TX orientation coordinate frames."""
        with self._renderer_update_batch():
            return helpers_update_tx_orientation_frames(self.visualizer, tx_orientations)

    def update_rx_orientation_frames(self, rx_orientations) -> bool:
        """Update RX orientation coordinate frames."""
        with self._renderer_update_batch():
            return helpers_update_rx_orientation_frames(self.visualizer, rx_orientations)

    def update_target_orientation_frames(self, target_orientations) -> bool:
        """Update target orientation coordinate frames."""
        with self._renderer_update_batch():
            return helpers_update_target_orientation_frames(
                self.visualizer,
                target_orientations,
            )

    def apply_node_coloring(self) -> None:
        """Apply node coloring based on the current mode."""
        viz = self.visualizer
        if not hasattr(viz, "tx_markers") or not hasattr(viz, "rx_markers"):
            return

        if viz.node_coloring_mode == "per_type":
            tx_color = [1.0, 0.0, 0.0]  # Red for TX
            rx_color = [0.0, 0.0, 1.0]  # Blue for RX (consistent with marker creation)
            for i, marker in enumerate(viz.tx_markers):
                self._paint_geometry_color(marker, tx_color)
                if i < len(viz.tx_labels):
                    self._node_render_sync.set_label_color(viz.tx_labels[i], tx_color)
            for i, marker in enumerate(viz.rx_markers):
                self._paint_geometry_color(marker, rx_color)
                if i < len(viz.rx_labels):
                    self._node_render_sync.set_label_color(viz.rx_labels[i], rx_color)
        else:
            import colorsys

            num_nodes = len(viz.tx_markers) + len(viz.rx_markers)
            colors = []
            for i in range(max(num_nodes, 1)):
                hue = i / num_nodes if num_nodes else 0
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                colors.append(list(rgb))
            viz.individual_node_colors = colors
            color_idx = 0
            for i, marker in enumerate(viz.tx_markers):
                self._paint_geometry_color(marker, colors[color_idx])
                if i < len(viz.tx_labels):
                    self._node_render_sync.set_label_color(viz.tx_labels[i], colors[color_idx])
                color_idx += 1
            for i, marker in enumerate(viz.rx_markers):
                self._paint_geometry_color(marker, colors[color_idx])
                if i < len(viz.rx_labels):
                    self._node_render_sync.set_label_color(viz.rx_labels[i], colors[color_idx])
                color_idx += 1

        if viz.vis_initialized:
            try:
                if self._node_entity_renderer_available():
                    self._sync_tx_rx_visual_entities()
                else:
                    logger.debug("Renderer does not expose entity sync for node coloring")
                viz.renderer.update_renderer()
                logger.debug("Node coloring update applied: %s", viz.node_coloring_mode)
            except (RuntimeError, ValueError) as exc:
                logger.error(f"Error updating renderer: {exc}")

    def update_node_coloring_legend(self) -> None:
        """Update the legend to reflect the current node coloring mode."""
        viz = self.visualizer
        tx_label = getattr(viz, "tx_legend_label", None)
        rx_label = getattr(viz, "rx_legend_label", None)
        legend_layout = getattr(viz, "tx_rx_legend_layout", None)
        if tx_label is None or rx_label is None or legend_layout is None:
            return

        if viz.node_coloring_mode == "per_type":
            tx_label.show()
            rx_label.show()

            tx_label.setText("Transmitters (TX1, TX2, ...)")
            tx_label.setStyleSheet("color: red; font-size: 10px; margin: 0px; padding: 0px;")

            rx_label.setText("Receivers (RX1, RX2, ...)")
            rx_label.setStyleSheet("color: blue; font-size: 10px; margin: 0px; padding: 0px;")

            for i in reversed(range(legend_layout.count())):
                child = legend_layout.itemAt(i).widget()
                if child and child not in (tx_label, rx_label):
                    child.deleteLater()
        else:
            tx_label.hide()
            rx_label.hide()

            for i in reversed(range(legend_layout.count())):
                child = legend_layout.itemAt(i).widget()
                if child and child not in (tx_label, rx_label):
                    child.deleteLater()

            for i, _marker in enumerate(viz.tx_markers):
                if i < len(viz.tx_markers) and hasattr(viz, "individual_node_colors"):
                    color = viz.individual_node_colors[i]
                    color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    )
                    text = self._node_label_text("tx", i)
                    node_label = QLabel(text)
                    node_label.setStyleSheet(
                        f"color: {color_hex}; font-size: 10px; margin-left: 10px;"
                    )
                    legend_layout.addWidget(node_label)

            for i, _marker in enumerate(viz.rx_markers):
                if i < len(viz.rx_markers) and hasattr(viz, "individual_node_colors"):
                    color_idx = len(viz.tx_markers) + i
                    if color_idx < len(viz.individual_node_colors):
                        color = viz.individual_node_colors[color_idx]
                        color_hex = "#{:02x}{:02x}{:02x}".format(
                            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                        )
                        text = self._node_label_text("rx", i)
                        node_label = QLabel(text)
                        node_label.setStyleSheet(
                            f"color: {color_hex}; font-size: 10px; margin-left: 10px;"
                        )
                        legend_layout.addWidget(node_label)

    def _refresh_comm_node_entries(self) -> None:
        """Maintain cached TX/RX entries for the object panel."""
        viz = self.visualizer

        def _to_list(arr):
            """Convert numpy-like arrays to plain lists without changing values."""
            if hasattr(arr, "tolist"):
                return arr.tolist()
            return list(arr)

        tx_positions = (
            _to_list(viz.current_tx_positions) if viz.current_tx_positions is not None else []
        )
        rx_positions = (
            _to_list(viz.current_rx_positions) if viz.current_rx_positions is not None else []
        )
        tx_orientations_raw = getattr(viz, "current_tx_orientations", None)
        rx_orientations_raw = getattr(viz, "current_rx_orientations", None)
        tx_orientations = _to_list(tx_orientations_raw) if tx_orientations_raw is not None else []
        rx_orientations = _to_list(rx_orientations_raw) if rx_orientations_raw is not None else []

        def _inventory_count(kind: str, existing: list[dict[str, Any]]) -> int:
            """Return stable node cardinality independent of one frame's anchors."""
            try:
                available = list(getattr(viz, f"available_{kind}", []) or [])
            except TypeError:
                available = []
            if available:
                try:
                    return max(int(index) for index in available) + 1
                except (TypeError, ValueError):
                    return len(available)
            try:
                markers = list(getattr(viz, f"{kind}_markers", []) or [])
            except TypeError:
                markers = []
            if markers:
                return len(markers)
            return len(existing)

        def _sync_entries(
            kind: str,
            positions: list[Any],
            orientations: list[Any],
            existing: List[Dict[str, Any]],
        ) -> bool:
            """Update dynamic node fields without replacing semantic entry intent."""
            count = max(
                _inventory_count(kind, existing),
                len(positions),
                len(orientations),
            )
            count_changed = len(existing) != count
            if len(existing) > count:
                del existing[count:]

            is_tx = kind == "tx"
            for idx in range(count):
                is_new = idx >= len(existing)
                entry = {} if is_new else existing[idx]
                entry.update(
                    {
                        "name": self._node_label_text(kind, idx),
                        "node_name": f"{kind}_{idx + 1}",
                        "entry_type": kind,
                        "node_index": idx,
                        "material_id": "TX" if is_tx else "RX",
                        "supports_position": True,
                        "supports_orientation": True,
                        "supports_scale": False,
                        "supports_label_toggle": True,
                        "supports_highlight_toggle": False,
                        "_frame_position_valid": idx < len(positions),
                        "_frame_orientation_valid": idx < len(orientations),
                    }
                )
                if idx < len(positions):
                    entry["position"] = [float(value) for value in positions[idx]]
                else:
                    entry.setdefault("position", [0.0, 0.0, 0.0])

                if idx < len(orientations):
                    orientation_vals = _to_list(orientations[idx])
                    entry["orientation"] = orientation_vals
                    entry["orientation_degrees"] = [
                        float(np.degrees(value)) for value in orientation_vals
                    ]
                else:
                    entry.setdefault("orientation", [0.0, 0.0, 0.0])
                    entry.setdefault("orientation_degrees", [0.0, 0.0, 0.0])

                entry.setdefault("visible", True)
                entry.setdefault("highlighted", False)
                entry.setdefault("show_label", True)
                entry.setdefault("color", [1.0, 0.55, 0.0] if is_tx else [0.2, 0.4, 1.0])
                ensure_node_entry_identity(entry, kind, idx)
                if is_new:
                    existing.append(entry)
            return count_changed

        tx_changed = _sync_entries("tx", tx_positions, tx_orientations, viz.tx_entries)
        rx_changed = _sync_entries("rx", rx_positions, rx_orientations, viz.rx_entries)

        if (
            (tx_changed or rx_changed)
            and hasattr(viz, "ui_manager")
            and "objects" in viz.ui_manager.panels
        ):
            try:
                viz.ui_controller.populate_controls()
            except Exception:
                logger.debug("Unable to refresh object panel after TX/RX update", exc_info=True)

    def _is_hidden_for_pov(self, node_type: str, index: int) -> bool:
        """Return whether POV mode hides the node used as the camera origin.

        Args:
            node_type: "tx", "rx", or "target"
            index: The 0-based index of the node

        Returns:
            True if this node should be hidden for POV mode
        """
        return is_hidden_for_pov(getattr(self.visualizer, "app_state", None), node_type, index)

    def _update_tx_rx_visibility(self) -> bool:
        """Update TX/RX marker and label visibility through visual entities."""
        if not self._node_entity_renderer_ready():
            logger.debug("Cannot update TX/RX visibility: renderer not initialized")
            return False

        with self._renderer_update_batch():
            nodes_synced = self._sync_tx_rx_visual_entities()
            self._request_redraw(
                context="TX/RX visual entity visibility change",
            )
        return nodes_synced

    def update_tx_marker_sizes(self) -> None:
        """Update the size of all TX markers."""
        viz = self.visualizer
        if not viz.vis_initialized or not hasattr(viz, "tx_markers"):
            logger.warning(
                "Cannot update TX marker sizes: visualizer not initialized or no TX markers"
            )
            return

        size = getattr(viz, "tx_marker_size", 0.3)  # Default to 0.3 if not set
        logger.debug("Updating TX marker sizes to: %s", size)

        def _do_tx_marker_updates():
            """Inner function to perform marker updates (can be batched)."""
            for i, marker in enumerate(viz.tx_markers):
                color = self._uniform_color(marker, default=[1.0, 0.0, 0.0])
                current_center = self._resolve_node_anchor("tx", i, marker)
                position = (
                    np.asarray(current_center, dtype=float)
                    if current_center is not None
                    else np.zeros(3, dtype=float)
                )
                new_marker = self._node_marker_handle(
                    "tx",
                    i,
                    size=size,
                    color=color,
                    visible=marker.visible,
                    position=position,
                )
                marker_points = render_state_points(new_marker)
                orig_vertices = marker_points - marker_points.mean(axis=0)

                base_vertices = getattr(viz, "_tx_marker_base_vertices", None)
                if not isinstance(base_vertices, list):
                    base_vertices = []
                    viz._tx_marker_base_vertices = base_vertices
                while len(base_vertices) <= i:
                    base_vertices.append(None)
                base_vertices[i] = orig_vertices

                viz.tx_markers[i] = new_marker

        # Use batch updates for performance (single redraw at the end)
        if hasattr(viz.renderer, "batch_updates"):
            with viz.renderer.batch_updates():
                _do_tx_marker_updates()
        else:
            _do_tx_marker_updates()

        # Reapply current node coloring (individual mode uses custom colors)
        self.apply_node_coloring()

        # Force visualizer update
        try:
            viz.renderer.update_renderer()
            logger.debug("Updated %d TX markers to size %s", len(viz.tx_markers), size)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error updating renderer after TX marker size change: {e}")

    def update_rx_marker_sizes(self) -> None:
        """Update the size of all RX markers."""
        viz = self.visualizer
        if not viz.vis_initialized or not hasattr(viz, "rx_markers"):
            logger.warning(
                "Cannot update RX marker sizes: visualizer not initialized or no RX markers"
            )
            return

        size = getattr(viz, "rx_marker_size", 0.3)  # Default to 0.3 if not set
        logger.debug("Updating RX marker sizes to: %s", size)

        def _do_rx_marker_updates():
            """Inner function to perform marker updates (can be batched)."""
            for i, marker in enumerate(viz.rx_markers):
                color = self._uniform_color(marker, default=[0.0, 0.0, 1.0])
                current_center = self._resolve_node_anchor("rx", i, marker)
                position = (
                    np.asarray(current_center, dtype=float)
                    if current_center is not None
                    else np.zeros(3, dtype=float)
                )
                new_marker = self._node_marker_handle(
                    "rx",
                    i,
                    size=size,
                    color=color,
                    visible=marker.visible,
                    position=position,
                )
                marker_points = render_state_points(new_marker)
                orig_vertices = marker_points - marker_points.mean(axis=0)

                base_vertices = getattr(viz, "_rx_marker_base_vertices", None)
                if not isinstance(base_vertices, list):
                    base_vertices = []
                    viz._rx_marker_base_vertices = base_vertices
                while len(base_vertices) <= i:
                    base_vertices.append(None)
                base_vertices[i] = orig_vertices

                viz.rx_markers[i] = new_marker

        # Use batch updates for performance (single redraw at the end)
        if hasattr(viz.renderer, "batch_updates"):
            with viz.renderer.batch_updates():
                _do_rx_marker_updates()
        else:
            _do_rx_marker_updates()

        # Reapply current node coloring (individual mode uses custom colors)
        self.apply_node_coloring()

        # Force visualizer update
        try:
            viz.renderer.update_renderer()
            logger.debug("Updated %d RX markers to size %s", len(viz.rx_markers), size)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error updating renderer after RX marker size change: {e}")

    def apply_label_offsets(self) -> None:
        """Apply current label offset values to all visible labels."""
        viz = self.visualizer
        if not viz.vis_initialized:
            return

        logger.debug(
            "Applying label offsets: X=%s, Y=%s, Z=%s",
            viz.label_offset_x,
            viz.label_offset_y,
            viz.label_offset_z,
        )

        offset = np.array(
            [
                float(viz.label_offset_x),
                float(viz.label_offset_y),
                float(viz.label_offset_z),
            ]
        )

        with self._renderer_update_batch():
            self._sync_tx_rx_visual_entities()

            # Update target labels by stable ID because the list may be sparse.
            labels_by_id = {
                label.id: label
                for label in viz.target_labels
                if isinstance(label, RenderObjectState)
            }
            for i, target_entry in enumerate(viz.target_entries):
                ensure_target_entry_identity(target_entry, i)
                label_name = make_target_entry_geometry_name(target_entry, "label")
                label = labels_by_id.get(label_name)
                if label is None:
                    continue
                pos = target_entry.get("position")
                if pos is None:
                    mesh = target_entry.get("mesh")
                    if mesh is not None:
                        pos = target_entry_anchor_position(target_entry)
                if pos is None:
                    continue
                visible = target_label_visible(
                    target_entry,
                    i,
                    self._is_hidden_for_pov,
                    show_target_labels=getattr(
                        getattr(viz, "app_state", None), "show_target_labels", True
                    ),
                )
                self._node_render_sync.sync_label(
                    label_id=label_name,
                    label=label,
                    visible=visible,
                    anchor_position=pos,
                    offset=offset,
                )

            self._request_redraw(context="label offset change")

        logger.debug("Label offsets applied")

    def create_tx_rx_markers(self) -> None:
        """Create TX/RX marker placeholders plus text labels."""
        viz = self.visualizer
        old_tx_count = max(len(viz.tx_markers), len(viz.tx_labels))
        old_rx_count = max(len(viz.rx_markers), len(viz.rx_labels))
        viz.tx_markers.clear()
        viz.rx_markers.clear()
        viz.tx_labels.clear()
        viz.rx_labels.clear()

        for i in range(old_tx_count):
            self._remove_node_marker_entity("tx", i)
        for i in range(old_rx_count):
            self._remove_node_marker_entity("rx", i)

        if viz.num_tx is None or viz.num_tx <= 0:
            logger.error("Cannot create markers: num_tx is %s", viz.num_tx)
            return
        if viz.num_rx is None or viz.num_rx <= 0:
            logger.error("Cannot create markers: num_rx is %s", viz.num_rx)
            return

        logger.debug("Creating %s TX and %s RX markers", viz.num_tx, viz.num_rx)

        font_size = getattr(viz, "label_font_size", 0.3)

        for i in range(viz.num_tx):
            marker = self._node_marker_handle(
                "tx",
                i,
                size=viz.tx_marker_size,
                color=[1.0, 0.0, 0.0],
                visible=True,
            )
            viz.tx_markers.append(marker)

            text = self._node_label_text("tx", i)
            label = self._create_node_label_state("tx", i, text, [1.0, 0.0, 0.0], font_size)
            viz.tx_labels.append(label)

        for i in range(viz.num_rx):
            marker = self._node_marker_handle(
                "rx",
                i,
                size=viz.rx_marker_size,
                color=[0.0, 0.0, 1.0],
                visible=True,
            )
            viz.rx_markers.append(marker)

            text = self._node_label_text("rx", i)
            label = self._create_node_label_state("rx", i, text, [0.0, 0.0, 1.0], font_size)
            viz.rx_labels.append(label)

        self.apply_node_coloring()
        logger.debug(
            "Created %s TX markers and %s RX markers", len(viz.tx_markers), len(viz.rx_markers)
        )

    def apply_orientation_to_marker(self, marker, orientation, marker_name):
        """
        Apply absolute orientation to a neutral TX/RX marker handle.

        Marker payloads stay in local space; orientation is expressed through
        the render handle transform so all renderers receive the same intent.

        Args:
            marker: Renderer-neutral marker handle.
            orientation: The absolute orientation (yaw, pitch, roll) in radians
            marker_name: Name for debugging (e.g., "TX1", "RX2")
        """
        try:
            if marker is None:
                return
            if not isinstance(marker, RenderObjectState):
                logger.warning(
                    "Ignoring non-neutral marker geometry for %s orientation update",
                    marker_name,
                )
                return

            world_center = render_state_center(marker)
            yaw, pitch, roll = np.asarray(orientation, dtype=float).reshape(-1)[:3]
            marker.world_transform = Transform(
                create_orientation_transform(world_center, yaw, pitch, roll)
            )

        except (AttributeError, ValueError, TypeError, RuntimeError):
            logger.exception("Error applying orientation to marker %s", marker_name)

    def update_tx_rx_orientations(
        self,
        tx_markers: List[Any],
        rx_markers: List[Any],
        tx_orientations: List[Any],
        rx_orientations: List[Any],
    ) -> None:
        """
        Update TX/RX orientations for the current animation step.

        This function applies orientations to TX/RX markers using the same center-based
        rotation system as targets, ensuring consistent behavior.

        Args:
            tx_markers: List of TX marker handles.
            rx_markers: List of RX marker handles.
            tx_orientations: List of TX orientations (yaw, pitch, roll) in radians
            rx_orientations: List of RX orientations (yaw, pitch, roll) in radians
        """
        for i, tx_marker in enumerate(tx_markers):
            if i < len(tx_orientations):
                self.apply_orientation_to_marker(tx_marker, tx_orientations[i], f"TX{i+1}")

        for i, rx_marker in enumerate(rx_markers):
            if i < len(rx_orientations):
                self.apply_orientation_to_marker(rx_marker, rx_orientations[i], f"RX{i+1}")

    def ensure_tx_rx_markers_created(self, tx_count: int, rx_count: int):
        """Ensure TX/RX marker entities are created when frame data arrives."""
        viz = self.visualizer
        tx_count = max(0, int(tx_count))
        rx_count = max(0, int(rx_count))
        logger.debug(
            f"Ensuring TX/RX markers: current TX={len(viz.tx_markers)}, RX={len(viz.rx_markers)}, requested TX={tx_count}, RX={rx_count}"
        )

        # Trim excess TX/RX entities when incoming frame has fewer nodes.
        while len(viz.tx_markers) > tx_count:
            idx = len(viz.tx_markers) - 1
            viz.tx_markers.pop()
            self._remove_node_marker_entity("tx", idx)
            if idx < len(viz.tx_labels):
                viz.tx_labels.pop()

        while len(viz.rx_markers) > rx_count:
            idx = len(viz.rx_markers) - 1
            viz.rx_markers.pop()
            self._remove_node_marker_entity("rx", idx)
            if idx < len(viz.rx_labels):
                viz.rx_labels.pop()

        while len(viz.tx_labels) > len(viz.tx_markers):
            idx = len(viz.tx_labels) - 1
            viz.tx_labels.pop()
            self._remove_node_marker_entity("tx", idx)

        while len(viz.rx_labels) > len(viz.rx_markers):
            idx = len(viz.rx_labels) - 1
            viz.rx_labels.pop()
            self._remove_node_marker_entity("rx", idx)

        while len(viz.tx_labels) < len(viz.tx_markers):
            idx = len(viz.tx_labels)
            font_size = getattr(viz, "label_font_size", 0.3)
            text = self._node_label_text("tx", idx)
            label = self._create_node_label_state("tx", idx, text, [1.0, 0.0, 0.0], font_size)
            viz.tx_labels.append(label)
            logger.debug("Created missing TX label %d", len(viz.tx_labels))

        while len(viz.rx_labels) < len(viz.rx_markers):
            idx = len(viz.rx_labels)
            font_size = getattr(viz, "label_font_size", 0.3)
            text = self._node_label_text("rx", idx)
            label = self._create_node_label_state("rx", idx, text, [0.0, 0.0, 1.0], font_size)
            viz.rx_labels.append(label)
            logger.debug("Created missing RX label %d", len(viz.rx_labels))

        tx_needed = tx_count - len(viz.tx_markers)
        if tx_needed > 0:
            font_size = getattr(viz, "label_font_size", 0.3)
            for _ in range(tx_needed):
                idx = len(viz.tx_markers)
                marker = self._node_marker_handle(
                    "tx",
                    idx,
                    size=viz.tx_marker_size,
                    color=[1.0, 0.0, 0.0],
                    visible=True,
                )
                viz.tx_markers.append(marker)
                text = self._node_label_text("tx", idx)
                label = self._create_node_label_state("tx", idx, text, [1.0, 0.0, 0.0], font_size)
                viz.tx_labels.append(label)
            logger.debug("Created %d TX marker render handles", tx_needed)

        rx_needed = rx_count - len(viz.rx_markers)
        if rx_needed > 0:
            font_size = getattr(viz, "label_font_size", 0.3)
            for _ in range(rx_needed):
                idx = len(viz.rx_markers)
                marker = self._node_marker_handle(
                    "rx",
                    idx,
                    size=viz.rx_marker_size,
                    color=[0.0, 0.0, 1.0],
                    visible=True,
                )
                viz.rx_markers.append(marker)
                text = self._node_label_text("rx", idx)
                label = self._create_node_label_state("rx", idx, text, [0.0, 0.0, 1.0], font_size)
                viz.rx_labels.append(label)
            logger.debug("Created %d RX marker render handles", rx_needed)

        logger.debug(f"Final marker counts: TX={len(viz.tx_markers)}, RX={len(viz.rx_markers)}")

        if self._node_entity_renderer_ready():
            self._sync_tx_rx_visual_entities()
            logger.debug(
                "Synced %d TX and %d RX marker entities",
                len(viz.tx_markers),
                len(viz.rx_markers),
            )
            return

        logger.debug("Visualizer renderer not initialized, cannot sync TX/RX entities")
