"""Pointer picking and tooltip helpers for the pygfx renderer.

The pygfx backend owns hover picking because tooltip text depends on native
``pick_info`` and the renderer's name-to-object maps. This mixin keeps picking
metadata backend-local while formatting tooltips from stable ORCHAV names and
frame-packet canonical MPC data when available.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from ..protocol import MpcPathSelectionCallback

logger = logging.getLogger(__name__)


class PygfxPickingMixin:
    """Own pygfx pointer-pick metadata and hover tooltips."""

    ITYPE_NAMES = {
        0: "LoS",
        1: "Specular",
        2: "Diffuse",
        4: "Refraction",
        8: "Diffraction",
        99: "Virtual",
    }
    HOVER_INFO_MODES = {"off", "essential", "inspect_all"}
    ESSENTIAL_HOVER_TYPES = {"mpc_lines", "mpc_points", "target", "tx", "rx"}
    MPC_CLICK_MAX_MOVEMENT_PX = 5.0
    MPC_CLICK_MAX_DURATION_S = 0.60

    @classmethod
    def _normalize_hover_info_mode(cls, mode: Any) -> str:
        """Return a supported hover-tooltip policy token."""
        value = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "none": "off",
            "disabled": "off",
            "disable": "off",
            "minimal": "essential",
            "essentials": "essential",
            "rays": "essential",
            "ray": "essential",
            "all": "inspect_all",
            "inspect": "inspect_all",
            "full": "inspect_all",
            "debug": "inspect_all",
        }
        value = aliases.get(value, value)
        return value if value in cls.HOVER_INFO_MODES else "essential"

    def set_hover_info_mode(self, mode: Any) -> bool:
        """Set hover tooltip policy without changing picking or selection behavior."""
        normalized = self._normalize_hover_info_mode(mode)
        if normalized == getattr(self, "_hover_info_mode", "essential"):
            return True
        self._hover_info_mode = normalized
        self._hide_tooltip()
        return True

    def get_hover_info_mode(self) -> str:
        """Return the active hover tooltip policy."""
        return self._normalize_hover_info_mode(getattr(self, "_hover_info_mode", "essential"))

    def set_mpc_path_selection_callback(
        self,
        callback: MpcPathSelectionCallback | None,
    ) -> None:
        """Enable path selection with the source frame-packet identity."""
        if callback is not None and not callable(callback):
            raise TypeError("MPC path selection callback must be callable or None")
        self._mpc_path_selection_callback = callback
        self._cancel_mpc_path_click_gesture()
        if callback is None:
            self._invalidate_mpc_pick_cache()

    def set_mpc_pick_segment_mapping(
        self,
        packet_identity: int,
        canonical_segment_indices: Any,
    ) -> bool:
        """Install a worker-prepared packet-to-canonical segment mapping.

        The Explorer builds this optional O(segment-count) permutation away
        from the GUI thread.  Packet identity prevents an obsolete query from
        being attached after playback has already presented another frame.
        """
        packet = getattr(self, "last_frame_packet", None)
        if packet is None or id(packet) != int(packet_identity):
            return False
        if canonical_segment_indices is None:
            self._mpc_pick_segment_map_packet = packet
            self._mpc_pick_canonical_segment_indices = np.empty((0,), dtype=np.int32)
            self._mpc_pick_identity_mapping = True
            return True
        # MpcPathCatalog prepares this mapping as contiguous int32 on the
        # query worker. Retain that exact array: converting to np.intp here
        # would allocate an avoidable int64 copy on the GUI thread.
        if not isinstance(canonical_segment_indices, np.ndarray):
            return False
        indices = canonical_segment_indices
        if indices.ndim != 1 or indices.dtype != np.int32 or not indices.flags.c_contiguous:
            return False
        packet_lines = getattr(packet, "mpc_lines", None)
        if packet_lines is not None and len(indices) != len(packet_lines):
            return False
        self._mpc_pick_segment_map_packet = packet
        self._mpc_pick_canonical_segment_indices = indices
        self._mpc_pick_identity_mapping = False
        return True

    def route_mpc_path_selection_event(self, event: Any) -> None:
        """Track one short unmodified click without consuming camera drags."""
        callback = getattr(self, "_mpc_path_selection_callback", None)
        if callback is None:
            # Explorer-off is a true fast path: inspect no event or frame data.
            return

        event_type = str(getattr(event, "type", ""))
        if event_type == "pointer_leave":
            self._cancel_mpc_path_click_gesture()
            return
        if event_type == "pointer_down":
            self._begin_mpc_path_click(event)
            return
        if event_type == "pointer_move":
            self._update_mpc_path_click(event)
            return
        if event_type == "pointer_up":
            self._finish_mpc_path_click(event, callback)

    def _begin_mpc_path_click(self, event: Any) -> None:
        """Record a candidate only for a picked bulk MPC line."""
        self._cancel_mpc_path_click_gesture()
        if not self._is_unmodified_left_button(event):
            return
        position = self._pointer_position(event)
        path_pick = self._canonical_mpc_path_pick_from_event(event)
        if position is None or path_pick is None:
            return
        path_id, packet_identity = path_pick
        self._mpc_path_click_candidate = {
            "pointer_id": getattr(event, "pointer_id", None),
            "position": position,
            "started_at": time.monotonic(),
            "path_id": path_id,
            "packet_identity": packet_identity,
        }

    def _update_mpc_path_click(self, event: Any) -> None:
        """Cancel the candidate once pointer movement becomes a camera drag."""
        candidate = getattr(self, "_mpc_path_click_candidate", None)
        if candidate is None:
            return
        if not self._same_pointer(candidate["pointer_id"], getattr(event, "pointer_id", None)):
            return
        position = self._pointer_position(event)
        if position is None or self._pointer_moved_too_far(candidate["position"], position):
            self._cancel_mpc_path_click_gesture()

    def _finish_mpc_path_click(
        self,
        event: Any,
        callback: MpcPathSelectionCallback,
    ) -> None:
        """Publish the canonical path ID after validating the complete gesture."""
        candidate = getattr(self, "_mpc_path_click_candidate", None)
        self._cancel_mpc_path_click_gesture()
        if candidate is None or not self._is_unmodified_left_button(event):
            return
        if not self._same_pointer(candidate["pointer_id"], getattr(event, "pointer_id", None)):
            return
        if time.monotonic() - float(candidate["started_at"]) > self.MPC_CLICK_MAX_DURATION_S:
            return
        position = self._pointer_position(event)
        if position is None or self._pointer_moved_too_far(candidate["position"], position):
            return
        path_pick = self._canonical_mpc_path_pick_from_event(event)
        if path_pick is None:
            return
        path_id, packet_identity = path_pick
        if path_id != candidate["path_id"] or packet_identity != candidate["packet_identity"]:
            return
        try:
            callback(path_id, packet_identity)
        except Exception:
            logger.exception("MPC viewport-selection callback failed")

    def _canonical_mpc_path_id_from_event(self, event: Any) -> int | None:
        """Resolve a picked bulk line segment to its canonical frame-local path."""
        path_pick = self._canonical_mpc_path_pick_from_event(
            event,
            allow_cold_mapping=True,
        )
        return None if path_pick is None else path_pick[0]

    def _canonical_mpc_path_pick_from_event(
        self,
        event: Any,
        *,
        allow_cold_mapping: bool = False,
    ) -> tuple[int, int] | None:
        """Resolve one line pick plus the exact packet used for its mapping."""
        target = getattr(event, "target", None)
        if target is None or getattr(self, "_reverse_objects", {}).get(id(target)) != "mpc_lines":
            return None
        vertex_index = self._picked_vertex_index(event)
        if vertex_index is None:
            return None
        packet = getattr(self, "last_frame_packet", None)
        if packet is None or getattr(packet, "canonical_data", None) is None:
            return None
        canonical_segment_index = self._canonical_mpc_segment_index(
            packet,
            vertex_index // 2,
            allow_cold_mapping=allow_cold_mapping,
        )
        if canonical_segment_index is None:
            return None

        canon = packet.canonical_data
        segment_path_ids = getattr(canon, "segment_path_id", None)
        if segment_path_ids is not None and canonical_segment_index < len(segment_path_ids):
            path_id = int(segment_path_ids[canonical_segment_index])
        else:
            start_index = self._canonical_mpc_segment_start_index(
                canon,
                canonical_segment_index,
            )
            point_path_ids = getattr(canon, "path_id", None)
            if start_index is None or point_path_ids is None or start_index >= len(point_path_ids):
                return None
            path_id = int(point_path_ids[start_index])
        if path_id < 0:
            return None
        return path_id, id(packet)

    def _invalidate_mpc_pick_cache(self) -> None:
        """Release packet-local canonical segment indices immediately."""
        self._mpc_pick_segment_map_packet = None
        self._mpc_pick_canonical_segment_indices = np.empty((0,), dtype=np.int32)
        self._mpc_pick_identity_mapping = False

    def _cancel_mpc_path_click_gesture(self) -> None:
        """Forget any partial click candidate."""
        self._mpc_path_click_candidate = None

    @staticmethod
    def _is_unmodified_left_button(event: Any) -> bool:
        """Return whether one event represents an unmodified primary button."""
        try:
            button = int(getattr(event, "button", 0) or 0)
        except (TypeError, ValueError):
            return False
        return button == 1 and not tuple(getattr(event, "modifiers", ()) or ())

    @staticmethod
    def _pointer_position(event: Any) -> tuple[float, float] | None:
        """Return finite logical pointer coordinates."""
        try:
            x = float(getattr(event, "x"))
            y = float(getattr(event, "y"))
        except (AttributeError, TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return x, y

    @staticmethod
    def _same_pointer(first: Any, second: Any) -> bool:
        """Treat missing pointer IDs as compatible for older rendercanvas events."""
        return first is None or second is None or first == second

    def _pointer_moved_too_far(
        self,
        start: tuple[float, float],
        current: tuple[float, float],
    ) -> bool:
        """Return whether logical movement crossed the click/drag threshold."""
        dx = current[0] - start[0]
        dy = current[1] - start[1]
        return dx * dx + dy * dy > self.MPC_CLICK_MAX_MOVEMENT_PX**2

    def _register_pick_metadata(self, name: str, metadata: dict) -> None:
        """Associate tooltip metadata with a named geometry."""
        self._pick_metadata[name] = metadata

    def _on_pointer_move(self, event: Any) -> None:
        """Handle pointer_move to show/update tooltip on hover."""
        hud_role_enabled = getattr(self, "_hud_role_enabled", None)
        if callable(hud_role_enabled) and not hud_role_enabled("annotation"):
            # HUD-off is a real fast path: do not infer metadata or format
            # tooltip content that cannot be displayed.
            self._hide_tooltip()
            return
        hover_mode = self.get_hover_info_mode()
        if hover_mode == "off":
            self._hide_tooltip()
            return

        target = getattr(event, "target", None)
        if target is None:
            self._hide_tooltip()
            return

        name = self._reverse_objects.get(id(target))
        if name is None:
            self._hide_tooltip()
            return
        if hover_mode == "essential" and self._name_is_scene_mesh(name):
            self._hide_tooltip()
            return

        meta = self._pick_metadata.get(name)
        if meta is None:
            meta = self._infer_pick_metadata(name)
            if meta is None:
                self._hide_tooltip()
                return
            self._pick_metadata[name] = meta
        elif self._name_is_scene_mesh(name):
            inferred = self._infer_pick_metadata(name)
            if inferred is not None:
                meta = {**inferred, **meta}
                if not meta.get("material"):
                    meta["material"] = inferred.get("material", "")

        hover_identity = self._hover_cache_identity(name, meta, event)
        if hover_identity == getattr(self, "_last_hover_identity", None):
            self._reposition_tooltip(event)
            return

        self._last_hover_identity = hover_identity
        tooltip_text = self._format_tooltip(name, meta, event)
        if tooltip_text:
            self._show_tooltip(tooltip_text, event)
        else:
            self._hide_tooltip()

    def _on_pointer_leave(self, event: Any) -> None:
        """Hide tooltip when pointer leaves the canvas."""
        self._hide_tooltip()

    def _format_tooltip(self, name: str, meta: dict, event: Any) -> str:
        """Build tooltip text from pick metadata and event info."""
        obj_type = meta.get("type", "")
        hover_mode = self.get_hover_info_mode()
        if hover_mode == "off":
            return ""
        if hover_mode == "essential" and obj_type not in self.ESSENTIAL_HOVER_TYPES:
            return ""

        if obj_type in ("building", "scene_merged"):
            parts = [self._format_scene_hover_name(name, meta)]
            mat = meta.get("material")
            if mat:
                parts.append(f"Material: {mat}")
            return "\n".join(parts)

        if obj_type == "target":
            return meta.get("name", name)

        if obj_type in ("tx", "rx"):
            prefix = obj_type.upper()
            idx = meta.get("index", "?")
            pos = meta.get("position")
            text = f"{prefix}{idx}"
            if pos is not None:
                text += f"\n({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            return text

        if obj_type == "mpc_lines":
            return self._format_mpc_tooltip(name, event)

        if obj_type == "mpc_points":
            return self._format_mpc_point_tooltip(event)

        return ""

    def _hover_cache_identity(self, name: str, meta: dict, event: Any) -> tuple[str, int | None]:
        """Return the geometry and picked element that determine tooltip text."""
        picked_index = self._picked_vertex_index(event)
        obj_type = meta.get("type", "")
        if obj_type == "mpc_lines" and picked_index is not None:
            return name, picked_index // 2
        if obj_type == "mpc_points":
            return name, picked_index
        return name, None

    @staticmethod
    def _picked_vertex_index(event: Any) -> int | None:
        """Return a non-negative pygfx vertex index from one pointer event."""
        pick_info = getattr(event, "pick_info", None)
        if not isinstance(pick_info, dict):
            return None
        for key in ("vertex_index", "index"):
            try:
                index = int(pick_info.get(key, -1))
            except (TypeError, ValueError):
                continue
            if index >= 0:
                return index
        return None

    def _format_mpc_tooltip(self, name: str, event: Any) -> str:
        """Format tooltip for an MPC line segment from pick_info.

        Filtered and Top-K packets map the packet-local picked segment index
        back to the unfiltered canonical segment index.
        """
        vertex_idx = self._picked_vertex_index(event)
        if vertex_idx is None:
            return "MPC segment"

        packet_segment_idx = vertex_idx // 2

        packet = self.last_frame_packet
        if packet is None or packet.canonical_data is None:
            return f"MPC segment #{packet_segment_idx}"

        canon = packet.canonical_data

        canonical_segment_idx = self._canonical_mpc_segment_index(
            packet,
            packet_segment_idx,
        )
        if canonical_segment_idx is None:
            return f"MPC segment #{packet_segment_idx}"

        canonical_start_idx = self._canonical_mpc_segment_start_index(
            canon,
            canonical_segment_idx,
        )

        parts: list[str] = []

        itype_value = None
        if packet.mpc_line_itypes is not None and packet_segment_idx < len(packet.mpc_line_itypes):
            itype_value = int(packet.mpc_line_itypes[packet_segment_idx])
        elif getattr(canon, "segment_itype", None) is not None and canonical_segment_idx < len(
            canon.segment_itype
        ):
            itype_value = int(canon.segment_itype[canonical_segment_idx])
        elif (
            canon.itype is not None
            and canonical_start_idx is not None
            and canonical_start_idx < canon.itype.size
        ):
            itype_value = int(canon.itype[canonical_start_idx])
        if itype_value is not None:
            itype_name = self.ITYPE_NAMES.get(itype_value, f"type_{itype_value}")
            parts.append(itype_name)

        order_value = None
        if getattr(canon, "segment_order", None) is not None and canonical_segment_idx < len(
            canon.segment_order
        ):
            order_value = int(canon.segment_order[canonical_segment_idx])
        elif (
            canon.order is not None
            and canonical_start_idx is not None
            and canonical_start_idx < canon.order.size
        ):
            order_value = int(canon.order[canonical_start_idx])
        if order_value is not None:
            parts.append(f"order {order_value}")

        path_idx = None
        if getattr(canon, "segment_path_id", None) is not None and canonical_segment_idx < len(
            canon.segment_path_id
        ):
            path_idx = int(canon.segment_path_id[canonical_segment_idx])
        elif (
            canon.path_id is not None
            and canonical_start_idx is not None
            and canonical_start_idx < canon.path_id.size
        ):
            path_idx = int(canon.path_id[canonical_start_idx])

        if path_idx is not None and path_idx >= 0:
            if canon.path_delays is not None and path_idx < canon.path_delays.size:
                delay_ns = float(canon.path_delays[path_idx])
                if np.isfinite(delay_ns):
                    parts.append(f"delay {delay_ns:.2f} ns")
            if canon.path_losses is not None and path_idx < canon.path_losses.size:
                loss_db = float(canon.path_losses[path_idx])
                if np.isfinite(loss_db):
                    parts.append(f"loss {loss_db:.2f} dB")

        if not parts:
            return f"MPC segment #{packet_segment_idx}"
        return " | ".join(parts)

    def _format_mpc_point_tooltip(self, event: Any) -> str:
        """Format one physical MPC interaction point without line-index remapping."""
        point_idx = self._picked_vertex_index(event)
        if point_idx is None:
            return "MPC interaction point"

        parts: list[str] = []
        packet = self.last_frame_packet
        itypes = None if packet is None else getattr(packet, "mpc_bounce_itypes", None)
        if itypes is not None and point_idx < len(itypes):
            itype_value = int(itypes[point_idx])
            parts.append(self.ITYPE_NAMES.get(itype_value, f"type_{itype_value}"))
        parts.append(f"interaction point #{point_idx}")
        return " | ".join(parts)

    def _canonical_mpc_segment_index(
        self,
        packet: Any,
        packet_segment_idx: int,
        *,
        allow_cold_mapping: bool = True,
    ) -> int | None:
        """Map one filtered packet segment index to its canonical segment index."""
        if packet_segment_idx < 0:
            return None

        packet_lines = getattr(packet, "mpc_lines", None)
        if packet_lines is not None and packet_segment_idx >= len(packet_lines):
            return None

        segment_mask = getattr(packet, "segment_mask", None)
        if segment_mask is None:
            return packet_segment_idx

        mask = np.asarray(segment_mask)
        if mask.ndim != 1:
            return None

        cached_packet = getattr(self, "_mpc_pick_segment_map_packet", None)
        if cached_packet is not packet:
            if not allow_cold_mapping:
                return None
            self._mpc_pick_segment_map_packet = packet
            self._mpc_pick_canonical_segment_indices = np.flatnonzero(mask).astype(
                np.int32,
                copy=False,
            )
            self._mpc_pick_identity_mapping = False
        elif bool(getattr(self, "_mpc_pick_identity_mapping", False)):
            return packet_segment_idx

        canonical_indices = self._mpc_pick_canonical_segment_indices
        if packet_segment_idx >= len(canonical_indices):
            return None
        return int(canonical_indices[packet_segment_idx])

    @staticmethod
    def _canonical_mpc_segment_start_index(canon: Any, segment_idx: int) -> int | None:
        """Return a canonical point index for per-point segment metadata fallback."""
        start_indices = getattr(canon, "segment_start_indices", None)
        if start_indices is not None and segment_idx < len(start_indices):
            start_idx = int(start_indices[segment_idx])
            return start_idx if start_idx >= 0 else None

        lines = getattr(canon, "lines", None)
        if lines is None or segment_idx >= len(lines):
            return None
        start_idx = int(lines[segment_idx, 0])
        return start_idx if start_idx >= 0 else None

    def _infer_pick_metadata(self, name: str) -> Optional[dict]:
        """Derive pick metadata from geometry name conventions."""
        if "::mesh" in name and (name.startswith("scene:") or "merged" in name):
            display = name.split("::")[0].replace("scene:", "")
            material_name = ""
            obj_type = "building"
            mesh_count = None
            state = self._applied_state_for_geometry(name)
            if state is not None:
                mat_info = self._state_field(state, "material", {}) or {}
                if isinstance(mat_info, dict):
                    material_name = mat_info.get("material_name", "")
                metadata = self._state_field(state, "metadata", {}) or {}
                if isinstance(metadata, dict) and metadata.get("type") == "scene_merged":
                    obj_type = "scene_merged"
                    mesh_ids = metadata.get("mesh_ids")
                    if isinstance(mesh_ids, (list, tuple)):
                        mesh_count = len(mesh_ids)
            elif display.startswith("merged_"):
                obj_type = "scene_merged"
            meta = {"type": obj_type, "name": display, "material": material_name}
            if mesh_count is not None:
                meta["mesh_count"] = mesh_count
            return meta

        if name.startswith("node:tx_") and name.endswith("::marker"):
            try:
                idx = int(name.split("node:tx_", 1)[1].split("::", 1)[0])
            except (ValueError, IndexError):
                idx = 0
            pos = self._positions.get(name)
            return {"type": "tx", "index": idx + 1, "position": pos}

        if name.startswith("node:rx_") and name.endswith("::marker"):
            try:
                idx = int(name.split("node:rx_", 1)[1].split("::", 1)[0])
            except (ValueError, IndexError):
                idx = 0
            pos = self._positions.get(name)
            return {"type": "rx", "index": idx + 1, "position": pos}

        if name.startswith("target:") and "::mesh" in name:
            display = name.split("::")[0].replace("target:", "")
            return {"type": "target", "name": display}

        if name == "mpc_lines":
            return {"type": "mpc_lines"}

        if name == "mpc_points":
            return {"type": "mpc_points"}

        return None

    @staticmethod
    def _name_is_scene_mesh(name: str) -> bool:
        """Return True for scene mesh objects, including merged scene groups."""
        return "::mesh" in name and name.startswith("scene:")

    @staticmethod
    def _state_field(state: Any, field: str, default: Any = None) -> Any:
        """Read a field from dict-like or dataclass applied-state entries."""
        if isinstance(state, dict):
            return state.get(field, default)
        return getattr(state, field, default)

    def _applied_state_for_geometry(self, geometry_name: str) -> Any:
        """Return renderer-owned material and metadata for hover formatting."""
        material = self._materials.get(geometry_name)
        metadata = self._pick_metadata.get(geometry_name, {})
        if material is None and not metadata:
            return None
        return {"material": material, "metadata": metadata}

    @staticmethod
    def _format_scene_hover_name(name: str, meta: dict) -> str:
        """Return a user-facing label for scene hover tooltips."""
        if meta.get("type") == "scene_merged":
            mesh_count = meta.get("mesh_count")
            if mesh_count:
                return f"Scene group ({int(mesh_count)} meshes)"
            return "Scene group"
        display = str(meta.get("name") or name)
        if display.startswith("merged_") or display.startswith("scene:merged_"):
            return "Scene group"
        return display
