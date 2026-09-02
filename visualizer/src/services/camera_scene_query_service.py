"""Read-only scene and frame queries used by camera workflows.

Camera controllers express user intent, while this service owns the storage
details for renderer bounds, ViewModel positions, cached frame orientations,
and current focus dropdown selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from shared.logging import get_logger

from .base import BaseService
from .object_identity import normalize_token

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.camera_scene_query_service")


class CameraSceneQueryService(BaseService):
    """Resolve scene bounds and selected entity data for camera controllers."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind camera scene queries to the active visualizer state."""
        super().__init__()
        self.visualizer = visualizer

    def compute_scene_bounds(
        self, scope: str = "visible"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return center and extent from the renderer's successfully applied scene."""
        renderer = getattr(self.visualizer, "renderer", None)
        compute_renderer_bounds = getattr(renderer, "compute_scene_bounds", None)
        if not callable(compute_renderer_bounds):
            return None
        try:
            bbox = compute_renderer_bounds(scope=scope)
            if bbox is None:
                return None
            center = np.asarray(bbox.get_center(), dtype=float)
            extent = np.asarray(bbox.get_extent(), dtype=float)
            if center.shape != (3,) or extent.shape != (3,):
                return None
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
                return None
            return center, extent
        except (RuntimeError, AttributeError, TypeError, ValueError):
            logger.debug("Renderer scene bounds are unavailable", exc_info=True)
            return None

    def viewport_aspect(self) -> float:
        """Return renderer viewport aspect ratio, or a stable default."""
        renderer = getattr(self.visualizer, "renderer", None)
        width = float(getattr(renderer, "_width", 0.0))
        height = float(getattr(renderer, "_height", 0.0))
        if width > 1.0 and height > 1.0:
            return width / height
        return 16.0 / 9.0

    def get_focus_position(self) -> Optional[list[float]]:
        """Get the selected target, TX, or RX position for focus/follow mode."""
        viz = self.visualizer
        try:
            dropdown = getattr(viz, "target_focus_dropdown", None)
            if dropdown is None:
                return None

            selection = dropdown.currentData()
            if not isinstance(selection, dict):
                return None

            focus_type = selection.get("type")
            index = selection.get("index")
            if focus_type == "auto":
                return self.resolve_auto_focus_position()
            if focus_type == "target":
                return self.resolve_target_selection_position(selection)
            if focus_type == "tx" and index is not None:
                return self.resolve_tx_position(int(index))
            if focus_type == "rx" and index is not None:
                return self.resolve_rx_position(int(index))
            return None

        except (KeyError, AttributeError, ValueError, TypeError, IndexError) as exc:
            logger.error("Error resolving camera focus position: %s", exc)
            return None

    def resolve_auto_focus_position(self) -> Optional[list[float]]:
        """Return the best available focus position for automatic camera targets."""
        viz = self.visualizer
        vm = getattr(viz, "current_view_model", None)
        if vm is not None:
            pos = self._first_position(getattr(vm, "target_positions", None))
            if pos:
                return pos
            pos = self._first_position(getattr(vm, "tx_positions", None))
            if pos:
                return pos
            pos = self._first_position(getattr(vm, "rx_positions", None))
            if pos:
                return pos
        pos = self._first_position(getattr(viz, "current_tx_positions", None))
        if pos:
            return pos
        pos = self._first_position(getattr(viz, "current_rx_positions", None))
        if pos:
            return pos
        return None

    def target_focus_selection(self, metadata_index: int) -> Optional[dict[str, Any]]:
        """Return stable dropdown data for one target in the active frame.

        Target arrays follow ``ViewModel.target_metadata`` order, while POV
        visibility follows the persistent ``target_entries`` order.  A focus
        selection therefore carries both the target's stable identity and its
        canonical entry index; a frame-array index must never double as object
        identity.
        """
        metadata = self._target_metadata()
        if not self._has_index(metadata, metadata_index):
            return None

        metadata_aliases = self._target_identity_aliases(metadata[metadata_index])
        if not metadata_aliases:
            return None

        canonical_index = self._canonical_target_index(metadata_aliases)
        if canonical_index is None:
            return None

        entry = self.visualizer.target_entries[canonical_index]
        stable_target_id = self._stable_target_id(entry, canonical_index)
        return {
            "type": "target",
            "stable_target_id": stable_target_id,
            "index": canonical_index,
        }

    def resolve_target_selection_position(self, selection: dict[str, Any]) -> Optional[list[float]]:
        """Resolve a selected target position by identity in the current frame."""
        resolved = self._resolve_target_selection(selection)
        if resolved is None:
            return None
        _, canonical_index, metadata_index = resolved

        vm = getattr(self.visualizer, "current_view_model", None)
        positions = getattr(vm, "target_positions", None) if vm is not None else None
        pos = self._extract_position(positions, metadata_index)
        if pos:
            return pos

        entries = getattr(self.visualizer, "target_entries", None)
        if self._has_index(entries, canonical_index):
            return self._normalize_position(entries[canonical_index].get("position"))
        return None

    def resolve_tx_position(self, index: int) -> Optional[list[float]]:
        """Resolve one TX position from ViewModel or current frame state."""
        viz = self.visualizer
        vm = getattr(viz, "current_view_model", None)
        positions = getattr(vm, "tx_positions", None) if vm is not None else None
        pos = self._extract_position(positions, index)
        if pos:
            return pos
        pos = self._extract_position(getattr(viz, "current_tx_positions", None), index)
        if pos:
            return pos
        return None

    def resolve_rx_position(self, index: int) -> Optional[list[float]]:
        """Resolve one RX position from ViewModel or current frame state."""
        viz = self.visualizer
        vm = getattr(viz, "current_view_model", None)
        positions = getattr(vm, "rx_positions", None) if vm is not None else None
        pos = self._extract_position(positions, index)
        if pos:
            return pos
        pos = self._extract_position(getattr(viz, "current_rx_positions", None), index)
        if pos:
            return pos
        return None

    def format_target_label(self, index: int, *, canonical_index: Optional[int] = None) -> str:
        """Return the dropdown label for one target focus entry."""
        label_index = index if canonical_index is None else canonical_index
        base_label = f"Target {label_index + 1}"
        name = self.get_target_name(index, canonical_index=canonical_index)
        if name:
            return f"{base_label} - {name}"
        return base_label

    def get_target_name(
        self,
        index: int,
        *,
        canonical_index: Optional[int] = None,
    ) -> Optional[str]:
        """Return a target display name from ViewModel metadata or scene entries."""
        viz = self.visualizer
        vm = getattr(viz, "current_view_model", None)
        metadata = getattr(vm, "target_metadata", None) if vm else None
        if metadata and 0 <= index < len(metadata):
            name = metadata[index].get("name") or metadata[index].get("id")
            if name:
                return str(name)
        entry_index = index if canonical_index is None else canonical_index
        if getattr(viz, "target_entries", None) and 0 <= entry_index < len(viz.target_entries):
            entry = viz.target_entries[entry_index]
            name = entry.get("target_name") or entry.get("name")
            if name:
                return str(name)
        return None

    def get_entity_position_orientation_and_info(
        self,
    ) -> tuple[Optional[list[float]], Optional[list[float]], Optional[dict]]:
        """Get position, orientation, and entity info for the selected entity."""
        viz = self.visualizer
        try:
            dropdown = getattr(viz, "target_focus_dropdown", None)
            if dropdown is None:
                return None, None, None

            selection = dropdown.currentData()
            if not isinstance(selection, dict):
                return None, None, None

            focus_type = selection.get("type")
            index = selection.get("index")

            if focus_type == "auto":
                target_selection = self.target_focus_selection(0)
                if target_selection is not None:
                    resolved = self._target_position_orientation_and_info(target_selection)
                    if resolved[0] is not None:
                        return resolved
                pos, orient = self.get_tx_position_and_orientation(0)
                if pos is not None:
                    return pos, orient, {"type": "tx", "index": 0}
                pos, orient = self.get_rx_position_and_orientation(0)
                return pos, orient, {"type": "rx", "index": 0}

            if focus_type == "target":
                return self._target_position_orientation_and_info(selection)

            if focus_type == "tx" and index is not None:
                pos, orient = self.get_tx_position_and_orientation(int(index))
                return pos, orient, {"type": "tx", "index": int(index)}

            if focus_type == "rx" and index is not None:
                pos, orient = self.get_rx_position_and_orientation(int(index))
                return pos, orient, {"type": "rx", "index": int(index)}

            return None, None, None

        except (KeyError, AttributeError, ValueError, TypeError, IndexError) as exc:
            logger.error("Error getting camera entity info: %s", exc)
            return None, None, None

    def _target_position_orientation_and_info(
        self, selection: dict[str, Any]
    ) -> tuple[Optional[list[float]], Optional[list[float]], Optional[dict]]:
        """Resolve a target pose and canonical POV identity from dropdown data."""
        resolved = self._resolve_target_selection(selection)
        if resolved is None:
            return None, None, None
        stable_target_id, canonical_index, metadata_index = resolved

        position = self.resolve_target_selection_position(selection)
        orientation = self._target_orientation_from_frame_identity(
            stable_target_id,
            metadata_index,
        )
        if orientation is None:
            orientation = self._target_orientation_from_view_model(metadata_index)
        if orientation is None:
            logger.debug("No orientation found for target %s", stable_target_id)
        return (
            position,
            orientation,
            {
                "type": "target",
                "stable_target_id": stable_target_id,
                "index": canonical_index,
            },
        )

    def get_tx_position_and_orientation(
        self, index: int
    ) -> tuple[Optional[list[float]], Optional[list[float]]]:
        """Return TX position plus yaw/pitch/roll from frame data."""
        position = self.resolve_tx_position(index)
        orientation = self._orientation_array_from_frame("tx_orientations", index)
        if orientation is None:
            orientation = self._orientation_array_from_view_model("tx_orientations", index)
        if orientation is None:
            logger.debug("No orientation found for TX %d", index + 1)
        return position, orientation

    def get_rx_position_and_orientation(
        self, index: int
    ) -> tuple[Optional[list[float]], Optional[list[float]]]:
        """Return RX position plus yaw/pitch/roll from frame data."""
        position = self.resolve_rx_position(index)
        orientation = self._orientation_array_from_frame("rx_orientations", index)
        if orientation is None:
            orientation = self._orientation_array_from_view_model("rx_orientations", index)
        if orientation is None:
            logger.debug("No orientation found for RX %d", index + 1)
        return position, orientation

    def _target_orientation_from_frame_identity(
        self,
        stable_target_id: str,
        fallback_metadata_index: int,
    ) -> Optional[list[float]]:
        """Return cached target orientation after matching stable identity."""
        frame_data = self._current_frame_data()
        if frame_data is None:
            return None
        targets_metadata = frame_data.get("targets_metadata", [])
        aliases = self._selection_target_aliases(stable_target_id)
        metadata_index = self._metadata_index_for_aliases(targets_metadata, aliases)
        if metadata_index is None:
            metadata_index = fallback_metadata_index
        if self._has_index(targets_metadata, metadata_index):
            return self._normalize_orientation(targets_metadata[metadata_index].get("orientation"))
        return None

    def _target_orientation_from_view_model(self, index: int) -> Optional[list[float]]:
        """Return target orientation from the active ViewModel metadata."""
        vm = getattr(self.visualizer, "current_view_model", None)
        if vm is None or not getattr(vm, "target_metadata", None):
            return None
        if 0 <= index < len(vm.target_metadata):
            return self._normalize_orientation(vm.target_metadata[index].get("orientation"))
        return None

    def _orientation_array_from_frame(self, field_name: str, index: int) -> Optional[list[float]]:
        """Return one orientation array from cached provider frame data."""
        frame_data = self._current_frame_data()
        if frame_data is None:
            return None
        orientations = frame_data.get(field_name, [])
        if 0 <= index < len(orientations):
            return self._normalize_orientation(orientations[index])
        return None

    def _orientation_array_from_view_model(
        self, field_name: str, index: int
    ) -> Optional[list[float]]:
        """Return one orientation array from ViewModel orientation fields."""
        vm = getattr(self.visualizer, "current_view_model", None)
        orientations = getattr(vm, field_name, None) if vm is not None else None
        if orientations is not None and 0 <= index < len(orientations):
            return self._normalize_orientation(orientations[index])
        return None

    def _current_frame_data(self) -> Optional[dict[str, Any]]:
        """Return cached frame data for the current visualizer step."""
        viz = self.visualizer
        if not hasattr(viz, "current_step"):
            return None
        cache_service = getattr(viz, "cache_service", None)
        get_frame = getattr(cache_service, "get_frame", None)
        if not callable(get_frame):
            return None
        return get_frame(viz.current_step)

    def _target_metadata(self) -> Any:
        """Return active target metadata without assuming a concrete sequence type."""
        vm = getattr(self.visualizer, "current_view_model", None)
        return getattr(vm, "target_metadata", None) if vm is not None else None

    def _resolve_target_selection(
        self,
        selection: dict[str, Any],
    ) -> Optional[tuple[str, int, int]]:
        """Return stable ID, canonical index, and current metadata index."""
        raw_stable_id = selection.get("stable_target_id")
        if raw_stable_id is None or selection.get("index") is None:
            return None
        selected_canonical_index = int(selection["index"])

        stable_target_id = normalize_token(raw_stable_id)
        aliases = self._selection_target_aliases(stable_target_id)
        canonical_index = self._canonical_target_index(aliases)
        metadata_index = self._metadata_index_for_aliases(self._target_metadata(), aliases)
        if canonical_index is None or metadata_index is None:
            return None
        if selected_canonical_index != canonical_index:
            logger.debug(
                "Resolved stale target focus index %d to canonical index %d for %s",
                selected_canonical_index,
                canonical_index,
                stable_target_id,
            )
        return stable_target_id, canonical_index, metadata_index

    def _selection_target_aliases(self, stable_target_id: str) -> set[str]:
        """Return aliases for a stable target selection and its canonical entry."""
        aliases = {normalize_token(stable_target_id)}
        canonical_index = self._canonical_target_index(aliases)
        entries = getattr(self.visualizer, "target_entries", None)
        if canonical_index is not None and self._has_index(entries, canonical_index):
            aliases.update(self._target_identity_aliases(entries[canonical_index]))
        return aliases

    def _canonical_target_index(self, aliases: set[str]) -> Optional[int]:
        """Return the persistent target-entry index matching any identity alias."""
        entries = getattr(self.visualizer, "target_entries", None) or ()
        for index, entry in enumerate(entries):
            if aliases.intersection(self._target_identity_aliases(entry)):
                return index
        return None

    def _metadata_index_for_aliases(self, metadata: Any, aliases: set[str]) -> Optional[int]:
        """Return the current frame-array index matching any target identity alias."""
        if metadata is None:
            return None
        for index, item in enumerate(metadata):
            if aliases.intersection(self._target_identity_aliases(item)):
                return index
        return None

    @staticmethod
    def _target_identity_aliases(target: Any) -> set[str]:
        """Return normalized stable/name aliases from target metadata or entries."""
        if not hasattr(target, "get"):
            return set()
        aliases: set[str] = set()
        for key in ("stable_target_id", "target_name", "node_name", "name", "id"):
            value = target.get(key)
            if value is not None and str(value).strip():
                aliases.add(normalize_token(value))
        return aliases

    @staticmethod
    def _stable_target_id(entry: Any, canonical_index: int) -> str:
        """Return the canonical stable ID of one persistent target entry."""
        if hasattr(entry, "get"):
            for key in ("stable_target_id", "target_name", "node_name", "name", "id"):
                value = entry.get(key)
                if value is not None and str(value).strip():
                    return normalize_token(value)
        return normalize_token(f"target_{canonical_index}")

    def _first_position(self, seq: Any) -> Optional[list[float]]:
        """Return the first normalized position from a sequence-like value."""
        return self._extract_position(seq, 0)

    def _extract_position(self, seq: Any, index: int) -> Optional[list[float]]:
        """Return a normalized position from ``seq[index]`` when present."""
        if not self._has_index(seq, index):
            return None
        try:
            value = seq[index]
        except (IndexError, KeyError, TypeError):
            return None
        return self._normalize_position(value)

    @staticmethod
    def _has_index(seq: Any, index: int) -> bool:
        """Return whether ``seq`` has a non-negative index."""
        if seq is None or index < 0:
            return False
        try:
            length = len(seq)
        except TypeError:
            return False
        return index < length

    @staticmethod
    def _normalize_position(value: Any) -> Optional[list[float]]:
        """Coerce list-like or scalar positions into a 3-element list."""
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (tuple, list)):
            return [float(v) for v in value[:3]]
        try:
            return [float(value), 0.0, 0.0]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_orientation(value: Any) -> Optional[list[float]]:
        """Coerce orientation-like values to yaw/pitch/roll triples."""
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            if len(value) < 3:
                return None
            return [float(v) for v in value[:3]]
        except (TypeError, ValueError):
            return None
