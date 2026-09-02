"""Runtime frame override support and overrides-panel synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from shared.frames import StandardMPCFrame
from shared.logging import get_logger

from ..io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    visual_frame_read_request_for_visualizer,
)
from ..services.base import BaseService
from .cache_service import CacheInvalidationScope

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.override_service")


class OverrideService(BaseService):
    """Handles runtime overrides and the overrides panel synchronization."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind override operations to the visualizer frame source and UI."""
        super().__init__()
        self.visualizer = visualizer

    def apply_position_overrides(self, overrides: List[Dict[str, Any]]) -> None:
        """Apply overrides via the frame source and refresh caches/UI."""
        viz = self.visualizer

        if not viz.frame_source or not hasattr(viz.frame_source, "load_frame_with_overrides"):
            logger.warning("Position overrides not supported for this data source")
            return

        current_step = getattr(viz.app_state, "step", viz.animation_step)
        logger.debug(
            "Applying position overrides for step %s: %d overrides", current_step, len(overrides)
        )

        try:
            panel = self._get_override_panel()
            if panel:
                panel.set_busy(True)
                panel.set_status_text("Computing overrides...")

            updated_frame = viz.frame_source.load_frame_with_overrides(current_step, overrides)
            if updated_frame is None:
                logger.error("Failed to compute frame with overrides")
                if panel:
                    panel.set_busy(False)
                    panel.set_status_text("Override computation failed.", error=True)
                return
            if not isinstance(updated_frame, StandardMPCFrame):
                raise TypeError("load_frame_with_overrides() must return StandardMPCFrame or None")

            visual_frame = standard_frame_to_visual_frame(
                updated_frame,
                request=visual_frame_read_request_for_visualizer(viz),
                points_dtype=getattr(
                    getattr(viz, "mpc_core", None),
                    "canon_points_dtype",
                    "float32",
                ),
            )

            viz.cache_service.store_frame(current_step, visual_frame, source="override")
            viz.cache_service.invalidate(
                CacheInvalidationScope.MPC_RENDER_SETTINGS,
                reason="position_override",
            )
            invalidate_canonical_step = getattr(
                viz.cache_service,
                "invalidate_canonical_step",
                None,
            )
            if callable(invalidate_canonical_step):
                invalidate_canonical_step(current_step, reason="position_override")

            self._apply_updated_node_state(updated_frame)
            viz.node_service.update_tx_rx_positions(
                viz.current_tx_positions,
                viz.current_rx_positions,
            )

            self.update_override_panel_with_frame_data(updated_frame)

            viz.force_update_next_frame = True
            viz.schedule_update()

            logger.debug("Position overrides applied successfully")
            if panel:
                panel.set_busy(False)
                panel.set_status_text("Overrides applied successfully.")
        except (ValueError, KeyError, AttributeError, RuntimeError, TypeError) as exc:
            logger.error("Error applying position overrides: %s", exc)
            if panel:
                panel.set_busy(False)
                panel.set_status_text(f"Error: {exc}", error=True)

    def update_override_panel_with_frame_data(self, frame_data: StandardMPCFrame) -> None:
        """Populate the overrides panel from one canonical frame."""
        panel = self._get_override_panel()
        if not panel:
            return

        logger.debug("Override panel: frame_data type: %s", type(frame_data))
        objects_data = self._extract_objects_from_frame(frame_data)
        logger.debug(
            "Override panel: Objects data to send to panel: %s",
            objects_data,
        )
        panel.update_objects(objects_data)

    def update_override_panel_with_current_frame(self, step: int) -> None:
        """Load the processed frame for a step and refresh the overrides panel."""
        viz = self.visualizer
        panel = self._get_override_panel()
        if not panel or not getattr(viz, "pipeline", None):
            return

        logger.debug("Override panel: Updating panel for step %s", step)

        raw_frame = viz.pipeline.load_frame(step)
        if raw_frame is None:
            logger.warning("Override panel: No frame data available for step %s", step)
            return

        objects_data = self._extract_objects_from_dict(raw_frame)
        panel.update_objects(objects_data)

    # Internal helpers
    def _apply_updated_node_state(self, frame: StandardMPCFrame) -> None:
        """Publish the canonical frame's TX/RX position and orientation state."""
        viz = self.visualizer
        viz.current_tx_positions = self._to_list(frame.tx_positions)
        viz.current_rx_positions = self._to_list(frame.rx_positions)
        viz.current_tx_orientations = self._to_list(frame.tx_orientations)
        viz.current_rx_orientations = self._to_list(frame.rx_orientations)

    def _get_override_panel(self):
        """Return the overrides panel if the UI manager has created one."""
        viz = self.visualizer
        if hasattr(viz, "ui_manager") and viz.ui_manager is not None:
            return viz.ui_manager.panels.get("overrides")
        return None

    def _to_list(self, data):
        """Convert array-like frame fields to plain Python lists."""
        if hasattr(data, "tolist"):
            try:
                return data.tolist()
            except (AttributeError, TypeError):
                return list(data)
        return list(data)

    def _extract_objects_from_frame(
        self,
        frame_data: StandardMPCFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract editable object rows from one canonical frame."""
        objects_data: Dict[str, Dict[str, Any]] = {}
        self._add_node_entries(
            objects_data,
            positions=frame_data.tx_positions,
            names=frame_data.tx_names,
            orientations=frame_data.tx_orientations,
            prefix="tx",
            node_type="tx",
        )
        self._add_node_entries(
            objects_data,
            positions=frame_data.rx_positions,
            names=frame_data.rx_names,
            orientations=frame_data.rx_orientations,
            prefix="rx",
            node_type="rx",
        )

        target_data = self._extract_target_data(frame_data)
        if target_data:
            objects_data.update(target_data)

        return objects_data

    def _extract_objects_from_dict(self, raw_frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract editable object rows from a raw frame dictionary."""
        objects_data: Dict[str, Dict[str, Any]] = {}

        if raw_frame.get("tx_positions") is not None:
            self._add_node_entries(
                objects_data,
                positions=raw_frame["tx_positions"],
                names=raw_frame.get("tx_names", []),
                orientations=raw_frame.get("tx_orientations", []),
                prefix="tx",
                node_type="tx",
            )

        if raw_frame.get("rx_positions") is not None:
            self._add_node_entries(
                objects_data,
                positions=raw_frame["rx_positions"],
                names=raw_frame.get("rx_names", []),
                orientations=raw_frame.get("rx_orientations", []),
                prefix="rx",
                node_type="rx",
            )

        if raw_frame.get("targets_metadata"):
            for i, target_meta in enumerate(raw_frame["targets_metadata"]):
                name = target_meta.get("name", f"target_{i+1}")
                position = target_meta.get("current_position", [0, 0, 0])
                orientation = target_meta.get("orientation", [0, 0, 0])
                objects_data[name] = {
                    "type": "target",
                    "position": self._as_float_list(position),
                    "orientation": self._as_float_list(orientation),
                }

        return objects_data

    def _add_node_entries(
        self,
        objects_data: Dict[str, Dict[str, Any]],
        *,
        positions,
        names,
        orientations,
        prefix: str,
        node_type: str,
    ) -> None:
        """Append TX/RX node rows with position and orientation triples."""
        for idx, pos in enumerate(positions):
            name = names[idx] if idx < len(names) else f"{prefix}_{idx+1}"
            orientation = orientations[idx] if idx < len(orientations) else [0.0, 0.0, 0.0]
            objects_data[name] = {
                "type": node_type,
                "position": self._as_float_list(pos),
                "orientation": self._as_float_list(orientation),
            }

    def _extract_target_data(
        self,
        frame_data: StandardMPCFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract editable target rows from canonical target state."""
        objects_data: Dict[str, Dict[str, Any]] = {}
        for index, position in enumerate(frame_data.target_positions_m):
            metadata = frame_data.targets_metadata[index]
            name = str(metadata.get("name") or f"target_{index + 1}")
            orientation = metadata.get("orientation")
            if orientation is None:
                orientation = [0.0, 0.0, 0.0]
            objects_data[name] = {
                "type": "target",
                "position": self._as_float_list(position),
                "orientation": self._as_float_list(orientation),
            }

        return objects_data

    def _as_float_list(self, data) -> List[float]:
        """Coerce a 3-vector into floats, falling back to the origin."""
        try:
            return [float(data[0]), float(data[1]), float(data[2])]
        except (TypeError, ValueError, IndexError):
            return [0.0, 0.0, 0.0]
