"""Beamforming control-state synchronization for Antennas UI widgets."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSignalBlocker

from shared.logging import get_logger

logger = get_logger("orchav.beamforming_ui_controller")


class BeamformingUIController:
    """Keep beamforming AppState, selector labels, and mode controls in sync."""

    def __init__(self, visualizer: Any) -> None:
        """Bind to the active visualizer shell."""
        self.visualizer = visualizer

    def update_resolution_controls(self) -> None:
        """Synchronize beamforming resolution controls with current state."""
        viz = self.visualizer
        if getattr(viz, "beam_azimuth_spin", None):
            with QSignalBlocker(viz.beam_azimuth_spin):
                viz.beam_azimuth_spin.setValue(viz.app_state.beamforming_azimuth_samples)
                viz.beam_azimuth_spin.setEnabled(True)
                viz.beam_azimuth_spin.setReadOnly(False)
        if getattr(viz, "beam_elevation_spin", None):
            with QSignalBlocker(viz.beam_elevation_spin):
                viz.beam_elevation_spin.setValue(viz.app_state.beamforming_elevation_samples)
                viz.beam_elevation_spin.setEnabled(True)
                viz.beam_elevation_spin.setReadOnly(False)
        if getattr(viz, "beam_tx_scale_spin", None):
            with QSignalBlocker(viz.beam_tx_scale_spin):
                viz.beam_tx_scale_spin.setValue(viz.app_state.beamforming_tx_scale)
                viz.beam_tx_scale_spin.setEnabled(True)
                viz.beam_tx_scale_spin.setReadOnly(False)
        if getattr(viz, "beam_rx_scale_spin", None):
            with QSignalBlocker(viz.beam_rx_scale_spin):
                viz.beam_rx_scale_spin.setValue(viz.app_state.beamforming_rx_scale)
                viz.beam_rx_scale_spin.setEnabled(True)
                viz.beam_rx_scale_spin.setReadOnly(False)

        self.apply_selector_state()
        self.update_standalone_buttons_state()

    def update_standalone_buttons_state(self) -> None:
        """Enable or disable standalone helper buttons from current frame state."""

    def set_frame_beamforming_available(self, available: bool) -> None:
        """Enable Frame Data mode only when current frames carry beamforming metadata."""
        viz = self.visualizer
        available = bool(available)
        viz._frame_beamforming_available = available

        frame_button = getattr(viz, "standalone_mode_frame", None)
        standalone_button = getattr(viz, "standalone_mode_standalone", None)
        if frame_button is not None:
            with QSignalBlocker(frame_button):
                frame_button.setEnabled(available)
                frame_button.setToolTip(
                    "Advanced: use beamforming weights stored in loaded or streamed frame metadata"
                    if available
                    else "Unavailable: loaded frames do not contain beamforming metadata"
                )

        if not available and viz.app_state.standalone_beamforming_mode == "frame":
            viz.set_state(standalone_beamforming_mode="standalone")
            if frame_button is not None:
                with QSignalBlocker(frame_button):
                    frame_button.setChecked(False)
            if standalone_button is not None:
                with QSignalBlocker(standalone_button):
                    standalone_button.setChecked(True)

        panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("beam_pattern")
        update_visibility = getattr(panel, "_update_standalone_visibility", None)
        if callable(update_visibility):
            update_visibility()

    def update_node_options(
        self,
        info: Optional[Dict[str, Any]],
        pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Cache latest beamforming metadata and refresh selector controls."""
        viz = self.visualizer
        # A pipeline update has completed even when it produced no beam payload.
        # Always clear the transient flag so the UI cannot remain on Computing.
        viz._beamforming_computing = False
        viz._beamforming_completed_without_result = not bool(info or pairs)
        viz._beamforming_error_message = None
        viz._latest_beamforming_info = info
        info_pairs = info.get("pairs") if isinstance(info, dict) else None
        viz._latest_beamforming_pairs = list(pairs or info_pairs or [])
        self.apply_selector_state()

    def clear_result_metadata(self) -> None:
        """Drop result metadata that belongs to the superseded node pair."""
        viz = self.visualizer
        viz._latest_beamforming_info = None
        viz._latest_beamforming_pairs = []
        viz._beamforming_tx_nodes = []
        viz._beamforming_rx_nodes = []
        viz._beamforming_computing = False
        viz._beamforming_completed_without_result = False
        viz._beamforming_error_message = None

    def begin_computation(self) -> None:
        """Show a pending result for one visible, concrete TX/RX pair."""
        viz = self.visualizer
        pair_ready = viz.app_state.selected_tx not in {
            "all",
            None,
        } and viz.app_state.selected_rx not in {"all", None}
        viz._beamforming_computing = bool(viz.app_state.show_beamforming and pair_ready)
        viz._beamforming_completed_without_result = False
        viz._beamforming_error_message = None
        self.apply_selector_state()

    def fail_computation(self, reason: str) -> None:
        """End a pending beam update that cannot reach result synchronization."""
        viz = self.visualizer
        if not getattr(viz, "_beamforming_computing", False):
            return
        viz._beamforming_computing = False
        viz._beamforming_completed_without_result = False
        viz._beamforming_error_message = str(reason).strip() or "Frame update failed"
        self.apply_selector_state()

    def apply_selector_state(self) -> None:
        """Synchronize read-only beamforming node displays with Context selection."""
        viz = self.visualizer
        tx_widget = getattr(viz, "beam_tx_selector", None)
        rx_widget = getattr(viz, "beam_rx_selector", None)
        if tx_widget is None or rx_widget is None:
            return

        def _node_name(value: object, prefix: str) -> str:
            """Return a normalized one-based TX/RX node display name."""
            if isinstance(value, str) and value.startswith(f"{prefix}_"):
                return value
            return f"{prefix}_{int(value) + 1}"

        def _names_from_pairs(
            pairs_meta: list[dict[str, Any]], key: str, index_key: str
        ) -> list[str]:
            """Extract sorted TX/RX display names from beamforming pair metadata."""
            names: list[str] = []
            prefix = "tx" if key.startswith("tx") else "rx"
            for entry in pairs_meta:
                value = entry.get(key) or entry.get(index_key)
                if value is None:
                    continue
                try:
                    names.append(_node_name(value, prefix))
                except (TypeError, ValueError):
                    names.append(str(value))
            return sorted(set(names))

        info = getattr(viz, "_latest_beamforming_info", None) or {}
        pairs_meta = info.get("pairs") or getattr(viz, "_latest_beamforming_pairs", None) or []
        available_tx = list(info.get("available_tx_nodes") or [])
        available_rx = list(info.get("available_rx_nodes") or [])
        if pairs_meta:
            if not available_tx:
                available_tx = _names_from_pairs(pairs_meta, "tx_name", "tx_index")
            if not available_rx:
                available_rx = _names_from_pairs(pairs_meta, "rx_name", "rx_index")
        if not available_tx:
            available_tx = [f"tx_{int(idx) + 1}" for idx in getattr(viz, "available_tx", [])]
        if not available_rx:
            available_rx = [f"rx_{int(idx) + 1}" for idx in getattr(viz, "available_rx", [])]

        def _from_global_selection(selection: object, prefix: str) -> str | None:
            """Normalize concrete global selection without consulting stale results."""
            if selection not in {"all", None}:
                try:
                    return _node_name(selection, prefix)
                except (TypeError, ValueError):
                    return str(selection)
            return None

        selected_tx_value = viz.app_state.selected_tx
        selected_rx_value = viz.app_state.selected_rx
        desired_tx = _from_global_selection(selected_tx_value, "tx")
        desired_rx = _from_global_selection(selected_rx_value, "rx")

        def _selection_display(selection: object, prefix: str) -> str:
            """Return the Context selector's user-facing text when available."""
            if selection in {"all", None}:
                return f"All {prefix.upper()}"
            dropdown = getattr(viz, f"{prefix}_dropdown", None)
            if dropdown is not None and hasattr(dropdown, "findData"):
                try:
                    item_index = dropdown.findData(int(selection))
                except (TypeError, ValueError):
                    item_index = -1
                if item_index >= 0:
                    return str(dropdown.itemText(item_index))
            try:
                return f"{prefix.upper()}{int(selection) + 1}"
            except (TypeError, ValueError):
                return str(selection)

        display_tx = _selection_display(selected_tx_value, "tx")
        display_rx = _selection_display(selected_rx_value, "rx")

        def _set_node_display(widget: object, text: str) -> None:
            """Write global scope into either label or combo-box widget forms."""
            if hasattr(widget, "setText") and not hasattr(widget, "setCurrentText"):
                widget.setText(text)
                # These labels are read-only context, not unavailable controls.
                # Keep "All TX/RX" legible instead of rendering it disabled.
                widget.setEnabled(True)
                return
            if not all(hasattr(widget, attr) for attr in ("blockSignals", "clear", "addItem")):
                return
            with QSignalBlocker(widget):
                widget.clear()
                widget.addItem(text)
                if hasattr(widget, "setCurrentText"):
                    widget.setCurrentText(text)
                widget.setEnabled(False)

        _set_node_display(tx_widget, display_tx)
        _set_node_display(rx_widget, display_rx)
        viz._beamforming_tx_nodes = list(available_tx)
        viz._beamforming_rx_nodes = list(available_rx)

        state_updates = {}
        requested_tx_node = desired_tx or "auto"
        requested_rx_node = desired_rx or "auto"
        if requested_tx_node != viz.app_state.beamforming_tx_node:
            state_updates["beamforming_tx_node"] = requested_tx_node
        if requested_rx_node != viz.app_state.beamforming_rx_node:
            state_updates["beamforming_rx_node"] = requested_rx_node
        if state_updates:
            viz.set_state(**state_updates)

        pair_metadata = info.get("pairs") or pairs_meta
        result_matches_selection = bool(
            desired_tx
            and desired_rx
            and any(
                entry.get("tx_index") == selected_tx_value
                and entry.get("rx_index") == selected_rx_value
                for entry in pair_metadata
                if isinstance(entry, dict)
            )
        )
        if desired_tx and desired_rx and not result_matches_selection:
            result_matches_selection = bool(
                info.get("requested_tx_index") == selected_tx_value
                and info.get("requested_rx_index") == selected_rx_value
            )
        if not pair_metadata and desired_tx and desired_rx:
            result_matches_selection = bool(
                result_matches_selection
                or (
                    info.get("resolved_tx_node") == desired_tx
                    and info.get("resolved_rx_node") == desired_rx
                )
                or str(info.get("status", "")).startswith("Frame Data unavailable")
            )

        self._update_status_labels(
            info=info if result_matches_selection else {},
            desired_tx=desired_tx,
            desired_rx=desired_rx,
            display_tx=display_tx,
            display_rx=display_rx,
            result_matches_selection=result_matches_selection,
        )

    def _update_status_labels(
        self,
        *,
        info: dict[str, Any],
        desired_tx: str | None,
        desired_rx: str | None,
        display_tx: str,
        display_rx: str,
        result_matches_selection: bool,
    ) -> None:
        """Refresh beam-pattern status and gain metric labels."""
        viz = self.visualizer
        incomplete_status = "Select one TX and one RX to render beam patterns"
        pair_ready = bool(desired_tx and desired_rx)
        computing = bool(getattr(viz, "_beamforming_computing", False))
        completed_without_result = bool(
            getattr(viz, "_beamforming_completed_without_result", False)
        )
        error_message = str(getattr(viz, "_beamforming_error_message", "") or "").strip()
        show_beamforming = bool(getattr(viz.app_state, "show_beamforming", False))
        metrics_info = info
        if not show_beamforming:
            status = "Hidden"
            if pair_ready:
                status = f"Hidden: {display_tx} -> {display_rx}"
            metrics_info = {}
        elif not pair_ready:
            status = incomplete_status
            metrics_info = {}
        elif computing:
            status = f"Computing: {display_tx} -> {display_rx}..."
            metrics_info = {}
        elif error_message:
            status = f"Error: {display_tx} -> {display_rx}. {error_message}"
            metrics_info = {}
        elif completed_without_result:
            status = f"Unavailable: no beam pattern result for {display_tx} -> {display_rx}"
            metrics_info = {}
        elif result_matches_selection:
            raw_status = str(info.get("status") or "").strip()
            status_lower = raw_status.lower()
            if any(marker in status_lower for marker in ("error", "failed", "failure")):
                status = f"Error: {display_tx} -> {display_rx}"
                metrics_info = {}
            elif "partial" in status_lower:
                detail = raw_status.split(":", 1)[-1].strip()
                status = f"Partial: {display_tx} -> {display_rx}. {detail}"
            elif raw_status and any(
                marker in status_lower
                for marker in ("unavailable", "no beamforming", "not found", "could not")
            ):
                status = f"Unavailable: {display_tx} -> {display_rx}"
                metrics_info = {}
            else:
                status = f"Ready: {display_tx} -> {display_rx}"
        else:
            status = f"Waiting for result: {display_tx} -> {display_rx}"
            metrics_info = {}

        sampling_by_role = info.get("sampling_by_role") or {}
        requested_azimuth = int(getattr(viz.app_state, "beamforming_azimuth_samples", 0))
        requested_elevation = int(getattr(viz.app_state, "beamforming_elevation_samples", 0))
        sampling_limited = False
        for samples in sampling_by_role.values():
            if not isinstance(samples, dict):
                continue
            try:
                actual_azimuth = int(samples.get("azimuth", requested_azimuth))
                actual_elevation = int(samples.get("elevation", requested_elevation))
            except (TypeError, ValueError):
                continue
            if actual_azimuth < requested_azimuth or actual_elevation < requested_elevation:
                sampling_limited = True
                break
        if sampling_limited and metrics_info is info:
            status = f"{status}. Sampling limited for memory safety"
        pattern_status = getattr(viz.app_state, "beamforming_pattern_status", "")
        if (
            pattern_status
            and show_beamforming
            and result_matches_selection
            and not computing
            and pattern_status not in status
        ):
            status = f"{status}. {pattern_status}" if status else pattern_status
        status_label = getattr(viz, "beam_status_label", None)
        if status_label is not None:
            status_label.setText(status)

        gain_label = getattr(viz, "beam_gain_label", None)
        if gain_label is not None:
            gain_label.setText(self._format_gain_metrics(metrics_info))

    def _format_gain_metrics(self, info: dict[str, Any]) -> str:
        """Return the compact TX/RX gain metric label text."""
        gain_by_role = info.get("gain_by_role") or {}
        metrics_by_role = info.get("metrics_by_role") or {}

        def _finite_float(value: object) -> float | None:
            """Return a finite float or ``None`` for invalid gain values."""
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None

        def _linear_gain_to_db(value: object) -> float | None:
            """Convert positive linear gain to dB for beamforming labels."""
            gain = _finite_float(value)
            if gain is None or gain <= 0.0:
                return None
            return 10.0 * math.log10(gain)

        parts = []
        for role in ("tx", "rx"):
            role_metrics = metrics_by_role.get(role) or {}
            if not isinstance(role_metrics, dict):
                role_metrics = {}

            role_parts = []
            if role in gain_by_role:
                gain_db = _linear_gain_to_db(gain_by_role[role])
                if gain_db is not None:
                    role_parts.append(f"Gain {gain_db:.1f} dB")
            elif "peak_gain_dbi" in role_metrics:
                peak_gain = _finite_float(role_metrics.get("peak_gain_dbi"))
                if peak_gain is not None:
                    role_parts.append(f"Peak {peak_gain:.1f} dBi")

            hpbw_az = _finite_float(role_metrics.get("hpbw_az_deg"))
            hpbw_el = _finite_float(role_metrics.get("hpbw_el_deg"))
            if hpbw_az is not None and hpbw_el is not None:
                role_parts.append(f"HPBW {hpbw_az:.0f}/{hpbw_el:.0f} deg")

            sll = _finite_float(role_metrics.get("sll_db"))
            if sll is not None:
                role_parts.append(f"SLL {sll:.1f} dB")

            if role_parts:
                parts.append(f"{role.upper()} " + ", ".join(role_parts))

        return "Metrics: " + " | ".join(parts) if parts else "Metrics: \u2014"
