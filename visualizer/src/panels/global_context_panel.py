"""Persistent TX/RX scope and viewport-layer context controls."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..renderers.protocol import renderer_capabilities
from .base import BasePanel


class GlobalContextPanel(BasePanel):
    """Own controls whose state affects every visualizer workflow tab."""

    _OPTIONAL_OVERLAYS: tuple[tuple[str, str], ...] = (
        ("show_coverage", "Coverage"),
        ("show_beamforming", "Beams"),
        ("show_rf_xray", "RF X-Ray"),
        ("show_ground_grid", "Grid"),
        ("show_camera_minimap", "Minimap"),
    )

    def __init__(self, parent_widget: Any) -> None:
        """Initialize persistent context presentation state."""
        super().__init__(parent_widget)
        self._filters_active = False
        self._frame_data_available = True

    def create_panel(self) -> QGroupBox:
        """Create the persistent context strip."""
        group = self.create_group_box("Context")
        group.setObjectName("globalContextPanel")
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        stacked = getattr(self.parent, "_layout_profile", "auto") == "capture-workspace"
        layout = QVBoxLayout(group) if stacked else QHBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        scope_widget = self._create_scope_controls()
        layer_widget = self._create_layer_controls()
        layout.addWidget(scope_widget, stretch=1)
        layout.addWidget(layer_widget, stretch=0)

        self.sync_from_state()
        return group

    def _create_scope_controls(self) -> QWidget:
        """Create the authoritative TX/RX scope selectors."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("TX/RX Scope:")
        title.setToolTip(
            "Global communication scope used by paths, analysis, apertures, and antennas."
        )
        layout.addWidget(title)

        layout.addWidget(QLabel("TX"))
        tx_dropdown = QComboBox()
        tx_dropdown.setObjectName("globalTxSelector")
        tx_dropdown.addItem("All TX")
        tx_dropdown.setMinimumWidth(105)
        tx_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        tx_dropdown.setToolTip("Select the transmitter scope used throughout the visualizer.")
        self.widgets["tx_dropdown"] = tx_dropdown
        layout.addWidget(tx_dropdown, stretch=1)

        arrow = QLabel("\u2192")
        arrow.setProperty("role", "secondary")
        layout.addWidget(arrow)

        layout.addWidget(QLabel("RX"))
        rx_dropdown = QComboBox()
        rx_dropdown.setObjectName("globalRxSelector")
        rx_dropdown.addItem("All RX")
        rx_dropdown.setMinimumWidth(105)
        rx_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        rx_dropdown.setToolTip("Select the receiver scope used throughout the visualizer.")
        self.widgets["rx_dropdown"] = rx_dropdown
        layout.addWidget(rx_dropdown, stretch=1)
        return widget

    def _create_layer_controls(self) -> QWidget:
        """Create global layer switches and compact active-state readouts."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        mpc_layer = QCheckBox("MPC")
        mpc_layer.setObjectName("globalMpcLayerToggle")
        mpc_layer.setChecked(True)
        mpc_layer.setToolTip("Master visibility for MPC paths and physical bounce points.")
        self.widgets["mpc_layer_cb"] = mpc_layer
        layout.addWidget(mpc_layer)

        hud = QCheckBox("HUD")
        hud.setObjectName("globalViewportHudToggle")
        hud.setToolTip(
            "Show or hide the complete viewport HUD. Configure its detail and "
            "content categories in Rendering > Viewport HUD."
        )
        self.widgets["viewport_hud_cb"] = hud
        layout.addWidget(hud)

        overlay_status = QLabel("Overlays: none")
        overlay_status.setObjectName("globalOverlayStatus")
        overlay_status.setProperty("role", "secondary")
        self.widgets["overlay_status_label"] = overlay_status
        layout.addWidget(overlay_status)

        filter_status = QLabel("Paths filtered")
        filter_status.setObjectName("globalFilterStatus")
        filter_status.setToolTip(
            "One or more path order, type, material, range, angle, or Top-K filters are active."
        )
        filter_status.setVisible(False)
        self.widgets["filter_status_label"] = filter_status
        layout.addWidget(filter_status)

        frame_status = QLabel("No frame data")
        frame_status.setObjectName("globalFrameStatus")
        frame_status.setProperty("role", "warning")
        frame_status.setVisible(False)
        self.widgets["frame_status_label"] = frame_status
        layout.addWidget(frame_status)
        return widget

    @staticmethod
    def _sync_selection(dropdown: QComboBox, selection: object) -> None:
        """Reflect one canonical selection without emitting user intent."""
        with QSignalBlocker(dropdown):
            if selection == "all":
                dropdown.setCurrentIndex(0)
                return
            try:
                index = dropdown.findData(int(selection))
            except (TypeError, ValueError):
                return
            if index >= 0:
                dropdown.setCurrentIndex(index)

    def sync_from_state(self, state: object | None = None) -> None:
        """Synchronize controls and readouts from canonical application state."""
        if state is None:
            state = getattr(self.parent, "app_state", None)

        if state is not None:
            tx_dropdown = self.widgets.get("tx_dropdown")
            rx_dropdown = self.widgets.get("rx_dropdown")
            if tx_dropdown is not None:
                self._sync_selection(tx_dropdown, getattr(state, "selected_tx", "all"))
            if rx_dropdown is not None:
                self._sync_selection(rx_dropdown, getattr(state, "selected_rx", "all"))

            mpc_layer = self.widgets.get("mpc_layer_cb")
            visibility = getattr(state, "mpc_visibility", None)
            if mpc_layer is not None and visibility is not None:
                with QSignalBlocker(mpc_layer):
                    mpc_layer.setChecked(bool(getattr(visibility, "enabled", True)))

            hud = self.widgets.get("viewport_hud_cb")
            if hud is not None:
                supports_hud = renderer_capabilities(
                    getattr(self.parent, "renderer", None)
                ).viewport_hud
                enabled_value = getattr(state, "viewport_hud_enabled", None)
                enabled = (
                    str(getattr(state, "viewport_hud_mode", "compact")).strip().lower() != "off"
                    if enabled_value is None
                    else bool(enabled_value)
                )
                with QSignalBlocker(hud):
                    hud.setChecked(supports_hud and enabled)
                hud.setEnabled(supports_hud)
                hud.setToolTip(
                    "Show or hide the complete viewport HUD. Configure its detail and "
                    "content categories in Rendering > Viewport HUD."
                    if supports_hud
                    else "The active renderer does not provide a viewport HUD."
                )

        overlays = [
            label
            for field, label in self._OPTIONAL_OVERLAYS
            if state is not None and bool(getattr(state, field, False))
        ]
        if any(
            state is not None and bool(getattr(state, field, False))
            for field in ("show_tx_trajectory", "show_rx_trajectory", "show_target_trajectory")
        ):
            overlays.append("Trajectories")
        if state is not None and (
            bool(getattr(state, "show_aoa_aperture", False))
            or bool(getattr(state, "show_aod_aperture", False))
        ):
            overlays.append("Angular guides")

        overlay_status = self.widgets.get("overlay_status_label")
        if overlay_status is not None:
            overlay_status.setText(
                f"Overlays: {', '.join(overlays)}" if overlays else "Overlays: none"
            )
            overlay_status.setToolTip(
                "Active optional viewport overlays: " + ", ".join(overlays)
                if overlays
                else "No optional viewport overlays are active."
            )

        clip_axes = [
            axis.upper()
            for axis in ("x", "y", "z")
            if state is not None and bool(getattr(state, f"clip_{axis}_enabled", False))
        ]
        filter_status = self.widgets.get("filter_status_label")
        if filter_status is not None:
            status_parts = []
            if self._filters_active:
                status_parts.append("Paths filtered")
            if clip_axes:
                status_parts.append(f"Clip {'/'.join(clip_axes)}")
            filter_status.setText(" \u00b7 ".join(status_parts) if status_parts else "")
            filter_status.setVisible(bool(status_parts))

    def set_filters_active(self, active: bool) -> None:
        """Update the persistent neutral path-filter indicator."""
        self._filters_active = bool(active)
        self.sync_from_state()

    def set_frame_data_available(self, available: bool) -> None:
        """Keep Context visible while disabling frame-dependent controls."""
        self._frame_data_available = bool(available)
        for key in ("tx_dropdown", "rx_dropdown", "mpc_layer_cb"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(self._frame_data_available)
        frame_status = self.widgets.get("frame_status_label")
        if frame_status is not None:
            frame_status.setVisible(not self._frame_data_available)
