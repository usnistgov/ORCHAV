"""Antenna beam-pattern controls for frame-backed and standalone previews.

The panel exposes source selection, standalone array parameters, and display
sampling controls. It keeps TX/RX selection read-only because node selection is
owned by the persistent Context controls and synchronized through the controller layer. The
controller layer consumes these widgets and builds the renderer payloads.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..beamforming.extensions import registered_beamforming_modes
from ..beamforming.visualization import _get_colormap_lut
from ..utils.antenna_utils import spacing_m_to_wavelengths
from .base import BasePanel

# Keep the largest GUI-selectable preview below roughly 17 million complex
# element/direction samples.  The vectorized array-factor implementation builds
# several arrays of this shape, so the old 64x64 at 720x361 maximum could exhaust
# system memory before the renderer received a mesh.
MAX_STANDALONE_ARRAY_DIMENSION = 32
MAX_BEAM_AZIMUTH_SAMPLES = 180
MAX_BEAM_ELEVATION_SAMPLES = 91


def _beam_colormap_gradient_pixmap(colormap: str, width: int = 160, height: int = 12) -> QPixmap:
    """Create a horizontal gradient preview for the selected beam colormap."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, width, 0)
    lut = _get_colormap_lut(colormap or "jet", 16)
    for idx, (r, g, b) in enumerate(lut):
        position = idx / max(1, len(lut) - 1)
        gradient.setColorAt(position, QColor(int(r * 255), int(g * 255), int(b * 255)))
    painter.fillRect(0, 0, width, height, gradient)
    painter.end()
    return pixmap


class BeamPatternPanel(BasePanel):
    """Build the Antennas-tab controls for beam-pattern rendering."""

    def __init__(self, parent_widget):
        """Initialize containers whose visibility depends on beam mode."""
        self._parameters_container: QWidget | None = None
        self._array_container: QWidget | None = None
        self._steering_container: QWidget | None = None
        self._angle_container: QWidget | None = None
        super().__init__(parent_widget)

    def create_panel(self):
        """Create source, parameter, and display groups for beam patterns."""
        group = self.create_group_box("Antennas")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self._create_source_group())
        self._parameters_container = self._create_parameters_group()
        self._array_container = self._parameters_container
        layout.addWidget(self._parameters_container)
        layout.addWidget(self._create_display_group())
        self._update_standalone_visibility()
        self._connect_beam_colorbar_updates()
        self.update_beam_colorbar()
        return group

    # Pattern-source controls.

    def _create_source_group(self) -> QWidget:
        """Create beam source controls and read-only TX/RX selection labels."""
        group = self.create_subgroup_box("Pattern Source")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.widgets["beamforming_cb"] = QCheckBox("Show Beam Patterns")
        self.widgets["beamforming_cb"].setChecked(False)
        self.widgets["beamforming_cb"].setToolTip(
            "Display 3D radiation patterns showing antenna directivity"
        )
        layout.addWidget(self.widgets["beamforming_cb"])

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.widgets["mode_frame"] = QRadioButton("Frame Data")
        frame_available = bool(getattr(self.parent, "_frame_beamforming_available", False))
        self.widgets["mode_frame"].setEnabled(frame_available)
        self.widgets["mode_frame"].setToolTip(
            "Advanced: use beamforming weights stored in loaded or streamed frame metadata"
            if frame_available
            else "Unavailable: loaded frames do not contain beamforming metadata"
        )
        self.widgets["mode_standalone"] = QRadioButton("Standalone")
        self.widgets["mode_standalone"].setToolTip("Define custom antenna array parameters")
        # Extension modes come from optional beamforming integrations; this
        # panel only creates their selectors.
        optional_modes = registered_beamforming_modes()
        optional_mode_widgets: dict[str, QRadioButton] = {}
        for extension in optional_modes:
            widget_key = f"mode_optional_{extension.key}"
            button = QRadioButton(extension.label)
            button.setToolTip(extension.tooltip)
            self.widgets[widget_key] = button
            optional_mode_widgets[extension.key] = button
        mode_group = QButtonGroup(group)
        mode_group.addButton(self.widgets["mode_frame"], 0)
        mode_group.addButton(self.widgets["mode_standalone"], 1)
        for index, button in enumerate(optional_mode_widgets.values(), start=2):
            mode_group.addButton(button, index)
        mode_default = (
            getattr(self.parent.app_state, "standalone_beamforming_mode", "standalone")
            if hasattr(self.parent, "app_state")
            else "standalone"
        )
        if mode_default == "standalone" or (mode_default == "frame" and not frame_available):
            self.widgets["mode_standalone"].setChecked(True)
        elif mode_default in optional_mode_widgets:
            optional_mode_widgets[mode_default].setChecked(True)
        else:
            self.widgets["mode_frame"].setChecked(True)
        mode_row.addWidget(self.widgets["mode_frame"])
        mode_row.addWidget(self.widgets["mode_standalone"])
        for button in optional_mode_widgets.values():
            mode_row.addWidget(button)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self.widgets["mode_standalone"].toggled.connect(self._update_standalone_visibility)

        txrx_row = QHBoxLayout()
        txrx_row.setSpacing(8)
        txrx_row.addWidget(QLabel("TX:"))
        self.widgets["beam_tx_selector"] = QLabel("N/A")
        self.widgets["beam_tx_selector"].setMinimumWidth(80)
        self.widgets["beam_tx_selector"].setStyleSheet("QLabel { font-weight: 600; }")
        self.widgets["beam_tx_selector"].setToolTip(
            "Effective transmitter for beam pattern rendering, read from Context"
        )
        txrx_row.addWidget(self.widgets["beam_tx_selector"])
        txrx_row.addWidget(QLabel("RX:"))
        self.widgets["beam_rx_selector"] = QLabel("N/A")
        self.widgets["beam_rx_selector"].setMinimumWidth(80)
        self.widgets["beam_rx_selector"].setStyleSheet("QLabel { font-weight: 600; }")
        self.widgets["beam_rx_selector"].setToolTip(
            "Effective receiver for beam pattern rendering, read from Context"
        )
        txrx_row.addWidget(self.widgets["beam_rx_selector"])
        txrx_row.addStretch()
        layout.addLayout(txrx_row)

        self.widgets["beam_status_label"] = QLabel(
            "Select one TX and one RX to render beam patterns"
        )
        self.widgets["beam_status_label"].setStyleSheet(
            "QLabel { color: #555555; font-size: 10px; }"
        )
        self.widgets["beam_status_label"].setWordWrap(True)
        layout.addWidget(self.widgets["beam_status_label"])

        return group

    # Standalone beamforming parameters.

    def _create_parameters_group(self) -> QWidget:
        """Create standalone array and steering controls for beam patterns."""
        group = self.create_subgroup_box("Shared TX/RX Parameters")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        if hasattr(self.parent, "app_state"):
            state = self.parent.app_state
            rows_default = getattr(state, "standalone_antenna_rows", 1)
            cols_default = getattr(state, "standalone_antenna_cols", 1)
            h_spacing_default = getattr(state, "standalone_horizontal_spacing_m", 0.00535343675)
            v_spacing_default = getattr(state, "standalone_vertical_spacing_m", 0.00535343675)
            freq_default = getattr(state, "standalone_carrier_frequency_ghz", 28.0)
        else:
            rows_default = cols_default = 1
            h_spacing_default = v_spacing_default = 0.00535343675
            freq_default = 28.0
        # AppState stores spacing in meters; the panel edits it in wavelengths
        # because array design is usually tuned relative to carrier frequency.
        h_spacing_lambda_default = spacing_m_to_wavelengths(h_spacing_default, freq_default)
        v_spacing_lambda_default = spacing_m_to_wavelengths(v_spacing_default, freq_default)
        if hasattr(self.parent, "app_state"):
            strategy_default = getattr(state, "standalone_steering_strategy", "svd")
            az_default = getattr(state, "standalone_azimuth_deg", 0.0)
            el_default = getattr(state, "standalone_elevation_deg", 0.0)
        else:
            strategy_default = "svd"
            az_default = el_default = 0.0

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Rows:"))
        self.widgets["standalone_rows"] = QSpinBox()
        self.widgets["standalone_rows"].setRange(1, MAX_STANDALONE_ARRAY_DIMENSION)
        self.widgets["standalone_rows"].setKeyboardTracking(False)
        self.widgets["standalone_rows"].setValue(rows_default)
        self.widgets["standalone_rows"].setToolTip(
            "Shared TX/RX vertical element count (maximum 32 to bound preview memory)"
        )
        row1.addWidget(self.widgets["standalone_rows"])
        row1.addWidget(QLabel("Cols:"))
        self.widgets["standalone_cols"] = QSpinBox()
        self.widgets["standalone_cols"].setRange(1, MAX_STANDALONE_ARRAY_DIMENSION)
        self.widgets["standalone_cols"].setKeyboardTracking(False)
        self.widgets["standalone_cols"].setValue(cols_default)
        self.widgets["standalone_cols"].setToolTip(
            "Shared TX/RX horizontal element count (maximum 32 to bound preview memory)"
        )
        row1.addWidget(self.widgets["standalone_cols"])
        row1.addWidget(QLabel("Carrier:"))
        self.widgets["standalone_freq"] = QDoubleSpinBox()
        self.widgets["standalone_freq"].setRange(0.1, 100.0)
        self.widgets["standalone_freq"].setDecimals(1)
        self.widgets["standalone_freq"].setKeyboardTracking(False)
        self.widgets["standalone_freq"].setSuffix(" GHz")
        self.widgets["standalone_freq"].setValue(freq_default)
        self.widgets["standalone_freq"].setToolTip(
            "Carrier frequency in GHz used for standalone beam-pattern inspection"
        )
        row1.addWidget(self.widgets["standalone_freq"])
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("H Spacing:"))
        self.widgets["standalone_h_spacing"] = QDoubleSpinBox()
        self.widgets["standalone_h_spacing"].setRange(0.05, 2.0)
        self.widgets["standalone_h_spacing"].setDecimals(2)
        self.widgets["standalone_h_spacing"].setKeyboardTracking(False)
        self.widgets["standalone_h_spacing"].setSingleStep(0.05)
        self.widgets["standalone_h_spacing"].setSuffix(" lambda")
        self.widgets["standalone_h_spacing"].setValue(h_spacing_lambda_default)
        self.widgets["standalone_h_spacing"].setToolTip(
            "Horizontal element spacing in wavelengths. 0.5 lambda is the usual starting point; larger values, especially near 1.0 lambda, can create grating lobes."
        )
        row2.addWidget(self.widgets["standalone_h_spacing"])
        row2.addWidget(QLabel("V Spacing:"))
        self.widgets["standalone_v_spacing"] = QDoubleSpinBox()
        self.widgets["standalone_v_spacing"].setRange(0.05, 2.0)
        self.widgets["standalone_v_spacing"].setDecimals(2)
        self.widgets["standalone_v_spacing"].setKeyboardTracking(False)
        self.widgets["standalone_v_spacing"].setSingleStep(0.05)
        self.widgets["standalone_v_spacing"].setSuffix(" lambda")
        self.widgets["standalone_v_spacing"].setValue(v_spacing_lambda_default)
        self.widgets["standalone_v_spacing"].setToolTip(
            "Vertical element spacing in wavelengths. 0.5 lambda is the usual starting point; larger values, especially near 1.0 lambda, can create grating lobes."
        )
        row2.addWidget(self.widgets["standalone_v_spacing"])
        row2.addStretch()
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Strategy:"))
        self.widgets["standalone_strategy"] = QComboBox()
        self.widgets["standalone_strategy"].addItems(
            ["SVD (Current MPCs)", "LOS Steering", "Manual Steering"]
        )
        if strategy_default == "los":
            self.widgets["standalone_strategy"].setCurrentIndex(1)
        elif strategy_default == "manual":
            self.widgets["standalone_strategy"].setCurrentIndex(2)
        else:
            self.widgets["standalone_strategy"].setCurrentIndex(0)
        self.widgets["standalone_strategy"].setToolTip(
            "SVD: maximum-gain beams from the selected pair's current MPC channel "
            "(with LOS fallback when no MPCs are available); LOS: steer toward the peer; "
            "Manual: use the shared TX/RX angles"
        )
        self.widgets["standalone_strategy"].currentIndexChanged.connect(
            self._update_angle_visibility
        )
        row3.addWidget(self.widgets["standalone_strategy"])

        self._angle_container = QWidget()
        angle_layout = QHBoxLayout(self._angle_container)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.addWidget(QLabel("Azimuth:"))
        self.widgets["standalone_azimuth"] = QDoubleSpinBox()
        self.widgets["standalone_azimuth"].setRange(-180.0, 180.0)
        self.widgets["standalone_azimuth"].setDecimals(1)
        self.widgets["standalone_azimuth"].setKeyboardTracking(False)
        self.widgets["standalone_azimuth"].setValue(az_default)
        self.widgets["standalone_azimuth"].setToolTip(
            "Manual steering azimuth angle in degrees (-180 to 180, 0=forward)"
        )
        angle_layout.addWidget(self.widgets["standalone_azimuth"])
        angle_layout.addWidget(QLabel("Elevation:"))
        self.widgets["standalone_elevation"] = QDoubleSpinBox()
        self.widgets["standalone_elevation"].setRange(-90.0, 90.0)
        self.widgets["standalone_elevation"].setDecimals(1)
        self.widgets["standalone_elevation"].setKeyboardTracking(False)
        self.widgets["standalone_elevation"].setValue(el_default)
        self.widgets["standalone_elevation"].setToolTip(
            "Manual steering elevation angle in degrees (-90 to 90, 0=horizontal)"
        )
        angle_layout.addWidget(self.widgets["standalone_elevation"])
        angle_layout.addStretch()
        row3.addWidget(self._angle_container)
        row3.addStretch()
        layout.addLayout(row3)
        self._update_angle_visibility()

        return group

    # Beam-pattern display controls.

    def _create_display_group(self) -> QWidget:
        """Create sampling, scale, colormap, and element-pattern controls."""
        group = self.create_subgroup_box("Display")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        if hasattr(self.parent, "app_state"):
            state = self.parent.app_state
            az_default = getattr(state, "beamforming_azimuth_samples", 72)
            el_default = getattr(state, "beamforming_elevation_samples", 37)
            tx_default = getattr(state, "beamforming_tx_scale", 1.5)
            rx_default = getattr(state, "beamforming_rx_scale", 1.5)
            db_default = getattr(state, "beamforming_db_scale", False)
            dr_default = getattr(state, "beamforming_dynamic_range_db", 40.0)
            cmap_default = getattr(state, "beamforming_colormap", "jet")
            tx_elem_default = getattr(state, "beamforming_tx_element_pattern", "isotropic")
            rx_elem_default = getattr(state, "beamforming_rx_element_pattern", "isotropic")
        else:
            az_default = 72
            el_default = 37
            tx_default = rx_default = 1.5
            db_default = False
            dr_default = 40.0
            cmap_default = "jet"
            tx_elem_default = rx_elem_default = "isotropic"

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Az:"))
        self.widgets["beam_azimuth_spin"] = QSpinBox()
        self.widgets["beam_azimuth_spin"].setRange(12, MAX_BEAM_AZIMUTH_SAMPLES)
        self.widgets["beam_azimuth_spin"].setSingleStep(12)
        self.widgets["beam_azimuth_spin"].setKeyboardTracking(False)
        self.widgets["beam_azimuth_spin"].setValue(int(az_default))
        self.widgets["beam_azimuth_spin"].setToolTip(
            "Azimuth sample count (maximum 180 to bound preview memory)"
        )
        row1.addWidget(self.widgets["beam_azimuth_spin"])
        row1.addWidget(QLabel("El:"))
        self.widgets["beam_elevation_spin"] = QSpinBox()
        self.widgets["beam_elevation_spin"].setRange(9, MAX_BEAM_ELEVATION_SAMPLES)
        self.widgets["beam_elevation_spin"].setSingleStep(4)
        self.widgets["beam_elevation_spin"].setKeyboardTracking(False)
        self.widgets["beam_elevation_spin"].setValue(int(el_default))
        self.widgets["beam_elevation_spin"].setToolTip(
            "Elevation sample count (maximum 91 to bound preview memory)"
        )
        row1.addWidget(self.widgets["beam_elevation_spin"])
        row1.addWidget(QLabel("TX Scale:"))
        self.widgets["beam_tx_scale_spin"] = QDoubleSpinBox()
        self.widgets["beam_tx_scale_spin"].setDecimals(2)
        self.widgets["beam_tx_scale_spin"].setRange(0.1, 10.0)
        self.widgets["beam_tx_scale_spin"].setSingleStep(0.1)
        self.widgets["beam_tx_scale_spin"].setKeyboardTracking(False)
        self.widgets["beam_tx_scale_spin"].setValue(float(tx_default))
        self.widgets["beam_tx_scale_spin"].setToolTip("Visual scale for TX pattern")
        row1.addWidget(self.widgets["beam_tx_scale_spin"])
        row1.addWidget(QLabel("RX Scale:"))
        self.widgets["beam_rx_scale_spin"] = QDoubleSpinBox()
        self.widgets["beam_rx_scale_spin"].setDecimals(2)
        self.widgets["beam_rx_scale_spin"].setRange(0.1, 10.0)
        self.widgets["beam_rx_scale_spin"].setSingleStep(0.1)
        self.widgets["beam_rx_scale_spin"].setKeyboardTracking(False)
        self.widgets["beam_rx_scale_spin"].setValue(float(rx_default))
        self.widgets["beam_rx_scale_spin"].setToolTip("Visual scale for RX pattern")
        row1.addWidget(self.widgets["beam_rx_scale_spin"])
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.widgets["beam_db_scale_cb"] = QCheckBox("dB Scale")
        self.widgets["beam_db_scale_cb"].setChecked(db_default)
        self.widgets["beam_db_scale_cb"].setToolTip("Display pattern in dB scale")
        row2.addWidget(self.widgets["beam_db_scale_cb"])
        self.widgets["beam_dynamic_range_label"] = QLabel("Dynamic Range:")
        row2.addWidget(self.widgets["beam_dynamic_range_label"])
        self.widgets["beam_dynamic_range"] = QDoubleSpinBox()
        self.widgets["beam_dynamic_range"].setRange(10.0, 80.0)
        self.widgets["beam_dynamic_range"].setSingleStep(5.0)
        self.widgets["beam_dynamic_range"].setDecimals(0)
        self.widgets["beam_dynamic_range"].setKeyboardTracking(False)
        self.widgets["beam_dynamic_range"].setValue(dr_default)
        self.widgets["beam_dynamic_range"].setSuffix(" dB")
        self.widgets["beam_dynamic_range"].setToolTip("Dynamic range floor below peak")
        row2.addWidget(self.widgets["beam_dynamic_range"])
        row2.addWidget(QLabel("Colormap:"))
        self.widgets["beam_colormap"] = QComboBox()
        self.widgets["beam_colormap"].addItems(["jet", "viridis", "hot", "coolwarm"])
        idx = self.widgets["beam_colormap"].findText(cmap_default)
        if idx >= 0:
            self.widgets["beam_colormap"].setCurrentIndex(idx)
        self.widgets["beam_colormap"].setToolTip("Colormap for beam pattern amplitude")
        row2.addWidget(self.widgets["beam_colormap"])
        row2.addStretch()
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("TX Element:"))
        self.widgets["beam_tx_element_pattern"] = QComboBox()
        self.widgets["beam_tx_element_pattern"].addItems(["isotropic", "dipole", "tr38901"])
        idx_tx = self.widgets["beam_tx_element_pattern"].findText(tx_elem_default)
        if idx_tx >= 0:
            self.widgets["beam_tx_element_pattern"].setCurrentIndex(idx_tx)
        self.widgets["beam_tx_element_pattern"].setToolTip("TX per-element radiation pattern")
        row3.addWidget(self.widgets["beam_tx_element_pattern"])
        row3.addWidget(QLabel("RX Element:"))
        self.widgets["beam_rx_element_pattern"] = QComboBox()
        self.widgets["beam_rx_element_pattern"].addItems(["isotropic", "dipole", "tr38901"])
        idx_rx = self.widgets["beam_rx_element_pattern"].findText(rx_elem_default)
        if idx_rx >= 0:
            self.widgets["beam_rx_element_pattern"].setCurrentIndex(idx_rx)
        self.widgets["beam_rx_element_pattern"].setToolTip("RX per-element radiation pattern")
        row3.addWidget(self.widgets["beam_rx_element_pattern"])
        row3.addStretch()
        layout.addLayout(row3)

        self.widgets["beam_gain_label"] = QLabel("Metrics: \u2014")
        self.widgets["beam_gain_label"].setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        self.widgets["beam_gain_label"].setWordWrap(True)
        self.widgets["beam_gain_label"].setToolTip(
            "Gain, half-power beamwidth, and sidelobe summary when available"
        )
        layout.addWidget(self.widgets["beam_gain_label"])

        self.widgets["beam_complexity_note"] = QLabel(
            "Preview limits: 32 x 32 elements, 180 x 91 samples, "
            "and 8 million combined work items"
        )
        self.widgets["beam_complexity_note"].setStyleSheet(
            "QLabel { color: #777777; font-size: 9px; }"
        )
        self.widgets["beam_complexity_note"].setWordWrap(True)
        self.widgets["beam_complexity_note"].setToolTip(
            "These limits prevent beam-pattern temporary arrays from exhausting system memory"
        )
        layout.addWidget(self.widgets["beam_complexity_note"])

        colorbar = QWidget()
        colorbar_layout = QHBoxLayout(colorbar)
        colorbar_layout.setSpacing(4)
        colorbar_layout.setContentsMargins(0, 0, 0, 0)
        colorbar_layout.addWidget(QLabel("Beam Color:"))
        self.widgets["beam_colorbar_min_label"] = QLabel("")
        self.widgets["beam_colorbar_min_label"].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.widgets["beam_colorbar_min_label"].setFixedWidth(48)
        self.widgets["beam_colorbar_min_label"].setStyleSheet("font-size: 9px; color: #444444;")
        colorbar_layout.addWidget(self.widgets["beam_colorbar_min_label"])
        self.widgets["beam_colorbar_gradient"] = QLabel()
        self.widgets["beam_colorbar_gradient"].setFixedSize(160, 12)
        self.widgets["beam_colorbar_gradient"].setPixmap(
            _beam_colormap_gradient_pixmap(cmap_default, 160, 12)
        )
        self.widgets["beam_colorbar_gradient"].setToolTip(
            "Beam-pattern color scale for the selected colormap"
        )
        colorbar_layout.addWidget(self.widgets["beam_colorbar_gradient"])
        self.widgets["beam_colorbar_max_label"] = QLabel("")
        self.widgets["beam_colorbar_max_label"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.widgets["beam_colorbar_max_label"].setFixedWidth(48)
        self.widgets["beam_colorbar_max_label"].setStyleSheet("font-size: 9px; color: #444444;")
        colorbar_layout.addWidget(self.widgets["beam_colorbar_max_label"])
        colorbar_layout.addStretch()
        self.widgets["beam_colorbar_container"] = colorbar
        layout.addWidget(colorbar)

        return group

    # Visibility helpers.

    def _update_standalone_visibility(self, _checked: bool = False) -> None:
        """Show standalone array config only when Standalone mode is selected."""
        is_standalone = self.widgets.get("mode_standalone")
        visible = is_standalone is not None and is_standalone.isChecked()
        if self._parameters_container is not None:
            self._parameters_container.setVisible(visible)
        if self._steering_container is not None:
            self._steering_container.setVisible(visible)

    def _update_angle_visibility(self, _index: int = 0) -> None:
        """Show steering angles only when Manual strategy is selected."""
        if self._angle_container is None:
            return
        strategy = self.widgets.get("standalone_strategy")
        self._angle_container.setVisible(strategy is not None and strategy.currentIndex() == 2)

    def _connect_beam_colorbar_updates(self) -> None:
        """Keep the local colorbar preview synchronized with display widgets."""
        for widget_key, signal_name in (
            ("beamforming_cb", "stateChanged"),
            ("beam_db_scale_cb", "stateChanged"),
            ("beam_dynamic_range", "valueChanged"),
            ("beam_colormap", "currentTextChanged"),
        ):
            widget = self.widgets.get(widget_key)
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(lambda *_args: self.update_beam_colorbar())

    def update_beam_colorbar(
        self,
        *,
        show_beamforming: bool | None = None,
        db_scale: bool | None = None,
        dynamic_range_db: float | None = None,
        colormap: str | None = None,
    ) -> None:
        """Show and update the beam-pattern colorbar preview."""
        container = self.widgets.get("beam_colorbar_container")
        if container is None:
            return

        if show_beamforming is None:
            checkbox = self.widgets.get("beamforming_cb")
            show_beamforming = bool(checkbox.isChecked()) if checkbox is not None else False
        container.setVisible(bool(show_beamforming))

        if db_scale is None:
            db_checkbox = self.widgets.get("beam_db_scale_cb")
            db_scale = bool(db_checkbox.isChecked()) if db_checkbox is not None else False
        dynamic_range = self.widgets.get("beam_dynamic_range")
        dynamic_range_label = self.widgets.get("beam_dynamic_range_label")
        if dynamic_range is not None:
            dynamic_range.setEnabled(bool(db_scale))
        if dynamic_range_label is not None:
            dynamic_range_label.setEnabled(bool(db_scale))
        if dynamic_range_db is None:
            dr_spin = self.widgets.get("beam_dynamic_range")
            dynamic_range_db = float(dr_spin.value()) if dr_spin is not None else 40.0
        if colormap is None:
            cmap_combo = self.widgets.get("beam_colormap")
            colormap = str(cmap_combo.currentText()) if cmap_combo is not None else "jet"

        gradient = self.widgets.get("beam_colorbar_gradient")
        if gradient is not None:
            gradient.setPixmap(_beam_colormap_gradient_pixmap(colormap, 160, 12))
            gradient.setToolTip(f"Beam-pattern color scale: {colormap}")

        min_label = self.widgets.get("beam_colorbar_min_label")
        max_label = self.widgets.get("beam_colorbar_max_label")
        if db_scale:
            # Beam payloads normalize the peak to 0 dB; dynamic range sets the
            # negative floor shown at the left of this local preview.
            floor = -abs(float(dynamic_range_db))
            min_text = f"{floor:.0f} dB"
            max_text = "0 dB"
        else:
            min_text = "0"
            max_text = "1"
        if min_label is not None:
            min_label.setText(min_text)
        if max_label is not None:
            max_label.setText(max_text)
