"""Coverage overlay controls for metrics, height slices, and display options.

The panel is deliberately a thin Qt view. Coverage data selection, derived
analysis state, cache invalidation, and rendering remain controller/service
responsibilities.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any, Dict, Optional

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap, QValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..coverage.analysis import (
    compute_coverage_slice_summary,
    compute_serving_tx_coverage_summary,
    coverage_metric_color_scale,
    coverage_metric_colormap,
    coverage_metric_comparator,
    coverage_metric_label,
    default_coverage_threshold,
    format_coverage_slice_summary,
    format_coverage_value,
    format_serving_tx_coverage_summary,
    is_serving_tx_metric,
    serving_tx_color_hex,
    serving_tx_labels,
    supports_coverage_threshold,
)
from .base import BasePanel

logger = get_logger("orchav.coverage_panel")

_SERVING_METRIC_FAMILIES = {
    "serving_path_gain_linear": "path_gain_linear",
    "best_path_loss_db": "path_loss_db",
    "best_rss_dbm": "rss_dbm",
}
_SELECTABLE_SERVING_METRICS = frozenset({"sinr_linear", "sinr_db"})


def _coverage_colormap_gradient_pixmap(
    colormap: str,
    width: int = 180,
    height: int = 12,
) -> QPixmap:
    """Render a compact preview of the semantic coverage colormap."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, width, 0)
    try:
        import matplotlib as mpl

        cmap = mpl.colormaps.get_cmap(colormap)
        colors = [cmap(index / 15.0)[:3] for index in range(16)]
    except (ImportError, KeyError, ValueError, AttributeError):
        # Coverage currently uses the red/yellow/green semantic pair; retain
        # that meaning if matplotlib is unavailable during a minimal Qt run.
        colors = [(0.84, 0.19, 0.15), (1.0, 0.92, 0.48), (0.10, 0.62, 0.33)]
        if str(colormap).endswith("_r"):
            colors.reverse()
    for index, (red, green, blue) in enumerate(colors):
        position = index / max(1, len(colors) - 1)
        gradient.setColorAt(
            position,
            QColor(int(red * 255), int(green * 255), int(blue * 255)),
        )
    painter.fillRect(0, 0, width, height, gradient)
    painter.end()
    return pixmap


class _CoverageThresholdSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that can expose very small linear values accurately."""

    def __init__(self) -> None:
        super().__init__()
        self._scientific = False

    def set_scientific_mode(self, enabled: bool) -> None:
        """Switch between ordinary fixed-point and compact scientific text."""
        self._scientific = bool(enabled)
        self.lineEdit().setText(self.textFromValue(self.value()) + self.suffix())

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        """Format linear path-gain thresholds without rounding them to zero."""
        if self._scientific:
            return f"{float(value):.6g}"
        return super().textFromValue(value)

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt override
        """Accept scientific notation when the active metric is linear."""
        if not self._scientific:
            return super().valueFromText(text)
        value_text = str(text).strip()
        suffix = self.suffix().strip()
        if suffix and value_text.endswith(suffix):
            value_text = value_text[: -len(suffix)].strip()
        try:
            return float(value_text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):  # noqa: N802 - Qt override
        """Permit complete and in-progress scientific-notation edits."""
        if not self._scientific:
            return super().validate(text, pos)
        value_text = str(text).strip()
        if value_text in {"", "+", "-", ".", "+.", "-."}:
            return QValidator.Intermediate, text, pos
        try:
            float(value_text)
        except ValueError:
            lowered = value_text.lower()
            if lowered.endswith(("e", "e+", "e-")):
                return QValidator.Intermediate, text, pos
            return QValidator.Invalid, text, pos
        return QValidator.Acceptable, text, pos


class CoverageMapPanel(BasePanel):
    """Build and synchronize the coverage-map control panel."""

    _INTERPOLATION_LABELS = {
        "none": "Raw",
        "linear": "Smooth",
        "cubic": "Smooth+",
    }

    def __init__(self, parent_widget):
        """Initialize height-slice state tracked by the panel widgets."""
        super().__init__(parent_widget)
        self.heights: list[float] = []
        self.current_height_index = 0
        self._last_coverage_metadata: Optional[Dict[str, Any]] = None
        self._last_threshold_metadata: Optional[Dict[str, Any]] = None
        self._threshold_metric: Optional[str] = None
        self._pending_isoline_count = 6
        self._use_serving_tx = True
        self._selected_tx_name: Optional[str] = None
        self._metric_preference_initialized = False
        self._tx_count: Optional[int] = None
        self._metric_variants: dict[
            str,
            list[tuple[str, Optional[str], bool]],
        ] = {}

    def create_panel(self) -> QGroupBox:
        """Create compact, logically grouped coverage controls."""
        group = self.create_group_box("Coverage Map")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        toggle = QCheckBox("Show coverage map")
        toggle.setAccessibleName("Show coverage map")
        toggle.setEnabled(False)
        toggle.toggled.connect(self._on_coverage_toggled)
        self.widgets["coverage_toggle"] = toggle
        layout.addWidget(toggle)

        status = QLabel(self._no_data_text())
        status.setAccessibleName("Coverage data status")
        status.setWordWrap(True)
        self.widgets["coverage_status"] = status
        self._set_label_emphasis(status, active=False)
        layout.addWidget(status)

        layout.addWidget(self._create_display_group())
        layout.addWidget(self._create_analysis_group())
        self._height_animation_group = self._create_height_animation_group()
        self._height_animation_group.setVisible(False)
        layout.addWidget(self._height_animation_group)
        return group

    def _create_display_group(self) -> QGroupBox:
        """Create selectors and renderer-facing display controls."""
        group = self.create_subgroup_box("Layer")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)

        metric_label = QLabel("&Metric:")
        metric_combo = QComboBox()
        metric_combo.setAccessibleName("Coverage metric")
        metric_combo.setEnabled(False)
        metric_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        metric_combo.setToolTip("Choose the coverage quantity shown on the map")
        metric_combo.currentTextChanged.connect(self._on_metric_changed)
        metric_label.setBuddy(metric_combo)
        self.widgets["coverage_metric_combo"] = metric_combo
        layout.addWidget(metric_label, 0, 0)
        layout.addWidget(metric_combo, 0, 1)

        self._tx_label = QLabel("&Transmitter:")
        tx_combo = QComboBox()
        tx_combo.setAccessibleName("Coverage transmitter")
        tx_combo.setToolTip("Choose the transmitter for this per-transmitter metric")
        tx_combo.currentTextChanged.connect(self._on_tx_changed)
        self._tx_label.setBuddy(tx_combo)
        self.widgets["coverage_tx_combo"] = tx_combo
        layout.addWidget(self._tx_label, 0, 2)
        layout.addWidget(tx_combo, 0, 3)
        self._set_tx_selector_visible(False)

        serving_toggle = QCheckBox("Use serving transmitter")
        serving_toggle.setAccessibleName("Use serving transmitter for coverage metric")
        serving_toggle.setToolTip(
            "Use the transmitter with the strongest received signal at each grid cell"
        )
        serving_toggle.toggled.connect(self._on_serving_mode_changed)
        serving_toggle.setVisible(False)
        self.widgets["coverage_serving_tx_toggle"] = serving_toggle
        layout.addWidget(serving_toggle, 1, 0, 1, 2)

        height_label = QLabel("&Height:")
        height_combo = QComboBox()
        height_combo.setAccessibleName("Coverage height")
        height_combo.setEnabled(False)
        height_combo.setToolTip("Choose the coverage height above the scene datum")
        height_combo.currentIndexChanged.connect(self._on_height_changed)
        height_label.setBuddy(height_combo)
        self.widgets["coverage_height_text"] = height_label
        self.widgets["coverage_height_combo"] = height_combo
        layout.addWidget(height_label, 2, 0)
        layout.addWidget(height_combo, 2, 1)

        smoothing_label = QLabel("&Smoothing:")
        interpolation_combo = QComboBox()
        interpolation_combo.addItems(["Raw", "Smooth", "Smooth+"])
        interpolation_combo.setCurrentText("Raw")
        interpolation_combo.setAccessibleName("Coverage smoothing")
        interpolation_combo.setEnabled(False)
        interpolation_combo.setToolTip("NaN-aware spatial smoothing for scalar coverage data")
        interpolation_combo.currentTextChanged.connect(self._on_interpolation_changed)
        smoothing_label.setBuddy(interpolation_combo)
        self.widgets["coverage_interpolation"] = interpolation_combo
        layout.addWidget(smoothing_label, 2, 2)
        layout.addWidget(interpolation_combo, 2, 3)

        self._opacity_container = QWidget()
        opacity_layout = QHBoxLayout(self._opacity_container)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(6)
        opacity_label = QLabel("&Opacity:")
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setAccessibleName("Coverage opacity")
        opacity_slider.setRange(10, 100)
        opacity_slider.setValue(100)
        opacity_slider.setEnabled(False)
        opacity_slider.setToolTip("Coverage-map opacity (10 to 100 percent)")
        opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_label.setBuddy(opacity_slider)
        opacity_value = QLabel("100%")
        opacity_value.setAccessibleName("Coverage opacity value")
        opacity_value.setMinimumWidth(40)
        self.widgets["coverage_opacity"] = opacity_slider
        self.widgets["opacity_label"] = opacity_value
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(opacity_slider, stretch=1)
        opacity_layout.addWidget(opacity_value)
        layout.addWidget(self._opacity_container, 3, 0, 1, 4)

        self._legend_container = QWidget()
        legend_layout = QHBoxLayout(self._legend_container)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(5)
        legend_title = QLabel("Scale:")
        self.widgets["coverage_legend_title"] = legend_title
        legend_layout.addWidget(legend_title)
        legend_min = QLabel("—")
        legend_min.setAccessibleName("Coverage color scale minimum")
        legend_gradient = QLabel()
        legend_gradient.setAccessibleName("Coverage color scale")
        legend_gradient.setMinimumWidth(120)
        legend_gradient.setMaximumHeight(16)
        legend_gradient.setWordWrap(True)
        legend_max = QLabel("—")
        legend_max.setAccessibleName("Coverage color scale maximum")
        missing_label = QLabel("Missing cells: hidden (transparent)")
        missing_label.setAccessibleName("Coverage missing-cell behavior")
        self.widgets["coverage_legend_min"] = legend_min
        self.widgets["coverage_legend_gradient"] = legend_gradient
        self.widgets["coverage_legend_max"] = legend_max
        self.widgets["coverage_missing_label"] = missing_label
        legend_layout.addWidget(legend_min)
        legend_layout.addWidget(legend_gradient, stretch=1)
        legend_layout.addWidget(legend_max)
        legend_layout.addSpacing(8)
        legend_layout.addWidget(missing_label)
        self._legend_container.setVisible(False)
        layout.addWidget(self._legend_container, 4, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        return group

    def _create_analysis_group(self) -> QGroupBox:
        """Create threshold, mask, and isoline controls."""
        group = self.create_subgroup_box("Analysis")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)

        slice_label = QLabel("Raw selected slice:")
        slice_label.setToolTip(
            "Percentiles use the stored selected slice before display smoothing."
        )
        slice_summary = QLabel("Unavailable")
        slice_summary.setAccessibleName("Selected coverage slice statistics")
        slice_summary.setWordWrap(True)
        self.widgets["coverage_slice_summary"] = slice_summary
        self._set_label_emphasis(slice_summary, active=False)
        layout.addWidget(slice_label, 0, 0)
        layout.addWidget(slice_summary, 0, 1, 1, 4)

        threshold_label = QLabel("Threshold:")
        threshold_toggle = QCheckBox("Enable")
        threshold_toggle.setAccessibleName("Enable coverage threshold")
        threshold_toggle.setEnabled(False)
        threshold_toggle.setToolTip(
            "Measure cells in the displayed slice after any enabled smoothing"
        )
        threshold_toggle.toggled.connect(self._on_threshold_toggled)
        self.widgets["coverage_threshold_toggle"] = threshold_toggle

        comparator = QLabel("")
        comparator.setAccessibleName("Coverage threshold comparator")
        comparator.setMinimumWidth(14)
        self.widgets["coverage_threshold_comparator"] = comparator

        threshold_spin = _CoverageThresholdSpinBox()
        threshold_spin.setAccessibleName("Coverage threshold value")
        threshold_spin.setDecimals(2)
        threshold_spin.setRange(-1.0e12, 1.0e12)
        threshold_spin.setSingleStep(1.0)
        threshold_spin.setEnabled(False)
        threshold_spin.setToolTip("Threshold value for the selected coverage metric")
        threshold_spin.valueChanged.connect(self._on_threshold_value_changed)
        self.widgets["coverage_threshold_value"] = threshold_spin

        mask_toggle = QCheckBox("Dim outside")
        mask_toggle.setAccessibleName("Dim cells outside coverage threshold")
        mask_toggle.setEnabled(False)
        mask_toggle.setToolTip("Dim coverage cells that do not satisfy the threshold")
        mask_toggle.toggled.connect(self._on_threshold_mask_toggled)
        self.widgets["coverage_threshold_mask_toggle"] = mask_toggle

        layout.addWidget(threshold_label, 1, 0)
        layout.addWidget(threshold_toggle, 1, 1)
        layout.addWidget(comparator, 1, 2)
        layout.addWidget(threshold_spin, 1, 3)
        layout.addWidget(mask_toggle, 1, 4)

        summary = QLabel("Off")
        summary.setAccessibleName("Coverage threshold summary")
        summary.setWordWrap(True)
        self.widgets["coverage_threshold_summary"] = summary
        self._set_label_emphasis(summary, active=False)
        layout.addWidget(summary, 2, 1, 1, 4)

        isoline_label = QLabel("Contours:")
        isolines_toggle = QCheckBox("Show")
        isolines_toggle.setAccessibleName("Show coverage contours")
        isolines_toggle.setEnabled(False)
        isolines_toggle.setToolTip(
            "Draw contours for the active scalar metric; complex grids may take a moment"
        )
        isolines_toggle.toggled.connect(self._on_isolines_toggled)
        self.widgets["coverage_isolines_toggle"] = isolines_toggle

        levels_label = QLabel("&Levels:")
        isoline_count = QSpinBox()
        isoline_count.setAccessibleName("Coverage contour levels")
        isoline_count.setRange(2, 12)
        isoline_count.setValue(6)
        isoline_count.setEnabled(False)
        isoline_count.setToolTip("Number of evenly spaced contour levels")
        isoline_count.valueChanged.connect(self._on_isoline_count_changed)
        levels_label.setBuddy(isoline_count)
        self.widgets["coverage_isoline_count"] = isoline_count

        self._isoline_count_timer = QTimer(isoline_count)
        self._isoline_count_timer.setSingleShot(True)
        self._isoline_count_timer.setInterval(180)
        self._isoline_count_timer.timeout.connect(self._flush_isoline_count_change)

        layout.addWidget(isoline_label, 3, 0)
        layout.addWidget(isolines_toggle, 3, 1)
        layout.addWidget(levels_label, 3, 2)
        layout.addWidget(isoline_count, 3, 3)
        layout.setColumnStretch(3, 1)
        return group

    def _create_height_animation_group(self) -> QGroupBox:
        """Create playback controls shown only for multi-height data."""
        group = self.create_subgroup_box("Height playback")
        layout = QHBoxLayout(group)
        layout.setSpacing(6)

        play_btn = QPushButton("Play heights")
        play_btn.setAccessibleName("Play or pause coverage heights")
        play_btn.setEnabled(False)
        play_btn.setToolTip("Animate through all coverage heights; click again to pause")
        play_btn.clicked.connect(self._on_height_animation_play_clicked)
        self.widgets["coverage_height_play_btn"] = play_btn
        layout.addWidget(play_btn)

        stop_btn = QPushButton("Stop")
        stop_btn.setAccessibleName("Stop coverage height playback")
        stop_btn.setEnabled(False)
        stop_btn.setToolTip("Stop height playback")
        stop_btn.clicked.connect(self._on_height_animation_stop_clicked)
        self.widgets["coverage_height_stop_btn"] = stop_btn
        layout.addWidget(stop_btn)

        speed_label = QLabel("&Interval:")
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setAccessibleName("Coverage height playback interval")
        speed_slider.setRange(1, 10)
        speed_slider.setValue(3)
        speed_slider.setEnabled(False)
        speed_slider.setToolTip("Time between height changes, from 2.00 s to 0.20 s")
        speed_slider.valueChanged.connect(self._on_height_animation_speed_changed)
        speed_label.setBuddy(speed_slider)
        self.widgets["coverage_height_speed"] = speed_slider
        layout.addWidget(speed_label)
        layout.addWidget(speed_slider, stretch=1)

        interval_label = QLabel(self._animation_interval_text(3))
        interval_label.setAccessibleName("Coverage height playback interval value")
        interval_label.setMinimumWidth(48)
        self.widgets["coverage_height_speed_label"] = interval_label
        layout.addWidget(interval_label)
        return group

    # --- synchronization helpers --------------------------------------
    @staticmethod
    def _no_data_text() -> str:
        return (
            "No coverage data is available. Enable coverage in scenario.yaml "
            "and regenerate the scenario."
        )

    @staticmethod
    def _set_label_emphasis(label: QLabel, *, active: bool) -> None:
        """Use theme-neutral font emphasis for status labels."""
        font = label.font()
        font.setBold(bool(active))
        font.setItalic(not active)
        label.setFont(font)

    @staticmethod
    def _animation_interval_text(speed: int) -> str:
        return f"{2.0 / max(1, int(speed)):.2f} s"

    @staticmethod
    def _metric_display_text(metric: Any) -> str:
        label, unit = coverage_metric_label(metric)
        return f"{label} ({unit})" if unit else label

    @staticmethod
    def _metadata_heights(metadata: Optional[Dict[str, Any]]) -> Any:
        if not metadata:
            return []
        heights = metadata.get("heights")
        if heights is None:
            nested = metadata.get("metadata")
            if isinstance(nested, dict):
                heights = nested.get("heights")
        return [] if heights is None else heights

    @staticmethod
    def _format_heights(raw_heights: Any) -> Optional[str]:
        if raw_heights is None:
            return None
        if hasattr(raw_heights, "tolist"):
            raw_heights = raw_heights.tolist()
        try:
            values = [float(height) for height in raw_heights]
        except (TypeError, ValueError):
            return None
        if not values:
            return None
        if len(values) == 1:
            return f"{values[0]:.2f} m"
        if len(values) <= 3:
            return ", ".join(f"{height:.2f}" for height in values) + " m"
        return f"{values[0]:.2f}…{values[-1]:.2f} m ({len(values)} levels)"

    def update_coverage_status(
        self,
        has_data: bool,
        metadata: Optional[Dict[str, Any]] = None,
        supports_transparency: bool = True,
    ) -> None:
        """Refresh availability, metadata text, and enabled controls."""
        if has_data and metadata:
            self._last_coverage_metadata = metadata
            metric = str(metadata.get("metric_name", "coverage"))
            label, unit = coverage_metric_label(metric)
            raw_grid_shape = metadata.get("grid_shape")
            grid_shape = list(raw_grid_shape) if raw_grid_shape is not None else [0, 0, 0]
            grid_shape = (grid_shape + [0, 0, 0])[:3]
            value_min = format_coverage_value(metadata.get("value_min"), unit)
            value_max = format_coverage_value(metadata.get("value_max"), unit)
            range_unit = f" {unit}" if unit and unit != "index" else ""
            status_text = (
                f"{label} · {grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]} cells · "
                f"{value_min}–{value_max}{range_unit}"
            )
            heights = self._metadata_heights(metadata)
            heights_text = self._format_heights(heights)
            if heights_text:
                status_text += f" · Heights: {heights_text}"
            status = self.widgets["coverage_status"]
            status.setText(status_text)
            self._set_label_emphasis(status, active=True)
            self.widgets["coverage_toggle"].setEnabled(True)
            self.widgets["coverage_opacity"].setEnabled(True)
            tx_names = metadata.get("tx_names")
            try:
                tx_count = len(tx_names) if tx_names is not None else 0
            except TypeError:
                tx_count = 0
            if tx_count <= 0:
                try:
                    tx_count = int(metadata.get("tx_count", metadata.get("serving_tx_count", 0)))
                except (TypeError, ValueError):
                    tx_count = 0
            self.set_metrics(
                metadata.get("available_metrics") or [metric],
                metric,
                tx_count=tx_count or None,
            )
            self.set_heights(heights)
            self.configure_metric_controls(True, metric)
            self.configure_threshold_controls(True, metadata)
            self._update_scalar_legend(metric, metadata)
            self.update_slice_summary(metadata)
        else:
            self._last_coverage_metadata = None
            status = self.widgets["coverage_status"]
            status.setText(self._no_data_text())
            self._set_label_emphasis(status, active=False)
            self.widgets["coverage_toggle"].setEnabled(False)
            self.widgets["coverage_toggle"].setChecked(False)
            self.widgets["coverage_opacity"].setEnabled(False)
            self.set_metrics([], None, tx_count=0)
            self.set_heights([])
            self.configure_metric_controls(False, None)
            self.configure_threshold_controls(False, None)
            self.update_slice_summary(None)
            self._legend_container.setVisible(False)
        self.set_renderer_capabilities(supports_transparency=supports_transparency)

    def reset_view_state(self) -> None:
        """Restore deterministic coverage-widget defaults without callbacks."""
        self._isoline_count_timer.stop()
        self._pending_isoline_count = 6
        reset_values = (
            ("coverage_toggle", "setChecked", False),
            ("coverage_opacity", "setValue", 100),
            ("coverage_interpolation", "setCurrentText", "Raw"),
            ("coverage_serving_tx_toggle", "setChecked", False),
            ("coverage_threshold_toggle", "setChecked", False),
            ("coverage_threshold_value", "setValue", 0.0),
            ("coverage_threshold_mask_toggle", "setChecked", False),
            ("coverage_isolines_toggle", "setChecked", False),
            ("coverage_isoline_count", "setValue", 6),
            ("coverage_height_speed", "setValue", 3),
        )
        for key, method_name, value in reset_values:
            widget = self.widgets.get(key)
            if widget is None:
                continue
            with QSignalBlocker(widget):
                getattr(widget, method_name)(value)
        self.widgets["opacity_label"].setText("100%")
        self.widgets["coverage_height_speed_label"].setText(self._animation_interval_text(3))
        self.set_metrics([], None, tx_count=0)
        self._metric_variants = {}
        self._use_serving_tx = True
        self._selected_tx_name = None
        self._metric_preference_initialized = False
        self._tx_count = None
        tx_combo = self.widgets["coverage_tx_combo"]
        with QSignalBlocker(tx_combo):
            tx_combo.clear()
        self._set_tx_selector_visible(False)
        self._set_serving_selector_visible(False)
        self.heights = []
        self.current_height_index = 0
        height_combo = self.widgets["coverage_height_combo"]
        with QSignalBlocker(height_combo):
            height_combo.clear()
            height_combo.setEnabled(False)
        self._threshold_metric = None
        self._last_coverage_metadata = None
        self._last_threshold_metadata = None
        self._legend_container.setVisible(False)
        self._reset_animation_widgets(multi_height=False)
        self.configure_metric_controls(False, None)
        self.configure_threshold_controls(False, None)
        self.update_slice_summary(None)

    def set_heights(self, heights: Any) -> None:
        """Populate available coverage heights and playback eligibility."""
        combo = self.widgets.get("coverage_height_combo")
        if combo is None:
            return
        if heights is None:
            heights = []
        if hasattr(heights, "tolist"):
            heights = heights.tolist()
        try:
            self.heights = [float(height) for height in heights]
        except (TypeError, ValueError):
            self.heights = []

        multi = len(self.heights) > 1
        with QSignalBlocker(combo):
            combo.clear()
            for height in self.heights:
                combo.addItem(f"{height:.2f} m", height)
            if not self.heights:
                self.current_height_index = 0
            else:
                self.current_height_index = min(self.current_height_index, len(self.heights) - 1)
                combo.setCurrentIndex(self.current_height_index)
            combo.setEnabled(multi)

        self._height_animation_group.setVisible(multi)
        self._reset_animation_widgets(multi_height=multi)
        if not multi:
            controller = getattr(self.parent, "ui_controller", None)
            handler = getattr(controller, "handle_coverage_height_animation_stop", None)
            if handler:
                handler()

    def _reset_animation_widgets(self, *, multi_height: bool) -> None:
        play_btn = self.widgets["coverage_height_play_btn"]
        play_btn.setText("Play heights")
        play_btn.setEnabled(bool(multi_height))
        self.widgets["coverage_height_stop_btn"].setEnabled(False)
        self.widgets["coverage_height_speed"].setEnabled(bool(multi_height))

    def set_metrics(
        self,
        metrics: Any,
        selected_metric: Optional[str] = None,
        *,
        tx_count: Optional[int] = None,
    ) -> None:
        """Populate metric families and their serving/per-TX selectors."""
        combo = self.widgets.get("coverage_metric_combo")
        if combo is None:
            return
        raw_metrics = [str(metric) for metric in (metrics or [])]
        self._metric_variants = {}
        for metric in raw_metrics:
            base, separator, selector = metric.partition("/")
            selector_value = selector if separator else None
            family, uses_serving_tx = self._metric_family(
                base,
                selector_value,
            )
            self._metric_variants.setdefault(family, []).append(
                (metric, selector_value, uses_serving_tx)
            )
        if tx_count is not None:
            self._tx_count = max(0, int(tx_count))
        elif self._tx_count is None:
            selectors = {
                selector
                for variants in self._metric_variants.values()
                for _metric, selector, _uses_serving_tx in variants
                if selector is not None
            }
            self._tx_count = len(selectors)
        selected_base, separator, selected_selector = str(selected_metric or "").partition("/")
        selected_family, _ = self._metric_family(
            selected_base,
            selected_selector if separator else None,
        )
        with QSignalBlocker(combo):
            combo.clear()
            for family in self._metric_variants:
                combo.addItem(self._metric_display_text(family), family)
            combo.setEnabled(bool(self._metric_variants))
            selected_index = combo.findData(selected_family) if selected_metric else -1
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
        current_family = combo.currentData()
        initialize_preference = bool(raw_metrics and not self._metric_preference_initialized)
        self._populate_tx_selector(
            str(current_family or ""),
            selected_metric=str(selected_metric or ""),
            initialize_preference=initialize_preference,
        )
        if raw_metrics:
            self._metric_preference_initialized = True

    @staticmethod
    def _metric_family(base: str, selector: Optional[str]) -> tuple[str, bool]:
        """Return the GUI family and serving-transmitter mode for one metric."""
        serving_family = _SERVING_METRIC_FAMILIES.get(base)
        if serving_family is not None:
            return serving_family, True
        if base in _SELECTABLE_SERVING_METRICS and selector is None:
            return base, True
        return base, False

    def _populate_tx_selector(
        self,
        family: str,
        *,
        selected_metric: str = "",
        initialize_preference: bool = False,
    ) -> None:
        """Synchronize serving mode and TX choices for one metric family."""
        tx_combo = self.widgets.get("coverage_tx_combo")
        serving_toggle = self.widgets.get("coverage_serving_tx_toggle")
        if tx_combo is None or serving_toggle is None:
            return
        variants = self._metric_variants.get(family, [])
        serving_variants = [variant for variant in variants if variant[2]]
        per_tx_variants = [variant for variant in variants if variant[1] is not None]
        tx_count = int(self._tx_count or 0)
        can_choose_mode = bool(serving_variants and per_tx_variants and tx_count > 1)
        if initialize_preference and selected_metric and can_choose_mode:
            if any(variant[0] == selected_metric for variant in serving_variants):
                self._use_serving_tx = True
            elif any(variant[0] == selected_metric for variant in per_tx_variants):
                self._use_serving_tx = False
        selected_selector = next(
            (variant[1] for variant in per_tx_variants if variant[0] == selected_metric),
            None,
        )
        if initialize_preference and selected_selector is not None:
            self._selected_tx_name = selected_selector
        use_serving_tx = bool(
            serving_variants and (not per_tx_variants or (can_choose_mode and self._use_serving_tx))
        )
        with QSignalBlocker(tx_combo):
            tx_combo.clear()
            for full_name, selector, _uses_serving_tx in per_tx_variants:
                tx_combo.addItem(str(selector), full_name)
            selected_index = tx_combo.findData(selected_metric) if selected_metric else -1
            if selected_index < 0 and self._selected_tx_name is not None:
                selected_index = next(
                    (
                        index
                        for index in range(tx_combo.count())
                        if tx_combo.itemText(index) == self._selected_tx_name
                    ),
                    -1,
                )
            if selected_index >= 0:
                tx_combo.setCurrentIndex(selected_index)
            if tx_combo.count() and self._selected_tx_name is None:
                self._selected_tx_name = str(tx_combo.currentText())
        with QSignalBlocker(serving_toggle):
            serving_toggle.setChecked(use_serving_tx)
        self._set_serving_selector_visible(can_choose_mode)
        self._set_tx_selector_visible(bool(per_tx_variants) and tx_count > 1 and not use_serving_tx)

    def _selected_metric_for_family(self, family: str) -> Optional[str]:
        """Return the exact logical metric selected by the family controls."""
        variants = self._metric_variants.get(family, [])
        serving_toggle = self.widgets.get("coverage_serving_tx_toggle")
        if serving_toggle is not None and serving_toggle.isChecked():
            serving_metric = next((variant[0] for variant in variants if variant[2]), None)
            if serving_metric is not None:
                return serving_metric
        tx_combo = self.widgets.get("coverage_tx_combo")
        if tx_combo is not None and tx_combo.count():
            selected_tx_metric = tx_combo.currentData()
            if selected_tx_metric:
                return str(selected_tx_metric)
        return variants[0][0] if variants else None

    def _set_serving_selector_visible(self, visible: bool) -> None:
        """Show serving-transmitter mode only when both modes are meaningful."""
        toggle = self.widgets.get("coverage_serving_tx_toggle")
        if toggle is not None:
            toggle.setVisible(bool(visible))
            toggle.setEnabled(bool(visible))

    def _set_tx_selector_visible(self, visible: bool) -> None:
        """Keep TX selection out of aggregate and categorical workflows."""
        if hasattr(self, "_tx_label"):
            self._tx_label.setVisible(bool(visible))
        combo = self.widgets.get("coverage_tx_combo")
        if combo is not None:
            combo.setVisible(bool(visible))
            combo.setEnabled(bool(visible))

    def configure_metric_controls(self, has_data: bool, metric: Optional[str]) -> None:
        """Disable unsafe smoothing for categorical serving-TX layers."""
        interpolation = self.widgets.get("coverage_interpolation")
        if interpolation is None:
            return
        categorical = bool(has_data and is_serving_tx_metric(metric))
        if categorical:
            with QSignalBlocker(interpolation):
                interpolation.setCurrentText("Raw")
            interpolation.setToolTip(
                "Smoothing is disabled for categorical serving-transmitter indices"
            )
        else:
            interpolation.setToolTip("NaN-aware spatial smoothing for scalar coverage data")
        interpolation.setEnabled(bool(has_data and not categorical))

    def _update_scalar_legend(
        self,
        metric: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Show the semantic scalar color scale and transparent-missing rule."""
        colormap = coverage_metric_colormap(metric)
        if colormap is None:
            raw_names = metadata.get("tx_names")
            if raw_names is None:
                raw_names = []
            if hasattr(raw_names, "tolist"):
                raw_names = raw_names.tolist()
            try:
                tx_count = int(metadata.get("serving_tx_count", len(raw_names)))
            except (TypeError, ValueError):
                tx_count = len(raw_names)
            tx_names = serving_tx_labels(raw_names, tx_count)
            visible_count = min(len(tx_names), 8)
            entries = [
                (
                    f'<span style="color:{serving_tx_color_hex(index)}">■</span> '
                    f"{escape(tx_names[index])}"
                )
                for index in range(visible_count)
            ]
            if len(tx_names) > visible_count:
                entries.append(f"+{len(tx_names) - visible_count} more")
            self.widgets["coverage_legend_title"].setText("Transmitters:")
            self.widgets["coverage_legend_min"].setVisible(False)
            categorical_legend = self.widgets["coverage_legend_gradient"]
            categorical_legend.clear()
            categorical_legend.setText(" · ".join(entries) if entries else "No transmitters")
            categorical_legend.setToolTip("Serving-transmitter category colors")
            # At narrow panel widths the capped legend can wrap to five lines;
            # leave enough height for the final "+N more" overflow marker.
            categorical_legend.setMaximumHeight(80)
            categorical_legend.setVisible(True)
            self.widgets["coverage_legend_max"].setVisible(False)
            self._legend_container.setVisible(True)
            return
        color_scale = coverage_metric_color_scale(metric)
        self.widgets["coverage_legend_title"].setText(
            "Scale (logarithmic):" if color_scale == "logarithmic" else "Scale:"
        )
        self.widgets["coverage_legend_min"].setVisible(True)
        self.widgets["coverage_legend_gradient"].setVisible(True)
        self.widgets["coverage_legend_max"].setVisible(True)
        _label, unit = coverage_metric_label(metric)
        value_min = format_coverage_value(metadata.get("value_min"), unit)
        value_max = format_coverage_value(metadata.get("value_max"), unit)
        suffix = f" {unit}" if unit and unit != "index" else ""
        self.widgets["coverage_legend_min"].setText(f"{value_min}{suffix}")
        self.widgets["coverage_legend_max"].setText(f"{value_max}{suffix}")
        gradient = self.widgets["coverage_legend_gradient"]
        gradient.setMaximumHeight(16)
        gradient.setPixmap(_coverage_colormap_gradient_pixmap(colormap))
        scale_description = "logarithmic" if color_scale == "logarithmic" else "linear"
        gradient.setToolTip(f"Coverage color scale: {colormap} ({scale_description})")
        self._legend_container.setVisible(True)

    def configure_threshold_controls(
        self,
        has_data: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Enable and accurately range threshold controls for the active metric."""
        self._last_threshold_metadata = metadata
        toggle = self.widgets.get("coverage_threshold_toggle")
        spin = self.widgets.get("coverage_threshold_value")
        mask_toggle = self.widgets.get("coverage_threshold_mask_toggle")
        isolines_toggle = self.widgets.get("coverage_isolines_toggle")
        isoline_count = self.widgets.get("coverage_isoline_count")
        comparator_label = self.widgets.get("coverage_threshold_comparator")
        if toggle is None or spin is None:
            return

        metric = str(metadata.get("metric_name")) if has_data and metadata else None
        thresholdable = bool(has_data and supports_coverage_threshold(metric))
        metric_changed = metric != self._threshold_metric
        self._threshold_metric = metric

        blockers = [QSignalBlocker(widget) for widget in (toggle, spin)]
        try:
            toggle.setEnabled(thresholdable)
            if not thresholdable:
                toggle.setChecked(False)
                spin.setEnabled(False)
                spin.setSuffix("")
                spin.set_scientific_mode(False)
                if comparator_label is not None:
                    comparator_label.setText("")
                for widget in (mask_toggle, isolines_toggle):
                    if widget is not None:
                        with QSignalBlocker(widget):
                            widget.setChecked(False)
                            widget.setEnabled(False)
                if isoline_count is not None:
                    isoline_count.setEnabled(False)
                self.set_threshold_summary("Not available for this metric", active=False)
                return

            try:
                value_min = float(metadata.get("value_min", 0.0))
                value_max = float(metadata.get("value_max", 1.0))
            except (TypeError, ValueError):
                value_min, value_max = 0.0, 1.0
            if not math.isfinite(value_min) or not math.isfinite(value_max):
                value_min, value_max = 0.0, 1.0
            if value_min > value_max:
                value_min, value_max = value_max, value_min

            _label, unit = coverage_metric_label(metric)
            if unit in {"linear", "W"}:
                scale = max(abs(value_min), abs(value_max), 1.0e-30)
                data_span = max(value_max - value_min, scale * 0.01, 1.0e-30)
                lower = value_min - data_span
                if value_min >= 0.0:
                    lower = 0.0
                spin.setDecimals(30)
                spin.setRange(lower, value_max + data_span)
                spin.setSingleStep(max(data_span / 100.0, 1.0e-30))
                spin.setSuffix("")
                spin.set_scientific_mode(True)
            else:
                span = max(value_max - value_min, 1.0)
                spin.set_scientific_mode(False)
                spin.setDecimals(2)
                spin.setRange(value_min - span, value_max + span)
                spin.setSingleStep(max(span / 100.0, 0.01))
                spin.setSuffix(f" {unit}" if unit and unit != "index" else "")

            comparator = coverage_metric_comparator(metric)
            if comparator_label is not None:
                comparator_label.setText("≤" if comparator == "<=" else "≥")

            try:
                default_value = default_coverage_threshold(metadata, self.current_height_index)
            except (TypeError, ValueError, KeyError, IndexError):
                default_value = (value_min + value_max) / 2.0
            current_value = float(spin.value())
            if (
                metric_changed
                or current_value < spin.minimum()
                or current_value > spin.maximum()
                or not toggle.isChecked()
            ):
                spin.setValue(default_value)

            spin.setEnabled(toggle.isChecked())
            if mask_toggle is not None:
                mask_toggle.setEnabled(toggle.isChecked())
            if isolines_toggle is not None:
                isolines_toggle.setEnabled(True)
            if isoline_count is not None:
                isoline_count.setEnabled(
                    bool(isolines_toggle is not None and isolines_toggle.isChecked())
                )
            if not toggle.isChecked():
                self.set_threshold_summary("Off", active=False)
        finally:
            del blockers

    def set_analysis_state(
        self,
        *,
        threshold_enabled: bool,
        threshold_value: Optional[float],
        mask_enabled: bool,
        isolines_enabled: bool,
        isoline_count: int,
        interpolation: str,
    ) -> None:
        """Atomically mirror controller-owned analysis state into widgets."""
        metric = self._threshold_metric
        supported = supports_coverage_threshold(metric)
        threshold_enabled = bool(threshold_enabled and supported)
        mask_enabled = bool(mask_enabled and threshold_enabled)
        isolines_enabled = bool(isolines_enabled and supported)
        count = max(2, min(int(isoline_count), 12))

        toggle = self.widgets["coverage_threshold_toggle"]
        spin = self.widgets["coverage_threshold_value"]
        mask = self.widgets["coverage_threshold_mask_toggle"]
        isolines = self.widgets["coverage_isolines_toggle"]
        count_widget = self.widgets["coverage_isoline_count"]
        for widget in (toggle, spin, mask, isolines, count_widget):
            widget.blockSignals(True)
        try:
            toggle.setChecked(threshold_enabled)
            spin.setEnabled(threshold_enabled)
            if threshold_value is not None:
                spin.setValue(float(threshold_value))
            mask.setChecked(mask_enabled)
            mask.setEnabled(threshold_enabled)
            isolines.setChecked(isolines_enabled)
            isolines.setEnabled(supported)
            count_widget.setValue(count)
            count_widget.setEnabled(isolines_enabled)
            self.set_interpolation_method(interpolation)
        finally:
            for widget in (toggle, spin, mask, isolines, count_widget):
                widget.blockSignals(False)

    def restore_session_controls(
        self,
        *,
        opacity: float,
        threshold_enabled: bool,
        threshold_value: Optional[float],
        mask_enabled: bool,
        isolines_enabled: bool,
        isoline_count: int,
        interpolation: str,
        height_animation_speed: int,
    ) -> None:
        """Mirror a controller-owned session snapshot without emitting commands."""
        opacity_percent = int(round(float(opacity) * 100.0))
        speed = max(1, min(int(height_animation_speed), 10))
        opacity_widget = self.widgets.get("coverage_opacity")
        speed_widget = self.widgets.get("coverage_height_speed")
        for widget, value in (
            (opacity_widget, opacity_percent),
            (speed_widget, speed),
        ):
            if widget is not None:
                with QSignalBlocker(widget):
                    widget.setValue(value)
        opacity_label = self.widgets.get("opacity_label")
        if opacity_label is not None:
            opacity_label.setText(f"{opacity_percent}%")
        speed_label = self.widgets.get("coverage_height_speed_label")
        if speed_label is not None:
            speed_label.setText(self._animation_interval_text(speed))
        self.set_analysis_state(
            threshold_enabled=threshold_enabled,
            threshold_value=threshold_value,
            mask_enabled=mask_enabled,
            isolines_enabled=isolines_enabled,
            isoline_count=isoline_count,
            interpolation=interpolation,
        )

    def set_interpolation_method(self, method: str) -> None:
        """Select a smoothing label without notifying the controller."""
        combo = self.widgets.get("coverage_interpolation")
        if combo is None:
            return
        label = self._INTERPOLATION_LABELS.get(str(method).lower(), "Raw")
        with QSignalBlocker(combo):
            combo.setCurrentText(label)

    def set_threshold_summary(self, text: str, *, active: bool) -> None:
        """Set the user-visible threshold summary text and emphasis."""
        label = self.widgets.get("coverage_threshold_summary")
        if label is None:
            return
        label.setText(str(text))
        self._set_label_emphasis(label, active=active)

    def update_slice_summary(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Show statistics for only the selected metric and height slice."""
        label = self.widgets.get("coverage_slice_summary")
        if label is None:
            return
        if not metadata:
            label.setText("Unavailable")
            self._set_label_emphasis(label, active=False)
            return

        try:
            if is_serving_tx_metric(metadata.get("metric_name")):
                summary = compute_serving_tx_coverage_summary(
                    metadata,
                    height_index=self.current_height_index,
                )
                text = format_serving_tx_coverage_summary(summary)
            else:
                summary = compute_coverage_slice_summary(
                    metadata,
                    height_index=self.current_height_index,
                )
                text = format_coverage_slice_summary(summary)
        except (IndexError, KeyError, TypeError, ValueError):
            label.setText("Unavailable")
            self._set_label_emphasis(label, active=False)
            return

        label.setText(text)
        self._set_label_emphasis(label, active=True)

    def set_threshold_value(self, value: float) -> None:
        """Set the threshold value without notifying controllers."""
        spin = self.widgets.get("coverage_threshold_value")
        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(float(value))

    def current_threshold_value(self) -> Optional[float]:
        """Return the current threshold spin value, if present."""
        spin = self.widgets.get("coverage_threshold_value")
        return None if spin is None else float(spin.value())

    def set_height_index(self, index: int) -> None:
        """Select a coverage height by index while clamping valid bounds."""
        combo = self.widgets.get("coverage_height_combo")
        if combo is None:
            return
        max_index = combo.count() - 1
        clamped = max(0, min(int(index), max_index)) if max_index >= 0 else 0
        self.current_height_index = clamped
        with QSignalBlocker(combo):
            combo.setCurrentIndex(clamped)
        self.update_slice_summary(self._last_coverage_metadata)

    def set_renderer_capabilities(self, *, supports_transparency: bool) -> None:
        """Show opacity only for renderers that support coverage transparency."""
        if hasattr(self, "_opacity_container"):
            self._opacity_container.setVisible(bool(supports_transparency))

    # --- callbacks -----------------------------------------------------
    def _on_coverage_toggled(self, checked: bool) -> None:
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_toggled", None)
        if handler:
            handler(checked)
        elif hasattr(self.parent, "on_coverage_toggled"):
            self.parent.on_coverage_toggled(checked)

    def _on_opacity_changed(self, value: int) -> None:
        self.widgets["opacity_label"].setText(f"{value}%")
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_opacity_changed", None)
        if handler:
            handler(value)
        elif hasattr(self.parent, "on_coverage_opacity_changed"):
            self.parent.on_coverage_opacity_changed(value)

    def _on_height_changed(self, index: int) -> None:
        if index < 0:
            return
        self.current_height_index = index
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_height_changed", None)
        if handler:
            handler(index)
        elif hasattr(self.parent, "on_coverage_height_changed"):
            self.parent.on_coverage_height_changed(index)

    def _on_interpolation_changed(self, method: str) -> None:
        method_map = {"raw": "none", "smooth": "linear", "smooth+": "cubic"}
        method_key = method_map.get(str(method).lower(), str(method).lower())
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_interpolation_changed", None)
        if handler:
            handler(method_key)
        elif hasattr(self.parent, "on_coverage_interpolation_changed"):
            self.parent.on_coverage_interpolation_changed(method_key)

    def _on_threshold_toggled(self, checked: bool) -> None:
        spin = self.widgets.get("coverage_threshold_value")
        mask = self.widgets.get("coverage_threshold_mask_toggle")
        if spin is not None:
            spin.setEnabled(bool(checked))
        if mask is not None:
            if not checked:
                with QSignalBlocker(mask):
                    mask.setChecked(False)
            mask.setEnabled(bool(checked))
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_threshold_changed", None)
        if handler and spin is not None:
            handler(bool(checked), float(spin.value()))
        elif hasattr(self.parent, "on_coverage_threshold_changed") and spin is not None:
            self.parent.on_coverage_threshold_changed(bool(checked), float(spin.value()))

    def _on_threshold_value_changed(self, value: float) -> None:
        toggle = self.widgets.get("coverage_threshold_toggle")
        if toggle is not None and not toggle.isChecked():
            return
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_threshold_changed", None)
        if handler:
            handler(True, float(value))
        elif hasattr(self.parent, "on_coverage_threshold_changed"):
            self.parent.on_coverage_threshold_changed(True, float(value))

    def _on_threshold_mask_toggled(self, checked: bool) -> None:
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_threshold_mask_changed", None)
        if handler:
            handler(bool(checked))
        elif hasattr(self.parent, "on_coverage_threshold_mask_changed"):
            self.parent.on_coverage_threshold_mask_changed(bool(checked))

    def _on_isolines_toggled(self, checked: bool) -> None:
        self._isoline_count_timer.stop()
        count = self.widgets.get("coverage_isoline_count")
        if count is not None:
            count.setEnabled(bool(checked))
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_isolines_changed", None)
        if handler and count is not None:
            handler(bool(checked), int(count.value()))
        elif hasattr(self.parent, "on_coverage_isolines_changed") and count is not None:
            self.parent.on_coverage_isolines_changed(bool(checked), int(count.value()))

    def _on_isoline_count_changed(self, value: int) -> None:
        toggle = self.widgets.get("coverage_isolines_toggle")
        if toggle is None or not toggle.isChecked():
            return
        self._pending_isoline_count = int(value)
        self._isoline_count_timer.start()

    def _flush_isoline_count_change(self) -> None:
        toggle = self.widgets.get("coverage_isolines_toggle")
        if toggle is None or not toggle.isChecked():
            return
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_isolines_changed", None)
        if handler:
            handler(True, self._pending_isoline_count)
        elif hasattr(self.parent, "on_coverage_isolines_changed"):
            self.parent.on_coverage_isolines_changed(True, self._pending_isoline_count)

    def _on_metric_changed(self, _display_text: str) -> None:
        combo = self.widgets.get("coverage_metric_combo")
        family = combo.currentData() if combo is not None else None
        if not family:
            return
        family = str(family)
        self._populate_tx_selector(family)
        metric = self._selected_metric_for_family(family)
        if not metric:
            return
        self._forward_metric_selection(str(metric))

    def _on_serving_mode_changed(self, checked: bool) -> None:
        """Switch between the serving layer and a selected transmitter."""
        self._use_serving_tx = bool(checked)
        combo = self.widgets.get("coverage_metric_combo")
        family = combo.currentData() if combo is not None else None
        if not family:
            return
        variants = self._metric_variants.get(str(family), [])
        per_tx_count = sum(variant[1] is not None for variant in variants)
        self._set_tx_selector_visible(
            per_tx_count > 0 and int(self._tx_count or 0) > 1 and not checked
        )
        metric = self._selected_metric_for_family(str(family))
        if metric:
            self._forward_metric_selection(metric)

    def _on_tx_changed(self, _display_text: str) -> None:
        tx_combo = self.widgets.get("coverage_tx_combo")
        metric = tx_combo.currentData() if tx_combo is not None else None
        if metric:
            self._selected_tx_name = str(tx_combo.currentText())
            self._forward_metric_selection(str(metric))

    def _forward_metric_selection(self, metric: str) -> None:
        """Forward one resolved aggregate or per-TX metric key."""
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_metric_changed", None)
        if handler:
            handler(metric)

    def _on_height_animation_play_clicked(self) -> None:
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_height_animation_play", None)
        if handler:
            handler()
        elif hasattr(self.parent, "on_coverage_height_animation_play"):
            self.parent.on_coverage_height_animation_play()

    def _on_height_animation_stop_clicked(self) -> None:
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_height_animation_stop", None)
        if handler:
            handler()
        elif hasattr(self.parent, "on_coverage_height_animation_stop"):
            self.parent.on_coverage_height_animation_stop()

    def _on_height_animation_speed_changed(self, value: int) -> None:
        self.widgets["coverage_height_speed_label"].setText(self._animation_interval_text(value))
        controller = getattr(self.parent, "ui_controller", None)
        handler = getattr(controller, "handle_coverage_height_animation_speed_changed", None)
        if handler:
            handler(value)
        elif hasattr(self.parent, "on_coverage_height_animation_speed_changed"):
            self.parent.on_coverage_height_animation_speed_changed(value)
