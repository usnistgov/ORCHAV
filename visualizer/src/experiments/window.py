"""Dockable arrival-time and delay–angle views linked to Path Explorer."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..panels.collapsible_section import CollapsibleSection
from ..widgets.auxiliary_dock import AuxiliaryDock
from ..widgets.plot_lifecycle import ManagedPlotWidget
from ..widgets.responsive import wrap_control_row
from ..widgets.wrapping_layout import WrappingLayout

ANALYSIS_PLOT_MIN_HEIGHT_PX = 160
SELECTED_PATH_COLOR = "#ff70cf"
SELECTED_PATH_MARKER_SIZE_PX = 14
POWER_WEIGHT_TOOLTIP = (
    "Relative weights from exported path loss: strongest path = 1; "
    "10 dB weaker = 0.1. Weights at the same delay are summed. "
    "This is a power proxy, not received power in watts or dBm; phase is not combined."
)
ANGLE_PLOT_TOOLTIP = (
    "Numerical delay/angle bins, not physical scatterer clusters. "
    "Click a bin to select its recorded paths. Scene direction guides have a fixed 2 m display length."
)
INTERACTION_LABEL_TOOLTIP = (
    "Show up to 32 labels: P is the frame-local path ID; I is the interaction ordinal along "
    "that path, not its reflection-order color category. Existing MPC type markers add recorded "
    "type names and glyphs. Labels are drawn on top: they can overlap or show through buildings. "
    "Select one path or a small gate for readability."
)


class ExperimentWindow(AuxiliaryDock):
    """Plots select recorded rows; only the pygfx adapter changes scene overlays."""

    options_changed = Signal()
    gate_changed = Signal(float, float)
    bin_selected = Signal(int, int)
    clock_changed = Signal(float)
    play_changed = Signal(bool)
    cinematic_changed = Signal()
    interaction_mode_changed = Signal()
    presentation_speed_changed = Signal(float)
    presentation_style_changed = Signal()
    restart_requested = Signal()
    interaction_step_requested = Signal(int)
    paths_requested = Signal()
    cleared = Signal()
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__("Path Analysis", parent)
        self.setObjectName("pathAnalysis")
        self.resize(1040, 800)
        self._plots_retired = False
        self._updating = False
        self._histogram = None
        self._analysis_data = None
        self._selected_details = None
        self._arrival_delays = np.empty(0)
        self._arrival_weights = np.empty(0)
        self._clock_end = 1.0
        self._displayed_bounds = (0.0, 1.0)
        self._analysis_tab_index = 0
        widget = QWidget()
        self.setWidget(widget)
        layout = QVBoxLayout(widget)
        self._interaction_note = ""
        self.selection_context = QLabel()
        self.selection_context.setWordWrap(True)
        self.selection_context.hide()
        layout.addWidget(self.selection_context)
        row = QHBoxLayout()
        self.pair = QComboBox()
        self.set_frame_identity("Waiting for an accepted frame")
        self.scope = QComboBox()
        self.scope.setToolTip(
            "All uses every recorded path for this pair. Filtered uses the active Paths filters. "
            "Rendered uses only the paths currently drawn in the main scene."
        )
        for label, value in (
            ("All paths", "all"),
            ("Filtered paths", "filtered"),
            ("Rendered paths", "rendered"),
        ):
            self.scope.addItem(label, value)
        self.weight = QComboBox()
        self.weight.addItem("Path count", "count")
        self.weight.addItem("Relative power", "power")
        self.weight.setToolTip(POWER_WEIGHT_TOOLTIP)
        for label, combo in (("Pair", self.pair), ("Scope", self.scope), ("Weight", self.weight)):
            row.addWidget(QLabel(label))
            row.addWidget(combo, 1)
            combo.currentIndexChanged.connect(self._changed)
        layout.addLayout(wrap_control_row(row, [2, 2, 2]))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._arrival_tab()
        self._angle_tab()
        self.availability_note = QLabel()
        self.availability_note.setWordWrap(True)
        layout.addWidget(self.availability_note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.hide()
        layout.addWidget(self.status)
        self.matching_label = QLabel("Matching paths: none")
        self.matching_label.setWordWrap(True)
        layout.addWidget(self.matching_label)
        self.selected_label = QLabel("Selected path: none — choose a row in Path Explorer")
        self.selected_label.setWordWrap(True)
        self.selected_label.hide()
        layout.addWidget(self.selected_label)
        paths = QPushButton("Show paths")
        paths.setToolTip(
            "Open Path Explorer to inspect the matching paths and select an individual row."
        )
        paths.clicked.connect(self.paths_requested.emit)
        clear = QPushButton("Clear matching set")
        clear.setToolTip(
            "Clear the analysis gate/bin highlight without clearing an individually selected path."
        )
        clear.clicked.connect(self.cleared.emit)
        row = QHBoxLayout()
        row.addWidget(paths)
        save = QPushButton("Save workspace PNG…")
        save.setToolTip(
            "Save an image of this Path Analysis pane, including its visible plots and controls."
        )
        save.clicked.connect(self._save_image)
        row.addWidget(save)
        row.addWidget(clear)
        row.addWidget(self.create_placement_button())
        layout.addLayout(wrap_control_row(row, [1, 1, 1]))
        self._set_available_data(None)
        self.attach()

    def _plot(self, title):
        plot = ManagedPlotWidget(title=title, background="#111925")
        plot.setMinimumHeight(ANALYSIS_PLOT_MIN_HEIGHT_PX)
        plot.showGrid(x=True, y=True, alpha=0.18)
        return plot

    def _arrival_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.arrival_plot = self._plot("Arrival distribution · drag the shaded delay gate")
        self.arrival_plot.setLabel("bottom", "Path delay", units="ns")
        self.arrival_plot.setLabel("left", "Path count")
        self.arrivals = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7, symbolBrush="#4fded0")
        self.arrival_plot.addItem(self.arrivals)
        self.arrival_stems = pg.PlotDataItem(pen=pg.mkPen("#4fded0", width=2))
        self.arrival_plot.addItem(self.arrival_stems)
        self.gate = pg.LinearRegionItem((0, 1), brush=pg.mkBrush(255, 224, 105, 35))
        self.gate.sigRegionChangeFinished.connect(self._gate)
        self.arrival_plot.addItem(self.gate)
        self.selected_delay = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(SELECTED_PATH_COLOR, width=2, style=Qt.DashLine),
        )
        self.selected_delay.setZValue(10)
        self.selected_delay.hide()
        self.arrival_plot.addItem(self.selected_delay, ignoreBounds=True)
        self.selected_arrival = self._selection_marker(self.arrival_plot)
        layout.addWidget(self.arrival_plot, 1)
        row = QHBoxLayout()
        self.lower_bound = QDoubleSpinBox()
        self.upper_bound = QDoubleSpinBox()
        for index, box in enumerate((self.lower_bound, self.upper_bound)):
            box.setDecimals(3)
            box.setRange(-1e9, 1e9)
            box.setSuffix(" ns")
            box.editingFinished.connect(lambda index=index: self._boxes(index))
            box.setToolTip(
                ("Lower" if index == 0 else "Upper")
                + " delay-gate bound in nanoseconds. Editing it selects matching recorded paths."
            )
        row.addWidget(QLabel("Delay gate"))
        row.addWidget(self.lower_bound)
        row.addWidget(self.upper_bound)
        self.play = QPushButton("Play geometric clock")
        self.play.setToolTip(
            "Animate the selected recorded polylines using segment length / c. "
            "Pause scene playback first; this is a separate illustrative travel clock."
        )
        self.play.setCheckable(True)
        self.play.toggled.connect(self.play_changed.emit)
        row.addWidget(QLabel("Animate"))
        self.cinematic_limit = QComboBox()
        self.cinematic_limit.addItem("All paths", None)
        for count in (10, 50, 100):
            self.cinematic_limit.addItem(f"Top {count}", count)
        self.cinematic_limit.setToolTip(
            "Animate all selected paths, or limit the display to the strongest paths."
        )
        self.cinematic_limit.currentIndexChanged.connect(lambda: self.cinematic_changed.emit())
        row.addWidget(self.cinematic_limit)
        row.addWidget(self.play)
        layout.addLayout(wrap_control_row(row, [3, 2, 1]))
        self.presentation_options = CollapsibleSection("Presentation options", start_open=False)
        self.presentation_options.setEnabled(True)
        options_layout = self.presentation_options.content_layout()
        layout.addWidget(self.presentation_options)
        groups = WrappingLayout()
        groups.setSpacing(8)
        options_layout.addLayout(groups)
        self.presentation_groups = {}
        forms = {}
        for title in ("Playback", "Interactions", "Appearance"):
            group = QGroupBox(title)
            group.setFlat(True)
            form = QFormLayout(group)
            form.setContentsMargins(8, 4, 8, 4)
            form.setHorizontalSpacing(6)
            form.setVerticalSpacing(4)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            groups.addWidget(group)
            self.presentation_groups[title] = group
            forms[title] = form
        self.interaction_mode = QComboBox()
        for label, mode in (
            ("Flash", "flash"),
            ("Keep reached", "reached"),
            ("Show all", "all"),
            ("Dwell then fade", "dwell"),
        ):
            self.interaction_mode.addItem(label, mode)
        self.interaction_mode.setToolTip(
            "Recorded interactions of the animated subset, not reflection-only markers. "
            "Keep reached follows the current geometric time, including backward scrubbing."
        )
        self.interaction_mode.setEnabled(True)
        self.interaction_mode.currentIndexChanged.connect(
            lambda: self.interaction_mode_changed.emit()
        )
        forms["Interactions"].addRow("Show", self.interaction_mode)
        self.presentation_speed = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0):
            self.presentation_speed.addItem(f"{speed:g}×", speed)
        self.presentation_speed.setCurrentIndex(2)
        self.presentation_speed.setToolTip(
            "Cinematic presentation pacing only. 1× traverses the selected geometric "
            "clock in about 5 seconds. Geometry / c times and arrival relationships do not change."
        )
        self.presentation_speed.setEnabled(True)
        self.presentation_speed.currentIndexChanged.connect(
            lambda: self.presentation_speed_changed.emit(self.presentation_speed.currentData())
        )
        forms["Playback"].addRow("Speed", self.presentation_speed)
        self.interaction_dwell = QDoubleSpinBox()
        self.interaction_dwell.setRange(0.05, 10.0)
        self.interaction_dwell.setDecimals(2)
        self.interaction_dwell.setSingleStep(0.25)
        self.interaction_dwell.setValue(1.0)
        self.interaction_dwell.setSuffix(" s")
        self.interaction_dwell.setEnabled(False)
        self.interaction_dwell.setToolTip(
            "Dwell then fade: hold duration on the 1× presentation timeline. "
            "2× halves wall duration; pausing freezes it. Scrubbing and looping recompute it."
        )
        self.interaction_dwell.valueChanged.connect(lambda: self.presentation_style_changed.emit())
        self.interaction_mode.currentIndexChanged.connect(
            lambda: self.interaction_dwell.setEnabled(
                self.interaction_mode.currentData() == "dwell"
            )
        )
        forms["Interactions"].addRow("Hold at 1×", self.interaction_dwell)
        self.match_mpc_colors = QCheckBox("Match MPC colors")
        self.match_mpc_colors.setChecked(True)
        self.match_mpc_colors.setToolTip(
            "Use the main MPC color mode/palette; material means the recorded vertex/outgoing material."
        )
        self.match_mpc_colors.toggled.connect(lambda: self.presentation_style_changed.emit())
        forms["Appearance"].addRow(self.match_mpc_colors)
        self.interaction_labels = QCheckBox("Order labels (max 32)")
        self.interaction_labels.setToolTip(INTERACTION_LABEL_TOOLTIP)
        self.interaction_labels.toggled.connect(lambda: self.presentation_style_changed.emit())
        forms["Interactions"].addRow(self.interaction_labels)
        self.presentation_end = QComboBox()
        for label, mode in (
            ("Hold final state", "hold"),
            ("Play once, clear pulses", "once"),
            ("Loop", "loop"),
        ):
            self.presentation_end.addItem(label, mode)
        self.presentation_end.setToolTip(
            "Hold final keeps the final pulse/interaction display with no timer. "
            "Play once clears pulse markers at the end; routes remain. Loop resets reached markers each cycle."
        )
        forms["Playback"].addRow("At end", self.presentation_end)
        self.restart = QPushButton("Restart")
        self.restart.setToolTip("Restart cinematic playback from geometric time zero.")
        self.restart.clicked.connect(self.restart_requested.emit)
        forms["Playback"].addRow(self.restart)
        self.trail_mode = QComboBox()
        for label, mode in (("Short", "short"), ("Medium", "medium"), ("Long", "long")):
            self.trail_mode.addItem(label, mode)
        self.trail_mode.setCurrentIndex(1)
        self.trail_mode.setToolTip(
            "Choose the length of the fading pulse trail. This changes presentation only, "
            "not path geometry, attenuation or travel time."
        )
        self.trail_mode.currentIndexChanged.connect(lambda: self.presentation_style_changed.emit())
        forms["Appearance"].insertRow(0, "Trail", self.trail_mode)
        self.halo_opacity = QDoubleSpinBox()
        self.halo_opacity.setRange(0, 0.30)
        self.halo_opacity.setSingleStep(0.01)
        self.halo_opacity.setDecimals(2)
        self.halo_opacity.setValue(0.13)
        self.halo_opacity.setToolTip("Route halo opacity: presentation only, not physical power.")
        self.halo_opacity.valueChanged.connect(lambda: self.presentation_style_changed.emit())
        forms["Appearance"].insertRow(1, "Halo", self.halo_opacity)
        steps_row = QHBoxLayout()
        self.previous_interaction = QPushButton("Previous")
        self.next_interaction = QPushButton("Next")
        self.previous_interaction.setAccessibleName("Previous interaction")
        self.next_interaction.setAccessibleName("Next interaction")
        self.previous_interaction.clicked.connect(lambda: self.interaction_step_requested.emit(-1))
        self.next_interaction.clicked.connect(lambda: self.interaction_step_requested.emit(1))
        for button in (self.previous_interaction, self.next_interaction):
            button.setToolTip(
                "Pause at a known interaction of the animated subset, using geometric length / c."
            )
            steps_row.addWidget(button)
        forms["Interactions"].addRow("Step", steps_row)
        self.cinematic_note = QLabel(
            "Select paths, then play their recorded geometry. Pause scene playback first."
        )
        self.cinematic_note.setWordWrap(True)
        layout.addWidget(self.cinematic_note)
        self.clock = QSlider(Qt.Orientation.Horizontal)
        self.clock.setRange(0, 1000)
        self.clock.valueChanged.connect(
            lambda value: self.clock_changed.emit(value * self._clock_end / 1000)
        )
        layout.addWidget(self.clock)
        self.clock.setToolTip(
            "Geometry-only travel: segment arc length / c, independent of exported delay"
        )
        self.tabs.addTab(tab, "Arrival time")

    def _angle_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.angle = QComboBox()
        self.angle.setToolTip(
            "Choose the recorded arrival or departure angle in world coordinates."
        )
        for label, key in (
            ("Arrival azimuth", "aoa_azimuth"),
            ("Departure azimuth", "aod_azimuth"),
            ("Arrival elevation", "aoa_elevation"),
            ("Departure elevation", "aod_elevation"),
        ):
            self.angle.addItem(label, key)
        self.angle.currentIndexChanged.connect(self._changed)
        row.addWidget(self.angle)
        self.angular_bin = QDoubleSpinBox()
        self.angular_bin.setRange(1, 90)
        self.angular_bin.setValue(10)
        self.angular_bin.setSuffix("°")
        self.angular_bin.setToolTip(
            "Angular-bin width in degrees; unavailable angle data is never inferred."
        )
        self.delay_bin = QDoubleSpinBox()
        self.delay_bin.setRange(0.001, 1e6)
        self.delay_bin.setDecimals(3)
        self.delay_bin.setValue(10)
        self.delay_bin.setSuffix(" ns")
        self.delay_bin.setToolTip(
            "Delay-bin width in nanoseconds. Bins group recorded paths, not physical scatterers."
        )
        for label, box in (("Angular bin", self.angular_bin), ("Delay bin", self.delay_bin)):
            row.addWidget(QLabel(label))
            row.addWidget(box)
            box.valueChanged.connect(self._changed)
        layout.addLayout(wrap_control_row(row, [1, 2, 2]))
        self.angle_plot = self._plot("Joint delay–angle mass · click a cell to reveal its paths")
        self.angle_plot.setLabel("bottom", "World angle", units="°")
        self.angle_plot.setLabel("left", "Path delay", units="ns")
        self.image = pg.PColorMeshItem(colorMap=pg.colormap.get("viridis"), enableAutoLevels=False)
        self.angle_plot.addItem(self.image)
        self.selected_angle = self._selection_marker(self.angle_plot)
        self.colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=pg.colormap.get("viridis"),
            label="Bin weight",
            interactive=False,
        )
        self.colorbar.setImageItem(self.image, insert_in=self.angle_plot.plotItem)
        self.angle_plot.scene().sigMouseClicked.connect(self._image_click)
        layout.addWidget(self.angle_plot, 1)
        self.angle_plot.setToolTip(ANGLE_PLOT_TOOLTIP)
        self.tabs.addTab(tab, "Delay–angle")

    def _changed(self, *_):
        if not self._updating:
            self.options_changed.emit()

    def settings(self):
        return dict(
            pair=self.pair.currentData(),
            scope=self.scope.currentData(),
            weight_mode=self.weight.currentData(),
            angle=self.angle.currentData(),
            angular_bin_width_deg=self.angular_bin.value(),
            delay_bin_width_ns=self.delay_bin.value(),
        )

    def set_pairs(self, pairs):
        current = self.pair.currentData()
        with QSignalBlocker(self.pair):
            self.pair.clear()
            for tx, rx in pairs:
                self.pair.addItem(f"TX {tx} → RX {rx}", (int(tx), int(rx)))
            # QVariant lookup does not compare opaque Python tuples by value.
            index = next((i for i, pair in enumerate(pairs) if pair == current), -1)
            if index >= 0:
                self.pair.setCurrentIndex(index)

    def clear_analysis(self):
        """Remove stale plot data before a new frame or selection is computed."""
        self._histogram = None
        self._analysis_data = None
        self._arrival_delays = np.empty(0)
        self._arrival_weights = np.empty(0)
        self.arrivals.setData([], [])
        self.arrival_stems.setData([], [])
        self.image.setData()
        self.set_matching_paths("")
        self.gate.hide()
        self._set_available_data(None)
        self.matching_label.setToolTip("")
        self.status.setToolTip("")
        self.angle_plot.setToolTip(ANGLE_PLOT_TOOLTIP)
        self.set_interaction_note("")
        self._refresh_selected_path()

    def _set_available_data(self, data):
        """Grey out unsupported choices for this frame, pair, scope and weighting."""
        if self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self._analysis_tab_index = self.tabs.currentIndex()
        has_arrival = data is not None and bool(
            np.any(np.isfinite(data.delays_ns) & (data.weights > 0))
        )
        angles = data.available_angles if data is not None else ()
        for index in range(self.angle.count()):
            available = self.angle.itemData(index) in angles
            item = self.angle.model().item(index)
            item.setEnabled(available)
            item.setToolTip("" if available else "No usable delay/angle pairs in this selection.")
        power = data is not None and data.power_available
        power_item = self.weight.model().item(self.weight.findData("power"))
        power_item.setEnabled(power)
        power_item.setToolTip(
            POWER_WEIGHT_TOOLTIP
            if power
            else "Unavailable: no finite exported path losses in this selection."
        )
        for index, available, reason in (
            (0, has_arrival, "No usable path delays in this selection."),
            (1, bool(angles), "No usable delay/angle pairs in this selection."),
        ):
            self.tabs.setTabEnabled(index, available)
            self.tabs.setTabToolTip(index, "" if available else reason)
        if self.tabs.isTabEnabled(self._analysis_tab_index):
            self.tabs.setCurrentIndex(self._analysis_tab_index)
        for control in (self.lower_bound, self.upper_bound, self.play, self.clock):
            control.setEnabled(has_arrival)
        notes = []
        if data is None:
            notes.append("Waiting for path data.")
        else:
            if not has_arrival:
                notes.append("No usable path delays in this selection.")
            else:
                finite = np.isfinite(data.delays_ns)
                if np.any(data.delay_provenance_codes[finite] == 1):
                    notes.append("Delays include geometry estimates.")
                elif np.any(data.delay_provenance_codes[finite] == -1):
                    notes.append("Delay provenance is unknown.")
            if not angles:
                notes.append("Delay–angle unavailable.")
            if not power:
                notes.append("Relative power unavailable.")
        self.availability_note.setText(" ".join(notes))
        self.availability_note.setVisible(bool(notes))

    def set_status(self, message, details=""):
        """Keep actionable feedback visible and place numerical context on hover."""
        self.status.setText(message)
        self.status.setVisible(bool(message))
        self.status.setToolTip(details)
        self.matching_label.setToolTip(details)

    def set_frame_identity(self, description):
        """Attach accepted-frame provenance to the pair selector."""
        self.pair.setToolTip(f"Choose the transmitter/receiver pair for analysis.\n{description}")

    def set_interaction_note(self, note):
        """Keep optional marker/color disclosures beside the control they explain."""
        if note != self._interaction_note:
            self._interaction_note = note
            self.interaction_labels.setToolTip(f"{INTERACTION_LABEL_TOOLTIP}\n{note}")
            self.match_mpc_colors.setToolTip(
                "Use the main MPC color mode/palette; material means the recorded "
                f"vertex/outgoing material.\n{note}"
            )

    def set_analysis(self, data, histogram, *, reset_gate, clock_end):
        self._analysis_data = data
        self._arrival_delays = np.empty(0)
        self._arrival_weights = np.empty(0)
        self.gate.show()
        self._updating = True
        try:
            with QSignalBlocker(self.weight), QSignalBlocker(self.angle):
                self.weight.setCurrentIndex(self.weight.findData(data.weight_mode.value))
                self.angle.setCurrentIndex(self.angle.findData(histogram.angle))
            self._set_available_data(data)
            self.gate.setVisible(self.tabs.isTabEnabled(0))
            finite = np.isfinite(data.delays_ns)
            delays = data.delays_ns[finite]
            weights = data.weights[finite]
            if len(delays):
                unique, inverse = np.unique(delays, return_inverse=True)
                totals = np.bincount(inverse, weights=weights)
                self._arrival_delays, self._arrival_weights = unique, totals
                self.arrivals.setData(unique, totals)
                stems = np.column_stack((np.zeros_like(totals), totals)).reshape(-1)
                self.arrival_stems.setData(np.repeat(unique, 2), stems, connect="pairs")
                bounds = (float(np.min(delays)), float(np.max(delays)))
                if reset_gate:
                    self.gate.setRegion(bounds)
                    self.lower_bound.setValue(bounds[0])
                    self.upper_bound.setValue(bounds[1])
                    self._displayed_bounds = (self.lower_bound.value(), self.upper_bound.value())
                self.arrival_plot.enableAutoRange()
            else:
                self.arrivals.setData([], [])
                self.arrival_stems.setData([], [])
            self.arrival_plot.setLabel("left", data.weight_label)
            delay_label = (
                "Path delay (includes estimates)"
                if np.any(data.delay_provenance_codes == 1)
                else "Path delay"
            )
            self.arrival_plot.setLabel("bottom", delay_label, units="ns")
            self.angle_plot.setLabel("left", delay_label, units="ns")
            self._histogram = histogram
            values = histogram.values
            x, y = histogram.angle_edges_deg, histogram.delay_edges_ns
            x_grid: np.ndarray
            y_grid: np.ndarray
            x_grid, y_grid = np.meshgrid(x, y)
            self.image.setData(x_grid, y_grid, values)
            self.colorbar.setLevels((0, max(float(values.max(initial=0)), 1e-12)))
            self.colorbar.getAxis("left").setLabel(data.weight_label)
            self.angle_plot.enableAutoRange()
            self._clock_end = max(float(clock_end), 1e-3)
            self.angle_plot.setToolTip(
                f"{histogram.occupied_bins} occupied bins · dominant bin {histogram.dominant_bin_share:.1%} · {histogram.min_bins_90} bins contain 90% of displayed mass.\n{ANGLE_PLOT_TOOLTIP}"
            )
        finally:
            self._updating = False
        self._refresh_selected_path()
        if self.tabs.isTabEnabled(0):
            self._gate()
        else:
            self.set_status("", " ".join(data.disclosures))

    def set_matching_paths(self, description):
        """Keep the matching group distinct from the individually selected path."""
        self.matching_label.setText(description or "Matching paths: none")

    def set_selected_path(self, details):
        """Follow a newly selected pair and mark the path in both analysis plots."""
        previous = getattr(self._selected_details, "identity", None)
        self._selected_details = details
        if details is not None and details.identity != previous:
            pair = (details.tx_id, details.rx_id)
            index = next((i for i in range(self.pair.count()) if self.pair.itemData(i) == pair), -1)
            if index >= 0 and index != self.pair.currentIndex():
                self.pair.setCurrentIndex(index)
        self._refresh_selected_path()

    @staticmethod
    def _selection_marker(plot):
        """Draw a visible location cue without changing the chart's data or bounds."""
        marker = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=SELECTED_PATH_MARKER_SIZE_PX,
            symbolPen=pg.mkPen(SELECTED_PATH_COLOR, width=3),
            symbolBrush=None,
        )
        marker.setZValue(10)
        marker.hide()
        plot.addItem(marker, ignoreBounds=True)
        return marker

    def _refresh_selected_path(self):
        """Resolve one ID in the sorted analysis snapshot; never infer missing metrics."""
        for item in (self.selected_delay, self.selected_arrival, self.selected_angle):
            item.hide()
        details, data = self._selected_details, self._analysis_data
        self.selected_label.setVisible(details is not None)
        if details is None:
            self.selected_label.setText("Selected path: none — choose a row in Path Explorer")
            return
        prefix = (
            f"Selected path: {details.canonical_path_id} · TX {details.tx_id} → RX {details.rx_id}"
        )
        if data is None:
            self.selected_label.setText(f"{prefix} · waiting for analysis")
            return
        row = int(np.searchsorted(data.path_ids, details.canonical_path_id))
        if row >= len(data.path_ids) or data.path_ids[row] != details.canonical_path_id:
            self.selected_label.setText(f"{prefix} · outside the displayed pair/scope")
            return
        delay = float(data.delays_ns[row])
        if not np.isfinite(delay):
            self.selected_label.setText(f"{prefix} · recorded delay unavailable")
            return
        self.selected_delay.setValue(delay)
        self.selected_delay.show()
        arrival = int(np.searchsorted(self._arrival_delays, delay))
        self.selected_arrival.setData([delay], [self._arrival_weights[arrival]])
        self.selected_arrival.show()
        angle = float(data.angles_deg(self.angle.currentData())[row])
        note = "pink markers show this path"
        if not np.isfinite(angle):
            note = "pink delay marker; selected angle unavailable"
        elif data.weights[row] <= 0:
            note = "pink delay marker; zero weight in this view"
        else:
            # Match the histogram's [-180, 180) convention, including its seam.
            angle = float(np.fmod(angle, 360.0))
            if angle < -180.0:
                angle += 360.0
            elif angle >= 180.0:
                angle -= 360.0
            self.selected_angle.setData([angle], [delay])
            self.selected_angle.show()
        self.selected_label.setText(f"{prefix} · {note}")

    def _gate(self):
        if self._updating:
            return
        low, high = self.gate.getRegion()
        self.lower_bound.setValue(low)
        self.upper_bound.setValue(high)
        self._displayed_bounds = (self.lower_bound.value(), self.upper_bound.value())
        self.gate_changed.emit(float(low), float(high))

    def _boxes(self, index):
        boxes = (self.lower_bound, self.upper_bound)
        if boxes[index].value() == self._displayed_bounds[index]:
            return
        bounds = list(self.gate.getRegion())
        bounds[index] = boxes[index].value()
        self.gate.setRegion(sorted(bounds))

    def _image_click(self, event):
        h = self._histogram
        if h is None or not self.angle_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.angle_plot.plotItem.vb.mapSceneToView(event.scenePos())
        a = int(np.searchsorted(h.angle_edges_deg, point.x(), side="right") - 1)
        d = int(np.searchsorted(h.delay_edges_ns, point.y(), side="right") - 1)
        if 0 <= a < h.values.shape[1] and 0 <= d < h.values.shape[0]:
            self.bin_selected.emit(d, a)

    def _save_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save path analysis", "path-analysis.png", "PNG image (*.png)"
        )
        if filename and not self.grab().save(filename, "PNG"):
            self.set_status("Could not save the workspace image")

    def set_clock_extent(self, end):
        self._clock_end = max(float(end), 1e-3)
        self.set_clock(0.0)

    def set_clock(self, value):
        with QSignalBlocker(self.clock):
            self.clock.setValue(int(np.clip(value / self._clock_end, 0, 1) * 1000))
        self.clock.setToolTip(
            f"Geometric travel clock {value:.3f} ns / {self._clock_end:.3f} ns · arc length / c, not measured bounce timestamps"
        )

    def retire_plots(self):
        """Unregister owned views before Qt destroys this scenario's plots."""
        if self._plots_retired:
            return
        # Parent deletion destroys the Qt objects before their Python wrappers
        # leave pyqtgraph's weak registry. Its later shutdown then disconnects
        # already-deleted objects unless we unregister these views explicitly.
        for plot in (self.arrival_plot, self.angle_plot, self.colorbar):
            plot.getViewBox().unregister()
        self._plots_retired = True

    def closeEvent(self, event):
        self.play.setChecked(False)
        self.closed.emit()
        super().closeEvent(event)
