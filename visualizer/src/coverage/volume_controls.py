"""One progressive control layout for embedded and scripted coverage views."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.coverage.volume import ALL_TRANSMITTERS

from ..panels.collapsible_section import CollapsibleSection
from ..sensing.volume_inspection_controls import LinkedSlices
from ..widgets.form_layout import AlignedFormLayout
from ..widgets.wrapping_layout import WrappingLayout


def build_controls(self, opening, view):
    """Build controls once; the owner handles source loading and scene updates."""
    outer = QVBoxLayout(self)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setAlignment(Qt.AlignTop)
    if self.embedded:
        layout = outer
    else:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        layout = QVBoxLayout(body)
    layout.setSpacing(6)
    self.sections = {}

    def control_row(*widgets, wrap=False):
        row_widget = QWidget()
        row = WrappingLayout(row_widget) if wrap else QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for widget in widgets:
            row.addWidget(widget)
        return row_widget

    def section(title, *, opened=False, parent_layout=None):
        widget = CollapsibleSection(title, start_open=opened, flat_header=parent_layout is not None)
        self.sections[title] = widget
        (parent_layout or layout).addWidget(widget)
        form = AlignedFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        widget.content_layout().addLayout(form)
        return form

    self.status = QLabel("Loading saved coverage…")
    self.status.setWordWrap(True)
    layout.addWidget(self.status)
    self.tx = QComboBox(self)
    self.tx.addItems(self.source["tx_names"])
    if len(self.source["tx_names"]) > 1:
        self.tx.addItem("All transmitters — strongest link", ALL_TRANSMITTERS)
    self.quantity = QComboBox(self)
    self.quantity.addItem("Path gain (dB)", "path_gain_db")
    self.quantity.addItem("Received power (dBm)", "rss_dbm")
    self.quantity.addItem("Path loss (dB)", "path_loss_db")
    if not self.embedded:
        data = section("Data", opened=True)
        data.addRow("Transmitter", self.tx)
        data.addRow("Quantity", self.quantity)
        if opening and opening["note"]:
            note = QLabel(opening["note"])
            note.setWordWrap(True)
            data.addRow(note)
    else:
        # Main Coverage panel owns data selection; never present a second selector.
        self.tx.hide()
        self.quantity.hide()

    view_form = section("View", opened=True)
    self.mode = QComboBox()
    self.mode.addItem("Slice", "slice")
    self.mode.addItem("3D volume", "iso")
    self.mode.addItem("Maximum strength along view", "mip")
    self.mode.setToolTip("Choose a slice, a 3D coverage region, or a projection")
    view_form.addRow("Display", self.mode)
    self.orientation = QComboBox()
    for label, axis in (("XY · height", 2), ("XZ · north–south", 1), ("YZ · east–west", 0)):
        self.orientation.addItem(label, axis)
    view_form.addRow("Plane", self.orientation)
    self.coordinate = QComboBox()
    view_form.addRow("Position", self.coordinate)
    self.smoothing = QComboBox()
    for label, key in (("Raw", "none"), ("Smooth", "linear"), ("Smooth+", "cubic")):
        self.smoothing.addItem(label, key)
    self.smoothing.setToolTip(
        "Gaussian display smoothing within the selected plane. Missing cells and original values are preserved."
    )
    self.contours = QCheckBox("Contours")
    self.contour_count = QSpinBox()
    self.contour_count.setRange(2, 12)
    self.contour_count.setValue(6)
    self.contour_count.setKeyboardTracking(False)
    self.filter_enabled = QCheckBox("Apply threshold")
    self.filter_enabled.setToolTip(
        "Filter original saved values in every view. Smoothing affects colours and contours, not this criterion."
    )
    view_form.addRow(self.filter_enabled)
    self.threshold = self._spin(-250, 250, -95, " dB")
    self.threshold.setToolTip(
        "RF threshold in dB or dBm. Arrow keys change it by 0.5 dB; saved values are unchanged."
    )
    self.threshold_label = QLabel("Threshold ≥")
    view_form.addRow(self.threshold_label, self.threshold)
    self.region_tx = QComboBox()
    # Store transmitter IDs separately from menu positions so older bookmarks
    # still select the same transmitter after adding the all-regions default.
    self.region_tx.addItem("All transmitters", -1)
    for index, name in enumerate(self.source["tx_names"]):
        self.region_tx.addItem(name, index)
    view_form.addRow("Serving region", self.region_tx)
    self.view_form = view_form
    self.play = QPushButton("Sweep slice")
    self.play.clicked.connect(self._play)
    self.cut_axis = QComboBox()
    for axis, name in enumerate("XYZ"):
        self.cut_axis.addItem(name, axis)
    self.cut_axis.setCurrentIndex(2)
    self.cut_axis.setToolTip("3D sweep axis")
    self.cut_axis.setAccessibleName("3D sweep axis")
    self.cut_enabled = QCheckBox("Cutaway ≤")
    self.cut_enabled.setToolTip("Keep cells through this coordinate along the sweep axis.")
    self.cut_position = QComboBox()
    self.cut_position.setAccessibleName("3D cutaway coordinate")
    view_form.addRow(self.cut_enabled, self.cut_position)
    self.rate = self._spin(0.2, 20, 2, " planes/s")
    self.rate.setDecimals(1)
    self.rate.setSingleStep(0.2)
    self.rate.setAccessibleName("Sweep rate")
    self.rate.setToolTip("Spatial planes per second. Scenario time stays unchanged.")
    self.rate.valueChanged.connect(self._rate_changed)
    self.loop = QCheckBox("Loop")
    self.sweep_controls = control_row(self.play, self.cut_axis, self.rate, self.loop, wrap=True)
    view_form.addRow(self.sweep_controls)

    view_details = self.sections["View"].content_layout()
    self.opacity = self._spin(0.05, 1, 0.65)
    self.opacity.setSingleStep(0.05)
    view_form.addRow("Opacity", self.opacity)
    appearance = section("Display details", parent_layout=view_details)
    appearance.addRow("Smoothing", self.smoothing)
    self.contour_controls = control_row(self.contours, self.contour_count)
    self.contour_count.setAccessibleName("Contour count")
    # Retain scripted/bookmark controls for compatibility, outside the GUI.
    # Interactive views always use the metric's scalar or transmitter palette.
    self.colour_mode = QComboBox(self)
    self.colour_mode.addItem("Metric colours", "metric")
    self.colour_mode.addItem("Uniform colour", "uniform")
    self.colour_mode.hide()
    self.solid_colour = QComboBox(self)
    for name, rgba in (
        ("Cyan", (0.1, 0.8, 0.95, 1)),
        ("Blue", (0.2, 0.4, 0.9, 1)),
        ("Orange", (0.9, 0.5, 0.15, 1)),
    ):
        self.solid_colour.addItem(name, rgba)
    self.solid_colour.hide()
    self.uniform_colour = QCheckBox("Unlit surface", self)
    self.uniform_colour.setChecked(True)
    self.uniform_colour.setToolTip(
        "Display colour independently of surface lighting. Opacity still blends with the scene."
    )
    self.uniform_colour.hide()
    self.visible = QCheckBox("Show coverage")
    self.visible.setChecked(True)
    if not self.embedded:
        appearance.addRow(self.visible)
    else:
        self.visible.setParent(self)
        self.visible.hide()
    self.show_nonfinite = QCheckBox("Grey no-data")
    self.show_nonfinite.setToolTip(
        "Neutral grey lies outside the colour scale. Cell inspection distinguishes zero contributions from unavailable estimates."
    )
    self.cell_edges = QCheckBox("Cell edges")
    self.cell_edges.setToolTip(
        "Outline saved sampling cells; this does not change the values or criterion."
    )
    self.threshold_edges = QCheckBox("Selected only")
    self.threshold_edges.setToolTip(
        "Draw edges only for cells in the selected threshold or serving region."
    )
    self.unavailable_edges = QCheckBox("Include unavailable")
    self.unavailable_edges.setToolTip("Also outline cells without an available estimate.")
    appearance.addRow(
        control_row(
            self.contour_controls,
            self.show_nonfinite,
            self.cell_edges,
            self.unavailable_edges,
            self.threshold_edges,
            wrap=True,
        )
    )
    self.low = self._spin(-250, 250, -130)
    self.high = self._spin(-250, 250, -50)
    self.low.setDecimals(6)
    self.high.setDecimals(6)
    self.colormap = QComboBox()
    self.colormap.addItems(["viridis", "RdYlGn", "RdYlGn_r"])
    self.auto_range = QPushButton("Auto")
    self.auto_range.setToolTip("Fit the colour scale to the saved volume.")
    self.auto_range.clicked.connect(self._fit_scale)
    self.colormap_controls = control_row(self.colormap, self.auto_range)
    self.colormap_controls.layout().setStretch(0, 1)
    appearance.addRow("Colormap", self.colormap_controls)
    self.low.setAccessibleName("Colormap minimum")
    self.high.setAccessibleName("Colormap maximum")
    self.low.setToolTip("Minimum colour-scale value; values below it keep the lowest colour.")
    self.high.setToolTip("Maximum colour-scale value; values above it keep the highest colour.")
    for spin in (self.low, self.high):
        # The full numerical range can be much wider than the displayed RF
        # limits. Let these paired fields share the available panel width.
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        spin.setMinimumWidth(90)
    self.scale_controls = control_row(QLabel("Min"), self.low, QLabel("Max"), self.high)
    for index in (1, 3):
        self.scale_controls.layout().setStretch(index, 1)
    appearance.addRow(self.scale_controls)
    self.details = self.sections["Display details"]
    self.appearance_form = appearance

    # Preserve scripted figure/bookmark settings without duplicating the main
    # scene's label controls in the interactive coverage panel.
    self.show_transmitter = QCheckBox("Show transmitter labels", self)
    self.show_transmitter.setChecked(True)
    self.readable_transmitter = QCheckBox("Readable transmitter locators", self)
    self.readable_transmitter.setChecked(self.embedded)
    self.show_transmitter.hide()
    self.readable_transmitter.hide()

    inspection = section("Inspect cells")
    self.cell_note = QLabel("Click a cell to inspect it.")
    self.cell_note.setWordWrap(True)
    inspection.addRow(self.cell_note)
    self.highlight_cell = QCheckBox("Highlight cell")
    self.highlight_cell.setChecked(self.embedded)
    self.highlight_cell.setToolTip(
        "Cyan outline with a dark border marks the selected saved cell. This selection guide remains visible through surfaces."
    )
    self.show_slices = QCheckBox("Linked slices")
    self.show_slices.setToolTip("Show linked XY, XZ and YZ slices through the selected cell.")
    inspection.addRow(control_row(self.highlight_cell, self.show_slices, wrap=True))
    self.slice_view = LinkedSlices()
    self.slice_view.selected.connect(self._select_slice)
    self.slice_view.hide()
    inspection.addRow(self.slice_view)

    from .cell_inspection import CoverageCellTable

    self.cell_table = CoverageCellTable(
        self.path, self.identity, self, tx_count=len(self.source["tx_names"])
    )
    inspection.addRow(self.cell_table)

    self.region_form = view_form
    self.region_enabled = QCheckBox("Restrict to XYZ region")
    self.region_enabled.setToolTip(
        "Include saved cell centres inside these bounds. Data and transmitter comparisons use the same region."
    )
    self.region_form.addRow(self.region_enabled)
    self.region_bounds = []
    self.region_bound_rows = []
    for axis in "XYZ":
        bounds = [self._spin(-1e9, 1e9, 0, " m"), self._spin(-1e9, 1e9, 0, " m")]
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        for spin in bounds:
            spin.setKeyboardTracking(False)
            spin.setDecimals(12)
            spin.set_scientific_mode(True)
            row.addWidget(spin)
        self.region_form.addRow(f"{axis} min / max", row_widget)
        self.region_bounds.append(bounds)
        self.region_bound_rows.append(row_widget)
    analysis = section("Analysis")
    self.analysis_summary = QLabel("Loading saved cells…")
    self.analysis_summary.setWordWrap(True)
    analysis.addRow(self.analysis_summary)
    from .height_comparison import CoverageHeightComparison

    self.height_comparison = CoverageHeightComparison(self.path, self.identity, self)
    analysis.addRow(self.height_comparison)

    tools = section("Saved views")
    saved_row = QHBoxLayout()
    self.save_view = QPushButton("Save view…")
    self.load_view = QPushButton("Load view…")
    for button in (self.save_view, self.load_view):
        button.setEnabled(False)
        saved_row.addWidget(button)
    self.save_view.clicked.connect(lambda: self._view_action(save=True))
    self.load_view.clicked.connect(lambda: self._view_action(save=False))
    tools.addRow(saved_row)

    if opening:
        (
            self.tx.setCurrentIndex(self.tx.findData(ALL_TRANSMITTERS))
            if opening["transmitter"] == ALL_TRANSMITTERS
            else self.tx.setCurrentText(opening["transmitter"])
        )
        self.quantity.setCurrentIndex(self.quantity.findData(opening["quantity"]))
        self.colormap.setCurrentText(opening["colormap"])
        for key in ("threshold", "opacity", "low", "high"):
            getattr(self, key).setValue(opening[key])
    if self.embedded:
        self._configure_metric(self.source["metric_name"])
        self.visible.setChecked(bool(self.visualizer.app_state.show_coverage))
        self.opacity.setValue(float(getattr(self.visualizer, "coverage_opacity", 1)))
    if view is not None:
        self._apply_view_controls(view)
    select_volume = getattr(self.visualizer.renderer, "set_volume_cell_selection_callback", None)
    if callable(select_volume):
        select_volume(self._select_volume_cell, object_ids=(self.OBJECT_ID,))
    self.tx.currentIndexChanged.connect(self._request)
    self.quantity.currentIndexChanged.connect(self._request)
    self.orientation.currentIndexChanged.connect(self._orient)
    self.coordinate.currentIndexChanged.connect(self._coordinate_changed)
    self.mode.currentIndexChanged.connect(self._mode_changed)
    self.cut_axis.currentIndexChanged.connect(self._cut_axis_changed)
    self.cut_position.currentIndexChanged.connect(self._cut_position_changed)
    self.cut_enabled.toggled.connect(self._cut_toggled)
    for combo in (
        self.colormap,
        self.region_tx,
        self.smoothing,
        self.colour_mode,
        self.solid_colour,
    ):
        combo.currentIndexChanged.connect(self._draw)
    for spin in (self.threshold, self.opacity, self.low, self.high):
        spin.valueChanged.connect(self._draw)
    self.contour_count.valueChanged.connect(self._draw)
    for bounds in self.region_bounds:
        for spin in bounds:
            spin.valueChanged.connect(self._draw)
    for toggle in (
        self.visible,
        self.show_nonfinite,
        self.cell_edges,
        self.uniform_colour,
        self.threshold_edges,
        self.unavailable_edges,
        self.show_transmitter,
        self.readable_transmitter,
        self.show_slices,
        self.highlight_cell,
        self.filter_enabled,
        self.contours,
        self.region_enabled,
    ):
        toggle.toggled.connect(self._draw)
    self._request()
    self.restoring = view
