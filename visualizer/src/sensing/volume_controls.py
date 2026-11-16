"""Focused view/analysis controls with shared display and selected-cell tools."""

from importlib.util import find_spec

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..panels.collapsible_section import CollapsibleSection
from ..widgets.form_layout import AlignedFormLayout
from ..widgets.input_guard import protect_controls
from ..widgets.wrapping_layout import WrappingLayout
from .volume_antenna_controls import ReceivePatternControls
from .volume_charts import ContributionPlot, VolumeSummaryPlots
from .volume_cutaway import VolumeCutawayControls
from .volume_evaluation import VolumeEvaluationControls
from .volume_inspection_controls import InspectionControls


class VolumeTabs(QTabWidget):
    """Size to the current task, rather than reserving space for hidden charts."""

    def heightForWidth(self, width):
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        height = page.layout().totalHeightForWidth(width)
        if height < 0:
            height = page.sizeHint().height()
        return height + self.tabBar().sizeHint().height() + 8

    def sizeHint(self):
        page = self.currentWidget()
        return (
            page.sizeHint() + QSize(0, self.tabBar().sizeHint().height() + 8)
            if page
            else super().sizeHint()
        )

    def minimumSizeHint(self):
        page = self.currentWidget()
        return (
            page.minimumSizeHint() + QSize(0, self.tabBar().sizeHint().height() + 8)
            if page
            else super().minimumSizeHint()
        )


def section(layout, title, *, expanded=False):
    widget = CollapsibleSection(title, start_open=expanded)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    widget.layout().setAlignment(Qt.AlignTop)
    widget.content_layout().setAlignment(Qt.AlignTop)
    layout.addWidget(widget)
    return widget, widget.content_layout()


def check(text, callback, *, checked=False):
    widget = QCheckBox(text)
    widget.setChecked(checked)
    widget.toggled.connect(callback)
    return widget


def compact_row(*widgets):
    row = QWidget()
    layout = WrappingLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return row


def build_controls(w):
    outer = QVBoxLayout(w)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    w.scroll = scroll
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.setAlignment(Qt.AlignTop)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(8)
    scroll.setWidget(body)
    outer.addWidget(scroll)
    w.source_note = QLabel("Saved sensing experiment")
    w.source_note.setWordWrap(True)
    layout.addWidget(w.source_note)
    w.conditions = QComboBox()
    w.conditions.setToolTip("Open another saved condition.")
    w.conditions.currentIndexChanged.connect(w._condition_changed)
    layout.addWidget(w.conditions)
    w.visible = check("Show volume", w._draw, checked=True)
    w.reset = QPushButton("Frame volume")
    w.reset.clicked.connect(w._frame_grid)
    layout.addWidget(compact_row(w.visible, w.reset))
    w.status = QLabel("Loading recorded outcomes…")
    w.status.setWordWrap(True)
    layout.addWidget(w.status)

    w.inspection_controls = c = InspectionControls()
    w.tabs = VolumeTabs()
    w.tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    w.tabs.currentChanged.connect(w.tabs.updateGeometry)
    w.tabs.setDocumentMode(True)
    layout.addWidget(w.tabs)
    pages = {}
    for title in ("View", "Analysis"):
        page = QWidget()
        body_layout = QVBoxLayout(page)
        body_layout.setContentsMargins(0, 8, 0, 4)
        body_layout.setSpacing(8)
        body_layout.setAlignment(Qt.AlignTop)
        w.tabs.addTab(page, title)
        pages[title] = body_layout
    view, analysis = pages.values()
    w.controls = QWidget()
    form = AlignedFormLayout(w.controls)
    form.setContentsMargins(0, 0, 0, 0)
    w.heading = QComboBox()
    w.heading.currentIndexChanged.connect(w._draw)
    w.single_heading = QLabel()
    w.motion_label = QLabel("Target motion")
    form.addRow(w.motion_label, compact_row(w.heading, w.single_heading))
    w.quantity = QComboBox()
    for title, value in (
        ("Matched / missed", "outcomes"),
        ("Outcome detail", "details"),
        ("Unmatched reports per trial", "extras"),
        ("Nearest report position error", "reports"),
    ):
        w.quantity.addItem(title, value)
    w.quantity.currentIndexChanged.connect(w._draw)
    form.addRow("Colour by", w.quantity)
    w.mode = QComboBox()
    for title, value in (("3D volume", "both"), ("Slice", "slice"), ("Maximum along view", "mip")):
        w.mode.addItem(title, value)
    # Retain old scripted view values; interactive filtering has a single owner.
    for title, value in (("Matched cells", "hit"), ("Missed cells", "miss")):
        w.mode.addItem(title, value)
    w.mode.currentIndexChanged.connect(w._draw)
    form.addRow("View", w.mode)
    c.orientation_label = QLabel("Plane")
    form.addRow(c.orientation_label, c.orientation)
    w.slice_coordinate = QComboBox()
    w.slice_coordinate_label = QLabel("Position")
    w.slice_coordinate.currentIndexChanged.connect(w._slice_coordinate_changed)
    form.addRow(w.slice_coordinate_label, w.slice_coordinate)
    w.cutaway = VolumeCutawayControls()
    w.cutaway.changed.connect(w._draw)
    form.addRow(w.cutaway)
    form.addRow(c.sweep_controls)
    view.addWidget(w.controls)
    w.controls.setEnabled(False)
    # One cell filter: no additional matched/missed rendering modes or category checkboxes.
    view.addWidget(c.focus_body)
    c.focus_expand.hide()
    w.legend, w.display_note = QLabel(w), QLabel(w)
    w.legend.hide()
    w.display_note.hide()

    w.summary_detail = QLabel(
        "All measured cells; view filters and cuts do not change these statistics."
    )
    w.summary_detail.setWordWrap(True)
    analysis.addWidget(w.summary_detail)
    w.contribution_plot = ContributionPlot()
    w.contribution_plot.hide()
    analysis.addWidget(w.contribution_plot)
    w.summary_plots = VolumeSummaryPlots()
    analysis.addWidget(w.summary_plots)
    w.evaluation_section, matching = section(analysis, "Rescore saved reports")
    explanation = QLabel(
        "Change how a report is counted as a match. This updates result colours and charts; it does not filter cells or rerun the detector."
    )
    explanation.setWordWrap(True)
    matching.addWidget(explanation)
    w.evaluation_controls = VolumeEvaluationControls(collapsible=False)
    w.evaluation_controls.setEnabled(False)
    w.evaluation_controls.changed.connect(w._draw)
    w.evaluation_controls.retry_requested.connect(w._load_evaluation)
    matching.addWidget(w.evaluation_controls)
    reset_gate = QPushButton("Restore recorded matching rule")
    reset_gate.clicked.connect(w._reset_evaluation)
    matching.addWidget(reset_gate)
    w.compare_section, comparison = section(analysis, "Compare conditions")
    w.compare_saved = QPushButton("Compare saved volumes…")
    w.compare_saved.setEnabled(False)
    w.compare_saved.setVisible(find_spec("visualizer.src.sensing.volume_comparison") is not None)
    w.compare_saved.clicked.connect(w._compare_saved_volumes)
    comparison.addWidget(w.compare_saved)
    w.pair_source = QComboBox()
    w.pair_source.currentIndexChanged.connect(w._pair_source_changed)
    comparison.addWidget(QLabel("Reference for per-cell change"))
    comparison.addWidget(w.pair_source)

    w.appearance_section, style_layout = section(layout, "Display", expanded=True)
    layout.removeWidget(w.appearance_section)
    layout.insertWidget(layout.indexOf(w.tabs), w.appearance_section)
    style = AlignedFormLayout()
    style_layout.addLayout(style)
    w.opacity = QDoubleSpinBox()
    w.opacity.setRange(0.05, 1)
    w.opacity.setSingleStep(0.05)
    w.opacity.setValue(1)
    w.opacity.valueChanged.connect(w._draw)
    w.uniform_colour = check("Unlit colours", w._draw, checked=True)
    style.addRow("Opacity", compact_row(w.opacity, w.uniform_colour))
    w.cell_edges = check("Cell edges", w._draw, checked=True)
    w.unavailable_edges = check("Include unmeasured", w._draw)
    w.unavailable_edges.setEnabled(True)
    w.cell_edges.toggled.connect(w.unavailable_edges.setEnabled)
    style_layout.addWidget(compact_row(w.cell_edges, w.unavailable_edges))
    w.scene_label_settings = QPushButton("Scene label style…")
    w.scene_label_settings.setToolTip(
        "Open shared scene labels: Role, Device Name or Sensing role text; Plain text, Callout or Card style."
    )
    w.scene_label_settings.clicked.connect(w._show_label_settings)
    style_layout.addWidget(w.scene_label_settings)
    w.count_scale_row = compact_row(QLabel("Colour scale maximum"), c.count_max)
    c.count_max.setSuffix(" reports/trial")
    c.count_max.show()
    style_layout.addWidget(w.count_scale_row)
    style_layout.addWidget(w.evaluation_controls.error_settings)
    w.count_scale_row.hide()
    w.readable_sensor = QCheckBox(w)  # Legacy saved-view field; scene labels own radios.
    w.readable_sensor.hide()

    w.selection_hint = QLabel(
        "Click a cell to inspect reports or open its trial in the visualizer."
    )
    w.selection_hint.setWordWrap(True)
    layout.addWidget(w.selection_hint)
    w.inspect_section, detail_layout = section(layout, "Selected cell", expanded=True)
    w.inspect_section.hide()
    w.inspect = check("Selected cell", w._draw)
    w.inspect.setParent(w)
    w.inspect.hide()
    w.clear_selection = QPushButton("Clear selection")
    w.clear_selection.clicked.connect(lambda: w.inspect.setChecked(False))
    nav = c._buttons(
        AlignedFormLayoutProxy(detail_layout), [("previous", "Previous"), ("next", "Next")]
    )
    nav.layout().addWidget(w.clear_selection)
    w.altitude, w.x, w.y = QComboBox(), QComboBox(), QComboBox()
    w.altitude_label = QLabel("Z")
    w.coordinate_labels = {"x": QLabel("X"), "y": QLabel("Y"), "z": w.altitude_label}
    coordinates = []
    for axis, combo in (("x", w.x), ("y", w.y), ("z", w.altitude)):
        combo.currentIndexChanged.connect(w._draw)
        combo.setMinimumContentsLength(6)
        coordinates.extend((w.coordinate_labels[axis], combo))
    detail_layout.addWidget(compact_row(*coordinates))
    w.cell_summary = QLabel()
    w.cell_summary.setWordWrap(True)
    detail_layout.addWidget(w.cell_summary)
    provenance, provenance_layout = section(detail_layout, "Source cell details", expanded=False)
    w.cell_note = QLabel()
    w.cell_note.setWordWrap(True)
    w.cell_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
    provenance_layout.addWidget(w.cell_note)
    w.inspector = QWidget()
    detail = QVBoxLayout(w.inspector)
    detail.setContentsMargins(0, 0, 0, 0)
    w.open_native = QPushButton("Open trial in visualizer")
    w.open_native.clicked.connect(lambda: w._open_native())
    w.hide_volume = check("Hide volume", w._draw)
    w.hide_volume.setToolTip(
        "Hide the fill and grid while inspecting reports. Clear the cell selection to show them again."
    )
    detail.addWidget(compact_row(w.open_native, c.native_b, w.hide_volume))
    w.contributor_note = QLabel()
    w.contributor_note.setWordWrap(True)
    w.contributor_note.hide()
    detail.addWidget(w.contributor_note)
    w.native_note = QLabel()
    w.native_note.setWordWrap(True)
    detail.addWidget(w.native_note)
    detail_layout.addWidget(w.inspector)
    detail_layout.addWidget(c.selected)
    w.dim_context = check("Dim volume", w._draw)
    w.dim_context.setParent(w)
    w.dim_context.hide()  # Older saved views can still restore dimming.
    w.highlight_cell = check("Highlight cell", w._draw, checked=True)
    detail_layout.addWidget(compact_row(w.highlight_cell, c.slice_plot))
    detail_layout.addWidget(c.slice_view)
    c.slice_expand.hide()
    c.slice_view.hide()

    w.geometry_section, context = section(layout, "Geometry and array")
    context.addWidget(c.geometry_body)
    c.geometry_expand.hide()
    w.diagnostics_section, diagnostics = section(context, "Array response")
    w.antenna = ReceivePatternControls()
    w.antenna.changed.connect(w._draw)
    diagnostics.addWidget(w.antenna)
    c._buttons(
        AlignedFormLayoutProxy(layout),
        [
            ("bookmark_save", "Save view bookmark…"),
            ("bookmark_load", "Open view bookmark…"),
            ("capture", "Save image…"),
        ],
    )
    c.buttons["bookmark_save"].setToolTip(
        "Save the camera, display settings and selected cell for this volume in a small JSON bookmark."
    )
    c.buttons["bookmark_load"].setToolTip(
        "Open a saved bookmark to restore its volume, camera, display settings and selected cell."
    )
    # Keep the signal owner alive after relocating its controls.
    c.setParent(w)
    c.hide()
    layout.addStretch()
    protect_controls(w)


class AlignedFormLayoutProxy:
    def __init__(self, layout):
        self.layout = layout

    def addRow(self, widget):
        self.layout.addWidget(widget)
