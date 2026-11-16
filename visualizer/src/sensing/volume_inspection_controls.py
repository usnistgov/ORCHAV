"""Progressive controls and a linked physical-coordinate slice, without processing."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.sensing.volume.inspection import (
    DETAIL_LABELS,
    FOCUS_LABELS,
    slice_data,
    zero_radial_polygon,
)

from ..panels.collapsible_section import CollapsibleSection
from ..widgets.form_layout import AlignedFormLayout
from ..widgets.wrapping_layout import WrappingLayout


def group(layout, title):
    section = CollapsibleSection(title, start_open=False)
    layout.addWidget(section)
    body = QWidget()
    form = AlignedFormLayout(body)
    section.content_layout().addWidget(body)
    return section, body, form


def combo(items):
    widget = QComboBox()
    for key, title in items:
        widget.addItem(title, key)
    return widget


class InspectionControls(QWidget):
    changed = Signal()
    report_selected = Signal(object)
    action = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.focus_expand, self.focus_body, form = group(layout, "Filter cells")
        self.focus_form = form
        self.buttons = {}
        self.focus = combo(FOCUS_LABELS.items())
        reset_filter = QPushButton("Reset")
        reset_filter.clicked.connect(self.reset_filter)
        filter_row = QWidget()
        filter_layout = WrappingLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(self.focus)
        filter_layout.addWidget(reset_filter)
        form.addRow("Show", filter_row)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1e6)
        self.threshold.setValue(5)
        self.threshold.setSuffix(" m")
        self.threshold.setKeyboardTracking(False)
        self.minimum = QSpinBox()
        self.minimum.setRange(1, 100000)
        self.threshold_label = QLabel("Error ≥")
        form.addRow(self.threshold_label, self.threshold)
        self.minimum_label = QLabel("Extra reports ≥")
        form.addRow(self.minimum_label, self.minimum)
        self.category_filters = []
        categories = QWidget()
        category_layout = WrappingLayout(categories)
        category_layout.setContentsMargins(0, 0, 0, 0)
        for label in DETAIL_LABELS:
            check = QCheckBox(label)
            check.setChecked(True)
            category_layout.addWidget(check)
            self.category_filters.append(check)
        self.category_options = categories
        form.addRow(categories)
        categories.hide()
        self.focus_note = QLabel()
        self.focus_note.setWordWrap(True)
        form.addRow(self.focus_note)
        self.count_max = QSpinBox()
        self.count_max.setRange(1, 100000)
        self.count_max.setValue(5)
        self.count_max.setParent(self)
        self.count_max.hide()
        self.geometry_expand, self.geometry_body, geo = group(layout, "Geometry context")
        self.forward = QCheckBox("Array forward direction")
        self.motion = QCheckBox("Recorded translation vector")
        self.zero = QCheckBox("Zero direct radial velocity plane")
        for check in (self.forward, self.motion, self.zero):
            geo.addRow(check)
        self.geometry_note = QLabel(
            "Geometric references; no visibility or antenna-pattern guarantee."
        )
        self.geometry_note.setWordWrap(True)
        geo.addRow(self.geometry_note)
        self.slice_expand, self.slice_body, slices = group(layout, "Linked slices")
        self.orientation = combo(
            [("xy", "XY — fixed Z"), ("xz", "XZ — fixed Y"), ("yz", "YZ — fixed X")]
        )
        self.slice_plot = QCheckBox("Show XY, XZ and YZ slices")
        slices.addRow(self.slice_plot)
        self.sweep_controls = QWidget()
        sweep = WrappingLayout(self.sweep_controls)
        sweep.setContentsMargins(0, 0, 0, 0)
        for name, label in (("plane_previous", "‹"), ("play", "Sweep slice"), ("plane_next", "›")):
            button = QPushButton(label)
            if name != "play":
                button.setToolTip("Previous plane" if name == "plane_previous" else "Next plane")
            button.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            self.buttons[name] = button
            sweep.addWidget(button)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.2, 20)
        self.rate.setValue(2)
        self.rate.setSuffix(" planes/s")
        self.rate.setToolTip("Spatial planes per second; scenario time is unchanged.")
        self.loop = QCheckBox("Loop")
        sweep.addWidget(self.rate)
        sweep.addWidget(self.loop)
        self.sweep_note = QLabel()
        self.sweep_note.hide()
        self.slice_view = LinkedSlices()
        layout.addWidget(self.slice_view)
        self.slice_view.hide()
        self.selected = QWidget()
        selected = QVBoxLayout(self.selected)
        selected.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Report",
                "Association",
                "Position (m)",
                "Error (m)",
                "Range (m)",
                "Radial (m/s)",
                "Sampling cell",
                "Contributors",
            ]
        )
        self.table.setColumnHidden(7, True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(170)
        self.table.itemSelectionChanged.connect(self._report_changed)
        selected.addWidget(self.table)
        self.vectors = QCheckBox("Error vectors")
        self.vectors.setToolTip("Position error for the selected report and match.")
        self.through = QCheckBox("Reports through geometry")
        self.through.setToolTip(
            "Disables depth occlusion for report/reference markers. This does not mean the radar sees through the scene."
        )
        # Retain saved-view compatibility without exposing diagnostic clutter.
        for control in (self.vectors, self.through):
            control.setParent(self)
            control.hide()
        self.native_b = QPushButton("Open B trial in visualizer")
        self.native_b.clicked.connect(lambda: self.action.emit("native_b"))
        selected.addWidget(self.native_b)
        self.native_b.hide()
        self.report_note = QLabel()
        self.report_note.setWordWrap(True)
        self.report_note.setParent(self)
        self.report_note.hide()
        row = WrappingLayout()
        selected.addLayout(row)
        for name, title in [
            ("locate_report", "Locate report"),
            ("frame_reports", "Frame reports"),
            ("side_reports", "Side view"),
        ]:
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            row.addWidget(button)
        layout.addWidget(self.selected)
        self.selected.hide()
        self._buttons(QFormLayoutProxy(layout), [("capture", "Capture image + state")])
        for widget in (self.focus, self.orientation):
            widget.currentIndexChanged.connect(self.changed)
        self.focus.currentIndexChanged.connect(self._clear_category_filter)
        for widget in (self.threshold, self.minimum, self.count_max, self.rate):
            widget.valueChanged.connect(self.changed)
        for widget in (
            *self.category_filters,
            self.forward,
            self.motion,
            self.zero,
            self.slice_plot,
            self.loop,
            self.vectors,
            self.through,
        ):
            widget.toggled.connect(self.changed)

    def _clear_category_filter(self, *_):
        from PySide6.QtCore import QSignalBlocker

        with QSignalBlocker(self):
            for checkbox in self.category_filters:
                checkbox.setChecked(True)
        self.changed.emit()

    def reset_filter(self):
        self.focus.setCurrentIndex(self.focus.findData("all"))
        self._clear_category_filter()

    def set_geometry_context(self, sensor, velocity, axes, *, forward_available):
        """Explain absent reference geometry instead of offering a silent no-op."""
        sensor, velocity = np.asarray(sensor, float), np.asarray(velocity, float)
        speed = np.linalg.norm(velocity)
        self.forward.setEnabled(forward_available)
        self.motion.setEnabled(bool(speed > 0))
        intersects = len(zero_radial_polygon(sensor, velocity, axes)) >= 3
        self.zero.setEnabled(intersects)
        if speed == 0:
            note = "Stationary translation: direct radial velocity is zero everywhere; no unique plane."
        else:
            nonzero = np.flatnonzero(velocity)
            if len(nonzero) == 1:
                index = nonzero[0]
                location = f"World {'XYZ'[index]} = {sensor[index]:g} m."
            else:
                normal = ", ".join(f"{value:g}" for value in velocity / speed)
                origin = ", ".join(f"{value:g}" for value in sensor)
                location = f"Plane through sensor ({origin}) m, normal ({normal})."
            note = "Zero direct radial velocity: " + location
            if intersects:
                note += " Overlay is clipped to the sampled cell bounds."
            else:
                note += " No plane area intersects the sampled cell bounds; nothing to draw."
                if len(nonzero) == 1:
                    centres = axes["xyz"[index]]
                    half_cell = (centres[1] - centres[0]) / 2
                    note += (
                        f" Sampled {'XYZ'[index]} cell bounds: "
                        f"{centres[0] - half_cell:g} to {centres[-1] + half_cell:g} m."
                    )
        self.zero.setToolTip(note)
        self.geometry_note.setText(
            ("Array orientation not recorded. " if not forward_available else "")
            + note
            + " Geometric references; no detection guarantee. Motion vector is not attitude."
        )

    def _buttons(self, form, items):
        row = QWidget()
        layout = WrappingLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for name, title in items:
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            layout.addWidget(button)
            self.buttons[name] = button
        form.addRow(row)
        return row

    def _report_changed(self):
        rows = self.table.selectionModel().selectedRows()
        self.report_selected.emit(self.table.item(rows[0].row(), 0).text() if rows else None)

    def set_reports(self, rows, selected):
        from PySide6.QtCore import QSignalBlocker

        with QSignalBlocker(self.table):
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                position = row["position_m"]

                def fmt(value):
                    return "not recorded" if value is None else f"{value:.3f}"

                values = [
                    row["detection_id"],
                    row.get("association", "Matched" if row["matched"] else "Extra"),
                    (
                        ", ".join(f"{v:.3f}" for v in position)
                        if position is not None
                        else "not recorded"
                    ),
                    fmt(row["error_m"]),
                    fmt(row["range_m"]),
                    fmt(row["closing_velocity_m_s"]),
                    (
                        "Not available"
                        if row.get("inside_sample_cell") is None
                        else "Inside" if row["inside_sample_cell"] else "Outside"
                    ),
                    ", ".join(name.title() for name in row.get("sensor_support", [])),
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(row["scoped_id"])
                    self.table.setItem(i, col, item)
                if row["detection_id"] == selected:
                    self.table.selectRow(i)
            self.table.resizeColumnsToContents()

    def state(self):
        return dict(
            focus=self.focus.currentData(),
            threshold=self.threshold.value(),
            minimum=self.minimum.value(),
            categories=[c.isChecked() for c in self.category_filters],
            count_max=self.count_max.value(),
            orientation=self.orientation.currentData(),
            rate=self.rate.value(),
            **{
                n: getattr(self, n).isChecked()
                for n in ("forward", "motion", "zero", "slice_plot", "loop", "vectors", "through")
            },
        )

    def restore(self, state):
        from PySide6.QtCore import QSignalBlocker

        with QSignalBlocker(self):
            for name in ("focus", "orientation"):
                if name in state:
                    getattr(self, name).setCurrentIndex(getattr(self, name).findData(state[name]))
            for name in ("threshold", "minimum", "count_max", "rate"):
                if name in state:
                    getattr(self, name).setValue(state[name])
            for name in ("forward", "motion", "zero", "slice_plot", "loop", "vectors", "through"):
                if name in state:
                    getattr(self, name).setChecked(state[name])
            for check, value in zip(
                self.category_filters, state.get("categories", [True] * 4), strict=True
            ):
                check.setChecked(value)


class QFormLayoutProxy:
    def __init__(self, layout):
        self.layout = layout

    def addRow(self, widget):
        self.layout.addWidget(widget)


class LinkedSlice(QWidget):
    selected = Signal(str, float, float)

    def __init__(self):
        super().__init__()
        self.axes = None
        self.orientation = "xy"
        self.plot = None

    def _ensure_plot(self):
        if self.plot is not None:
            return
        import pyqtgraph as pg

        from ..widgets.plot_lifecycle import ManagedPlotWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = ManagedPlotWidget()
        self.plot.setFixedHeight(240)
        self.plot.setAspectLocked(True)
        self.image = pg.ImageItem(axisOrder="row-major", autoDownsample=False)
        self.plot.addItem(self.image)
        self.marker = pg.ScatterPlotItem(symbol="+", size=16, pen=pg.mkPen("white", width=2))
        self.plot.addItem(self.marker)
        self.horizontal = pg.InfiniteLine(angle=0, pen=pg.mkPen("white", width=1))
        self.vertical = pg.InfiniteLine(angle=90, pen=pg.mkPen("white", width=1))
        self.plot.addItem(self.horizontal)
        self.plot.addItem(self.vertical)
        self.plot.scene().sigMouseClicked.connect(self._click)
        layout.addWidget(self.plot)

    def _apply_theme(self):
        if getattr(self, "plot", None) is not None:
            from ..app.plot_theme import apply_pyqtgraph_plot_theme

            apply_pyqtgraph_plot_theme(
                self.plot, title=self.plot.getPlotItem().titleLabel.text, show_grid=False
            )

    def changeEvent(self, event):
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._apply_theme()
        super().changeEvent(event)

    def update_slice(self, payload, reader, orientation, key, *, show_selection=True):
        self._ensure_plot()
        normal = {"xy": "z", "xz": "y", "yz": "x"}[orientation]
        index = key[{"x": 3, "y": 2, "z": 1}[normal]]
        values, h, v, _, extent = slice_data(payload.values, reader.axes, orientation, index)
        normalized = (values - payload.limits[0]) / (payload.limits[1] - payload.limits[0])
        indices = np.clip(
            np.floor(normalized * len(payload.palette)).astype(int), 0, len(payload.palette) - 1
        )
        rgba = np.rint(payload.palette[indices] * 255).astype(np.uint8)
        self.image.setImage(rgba, autoLevels=False)
        self.image.setRect(
            QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2])
        )
        self.plot.setLabel("bottom", f"World {h.upper()}", units="m")
        self.plot.setLabel("left", f"World {v.upper()}", units="m")
        self.plot.setTitle(
            f"{orientation.upper()} · {normal.upper()} = {reader.axes[normal][index]:g} m"
        )
        self._apply_theme()
        coords = {n: reader.axes[n][key[{"x": 3, "y": 2, "z": 1}[n]]] for n in ("x", "y", "z")}
        self.marker.setData([coords[h]], [coords[v]])
        self.horizontal.setPos(coords[v])
        self.vertical.setPos(coords[h])
        for guide in (self.marker, self.horizontal, self.vertical):
            guide.setVisible(show_selection)
        signature = (reader.identity, orientation)
        if getattr(self, "signature", None) != signature:
            self.plot.setRange(xRange=extent[:2], yRange=extent[2:], padding=0.05)
            self.signature = signature
        self.axes = reader.axes
        self.orientation = orientation
        self.extent = extent

    def _click(self, event):
        if self.axes is None or event.button() != Qt.LeftButton:
            return
        p = self.plot.getViewBox().mapSceneToView(event.scenePos())
        x0, x1, y0, y1 = self.extent
        if x0 <= p.x() <= x1 and y0 <= p.y() <= y1:
            self.selected.emit(self.orientation, p.x(), p.y())


class LinkedSlices(QWidget):
    """Three native-grid sections driven by one source-cell selection."""

    selected = Signal(str, float, float)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel(
            "Click any cell to inspect it. All three crosshairs share the same source "
            "indices. These full-source slices include cells hidden by 3D filters or "
            "cutaways; transparent cells are unavailable."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        layout.addLayout(grid)
        self.views = {}
        for index, orientation in enumerate(("xy", "xz", "yz")):
            view = LinkedSlice()
            view.selected.connect(self.selected.emit)
            self.views[orientation] = view
            grid.addWidget(view, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def update_slice(self, payload, reader, orientation, key, *, show_selection=True):
        for plane, view in self.views.items():
            view.update_slice(payload, reader, plane, key, show_selection=show_selection)
