"""Focused before/after and input views below the dashboard's detail selector."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..app.plot_theme import apply_pyqtgraph_plot_theme
from ..app.theme import get_theme_manager
from ..widgets.dashboard_plot import DashboardPlot
from ..widgets.plot_lifecycle import ManagedPlotWidget
from ..widgets.wrapping_layout import WrappingLayout
from .inspection_display import PowerScale, regular_axis
from .processing_inspection import ProcessingInspection
from .spectrum_surface import SpectrumSurface


class _ProcessingLayout(QVBoxLayout):
    """Size plots by their minima while allowing child toolbars to wrap."""

    def hasHeightForWidth(self):
        # Controls wrap inside this layout. Exporting their height-for-width
        # through the movable plot grid leaves Qt caching the previous, taller
        # detector view's height when this panel replaces it.
        return False


class ProcessingPanel(QWidget):
    """Show one processing question at a time; deeper controls are opt-in."""

    reconstruction_requested = Signal()
    cell_selected = Signal(int, object, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = None
        self.mode = ""
        self._span = 60.0
        self.scale = PowerScale()
        self._blocked_selection = False
        self._selection_cleared = False
        self._image_key = None
        self._input_scope = None
        self._input_bins = {}
        self._input_elements = {}
        self._input_ranges = None
        root = _ProcessingLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        controls = WrappingLayout()
        self.range_label = QLabel("Range bin:")
        controls.addWidget(self.range_label)
        self.range_bin = QSpinBox()
        self.range_bin.valueChanged.connect(self._select_bin)
        controls.addWidget(self.range_bin)
        self.coordinate = QLabel()
        controls.addWidget(self.coordinate)
        self.element = QComboBox()
        self.element.setMinimumContentsLength(11)
        self.element.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.element.currentIndexChanged.connect(self._select_element)
        controls.addWidget(self.element)
        self.surface_toggle = QPushButton("3D surface")
        self.surface_toggle.setCheckable(True)
        self.surface_toggle.toggled.connect(self.render)
        controls.addWidget(self.surface_toggle)
        self.reconstruct = QPushButton("Reconstruct missing spectrum")
        self.reconstruct.clicked.connect(self.reconstruction_requested)
        controls.addWidget(self.reconstruct)
        self.details_toggle = QPushButton("Processing details")
        self.details_toggle.setCheckable(True)
        controls.addWidget(self.details_toggle)
        root.addLayout(controls)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        root.addWidget(self.notice)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.hide()
        self.details_toggle.toggled.connect(self.details.setVisible)
        root.addWidget(self.details)
        self.maps = QWidget()
        grid = QGridLayout(self.maps)
        grid.setContentsMargins(0, 0, 0, 0)
        self.before_plot, self.after_plot = ManagedPlotWidget(), ManagedPlotWidget()
        self.before_image, self.after_image = pg.ImageItem(axisOrder="row-major"), pg.ImageItem(
            axisOrder="row-major"
        )
        self._lines = []
        self.bars = []
        for i, (plot, image) in enumerate(
            ((self.before_plot, self.before_image), (self.after_plot, self.after_image))
        ):
            plot.setMinimumSize(180, 200)
            plot.addItem(image)
            plot.setLabel("bottom", "Range", units="m")
            plot.setLabel("left", "Doppler", units="Hz")
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#edca55", width=2))
            plot.addItem(line)
            self._lines.append(line)
            bar = pg.ColorBarItem(
                colorMap=pg.colormap.get("viridis"), label="Power (dB)", interactive=False, width=10
            )
            bar.setImageItem(image, insert_in=plot.getPlotItem())
            self.bars.append(bar)
            plot.scene().sigMouseClicked.connect(lambda event, p=plot: self._select_range(event, p))
            grid.addWidget(plot, 0, i)
        self.after_plot.setXLink(self.before_plot)
        self.after_plot.setYLink(self.before_plot)
        root.addWidget(self.maps, 3)
        self.surface = SpectrumSurface()
        self.surface.hide()
        root.addWidget(self.surface, 4)
        self.slice = DashboardPlot()
        self.slice.setMinimumHeight(230)
        self.slice.getAxis("left").enableAutoSIPrefix(False)
        self.slice.add_external_legend(labelTextSize="8pt")
        self.before_curve = self.slice.plot(pen=pg.mkPen("#3d9fec", width=2), name="Before")
        self.after_curve = self.slice.plot(pen=pg.mkPen("#ec7955", width=2), name="After")
        root.addWidget(self.slice, 2)
        self.phase = DashboardPlot()
        self.phase_curve = self.phase.plot(pen=pg.mkPen("#be87db", width=1.5))
        root.addWidget(self.phase, 2)
        self._selected_doppler = None
        self.set_model(None, "clutter")
        self.apply_theme()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = event.size().width() < 760
        grid = self.maps.layout()
        grid.removeWidget(self.after_plot)
        grid.addWidget(self.after_plot, 1 if compact else 0, 0 if compact else 1)

    def apply_theme(self) -> None:
        for plot in (self.before_plot, self.after_plot, self.slice, self.phase):
            apply_pyqtgraph_plot_theme(plot, title=plot.getPlotItem().titleLabel.text)
        theme = get_theme_manager().current_theme
        legend = self.slice.getPlotItem().legend
        legend.setBrush(pg.mkBrush(theme.bg_secondary))
        legend.setPen(pg.mkPen(theme.border_primary))
        legend.setLabelTextColor(theme.text_primary)
        for _, label in legend.items:
            label.setText(label.text)

    def set_model(self, model: ProcessingInspection | None, mode: str) -> None:
        """Replace all event-owned data; a missing/new source cannot retain old plots."""
        if model is self.model and mode == self.mode:
            return
        previous = self.model
        self.model, self.mode = model, mode
        if model is not None and (
            model.samples is not None or model.sample_power is not None or model.after is not None
        ):
            self._sync_input_scope(model)
        self.notice.setText(model.notice if model else "Loading evidence…")
        self.details.setText(model.details if model else "")
        self.reconstruct.setVisible(bool(model and model.can_reconstruct))
        if (
            model is not None
            and model.ranges is not None
            and model.doppler is not None
            and (
                previous is None
                or previous.ranges is None
                or not np.array_equal(previous.ranges, model.ranges)
            )
        ):
            self.before_plot.setRange(
                xRange=(model.ranges[0], model.ranges[-1]),
                yRange=(model.doppler[0], model.doppler[-1]),
                padding=0.02,
            )
        with QSignalBlocker(self.range_bin):
            self.range_bin.setRange(
                0,
                (
                    len(model.times) - 1
                    if model and mode == "frequency" and model.times is not None
                    else len(model.ranges) - 1 if model and model.ranges is not None else 0
                ),
            )
            if model is not None and mode in {"inputs", "frequency"}:
                chosen = self._input_bins.get(mode)
                if mode == "inputs" and self._input_ranges is not None:
                    if not np.array_equal(self._input_ranges, model.ranges):
                        chosen = None
                        self._input_bins.pop(mode, None)
                if chosen is not None and not 0 <= chosen <= self.range_bin.maximum():
                    chosen = None
                    self._input_bins.pop(mode, None)
                self.range_bin.setValue(
                    chosen if chosen is not None else (model.default_coordinate or 0)
                )
            elif previous is None and model is not None:
                power = (
                    model.after
                    if model.after is not None
                    else (
                        np.sum(np.abs(model.samples) ** 2, axis=(0, 1))[None, :]
                        if model.samples is not None
                        else None
                    )
                )
                if power is not None:
                    self.range_bin.setValue(int(np.argmax(np.sum(power, axis=0))))
        with QSignalBlocker(self.element):
            selected = self._input_elements.get(mode, 0)
            self.element.clear()
            self.element.addItems(model.elements if model else ())
            self.element.setCurrentIndex(max(0, min(selected, self.element.count() - 1)))
        self._blocked_selection = False
        self.render()

    def _sync_input_scope(self, model) -> None:
        # Coordinate choices may follow acquisitions within one processing run
        # and pair, but never carry over to a different scientific source.
        scope = (model.scale_scope[:3], model.range_label)
        if scope != self._input_scope:
            self._input_scope = scope
            self._input_bins.clear()
            self._input_elements.clear()
            self._input_ranges = None

    def _remember_coordinate(self) -> None:
        if self.model is None:
            return
        self._sync_input_scope(self.model)
        mode = "frequency" if self.mode == "frequency" else "inputs"
        self._input_bins[mode] = self.range_bin.value()
        if mode == "inputs":
            self._input_ranges = self.model.ranges

    def _select_element(self, *_args) -> None:
        if self.mode in {"inputs", "frequency"} and self.element.currentIndex() >= 0:
            self._input_elements[self.mode] = self.element.currentIndex()
        self.render()

    def reset(self, *, preserve_options: bool = False) -> None:
        self.model = None
        self._selection_cleared = False
        self._selected_doppler = None
        self.scale.reset()
        self._image_key = None
        self._input_scope = None
        self._input_bins.clear()
        self._input_elements.clear()
        self._input_ranges = None
        if not preserve_options:
            with QSignalBlocker(self.surface_toggle):
                self.surface_toggle.setChecked(False)
        self.set_model(None, "")
        self.surface.clear()

    def _select_bin(self, *_args, toggle=False) -> None:
        self._blocked_selection = False
        self._selection_cleared = False
        self._remember_coordinate()
        if self.mode != "frequency":
            self.cell_selected.emit(self.range_bin.value(), self._selected_doppler, toggle)
        self.render()

    def apply_cell(self, cell, *, explicit=False, cleared=False) -> None:
        if self.mode == "frequency":
            return
        model = self.model
        compatible = (
            cell is not None
            and model is not None
            and model.ranges is not None
            and np.array_equal(model.ranges, cell.ranges)
            and (
                model.doppler is None
                or not cell.doppler
                or np.array_equal(model.doppler, cell.doppler)
            )
        )
        self._blocked_selection = explicit and not compatible
        self._selection_cleared = cleared and self.mode != "inputs"
        if compatible:
            with QSignalBlocker(self.range_bin):
                self.range_bin.setValue(cell.range_bin)
            self._selected_doppler = cell.doppler_bin
            if self.mode == "inputs":
                self._remember_coordinate()
        else:
            self._selected_doppler = None
        self.render()
        if self._blocked_selection:
            self.notice.setText("Selected cell unavailable in this result")
        elif model is not None:
            self.notice.setText(model.notice)
            self.notice.setToolTip(
                "The selected range coordinate is shared; receiver combination may differ from the detector."
                if explicit
                else ""
            )

    def _select_range(self, event, plot) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self.model is None
            or self.model.ranges is None
            or self.model.doppler is None
            or not plot.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        point = plot.getViewBox().mapSceneToView(event.scenePos())
        self._selected_doppler = int(np.argmin(abs(self.model.doppler - point.y())))
        with QSignalBlocker(self.range_bin):
            self.range_bin.setValue(int(np.argmin(abs(self.model.ranges - point.x()))))
        self._select_bin(toggle=True)
        event.accept()

    def _set_magnitude(self, coordinates, magnitude) -> None:
        """Show absolute magnitude without magnifying floating-point noise.

        A stationary return can be constant or differ only at machine precision.
        Generic autorange then either uses a unit-wide interval, hiding a small
        return near zero, or fills the plot with insignificant fluctuations.
        A zero baseline and the saved peak give both input views a useful scale.
        """
        self.before_curve.setData(coordinates, magnitude)
        self.slice.enableAutoRange(x=True, y=False)
        peak = float(np.max(magnitude, initial=0))
        self.slice.setYRange(0, 1.05 * peak if peak > 0 else 1, padding=0)

    def render(self, *_args) -> None:
        model = self.model
        if self.mode == "frequency":
            available = model is not None and (
                model.samples is not None or model.sample_power is not None
            )
            complex_available = bool(available and model.samples is not None)
            self.maps.hide()
            self.surface.hide()
            self.surface_toggle.hide()
            self.element.setVisible(available)
            self.element.setEnabled(complex_available)
            self.element.setToolTip(
                ""
                if complex_available
                else "Only the recorded receiver-combined power was retained."
            )
            self.range_bin.setEnabled(available)
            self.range_label.setText("Slow-time sample:")
            self.slice.setVisible(available)
            self.phase.setVisible(available)
            self.phase.setEnabled(complex_available)
            self.after_curve.hide()
            self.slice.getPlotItem().legend.hide()
            self.coordinate.clear()
            self.coordinate.setToolTip("")
            if available:
                index = self.range_bin.value()
                if complex_available:
                    values = model.samples[max(0, self.element.currentIndex()), index]
                    magnitude = abs(values)
                    phase = np.angle(values)
                    phase[
                        magnitude <= max(float(magnitude.max()) * 1e-10, np.finfo(float).tiny)
                    ] = np.nan
                else:
                    magnitude = np.sqrt(model.sample_power[0, index])
                frequencies = model.frequencies / 1e6
                self._set_magnitude(frequencies, magnitude)
                if complex_available:
                    self.phase_curve.setData(frequencies, phase, connect="finite")
                else:
                    self.phase_curve.setData([], [])
                self.coordinate.setText(f"{model.times[index]:.6g} s")
                if "frequency" not in self._input_bins:
                    self.coordinate.setToolTip(
                        "Automatic sample: greatest saved CFR energy across receiver elements and frequencies."
                    )
                self.phase.enableAutoRange(x=True, y=True)
                self.slice.setTitle(
                    "Saved CFR magnitude"
                    if complex_available
                    else "Saved receiver-combined CFR magnitude"
                )
                self.phase.setTitle(
                    "Saved CFR phase"
                    if complex_available
                    else "Phase unavailable · complex observations not retained"
                )
                self.slice.setLabel("bottom", "MHz offset from carrier")
                self.phase.setLabel("bottom", "MHz offset from carrier")
                self.slice.setLabel("left", "Magnitude (relative)")
                self.phase.setLabel("left", "Phase", units="rad")
            else:
                self.before_curve.setData([], [])
                self.phase_curve.setData([], [])
            return
        inputs = self.mode == "inputs"
        available = (
            not self._blocked_selection
            and model is not None
            and (
                (model.samples is not None or model.sample_power is not None)
                if inputs
                else model.after is not None
            )
        )
        complex_available = bool(available and model.samples is not None)
        self.element.setVisible(inputs)
        self.element.setEnabled(complex_available)
        self.element.setToolTip(
            "" if complex_available else "Only the recorded receiver-combined power was retained."
        )
        self.surface_toggle.setVisible(not inputs)
        self.surface_toggle.setEnabled(available)
        self.range_bin.setEnabled(model is not None and model.ranges is not None)
        self.surface_toggle.setText(
            "Return to 2D" if self.surface_toggle.isChecked() else "3D surface"
        )
        surface = not inputs and self.surface_toggle.isChecked() and available
        self.maps.setVisible(not inputs and not surface and available)
        self.surface.setVisible(surface)
        self.layout().setStretchFactor(self.slice, 1 if surface else 2)
        self.slice.setVisible(available)
        self.phase.setVisible(inputs and available)
        self.phase.setEnabled(complex_available)
        if not available:
            for image in (self.before_image, self.after_image):
                image.clear()
            for curve in (self.before_curve, self.after_curve, self.phase_curve):
                curve.setData([], [])
            self.coordinate.clear()
            self.surface.clear()
            return
        r = self.range_bin.value()
        self.range_label.setText(model.range_label + " bin:")
        self.coordinate.setText(f"{model.ranges[r]:.6g} m")
        self.coordinate.setToolTip("")
        self.after_curve.setVisible(not inputs and model.before is not None)
        if inputs:
            if complex_available:
                values = model.samples[max(0, self.element.currentIndex()), :, r]
                magnitude = np.abs(values)
                phase = np.angle(values)
                phase[magnitude == 0] = np.nan
            else:
                magnitude = np.sqrt(model.sample_power[0, :, r])
            self._set_magnitude(model.times, magnitude)
            if complex_available:
                self.phase_curve.setData(model.times, phase, connect="finite")
            else:
                self.phase_curve.setData([], [])
            self.slice.getPlotItem().legend.hide()
            self.slice.setTitle(
                "Reconstructed CIR magnitude"
                if model.reconstructed
                else (
                    "Saved CIR magnitude"
                    if complex_available
                    else "Saved receiver-combined CIR magnitude"
                )
            )
            self.slice.setLabel("bottom", "Scenario time", units="s")
            self.slice.setLabel("left", "Magnitude")
            self.phase.setTitle(
                "Reconstructed CIR phase"
                if model.reconstructed
                else (
                    "Saved CIR phase"
                    if complex_available
                    else "Phase unavailable · complex observations not retained"
                )
            )
            self.phase.setLabel("bottom", "Scenario time", units="s")
            self.phase.setLabel("left", "Phase", units="rad")
            if "inputs" not in self._input_bins:
                self.coordinate.setToolTip(
                    "Automatic range bin: greatest saved CIR energy across receiver elements and slow time."
                )
            return
        self.slice.getPlotItem().legend.show()
        self.slice.setTitle(f"Doppler slice at {model.ranges[r]:.3g} m")
        self.slice.setLabel("bottom", "Doppler", units="Hz")
        self.slice.setLabel("left", "Power (dB)")
        self.slice.setToolTip(
            "Power in dB relative to 1 recorded linear unit; not peak-normalized."
        )
        self.slice.setXRange(float(model.doppler[0]), float(model.doppler[-1]), padding=0.02)
        (
            self.before_curve.setData(
                model.doppler, 10 * np.log10(np.maximum(model.before[:, r], 1e-30))
            )
            if model.before is not None
            else self.before_curve.setData([], [])
        )
        self.after_curve.setVisible(True)
        self.after_curve.setData(model.doppler, 10 * np.log10(np.maximum(model.after[:, r], 1e-30)))
        levels = self.scale.levels(
            (model.before, model.after),
            identity=(*model.identity, id(model)),
            scope=model.scale_scope,
            span=self._span,
        )
        self.slice.setYRange(*levels, padding=0.03)
        uniform = regular_axis(model.ranges) and regular_axis(model.doppler)
        image_key = (id(model), model.identity, levels)
        for plot, image, power, title, bar, line in zip(
            (self.before_plot, self.after_plot),
            (self.before_image, self.after_image),
            (model.before, model.after),
            (
                "Before · reconstructed" if model.reconstructed else "Before · saved",
                "After · saved",
            ),
            self.bars,
            self._lines,
            strict=True,
        ):
            plot.setLabel("bottom", model.range_label, units="m")
            plot.setTitle(title if power is not None else "Before · not recorded")
            line.setPos(model.ranges[r])
            line.setVisible(not self._selection_cleared)
            if power is None or not uniform:
                image.clear()
                if power is not None:
                    plot.setTitle("Nonuniform coordinates")
                continue
            if image_key == self._image_key:
                continue
            dr = float(np.diff(model.ranges)[0]) if len(model.ranges) > 1 else 1
            dd = float(np.diff(model.doppler)[0]) if len(model.doppler) > 1 else 1
            image.setImage(10 * np.log10(np.maximum(power, 1e-30)), autoLevels=False)
            image.setRect(
                QRectF(
                    model.ranges[0] - dr / 2,
                    model.doppler[0] - dd / 2,
                    len(model.ranges) * dr,
                    len(model.doppler) * dd,
                )
            )
            bar.setLevels(levels)
        self._image_key = image_key
        if self._selection_cleared:
            self.before_curve.clear()
            self.after_curve.clear()
            self.coordinate.setText("No cell selected")
            self.slice.setTitle("Doppler slice")
            self.surface.clear("Select an RD cell to inspect its surface")
        elif surface:
            d = (
                self._selected_doppler
                if self._selected_doppler is not None
                else int(np.argmin(abs(model.doppler)))
            )
            self.surface.set_spectra(
                model.ranges,
                model.doppler,
                model.before if model.before is not None else model.after,
                model.after if model.before is not None else None,
                selected=(min(d, len(model.doppler) - 1), r),
                labels=(
                    (
                        "Before · reconstructed" if model.reconstructed else "Before · saved",
                        "After · saved",
                    )
                    if model.before is not None
                    else ("After · saved", "")
                ),
                span_db=self._span,
                levels=levels,
                range_label=model.range_label,
                cut="range",
                cache_key=(*model.identity, id(model)),
            )

    def release(self) -> None:
        self.model = None
        self.scale.reset()
        self._image_key = None
        self.surface.release()
        for plot in (self.before_plot, self.after_plot, self.slice, self.phase):
            if plot.plotItem is not None:
                plot.close()
                # PlotWidget.close detaches from its Qt parent. Restore ownership
                # so dashboard deferred deletion also destroys the graphics scene.
                plot.setParent(self)
