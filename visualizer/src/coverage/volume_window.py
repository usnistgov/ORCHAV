"""Progressive coverage-volume inspection in the existing physical scene."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import QDialog, QFileDialog

from shared.coverage.availability import CoverageAvailability
from shared.coverage.schema import (
    COVERAGE_PER_TX_METRICS,
    coverage_metric_base,
    resolve_coverage_tx_index,
)
from shared.coverage.spatial import coverage_roi_mask
from shared.coverage.volume import (
    ALL_TRANSMITTERS,
    coverage_file_identity,
    coverage_threshold_mask,
    load_coverage_metric_volume,
    load_coverage_volume,
)

from ..model import RenderObject, Transform
from ..renderers.protocol import volume_rendering_unavailable_reason
from ..services.label_callout_service import objects_with_callouts
from ..services.viewport_hud_service import AnalysisLegend
from ..types.camera_state import CameraState
from ..types.render_payloads import VolumePayload
from ..utils.cell_edges import grid_edge_objects, selected_cell_objects
from ..utils.volume_cut import cut_volume
from .analysis import (
    coverage_metric_color_scale,
    coverage_metric_colormap,
    coverage_metric_comparator,
    coverage_metric_label,
    coverage_metric_valid_mask,
    coverage_quantity_family,
    is_serving_tx_metric,
    serving_tx_color_hex,
    serving_tx_color_rgb,
)
from .scene_labels import CoverageSceneLabels
from .value_spinbox import CoverageValueSpinBox
from .volume_analysis import contour_object, plane_model, plane_values, region_summary
from .volume_controls import build_controls
from .volume_defaults import initial_volume_controls
from .volume_inspection import categorical_template, cell_summary, describe_cell, scalar_template
from .volume_sites import transmitter_object_ids, transmitter_objects, transmitter_position
from .volume_views import read_view, validate_view, write_view


class CoverageVolumeWindow(QDialog):
    """Own one bounded source layer, texture and spatial timer; never solve RT."""

    OBJECT_ID = "analysis:coverage_volume"
    EDGE_PREFIX = "analysis:coverage_cell_edges"
    CONTOUR_ID = "analysis:coverage_contours"
    SELECTION_PREFIX = "analysis:coverage_selected_cell"

    def __init__(self, visualizer, *, view=None, embedded=False, on_failure=None):
        super().__init__(visualizer)
        self.embedded = embedded
        self.on_failure = on_failure
        if embedded:
            self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("ORCHAV — Coverage volume")
        self.resize(520, 680)
        self.visualizer = visualizer
        self.source = visualizer.coverage_data
        opening = initial_volume_controls(visualizer) if view is None and not embedded else None
        self.initial_height_index = (
            getattr(visualizer, "coverage_height_index", 0)
            if embedded
            else opening["height_index"] if opening else None
        )
        self.path = Path(self.source["coverage_file"])
        self.identity = coverage_file_identity(self.path)
        self.old_planar_visibility = bool(visualizer.app_state.show_coverage)
        self.closed = False
        self.model = None
        self.pending = None
        self.request_key = None
        self.loaded_key = None
        self.template = None
        self.template_key = None
        self.edge_key = None
        self.edge_model = None
        self.edge_objects = ()
        self.edge_lines = ()
        self.selected_indices = None
        # Slice navigation needs planes, but opening them is not a cell pick.
        self.slice_indices = None
        self.cut_indices = None
        self._cut_source = self._cut_template = self._cut_key = None
        self.selection_origin = "Source grid selection"
        self.stepping = False
        self.restoring = None
        self._display_settings = {}
        self._configured_family = None
        self._region_key = self._plane_key = self._contour_key = self._summary_key = None
        self._region_mask = self._plane_values = None
        self._plane_template = False
        self._region_initialized = False
        self._selection_key = None
        self._selection_objects = ()
        self.scene_labels = CoverageSceneLabels(visualizer, self.source)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coverage-volume")
        self.poll = QTimer(self)
        self.poll.setInterval(50)
        self.poll.timeout.connect(self._poll)
        self.sweep = QTimer(self)
        self.sweep.setInterval(500)
        self.sweep.timeout.connect(self._step)
        build_controls(self, opening, view)

    def _open_comparison(self):
        """Reveal the current map's inline comparison without opening a dialog."""
        self.sections["Analysis"].expand()
        self.height_comparison.enabled.setChecked(True)

    @staticmethod
    def _spin(low, high, value, suffix=""):
        spin = CoverageValueSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _configure_metric(self, metric):
        """Use the main panel's exact metric; no second derivation or selector."""
        family = coverage_quantity_family(metric)
        if self._configured_family is not None:
            self._display_settings[self._configured_family] = dict(
                low=self.low.value(),
                high=self.high.value(),
                threshold=self.threshold.value(),
                colormap=self.colormap.currentText(),
                filter_enabled=self.filter_enabled.isChecked(),
            )
        saved = self._display_settings.get(family)
        self._configured_family = family
        base, selector = coverage_metric_base(metric)
        names = self.source["tx_names"]
        transmitter = (
            names[resolve_coverage_tx_index(selector, names, len(names))]
            if selector is not None or base in COVERAGE_PER_TX_METRICS
            else ALL_TRANSMITTERS
        )
        if transmitter == ALL_TRANSMITTERS and self.tx.findData(transmitter) < 0:
            self.tx.addItem("All transmitters", transmitter)
        with QSignalBlocker(self.tx), QSignalBlocker(self.quantity):
            if transmitter == ALL_TRANSMITTERS:
                self.tx.setCurrentIndex(self.tx.findData(transmitter))
            else:
                self.tx.setCurrentText(transmitter)
            label, _unit = coverage_metric_label(metric)
            index = self.quantity.findData(metric)
            if index < 0:
                self.quantity.addItem(label, metric)
                index = self.quantity.count() - 1
            self.quantity.setCurrentIndex(index)
        logarithmic = coverage_metric_color_scale(metric) == "logarithmic"
        for spin in (self.low, self.high, self.threshold):
            with QSignalBlocker(spin):
                spin.set_metric_display(metric)
                spin.setRange(1e-30 if logarithmic else -1e12, 1e12)
        low, high = self.source.get("value_min"), self.source.get("value_max")
        if low is None or high is None or not np.isfinite([low, high]).all() or low >= high:
            low, high = (1e-15, 1e-5) if logarithmic else (-130, -50)
        if logarithmic:
            low, high = max(float(low), 1e-30), max(float(high), 1e-29)
        with (
            QSignalBlocker(self.low),
            QSignalBlocker(self.high),
            QSignalBlocker(self.threshold),
            QSignalBlocker(self.colormap),
            QSignalBlocker(self.filter_enabled),
        ):
            self.low.setValue(low)
            self.high.setValue(high)
            self.threshold.setValue(np.sqrt(low * high) if logarithmic else (low + high) / 2)
            self.colormap.setCurrentText(coverage_metric_colormap(metric))
            if saved:
                for key in ("low", "high", "threshold"):
                    getattr(self, key).setValue(saved[key])
                self.colormap.setCurrentText(saved["colormap"])
                self.filter_enabled.setChecked(saved["filter_enabled"])
        self._configure_modes()

    def _configure_modes(self):
        metric = self.quantity.currentData()
        categorical = is_serving_tx_metric(metric)
        loss = coverage_metric_comparator(metric) == "<="
        self.threshold_label.setText("Threshold ≤" if loss else "Threshold ≥")
        self.mode.setItemText(1, "3D volume")
        self.mode.setItemText(2, "Minimum loss along view" if loss else "Maximum along view")
        self.mode.model().item(2).setEnabled(not categorical)
        if categorical and self.mode.currentData() == "mip":
            with QSignalBlocker(self.mode):
                self.mode.setCurrentIndex(0)

    def sync_from_panel(self):
        """Synchronize only changed state; unchanged refreshes never read HDF5."""
        metric = self.source["metric_name"]
        if metric != self.quantity.currentData():
            self._configure_metric(metric)
            self._request()
        visible = bool(self.visualizer.app_state.show_coverage)
        changed = self.visible.isChecked() != visible
        with QSignalBlocker(self.visible):
            self.visible.setChecked(visible)
        if self.model is not None and self.orientation.currentData() == 2:
            index = int(getattr(self.visualizer, "coverage_height_index", 0))
            if index != self.coordinate.currentIndex():
                with QSignalBlocker(self.coordinate):
                    self.coordinate.setCurrentIndex(index)
                self.slice_indices = (*self.slice_indices[:2], index)
                if self.selected_indices is not None:
                    self.selected_indices = (*self.selected_indices[:2], index)
                changed = True
        if changed:
            self._draw()

    def _fit_scale(self):
        if self.model is None:
            return
        valid = (
            coverage_metric_valid_mask(self.model.values, self.model.quantity)
            & ~self.model.zero_contributions
        )
        values = self.model.values[valid]
        if not values.size:
            self.status.setText("No finite values available to set the scale.")
            return
        low, high = float(values.min()), float(values.max())
        if low == high:
            high = (
                low * 10
                if coverage_metric_color_scale(self.model.quantity) == "logarithmic"
                else low + 1
            )
        with QSignalBlocker(self.low), QSignalBlocker(self.high):
            self.low.setValue(low)
            self.high.setValue(high)
        self._draw()

    def _region_bounds(self):
        return (
            [[spin.value() for spin in bounds] for bounds in self.region_bounds]
            if self.region_enabled.isChecked()
            else None
        )

    def _analysis_selection(self, threshold):
        """Cache the centre selection and criterion independently of appearance."""
        bounds = self._region_bounds()
        key = (self.loaded_key, tuple(map(tuple, bounds)) if bounds is not None else None)
        if key != self._region_key:
            self._region_mask = coverage_roi_mask(self.model.axes, bounds)
            self._region_key = key
        for bounds_widgets in self.region_bounds:
            for spin in bounds_widgets:
                spin.setEnabled(self.region_enabled.isChecked())
        for row in self.region_bound_rows:
            self.region_form.setRowVisible(row, self.region_enabled.isChecked())
        threshold_key = (self.loaded_key, threshold)
        if threshold_key != getattr(self, "_filter_key", None):
            self._filter_mask = (
                coverage_threshold_mask(self.model, threshold) if threshold is not None else None
            )
            self._filter_key = threshold_key
        return self._region_mask, self._filter_mask

    def _display_plane(self, axis, index, smoothing, region):
        key = (self.loaded_key, axis, index, smoothing, self._region_key)
        if key != self._plane_key:
            self._plane_values = plane_values(self.model, axis, index, smoothing, region)
            self._plane_key = key
        return self._plane_values

    def _draw_contours(self, mode, axis, index, smoothing, region, passed):
        enabled = (
            mode == "slice"
            and self.contours.isChecked()
            and self.visible.isChecked()
            and not is_serving_tx_metric(self.model.quantity)
        )
        key = (
            self.loaded_key,
            enabled,
            axis,
            index,
            smoothing,
            self._region_key,
            self._filter_key,
            self.contour_count.value(),
        )
        if key == self._contour_key:
            return
        self.visualizer.renderer.remove_object(self.CONTOUR_ID)
        if enabled:
            plane = self._display_plane(axis, index, smoothing, region)
            if passed is not None:
                plane = np.where(np.take(passed, index, axis=2 - axis), plane, np.nan)
            obj, levels = contour_object(
                self.model, plane, axis, index, self.contour_count.value(), self.CONTOUR_ID
            )
            if obj is not None and not self.visualizer.renderer.ensure_object(obj):
                raise RuntimeError("Renderer could not display coverage contours")
            self.contours.setToolTip("Contour levels: " + ", ".join(f"{v:g}" for v in levels))
        self._contour_key = key

    def _update_analysis_summary(self, region, passed, axis, index):
        key = (self._region_key, self._filter_key, axis, index)
        if key == self._summary_key:
            return
        lines = []
        details = []
        plane_name = "".join(d for i, d in enumerate("XYZ") if i != axis)
        plane_label = f"{plane_name} plane at {'XYZ'[axis]} = {self.model.axes[axis][index]:g} m"
        for label, normal in (("Slice", axis), ("Volume", None)):
            if normal is None:
                volume_key = (self._region_key, self._filter_key)
                if getattr(self, "_volume_summary_key", None) != volume_key:
                    self._volume_summary = region_summary(self.model, region, passed)
                    self._volume_summary_key = volume_key
                summary = self._volume_summary
            else:
                summary = region_summary(self.model, region, passed, axis=normal, index=index)
            usable = summary["finite"] + summary["zero"]
            line = f"{label}: {usable}/{summary['total']} usable cells"
            if passed is not None:
                line = f"{label}: {summary['passing']}/{summary['total']} pass ({summary['percent']:.1f}%)"
            lines.append(line)
            details.append(
                f"{plane_label if normal is not None else 'Selected volume'}: "
                f"{summary['finite']} finite · {summary['zero']} zero · "
                f"{summary['unknown']} unknown · {summary['invalid']} invalid"
            )
        if passed is not None:
            details.append(
                "Unknown/invalid cells remain in the denominator; the passing fraction is a lower bound when these are present."
            )
        details.append("Counts use raw samples; smoothing, colours and opacity do not change them.")
        self.analysis_summary.setText("\n".join(lines))
        self.analysis_summary.setToolTip("\n".join(details))
        self._summary_key = key

    def _request(self):
        self._stop()
        self.restoring = None
        previous = self.request_key
        key = (self.tx.currentData() or self.tx.currentText(), self.quantity.currentData())
        if key == previous:
            return
        self.save_view.setEnabled(False)
        self.load_view.setEnabled(False)
        self.request_key = key
        loss = coverage_metric_comparator(key[1]) == "<="
        if not self.embedded and previous and (previous[1] == "path_loss_db") != loss:
            low, high = self.low.value(), self.high.value()
            for widget, value in (
                (self.low, -high),
                (self.high, -low),
                (self.threshold, -self.threshold.value()),
            ):
                with QSignalBlocker(widget):
                    widget.setValue(value)
            if self.colormap.currentText() in ("RdYlGn", "RdYlGn_r"):
                with QSignalBlocker(self.colormap):
                    self.colormap.setCurrentText("RdYlGn_r" if loss else "RdYlGn")
        self._configure_modes()
        self.visualizer.renderer.remove_object(self.OBJECT_ID)
        self.visualizer.renderer.remove_object(self.CONTOUR_ID)
        self._contour_key = None
        self._clear_cell_edges()
        self._clear_selection()
        for object_id in transmitter_object_ids(self.source):
            self.visualizer.renderer.remove_object(object_id)
        self.visualizer.renderer.set_analysis_legend(None)
        with QSignalBlocker(self.threshold):
            self.threshold.set_metric_display(key[1])
        self.status.setText("Loading saved layer…")
        self.selection_origin = "Source grid selection"
        self.cell_note.setText("Loading selected source layer…")
        self.cell_note.setToolTip("")
        self.slice_view.hide()
        if self.pending is None:
            self._start_read()

    def _start_read(self):
        self.pending_key = self.request_key
        if self.embedded:
            self.pending = self.executor.submit(
                load_coverage_metric_volume, self.path, self.pending_key[1]
            )
        else:
            self.pending = self.executor.submit(load_coverage_volume, self.path, *self.pending_key)
        self.poll.start()

    def _poll(self):
        if self.closed:
            return
        if self.visualizer.coverage_data is not self.source:
            self.close()
            return
        if self.pending is None or not self.pending.done():
            return
        self.poll.stop()
        future, key = self.pending, self.pending_key
        self.pending = None
        if key != self.request_key:
            self._start_read()
            return
        try:
            model = future.result()
            if model.source_identity != self.identity:
                raise ValueError("Coverage file changed; reopen the scenario")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.status.setText(f"Coverage volume unavailable: {exc}")
            self.visualizer.renderer.remove_object(self.OBJECT_ID)
            self.model = None
            if self.on_failure:
                self.on_failure(str(exc))
            return
        first = self.model is None
        self.model, self.loaded_key = model, key
        for axis, bounds in zip(model.axes, self.region_bounds, strict=True):
            for spin, value in zip(bounds, (axis[0], axis[-1]), strict=True):
                with QSignalBlocker(spin):
                    step = float(axis[1] - axis[0])
                    spin.setRange(float(axis[0]), float(axis[-1]))
                    spin.setSingleStep(step)
                    spin.setToolTip(f"Arrow step: {step:g} m (recorded grid spacing).")
                    if not self._region_initialized:
                        spin.setValue(value)
        self._region_initialized = True
        self._region_key = self._plane_key = self._contour_key = self._summary_key = None
        # Grid outlines include known zero solver contributions: these are
        # computed cells even though their power fill is transparent. Unknown
        # cells remain separate and require the unavailable-edge option.
        self.valid_cells = (
            np.isfinite(model.values) & (model.values >= 0)
            if is_serving_tx_metric(model.quantity)
            else coverage_metric_valid_mask(model.values, model.quantity) | model.zero_contributions
        )
        self.valid_count = int(np.count_nonzero(self.valid_cells))
        self.reason_counts = (
            tuple(
                int(np.count_nonzero(model.availability == code))
                for code in (
                    CoverageAvailability.ZERO_CONTRIBUTION,
                    CoverageAvailability.INVALID,
                    CoverageAvailability.UNKNOWN,
                )
            )
            if model.availability is not None
            else None
        )
        self.threshold_count_key = None
        if self.embedded:
            self.source["_volume_layer"] = model
        if self.slice_indices is None:
            self.slice_indices = tuple(len(axis) // 2 for axis in model.axes)
            if self.initial_height_index is not None:
                index = int(np.clip(self.initial_height_index, 0, len(model.axes[2]) - 1))
                self.slice_indices = (*self.slice_indices[:2], index)
        self.template = None
        self.template_key = None
        # Hide the ordinary plane while inspecting; restore only for this source.
        if self.embedded:
            self.visualizer.coverage_volume_active = True
            controller = getattr(self.visualizer.ui_controller, "coverage_controller", None)
            if controller is not None:
                # Ownership changed outside AppState. Invalidate the stationary
                # frame shortcut so the renderer actually retires its old plane.
                self.visualizer.force_update_next_frame = True
                controller._render_current_coverage()
        else:
            self.visualizer.ui_controller.handle_coverage_toggled(False)
        restore, self.restoring = self.restoring, None
        self._populate_cut_positions(coordinate=restore.get("cut_coordinate") if restore else None)
        self._orient(coordinate=restore["coordinate"] if restore else None)
        if restore:
            if not self.visualizer.renderer.set_camera_state(
                CameraState.from_dict(restore["camera"])
            ):
                self.status.setText(
                    "View controls restored, but the renderer could not restore its camera."
                )
        elif first and not self.embedded:
            self._frame()
        self.save_view.setEnabled(True)
        self.load_view.setEnabled(True)

    def _orient(self, *_args, coordinate=None):
        self._stop()
        self.restoring = None
        if self.model is None:
            return
        with QSignalBlocker(self.coordinate):
            self.coordinate.clear()
            for v in self.model.axes[self.orientation.currentData()]:
                self.coordinate.addItem(f"{v:g} m", float(v))
            index = (
                self.selected_indices[self.orientation.currentData()]
                if self.selected_indices is not None
                else self.slice_indices[self.orientation.currentData()]
            )
            if coordinate is not None:
                index = int(
                    np.argmin(np.abs(self.model.axes[self.orientation.currentData()] - coordinate))
                )
            self.coordinate.setCurrentIndex(index)
        navigation = list(self.slice_indices)
        navigation[self.orientation.currentData()] = index
        self.slice_indices = tuple(navigation)
        if self.selected_indices is not None:
            selected = list(self.selected_indices)
            selected[self.orientation.currentData()] = index
            self.selected_indices = tuple(selected)
        self._draw()
        self._publish_xy_height()

    def _coordinate_changed(self):
        if self.slice_indices is not None:
            navigation = list(self.slice_indices)
            navigation[self.orientation.currentData()] = self.coordinate.currentIndex()
            self.slice_indices = tuple(navigation)
        if self.selected_indices is not None:
            selected = list(self.selected_indices)
            selected[self.orientation.currentData()] = self.coordinate.currentIndex()
            self.selected_indices = tuple(selected)
            self.selection_origin = "Source grid selection"
        if not self.stepping:
            self._stop()
        self._draw()
        self._publish_xy_height()

    def _mode_changed(self):
        self._stop()
        self._draw()

    def _populate_cut_positions(self, *, coordinate=None):
        if self.model is None:
            return
        if self.cut_indices is None:
            self.cut_indices = [len(axis) - 1 for axis in self.model.axes]
        axis = self.cut_axis.currentData()
        positions = self.model.axes[axis]
        index = self.cut_indices[axis]
        if coordinate is not None:
            index = int(np.argmin(np.abs(positions - coordinate)))
        with QSignalBlocker(self.cut_position):
            self.cut_position.clear()
            for value in positions:
                self.cut_position.addItem(f"{value:g} m", float(value))
            self.cut_position.setCurrentIndex(index)
        self.cut_indices[axis] = index

    def _cut_axis_changed(self):
        self._stop()
        self._populate_cut_positions()
        self._draw()

    def _cut_position_changed(self):
        if self.cut_indices is not None:
            self.cut_indices[self.cut_axis.currentData()] = self.cut_position.currentIndex()
        if not self.stepping:
            self._stop()
        self._draw()

    def _cut_toggled(self, enabled):
        if not enabled:
            self._stop()
        self._draw()

    def _cut_state(self):
        if self.mode.currentData() == "iso" and self.cut_enabled.isChecked():
            return ("xyz"[self.cut_axis.currentData()], self.cut_position.currentIndex(), True)
        return None

    def _display_template(self):
        """Retain at most one cutaway; appearance edits reuse its texture array."""
        cut = self._cut_state()
        if cut is None:
            self._cut_source = self._cut_template = self._cut_key = None
            return self.template
        if self._cut_source is not self.template or self._cut_key != cut:
            self._cut_template = cut_volume(self.template, cut)
            self._cut_source, self._cut_key = self.template, cut
        return self._cut_template

    def _publish_xy_height(self):
        """Keep horizontal slice graphs aligned without changing orthogonal cuts."""
        if (
            self.embedded
            and self.orientation.currentData() == 2
            and self.mode.currentData() == "slice"
            and self.coordinate.currentIndex() >= 0
        ):
            self.visualizer.ui_controller.handle_coverage_height_changed(
                self.coordinate.currentIndex()
            )

    def _draw(self):
        self.restoring = None
        self.unavailable_edges.setEnabled(self.cell_edges.isChecked())
        if self.model is None or self.loaded_key != self.request_key or self.closed:
            self._clear_cell_edges()
            self._clear_selection()
            return
        categorical = is_serving_tx_metric(self.model.quantity)
        valid = self.valid_cells
        mode = self.mode.currentData()
        if self.embedded:
            panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get(
                "coverage"
            )
            if panel is not None:
                panel.set_volume_controls_active(True)
            self.visualizer.coverage_volume_active = True
            self.visualizer.coverage_opacity = self.opacity.value()
        if mode not in ("slice", "iso") or not self.visible.isChecked():
            self._stop()
        self.view_form.setRowVisible(self.sweep_controls, mode in ("slice", "iso"))
        self.cut_axis.setVisible(mode == "iso")
        self.view_form.setRowVisible(self.cut_position, mode == "iso")
        self.cut_position.setEnabled(self.cut_enabled.isChecked())
        for widget in (self.coordinate, self.orientation):
            self.view_form.setRowVisible(widget, mode == "slice")
        self.view_form.setRowVisible(self.filter_enabled, not categorical)
        active_threshold = not categorical and (
            self.filter_enabled.isChecked() or (not self.embedded and mode == "iso")
        )
        self.view_form.setRowVisible(self.threshold, active_threshold)
        self.threshold.setEnabled(active_threshold)
        self.appearance_form.setRowVisible(self.smoothing, mode == "slice" and not categorical)
        self.contour_controls.setVisible(mode == "slice" and not categorical)
        self.contour_count.setVisible(
            mode == "slice" and not categorical and self.contours.isChecked()
        )
        self.smoothing.setEnabled(not categorical)
        self.contours.setEnabled(not categorical)
        self.contour_count.setEnabled(not categorical and self.contours.isChecked())
        self.view_form.setRowVisible(
            self.region_tx, categorical and len(self.source["tx_names"]) > 1
        )
        self.threshold_edges.setVisible(
            self.cell_edges.isChecked() and (active_threshold or (categorical and mode == "iso")),
        )
        self.unavailable_edges.setVisible(self.cell_edges.isChecked())
        if self.embedded:
            with QSignalBlocker(self.colour_mode):
                self.colour_mode.setCurrentIndex(0)
        uniform = (
            not self.embedded
            and not categorical
            and (self.colour_mode.currentData() == "uniform" or mode == "iso")
        )
        for widget in (self.scale_controls, self.colormap_controls):
            self.appearance_form.setRowVisible(widget, not categorical and not uniform)
        self.uniform_colour.setEnabled(mode == "iso")
        self.threshold_edges.setEnabled(
            self.cell_edges.isChecked() and (active_threshold or (categorical and mode == "iso"))
        )
        self.show_nonfinite.setVisible(mode != "iso")
        self.show_nonfinite.setEnabled(mode != "iso")
        self.show_nonfinite.setToolTip(
            "Available in Slice and Projection views. The 3D volume surface excludes no-data cells."
            if mode == "iso"
            else "Neutral grey lies outside the colour scale. Cell inspection distinguishes zero contributions from unavailable estimates."
        )
        for widget in (self.coordinate, self.orientation):
            widget.setEnabled(mode == "slice")
        for widget in (self.play, self.rate, self.loop):
            widget.setEnabled(mode in ("slice", "iso"))
        low, high = self.low.value(), self.high.value()
        threshold = self.threshold.value() if active_threshold else None
        colormap = self.colormap.currentText()
        show_nonfinite = self.show_nonfinite.isChecked()
        axis = self.orientation.currentData()
        coordinate = self.coordinate.currentData()
        index = self.coordinate.currentIndex()
        smoothing = self.smoothing.currentData() if mode == "slice" and not categorical else "none"
        try:
            region, passed = self._analysis_selection(threshold)
        except ValueError as exc:
            self.status.setText(str(exc))
            self.visualizer.renderer.remove_object(self.OBJECT_ID)
            self.visualizer.renderer.remove_object(self.CONTOUR_ID)
            self._contour_key = None
            self._clear_cell_edges()
            self._clear_selection()
            return
        displayed = (
            region
            if passed is None
            else region & (passed | ~np.isfinite(self.model.values) | self.model.zero_contributions)
        )
        # Filtering zeros never turns them into threshold-qualified cells;
        # optional grey is an availability cue only.
        key = (
            self.loaded_key,
            self._region_key,
            smoothing,
            (axis, index) if smoothing != "none" else None,
            uniform,
            self.solid_colour.currentData(),
            low,
            high,
            threshold,
            "iso" if mode == "iso" else "scalar",
            colormap,
            show_nonfinite,
            self.region_tx.currentData() if categorical else None,
        )
        try:
            if key != self.template_key:
                if categorical:
                    if mode == "mip":
                        raise ValueError("Serving identities cannot be maximum-projected")
                    region_index = self.region_tx.currentData()
                    if mode == "iso" and region_index >= 0:
                        values = (self.model.values == region_index).astype(np.float32)
                        colour = (*serving_tx_color_rgb(region_index), 1)
                        self.template = VolumePayload(
                            values, np.array([[0, 0, 0, 0], colour], np.float32), (0, 1)
                        )
                    else:
                        self.template = categorical_template(
                            self.model,
                            show_nonfinite=show_nonfinite and mode != "iso",
                            transmitter_index=region_index if region_index >= 0 else None,
                        )
                elif mode == "iso" and uniform:
                    membership = region & (
                        passed
                        if passed is not None
                        else coverage_metric_valid_mask(self.model.values, self.model.quantity)
                        & ~self.model.zero_contributions
                    )
                    self.template = VolumePayload(
                        membership.astype(np.float32),
                        np.array([[0, 0, 0, 0], self.solid_colour.currentData()], np.float32),
                        (0, 1),
                    )
                else:
                    display_model = self.model
                    display_mask = (
                        displayed
                        if threshold is not None or self.region_enabled.isChecked()
                        else None
                    )
                    if smoothing != "none":
                        plane = self._display_plane(axis, index, smoothing, region)
                        display_model = plane_model(self.model, plane, axis, index)
                        if display_mask is not None:
                            display_mask = np.expand_dims(
                                np.take(display_mask, index, axis=2 - axis), 2 - axis
                            )
                    self.template = scalar_template(
                        display_model,
                        low,
                        high,
                        colormap=colormap,
                        show_nonfinite=show_nonfinite and mode != "iso",
                        visible_mask=display_mask,
                        solid_colour=self.solid_colour.currentData() if uniform else None,
                    )
                if categorical and self.region_enabled.isChecked():
                    self.template = replace(
                        self.template, values=np.where(region, self.template.values, 0)
                    )
                self.template_key = key
            axis = self.orientation.currentData()
            coordinate = self.coordinate.currentData()
            plane = tuple(float(i == axis) for i in range(3)) + (-coordinate,)
            payload = replace(
                self._display_template(),
                mode=mode,
                plane=plane,
                opacity=self.opacity.value(),
                lighting=not self.embedded and not self.uniform_colour.isChecked(),
                surface_grid=self.cell_edges.isChecked() and mode == "iso",
            )
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(self.model.spacing)
            matrix[:3, 3] = [v[0] for v in self.model.axes]
            self._plane_template = smoothing != "none"
            if self._plane_template:
                matrix[axis, 3] = coordinate
            obj = RenderObject(
                id=self.OBJECT_ID,
                payload=payload,
                transform=Transform(matrix),
                visibility=self.visible.isChecked(),
            )
            if not self.visualizer.renderer.ensure_object(obj):
                raise RuntimeError("Renderer could not display the volume")
            self.sync_transmitter_labels()
            edge_lines = self._draw_cell_edges(valid, mode, axis, threshold)
            self._draw_contours(mode, axis, index, smoothing, region, passed)
            self._update_analysis_summary(region, passed, axis, index)
            self._draw_selection(mode, axis)
            self.cell_table.charts.set_context(self.model, self.selected_indices, threshold)
            self.cell_table.request(self.selected_indices)
            self.height_comparison.set_context(self.model.axes, self._region_bounds())
        except (ValueError, RuntimeError) as exc:
            self._clear_cell_edges()
            self.status.setText(str(exc))
            return
        self.slice_view.setVisible(self.show_slices.isChecked())
        if self.selected_indices is not None:
            self.cell_note.setText(cell_summary(self.model, self.selected_indices))
            self.cell_note.setToolTip(
                describe_cell(self.model, self.selected_indices, self.selection_origin)
            )
        else:
            self.cell_note.setText("Click a cell to inspect it.")
            self.cell_note.setToolTip("")
        if self.show_slices.isChecked():
            slice_key = (self.loaded_key, low, high, colormap, show_nonfinite)
            if getattr(self, "_slice_template_key", None) != slice_key:
                self._slice_template = (
                    categorical_template(self.model, show_nonfinite=show_nonfinite)
                    if categorical
                    else scalar_template(
                        self.model, low, high, colormap=colormap, show_nonfinite=show_nonfinite
                    )
                )
                self._slice_template_key = slice_key
            reader = SimpleNamespace(
                axes=dict(zip("xyz", self.model.axes, strict=True)), identity=self.identity
            )
            self.slice_view.update_slice(
                self._slice_template,
                reader,
                "xy",
                (0, *reversed(self.selected_indices or self.slice_indices)),
                show_selection=self.selected_indices is not None,
            )
        values = self.model.values
        if threshold is not None:
            count_key = (self.loaded_key, self.threshold.value(), self._region_key)
            if self.threshold_count_key != count_key:
                self.threshold_count = int(np.count_nonzero(self._filter_mask & region))
                self.threshold_count_key = count_key
        passed = getattr(self, "threshold_count", 0)
        loss = coverage_metric_comparator(self.model.quantity) == "<="
        comparator = "≤" if loss else "≥"
        _, unit = coverage_metric_label(self.model.quantity)
        selected_count = self._volume_summary["finite"] + self._volume_summary["zero"]
        denominator = self._volume_summary["total"] if self.embedded else selected_count
        denominator_label = "selected cells" if self.embedded else "usable cells"
        # Keep routine grid diagnostics on hover; a failed edge request must
        # still be visible even when the display-details section is collapsed.
        edge_warning = edge_lines[:1] if edge_lines and not self.edge_objects else ()
        edge_warning = tuple(
            line for line in edge_warning if line.startswith("Cell edges disabled")
        )
        self.status.setText(
            f"{values.shape[2]} × {values.shape[1]} × {values.shape[0]} cells · "
            f"{self._volume_summary['total'] - selected_count} unavailable"
            + (" in selected region" if self.region_enabled.isChecked() else "")
            + (
                f"\n{passed}/{denominator} {denominator_label} {comparator} {self.threshold.text()}"
                if threshold is not None
                else ""
            )
            + ("\n" + "\n".join(edge_warning) if edge_warning else "")
        )
        self.status.setToolTip(self.analysis_summary.toolTip() + "\n" + "\n".join(edge_lines))
        selection = (
            "All transmitters"
            if self.loaded_key[0] == ALL_TRANSMITTERS
            else str(self.loaded_key[0])
        )
        view_label = (
            f"{'XYZ'[axis]} = {coordinate:g} m"
            if mode == "slice"
            else (
                "3D volume"
                if mode == "iso"
                else "Minimum loss along view" if loss else "Maximum along view"
            )
        )
        if categorical:
            selection = (
                "All transmitters"
                if self.region_tx.currentData() < 0
                else self.region_tx.currentText()
            )
        lines = [f"{selection} · {view_label}"]
        filters = []
        if threshold is not None:
            filters.append(f"Filter {comparator} {self.threshold.text()}")
        if self.region_enabled.isChecked():
            filters.append("Selected region")
        if filters:
            lines.append(" · ".join(filters))
        if self._cut_state() is not None:
            lines[
                0
            ] += f" · {'XYZ'[self.cut_axis.currentData()]} ≤ {self.cut_position.currentData():g} m"
        metric_label, _ = coverage_metric_label(coverage_metric_base(self.model.quantity)[0])
        scale_label = unit
        if coverage_metric_color_scale(self.model.quantity) == "logarithmic":
            scale_label += " · log scale"
        self.visualizer.renderer.set_analysis_legend(
            AnalysisLegend(
                title="Coverage — " + metric_label,
                lines=tuple(lines),
                swatches=(
                    tuple(
                        (name, serving_tx_color_hex(i))
                        for i, name in enumerate(self.source["tx_names"])
                        if self.region_tx.currentData() in (-1, i)
                    )
                    if categorical
                    else (
                        (
                            (
                                "Selected cells",
                                "#%02x%02x%02x"
                                % tuple(
                                    round(c * 255) for c in self.solid_colour.currentData()[:3]
                                ),
                            ),
                        )
                        if uniform
                        else (("No data", "#808080"),) if show_nonfinite and mode != "iso" else ()
                    )
                ),
                colorbar_label=scale_label if not uniform and not categorical else "",
                colorbar_range=(low, high) if not uniform and not categorical else None,
                colorbar_colormap=colormap,
            )
            if self.visible.isChecked()
            else None
        )

    def _clear_selection(self):
        for obj in self._selection_objects:
            self.visualizer.renderer.remove_object(obj.id)
        self._selection_objects = ()
        self._selection_key = None

    def _draw_selection(self, mode, axis):
        cut = self._cut_state()
        if (
            not self.highlight_cell.isChecked()
            or not self.visible.isChecked()
            or self.selected_indices is None
            or (cut is not None and self.selected_indices["xyz".index(cut[0])] > cut[1])
        ):
            self._clear_selection()
            return
        normal = "xyz"[axis] if mode == "slice" else None
        key = (self.loaded_key, self.selected_indices, normal)
        if key != self._selection_key:
            self._selection_objects = selected_cell_objects(
                dict(zip("xyz", self.model.axes, strict=True)),
                self.selected_indices,
                normal=normal,
                prefix=self.SELECTION_PREFIX,
            )
            self._selection_key = key
        for obj in self._selection_objects:
            if not self.visualizer.renderer.ensure_object(obj):
                raise RuntimeError("Renderer could not highlight the selected cell")

    def _draw_cell_edges(self, valid, mode, axis, threshold):
        """Reuse exact edges while only volume material or labels change.

        A loaded CoverageVolume is immutable. Keep its identity with the one
        cached layer, and still reconcile every object so a removed native
        object can be repaired. Failed draws never commit a cache hit.
        """
        selected_region = self.threshold_edges.isChecked() and (
            threshold is not None or (mode == "iso" and is_serving_tx_metric(self.model.quantity))
        )
        key = (
            self.loaded_key,
            self._region_key,
            self.cell_edges.isChecked(),
            self.unavailable_edges.isChecked(),
            self.visible.isChecked(),
            (axis, self.coordinate.currentIndex()) if mode == "slice" else None,
            self._cut_state(),
            (threshold, self.region_tx.currentData()) if selected_region else None,
        )
        if self.edge_model is self.model and self.edge_key == key:
            objects, lines = self.edge_objects, self.edge_lines
        else:
            objects, lines = grid_edge_objects(
                dict(zip("xyz", self.model.axes)),
                valid,
                enabled=self.cell_edges.isChecked(),
                include_unavailable=self.unavailable_edges.isChecked(),
                display="slice" if mode == "slice" else "both",
                normal="xyz"[axis],
                index=self.coordinate.currentIndex(),
                cut=self._cut_state(),
                visible=self.visible.isChecked(),
                prefix=self.EDGE_PREFIX,
                available_label=(
                    (
                        "Below-threshold saved"
                        if self.model.quantity == "path_loss_db"
                        else "Above-threshold saved"
                    )
                    if selected_region
                    else "Usable saved"
                ),
                unavailable_label="Unavailable",
                selected=(
                    (
                        (
                            valid
                            if self.region_tx.currentData() < 0
                            else self.model.values == self.region_tx.currentData()
                        )
                        & self._region_mask
                        if is_serving_tx_metric(self.model.quantity)
                        else coverage_threshold_mask(self.model, threshold) & self._region_mask
                    )
                    if selected_region
                    else self._region_mask if self.region_enabled.isChecked() else None
                ),
            )
        desired_ids = {obj.id for obj in objects}
        # Reconcile absence too: a failed native removal remains retryable on
        # an identical redraw, even after the desired cached layer is empty.
        for suffix in ("available", "unavailable"):
            object_id = f"{self.EDGE_PREFIX}:{suffix}"
            if object_id not in desired_ids:
                self.visualizer.renderer.remove_object(object_id)
        for obj in objects:
            if not self.visualizer.renderer.ensure_object(obj):
                raise RuntimeError("Renderer could not display coverage cell edges")
        self.edge_model = self.model
        self.edge_key = key
        self.edge_objects, self.edge_lines = tuple(objects), lines
        return lines

    def sync_transmitter_labels(self):
        """Update annotations alone when scene nodes or shared preferences change."""
        if self.closed or self.model is None:
            return
        if self.embedded:
            self.scene_labels.sync(visible=self.visible.isChecked())
            return
        # The scripted figure API retains its explicit locator options.
        categorical = is_serving_tx_metric(self.model.quantity)
        site_objects = (
            transmitter_objects(
                self.source,
                (
                    self.source["tx_names"][self.region_tx.currentData()]
                    if categorical and self.region_tx.currentData() >= 0
                    else self.loaded_key[0]
                ),
                self.model.axes,
                readable=self.readable_transmitter.isChecked(),
            )
            if self.show_transmitter.isChecked() and self.visible.isChecked()
            else []
        )
        desired = {obj.id for obj in site_objects}
        for object_id in transmitter_object_ids(self.source):
            if object_id not in desired:
                self.visualizer.renderer.remove_object(object_id)
        for obj in objects_with_callouts(site_objects, self.visualizer):
            if not self.visualizer.renderer.ensure_object(obj):
                raise RuntimeError("Renderer could not display the transmitter locator")

    def _clear_cell_edges(self):
        """Release both status layers and cached geometry on failure/change/close."""
        self.edge_key = None
        self.edge_model = None
        self.edge_objects, self.edge_lines = (), ()
        for suffix in ("available", "unavailable"):
            self.visualizer.renderer.remove_object(f"{self.EDGE_PREFIX}:{suffix}")

    def _frame(self):
        if self.model is None:
            return
        lo = np.array([a[0] for a in self.model.axes]) - np.asarray(self.model.spacing) / 2
        hi = np.array([a[-1] for a in self.model.axes]) + np.asarray(self.model.spacing) / 2
        names = (
            self.source["tx_names"]
            if self.loaded_key[0] == ALL_TRANSMITTERS
            else [self.loaded_key[0]]
        )
        if self.show_transmitter.isChecked():
            for name in names:
                position = transmitter_position(self.source, name)
                if position is not None:
                    lo, hi = np.minimum(lo, position), np.maximum(hi, position)
                    if not self.embedded:
                        span = max(float(np.ptp(axis)) for axis in self.model.axes)
                        hi[2] = max(
                            hi[2],
                            max(position[2], float(self.model.axes[2][-1]))
                            + max(2.0, span * 0.08)
                            * (2 if self.readable_transmitter.isChecked() else 1),
                        )
        self.visualizer.renderer.set_overview_camera("isometric", ((hi + lo) / 2, (hi - lo) * 1.3))

    def _play(self):
        if self.sweep.isActive():
            self._stop()
        elif (
            self.model is not None
            and self.loaded_key == self.request_key
            and self.mode.currentData() in ("slice", "iso")
            and self.visible.isChecked()
        ):
            if self.mode.currentData() == "iso":
                self.cut_enabled.setChecked(True)
            coordinate = self._sweep_coordinate()
            if coordinate.currentIndex() == coordinate.count() - 1:
                coordinate.setCurrentIndex(0)
            self.sweep.start()
            self.play.setText("Pause sweep")

    def _rate_changed(self):
        self.sweep.setInterval(max(50, round(1000 / self.rate.value())))

    def _stop(self):
        self.sweep.stop()
        self.play.setText("Sweep 3D" if self.mode.currentData() == "iso" else "Sweep slice")

    def _sweep_coordinate(self):
        return self.cut_position if self.mode.currentData() == "iso" else self.coordinate

    def _step(self):
        coordinate = self._sweep_coordinate()
        index = coordinate.currentIndex() + 1
        if index >= coordinate.count():
            if self.loop.isChecked() and coordinate.count() > 1:
                index = 0
            else:
                self._stop()
                return
        self.stepping = True
        try:
            coordinate.setCurrentIndex(index)
        finally:
            self.stepping = False

    def _apply_view_controls(self, view):
        """Restore validated controls without emitting partially restored views."""
        if self.embedded and view.get("metric"):
            self._configure_metric(view["metric"])
        for key, data in (
            ("tx", "transmitter"),
            ("quantity", "quantity"),
            ("mode", "mode"),
            ("orientation", "axis"),
            ("smoothing", "smoothing"),
            ("colour_mode", "colour_mode"),
        ):
            widget = getattr(self, key)
            with QSignalBlocker(widget):
                value = view.get(data, "none" if key == "smoothing" else "metric")
                index = widget.findData(value)
                if key == "tx" and index < 0:
                    index = widget.findText(view[data])
                widget.setCurrentIndex(index)
        with QSignalBlocker(self.colormap):
            self.colormap.setCurrentText(view.get("colormap", "viridis"))
        for key in ("threshold", "opacity", "low", "high", "rate"):
            with QSignalBlocker(getattr(self, key)):
                getattr(self, key).setValue(view[key])
        for key, default in (
            ("visible", True),
            ("loop", False),
            ("filter_enabled", False),
            ("contours", False),
            ("region_enabled", False),
            ("highlight_cell", False),
            ("cut_enabled", False),
            ("show_nonfinite", False),
            ("cell_edges", False),
            ("uniform_colour", False),
            ("threshold_edges", False),
            ("unavailable_edges", False),
            ("show_transmitter", True),
            ("readable_transmitter", False),
        ):
            with QSignalBlocker(getattr(self, key)):
                getattr(self, key).setChecked(view.get(key, default))
        with QSignalBlocker(self.cut_axis):
            self.cut_axis.setCurrentIndex(self.cut_axis.findData(view.get("cut_axis", 2)))
        with QSignalBlocker(self.region_tx):
            self.region_tx.setCurrentIndex(self.region_tx.findData(view.get("region_tx", -1)))
        indices = view.get("selected_indices")
        self.selected_indices = tuple(indices) if indices is not None else None
        self.selection_origin = "Saved cell selection"
        with QSignalBlocker(self.contour_count), QSignalBlocker(self.solid_colour):
            self.contour_count.setValue(view.get("contour_count", 6))
            self.solid_colour.setCurrentIndex(view.get("solid_colour", 0))
        if view.get("region_enabled", False):
            for widgets, bounds in zip(self.region_bounds, view["region_bounds"], strict=True):
                for spin, value in zip(widgets, bounds, strict=True):
                    with QSignalBlocker(spin):
                        spin.setValue(value)
            self._region_initialized = True
        self._rate_changed()

    def _view_state(self):
        if self.visualizer.coverage_data is not self.source:
            raise ValueError("Coverage result changed; reopen the inspector")
        if self.model is None or self.loaded_key != self.request_key:
            raise ValueError("Wait for the coverage layer to load")
        camera = self.visualizer.renderer.get_camera_state()
        if camera is None:
            raise ValueError("Camera is unavailable")
        state = dict(
            schema="orchav.coverage-view.v3",
            canonical=self.embedded,
            filter_enabled=self.filter_enabled.isChecked(),
            highlight_cell=self.highlight_cell.isChecked(),
            selected_indices=(
                list(self.selected_indices) if self.selected_indices is not None else None
            ),
            smoothing=self.smoothing.currentData(),
            contours=self.contours.isChecked(),
            contour_count=self.contour_count.value(),
            colour_mode=self.colour_mode.currentData(),
            solid_colour=self.solid_colour.currentIndex(),
            region_enabled=self.region_enabled.isChecked(),
            region_bounds=self._region_bounds(),
            source=list(self.identity),
            transmitter=self.tx.currentData() or self.tx.currentText(),
            quantity=self.quantity.currentData(),
            colormap=self.colormap.currentText(),
            mode=self.mode.currentData(),
            axis=self.orientation.currentData(),
            coordinate=self.coordinate.currentData(),
            cut_enabled=self.cut_enabled.isChecked(),
            cut_axis=self.cut_axis.currentData(),
            cut_coordinate=self.cut_position.currentData(),
            visible=self.visible.isChecked(),
            show_nonfinite=self.show_nonfinite.isChecked(),
            cell_edges=self.cell_edges.isChecked(),
            uniform_colour=self.uniform_colour.isChecked(),
            threshold_edges=self.threshold_edges.isChecked(),
            unavailable_edges=self.unavailable_edges.isChecked(),
            show_transmitter=self.show_transmitter.isChecked(),
            readable_transmitter=self.readable_transmitter.isChecked(),
            loop=self.loop.isChecked(),
            camera=camera.to_dict(),
        )
        if self.embedded:
            state.update(metric=self.quantity.currentData(), region_tx=self.region_tx.currentData())
        state.update(
            {
                key: getattr(self, key).value()
                for key in ("threshold", "opacity", "low", "high", "rate")
            }
        )
        return validate_view(
            state, coverage_file_identity(self.path), self.source["tx_names"], self.model.axes
        )

    def _save_view(self, path):
        write_view(path, self._view_state())

    def _load_view(self, path):
        if self.visualizer.coverage_data is not self.source:
            raise ValueError("Coverage result changed; reopen the inspector")
        state = validate_view(
            read_view(path),
            coverage_file_identity(self.path),
            self.source["tx_names"],
            self.model.axes,
        )
        if tuple(state["source"]) != self.identity:
            raise ValueError("Coverage file changed; reopen the scenario")
        if self.embedded:
            if not (state["schema"] == "orchav.coverage-view.v2" or state.get("canonical", False)):
                raise ValueError(
                    "This legacy view uses a different aggregate contract. Open it with the scripted volume-view command."
                )
            metric = state["metric"]
            if metric not in self.source.get("available_metrics", []):
                raise ValueError("Saved metric is unavailable in the active recording")
            self.visualizer.ui_controller.handle_coverage_metric_changed(metric)
            self._apply_view_controls(state)
            self._request()
            self.restoring = state
            self.visualizer.ui_controller.handle_coverage_toggled(state["visible"])
            if self.model is not None and self.loaded_key == self.request_key:
                self._populate_cut_positions(coordinate=state.get("cut_coordinate"))
                self._orient(coordinate=state["coordinate"])
                self.visualizer.renderer.set_camera_state(CameraState.from_dict(state["camera"]))
            return self
        return open_coverage_volume(self.visualizer, view=state)

    def _view_action(self, *, save):
        dialog = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        path, _ = dialog(
            self,
            "Save coverage view" if save else "Load coverage view",
            "coverage-view.json" if save else "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            if save:
                if not Path(path).suffix:
                    path += ".json"
                self._save_view(path)
                self.status.setText(f"View saved to {path}")
            else:
                self._load_view(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"Coverage view unavailable: {exc}")

    def _select_volume_cell(self, object_id, indices, mode):
        if (
            self.closed
            or self.model is None
            or self.loaded_key != self.request_key
            or object_id != self.OBJECT_ID
        ):
            return
        if mode == "slice" and self._plane_template:
            indices = list(indices)
            indices[self.orientation.currentData()] = self.coordinate.currentIndex()
        if any(
            not 0 <= index < len(axis) for index, axis in zip(indices, self.model.axes, strict=True)
        ):
            return
        self._stop()
        self.selected_indices = tuple(indices)
        self.slice_indices = self.selected_indices
        self.selection_origin = {
            "iso": "Visible ISO surface hit (not a hidden cell)",
            "mip": (
                "Ray-minimum loss hit"
                if self.model.quantity == "path_loss_db"
                else "Ray-maximum hit (MIP; strongest saved value along this ray)"
            ),
            "slice": "Visible slice cell",
        }[mode]
        with QSignalBlocker(self.coordinate):
            self.coordinate.setCurrentIndex(indices[self.orientation.currentData()])
        with QSignalBlocker(self.show_slices):
            self.show_slices.setChecked(True)
        self._draw()
        self._publish_xy_height()

    def _select_slice(self, orientation, horizontal, vertical):
        if self.model is None or self.loaded_key != self.request_key:
            return
        self._stop()
        indices = list(self.selected_indices or self.slice_indices)
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[orientation]
        for axis, value in zip(axes, (horizontal, vertical), strict=True):
            indices[axis] = int(np.argmin(abs(self.model.axes[axis] - value)))
        self.selected_indices = tuple(indices)
        self.slice_indices = self.selected_indices
        self.selection_origin = f"{orientation.upper()} source slice cell"
        with QSignalBlocker(self.coordinate):
            self.coordinate.setCurrentIndex(indices[self.orientation.currentData()])
        self._draw()
        self._publish_xy_height()

    def reject(self):
        """Escape may close scripted dialogs, but never remove embedded controls."""
        if not self.embedded:
            super().reject()

    def closeEvent(self, event):
        self.closed = True
        select_volume = getattr(
            self.visualizer.renderer, "set_volume_cell_selection_callback", None
        )
        if callable(select_volume):
            select_volume(None, object_ids=(self.OBJECT_ID,))
        self.poll.stop()
        self._stop()
        if self.pending is not None:
            self.pending.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.cell_table.shutdown()
        self.height_comparison.shutdown()
        self.scene_labels.clear()
        self.visualizer.renderer.remove_object(self.OBJECT_ID)
        self.visualizer.renderer.remove_object(self.CONTOUR_ID)
        self._contour_key = None
        self._clear_cell_edges()
        self._clear_selection()
        for object_id in transmitter_object_ids(self.source):
            self.visualizer.renderer.remove_object(object_id)
        self.visualizer.renderer.set_analysis_legend(None)
        if self.embedded:
            if self.source.get("_volume_layer") is self.model:
                self.source.pop("_volume_layer", None)
            self.visualizer.coverage_volume_active = False
        elif self.visualizer.coverage_data is self.source:
            self.visualizer.ui_controller.handle_coverage_toggled(self.old_planar_visibility)
        self.model = self.template = self.valid_cells = None
        self._cut_source = self._cut_template = self._cut_key = None
        self._slice_template = None
        if self.embedded:
            # QDialog.closeEvent calls reject(), which intentionally ignores
            # Escape here. Accept explicit teardown so Qt hides/deletes the old
            # controls instead of leaving a stale panel above the next source.
            event.accept()
        else:
            super().closeEvent(event)


def open_coverage_volume(visualizer, *, view=None):
    """Open one capability-checked volume inspector for the active coverage file."""
    reason = volume_rendering_unavailable_reason(visualizer.renderer)
    if reason:
        raise ValueError(f"{reason} Ordinary coverage height planes remain available.")
    if not (visualizer.coverage_data or {}).get("coverage_file"):
        raise ValueError("Load a saved multi-height coverage result first")
    for attr in ("coverage_volume_window", "sensing_volume_window"):
        old = getattr(visualizer, attr, None)
        if old is not None and not old.closed:
            old.close()
    window = CoverageVolumeWindow(visualizer, view=view)
    visualizer.coverage_volume_window = window
    window.show()
    return window
