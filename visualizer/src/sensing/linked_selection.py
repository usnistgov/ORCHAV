"""Lightweight linked-selection presentation for the sensing dashboard.

Selections change marker emphasis and saved-bin overlays. Opening the explanation
also requests saved detector evidence through the dashboard session.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets.dashboard_host import dashboard_is_visible
from ..widgets.wrapping_layout import WrappingLayout
from .display_style import HEATMAP_MARKER_OUTLINE, detection_color
from .selection import (
    catalog_identity,
    inspection_cell,
    matches_detection,
    select_estimate,
    selection_scope,
)


class LinkedSelectionMixin:
    """Connect saved detection/track identities across the dashboard's plots."""

    def setup_linked_selection(self, layout) -> None:
        self._linked_selection = None
        self._cleared_cell_context = None
        self.selection_tools = QWidget()
        layout.addWidget(self.selection_tools)
        layout = QVBoxLayout(self.selection_tools)
        layout.setContentsMargins(0, 0, 0, 0)
        self.selection_tools_toggle = QPushButton("Selection && 3D explanation")
        self.selection_tools_toggle.setCheckable(True)
        self.selection_tools_toggle.setToolTip(
            "Inspect the selected bin, detection or track and explain its geometry in the 3D scene."
        )
        self.selection_tools_toggle.toggled.connect(self._sync_selection_details)
        self.selection_tools_toggle.toggled.connect(self._request_cell_explanation)
        self.selection_tools_toggle.hide()
        self.selection_tools.hide()
        self.selection_label = QLabel("No selection")
        self.selection_label.setWordWrap(True)
        row = WrappingLayout()
        row.addWidget(self.selection_label)
        self.locate_selection_button = QPushButton("Center selection in 3D")
        self.locate_selection_button.setEnabled(False)
        self.locate_selection_button.setToolTip(
            "Center the scene camera on the saved spatial position."
        )
        self.locate_selection_button.clicked.connect(self.locate_selection_requested.emit)
        clear = QPushButton("Clear selection")
        clear.clicked.connect(lambda: self.set_linked_selection(None, notify=True))
        row.addWidget(clear)
        layout.addLayout(row)
        from .cell_explanation import CellExplanation

        self.cell_explanation = CellExplanation(self)
        self.cell_explanation.paths_requested.connect(self.path_inspection_requested.emit)
        self.cell_explanation.detector_requested.connect(self._open_cell_detector)
        layout.addWidget(self.cell_explanation)
        self.scene_selection_controls = QWidget()
        shell_row = WrappingLayout(self.scene_selection_controls)
        layout.addWidget(self.scene_selection_controls)
        self.range_shell_enabled = QCheckBox("Show range shell")
        self.range_shell_enabled.setChecked(False)
        self.range_shell_enabled.setToolTip("Show the selected range-bin geometry in the 3D scene.")
        self.range_shell_cutaway = QCheckBox("Horizontal cutaway")
        self.range_shell_cutaway.setToolTip(
            "Clip only the explanatory shell at the sensor midpoint"
        )
        for control in (self.range_shell_enabled, self.range_shell_cutaway):
            control.toggled.connect(lambda _checked: self.range_shell_changed.emit())
            shell_row.addWidget(control)
        frame_shell = QPushButton("Frame range shell")
        frame_shell.setToolTip(
            "Fit the complete numerical range shell in the scene; this is not a target location"
        )
        frame_shell.clicked.connect(lambda _checked=False: self.range_shell_frame_requested.emit())
        shell_row.addWidget(frame_shell)
        shell_row.addWidget(self.locate_selection_button)
        save_view = QPushButton("Save view PNG…")
        save_view.setToolTip(
            "Export the current dashboard, including the selected bin and range-shell controls"
        )
        save_view.clicked.connect(self._save_selection_view)
        row.addWidget(save_view)
        self.range_shell_status = QLabel("Range shell off")
        self.range_shell_status.setWordWrap(True)
        layout.addWidget(self.range_shell_status)
        from .scene_explanation import SceneExplanationControls

        self.scene_explanation = SceneExplanationControls(self.range_shell_enabled, self)
        self.scene_explanation.changed.connect(self.range_shell_changed.emit)
        layout.addWidget(self.scene_explanation)
        self.rd_plot.scene().sigMouseClicked.connect(self._select_rd_cell)
        self._selected_rd_cells = pg.ScatterPlotItem(
            symbol="s",
            size=18,
            pen=pg.mkPen("#ffe45c", width=2),
            brush=pg.mkBrush(None),
            name="Selected saved bin",
        )
        self.rd_plot.addItem(self._selected_rd_cells)
        self._selected_rd_cells.setZValue(22)
        # The marker consumes clicks before the scene's heatmap handler runs.
        self._selected_rd_cells.sigClicked.connect(self._clear_rd_cell)
        for scatter in (
            self.rd_det_scatter,
            self.range_azimuth_det_scatter,
            self.doppler_azimuth_det_scatter,
            self.angular_spectrum_det_scatter,
        ):
            scatter.sigClicked.connect(self._select_detection_point)

    def _sync_selection_details(self, *_args) -> None:
        """Keep scene-explanation controls local to an actual selected result."""
        available = (
            self._linked_selection is not None
            and self._last_request is not None
            and self._selected_view_mode() not in {"summary", "evaluation"}
        )
        self.selection_tools_toggle.setVisible(available)
        self.selection_tools.setVisible(available and self.selection_tools_toggle.isChecked())
        if hasattr(self, "result_scroll"):
            self._sync_browser_visibility()

    def _save_selection_view(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sensing view", "sensing-view.png", "PNG image (*.png)"
        )
        if path and not self.grab().save(path, "PNG"):
            self.range_shell_status.setText("Could not save the sensing view image")

    def _request_cell_explanation(self, checked) -> None:
        if self._last_request is not None:
            self.apply_linked_selection()
        if self._last_request is not None and self._selected_view_mode() in {"current", "analysis"}:
            self.view_changed.emit()

    def _open_cell_detector(self) -> None:
        self._set_view_mode("diagnostics")
        self._support_box.setCurrentIndex(self._support_box.findData("cfar"))
        self._on_view_mode_changed()

    def _select_rd_cell(self, event) -> None:
        """Select only an in-bounds numerical RD cell, without loading data."""
        request = self._last_request
        if getattr(event, "isAccepted", lambda: False)():
            return
        if (
            event.button() != Qt.MouseButton.LeftButton
            or request is None
            or request.rd_result is None
        ):
            return
        view = self.rd_plot.getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view.mapSceneToView(event.scenePos())
        rd = request.rd_result
        ranges = np.asarray(rd.range_axis_m)
        doppler = np.asarray(rd.doppler_axis_hz)
        y = (
            doppler
            if self._use_doppler_hz
            else self._doppler_to_velocity(doppler, request.carrier_freq_hz)
        )

        def cell_index(axis, coordinate):
            if len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                return None
            edges = np.r_[
                axis[0] - (axis[1] - axis[0]) / 2,
                (axis[:-1] + axis[1:]) / 2,
                axis[-1] + (axis[-1] - axis[-2]) / 2,
            ]
            if coordinate < edges[0] or coordinate >= edges[-1]:
                return None
            return int(np.searchsorted(edges, coordinate, side="right") - 1)

        r, d = cell_index(ranges, point.x()), cell_index(y, point.y())
        if r is None or d is None:
            self.set_linked_selection(None, notify=True)
        else:
            self._select_inspection_coordinate(r, d, toggle=True)
        event.accept()

    def _clear_rd_cell(self, _scatter, _points, event) -> None:
        self.set_linked_selection(None, notify=True)
        event.accept()

    def _select_detection_point(self, _scatter, points, event) -> None:
        if points:
            self.select_detection_index(int(points[0].data()), toggle=True)
            event.accept()

    def select_detection_index(self, index: int, *, toggle: bool = False) -> None:
        request = self._last_request
        if request is not None:
            self.set_linked_selection(
                select_estimate(request.display_frame["sensing"], detection_index=index),
                notify=True,
                toggle=toggle,
            )

    def select_track(self, track_id: str) -> None:
        if self._last_request is not None:
            self.set_linked_selection(
                select_estimate(
                    self._last_request.display_frame["sensing"], track_id=int(track_id)
                ),
                notify=True,
            )

    def _cell_context(self):
        request = self._last_request
        if request is None:
            return None
        sensing = request.display_frame["sensing"]
        return (
            (
                request.source_identity,
                selection_scope(sensing),
                sensing["temporal"]["result"]["processing_recipe_id"],
                request.variant,
            ),
            sensing["temporal"]["acquisition"]["acquisition_id"],
            catalog_identity(sensing),
        )

    def _cell_selection_cleared(self) -> bool:
        """Keep detector deselection while stepping results in the same recording."""
        if self._linked_selection is None:
            return True
        previous = self._cleared_cell_context
        current = self._cell_context()
        if previous is None or current is None:
            return False
        if (
            self._selected_view_mode() == "diagnostics"
            and self._support_box.currentData() == "cfar"
        ):
            # No bin is selected, so changing acquisitions cannot make the
            # choice stale. Do reset it when the scientific/source scope changes.
            return previous[0] == current[0]
        return previous == current

    def set_linked_selection(
        self, selection, *, notify: bool = False, toggle: bool = False
    ) -> None:
        """Highlight an identity without recomputing spectra or reloading a scene."""
        request = self._last_request
        if selection is not None and request is not None:
            selection = replace(selection, source=selection.source or request.source_identity)
            selection = self._bind_loaded_detection_cell(selection)
        current = self._linked_selection
        if toggle and selection is not None and current is not None:
            same_cell = (
                selection.source == current.source
                and selection.scope == current.scope
                and selection.acquisition_id == current.acquisition_id
                and selection.cell is not None
                and current.cell is not None
                and selection.cell == replace(current.cell, origin=selection.cell.origin)
            )
            if same_cell or selection == current:
                selection = None
        self._cleared_cell_context = self._cell_context() if selection is None else None
        self._linked_selection = selection
        self._recorded_scatter_state = None
        if hasattr(self, "tracking_inspector") and (
            selection is None or selection.track_id is None or selection.detection_index is not None
        ):
            self.tracking_inspector.select_track(None)
        if selection is None and self._last_request is not None:
            history = self._last_request.recording_history
            if history is not None and self._selected_view_mode() in {
                "current",
                "analysis",
                "tracking",
            }:
                self._render_recording(self._last_request)
        self.apply_linked_selection()
        self._sync_selected_stage_cell(self._linked_selection)
        self.range_shell_changed.emit()
        if notify:
            self.selection_changed.emit(self._linked_selection)

    def _bind_loaded_detection_cell(self, selection):
        """Attach axes arriving after a report click; never request another product."""
        request = self._last_request
        if selection is None or selection.cell is not None or request is None:
            return selection
        sensing = request.display_frame["sensing"]
        if (
            selection.source != request.source_identity
            or selection.acquisition_id != sensing["temporal"]["acquisition"]["acquisition_id"]
            or request.variant not in {"auto", "primary", "clean"}
        ):
            return selection
        for index, detection in enumerate(sensing.get("detections", [])):
            if matches_detection(selection, sensing, index, detection):
                r, d = detection.get("range_bin"), detection.get("doppler_bin")
                if type(r) is int and type(d) is int:
                    cell = inspection_cell(request, r, d, origin="detection")
                    if cell is not None:
                        return replace(selection, cell=cell)
                break
        return selection

    def apply_linked_selection(self) -> None:
        request = self._last_request
        self._selected_rd_cells.setData([], [])
        if request is None:
            self._sync_selection_details()
            self.cell_explanation.update_cell(None, None)
            return
        sensing = request.display_frame["sensing"]
        selection = self._linked_selection
        if selection is not None and (
            selection.scope != selection_scope(sensing)
            or selection.source != request.source_identity
            or selection.detection_collection
            != sensing.get("_display_detection_collection", "detections")
            or (
                selection.detection_index is not None
                and selection.acquisition_id != sensing["temporal"]["acquisition"]["acquisition_id"]
            )
            # A coordinate probe belongs to one acquisition. Keep saved
            # detection/track history selections, but never carry a raw bin
            # highlight onto a different observation at the same coordinates.
            or (
                selection.cell is not None
                and selection.detection_index is None
                and selection.track_id is None
                and selection.acquisition_id != sensing["temporal"]["acquisition"]["acquisition_id"]
            )
        ):
            self._linked_selection = selection = None
            self.selection_changed.emit(None)
        bound = self._bind_loaded_detection_cell(selection)
        if bound is not selection:
            self._linked_selection = selection = bound
            # This enriches an existing selection, not a new user click that
            # should pause playback or seek the scene again.
            self.range_shell_changed.emit()
        self._sync_selection_details()
        self.cell_explanation.update_cell(
            request if self.selection_tools_toggle.isChecked() else None,
            selection,
            velocity=not self._use_doppler_hz,
        )
        self._sync_selection_location_button()
        detections = sensing.get("detections") or []
        chosen = [
            (i, d) for i, d in enumerate(detections) if matches_detection(selection, sensing, i, d)
        ]
        indices = {i for i, _ in chosen}
        # Probing a saved bin does not select an estimate: keep every detection
        # readable while the coordinate highlight is shown.
        estimate_selected = selection is not None and (
            selection.detection_index is not None or selection.track_id is not None
        )
        if (
            selection is not None
            and selection.track_id is not None
            and selection.detection_index is None
        ):
            self.result_browser.select_track(selection.track_id)
        elif chosen:
            self.result_browser.select_detection(chosen[0][0])
        elif not (
            selection is None
            and request.saved_tracking is not None
            and self.tracking_inspector.selected_id is not None
        ):
            self.result_browser.clear_selection()
        for plot, scatter in (
            (self.rd_plot, self.rd_det_scatter),
            (self.range_azimuth_plot, self.range_azimuth_det_scatter),
            (self.doppler_azimuth_plot, self.doppler_azimuth_det_scatter),
            (self.angular_spectrum_plot, self.angular_spectrum_det_scatter),
        ):
            # Invisible plots may retain another event. Update only plots
            # rendered for this request; their point data is a saved row index.
            if not scatter.isVisible() or (
                request.recording_managed and not dashboard_is_visible(plot)
            ):
                continue
            pens, brushes, sizes = [], [], []
            for point in scatter.points():
                index = point.data()
                if not isinstance(index, (int, np.integer)) or index >= len(detections):
                    pens.append(point.pen())
                    brushes.append(point.brush())
                    sizes.append(point.size())
                    continue
                color = pg.mkColor(detection_color(detections[index]))
                if estimate_selected and index not in indices:
                    color.setAlphaF(0.22)
                if scatter in (self.rd_det_scatter, self.angular_spectrum_det_scatter):
                    outline = pg.mkColor(HEATMAP_MARKER_OUTLINE)
                    outline.setAlphaF(color.alphaF())
                    pens.append(pg.mkPen(outline, width=1.2))
                    sizes.append(16 if index in indices else 12)
                else:
                    pens.append(pg.mkPen(color, width=3 if index in indices else 2))
                    sizes.append(16 if index in indices else 10)
                brushes.append(pg.mkBrush(color))
            if sizes:
                # Native array setters update the symbol atlas once per property,
                # rather than once per property per point (including hidden maps).
                scatter.setPen(pens)
                scatter.setBrush(brushes)
                scatter.setSize(sizes)
        for tid, curve in self._tracking_curves.items():
            if selection is not None:
                curve.setOpacity(1.0 if int(tid) == selection.track_id else 0.18)
        if selection is None:
            self.selection_label.setText("No selection")
            return
        if (
            selection.cell is not None
            and selection.detection_index is None
            and selection.track_id is None
        ):
            self._render_detection_history()
            cell = selection.cell
            self.selection_label.setText(
                f"Bin {cell.range_bin} · {cell.ranges[cell.range_bin]:.6g} m"
            )
            if cell.doppler_bin is not None and 0 <= cell.doppler_bin < len(cell.doppler):
                y = cell.doppler[cell.doppler_bin]
                if not self._use_doppler_hz:
                    y = self._doppler_to_velocity(np.array([y]), request.carrier_freq_hz)[0]
                self._selected_rd_cells.setData([cell.ranges[cell.range_bin]], [y])
            return
        name = (
            f"Detection {selection.detection_index + 1}"
            if selection.detection_index is not None
            else f"Track {selection.track_id}"
        )
        if selection.detection_index is not None and selection.track_id is not None:
            name += f" · Track {selection.track_id}"
        if not chosen:
            self.selection_label.setText(f"{name} · no detection in this result")
        else:
            d = chosen[0][1]
            details = [f"{name} · {len(chosen)} detection(s)"]
            for field, label, unit in (
                ("range_m", "range", "m"),
                ("velocity_m_s", "velocity", "m/s"),
                ("aoa_az_deg", "azimuth", "°"),
                ("aoa_el_deg", "elevation", "°"),
            ):
                value = d.get(field)
                if value is not None and np.isfinite(value):
                    details.append(f"{label} {value:.2f} {unit}")
            self.selection_label.setText(" · ".join(details))
        rd = request.rd_result
        cells = []
        if rd is not None and request.variant in {"auto", "primary", "clean"}:
            for _, d in chosen:
                r, f = d.get("range_bin"), d.get("doppler_bin")
                if (
                    isinstance(r, int)
                    and isinstance(f, int)
                    and 0 <= r < len(rd.range_axis_m)
                    and 0 <= f < len(rd.doppler_axis_hz)
                ):
                    y = rd.doppler_axis_hz[f]
                    if not self._use_doppler_hz:
                        y = self._doppler_to_velocity(np.asarray([y]), request.carrier_freq_hz)[0]
                    cells.append((rd.range_axis_m[r], y))
            if cells:
                self._selected_rd_cells.setData(*np.array(cells).T)
        history = request.recording_history
        if history is not None:
            positions, ranges, speeds, colors, brushes = [], [], [], [], []
            for record in history.records:
                if record.time_s > self.result_time_s() + 1e-12:
                    continue
                for index, d in enumerate(record.detections):
                    selected = selection.scope == record.identity[:3] and (
                        d.get("track_id") == selection.track_id
                        if selection.track_id is not None
                        else record.identity[3] == selection.acquisition_id
                        and index == selection.detection_index
                    )
                    pen, brush = self._history_marker_style(detection_color(d), selected=selected)
                    positions.append(record.time_s)
                    ranges.append(d.get("range_m"))
                    speeds.append(d.get("velocity_m_s"))
                    colors.append(pen)
                    brushes.append(brush)
            self._gt_range_det_scatter.setData(positions, ranges, pen=colors, brush=brushes)
            self._gt_speed_det_scatter.setData(positions, speeds, pen=colors, brush=brushes)

    def selected_world_position(self):
        """Return the current saved anchor even when scene overlays are hidden."""
        from .selection_geometry import selected_world_position

        selection = self._linked_selection
        inspector = self.tracking_inspector
        track = inspector.selected_track() if selection is None else None
        if selection is not None and selection.detection_index is None:
            track = next(
                (row for row in inspector.snapshot.states if row["track_id"] == selection.track_id),
                None,
            )
        return selected_world_position(self._last_request, selection, track=track)

    def _sync_selection_location_button(self):
        available = self.selected_world_position() is not None
        self.locate_selection_button.setEnabled(available)
        self.locate_selection_button.setToolTip(
            "Center on the saved spatial position."
            if available
            else "No saved 3D position. Range-only results can be inspected with a range shell."
        )
