"""Draw saved tracking estimates without conflating them with measurements."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt

from .display_style import identity_color


def _recorded_symbol_brushes(curve, color, coast):
    """Reuse two Qt brushes per curve across the immutable history prefixes."""
    brushes = getattr(curve, "_sensing_symbol_brushes", None)
    if brushes is None or brushes[0] != color:
        brushes = (color, pg.mkBrush(color), pg.mkBrush(None))
        curve._sensing_symbol_brushes = brushes
    return [brushes[2] if predicted else brushes[1] for predicted in coast]


class TrackingViewsMixin:
    def _tracking_figures_active(self):
        """A host with detached figures can extend this ordinary workspace demand."""
        return self._selected_view_mode() == "tracking"

    def _live_plot_needs_update(self, _plot):
        """Ordinary tracking hosts update all figures; dock hosts refine visibility."""
        return True

    def apply_linked_selection(self):
        super().apply_linked_selection()
        self._highlight_track_selection()

    def _highlight_track_selection(self):
        """Apply one selection to spatial/radial states and their coasting segments."""
        request = getattr(self, "_last_request", None)
        if request is not None and request.saved_tracking is not None:
            selected = self.tracking_inspector.selected_id
        else:
            selected = getattr(getattr(self, "_linked_selection", None), "track_id", None)
        selected = None if selected is None else str(selected)
        for curves, radial in (
            (getattr(self, "_tracking_curves", {}), False),
            (getattr(self, "_tracking_coast_curves", {}), False),
            (getattr(self, "_track_estimate_curves", {}), True),
            (getattr(self, "_track_estimate_coasts", {}), True),
        ):
            for key, curve in curves.items():
                tid = key[0] if radial else key
                base = getattr(curve, "_sensing_base_opacity", 1.0)
                curve.setOpacity(base if selected is None or str(tid) == selected else 0.18 * base)

    def _set_tracking_projection_capability(self, prepared):
        """Do not expose an elevation projection for synthetic zero-Z 2D points."""
        dimensions = getattr(prepared, "spatial_dimensions", 0)
        for index in range(1, self.tracking_projection.count()):
            self.tracking_projection.model().item(index).setEnabled(dimensions == 3)
        if dimensions != 3 and self.tracking_projection.currentIndex() != 0:
            with QSignalBlocker(self.tracking_projection):
                self.tracking_projection.setCurrentIndex(0)

    def _draw_track_paths(self, history, now):
        prepared = history.tracking_plot_data
        self._set_tracking_projection_capability(prepared)
        plane = self.tracking_projection.currentText().lower()
        self.tracking_plot.setLabel("bottom", plane[0].upper(), units="m")
        self.tracking_plot.setLabel("left", plane[1].upper(), units="m")
        for series, curves, truth in (
            (prepared.tracks, self._tracking_curves, False),
            (prepared.truths, self._tracking_truth_curves, True),
        ):
            present = set()
            for tid, values in sorted(series.items()):
                times, points = values.projected(plane).through(now)
                if not len(times):
                    continue
                present.add(tid)
                color = identity_color(tid, truth=truth)
                if tid not in curves:
                    curves[tid] = self.tracking_plot.plot(
                        pen=pg.mkPen(color, width=2, style=Qt.DashLine if truth else Qt.SolidLine),
                        symbol="+" if truth else "d",
                        symbolSize=8,
                        symbolPen=pg.mkPen(color, width=2),
                        symbolBrush=pg.mkBrush(color),
                        name=f"GT {self._get_target_label(tid)}" if truth else f"Track {tid}",
                    )
                    if not truth:
                        curves[tid].setCurveClickable(True, width=10)
                        curves[tid].sigClicked.connect(lambda _c, _e, t=tid: self.select_track(t))
                        curves[tid].sigPointsClicked.connect(
                            lambda _c, _p, _e, t=tid: self.select_track(t)
                        )
                coast = np.zeros(len(times), dtype=bool)
                statuses = prepared.statuses.get(tid)
                if not truth and statuses is not None:
                    status_times, states = statuses.through(now)
                    by_time = dict(zip(status_times, states[:, 0], strict=True))
                    coast = np.array([by_time.get(t) == 1 for t in times])
                sizes = np.full(len(times), 8.0)
                if abs(times[-1] - now) < 1e-9:
                    sizes[-1] = 18
                connect = np.ones(len(times), dtype=bool)
                connect[:-1] = ~(coast[:-1] | coast[1:])
                connect[-1] = False
                curves[tid].setData(
                    points[:, 0],
                    points[:, 1],
                    symbolSize=sizes,
                    symbolBrush=_recorded_symbol_brushes(curves[tid], color, coast),
                    connect=connect,
                )
                opacity = 1.0 if truth or abs(times[-1] - now) < 1e-9 else 0.35
                curves[tid]._sensing_base_opacity = opacity
                curves[tid].setOpacity(opacity)
                curves[tid].setVisible(
                    self._show_gt_cb.isChecked() if truth else self._show_det_cb.isChecked()
                )
                if not truth:
                    if tid not in self._tracking_coast_curves:
                        self._tracking_coast_curves[tid] = self.tracking_plot.plot(
                            pen=pg.mkPen(color, width=2, style=Qt.DashLine)
                        )
                    coast_connect = ~connect
                    coast_connect[-1] = False
                    self._tracking_coast_curves[tid].setData(
                        points[:, 0], points[:, 1], connect=coast_connect
                    )
                    self._tracking_coast_curves[tid]._sensing_base_opacity = opacity
                    self._tracking_coast_curves[tid].setVisible(self._show_det_cb.isChecked())
            for expired in set(curves) - present:
                self.tracking_plot.removeItem(curves.pop(expired))
                if not truth and expired in self._tracking_coast_curves:
                    self.tracking_plot.removeItem(self._tracking_coast_curves.pop(expired))
        request = getattr(self, "_last_request", None)
        self.tracking_plot.setTitle(
            (
                "Saved confirmed 3D tracks · includes coasting"
                if request is not None and request.tracking_history is not None
                else "Saved tracks and ground truth"
            ),
        )

    def _draw_track_estimates(self, prepared, now):
        active = self._tracking_figures_active()
        present = set()
        if active:
            for tid, series in prepared.estimates.items():
                times, points = series.through(now)
                for column, plot in enumerate((self.gt_range_plot, self.gt_speed_plot)):
                    if not self._live_plot_needs_update(plot):
                        continue
                    if not len(times) or not np.isfinite(points[:, column]).any():
                        continue
                    key = (tid, column)
                    present.add(key)
                    if key not in self._track_estimate_curves:
                        color = identity_color(tid)
                        curve = plot.plot(
                            pen=pg.mkPen(color, width=2),
                            symbol="o",
                            symbolSize=6,
                            symbolPen=pg.mkPen(color),
                            name=f"Track {tid} estimate",
                        )
                        curve.setCurveClickable(True, width=10)
                        curve.sigClicked.connect(lambda _c, _e, t=tid: self.select_track(t))
                        self._track_estimate_curves[key] = curve
                        self._track_estimate_coasts[key] = plot.plot(
                            pen=pg.mkPen(color, width=2, style=Qt.DashLine)
                        )
                    color = identity_color(tid)
                    status = prepared.statuses.get(tid)
                    by_time = (
                        dict(zip(status.times, status.values[:, 0], strict=True))
                        if status is not None
                        else {}
                    )
                    coast = np.array([by_time.get(t) == 1 for t in times])
                    connect = np.ones(len(times), dtype=bool)
                    connect[:-1] = ~(coast[:-1] | coast[1:])
                    connect[-1] = False
                    self._track_estimate_curves[key].setData(
                        times,
                        points[:, column],
                        connect=connect,
                        symbolBrush=_recorded_symbol_brushes(
                            self._track_estimate_curves[key], color, coast
                        ),
                    )
                    coast_connect = ~connect
                    coast_connect[-1] = False
                    self._track_estimate_coasts[key].setData(
                        times, points[:, column], connect=coast_connect
                    )
                    opacity = 1.0 if abs(times[-1] - now) < 1e-9 else 0.35
                    self._track_estimate_curves[key]._sensing_base_opacity = opacity
                    self._track_estimate_coasts[key]._sensing_base_opacity = opacity
        self._track_estimate_present = present
        for key, curve in self._track_estimate_curves.items():
            curve.setVisible(key in present and self._show_det_cb.isChecked())
            self._track_estimate_coasts[key].setVisible(curve.isVisible())
        self._highlight_track_selection()

    def _tracking_plot_labels(self):
        tracking = self._tracking_figures_active()
        tracking_workspace = self._selected_view_mode() == "tracking"
        self.tracking_measurements.setVisible(
            tracking_workspace
            and (self._last_request is None or self._last_request.saved_tracking is None)
        )
        for key, curve in self._track_estimate_curves.items():
            curve.setVisible(
                tracking
                and key in getattr(self, "_track_estimate_present", ())
                and self._show_det_cb.isChecked()
            )
            self._track_estimate_coasts[key].setVisible(curve.isVisible())
        if not tracking:
            return
        saved_world = (
            self._last_request is not None and self._last_request.saved_tracking is not None
        )
        show_measurements = self._show_det_cb.isChecked() and (
            saved_world or self.tracking_measurements.isChecked()
        )
        self._gt_range_det_scatter.setVisible(show_measurements)
        self._gt_speed_det_scatter.setVisible(show_measurements)
        self._show_det_cb.setText("Tracks" if tracking_workspace else "Detections")
        for plot, scatter in (
            (self.gt_range_plot, self._gt_range_det_scatter),
            (self.gt_speed_plot, self._gt_speed_det_scatter),
        ):
            label = plot.getPlotItem().legend.getLabel(scatter)
            text = "Projected states" if saved_world else "Measurements"
            if label is not None and label.text != text:
                label.setText(text)
        for column, (plot, quantity) in enumerate(
            ((self.gt_range_plot, "Range"), (self.gt_speed_plot, "Radial velocity"))
        ):
            has_estimates = any(
                key[1] == column for key in getattr(self, "_track_estimate_present", ())
            )
            plot.setTitle(
                f"Projected track {quantity.lower()}"
                if saved_world
                else f"{quantity} estimates" if has_estimates else f"Measured {quantity.lower()}"
            )
        for curve in self._tracking_coast_curves.values():
            curve.setVisible(self._show_det_cb.isChecked())
