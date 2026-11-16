"""Target-specific framing of existing, time-aligned ground-truth displays.

This controller changes view limits only. It consumes the same saved truth as
the plotted overlays and never associates detections with targets or loads data.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker
from PySide6.QtWidgets import QComboBox

from .selection import selection_scope


def _bounds(values, minimum_span: float) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    low, high = float(finite.min()), float(finite.max())
    center = (low + high) / 2
    half_span = max((high - low) * 0.6, minimum_span / 2)
    return center - half_span, center + half_span


def _neighborhood(axis, minimum_span: float) -> float:
    """Keep at least eight saved cells visible around a single truth point."""
    spacing = np.abs(np.diff(np.asarray(axis, dtype=float)))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    return max(minimum_span, 8 * float(np.median(spacing))) if spacing.size else minimum_span


class TruthZoomController(QObject):
    """Keep a selected truth identity framed across playback and projection changes."""

    PLOTS = ("rd_plot", "rp_plot", "gt_range_plot", "gt_speed_plot", "tracking_plot")
    MODES = frozenset({"current", "analysis", "tracking"})

    def __init__(self, dashboard):
        super().__init__(dashboard)
        self.dashboard = dashboard
        self.target_id: str | None = None
        self._scope = None
        self._choices = None
        self.combo = QComboBox(dashboard)
        self.combo.setObjectName("sensingTruthZoom")
        self.combo.setMaximumWidth(260)
        self.combo.addItem("Zoom to GT: Off", None)
        self.combo.currentIndexChanged.connect(self._selected)
        dashboard._zoom_data_cb.toggled.disconnect(dashboard._rerender_last_request)
        dashboard._zoom_data_cb.toggled.connect(self._data_zoom_changed)
        dashboard._show_gt_cb.toggled.connect(self._truth_visibility_changed)
        self.menus = {}
        for name in self.PLOTS:
            menu = getattr(dashboard, name).getViewBox().menu.addMenu("Zoom to ground truth")
            menu.aboutToShow.connect(lambda m=menu: self._populate_menu(m))
            self.menus[name] = menu
        self.sync()

    def _populate_menu(self, menu):
        self.sync()
        menu.clear()
        for index in range(self.combo.count()):
            identity = self.combo.itemData(index)
            label = self.combo.itemText(index).removeprefix("Zoom to GT: ")
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(identity == self.target_id)
            action.setEnabled(identity is None or self.combo.isEnabled())
            action.triggered.connect(lambda _checked=False, i=index: self._choose_index(i))

    def close(self):
        """Release callbacks and the Python owner before Qt deletes the window."""
        owner = self.dashboard
        if owner is None:
            return
        self.combo.currentIndexChanged.disconnect(self._selected)
        owner._zoom_data_cb.toggled.disconnect(self._data_zoom_changed)
        owner._show_gt_cb.toggled.disconnect(self._truth_visibility_changed)
        for menu in self.menus.values():
            menu.aboutToShow.disconnect()
            menu.clear()
            menu.deleteLater()
        self.menus.clear()
        self.dashboard = None

    def _choose_index(self, index):
        with QSignalBlocker(self.combo):
            self.combo.setCurrentIndex(index)
        self._selected(index)

    def reset(self):
        self.target_id = None
        self._scope = None
        self.sync()

    def _current_truth(self):
        owner = self.dashboard
        if owner is None:
            return {}
        request = owner._last_request
        if request is None:
            return {}
        entry = owner._extract_bound_gt_entry(request.display_frame.get("sensing") or {})
        if entry is None:
            return {}
        truth = entry[1]
        return {
            str(identity): (distance, speed if valid else None)
            for identity, distance, speed, valid in zip(
                truth["target_ids"],
                truth["target_ranges"],
                truth["target_speeds"],
                truth["target_velocity_valid"],
                strict=True,
            )
        }

    def sync(self):
        """Refresh available identities without replacing a user's target on a seek."""
        owner = self.dashboard
        if owner is None:
            return {}
        request = owner._last_request
        scope = (
            (request.source_identity, selection_scope(request.display_frame.get("sensing") or {}))
            if request is not None
            else None
        )
        if request is not None and scope != self._scope:
            self.target_id = None
            self._scope = scope
        current = self._current_truth()
        ids = set(current)
        if request is not None:
            ids.update(str(key) for key in owner._target_history)
            ids.update(str(key) for key in owner._tracking_truth_curves)
        available = self.target_id is None or self.target_id in ids
        if self.target_id is not None:
            ids.add(self.target_id)
        choices = tuple((key, owner._get_target_label(key)) for key in sorted(ids))
        signature = (choices, self.target_id, available)
        if signature != self._choices:
            self._choices = signature
            with QSignalBlocker(self.combo):
                self.combo.clear()
                self.combo.addItem("Zoom to GT: Off", None)
                for key, label in choices:
                    suffix = " (unavailable)" if key == self.target_id and not available else ""
                    self.combo.addItem(f"Zoom to GT: {label}{suffix}", key)
                self.combo.setCurrentIndex(max(0, self.combo.findData(self.target_id)))
        self.combo.setEnabled(
            request is not None and bool(choices) and owner._show_gt_cb.isChecked()
        )
        self.combo.setVisible(owner._selected_view_mode() in self.MODES)
        self.combo.setToolTip(
            "No saved ground truth is available for these figures."
            if not choices
            else (
                "Enable Ground truth to frame a target."
                if not owner._show_gt_cb.isChecked()
                else (
                    "This target has no ground truth at the displayed result."
                    if not available
                    else "Follow one recorded target in range–velocity, range profile, range/velocity "
                    "history and spatial tracking. History views fit its displayed path. Off restores "
                    "normal framing. This changes only the view; detections and tracks stay unchanged."
                )
            )
        )

    def _selected(self, _index):
        self.target_id = self.combo.currentData()
        if self.target_id is not None:
            with QSignalBlocker(self.dashboard._zoom_data_cb):
                self.dashboard._zoom_data_cb.setChecked(False)
            # A deliberate framing choice takes ownership back from manually
            # inspected live figures, without changing their selected reports.
            for key, pinned in self.dashboard._live_figures.items():
                if key in self.PLOTS:
                    pinned.inspected = False
            self.apply()
        else:
            self.dashboard._rerender_last_request()

    def _data_zoom_changed(self, checked):
        if checked:
            self.target_id = None
            self.sync()
        self.dashboard._rerender_last_request()

    def _truth_visibility_changed(self, checked):
        if not checked and self.target_id is not None:
            self.target_id = None
            self.dashboard._rerender_last_request()
        self.sync()

    def apply(self):
        """Frame current saved truth and displayed history, preserving all samples."""
        self.sync()
        owner = self.dashboard
        identity = self.target_id
        if identity is None or not owner._show_gt_cb.isChecked():
            return
        request = owner._last_request
        if request is None:
            return
        current = self._current_truth().get(identity)
        rd = request.rd_result
        rp = request.rp_result
        if current is not None and (rd is not None or rp is not None):
            ranges = (rd if rd is not None else rp).range_axis_m
            x = _bounds([current[0]], _neighborhood(ranges, 2.0))
            # RD and the range profile already share their horizontal axis.
            if x is not None and any(self._visible(key) for key in ("rd_plot", "rp_plot")):
                owner.rd_plot.setXRange(*x, padding=0)
                owner._fit_range_profile_axis()
            if rd is not None and self._visible("rd_plot") and current[1] is not None:
                if owner._use_doppler_hz:
                    y = owner._velocity_to_doppler_hz(
                        np.asarray([current[1]]), request.carrier_freq_hz
                    )
                    axis = rd.doppler_axis_hz
                    minimum = abs(
                        float(
                            owner._velocity_to_doppler_hz(
                                np.asarray([1.0]), request.carrier_freq_hz
                            )[0]
                        )
                    )
                else:
                    y = [current[1]]
                    axis = owner._doppler_to_velocity(rd.doppler_axis_hz, request.carrier_freq_hz)
                    minimum = 1.0
                bounds = _bounds(y, _neighborhood(axis, minimum))
                if bounds is not None:
                    owner.rd_plot.setYRange(*bounds, padding=0)
        history = next(
            (value for key, value in owner._target_history.items() if str(key) == identity), None
        )
        if history is not None:
            for name, field, span in (
                ("gt_range_plot", "range", 2.0),
                ("gt_speed_plot", "speed", 1.0),
            ):
                times = np.asarray(history["times_s"], dtype=float)
                values = np.asarray(history[field], dtype=float)
                finite = np.isfinite(times) & np.isfinite(values)
                x, y = _bounds(times[finite], 1.0), _bounds(values[finite], span)
                if x is not None and y is not None and self._visible(name):
                    getattr(owner, name).setRange(xRange=x, yRange=y, padding=0)
        curve = next(
            (value for key, value in owner._tracking_truth_curves.items() if str(key) == identity),
            None,
        )
        if curve is not None and self._visible("tracking_plot"):
            xs, ys = curve.getData()
            if xs is not None and ys is not None:
                finite = np.isfinite(xs) & np.isfinite(ys)
                x, y = _bounds(xs[finite], 1.0), _bounds(ys[finite], 1.0)
                if x is not None and y is not None:
                    owner.tracking_plot.setRange(xRange=x, yRange=y, padding=0)

    def _visible(self, name):
        plot = getattr(self.dashboard, name)
        return plot.isVisible() and not self.dashboard._live_plot_is_inspected(plot)
