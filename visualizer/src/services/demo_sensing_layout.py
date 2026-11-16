"""Two existing sensing figures beside the scene, with reversible display choices."""

from PySide6.QtCore import Qt, QTimer
from shiboken6 import isValid

from ..sensing.runtime import state_value, update_state
from ..widgets.live_figure_dock import MAX_LIVE_FIGURES, LiveFigureManager
from .demo_layout import ExhibitionLayout


class SensingExhibitionLayout(ExhibitionLayout):
    """Borrow the saved range–velocity map and range profile for a presentation."""

    FIGURES = ("Sensing:rd_plot", "Sensing:rp_plot")
    interactive_figures = True
    OVERLAYS = {
        "show_detections": True,
        "scene_mode": "full",
        "show_track_labels": True,
        "show_truth": False,
        "show_tracking_uncertainty": False,
    }

    def __init__(self, visualizer, failed):
        super().__init__(visualizer)
        self.failed = failed
        self.manager = None
        self.chosen = []
        self.pending_figures = []
        self.borrowed_choices = []
        self.checkboxes = []
        self.mouse_propagation = []
        self.overlays = None
        self._timer = QTimer(visualizer)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._select_figures)

    @staticmethod
    def supports_dashboard(dashboard):
        """Reject known missing products without opening a reader in a UI callback."""
        request = getattr(dashboard, "_last_request", None)
        if request is None:
            return True  # The normal asynchronous catalog has not arrived yet.
        kinds = request.available_product_kinds
        if kinds is None:
            kinds = {
                item.get("kind")
                for item in (
                    request.display_frame.get("sensing", {}).get("products") or {}
                ).values()
            }
        return "range_doppler" in kinds

    def _check(self, checkbox, value):
        if checkbox is not None:
            self.checkboxes.append((checkbox, checkbox.isChecked()))
            checkbox.setChecked(value)

    def enter(self):
        controls, docks = self._capture_layout()
        self.viz.ui_manager.ensure_panel("sensing")
        panel = self.viz.ui_manager.panels.get("sensing")
        if panel is None:
            raise ValueError("This recording has no sensing panel")
        self._check(panel.widgets.get("load_sensing_figures"), True)
        self._check(panel.widgets.get("sensing_playback"), True)
        dashboard = panel.ensure_live_figure_dashboard()
        if dashboard is None or not self.supports_dashboard(dashboard):
            raise ValueError("This recording has no saved range–velocity map")
        self._check(dashboard._show_det_cb, True)
        self._check(dashboard._show_gt_cb, True)
        self._check(dashboard._profile_detections_cb, True)
        self._check(dashboard._profile_full_dynamic_range_cb, True)
        self.overlays = {key: state_value(self.viz.app_state, key) for key in self.OVERLAYS}
        update_state(self.viz, **self.OVERLAYS)
        self.manager = getattr(self.viz, "live_figure_manager", None)
        if self.manager is None:
            self.manager = LiveFigureManager(self.viz)
            self.viz.live_figure_manager = self.manager
            self.owned_manager = self.manager
        self.manager.prepare_sources()
        for dock in docks:
            dock.hide()
        controls.hide()
        # Reuse already selected figures before allocating another dock. At
        # the four-dock limit, temporarily borrow an unrelated dock and retain
        # its choice so stopping the demo can put it back exactly.
        reserved = {
            view.choice_id: view for view in self.manager.views if view.choice_id in self.FIGURES
        }
        for key in self.FIGURES:
            view = reserved.get(key)
            if view is not None:
                self.borrowed_choices.append((view, view.choice_id))
            if view is None and len(self.manager.views) < MAX_LIVE_FIGURES:
                view = self.manager.add()
                if view is not None:
                    self.created.append(view)
            if view is None:
                view = next(
                    (
                        v
                        for v in self.manager.views
                        if v not in self.chosen and v not in reserved.values()
                    ),
                    None,
                )
                if view is not None:
                    self.borrowed_choices.append((view, view.choice_id))
                    view.release_figure()
            if view is None:
                raise ValueError("Could not open the sensing presentation figures")
            self.chosen.append(view)
            self.mouse_propagation.append((view, view.testAttribute(Qt.WA_NoMousePropagation)))
            # A plot can ignore a press while deciding between a click and a
            # drag. Keep that event inside the dock so it cannot bubble to
            # the main window's ordinary-control takeover handler.
            view.setAttribute(Qt.WA_NoMousePropagation, True)
            view.setFloating(False)
            self.viz.addDockWidget(Qt.RightDockWidgetArea, view)
            view.show()
        self.viz.splitDockWidget(*self.chosen, Qt.Vertical)
        self.viz.resizeDocks(
            self.chosen, [max(320, int(self.viz.width() * 0.36))] * 2, Qt.Horizontal
        )
        self.viz.resizeDocks(
            self.chosen, [int(self.viz.height() * 0.6), int(self.viz.height() * 0.4)], Qt.Vertical
        )
        self.pending_figures = list(zip(self.chosen, self.FIGURES))
        self._select_figures()
        if self.pending_figures:
            self._timer.start()

    def _select_figures(self):
        """Bind the lazy catalog once; later manual figure selections stay selected."""
        if self.state is None or self.manager is None:
            return
        dashboard = self.manager.sources.get("Sensing")
        if dashboard is not None and not self.supports_dashboard(dashboard):
            self._timer.stop()
            self.failed("This recording has no saved range–velocity map.")
            return
        self.manager.refresh_choices()
        remaining = []
        for view, key in self.pending_figures:
            if not isValid(view) or view.closed:
                continue
            index = view.figure_combo.findData(key)
            if index >= 0 and not self.manager.is_used(key, except_view=view):
                view.figure_combo.setCurrentIndex(index)
            if view.choice_id != key:
                remaining.append((view, key))
        self.pending_figures = remaining
        if not remaining:
            self._timer.stop()

    def owns_widget(self, widget):
        """Recognize all presentation plots, including reused docks."""
        return any(
            isValid(view) and (widget is view or view.isAncestorOf(widget)) for view in self.chosen
        )

    def restore(self, *, restore_docks=True):
        if isValid(self._timer):
            self._timer.stop()
            self._timer.deleteLater()
        for view, enabled in self.mouse_propagation:
            if isValid(view):
                view.setAttribute(Qt.WA_NoMousePropagation, enabled)
        self.mouse_propagation.clear()
        if restore_docks:
            # Release temporary selections first so no other dock owns a
            # figure when its original choice is restored.
            for view, _choice in self.borrowed_choices:
                if isValid(view) and not view.closed:
                    view.release_figure()
            for view in self.created:
                if isValid(view) and not view.closed:
                    view.release_figure()
            if self.manager is not None:
                self.manager.refresh_choices()
            for view, choice in self.borrowed_choices:
                if isValid(view) and not view.closed and choice is not None:
                    view.figure_combo.setCurrentIndex(view.figure_combo.findData(choice))
        self.borrowed_choices.clear()
        super().restore(restore_docks=restore_docks)
        self.chosen.clear()
        self.pending_figures.clear()
        if restore_docks:
            if self.overlays is not None:
                update_state(self.viz, **self.overlays)
            for checkbox, checked in reversed(self.checkboxes):
                if isValid(checkbox):
                    checkbox.setChecked(checked)
        self.checkboxes.clear()
        self.overlays = None
