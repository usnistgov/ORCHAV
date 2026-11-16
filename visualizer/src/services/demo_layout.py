"""Reversible exhibition placement of the existing scene and optional docks."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget
from shiboken6 import isValid


class ExhibitionLayout:
    """Borrow one target view and one figure without changing their selections."""

    def __init__(self, visualizer):
        self.viz = visualizer
        self.state = None
        self.created = []
        self.catalog_signal = None
        self.figure = None
        self.owned_manager = None

    def enter(self):
        """Use most of an embedded window for the scene until manual takeover."""
        controls, docks = self._capture_layout()
        self._enter_actor_layout(controls, docks)

    def _capture_layout(self):
        """Save placement before borrowing any existing docks."""
        workspace = getattr(self.viz, "_visualization_workspace", None)
        splitter = getattr(workspace, "splitter", None)
        if splitter is None or splitter.count() < 2:
            raise ValueError("Exhibition layout requires the embedded scene viewport")
        controls = splitter.widget(0)
        docks = self.viz.findChildren(QDockWidget)
        self.state = {
            "window": self.viz.saveState(),
            "splitter": splitter,
            "splitter_state": splitter.saveState(),
            "controls": controls,
            "controls_visible": not controls.isHidden(),
            "docks": [(dock, not dock.isHidden()) for dock in docks],
        }
        return controls, docks

    def _enter_actor_layout(self, controls, docks):
        target_views = list(getattr(self.viz.renderer, "get_target_views", lambda: ())())
        figure_manager = getattr(self.viz, "live_figure_manager", None)
        figure_views = list(getattr(figure_manager, "views", ()))
        target = self._preferred(target_views)
        figure = self._preferred(figure_views)
        for dock in docks:
            dock.hide()
        controls.hide()
        if target is None:
            create = getattr(self.viz.renderer, "open_target_view", None)
            target = create() if callable(create) else None
            if target is not None:
                self.created.append(target)
        if figure is None:
            if figure_manager is None:
                from ..widgets.live_figure_dock import LiveFigureManager

                figure_manager = LiveFigureManager(self.viz)
                self.viz.live_figure_manager = figure_manager
                self.owned_manager = figure_manager
            figure = figure_manager.add()
            if figure is not None:
                self.created.append(figure)
                self.figure = figure
                owner = figure_manager.sources.get("Sensing")
                signal = getattr(owner, "live_figures_changed", None)
                if signal is not None:
                    self.catalog_signal = signal
                    signal.connect(self._prefer_sensing_figure)
                self._prefer_sensing_figure()
        chosen = [dock for dock in (target, figure) if dock is not None]
        for dock in chosen:
            dock.setFloating(False)
            self.viz.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.show()
        if len(chosen) == 2:
            self.viz.splitDockWidget(chosen[0], chosen[1], Qt.Vertical)
        if chosen:
            self.viz.resizeDocks(
                chosen, [max(300, self.viz.width() // 4)] * len(chosen), Qt.Horizontal
            )
            self.viz.resizeDocks(
                chosen, [self.viz.height() // len(chosen)] * len(chosen), Qt.Vertical
            )

    @staticmethod
    def _preferred(views):
        usable = [view for view in views if isValid(view) and not getattr(view, "closed", False)]
        return next((view for view in usable if not view.isHidden()), usable[0] if usable else None)

    def _prefer_sensing_figure(self):
        """Use the saved range–velocity plot when its lazy catalog is ready."""
        view = self.figure
        if view is None or not isValid(view) or view.closed:
            return
        index = view.figure_combo.findData("Sensing:rd_plot")
        if index >= 0:
            view.figure_combo.setCurrentIndex(index)

    def owns_widget(self, widget):
        """Identify input aimed at a temporary dock that takeover will close."""
        return any(
            isValid(dock) and (widget is dock or dock.isAncestorOf(widget)) for dock in self.created
        )

    def restore(self, *, restore_docks=True):
        """Close only demo-created docks and restore the user's exact arrangement."""
        if self.catalog_signal is not None:
            try:
                self.catalog_signal.disconnect(self._prefer_sensing_figure)
            except RuntimeError:
                pass  # Scenario teardown may already have destroyed the owner.
            self.catalog_signal = None
        for dock in self.created:
            if isValid(dock):
                dock.close()
        self.created.clear()
        if self.owned_manager is not None:
            self.owned_manager.close()
            self.owned_manager = None
        state, self.state = self.state, None
        if state is None:
            return
        if restore_docks:
            self.viz.restoreState(state["window"])
            for dock, visible in state["docks"]:
                if isValid(dock):
                    dock.setVisible(visible)
        if isValid(state["controls"]):
            state["controls"].setVisible(state["controls_visible"])
        if isValid(state["splitter"]):
            state["splitter"].restoreState(state["splitter_state"])
