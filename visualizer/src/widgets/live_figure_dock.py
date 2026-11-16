"""Optional docks that borrow plots from the existing live dashboards.

Dashboards retain ownership of data, selection and refresh pacing. These docks
only choose and place their existing widgets, returning each plot on close.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QEvent, QSignalBlocker, Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from shiboken6 import isValid

from .auxiliary_dock import AuxiliaryDock
from .dashboard_plot import DashboardPlot

logger = logging.getLogger(__name__)
MAX_LIVE_FIGURES = 4


class _FigureCombo(QComboBox):
    def __init__(self, parent: QWidget, refresh: Any) -> None:
        super().__init__(parent)
        self._refresh = refresh

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        self._refresh()
        super().showPopup()


class LiveFigureDock(AuxiliaryDock):
    """One selectable live chart, sharing the normal dashboard's plot object."""

    def __init__(self, manager: LiveFigureManager, serial: int) -> None:
        super().__init__(f"Live figure {serial}", manager.visualizer)
        self.manager = manager
        self.owner: Any = None
        self.figure_key: str | None = None
        self.choice_id: str | None = None
        self.plot: QWidget | None = None
        self.closed = False
        self.setObjectName(f"liveFigure{serial}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        content = QWidget(self)
        content.setMinimumSize(240, 160)
        self.setWidget(content)
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.figure_combo = _FigureCombo(self, manager.refresh_choices)
        self.figure_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.figure_combo.setMinimumContentsLength(24)
        self.figure_combo.setToolTip(
            "Choose a live figure; resize its window or drag the title bar to dock it. "
            "A figure can occupy one window at a time."
        )
        self.figure_combo.currentIndexChanged.connect(self._select_figure)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.figure_combo, 1)
        toolbar.addWidget(self.create_placement_button())
        self.content_layout.addLayout(toolbar)
        self.notice = QLabel("Choose a live figure.", content)
        self.notice.setWordWrap(True)
        self.content_layout.addWidget(self.notice)
        self.visibilityChanged.connect(self._visibility_changed)

    def _select_figure(self, index: int) -> None:
        choice_id = self.figure_combo.itemData(index)
        if choice_id is None or choice_id == self.choice_id:
            return
        choice = self.manager.choices.get(choice_id)
        if choice is None or self.manager.is_used(choice_id, except_view=self):
            return
        owner, key, title = choice
        self.release_figure()
        try:
            plot = owner.acquire_live_figure(key, self.widget())
        except (RuntimeError, ValueError, KeyError):
            logger.exception("Unable to open live figure %s", title)
            self.notice.setText("This figure is unavailable for the current recording.")
            self.notice.show()
            return
        self.owner, self.figure_key, self.choice_id = owner, key, choice_id
        self.plot = plot
        for chart in self._charts(plot):
            chart.set_live_figure(True)
        self.content_layout.addWidget(plot, 1)
        self.notice.hide()
        self.setWindowTitle(title)
        plot.show()
        self._visibility_changed()
        self.manager.refresh_choices()

    def release_figure(self) -> None:
        """Return the borrowed widget before this dock or its owner is destroyed."""
        owner, key, plot = self.owner, self.figure_key, self.plot
        self.owner = self.figure_key = self.choice_id = self.plot = None
        if plot is not None and isValid(plot):
            for chart in self._charts(plot):
                chart.set_live_figure(False)
            self.content_layout.removeWidget(plot)
        if owner is not None and isValid(owner):
            owner.release_live_figure(key)

    @staticmethod
    def _charts(widget: QWidget) -> list[DashboardPlot]:
        return ([widget] if isinstance(widget, DashboardPlot) else []) + widget.findChildren(
            DashboardPlot
        )

    def _visibility_changed(self, *_args: Any) -> None:
        if self.owner is not None and isValid(self.owner):
            self.owner.refresh_live_figure_visibility()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._visibility_changed()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if not self.closed:
            self.closed = True
            self.release_figure()
            self.manager.remove(self)
        super().closeEvent(event)


class LiveFigureManager:
    """Place up to four figures; their dashboards own all data and timers."""

    def __init__(self, visualizer: Any) -> None:
        self.visualizer = visualizer
        self.views: list[LiveFigureDock] = []
        self.sources: dict[str, Any] = {}
        self._source_connections: dict[str, Any] = {}
        self.choices: dict[str, tuple[Any, str, str]] = {}
        self._serial = 0

    def _load_sources(self) -> None:
        providers = {
            "Channel": getattr(self.visualizer, "metrics_service", None),
            "Sensing": getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get(
                "sensing"
            ),
        }
        for name, provider in providers.items():
            if name in self.sources and isValid(self.sources[name]):
                continue
            ensure = getattr(provider, "ensure_live_figure_dashboard", None)
            if not callable(ensure):
                continue
            owner = ensure()
            if owner is not None:
                self.sources[name] = owner

                def callback(_obj=None, source=name):
                    self._source_destroyed(source)

                self._source_connections[name] = callback
                owner.destroyed.connect(callback)
                changed = getattr(owner, "live_figures_changed", None)
                if changed is not None:
                    changed.connect(self.refresh_choices)

    def prepare_sources(self) -> None:
        """Start lazy dashboard catalogs without allocating another figure dock."""
        self._load_sources()
        self.refresh_choices()

    def _source_destroyed(self, name: str) -> None:
        owner = self.sources.pop(name, None)
        self._source_connections.pop(name, None)
        for view in tuple(self.views):
            if view.owner is owner:
                view.close()

    def refresh_choices(self) -> None:
        """Recheck availability on demand, without polling or reading frames."""
        choices = {}
        for source, owner in self.sources.items():
            if not isValid(owner):
                continue
            for key, title in owner.available_live_figures():
                choices[f"{source}:{key}"] = (owner, key, f"{source} · {title}")
        self.choices = choices
        for view in self.views:
            combo = view.figure_combo
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem("Choose a live figure…", None)
                for choice_id, (_owner, _key, title) in choices.items():
                    combo.addItem(title, choice_id)
                    if self.is_used(choice_id, except_view=view):
                        combo.model().item(combo.count() - 1).setEnabled(False)
                combo.setCurrentIndex(max(0, combo.findData(view.choice_id)))

    def is_used(self, choice_id: str, *, except_view: LiveFigureDock | None = None) -> bool:
        return any(view is not except_view and view.choice_id == choice_id for view in self.views)

    def add(self) -> LiveFigureDock | None:
        if len(self.views) >= MAX_LIVE_FIGURES:
            self.visualizer.statusBar().showMessage("At most four live figures can be open.", 4000)
            return None
        if (
            getattr(self.visualizer, "frame_source", None) is None
            and getattr(self.visualizer, "current_view_model", None) is None
        ):
            self.visualizer.statusBar().showMessage("Open a recording first.", 4000)
            return None
        self.prepare_sources()
        if not self.sources:
            self.visualizer.statusBar().showMessage(
                "No live figures are available for this recording.", 4000
            )
            return None
        self._serial += 1
        view = LiveFigureDock(self, self._serial)
        self.views.append(view)
        self.visualizer.addDockWidget(Qt.RightDockWidgetArea, view)
        # A single dock fills an entire side of QMainWindow. Start floating so
        # both dimensions are adjustable without rearranging the user's docks.
        view.setFloating(True)
        view.show()
        self.refresh_choices()
        preferred = ("Sensing:rd_plot", "Channel:pdp", *self.choices)
        selected = next(
            (key for key in preferred if key in self.choices and not self.is_used(key)), None
        )
        if selected is not None:
            view.figure_combo.setCurrentIndex(view.figure_combo.findData(selected))
        view.resize(440, 360)
        return view

    def remove(self, view: LiveFigureDock) -> None:
        if view in self.views:
            self.views.remove(view)
        self.refresh_choices()

    def close(self) -> None:
        """Return all figures before scenario replacement or dashboard cleanup."""
        for view in tuple(self.views):
            view.close()
        for name, owner in self.sources.items():
            if isValid(owner):
                owner.destroyed.disconnect(self._source_connections[name])
                changed = getattr(owner, "live_figures_changed", None)
                if changed is not None:
                    changed.disconnect(self.refresh_choices)
        self.choices.clear()
        self.sources.clear()
        self._source_connections.clear()
