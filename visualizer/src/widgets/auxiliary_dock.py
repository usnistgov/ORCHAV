"""Shared placement controls for views that belong to the visualizer window.

Keep Qt's native title bar for dragging, floating and tab docking. The title-bar
menu also provides explicit placement actions, including joining another view.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QMenu, QToolButton, QWidget

DOCK_AREAS = {
    "top": Qt.TopDockWidgetArea,
    "right": Qt.RightDockWidgetArea,
    "bottom": Qt.BottomDockWidgetArea,
}
INITIAL_DOCK_FRACTION = 0.4


class AuxiliaryDock(QDockWidget):
    """An owned floating window that can also share the main window's dock areas."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setAllowedAreas(
            Qt.TopDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.setToolTip(
            "Drag the title bar to dock or float. Right-click it for placement options."
        )
        if isinstance(parent, QMainWindow):
            parent.setDockOptions(
                parent.dockOptions() | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks
            )

    def attach(self, *, floating: bool = True) -> None:
        """Register with the owner before showing, preserving native docking."""
        owner = self.parentWidget()
        if isinstance(owner, QMainWindow):
            owner.addDockWidget(Qt.RightDockWidgetArea, self)
            self.setFloating(floating)
        self.hide()

    def place(self, position: str) -> None:
        """Move an existing view without replacing its content or selection."""
        owner = self.parentWidget()
        if not isinstance(owner, QMainWindow):
            return
        if position == "float":
            self.setFloating(True)
        else:
            area = DOCK_AREAS[position]
            owner.addDockWidget(area, self)
            self.setFloating(False)
            direction = Qt.Horizontal if area == Qt.RightDockWidgetArea else Qt.Vertical
            extent = owner.width() if direction == Qt.Horizontal else owner.height()
            owner.resizeDocks([self], [int(extent * INITIAL_DOCK_FRACTION)], direction)
        self.show()
        self.raise_()

    def join(self, other: AuxiliaryDock, *, tabbed: bool) -> None:
        """Share another view's dock area as tabs or adjacent panes."""
        owner = self.parentWidget()
        if not isinstance(owner, QMainWindow) or other.parentWidget() is not owner:
            return
        if other.isFloating():
            other.place("right")
        area = owner.dockWidgetArea(other)
        owner.addDockWidget(area, self)
        self.setFloating(False)
        if tabbed:
            owner.tabifyDockWidget(other, self)
        else:
            direction = Qt.Vertical if area == Qt.RightDockWidgetArea else Qt.Horizontal
            extent = other.height() if direction == Qt.Vertical else other.width()
            owner.splitDockWidget(other, self, direction)
            owner.resizeDocks([other, self], [extent // 2, extent // 2], direction)
        self.show()
        self.raise_()

    def create_placement_button(self) -> QToolButton:
        """Expose placement even when the window manager owns a floating title bar."""
        button = QToolButton(self)
        button.setText("Dock")
        button.setToolTip("Dock, float, or arrange this view with another view.")
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        menu.aboutToShow.connect(lambda: self.placement_menu(menu))
        button.setMenu(menu)
        return button

    def placement_menu(self, menu: QMenu | None = None) -> QMenu:
        """Build actions on demand so the list of open peer views stays current."""
        if menu is None:
            menu = QMenu(self)
        else:
            menu.clear()
        for position in DOCK_AREAS:
            menu.addAction(f"Dock {position}", lambda position=position: self.place(position))
        menu.addAction("Float", lambda: self.place("float"))
        owner = self.parentWidget()
        if isinstance(owner, QMainWindow):
            peers = [
                dock
                for dock in owner.findChildren(AuxiliaryDock)
                if dock is not self and dock.parentWidget() is owner and not dock.isHidden()
            ]
            if peers:
                tabs = menu.addMenu("Tab with")
                adjacent = menu.addMenu("Place beside")
                for dock in peers:
                    tabs.addAction(
                        dock.windowTitle(), lambda dock=dock: self.join(dock, tabbed=True)
                    )
                    adjacent.addAction(
                        dock.windowTitle(), lambda dock=dock: self.join(dock, tabbed=False)
                    )
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = self.placement_menu()
        menu.exec(event.globalPos())
        menu.deleteLater()
