"""Protect widget keyboard actions, with optional Inspector wheel protection."""

from __future__ import annotations

from PySide6.QtCore import QChildEvent, QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QTabBar,
    QTextEdit,
    QWidget,
)


class ControlInputGuard(QObject):
    """Keep native wheel editing unless an Inspector explicitly opts into protection."""

    def __init__(self, root: QWidget, *, require_wheel_focus: bool = False) -> None:
        super().__init__(root)
        self._require_wheel_focus = require_wheel_focus
        self.watch(root)

    def watch(self, obj: QObject) -> None:
        if not isinstance(obj, QWidget):
            return
        obj.installEventFilter(self)
        self._focus_policy(obj)
        for child in obj.findChildren(QWidget):
            child.installEventFilter(self)
            self._focus_policy(child)

    def _focus_policy(self, widget: QObject) -> None:
        if self._require_wheel_focus and isinstance(
            widget, (QComboBox, QAbstractSpinBox, QAbstractSlider)
        ):
            widget.setFocusPolicy(Qt.StrongFocus)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Qt can announce children before their concrete widget is constructed.
        # Wait for polishing before asking Shiboken for its Python wrapper.
        # Also tolerate an invalid transient wrapper at this binding boundary.
        if not isinstance(event, QEvent):
            return False
        if isinstance(event, QChildEvent) and event.type() == QEvent.ChildPolished:
            self.watch(event.child())
        elif event.type() == QEvent.Polish:
            self._focus_policy(watched)
        elif isinstance(event, QKeyEvent) and event.type() == QEvent.ShortcutOverride:
            if (
                event.modifiers() == Qt.NoModifier
                and event.key()
                in (
                    Qt.Key_Space,
                    Qt.Key_Left,
                    Qt.Key_Right,
                )
                and isinstance(
                    watched,
                    (
                        QAbstractButton,
                        QAbstractItemView,
                        QAbstractSlider,
                        QAbstractSpinBox,
                        QComboBox,
                        QLineEdit,
                        QPlainTextEdit,
                        QTabBar,
                        QTextEdit,
                    ),
                )
            ):
                event.accept()
                return True
        elif (
            self._require_wheel_focus
            and isinstance(event, QWheelEvent)
            and isinstance(watched, QWidget)
        ):
            control = watched
            while control is not None and control is not self.parent():
                if isinstance(control, (QComboBox, QAbstractSpinBox, QAbstractSlider)):
                    break
                control = control.parentWidget()
            if not isinstance(control, (QComboBox, QAbstractSpinBox, QAbstractSlider)):
                return False
            # Scroll bars must always retain ordinary wheel scrolling.
            from PySide6.QtWidgets import QScrollBar

            if isinstance(control, QScrollBar):
                return False
            focus = QApplication.focusWidget()
            if control.hasFocus() or (focus is not None and control.isAncestorOf(focus)):
                return False
            scroll = control.parentWidget()
            while scroll is not None and not isinstance(scroll, QScrollArea):
                scroll = scroll.parentWidget()
            if scroll is not None:
                viewport = scroll.viewport()
                forwarded = QWheelEvent(
                    QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
                    event.globalPosition(),
                    event.pixelDelta(),
                    event.angleDelta(),
                    event.buttons(),
                    event.modifiers(),
                    event.phase(),
                    event.inverted(),
                    event.source(),
                    event.pointingDevice(),
                )
                QApplication.sendEvent(viewport, forwarded)
            event.accept()
            return True
        return False


def protect_controls(root: QWidget, *, require_wheel_focus: bool = False) -> ControlInputGuard:
    """Guard current/future controls, preserving ordinary wheel editing by default."""
    guard = getattr(root, "_control_input_guard", None)
    if guard is None:
        guard = ControlInputGuard(root, require_wheel_focus=require_wheel_focus)
        root._control_input_guard = guard
    return guard
