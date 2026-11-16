"""Real Qt checks for keyboard protection and ordinary wheel editing."""

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence, QShortcut, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from visualizer.src.widgets.input_guard import protect_controls


def test_space_toggles_focused_checkbox_without_playback(qapp, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    protect_controls(root)
    layout = QVBoxLayout(root)
    check = QCheckBox("Enabled")
    layout.addWidget(check)
    calls = []
    shortcut = QShortcut(QKeySequence("Space"), root)
    shortcut.activated.connect(lambda: calls.append("play"))
    root.show()
    root.activateWindow()
    qtbot.waitUntil(root.isActiveWindow)
    check.setFocus()
    qtbot.waitUntil(check.hasFocus)
    QTest.keyClick(check, Qt.Key_Space)
    assert check.isChecked()
    assert calls == []
    root.setFocus()
    qtbot.waitUntil(root.hasFocus)
    QTest.keyClick(root, Qt.Key_Space)
    assert calls == ["play"]
    root.close()


@pytest.mark.parametrize("in_scroll_area", [False, True])
@pytest.mark.parametrize("kind", ["spin", "combo", "slider"])
def test_wheel_edits_hovered_field_without_clicking(qtbot, kind, in_scroll_area):
    root = QWidget()
    qtbot.addWidget(root)
    protect_controls(root)
    layout = QVBoxLayout(root)
    focus = QPushButton("Keyboard focus stays here")
    layout.addWidget(focus)
    scroll = None
    if in_scroll_area:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        page = QWidget()
        page.setMinimumHeight(1000)
        fields = QVBoxLayout(page)
        scroll.setWidget(page)
    else:
        fields = layout

    root.resize(350, 250)
    root.show()
    root.activateWindow()
    # Add fields after the guard and visible panel, as lazy sections do.
    if kind == "combo":
        control = QComboBox()
        control.addItems(["First", "Second", "Third"])
        control.setCurrentIndex(1)
        value = control.currentIndex
    else:
        control = QSlider(Qt.Horizontal) if kind == "slider" else QDoubleSpinBox()
        control.setValue(10)
        value = control.value
    native_focus_policy = control.focusPolicy()
    fields.addWidget(control)
    fields.addStretch()
    focus.setFocus()
    qtbot.waitUntil(control.isVisible)
    QApplication.processEvents()
    assert not control.hasFocus()
    assert control.focusPolicy() == native_focus_policy
    before = value()
    pos = control.rect().center()
    event = QWheelEvent(
        QPointF(pos),
        QPointF(control.mapToGlobal(pos)),
        QPoint(),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(control, event)
    assert value() != before
    if scroll is not None:
        assert scroll.verticalScrollBar().value() == 0


def test_wheel_focus_protection_is_opt_in(qapp, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    protect_controls(root, require_wheel_focus=True)
    layout = QVBoxLayout(root)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)
    page = QWidget()
    content = QVBoxLayout(page)
    spin = QDoubleSpinBox()
    spin.setValue(10)
    content.addWidget(spin)
    content.addWidget(QWidget(), 1)
    page.setMinimumHeight(1000)
    scroll.setWidget(page)
    root.resize(350, 250)
    root.show()
    root.activateWindow()
    qtbot.waitUntil(root.isActiveWindow)
    root.setFocus()
    qtbot.waitUntil(root.hasFocus)

    def wheel():
        pos = spin.rect().center()
        event = QWheelEvent(
            QPointF(pos),
            QPointF(spin.mapToGlobal(pos)),
            QPoint(),
            QPoint(0, -120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.ScrollUpdate,
            False,
        )
        qapp.sendEvent(spin, event)

    wheel()
    assert spin.value() == 10
    assert scroll.verticalScrollBar().value() > 0
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus)
    wheel()
    assert spin.value() == 9
    root.close()


def test_guard_ignores_non_event_wrapper(qtbot):
    from PySide6.QtGui import QStandardItem

    root = QWidget()
    qtbot.addWidget(root)
    guard = protect_controls(root)
    assert guard.eventFilter(root, QStandardItem("transient wrapper")) is False
