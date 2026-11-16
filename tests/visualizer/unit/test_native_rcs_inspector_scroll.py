"""Scrolling an Inspector must not silently change a target's fields."""

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visualizer.src.authoring.inspector_scroll import InspectorScrollArea


def _wheel(control, delta=-120):
    local = control.rect().center()
    event = QWheelEvent(
        QPointF(local),
        QPointF(control.mapToGlobal(local)),
        QPoint(),
        QPoint(0, delta),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(control, event)


def _form(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    focus = QPushButton("Keep keyboard focus")
    layout.addWidget(focus)
    scroll = InspectorScrollArea(host)
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)
    content = QWidget()
    rows = QVBoxLayout(content)
    combo = QComboBox()
    combo.addItems(("Fixed", "Keyframes", "Spin"))
    spin = QSpinBox()
    spin.setRange(-360, 360)
    spin.setValue(180)
    rows.addWidget(combo)
    rows.addWidget(spin)
    rows.addSpacing(900)
    scroll.setWidget(content)
    host.resize(320, 240)
    host.show()
    focus.setFocus()
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)
    return host, focus, scroll, rows, combo, spin


def test_unfocused_choice_and_spin_wheels_scroll_without_mutation(qtbot):
    _host, focus, scroll, _rows, combo, spin = _form(qtbot)
    for control in (combo, spin, spin.lineEdit()):
        scroll.verticalScrollBar().setValue(0)
        focus.setFocus()
        assert not control.hasFocus()
        _wheel(control)
        assert combo.currentText() == "Fixed"
        assert spin.value() == 180
        assert scroll.verticalScrollBar().value() > 0


def test_focused_control_can_be_deliberately_wheel_edited(qtbot):
    _host, _focus, scroll, _rows, combo, spin = _form(qtbot)
    scroll.verticalScrollBar().setValue(0)
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus)
    _wheel(spin, 120)
    assert spin.value() == 181
    assert combo.currentText() == "Fixed"


def test_dynamic_editor_controls_receive_the_same_guard(qtbot):
    _host, focus, scroll, rows, _combo, _spin = _form(qtbot)
    dynamic = QSpinBox(scroll.widget())
    dynamic.setValue(42)
    rows.insertWidget(0, dynamic)
    dynamic.show()
    focus.setFocus()
    qtbot.waitUntil(lambda: dynamic.focusPolicy() == Qt.StrongFocus)
    _wheel(dynamic)
    assert dynamic.value() == 42
    assert scroll.verticalScrollBar().value() > 0
