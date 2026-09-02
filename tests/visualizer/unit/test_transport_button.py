"""Tests for the QPainter-rendered transport buttons."""

import os

import pytest

# Ensure headless mode for CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

# Ensure QApplication exists
_app = QApplication.instance() or QApplication([])


def test_all_icon_types_create():
    from visualizer.src.widgets.transport_button import TransportButton

    for icon_type in ("prev", "rev_play", "play", "stop", "next"):
        btn = TransportButton(icon_type)
        assert btn.sizeHint().width() == 34
        assert btn.sizeHint().height() == 34


def test_invalid_icon_type_raises():
    from visualizer.src.widgets.transport_button import TransportButton

    with pytest.raises(ValueError, match="Unknown icon_type"):
        TransportButton("invalid_type")


def test_play_and_rev_play_are_checkable():
    from visualizer.src.widgets.transport_button import TransportButton

    play = TransportButton("play")
    rev = TransportButton("rev_play")
    assert play.isCheckable()
    assert rev.isCheckable()


def test_non_play_buttons_are_not_checkable():
    from visualizer.src.widgets.transport_button import TransportButton

    for icon_type in ("prev", "stop", "next"):
        btn = TransportButton(icon_type)
        assert not btn.isCheckable()


def test_custom_size():
    from visualizer.src.widgets.transport_button import TransportButton

    btn = TransportButton("play", size=48)
    assert btn.sizeHint().width() == 48
    assert btn.minimumSizeHint().height() == 48


def test_paint_event_runs_without_crash():
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QPaintEvent

    from visualizer.src.widgets.transport_button import TransportButton

    for icon_type in ("prev", "rev_play", "play", "stop", "next"):
        btn = TransportButton(icon_type)
        btn.resize(34, 34)
        # Trigger a paint event
        event = QPaintEvent(QRect(0, 0, 34, 34))
        btn.paintEvent(event)


def test_checked_state_paint():
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QPaintEvent

    from visualizer.src.widgets.transport_button import TransportButton

    btn = TransportButton("play")
    btn.setChecked(True)
    event = QPaintEvent(QRect(0, 0, 34, 34))
    btn.paintEvent(event)
