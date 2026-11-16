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


@pytest.mark.parametrize("size", [34, 44])
def test_scaled_transport_icon_and_shell_fill_current_geometry(size, qtbot):
    from PySide6.QtGui import QColor

    from visualizer.src.app.theme import current_theme
    from visualizer.src.widgets.transport_button import TransportButton

    button = TransportButton("stop")
    qtbot.addWidget(button)
    button.setFixedSize(size, size)
    image = button.grab().toImage()
    foreground = QColor(current_theme().text_primary)
    pixels = [
        (x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y) == foreground
    ]
    assert pixels
    xs, ys = zip(*pixels, strict=True)
    assert abs((min(xs) + max(xs)) / 2 - (size - 1) / 2) <= 0.5
    assert abs((min(ys) + max(ys)) / 2 - (size - 1) / 2) <= 0.5
    assert max(xs) - min(xs) + 1 >= int(size * 0.25)
    # An enlarged button must paint its right edge, not leave an empty strip.
    assert image.pixelColor(size - 1, size // 2) == QColor(current_theme().border_primary)
