"""QPainter-rendered transport buttons used by playback panels.

The widget owns only icon drawing, fixed sizing, hover state, and checkable
play-button presentation. Playback semantics and signal routing stay in the
animation panel/controller layer.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

from ..app.theme import current_theme, get_theme_manager

_ICON_TYPES = {"prev", "rev_play", "play", "stop", "next"}


class TransportButton(QAbstractButton):
    """A transport-control button rendered entirely via QPainter.

    Args:
        icon_type: One of ``"prev"``, ``"rev_play"``, ``"play"``,
            ``"stop"``, ``"next"``.
        size: Button diameter in logical pixels (default 34).
    """

    def __init__(self, icon_type: str, size: int = 34, parent=None):
        """Create a fixed-size transport button for one supported icon type."""
        super().__init__(parent)
        if icon_type not in _ICON_TYPES:
            raise ValueError(f"Unknown icon_type {icon_type!r}, expected one of {_ICON_TYPES}")
        self._icon_type = icon_type
        self._size = size
        self._hovered = False

        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

        # Play controls mirror playback toggle state; skip/stop controls remain momentary.
        if icon_type in ("play", "rev_play"):
            self.setCheckable(True)

        get_theme_manager().theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, _theme: object) -> None:
        """Request repaint when application theme tokens change."""
        self.update()

    def sizeHint(self) -> QSize:
        """Return the fixed logical button size expected by panel layouts."""
        return QSize(self._size, self._size)

    def minimumSizeHint(self) -> QSize:
        """Keep Qt from shrinking transport controls below their icon geometry."""
        return QSize(self._size, self._size)

    def enterEvent(self, event):
        """Track hover state so the painter can apply themed affordances."""
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Clear hover state and request repaint after the pointer exits."""
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        """Paint the themed button shell and icon without relying on font glyphs."""
        t = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(0.5, 0.5, self._size - 1, self._size - 1)

        if self.isDown():
            bg = QColor(t.border_subtle)
        elif self.isChecked():
            bg = QColor(t.accent_active)
        elif self._hovered:
            bg = QColor(t.bg_tertiary)
        else:
            bg = QColor(t.bg_secondary)

        painter.setPen(QPen(QColor(t.border_primary), 1.0))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 6, 6)

        if self.isChecked():
            icon_color = QColor("#ffffff")
        elif self._hovered:
            icon_color = QColor(t.accent)
        else:
            icon_color = QColor(t.text_primary)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(icon_color))

        cx = self._size / 2.0
        cy = self._size / 2.0
        # Keep icon proportions independent of the fixed logical button size.
        s = self._size * 0.4

        if self._icon_type == "play":
            self._draw_play(painter, cx, cy, s)
        elif self._icon_type == "rev_play":
            self._draw_play(painter, cx, cy, s, reverse=True)
        elif self._icon_type == "stop":
            self._draw_stop(painter, cx, cy, s)
        elif self._icon_type == "prev":
            self._draw_skip(painter, cx, cy, s, reverse=True)
        elif self._icon_type == "next":
            self._draw_skip(painter, cx, cy, s)

        painter.end()

    @staticmethod
    def _draw_play(painter: QPainter, cx: float, cy: float, s: float, reverse: bool = False):
        """Draw a triangle pointing right (or left if reverse)."""
        path = QPainterPath()
        hs = s / 2.0
        if reverse:
            path.moveTo(QPointF(cx + hs * 0.3, cy - hs))
            path.lineTo(QPointF(cx - hs * 0.9, cy))
            path.lineTo(QPointF(cx + hs * 0.3, cy + hs))
        else:
            path.moveTo(QPointF(cx - hs * 0.3, cy - hs))
            path.lineTo(QPointF(cx + hs * 0.9, cy))
            path.lineTo(QPointF(cx - hs * 0.3, cy + hs))
        path.closeSubpath()
        painter.drawPath(path)

    @staticmethod
    def _draw_stop(painter: QPainter, cx: float, cy: float, s: float):
        """Draw a square."""
        hs = s * 0.4
        painter.drawRoundedRect(QRectF(cx - hs, cy - hs, hs * 2, hs * 2), 2, 2)

    @staticmethod
    def _draw_skip(painter: QPainter, cx: float, cy: float, s: float, reverse: bool = False):
        """Draw a skip icon (triangle + bar)."""
        hs = s / 2.0
        bar_w = s * 0.12
        gap = s * 0.05

        if reverse:
            bar_x = cx - hs * 0.7
            painter.drawRoundedRect(QRectF(bar_x, cy - hs * 0.8, bar_w, hs * 1.6), 1, 1)
            tri = QPainterPath()
            tri.moveTo(QPointF(cx + hs * 0.5, cy - hs * 0.8))
            tri.lineTo(QPointF(bar_x + bar_w + gap, cy))
            tri.lineTo(QPointF(cx + hs * 0.5, cy + hs * 0.8))
            tri.closeSubpath()
            painter.drawPath(tri)
        else:
            bar_x = cx + hs * 0.7 - bar_w
            painter.drawRoundedRect(QRectF(bar_x, cy - hs * 0.8, bar_w, hs * 1.6), 1, 1)
            tri = QPainterPath()
            tri.moveTo(QPointF(cx - hs * 0.5, cy - hs * 0.8))
            tri.lineTo(QPointF(bar_x - gap, cy))
            tri.lineTo(QPointF(cx - hs * 0.5, cy + hs * 0.8))
            tri.closeSubpath()
            painter.drawPath(tri)
