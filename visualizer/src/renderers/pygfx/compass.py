"""Geographic-scene compass using the existing Qt HUD and capture layer."""

from __future__ import annotations

import math

import numpy as np

_OVERLAY_ID = "compass"
_SIZE = 72
_UNAVAILABLE_REASON = "This scene has no geographic coordinate metadata."


def north_heading(camera_matrix, north_direction_xy=(0.0, 1.0)) -> float | None:
    """Return screen-clockwise scene-north heading, independent of projection.

    Camera right projected onto the XY ground plane defines the horizontal
    heading. Unlike projecting a +Y line, it does not collapse or flip when
    looking north along the horizon. Overhead cameras retain their screen-up
    convention. A rolled camera with vertical screen-right uses screen-up.
    """
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix[:3, :3]).all():
        return None
    right = matrix[:2, 0]
    if np.linalg.norm(right) < 1e-8:
        up = matrix[:2, 1]
        right = np.array([up[1], -up[0]])
    if np.linalg.norm(right) < 1e-8:
        return None
    east, north = north_direction_xy
    return math.atan2(
        float(right[0] * east + right[1] * north),
        float(-right[1] * east + right[0] * north),
    )


def _compass_pixmap(heading: float, style: str, device_ratio: float):
    """Rasterize a tiny vector glyph only after the heading or palette changes."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF

    pixmap = QPixmap(round(_SIZE * device_ratio), round(_SIZE * device_ratio))
    pixmap.setDevicePixelRatio(device_ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        foreground = QColor("#203b6b" if style == "card" else "#f5f7fc")
        muted = QColor(foreground)
        muted.setAlpha(95)
        accent = QColor("#c27622" if style == "card" else "#f0bc63")
        center = QPointF(_SIZE / 2, _SIZE / 2)
        painter.setPen(QPen(muted, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, 23, 23)
        direction = QPointF(math.sin(heading), -math.cos(heading))
        side = QPointF(-direction.y(), direction.x())
        painter.setPen(Qt.NoPen)
        painter.setBrush(muted)
        painter.drawPolygon(
            QPolygonF([center - direction * 15, center + side * 4, center - side * 4])
        )
        painter.setBrush(accent)
        painter.drawPolygon(
            QPolygonF([center + direction * 18, center + side * 5, center - side * 5])
        )
        painter.setBrush(foreground)
        painter.drawEllipse(center, 2, 2)
        painter.setPen(QPen(foreground))
        font = QFont()
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        letter = center + direction * 28
        painter.drawText(QRectF(letter.x() - 8, letter.y() - 8, 16, 16), Qt.AlignCenter, "N")
    finally:
        painter.end()
    return pixmap


class PygfxCompassMixin:
    """Retain compass intent; the master HUD switch owns effective visibility."""

    def configure_compass(
        self, north_direction_xy=None, *, label="Geographic north at scene origin", reason=""
    ) -> bool:
        """Publish validated geographic orientation; each source starts disabled."""
        if north_direction_xy is not None:
            direction = np.asarray(north_direction_xy, dtype=float)
            if direction.shape != (2,) or not np.isfinite(direction).all():
                raise ValueError("Compass north must be a finite horizontal direction")
            length = float(np.linalg.norm(direction))
            if length < 1e-8:
                raise ValueError("Compass north direction must be nonzero")
            north_direction_xy = tuple(direction / length)
        self._clear_compass()
        self._compass_north = north_direction_xy
        self._compass_label = str(label)
        self._compass_reason = (
            "" if north_direction_xy is not None else str(reason or _UNAVAILABLE_REASON)
        )
        return True

    def get_compass_state(self) -> dict:
        available = getattr(self, "_compass_north", None) is not None
        return {
            "available": available,
            "enabled": available and self.get_compass_visible(),
            "label": getattr(self, "_compass_label", "Geographic north at scene origin"),
            "reason": "" if available else getattr(self, "_compass_reason", _UNAVAILABLE_REASON),
        }

    def set_compass_visible(self, visible: bool) -> bool:
        if visible and getattr(self, "_compass_north", None) is None:
            return False
        if bool(visible) == self.get_compass_visible():
            return True
        if not visible:
            self._retire_compass_overlay()
        else:
            self._compass_visible = True
            self._sync_compass()
        return True

    def get_compass_visible(self) -> bool:
        return bool(getattr(self, "_compass_visible", False))

    def _sync_compass(self) -> None:
        """Repaint only a changed orientation; no timer, scene objects or readback."""
        if not self.get_compass_visible():
            return
        if not self._viewport_hud_policy().enabled:
            self._clear_hud_overlay(_OVERLAY_ID)
            self._compass_signature = None
            return
        camera = getattr(self, "_camera", None)
        container = getattr(self, "_container", None)
        if camera is None or container is None:
            return
        heading = north_heading(camera.world.matrix, self._compass_north)
        if heading is None:
            self._clear_hud_overlay(_OVERLAY_ID)
            self._compass_signature = None
            return
        # Subpixel angular movement cannot improve this small glyph. Bounding
        # precision also avoids repaints caused by floating-point camera noise.
        heading = round(math.degrees(heading) * 4) / 4
        style = self._hud_style()
        ratio = float(container.devicePixelRatioF())
        signature = (heading, style, ratio)
        spec = getattr(self, "_hud_overlay_specs", {}).get(_OVERLAY_ID, {})
        if getattr(self, "_compass_signature", None) == signature and spec.get("visible"):
            return
        self._set_hud_overlay(
            _OVERLAY_ID, html="", visible=True, role="guide", corner="top_right", priority=5
        )
        label = getattr(self, "_hud_overlay_labels", {}).get(_OVERLAY_ID)
        if label is None:
            return
        label.setPixmap(_compass_pixmap(math.radians(heading), style, ratio))
        label.setToolTip(self._compass_label)
        self._compass_signature = signature
        if not spec.get("visible") or spec.get("style") != style:
            label.adjustSize()
            self._reposition_hud_overlays()

    def _clear_compass(self) -> None:
        """Forget the previous source's geographic mapping and optional widget."""
        self._retire_compass_overlay()
        self._compass_north = None
        self._compass_label = "Geographic north at scene origin"
        self._compass_reason = _UNAVAILABLE_REASON

    def _retire_compass_overlay(self) -> None:
        """Release the optional widget and pixmap while retaining scene availability."""
        self._compass_visible = False
        self._compass_signature = None
        specs = getattr(self, "_hud_overlay_specs", {})
        spec = specs.pop(_OVERLAY_ID, None)
        label = getattr(self, "_hud_overlay_labels", {}).pop(_OVERLAY_ID, None)
        if label is not None:
            label.hide()
            label.clear()
            label.deleteLater()
        if spec is not None or label is not None:
            self._reposition_hud_overlays()
            self._request_hud_background_redraw()
