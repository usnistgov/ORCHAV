"""Shared metric top-down view for actor edits and street placement."""

from __future__ import annotations

import math
import threading

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImageReader, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..scene.street_plan import (
    BACKGROUND,
    GRID_COLOR,
    STREET_COLOR,
    grid_spacing,
    load_street_polylines,
)


class MetricView(QGraphicsView):
    """A metric XY view: positive Y points up, with a scale-aware grid."""

    manual_navigation = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor(BACKGROUND))
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setToolTip(
            "Scroll to zoom at the pointer; drag to pan. Use Fit trajectory to restore the full path."
        )

    def zoom_by(self, factor, anchor=None):
        """Zoom around the pointer, or the viewport centre for button actions."""
        anchor = anchor if anchor is not None else self.viewport().rect().center()
        before = self.mapToScene(anchor)
        scale = max(1e-6, min(1e4, self.transform().m11() * factor))
        factor = scale / self.transform().m11()
        self.manual_navigation.emit()
        self.scale(factor, factor)
        after = self.mapToScene(anchor)
        self.translate(after.x() - before.x(), after.y() - before.y())

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        if not delta:
            delta = event.pixelDelta().y() / 60
        if delta:
            self.zoom_by(1.2 ** max(-4, min(4, delta)), event.position().toPoint())
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.manual_navigation.emit()
        super().mouseMoveEvent(event)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        spacing = grid_spacing(rect.width(), rect.height())
        pen = QPen(QColor(GRID_COLOR))
        pen.setCosmetic(True)
        painter.setPen(pen)
        for x in np.arange(math.floor(rect.left() / spacing) * spacing, rect.right(), spacing):
            painter.drawLine(QPointF(float(x), rect.top()), QPointF(float(x), rect.bottom()))
        for y in np.arange(math.floor(rect.top() / spacing) * spacing, rect.bottom(), spacing):
            painter.drawLine(QPointF(rect.left(), float(y)), QPointF(rect.right(), float(y)))


class TrajectoryPreview(QWidget):
    """Render exactly the paths and current poses supplied to the 3D preview."""

    streets_ready = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("Top view · X / Y (m)")
        layout.addWidget(self.title)
        self.view = MetricView(self)
        self.canvas = QGraphicsScene(self.view)
        self.view.setScene(self.canvas)
        self.view.setMinimumHeight(140)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(3)
        self.zoom_in_button = QPushButton("+")
        self.zoom_out_button = QPushButton("−")
        for button, label in ((self.zoom_in_button, "Zoom in"), (self.zoom_out_button, "Zoom out")):
            button.setFixedWidth(28)
            button.setToolTip(label)
            button.setAccessibleName(label)
        self.zoom_in_button.clicked.connect(lambda: self.view.zoom_by(1.2))
        self.zoom_out_button.clicked.connect(lambda: self.view.zoom_by(1 / 1.2))
        self.fit_button = QPushButton("Fit trajectory")
        self.fit_button.setToolTip(
            "Fit committed and proposed trajectories, including their start and end points."
        )
        self.fit_button.clicked.connect(self.fit_trajectory)
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.zoom_in_button)
        zoom_row.addWidget(self.fit_button, 1)
        layout.addLayout(zoom_row)
        self.auto_fit = QCheckBox("Auto fit")
        self.auto_fit.setChecked(True)
        self.auto_fit.setToolTip(
            "Keep the trajectories in view when selecting an actor, editing, or resizing. Manual zoom and pan turn this off."
        )
        self.auto_fit.toggled.connect(lambda enabled: self.fit_trajectory() if enabled else None)
        self.view.manual_navigation.connect(lambda: self.auto_fit.setChecked(False))
        layout.addWidget(self.auto_fit)
        layout.addWidget(self.view, 1)
        self.height_label = QLabel("Height Z: —")
        self.height_label.setWordWrap(True)
        layout.addWidget(self.height_label)
        self.legend = QLabel("Blue: committed · Cyan: proposed\nGreen: start · Red: end")
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self._context_key = None
        self._background = []
        self._paths = []
        self._bounds = QRectF(-10, -10, 20, 20)
        self._map_credit = ""
        self.committed = ()
        self.proposed = ()
        self.streets_ready.connect(self._show_streets)

    @property
    def context_key(self):
        return self._context_key

    def load_streets(self, path):
        """Prepare graph geometry off the GUI thread; discard stale contexts."""
        if path is None:
            return
        key = self._context_key
        self.title.setText("Top view · loading streets…")

        def prepare():
            try:
                result = load_street_polylines(path)
            except (OSError, ValueError, RuntimeError) as exc:
                result = str(exc)
            try:
                self.streets_ready.emit(key, result)
            except RuntimeError:
                pass  # The workspace closed before the graph finished.

        threading.Thread(target=prepare, name="orchav-street-preview", daemon=True).start()

    def _show_streets(self, key, result):
        if key != self._context_key:
            return
        if isinstance(result, str):
            self.title.setText("Street preview unavailable · metric grid")
            self.title.setToolTip(result)
            return
        self._add_streets(result)
        self.title.setText("Top view · X / Y (m)" + self._map_credit)

    def _add_streets(self, streets):
        path = QPainterPath()
        for points in streets:
            if not len(points):
                continue
            path.moveTo(float(points[0][0]), -float(points[0][1]))
            for x, y, *_ in points[1:]:
                path.lineTo(float(x), -float(y))
        if not path.isEmpty():
            pen = QPen(QColor(STREET_COLOR))
            pen.setCosmetic(True)
            item = self.canvas.addPath(path, pen)
            item.setZValue(-1)
            self._background.append(item)

    def set_context(self, assets=(), streets=(), *, key=None):
        if key is not None and key == self._context_key:
            return
        self._context_key = key
        for item in self._background:
            self.canvas.removeItem(item)
        self._background.clear()
        self._map_credit = ""
        for asset in assets:
            if not asset.attribution or not asset.material.texture_path:
                continue
            # Recheck the format at display time: the file may have changed
            # since read_map validated it. Bypass QPixmap's filename cache and
            # never fall back to Qt's SVG or other content-selected decoders.
            reader = QImageReader(asset.material.texture_path, b"png")
            reader.setAutoDetectImageFormat(False)
            reader.setDecideFormatFromContent(False)
            image = QPixmap.fromImage(reader.read())
            uv = np.asarray(asset.mesh.triangle_uvs)
            vertices = np.asarray(asset.mesh.vertices)
            if image.isNull() or uv.shape != (len(vertices), 2):
                continue
            pixels = uv * (image.width(), image.height())
            matrix = np.linalg.lstsq(
                np.column_stack((pixels, np.ones(len(pixels)))),
                vertices[:, :2] * (1, -1),
                rcond=None,
            )[0]
            item = self.canvas.addPixmap(image)
            item.setTransform(QTransform(*matrix[0], *matrix[1], *matrix[2]))
            item.setZValue(-2)
            item.setOpacity(0.65)
            self._background.append(item)
            self._map_credit = " · © OpenStreetMap contributors"
        self._add_streets(streets)

    def _path(self, points, colour, width, *, dashed=False):
        if not len(points):
            return None
        path = QPainterPath()
        path.moveTo(float(points[0][0]), -float(points[0][1]))
        # Rendering may decimate paths; their source samples stay exact.
        stride = max(1, len(points) // 2000)
        for x, y, *_ in (*points[::stride], points[-1]):
            path.lineTo(float(x), -float(y))
        pen = QPen(QColor(colour))
        pen.setWidthF(width)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        return self.canvas.addPath(path, pen)

    def set_paths(
        self,
        committed=(),
        proposed=(),
        *,
        current=None,
        proposed_current=None,
        last_valid=False,
        routes=(),
    ):
        self.committed, self.proposed = tuple(committed), tuple(proposed)
        for item in self._paths:
            self.canvas.removeItem(item)
        self._paths.clear()
        paths = [(self.committed, "#4499ff", False), (self.proposed, "#32e4d0", True)]
        paths.extend((tuple(route), "#32e4d0", True) for route in routes)
        for points, colour, dashed in paths:
            item = self._path(points, colour, 3, dashed=dashed)
            if item is not None:
                self._paths.append(item)
        # Fit the trajectories themselves, independently of a city-sized map.
        xy = [(float(p[0]), -float(p[1])) for points, _, _ in paths for p in points]
        if not xy:
            xy = [(float(p[0]), -float(p[1])) for p in (current, proposed_current) if p is not None]
        if xy:
            points = np.asarray(xy)
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2
            extent = np.maximum(high - low, 2)
            bounds = QRectF(*(center - extent / 2), *extent)
        else:
            bounds = self.canvas.itemsBoundingRect()
        radius = max(bounds.width(), bounds.height(), 10) / 90
        for points, _, _ in paths:
            if points:
                for p, colour in ((points[0], "#35d589"), (points[-1], "#ff6464")):
                    self._paths.append(
                        self.canvas.addEllipse(
                            p[0] - radius,
                            -p[1] - radius,
                            radius * 2,
                            radius * 2,
                            QPen(Qt.NoPen),
                            QColor(colour),
                        )
                    )
        for p, colour in ((current, "#ffffff"), (proposed_current, "#32e4d0")):
            if p is not None:
                pen = QPen(QColor(colour))
                pen.setCosmetic(True)
                self._paths.append(
                    self.canvas.addEllipse(
                        p[0] - radius,
                        -p[1] - radius,
                        radius * 2,
                        radius * 2,
                        pen,
                        QColor(colour),
                    )
                )
        self._bounds = (
            bounds.adjusted(-radius * 5, -radius * 5, radius * 5, radius * 5)
            if not bounds.isNull()
            else QRectF(-10, -10, 20, 20)
        )
        center = self.view.mapToScene(self.view.viewport().rect().center())
        scene_bounds = self.canvas.itemsBoundingRect().united(self._bounds)
        padding = max(scene_bounds.width(), scene_bounds.height(), 100) * 10
        self.view.setSceneRect(scene_bounds.adjusted(-padding, -padding, padding, padding))
        if self.auto_fit.isChecked():
            self.fit_trajectory()
        else:
            self.view.centerOn(center)
        detail = f"Height Z: {current[2]:.3f} m" if current is not None else "Height Z: —"
        if proposed_current is not None:
            detail += f" · proposed {proposed_current[2]:.3f} m"
        self.height_label.setText(detail)
        self.title.setText(
            ("Last valid preview" if last_valid else "Top view · X / Y (m)") + self._map_credit
        )
        self.title.setWordWrap(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit.isChecked():
            self.fit_trajectory()

    def fit_trajectory(self):
        """Frame both paths once without changing the automatic-fit preference."""
        self.view.fitInView(self._bounds, Qt.KeepAspectRatio)
