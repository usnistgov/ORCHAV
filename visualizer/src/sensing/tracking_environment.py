"""Optional world-XY context from the Visualizer's loaded static geometry.

Only immutable neutral mesh snapshots enter the worker. No scene is reopened,
no renderer is read back, and no sensing values or camera state are changed.
The single retained projection is invalidated by source, payload or transform
replacement. Actor meshes are deliberately outside this static context layer.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QPainterPath
from PySide6.QtWidgets import QCheckBox, QGraphicsPathItem, QLabel, QSpinBox

from ..model import RenderObjectState
from ..types.render_payloads import MeshPayload

# Refuse an oversized projection explicitly rather than silently dropping buildings.
MAX_PROJECTION_TRIANGLES = 500_000
MAX_PROJECTION_SEGMENTS = 500_000
_PARALLEL_NORMAL_TOLERANCE = 1e-6
_CONTEXT_NOTE = (
    "Top-down scene edges in world XY; roofs and levels overlap. Not a floor plan. "
    "Uses the loaded static scene, independently of its 3D visibility settings. "
    "Pan/zoom normally; the background does not change automatic track fitting."
)


def project_scene_edges(
    meshes: tuple[tuple[MeshPayload, np.ndarray], ...], cancelled: Event
) -> np.ndarray:
    """Project boundary/crease edges, suppressing internal coplanar triangulation.

    Scene I/O already bakes authored XML transforms into vertices. Apply only
    the render state's remaining world transform, exactly once. Exact duplicate
    vertices are welded for adjacency; distinct surfaces are not approximated.
    """
    if sum(len(mesh.triangles) for mesh, _ in meshes) > MAX_PROJECTION_TRIANGLES:
        raise ValueError("Scene exceeds the 500,000-triangle background limit.")
    projected: list[np.ndarray] = []
    count = 0
    for mesh, transform in meshes:
        if cancelled.is_set():
            return np.empty((0, 2, 2))
        vertices = np.asarray(mesh.vertices, dtype=float)
        triangles = np.asarray(mesh.triangles)
        if not len(vertices) or not len(triangles):
            continue
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(transform)):
            raise ValueError("Scene contains nonfinite coordinates.")
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        triangles = inverse[triangles]
        faces = vertices[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0
        triangles, normals = triangles[valid], normals[valid] / lengths[valid, None]
        edges = np.sort(
            np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
            axis=1,
        )
        unique, first, inverse, counts = np.unique(
            edges, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        last = np.zeros(len(unique), dtype=np.intp)
        np.maximum.at(last, inverse, np.arange(len(edges)))
        edge_normals = np.tile(normals, (3, 1))
        parallel = np.abs(np.sum(edge_normals[first] * edge_normals[last], axis=1))
        keep = (counts != 2) | (parallel < 1 - _PARALLEL_NORMAL_TOLERANCE)
        xy = vertices[unique[keep], :2]
        xy = xy[np.any(xy[:, 0] != xy[:, 1], axis=1)]
        count += len(xy)
        if count > MAX_PROJECTION_SEGMENTS:
            raise ValueError("Scene exceeds the 500,000-edge background limit.")
        projected.append(xy)
    if cancelled.is_set() or not projected:
        return np.empty((0, 2, 2))
    result = np.concatenate(projected)
    result.setflags(write=False)
    return result


class TrackingEnvironment(QObject):
    """Own a lazy, cancellable projection and its noninteractive plot layer."""

    def __init__(self, plot: pg.PlotWidget, parent: QObject) -> None:
        super().__init__(parent)
        self.toggle = QCheckBox("Scene background")
        self.toggle.setToolTip(_CONTEXT_NOTE)
        self.toggle.setEnabled(False)
        self.opacity = QSpinBox()
        self.opacity.setRange(5, 100)
        self.opacity.setValue(35)
        self.opacity.setEnabled(False)
        self.opacity.setSuffix(" %")
        self.opacity.setToolTip("Opacity of scene context beneath the tracks.")
        self.note = QLabel("Scene background unavailable: no loaded environment.")
        self.note.setWordWrap(True)
        self.item = QGraphicsPathItem()
        self.item.setZValue(-30)
        self.item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.item.setAcceptHoverEvents(False)
        plot.addItem(self.item, ignoreBounds=True)
        self.item.hide()
        self._key: tuple | None = None
        self._meshes: tuple[tuple[MeshPayload, np.ndarray], ...] = ()
        self._future: Future | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._cancelled = Event()
        self._ready = False
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll)
        self.toggle.toggled.connect(self._update_visibility)
        self.opacity.valueChanged.connect(self._update_visibility)

    def set_scene(self, identity: Any, entries: list[dict]) -> None:
        """Bind only loaded static mesh entries, with their immutable world pose."""
        meshes = []
        for entry in entries:
            state = entry.get("mesh")
            if isinstance(state, RenderObjectState) and isinstance(state.payload, MeshPayload):
                meshes.append((state.payload, state.world_transform.matrix))
            elif isinstance(state, MeshPayload):
                meshes.append((state, np.eye(4)))
        key = (identity, tuple((id(p), t.tobytes()) for p, t in meshes))
        if self._closed or key == self._key:
            return
        self.clear()
        self._key, self._meshes = key, tuple(meshes)
        self.toggle.setEnabled(bool(meshes))
        if meshes:
            self.note.setText(_CONTEXT_NOTE)
            self.toggle.setToolTip(_CONTEXT_NOTE)
        self._update_visibility()

    def clear(self) -> None:
        """Invalidate visible data and detach any previous worker completion."""
        self._cancelled.set()
        if self._future is not None:
            self._future.cancel()
        self._future = None
        self._timer.stop()
        self._meshes, self._key, self._ready = (), None, False
        self.item.hide()
        self.item.setPath(QPainterPath())
        self.toggle.setEnabled(False)
        self.opacity.setEnabled(False)
        reason = "Scene background unavailable: no loaded environment."
        self.note.setText(reason)
        self.toggle.setToolTip(reason)

    def set_color(self, color: str) -> None:
        """Use a muted theme-compatible line, independent of scientific colors."""
        self.item.setPen(pg.mkPen(color, width=1))

    def _update_visibility(self, *_args: Any) -> None:
        requested = self.toggle.isChecked() and self.toggle.isEnabled() and not self._closed
        self.opacity.setEnabled(requested)
        self.item.setOpacity(self.opacity.value() / 100)
        self.item.setVisible(requested and self._ready)
        if requested and not self._ready and self._future is None:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-map")
            self._cancelled = Event()
            self._future = self._executor.submit(project_scene_edges, self._meshes, self._cancelled)
            self.note.setText("Preparing scene background… " + _CONTEXT_NOTE)
            self._timer.start()

    def _poll(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        self._timer.stop()
        try:
            segments = future.result()
            if not len(segments):
                raise ValueError("The loaded scene has no projected edges.")
            xy = segments.reshape(-1, 2)
            self.item.setPath(pg.arrayToQPath(xy[:, 0], xy[:, 1], connect="pairs"))
        except Exception as exc:  # Worker/Qt boundary: keep sensing usable after geometry failure.
            self.note.setText(f"Scene background unavailable: {exc}")
            self.toggle.setToolTip(self.note.text())
            self.toggle.setEnabled(False)
        else:
            self._ready = True
            self.note.setText(_CONTEXT_NOTE)
        self._update_visibility()

    def close(self) -> None:
        """Release workers without blocking Qt or accepting late results."""
        self._closed = True
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
