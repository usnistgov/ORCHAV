"""Optional actor cameras sharing the main scene, with bounded GUI draw work.

These docks never load frames or copy scene geometry. The main renderer remains
the only scene producer; auxiliary cameras sample its latest complete state.
Text callouts and editing handles belong to the main viewport only.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...scene.geometry_helpers import resolve_node_label
from ...services.object_identity import make_node_geometry_name, make_target_entry_geometry_name
from ...widgets.auxiliary_dock import AuxiliaryDock
from .canvas import retire_canvas_readback

logger = logging.getLogger(__name__)

_REFRESH_POLICIES = {
    "main_priority": ("Main priority", 15.0, 0.15),
    "smooth": ("Smooth actors", 30.0, 0.30),
}
_INTERACTION_HZ = 30.0
_INTERACTION_BUDGET = 0.5


def target_choices(backend: Any) -> tuple[tuple[str, str], ...]:
    """List targets and TX/RX markers by stable renderer identity.

    Keep the existing target-view API and dock names for saved layouts.
    """
    choices = []
    for entry in getattr(backend.visualizer, "target_entries", ()):
        name = make_target_entry_geometry_name(entry)
        if name in backend._objects:
            choices.append((name, str(entry.get("name") or entry.get("target_name") or name)))
    state = getattr(backend.visualizer, "app_state", None)
    for kind in ("tx", "rx"):
        for index, _entry in enumerate(getattr(backend.visualizer, f"{kind}_entries", ())):
            name = make_node_geometry_name(kind, index, "marker")
            if name in backend._objects:
                label = resolve_node_label(
                    kind.upper(),
                    index,
                    getattr(state, f"{kind}_labels", ()),
                    label_mode=getattr(state, "node_label_mode", "role"),
                    device_names=getattr(state, f"{kind}_device_names", ()),
                )
                choices.append((name, f"{kind.upper()} · {label}"))
    return tuple(choices)


@contextmanager
def without_main_view_annotations(backend: Any, *, guides_only: bool = False):
    """Temporarily omit camera-dependent labels/handles from one synchronous draw."""
    objects = (
        []
        if guides_only
        else [obj for name, obj in backend._objects.items() if backend._kinds.get(name) == "text"]
    )
    guide_labels = getattr(backend, "_scene_guide_labels", None)
    if callable(guide_labels):
        objects.extend(guide_labels())
    gizmo = getattr(backend, "_transform_gizmo", None)
    if gizmo is not None and not guides_only:
        objects.append(gizmo)
    previous = [(obj, obj.visible) for obj in {id(obj): obj for obj in objects}.values()]
    try:
        for obj, _visible in previous:
            obj.visible = False
        yield
    finally:
        for obj, visible in previous:
            obj.visible = visible


class TargetViewDock(AuxiliaryDock):
    """One independent orbit camera following a named actor in the shared scene."""

    def __init__(self, manager: TargetViewManager, serial: int) -> None:
        super().__init__(f"Actor view {serial}", manager.backend.visualizer)
        self.manager = manager
        self.backend = manager.backend
        self.setObjectName(f"targetView{serial}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.closed = False
        self.dirty = True
        self._requested = False
        self._last_center: np.ndarray | None = None
        self._last_render_size = None
        self._drag_point = None
        self._choices = ()
        self._selected_target: str | None = None
        self._selection_initialized = False
        self.failed = False
        self.render_count = 0
        self.render_failures = 0
        self.render_submit_ms_total = 0.0
        content = QWidget(self)
        content.setMinimumSize(240, 180)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.target_combo = QComboBox(content)
        self.target_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.target_combo.setMinimumContentsLength(12)
        self.target_combo.setToolTip("Choose the actor this camera follows")
        row.addWidget(self.target_combo, 1)
        fit = QPushButton("Fit", content)
        fit.clicked.connect(self.refit)
        row.addWidget(fit)
        row.addWidget(self.create_placement_button())
        layout.addLayout(row)
        refresh_row = QHBoxLayout()
        refresh_row.addWidget(QLabel("Refresh", content))
        self.refresh_combo = QComboBox(content)
        for key, (label, _hz, _budget) in _REFRESH_POLICIES.items():
            self.refresh_combo.addItem(label, key)
        self.refresh_combo.setCurrentIndex(self.refresh_combo.findData(manager.refresh_policy))
        self.refresh_combo.setToolTip(
            "Applies to all actor views. Main priority reserves more time for the main view; "
            "Smooth actors gives actor cameras more rendering time. Actual refresh depends on scene cost."
        )
        self.refresh_combo.currentIndexChanged.connect(self._select_refresh_policy)
        refresh_row.addWidget(self.refresh_combo, 1)
        layout.addLayout(refresh_row)
        self.status = QLabel("Drag to orbit · wheel to zoom", content)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        # The manager owns pacing. Faster canvas polling avoids adding another
        # low-frequency wait after an auxiliary draw has already been approved.
        kwargs = dict(parent=content, update_mode="ondemand", min_fps=0, max_fps=60)
        method = getattr(self.backend, "_canvas_present_method", "")
        if method in {"screen", "bitmap"}:
            kwargs["present_method"] = method
        try:
            self.canvas = self.backend._WgpuWidget(**kwargs)
        except Exception:
            self.closed = True
            self.deleteLater()
            raise
        size_policy = self.canvas.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.canvas.setSizePolicy(size_policy)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.installEventFilter(self)
        layout.addWidget(self.canvas, 1)
        gfx = self.backend._gfx
        # Disable native scene picking: an auxiliary camera must not trigger
        # gizmo/object handlers registered on the shared scene.
        try:
            self.renderer = gfx.WgpuRenderer(
                self.canvas, pixel_scale=1, ppaa="fxaa", enable_events=False
            )
            self.camera = gfx.PerspectiveCamera(45, 1.3)
            self.camera.world.reference_up = (0, 0, 1)
            self.controller = gfx.OrbitController(self.camera)
        except Exception:
            # Construction has not registered this dock with the manager yet.
            # Retire native resources here rather than relying on scene close.
            self.closed = True
            self.canvas.removeEventFilter(self)
            try:
                retire_canvas_readback(self.canvas)
            finally:
                self.canvas.close()
                self.deleteLater()
            raise
        self.canvas.request_draw(self._draw)
        self.target_combo.currentIndexChanged.connect(self._select_target)
        self.refresh_targets()
        self.refit(interactive=False)

    def _select_refresh_policy(self, index: int) -> None:
        self.manager.set_refresh_policy(self.refresh_combo.itemData(index))

    def refresh_targets(self) -> None:
        choices = target_choices(self.backend)
        if choices == self._choices:
            return
        selected = self._selected_target
        self._choices = choices
        with QSignalBlocker(self.target_combo):
            self.target_combo.clear()
            for name, label in choices:
                self.target_combo.addItem(label, name)
            index = self.target_combo.findData(selected)
            if not self._selection_initialized and choices:
                index = 0
                self._selected_target = choices[0][0]
                self._selection_initialized = True
            self.target_combo.setCurrentIndex(index)
        if selected is not None and index < 0:
            self._last_center = None
            self.status.setText("Actor unavailable. Choose another actor.")

    def _select_target(self, _index: int) -> None:
        self._selected_target = self.target_combo.currentData()
        self._selection_initialized = True
        self.refit()

    def drawable(self) -> bool:
        """Hidden dock contents and minimized windows consume no draw budget."""
        return bool(
            not self.closed
            and not self.failed
            and self.isVisible()
            and self.canvas.isVisible()
            and not self.window().isMinimized()
        )

    def _target_bounds(self):
        obj = self.backend._objects.get(self.target_combo.currentData())
        if obj is None:
            return None
        bounds = obj.get_world_bounding_box()
        if bounds is None:
            return None
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            return None
        return bounds

    def refit(self, *_args, interactive: bool = True) -> None:
        if self.closed:
            return
        bounds = self._target_bounds()
        if bounds is None:
            return
        center = bounds.mean(axis=0)
        radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.5, 0.5)
        self.camera.show_object(
            (*center, radius), view_dir=(-0.8, 1.0, -0.55), up=(0, 0, 1), scale=1.5
        )
        self.controller.target = center
        self._last_center = center
        self.status.setText("Drag to orbit · wheel to zoom")
        self.failed = False
        self.dirty = True
        if interactive:
            self.manager.request_interaction(self)
        else:
            self.manager.schedule()

    def _follow_target(self) -> bool:
        bounds = self._target_bounds()
        if bounds is None:
            return False
        center = bounds.mean(axis=0)
        if self._last_center is not None:
            self.camera.local.position = (
                np.asarray(self.camera.local.position) + center - self._last_center
            )
        self.controller.target = center
        self._last_center = center
        return True

    def _draw(self) -> None:
        self._requested = False
        if not self.drawable() or not self.backend._initialized:
            self.manager.schedule()
            return
        if self.backend._frame_update_paused:
            return
        get_size = getattr(self.canvas, "get_physical_size", None)
        size = tuple(get_size()) if get_size else (self.canvas.width(), self.canvas.height())
        # rendercanvas can request a size-change draw independently of our
        # dirty flag (including display pixel-ratio changes).
        if size != self._last_render_size:
            self.dirty = True
        if not self.dirty:
            self.manager.schedule()
            return
        if time.perf_counter() < self.manager.next_draw_at:
            self.manager.schedule()
            return
        self.refresh_targets()
        if self._last_center is None:
            self.refit(interactive=False)
        if not self._follow_target():
            self.dirty = False
            self.status.setText("Actor unavailable. Choose another actor.")
            self.manager.schedule()
            return
        self.dirty = False
        start = time.perf_counter()
        succeeded = False
        try:
            with without_main_view_annotations(self.backend):
                self.renderer.render(self.backend._scene, self.camera)
            self.render_count += 1
            self._last_render_size = size
            succeeded = True
            self.render_submit_ms_total += (time.perf_counter() - start) * 1000
        except Exception:
            logger.exception("Actor view rendering failed")
            self.failed = True
            self.render_failures += 1
            self.status.setText("View unavailable; close and reopen it.")
        finally:
            self.manager.record_draw(time.perf_counter() - start, succeeded=succeeded)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if watched is self.canvas and not self.closed:
            kind = event.type()
            rect = (0, 0, max(self.canvas.width(), 1), max(self.canvas.height(), 1))
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_point = event.position()
            elif kind == QEvent.MouseButtonRelease:
                self._drag_point = None
            elif kind == QEvent.MouseMove and self._drag_point is not None:
                delta = event.position() - self._drag_point
                self._drag_point = event.position()
                self.controller.rotate((delta.x() * 0.005, delta.y() * 0.005), rect)
                self.manager.request_interaction(self)
            elif kind == QEvent.Wheel:
                self.controller.zoom(float(np.clip(event.angleDelta().y() / 1200, -0.5, 0.5)), rect)
                self.manager.request_interaction(self)
            elif kind == QEvent.Resize:
                self.dirty = True
                self.manager.schedule()
            elif kind in {QEvent.Hide, QEvent.Show}:
                self.dirty = True
                self.manager.schedule()
        return super().eventFilter(watched, event)

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        self.dirty = True
        self.manager.schedule()

    def hideEvent(self, event):  # noqa: N802 - Qt API
        super().hideEvent(event)
        self.manager.schedule()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if not self.closed:
            self.closed = True
            self.manager.remove(self)
            self.canvas.removeEventFilter(self)
            retire_canvas_readback(self.canvas)
            self.canvas.close()
            self.controller.remove_camera(self.camera)
            self.renderer = None
        super().closeEvent(event)


class TargetViewManager:
    """Coalesce latest-scene draws, sharing one auxiliary submission budget."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.views: list[TargetViewDock] = []
        self._disposed = False
        self._serial = 0
        self.next_draw_at = 0.0
        self.refresh_policy = "main_priority"
        self._last_draw_end = 0.0
        self._last_draw_duration = 0.0
        self._synchronous_draw = False
        self._preferred_view: TargetViewDock | None = None
        self._last_dispatched: TargetViewDock | None = None
        self.render_count = 0
        self.render_failures = 0
        self.render_submit_ms_total = 0.0
        self.timer = QTimer(backend.visualizer)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._dispatch)

    def add(self) -> TargetViewDock | None:
        if not self.backend._initialized or not target_choices(self.backend):
            return None
        self._serial += 1
        view = TargetViewDock(self, self._serial)
        self.views.append(view)
        window = self.backend.visualizer
        previous = self.views[-2] if len(self.views) > 1 else None
        window.addDockWidget(Qt.RightDockWidgetArea, view)
        if previous is not None:
            window.splitDockWidget(previous, view, Qt.Vertical)
        window.resizeDocks([view], [350], Qt.Horizontal)
        view.show()
        docked = [item for item in self.views if not item.isFloating()]
        window.resizeDocks(
            docked, [max(180, window.height() // len(docked))] * len(docked), Qt.Vertical
        )
        self.schedule()
        return view

    def scene_presented(self) -> None:
        for view in self.views:
            view.dirty = True
        self.schedule()

    @staticmethod
    def _cooldown(duration: float, hz: float, budget: float) -> float:
        """Measure cadence from render start, retaining a submission duty ceiling."""
        return max(0.0, 1 / hz - duration, duration * (1 - budget) / budget)

    def set_refresh_policy(self, policy: str) -> None:
        """Apply one shared rendering allowance and mirror it in every dock."""
        if policy not in _REFRESH_POLICIES:
            raise ValueError(f"Unknown target-view refresh policy: {policy}")
        self.refresh_policy = policy
        _label, hz, budget = _REFRESH_POLICIES[policy]
        self.next_draw_at = self._last_draw_end + self._cooldown(
            self._last_draw_duration, hz, budget
        )
        for view in self.views:
            with QSignalBlocker(view.refresh_combo):
                view.refresh_combo.setCurrentIndex(view.refresh_combo.findData(policy))
        self.timer.stop()
        self.schedule()

    def request_interaction(self, view: TargetViewDock) -> None:
        """Prioritize one latest camera state without bypassing bounded GUI work."""
        if view not in self.views or not view.drawable():
            return
        view.dirty = True
        self._preferred_view = view
        ready = self._last_draw_end + self._cooldown(
            self._last_draw_duration, _INTERACTION_HZ, _INTERACTION_BUDGET
        )
        self.next_draw_at = min(self.next_draw_at, max(time.perf_counter(), ready))
        self.schedule()

    def schedule(self) -> None:
        if self._disposed:
            return
        if self._synchronous_draw:
            # force_draw also performs bitmap readback/presentation. Its full
            # cost must be known before another auxiliary draw is scheduled.
            self.timer.stop()
            return
        pending = [view for view in self.views if view.drawable() and view.dirty]
        if not pending or self.backend._frame_update_paused:
            self.timer.stop()
            return
        if any(view._requested and view.drawable() for view in self.views):
            self.timer.stop()
            return
        delay = max(0, int((self.next_draw_at - time.perf_counter()) * 1000) + 1)
        if not self.timer.isActive() or delay < self.timer.remainingTime():
            self.timer.start(delay)

    def _dispatch(self) -> None:
        if self._disposed or self._synchronous_draw or self.backend._frame_update_paused:
            return
        if time.perf_counter() < self.next_draw_at:
            self.schedule()
            return
        if any(view._requested and view.drawable() for view in self.views):
            return
        pending = [view for view in self.views if view.drawable() and view.dirty]
        if not pending:
            return
        view = pending[0]
        preferred = self._preferred_view
        if preferred in pending and (preferred is not self._last_dispatched or len(pending) == 1):
            view = preferred
        interactive = view is preferred
        self._preferred_view = None
        self._last_dispatched = view
        self.views.remove(view)
        self.views.append(view)
        view._requested = True
        if self.refresh_policy != "smooth" and not interactive:
            view.canvas.request_draw()
            return
        # Use the same public synchronous path as main-view frame playback.
        # This bypasses another canvas scheduling wait only for opted-in smooth
        # views or direct input; idle/main-priority draws remain asynchronous.
        start = time.perf_counter()
        self._synchronous_draw = True
        try:
            view.canvas.force_draw()
        except Exception:
            logger.exception("Actor view presentation failed")
            if not view.failed:
                view.render_failures += 1
                self.render_failures += 1
            view.failed = True
            view.status.setText("View unavailable; close and reopen it.")
        finally:
            view._requested = False
            self._synchronous_draw = False
            self._set_cooldown(time.perf_counter() - start)
            self.schedule()

    def _set_cooldown(self, duration: float) -> None:
        self._last_draw_end = time.perf_counter()
        self._last_draw_duration = duration
        _label, hz, budget = _REFRESH_POLICIES[self.refresh_policy]
        self.next_draw_at = self._last_draw_end + self._cooldown(duration, hz, budget)

    def record_draw(self, duration: float, *, succeeded: bool) -> None:
        self.render_count += int(succeeded)
        self.render_failures += int(not succeeded)
        self.render_submit_ms_total += duration * 1000
        self._set_cooldown(duration)
        self.schedule()

    def remove(self, view: TargetViewDock) -> None:
        if view in self.views:
            self.views.remove(view)
        if self._preferred_view is view:
            self._preferred_view = None
        if self._last_dispatched is view:
            self._last_dispatched = None
        self.schedule()

    def close(self) -> None:
        self.timer.stop()
        for view in tuple(self.views):
            view.close()
        self.views.clear()
        self.timer.stop()

    def dispose(self) -> None:
        """Release the callback retaining this renderer across scenario replacement."""
        if self._disposed:
            return
        self.close()
        self._disposed = True
        self.timer.timeout.disconnect(self._dispatch)
        self.timer.deleteLater()
