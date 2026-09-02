from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QRect, QSize

from visualizer.src.metrics.mpc_stats import MPCStatsComputer
from visualizer.src.services import metrics_service
from visualizer.src.services.metrics_service import MetricsService
from visualizer.src.state import create_initial_state


class DummySignal:
    def __init__(self):
        self._callbacks: list = []

    def connect(self, callback):
        self._callbacks.append(callback)


class DummyWindow:
    def __init__(self, parent, update_hz=0, frame_stats_provider=None):
        self.destroyed = DummySignal()
        self.enqueued: list = []
        self.frame_stats_provider = frame_stats_provider
        self._visible = True
        self.updates_paused = False
        self.closed = False
        self.geometry = None
        self.move_calls = []
        self.shown = False
        self.raised = False
        self.activated = False

    def move(self, *args):
        self.move_calls.append(args)

    @staticmethod
    def size():
        return QSize(1040, 760)

    def setGeometry(self, geometry):
        self.geometry = geometry

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def activateWindow(self):
        self.activated = True

    def isVisible(self):
        return self._visible

    def enqueue(self, view_model, frame_stats, context=None):
        self.enqueued.append((view_model, frame_stats, context or {}))

    def close(self):
        self.closed = True


class DummyMpcCore:
    def __init__(self):
        self._last_reflection_order_counts = None
        self.current_delay_range = None
        self.current_path_loss_range = None
        self.stats_calls = 0

    def stats(self, payload):
        self.stats_calls += 1
        return {"stats_payload": payload}


class DummyVisualizer:
    def __init__(self):
        self.metrics_window = None
        self.mpc_core = DummyMpcCore()
        self.available_geometry = QRect(0, 0, 1920, 1080)

    def geometry(self):
        return SimpleNamespace(left=0, width=1, top=0)

    def screen(self):
        return SimpleNamespace(availableGeometry=lambda: self.available_geometry)


def _setup_service(monkeypatch):
    monkeypatch.setattr(metrics_service, "MetricsWindow", DummyWindow)
    monkeypatch.setattr(metrics_service, "METRICS_WINDOW_AVAILABLE", True)

    viz = DummyVisualizer()
    service = MetricsService(viz)
    return service, viz


def test_metrics_window_toggle(monkeypatch):
    service, viz = _setup_service(monkeypatch)

    service.toggle_window()
    assert isinstance(service.window, DummyWindow)
    assert viz.metrics_window is service.window
    assert service.window.frame_stats_provider == service._compute_frame_stats
    assert service.window.move_calls == []
    assert service.window.geometry == QRect(440, 160, 1040, 760)
    assert service.window.shown is True
    assert service.window.raised is True
    assert service.window.activated is True

    service.toggle_window()
    assert service.window is None
    assert viz.metrics_window is None


def test_metrics_window_fits_secondary_monitor_available_area(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    viz.available_geometry = QRect(-1280, 0, 1280, 720)

    service.toggle_window()

    assert service.window is not None
    assert service.window.geometry == QRect(-1160, 16, 1040, 688)


def test_metrics_update_dispatches(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    service.window = DummyWindow(None)

    view_model = SimpleNamespace(
        canonical_data=SimpleNamespace(),
        mpc_points=[1, 2, 3],
    )

    service.update_metrics(view_model)

    assert len(service.window.enqueued) == 1
    enqueued_vm, stats, context = service.window.enqueued[0]
    assert enqueued_vm is view_model
    assert stats is None
    assert viz.mpc_core.stats_calls == 0
    assert context["selected_tx"] == "all"
    assert context["selected_rx"] == "all"

    computed = service._compute_frame_stats(view_model)
    assert "stats_payload" in computed
    assert viz.mpc_core.stats_calls == 1


def test_metrics_update_without_window_is_noop(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    service.window = None

    view_model = SimpleNamespace(
        canonical_data=SimpleNamespace(),
        mpc_points=[1, 2, 3],
    )

    service.update_metrics(view_model)

    assert viz.mpc_core.stats_calls == 0


def test_metrics_pause_skips_advanced_statistics_and_enqueue(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    service.window = DummyWindow(None)
    service.window.updates_paused = True
    view_model = SimpleNamespace(canonical_data=SimpleNamespace())

    service.update_metrics(view_model)

    assert viz.mpc_core.stats_calls == 0
    assert service.window.enqueued == []


def test_empty_mpc_allow_lists_are_reported_as_active_filters() -> None:
    viz = SimpleNamespace(
        app_state=create_initial_state(
            mpc_allowed_orders=frozenset(),
            mpc_allowed_types=frozenset(),
        )
    )
    service = MetricsService(viz)

    assert service._active_filter_labels()[:2] == ["orders", "types"]


def test_3d_render_topk_is_not_reported_as_a_metrics_filter() -> None:
    viz = SimpleNamespace(
        app_state=SimpleNamespace(
            topk_render_enabled=True,
            topk_render_max_paths=5000,
        )
    )
    service = MetricsService(viz)

    assert service._active_filter_labels() == []


def test_metrics_provider_uses_canonical_view_model(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    service.window = DummyWindow(None)

    view_model = SimpleNamespace(
        canonical_data=SimpleNamespace(path_delays=np.array([1.0, 2.0, 3.0])),
        path_mask=[True, False, True],
        mpc_points=[1, 2, 3, 4, 5],
    )

    service.update_metrics(view_model)

    assert len(service.window.enqueued) == 1
    _, stats, context = service.window.enqueued[0]
    assert stats is None
    assert viz.mpc_core.stats_calls == 0
    assert context["visible_paths"] is None
    assert context["total_paths"] == 3

    stats = service._compute_frame_stats(view_model)
    assert viz.mpc_core.stats_calls == 1
    assert stats["stats_payload"]["canonical_data"] is view_model.canonical_data
    assert stats["stats_payload"]["metrics_visible"] is True
    assert stats["stats_payload"]["path_mask"] == [True, False, True]


def test_metrics_provider_computes_corrected_power_delay_profile(monkeypatch):
    service, viz = _setup_service(monkeypatch)
    service.window = DummyWindow(None)

    class CanonicalStatsCore:
        @staticmethod
        def stats(payload):
            return MPCStatsComputer().compute_frame_stats(
                payload["canonical_data"],
                include_advanced=payload["metrics_visible"],
                path_mask=payload["path_mask"],
            )

    viz.mpc_core = CanonicalStatsCore()
    canonical = SimpleNamespace(
        path_orders=np.array([0, 1, 2]),
        path_delays=np.array([10.0, 20.0, 30.0]),
        path_losses=np.array([80.0, 90.0, 100.0]),
        path_delay_is_estimated=np.zeros(3, dtype=bool),
        path_loss_is_estimated=np.zeros(3, dtype=bool),
    )
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.array([True, True, True]),
    )

    service.update_metrics(view_model)

    _, queued_stats, _ = service.window.enqueued[0]
    assert queued_stats is None
    stats = service._compute_frame_stats(view_model)
    assert not hasattr(stats, "power_delay_profile")
    assert stats.binned_power_delay_profile is not None
