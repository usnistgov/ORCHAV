"""Summary UI and asynchronous lifetime contracts without a 3D renderer."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication

from visualizer.src.sensing.run_summary_data import RunSummaryData
from visualizer.src.sensing.run_summary_panel import RunSummaryPanel
from visualizer.src.sensing.run_summary_session import RunSummarySession

APP = QApplication.instance() or QApplication([])


def test_summary_has_all_figures_and_explicit_apply(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    assert not panel.plots  # Heavy plot objects are lazy.
    panel._ensure_plots()
    assert len(panel.plots) == 10
    old = panel.settings()
    panel.range_gate.setValue(9)
    assert panel.settings() == old
    panel.apply.click()
    assert panel.settings()[0] == 9
    panel.set_summary(RunSummaryData(("clock", "run", "pair")))
    assert "unavailable" in panel.cards.text().lower()
    assert not panel.tabs.isTabEnabled(2)
    panel.clear()
    assert all(not plot.listDataItems() for plot in panel.plots.values())


def test_identity_samples_and_coverage_render(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    data = RunSummaryData(
        ("clock", "run", "pair"),
        summary={
            "total_tp": 1,
            "total_fp": 0,
            "total_fn": 1,
            "recall": 0.5,
            "precision": 1.0,
            "id_switches": 0,
            "per_target": {"target": {"recall": 0.5, "matched_count": 1, "eligible_count": 2}},
        },
        acquisition_count=2,
        evaluated_count=2,
        tracking_time_s=np.array([0.0]),
        tracking_target_ids=("target",),
        tracking_track_ids=(7,),
    )
    panel.set_summary(data)
    assert panel.tabs.isTabEnabled(2)
    assert "50.0%" in panel.cards.text()
    assert "1 / 2" in panel.plots["coverage"].toolTip()
    previous = panel.plots["identity"].listDataItems()[0]
    panel.set_summary(data)
    assert panel.plots["identity"].listDataItems()[0] is previous
    panel.set_loading()
    assert panel.data is None
    assert not panel.tabs.isTabEnabled(2)


class _Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def shutdown(self, **kwargs):
        pass


def test_session_reuses_exact_key_invalidates_and_rejects_late_completion(monkeypatch):
    from visualizer.src.sensing import run_summary_session as module

    executor = _Executor()
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda **kwargs: executor)
    session = RunSummarySession(SimpleNamespace())
    context = ("c", "r", "p")
    settings = (2.0, 2.0, None)
    session.request(context, (1,), settings)
    old, _, args = executor.jobs[0]
    old.set_running_or_notify_cancel()
    old_cancel = args[-1]
    session.request(context, (1,), settings)
    assert len(executor.jobs) == 1
    session.request(context, (2,), settings)
    assert old_cancel.is_set()
    old.set_result(RunSummaryData(context, acquisition_count=99))
    assert not session.poll()
    new = executor.jobs[1][0]
    wanted = RunSummaryData(context, acquisition_count=2)
    new.set_result(wanted)
    assert session.poll() and session.data is wanted
    session.request(context, (2,), settings)
    assert len(executor.jobs) == 2
    session.close()
    session.request(context, (3,), settings)
    assert len(executor.jobs) == 2


def test_session_errors_are_visible_and_cancelled_totals_never_publish(monkeypatch):
    from visualizer.src.sensing import run_summary_session as module

    executor = _Executor()
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda **kwargs: executor)
    session = RunSummarySession(None)
    session.request(("c", "r", "p"), 1, (2, 2, None))
    executor.jobs[-1][0].set_exception(ValueError("unreadable recording"))
    assert session.poll()
    assert session.error == "unreadable recording"
    session.request(("c", "r", "p"), 2, (2, 2, None))
    executor.jobs[-1][0].set_result(RunSummaryData(("c", "r", "p"), cancelled=True))
    assert not session.poll()
    assert session.data is None
    session.close()


def test_reopened_summary_reuses_completed_data_but_rescores_changed_gates(monkeypatch):
    from visualizer.src.sensing import run_summary_session as module

    executor = _Executor()
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda **kwargs: executor)
    provider = SimpleNamespace()
    first = RunSummarySession(provider)
    context, settings = ("clock", "run", "pair"), (2.0, 2.0, None)
    first.request(context, (1,), settings)
    data = RunSummaryData(context, acquisition_count=16)
    executor.jobs[-1][0].set_result(data)
    assert first.poll()
    first.close()
    reopened = RunSummarySession(provider, previous=first)
    reopened.request(context, (1,), settings)
    assert reopened.data is data
    assert len(executor.jobs) == 1
    reopened.request(context, (1,), (1.0, 2.0, None))
    assert len(executor.jobs) == 2 and reopened.data is None
    reopened.close()
    other = RunSummarySession(SimpleNamespace(), previous=first)
    assert other.data is None
    other.close()


def test_dashboard_summary_selection_is_compact_and_seekable(qtbot, monkeypatch):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _frame
    from visualizer.src.sensing.dashboard_data import load_dashboard_request
    from visualizer.src.sensing.sensing_window import SensingWindow

    frame = _frame()
    selection = SimpleNamespace(
        display_frame=frame,
        bound_scene_frame_id=0,
        stale_for_current_scene=False,
        replay_event=None,
        storage_frame_id=0,
    )
    requests = []
    service = SimpleNamespace(
        dashboard_scene_frame=lambda step: frame,
        resolve_display_sensing_frame=lambda *args, **kwargs: (
            requests.append(kwargs) or selection
        ),
        dashboard_products=lambda *args: (_ for _ in ()).throw(
            AssertionError("Summary must not read dense products")
        ),
        dashboard_product_kinds=lambda step: frozenset(),
    )
    request = load_dashboard_request(service, 0, "summary", "primary", True, "auto")
    assert tuple(requests[0]["product_names"]) == ("config_resolved",)
    assert request.rp_result is None and request.rd_result is None
    window = SensingWindow()
    qtbot.addWidget(window)
    assert window.request_view_mode() == "summary"
    window.update_from_request(request)
    assert window.request_view_mode() == "summary"
    assert not window.wants_recording_history()
    assert window._last_request is request


def test_summary_worker_does_not_block_qt_and_close_discards_it(qtbot, monkeypatch):
    from threading import Event

    from PySide6.QtCore import QTimer

    started, release = Event(), Event()
    context = ("clock", "run", "pair")

    def slow_load(self, key, cancelled):
        started.set()
        assert release.wait(3)
        return RunSummaryData(context, acquisition_count=123)

    monkeypatch.setattr(RunSummarySession, "_load", slow_load)
    session = RunSummarySession(None)
    try:
        session.request(context, 1, (2, 2, None))
        qtbot.waitUntil(started.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        assert session.pending
        session.close()
        release.set()
        assert not session.poll()
        assert session.data is None
    finally:
        release.set()
        session.close()


def test_plot_seek_requires_unmodified_left_click_inside_view(qtbot):
    from PySide6.QtCore import QPointF, QRectF, Qt

    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    emitted = []
    panel.time_selected.connect(emitted.append)
    view = SimpleNamespace(
        sceneBoundingRect=lambda: QRectF(10, 10, 100, 100),
        mapSceneToView=lambda point: point,
    )
    plot = SimpleNamespace(getPlotItem=lambda: SimpleNamespace(vb=view))
    for button, modifiers, point in (
        (Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPointF(50, 50)),
        (Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPointF(50, 50)),
        (Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, QPointF(50, 50)),
        (Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(2, 2)),
    ):
        event = SimpleNamespace(
            button=lambda: button,
            modifiers=lambda: modifiers,
            double=lambda: False,
            scenePos=lambda: point,
        )
        panel._clicked(plot, event)
        assert not emitted
    event = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        modifiers=lambda: Qt.KeyboardModifier.NoModifier,
        double=lambda: False,
        scenePos=lambda: QPointF(50, 50),
    )
    panel._clicked(plot, event)
    assert emitted == [50.0]
