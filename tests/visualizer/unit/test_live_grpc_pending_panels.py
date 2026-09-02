"""Tests for live-gRPC pending-request panel integration."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter

from visualizer.src.io.frame_sources import LiveGrpcSource
from visualizer.src.io.grpc_provider import PendingFrameRequest
from visualizer.src.panels.data_source import performance_section
from visualizer.src.panels.data_source.connection_section import ConnectionStatusSection
from visualizer.src.panels.data_source.performance_section import PerformanceSection
from visualizer.src.panels.data_source.streaming_section import StreamingControlSection
from visualizer.src.panels.data_source.widgets import FrameTimelineWidget


class _Button:
    def __init__(self) -> None:
        self.enabled = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = str(text)


class _ProgressBar:
    def __init__(self) -> None:
        self.range = None
        self.value = None
        self.format = ""

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (int(minimum), int(maximum))

    def setValue(self, value: int) -> None:
        self.value = int(value)

    def setFormat(self, text: str) -> None:
        self.format = str(text)


class _Timeline:
    def __init__(self) -> None:
        self.frame_states = None
        self.current_frame = None

    def set_frame_states(self, frame_states: dict[int, str], current_frame: int) -> None:
        self.frame_states = dict(frame_states)
        self.current_frame = int(current_frame)


def test_streaming_timeline_uses_pending_snapshot_api():
    timeline = _Timeline()
    section = StreamingControlSection(
        SimpleNamespace(animation_step=2),
        {"frame_timeline": timeline},
        lambda: "",
    )
    provider = SimpleNamespace(
        get_pending_frame_requests=lambda: [
            PendingFrameRequest(frame_idx=2, request_timestamp=10.0)
        ],
        _get_frame_info=lambda: {"available_frames": [1, 3]},
    )

    section._update_frame_timeline(
        provider,
        {"available_frames": [4]},
        [{"frame_idx": 5, "success": False}],
    )

    assert timeline.current_frame == 2
    assert timeline.frame_states == {
        1: "computed",
        2: "requested",
        3: "computed",
        4: "buffered",
        5: "failed",
    }


def test_frame_timeline_paints_with_qt6_font_metrics(qapp):
    widget = FrameTimelineWidget()
    widget.resize(520, 160)
    widget.set_frame_states(
        {
            1: "computed",
            2: "buffered",
            3: "requested",
            4: "failed",
        },
        current_frame=3,
    )

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    try:
        widget.render(painter, QPoint(0, 0))
    finally:
        painter.end()

    assert not image.isNull()


def test_performance_progress_uses_pending_snapshot_api(monkeypatch):
    monkeypatch.setattr(performance_section.time, "time", lambda: 120.0)
    button = _Button()
    progress = _ProgressBar()
    label = _Label()
    section = PerformanceSection(
        SimpleNamespace(),
        {
            "cancel_request_btn": button,
            "frame_progress_bar": progress,
            "frame_progress_label": label,
        },
        lambda: "",
    )
    provider = SimpleNamespace(
        get_pending_frame_requests=lambda: [
            PendingFrameRequest(frame_idx=7, request_timestamp=105.0)
        ]
    )

    section._update_cancel_button_state(provider, [])
    section._update_frame_progress(provider)

    assert button.enabled is True
    assert progress.range == (0, 0)
    assert progress.value is None
    assert progress.format == "1 pending - oldest 15.0s"
    assert label.text == ("Waiting for generator responses; exact solver progress is unavailable")


def test_streaming_prefetch_changes_use_controller_and_provider_public_apis():
    calls = []
    provider = SimpleNamespace(
        set_prefetch_lookahead=lambda value: calls.append(("lookahead", value))
    )
    frame_source = LiveGrpcSource("grpc://unit-test")
    frame_source.provider = provider
    parent = SimpleNamespace(
        frame_source=frame_source,
        animation_controller=SimpleNamespace(
            sync_prefetch_settings=lambda: calls.append(("sync", None))
        ),
    )
    section = StreamingControlSection(parent, {}, lambda: "")

    section._on_prefetch_pause_changed(0)
    section._on_prefetch_lookahead_changed(12)

    assert calls == [("sync", None), ("lookahead", 12)]


def test_reset_live_frames_action_names_and_requests_full_reset(qapp):
    reasons = []
    frame_source = LiveGrpcSource("grpc://unit-test")
    frame_source.request_cache_flush = lambda reason: reasons.append(reason)
    widgets = {}
    section = StreamingControlSection(
        SimpleNamespace(frame_source=frame_source), widgets, lambda: ""
    )
    group = section._create_online_actions_group()

    assert widgets["online_clear_btn"].text() == "Reset Live Frames"
    assert "client and generator" in widgets["online_clear_btn"].toolTip()

    section._on_clear_buffer_clicked()

    assert reasons == ["UI Reset Live Frames"]
    group.deleteLater()


def test_cancel_request_uses_provider_cancel_api(monkeypatch):
    cancelled = []
    messages = []
    provider = SimpleNamespace(
        get_pending_frame_requests=lambda: [
            PendingFrameRequest(frame_idx=11, request_timestamp=100.0)
        ],
        cancel_pending_frame_request=lambda frame_idx: cancelled.append(frame_idx) or True,
    )
    frame_source = LiveGrpcSource("grpc://unit-test")
    frame_source.provider = provider
    section = PerformanceSection(
        SimpleNamespace(frame_source=frame_source),
        {},
        lambda: "",
    )
    monkeypatch.setattr(
        performance_section.QMessageBox,
        "information",
        lambda _parent, title, text: messages.append((title, text)),
    )
    monkeypatch.setattr(
        performance_section.QMessageBox,
        "warning",
        lambda _parent, title, text: messages.append((title, text)),
    )

    section._on_cancel_request_clicked()

    assert cancelled == [11]
    assert messages == [("Request Cancelled", "Cancelled request for frame 11")]


def test_live_panels_do_not_claim_client_identity_or_automatic_reconnect(qapp):
    assert qapp is not None
    connection_widgets = {}
    connection = ConnectionStatusSection(SimpleNamespace(), connection_widgets)
    connection._create_connection_info_group()
    connection._create_health_metrics_group()

    performance_widgets = {}
    performance = PerformanceSection(SimpleNamespace(), performance_widgets, lambda: "")
    performance._create_connection_diagnostics_group()

    assert "online_client_id" not in connection_widgets
    assert "online_reconnections" not in connection_widgets
    assert "retry_timeline" not in performance_widgets
    performance.cleanup()


def test_live_performance_graphs_are_not_created_until_expanded(qapp):
    widgets = {}
    performance = PerformanceSection(SimpleNamespace(), widgets, lambda: "")

    content = performance.create_content()

    assert performance._performance_graphs_initialized is False
    assert "perf_graph_frame_time" not in widgets
    assert "perf_graph_buffer" not in widgets
    content.deleteLater()
    performance.cleanup()
