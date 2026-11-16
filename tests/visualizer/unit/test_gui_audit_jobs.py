"""GUI regressions for optional analysis and asynchronous network updates."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget

from visualizer.src.io.frame_sources import LiveGrpcSource
from visualizer.src.panels.collapsible_section import CollapsibleSection
from visualizer.src.panels.data_source.raytracing_section import RaytracingControlSection
from visualizer.src.panels.statistics_panel import StatisticsPanel


@pytest.mark.parametrize("outcome", ["success", "failure", "source_changed"])
def test_parameter_wait_keeps_qt_responsive_and_ignores_retired_source(qtbot, outcome):
    entered, release = threading.Event(), threading.Event()

    def update(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if outcome == "failure":
            raise OSError("connection closed")
        return SimpleNamespace(success=True, message="Applied")

    source = LiveGrpcSource("unused")
    source.provider = SimpleNamespace(update_raytracing_params=update)
    refresh = Mock(return_value=True)
    owner = SimpleNamespace(
        frame_source=source,
        frame_refresh_service=SimpleNamespace(refresh_current_frame_after_data_change=refresh),
    )
    section = RaytracingControlSection(owner, {}, lambda: "")
    widget = section.create_content()
    qtbot.addWidget(widget)
    ticks = []
    timer = QTimer(widget)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        section._on_rt_apply_clicked()
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert not section.widgets["rt_apply_btn"].isEnabled()
        if outcome == "source_changed":
            owner.frame_source = object()
        release.set()
        qtbot.waitUntil(lambda: section.widgets["rt_apply_btn"].isEnabled())
        assert refresh.call_count == (1 if outcome == "success" else 0)
        if outcome == "failure":
            assert "connection closed" in section.widgets["rt_status_label"].text()
    finally:
        release.set()


def test_hidden_expanded_graphs_defer_render_until_their_tab_is_visible(qtbot):
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    panel = StatisticsPanel(SimpleNamespace())
    section = CollapsibleSection("Graphs", start_open=True)
    panel.bind_graphs_section(section)
    tabs.addTab(section, "Analysis")
    tabs.addTab(QWidget(), "Scene")
    panel.stats = {"total_frames": 1}
    panel._graphs_panel_created = True
    render = Mock()
    panel._render_statistics_graphs = render
    tabs.show()
    qtbot.waitUntil(lambda: panel._graphs_visible)
    tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: not panel._graphs_visible)
    render.reset_mock()
    panel._mark_graphs_dirty()
    assert render.call_count == 0
    tabs.setCurrentIndex(0)
    qtbot.waitUntil(lambda: render.call_count == 1)
    panel.cleanup()


def test_statistics_cancel_retires_queued_result_and_retains_previous_summary(qtbot):
    service = SimpleNamespace(cancel_collection=Mock())
    panel = StatisticsPanel(SimpleNamespace(scenario_statistics_service=service))
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    panel._statistics_provider = object()
    panel._provider_stats_generation = 3
    previous = {"total_frames": 4}
    panel.stats = previous
    panel._cancel_statistics()
    panel._on_provider_stats_ready({"total_frames": 99}, False, 3)
    assert panel.stats is previous
    assert panel._compute_btn.isEnabled()
    service.cancel_collection.assert_called_once()
    panel.cleanup()


def test_video_dialog_translates_display_frames_to_internal_indices(qtbot, monkeypatch, tmp_path):
    from visualizer.src.panels.video_export_dialog import QMessageBox, VideoExportDialog

    parent = QMainWindow()
    qtbot.addWidget(parent)
    parent.export_video = Mock(return_value=True)
    dialog = VideoExportDialog(parent, 10)
    qtbot.addWidget(dialog)
    assert (dialog.start_frame_spin.value(), dialog.end_frame_spin.value()) == (1, 10)
    dialog.file_input.setText(str(tmp_path / "movie.mp4"))
    dialog.start_frame_spin.setValue(2)
    dialog.end_frame_spin.setValue(8)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    dialog._export()
    assert parent.export_video.call_args.kwargs["start_frame"] == 1
    assert parent.export_video.call_args.kwargs["end_frame"] == 7


def test_new_statistics_source_clears_previous_summary_before_cache_lookup(
    qtbot, monkeypatch, tmp_path
):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QLabel

    from visualizer.src.panels import statistics_panel

    # Fresh settings keep this manual-collection case independent of user preferences.
    settings = QSettings(str(tmp_path / "statistics.ini"), QSettings.IniFormat)
    monkeypatch.setattr(statistics_panel, "QSettings", lambda: settings)
    service = SimpleNamespace(start_collection=Mock(return_value=7), cancel_collection=Mock())
    panel = StatisticsPanel(SimpleNamespace(scenario_statistics_service=service))
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    old = QLabel("42 old frames", widget)
    panel.widgets["old_summary"] = old
    panel.stats = {"total_frames": 42}
    provider = object()
    monkeypatch.setattr(statistics_panel, "resolve_statistics_provider", lambda source: provider)
    assert panel.set_statistics_source(provider)
    assert panel.stats is None
    assert old.text() == statistics_panel.EMPTY_VALUE
    assert service.start_collection.call_args.kwargs["cache_only"]
    panel._on_provider_stats_ready(None, False, 7)
    assert panel._compute_btn.isEnabled()
    assert "No cached statistics" in panel._compute_status.text()
    panel.cleanup()


def test_statistics_error_keeps_details_and_enables_retry(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    panel._statistics_provider = object()
    panel._provider_stats_generation = 2
    panel._on_provider_stats_error("Frame 17: read failed", 2)
    assert panel._compute_btn.isEnabled()
    assert panel._compute_btn.text() == "Retry Statistics"
    assert not panel._stats_error_section.isHidden()
    assert panel._stats_error_text.toPlainText() == "Frame 17: read failed"
    panel.cleanup()
