"""Statistics source ownership, reload overrides, and live connection status."""

from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.app import scenario_workflow
from visualizer.src.controllers.telemetry_controller import TelemetryController
from visualizer.src.io.frame_sources import LiveGrpcSource
from visualizer.src.panels import statistics_panel
from visualizer.visualizer import OrchavVisualizer


def test_disabled_statistics_panel_rebinds_without_automatic_collection():
    bind = Mock(return_value=True)
    source = object()
    viz = SimpleNamespace(
        pipeline=None,
        scenario_config=SimpleNamespace(visualizer_cfg={"panels": {"statistics": False}}),
        ui_manager=SimpleNamespace(
            panels={"statistics": SimpleNamespace(set_statistics_source=bind)}
        ),
    )
    assert scenario_workflow._start_scenario_statistics(viz, source)
    bind.assert_called_once_with(source, start_loading=False)


def test_statistics_source_can_be_retired_and_manually_rebound(monkeypatch):
    provider = object()
    monkeypatch.setattr(statistics_panel, "resolve_statistics_provider", lambda source: source)
    service = SimpleNamespace(start_collection=Mock(return_value=12), cancel_collection=Mock())
    panel = statistics_panel.StatisticsPanel(SimpleNamespace(scenario_statistics_service=service))
    assert panel.set_statistics_source(provider, start_loading=False)
    service.start_collection.assert_not_called()
    panel._on_compute_clicked()
    assert service.start_collection.call_args.args == (provider,)
    assert service.start_collection.call_args.kwargs["force"] is True
    panel.set_statistics_source(None)
    service.start_collection.reset_mock()
    assert panel._statistics_provider is None
    assert panel._provider_stats_generation is None
    panel._on_compute_clicked()
    service.start_collection.assert_not_called()


def test_reopen_preserves_same_yaml_overrides_without_leaking_to_other_scenarios(
    monkeypatch, tmp_path
):
    scenario = tmp_path / "stream"
    scenario.mkdir()
    config = scenario / "scenario.yaml"
    config.write_text("scene: empty\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "scenario.yaml").write_text("scene: empty\n")
    app_open = Mock(return_value=SimpleNamespace(succeeded=True))
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    viz = SimpleNamespace(workspace_mode_controller=None)
    OrchavVisualizer.open_scenario(
        viz, str(scenario), data_mode_override="live_grpc", grpc_port_override=50053
    )
    OrchavVisualizer.open_scenario(viz, str(config))
    assert app_open.call_args.kwargs["grpc_port_override"] == 50053
    assert app_open.call_args.kwargs["data_mode_override"] == "live_grpc"
    OrchavVisualizer.open_scenario(viz, str(config), grpc_port_override=50054)
    assert app_open.call_args.kwargs["grpc_port_override"] == 50054
    OrchavVisualizer.open_scenario(viz, str(config), data_mode_override="files")
    assert app_open.call_args.kwargs["data_mode_override"] == "files"
    assert "grpc_port_override" not in app_open.call_args.kwargs
    OrchavVisualizer.open_scenario(viz, str(other))
    assert "grpc_port_override" not in app_open.call_args.kwargs
    assert "data_mode_override" not in app_open.call_args.kwargs
    assert config.read_text() == "scene: empty\n"


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("same_recording", [False, True])
def test_unsuccessful_open_keeps_active_overrides_and_attempt_retry(
    monkeypatch, tmp_path, cancelled, same_recording
):
    active = tmp_path / "active.yaml"
    active.write_text("scene: empty\n")
    attempted = active if same_recording else tmp_path / "attempted.yaml"
    attempted.write_text("scene: empty\n")
    success = scenario_workflow.ScenarioOpenOutcome.success(tmp_path, frame_source_ready=True)
    unsuccessful = (
        scenario_workflow.ScenarioOpenOutcome.cancelled()
        if cancelled
        else scenario_workflow.ScenarioOpenOutcome.failure("Invalid recording")
    )
    app_open = Mock(return_value=success)
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    viz = SimpleNamespace(workspace_mode_controller=None)
    viz.open_scenario = lambda *args, **kwargs: OrchavVisualizer.open_scenario(viz, *args, **kwargs)
    viz.open_scenario(str(active), data_mode_override="live_grpc", grpc_port_override=50053)
    app_open.return_value = unsuccessful
    viz.open_scenario(str(attempted), data_mode_override="files")
    assert viz._last_scenario_attempt == str(attempted)
    assert OrchavVisualizer.retry_last_scenario(viz) is False
    assert app_open.call_args.args[1] == str(attempted)
    assert app_open.call_args.kwargs["data_mode_override"] == "files"
    assert "grpc_port_override" not in app_open.call_args.kwargs

    app_open.return_value = success
    viz.open_scenario(str(active))
    assert app_open.call_args.kwargs["data_mode_override"] == "live_grpc"
    assert app_open.call_args.kwargs["grpc_port_override"] == 50053


def test_cancelled_authoring_transition_keeps_active_source_overrides(monkeypatch, tmp_path):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scene: empty\n")
    app_open = Mock(return_value=SimpleNamespace(succeeded=True))
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    prepare = Mock(return_value=True)
    viz = SimpleNamespace(
        workspace_mode_controller=SimpleNamespace(prepare_normal_scenario_open=prepare)
    )
    OrchavVisualizer.open_scenario(
        viz, str(scenario), data_mode_override="live_grpc", grpc_port_override=50053
    )
    prepare.return_value = False
    result = OrchavVisualizer.open_scenario(viz, str(scenario), data_mode_override="files")
    assert result.status is scenario_workflow.ScenarioOpenStatus.CANCELLED
    prepare.return_value = True
    OrchavVisualizer.open_scenario(viz, str(scenario))
    assert app_open.call_args.kwargs["data_mode_override"] == "live_grpc"
    assert app_open.call_args.kwargs["grpc_port_override"] == 50053


def test_nested_open_does_not_replace_retry_source(monkeypatch, tmp_path):
    app_open = Mock(return_value=scenario_workflow.ScenarioOpenOutcome.cancelled())
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    viz = SimpleNamespace(
        _scenario_load_in_progress=True,
        _last_scenario_attempt="loading.yaml",
        _last_scenario_data_mode_override="live_grpc",
        _last_scenario_grpc_port_override=50053,
    )
    result = OrchavVisualizer.open_scenario(viz, str(tmp_path / "other.yaml"))
    assert result.status is scenario_workflow.ScenarioOpenStatus.CANCELLED
    assert viz._last_scenario_attempt == "loading.yaml"
    assert viz._last_scenario_grpc_port_override == 50053
    app_open.assert_not_called()


@pytest.mark.parametrize("retire", [False, True])
def test_statistics_source_change_clears_previous_error(qtbot, monkeypatch, retire):
    monkeypatch.setattr(statistics_panel, "resolve_statistics_provider", lambda source: source)
    service = SimpleNamespace(start_collection=Mock(return_value=12), cancel_collection=Mock())
    panel = statistics_panel.StatisticsPanel(SimpleNamespace(scenario_statistics_service=service))
    widget = panel._create_compute_section()
    qtbot.addWidget(widget)
    panel.set_statistics_source(object())
    panel._on_provider_stats_error("Previous source is unreadable", 12)
    assert not panel._stats_error_section.isHidden()
    assert panel._compute_btn.text() == "Retry Statistics"
    panel.set_statistics_source(None if retire else object(), start_loading=False)
    assert panel._stats_error_section.isHidden()
    assert panel._stats_error_text.toPlainText() == ""
    assert panel._compute_btn.text() == "Compute Statistics"


@pytest.mark.parametrize("binding", ["retired", "hidden", "loading"])
def test_queued_statistics_callbacks_do_not_cross_source_binding(qtbot, monkeypatch, binding):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(statistics_panel, "resolve_statistics_provider", lambda source: source)
    service = SimpleNamespace(start_collection=Mock(side_effect=[12, 13]), cancel_collection=Mock())
    panel = statistics_panel.StatisticsPanel(SimpleNamespace(scenario_statistics_service=service))
    widget = panel._create_compute_section()
    qtbot.addWidget(widget)
    panel.set_statistics_source(object())

    def emit_old_callbacks():
        panel._stats_signals.provider_finished.emit({"old_source": True}, False, 12)
        panel._stats_signals.provider_progress.emit(7, 10, 12)
        panel._stats_signals.provider_failed.emit("Old source error", 12)

    worker = Thread(target=emit_old_callbacks)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    panel.set_statistics_source(
        None if binding == "retired" else object(), start_loading=binding == "loading"
    )
    QApplication.processEvents()
    assert panel.stats is None
    assert panel._stats_error_section.isHidden()
    assert panel._stats_error_text.toPlainText() == ""
    assert panel._progress_bar.isHidden()
    assert panel._progress_bar.value() == 0


@pytest.mark.parametrize(
    "connected,streaming,expected",
    [
        (True, True, "Connected · streaming"),
        (True, False, "Connected · stream inactive"),
        (False, False, "Disconnected · stream inactive"),
        (False, True, "Disconnected · stream inactive"),
    ],
)
def test_stream_footer_tracks_connection_state(connected, streaming, expected):
    source = LiveGrpcSource("grpc://127.0.0.1:50053")
    source.get_connection_status = lambda: {"connected": connected, "streaming_active": streaming}
    label = Mock()
    viz = SimpleNamespace(frame_source=source, status_scenario_label=label)
    TelemetryController(SimpleNamespace(visualizer=viz)).update_file_source_summary()
    label.setText.assert_called_with(expected)


def test_periodic_live_status_updates_footer_without_recursive_panel_poll(monkeypatch):
    from visualizer.src.io import grpc_provider
    from visualizer.src.panels.live_grpc_mode_panel import LiveGrpcModePanel

    provider = SimpleNamespace(
        get_connection_status=lambda: {"connected": False},
        get_buffer_status=lambda: {},
        get_latency_statistics=lambda: {},
        get_request_history=lambda **kwargs: [],
    )
    monkeypatch.setattr(grpc_provider, "GrpcProvider", SimpleNamespace)
    update_summary = Mock()
    panel = SimpleNamespace(
        parent=SimpleNamespace(
            frame_source=SimpleNamespace(provider=provider),
            ui_controller=SimpleNamespace(update_file_source_summary=update_summary),
        ),
        widgets={},
        _connection=Mock(),
        _streaming=Mock(),
        _performance=Mock(),
    )
    LiveGrpcModePanel.update_live_grpc_mode_info(panel)
    update_summary.assert_called_once_with(refresh_source=False)


def test_live_footer_and_panel_refresh_have_bounded_call_chain(monkeypatch):
    from visualizer.src.io import grpc_provider
    from visualizer.src.panels.live_grpc_mode_panel import LiveGrpcModePanel

    provider = SimpleNamespace(
        get_connection_status=lambda: {"connected": True, "streaming_active": True},
        get_buffer_status=lambda: {},
        get_latency_statistics=lambda: {},
        get_request_history=lambda **kwargs: [],
    )
    monkeypatch.setattr(grpc_provider, "GrpcProvider", SimpleNamespace)
    source = LiveGrpcSource("grpc://127.0.0.1:50053")
    source.provider = provider
    source.is_initialized = True
    viz = SimpleNamespace(frame_source=source, status_scenario_label=Mock())
    telemetry = TelemetryController(SimpleNamespace(visualizer=viz))
    viz.ui_controller = SimpleNamespace(
        update_file_source_summary=telemetry.update_file_source_summary
    )
    panel = SimpleNamespace(
        parent=viz,
        widgets={},
        _connection=Mock(),
        _streaming=Mock(),
        _performance=Mock(),
    )
    refresh = Mock(side_effect=lambda: LiveGrpcModePanel.update_live_grpc_mode_info(panel))
    viz.ui_manager = SimpleNamespace(
        panels={"data_source": SimpleNamespace(refresh_source_status=refresh)}
    )
    telemetry.update_file_source_summary()
    refresh.assert_called_once_with()
    panel._connection.update.assert_called_once()
    assert viz.status_scenario_label.setText.call_count == 2
    viz.status_scenario_label.setText.assert_called_with("Connected · streaming")
