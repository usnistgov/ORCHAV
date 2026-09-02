"""Focused ownership tests for main-window menu actions."""

from datetime import datetime
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow

from visualizer.src.controllers.menu_controller import MenuController
from visualizer.src.io import scenario_config
from visualizer.src.io.config_handlers import RecentFilesHandler
from visualizer.visualizer import OrchavVisualizer


def test_menu_actions_call_controller_and_scene_appearance_owners(qapp) -> None:
    """Metrics/background actions should not require root command delegates."""
    window = QMainWindow()
    scene_appearance = SimpleNamespace(
        set_background_color=Mock(),
        pick_custom_background_color=Mock(),
        reset_to_default_background=Mock(),
    )
    window.scene_appearance_service = scene_appearance
    window.recent_files = []
    window.save_session_dialog = Mock()
    window.load_session_dialog = Mock()
    window._show_help_dialog = Mock()
    window._set_status_message = Mock()
    ui_controller = SimpleNamespace(
        visualizer=window,
        toggle_metrics_window=Mock(),
    )

    MenuController(ui_controller).setup_menus(window, metrics_available=True)
    actions = {
        action.text(): action for action in window.findChildren(QAction) if not action.isSeparator()
    }

    actions["&Metrics Window"].trigger()
    actions["&Black"].trigger()
    actions["&Custom..."].trigger()
    actions["&Reset to Default (Dark Gray)"].trigger()

    ui_controller.toggle_metrics_window.assert_called_once_with()
    scene_appearance.set_background_color.assert_called_once_with([0.0, 0.0, 0.0])
    scene_appearance.pick_custom_background_color.assert_called_once_with()
    scene_appearance.reset_to_default_background.assert_called_once_with()
    window.close()


def test_workspace_menu_describes_recent_snapshot(qapp, tmp_path) -> None:
    """Recent workspaces should identify their scenario and saved frame."""
    window = QMainWindow()
    snapshot_path = tmp_path / "munich_filtering_dense_autosave_abcd1234.json"
    summary = SimpleNamespace(
        path=snapshot_path,
        scenario_root=tmp_path / "munich_filtering_dense",
        scenario_name="munich_filtering_dense",
        created_at=datetime(2026, 7, 18, 8, 30),
        frame=27,
        is_autosave=True,
    )
    window.session_service = SimpleNamespace(list_workspace_summaries=Mock(return_value=[summary]))
    window.recent_files = []
    window.save_session_dialog = Mock()
    window.load_session_dialog = Mock()
    window.load_session_file = Mock()
    window._show_help_dialog = Mock()
    window._set_status_message = Mock()
    window.scene_appearance_service = SimpleNamespace(
        set_background_color=Mock(),
        pick_custom_background_color=Mock(),
        reset_to_default_background=Mock(),
    )
    ui_controller = SimpleNamespace(visualizer=window, toggle_metrics_window=Mock())

    MenuController(ui_controller).setup_menus(window, metrics_available=False)
    actions = [action for action in window.findChildren(QAction) if not action.isSeparator()]
    recent_action = next(action for action in actions if "Munich Filtering Dense" in action.text())

    assert "frame 27" in recent_action.text()
    assert "(Auto)" in recent_action.text()
    assert str(summary.scenario_root) in recent_action.toolTip()
    recent_action.trigger()
    window.load_session_file.assert_called_once_with(snapshot_path)
    assert any(action.text() == "&Save Workspace Snapshot..." for action in actions)
    assert any(action.text() == "&Open Workspace Snapshot..." for action in actions)
    window.close()


def test_manual_workspace_menu_includes_name_time_and_kind(qapp, tmp_path) -> None:
    """Named snapshots should remain recognizable without exposing raw paths."""
    window = QMainWindow()
    summary = SimpleNamespace(
        path=tmp_path / "roof_analysis.json",
        scenario_root=tmp_path / "munich_filtering_dense",
        scenario_name="munich_filtering_dense",
        created_at=datetime(2026, 7, 18, 9, 5),
        frame=12,
        is_autosave=False,
    )
    window.session_service = SimpleNamespace(list_workspace_summaries=Mock(return_value=[summary]))
    window.recent_files = []
    window.save_session_dialog = Mock()
    window.load_session_dialog = Mock()
    window.load_session_file = Mock()
    window._show_help_dialog = Mock()
    window._set_status_message = Mock()
    window.scene_appearance_service = SimpleNamespace(
        set_background_color=Mock(),
        pick_custom_background_color=Mock(),
        reset_to_default_background=Mock(),
    )
    ui_controller = SimpleNamespace(visualizer=window, toggle_metrics_window=Mock())

    MenuController(ui_controller).setup_menus(window, metrics_available=False)
    recent_action = next(
        action for action in window.findChildren(QAction) if "Roof Analysis" in action.text()
    )

    assert "Munich Filtering Dense" in recent_action.text()
    assert "frame 12" in recent_action.text()
    assert "Jul 18 09:05" in recent_action.text()
    assert "(Manual)" in recent_action.text()
    window.close()


def test_open_recent_scenario_directory_reaches_scenario_workflow(tmp_path, monkeypatch) -> None:
    """Recent scenario directories must not return before opening."""
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    app_open = Mock()
    prepare_normal_open = Mock(return_value=True)
    visualizer = SimpleNamespace(
        workspace_mode_controller=SimpleNamespace(
            prepare_normal_scenario_open=prepare_normal_open,
        ),
        recent_files=[],
    )
    visualizer.open_scenario = MethodType(OrchavVisualizer.open_scenario, visualizer)
    controller = MenuController(SimpleNamespace(visualizer=visualizer))
    controller.add_recent_file = Mock()
    monkeypatch.setattr(scenario_config, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)

    controller.open_recent_file(str(scenario_dir))

    prepare_normal_open.assert_called_once_with()
    app_open.assert_called_once_with(
        visualizer,
        "scenario_one",
        pending_camera=None,
        autorun_initial_frame=True,
    )
    controller.add_recent_file.assert_not_called()


def test_add_recent_file_persists_menu_state(tmp_path) -> None:
    """The menu owner persists a successful scenario recorded by the workflow."""
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    config_file = tmp_path / "visualizer.json"
    visualizer = SimpleNamespace(
        config_file=str(config_file),
        max_recent_files=3,
        recent_files=[],
    )
    controller = MenuController(SimpleNamespace(visualizer=visualizer))

    controller.add_recent_file(str(scenario_dir.resolve()))

    expected = [str(scenario_dir.resolve())]
    assert visualizer.recent_files == expected
    assert RecentFilesHandler.load_recent_files(str(config_file), 3) == expected


def test_open_recent_xml_records_once_after_success(tmp_path) -> None:
    """Direct XML remains a separate recent-file path and records only on success."""
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>", encoding="utf-8")
    load_scene = Mock()
    visualizer = SimpleNamespace(
        main_controller=SimpleNamespace(load_scene=load_scene),
        recent_files=[],
    )
    controller = MenuController(SimpleNamespace(visualizer=visualizer))
    controller.add_recent_file = Mock()

    controller.open_recent_file(str(xml_path))

    load_scene.assert_called_once_with(str(xml_path))
    controller.add_recent_file.assert_called_once_with(str(xml_path))


def test_open_recent_xml_reports_load_failure_without_recording_it(tmp_path, monkeypatch) -> None:
    """A failed direct XML load must remain absent from recent-file state."""
    xml_path = tmp_path / "broken.xml"
    xml_path.write_text("<scene>", encoding="utf-8")
    visualizer = SimpleNamespace(
        main_controller=SimpleNamespace(load_scene=Mock(side_effect=ValueError("parse-fail"))),
        recent_files=[],
    )
    controller = MenuController(SimpleNamespace(visualizer=visualizer))
    controller.add_recent_file = Mock()
    critical = Mock()
    monkeypatch.setattr(
        "visualizer.src.controllers.menu_controller.QMessageBox.critical",
        critical,
    )

    controller.open_recent_file(str(xml_path))

    visualizer.main_controller.load_scene.assert_called_once_with(str(xml_path))
    controller.add_recent_file.assert_not_called()
    assert "parse-fail" in critical.call_args.args[2]
