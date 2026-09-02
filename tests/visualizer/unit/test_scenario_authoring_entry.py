"""Feature-gate, CLI, menu, and workspace-mode tests for authoring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMainWindow, QMenu, QMessageBox, QStackedWidget, QWidget

from visualizer.src.app import startup_workflow
from visualizer.src.authoring import mode_controller as mode_controller_module
from visualizer.src.authoring.compiler import ScenarioCompiler
from visualizer.src.authoring.feature import SCENARIO_BUILDER_ENV, scenario_builder_enabled
from visualizer.src.authoring.mode_controller import WorkspaceMode, WorkspaceModeController
from visualizer.src.controllers.menu_controller import MenuController
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.visualizer import OrchavVisualizer


def test_feature_gate_requires_exact_one() -> None:
    assert scenario_builder_enabled({}) is False
    assert scenario_builder_enabled({SCENARIO_BUILDER_ENV: "true"}) is False
    assert scenario_builder_enabled({SCENARIO_BUILDER_ENV: "1"}) is True


def test_author_cli_is_gated_and_pygfx_only(monkeypatch) -> None:
    monkeypatch.delenv(SCENARIO_BUILDER_ENV, raising=False)
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--author"])

    monkeypatch.setenv(SCENARIO_BUILDER_ENV, "1")
    args = startup_workflow.parse_cli_args(["--author", "--scenario", "example"])
    assert args.author is True
    assert args.scenario == "example"
    assert args.renderer == "pygfx"
    assert startup_workflow.workspace_resume_enabled(args) is False

    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--author", "--renderer", "open3d"])
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--author", "--benchmark", "1"])


def test_author_startup_routes_scenario_without_normal_open(monkeypatch) -> None:
    monkeypatch.setenv(SCENARIO_BUILDER_ENV, "1")
    args = startup_workflow.parse_cli_args(["--author", "--scenario", "scenario.yaml"])
    viz = SimpleNamespace(
        _deferred_init=Mock(),
        open_for_authoring=Mock(),
        new_authoring_scenario=Mock(),
        open_scenario=Mock(),
    )
    app = SimpleNamespace(processEvents=Mock())
    reporter = SimpleNamespace(note=Mock())

    startup_workflow._complete_cli_startup(args=args, app=app, reporter=reporter, viz=viz)

    viz._deferred_init.assert_called_once_with(pending_camera=None)
    viz.open_for_authoring.assert_called_once_with("scenario.yaml")
    viz.new_authoring_scenario.assert_not_called()
    viz.open_scenario.assert_not_called()


def _menu_window() -> QMainWindow:
    window = QMainWindow()
    window.renderer = SimpleNamespace(capabilities=RendererCapabilities(scenario_authoring=True))
    window.new_authoring_scenario = Mock()
    window.open_for_authoring_dialog = Mock()
    window.edit_current_scenario = Mock()
    window.copy_current_scenario_and_edit = Mock()
    window.resume_authoring_draft = Mock()
    window.save_authoring_scenario = Mock()
    window.save_authoring_scenario_as = Mock()
    window.return_to_visualization = Mock()
    window.recent_files = []
    window.save_session_dialog = Mock()
    window.load_session_dialog = Mock()
    window._show_help_dialog = Mock()
    window._set_status_message = Mock()
    window.scene_appearance_service = SimpleNamespace(
        set_background_color=Mock(),
        pick_custom_background_color=Mock(),
        reset_to_default_background=Mock(),
    )
    return window


def test_scenario_builder_menu_exists_only_behind_gate(qapp, monkeypatch) -> None:
    monkeypatch.setenv(SCENARIO_BUILDER_ENV, "1")
    window = _menu_window()
    menu_controller = MenuController(
        SimpleNamespace(visualizer=window, toggle_metrics_window=Mock())
    )
    menu_controller.setup_menus(window, metrics_available=False)
    assert [action.text().replace("&", "") for action in window.menuBar().actions()] == [
        "File",
        "Scenario Builder",
        "View",
        "Display",
        "Help",
    ]
    builder_menu = window.scenario_builder_menu
    assert [(action.text(), action.isSeparator()) for action in builder_menu.actions()] == [
        ("Create", True),
        ("New Scenario", False),
        ("Copy Current Scenario and Edit...", False),
        ("Edit or Continue", True),
        ("Open Scenario for Authoring...", False),
        ("Edit Current Scenario", False),
        ("Resume Authoring Draft", False),
        ("Save", True),
        ("Save Scenario", False),
        ("Save Scenario As...", False),
        ("Workspace", True),
        ("Return to Visualization", False),
    ]
    actions = {action.text(): action for action in window.findChildren(QAction)}
    for label in (
        "New Scenario",
        "Copy Current Scenario and Edit...",
        "Open Scenario for Authoring...",
        "Edit Current Scenario",
        "Resume Authoring Draft",
        "Save Scenario",
        "Save Scenario As...",
        "Return to Visualization",
    ):
        assert [
            owner for owner in actions[label].associatedObjects() if isinstance(owner, QMenu)
        ] == [builder_menu]
    actions["New Scenario"].trigger()
    actions["Open Scenario for Authoring..."].trigger()
    assert not actions["Edit Current Scenario"].isEnabled()
    assert not actions["Copy Current Scenario and Edit..."].isEnabled()
    actions["Copy Current Scenario and Edit..."].setEnabled(True)
    actions["Copy Current Scenario and Edit..."].trigger()
    actions["Edit Current Scenario"].setEnabled(True)
    actions["Edit Current Scenario"].trigger()
    actions["Resume Authoring Draft"].setEnabled(True)
    actions["Resume Authoring Draft"].trigger()
    actions["Save Scenario"].setEnabled(True)
    actions["Save Scenario"].trigger()
    actions["Save Scenario As..."].setEnabled(True)
    actions["Save Scenario As..."].trigger()
    actions["Return to Visualization"].setEnabled(True)
    actions["Return to Visualization"].trigger()
    window.new_authoring_scenario.assert_called_once_with()
    window.open_for_authoring_dialog.assert_called_once_with()
    window.copy_current_scenario_and_edit.assert_called_once_with()
    window.edit_current_scenario.assert_called_once_with()
    window.resume_authoring_draft.assert_called_once_with()
    window.save_authoring_scenario.assert_called_once_with()
    window.save_authoring_scenario_as.assert_called_once_with()
    window.return_to_visualization.assert_called_once_with()
    assert "Resume Authoring Draft" in actions
    assert "Save Scenario" in actions
    assert "Save Scenario As..." in actions
    assert actions["Save Scenario"].shortcut().toString() == "Ctrl+S"
    assert actions["Save Scenario As..."].shortcut().toString() == "Ctrl+Shift+S"
    window.close()

    monkeypatch.delenv(SCENARIO_BUILDER_ENV, raising=False)
    disabled = _menu_window()
    disabled_menu_controller = MenuController(
        SimpleNamespace(visualizer=disabled, toggle_metrics_window=Mock())
    )
    disabled_menu_controller.setup_menus(disabled, metrics_available=False)
    disabled_menu_titles = {
        action.text().replace("&", "") for action in disabled.menuBar().actions()
    }
    assert "Scenario Builder" not in disabled_menu_titles
    labels = {action.text() for action in disabled.findChildren(QAction)}
    assert "New Scenario" not in labels
    assert "Open Scenario for Authoring..." not in labels
    assert "Copy Current Scenario and Edit..." not in labels
    assert "Edit Current Scenario" not in labels
    disabled.close()

    monkeypatch.setenv(SCENARIO_BUILDER_ENV, "1")
    incapable = _menu_window()
    incapable.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(scenario_authoring=False)
    )
    incapable_menu_controller = MenuController(
        SimpleNamespace(visualizer=incapable, toggle_metrics_window=Mock())
    )
    incapable_menu_controller.setup_menus(incapable, metrics_available=False)
    assert "Scenario Builder" not in {
        action.text().replace("&", "") for action in incapable.menuBar().actions()
    }
    incapable.close()


def test_builder_actions_follow_pending_waypoint_session(qapp) -> None:
    new_action = QAction()
    open_action = QAction()
    save_action = QAction()
    save_as_action = QAction()
    return_action = QAction()
    workspace = SimpleNamespace(has_pending_waypoint_session=True)
    visualizer = SimpleNamespace(
        workspace_mode_controller=SimpleNamespace(
            authoring_document=SimpleNamespace(read_only=False),
            mode=SimpleNamespace(value="authoring"),
            workspace=workspace,
        ),
        authoring_generation_controller=SimpleNamespace(running=False),
        new_authoring_scenario_action=new_action,
        open_for_authoring_action=open_action,
        save_authoring_scenario_action=save_action,
        save_authoring_scenario_as_action=save_as_action,
        return_to_visualization_action=return_action,
    )

    OrchavVisualizer._refresh_authoring_actions(visualizer)

    assert not new_action.isEnabled()
    assert not open_action.isEnabled()
    assert not save_action.isEnabled()
    assert not save_as_action.isEnabled()
    assert not return_action.isEnabled()

    workspace.has_pending_waypoint_session = False
    OrchavVisualizer._refresh_authoring_actions(visualizer)

    assert new_action.isEnabled()
    assert open_action.isEnabled()
    assert save_action.isEnabled()
    assert save_as_action.isEnabled()
    assert return_action.isEnabled()

    visualizer.authoring_generation_controller.running = True
    OrchavVisualizer._refresh_authoring_actions(visualizer)

    assert not new_action.isEnabled()
    assert not open_action.isEnabled()
    assert not save_action.isEnabled()
    assert not save_as_action.isEnabled()
    assert not return_action.isEnabled()


def test_edit_current_scenario_action_tracks_active_visualization_scenario(
    qapp,
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    edit_action = QAction()
    copy_action = QAction()
    controller = SimpleNamespace(
        authoring_document=None,
        mode=SimpleNamespace(value="visualization"),
        workspace=None,
    )
    visualizer = SimpleNamespace(
        workspace_mode_controller=controller,
        authoring_generation_controller=SimpleNamespace(running=False),
        edit_current_scenario_action=edit_action,
        copy_current_scenario_action=copy_action,
        current_scenario_path=str(scenario_root),
    )

    OrchavVisualizer._refresh_authoring_actions(visualizer)
    assert edit_action.isEnabled()
    assert copy_action.isEnabled()

    controller.mode = SimpleNamespace(value="authoring")
    OrchavVisualizer._refresh_authoring_actions(visualizer)
    assert not edit_action.isEnabled()
    assert not copy_action.isEnabled()

    controller.mode = SimpleNamespace(value="visualization")
    (scenario_root / "scenario.yaml").unlink()
    OrchavVisualizer._refresh_authoring_actions(visualizer)
    assert not edit_action.isEnabled()
    assert not copy_action.isEnabled()


def test_edit_current_scenario_reuses_open_for_authoring_path(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    visualizer = SimpleNamespace(
        current_scenario_path=str(scenario_root),
        open_for_authoring=Mock(return_value=True),
    )

    assert OrchavVisualizer.edit_current_scenario(visualizer)

    visualizer.open_for_authoring.assert_called_once_with(str(scenario_root))


def test_copy_current_scenario_uses_active_canonical_source(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    source = scenario_root / "scenario.yaml"
    source.write_text("schema_version: 2\n", encoding="utf-8")
    controller = SimpleNamespace(create_copy_for_authoring=Mock(return_value=True))
    visualizer = SimpleNamespace(
        current_scenario_path=str(scenario_root),
        _authoring_mode_controller=Mock(return_value=controller),
    )

    assert OrchavVisualizer.copy_current_scenario_and_edit(visualizer)

    controller.create_copy_for_authoring.assert_called_once_with(source)


class _FakeWorkspace(QWidget):
    """Small workspace double with the signals consumed by the controller."""

    from PySide6.QtCore import Signal

    save_requested = Signal()
    save_as_requested = Signal()
    leave_authoring_requested = Signal()
    preview_generated_requested = Signal(object)
    generate_requested = Signal()
    cancel_generation_requested = Signal()
    title_changed = Signal(str)
    dirty_changed = Signal(bool)

    def __init__(self, _visualizer, document, parent=None):
        super().__init__(parent)
        self.document = document
        self.document_label = SimpleNamespace(text=lambda: "Untitled Scenario *")
        self.compilation_lock = object()
        self.closed = False
        self.generation_log = []

    def bind_document(self, document):
        self.document = document

    def close_workspace(self):
        self.closed = True

    def confirm_replace(self):
        return True

    def commit_pending_edits(self):
        return None

    def show_read_only_import(self, _result):
        self.document = None

    def append_generation_log(self, text):
        self.generation_log.append(text)


def test_mode_controller_opens_unmarked_canonical_configuration_in_place_without_copy_dialog(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_root = tmp_path / "source"
    scenario_root.mkdir()
    source = scenario_root / "scenario.yaml"
    mapping = {
        "schema_version": 2,
        "scene": {"source": "library", "id": "empty/empty.xml"},
        "debug_level": "WARNING",
        "view_defaults": {"merge_scene_meshes": True},
        "timeline": {"steps": 30, "duration_s": 3.0},
        "raytracing": {
            "enabled": True,
            "quality": {
                "preset": "custom",
                "custom": {"max_depth": 4, "samples_per_src": 75_000},
            },
            "materials": {
                "concrete": {"scattering_coefficient": 0.3},
            },
        },
        "actors": {
            "tx": [
                {
                    "name": "TX1",
                    "mobility": {
                        "type": "stationary",
                        "position_m": [0.0, 0.0, 1.0],
                    },
                }
            ],
            "rx": [
                {
                    "name": "RX1",
                    "mobility": {
                        "type": "stationary",
                        "position_m": [2.0, 0.0, 1.0],
                    },
                }
            ],
        },
    }
    source.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    source_bytes = source.read_bytes()
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    monkeypatch.setattr(
        mode_controller_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: pytest.fail("Open for Authoring must not choose a destination"),
    )

    assert controller.open_document(source)
    assert controller.mode is WorkspaceMode.AUTHORING
    assert controller.workspace is not None
    document = controller.workspace.document
    assert document is not None
    assert document.path == source.resolve()
    assert [actor.name for actor in document.scenario.actors] == ["TX1", "RX1"]
    assert document.scenario.source_snapshot.has_path("raytracing.quality.custom")
    assert source.read_bytes() == source_bytes

    assert controller.leave_authoring(restore_visualization=False)
    visualizer.close()


def test_mode_controller_creates_persisted_copy_and_enters_it_for_authoring(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "orchav-project"
    library_scene = project_root / "libraries" / "scenes" / "empty" / "empty.xml"
    library_scene.parent.mkdir(parents=True)
    library_scene.write_bytes(
        (
            Path(__file__).resolve().parents[3] / "libraries" / "scenes" / "empty" / "empty.xml"
        ).read_bytes()
    )
    compiler = ScenarioCompiler(project_root)
    persistence_save_document = mode_controller_module.save_document
    monkeypatch.setattr(
        mode_controller_module,
        "save_document",
        lambda document: persistence_save_document(document, compiler=compiler),
    )
    monkeypatch.setattr(
        mode_controller_module.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: pytest.fail("successful copy must not open a warning dialog"),
    )
    monkeypatch.setattr(
        mode_controller_module.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: pytest.fail("successful copy must not open an error dialog"),
    )

    source_root = project_root / "scenarios" / "source"
    source_root.mkdir(parents=True)
    source = source_root / "scenario.yaml"
    mapping = {
        "schema_version": 2,
        "scene": {"source": "library", "id": "empty/empty.xml"},
        "timeline": {"steps": 2, "duration_s": 1.0},
        "raytracing": {"enabled": True},
        "actors": {
            "tx": [
                {
                    "name": "TX1",
                    "mobility": {"type": "stationary", "position_m": [0, 0, 1]},
                }
            ],
            "rx": [
                {
                    "name": "RX1",
                    "mobility": {"type": "stationary", "position_m": [1, 0, 1]},
                }
            ],
        },
    }
    source.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    source_bytes = source.read_bytes()
    (source_root / "frames").mkdir()
    (source_root / "frames" / "frame_00000.h5").write_text(
        "generated output",
        encoding="utf-8",
    )
    destination = project_root / "scenarios" / "copy"
    visualizer = QMainWindow()
    visualizer.setCentralWidget(QWidget())
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)

    assert controller.create_copy_for_authoring(source, destination=destination)
    assert controller.workspace is not None
    document = controller.workspace.document
    assert document is not None
    assert document.path == (destination / "scenario.yaml").resolve()
    assert not document.dirty
    assert source.read_bytes() == source_bytes
    assert (destination / "scenario.yaml").is_file()
    assert not (destination / "frames").exists()
    assert controller._normal_scenario_path == str(destination.resolve())

    assert controller.leave_authoring(restore_visualization=False)
    visualizer.close()


def test_mode_controller_restores_normal_widget_and_retains_document(qapp) -> None:
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    visualizer.setWindowTitle("ORCHAV")
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)

    assert controller.new_document() is True
    document = controller.authoring_document
    assert controller.mode is WorkspaceMode.AUTHORING
    assert visualizer.centralWidget() is controller.workspace

    assert controller.leave_authoring() is True
    assert controller.mode is WorkspaceMode.VISUALIZATION
    assert visualizer.centralWidget() is normal_widget
    assert controller.authoring_document is document
    assert controller.resume_document() is True
    assert controller.workspace.document is document
    assert controller.workspace.document.undo_stack is document.undo_stack
    visualizer.close()


def test_mode_controller_retires_live_scenario_before_renderer_close_and_reopens_it(
    qapp,
    monkeypatch,
) -> None:
    visualizer = QMainWindow()
    stack = QStackedWidget(visualizer)
    normal_widget = QWidget(stack)
    stack.addWidget(normal_widget)
    stack.setCurrentWidget(normal_widget)
    visualizer.setCentralWidget(stack)
    visualizer._workspace_stack = stack
    visualizer._visualization_workspace = normal_widget
    visualizer._viewport_mode = "detached"
    visualizer.current_scenario_path = "scenario"
    visualizer.vis_initialized = True
    visualizer.open_scenario = Mock()
    lifecycle_events = []

    def cleanup_active_scenario(viz):
        lifecycle_events.append(("cleanup", stack.currentWidget() is normal_widget))
        viz.current_scenario_path = None

    monkeypatch.setattr(
        mode_controller_module,
        "cleanup_previous_scene",
        cleanup_active_scenario,
    )
    monkeypatch.setattr(
        mode_controller_module.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    visualizer.renderer = SimpleNamespace(
        close=lambda: lifecycle_events.append(
            ("renderer_close", stack.currentWidget() is normal_widget)
        )
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)

    assert controller.new_document() is True
    workspace = controller.workspace

    assert lifecycle_events == [("cleanup", True), ("renderer_close", True)]
    assert controller._normal_scenario_path == "scenario"
    assert stack.currentWidget() is workspace
    assert controller.leave_authoring() is True
    assert stack.currentWidget() is normal_widget
    assert visualizer.centralWidget() is stack
    assert workspace.closed is True
    visualizer.open_scenario.assert_called_once_with("scenario")
    QCoreApplication.sendPostedEvents(workspace, QEvent.Type.DeferredDelete)
    visualizer.close()


def test_mode_controller_replaces_scene_loading_status_while_authoring(
    qapp,
    monkeypatch,
) -> None:
    visualizer = QMainWindow()
    visualizer.setCentralWidget(QWidget())
    visualizer.current_scenario_path = "scenario"
    visualizer.vis_initialized = False
    visualizer._set_status_message = Mock()
    visualizer.status_scenario_label = QLabel()
    visualizer.status_scenario_label.setToolTip("Retired scenario frames")
    telemetry = SimpleNamespace(_scenario_summary_text="Retired scenario")
    visualizer.ui_controller = SimpleNamespace(_telemetry_ctrl=telemetry)

    def cleanup_active_scenario(viz):
        viz._set_status_message("Loading new scene...")

    monkeypatch.setattr(
        mode_controller_module,
        "cleanup_previous_scene",
        cleanup_active_scenario,
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)

    assert controller.new_document() is True
    assert visualizer._set_status_message.call_args_list == [
        call("Loading new scene..."),
        call("Scenario Builder ready"),
    ]
    assert telemetry._scenario_summary_text == "Scenario Builder ready"
    assert visualizer.status_scenario_label.toolTip() == ""

    assert controller.leave_authoring(restore_visualization=False) is True
    visualizer._set_status_message.assert_called_with("Ready")
    assert telemetry._scenario_summary_text == "Ready"
    visualizer.close()


def test_new_status_invalidates_an_older_transient_restore(qapp, monkeypatch) -> None:
    callbacks = []
    visualizer = SimpleNamespace(
        status_scenario_label=QLabel(),
        ui_controller=SimpleNamespace(
            _telemetry_ctrl=SimpleNamespace(_scenario_summary_text="Retired scenario")
        ),
    )
    monkeypatch.setattr(
        "visualizer.visualizer.QTimer.singleShot",
        lambda _timeout, callback: callbacks.append(callback),
    )

    OrchavVisualizer._set_status_message(visualizer, "Scenario loaded successfully", 5000)
    OrchavVisualizer._set_status_message(visualizer, "Scenario Builder ready")
    callbacks[0]()

    assert visualizer.status_scenario_label.text() == "Scenario Builder ready"


def test_mode_controller_rebinds_generation_state_after_workspace_resume(qapp) -> None:
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    bind_workspace = Mock()
    visualizer.authoring_generation_controller = SimpleNamespace(
        running=False,
        bind_workspace=bind_workspace,
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)

    assert controller.new_document() is True
    first_workspace = controller.workspace
    bind_workspace.assert_called_once_with(first_workspace)
    assert controller.leave_authoring() is True
    assert controller.resume_document() is True
    resumed_workspace = controller.workspace

    assert resumed_workspace is not first_workspace
    assert bind_workspace.call_args_list == [
        call(first_workspace),
        call(resumed_workspace),
    ]
    visualizer.close()


def test_failed_generated_preview_restores_draft_and_undo_stack(qapp, monkeypatch) -> None:
    visualizer = QMainWindow()
    stack = QStackedWidget(visualizer)
    normal_widget = QWidget(stack)
    stack.addWidget(normal_widget)
    stack.setCurrentWidget(normal_widget)
    visualizer.setCentralWidget(stack)
    visualizer._workspace_stack = stack
    visualizer._visualization_workspace = normal_widget
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    visualizer.open_scenario = Mock(
        return_value=SimpleNamespace(succeeded=False, message="Preview load failed")
    )
    monkeypatch.setattr(
        mode_controller_module.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    document = controller.authoring_document
    assert document is not None
    undo_stack = document.undo_stack

    controller.preview_generated_result("generated-scenario")

    visualizer.open_scenario.assert_called_once_with("generated-scenario")
    assert controller.mode is WorkspaceMode.AUTHORING
    assert controller.authoring_document is document
    assert controller.workspace is not None
    assert controller.workspace.document is document
    assert controller.workspace.document.undo_stack is undo_stack
    assert controller.workspace.generation_log == ["Preview load failed"]
    assert stack.currentWidget() is controller.workspace
    visualizer.close()


def test_generated_preview_exception_restores_draft_and_undo_stack(qapp, monkeypatch) -> None:
    visualizer = QMainWindow()
    stack = QStackedWidget(visualizer)
    normal_widget = QWidget(stack)
    stack.addWidget(normal_widget)
    stack.setCurrentWidget(normal_widget)
    visualizer.setCentralWidget(stack)
    visualizer._workspace_stack = stack
    visualizer._visualization_workspace = normal_widget
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    visualizer.open_scenario = Mock(side_effect=RuntimeError("renderer initialization failed"))
    monkeypatch.setattr(
        mode_controller_module.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    document = controller.authoring_document
    assert document is not None
    undo_stack = document.undo_stack

    controller.preview_generated_result("generated-scenario")

    visualizer.open_scenario.assert_called_once_with("generated-scenario")
    assert controller.mode is WorkspaceMode.AUTHORING
    assert controller.authoring_document is document
    assert controller.workspace is not None
    assert controller.workspace.document is document
    assert controller.workspace.document.undo_stack is undo_stack
    assert controller.workspace.generation_log == [
        "Generated preview could not be opened: renderer initialization failed"
    ]
    assert stack.currentWidget() is controller.workspace
    visualizer.close()


def test_preview_without_generated_path_keeps_authoring_active(qapp) -> None:
    visualizer = QMainWindow()
    visualizer.setCentralWidget(QWidget())
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    workspace = controller.workspace

    controller.preview_generated_result()

    assert controller.mode is WorkspaceMode.AUTHORING
    assert controller.workspace is workspace
    assert workspace.generation_log == [
        "Generated preview is unavailable because no scenario path was produced."
    ]
    visualizer.close()


def test_normal_open_preparation_preserves_draft_without_restoring_old_scenario(
    qapp, monkeypatch
) -> None:
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = "old-scenario"
    visualizer.vis_initialized = False
    monkeypatch.setattr(
        mode_controller_module,
        "cleanup_previous_scene",
        lambda viz: setattr(viz, "current_scenario_path", None),
    )
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    document = controller.authoring_document
    assert document is not None
    undo_stack = document.undo_stack
    question = Mock(return_value=QMessageBox.Yes)
    monkeypatch.setattr(mode_controller_module.QMessageBox, "question", question)

    assert controller.prepare_normal_scenario_open() is True

    assert "will be preserved" in question.call_args.args[2]
    assert controller.mode is WorkspaceMode.VISUALIZATION
    assert controller.authoring_document is document
    assert visualizer.centralWidget() is normal_widget
    assert controller.resume_document() is True
    assert controller.workspace is not None
    assert controller.workspace.document is document
    assert controller.workspace.document.undo_stack is undo_stack
    visualizer.close()


def test_normal_open_preparation_honors_authoring_switch_cancellation(qapp, monkeypatch) -> None:
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    workspace = controller.workspace
    assert workspace is not None
    document = controller.authoring_document
    undo_stack = document.undo_stack
    question = Mock(return_value=QMessageBox.Cancel)
    monkeypatch.setattr(mode_controller_module.QMessageBox, "question", question)

    assert controller.prepare_normal_scenario_open() is False

    assert "will be preserved" in question.call_args.args[2]
    assert controller.mode is WorkspaceMode.AUTHORING
    assert visualizer.centralWidget() is workspace
    assert controller.authoring_document is document
    assert controller.authoring_document.undo_stack is undo_stack
    assert workspace.closed is False
    visualizer.close()


def test_return_to_visualization_signal_uses_guarded_switch_policy(qapp, monkeypatch) -> None:
    visualizer = QMainWindow()
    normal_widget = QWidget()
    visualizer.setCentralWidget(normal_widget)
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    workspace = controller.workspace
    assert workspace is not None
    question = Mock(return_value=QMessageBox.Cancel)
    monkeypatch.setattr(mode_controller_module.QMessageBox, "question", question)

    workspace.leave_authoring_requested.emit()

    question.assert_called_once()
    assert controller.mode is WorkspaceMode.AUTHORING
    assert visualizer.centralWidget() is workspace
    assert workspace.closed is False
    visualizer.close()


def test_direct_normal_open_runs_only_after_authoring_preparation(monkeypatch) -> None:
    app_open = Mock()
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    prepare = Mock(return_value=False)
    visualizer = SimpleNamespace(
        workspace_mode_controller=SimpleNamespace(
            prepare_normal_scenario_open=prepare,
        )
    )

    OrchavVisualizer.open_scenario(visualizer, "blocked-scenario")

    prepare.assert_called_once_with()
    app_open.assert_not_called()

    prepare.return_value = True
    OrchavVisualizer.open_scenario(
        visualizer,
        "next-scenario",
        pending_camera={"position": [1.0, 2.0, 3.0]},
        autorun_initial_frame=False,
    )

    app_open.assert_called_once_with(
        visualizer,
        "next-scenario",
        pending_camera={"position": [1.0, 2.0, 3.0]},
        autorun_initial_frame=False,
    )


def test_normal_open_and_retry_retain_process_local_data_source_overrides(monkeypatch) -> None:
    outcome = SimpleNamespace(succeeded=True)
    app_open = Mock(return_value=outcome)
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)
    visualizer = SimpleNamespace(workspace_mode_controller=None)

    assert (
        OrchavVisualizer.open_scenario(
            visualizer,
            "live-scenario",
            data_mode_override="live_grpc",
            grpc_port_override=50052,
        )
        is outcome
    )
    assert visualizer._last_scenario_data_mode_override == "live_grpc"
    assert visualizer._last_scenario_grpc_port_override == 50052
    app_open.assert_called_once_with(
        visualizer,
        "live-scenario",
        pending_camera=None,
        autorun_initial_frame=True,
        data_mode_override="live_grpc",
        grpc_port_override=50052,
    )

    app_open.reset_mock()
    visualizer.open_scenario = lambda *args, **kwargs: OrchavVisualizer.open_scenario(
        visualizer,
        *args,
        **kwargs,
    )
    assert OrchavVisualizer.retry_last_scenario(visualizer) is True
    app_open.assert_called_once_with(
        visualizer,
        "live-scenario",
        pending_camera=None,
        autorun_initial_frame=True,
        data_mode_override="live_grpc",
        grpc_port_override=50052,
    )


def test_save_as_selects_directory_and_uses_canonical_persistence(qapp, monkeypatch) -> None:
    visualizer = QMainWindow()
    visualizer.setCentralWidget(QWidget())
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True
    destination = Path(__file__).parent / "authoring-save-as"
    monkeypatch.setattr(
        mode_controller_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )

    def save(document, selected, *, compiler=None, compile_lock=None):
        assert Path(selected) == destination
        assert compile_lock is controller.workspace.compilation_lock
        path = destination / "scenario.yaml"
        document.mark_saved(path)
        return path

    save_mock = Mock(side_effect=save)
    monkeypatch.setattr(mode_controller_module, "save_document", save_mock)

    assert controller.save_as() is True
    assert controller.authoring_document.path == (destination / "scenario.yaml").resolve()
    save_mock.assert_called_once()
    visualizer.close()


def test_save_as_reports_project_library_destination_problem(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    visualizer = QMainWindow()
    visualizer.setCentralWidget(QWidget())
    visualizer.current_scenario_path = None
    visualizer.vis_initialized = False
    controller = WorkspaceModeController(visualizer, workspace_factory=_FakeWorkspace)
    assert controller.new_document() is True

    active_project = tmp_path / "active-project"
    destination = tmp_path / "outside-project"
    active_project.mkdir()
    destination.mkdir()
    controller.workspace.compiler = ScenarioCompiler(active_project)
    controller.workspace.refresh_now = lambda: None
    monkeypatch.setattr(
        mode_controller_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    warnings = []
    monkeypatch.setattr(
        mode_controller_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert controller.save_as() is False
    assert not (destination / "scenario.yaml").exists()
    assert len(warnings) == 1
    assert warnings[0][0] == "Scenario Has Problems"
    assert "inside the active ORCHAV project root" in warnings[0][1]
    visualizer.close()


def test_legacy_builder_is_not_registered_in_normal_panel_manager() -> None:
    from visualizer.src.app import panel_manager

    assert not hasattr(panel_manager, "TrajectoryBuilderPanel")
    assert "trajectory_builder" not in {
        key
        for _tab, definitions in panel_manager.UIPanelManager._CORE_TABS
        for key, _ in definitions
    }
