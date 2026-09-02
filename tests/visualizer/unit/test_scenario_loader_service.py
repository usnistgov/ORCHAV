import logging
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from shared.scenarios.actors import (
    ActorsSpec,
    RxActorSpec,
    StationaryMobilitySpec,
    TimelineSpec,
    TxActorSpec,
)
from visualizer.src.scene.defaults import (
    DEFAULT_NODE_MARKER_SIZE_M,
    NODE_MARKER_SIZE_BOUNDS_M,
)
from visualizer.src.scene.io import XMLSceneHandler
from visualizer.src.services.scenario_loader_service import (
    ScenarioFrameSourcePreparation,
    ScenarioLoaderService,
)
from visualizer.visualizer import OrchavVisualizer


class DummyDialogManager:
    def __init__(self):
        self.scenario_path = None
        self.save_path = None
        self.messages = []

    def select_scenario_file(self, default_dir):
        return self.scenario_path

    def select_save_file(self, title="", default_name="", filter_str=""):
        return self.save_path

    def show_info(self, title, message):
        self.messages.append(("info", title, message))

    def show_warning(self, title, message):
        self.messages.append(("warning", title, message))

    def show_error(self, title, message):
        self.messages.append(("error", title, message))


def test_explicit_cli_scenario_error_is_nonmodal():
    set_status = Mock()
    visualizer = SimpleNamespace(
        _cli_driven_frame_run=False,
        _explicit_cli_scenario_startup=True,
        _viewport_mode="detached",
        _set_status_message=set_status,
    )
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(visualizer, dialog)

    loader._report_open_error("live generator is busy")

    set_status.assert_called_once_with("live generator is busy", 5000)
    assert dialog.messages == []


def test_open_scenario_via_dialog_uses_mode_safe_normal_open(tmp_path, monkeypatch):
    dialog = DummyDialogManager()
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    dialog.scenario_path = scenario_dir
    app_open = Mock()
    prepare_normal_open = Mock(return_value=True)
    visualizer = SimpleNamespace(
        workspace_mode_controller=SimpleNamespace(
            prepare_normal_scenario_open=prepare_normal_open,
        ),
        reset_frame_retry_state=Mock(),
    )
    visualizer.open_scenario = MethodType(OrchavVisualizer.open_scenario, visualizer)
    monkeypatch.setattr("visualizer.visualizer.app_open_scenario", app_open)

    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.open_scenario_via_dialog() is True
    prepare_normal_open.assert_called_once_with()
    visualizer.reset_frame_retry_state.assert_called_once_with()
    app_open.assert_called_once_with(
        visualizer,
        "scenario_one",
        pending_camera=None,
        autorun_initial_frame=True,
    )


def test_save_scene_xml_invokes_handlers(tmp_path, monkeypatch):
    dialog = DummyDialogManager()
    save_path = tmp_path / "scene_modified.xml"
    dialog.save_path = str(save_path)

    saved = {}
    monkeypatch.setattr(
        XMLSceneHandler,
        "debug_xml_structure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        XMLSceneHandler,
        "save_xml_scene",
        lambda root, path: saved.update({"root": root, "path": path}),
    )

    loader = ScenarioLoaderService(
        SimpleNamespace(open_scenario=lambda _: None),
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    xml_root = {"objects": 3}
    assert loader.save_scene_xml(xml_root) is True
    assert saved == {"root": xml_root, "path": str(save_path)}
    assert dialog.messages[-1][0] == "info"


def test_validate_node_editing_context_handles_modes():
    dialog = DummyDialogManager()

    loader = ScenarioLoaderService(
        SimpleNamespace(open_scenario=lambda _: None),
        dialog,
        project_root_resolver=lambda: Path("."),
    )

    # Force missing live gRPC source class
    loader._online_source_cls = None

    def _force_missing_class():
        return None

    loader._resolve_online_source_cls = _force_missing_class
    assert loader.validate_node_editing_context(object()) is False
    assert dialog.messages[-1][0] == "warning"

    class FakeOnline:
        pass

    loader._resolve_online_source_cls = lambda: FakeOnline
    loader._online_source_cls = FakeOnline
    assert loader.validate_node_editing_context(FakeOnline()) is True
    assert loader.validate_node_editing_context(object()) is False
    assert dialog.messages[-1][0] == "info"


def test_sync_node_actor_names_from_scenario_updates_state():
    """Scenario actor names are copied into visualizer label state."""
    dialog = DummyDialogManager()
    set_state = Mock()
    viz = SimpleNamespace(set_state=set_state, mpc_view_cache={"old": object()})
    loader = ScenarioLoaderService(
        viz,
        dialog,
        project_root_resolver=lambda: Path("."),
    )
    scenario = SimpleNamespace(
        actors=ActorsSpec(
            tx=(
                TxActorSpec(
                    name="MainTransmitter",
                    mobility=StationaryMobilitySpec(position_m=(0.0, 0.0, 10.0)),
                ),
            ),
            rx=(
                RxActorSpec(
                    name="WalkingReceiver",
                    mobility=StationaryMobilitySpec(position_m=(5.0, 0.0, 1.5)),
                ),
                RxActorSpec(
                    name="StaticReceiver",
                    mobility=StationaryMobilitySpec(position_m=(1.0, 0.0, 1.5)),
                ),
            ),
        )
    )

    loader._sync_node_actor_names_from_scenario(scenario)

    set_state.assert_called_once_with(
        tx_device_names=("MainTransmitter",),
        rx_device_names=("WalkingReceiver", "StaticReceiver"),
    )
    assert viz.mpc_view_cache == {}


def test_scenario_log_level_preserves_env_and_syncs_combo_without_signal(qapp, monkeypatch):
    combo = QComboBox()
    combo.addItems(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    combo.setCurrentText("WARNING")
    emitted_levels = []
    combo.currentTextChanged.connect(emitted_levels.append)
    applied_levels = []
    monkeypatch.setenv("ORCHAV_LOG_LEVEL", "ERROR")
    monkeypatch.setattr(
        "visualizer.src.services.scenario_loader_service.set_log_level",
        lambda level: applied_levels.append(level) or level,
    )
    monkeypatch.setattr(
        "visualizer.src.services.scenario_loader_service.get_current_log_level_name",
        lambda: "ERROR",
    )
    loader = ScenarioLoaderService(
        SimpleNamespace(log_level_combo=combo),
        DummyDialogManager(),
    )

    assert loader._apply_scenario_log_level("DEBUG") == "ERROR"
    assert applied_levels == [logging.ERROR]
    assert combo.currentText() == "ERROR"
    assert emitted_levels == []
    assert combo.signalsBlocked() is False


def test_sync_node_marker_config_from_scenario_updates_marker_state(tmp_path):
    """Scenario YAML marker config is copied and relative mesh paths are resolved."""
    marker_mesh = tmp_path / "marker.obj"
    marker_mesh.write_text("# custom marker\n", encoding="utf-8")

    class DummySpin:
        def __init__(self):
            self.value = None
            self.signals_blocked = False

        def blockSignals(self, blocked):
            previous = self.signals_blocked
            self.signals_blocked = bool(blocked)
            return previous

        def setValue(self, value):
            self.value = value

    marker_cache = {"old": object()}
    viz = SimpleNamespace(
        node_marker_config={},
        node_service=SimpleNamespace(_node_marker_payload_cache=marker_cache),
        tx_marker_size=0.3,
        rx_marker_size=0.3,
        tx_marker_size_spin=DummySpin(),
        rx_marker_size_spin=DummySpin(),
    )
    loader = ScenarioLoaderService(
        viz,
        DummyDialogManager(),
        project_root_resolver=lambda: tmp_path,
    )
    scenario = SimpleNamespace(
        root=tmp_path,
        visualizer_cfg={
            "node_markers": {
                "default": {"shape": "sphere", "center": True},
                "tx": {"shape": "box", "size": 0.45},
                "rx": {
                    "shape": "mesh",
                    "mesh_path": "marker.obj",
                    "marker_size": 0.25,
                    "scale": 0.5,
                },
            }
        },
    )

    loader._sync_node_marker_config_from_scenario(scenario)

    assert viz.node_marker_config["tx"]["shape"] == "box"
    assert viz.node_marker_config["rx"]["mesh_path"] == str(marker_mesh.resolve(strict=False))
    assert viz.tx_marker_size == 0.45
    assert viz.rx_marker_size == 0.25
    assert viz.tx_marker_size_spin.value == 0.45
    assert viz.rx_marker_size_spin.value == 0.25
    assert marker_cache == {}


def test_sync_node_marker_config_resets_previous_scenario_defaults():
    marker_cache = {"old": object()}
    viz = SimpleNamespace(
        node_marker_config={"tx": {"shape": "box"}},
        node_service=SimpleNamespace(_node_marker_payload_cache=marker_cache),
        tx_marker_size=2.0,
        rx_marker_size=3.0,
        tx_marker_size_spin=Mock(),
        rx_marker_size_spin=Mock(),
    )
    loader = ScenarioLoaderService(viz, DummyDialogManager())

    loader._sync_node_marker_config_from_scenario(
        SimpleNamespace(root=Path("."), visualizer_cfg={})
    )

    assert viz.node_marker_config == {"default": {"shape": "sphere", "center": True}}
    assert viz.tx_marker_size == DEFAULT_NODE_MARKER_SIZE_M
    assert viz.rx_marker_size == DEFAULT_NODE_MARKER_SIZE_M
    assert marker_cache == {}


def test_sync_node_marker_config_rejects_values_outside_shared_ui_bounds(qapp):
    """Scenario values cannot diverge from the bounded controls that persist them."""
    tx_spin = QDoubleSpinBox()
    rx_spin = QDoubleSpinBox()
    tx_spin.setRange(*NODE_MARKER_SIZE_BOUNDS_M)
    rx_spin.setRange(*NODE_MARKER_SIZE_BOUNDS_M)
    viz = SimpleNamespace(
        node_marker_config={},
        node_service=SimpleNamespace(_node_marker_payload_cache={}),
        tx_marker_size=42.0,
        rx_marker_size=42.0,
        tx_marker_size_spin=tx_spin,
        rx_marker_size_spin=rx_spin,
    )
    loader = ScenarioLoaderService(viz, DummyDialogManager())
    minimum, maximum = NODE_MARKER_SIZE_BOUNDS_M
    scenario = SimpleNamespace(
        root=Path("."),
        visualizer_cfg={
            "node_markers": {
                "tx": {"size": maximum * 2.0},
                "rx": {"size": minimum / 2.0},
            }
        },
    )

    loader._sync_node_marker_config_from_scenario(scenario)

    assert viz.tx_marker_size == DEFAULT_NODE_MARKER_SIZE_M
    assert viz.rx_marker_size == DEFAULT_NODE_MARKER_SIZE_M
    assert tx_spin.value() == DEFAULT_NODE_MARKER_SIZE_M
    assert rx_spin.value() == DEFAULT_NODE_MARKER_SIZE_M


def test_load_scene_xml_uses_composed_main_controller(tmp_path):
    """Scenario loading routes XML scene work through the composed controller."""
    dialog = DummyDialogManager()
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>", encoding="utf-8")
    calls = []

    main_controller = SimpleNamespace(
        load_scene=lambda path, *, render_immediately, cleanup_first: calls.append(
            (path, render_immediately, cleanup_first)
        )
    )
    viz = SimpleNamespace(
        main_controller=main_controller,
    )
    loader = ScenarioLoaderService(
        viz,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    loader._load_scene_xml(xml_path)

    assert calls == [(xml_path, False, True)]


def test_local_scene_resolution_fails_then_succeeds_when_xml_appears(tmp_path):
    """A missing scene must fail instead of committing an empty scenario."""
    calls = []
    main_controller = SimpleNamespace(
        load_scene=lambda path, *, render_immediately, cleanup_first: calls.append(
            (path, render_immediately, cleanup_first)
        )
    )
    loader = ScenarioLoaderService(
        SimpleNamespace(main_controller=main_controller),
        DummyDialogManager(),
        project_root_resolver=lambda: tmp_path,
    )
    scenario = SimpleNamespace(
        root=tmp_path,
        scene_spec={"source": "local", "id": "scene.xml"},
    )
    app_config = SimpleNamespace(scenes=tmp_path)

    with pytest.raises(FileNotFoundError, match="Could not resolve scene XML"):
        loader._load_scene_from_scenario(scenario, app_config)

    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>", encoding="utf-8")
    loader._load_scene_from_scenario(scenario, app_config)

    assert calls == [(xml_path, False, True)]


def test_load_scene_xml_installs_preflighted_payload(tmp_path):
    xml_path = tmp_path / "scene.xml"
    xml_root = object()
    mesh_entries = [{"name": "mesh"}]
    load_prepared = Mock()
    viz = SimpleNamespace(
        main_controller=SimpleNamespace(
            load_prepared_scene=load_prepared,
            load_scene=Mock(),
        )
    )
    loader = ScenarioLoaderService(viz, DummyDialogManager())

    loader._load_scene_xml(
        xml_path,
        xml_root=xml_root,
        mesh_entries=mesh_entries,
        cleanup_first=False,
    )

    load_prepared.assert_called_once_with(
        xml_path,
        xml_root,
        mesh_entries,
        render_immediately=False,
        cleanup_first=False,
    )
    viz.main_controller.load_scene.assert_not_called()


def test_preflight_validates_local_inputs_without_mutating_visualizer(tmp_path, monkeypatch):
    """Preflight resolves YAML, source config, and XML before app teardown."""
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "timeline:",
                "  steps: 1",
                "  duration_s: 0.0",
                "scene:",
                "  source: local",
                "  id: scene.xml",
                "data:",
                "  mode: files",
                "  files:",
                "    format: h5",
            ]
        ),
        encoding="utf-8",
    )
    scene_xml = scenario_dir / "scene.xml"
    scene_xml.write_text("<scene version='2'></scene>", encoding="utf-8")
    active_scenario = object()
    visualizer = SimpleNamespace(
        scenario=active_scenario,
        current_scenario_path="active-scenario",
        ready=True,
    )
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    def _unexpected_frame_source_construction(_scenario):
        raise AssertionError("preflight must not construct a frame source")

    monkeypatch.setattr(
        "visualizer.src.services.scenario_loader_service.make_frame_source",
        _unexpected_frame_source_construction,
    )

    result = loader.preflight_scenario("scenario_one")

    assert result is not None
    assert result.scenario.root == scenario_dir.resolve()
    assert result.scene_xml_path == scene_xml
    assert result.scene_xml_root.tag == "scene"
    assert result.scene_mesh_entries == []
    assert visualizer.scenario is active_scenario
    assert visualizer.current_scenario_path == "active-scenario"
    assert visualizer.ready is True
    assert dialog.messages == []


def test_prepare_frame_source_probes_first_local_frame_without_mutating_visualizer(
    monkeypatch,
):
    active_scenario = object()
    active_source = object()
    visualizer = SimpleNamespace(
        scenario=active_scenario,
        frame_source=active_source,
    )
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(visualizer, dialog)
    candidate = SimpleNamespace(
        open=Mock(),
        list_frames=Mock(return_value=[9, 3, 7]),
        load_frame=Mock(return_value=object()),
        close=Mock(),
    )
    monkeypatch.setattr(
        "visualizer.src.services.scenario_loader_service.make_frame_source",
        Mock(return_value=candidate),
    )
    preflight = SimpleNamespace(
        scenario=SimpleNamespace(data_mode="files"),
    )

    prepared = loader.prepare_frame_source(preflight)

    assert prepared is not None
    assert prepared.frame_source is candidate
    assert prepared.frame_source_ready is True
    assert prepared.available_frames == (3, 7, 9)
    candidate.open.assert_called_once_with()
    candidate.load_frame.assert_called_once_with(3)
    candidate.close.assert_not_called()
    assert visualizer.scenario is active_scenario
    assert visualizer.frame_source is active_source
    assert dialog.messages == []


def test_prepare_frame_source_failure_closes_candidate_and_preserves_visualizer(
    monkeypatch,
):
    active_scenario = object()
    active_source = object()
    visualizer = SimpleNamespace(
        scenario=active_scenario,
        frame_source=active_source,
    )
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(visualizer, dialog)
    candidate = SimpleNamespace(
        open=Mock(side_effect=OSError("server unavailable")),
        close=Mock(),
    )
    monkeypatch.setattr(
        "visualizer.src.services.scenario_loader_service.make_frame_source",
        Mock(return_value=candidate),
    )

    prepared = loader.prepare_frame_source(
        SimpleNamespace(scenario=SimpleNamespace(data_mode="remote_hdf5"))
    )

    assert prepared is None
    candidate.close.assert_called_once_with()
    assert visualizer.scenario is active_scenario
    assert visualizer.frame_source is active_source
    assert dialog.messages[-1][0] == "error"
    assert loader.last_frame_source_preparation_error == (
        "Failed to prepare scenario data source: server unavailable"
    )


def test_prepared_frame_source_closes_on_unexpected_scene_install_failure(tmp_path):
    candidate = SimpleNamespace(close=Mock())
    visualizer = SimpleNamespace(
        progress=SimpleNamespace(note=Mock(), task=lambda _message: nullcontext()),
        playback_cadence=None,
        ui_controller=SimpleNamespace(update_file_source_summary=Mock()),
        set_state=Mock(),
        node_service=SimpleNamespace(_node_marker_payload_cache={}),
        frame_source=None,
    )
    loader = ScenarioLoaderService(visualizer, DummyDialogManager())
    loader._load_scene_xml = Mock(side_effect=TypeError("unexpected scene install failure"))
    scenario = SimpleNamespace(
        root=tmp_path,
        debug_level="INFO",
        actors=ActorsSpec(),
        timeline=TimelineSpec(steps=1, duration_s=0.0),
        raytracing={},
        visualizer_cfg={},
    )
    preflight = SimpleNamespace(
        scenario=scenario,
        app_config=object(),
        path_policy=SimpleNamespace(project_root=tmp_path, config_dir=tmp_path),
        scene_xml_path=tmp_path / "scene.xml",
        scene_xml_root=object(),
        scene_mesh_entries=[],
    )
    prepared = ScenarioFrameSourcePreparation(candidate, True, (0,))

    with pytest.raises(TypeError, match="unexpected scene install failure"):
        loader.load_scenario(
            str(tmp_path),
            preflight=preflight,
            prepared_frame_source=prepared,
        )

    candidate.close.assert_called_once_with()
    assert visualizer.frame_source is None


def test_preflight_rejects_missing_scene_without_mutating_visualizer(tmp_path):
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "timeline:",
                "  steps: 1",
                "  duration_s: 0.0",
                "scene:",
                "  source: local",
                "  id: missing.xml",
                "data:",
                "  mode: files",
            ]
        ),
        encoding="utf-8",
    )
    active_scenario = object()
    visualizer = SimpleNamespace(scenario=active_scenario, ready=True)
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.preflight_scenario("scenario_one") is None
    assert visualizer.scenario is active_scenario
    assert visualizer.ready is True
    assert dialog.messages[-1][0] == "error"


def test_preflight_rejects_non_mapping_yaml_without_mutating_visualizer(tmp_path):
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    active_scenario = object()
    visualizer = SimpleNamespace(scenario=active_scenario, ready=True)
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.preflight_scenario("scenario_one") is None
    assert visualizer.scenario is active_scenario
    assert visualizer.ready is True
    assert "must be a mapping" in dialog.messages[-1][2]


def test_preflight_rejects_missing_mesh_without_mutating_visualizer(tmp_path):
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "timeline:",
                "  steps: 1",
                "  duration_s: 0.0",
                "scene:",
                "  source: local",
                "  id: scene.xml",
                "data:",
                "  mode: files",
            ]
        ),
        encoding="utf-8",
    )
    (scenario_dir / "scene.xml").write_text(
        """<scene version="2">
  <bsdf type="diffuse" id="material">
    <rgb name="reflectance" value="0.5,0.5,0.5"/>
  </bsdf>
  <shape type="ply">
    <string name="filename" value="missing.ply"/>
    <ref id="material"/>
  </shape>
</scene>
""",
        encoding="utf-8",
    )
    active_scenario = object()
    visualizer = SimpleNamespace(scenario=active_scenario, ready=True)
    loader = ScenarioLoaderService(
        visualizer,
        DummyDialogManager(),
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.preflight_scenario("scenario_one") is None
    assert visualizer.scenario is active_scenario
    assert visualizer.ready is True


def test_preflight_reports_unsupported_mesh_transform_without_mutating_visualizer(tmp_path):
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        """schema_version: 2
timeline:
  steps: 1
  duration_s: 0.0
scene:
  source: local
  id: scene.xml
data:
  mode: files
""",
        encoding="utf-8",
    )
    (scenario_dir / "scene.xml").write_text(
        """<scene version="2">
  <bsdf type="diffuse" id="material"/>
  <shape type="ply" id="matrix-wall">
    <string name="filename" value="missing.ply"/>
    <ref id="material"/>
    <transform name="to_world"><matrix value="1 0 0 0 1 0 0 0 1"/></transform>
  </shape>
</scene>
""",
        encoding="utf-8",
    )
    active_scenario = object()
    visualizer = SimpleNamespace(scenario=active_scenario, ready=True)
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.preflight_scenario("scenario_one") is None
    assert visualizer.scenario is active_scenario
    assert visualizer.ready is True
    assert dialog.messages[-1][0] == "error"
    assert "Unsupported lightweight mesh transform" in dialog.messages[-1][2]
    assert "matrix-wall" in dialog.messages[-1][2]


def test_preflight_rejects_malformed_scene_xml_without_mutating_visualizer(tmp_path):
    scenario_dir = tmp_path / "scenario_one"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "timeline:",
                "  steps: 1",
                "  duration_s: 0.0",
                "scene:",
                "  source: local",
                "  id: scene.xml",
                "data:",
                "  mode: files",
            ]
        ),
        encoding="utf-8",
    )
    (scenario_dir / "scene.xml").write_text("<scene>", encoding="utf-8")
    active_scenario = object()
    visualizer = SimpleNamespace(scenario=active_scenario, ready=True)
    dialog = DummyDialogManager()
    loader = ScenarioLoaderService(
        visualizer,
        dialog,
        project_root_resolver=lambda: tmp_path,
    )

    assert loader.preflight_scenario("scenario_one") is None
    assert visualizer.scenario is active_scenario
    assert visualizer.ready is True
    assert dialog.messages[-1][0] == "error"


def test_animation_timing_uses_parsed_scenario_without_reopening_yaml():
    """Animation timing comes from the canonical in-memory scenario model."""
    update_total_steps = Mock()
    viz = SimpleNamespace(
        total_animation_steps=1,
        _frame_duration=0.0,
        _mesh_update_interval_s=0.0,
        ui_manager=SimpleNamespace(update_total_steps=update_total_steps),
    )
    loader = ScenarioLoaderService(
        SimpleNamespace(),
        DummyDialogManager(),
    )
    scenario = SimpleNamespace(
        timeline=TimelineSpec(steps=24, duration_s=2.5),
        raytracing={"mesh_update_interval_s": 0.2},
    )

    loader._update_animation_steps_from_scenario(scenario, viz)

    assert viz.total_animation_steps == 24
    assert viz._frame_duration == 2.5
    assert viz._mesh_update_interval_s == 0.2
    update_total_steps.assert_called_once_with(24)


def test_frame_source_open_failure_is_not_downgraded_to_scene_only():
    loader = ScenarioLoaderService(SimpleNamespace(), DummyDialogManager())
    source = SimpleNamespace(open=Mock(side_effect=OSError("server unavailable")))

    with pytest.raises(OSError, match="server unavailable"):
        loader._setup_frame_source(source)
