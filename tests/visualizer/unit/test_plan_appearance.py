"""Plan reuses Builder's schematic, independently of downloaded map imagery."""

from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox, QWidget

from tests.visualizer.authoring.test_builder_scene_tools import graph_path as graph_path
from tests.visualizer.unit.test_appearance_presets import service as service
from visualizer.src.authoring.street_routes import StreetNetwork
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.materials.view_styles import PRESETS
from visualizer.src.panels.appearance_panel import AppearancePanel
from visualizer.src.scene.street_plan import load_street_polylines, schematic_payloads
from visualizer.src.scene.visibility_policy import effective_entry_visibility
from visualizer.src.services.plan_view_service import PlanViewService
from visualizer.src.services.session_service import SessionService
from visualizer.src.state import create_initial_state, update_state


def test_plan_and_builder_use_identical_curved_streets(graph_path):
    streets = load_street_polylines(graph_path)
    assert streets == tuple(StreetNetwork(graph_path, "pedestrian").edges())
    assert any(len(points) > 2 for points in streets)
    original = deepcopy(streets)
    grid, roads = schematic_payloads(((-10, -10), (40, 30)), streets)
    assert len(grid.lines) < 250
    np.testing.assert_allclose(roads.points[:, :2], [p[:2] for road in streets for p in road])
    np.testing.assert_allclose(roads.points[:, 2], 0.02)
    assert streets == original


@pytest.fixture
def plan(qapp, graph_path):
    viz = SimpleNamespace(
        renderer=SimpleNamespace(
            ensure_object=Mock(return_value=True),
            remove_object=Mock(),
            get_camera_state=Mock(return_value="old-camera"),
            set_camera_state=Mock(),
        ),
        mesh_entries=[
            {"visible": True},
            {"visible": False},
            {"visible": True, "_geographic_layer_visible": False},
        ],
        target_entries=[{"visible": True, "name": "actor"}],
        camera_controller=SimpleNamespace(set_overview_view=Mock()),
        object_appearance_service=SimpleNamespace(refresh_object_visibility_batch=Mock()),
        scene_service=SimpleNamespace(map_layer=SimpleNamespace(set_enabled=Mock())),
        app_state=create_initial_state(),
        trajectory_load_coordinator=SimpleNamespace(supports_source=Mock(return_value=True)),
    )
    viz.set_state = lambda **values: setattr(
        viz, "app_state", update_state(viz.app_state, **values)
    )
    viz.ui_controller = SimpleNamespace(
        visualizer=viz,
        _ensure_trajectory_loaded=Mock(),
        _remove_trajectory_geometry=Mock(),
        _hide_colorbar_if_all_off=Mock(),
        _render_trajectory_if_enabled=Mock(),
        _set_trajectory_status=Mock(),
        _update_trajectory_colorbar=Mock(),
    )
    for kind in ("tx", "rx", "target"):
        name = f"handle_{kind}_trajectory_toggled"
        handler = MethodType(getattr(UIController, name), viz.ui_controller)
        setattr(viz.ui_controller, name, handler)
        checkbox = QCheckBox()
        checkbox.toggled.connect(handler)
        setattr(viz, f"{kind}_trajectory_cb", checkbox)
    result = PlanViewService(viz)
    viz.scene_service.plan_view = result
    result.set_scene(graph_path.with_name("scene.xml"))
    yield result
    result.close()


def test_plan_with_local_graph_preserves_camera_and_visibility(plan, qtbot):
    entries = deepcopy(plan.viz.mesh_entries)
    actors = deepcopy(plan.viz.target_entries)
    plan.set_enabled(True)
    qtbot.waitUntil(lambda: not plan._loading)
    assert plan.enabled and plan.available
    assert all(not effective_entry_visibility(entry) for entry in plan.viz.mesh_entries)
    assert plan.viz.target_entries == actors
    plan.viz.scene_service.map_layer.set_enabled.assert_not_called()
    plan.viz.camera_controller.set_overview_view.assert_not_called()
    shown = plan.viz.renderer.ensure_object.call_args.args[0]
    assert shown.id == "plan:streets" and shown.visible and not shown.metadata["pickable"]
    payload = shown.payload
    plan.set_enabled(False)
    assert [effective_entry_visibility(e) for e in plan.viz.mesh_entries] == [True, False, False]
    assert [e["visible"] for e in plan.viz.mesh_entries] == [e["visible"] for e in entries]
    plan.viz.renderer.get_camera_state.assert_not_called()
    plan.viz.renderer.set_camera_state.assert_not_called()
    plan.set_enabled(True)
    assert plan.viz.renderer.ensure_object.call_args.args[0].payload is payload


def test_missing_graph_cannot_hide_environment_or_enable_trajectories(plan):
    plan.set_scene(None)
    before = deepcopy(plan.viz.mesh_entries)
    plan.set_enabled(True)
    assert not plan.available and not plan.enabled
    assert plan.viz.mesh_entries == before
    plan.viz.renderer.ensure_object.assert_not_called()
    plan.viz.ui_controller._ensure_trajectory_loaded.assert_not_called()


def test_plan_choice_follows_scene_capability_without_manual_refresh(
    plan, service, graph_path, qtbot
):
    service.viz.scene_service = plan.viz.scene_service
    service.viz.appearance_preset_service = service
    plan.viz.appearance_preset_service = service
    manager = SimpleNamespace(
        panels={
            "materials": SimpleNamespace(
                surface_controls=QWidget(), sync_procedural_controls=Mock()
            ),
            "render": SimpleNamespace(_sync_from_visualizer=Mock()),
        },
        sections={},
    )
    panel = AppearancePanel(service.viz, manager)
    qtbot.addWidget(panel.create_panel())
    try:
        assert panel.combo.findText("Plan") >= 0
        service.apply("Plan")
        qtbot.waitUntil(lambda: not plan._loading)
        saved = service.capture()
        plan.set_scene(None)
        assert panel.combo.findText("Plan") == -1
        assert not plan.enabled
        assert panel.combo.currentText() != "Plan"
        assert not panel.widgets["Reset"].isEnabled()
        # A saved workspace/recipe must not bypass the same capability check.
        service.restore(saved, name="Plan")
        assert saved["plan_view"] is True
        assert not plan.enabled and not service.capture()["plan_view"]
        assert "3D environment" in service.notice
        assert all(e.get("_plan_view_visible", True) for e in plan.viz.mesh_entries)
        plan.set_scene(graph_path.with_name("scene.xml"))
        assert panel.combo.findText("Plan") >= 0
    finally:
        panel.cleanup()


@pytest.mark.parametrize("look", [name for name in PRESETS if name != "Plan"])
def test_each_appearance_restores_mixed_trajectory_visibility(plan, service, look):
    service.viz.scene_service = plan.viz.scene_service
    # Exercise the actual trajectory handlers and checkbox signals.
    plan.viz.rx_trajectory_cb.setChecked(True)
    previous = plan._trajectory_visibility()
    service.apply("Plan")
    assert plan._trajectory_visibility() == dict.fromkeys(("tx", "rx", "target"), True)
    assert all(getattr(plan.viz, f"{kind}_trajectory_cb").isChecked() for kind in previous)
    assert {c.args[0] for c in plan.viz.ui_controller._ensure_trajectory_loaded.call_args_list} == {
        "tx",
        "rx",
        "target",
    }
    # Reapplying Plan must not overwrite the preferences saved on entry.
    service.apply("Plan")
    service.apply(look)
    assert plan._trajectory_visibility() == previous
    assert {
        kind: getattr(plan.viz, f"{kind}_trajectory_cb").isChecked() for kind in previous
    } == previous
    # A load completing after leaving Plan must respect restored visibility.
    controller = plan.viz.ui_controller
    UIController._on_trajectory_loaded(controller, SimpleNamespace(frames_loaded=[0, 1]))
    assert [c.args[0] for c in controller._render_trajectory_if_enabled.call_args_list] == ["rx"]


def test_plan_without_recorded_frames_does_not_request_trajectories(plan):
    plan.viz.trajectory_load_coordinator.supports_source.return_value = False
    plan.set_enabled(True)
    plan.viz.ui_controller._ensure_trajectory_loaded.assert_not_called()
    assert plan.capture_trajectory_state() is None


def test_scene_change_restores_preferences_without_loading_retired_source(plan):
    plan.viz.rx_trajectory_cb.setChecked(True)
    plan.set_enabled(True)
    plan.viz.ui_controller._ensure_trajectory_loaded.reset_mock()
    plan.set_scene(None)
    assert plan._trajectory_visibility() == {"tx": False, "rx": True, "target": False}
    plan.viz.ui_controller._ensure_trajectory_loaded.assert_not_called()


def test_workspace_retains_pre_plan_preferences_and_in_plan_edits(plan, service):
    plan.viz.rx_trajectory_cb.setChecked(True)
    service.viz.scene_service = plan.viz.scene_service
    service.apply("Plan")
    plan.viz.tx_trajectory_cb.setChecked(False)
    plan_state = plan.capture_trajectory_state()
    appearance = service.capture()
    # The same reset precedes AppState restoration when loading a workspace.
    SessionService(plan.viz)._reset_transient_appearance_state()
    service.restore(appearance)
    plan.restore_trajectory_state(plan_state)
    assert plan._trajectory_visibility() == {"tx": False, "rx": True, "target": True}
    service.apply("Detailed")
    assert plan._trajectory_visibility() == {"tx": False, "rx": True, "target": False}


def test_async_graph_is_shared_cached_and_stale_results_are_discarded(plan, graph_path, qtbot):
    plan.set_enabled(True)
    assert effective_entry_visibility(plan.viz.mesh_entries[0])
    qtbot.waitUntil(lambda: not plan._loading)
    assert plan._streets == load_street_polylines(graph_path)
    payloads = plan._payloads
    plan.set_enabled(False)
    plan.set_enabled(True)
    assert plan._payloads is payloads and not plan._loading
    generation = plan._generation
    plan.set_scene(None)
    plan._prepared(generation, load_street_polylines(graph_path))
    assert plan._payloads is None and not plan.enabled


def test_bad_graph_keeps_environment_and_restores_trajectories(plan, graph_path, qtbot):
    plan.viz.rx_trajectory_cb.setChecked(True)
    graph_path.write_text("broken")
    plan.set_enabled(True)
    qtbot.waitUntil(lambda: not plan._loading)
    assert not plan.enabled and not plan.available and "unavailable" in plan.status
    assert effective_entry_visibility(plan.viz.mesh_entries[0])
    assert plan._trajectory_visibility() == {"tx": False, "rx": True, "target": False}


def test_empty_graph_disables_plan_without_hiding_environment(plan, monkeypatch, qtbot):
    monkeypatch.setattr(
        "visualizer.src.services.plan_view_service.load_street_polylines", lambda _: ()
    )
    plan.set_enabled(True)
    qtbot.waitUntil(lambda: not plan._loading)
    assert not plan.enabled and not plan.available
    assert effective_entry_visibility(plan.viz.mesh_entries[0])


def test_appearance_plan_roundtrip_leaves_downloaded_map_independent(service):
    plan = SimpleNamespace(enabled=False, available=True)

    def enable(value):
        plan.enabled = value

    plan.set_enabled = Mock(side_effect=enable)
    map_layer = SimpleNamespace(enabled=True, set_enabled=Mock())
    service.viz.scene_service = SimpleNamespace(plan_view=plan, map_layer=map_layer)
    service.apply("Detailed")
    service.apply("Plan")
    assert service.active_name == "Plan" and plan.enabled
    assert service.capture()["plan_view"] is True
    assert map_layer.enabled
    map_layer.set_enabled.assert_not_called()
    service.save("My plan")
    service.apply("Detailed")
    assert not plan.enabled
    service.apply("My plan")
    assert plan.enabled
    service.restore(service.previous)
    assert not plan.enabled


def test_legacy_appearance_recipe_remains_3d(service):
    plan = SimpleNamespace(enabled=True, set_enabled=Mock())
    service.viz.scene_service = SimpleNamespace(plan_view=plan)
    data = service.capture()
    data.pop("plan_view")
    service.restore(data)
    plan.set_enabled.assert_called_once_with(False)
