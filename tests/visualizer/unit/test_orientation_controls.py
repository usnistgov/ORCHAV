"""Independent single-actor orientation guides, persistence and frame changes."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.app.scenario_workflow import _clear_scenario_identity
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.orientation_controls import OrientationControls
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.aperture_service import ApertureService
from visualizer.src.state import AppState, create_initial_state
from visualizer.src.types.orientation_reference import OrientationReferenceSettings


@pytest.fixture
def rig(qtbot):
    viz = SimpleNamespace(
        app_state=create_initial_state(camera_mode="follow"),
        current_tx_positions=[[1, 2, 3], [3, 4, 5]],
        current_rx_positions=[[6, 7, 8]],
        current_tx_orientations=[[0, 0, 0], [0, 0, 0]],
        current_rx_orientations=[[0, np.pi / 4, 0]],
        target_entries=[
            {
                "name": "Car",
                "stable_target_id": "car-a",
                "position": [10, 0, 1],
                "orientation_radians": [np.pi / 2, 0, 0],
            },
            {
                "name": "Car",
                "stable_target_id": "car-b",
                "position": [20, 0, 1],
                "orientation_radians": [0, 0, 0],
            },
        ],
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(aperture_preview=True, angular_preview=True),
            ensure_object=Mock(return_value=True),
            remove_object=Mock(return_value=True),
            request_redraw=Mock(),
        ),
        vis_initialized=True,
    )
    viz.set_state = lambda **changes: setattr(viz, "app_state", replace(viz.app_state, **changes))
    viz.ui_controller = UIController.__new__(UIController)
    viz.ui_controller.visualizer = viz
    viz.aperture_service = ApertureService(viz)
    controls = OrientationControls(viz)
    qtbot.addWidget(controls)
    return viz, controls


def test_one_actor_selector_keeps_context_and_camera_independent(rig):
    viz, controls = rig
    assert controls.actor.count() == 5
    assert len({controls.actor.itemText(i) for i in range(5)}) == 5
    assert controls.actor.currentIndex() == -1
    assert not controls.local_frame.isEnabled()
    controls.actor.setCurrentIndex(controls.actor.findData("target:car-a"))
    controls.global_frame.setChecked(True)
    controls.local_frame.setChecked(True)
    assert (viz.app_state.selected_tx, viz.app_state.selected_rx, viz.app_state.camera_mode) == (
        "all",
        "all",
        "follow",
    )
    active = viz.aperture_service._active_reference_geometries
    assert len(active) == 14
    assert all("target_car-a" in name for name in active)
    assert "Yaw 90.0°" in controls.status.text()
    assert viz.app_state.aperture_radius_m == 5
    controls.radius.setValue(2)
    assert viz.app_state.orientation_reference.radius_m == 2
    assert viz.app_state.aperture_radius_m == 5
    local = active["angular_reference_local_target_car-a"].payload
    global_frame = active["angular_reference_global_target_car-a"].payload
    np.testing.assert_allclose(local.points[local.lines[-6, 1]], [10, 1.8, 1], atol=1e-10)
    np.testing.assert_allclose(
        global_frame.points[global_frame.lines[-6, 1]], [12.34, 0, 1], atol=1e-10
    )


def test_guides_follow_motion_reordering_and_remove_missing_actor(rig):
    viz, controls = rig
    controls.actor.setCurrentIndex(controls.actor.findData("target:car-a"))
    controls.local_frame.setChecked(True)
    target = viz.target_entries[0]
    target.update(position=[12, 3, 1], orientation_radians=[0, 0, 0])
    viz.target_entries.reverse()
    controls.sync_from_state()
    assert controls.actor.currentData() == "target:car-a"
    assert viz.aperture_service.update_apertures()
    active = viz.aperture_service._active_reference_geometries
    payload = active["angular_reference_local_target_car-a"].payload
    np.testing.assert_allclose(payload.points[0], [12, 3, 1])
    np.testing.assert_allclose(payload.points[payload.lines[-6, 1]], [16.5, 3, 1])
    viz.target_entries.remove(target)
    controls.sync_from_state()
    assert controls.actor.currentIndex() == -1
    assert viz.app_state.orientation_reference.actor == "target:car-a"
    assert "unavailable" in controls.status.text()
    assert viz.aperture_service.update_apertures()
    assert not active


def test_global_frame_remains_available_without_local_pose_and_stale_guides_clear(rig):
    viz, controls = rig
    controls.actor.setCurrentIndex(controls.actor.findData("rx:0"))
    controls.local_frame.setChecked(True)
    assert viz.aperture_service._active_reference_geometries
    viz.current_rx_orientations = []
    controls.sync_from_state()
    assert not controls.local_frame.isEnabled()
    assert controls.global_frame.isEnabled()
    assert viz.aperture_service.update_apertures()
    assert not viz.aperture_service._active_reference_geometries
    controls.global_frame.setChecked(True)
    assert all("global_rx_0" in name for name in viz.aperture_service._active_reference_geometries)
    viz.current_rx_positions = [[np.nan, 0, 0]]
    assert viz.aperture_service.update_apertures()
    assert not viz.aperture_service._active_reference_geometries


def test_frame_sync_preserves_popup_model_and_actor_change_replaces_previous_guides(rig):
    viz, controls = rig
    controls.actor.setCurrentIndex(controls.actor.findData("tx:1"))
    controls.global_frame.setChecked(True)
    reset = Mock()
    controls.actor.model().modelReset.connect(reset)
    viz.current_tx_positions[1] = [4, 5, 6]
    controls.sync_from_state()
    reset.assert_not_called()
    controls.actor.setCurrentIndex(controls.actor.findData("rx:0"))
    assert all("global_rx_0" in name for name in viz.aperture_service._active_reference_geometries)
    controls.global_frame.setChecked(False)
    assert not viz.aperture_service._active_reference_geometries


def test_orientation_settings_roundtrip_and_migrate_one_legacy_context_actor(rig):
    viz, controls = rig
    controls.actor.setCurrentIndex(controls.actor.findData("target:car-b"))
    controls.radius.setValue(7.25)
    controls.local_frame.setChecked(True)
    restored = AppState.from_dict(viz.app_state.to_dict())
    assert restored.orientation_reference == OrientationReferenceSettings("target:car-b", 7.25)
    assert restored.show_local_angular_reference
    old = viz.app_state.to_dict()
    old.pop("orientation_reference")
    old.update(selected_tx=1, selected_rx=0, aperture_radius_m=3.5)
    migrated = AppState.from_dict(old)
    assert migrated.orientation_reference == OrientationReferenceSettings("tx:1", 3.5)
    invalid = OrientationReferenceSettings.from_dict({"actor": "tx:-1", "radius_m": float("nan")})
    assert invalid == OrientationReferenceSettings()


def test_source_change_retires_actor_selection_and_unsupported_renderer_disables_guides(rig):
    viz, controls = rig
    controls.actor.setCurrentIndex(controls.actor.findData("target:car-a"))
    controls.radius.setValue(9)
    _clear_scenario_identity(viz)
    controls.sync_from_state()
    assert viz.app_state.orientation_reference == OrientationReferenceSettings(radius_m=9)
    assert controls.actor.currentIndex() == -1
    viz.renderer.capabilities = RendererCapabilities()
    controls.sync_from_state()
    assert not controls.actor.isEnabled()
    assert not controls.global_frame.isEnabled()
    assert "pygfx" in controls.status.text()
