from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.controllers.camera_controller import CameraController
from visualizer.src.services.camera_scene_query_service import CameraSceneQueryService

try:
    from visualizer.src.controllers.ui_controller import UIController

    UI_CONTROLLER_AVAILABLE = True
except ImportError:
    UI_CONTROLLER_AVAILABLE = False

try:
    from visualizer.src.renderers.open3d.renderer import Open3DRenderer

    OPEN3D_RENDERER_AVAILABLE = True
except ImportError:
    OPEN3D_RENDERER_AVAILABLE = False


class _FakeCamera:
    def __init__(self, eye: np.ndarray, lookat: np.ndarray, fov: float = 60.0) -> None:
        self.eye = np.asarray(eye, dtype=np.float64)
        self.lookat = np.asarray(lookat, dtype=np.float64)
        self.fov = fov

    def get_field_of_view(self) -> float:
        return self.fov

    def get_view_matrix(self) -> np.ndarray:
        forward = self.lookat - self.eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward = forward / forward_norm

        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(forward, up)) > 0.999:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-9)
        true_up = np.cross(right, forward)
        true_up = true_up / (np.linalg.norm(true_up) + 1e-9)

        view = np.eye(4, dtype=np.float64)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ self.eye
        return view


class _FakeO3DVisualizer:
    def __init__(self, camera: _FakeCamera) -> None:
        self.scene = SimpleNamespace(camera=camera)
        self.calls: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    def setup_camera(
        self, fov: float, lookat: list[float], eye: list[float], up: list[float]
    ) -> None:
        lookat_arr = np.asarray(lookat, dtype=np.float64)
        eye_arr = np.asarray(eye, dtype=np.float64)
        up_arr = np.asarray(up, dtype=np.float64)
        self.calls.append((float(fov), lookat_arr, eye_arr, up_arr))
        self.scene.camera.fov = float(fov)
        self.scene.camera.lookat = lookat_arr
        self.scene.camera.eye = eye_arr


def _make_controller(viz) -> CameraController:
    return CameraController(viz, CameraSceneQueryService(viz))


@pytest.mark.skipif(
    not OPEN3D_RENDERER_AVAILABLE, reason="Open3D renderer dependencies unavailable"
)
def test_set_lookat_bootstraps_non_degenerate_follow_from_pov_like_state() -> None:
    target = np.array([4.0, -2.0, 1.0], dtype=np.float64)
    camera = _FakeCamera(eye=target.copy(), lookat=target.copy())
    o3d_vis = _FakeO3DVisualizer(camera)

    renderer = Open3DRenderer.__new__(Open3DRenderer)
    renderer._o3d_vis = o3d_vis
    renderer._stored_lookat = None
    renderer._last_camera_look_distance = 10.0
    renderer._set_far_clipping_plane = Mock()
    renderer._post_redraw = Mock()

    assert renderer.set_lookat(target)
    assert o3d_vis.calls, "setup_camera should be called"

    _, lookat, eye, _ = o3d_vis.calls[-1]
    eye_to_target = np.linalg.norm(lookat - eye)
    assert eye_to_target > 1.0
    assert np.allclose(renderer._stored_lookat, target)
    assert renderer._last_camera_look_distance > 1.0


@pytest.mark.skipif(
    not OPEN3D_RENDERER_AVAILABLE, reason="Open3D renderer dependencies unavailable"
)
def test_set_lookat_preserves_radius_across_follow_updates() -> None:
    camera = _FakeCamera(eye=np.array([0.0, -10.0, 5.0]), lookat=np.array([0.0, 0.0, 0.0]))
    o3d_vis = _FakeO3DVisualizer(camera)

    renderer = Open3DRenderer.__new__(Open3DRenderer)
    renderer._o3d_vis = o3d_vis
    renderer._stored_lookat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    renderer._last_camera_look_distance = np.linalg.norm(camera.lookat - camera.eye)
    renderer._set_far_clipping_plane = Mock()
    renderer._post_redraw = Mock()

    targets = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 0.0, 0.0], dtype=np.float64),
        np.array([3.0, 0.0, 0.0], dtype=np.float64),
    ]
    expected_radius = np.linalg.norm(np.array([0.0, -10.0, 5.0], dtype=np.float64))

    for target in targets:
        assert renderer.set_lookat(target)
        actual_radius = np.linalg.norm(o3d_vis.scene.camera.eye - target)
        assert actual_radius == pytest.approx(expected_radius, rel=1e-6, abs=1e-6)


def test_focus_on_target_updates_last_follow_position_only_on_success() -> None:
    renderer = SimpleNamespace(
        renderer_type="open3d",
        reset_follow_state=Mock(),
        focus_camera=Mock(return_value=False),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
    )
    controller = _make_controller(viz)
    controller.scene_query.get_focus_position = Mock(return_value=[1.0, 2.0, 3.0])

    controller.focus_on_target()
    assert controller._last_follow_target_pos is None

    renderer.focus_camera.return_value = True
    controller.focus_on_target()
    assert np.allclose(controller._last_follow_target_pos, np.array([1.0, 2.0, 3.0]))


def test_update_follow_camera_focus_skips_updates_outside_follow_mode() -> None:
    renderer = SimpleNamespace(
        renderer_type="open3d",
        reset_follow_state=Mock(),
        update_follow_camera=Mock(return_value=True),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=SimpleNamespace(camera_mode="pov"),
    )
    controller = _make_controller(viz)
    controller.scene_query.get_focus_position = Mock(return_value=[1.0, 2.0, 3.0])

    controller.update_follow_camera_focus()
    renderer.update_follow_camera.assert_not_called()
    renderer.reset_follow_state.assert_called_once()

    viz.app_state.camera_mode = "follow"
    controller.update_follow_camera_focus()
    renderer.update_follow_camera.assert_called_once()


def test_set_pov_camera_bootstraps_frame_data_when_entity_not_ready() -> None:
    renderer = SimpleNamespace(
        renderer_type="open3d",
        _frame_update_in_progress=False,
        set_pov_camera=Mock(return_value=True),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=SimpleNamespace(pov_axis="forward", step=7),
        animation_step=7,
        _process_frame_step=Mock(),
    )
    controller = _make_controller(viz)
    controller._hide_pov_entity = Mock()
    controller.scene_query.get_entity_position_orientation_and_info = Mock(
        side_effect=[
            (None, None, None),
            ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], {"type": "tx", "index": 0}),
        ]
    )

    controller.set_pov_camera()

    viz._process_frame_step.assert_called_once_with(7)
    renderer.set_pov_camera.assert_called_once_with(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        "forward",
        defer_redraw=False,
    )


def test_set_pov_camera_does_not_bootstrap_during_frame_update() -> None:
    renderer = SimpleNamespace(
        renderer_type="open3d",
        _frame_update_in_progress=True,
        set_pov_camera=Mock(),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=SimpleNamespace(pov_axis="forward", step=3),
        animation_step=3,
        _process_frame_step=Mock(),
    )
    controller = _make_controller(viz)
    controller.scene_query.get_entity_position_orientation_and_info = Mock(
        return_value=(None, None, None)
    )

    controller.set_pov_camera()

    viz._process_frame_step.assert_not_called()
    renderer.set_pov_camera.assert_not_called()


@pytest.mark.skipif(not UI_CONTROLLER_AVAILABLE, reason="UI controller dependencies unavailable")
def test_handle_target_focus_changed_routes_pov_and_follow_paths() -> None:
    camera = SimpleNamespace(
        remember_focus_selection=Mock(),
        update_follow_camera_focus=Mock(),
        set_pov_camera=Mock(),
    )
    viz = SimpleNamespace(app_state=SimpleNamespace(camera_mode="pov"), camera_controller=camera)

    ui_controller = UIController.__new__(UIController)
    ui_controller.visualizer = viz

    ui_controller.handle_target_focus_changed("Target 1")
    camera.remember_focus_selection.assert_called_once()
    camera.set_pov_camera.assert_called_once()
    camera.update_follow_camera_focus.assert_not_called()

    camera.remember_focus_selection.reset_mock()
    camera.update_follow_camera_focus.reset_mock()
    camera.set_pov_camera.reset_mock()
    viz.app_state.camera_mode = "follow"

    ui_controller.handle_target_focus_changed("Target 1")
    camera.remember_focus_selection.assert_called_once()
    camera.update_follow_camera_focus.assert_called_once()
    camera.set_pov_camera.assert_not_called()
