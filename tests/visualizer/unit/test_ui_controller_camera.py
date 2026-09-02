from __future__ import annotations

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.state import create_initial_state, update_state


class _DummyToggle:
    def __init__(self, checked: bool = False):
        self._checked = bool(checked)
        self.visible = True

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def blockSignals(self, _blocked: bool) -> None:
        return None

    def setVisible(self, visible: bool) -> None:
        self.visible = bool(visible)


class _DummyRenderer:
    def __init__(self) -> None:
        self.fly_mode_calls: list[bool] = []
        self.reset_camera_bounds_calls = 0
        self.reset_follow_state_calls = 0
        self.update_renderer_calls = 0

    def set_fly_mode(self, enabled: bool) -> bool:
        self.fly_mode_calls.append(bool(enabled))
        return True

    def reset_camera_bounds(self) -> None:
        self.reset_camera_bounds_calls += 1

    def reset_follow_state(self) -> None:
        self.reset_follow_state_calls += 1

    def update_renderer(self) -> None:
        self.update_renderer_calls += 1


class _DummyCameraController:
    def __init__(self) -> None:
        self.restore_calls = 0
        self.restore_update_flags: list[bool] = []
        self.restore_pre_pov_calls = 0
        self.restore_pre_pov_update_flags: list[bool] = []
        self.clear_pre_pov_calls = 0
        self.capture_pre_pov_calls = 0
        self.focus_calls = 0
        self.pov_calls = 0
        self.pov_result = True
        self.restore_pre_pov_result = True

    def restore_pov_entity_visibility(self, update_renderer: bool = True) -> None:
        self.restore_calls += 1
        self.restore_update_flags.append(bool(update_renderer))

    def restore_pre_pov_camera_state(self, update_renderer: bool = True) -> bool:
        self.restore_pre_pov_calls += 1
        self.restore_pre_pov_update_flags.append(bool(update_renderer))
        return self.restore_pre_pov_result

    def clear_pre_pov_camera_state(self) -> None:
        self.clear_pre_pov_calls += 1

    def capture_pre_pov_camera_state(self) -> bool:
        self.capture_pre_pov_calls += 1
        return True

    def focus_on_target(self) -> None:
        self.focus_calls += 1

    def set_pov_camera(self, **_kwargs) -> bool:
        self.pov_calls += 1
        return self.pov_result


class _DummyViz:
    def __init__(self, *, camera_mode: str = "overview", fly_mode: bool = False) -> None:
        self.app_state = create_initial_state(camera_mode=camera_mode, fly_mode=fly_mode)
        self.renderer = _DummyRenderer()
        self.camera_controller = _DummyCameraController()
        self.fly_mode_cb = _DummyToggle(fly_mode)
        self.track_group = _DummyToggle()
        self.pov_axis_container = _DummyToggle()

    def set_state(self, **changes) -> None:
        previous_state = self.app_state
        self.app_state = update_state(self.app_state, **changes)
        if previous_state.fly_mode != self.app_state.fly_mode:
            self.fly_mode_cb.blockSignals(True)
            self.fly_mode_cb.setChecked(self.app_state.fly_mode)
            self.fly_mode_cb.blockSignals(False)
            self.renderer.set_fly_mode(self.app_state.fly_mode)


def _make_controller(viz: _DummyViz) -> UIController:
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    controller._camera_debug_enabled = False
    return controller


def test_enabling_fly_from_follow_forces_overview_first() -> None:
    viz = _DummyViz(camera_mode="follow", fly_mode=False)
    controller = _make_controller(viz)

    controller.handle_fly_mode_toggled(True)

    assert viz.app_state.camera_mode == "overview"
    assert viz.app_state.fly_mode is True
    assert viz.renderer.fly_mode_calls == [True]
    assert viz.camera_controller.restore_calls == 1
    assert viz.camera_controller.restore_update_flags == [True]


def test_follow_mode_turns_off_active_fly_mode() -> None:
    viz = _DummyViz(camera_mode="overview", fly_mode=True)
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("follow", True)

    assert viz.app_state.camera_mode == "follow"
    assert viz.app_state.fly_mode is False
    assert viz.renderer.fly_mode_calls == [False]
    assert viz.camera_controller.focus_calls == 1


def test_pov_mode_turns_off_active_fly_mode_and_captures_pre_pov_view() -> None:
    viz = _DummyViz(camera_mode="overview", fly_mode=True)
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("pov", True)

    assert viz.app_state.camera_mode == "pov"
    assert viz.app_state.fly_mode is False
    assert viz.renderer.fly_mode_calls == [False]
    assert viz.camera_controller.capture_pre_pov_calls == 1
    assert viz.camera_controller.pov_calls == 1


def test_failed_pov_setup_rolls_back_camera_mode_fly_mode_and_controls() -> None:
    viz = _DummyViz(camera_mode="overview", fly_mode=True)
    viz.camera_controller.pov_result = False
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("pov", True)

    assert viz.app_state.camera_mode == "overview"
    assert viz.app_state.fly_mode is True
    assert viz.camera_controller.capture_pre_pov_calls == 1
    assert viz.camera_controller.clear_pre_pov_calls == 1
    assert viz.camera_controller.pov_calls == 1
    assert viz.track_group.visible is False
    assert viz.pov_axis_container.visible is False


def test_leaving_pov_for_overview_restores_saved_view_without_bounds_reset() -> None:
    viz = _DummyViz(camera_mode="pov", fly_mode=False)
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("overview", True)

    assert viz.app_state.camera_mode == "overview"
    assert viz.camera_controller.restore_calls == 1
    assert viz.camera_controller.restore_update_flags == [False]
    assert viz.camera_controller.restore_pre_pov_calls == 1
    assert viz.camera_controller.restore_pre_pov_update_flags == [True]
    assert viz.renderer.reset_camera_bounds_calls == 0


def test_leaving_pov_for_overview_redraws_if_saved_view_missing() -> None:
    viz = _DummyViz(camera_mode="pov", fly_mode=False)
    viz.camera_controller.restore_pre_pov_result = False
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("overview", True)

    assert viz.camera_controller.restore_update_flags == [False]
    assert viz.camera_controller.restore_pre_pov_calls == 1
    assert viz.renderer.update_renderer_calls == 1


def test_leaving_pov_for_follow_clears_saved_view_without_restoring_it() -> None:
    viz = _DummyViz(camera_mode="pov", fly_mode=False)
    controller = _make_controller(viz)

    controller.handle_camera_mode_changed("follow", True)

    assert viz.app_state.camera_mode == "follow"
    assert viz.camera_controller.restore_calls == 1
    assert viz.camera_controller.restore_update_flags == [False]
    assert viz.camera_controller.restore_pre_pov_calls == 0
    assert viz.camera_controller.clear_pre_pov_calls == 1
    assert viz.camera_controller.focus_calls == 1
    assert viz.renderer.reset_follow_state_calls == 1
    assert viz.renderer.reset_camera_bounds_calls == 0


def test_view_mode_change_routes_directly_to_material_controller() -> None:
    controller = UIController.__new__(UIController)
    calls = []

    class _MaterialController:
        def update_object_list_display(self, mode: str) -> None:
            calls.append(mode)

    controller._material_ctrl = _MaterialController()

    controller.handle_view_mode_changed("Type")

    assert calls == ["Type"]
