from types import SimpleNamespace

import numpy as np

from visualizer.src.controllers.camera_controller import CameraController
from visualizer.src.services.camera_scene_query_service import CameraSceneQueryService
from visualizer.src.state import create_initial_state, update_state
from visualizer.src.types.camera_state import CameraState
from visualizer.visualizer import OrchavVisualizer


def _make_orbit_state() -> CameraState:
    return CameraState(
        eye=(10.0, 5.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )


class _FakeButton:
    def __init__(self, preset_num: int | None = None):
        self._preset_num = preset_num
        self._checked = False
        self.tooltip = ""
        self.stylesheet = ""
        self.signal_blocks: list[bool] = []

    def property(self, name: str):
        if name == "preset_num":
            return self._preset_num
        return None

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def blockSignals(self, blocked: bool) -> None:
        self.signal_blocks.append(bool(blocked))

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setStyleSheet(self, text: str) -> None:
        self.stylesheet = text


class _FakeDropdown:
    def __init__(self) -> None:
        self.items: list[tuple[str, object]] = []
        self.current_index = -1
        self.signal_blocks: list[bool] = []

    def currentData(self):
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][1]
        return None

    def currentText(self) -> str:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][0]
        return ""

    def blockSignals(self, blocked: bool) -> None:
        self.signal_blocks.append(bool(blocked))

    def clear(self) -> None:
        self.items.clear()
        self.current_index = -1

    def addItem(self, text: str, data) -> None:
        self.items.append((text, data))
        if self.current_index < 0:
            self.current_index = 0

    def setCurrentIndex(self, index: int) -> None:
        self.current_index = int(index)

    def count(self) -> int:
        return len(self.items)

    def itemData(self, index: int):
        return self.items[index][1]

    def findText(self, text: str) -> int:
        for index, item in enumerate(self.items):
            if item[0] == text:
                return index
        return -1


def _make_controller(viz) -> CameraController:
    return CameraController(viz, CameraSceneQueryService(viz))


def test_save_camera_preset_stores_neutral_camera_state():
    renderer = SimpleNamespace(
        get_camera_state=lambda: _make_orbit_state(),
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    ok = controller.save_camera_preset(1, name="Orbit Preset")

    assert ok is True
    preset = controller._camera_presets[1]
    assert isinstance(preset["state"], CameraState)


def test_save_camera_preset_fails_when_camera_state_unavailable():
    renderer = SimpleNamespace(
        get_camera_state=lambda: None,
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    ok = controller.save_camera_preset(1, name="Legacy Preset")

    assert ok is False
    assert controller._camera_presets == {}


def test_camera_presets_are_limited_to_four_slots():
    renderer = SimpleNamespace(
        get_camera_state=lambda: _make_orbit_state(),
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    assert controller.MAX_PRESETS == 4
    assert controller.save_camera_preset(4, name="Last Slot") is True
    assert controller.save_camera_preset(5, name="Hidden Slot") is False
    assert 4 in controller._camera_presets
    assert 5 not in controller._camera_presets


def test_load_camera_preset_uses_set_camera_state_for_orbit_presets():
    calls: list[CameraState] = []

    def _set_camera_state(state: CameraState) -> bool:
        calls.append(state)
        return True

    renderer = SimpleNamespace(
        set_camera_state=_set_camera_state,
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)
    state = _make_orbit_state()
    controller._camera_presets[1] = {"state": state, "name": "Orbit Preset"}

    ok = controller.load_camera_preset(1)

    assert ok is True
    assert len(calls) == 1
    assert calls[0] == state


def test_set_overview_view_uses_whole_scene_bounds_for_renderer_camera():
    class _BBox:
        def get_center(self):
            return np.array([20.0, 30.0, 5.0], dtype=float)

        def get_extent(self):
            return np.array([100.0, 50.0, 10.0], dtype=float)

    scopes: list[str] = []
    calls: list[tuple[str, tuple[np.ndarray, np.ndarray], float, float]] = []

    def _compute_scene_bounds(*, scope="visible"):
        scopes.append(scope)
        return _BBox()

    def _set_overview_camera(view, bounds, fov, distance=None):
        calls.append((view, bounds, fov, distance))
        return True

    renderer = SimpleNamespace(
        _width=1200,
        _height=800,
        compute_scene_bounds=_compute_scene_bounds,
        set_overview_camera=_set_overview_camera,
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    ok = controller.set_overview_view("top")

    assert ok is True
    assert scopes == ["whole"]
    assert len(calls) == 1
    view, bounds, fov, distance = calls[0]
    center, extent = bounds
    assert view == "top"
    np.testing.assert_allclose(center, [20.0, 30.0, 5.0])
    np.testing.assert_allclose(extent, [100.0, 50.0, 10.0])
    assert fov == 60.0
    assert distance > 0.0


def test_set_overview_view_uses_renderer_auto_distance_adjustment():
    class _BBox:
        def get_center(self):
            return np.array([0.0, 0.0, 0.0], dtype=float)

        def get_extent(self):
            return np.array([100.0, 50.0, 10.0], dtype=float)

    adjustments: list[tuple[float, float, float, str, np.ndarray]] = []
    calls: list[float] = []

    def _compute_scene_bounds(*, scope="visible"):
        return _BBox()

    def _adjust(distance, *, fov, aspect, view, extent, **_kwargs):
        adjustments.append((distance, fov, aspect, view, np.asarray(extent, dtype=float)))
        return distance * 1.5

    def _set_overview_camera(_view, _bounds, _fov, distance=None):
        calls.append(distance)
        return True

    renderer = SimpleNamespace(
        _width=1200,
        _height=800,
        compute_scene_bounds=_compute_scene_bounds,
        adjust_overview_distance=_adjust,
        set_overview_camera=_set_overview_camera,
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    assert controller.set_overview_view("top") is True

    assert len(adjustments) == 1
    raw_distance, fov, aspect, view, extent = adjustments[0]
    assert fov == 60.0
    assert aspect == 1200 / 800
    assert view == "top"
    np.testing.assert_allclose(extent, [100.0, 50.0, 10.0])
    assert calls == [raw_distance * 1.5]


def test_set_overview_view_keeps_explicit_camera_distance_unadjusted():
    class _BBox:
        def get_center(self):
            return np.array([0.0, 0.0, 0.0], dtype=float)

        def get_extent(self):
            return np.array([100.0, 50.0, 10.0], dtype=float)

    calls: list[float] = []

    def _adjust(*_args, **_kwargs):
        raise AssertionError("explicit camera distance should not be adjusted")

    def _set_overview_camera(_view, _bounds, _fov, distance=None):
        calls.append(distance)
        return True

    renderer = SimpleNamespace(
        _width=1200,
        _height=800,
        compute_scene_bounds=lambda *, scope="visible": _BBox(),
        adjust_overview_distance=_adjust,
        set_overview_camera=_set_overview_camera,
    )
    viz = SimpleNamespace(vis_initialized=True, vis=None, renderer=renderer)
    controller = _make_controller(viz)

    assert controller.set_overview_view("top", camera_dist=42.0) is True

    assert calls == [42.0]


def test_reset_camera_uses_isometric_when_no_scenario_default():
    calls: list[tuple] = []

    def _apply_camera_view(view, camera_dist=None, fov=None):
        calls.append(("apply", view, camera_dist, fov))
        return True

    renderer = SimpleNamespace(
        reset_camera_bounds=lambda: calls.append(("bounds",)),
        poll_events=lambda: calls.append(("poll",)),
        update_renderer=lambda: calls.append(("update",)),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=None,
        renderer=renderer,
        default_camera_view=None,
        default_camera_dist=None,
        default_camera_fov=None,
        apply_camera_view=_apply_camera_view,
    )

    OrchavVisualizer.reset_camera_to_overview(viz)

    assert calls == [("apply", "isometric", None, None)]
    assert not hasattr(viz, "good_camera_position")


def test_reset_camera_switches_to_overview_and_clears_pov_state():
    calls: list[str] = []

    class _Controller:
        def __init__(self):
            self.restore_calls = 0
            self.clear_calls = 0

        def restore_pov_entity_visibility(self, update_renderer=True) -> None:
            self.restore_calls += 1
            assert update_renderer is False

        def clear_pre_pov_camera_state(self) -> None:
            self.clear_calls += 1

    class _Viz:
        def __init__(self):
            self.vis_initialized = True
            self.vis = None
            self.renderer = SimpleNamespace(
                reset_camera_bounds=lambda: calls.append("bounds"),
                reset_follow_state=lambda: calls.append("follow_reset"),
                poll_events=lambda: calls.append("poll"),
                update_renderer=lambda: calls.append("update"),
            )
            self.camera_controller = _Controller()
            self.app_state = create_initial_state(
                camera_mode="pov",
                pov_hidden_node=("tx", 0),
            )
            self.default_camera_view = None
            self.default_camera_dist = None
            self.default_camera_fov = None

        def set_state(self, **changes) -> None:
            self.app_state = update_state(self.app_state, **changes)

        def apply_camera_view(self, view, camera_dist=None, fov=None):
            calls.append(f"apply:{view}:{camera_dist}:{fov}")
            return True

    viz = _Viz()

    OrchavVisualizer.reset_camera_to_overview(viz)

    assert viz.app_state.camera_mode == "overview"
    assert viz.app_state.pov_hidden_node is None
    assert viz.camera_controller.restore_calls == 1
    assert viz.camera_controller.clear_calls == 1
    assert calls == ["follow_reset", "apply:isometric:None:None"]


def test_reset_camera_prefers_scenario_default_over_bounds():
    calls: list[tuple] = []

    def _apply_camera_view(view, camera_dist=None, fov=None):
        calls.append(("apply", view, camera_dist, fov))
        return True

    renderer = SimpleNamespace(
        reset_camera_bounds=lambda: calls.append(("bounds",)),
        poll_events=lambda: calls.append(("poll",)),
        update_renderer=lambda: calls.append(("update",)),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=None,
        renderer=renderer,
        default_camera_view="isometric",
        default_camera_dist=120.0,
        default_camera_fov=55.0,
        apply_camera_view=_apply_camera_view,
    )

    OrchavVisualizer.reset_camera_to_overview(viz)

    assert calls == [("apply", "isometric", 120.0, 55.0)]


def test_pre_pov_camera_state_restores_via_renderer_camera_state():
    state = _make_orbit_state()
    calls: list[tuple[str, CameraState]] = []

    def _set_camera_state(camera_state):
        calls.append(("set", camera_state))
        return True

    renderer = SimpleNamespace(
        get_camera_state=lambda: state,
        set_camera_state=_set_camera_state,
    )
    viz = SimpleNamespace(renderer=renderer)
    controller = _make_controller(viz)

    assert controller.capture_pre_pov_camera_state() is True
    assert controller.restore_pre_pov_camera_state() is True

    assert calls == [("set", state)]
    assert controller._pre_pov_camera_state is None
    assert controller.restore_pre_pov_camera_state() is False


def test_pre_pov_camera_restore_retains_state_without_renderer() -> None:
    state = _make_orbit_state()
    controller = _make_controller(SimpleNamespace(renderer=None))
    controller._pre_pov_camera_state = state

    assert controller.restore_pre_pov_camera_state() is False
    assert controller._pre_pov_camera_state == state


def test_pre_pov_camera_restore_retains_state_when_renderer_returns_false() -> None:
    state = _make_orbit_state()
    renderer = SimpleNamespace(set_camera_state=lambda _state: False)
    controller = _make_controller(SimpleNamespace(renderer=renderer))
    controller._pre_pov_camera_state = state

    assert controller.restore_pre_pov_camera_state() is False
    assert controller._pre_pov_camera_state == state


def test_pre_pov_camera_restore_retains_state_when_renderer_raises() -> None:
    state = _make_orbit_state()

    def fail(_state):
        raise RuntimeError("injected restore failure")

    controller = _make_controller(SimpleNamespace(renderer=SimpleNamespace(set_camera_state=fail)))
    controller._pre_pov_camera_state = state

    assert controller.restore_pre_pov_camera_state() is False
    assert controller._pre_pov_camera_state == state


def test_pre_pov_camera_restore_fail_once_retry_converges() -> None:
    state = _make_orbit_state()
    outcomes = iter((False, True))
    calls: list[CameraState] = []

    def restore(camera_state):
        calls.append(camera_state)
        return next(outcomes)

    controller = _make_controller(
        SimpleNamespace(renderer=SimpleNamespace(set_camera_state=restore))
    )
    controller._pre_pov_camera_state = state

    assert controller.restore_pre_pov_camera_state() is False
    assert controller._pre_pov_camera_state == state
    assert controller.restore_pre_pov_camera_state() is True
    assert controller._pre_pov_camera_state is None
    assert calls == [state, state]


def test_set_pov_camera_can_defer_renderer_redraw():
    calls: list[tuple[list[float], list[float], str, bool]] = []

    def _set_pov_camera(position, orientation, axis, *, defer_redraw=False):
        calls.append((position, orientation, axis, defer_redraw))
        return True

    renderer = SimpleNamespace(
        set_pov_camera=_set_pov_camera,
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=SimpleNamespace(pov_axis="forward"),
    )
    controller = _make_controller(viz)
    controller.scene_query.get_entity_position_orientation_and_info = lambda: (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        None,
    )
    controller._hide_pov_entity = lambda _entity_info: None

    controller.set_pov_camera(defer_redraw=True)

    assert calls == [([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "forward", True)]

    controller.set_pov_camera()

    assert calls[-1] == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "forward", False)


def test_target_focus_dropdown_carries_canonical_identity_when_metadata_reorders():
    dropdown = _FakeDropdown()
    viz = SimpleNamespace(
        target_focus_dropdown=dropdown,
        current_view_model=SimpleNamespace(
            target_metadata=[{"name": "beta"}, {"name": "alpha"}],
            target_positions=[[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            tx_positions=[],
            rx_positions=[],
        ),
        target_entries=[
            {"target_name": "alpha", "stable_target_id": "alpha"},
            {"target_name": "beta", "stable_target_id": "beta"},
        ],
        app_state=SimpleNamespace(
            tx_labels=(),
            rx_labels=(),
            tx_device_names=(),
            rx_device_names=(),
            node_label_mode="role",
        ),
    )
    controller = _make_controller(viz)
    controller._preferred_focus_data = {
        "type": "target",
        "stable_target_id": "alpha",
        "index": 0,
    }

    controller.update_target_focus_dropdown()

    assert dropdown.items == [
        ("Auto (First Target)", {"type": "auto"}),
        (
            "Target 2 - beta",
            {"type": "target", "stable_target_id": "beta", "index": 1},
        ),
        (
            "Target 1 - alpha",
            {"type": "target", "stable_target_id": "alpha", "index": 0},
        ),
    ]
    assert dropdown.currentData() == controller._preferred_focus_data


def test_failed_pov_camera_switch_restores_previous_semantic_visibility():
    state = SimpleNamespace(
        camera_mode="pov",
        pov_axis="forward",
        pov_hidden_node=("tx", 0),
    )
    semantic_syncs: list[tuple[tuple[str, int], ...]] = []

    def set_state(**changes) -> None:
        for name, value in changes.items():
            setattr(state, name, value)

    renderer = SimpleNamespace(
        set_pov_camera=lambda *_args, **_kwargs: False,
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=state,
        set_state=set_state,
        node_service=SimpleNamespace(
            sync_pov_entity_visibility=lambda refs: (semantic_syncs.append(tuple(refs)) or True)
        ),
    )
    controller = _make_controller(viz)
    controller.scene_query.get_entity_position_orientation_and_info = lambda: (
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        {"type": "target", "stable_target_id": "present", "index": 1},
    )

    controller.set_pov_camera()

    assert state.pov_hidden_node == ("tx", 0)
    assert semantic_syncs == [
        (("tx", 0), ("target", 1)),
        (("target", 1), ("tx", 0)),
    ]


def test_pov_camera_is_not_applied_when_visibility_sync_fails() -> None:
    camera_calls = []
    renderer = SimpleNamespace(
        set_pov_camera=lambda *_args, **_kwargs: camera_calls.append(True) or True,
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        app_state=SimpleNamespace(pov_axis="forward"),
    )
    controller = _make_controller(viz)
    controller.scene_query.get_entity_position_orientation_and_info = lambda: (
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        {"type": "rx", "index": 0},
    )
    controller._hide_pov_entity = lambda _entity_info: False

    assert controller.set_pov_camera() is False
    assert camera_calls == []


def test_saved_view_save_mode_saves_then_returns_to_load_mode():
    class _Controller:
        def __init__(self):
            self.saved: list[int] = []
            self.loaded: list[int] = []

        def save_camera_preset(self, preset_num: int) -> bool:
            self.saved.append(preset_num)
            return True

        def load_camera_preset(self, preset_num: int) -> bool:
            self.loaded.append(preset_num)
            return True

        def get_all_presets(self) -> dict[int, str]:
            return {num: f"Preset {num}" for num in self.saved}

    controller = _Controller()
    save_button = _FakeButton()
    slot_buttons = [_FakeButton(i) for i in range(1, 5)]
    viz = SimpleNamespace(
        camera_controller=controller,
        _camera_preset_save_mode=True,
        camera_preset_save_btn=save_button,
        camera_preset_buttons=slot_buttons,
    )
    viz._save_camera_preset = lambda preset_num: OrchavVisualizer._save_camera_preset(
        viz, preset_num
    )
    viz._set_camera_preset_save_mode = (
        lambda enabled: OrchavVisualizer._set_camera_preset_save_mode(viz, enabled)
    )
    viz._update_camera_preset_buttons = lambda: OrchavVisualizer._update_camera_preset_buttons(viz)

    OrchavVisualizer._handle_camera_preset_clicked(viz, 2)

    assert controller.saved == [2]
    assert controller.loaded == []
    assert viz._camera_preset_save_mode is False
    assert save_button.isChecked() is False

    OrchavVisualizer._handle_camera_preset_clicked(viz, 2)

    assert controller.loaded == [2]
