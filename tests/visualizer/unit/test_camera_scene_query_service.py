from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.services.camera_scene_query_service import CameraSceneQueryService


class _FakeBBox:
    def __init__(self, min_bound, max_bound) -> None:
        self.min_bound = np.asarray(min_bound, dtype=float)
        self.max_bound = np.asarray(max_bound, dtype=float)

    def get_center(self) -> np.ndarray:
        return (self.min_bound + self.max_bound) / 2.0

    def get_extent(self) -> np.ndarray:
        return self.max_bound - self.min_bound


class _FakeDropdown:
    def __init__(self, *, text: str, data) -> None:
        self._text = text
        self._data = data

    def currentText(self) -> str:
        return self._text

    def currentData(self):
        return self._data


def test_compute_scene_bounds_does_not_guess_from_unapplied_source_entries() -> None:
    viz = SimpleNamespace(
        renderer=None,
        mesh_entries=[{"visible": True, "mesh": object()}],
        target_entries=[{"visible": True, "mesh": object()}],
    )
    service = CameraSceneQueryService(viz)

    assert service.compute_scene_bounds(scope="visible") is None


def test_compute_scene_bounds_uses_renderer_contract_without_capability_gate() -> None:
    calls: list[str] = []

    def compute_scene_bounds(*, scope: str) -> _FakeBBox:
        calls.append(scope)
        return _FakeBBox([-2.0, -4.0, -6.0], [4.0, 8.0, 12.0])

    viz = SimpleNamespace(
        renderer=SimpleNamespace(compute_scene_bounds=compute_scene_bounds),
        mesh_entries=[],
        target_entries=[],
        tx_markers=[],
        rx_markers=[],
    )

    center, extent = CameraSceneQueryService(viz).compute_scene_bounds(scope="whole")

    assert calls == ["whole"]
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(extent, [6.0, 12.0, 18.0])


def test_compute_scene_bounds_returns_none_when_renderer_contract_fails() -> None:
    def compute_scene_bounds(*, scope: str) -> _FakeBBox:
        raise RuntimeError(f"bounds unavailable for {scope}")

    viz = SimpleNamespace(
        renderer=SimpleNamespace(compute_scene_bounds=compute_scene_bounds),
        mesh_entries=[{"visible": True, "mesh": object()}],
    )

    assert CameraSceneQueryService(viz).compute_scene_bounds(scope="visible") is None


def test_get_focus_position_resolves_structured_dropdown_selection() -> None:
    view_model = SimpleNamespace(
        target_positions=[],
        tx_positions=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        rx_positions=[],
    )
    viz = SimpleNamespace(
        current_view_model=view_model,
        target_focus_dropdown=_FakeDropdown(text="unused", data={"type": "tx", "index": 1}),
    )
    service = CameraSceneQueryService(viz)

    assert service.get_focus_position() == [4.0, 5.0, 6.0]

    viz.target_focus_dropdown = _FakeDropdown(text="TX2", data=None)

    assert service.get_focus_position() is None


def test_entity_position_orientation_prefers_cached_frame_orientation() -> None:
    view_model = SimpleNamespace(
        target_positions=[[10.0, 20.0, 30.0]],
        target_metadata=[{"name": "target_0", "orientation": [1.0, 2.0, 3.0]}],
    )
    frame_data = {"targets_metadata": [{"orientation": np.array([9.0, 8.0, 7.0])}]}
    cache_service = SimpleNamespace(get_frame=lambda step: frame_data if step == 3 else None)
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=3),
        current_view_model=view_model,
        cache_service=cache_service,
        target_entries=[
            {"target_name": "target_0", "stable_target_id": "target_0"},
        ],
        target_focus_dropdown=_FakeDropdown(
            text="Target 1",
            data={"type": "target", "stable_target_id": "target_0", "index": 0},
        ),
    )
    service = CameraSceneQueryService(viz)

    position, orientation, entity_info = service.get_entity_position_orientation_and_info()

    assert position == [10.0, 20.0, 30.0]
    assert orientation == [9.0, 8.0, 7.0]
    assert entity_info == {
        "type": "target",
        "stable_target_id": "target_0",
        "index": 0,
    }


def test_target_camera_pose_uses_stable_identity_when_metadata_order_changes() -> None:
    view_model = SimpleNamespace(
        target_positions=[[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        target_metadata=[
            {"name": "beta", "orientation": [2.0, 2.0, 2.0]},
            {"name": "alpha", "orientation": [1.0, 1.0, 1.0]},
        ],
    )
    frame_data = {
        "targets_metadata": [
            {"name": "alpha", "orientation": [9.0, 8.0, 7.0]},
            {"name": "beta", "orientation": [6.0, 5.0, 4.0]},
        ]
    }
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=4),
        current_view_model=view_model,
        cache_service=SimpleNamespace(get_frame=lambda step: frame_data if step == 4 else None),
        target_entries=[
            {"target_name": "alpha", "stable_target_id": "alpha"},
            {"target_name": "beta", "stable_target_id": "beta"},
        ],
        target_focus_dropdown=_FakeDropdown(
            text="Target 1 - alpha",
            data={"type": "target", "stable_target_id": "alpha", "index": 0},
        ),
    )

    position, orientation, entity_info = CameraSceneQueryService(
        viz
    ).get_entity_position_orientation_and_info()

    assert position == [10.0, 0.0, 0.0]
    assert orientation == [9.0, 8.0, 7.0]
    assert entity_info == {
        "type": "target",
        "stable_target_id": "alpha",
        "index": 0,
    }


def test_sparse_target_focus_selection_keeps_canonical_pov_index() -> None:
    view_model = SimpleNamespace(
        target_positions=[[7.0, 8.0, 9.0]],
        target_metadata=[{"name": "present", "orientation": [1.0, 2.0, 3.0]}],
    )
    viz = SimpleNamespace(
        current_view_model=view_model,
        target_entries=[
            {"target_name": "missing", "stable_target_id": "missing"},
            {"target_name": "present", "stable_target_id": "present"},
        ],
    )
    service = CameraSceneQueryService(viz)

    selection = service.target_focus_selection(0)

    assert selection == {
        "type": "target",
        "stable_target_id": "present",
        "index": 1,
    }
    viz.target_focus_dropdown = _FakeDropdown(text="Target 2 - present", data=selection)
    position, orientation, entity_info = service.get_entity_position_orientation_and_info()
    assert position == [7.0, 8.0, 9.0]
    assert orientation == [1.0, 2.0, 3.0]
    assert entity_info == selection


def test_motion_direction_matches_target_identity_and_ignores_model_orientation():
    frames = {
        0: {
            "targets_metadata": [
                {"name": "beta", "current_position": [500, 0, 0]},
                {"name": "alpha", "current_position": [0, 0, 0]},
            ]
        },
        1: {
            "targets_metadata": [
                {"name": "alpha", "current_position": [0, 2, 0]},
                {"name": "beta", "current_position": [600, 0, 0]},
            ]
        },
    }
    vm = SimpleNamespace(
        target_positions=[[0, 2, 0]], target_metadata=[{"name": "alpha", "orientation": [0, 0, 0]}]
    )
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=1),
        current_view_model=vm,
        cache_service=SimpleNamespace(get_frame=frames.get),
        target_entries=[{"stable_target_id": "alpha"}],
        target_focus_dropdown=_FakeDropdown(
            text="Alpha", data={"type": "target", "stable_target_id": "alpha", "index": 0}
        ),
    )
    query = CameraSceneQueryService(viz)
    np.testing.assert_allclose(query.get_focus_motion_direction(), [0, 2, 0])
    # Scrub backward to the first frame: physical travel remains +Y.
    viz.app_state.step = 0
    vm.target_positions = [[0, 0, 0]]
    np.testing.assert_allclose(query.get_focus_motion_direction(), [0, 2, 0])
    frames[1]["targets_metadata"] = [{"name": "beta", "current_position": [600, 0, 0]}]
    assert query.get_focus_motion_direction() is None


@pytest.mark.parametrize("kind", ["tx", "rx", "auto"])
def test_motion_direction_supports_radios_and_auto_selection(kind):
    vm = SimpleNamespace(tx_positions=[[2, 0, 0]], rx_positions=[[0, 3, 0]])
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=1),
        current_view_model=vm,
        cache_service=SimpleNamespace(
            get_frame=lambda step: (
                {"tx_positions": [[0, 0, 0]], "rx_positions": [[0, 0, 0]]} if step == 0 else None
            )
        ),
        target_focus_dropdown=_FakeDropdown(text="Radio", data={"type": kind, "index": 0}),
    )
    direction = CameraSceneQueryService(viz).get_focus_motion_direction()
    np.testing.assert_allclose(direction, [0, 3, 0] if kind == "rx" else [2, 0, 0])


def test_motion_projection_reads_only_neighbor_and_keeps_playback_cache_separate(monkeypatch):
    provider = SimpleNamespace(list_frames=lambda: [4, 8, 12])
    reads = []

    def load(_provider, step):
        reads.append(step)
        return {"rx_positions": [[0, step, 0]]}

    module = "visualizer.src.services.camera_scene_query_service"
    monkeypatch.setattr(f"{module}.visualizer_frame_provider", lambda viz: provider)
    monkeypatch.setattr(f"{module}.try_load_packed_trajectory_frame", load)
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=8),
        current_view_model=SimpleNamespace(rx_positions=[[0, 8, 0]]),
        target_focus_dropdown=_FakeDropdown(text="RX", data={"type": "rx", "index": 0}),
    )
    query = CameraSceneQueryService(viz)
    for _ in range(2):
        np.testing.assert_allclose(query.get_focus_motion_direction(), [0, 1, 0])
    assert reads == [4] and viz.app_state.step == 8
    provider = SimpleNamespace(list_frames=lambda: [4, 8, 12])
    query.get_focus_motion_direction()
    assert reads == [4, 4]  # Another recording cannot reuse a stale position read.


def test_stationary_motion_does_not_look_ahead_to_future_turn():
    frames = {
        0: {"rx_positions": [[0, 0, 0]]},
        2: {"rx_positions": [[10, 0, 0]]},
    }
    viz = SimpleNamespace(
        app_state=SimpleNamespace(step=1),
        current_view_model=SimpleNamespace(rx_positions=[[0, 0, 0]]),
        cache_service=SimpleNamespace(get_frame=frames.get),
        target_focus_dropdown=_FakeDropdown(text="RX", data={"type": "rx", "index": 0}),
    )
    np.testing.assert_allclose(CameraSceneQueryService(viz).get_focus_motion_direction(), [0, 0, 0])
