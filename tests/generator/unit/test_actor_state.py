from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

import generator.core.scenario_actors.state as actor_state
from generator.core.mobility import LinearMobility, RandomBoxMobility
from generator.core.orientation import (
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    LookAtOrientationSpec,
)
from generator.core.scenario_actors import PosePreparationError
from generator.core.scenario_actors.state import (
    ActorStateCache,
    ActorStateManager,
    ActorStateSnapshot,
    ActorStateVelocities,
)


def _xyz(point: Any) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


class _FakeMobility:
    def __init__(self, positions: list[tuple[float, float, float]]) -> None:
        self._positions = positions
        self._prepared_positions: list[tuple[float, float, float]] | None = None

    def prepare(
        self,
        scene_steps: int,
        scene_duration: float,
        start_pos: tuple[float, float, float] | None = None,
    ) -> None:
        del scene_duration, start_pos
        self._prepared_positions = list(self._positions[:scene_steps])

    def prepared_positions(self) -> list[tuple[float, float, float]]:
        if self._prepared_positions is None:
            raise RuntimeError("Fake mobility must be prepared before prepared_positions()")
        return self._prepared_positions

    def get_position(self, step: int) -> tuple[float, float, float]:
        return self.prepared_positions()[step]

    def get_positions(
        self,
        start_pos: tuple[float, float, float] | None,
        scene_steps: int,
        scene_duration: float,
    ) -> list[tuple[float, float, float]]:
        del start_pos, scene_duration
        return list(self._positions[:scene_steps])


class _CountingPreparedOrientationSource:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self._orientations: list[tuple[float, float, float]] = []

    def prepare(self, steps: int, duration: float | None = None, context=None) -> None:
        del duration, context
        self.prepare_calls += 1
        self._orientations = [(0.0, 0.0, 0.0)] * steps

    def orientations(self) -> list[tuple[float, float, float]]:
        return list(self._orientations)


def test_streaming_state_at_step_uses_cached_ply_target_positions(monkeypatch) -> None:
    ply_positions = [
        SimpleNamespace(x=1.0, y=2.0, z=3.0),
        SimpleNamespace(x=4.0, y=5.0, z=6.0),
    ]
    monkeypatch.setattr(
        actor_state,
        "positions_from_ply_aabb",
        lambda *args, **kwargs: ply_positions,
    )
    target_manager = SimpleNamespace(
        meshes=["target_000.ply"],
        config=SimpleNamespace(
            use_ply_position=True,
            mobility=None,
            initial_position=(0.0, 0.0, 0.0),
            orientation=FixedOrientationSpec(),
        ),
    )
    manager = ActorStateManager(
        [],
        [],
        [target_manager],
        steps=2,
        duration=1.0,
        motion_mode="step",
    )

    cache = manager.prepare_cached()
    snapshot = manager.state_at_step(1)

    assert isinstance(cache, ActorStateCache)
    assert isinstance(snapshot, ActorStateSnapshot)
    assert cache.target_positions == [[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]]
    assert len(snapshot.target_positions) == 1
    assert _xyz(snapshot.target_positions[0]) == pytest.approx((4.0, 5.0, 6.0))


def test_streaming_state_at_step_uses_prepared_lookat_orientations() -> None:
    tx_mobility = _FakeMobility([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    target_mobility = _FakeMobility([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    tx_orientation = LookAtOrientationSpec(actor="target")
    tx_config = SimpleNamespace(
        name="tx",
        mobility=tx_mobility,
        initial_position=(0.0, 0.0, 0.0),
        orientation=tx_orientation,
    )
    target_manager = SimpleNamespace(
        config=SimpleNamespace(
            name="target",
            mobility=target_mobility,
            initial_position=(1.0, 0.0, 0.0),
            use_ply_position=False,
            orientation=FixedOrientationSpec(),
        )
    )
    manager = ActorStateManager(
        [tx_config],
        [],
        [target_manager],
        steps=2,
        duration=1.0,
        motion_mode="step",
    )

    manager.prepare_cached()
    first_orientation = manager.state_at_step(0).tx_orientations[0]
    second_orientation = manager.state_at_step(1).tx_orientations[0]

    assert first_orientation == pytest.approx((0.0, 0.0, 0.0))
    assert second_orientation == pytest.approx((90.0, 0.0, 0.0))


def test_state_at_step_returns_named_state_not_tuple() -> None:
    tx_mobility = _FakeMobility([(0.0, 0.0, 0.0)])
    tx_config = SimpleNamespace(
        name="tx",
        mobility=tx_mobility,
        initial_position=(0.0, 0.0, 0.0),
        orientation=FixedOrientationSpec(),
    )
    manager = ActorStateManager([tx_config], [], [], steps=1, duration=1.0)

    cache = manager.prepare_cached()
    snapshot = manager.state_at_step(0)
    velocities = manager.compute_velocities(0)

    assert isinstance(cache, ActorStateCache)
    assert isinstance(snapshot, ActorStateSnapshot)
    assert isinstance(velocities, ActorStateVelocities)
    assert _xyz(snapshot.tx_positions[0]) == pytest.approx((0.0, 0.0, 0.0))
    with pytest.raises(TypeError):
        tuple(snapshot)
    with pytest.raises(TypeError):
        tuple(cache)


def test_compute_velocities_uses_tuple_caches() -> None:
    tx_mobility = _FakeMobility(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (5.0, 1.0, 0.0),
        ]
    )
    tx_config = SimpleNamespace(
        name="tx",
        mobility=tx_mobility,
        initial_position=(0.0, 0.0, 0.0),
        orientation=FixedOrientationSpec(),
    )
    manager = ActorStateManager([tx_config], [], [], steps=3, duration=2.0)

    manager.prepare_cached()

    assert manager.compute_velocities(0).tx[0] == pytest.approx((2.0, 0.0, 0.0))
    assert manager.compute_velocities(2).tx[0] == pytest.approx((3.0, 1.0, 0.0))


def test_programmatic_mobility_declares_physical_motion_capability() -> None:
    linear = LinearMobility((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    random_box = RandomBoxMobility(seed=7)

    assert linear.has_physical_motion
    assert not random_box.has_physical_motion


def test_random_box_rejects_align_motion_and_exposes_no_artificial_velocity() -> None:
    mobility = RandomBoxMobility(
        x_bounds=(-10.0, 10.0),
        y_bounds=(-10.0, 10.0),
        z_bounds=(1.0, 2.0),
        seed=7,
    )
    config = SimpleNamespace(
        name="random",
        mobility=mobility,
        initial_position=mobility.start_pos,
        orientation=AlignMotionOrientationSpec(),
    )
    manager = ActorStateManager([config], [], [], steps=3, duration=2.0)

    with pytest.raises(PosePreparationError, match="physical velocity"):
        manager.prepare_cached()

    mobility = RandomBoxMobility(
        x_bounds=(-10.0, 10.0),
        y_bounds=(-10.0, 10.0),
        z_bounds=(1.0, 2.0),
        seed=7,
    )
    config = SimpleNamespace(
        name="random",
        mobility=mobility,
        initial_position=mobility.start_pos,
        orientation=FixedOrientationSpec(),
    )
    manager = ActorStateManager([config], [], [], steps=3, duration=2.0)

    cache = manager.prepare_cached()

    assert len(set(cache.tx_positions[0])) == 3
    assert not manager._actor_mobility_lookup["random"].has_physical_velocity
    assert [manager.compute_velocities(step).tx[0] for step in range(3)] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]


def test_prepare_cached_rejects_malformed_positions() -> None:
    tx_mobility = _FakeMobility([(0.0, 0.0, 0.0), (1.0, 0.0)])
    tx_config = SimpleNamespace(
        name="tx",
        mobility=tx_mobility,
        initial_position=(0.0, 0.0, 0.0),
        orientation=FixedOrientationSpec(),
    )
    manager = ActorStateManager([tx_config], [], [], steps=2, duration=1.0)

    with pytest.raises(ValueError, match="TX 'tx' position at step 1"):
        manager.prepare_cached()


def test_prepare_cached_rejects_duplicate_names_across_actor_roles() -> None:
    tx_config = SimpleNamespace(
        name="shared",
        mobility=_FakeMobility([(0.0, 0.0, 0.0)]),
        initial_position=(0.0, 0.0, 0.0),
        orientation=FixedOrientationSpec(),
    )
    rx_config = SimpleNamespace(
        name="shared",
        mobility=_FakeMobility([(1.0, 0.0, 0.0)]),
        initial_position=(1.0, 0.0, 0.0),
        orientation=LookAtOrientationSpec(actor="shared"),
    )
    manager = ActorStateManager(
        [tx_config],
        [rx_config],
        [],
        steps=1,
        duration=1.0,
    )

    with pytest.raises(
        ValueError,
        match="canonical actor references require globally unique names",
    ):
        manager.prepare_cached()


def test_state_at_step_warns_when_step_is_clamped(caplog) -> None:
    tx_mobility = _FakeMobility([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    tx_config = SimpleNamespace(
        name="tx",
        mobility=tx_mobility,
        initial_position=(0.0, 0.0, 0.0),
        orientation=FixedOrientationSpec(),
    )
    manager = ActorStateManager([tx_config], [], [], steps=2, duration=1.0)
    manager.prepare_cached()

    orchav_logger = logging.getLogger("orchav")
    previous_propagate = orchav_logger.propagate
    orchav_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="orchav.generator.core.scenario_actors.state"):
            snapshot = manager.state_at_step(-1)
    finally:
        orchav_logger.propagate = previous_propagate

    assert _xyz(snapshot.tx_positions[0]) == pytest.approx((0.0, 0.0, 0.0))
    assert "state_at_step requested step -1 outside prepared range [0, 1]" in caplog.text


def test_state_at_step_does_not_reprepare_or_reread_ply_positions(monkeypatch) -> None:
    ply_calls = 0
    orientation = _CountingPreparedOrientationSource()

    def _fake_ply_positions(*args, **kwargs):
        nonlocal ply_calls
        ply_calls += 1
        return [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    monkeypatch.setattr(actor_state, "positions_from_ply_aabb", _fake_ply_positions)
    target_manager = SimpleNamespace(
        meshes=["target_000.ply"],
        config=SimpleNamespace(
            use_ply_position=True,
            mobility=None,
            initial_position=(0.0, 0.0, 0.0),
            orientation=orientation,
        ),
    )
    manager = ActorStateManager([], [], [target_manager], steps=2, duration=1.0)

    manager.prepare_cached()
    for _ in range(5):
        manager.state_at_step(0)
        manager.state_at_step(1)

    assert ply_calls == 1
    assert orientation.prepare_calls == 1
