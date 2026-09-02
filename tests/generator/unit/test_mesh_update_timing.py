from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")
if not hasattr(mi, "variants") or "llvm_ad_rgb" not in mi.variants():
    pytest.skip("Usable Mitsuba runtime required", allow_module_level=True)
try:
    mi.set_variant("llvm_ad_rgb")
except Exception as exc:
    pytest.skip(f"Usable Mitsuba runtime required: {exc}", allow_module_level=True)

from generator.core.propagation import (
    apply_target_state_to_scene,
    normalize_live_overrides,
    should_update_mesh_for_step,
)
from generator.core.propagation.actor_state_application import apply_transceiver_state_to_scene
from generator.core.target.mesh import mesh_update_step_interval, positions_from_ply_aabb
from shared.protos import visualizer_pb2


def _cfg(
    *,
    duration: float,
    num_steps: int,
    mesh_update_interval_s: float | None,
    cir_time_steps: int,
    start_step: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        duration=duration,
        num_steps=num_steps,
        mesh_update_interval_s=mesh_update_interval_s,
        cir_time_steps=cir_time_steps,
        start_step=start_step,
    )


def test_snapshot_mesh_updates_preserve_per_step_interval() -> None:
    cfg = _cfg(
        duration=2.0,
        num_steps=2000,
        mesh_update_interval_s=0.143,
        cir_time_steps=1,
    )

    assert should_update_mesh_for_step(0, cfg) is True
    assert should_update_mesh_for_step(142, cfg) is False
    assert should_update_mesh_for_step(143, cfg) is True
    assert should_update_mesh_for_step(286, cfg) is True


def test_coherent_mesh_updates_are_quantized_to_acquisition_boundaries() -> None:
    cfg = _cfg(
        duration=2.0,
        num_steps=2000,
        mesh_update_interval_s=0.143,
        cir_time_steps=128,
    )

    assert should_update_mesh_for_step(0, cfg) is True
    assert should_update_mesh_for_step(127, cfg) is False
    assert should_update_mesh_for_step(128, cfg) is False
    assert should_update_mesh_for_step(255, cfg) is False
    assert should_update_mesh_for_step(256, cfg) is True
    assert should_update_mesh_for_step(384, cfg) is False
    assert should_update_mesh_for_step(512, cfg) is True


def test_coherent_aligned_mesh_updates_can_advance_every_cpi() -> None:
    cfg = _cfg(
        duration=1.0,
        num_steps=1000,
        mesh_update_interval_s=0.128,
        cir_time_steps=128,
    )

    assert should_update_mesh_for_step(0, cfg) is True
    assert should_update_mesh_for_step(128, cfg) is True
    assert should_update_mesh_for_step(256, cfg) is True


def test_coherent_without_explicit_mesh_interval_updates_each_acquisition() -> None:
    cfg = _cfg(
        duration=1.0,
        num_steps=1000,
        mesh_update_interval_s=None,
        cir_time_steps=64,
    )

    assert should_update_mesh_for_step(0, cfg) is True
    assert should_update_mesh_for_step(63, cfg) is False
    assert should_update_mesh_for_step(64, cfg) is True


def test_partial_coherent_interval_keeps_mesh_fixed_after_acquisition() -> None:
    cfg = _cfg(
        duration=0.031,
        num_steps=32,
        mesh_update_interval_s=None,
        cir_time_steps=64,
    )

    assert should_update_mesh_for_step(0, cfg) is True
    assert should_update_mesh_for_step(1, cfg) is False
    assert should_update_mesh_for_step(31, cfg) is False


def test_transceiver_orientation_preserves_yaw_pitch_roll_order(monkeypatch) -> None:
    device = SimpleNamespace(position=None, orientation=None, velocity=None)
    monkeypatch.setattr(
        "generator.core.sionna_integration.adapters.mi.Point3f", lambda x, y, z: (x, y, z)
    )

    apply_transceiver_state_to_scene(
        [device],
        step_positions=[None],
        step_orientations=[(10.0, -20.0, 30.0)],
    )

    expected = tuple(math.radians(value) for value in (10.0, -20.0, 30.0))
    np.testing.assert_allclose(np.asarray(device.orientation, dtype=np.float64), expected)


def test_normalize_override_payload_accepts_mapping_fields() -> None:
    normalized = normalize_live_overrides(
        [
            {
                "category": "tx",
                "name": "BaseStation",
                "x": "1.0",
                "y": 2,
                "z": 3.5,
                "orientation": ["10.0", -20, 30.5],
                "scale": "2.0",
            }
        ]
    )

    entry = normalized["tx"]["basestation"]
    assert entry.name == "BaseStation"
    assert entry.position == (1.0, 2.0, 3.5)
    assert entry.orientation == (10.0, -20.0, 30.5)
    assert entry.scale is None


def test_normalize_override_payload_accepts_object_position_without_hasfield() -> None:
    override = SimpleNamespace(
        category="target",
        name="Walker",
        position=SimpleNamespace(x="4.0", y=5, z=6.25),
        orientation=("90.0", 0, -15),
        scale="1.25",
    )

    normalized = normalize_live_overrides([override])

    entry = normalized["target"]["walker"]
    assert entry.name == "Walker"
    assert entry.position == (4.0, 5.0, 6.25)
    assert entry.orientation == (90.0, 0.0, -15.0)
    assert entry.scale == 1.25


def test_normalize_live_overrides_accepts_protobuf_node_override() -> None:
    override = visualizer_pb2.NodeOverride(
        name="rx_1",
        type=visualizer_pb2.NODE_TYPE_RX,
        x=7.0,
        y=8.0,
        z=9.5,
        orientation=[5.0, 6.0, 7.0],
    )

    normalized = normalize_live_overrides([override])

    entry = normalized["rx"]["rx_1"]
    assert entry.name == "rx_1"
    assert entry.position == (7.0, 8.0, 9.5)
    assert entry.orientation == (5.0, 6.0, 7.0)


def test_normalize_live_overrides_rejects_invalid_category_or_name() -> None:
    normalized = normalize_live_overrides(
        [
            {"category": "not-a-node", "name": "tx_1", "position": [1, 2, 3]},
            {"category": "tx", "name": "", "position": [1, 2, 3]},
        ]
    )

    assert normalized == {"tx": {}, "rx": {}, "target": {}}


def test_normalize_live_overrides_keeps_scale_target_only() -> None:
    normalized = normalize_live_overrides(
        [
            {"category": "tx", "name": "tx_1", "scale": 2.0},
            {"category": "target", "name": "walker", "scale": "1.5"},
        ]
    )

    assert normalized["tx"]["tx_1"].scale is None
    assert normalized["target"]["walker"].scale == 1.5


class _DummyTargetObject:
    def __init__(self) -> None:
        self.velocity = None


class _DummyTargetManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            name="target_0",
            mobility=object(),
            use_ply_position=False,
            switch_meshes=True,
        )
        self.target_object = _DummyTargetObject()
        self.current_mesh_idx = 0
        self.mesh_updates: list[int] = []
        self.expected_call_counts: list[int | None] = []

    def apply_position_snapshot(self, _pos) -> None:
        return None

    def update_mesh_for_frame(self, frame_idx: int, expected_call_count: int | None = None):
        self.mesh_updates.append(frame_idx)
        self.expected_call_counts.append(expected_call_count)
        self.current_mesh_idx += 1
        self.target_object = _DummyTargetObject()
        return self.target_object

    def apply_orientation_snapshot(self, _orient) -> None:
        return None


def test_mesh_switch_preserves_velocity_on_replaced_target_object(monkeypatch) -> None:
    cfg = _cfg(
        duration=1.0,
        num_steps=1000,
        mesh_update_interval_s=0.128,
        cir_time_steps=128,
    )
    manager = _DummyTargetManager()
    monkeypatch.setattr(
        "generator.core.sionna_integration.adapters.mi.Point3f", lambda x, y, z: (x, y, z)
    )

    apply_target_state_to_scene(
        [manager],
        step_positions=[(1.0, 2.0, 3.0)],
        step_orientations=[(0.0, 0.0, 0.0)],
        frame_idx=128,
        step_velocities=[(0.0, -1.25, 0.5)],
        simulation_config=cfg,
    )

    assert manager.mesh_updates == [128]
    assert manager.expected_call_counts == [1]
    assert manager.target_object.velocity is not None
    velocity = np.asarray(manager.target_object.velocity, dtype=np.float32).reshape(-1)
    np.testing.assert_allclose(velocity, np.array([0.0, -1.25, 0.5], dtype=np.float32))


def test_ply_aabb_positions_use_shared_mesh_update_interval(monkeypatch) -> None:
    centers = {
        "mesh_000.ply": np.array([[0.0, 0.0, 0.0]]),
        "mesh_001.ply": np.array([[10.0, 0.0, 0.0]]),
        "mesh_002.ply": np.array([[20.0, 0.0, 0.0]]),
    }

    def _read_triangle_mesh(path: str):
        return SimpleNamespace(vertices=centers[path])

    monkeypatch.setitem(
        sys.modules,
        "open3d",
        SimpleNamespace(io=SimpleNamespace(read_triangle_mesh=_read_triangle_mesh)),
    )

    assert (
        mesh_update_step_interval(
            duration=1.0,
            num_steps=3,
            mesh_update_interval_s=0.34,
        )
        == 2
    )

    positions = positions_from_ply_aabb(
        ["mesh_000.ply", "mesh_001.ply", "mesh_002.ply"],
        steps=3,
        duration=1.0,
        mesh_update_interval_s=0.34,
    )

    np.testing.assert_allclose(
        np.asarray(positions, dtype=np.float64),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_ply_aabb_positions_hold_final_mesh_with_start_and_stride(monkeypatch) -> None:
    centers = {
        "mesh_000.ply": np.array([[0.0, 0.0, 0.0]]),
        "mesh_001.ply": np.array([[10.0, 0.0, 0.0]]),
        "mesh_002.ply": np.array([[20.0, 0.0, 0.0]]),
    }

    monkeypatch.setitem(
        sys.modules,
        "open3d",
        SimpleNamespace(
            io=SimpleNamespace(
                read_triangle_mesh=lambda path: SimpleNamespace(vertices=centers[path])
            )
        ),
    )

    positions = positions_from_ply_aabb(
        ["mesh_000.ply", "mesh_001.ply", "mesh_002.ply"],
        steps=4,
        duration=1.0,
        mesh_start_index=1,
        mesh_frame_stride=2,
        mesh_end_behavior="hold_last",
    )

    np.testing.assert_allclose(
        np.asarray(positions, dtype=np.float64),
        np.asarray(
            [
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
