"""Rigid target snapshots preserve geometry across repeated pose updates."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.target.manager import TargetManager
from generator.core.target.pose import MeshPoseReference

ROOT = Path(__file__).resolve().parents[3]


def _manager(mesh: Path) -> TargetManager:
    from sionna.rt import SceneObject, load_scene

    manager = TargetManager.__new__(TargetManager)
    manager.config = SimpleNamespace(name="rigid")
    manager.target_object = SceneObject(fname=str(mesh), name="rigid")
    # Pose setters operate on scene geometry in Sionna RT; match production
    # by attaching the object before capturing or updating its pose.
    manager._test_scene = load_scene()
    manager._test_scene.edit(add=manager.target_object)
    manager._mesh_pose = MeshPoseReference(manager.target_object)
    return manager


def _vertices(manager: TargetManager) -> np.ndarray:
    import mitsuba as mi

    return np.asarray(mi.traverse(manager.target_object.mi_mesh)["vertex_positions"]).reshape(-1, 3)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_absolute_pose_is_independent_of_intermediate_snapshots(tmp_path, asymmetric):
    mesh = ROOT / "libraries/targets/cube/cube.ply"
    if asymmetric:
        mesh = tmp_path / "tetrahedron.ply"
        mesh.write_text(
            "ply\nformat ascii 1.0\nelement vertex 4\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
            "0 0 0\n2 0 0\n0 1 0\n0 0 3\n"
            "3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n"
        )
    manager = _manager(mesh)
    active_object = manager.target_object
    for step in range(256):
        orientation = (90 * np.sin(step / 50), 35 * np.cos(step / 30), step / 7)
        position = (5 + step / 100, np.sin(step / 20), 0.5 + step / 200)
        scale = 1 + 0.1 * np.sin(step / 15)
        with manager.pose_update():
            manager.apply_orientation_snapshot(orientation)
            manager.apply_position_snapshot(position)
            manager.apply_scale_snapshot(scale)

    fresh = _manager(mesh)
    fresh.apply_orientation_snapshot(orientation)
    fresh.apply_position_snapshot(position)
    fresh.apply_scale_snapshot(scale)
    assert manager.target_object is active_object
    np.testing.assert_allclose(_vertices(manager), _vertices(fresh), rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(manager.target_object.position).reshape(3), position, rtol=0, atol=1e-6
    )


def test_repeating_the_same_pose_does_not_warp_a_flat_face():
    manager = _manager(ROOT / "libraries/targets/cube/cube.ply")
    pose = (168.6900675, 14.2184928, 0.0)
    position = (4.7992157, 0.9598431, 1.7011765)
    manager.apply_orientation_snapshot(pose)
    manager.apply_position_snapshot(position)
    original = _vertices(manager).copy()
    for _ in range(256):
        manager.apply_orientation_snapshot(pose)
        manager.apply_position_snapshot(position)
        manager.apply_scale_snapshot(1.0)
    np.testing.assert_array_equal(_vertices(manager), original)


def test_current_position_and_summary_marker_use_live_engine_coordinates():
    import mitsuba as mi

    from generator.figures.generator_summary_fig import _resolve_target_marker_position

    manager = TargetManager.__new__(TargetManager)
    manager.config = SimpleNamespace(initial_position=(0.0, 0.0, 0.0))
    manager.target_object = SimpleNamespace(position=mi.Point3f(4.25, -1.5, 2.0))
    assert manager.get_current_position() == (4.25, -1.5, 2.0)
    assert _resolve_target_marker_position(manager) == (4.25, -1.5, 2.0)
