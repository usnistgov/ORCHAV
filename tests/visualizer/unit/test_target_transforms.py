from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.model import RenderObjectState, Transform
from visualizer.src.scene.target_transforms import (
    build_sionna_rotation_matrix,
    closest_proper_rotation,
    mesh_aabb_center,
    sionna_orientation_from_rotation_matrix,
    sionna_orientation_from_transform,
    target_entry_anchor_position,
    target_mesh_world_center,
)
from visualizer.src.types.render_payloads import MeshPayload


def _payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def _state() -> RenderObjectState:
    return RenderObjectState(id="target:test::mesh", payload=_payload())


def test_target_mesh_world_center_applies_render_state_transform() -> None:
    state = RenderObjectState(
        id="target::mesh",
        payload=_payload(),
        world_transform=Transform.from_translation([9.0, 18.0, 27.0]),
    )

    np.testing.assert_allclose(mesh_aabb_center(state), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(target_mesh_world_center(state), [10.0, 20.0, 30.0])


def test_target_entry_anchor_prefers_position_fields_before_mesh_center() -> None:
    state = _state()

    np.testing.assert_allclose(
        target_entry_anchor_position({"mesh": state, "_target_position": [10.0, 20.0, 30.0]}),
        [10.0, 20.0, 30.0],
    )
    np.testing.assert_allclose(
        target_entry_anchor_position(
            {
                "mesh": state,
                "position": [4.0, 5.0, 6.0],
                "_target_position": [10.0, 20.0, 30.0],
            }
        ),
        [4.0, 5.0, 6.0],
    )


def test_target_entry_anchor_falls_back_to_render_state_aabb_center() -> None:
    np.testing.assert_allclose(target_entry_anchor_position({"mesh": _state()}), [1.0, 2.0, 3.0])


def test_mesh_aabb_center_rejects_native_target_mesh_fallbacks() -> None:
    class NativeMeshLike:
        vertices = [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]

        def get_center(self):
            return [3.0, 4.0, 5.0]

    with pytest.raises(TypeError, match="target render state"):
        mesh_aabb_center(NativeMeshLike())


def test_raw_mesh_payload_is_not_a_target_render_owner() -> None:
    with pytest.raises(TypeError, match="target render state"):
        mesh_aabb_center(_payload())
    assert target_entry_anchor_position({"mesh": _payload()}) is None


def test_target_entry_anchor_declines_unsupported_mesh() -> None:
    assert target_entry_anchor_position({"mesh": object()}) is None


def test_sionna_orientation_round_trip_discards_uniform_scale() -> None:
    expected = np.radians([37.0, -24.0, 11.0])
    rotation = build_sionna_rotation_matrix(*expected)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation * 2.5

    np.testing.assert_allclose(sionna_orientation_from_transform(transform), expected, atol=1e-10)
    np.testing.assert_allclose(
        sionna_orientation_from_rotation_matrix(rotation), expected, atol=1e-10
    )


def test_closest_proper_rotation_rejects_invalid_values() -> None:
    assert closest_proper_rotation(np.eye(4)) is None
    assert closest_proper_rotation(np.full((3, 3), np.nan)) is None
    assert sionna_orientation_from_transform(np.eye(3)) is None
