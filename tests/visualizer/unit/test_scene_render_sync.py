"""Unit tests for renderer-neutral scene synchronization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from visualizer.src.model import RenderObjectState, Transform, make_text_label_state
from visualizer.src.services.scene_render_sync import (
    SceneRenderSync,
    merge_scene_mesh_payloads,
    scene_mesh_has_triangle_uvs,
    scene_mesh_payload,
)
from visualizer.src.types.render_payloads import MeshPayload


def _payload(offset: float = 0.0, *, with_uvs: bool = False) -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray([[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]], dtype=float),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        triangle_uvs=(np.asarray([[0, 0], [1, 0], [0, 1]], dtype=float) if with_uvs else None),
    )


def _state(name: str = "wall") -> RenderObjectState:
    return RenderObjectState(
        id=f"scene:{name}::mesh",
        payload=_payload(),
        world_transform=Transform.from_translation([1, 2, 3]),
    )


def test_scene_payload_helpers_remain_renderer_neutral() -> None:
    first = _payload()
    second = _payload(2)
    state = RenderObjectState(id="scene:state::mesh", payload=first)

    assert scene_mesh_payload(first) is first
    assert scene_mesh_payload(state) is first
    assert scene_mesh_payload(object()) is None
    assert scene_mesh_has_triangle_uvs(_payload(with_uvs=True))
    merged = merge_scene_mesh_payloads([first, object(), second])
    assert merged is not None
    np.testing.assert_array_equal(
        merged.triangles,
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )


def test_labels_use_common_render_object_contract() -> None:
    renderer = SimpleNamespace(ensure_object=Mock(return_value=True))
    sync = SceneRenderSync(SimpleNamespace(renderer=renderer))
    label = make_text_label_state("scene:wall::label", "Wall", [0.8, 0.8, 0.8])

    assert sync.sync_label_geometry(name="scene:wall::label", geometry=label, visible=False)
    assert sync.sync_label_geometry(name="scene:wall::label", geometry=label, visible=True)
    assert [call.args[0].visible for call in renderer.ensure_object.call_args_list] == [False, True]
    assert label.visible is True

    label.visible = False
    assert sync.sync_label_geometry(name="scene:wall::label", geometry=label, visible=False)
    assert renderer.ensure_object.call_args.args[0].visible is False
    assert label.visible is False


def test_render_state_uses_object_contract() -> None:
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        ensure_named_geometry=Mock(return_value=True),
        set_visible=Mock(return_value=True),
        remove_object=Mock(return_value=True),
    )
    sync = SceneRenderSync(SimpleNamespace(renderer=renderer))
    state = _state()

    assert sync.ensure_object(state, effective_visible=False)
    obj = renderer.ensure_object.call_args.args[0]
    assert obj.id == state.id
    assert obj.visible is False
    renderer.ensure_named_geometry.assert_not_called()
    assert sync.set_object_visibility(state.id, True)
    renderer.set_visible.assert_called_once_with(state.id, True)
    assert sync.remove_object(state)
    renderer.remove_object.assert_called_once_with(state.id)


def test_render_state_does_not_require_or_fall_back_to_named_geometry() -> None:
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        ensure_named_geometry=Mock(return_value=True),
    )
    sync = SceneRenderSync(SimpleNamespace(renderer=renderer))
    state = _state()

    assert sync.ensure_object(state, effective_visible=True)
    renderer.ensure_named_geometry.assert_not_called()

    common_only = SimpleNamespace(ensure_object=Mock(return_value=True))
    common_sync = SceneRenderSync(SimpleNamespace(renderer=common_only))
    assert common_sync.ensure_object(state, effective_visible=True)


def test_material_update_is_carried_by_render_object() -> None:
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        ensure_named_geometry=Mock(return_value=True),
    )
    sync = SceneRenderSync(SimpleNamespace(renderer=renderer))
    state = _state("glass")

    assert sync.ensure_object(
        state,
        material={"base_color": [0.2, 0.3, 0.4, 0.5], "roughness": 0.25},
        effective_visible=True,
    )
    obj = renderer.ensure_object.call_args.args[0]
    assert obj.material_payload.base_color == (0.2, 0.3, 0.4, 0.5)
    assert obj.material_payload.roughness == 0.25
    renderer.ensure_named_geometry.assert_not_called()
