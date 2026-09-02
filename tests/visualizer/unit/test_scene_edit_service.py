"""Focused tests for persistent scene-edit renderer ownership."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from shared.geometry.transforms import parse_lightweight_shape_transform
from visualizer.src.model import RenderObjectState
from visualizer.src.services.scene_batches import SceneBatch
from visualizer.src.services.scene_edit_service import SceneEditService
from visualizer.src.services.scene_service import SceneService
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload


def _payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def test_scene_edit_replaces_payload_on_stable_render_state() -> None:
    original = _payload()
    material = MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0))
    mesh = RenderObjectState(
        id="scene:wall::mesh",
        payload=original,
        material=material,
        metadata={"type": "scene_mesh"},
    )
    entry = {
        "name": "Wall",
        "stable_mesh_id": "wall",
        "entry_type": "mesh",
        "mesh": mesh,
        "visible": True,
    }
    viz = SimpleNamespace(mesh_entries=[entry], target_entries=[])
    service = SceneEditService(viz)

    updated, _, center = service._apply_transform_to_entry_mesh(
        entry,
        mesh,
        np.asarray(original.vertices),
        np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        {"scale": 1.0, "rotation": [0.0, 0.0, 0.0], "translate": [2.0, 0.0, 0.0]},
    )

    assert updated is mesh
    assert entry["mesh"] is mesh
    assert mesh.id == "scene:wall::mesh"
    assert mesh.material is material
    assert mesh.payload is not original
    np.testing.assert_allclose(center, [7.0 / 3.0, 1.0 / 3.0, 0.0])


def test_scene_edit_serializes_multi_axis_rotation_for_strict_round_trip() -> None:
    shape = ET.fromstring("""<shape type="ply" id="wall">
  <string name="filename" value="wall.ply"/>
  <transform name="to_world">
    <scale value="1"/>
    <translate value="0 0 0"/>
  </transform>
</shape>""")
    service = SceneEditService(SimpleNamespace())
    state = {
        "scale": 2.5,
        "rotation": [30.0, 20.0, 10.0],
        "translate": [1.0, 2.0, 3.0],
    }

    service._update_xml_transform_from_state(shape, state)

    transform = shape.find("transform[@name='to_world']")
    assert transform is not None
    assert [operation.tag for operation in transform] == [
        "scale",
        "rotate",
        "rotate",
        "rotate",
        "translate",
    ]
    assert [
        next(axis for axis in ("x", "y", "z") if operation.get(axis) == "1")
        for operation in transform.findall("rotate")
    ] == ["x", "y", "z"]
    assert (
        parse_lightweight_shape_transform(
            shape,
            source_xml="scene.xml",
            shape_index=0,
        )
        == state
    )


def test_pov_hidden_target_edit_delegates_complete_target_snapshot() -> None:
    mesh = RenderObjectState(
        id="target:walker::mesh",
        payload=_payload(),
        visible=True,
        metadata={"type": "target_mesh"},
    )
    entry = {
        "name": "Walker",
        "target_name": "walker",
        "entry_type": "target",
        "node_index": 0,
        "mesh": mesh,
        "visible": True,
        "_frame_visible": True,
    }
    ensure_object = Mock(return_value=True)
    sync_snapshot = Mock(return_value=True)
    renderer = SimpleNamespace(ensure_object=ensure_object, update_renderer=Mock())
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_snapshot),
    )
    service = SceneEditService(viz)

    service._refresh_mesh_in_window(entry, mesh)

    sync_snapshot.assert_called_once_with(entry)
    ensure_object.assert_not_called()
    renderer.update_renderer.assert_called_once_with()


def test_scene_edit_delegates_unmerged_entry_snapshot_to_scene_service() -> None:
    mesh = RenderObjectState(
        id="scene:wall::mesh",
        payload=_payload(),
        metadata={"type": "scene_mesh"},
    )
    entry = {
        "name": "Wall",
        "stable_mesh_id": "wall",
        "entry_type": "mesh",
        "mesh": mesh,
        "visible": True,
    }
    ensure_object = Mock(return_value=True)
    sync_snapshot = Mock(return_value=True)
    renderer = SimpleNamespace(ensure_object=ensure_object, update_renderer=Mock())
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        scene_service=SimpleNamespace(sync_scene_entry_snapshot=sync_snapshot),
    )
    service = SceneEditService(viz)

    service._refresh_mesh_in_window(entry, mesh)

    sync_snapshot.assert_called_once_with(entry, geometry_changed=True)
    ensure_object.assert_not_called()
    renderer.update_renderer.assert_called_once_with()


def test_merged_scene_edit_rebuilds_aggregate_without_ensuring_member() -> None:
    edited_mesh = RenderObjectState(
        id="scene:edited::mesh",
        payload=_payload(),
        metadata={"type": "scene_mesh"},
    )
    neighbor_mesh = RenderObjectState(
        id="scene:neighbor::mesh",
        payload=MeshPayload(
            vertices=np.asarray(
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
        metadata={"type": "scene_mesh"},
    )
    edited_entry = {
        "name": "Edited",
        "stable_mesh_id": "edited",
        "entry_type": "mesh",
        "mesh": edited_mesh,
        "visible": True,
    }
    neighbor_entry = {
        "name": "Neighbor",
        "stable_mesh_id": "neighbor",
        "entry_type": "mesh",
        "mesh": neighbor_mesh,
        "visible": True,
    }
    ensure_object = Mock(return_value=True)
    renderer = SimpleNamespace(
        batch_updates=nullcontext,
        ensure_object=ensure_object,
        remove_object=Mock(return_value=True),
        update_renderer=Mock(),
    )
    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        mesh_entries=[edited_entry, neighbor_entry],
    )
    scene_service = SceneService(viz)
    viz.scene_service = scene_service
    edit_service = SceneEditService(viz)

    group_name = "scene:merged_test::mesh"
    edited_id = id(edited_mesh)
    neighbor_id = id(neighbor_mesh)
    meshes = [edited_mesh, neighbor_mesh]
    member_ids = [edited_id, neighbor_id]
    material = scene_service._scene_entry_base_material(edited_entry).payload
    baseline_geometry = scene_service._merge_scene_mesh_payloads(meshes)
    assert baseline_geometry is not None
    baseline_sources = tuple((mesh.id, mesh.payload) for mesh in meshes)
    scene_service._scene_batches.add_batch(
        SceneBatch(
            name=group_name,
            material_signature=scene_service._scene_material_signature(
                edited_entry,
                material,
            ),
            member_mesh_ids=member_ids,
            baseline_geometry=baseline_geometry,
            baseline_sources=baseline_sources,
            presentation={
                "geometry": baseline_geometry,
                "geometry_sources": baseline_sources,
            },
        )
    )
    for mesh_id, entry, mesh in zip(
        member_ids,
        [edited_entry, neighbor_entry],
        meshes,
        strict=True,
    ):
        mesh.material = material
        scene_service._scene_batches.register_entry(mesh_id, entry)

    translated = MeshPayload(
        vertices=np.asarray(edited_mesh.payload.vertices) + np.asarray([10.0, 0.0, 0.0]),
        triangles=np.asarray(edited_mesh.payload.triangles),
    )
    edited_mesh.replace_payload(translated)

    edit_service._refresh_mesh_in_window(edited_entry, edited_mesh)

    ensure_object.assert_called_once()
    aggregate = ensure_object.call_args.args[0]
    assert aggregate.id == group_name
    np.testing.assert_allclose(aggregate.payload.vertices[:3], translated.vertices)
    np.testing.assert_allclose(aggregate.payload.vertices[3:], neighbor_mesh.payload.vertices)
    assert aggregate.id not in {edited_mesh.id, neighbor_mesh.id}
    renderer.remove_object.assert_not_called()
    renderer.update_renderer.assert_called_once_with()
