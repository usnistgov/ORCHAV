"""Scene outline synchronization tests for the renderer object contract."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from visualizer.src.model import RenderObjectState
from visualizer.src.services.scene_appearance_service import SceneAppearanceService
from visualizer.src.services.scene_batches import SceneBatch, SceneBatchRegistry
from visualizer.src.services.scene_service import SceneService
from visualizer.src.types.render_payloads import MeshPayload


class _Renderer:
    def __init__(self) -> None:
        self.objects = {}
        self.visibility = {}
        self.redraws = 0
        self.ensure_calls = []
        self.set_visible_calls = []
        self.remove_calls = []
        self.ensure_success = True
        self.remove_success = True
        self.batch_entries = 0

    @contextmanager
    def batch_updates(self):
        self.batch_entries += 1
        yield

    def ensure_object(self, obj) -> bool:
        self.ensure_calls.append(obj)
        if not self.ensure_success:
            return False
        self.objects[obj.id] = obj
        self.visibility[obj.id] = obj.visible
        return True

    def set_visible(self, object_id: str, visible: bool) -> bool:
        self.set_visible_calls.append((object_id, bool(visible)))
        self.visibility[object_id] = bool(visible)
        return True

    def remove_object(self, object_id: str) -> bool:
        self.remove_calls.append(object_id)
        if not self.remove_success:
            return False
        self.objects.pop(object_id, None)
        self.visibility.pop(object_id, None)
        return True

    def request_redraw(self) -> None:
        self.redraws += 1

    def update_renderer(self) -> None:
        self.redraws += 1


def _mesh(
    offset: float = 0.0,
    *,
    object_id: str = "scene:building_a::mesh",
) -> RenderObjectState:
    return RenderObjectState(
        id=object_id,
        payload=MeshPayload(
            vertices=np.asarray(
                [[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _service():
    renderer = _Renderer()
    entry = {"name": "Building A", "mesh": _mesh(), "visible": True}
    viz = SimpleNamespace(
        renderer=renderer,
        mesh_entries=[entry],
        outline_color=[0.05, 0.05, 0.05],
        outlines_enabled=False,
        vis=object(),
        vis_initialized=True,
    )
    viz.scene_service = SceneService(viz)
    return SceneAppearanceService(viz), viz, renderer, entry


def _register_scene_batch(
    viz,
    entries: list[dict],
    *,
    group_name: str,
    geometry: RenderObjectState,
) -> SceneBatch:
    """Install one typed scene batch in the canonical fixture registry."""
    scene_service = viz.scene_service
    registry: SceneBatchRegistry = scene_service._scene_batches
    member_ids = [id(entry["mesh"]) for entry in entries]
    batch = SceneBatch(
        name=group_name,
        material_signature=("default", entries[0]["mesh"].material),
        member_mesh_ids=member_ids,
        presentation={"geometry": geometry},
    )
    registry.add_batch(batch)
    for mesh_id, entry in zip(member_ids, entries, strict=True):
        registry.register_entry(mesh_id, entry)
    scene_service._merge_enabled = True
    return batch


def test_scene_outline_enable_disable_uses_render_object_state() -> None:
    service, _viz, renderer, entry = _service()

    service.set_edge_visibility(True)

    outline = entry["outline_geometry"]
    assert isinstance(outline, RenderObjectState)
    assert outline.id == "scene_outline_scene:building_a::mesh"
    assert outline.is_edge is True
    assert renderer.visibility[outline.id] is True
    assert len(renderer.ensure_calls) == 1
    assert renderer.set_visible_calls == []

    service.set_edge_visibility(True)

    # The service submits visible desired state on every synchronization.
    # Detecting the no-op belongs to the renderer's applied-object cache.
    assert len(renderer.ensure_calls) == 2

    service.set_edge_visibility(False)

    assert outline.visible is False
    assert renderer.visibility[outline.id] is False
    assert len(renderer.ensure_calls) == 3
    assert renderer.set_visible_calls == []
    assert renderer.redraws == 0


def test_scene_outline_follows_parent_visibility() -> None:
    service, viz, renderer, entry = _service()
    viz.outlines_enabled = True
    service.sync_entry_outline_visibility(entry)
    outline = entry["outline_geometry"]
    assert renderer.visibility[outline.id] is True

    entry["visible"] = False
    service.sync_entry_outline_visibility(entry)

    assert outline.visible is False
    assert renderer.visibility[outline.id] is False
    assert renderer.set_visible_calls == []


def test_failed_scene_outline_ensure_is_retried_and_partial_state_is_hidden() -> None:
    service, _viz, renderer, entry = _service()
    renderer.ensure_success = False

    service.set_edge_visibility(True)

    outline = entry["outline_geometry"]
    assert outline.visible is True
    assert len(renderer.ensure_calls) == 1
    assert renderer.set_visible_calls == []
    assert renderer.redraws == 0

    service.set_edge_visibility(True)

    assert len(renderer.ensure_calls) == 2
    assert renderer.redraws == 0

    renderer.ensure_success = True
    service.set_edge_visibility(False)

    assert outline.visible is False
    assert len(renderer.ensure_calls) == 3
    assert renderer.visibility[outline.id] is False


def test_failed_hidden_transition_is_retried_until_renderer_accepts_it() -> None:
    service, _viz, renderer, entry = _service()
    service.set_edge_visibility(True)
    outline = entry["outline_geometry"]

    renderer.ensure_success = False
    service.set_edge_visibility(False)
    service.set_edge_visibility(False)

    assert [obj.visible for obj in renderer.ensure_calls[-2:]] == [False, False]
    assert outline.visible is False

    renderer.ensure_success = True
    service.set_edge_visibility(False)
    accepted_call_count = len(renderer.ensure_calls)

    # Once the pending transition succeeds, an already-hidden outline is a
    # service no-op. The renderer remains the sole owner of applied snapshots.
    service.set_edge_visibility(False)
    assert len(renderer.ensure_calls) == accepted_call_count
    assert renderer.visibility[outline.id] is False


def test_hidden_never_materialized_outline_is_removed_without_renderer_query() -> None:
    service, _viz, renderer, entry = _service()

    assert service.sync_scene_entry_outline_snapshot(
        entry,
        visible=False,
        rebuild=True,
    )
    assert renderer.ensure_calls == []
    assert "outline_geometry" not in entry

    assert service.remove_scene_entry_outline(entry)
    assert renderer.remove_calls == []


def test_failed_outline_attempt_is_removed_before_owner_is_released() -> None:
    service, _viz, renderer, entry = _service()
    renderer.ensure_success = False

    assert not service.sync_scene_entry_outline_snapshot(
        entry,
        visible=True,
        rebuild=True,
    )
    outline = entry["outline_geometry"]

    assert service.remove_scene_entry_outline(entry)
    assert renderer.remove_calls == [outline.id]


def test_merged_outline_rebuild_reuses_id_and_replaces_payload() -> None:
    service, _viz, renderer, _entry = _service()
    group_info = {"geometry": _mesh()}
    group_name = "scene:merged_group::mesh"

    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=True,
        rebuild=True,
    )
    outline = group_info["_merged_outline"]
    first_payload = outline.payload
    first_points = np.array(outline.payload.points, copy=True)
    assert renderer.objects[outline.id].payload is outline.payload

    group_info["geometry"] = _mesh(offset=4.0)
    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=True,
        rebuild=True,
    )

    assert group_info["_merged_outline"] is outline
    assert outline.payload is not first_payload
    assert not np.array_equal(outline.payload.points, first_points)
    assert renderer.objects[outline.id].payload is outline.payload
    assert len(renderer.ensure_calls) == 2
    assert renderer.set_visible_calls == []


def test_hidden_merged_outline_defers_rebuilt_payload_until_reenabled() -> None:
    service, _viz, renderer, _entry = _service()
    group_info = {"geometry": _mesh()}
    group_name = "scene:merged_group::mesh"

    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=True,
        rebuild=True,
    )
    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=False,
        rebuild=False,
    )
    assert len(renderer.ensure_calls) == 2
    outline = group_info["_merged_outline"]
    original_payload = outline.payload

    group_info["geometry"] = _mesh(offset=5.0)
    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=False,
        rebuild=True,
    )
    assert len(renderer.ensure_calls) == 2
    assert outline.payload is original_payload

    assert service.sync_merged_outline_geometry(
        group_name,
        group_info,
        visible=True,
        rebuild=False,
    )
    assert len(renderer.ensure_calls) == 3
    assert outline.payload is not original_payload
    assert renderer.objects[outline.id].payload is outline.payload


def test_visible_outline_is_resubmitted_after_renderer_replacement() -> None:
    service, viz, first_renderer, entry = _service()

    service.set_edge_visibility(True)
    assert len(first_renderer.ensure_calls) == 1

    second_renderer = _Renderer()
    viz.renderer = second_renderer
    service.set_edge_visibility(True)

    assert len(first_renderer.ensure_calls) == 1
    assert len(second_renderer.ensure_calls) == 1
    assert entry["outline_geometry"].id in second_renderer.objects


def test_individual_hidden_rebuild_is_deferred_until_reenabled() -> None:
    service, _viz, renderer, entry = _service()

    assert service.sync_scene_entry_outline_snapshot(
        entry,
        visible=True,
        rebuild=True,
    )
    outline = entry["outline_geometry"]
    original_payload = outline.payload
    assert service.sync_scene_entry_outline_snapshot(
        entry,
        visible=False,
    )

    entry["mesh"] = _mesh(offset=7.0)
    assert service.sync_scene_entry_outline_snapshot(
        entry,
        visible=False,
        rebuild=True,
    )

    assert outline.payload is original_payload
    assert len(renderer.ensure_calls) == 2

    assert service.sync_scene_entry_outline_snapshot(
        entry,
        visible=True,
    )
    assert outline.payload is not original_payload
    assert renderer.objects[outline.id].payload is outline.payload
    assert len(renderer.ensure_calls) == 3


def test_scene_outline_ids_use_stable_mesh_ids_not_display_names() -> None:
    service, viz, renderer, first_entry = _service()
    second_entry = {
        "name": first_entry["name"],
        "mesh": _mesh(object_id="scene:building_b::mesh"),
        "visible": True,
    }
    viz.mesh_entries = [first_entry, second_entry]

    service.set_edge_visibility(True)

    outline_ids = {obj.id for obj in renderer.ensure_calls}
    assert outline_ids == {
        "scene_outline_scene:building_a::mesh",
        "scene_outline_scene:building_b::mesh",
    }


def test_edge_toggle_does_not_request_an_extra_post_batch_redraw() -> None:
    service, _viz, renderer, _entry = _service()

    service.set_edge_visibility(True)
    service.set_edge_visibility(False)

    assert renderer.batch_entries == 2
    assert renderer.redraws == 0


def test_merged_outline_toggle_remains_subordinate_to_parent_visibility() -> None:
    service, viz, renderer, entry = _service()
    mesh = entry["mesh"]
    group_name = "scene:merged_group::mesh"
    batch = _register_scene_batch(
        viz,
        [entry],
        group_name=group_name,
        geometry=mesh,
    )
    entry["visible"] = False
    mesh.visible = False

    service.set_edge_visibility(True)

    assert batch.current_partition.aggregate_member_ids == ()
    assert "_merged_outline" not in batch
    assert renderer.ensure_calls == []

    entry["visible"] = True
    mesh.visible = True
    service.set_edge_visibility(True)

    assert batch.current_partition.aggregate_member_ids == (id(mesh),)
    outline = batch["_merged_outline"]
    assert outline.visible is True
    assert renderer.visibility[outline.id] is True
    assert len(renderer.ensure_calls) == 1


def test_highlighted_merged_member_uses_an_individual_outline_owner() -> None:
    service, viz, renderer, regular_entry = _service()
    regular_mesh = regular_entry["mesh"]
    highlighted_mesh = _mesh(object_id="scene:highlighted::mesh")
    highlighted_entry = {
        "name": "Building A",
        "mesh": highlighted_mesh,
        "visible": True,
        "highlighted": True,
    }
    viz.mesh_entries = [regular_entry, highlighted_entry]
    group_name = "scene:merged_group::mesh"
    batch = _register_scene_batch(
        viz,
        [regular_entry, highlighted_entry],
        group_name=group_name,
        geometry=regular_mesh,
    )

    service.set_edge_visibility(True)

    assert batch.current_partition.aggregate_member_ids == (id(regular_mesh),)
    assert batch.current_partition.individual_member_ids == (id(highlighted_mesh),)
    group_outline = batch["_merged_outline"]
    highlighted_outline = highlighted_entry["outline_geometry"]
    assert group_outline.visible is True
    assert highlighted_outline.visible is True
    assert highlighted_outline.id == "scene_outline_scene:highlighted::mesh"
    assert regular_entry.get("outline_geometry") is None
    assert {obj.id for obj in renderer.ensure_calls} == {
        group_outline.id,
        highlighted_outline.id,
    }

    service.set_edge_visibility(False)

    assert renderer.visibility[group_outline.id] is False
    assert renderer.visibility[highlighted_outline.id] is False
    assert len(renderer.ensure_calls) == 4
    assert renderer.redraws == 0
