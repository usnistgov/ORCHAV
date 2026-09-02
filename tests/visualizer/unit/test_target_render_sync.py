"""Unit tests for renderer-neutral target synchronization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from visualizer.src.model import RenderObjectState, make_text_label_state
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.target_render_sync import TargetRenderSync
from visualizer.src.types.render_payloads import (
    MeshPayload,
    SurfaceColorSource,
    TextLabelPayload,
)


class _EntityRecorder:
    def __init__(self, results: list[bool] | None = None) -> None:
        self.entities = []
        self._results = list(results or [])

    def sync_entity(self, entity):
        self.entities.append(entity)
        return self._results.pop(0) if self._results else True


def _payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def _state(name: str = "walker") -> RenderObjectState:
    return RenderObjectState(
        id=f"target:{name}::mesh",
        payload=_payload(),
        metadata={"type": "target_mesh"},
    )


def _sync(renderer=None, *, results: list[bool] | None = None):
    if renderer is not None and not hasattr(renderer, "capabilities"):
        renderer.capabilities = RendererCapabilities()
    recorder = _EntityRecorder(results)
    visualizer = SimpleNamespace(renderer=renderer, pipeline=None)
    return TargetRenderSync(visualizer, recorder), recorder


def test_render_state_builds_target_entity() -> None:
    sync, recorder = _sync(SimpleNamespace())
    state = _state()

    assert sync.sync_render_handle(state)
    [entity] = recorder.entities
    assert entity.entity_id == "target:walker"
    assert entity.render_object.payload is state.payload


def test_mesh_state_uses_effective_visibility_and_vertex_color_policy() -> None:
    sync, recorder = _sync(SimpleNamespace())
    state = _state("textured")
    state.replace_payload(
        MeshPayload(
            vertices=state.payload.vertices,
            triangles=state.payload.triangles,
            color_source=SurfaceColorSource.VERTEX,
        )
    )

    assert sync.sync_mesh_geometry(
        state,
        state.id,
        visible=False,
    )

    assert state.visible is True
    assert state.payload.color_source is SurfaceColorSource.VERTEX
    assert recorder.entities[0].visible is False


def test_raw_payload_and_mismatched_ids_never_create_named_aliases() -> None:
    renderer = SimpleNamespace(ensure_named_geometry=Mock(return_value=True))
    sync, recorder = _sync(renderer)
    state = _state()

    assert not sync.sync_mesh_geometry(_payload(), state.id)
    assert not sync.sync_mesh_geometry(state, "target:alias::mesh")
    assert not sync.sync_outline_geometry(state, "target:walker::outline")

    assert recorder.entities == []
    renderer.ensure_named_geometry.assert_not_called()


def test_labels_use_common_object_contract_and_carry_hidden_state() -> None:
    sync, recorder = _sync(SimpleNamespace())
    label = make_text_label_state(
        "target:walker::label",
        "Walker",
        [0.8, 0.8, 0.8],
    )

    assert sync.sync_target_label(
        geometry_name=label.id,
        label=label,
        visible=True,
        anchor_position=[1, 2, 3],
        offset=[0.5, 0, 1],
    )
    first = recorder.entities[-1].render_object
    assert first.id == label.id
    assert isinstance(first.payload, TextLabelPayload)
    assert first.payload.text == "Walker"
    assert first.visible is True
    np.testing.assert_allclose(first.transform.translation, [1.5, 2.0, 4.0])

    assert sync.sync_target_label(
        geometry_name=label.id,
        label=label,
        visible=False,
    )
    second = recorder.entities[-1].render_object
    assert second.id == first.id
    assert second.payload is first.payload
    assert second.visible is False
    assert label.visible is True


def test_label_failure_propagates_without_recording_an_alias() -> None:
    renderer = SimpleNamespace(ensure_label=Mock(), ensure_named_geometry=Mock())
    sync, recorder = _sync(renderer, results=[False])
    label = make_text_label_state(
        "target:walker::label",
        "Walker",
        [0.8, 0.8, 0.8],
    )

    assert not sync.sync_target_label(
        geometry_name=label.id,
        label=label,
        visible=True,
    )

    assert len(recorder.entities) == 1
    renderer.ensure_label.assert_not_called()
    renderer.ensure_named_geometry.assert_not_called()


def test_repeated_target_label_sync_keeps_one_stable_payload_identity() -> None:
    sync, recorder = _sync(SimpleNamespace())
    label = make_text_label_state(
        "target:walker::label",
        "Walker",
        [0.8, 0.8, 0.8],
        position=[1.0, 2.0, 3.0],
    )

    assert sync.sync_target_label(geometry_name=label.id, label=label, visible=True)
    assert sync.sync_target_label(geometry_name=label.id, label=label, visible=True)

    first, second = [entity.render_object for entity in recorder.entities]
    assert first.id == second.id == label.id
    assert first.payload is second.payload is label.payload


def test_outline_uses_exact_stable_id_and_effective_visibility() -> None:
    sync, recorder = _sync(SimpleNamespace())
    outline = _state("outline")
    outline.id = "target:outline::outline"
    outline.visible = True

    assert sync.sync_outline_geometry(outline, outline.id, visible=False)

    [entity] = recorder.entities
    assert entity.render_object.id == outline.id
    assert entity.render_object.visible is False
    assert outline.visible is True


def test_benchmark_metrics_cover_common_renderer_handoff() -> None:
    sync, _ = _sync(SimpleNamespace())
    sync.reset_benchmark_metrics(enabled=True)
    mesh = _state()
    label = make_text_label_state(
        "target:walker::label",
        "Target",
        [0.8, 0.8, 0.8],
    )

    assert sync.sync_mesh_geometry(mesh, mesh.id, visible=True)
    assert sync.sync_target_label(
        geometry_name=label.id,
        label=label,
        visible=True,
    )

    metrics = sync.get_benchmark_metrics()
    assert metrics["target_handoff_geometry_sync_count"] == 2.0
    assert metrics["target_handoff_sync_entity_count"] == 2.0
    assert metrics["target_handoff_sync_entity_success_count"] == 2.0
    assert metrics["target_label_ensure_call_count"] == 1.0


def test_mesh_vertex_stream_success_bypasses_full_entity_sync() -> None:
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(mesh_vertex_stream_updates=True),
        update_mesh_vertex_stream=Mock(return_value=True),
    )
    sync, recorder = _sync(renderer)
    mesh = _state()

    assert sync.sync_mesh_geometry(mesh, mesh.id, visible=False)

    renderer.update_mesh_vertex_stream.assert_called_once()
    [snapshot] = renderer.update_mesh_vertex_stream.call_args.args
    assert snapshot.payload is mesh.payload
    assert snapshot.visible is False
    assert recorder.entities == []


def test_mesh_vertex_stream_rejection_falls_back_to_full_entity_sync() -> None:
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(mesh_vertex_stream_updates=True),
        update_mesh_vertex_stream=Mock(return_value=False),
    )
    sync, recorder = _sync(renderer)
    mesh = _state()

    assert sync.sync_mesh_geometry(mesh, mesh.id, visible=True)

    renderer.update_mesh_vertex_stream.assert_called_once()
    assert len(recorder.entities) == 1


def test_mesh_vertex_stream_is_not_called_without_capability() -> None:
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        update_mesh_vertex_stream=Mock(return_value=True),
    )
    sync, recorder = _sync(renderer)

    assert sync.sync_mesh_geometry(_state(), "target:walker::mesh", visible=True)

    renderer.update_mesh_vertex_stream.assert_not_called()
    assert len(recorder.entities) == 1
