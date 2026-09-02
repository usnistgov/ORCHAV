"""Unit tests for TX/RX node renderer synchronization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from visualizer.src.model import RenderObjectState, make_text_label_state
from visualizer.src.services.node_render_sync import NodeRenderSync
from visualizer.src.services.object_identity import make_node_geometry_name
from visualizer.src.types.render_payloads import MeshPayload, TextLabelPayload


class _EntityRecorder:
    def __init__(self) -> None:
        self.synced = None
        self.removed = []

    def sync_entities(self, entities):
        self.synced = list(entities)
        return {"ok": True}

    def remove_entity(self, entity, *, label_id=None):
        self.removed.append((entity, label_id))
        return True


def _mesh_state(kind: str = "tx", index: int = 0) -> RenderObjectState:
    return RenderObjectState(
        id=make_node_geometry_name(kind, index, "marker"),
        payload=MeshPayload(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _sync(renderer=None):
    recorder = _EntityRecorder()
    viz = SimpleNamespace(renderer=renderer, vis_initialized=True, vis=object())
    return NodeRenderSync(viz, recorder), recorder


def test_entity_renderer_readiness_and_delegation() -> None:
    renderer = SimpleNamespace(ensure_object=Mock(), _initialized=True)
    sync, recorder = _sync(renderer)
    entities = [object(), object()]

    assert sync.entity_renderer_available()
    assert sync.entity_renderer_ready()
    assert sync.sync_entities(entities) == {"ok": True}
    assert recorder.synced == entities


def test_remove_marker_entity_uses_stable_ids() -> None:
    sync, recorder = _sync(SimpleNamespace())
    sync.remove_marker_entity("tx", 2)
    assert recorder.removed == [
        (
            make_node_geometry_name("tx", 2, "marker"),
            make_node_geometry_name("tx", 2, "label"),
        )
    ]


def test_render_handle_visibility_uses_object_contract() -> None:
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        set_visible=Mock(return_value=True),
    )
    sync, _ = _sync(renderer)
    handle = _mesh_state()

    assert sync.set_render_handle_visibility(handle, False)
    assert handle.visible is True
    renderer.ensure_object.assert_not_called()
    renderer.set_visible.assert_called_once_with(handle.id, False)


def test_label_helpers_use_common_object_contract() -> None:
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        remove_object=Mock(return_value=True),
    )
    sync, _ = _sync(renderer)
    label = make_text_label_state(
        "target:walker::label",
        "Walker",
        [0.8, 0.8, 0.8],
    )

    assert sync.sync_label(
        label_id="target:walker::label",
        label=label,
        visible=True,
        anchor_position=[1, 2, 3],
        offset=[0.5, 0, 1],
    )
    assert sync.remove_label("target:walker::label")
    ensured = renderer.ensure_object.call_args.args[0]
    assert ensured.id == "target:walker::label"
    assert isinstance(ensured.payload, TextLabelPayload)
    assert ensured.payload.text == "Walker"
    assert ensured.visible is True
    np.testing.assert_allclose(ensured.transform.translation, [1.5, 2.0, 4.0])
    assert ensured.metadata["layout_anchor"] == (1.0, 2.0, 3.0)
    assert ensured.metadata["layout_offset"] == (0.5, 0.0, 1.0)
    renderer.remove_object.assert_called_once_with("target:walker::label")


def test_repeated_label_sync_keeps_stable_payload_and_hidden_state() -> None:
    renderer = SimpleNamespace(ensure_object=Mock(return_value=True))
    sync, _ = _sync(renderer)
    label = make_text_label_state(
        "node:rx_0::label",
        "RX1",
        [0.0, 0.0, 1.0],
    )

    assert sync.sync_label(label_id=label.id, label=label, visible=True)
    assert sync.sync_label(label_id=label.id, label=label, visible=False)

    first, second = [call.args[0] for call in renderer.ensure_object.call_args_list]
    assert first.id == second.id == label.id
    assert first.payload is second.payload is label.payload
    assert first.visible is True
    assert second.visible is False
    assert label.visible is True
