from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from visualizer.src.model import RenderObject, Transform, VisualEntity, make_text_label_state
from visualizer.src.scene.geometry_payload_factory import make_sphere_payload
from visualizer.src.services.entity_render_service import EntityRenderService
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    SurfaceColorSource,
    TextLabelPayload,
)


class _Renderer:
    def __init__(self) -> None:
        self.objects = {}
        self.visibility = {}
        self.ensure_calls = []
        self.created_ids = []
        self.failed_ensure_ids = set()
        self.failed_remove_ids = set()

    @contextmanager
    def batch_updates(self):
        yield

    def ensure_object(self, obj):
        self.ensure_calls.append(obj)
        if obj.id in self.failed_ensure_ids:
            return False
        if obj.id not in self.objects:
            self.created_ids.append(obj.id)
        self.objects[obj.id] = obj
        self.visibility[obj.id] = obj.visible
        return True

    def remove_object(self, object_id):
        if object_id in self.failed_remove_ids:
            return False
        self.objects.pop(object_id, None)
        self.visibility.pop(object_id, None)
        return True


def _label_object(*, visible: bool = True) -> RenderObject:
    return make_text_label_state(
        "node:tx_0::label",
        "TX1",
        [1.0, 0.0, 0.0],
        position=[1.0, 2.0, 3.5],
        visible=visible,
    ).to_render_object()


def test_entity_render_service_syncs_primary_object_and_label() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    label = _label_object()

    entity = VisualEntity(
        entity_id="node:tx_0",
        category="tx",
        render_object=RenderObject(
            id="node:tx_0::marker",
            payload=make_sphere_payload(radius=0.5, color=[1.0, 0.0, 0.0]),
            material=MaterialPayload(base_color=(1.0, 0.0, 0.0, 1.0)),
            transform=transform,
            visibility=True,
        ),
        label_render_object=label,
    )

    assert service.sync_entity(entity) is True
    assert renderer.objects["node:tx_0::marker"].visible is True
    np.testing.assert_allclose(
        renderer.objects["node:tx_0::marker"].transform.translation,
        [1.0, 2.0, 3.0],
    )
    ensured_label = renderer.objects["node:tx_0::label"]
    assert isinstance(ensured_label.payload, TextLabelPayload)
    assert ensured_label.payload.text == "TX1"
    assert renderer.visibility["node:tx_0::label"] is True
    np.testing.assert_allclose(ensured_label.transform.translation, [1.0, 2.0, 3.5])


def test_entity_render_service_forwards_explicit_vertex_color_source() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    entity = VisualEntity(
        entity_id="target:pedestrian",
        category="target",
        render_object=RenderObject(
            id="target:pedestrian::mesh",
            payload=replace(
                make_sphere_payload(radius=0.5, color=[1.0, 1.0, 1.0]),
                color_source=SurfaceColorSource.VERTEX,
            ),
            material=MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0)),
        ),
    )

    assert service.sync_entity(entity) is True
    ensured = renderer.objects["target:pedestrian::mesh"]
    assert ensured.payload.color_source is SurfaceColorSource.VERTEX


def test_entity_render_service_ensures_hidden_label_state() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    label = make_text_label_state(
        "node:rx_0::label",
        "RX1",
        [0.0, 0.0, 1.0],
        visible=False,
    ).to_render_object()

    entity = VisualEntity(
        entity_id="node:rx_0",
        category="rx",
        render_object=RenderObject(
            id="node:rx_0::marker",
            payload=make_sphere_payload(radius=0.5, color=[0.0, 0.0, 1.0]),
            transform=Transform.identity(),
            visibility=False,
        ),
        label_render_object=label,
    )

    assert service.sync_entity(entity) is True
    assert renderer.visibility["node:rx_0::marker"] is False
    assert renderer.visibility["node:rx_0::label"] is False
    assert renderer.objects["node:rx_0::label"].visible is False


def test_repeated_entity_label_sync_is_stable_and_backend_idempotent() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    label = _label_object()

    entity = VisualEntity(
        entity_id="node:tx_0",
        category="tx",
        render_object=RenderObject(
            id="node:tx_0::marker",
            payload=make_sphere_payload(radius=0.5, color=[1.0, 0.0, 0.0]),
            transform=Transform.identity(),
            visibility=True,
        ),
        label_render_object=label,
    )

    assert service.sync_entity(entity) is True
    assert service.sync_entity(entity) is True

    label_calls = [obj for obj in renderer.ensure_calls if obj.id == label.id]
    assert len(label_calls) == 2
    assert all(isinstance(obj.payload, TextLabelPayload) for obj in label_calls)
    assert all(obj.payload is label.payload for obj in label_calls)
    assert renderer.created_ids.count(label.id) == 1
    assert set(renderer.objects) == {"node:tx_0::marker", "node:tx_0::label"}


def test_entity_sync_reports_label_failure_and_identical_retry_converges() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    label = _label_object()
    entity = VisualEntity(
        entity_id="node:tx_0",
        category="tx",
        render_object=RenderObject(
            id="node:tx_0::marker",
            payload=make_sphere_payload(radius=0.5, color=[1.0, 0.0, 0.0]),
        ),
        label_render_object=label,
    )

    renderer.failed_ensure_ids.add(label.id)
    assert service.sync_entity(entity) is False
    assert "node:tx_0::marker" in renderer.objects
    assert label.id not in renderer.objects

    renderer.failed_ensure_ids.clear()
    assert service.sync_entity(entity) is True
    assert set(renderer.objects) == {"node:tx_0::marker", label.id}


def test_entity_removal_reports_label_failure_and_identical_retry_converges() -> None:
    renderer = _Renderer()
    service = EntityRenderService(SimpleNamespace(renderer=renderer))
    label = _label_object()
    entity = VisualEntity(
        entity_id="node:tx_0",
        category="tx",
        render_object=RenderObject(
            id="node:tx_0::marker",
            payload=make_sphere_payload(radius=0.5, color=[1.0, 0.0, 0.0]),
        ),
        label_render_object=label,
    )
    assert service.sync_entity(entity) is True

    renderer.failed_remove_ids.add(label.id)
    assert service.remove_entity(entity) is False
    assert "node:tx_0::marker" not in renderer.objects
    assert label.id in renderer.objects

    renderer.failed_remove_ids.clear()
    assert service.remove_entity(entity) is True
    assert not renderer.objects
