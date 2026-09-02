"""Target outline synchronization tests for the renderer object contract."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from visualizer.src.model import RenderObjectState
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.target_service import TargetService
from visualizer.src.types.render_payloads import MeshPayload


class _Renderer:
    capabilities = RendererCapabilities()

    def __init__(self) -> None:
        self.objects = {}
        self.visibility = {}

    @contextmanager
    def batch_updates(self):
        yield

    def ensure_object(self, obj) -> bool:
        self.objects[obj.id] = obj
        self.visibility[obj.id] = obj.visible
        return True

    def remove_object(self, object_id: str) -> bool:
        self.objects.pop(object_id, None)
        self.visibility.pop(object_id, None)
        return True

    def set_visible(self, object_id: str, visible: bool) -> bool:
        self.visibility[object_id] = bool(visible)
        return True

    def has_named_geometry(self, name: str) -> bool:
        return name in self.objects

    def is_named_visible(self, name: str):
        return self.visibility.get(name)

    def set_transform(self, _object_id: str, _transform) -> bool:
        return True

    def request_redraw(self) -> None:
        pass


def _service():
    renderer = _Renderer()
    mesh = RenderObjectState(
        id="target:drone_1::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    entry = {
        "mesh": mesh,
        "visible": True,
        "target_name": "drone_1",
        "_target_position": [1.0, 2.0, 3.0],
        "_mesh_center": [0.0, 0.0, 0.0],
    }
    viz = SimpleNamespace(
        renderer=renderer,
        target_entries=[entry],
        target_outlines_enabled=False,
        outline_color=[0.05, 0.05, 0.05],
        vis=object(),
        vis_initialized=True,
    )
    return TargetService(viz), viz, renderer, entry


def test_target_outline_enable_disable_uses_render_object_state() -> None:
    service, _viz, renderer, entry = _service()

    service.set_target_edge_visibility(True)

    outline = entry["outline_geometry"]
    assert isinstance(outline, RenderObjectState)
    assert outline.is_edge is True
    assert renderer.visibility[outline.id] is True

    service.set_target_edge_visibility(False)

    assert outline.visible is False
    assert renderer.visibility[outline.id] is False


def test_invalidate_target_outline_preserves_backend_owned_object() -> None:
    service, _viz, renderer, entry = _service()
    service.set_target_edge_visibility(True)
    outline = entry["outline_geometry"]

    service._invalidate_target_outline(entry)

    assert outline.id in renderer.objects
    assert entry["outline_geometry"] is outline
    assert entry["_outline_payload_dirty"] is True
    assert entry["outline_visible"] is False

    service.set_target_edge_visibility(True)

    replacement = entry["outline_geometry"]
    assert replacement is outline
    assert replacement.id == outline.id
    assert renderer.objects[outline.id].payload is replacement.payload
    assert entry["_outline_payload_dirty"] is False


def test_invalidated_outline_is_hidden_and_retried_without_losing_state() -> None:
    service, _viz, renderer, entry = _service()
    service.set_target_edge_visibility(True)
    outline = entry["outline_geometry"]
    service._invalidate_target_outline(entry)
    entry["_frame_visible"] = False

    original_ensure = renderer.ensure_object
    failed_once = False

    def _fail_first_hidden_outline(obj):
        nonlocal failed_once
        if obj.id == outline.id and not obj.visible and not failed_once:
            failed_once = True
            return False
        return original_ensure(obj)

    renderer.ensure_object = _fail_first_hidden_outline

    assert not service.sync_target_entry_snapshot(entry)
    assert entry["_renderer_sync_pending"] is True
    assert renderer.visibility[outline.id] is True
    assert entry["outline_geometry"] is outline

    assert service.sync_target_entry_snapshot(entry)
    assert entry["_renderer_sync_pending"] is False
    assert renderer.visibility[outline.id] is False
    assert entry["outline_geometry"] is outline
