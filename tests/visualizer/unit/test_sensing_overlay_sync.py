"""Failure-injection tests for sensing-overlay renderer ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visualizer.src.model import RenderObject
from visualizer.src.services.sensing_overlay_sync import SensingOverlaySync
from visualizer.src.services.sensing_service import SensingSceneLayer, SensingService
from visualizer.src.types.render_payloads import MaterialPayload, PointCloudPayload


def _snapshot(object_id: str, *, x: float = 0.0) -> RenderObject:
    return RenderObject(
        id=object_id,
        payload=PointCloudPayload(
            points=np.asarray([[x, 0.0, 0.0]], dtype=np.float32),
        ),
        material=MaterialPayload(point_size=8.0),
        metadata={"type": "sensing_overlay"},
    )


@dataclass
class _FailureRenderer:
    objects: dict[str, RenderObject] = field(default_factory=dict)
    ensure_calls: list[str] = field(default_factory=list)
    remove_calls: list[str] = field(default_factory=list)
    ensure_failures: dict[str, int] = field(default_factory=dict)
    remove_failures: dict[str, int] = field(default_factory=dict)
    create_before_ensure_failure: bool = False

    def ensure_object(self, snapshot: RenderObject) -> bool:
        self.ensure_calls.append(snapshot.id)
        remaining = self.ensure_failures.get(snapshot.id, 0)
        if remaining:
            self.ensure_failures[snapshot.id] = remaining - 1
            if self.create_before_ensure_failure:
                self.objects[snapshot.id] = snapshot
            return False
        self.objects[snapshot.id] = snapshot
        return True

    def remove_object(self, object_id: str) -> bool:
        self.remove_calls.append(object_id)
        remaining = self.remove_failures.get(object_id, 0)
        if remaining:
            self.remove_failures[object_id] = remaining - 1
            return False
        self.objects.pop(object_id, None)
        return True


def test_identical_snapshot_is_handed_to_renderer_for_backend_idempotence() -> None:
    renderer = _FailureRenderer()
    sync = SensingOverlaySync()
    snapshot = _snapshot("sensing_detections")

    assert sync.sync(renderer, [snapshot]) is True
    assert sync.sync(renderer, [snapshot]) is True

    assert renderer.ensure_calls == ["sensing_detections", "sensing_detections"]
    assert set(renderer.objects) == {"sensing_detections"}


def test_failed_ensure_retains_owner_and_identical_retry_converges() -> None:
    renderer = _FailureRenderer(ensure_failures={"sensing_detections": 1})
    sync = SensingOverlaySync()
    snapshot = _snapshot("sensing_detections")

    assert sync.sync(renderer, [snapshot]) is False
    assert sync.owned_ids == {"sensing_detections"}

    assert sync.sync(renderer, [snapshot]) is True
    assert renderer.ensure_calls == ["sensing_detections", "sensing_detections"]
    assert sync.owned_ids == {"sensing_detections"}
    assert set(renderer.objects) == {"sensing_detections"}


def test_failed_ensure_without_native_object_is_still_removable() -> None:
    renderer = _FailureRenderer(ensure_failures={"sensing_detections": 1})
    sync = SensingOverlaySync()

    assert sync.sync(renderer, [_snapshot("sensing_detections")]) is False
    assert renderer.objects == {}

    assert sync.clear(renderer) is True
    assert renderer.remove_calls == ["sensing_detections"]
    assert sync.owned_ids == set()


def test_partially_created_failed_ensure_remains_removable() -> None:
    renderer = _FailureRenderer(
        ensure_failures={"sensing_detections": 1},
        create_before_ensure_failure=True,
    )
    sync = SensingOverlaySync()

    assert sync.sync(renderer, [_snapshot("sensing_detections")]) is False
    assert set(renderer.objects) == {"sensing_detections"}

    assert sync.clear(renderer) is True
    assert sync.owned_ids == set()
    assert renderer.objects == {}


def test_failed_stale_removal_blocks_replacement_then_retry_converges() -> None:
    renderer = _FailureRenderer()
    sync = SensingOverlaySync()
    old = _snapshot("sensing_detections")
    replacement = _snapshot("sensing_evidence", x=1.0)

    assert sync.sync(renderer, [old]) is True
    renderer.remove_failures[old.id] = 1

    assert sync.sync(renderer, [replacement]) is False
    assert renderer.ensure_calls == [old.id]
    assert set(renderer.objects) == {old.id}
    assert sync.owned_ids == {old.id}

    assert sync.sync(renderer, [replacement]) is True
    assert renderer.remove_calls == [old.id, old.id]
    assert renderer.ensure_calls == [old.id, replacement.id]
    assert set(renderer.objects) == {replacement.id}
    assert sync.owned_ids == {replacement.id}


def test_sensing_service_submits_complete_render_object_snapshots() -> None:
    renderer = _FailureRenderer()
    service = SensingService(object())
    payload = PointCloudPayload(
        points=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    material = MaterialPayload(base_color=(0.2, 0.4, 0.6, 1.0), point_size=12.0)
    layer = SensingSceneLayer(
        name="sensing_detections",
        payload=payload,
        material=material,
    )

    assert service.sync_scene_overlays(renderer, [layer]) is True

    snapshot = renderer.objects[layer.name]
    assert snapshot.payload is payload
    assert snapshot.material_payload is material
    assert snapshot.visible is True
    np.testing.assert_array_equal(snapshot.transform_matrix, np.eye(4))
    assert snapshot.metadata == {"type": "sensing_overlay"}
