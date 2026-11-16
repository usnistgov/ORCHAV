"""Display-binding gates for live 3D sensing overlays."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from shared.extensions.sensing import UnsupportedSensingSchemaError
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from visualizer.src.sensing.runtime import _sync_frame
from visualizer.src.services.sensing_service import SensingService


class _Cache:
    def __init__(self, frames: dict[int, dict], order: list[int]) -> None:
        self.frames = frames
        self.frame_cache = SimpleNamespace(keys=lambda: list(order))

    def get_frame(self, step: int) -> dict | None:
        return self.frames.get(step)


def _scene_frame(scene_id: int, *, timestamp_s: float | None = None) -> dict:
    frame = {
        "rx_positions": np.array([[float(scene_id), 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.zeros((1, 3), dtype=np.float32),
        "sensing": {},
    }
    if timestamp_s is not None:
        frame["_source"] = {"timestamp": timestamp_s}
    return frame


def _detection_product(*, bound_scene_id: int, storage_pose_x: float) -> dict:
    return {
        "rx_positions": np.array([[storage_pose_x, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.zeros((1, 3), dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]},
            scene_frame_id=bound_scene_id,
            sample_time_s=float(bound_scene_id),
        ),
    }


def _runtime(frames: dict[int, dict], order: list[int], *, pin_freshest: bool = False):
    renderer = SimpleNamespace(renderer_id="test")
    viz = SimpleNamespace(
        renderer=renderer,
        cache_service=_Cache(frames, order),
        frame_source=None,
        frame_loader=None,
        app_state=SimpleNamespace(
            extension_state={
                "sensing": {
                    "show_detections": True,
                    "scene_mode": "detections",
                    "history_frames": 4,
                    "pin_freshest": pin_freshest,
                }
            }
        ),
    )
    service = SensingService(viz)
    synced: list[list] = []
    cleared: list[bool] = []
    service.sync_scene_overlays = lambda _renderer, layers: synced.append(layers) or True
    service.clear_scene_overlays = lambda _renderer: cleared.append(True) or True
    return viz, service, synced, cleared


@pytest.mark.parametrize("order", [[1, 9, 7], [7, 9, 1]])
def test_runtime_uses_exact_binding_and_bound_scene_pose_regardless_of_cache_order(order) -> None:
    frames = {
        1: _detection_product(bound_scene_id=1, storage_pose_x=101.0),
        7: _detection_product(bound_scene_id=7, storage_pose_x=107.0),
        # A later storage row explicitly supplies the result for scene 5.
        9: _detection_product(bound_scene_id=5, storage_pose_x=109.0),
    }
    viz, service, synced, cleared = _runtime(frames, order)

    assert _sync_frame(viz, service, _scene_frame(5), 5) is True

    assert not cleared
    detection_layer = next(
        layer for layer in synced[-1] if layer.name == SensingService.DETECTION_GEOMETRY_NAME
    )
    # Local x=1 is transformed with scene 5's radar x=5, never storage row 9's x=109.
    np.testing.assert_allclose(detection_layer.payload.points, [[6.0, 0.0, 0.0]])


def test_runtime_reverse_and_seek_never_retain_mismatched_or_future_product() -> None:
    frames = {
        9: _detection_product(bound_scene_id=5, storage_pose_x=109.0),
        11: _detection_product(bound_scene_id=7, storage_pose_x=111.0),
    }
    viz, service, synced, cleared = _runtime(frames, [11, 9])

    assert _sync_frame(viz, service, _scene_frame(7), 7) is True
    assert _sync_frame(viz, service, _scene_frame(5), 5) is True
    # Seeking to an unbound scene clears instead of retaining scene 5 or using scene 7.
    assert _sync_frame(viz, service, _scene_frame(6), 6) is True

    assert len(synced) == 2
    assert len(cleared) == 1


def test_runtime_pin_uses_bound_pose_and_current_scene_age() -> None:
    frames = {
        10: _scene_frame(10),
        30: _detection_product(bound_scene_id=10, storage_pose_x=130.0),
    }
    viz, service, synced, cleared = _runtime(frames, [30, 10], pin_freshest=True)
    captured_provenance: list[dict] = []
    original_build = service._build_scene_overlays_from_validated

    def capture_build(frame, **kwargs):
        captured_provenance.append(dict(frame["_sensing_display_provenance"]))
        return original_build(frame, **kwargs)

    service._build_scene_overlays_from_validated = capture_build

    assert _sync_frame(viz, service, _scene_frame(20, timestamp_s=35.0), 20) is True

    assert not cleared
    detection_layer = next(
        layer for layer in synced[-1] if layer.name == SensingService.DETECTION_GEOMETRY_NAME
    )
    np.testing.assert_allclose(detection_layer.payload.points, [[11.0, 0.0, 0.0]])
    assert captured_provenance[-1]["current_scene_time_s"] == pytest.approx(35.0)
    assert captured_provenance[-1]["stale_age_s"] == pytest.approx(25.0)


@pytest.mark.parametrize("mode", ["tracks", "evidence", "full"])
@pytest.mark.parametrize("full_history", [False, True])
def test_history_survives_intermediate_scenes_and_rewinds_without_future_states(
    mode, full_history
) -> None:
    frames = {i: _scene_frame(i, timestamp_s=float(i)) for i in range(7)}
    for i in (1, 3, 5):
        product = _detection_product(bound_scene_id=i, storage_pose_x=100.0 + i)
        product["sensing"]["detections"][0].update(track_id=7, track_confirmed=True)
        frames[100 + i] = product
    viz, service, synced, cleared = _runtime(frames, [105, 103, 101])
    viz.app_state.extension_state["sensing"].update(
        scene_mode=mode, full_history=full_history, history_frames=2
    )
    for step, bound_ids in ((4, [1, 3]), (6, [1, 3, 5]), (2, [1])):
        assert _sync_frame(viz, service, frames[step], step)
        assert not cleared
        layers = {layer.name: layer for layer in synced[-1]}
        kept = bound_ids if full_history else bound_ids[-2:]
        points = [[i + 1.0, 0.0, 0.0] for i in kept]
        if mode in {"tracks", "full"}:
            np.testing.assert_allclose(layers["sensing_track_point_7"].payload.points, points[-1:])
            if len(points) > 1:
                np.testing.assert_allclose(layers["sensing_track_trail_7"].payload.points, points)
            else:
                assert "sensing_track_trail_7" not in layers
        if mode in {"evidence", "full"}:
            np.testing.assert_allclose(
                layers[service.EVIDENCE_GEOMETRY_NAME].payload.points, points
            )
        annotations, _ = service.resolve_scene_overlay_layers(
            frames[step],
            step=step,
            mode=mode,
            history_frames=2,
            use_full_history=full_history,
            show_velocity_arrows=False,
            pin_freshest=False,
        )
        assert annotations["stale"]
        assert annotations["result_time_s"] == bound_ids[-1]
        assert annotations["scene_time_s"] == step

    # Neither a rewind before the first result nor an unknown scene clock may
    # reuse the last visited result or borrow a future estimate.
    assert _sync_frame(viz, service, frames[0], 0)
    assert _sync_frame(viz, service, _scene_frame(4), 4)
    assert len(synced) == 3
    assert len(cleared) == 2


def test_runtime_binding_scan_propagates_pre_v3_regeneration_error() -> None:
    frames = {
        9: {
            "sensing": {
                "schema_version": 2,
                "detections": [{"x_m": 1.0, "y_m": 0.0}],
            }
        }
    }
    viz, service, _synced, _cleared = _runtime(frames, [9])

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        _sync_frame(viz, service, _scene_frame(5), 5)


def test_retirement_clears_presentation_when_next_scene_has_no_sensing(monkeypatch):
    from visualizer.src.sensing import runtime

    class Renderer:
        pass

    renderer = Renderer()
    calls = []
    adapter = SimpleNamespace(retire=lambda: calls.append("annotations_picking_hud"))
    monkeypatch.setitem(runtime._ANNOTATION_ADAPTERS, renderer, adapter)
    service = SimpleNamespace(clear_scene_overlays=lambda value: calls.append(value))
    viz = SimpleNamespace(renderer=renderer, sensing_selection=object())
    runtime._retire_service(viz, service)
    assert calls == [renderer, "annotations_picking_hud"]
    assert renderer not in runtime._ANNOTATION_ADAPTERS
    assert viz.sensing_selection is None


def test_retirement_clears_selection_before_renderer_initialization():
    from visualizer.src.sensing.runtime import _retire_service

    viz = SimpleNamespace(sensing_selection=object())
    _retire_service(viz, object())
    assert viz.sensing_selection is None
