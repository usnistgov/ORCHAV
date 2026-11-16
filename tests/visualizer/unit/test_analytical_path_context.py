"""Historical paths must never be inferred from an ordinary empty frame."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.pipeline.frame_pipeline import FramePipeline
from visualizer.src.sensing.path_context import AnalyticalPathContext, analytical_source_record
from visualizer.src.services.sensing_service import SensingService
from visualizer.src.state import create_initial_state


def _metadata():
    return {
        "config_resolved": {
            "processing": {"mode": "coherent_cpi"},
            "acquisition": {
                "sample_source": "coherent_one_solve",
                "rt_solve_stride_scene_steps": 4,
                "sample_period_s": 0.01,
            },
        },
        "temporal": {
            "acquisition": {
                "samples": [
                    {
                        "valid": True,
                        "phase_coherent": True,
                        "source": {"scene_frame_id": 0, "solve_id": "solve-0"},
                    }
                    for _ in range(4)
                ]
            },
            "result": {"reference_time_policy": "solve_start", "reference_time_s": 0.0},
            "display_binding": {"scene_frame_id": 0},
        },
    }


@pytest.mark.parametrize("mode", [None, "sequential_stft", "analytic_path_doppler_oracle"])
def test_non_analytical_modes_never_authorize_context(mode):
    metadata = _metadata()
    metadata["config_resolved"]["processing"]["mode"] = mode
    assert analytical_source_record(metadata) is None
    assert analytical_source_record({}) is None


@pytest.mark.parametrize("mutation", ["source", "solve", "phase", "invalid", "binding", "time"])
def test_ambiguous_or_invalid_evidence_never_authorizes_context(mutation):
    metadata = _metadata()
    sample = metadata["temporal"]["acquisition"]["samples"][-1]
    if mutation == "source":
        sample["source"]["scene_frame_id"] = 1
    elif mutation == "solve":
        sample["source"]["solve_id"] = "different-solve"
    elif mutation == "phase":
        sample["phase_coherent"] = False
    elif mutation == "invalid":
        sample["valid"] = False
    elif mutation == "binding":
        metadata["temporal"]["display_binding"]["scene_frame_id"] = 3
    else:
        metadata["temporal"]["result"]["reference_time_s"] = float("nan")
    assert analytical_source_record(metadata) is None


def _runtime(*, record=True, source_available=True):
    packet = FramePipeline._empty_scene_view_model().to_render_packet()
    source_packet = replace(
        packet,
        mpc_points=np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]]),
        mpc_lines=np.array([[0, 1]]),
        mpc_colors=np.ones((1, 3)),
    )
    source_raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.0}}
    calls = []
    objects = {}

    def load(step):
        calls.append(step)
        return (source_raw, source_packet) if source_available else None

    def ensure(obj):
        objects[obj.id] = obj
        return True

    def remove(object_id):
        objects.pop(object_id, None)
        return True

    proof = analytical_source_record(_metadata()) if record else None
    generation = ["generation-1"]
    service = SimpleNamespace(
        analytical_path_source=lambda raw, step: (proof, tuple(generation)),
        _scene_time_s=SensingService._scene_time_s,
    )
    viz = SimpleNamespace(
        app_state=replace(
            create_initial_state(), extension_state={"sensing": {"show_rt_context": True}}
        ),
        pipeline=SimpleNamespace(load_context_frame=load),
    )
    renderer = SimpleNamespace(ensure_object=ensure, remove_object=remove)
    return viz, service, packet, renderer, objects, calls, generation


def test_ordinary_empty_frame_remains_empty_even_with_control_enabled():
    viz, service, packet, renderer, objects, calls, _ = _runtime(record=False)
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    assert context.prepare(viz, service, raw, 1, packet) is packet
    assert context.sync(renderer)
    assert not objects and not calls and not context.status


def test_seek_reuses_fixed_source_geometry_and_clears_at_fresh_solve_or_toggle():
    viz, service, packet, renderer, objects, calls, _ = _runtime()
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.03}}
    original = deepcopy(raw)
    prepared = context.prepare(viz, service, raw, 3, packet)
    assert not prepared.mpc_visibility.enabled
    assert context.sync(renderer)
    line = objects["sensing:historical_rt_paths"]
    assert "Historical" in objects["sensing:historical_rt_label"].payload.text
    assert "30.0 ms" in context.status
    assert raw == original
    raw["_source"]["timestamp"] = 0.01
    context.prepare(viz, service, raw, 1, packet)
    context.sync(renderer)
    assert objects["sensing:historical_rt_paths"] is line
    assert calls == [0]
    np.testing.assert_array_equal(line.payload.points, [[0, 0, 1], [5, 0, 1]])
    raw["_source"]["timestamp"] = 0.0
    assert context.prepare(viz, service, raw, 0, packet) is packet
    context.sync(renderer)
    assert not objects
    raw["_source"]["timestamp"] = 0.02
    context.prepare(viz, service, raw, 2, packet)
    context.sync(renderer)
    viz.app_state = replace(viz.app_state, extension_state={"sensing": {"show_rt_context": False}})
    assert context.prepare(viz, service, raw, 2, packet) is packet
    context.sync(renderer)
    assert not objects


def test_mpc_master_alone_shows_labelled_verified_context_by_default():
    viz, service, packet, renderer, objects, calls, _ = _runtime()
    viz.app_state = create_initial_state()
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    context.prepare(viz, service, raw, 1, packet)
    context.sync(renderer)
    assert "sensing:historical_rt_paths" in objects
    assert "Historical" in objects["sensing:historical_rt_label"].payload.text
    assert calls == [0]
    viz.app_state = replace(
        viz.app_state, mpc_visibility=replace(viz.app_state.mpc_visibility, enabled=False)
    )
    context.prepare(viz, service, raw, 1, packet)
    context.sync(renderer)
    assert not objects


def test_fresh_rt_frame_clears_historical_context_without_indexing_metadata():
    viz, service, packet, renderer, objects, _, _ = _runtime()
    context = AnalyticalPathContext()
    context.prepare(
        viz, service, {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}, 1, packet
    )
    context.sync(renderer)
    assert objects
    service.analytical_path_source = lambda *args: pytest.fail("Indexed a fresh RT frame")
    raw = {"_source": {"source_rt_frame_idx": 4, "timestamp": 0.04}}
    assert context.prepare(viz, service, raw, 4, packet) is packet
    context.sync(renderer)
    assert not objects and not context.status


@pytest.mark.parametrize(
    "visibility", ({"enabled": False}, {"paths": False, "bounce_points": False})
)
def test_hidden_mpc_context_clears_without_indexing_metadata(visibility):
    viz, service, packet, renderer, objects, _, _ = _runtime()
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    context.prepare(viz, service, raw, 1, packet)
    context.sync(renderer)
    assert objects
    viz.app_state = replace(
        viz.app_state, mpc_visibility=replace(viz.app_state.mpc_visibility, **visibility)
    )
    service.analytical_path_source = lambda *args: pytest.fail("Indexed hidden paths")
    assert context.prepare(viz, service, raw, 1, packet) is packet
    context.sync(renderer)
    assert not objects


def test_bounce_only_historical_frame_still_validates_and_suppresses_stale_geometry():
    viz, service, packet, renderer, objects, calls, _ = _runtime()
    viz.app_state = replace(
        viz.app_state, mpc_visibility=replace(viz.app_state.mpc_visibility, paths=False)
    )
    lookups = []
    original = service.analytical_path_source
    service.analytical_path_source = lambda *args: (lookups.append(True), original(*args))[1]
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    prepared = context.prepare(viz, service, raw, 1, packet)
    assert lookups == [True]
    assert not prepared.mpc_visibility.enabled
    context.sync(renderer)
    assert not objects and not calls


@pytest.mark.parametrize("timestamp", [-0.01, 0.04, float("nan")])
def test_future_or_expired_source_is_not_presented(timestamp):
    viz, service, packet, renderer, objects, calls, _ = _runtime()
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": timestamp}}
    assert context.prepare(viz, service, raw, 1, packet) is packet
    context.sync(renderer)
    assert not objects and not calls


def test_missing_source_never_reuses_previous_geometry():
    viz, service, packet, renderer, objects, calls, _ = _runtime(source_available=False)
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    context.prepare(viz, service, raw, 1, packet)
    context.sync(renderer)
    assert not objects and calls == [0]
    assert "unavailable" in context.status


def test_provider_generation_and_material_filter_invalidate_snapshot():
    viz, service, packet, renderer, objects, calls, generation = _runtime()
    context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.01}}
    for _ in range(2):
        context.prepare(viz, service, raw, 1, packet)
    assert calls == [0]
    generation[0] = "generation-2"
    context.prepare(viz, service, raw, 1, packet)
    viz.mpc_allowed_materials = set()
    context.prepare(viz, service, raw, 1, packet)
    assert calls == [0, 0, 0]
