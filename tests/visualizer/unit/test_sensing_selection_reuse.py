"""Validated replay reuse must preserve data, provenance, and ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.frames.provider_base import ProviderInfo
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_services_sensing import (
    _CountingMultiFrameProvider,
    _detector_stage_sensing,
)
from visualizer.src.services import sensing_service
from visualizer.src.services.sensing_service import SensingReplayTelemetry, SensingService


class _Provider(_CountingMultiFrameProvider):
    generation = "generation-a"
    frame_set = "frame-set-a"

    @property
    def info(self):
        return ProviderInfo(
            name="replay-reuse-test",
            source="unit-test",
            generation_id=self.generation,
            frame_set_id=self.frame_set,
        )


def _scene(scene_id=5, *, timestamp=None):
    return {
        "timestamp_s": float(scene_id if timestamp is None else timestamp),
        "sensing": {},
        "tx_positions": np.array([[1.0, 2.0, 3.0]]),
        "tx_orientations": np.array([[0.1, 0.2, 0.3]]),
        "rx_positions": np.array([[4.0, 5.0, 6.0]]),
        "rx_orientations": np.array([[0.4, 0.5, 0.6]]),
    }


def _service():
    provider = _Provider(
        {10: replace(build_standard_mpc_frame(frame_idx=10), sensing=_detector_stage_sensing())}
    )
    service = SensingService(
        SimpleNamespace(frame_source=SimpleNamespace(provider=provider)),
        replay_telemetry=SensingReplayTelemetry(),
    )
    return service, provider


def _select(service, scene=None, *, scene_id=5, **kwargs):
    result = service.resolve_display_sensing_frame(
        scene or _scene(scene_id),
        scene_frame_id=scene_id,
        storage_frame_id=scene_id,
        candidate_storage_ids=(10,),
        product_names=kwargs.pop("product_names", ("range_doppler_map",)),
        **kwargs,
    )
    assert result is not None
    return result


def test_repeat_and_reverse_scrub_reuse_exact_validated_event(monkeypatch):
    service, provider = _service()
    first = _select(service)
    assert first.replay_event is not None
    original = first.replay_event.to_manifest()
    reads = len(provider.requests)
    builds = service.replay_telemetry_snapshot().replay_builds
    repeated = _select(service)
    assert repeated.replay_event is first.replay_event
    assert repeated.replay_event.to_manifest() == original
    assert service.replay_telemetry_snapshot().replay_builds == builds
    assert len(provider.requests) == reads

    bound_scene = _scene()
    pinned = _select(service, scene_id=8, pin_freshest=True, load_scene_frame=lambda _: bound_scene)
    assert pinned.replay_event is not None
    assert pinned.replay_event is not first.replay_event
    assert pinned.replay_event.manifest_id == first.replay_event.manifest_id
    assert pinned.replay_event.scene_binding.stale_age_s == 3.0
    assert pinned.replay_event.scene_binding.current_scene_frame_id == 8
    assert pinned.replay_event.preflight.disposition == "ready_with_warnings"
    assert _select(service).replay_event is first.replay_event
    assert (
        _select(
            service, scene_id=8, pin_freshest=True, load_scene_frame=lambda _: bound_scene
        ).replay_event
        is pinned.replay_event
    )

    # Both scientific hashes on the service hot path are reused, including
    # the stage lookup. Manifest serialization itself remains available.
    def forbidden_hash(_event):
        raise AssertionError("warm selection/stage lookup must not rehash the event")

    monkeypatch.setattr(sensing_service.ReplayEvent, "manifest_id", property(forbidden_hash))
    warm = _select(service)
    service.load_detection_stage_bundle(warm)
    service.load_detection_stage_bundle(warm)


def test_mutable_returned_metadata_and_array_headers_cannot_poison_reuse():
    service, _ = _service()
    first = _select(service)
    expected_manifest = first.replay_event.to_manifest()
    sensing = first.product_frame["sensing"]
    expected_array = sensing["range_doppler_by_pair"].copy()
    sensing["config_resolved"]["processing_recipe"]["untrusted_edit"] = True
    sensing["temporal"]["acquisition"]["samples"][0]["source"]["sample_offset"] = 999
    sensing["products"]["range_doppler_by_pair"]["axes"][0]["name"] = "wrong"
    sensing["detections"][0]["range_m"] = -999.0
    dense = sensing["range_doppler_by_pair"]
    with pytest.raises(ValueError):
        dense.setflags(write=True)
    with pytest.raises(ValueError):
        dense.flat[0] = 999
    dense.shape = (6,)
    dense.base.shape = (2, 3)
    first.display_frame["sensing"]["range_axis_m"] = np.array([100.0])

    warm = _select(service)
    assert warm.replay_event is first.replay_event
    assert warm.replay_event.to_manifest() == expected_manifest
    np.testing.assert_array_equal(
        warm.product_frame["sensing"]["range_doppler_by_pair"], expected_array
    )
    assert warm.product_frame["sensing"]["detections"][0]["range_m"] == 1.0
    assert (
        "untrusted_edit"
        not in warm.product_frame["sensing"]["config_resolved"]["processing_recipe"]
    )
    assert warm.product_frame is not first.product_frame
    assert warm.product_frame["sensing"] is not sensing


@pytest.mark.parametrize("field", ["generation", "frame_set"])
def test_provider_revision_invalidates_science_and_configuration(field):
    service, provider = _service()
    first = _select(service)
    original_manifest = first.replay_event.to_manifest()
    modified = deepcopy(provider.frames[10].sensing)
    modified["config_resolved"]["carrier_freq_hz"] *= 2
    modified["config"] = deepcopy(modified["config_resolved"])
    provider.frames[10] = replace(provider.frames[10], sensing=modified)
    setattr(provider, field, "revision-b")

    changed = _select(service)
    assert changed.replay_event is not None
    assert changed.replay_event is not first.replay_event
    assert changed.replay_event.to_manifest() != original_manifest
    assert (
        changed.product_frame["sensing"]["config_resolved"]["carrier_freq_hz"]
        == modified["config_resolved"]["carrier_freq_hz"]
    )
    assert service.replay_cache_state().validated_event_entries == 1
    assert service.replay_telemetry_snapshot().replay_builds == 2


def test_replacing_provider_invalidates_equal_storage_identity():
    service, provider = _service()
    first = _select(service)
    replacement = _Provider(deepcopy(provider.frames))
    service.visualizer.frame_source.provider = replacement
    second = _select(service)
    assert second.replay_event is not first.replay_event
    assert second.replay_event.to_manifest() == first.replay_event.to_manifest()
    assert replacement.requests


@pytest.mark.parametrize(
    "name", ["tx_positions", "tx_orientations", "rx_positions", "rx_orientations"]
)
def test_mutating_bound_pose_rebuilds_canonical_event(name):
    service, _ = _service()
    bound = _scene()
    first = _select(service, scene_id=6, pin_freshest=True, load_scene_frame=lambda _: bound)
    bound[name][0, 0] += 0.125
    changed = _select(service, scene_id=6, pin_freshest=True, load_scene_frame=lambda _: bound)
    assert changed.replay_event is not first.replay_event
    assert changed.replay_event.manifest_id != first.replay_event.manifest_id
    assert changed.replay_event.scene_binding == first.replay_event.scene_binding


def test_timestamp_pin_and_frame_index_are_part_of_exact_context():
    service, _ = _service()
    first = _select(service)
    changed_time = _select(service, _scene(timestamp=5.5))
    assert changed_time.replay_event is not first.replay_event
    assert changed_time.replay_event.scene_binding.current_scene_result_offset_s == 0.5
    pinned = _select(service, pin_freshest=True)
    assert pinned.replay_event is not first.replay_event
    assert pinned.replay_event.scene_binding.pinned is True
    invalid_frame = _scene()
    invalid_frame["frame_index"] = 777
    rejected = _select(service, invalid_frame)
    assert rejected.replay_event is None
    assert "storage_frame_id" in rejected.replay_reason
    assert _select(service).replay_event is first.replay_event


def test_product_requests_and_load_all_are_separate_validated_snapshots():
    service, _ = _service()
    first = _select(service)
    detections = _select(service, product_names=("detections",))
    assert detections.replay_event is not first.replay_event
    assert "range_doppler_by_pair" not in detections.product_frame["sensing"]
    assert _select(service).replay_event is first.replay_event
    complete = _select(service, load_all_sensing_products=True)
    assert complete.replay_event is not first.replay_event
    assert "range_doppler_cfar_mask_by_pair" in complete.product_frame["sensing"]
    assert service.replay_cache_state().selected_product_entries == 2


def test_failed_replay_build_is_retried_without_poisoning_cache(monkeypatch):
    service, _ = _service()
    builder = sensing_service.build_replay_event
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("transient validation failure")
        return builder(*args, **kwargs)

    monkeypatch.setattr(sensing_service, "build_replay_event", fail_once)
    failure = _select(service)
    assert failure.replay_event is None
    assert "transient validation failure" in failure.replay_reason
    assert service.replay_cache_state().validated_event_entries == 0
    success = _select(service)
    assert success.replay_event is not None
    assert calls == 2
    assert _select(service).replay_event is success.replay_event
    assert calls == 2


def test_inline_mutable_candidate_is_revalidated_even_with_a_cached_event():
    service, provider = _service()
    cached = _select(service)
    scene = _scene()
    scene["sensing"] = deepcopy(provider.frames[10].sensing)
    first = _select(service, scene)
    assert first.replay_event is not cached.replay_event
    scene["sensing"]["config_resolved"]["processing_recipe"]["tampered"] = True
    with pytest.raises(ValueError):
        _select(service, scene)


def test_custom_loader_never_reuses_a_mutable_projection():
    service, provider = _service()
    frame = {"sensing": deepcopy(provider.frames[10].sensing)}
    first = _select(service, load_candidate=lambda _: frame)
    second = _select(service, load_candidate=lambda _: frame)
    assert second.replay_event is not first.replay_event
    assert service.replay_cache_state().validated_event_entries == 0
    frame["sensing"]["config_resolved"]["processing_recipe"]["tampered"] = True
    with pytest.raises(ValueError):
        _select(service, load_candidate=lambda _: frame)


def test_validated_context_lru_is_bounded_and_evicted_context_is_rebuilt():
    service, _ = _service()
    bound = _scene()
    first = _select(service)
    capacity = service.replay_cache_state().validated_event_capacity
    for scene_id in range(6, 6 + capacity + 2):
        _select(service, scene_id=scene_id, pin_freshest=True, load_scene_frame=lambda _: bound)
        assert service.replay_cache_state().validated_event_entries <= capacity
    assert service.replay_cache_state().validated_event_entries == capacity
    fresh = _select(service)
    assert fresh.replay_event is not first.replay_event
    assert fresh.replay_event.to_manifest() == first.replay_event.to_manifest()


def test_mutating_returned_selection_cannot_poison_detector_stage_cache():
    service, _ = _service()
    selected = _select(service)
    selected.product_frame["sensing"]["products"].clear()
    selected.product_frame["sensing"]["temporal"].clear()
    bundle = service.load_detection_stage_bundle(selected)
    assert bundle.disposition is sensing_service.InspectionDisposition.AVAILABLE
    pristine = _select(service)
    assert service.load_detection_stage_bundle(pristine) is bundle


def test_evicted_product_snapshot_still_loads_stages_from_immutable_event():
    service, _ = _service()
    selected = _select(service)
    _select(service, product_names=("detections",))
    _select(service, load_all_sensing_products=True)
    selected.product_frame["sensing"].clear()
    bundle = service.load_detection_stage_bundle(selected)
    assert bundle.disposition is sensing_service.InspectionDisposition.AVAILABLE


def test_stage_projection_failure_can_recover(monkeypatch):
    service, _ = _service()
    selected = _select(service)
    loader = service._load_detection_stage_projection
    monkeypatch.setattr(service, "_load_detection_stage_projection", lambda *args: None)
    failed = service.load_detection_stage_bundle(selected)
    assert failed.disposition is sensing_service.InspectionDisposition.UNAVAILABLE
    assert service.replay_cache_state().detection_stage_entries == 0
    monkeypatch.setattr(service, "_load_detection_stage_projection", loader)
    recovered = service.load_detection_stage_bundle(selected)
    assert recovered.disposition is sensing_service.InspectionDisposition.AVAILABLE


def test_evicted_snapshot_without_stages_preserves_unavailable_disposition():
    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload

    service, provider = _service()
    provider.frames[10] = replace(
        provider.frames[10],
        sensing=build_visualizer_sensing_payload(
            {"detections": [{"range_m": 1.0}]}, scene_frame_id=5, sample_time_s=5.0
        ),
    )
    selected = _select(service, product_names=("detections",))
    expected = service.load_detection_stage_bundle(selected)
    assert expected.disposition is sensing_service.InspectionDisposition.UNAVAILABLE
    _select(service, product_names=("detections", "not-present"))
    _select(service, product_names=("detections",), load_all_sensing_products=True)
    selected.product_frame["sensing"].clear()
    evicted = service.load_detection_stage_bundle(selected)
    assert evicted.disposition == expected.disposition
    assert evicted.reason == expected.reason


@pytest.mark.parametrize(
    "change",
    [
        {"storage_frame_id": 999},
        {"bound_scene_frame_id": 999},
        {"current_scene_frame_id": 999},
        {"pinned": True},
        {"reference_time_s": 999.0},
        {"current_scene_time_s": 999.0},
        {"scene_result_time_offset_s": 999.0},
        {"stale_age_s": 999.0},
    ],
)
def test_replaced_selection_cannot_reuse_another_binding_stage_cache(change):
    service, _ = _service()
    selected = _select(service)
    service.load_detection_stage_bundle(selected)
    altered = replace(selected, **change)
    rejected = service.load_detection_stage_bundle(altered)
    assert rejected.disposition is sensing_service.InspectionDisposition.INVALID
    assert rejected.reason == "selection_binding_does_not_match_replay_event"


def test_forward_playback_rebinds_same_cpi_without_validation_or_hashing(monkeypatch):
    service, _ = _service()
    bound = _scene()
    first = _select(service)
    builder = sensing_service.build_replay_event
    for scene_id in range(6, 40):
        scene = _scene(scene_id)
        selected = _select(
            service, scene, scene_id=scene_id, pin_freshest=True, load_scene_frame=lambda _: bound
        )
        expected = builder(
            selected.display_frame,
            sensing_service.ReplaySceneContext(
                storage_frame_id=10,
                bound_scene_frame_id=5,
                current_scene_frame_id=scene_id,
                bound_scene_time_s=5.0,
                current_scene_time_s=float(scene_id),
                pinned=True,
            ),
            require_product_data=False,
        )
        assert selected.replay_event.to_json() == expected.to_json()
        assert selected.replay_event.pulses is first.replay_event.pulses
        assert selected.replay_event.recipe is first.replay_event.recipe
    telemetry = service.replay_telemetry_snapshot()
    assert telemetry.replay_builds == 1
    assert telemetry.replay_rebindings == 34
    assert service.replay_cache_state().validated_event_entries == 8

    def forbidden(*args, **kwargs):
        raise AssertionError("forward playback must reuse validated science")

    monkeypatch.setattr(sensing_service, "build_replay_event", forbidden)
    monkeypatch.setattr(sensing_service.ReplayEvent, "manifest_id", property(forbidden))
    final = _select(service, scene_id=40, pin_freshest=True, load_scene_frame=lambda _: bound)
    assert final.replay_event.scene_binding.stale_age_s == 35.0
    rewind = _select(service)
    assert rewind.replay_event.scene_binding.fresh
    assert rewind.replay_event.preflight == first.replay_event.preflight


def test_signed_zero_pose_change_rebuilds_scientific_identity():
    service, _ = _service()
    scene = _scene()
    scene["tx_positions"][0, 0] = -0.0
    first = _select(service, scene)
    scene["tx_positions"][0, 0] = 0.0
    second = _select(service, scene)
    assert second.replay_event is not first.replay_event
    assert second.replay_event.manifest_id != first.replay_event.manifest_id
    assert service.replay_telemetry_snapshot().replay_builds == 2


def test_signed_zero_current_time_rebinds_exact_view_manifest():
    service, provider = _service()
    provider.frames[10] = replace(
        provider.frames[10], sensing=_detector_stage_sensing(sample_time_s=0.0)
    )
    bound = _scene(timestamp=0.0)
    first = _select(
        service,
        _scene(6, timestamp=0.0),
        scene_id=6,
        pin_freshest=True,
        load_scene_frame=lambda _: bound,
    )
    second = _select(
        service,
        _scene(6, timestamp=-0.0),
        scene_id=6,
        pin_freshest=True,
        load_scene_frame=lambda _: bound,
    )
    assert second.replay_event is not first.replay_event
    assert second.replay_event.view_manifest_id != first.replay_event.view_manifest_id
    assert second.replay_event.manifest_id == first.replay_event.manifest_id
    assert service.replay_telemetry_snapshot().replay_builds == 1


def test_mutable_numpy_structured_scalar_is_detached_from_private_snapshot():
    service, provider = _service()
    source = provider.frames[10].sensing
    scalar = np.array([(1,)], dtype=[("value", "i4")])[0]
    source["config_resolved"]["extra_structured_scalar"] = scalar
    source["config"]["extra_structured_scalar"] = scalar
    selected = _select(service)
    public = selected.product_frame["sensing"]["config_resolved"]["extra_structured_scalar"]
    public["value"] = 99
    scalar["value"] = 7
    warm = _select(service)
    assert warm.product_frame["sensing"]["config_resolved"]["extra_structured_scalar"]["value"] == 1


def test_cpi_boundary_rewind_and_unpin_keep_the_correct_result():
    service, provider = _service()
    provider.frames[20] = replace(
        build_standard_mpc_frame(frame_idx=20),
        sensing=_detector_stage_sensing(
            scene_frame_id=8, sample_time_s=8.0, range_doppler_fill=2.0
        ),
    )
    first_core = None
    for scene_id, expected_storage in ((5, 10), (6, 10), (7, 10), (8, 20), (9, 20), (6, 10)):
        selected = service.resolve_display_sensing_frame(
            _scene(scene_id),
            scene_frame_id=scene_id,
            storage_frame_id=scene_id,
            candidate_storage_ids=(10, 20),
            product_names=("range_doppler_map",),
            pin_freshest=True,
            load_scene_frame=_scene,
        )
        assert selected is not None and selected.replay_event is not None
        assert selected.storage_frame_id == expected_storage
        expected_bound = 5 if expected_storage == 10 else 8
        assert selected.replay_event.scene_binding.bound_scene_frame_id == expected_bound
        assert selected.replay_event.scene_binding.stale_age_s == float(scene_id - expected_bound)
        assert selected.replay_event.scene_binding.fresh is (scene_id == expected_bound)
        if first_core is None:
            first_core = selected.replay_event.pulses
        if expected_storage == 10:
            assert selected.replay_event.pulses is first_core
        service.load_detection_stage_bundle(selected)
    assert service.replay_telemetry_snapshot().replay_builds == 2
    assert (
        service.resolve_display_sensing_frame(
            _scene(6),
            scene_frame_id=6,
            storage_frame_id=6,
            candidate_storage_ids=(10, 20),
            product_names=("range_doppler_map",),
            pin_freshest=False,
        )
        is None
    )


def test_invalid_revised_configuration_drops_cached_success_and_can_recover():
    service, provider = _service()
    original = _select(service)
    valid = provider.frames[10]
    invalid = deepcopy(valid.sensing)
    invalid["config_resolved"]["processing_recipe"]["tampered"] = True
    invalid["config"] = deepcopy(invalid["config_resolved"])
    provider.frames[10] = replace(valid, sensing=invalid)
    provider.generation = "invalid-config"
    with pytest.raises(ValueError, match="processing_recipe_id must match"):
        _select(service)
    assert service.replay_cache_state().validated_event_entries == 0
    provider.frames[10] = valid
    provider.generation = "repaired-config"
    repaired = _select(service)
    assert repaired.replay_event is not original.replay_event
    assert repaired.replay_event.to_json() == original.replay_event.to_json()


@pytest.mark.parametrize("load_all", [False, True])
def test_cache_miss_validates_owned_snapshot_once_before_binding_checks(monkeypatch, load_all):
    service, _ = _service()
    validator = sensing_service._assert_sensing_frame
    owned_validations = []

    def count(frame, **kwargs):
        if isinstance(frame, sensing_service._ValidatedSensingProductFrame):
            owned_validations.append(kwargs.get("require_described_products", False))
        return validator(frame, **kwargs)

    monkeypatch.setattr(sensing_service, "_assert_sensing_frame", count)
    first = _select(service, load_all_sensing_products=load_all)
    assert first.replay_event is not None
    assert owned_validations == [load_all]
    assert _select(service, load_all_sensing_products=load_all).replay_event is first.replay_event
    assert owned_validations == [load_all]


def test_incomplete_or_mismatched_new_snapshot_never_enters_dense_cache(monkeypatch):
    service, _ = _service()
    partial = _select(service).product_frame
    other, _ = _service()
    monkeypatch.setattr(other, "_load_sensing_candidate", lambda *args, **kwargs: partial)
    with pytest.raises(ValueError):
        _select(other, load_all_sensing_products=True)
    assert other.replay_cache_state().selected_product_entries == 0

    mismatched = {"sensing": _detector_stage_sensing(scene_frame_id=6, sample_time_s=6.0)}
    monkeypatch.setattr(other, "_load_sensing_candidate", lambda *args, **kwargs: mismatched)
    with pytest.raises(ValueError, match="Indexed sensing metadata disagrees"):
        _select(other)
    assert other.replay_cache_state().selected_product_entries == 0


@pytest.mark.parametrize("view", [lambda x: x[1:5], lambda x: x[::2], lambda x: x.reshape(2, 3).T])
def test_snapshot_copy_preserves_sliced_immutable_buffers(view):
    source = view(np.frombuffer(np.arange(6, dtype=float).tobytes()))
    result = sensing_service._copy_sensing_snapshot(source)
    np.testing.assert_array_equal(result, source)
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_new_owned_projection_validates_dense_payload_once(monkeypatch):
    service, _ = _service()
    checked = []
    original = sensing_service._assert_sensing_frame

    def validate(frame, **kwargs):
        if "range_doppler_by_pair" in frame.get("sensing", {}):
            checked.append(frame)
        return original(frame, **kwargs)

    monkeypatch.setattr(sensing_service, "_assert_sensing_frame", validate)
    frame = service._load_sensing_candidate(10, ("range_doppler",), own_snapshot=True)
    assert isinstance(frame, sensing_service._ValidatedSensingProductFrame)
    assert checked == [frame]
    assert not frame["sensing"]["range_doppler_by_pair"].flags.writeable
