"""View rebinding must equal a full canonical build without changing science."""

from dataclasses import replace

import pytest

from shared.sensing.replay import ReplaySceneContext, build_replay_event, rebind_replay_event
from tests.shared.test_sensing_replay import _event, _frame


def _context(**changes):
    return ReplaySceneContext(
        **dict(
            storage_frame_id=10,
            bound_scene_frame_id=101,
            current_scene_frame_id=101,
            bound_scene_time_s=0.001,
            current_scene_time_s=0.001,
            pinned=False,
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"pinned": True},
        {"current_scene_frame_id": 102, "current_scene_time_s": 0.002, "pinned": True},
        {"current_scene_frame_id": 102, "current_scene_time_s": 0.002, "pinned": False},
        {"current_scene_frame_id": 100, "current_scene_time_s": 0.0, "pinned": True},
        {"current_scene_frame_id": "101", "current_scene_time_s": 0.001, "pinned": True},
    ],
)
def test_rebinding_has_exact_full_builder_manifest_parity(changes):
    frame = _frame()
    original = _event(frame)
    context = _context(**changes)
    rebound = rebind_replay_event(original, context)
    fresh = build_replay_event(frame, context)
    assert rebound.to_json() == fresh.to_json()
    assert rebound.manifest_id == original.manifest_id
    for name in ("key", "pair_pose", "pulses", "recipe", "products", "evidence"):
        assert getattr(rebound, name) is getattr(original, name)
    assert rebind_replay_event(rebound, _context()).to_json() == original.to_json()


@pytest.mark.parametrize(
    "status,disposition", [("warning", "ready_with_warnings"), ("fail", "blocked")]
)
def test_rebinding_preserves_unrelated_preflight_warning_or_failure(status, disposition):
    original = _event(_frame())
    checks = tuple(
        replace(check, status=status) if check.check_id == "typed_products" else check
        for check in original.preflight.checks
    )
    modified = replace(
        original, preflight=replace(original.preflight, disposition=disposition, checks=checks)
    )
    stale = rebind_replay_event(
        modified, _context(current_scene_frame_id=102, current_scene_time_s=0.002, pinned=True)
    )
    fresh = rebind_replay_event(stale, _context())
    assert fresh.preflight.disposition == disposition
    assert fresh.preflight == modified.preflight


@pytest.mark.parametrize(
    "changes",
    [
        {"storage_frame_id": 11},
        {"storage_frame_id": "10"},
        {"bound_scene_frame_id": "101"},
        {"bound_scene_frame_id": 102, "current_scene_frame_id": 102},
        {"bound_scene_time_s": 0.002, "current_scene_time_s": 0.002},
    ],
)
def test_rebinding_rejects_changed_source_location_or_bound_time(changes):
    with pytest.raises(ValueError, match="requires unchanged"):
        rebind_replay_event(_event(_frame()), _context(**changes))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rebinding_context_rejects_nonfinite_times(value):
    with pytest.raises(ValueError):
        _context(current_scene_time_s=value)


def test_rebinding_context_rejects_contradictory_time_for_same_scene():
    with pytest.raises(ValueError, match="contradictory"):
        _context(current_scene_time_s=1.0)


def test_rebinding_preserves_signed_zero_view_times_and_rejects_bound_time_changes():
    frame = _frame()
    context = _context(bound_scene_time_s=0.0, current_scene_time_s=0.0)
    original = build_replay_event(frame, context)
    changed = replace(context, current_scene_time_s=-0.0)
    rebound = rebind_replay_event(original, changed)
    assert rebound.to_json() == build_replay_event(frame, changed).to_json()
    assert rebound.view_manifest_id != original.view_manifest_id
    with pytest.raises(ValueError, match="bound_scene_time_s"):
        rebind_replay_event(original, replace(context, bound_scene_time_s=-0.0))
