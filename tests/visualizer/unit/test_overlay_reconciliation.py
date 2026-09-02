"""Renderer-neutral coverage and beamforming retry-state contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from visualizer.src.renderers.shared.overlay_reconciliation import (
    CoverageSnapshot,
    capture_owned_snapshot_state,
    complete_owned_snapshot_reconciliation,
    owned_snapshot_plan_succeeded,
    plan_coverage_reconciliation,
    plan_owned_snapshot_reconciliation,
)


@pytest.mark.parametrize(
    ("change", "expected_operation"),
    [
        ("first_upload", "replace_mesh"),
        ("opacity", "update_opacity"),
        ("isolines", "update_isolines"),
        ("hide", "clear"),
    ],
)
def test_coverage_failure_retains_applied_state_and_same_desired_retries(
    change: str,
    expected_operation: str,
) -> None:
    applied = CoverageSnapshot(
        signature="mesh-1",
        isoline_signature="isoline-1",
        opacity=0.8,
        has_mesh=True,
        has_isolines=True,
    )
    desired = applied
    native_mesh = True
    native_isolines = True
    if change == "first_upload":
        applied = None
        desired = replace(desired, has_isolines=False, isoline_signature=None)
        native_mesh = False
        native_isolines = False
    elif change == "opacity":
        desired = replace(desired, opacity=0.3)
    elif change == "isolines":
        desired = replace(desired, isoline_signature="isoline-2")
    else:
        desired = replace(
            desired,
            signature=None,
            isoline_signature=None,
            has_mesh=False,
            has_isolines=False,
        )

    plan = plan_coverage_reconciliation(
        desired,
        applied,
        native_has_mesh=native_mesh,
        native_has_isolines=native_isolines,
    )
    assert getattr(plan, expected_operation) is True

    after_failure = plan.complete(
        succeeded=False,
        native_has_mesh=native_mesh,
        native_has_isolines=native_isolines,
    )
    assert after_failure is applied
    retry = plan_coverage_reconciliation(
        desired,
        after_failure,
        native_has_mesh=native_mesh,
        native_has_isolines=native_isolines,
    )
    assert getattr(retry, expected_operation) is True

    after_success = retry.complete(
        succeeded=True,
        native_has_mesh=desired.has_mesh,
        native_has_isolines=desired.has_isolines,
    )
    assert after_success == desired
    settled = plan_coverage_reconciliation(
        desired,
        after_success,
        native_has_mesh=desired.has_mesh,
        native_has_isolines=desired.has_isolines,
    )
    assert settled.is_noop


@pytest.mark.parametrize("realized_before_failure", [False, True])
def test_owned_snapshot_failure_retries_and_preserves_partial_native_ownership(
    realized_before_failure: bool,
) -> None:
    payload = object()
    desired = {"beam:one": payload}
    state = capture_owned_snapshot_state({}, set())
    plan = plan_owned_snapshot_reconciliation(
        desired,
        state,
        native_ids=set(),
        matches=lambda desired_value, applied_value: desired_value is applied_value,
    )
    assert plan.ensure_ids == ("beam:one",)

    after_failure = complete_owned_snapshot_reconciliation(
        state,
        plan,
        successful_snapshots={},
        realized_ids={"beam:one"} if realized_before_failure else set(),
        removed_ids=set(),
    )
    assert not owned_snapshot_plan_succeeded(
        plan,
        successful_ids=set(),
        removed_ids=set(),
    )
    assert ("beam:one" in after_failure.owned) is realized_before_failure
    assert "beam:one" not in after_failure.applied

    retry = plan_owned_snapshot_reconciliation(
        desired,
        after_failure,
        native_ids={"beam:one"} if realized_before_failure else set(),
        matches=lambda desired_value, applied_value: desired_value is applied_value,
    )
    assert retry.ensure_ids == ("beam:one",)

    after_success = complete_owned_snapshot_reconciliation(
        after_failure,
        retry,
        successful_snapshots={"beam:one": payload},
        realized_ids={"beam:one"},
        removed_ids=set(),
    )
    assert owned_snapshot_plan_succeeded(
        retry,
        successful_ids={"beam:one"},
        removed_ids=set(),
    )
    settled = plan_owned_snapshot_reconciliation(
        desired,
        after_success,
        native_ids={"beam:one"},
        matches=lambda desired_value, applied_value: desired_value is applied_value,
    )
    assert settled.is_noop


@pytest.mark.parametrize("removal_succeeds", [False, True])
def test_owned_snapshot_removal_forgets_state_only_after_confirmed_success(
    removal_succeeds: bool,
) -> None:
    payload = object()
    state = capture_owned_snapshot_state(
        {"beam:stale": payload},
        {"beam:stale"},
    )
    plan = plan_owned_snapshot_reconciliation(
        {},
        state,
        native_ids={"beam:stale"},
        matches=lambda desired_value, applied_value: desired_value is applied_value,
    )
    assert plan.remove_ids == ("beam:stale",)

    removed_ids = {"beam:stale"} if removal_succeeds else set()
    completed = complete_owned_snapshot_reconciliation(
        state,
        plan,
        successful_snapshots={},
        realized_ids=set(),
        removed_ids=removed_ids,
    )
    assert (
        owned_snapshot_plan_succeeded(
            plan,
            successful_ids=set(),
            removed_ids=removed_ids,
        )
        is removal_succeeds
    )
    if removal_succeeds:
        assert completed.applied == {}
        assert completed.owned == frozenset()
    else:
        assert completed.applied == state.applied
        assert completed.owned == state.owned
        retry = plan_owned_snapshot_reconciliation(
            {},
            completed,
            native_ids={"beam:stale"},
            matches=lambda desired_value, applied_value: desired_value is applied_value,
        )
        assert retry.remove_ids == ("beam:stale",)


def test_current_native_snapshot_repairs_missing_ownership_without_an_ensure() -> None:
    payload = object()
    desired = {"beam:current": payload}
    state = capture_owned_snapshot_state({"beam:current": payload}, set())

    plan = plan_owned_snapshot_reconciliation(
        desired,
        state,
        native_ids={"beam:current"},
        matches=lambda desired_value, applied_value: desired_value is applied_value,
    )

    assert plan.ensure_ids == ()
    assert plan.adopt_ids == ("beam:current",)
    repaired = complete_owned_snapshot_reconciliation(
        state,
        plan,
        successful_snapshots={},
        realized_ids=set(),
        removed_ids=set(),
    )
    assert repaired.owned == frozenset({"beam:current"})
