"""Pure reconciliation decisions for renderer-owned surface overlays.

Coverage and beamforming use different native objects in Open3D and pygfx, but
their retry contract is the same: compare desired state with the last
successfully applied snapshot, retain ownership of partially created native
objects, and never advance applied state after a failed operation.  This module
contains only those renderer-neutral decisions; adapters still perform every
geometry and material operation themselves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """Desired or successfully applied coverage overlay state."""

    signature: str | None
    isoline_signature: str | None
    opacity: float
    has_mesh: bool
    has_isolines: bool


@dataclass(frozen=True, slots=True)
class CoverageReconcilePlan:
    """Native operations needed to converge one coverage snapshot."""

    desired: CoverageSnapshot
    applied: CoverageSnapshot | None
    clear: bool
    replace_mesh: bool
    update_opacity: bool
    update_isolines: bool

    @property
    def is_noop(self) -> bool:
        """Return whether native coverage already matches ``desired``."""
        return not (self.clear or self.replace_mesh or self.update_opacity or self.update_isolines)

    def complete(
        self,
        *,
        succeeded: bool,
        native_has_mesh: bool,
        native_has_isolines: bool,
    ) -> CoverageSnapshot | None:
        """Return the next applied state without committing failed attempts."""
        native_matches = (
            bool(native_has_mesh) is self.desired.has_mesh
            and bool(native_has_isolines) is self.desired.has_isolines
        )
        if succeeded and native_matches:
            return self.desired
        return self.applied


def coverage_snapshot_from_packet(packet: Any) -> CoverageSnapshot:
    """Extract inexpensive coverage identity and presence from a frame packet."""

    def _has_rows(value: Any) -> bool:
        if value is None:
            return False
        try:
            return len(value) > 0
        except TypeError:
            return bool(getattr(value, "size", value))

    has_mesh = bool(
        getattr(packet, "show_coverage", False)
        and _has_rows(getattr(packet, "coverage_vertices", None))
        and _has_rows(getattr(packet, "coverage_triangles", None))
    )
    has_isolines = bool(
        has_mesh
        and _has_rows(getattr(packet, "coverage_isoline_points", None))
        and _has_rows(getattr(packet, "coverage_isoline_lines", None))
    )
    signature = getattr(packet, "coverage_signature", None) if has_mesh else None
    isoline_signature: str | None = None
    if has_isolines:
        metadata = getattr(packet, "coverage_metadata", None)
        if isinstance(metadata, dict) and metadata.get("isoline_signature") is not None:
            isoline_signature = str(metadata["isoline_signature"])
        else:
            isoline_signature = signature
    return CoverageSnapshot(
        signature=signature,
        isoline_signature=isoline_signature,
        opacity=float(getattr(packet, "coverage_opacity", 1.0)),
        has_mesh=has_mesh,
        has_isolines=has_isolines,
    )


def plan_coverage_reconciliation(
    desired: CoverageSnapshot,
    applied: CoverageSnapshot | None,
    *,
    native_has_mesh: bool,
    native_has_isolines: bool,
) -> CoverageReconcilePlan:
    """Plan coverage work from desired, applied, and actual native ownership."""
    native_has_mesh = bool(native_has_mesh)
    native_has_isolines = bool(native_has_isolines)
    if not desired.has_mesh:
        return CoverageReconcilePlan(
            desired=desired,
            applied=applied,
            clear=native_has_mesh or native_has_isolines,
            replace_mesh=False,
            update_opacity=False,
            update_isolines=False,
        )

    mesh_is_current = bool(
        desired.signature is not None
        and applied is not None
        and applied.has_mesh
        and applied.signature == desired.signature
        and native_has_mesh
    )
    if not mesh_is_current:
        return CoverageReconcilePlan(
            desired=desired,
            applied=applied,
            clear=False,
            replace_mesh=True,
            update_opacity=False,
            update_isolines=False,
        )

    isolines_are_current = bool(
        applied is not None
        and applied.has_isolines is desired.has_isolines
        and native_has_isolines is desired.has_isolines
        and (not desired.has_isolines or applied.isoline_signature == desired.isoline_signature)
    )
    opacity_is_current = bool(
        applied is not None
        and math.isclose(applied.opacity, desired.opacity, rel_tol=1e-5, abs_tol=1e-8)
    )
    return CoverageReconcilePlan(
        desired=desired,
        applied=applied,
        clear=False,
        replace_mesh=False,
        update_opacity=not opacity_is_current,
        update_isolines=not isolines_are_current,
    )


KeyT = TypeVar("KeyT")
DesiredT = TypeVar("DesiredT")
AppliedT = TypeVar("AppliedT")


@dataclass(frozen=True, slots=True)
class OwnedSnapshotState(Generic[KeyT, AppliedT]):
    """Successfully applied snapshots plus possibly partial native ownership."""

    applied: Mapping[KeyT, AppliedT]
    owned: frozenset[KeyT]


@dataclass(frozen=True, slots=True)
class OwnedSnapshotPlan(Generic[KeyT]):
    """Per-ID work required for a desired set of native overlay objects."""

    ensure_ids: tuple[KeyT, ...]
    remove_ids: tuple[KeyT, ...]
    adopt_ids: tuple[KeyT, ...]

    @property
    def is_noop(self) -> bool:
        """Return whether neither native work nor ownership repair is needed."""
        return not (self.ensure_ids or self.remove_ids or self.adopt_ids)


def capture_owned_snapshot_state(
    applied: Mapping[KeyT, AppliedT],
    owned: Set[KeyT],
) -> OwnedSnapshotState[KeyT, AppliedT]:
    """Copy mutable backend bookkeeping into one immutable reconciliation input."""
    return OwnedSnapshotState(
        applied=dict(applied),
        owned=frozenset(owned),
    )


def plan_owned_snapshot_reconciliation(
    desired: Mapping[KeyT, DesiredT],
    state: OwnedSnapshotState[KeyT, AppliedT],
    *,
    native_ids: Set[KeyT],
    matches: Callable[[DesiredT, AppliedT], bool],
) -> OwnedSnapshotPlan[KeyT]:
    """Plan per-ID ensures/removals without performing backend operations."""
    ensure_ids: list[KeyT] = []
    adopt_ids: list[KeyT] = []
    for object_id, desired_snapshot in desired.items():
        applied_snapshot = state.applied.get(object_id)
        is_current = bool(
            object_id in native_ids
            and applied_snapshot is not None
            and matches(desired_snapshot, applied_snapshot)
        )
        if not is_current:
            ensure_ids.append(object_id)
        elif object_id not in state.owned:
            adopt_ids.append(object_id)

    tracked_ids = set(state.owned) | set(state.applied)
    remove_ids = sorted(
        (object_id for object_id in tracked_ids if object_id not in desired),
        key=str,
    )
    return OwnedSnapshotPlan(
        ensure_ids=tuple(ensure_ids),
        remove_ids=tuple(remove_ids),
        adopt_ids=tuple(adopt_ids),
    )


def complete_owned_snapshot_reconciliation(
    state: OwnedSnapshotState[KeyT, AppliedT],
    plan: OwnedSnapshotPlan[KeyT],
    *,
    successful_snapshots: Mapping[KeyT, AppliedT],
    realized_ids: Set[KeyT],
    removed_ids: Set[KeyT],
) -> OwnedSnapshotState[KeyT, AppliedT]:
    """Apply successful results while retaining every failed operation for retry."""
    applied = dict(state.applied)
    owned = set(state.owned)
    owned.update(plan.adopt_ids)
    owned.update(realized_ids)

    for object_id, snapshot in successful_snapshots.items():
        applied[object_id] = snapshot
        owned.add(object_id)
    for object_id in removed_ids:
        applied.pop(object_id, None)
        owned.discard(object_id)

    return OwnedSnapshotState(applied=applied, owned=frozenset(owned))


def owned_snapshot_plan_succeeded(
    plan: OwnedSnapshotPlan[KeyT],
    *,
    successful_ids: Set[KeyT],
    removed_ids: Set[KeyT],
) -> bool:
    """Return whether every operation in ``plan`` completed successfully."""
    return set(plan.ensure_ids).issubset(successful_ids) and set(plan.remove_ids).issubset(
        removed_ids
    )
