"""Ground-truth to detection association for radar sensing evaluation.

Uses the Hungarian algorithm to optimally assign detections to ground-truth
targets, subject to range and velocity gating constraints. Computes standard
detection metrics (TP, FP, FN) and per-pair accuracy (RMSE).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Dict, List, Optional

import numpy as np

from shared.logging import get_logger

logger = get_logger(__name__)

ASSOCIATION_PROTOCOL = "bounded-range-velocity-v2"
MOT_PROTOCOL = "clear-mot-continuity-v1"
LEGACY_REFERENCE_PROTOCOL = "legacy_range_velocity_v1"
INSTANTANEOUS_REFERENCE_PROTOCOL = "instantaneous_velocity_support_v1"

DEFAULT_RANGE_GATE_M = 2.0
DEFAULT_VELOCITY_GATE_M_S = 2.0

_SENSING_INSTALL_COMMAND = "python -m pip install orchav"


def _solve_linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve one assignment while keeping SciPy outside the base import path."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as exc:
        if exc.name not in {"scipy", "scipy.optimize"}:
            raise
        raise RuntimeError(
            "Sensing association requires SciPy. "
            f"Install ORCHAV with: {_SENSING_INSTALL_COMMAND}"
        ) from exc
    return linear_sum_assignment(cost)


def _gated_linear_assignment(cost: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-cardinality, minimum-cost matching of finite nonnegative edges.

    A rectangular assignment has k=min(rows, columns) cells. After scaling valid
    costs to [0, 1], replacing an invalid edge by a valid edge saves k+1, more
    than any possible change in the sum of its k valid costs. Thus cardinality
    wins first; among equal cardinalities the original cost is minimized. This
    bounded penalty avoids the lost low-order bits of the old 1e18 sentinel.
    Exact ties may choose different pairs under permutation, but never change
    cardinality or the optimal numerical objective. No arbitrary epsilon tie
    breaker perturbs that objective.
    """
    cost = np.asarray(cost, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if cost.ndim != 2 or valid.shape != cost.shape:
        raise ValueError("Costs and gate mask must be aligned matrices")
    if np.any(~np.isfinite(cost[valid])) or np.any(cost[valid] < 0):
        raise ValueError("Valid assignment costs must be finite and nonnegative")
    if not valid.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    scale = float(np.max(cost[valid])) or 1.0
    bounded = np.full(cost.shape, min(cost.shape) + 1.0)
    bounded[valid] = cost[valid] / scale
    rows, cols = _solve_linear_assignment(bounded)
    keep = valid[rows, cols]
    return rows[keep], cols[keep]


@dataclass
class AssociationResult:
    """Result of associating detections to ground-truth targets.

    Attributes:
        gt_to_det: Mapping from GT index to matched detection index (None if missed).
        det_to_gt: Mapping from detection index to matched GT index (None if false alarm).
        true_positives: Number of correctly matched GT-detection pairs.
        false_positives: Number of unmatched detections (false alarms).
        false_negatives: Number of unmatched GT targets (missed detections).
        range_errors: Per-matched-pair range error in metres.
        velocity_errors: Per-matched-pair velocity error in m/s.
        range_rmse: Root-mean-square range error across matched pairs, or
            ``None`` when there are no matched pairs.
        velocity_rmse: Root-mean-square velocity error across matched pairs,
            or ``None`` when there are no matched pairs.
    """

    gt_to_det: Dict[int, Optional[int]]
    det_to_gt: Dict[int, Optional[int]]
    true_positives: int
    false_positives: int
    false_negatives: int
    range_errors: List[float] = field(default_factory=list)
    velocity_errors: List[float] = field(default_factory=list)
    range_rmse: Optional[float] = None
    velocity_rmse: Optional[float] = None
    velocity_matched_pair_count: int = 0
    velocity_excluded_matched_pair_count: int = 0
    velocity_unavailable_opportunity_count: int = 0
    evaluation_reference_protocol: str = LEGACY_REFERENCE_PROTOCOL


@dataclass
class ScenarioEvaluationResult:
    """Aggregate evaluation metrics across all evaluated scenario events.

    ``total_fp / evaluated_event_count`` is a mean false-alarm count, not the
    detection-theory probability of false alarm. ``matched_pair_count`` is the
    pooled range-error denominator; ``velocity_matched_pair_count`` counts
    matches with an eligible velocity reference for pooled velocity error.
    """

    evaluated_event_count: int
    per_event_tp: List[int]
    per_event_fp: List[int]
    per_event_fn: List[int]
    per_event_range_rmse: List[Optional[float]]
    per_event_velocity_rmse: List[Optional[float]]
    total_tp: int
    total_fp: int
    total_fn: int
    target_opportunity_count: int
    matched_pair_count: int
    cumulative_pd: Optional[float]
    mean_false_alarms_per_evaluated_event: Optional[float]
    cumulative_range_rmse: Optional[float]
    cumulative_velocity_rmse: Optional[float]
    all_range_errors: List[float]
    all_velocity_errors: List[float]
    per_target: dict[str, dict] = field(default_factory=dict)
    association_protocol: str = ASSOCIATION_PROTOCOL
    velocity_matched_pair_count: int = 0
    velocity_excluded_matched_pair_count: int = 0
    velocity_unavailable_opportunity_count: int = 0
    evaluation_reference_protocol: str = LEGACY_REFERENCE_PROTOCOL


@dataclass
class MOTMetrics:
    """CLEAR MOT metrics for multi-object tracking evaluation.

    Attributes:
        mota: Multi-Object Tracking Accuracy (1 - (misses + fp + id_switches) / total_gt).
        motp: Multi-Object Tracking Precision (mean range error of matched pairs, metres).
        total_gt: Total GT object appearances across all evaluated events.
        total_matches: Total matched GT-detection pairs across all evaluated events.
        total_misses: Total missed GT targets across all evaluated events.
        total_fp: Total false positive detections across all evaluated events.
        total_id_switches: Total identity switches across all evaluated events.
        num_tracks: Number of unique track IDs observed.
        num_gt_targets: Number of unique semantic GT target IDs.
        track_fragmentation: GT target ID to matched→missed→matched interruption count.
        distinct_track_id_count: GT target ID to number of distinct assigned track IDs.
        metric_protocol: Versioned temporal-association and fragmentation semantics.
        mostly_tracked: GT targets tracked in at least 80% of their lifespan events.
        mostly_lost: GT targets tracked in less than 20% of their lifespan events.
    """

    mota: Optional[float]
    motp: Optional[float]
    total_gt: int
    total_matches: int
    total_misses: int
    total_fp: int
    total_id_switches: int
    num_tracks: int
    num_gt_targets: int
    track_fragmentation: dict[str, int]
    mostly_tracked: int
    mostly_lost: int
    distinct_track_id_count: dict[str, int] = field(default_factory=dict)
    metric_protocol: str = MOT_PROTOCOL


@dataclass
class ThresholdSweepPoint:
    """Metrics for one operating point in a CFAR threshold sweep."""

    threshold_scale: float
    pd: Optional[float]
    mean_false_alarms_per_evaluated_event: Optional[float]
    miss_rate: Optional[float]
    evaluated_event_count: int
    target_opportunity_count: int
    matched_pair_count: int
    total_detections: int
    total_tp: int
    total_fp: int
    total_fn: int
    range_rmse: Optional[float]
    velocity_rmse: Optional[float]
    velocity_matched_pair_count: int = 0
    velocity_excluded_matched_pair_count: int = 0
    velocity_unavailable_opportunity_count: int = 0
    evaluation_reference_protocol: str = LEGACY_REFERENCE_PROTOCOL


def _rmse(errors: List[float]) -> Optional[float]:
    """Compute RMSE, returning ``None`` when its denominator is zero."""
    if not errors:
        return None
    return float(np.hypot.reduce(np.asarray(errors, dtype=float) / math.sqrt(len(errors))))


def _velocity_mask(values, count: int) -> np.ndarray:
    """Require explicit aligned booleans; omitted masks retain both-axis scoring."""
    if values is None:
        return np.ones(count, dtype=bool)
    array = np.asarray(values)
    if array.ndim == 1 and len(array) == count == 0:
        return np.empty(0, dtype=bool)
    if array.ndim != 1 or len(array) != count or array.dtype.kind != "b":
        raise ValueError("gt_velocity_valid must be an aligned one-dimensional boolean mask")
    return array


def associate_detections_to_gt(
    gt_ranges: np.ndarray,
    gt_velocities: np.ndarray,
    det_ranges: np.ndarray,
    det_velocities: np.ndarray,
    range_gate: float = DEFAULT_RANGE_GATE_M,
    velocity_gate: float = DEFAULT_VELOCITY_GATE_M_S,
    *,
    gt_velocity_valid: Sequence[bool] | None = None,
) -> AssociationResult:
    """Associate radar detections to ground-truth targets using the Hungarian algorithm.

    Builds a dimensionless Euclidean cost matrix by normalising range and
    velocity residuals by their respective gates, applies those gates, then
    solves the optimal assignment with
    :func:`scipy.optimize.linear_sum_assignment`.

    Args:
        gt_ranges: Ground-truth ranges in metres, shape ``(N_gt,)``.
        gt_velocities: Ground-truth velocities in m/s, shape ``(N_gt,)``.
        det_ranges: Detected ranges in metres, shape ``(N_det,)``.
        det_velocities: Detected velocities in m/s, shape ``(N_det,)``.
        range_gate: Maximum allowable range error in metres for a valid association.
        velocity_gate: Maximum allowable velocity error in m/s for a valid association.
        gt_velocity_valid: Optional explicit velocity support per target. False
            rows use only the range gate/cost and contribute no velocity error.
            Omitting the mask requires finite velocity for every target.

    Returns:
        An :class:`AssociationResult` containing mappings, detection counts, and
        per-pair accuracy metrics.
    """
    gt_ranges = np.asarray(gt_ranges, dtype=np.float64).ravel()
    gt_velocities = np.asarray(gt_velocities, dtype=np.float64).ravel()
    det_ranges = np.asarray(det_ranges, dtype=np.float64).ravel()
    det_velocities = np.asarray(det_velocities, dtype=np.float64).ravel()
    if gt_ranges.size != gt_velocities.size:
        raise ValueError("Ground-truth range and velocity arrays must have equal length")
    if det_ranges.size != det_velocities.size:
        raise ValueError("Detection range and velocity arrays must have equal length")
    velocity_valid = _velocity_mask(gt_velocity_valid, gt_ranges.size)
    reference_protocol = (
        LEGACY_REFERENCE_PROTOCOL if gt_velocity_valid is None else INSTANTANEOUS_REFERENCE_PROTOCOL
    )
    unavailable = int(np.count_nonzero(~velocity_valid))
    if not all(
        np.all(np.isfinite(values))
        for values in (gt_ranges, gt_velocities[velocity_valid], det_ranges, det_velocities)
    ):
        raise ValueError("Association inputs must contain only finite values")
    if not math.isfinite(float(range_gate)) or float(range_gate) <= 0.0:
        raise ValueError("range_gate must be finite and positive")
    if not math.isfinite(float(velocity_gate)) or float(velocity_gate) <= 0.0:
        raise ValueError("velocity_gate must be finite and positive")

    n_gt = len(gt_ranges)
    n_det = len(det_ranges)

    # Initialise all GT as missed, all detections as false alarms.
    gt_to_det: Dict[int, Optional[int]] = {i: None for i in range(n_gt)}
    det_to_gt: Dict[int, Optional[int]] = {j: None for j in range(n_det)}

    # --- Edge cases: empty GT or empty detections ---
    if n_gt == 0:
        logger.debug("No GT targets; all %d detections are false positives", n_det)
        return AssociationResult(
            gt_to_det=gt_to_det,
            det_to_gt=det_to_gt,
            true_positives=0,
            false_positives=n_det,
            false_negatives=0,
            evaluation_reference_protocol=reference_protocol,
        )

    if n_det == 0:
        logger.debug("No detections; all %d GT targets are missed", n_gt)
        return AssociationResult(
            gt_to_det=gt_to_det,
            det_to_gt=det_to_gt,
            true_positives=0,
            false_positives=0,
            false_negatives=n_gt,
            velocity_unavailable_opportunity_count=unavailable,
            evaluation_reference_protocol=reference_protocol,
        )

    # --- Build cost matrix with gating ---
    range_diff = gt_ranges[:, np.newaxis] - det_ranges[np.newaxis, :]  # (N_gt, N_det)
    # Unavailable rows never use their numeric placeholder in a gate or error.
    velocity = np.where(velocity_valid, gt_velocities, 0.0)
    vel_diff = velocity[:, np.newaxis] - det_velocities[np.newaxis, :]
    vel_diff = np.where(velocity_valid[:, np.newaxis], vel_diff, 0.0)

    cost = np.hypot(range_diff / float(range_gate), vel_diff / float(velocity_gate))

    valid = (np.abs(range_diff) <= range_gate) & (np.abs(vel_diff) <= velocity_gate)
    row_ind, col_ind = _gated_linear_assignment(cost, valid)

    range_errors: List[float] = []
    velocity_errors: List[float] = []

    for gt_idx, det_idx in zip(row_ind, col_ind, strict=True):
        gt_to_det[int(gt_idx)] = int(det_idx)
        det_to_gt[int(det_idx)] = int(gt_idx)
        range_errors.append(float(range_diff[gt_idx, det_idx]))
        if velocity_valid[gt_idx]:
            velocity_errors.append(float(vel_diff[gt_idx, det_idx]))

    tp = len(range_errors)
    fp = n_det - tp
    fn = n_gt - tp

    logger.debug(
        "Association: TP=%d  FP=%d  FN=%d  (N_gt=%d, N_det=%d)",
        tp,
        fp,
        fn,
        n_gt,
        n_det,
    )

    return AssociationResult(
        gt_to_det=gt_to_det,
        det_to_gt=det_to_gt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        range_errors=range_errors,
        velocity_errors=velocity_errors,
        range_rmse=_rmse(range_errors),
        velocity_rmse=_rmse(velocity_errors),
        velocity_matched_pair_count=len(velocity_errors),
        velocity_excluded_matched_pair_count=tp - len(velocity_errors),
        velocity_unavailable_opportunity_count=unavailable,
        evaluation_reference_protocol=reference_protocol,
    )


def evaluate_scenario(
    per_event_gt: List[tuple],
    per_event_detections: List[tuple],
    *,
    range_gate: float = DEFAULT_RANGE_GATE_M,
    velocity_gate: float = DEFAULT_VELOCITY_GATE_M_S,
    per_event_target_ids: Sequence[Sequence[str]] | None = None,
    per_event_velocity_valid: Sequence[Sequence[bool]] | None = None,
) -> ScenarioEvaluationResult:
    """Compute cumulative detection metrics across evaluated events.

    Args:
        per_event_gt: ``(gt_ranges, gt_velocities)`` arrays per evaluated event.
        per_event_detections: Detection range/velocity arrays per evaluated event.
        per_event_target_ids: Optional stable target IDs for per-target accuracy,
            using the same global assignment as the aggregate metrics.
        per_event_velocity_valid: Optional explicit support masks. Range-only
            matches remain counted; velocity denominators exclude those matches.
        range_gate: Maximum range error for a valid association (metres).
        velocity_gate: Maximum velocity error for a valid association (m/s).

    Returns:
        Aggregated evaluation result with per-event and cumulative metrics.
    """
    if len(per_event_gt) != len(per_event_detections):
        raise ValueError(
            "Ground-truth and detection event sequences must have equal length; "
            "evaluation never truncates events"
        )
    evaluated_event_count = len(per_event_gt)
    if per_event_target_ids is not None and len(per_event_target_ids) != evaluated_event_count:
        raise ValueError("Target IDs must align with evaluated events")
    if (
        per_event_velocity_valid is not None
        and len(per_event_velocity_valid) != evaluated_event_count
    ):
        raise ValueError("Velocity support masks must align with evaluated events")
    per_target = {}

    per_event_tp: List[int] = []
    per_event_fp: List[int] = []
    per_event_fn: List[int] = []
    per_event_range_rmse: List[Optional[float]] = []
    per_event_velocity_rmse: List[Optional[float]] = []
    all_range_errors: List[float] = []
    all_velocity_errors: List[float] = []
    velocity_unavailable_opportunities = 0

    for event_index in range(evaluated_event_count):
        gt_r, gt_v = per_event_gt[event_index]
        det_r, det_v = per_event_detections[event_index]
        gt_r = np.asarray(gt_r, dtype=np.float64).ravel()
        gt_v = np.asarray(gt_v, dtype=np.float64).ravel()
        det_r = np.asarray(det_r, dtype=np.float64).ravel()
        det_v = np.asarray(det_v, dtype=np.float64).ravel()
        mask = None if per_event_velocity_valid is None else per_event_velocity_valid[event_index]
        velocity_valid = _velocity_mask(mask, len(gt_r))

        result = associate_detections_to_gt(
            gt_ranges=gt_r,
            gt_velocities=gt_v,
            det_ranges=det_r,
            det_velocities=det_v,
            range_gate=range_gate,
            velocity_gate=velocity_gate,
            gt_velocity_valid=mask,
        )
        velocity_unavailable_opportunities += result.velocity_unavailable_opportunity_count

        if per_event_target_ids is not None:
            ids = per_event_target_ids[event_index]
            if len(ids) != len(gt_r) or len(set(ids)) != len(ids):
                raise ValueError("Target IDs must be unique and align with ground truth")
            for gi, tid in enumerate(ids):
                row = per_target.setdefault(
                    tid,
                    dict(
                        eligible=0,
                        matched=0,
                        range_errors=[],
                        velocity_errors=[],
                        velocity_matched=0,
                        velocity_excluded_matched=0,
                        velocity_unavailable_opportunities=0,
                    ),
                )
                row["eligible"] += 1
                row["velocity_unavailable_opportunities"] += int(not velocity_valid[gi])
                di = result.gt_to_det[gi]
                if di is not None:
                    row["matched"] += 1
                    row["range_errors"].append(float(gt_r[gi] - det_r[di]))
                    if velocity_valid[gi]:
                        row["velocity_errors"].append(float(gt_v[gi] - det_v[di]))
                        row["velocity_matched"] += 1
                    else:
                        row["velocity_excluded_matched"] += 1

        per_event_tp.append(result.true_positives)
        per_event_fp.append(result.false_positives)
        per_event_fn.append(result.false_negatives)
        per_event_range_rmse.append(result.range_rmse)
        per_event_velocity_rmse.append(result.velocity_rmse)
        all_range_errors.extend(result.range_errors)
        all_velocity_errors.extend(result.velocity_errors)

    total_tp = sum(per_event_tp)
    total_fp = sum(per_event_fp)
    total_fn = sum(per_event_fn)
    target_opportunity_count = total_tp + total_fn

    cumulative_pd = total_tp / target_opportunity_count if target_opportunity_count > 0 else None
    mean_false_alarms = total_fp / evaluated_event_count if evaluated_event_count > 0 else None

    for row in per_target.values():
        row["missed"] = row["eligible"] - row["matched"]
        row["coverage"] = row["matched"] / row["eligible"]
        row["range_rmse"] = _rmse(row.pop("range_errors"))
        row["velocity_rmse"] = _rmse(row.pop("velocity_errors"))

    return ScenarioEvaluationResult(
        per_target=per_target,
        evaluated_event_count=evaluated_event_count,
        per_event_tp=per_event_tp,
        per_event_fp=per_event_fp,
        per_event_fn=per_event_fn,
        per_event_range_rmse=per_event_range_rmse,
        per_event_velocity_rmse=per_event_velocity_rmse,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        target_opportunity_count=target_opportunity_count,
        matched_pair_count=total_tp,
        cumulative_pd=cumulative_pd,
        mean_false_alarms_per_evaluated_event=mean_false_alarms,
        cumulative_range_rmse=_rmse(all_range_errors),
        cumulative_velocity_rmse=_rmse(all_velocity_errors),
        all_range_errors=all_range_errors,
        all_velocity_errors=all_velocity_errors,
        velocity_matched_pair_count=len(all_velocity_errors),
        velocity_excluded_matched_pair_count=total_tp - len(all_velocity_errors),
        velocity_unavailable_opportunity_count=velocity_unavailable_opportunities,
        evaluation_reference_protocol=(
            LEGACY_REFERENCE_PROTOCOL
            if per_event_velocity_valid is None
            else INSTANTANEOUS_REFERENCE_PROTOCOL
        ),
    )


def _validated_track_ids(detections: Sequence[dict], event_index: int) -> List[int]:
    """Return exact integer track IDs after enforcing one ID per detection/event."""

    track_ids: List[int] = []
    seen: set[int] = set()
    for detection_idx, detection in enumerate(detections):
        if "track_id" not in detection:
            raise ValueError(
                f"MOT event {event_index} detection {detection_idx} is missing track_id"
            )
        raw_track_id = detection["track_id"]
        if isinstance(raw_track_id, (bool, np.bool_)) or not isinstance(raw_track_id, Real):
            raise ValueError(
                f"MOT event {event_index} detection {detection_idx} track_id "
                "must be a finite integer"
            )
        if isinstance(raw_track_id, Integral):
            track_id = int(raw_track_id)
        else:
            numeric_track_id = float(raw_track_id)
            if not math.isfinite(numeric_track_id) or not numeric_track_id.is_integer():
                raise ValueError(
                    f"MOT event {event_index} detection {detection_idx} track_id "
                    "must be a finite integer"
                )
            track_id = int(numeric_track_id)
        if track_id in seen:
            raise ValueError(f"MOT event {event_index} has duplicate track_id {track_id}")
        seen.add(track_id)
        track_ids.append(track_id)
    return track_ids


def compute_mot_metrics(
    per_event_gt: Sequence[tuple[Sequence[str], np.ndarray, np.ndarray]],
    per_event_detections: List[List[dict]],
    *,
    range_gate: float = DEFAULT_RANGE_GATE_M,
    velocity_gate: float = DEFAULT_VELOCITY_GATE_M_S,
) -> MOTMetrics:
    """Compute CLEAR MOT tracking metrics across evaluated events.

    Preserves each previous GT/track association whenever it remains within
    both gates, then optimally assigns remaining pairs. This matches CLEAR MOT
    continuity (motmetrics' default unlimited switch history). Missed or absent
    events do not erase identity history. Fragmentation counts observed misses
    between successful matches; absence from GT is not a missed opportunity.
    MOTP remains mean absolute range error, not normalized association distance.

    Args:
        per_event_gt: Sequence of ``(target_ids, gt_ranges, gt_velocities)``
            records per evaluated event. IDs are exact semantic target identities;
            row order may change independently on every event.
        per_event_detections: Detection dictionaries per evaluated event. Each
            detection dict must contain ``range_m``, ``velocity_m_s``, and
            ``track_id`` keys.
        range_gate: Maximum range error for a valid association (metres).
        velocity_gate: Maximum velocity error for a valid association (m/s).

    Returns:
        :class:`MOTMetrics` with MOTA, MOTP, ID switches, fragmentation, and
        track-level statistics.
    """
    if len(per_event_gt) != len(per_event_detections):
        raise ValueError(
            "Ground-truth and tracking event sequences must have equal length; "
            "MOT evaluation never truncates events"
        )
    evaluated_event_count = len(per_event_gt)

    total_gt = 0
    total_matches = 0
    total_fp = 0
    total_misses = 0
    total_id_switches = 0
    all_range_errors: List[float] = []
    all_track_ids: set[int] = set()

    # Every cross-event join is keyed by exact semantic target ID. Array row
    # indices exist only inside one Hungarian assignment.
    gt_track_ids: dict[str, set[int]] = {}
    gt_lifespan: dict[str, int] = {}
    gt_tracked_count: dict[str, int] = {}
    all_gt_ids: set[str] = set()

    # Last matched exact target ID -> track_id mapping for switch detection.
    # This intentionally survives missed events so a reappearing target with a
    # new identity counts as a switch.
    last_gt_to_track: dict[str, int] = {}
    interrupted: set[str] = set()
    fragmentation: dict[str, int] = {}

    for event_index in range(evaluated_event_count):
        gt_event = per_event_gt[event_index]
        if len(gt_event) != 3:
            raise ValueError(
                f"MOT event {event_index} must contain target_ids, ranges, and velocities"
            )
        target_ids_raw, gt_r, gt_v = gt_event
        target_ids = tuple(target_ids_raw)
        if any(not isinstance(target_id, str) or not target_id for target_id in target_ids):
            raise ValueError(
                f"MOT event {event_index} target_ids must be non-empty semantic strings"
            )
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"MOT event {event_index} has duplicate target_ids")
        gt_r = np.asarray(gt_r, dtype=np.float64).ravel()
        gt_v = np.asarray(gt_v, dtype=np.float64).ravel()
        n_gt = len(gt_r)
        if len(target_ids) != n_gt or len(gt_v) != n_gt:
            raise ValueError(
                f"MOT event {event_index} target_ids/ranges/velocities must align exactly"
            )
        all_gt_ids.update(target_ids)
        total_gt += n_gt

        # Record GT lifespan
        for target_id in target_ids:
            gt_lifespan[target_id] = gt_lifespan.get(target_id, 0) + 1

        dets = per_event_detections[event_index]
        track_ids = _validated_track_ids(dets, event_index)
        all_track_ids.update(track_ids)
        det_r = np.array([d["range_m"] for d in dets], dtype=np.float64) if dets else np.array([])
        det_v = (
            np.array([d["velocity_m_s"] for d in dets], dtype=np.float64) if dets else np.array([])
        )

        # Associate detections to GT
        assoc = associate_detections_to_gt(
            gt_ranges=gt_r,
            gt_velocities=gt_v,
            det_ranges=det_r,
            det_velocities=det_v,
            range_gate=range_gate,
            velocity_gate=velocity_gate,
        )

        # CLEAR continuity is intentionally higher priority than reoptimizing
        # residuals or cardinality. Validate the full event above, then lock
        # feasible prior pairs before solving the remaining subproblem.
        preserved: dict[int, int] = {}
        available = {track_id: di for di, track_id in enumerate(track_ids)}
        for gi, target_id in enumerate(target_ids):
            di = available.get(last_gt_to_track.get(target_id))
            if di is not None and (
                abs(gt_r[gi] - det_r[di]) <= range_gate
                and abs(gt_v[gi] - det_v[di]) <= velocity_gate
            ):
                preserved[gi] = di
                del available[track_ids[di]]
        matched_gt_to_det = assoc.gt_to_det
        if preserved:
            remaining_gt = [gi for gi in range(n_gt) if gi not in preserved]
            remaining_det = sorted(available.values())
            remainder = associate_detections_to_gt(
                gt_r[remaining_gt],
                gt_v[remaining_gt],
                det_r[remaining_det],
                det_v[remaining_det],
                range_gate=range_gate,
                velocity_gate=velocity_gate,
            )
            pairs = {
                **preserved,
                **{
                    remaining_gt[gi]: remaining_det[di]
                    for gi, di in remainder.gt_to_det.items()
                    if di is not None
                },
            }
            matched_gt_to_det = {gi: pairs.get(gi) for gi in range(n_gt)}

        event_matches = sum(di is not None for di in matched_gt_to_det.values())
        total_fp += len(dets) - event_matches
        total_misses += n_gt - event_matches
        total_matches += event_matches

        # Build this event's GT -> track_id mapping and detect ID switches.
        for gi, di in matched_gt_to_det.items():
            target_id = target_ids[gi]
            if di is None:
                if target_id in last_gt_to_track:
                    interrupted.add(target_id)
                continue
            if target_id in interrupted:
                fragmentation[target_id] = fragmentation.get(target_id, 0) + 1
                interrupted.remove(target_id)

            # Record range error for MOTP
            range_err = abs(float(gt_r[gi] - det_r[di]))
            all_range_errors.append(range_err)

            track_id = track_ids[di]
            target_id = target_ids[gi]

            # Record GT-to-track assignment for fragmentation
            if target_id not in gt_track_ids:
                gt_track_ids[target_id] = set()
            gt_track_ids[target_id].add(track_id)

            # Record that this GT target was tracked in this event.
            gt_tracked_count[target_id] = gt_tracked_count.get(target_id, 0) + 1

            # Check for an ID switch against the target's last successful
            # association, including reappearance after one or more misses.
            if target_id in last_gt_to_track and last_gt_to_track[target_id] != track_id:
                total_id_switches += 1
            last_gt_to_track[target_id] = track_id

    # Compute MOTA and MOTP
    if total_gt > 0:
        mota = 1.0 - (total_misses + total_fp + total_id_switches) / total_gt
    else:
        mota = None

    motp = float(np.mean(all_range_errors)) if all_range_errors else None

    track_fragmentation = {
        target_id: fragmentation.get(target_id, 0) for target_id in sorted(all_gt_ids)
    }
    distinct_track_id_count = {
        target_id: len(gt_track_ids.get(target_id, set())) for target_id in sorted(all_gt_ids)
    }

    # Compute mostly tracked / mostly lost
    mostly_tracked = 0
    mostly_lost = 0
    for target_id in sorted(all_gt_ids):
        lifespan = gt_lifespan.get(target_id, 0)
        if lifespan == 0:
            continue
        tracked_ratio = gt_tracked_count.get(target_id, 0) / lifespan
        if tracked_ratio >= 0.8:
            mostly_tracked += 1
        if tracked_ratio < 0.2:
            mostly_lost += 1

    logger.debug(
        "MOT: MOTA=%s  MOTP=%s  IDSW=%d  matches=%d  misses=%d  fp=%d  gt=%d",
        mota,
        motp,
        total_id_switches,
        total_matches,
        total_misses,
        total_fp,
        total_gt,
    )

    return MOTMetrics(
        mota=mota,
        motp=motp,
        total_gt=total_gt,
        total_matches=total_matches,
        total_misses=total_misses,
        total_fp=total_fp,
        total_id_switches=total_id_switches,
        num_tracks=len(all_track_ids),
        num_gt_targets=len(all_gt_ids),
        track_fragmentation=track_fragmentation,
        distinct_track_id_count=distinct_track_id_count,
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
    )
