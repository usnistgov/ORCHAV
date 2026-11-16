"""Read-only dashboard scoring using the shared range/radial evaluator.

Events are saved acquisitions, never playback visits. Missing or unaligned truth
excludes an entire event; an explicitly empty truth snapshot remains evaluable.
The cohort is all valid active targets in the stored pair truth, without inferred
visibility. Display variants and GT-marker visibility do not enter scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from shared.sensing.evaluation import (
    ASSOCIATION_PROTOCOL,
    DEFAULT_RANGE_GATE_M,
    DEFAULT_VELOCITY_GATE_M_S,
    MOT_PROTOCOL,
    AssociationResult,
    associate_detections_to_gt,
    compute_mot_metrics,
    evaluate_scenario,
)
from shared.sensing.evaluation_truth import (
    RECORDED_REFERENCE,
    SIGNAL_TRUTH_VERSION,
    evaluation_velocity_mask,
    extract_evaluation_truth,
)

EVALUATION_CACHE_PROTOCOL = f"{ASSOCIATION_PROTOCOL}/{MOT_PROTOCOL}/{SIGNAL_TRUTH_VERSION}"


def evaluation_cache_current(event: dict | None) -> bool:
    """Never reuse persisted scores computed with a different matching protocol."""
    return bool(event and event.get("evaluation_protocol") == EVALUATION_CACHE_PROTOCOL)


def evaluation_event(sensing: dict, frame: dict | None = None) -> dict:
    """Prepare compact evidence and default scores once, alongside saved products."""
    temporal = sensing.get("temporal") or {}
    result = temporal.get("result") or {}
    acquisition = temporal.get("acquisition") or {}
    identity = (
        (temporal.get("clock") or {}).get("clock_id"),
        result.get("processing_run_id"),
        acquisition.get("pair_id"),
        acquisition.get("acquisition_id"),
    )
    time = result.get("reference_time_s")
    event = {
        "identity": identity,
        "evaluation_protocol": EVALUATION_CACHE_PROTOCOL,
        "time_s": time,
        "truth": None,
        "detections": [
            {k: d[k] for k in ("range_m", "velocity_m_s", "track_id", "track_confirmed") if k in d}
            for d in sensing.get("detections", [])
        ],
        "tracking_enabled": bool(
            (sensing.get("config_resolved") or {}).get("track_confirmation_enabled")
        ),
        "reason": None,
    }
    if (
        not all(isinstance(v, str) and v for v in identity)
        or not isinstance(time, (int, float))
        or not np.isfinite(time)
    ):
        event["reason"] = "Saved result identity/time unavailable"
    elif sensing.get("truth") is None:
        event["reason"] = "Ground truth was not recorded"
    elif (temporal.get("truth") or {}).get("reference_time_s") != time:
        event["reason"] = "Ground truth is not aligned with this result"
    else:
        try:
            truth = extract_evaluation_truth(sensing, frame)
            event["truth"] = truth
            if truth is None or (
                evaluation_velocity_mask(truth) is None
                and any(v is None for v in truth["target_speeds"])
            ):
                event["reason"] = "Ground-truth radial velocity unavailable"
        except (ValueError, TypeError) as exc:
            event["reason"] = f"Ground truth unavailable: {exc}"
    event["default_summary"] = summarize_events([event])
    matched = match_event(event, DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S)
    if matched is not None:
        event["default_assignment"] = asdict(matched[2])
    return event


def _event_outputs(event, confirmed_only):
    """Apply identical evidence eligibility to aggregate scores and match plots."""
    if event is None:
        return [], "No saved result to evaluate"
    if event["reason"]:
        return [], event["reason"]
    detections = event["detections"]
    if confirmed_only:
        if not event["tracking_enabled"]:
            return [], "Track confirmation was not recorded"
        if any("track_confirmed" not in d for d in detections):
            return [], "Track confirmation state unavailable"
        detections = [d for d in detections if d["track_confirmed"]]
    if any(
        any(
            not isinstance(d.get(k), (int, float)) or not np.isfinite(d[k])
            for k in ("range_m", "velocity_m_s")
        )
        for d in detections
    ):
        return [], "Detection range/radial velocity unavailable"
    return detections, None


def match_event(event, range_gate, velocity_gate, *, confirmed_only=False):
    """Expose the shared evaluator's exact assignment for a selected-result plot."""
    detections, reason = _event_outputs(event, confirmed_only)
    if reason:
        return None
    truth = event["truth"]
    prepared = event.get("default_assignment")
    if (
        prepared is not None
        and evaluation_cache_current(event)
        and not confirmed_only
        and range_gate == DEFAULT_RANGE_GATE_M
        and velocity_gate == DEFAULT_VELOCITY_GATE_M_S
    ):
        # JSON caches stringify dictionary keys; restore exact integer indices.
        values = dict(prepared)
        for name in ("gt_to_det", "det_to_gt"):
            values[name] = {int(key): value for key, value in values[name].items()}
        return truth, detections, AssociationResult(**values)
    assignment = associate_detections_to_gt(
        truth["target_ranges"],
        truth["target_speeds"],
        [d["range_m"] for d in detections],
        [d["velocity_m_s"] for d in detections],
        range_gate=range_gate,
        velocity_gate=velocity_gate,
        gt_velocity_valid=evaluation_velocity_mask(truth),
    )
    return truth, detections, assignment


def summarize_events(
    events,
    range_gate=DEFAULT_RANGE_GATE_M,
    velocity_gate=DEFAULT_VELOCITY_GATE_M_S,
    *,
    confirmed_only=False,
) -> dict:
    """Score unique, time-ordered events; report exclusions and exact denominators."""
    unique = {}
    for event in events:
        if event is not None:
            unique[tuple(event["identity"])] = event
    ordered = sorted(
        unique.values(),
        key=lambda e: e["time_s"] if isinstance(e["time_s"], (int, float)) else -np.inf,
    )
    if len({tuple(e["identity"][:3]) for e in ordered}) > 1:
        return {"reason": "Select one clock, processing run and sensor pair"}
    skipped = Counter()
    gt, reports, velocity_masks = [], [], []
    references = {}
    reference_protocols = set()
    for event in ordered:
        detections, reason = _event_outputs(event, confirmed_only)
        if reason:
            skipped[reason] += 1
            continue
        truth = event["truth"]
        gt.append((truth["target_ids"], truth["target_ranges"], truth["target_speeds"]))
        reports.append(detections)
        velocity_masks.append(evaluation_velocity_mask(truth))
        if truth.get("evaluation_reference_protocol"):
            reference_protocols.add(truth["evaluation_reference_protocol"])
        for tid, reference in zip(
            truth["target_ids"],
            truth.get(
                "target_velocity_references", [RECORDED_REFERENCE] * len(truth["target_ids"])
            ),
            strict=True,
        ):
            references.setdefault(tid, set()).add(reference)
    if not gt:
        return {
            "reason": "; ".join(skipped) or "No saved results to evaluate",
            "skipped": dict(skipped),
        }
    metrics = evaluate_scenario(
        [(g[1], g[2]) for g in gt],
        [([d["range_m"] for d in ds], [d["velocity_m_s"] for d in ds]) for ds in reports],
        range_gate=range_gate,
        velocity_gate=velocity_gate,
        per_event_target_ids=[g[0] for g in gt],
        per_event_velocity_valid=(
            [
                mask if mask is not None else [True] * len(g[0])
                for mask, g in zip(velocity_masks, gt, strict=True)
            ]
            if any(mask is not None for mask in velocity_masks)
            else None
        ),
    )
    summary = asdict(metrics)
    summary["association_protocol"] = ASSOCIATION_PROTOCOL
    summary["metric_protocol"] = MOT_PROTOCOL
    summary["truth_reference_protocols"] = sorted(reference_protocols)
    for tid, row in summary["per_target"].items():
        row["velocity_reference"] = " / ".join(sorted(references[tid]))
    denominator = metrics.total_tp + metrics.total_fp
    summary.update(
        reason=None,
        skipped=dict(skipped),
        precision=metrics.total_tp / denominator if denominator else None,
        id_switches=None,
    )
    velocity_complete = all(mask is None or all(mask) for mask in velocity_masks)
    if not velocity_complete:
        summary["identity_metric_unavailable_reason"] = "Instantaneous velocity unavailable"
    if (
        velocity_complete
        and any(reports)
        and all(type(d.get("track_id")) is int for ds in reports for d in ds)
    ):
        try:
            mot = compute_mot_metrics(
                gt, reports, range_gate=range_gate, velocity_gate=velocity_gate
            )
            summary["id_switches"] = mot.total_id_switches
        except ValueError:
            # Duplicated or absent IDs cannot support an identity-switch claim.
            pass
    return summary


def metrics_text(summary: dict) -> str:
    if summary.get("reason"):
        return f"Unavailable: {summary['reason']}"

    def number(value, suffix=""):
        return "—" if value is None else f"{value:.3f}{suffix}"

    counts = ""
    if summary.get("velocity_excluded_matched_pair_count", 0):
        counts = (
            f" · radial pairs {summary['velocity_matched_pair_count']}"
            f"/{summary['matched_pair_count']} (range-only matches excluded)"
        )
    return (
        f"TP {summary['total_tp']} · FP {summary['total_fp']} · FN {summary['total_fn']}"
        f" · range RMSE {number(summary['cumulative_range_rmse'], ' m')}"
        f" · radial RMSE {number(summary['cumulative_velocity_rmse'], ' m/s')}" + counts
    )
