"""Whole-retained-run scoring from compact saved acquisitions, never dense spectra.

Plot samples are bounded; scalar denominators and squared-error sums are not.
No warm-up, visibility, confirmation, or probability-of-false-alarm inference is
made. Association and truth/reference selection use the dashboard evaluator.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, fields
from numbers import Real

import numpy as np

from shared.sensing.evaluation import INSTANTANEOUS_REFERENCE_PROTOCOL, LEGACY_REFERENCE_PROTOCOL
from shared.sensing.evaluation_truth import RECORDED_REFERENCE, evaluation_velocity_mask

from .component_cache import load_component, save_component
from .dashboard_evaluation import (
    DEFAULT_RANGE_GATE_M,
    DEFAULT_VELOCITY_GATE_M_S,
    EVALUATION_CACHE_PROTOCOL,
    SIGNAL_TRUTH_VERSION,
    evaluation_event,
    match_event,
)
from .time_interval import contains_time

SUMMARY_PROTOCOL = "run-summary-v1"


@dataclass(frozen=True)
class RunSummaryData:
    context: tuple[str, str, str]
    summary: dict = field(default_factory=dict)
    acquisition_count: int = 0
    evaluated_count: int = 0
    plot_sampled: bool = False
    cancelled: bool = False
    error: str | None = None
    uncertainty: tuple[str, ...] = ()
    time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_tp: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_fp: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_fn: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_gt_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_output_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_tp: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_fp: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_fn: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_recall: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_precision: np.ndarray = field(default_factory=lambda: np.empty(0))
    confirmed_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    tentative_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    tracking_population_available: bool = False
    range_errors_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    radial_errors_m_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_range_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_radial_m_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_target_ids: tuple[str, ...] = ()
    output_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    output_range_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    output_radial_m_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    output_track_ids: tuple[int | None, ...] = ()
    tracking_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    tracking_target_ids: tuple[str, ...] = ()
    tracking_track_ids: tuple[int, ...] = ()
    tracking_events: tuple[dict, ...] = ()


def summary_cache_key(
    context, range_gate, velocity_gate, interval, confirmed_only, max_plot_points
):
    """Bind all-event totals and bounded samples to their exact scoring settings."""
    return (
        *context,
        float(range_gate),
        float(velocity_gate),
        tuple(interval) if interval is not None else None,
        bool(confirmed_only),
        max_plot_points,
        EVALUATION_CACHE_PROTOCOL,
        SIGNAL_TRUTH_VERSION,
    )


def _load_cached_summary(cache, key, context, count, max_plot_points, interval, cancelled):
    defaults = RunSummaryData(context)
    expected = {field.name for field in fields(RunSummaryData)}
    expected_arrays = {name for name in expected if isinstance(getattr(defaults, name), np.ndarray)}
    loaded = load_component(
        cache,
        "run-summary",
        key,
        protocol=SUMMARY_PROTOCOL,
        byte_limit=max_plot_points * len(expected_arrays) * np.dtype(float).itemsize,
        cancelled=cancelled,
    )
    if loaded is None:
        return None
    metadata, arrays = loaded
    try:
        if set(metadata) & set(arrays) or set(metadata) | set(arrays) != expected:
            return None
        if set(arrays) != expected_arrays or any(
            value.ndim != 1 or value.dtype != np.dtype(float) or len(value) > max_plot_points
            for value in arrays.values()
        ):
            return None
        for name in expected - expected_arrays:
            if isinstance(getattr(defaults, name), tuple):
                metadata[name] = tuple(metadata[name])
        result = RunSummaryData(**metadata, **arrays)
        if (
            result.context != context
            or type(result.acquisition_count) is not int
            or result.acquisition_count != count
            or type(result.evaluated_count) is not int
            or not 0 <= result.evaluated_count <= count
            or result.cancelled
            or result.error is not None
            or not isinstance(result.summary, dict)
            or result.summary.get("evaluated_event_count") != result.evaluated_count
            or result.summary.get("excluded_event_count") != count - result.evaluated_count
            or result.summary.get("range_gate_m") != key[3]
            or result.summary.get("velocity_gate_m_s") != key[4]
            or result.summary.get("confirmed_only") != key[6]
            or (
                tuple(result.summary["interval"])
                if result.summary.get("interval") is not None
                else None
            )
            != key[5]
            or len(result.time_s) != min(count, max_plot_points)
        ):
            return None
        event_names = [name for name in arrays if name.startswith(("event_", "cumulative_"))]
        if any(len(arrays[name]) != len(result.time_s) for name in event_names):
            return None
        if any(
            len(arrays[name]) != len(result.time_s)
            for name in ("confirmed_count", "tentative_count")
        ):
            return None
        for prefix, id_name in (("gt", "gt_target_ids"), ("output", "output_track_ids")):
            length = len(arrays[f"{prefix}_time_s"])
            if len(getattr(result, id_name)) != length or any(
                len(arrays[f"{prefix}_{suffix}"]) != length for suffix in ("range_m", "radial_m_s")
            ):
                return None
        if (
            len(result.tracking_time_s) != len(result.tracking_target_ids)
            or len(result.tracking_time_s) != len(result.tracking_track_ids)
            or len(result.range_errors_m) != len(result.radial_errors_m_s)
            or len(result.tracking_events) > max_plot_points
        ):
            return None
        result.summary["interval"] = interval
        return result
    except (TypeError, KeyError, ValueError):
        return None


def _save_summary(cache, key, result, cancelled):
    if cache is None:
        return
    metadata, arrays = {}, {}
    for descriptor in fields(result):
        value = getattr(result, descriptor.name)
        (arrays if isinstance(value, np.ndarray) else metadata)[descriptor.name] = value
    saved = save_component(
        cache, "run-summary", key, metadata, arrays, protocol=SUMMARY_PROTOCOL, cancelled=cancelled
    )
    if cache._building and not saved:
        raise OSError("Could not save the complete run summary")


class _Sample:
    """Deterministic reservoir; each plot family has a strict point budget."""

    def __init__(self, limit):
        self.limit, self.count, self.rows = limit, 0, []
        self.rng = random.Random(1729)

    def add(self, row):
        self.count += 1
        if len(self.rows) < self.limit:
            self.rows.append(row)
        else:
            index = self.rng.randrange(self.count)
            if index < self.limit:
                self.rows[index] = row


def _identity(sensing):
    temporal = sensing.get("temporal") or {}
    return (
        (temporal.get("clock") or {}).get("clock_id"),
        (temporal.get("result") or {}).get("processing_run_id"),
        (temporal.get("acquisition") or {}).get("pair_id"),
        (temporal.get("acquisition") or {}).get("acquisition_id"),
    )


def _finite(value):
    return (
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)
    )


def _track_id(value):
    if not _finite(value) or int(value) != value:
        return None
    return int(value)


def _ratio(a, b):
    return a / b if b else None


def load_run_summary(
    service,
    context,
    cancelled,
    *,
    range_gate=DEFAULT_RANGE_GATE_M,
    velocity_gate=DEFAULT_VELOCITY_GATE_M_S,
    interval=None,
    confirmed_only=False,
    max_plot_points=4096,
):
    """Score every retained identity in one clock/run/pair, optionally in [a,b].

    All event arrays share ``time_s``. Excluded events have NaN event metrics;
    cumulative totals carry forward without treating missing truth as a miss.
    Confirmed/tentative populations describe saved outputs, not hidden tracker
    states. Trajectories and paired signed errors are uniform bounded samples.
    Cancellation returns an empty cancelled result, never partial run totals.
    """
    context = tuple(context)
    if len(context) != 3 or not all(isinstance(x, str) and x for x in context):
        raise ValueError("Select one clock, processing run and sensor pair")
    if any(not _finite(g) or g <= 0 for g in (range_gate, velocity_gate)):
        raise ValueError("Association gates must be finite and positive")
    if type(max_plot_points) is not int or max_plot_points < 2:
        raise ValueError("max_plot_points must be an integer of at least two")
    if interval is not None and (
        len(interval) != 2 or not all(_finite(x) for x in interval) or interval[0] > interval[1]
    ):
        raise ValueError("interval must be finite inclusive [start, end]")

    def stopped():
        return RunSummaryData(context, cancelled=True, error="Cancelled")

    if cancelled.is_set():
        return stopped()
    unique = {}
    for storage_id, frame in service.dashboard_result_rows():
        if cancelled.is_set():
            return stopped()
        sensing = frame.get("sensing") or {}
        ident = _identity(sensing)
        if ident[:3] != context or not isinstance(ident[3], str) or not ident[3]:
            continue
        time = ((sensing.get("temporal") or {}).get("result") or {}).get("reference_time_s")
        if interval is not None and (not _finite(time) or not contains_time(time, *interval)):
            continue
        # Duplicate storage occurrences are one acquisition. Last occurrence wins,
        # matching summarize_events' identity policy; timestamps are not identities.
        unique[ident] = (storage_id, time)
    rows = sorted(unique.items(), key=lambda row: row[1][1] if _finite(row[1][1]) else -math.inf)
    n = len(rows)
    cache = getattr(service, "recording_cache", None)
    cache_key = summary_cache_key(
        context, range_gate, velocity_gate, interval, confirmed_only, max_plot_points
    )
    try:
        cached = _load_cached_summary(
            cache, cache_key, context, n, max_plot_points, interval, cancelled
        )
    except CancelledError:
        return stopped()
    if cached is not None:
        progress = getattr(service, "report_read_progress", None)
        if callable(progress):
            progress("Cached run summary", n, n)
        return cached
    selected = set(np.linspace(0, n - 1, min(n, max_plot_points), dtype=int)) if n else set()
    samples = {
        name: _Sample(max_plot_points)
        for name in ("errors", "gt", "outputs", "tracking", "switches")
    }
    event_rows, skipped, targets = [], Counter(), {}
    tp = fp = fn = evaluated = switches = 0
    velocity_matched = velocity_excluded_matched = velocity_unavailable = 0
    reference_protocols = set()
    has_explicit_velocity_support = False
    range_sq = radial_sq = 0.0
    identity_valid = True
    had_track_output = population_available = False
    last_track = {}
    uncertainty = [
        "All retained acquisitions; completeness against an expected schedule is unknown.",
        (
            "Confirmed output rows are evaluated."
            if confirmed_only
            else "All saved output rows are evaluated."
        ),
        "Range and radial velocity errors describe matched pairs only, not 3D accuracy.",
        "GT trajectories show evaluated truth; output trajectories include finite saved outputs even when truth is unavailable.",
    ]
    finite_times = [value[1] for _, value in rows if _finite(value[1])]
    if len(set(finite_times)) < len(finite_times):
        uncertainty.append(
            "Distinct acquisitions share timestamps; equal-time continuity follows saved inventory order."
        )
    for index, (ident, (storage_id, metadata_time)) in enumerate(rows):
        if cancelled.is_set():
            return stopped()
        progress = getattr(service, "report_read_progress", None)
        if callable(progress):
            progress("Loading run summary", index, n)
        frame = service.dashboard_load_row(storage_id, ("config_resolved",))
        if cancelled.is_set():
            return stopped()
        t = float(metadata_time) if _finite(metadata_time) else math.nan
        counts = [math.nan] * 5
        confirmed = tentative = math.nan
        reason = None
        event = None
        if frame is None:
            reason = "Saved acquisition could not be loaded"
        elif _identity(frame.get("sensing") or {}) != ident:
            reason = "Loaded acquisition identity differs from inventory"
        else:
            sensing = frame["sensing"]
            pose = frame
            # Root-translation references need the explicitly bound sensor poses.
            scene_id = ((sensing.get("temporal") or {}).get("display_binding") or {}).get(
                "scene_frame_id"
            )
            if (
                "tx_positions" not in frame or "rx_positions" not in frame
            ) and scene_id is not None:
                load_scene = getattr(service, "dashboard_scene_frame", None)
                if callable(load_scene):
                    pose = load_scene(scene_id) or frame
            event = evaluation_event(sensing, pose)
            t = float(event["time_s"]) if _finite(event.get("time_s")) else math.nan
            ds = event["detections"]
            if confirmed_only:
                ds = [d for d in ds if d.get("track_confirmed") is True]
            if event["tracking_enabled"] and all(
                type(d.get("track_confirmed")) is bool for d in event["detections"]
            ):
                confirmed = sum(d["track_confirmed"] for d in ds)
                tentative = len(ds) - confirmed
                population_available = True
            for d in ds:
                if _finite(t) and _finite(d.get("range_m")) and _finite(d.get("velocity_m_s")):
                    samples["outputs"].add(
                        (t, d["range_m"], d["velocity_m_s"], _track_id(d.get("track_id")))
                    )
            reason = event.get("reason")
            try:
                matched = match_event(
                    event, range_gate, velocity_gate, confirmed_only=confirmed_only
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                matched = None
                reason = f"Invalid evaluation evidence: {exc}"
            if matched is None:
                if confirmed_only and (
                    not event["tracking_enabled"]
                    or any("track_confirmed" not in d for d in event["detections"])
                ):
                    reason = reason or "Track confirmation was not recorded"
                reason = reason or "Saved output range/radial velocity unavailable"
            else:
                truth, ds, assignment = matched
                tids = truth["target_ids"]
                mask = evaluation_velocity_mask(truth)
                has_explicit_velocity_support |= mask is not None
                velocity_valid = [True] * len(tids) if mask is None else mask
                reference_protocols.add(
                    truth.get("evaluation_reference_protocol", LEGACY_REFERENCE_PROTOCOL)
                )
                ids = [_track_id(d.get("track_id")) for d in ds]
                valid_ids = all(i is not None for i in ids) and len(set(ids)) == len(ids)
                valid_targets = all(isinstance(tid, str) and tid for tid in tids) and len(
                    set(tids)
                ) == len(tids)
                identity_valid &= valid_ids and valid_targets and all(velocity_valid)
                had_track_output |= bool(ds) and valid_ids
                if not valid_ids or not valid_targets:
                    last_track.clear()  # Never bridge unverifiable identity evidence.
                evaluated += 1
                etp, efp, efn = (
                    assignment.true_positives,
                    assignment.false_positives,
                    assignment.false_negatives,
                )
                tp += etp
                fp += efp
                fn += efn
                counts = [etp, efp, efn, len(tids), len(ds)]
                refs = truth.get("target_velocity_references", [RECORDED_REFERENCE] * len(tids))
                for gi, tid in enumerate(tids):
                    gr, gv = truth["target_ranges"][gi], truth["target_speeds"][gi]
                    samples["gt"].add((t, gr, gv, tid))
                    target = targets.setdefault(
                        tid,
                        dict(
                            eligible_count=0,
                            matched_count=0,
                            range_sq=0.0,
                            radial_sq=0.0,
                            references=set(),
                            tracks=set(),
                            velocity_matched=0,
                            velocity_excluded_matched=0,
                            velocity_unavailable_opportunities=0,
                        ),
                    )
                    target["eligible_count"] += 1
                    target["velocity_unavailable_opportunities"] += int(not velocity_valid[gi])
                    velocity_unavailable += int(not velocity_valid[gi])
                    target["references"].add(refs[gi])
                    di = assignment.gt_to_det.get(gi)
                    if di is None:
                        continue
                    d = ds[di]
                    er = float(d["range_m"] - gr)
                    ev = float(d["velocity_m_s"] - gv) if velocity_valid[gi] else math.nan
                    range_sq += er * er
                    target["range_sq"] += er * er
                    if velocity_valid[gi]:
                        radial_sq += ev * ev
                        target["radial_sq"] += ev * ev
                        target["velocity_matched"] += 1
                        velocity_matched += 1
                    else:
                        target["velocity_excluded_matched"] += 1
                        velocity_excluded_matched += 1
                    target["matched_count"] += 1
                    samples["errors"].add((er, ev))
                    if valid_ids and valid_targets:
                        track = ids[di]
                        target["tracks"].add(track)
                        samples["tracking"].add((t, tid, track))
                        if tid in last_track and last_track[tid] != track:
                            switches += 1
                            samples["switches"].add(
                                dict(
                                    time_s=t,
                                    target_id=tid,
                                    previous_track_id=last_track[tid],
                                    track_id=track,
                                )
                            )
                        last_track[tid] = track
        if reason:
            skipped[reason] += 1
        if index in selected:
            event_rows.append(
                (
                    t,
                    *counts,
                    tp,
                    fp,
                    fn,
                    _ratio(tp, tp + fn) if not reason else math.nan,
                    _ratio(tp, tp + fp) if not reason else math.nan,
                    confirmed,
                    tentative,
                )
            )
    if cancelled.is_set():
        return stopped()
    per_target = {}
    for tid, row in targets.items():
        m, e = row["matched_count"], row["eligible_count"]
        per_target[tid] = dict(
            eligible_count=e,
            matched_count=m,
            opportunity_count=e,
            detection_count=m,
            recall=_ratio(m, e),
            pd=_ratio(m, e),
            range_rmse=math.sqrt(row["range_sq"] / m) if m else None,
            velocity_rmse=(
                math.sqrt(row["radial_sq"] / row["velocity_matched"])
                if row["velocity_matched"]
                else None
            ),
            velocity_matched=row["velocity_matched"],
            velocity_excluded_matched=row["velocity_excluded_matched"],
            velocity_unavailable_opportunities=row["velocity_unavailable_opportunities"],
            velocity_reference=" / ".join(sorted(row["references"])),
            distinct_assigned_ids=(
                len(row["tracks"]) if identity_valid and had_track_output else None
            ),
        )
    trustworthy = identity_valid and had_track_output
    if not trustworthy:
        uncertainty.append(
            "Identity continuity unavailable: valid unique saved track IDs are required in every evaluated event."
        )
    summary = dict(
        total_tp=tp,
        total_fp=fp,
        total_fn=fn,
        target_opportunity_count=tp + fn,
        matched_pair_count=tp,
        evaluated_event_count=evaluated,
        cumulative_pd=_ratio(tp, tp + fn),
        recall=_ratio(tp, tp + fn),
        precision=_ratio(tp, tp + fp),
        mean_false_alarms_per_evaluated_event=_ratio(fp, evaluated),
        cumulative_range_rmse=math.sqrt(range_sq / tp) if tp else None,
        cumulative_velocity_rmse=(
            math.sqrt(radial_sq / velocity_matched) if velocity_matched else None
        ),
        velocity_matched_pair_count=velocity_matched,
        velocity_excluded_matched_pair_count=velocity_excluded_matched,
        velocity_unavailable_opportunity_count=velocity_unavailable,
        evaluation_reference_protocol=(
            INSTANTANEOUS_REFERENCE_PROTOCOL
            if has_explicit_velocity_support
            else LEGACY_REFERENCE_PROTOCOL
        ),
        truth_reference_protocols=sorted(reference_protocols),
        per_target=per_target,
        skipped=dict(skipped),
        id_switches=switches if trustworthy else None,
        reason=None if evaluated else "; ".join(skipped) or "No saved results to evaluate",
        range_gate_m=range_gate,
        velocity_gate_m_s=velocity_gate,
        interval=interval,
        confirmed_only=confirmed_only,
        excluded_event_count=n - evaluated,
    )
    arrays = {}
    names = (
        "time_s",
        "event_tp",
        "event_fp",
        "event_fn",
        "event_gt_count",
        "event_output_count",
        "cumulative_tp",
        "cumulative_fp",
        "cumulative_fn",
        "cumulative_recall",
        "cumulative_precision",
        "confirmed_count",
        "tentative_count",
    )
    for col, name in enumerate(names):
        arrays[name] = np.asarray([r[col] for r in event_rows], dtype=float)
    for family, prefix, id_name in (
        ("gt", "gt", "gt_target_ids"),
        ("outputs", "output", "output_track_ids"),
    ):
        records = sorted(samples[family].rows, key=lambda row: row[0])
        for col, suffix in enumerate(("time_s", "range_m", "radial_m_s")):
            arrays[f"{prefix}_{suffix}"] = np.asarray([r[col] for r in records], dtype=float)
        arrays[id_name] = tuple(r[3] for r in records)
    errors = samples["errors"].rows
    arrays["range_errors_m"] = np.asarray([r[0] for r in errors], dtype=float)
    arrays["radial_errors_m_s"] = np.asarray([r[1] for r in errors], dtype=float)
    tracking = sorted(samples["tracking"].rows, key=lambda row: row[0]) if trustworthy else []
    arrays.update(
        tracking_time_s=np.asarray([r[0] for r in tracking], dtype=float),
        tracking_target_ids=tuple(r[1] for r in tracking),
        tracking_track_ids=tuple(r[2] for r in tracking),
        tracking_events=(
            tuple(sorted(samples["switches"].rows, key=lambda r: r["time_s"]))
            if trustworthy
            else ()
        ),
    )
    sampled = n > max_plot_points or any(s.count > max_plot_points for s in samples.values())
    if sampled:
        uncertainty.append(
            "Plot points are bounded samples; scalar counts and pooled RMSE include every evaluated retained acquisition."
        )
    result = RunSummaryData(
        context,
        summary=summary,
        acquisition_count=n,
        evaluated_count=evaluated,
        plot_sampled=sampled,
        uncertainty=tuple(uncertainty),
        tracking_population_available=population_available,
        **arrays,
    )
    try:
        _save_summary(cache, cache_key, result, cancelled)
    except CancelledError:
        return stopped()
    return result
