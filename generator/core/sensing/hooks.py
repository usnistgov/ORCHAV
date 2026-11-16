"""Sensing bookkeeping helpers used by the file pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Integral
from typing import Optional

import numpy as np

from shared.sensing.delay import append_delay_domain_samples, extract_delay_domain_samples
from shared.sensing.signal_processing import GTEntry, compute_gt_entry, extract_target_positions
from shared.sensing.targets import INSTANTANEOUS_VELOCITY_REFERENCE_POLICY


def accumulate_sensing_gt(
    frame_data: dict,
    range_mode: str,
    dt: float | None,
    prev_target_ranges: Optional[Mapping[str, float]],
    sensing_data: Optional[dict] = None,
    step: Optional[int] = None,
    reference_time_s: float | None = None,
    *,
    velocity_reference_policy: str = "legacy",
) -> dict:
    """Extract per-frame GT ranges/speeds and detections for sensing summary.

    Delegates to :func:`shared.sensing.signal_processing.compute_gt_entry` for the actual math.
    Returns target ranges, explicitly named Doppler-comparable truth
    observables, and detection values. ``target_speeds`` remains a
    transitional alias for monostatic-equivalent velocity.

    Args:
        frame_data: Frame dict with positions and (optionally) sensing.
        range_mode: ``"monostatic"`` or ``"bistatic"``.
        dt: Exact elapsed time from the preceding retained truth reference
            epoch, or ``None`` when no preceding epoch exists.
        prev_target_ranges: Previous frame's target ranges (for speed calc).
        sensing_data: Explicit sensing dict.  When provided, detections are
            read from here instead of ``frame_data["sensing"]``.  This is
            needed in CPI mode where the sensing dict is buffered separately
            from the frame that carries the target positions.
    """
    import numpy as np

    tx_pos = frame_data.get("tx_positions_snapshot")
    rx_pos = frame_data.get("rx_positions_snapshot")
    tgt_pos = extract_target_positions(frame_data)
    tgt_vel = frame_data.get("target_velocities_snapshot")

    sensing = sensing_data or frame_data.get("sensing", {}) or {}
    detections = sensing.get("detections") or []

    reference_kwargs = {}
    tx_index = rx_index = 0
    if velocity_reference_policy != "legacy":
        tx_index, rx_index, pair_id = _reference_sensor_pair(frame_data, sensing)
        reference_kwargs = {
            "velocity_reference_policy": velocity_reference_policy,
            "pair_id": pair_id,
            "tx_velocity_m_s": _sensor_velocity(frame_data, "tx", tx_index),
            "rx_velocity_m_s": _sensor_velocity(frame_data, "rx", rx_index),
        }

    if tgt_pos is None:
        entry = compute_gt_entry(
            np.zeros(3),
            np.zeros(3),
            [],
            range_mode=range_mode,
            dt=dt,
            prev_target_ranges=None,
            detections=detections,
            target_ids=[],
            reference_time_s=reference_time_s,
            **reference_kwargs,
        )
        return _gt_entry_dict(entry, step=step, policy=velocity_reference_policy, sensing=sensing)

    if len(tgt_pos) > 0 and (
        tx_pos is None or rx_pos is None or len(tx_pos) == 0 or len(rx_pos) == 0
    ):
        raise ValueError("Sensing target truth requires exact TX and RX positions")

    tx = np.asarray(tx_pos[tx_index]) if len(tx_pos) > 0 else np.zeros(3)
    rx = np.asarray(rx_pos[rx_index]) if len(rx_pos) > 0 else np.zeros(3)

    target_ids = _target_ids(frame_data, len(tgt_pos))
    prior_ranges = (
        None
        if prev_target_ranges is None
        else [
            (float(prev_target_ranges[target_id]) if target_id in prev_target_ranges else None)
            for target_id in target_ids
        ]
    )

    entry = compute_gt_entry(
        tx,
        rx,
        list(tgt_pos),
        range_mode=range_mode,
        dt=dt,
        prev_target_ranges=prior_ranges,
        detections=detections,
        target_velocities=tgt_vel,
        target_ids=target_ids,
        reference_time_s=reference_time_s,
        **reference_kwargs,
    )
    return _gt_entry_dict(entry, step=step, policy=velocity_reference_policy, sensing=sensing)


def _reference_sensor_pair(frame_data: dict, sensing: dict) -> tuple[int, int, str]:
    """Resolve the acquisition's exact RX-major/TX-minor sensor rows."""
    from shared.extensions.sensing import canonical_sensing_pair_id

    temporal = sensing.get("temporal") or {}
    acquisition = temporal.get("acquisition") or {}
    pair_index = sensing.get("pair_index")
    tx_devices = frame_data.get("tx_list") or []
    rx_devices = frame_data.get("rx_list") or []
    if (
        not isinstance(pair_index, Integral)
        or isinstance(pair_index, bool)
        or not tx_devices
        or not rx_devices
        or not 0 <= pair_index < len(tx_devices) * len(rx_devices)
    ):
        raise ValueError("Instantaneous truth requires an exact acquisition sensor pair")
    tx_index = int(pair_index) % len(tx_devices)
    rx_index = int(pair_index) // len(tx_devices)
    tx_id = getattr(tx_devices[tx_index], "name", None)
    rx_id = getattr(rx_devices[rx_index], "name", None)
    tx_id = tx_id.strip() if isinstance(tx_id, str) else tx_id
    rx_id = rx_id.strip() if isinstance(rx_id, str) else rx_id
    pair_id = canonical_sensing_pair_id(tx_id, rx_id)
    if (
        acquisition.get("pair_id") != pair_id
        or acquisition.get("tx_sensor_id") != tx_id
        or acquisition.get("rx_sensor_id") != rx_id
    ):
        raise ValueError("Instantaneous truth sensor pair disagrees with acquisition identity")
    for role, devices in (("tx", tx_devices), ("rx", rx_devices)):
        positions = np.asarray(frame_data.get(f"{role}_positions_snapshot"), dtype=float)
        if positions.shape != (len(devices), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"Instantaneous truth requires exact finite {role.upper()} positions")
    return tx_index, rx_index, pair_id


def _sensor_velocity(frame_data: dict, role: str, index: int) -> np.ndarray | None:
    """Return known sensor velocity, preserving absent state as unavailable."""
    values = frame_data.get(f"{role}_velocities_snapshot")
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (len(frame_data[f"{role}_list"]), 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"Instantaneous truth requires aligned finite {role.upper()} velocities")
    return array[index]


def _gt_entry_dict(
    entry: GTEntry, *, step: Optional[int], policy: str = "legacy", sensing: dict | None = None
) -> dict:
    """Serialize one :class:`GTEntry` without dropping validity evidence."""

    result = {
        "target_ids": entry.target_ids,
        "reference_time_s": entry.reference_time_s,
        "target_ranges": entry.target_ranges,
        "target_speeds": entry.target_speeds,
        "target_speed_semantics": entry.target_speed_semantics,
        "target_path_closing_rates_m_s": entry.target_path_closing_rates_m_s,
        "target_monostatic_equivalent_velocities_m_s": (
            entry.target_monostatic_equivalent_velocities_m_s
        ),
        "target_bistatic_bisector_velocities_m_s": (entry.target_bistatic_bisector_velocities_m_s),
        "target_velocity_valid": entry.target_velocity_valid,
        "target_velocity_sources": entry.target_velocity_sources,
        "target_velocity_reference_dt_s": entry.target_velocity_reference_dt_s,
        "target_velocity_unavailable_reasons": entry.target_velocity_unavailable_reasons,
        "det_ranges": entry.det_ranges,
        "det_velocities": entry.det_velocities,
    }
    if step is not None:
        result["step"] = int(step)
    if policy != "legacy":
        result["velocity_reference_policy"] = policy
        result["target_velocity_reference_evidence"] = entry.target_velocity_reference_evidence
        temporal_result = ((sensing or {}).get("temporal") or {}).get("result")
        result["evaluation_reference_valid"] = (
            isinstance(temporal_result, dict)
            and temporal_result.get("reference_time_s") == entry.reference_time_s
        )
    return result


def _target_ids(frame_data: dict, target_count: int) -> list[str]:
    """Bind positional runtime rows to exact authored target identities once."""

    internal_ids = frame_data.get("_sensing_target_ids")
    if isinstance(internal_ids, (list, tuple)):
        ids = list(internal_ids)
    else:
        metadata = frame_data.get("targets_metadata") or []
        ids = [
            metadata[index].get("name") if isinstance(metadata[index], dict) else None
            for index in range(min(len(metadata), target_count))
        ]
    if len(ids) != target_count or any(
        not isinstance(target_id, str)
        or not target_id
        or re.fullmatch(r"target_\d+", target_id) is not None
        for target_id in ids
    ):
        raise ValueError(
            "Sensing target truth requires one exact authored target ID per runtime row"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("Sensing target truth IDs must be unique")
    return ids


def build_sensing_gt_channel(frame_data: dict, gt_entry: dict) -> dict:
    """Build canonical ID-keyed evaluation truth without positional aliases."""
    target_ids = gt_entry.get("target_ids")
    target_ranges = [float(v) for v in gt_entry.get("target_ranges", [])]
    target_speeds = list(gt_entry.get("target_speeds", []))
    path_closing_rates = list(gt_entry.get("target_path_closing_rates_m_s", []))
    monostatic_equivalent_velocities = list(
        gt_entry.get("target_monostatic_equivalent_velocities_m_s", [])
    )
    bistatic_bisector_velocities = list(gt_entry.get("target_bistatic_bisector_velocities_m_s", []))
    velocity_valid = list(gt_entry.get("target_velocity_valid", []))
    velocity_sources = list(gt_entry.get("target_velocity_sources", []))
    velocity_reference_dt_s = list(gt_entry.get("target_velocity_reference_dt_s", []))
    velocity_unavailable_reasons = list(gt_entry.get("target_velocity_unavailable_reasons", []))
    reference_policy = gt_entry.get("velocity_reference_policy")
    reference_evidence = gt_entry.get("target_velocity_reference_evidence")
    if reference_policy is not None:
        if reference_policy != INSTANTANEOUS_VELOCITY_REFERENCE_POLICY:
            raise ValueError("Unknown canonical truth velocity reference policy")
        if not isinstance(reference_evidence, list) or len(reference_evidence) != len(
            target_ranges
        ):
            raise ValueError("Canonical velocity reference evidence must align by target ID")

    targets = []
    if not isinstance(target_ids, list) or len(target_ids) != len(target_ranges):
        raise ValueError("Canonical sensing truth requires target_ids aligned at its source")
    if any(not np.isfinite(value) or value < 0.0 for value in target_ranges):
        raise ValueError("Canonical sensing truth ranges must be finite and non-negative")
    expected_ids = _target_ids(frame_data, len(target_ranges))
    if target_ids != expected_ids:
        raise ValueError("Canonical sensing truth target IDs changed after source-row binding")
    aligned_fields = {
        "target_speeds": target_speeds,
        "target_path_closing_rates_m_s": path_closing_rates,
        "target_monostatic_equivalent_velocities_m_s": monostatic_equivalent_velocities,
        "target_bistatic_bisector_velocities_m_s": bistatic_bisector_velocities,
        "target_velocity_valid": velocity_valid,
        "target_velocity_sources": velocity_sources,
        "target_velocity_reference_dt_s": velocity_reference_dt_s,
        "target_velocity_unavailable_reasons": velocity_unavailable_reasons,
    }
    for field_name, values in aligned_fields.items():
        if len(values) != len(target_ranges):
            raise ValueError(f"Canonical sensing truth {field_name} must align by target ID")
    for idx, target_id in enumerate(target_ids):
        valid = velocity_valid[idx]
        if not isinstance(valid, (bool, np.bool_)):
            raise ValueError("Canonical sensing truth velocity validity must be boolean")
        source = velocity_sources[idx]
        if source not in {
            "exact_epoch_finite_difference",
            "analytic_velocity_projection",
            "unavailable",
        }:
            raise ValueError("Canonical sensing truth has an unknown velocity derivation source")
        if valid:
            if source == "unavailable":
                raise ValueError("Canonical sensing truth valid velocity requires a valid source")
            path_closing_rate = float(path_closing_rates[idx])
            monostatic_velocity = float(monostatic_equivalent_velocities[idx])
            speed_m_s = float(target_speeds[idx])
            if not np.all(np.isfinite([path_closing_rate, monostatic_velocity, speed_m_s])):
                raise ValueError("Canonical sensing truth valid velocities must be finite")
            reference_dt_s = velocity_reference_dt_s[idx]
            if source == "exact_epoch_finite_difference":
                reference_dt_s = float(reference_dt_s)
                if not np.isfinite(reference_dt_s) or reference_dt_s <= 0.0:
                    raise ValueError(
                        "Finite-difference truth velocity requires a positive exact epoch delta"
                    )
            elif reference_dt_s is not None:
                raise ValueError("Analytic truth velocity must not claim a finite-difference dt")
            unavailable_reason = None
            if velocity_unavailable_reasons[idx] is not None:
                raise ValueError(
                    "Canonical sensing truth valid velocity cannot have an error reason"
                )
        else:
            if source != "unavailable":
                raise ValueError(
                    "Canonical sensing truth invalid velocity source must be unavailable"
                )
            if any(
                value is not None
                for value in (
                    path_closing_rates[idx],
                    monostatic_equivalent_velocities[idx],
                    target_speeds[idx],
                )
            ):
                raise ValueError("Canonical sensing truth unavailable velocities must be null")
            path_closing_rate = None
            monostatic_velocity = None
            speed_m_s = None
            unavailable_reason = velocity_unavailable_reasons[idx]
            if not isinstance(unavailable_reason, str) or not unavailable_reason:
                raise ValueError("Canonical sensing truth unavailable velocity requires a reason")
            reference_dt_s = velocity_reference_dt_s[idx]
            if reference_dt_s is not None:
                raise ValueError("Unavailable truth velocity must not claim a reference dt")
        bisector_velocity = bistatic_bisector_velocities[idx]
        if bisector_velocity is not None:
            bisector_velocity = float(bisector_velocity)
            if not np.isfinite(bisector_velocity):
                raise ValueError("Canonical sensing truth bisector velocity must be finite or null")
        if not valid and bisector_velocity is not None:
            raise ValueError("Unavailable truth velocity cannot have a bisector velocity")
        targets.append(
            {
                "target_id": target_id,
                "range_m": target_ranges[idx],
                "path_closing_rate_m_s": path_closing_rate,
                "monostatic_equivalent_velocity_m_s": monostatic_velocity,
                "bistatic_bisector_velocity_m_s": bisector_velocity,
                "velocity_valid": bool(valid),
                "velocity_derivation_source": source,
                "velocity_reference_dt_s": reference_dt_s,
                "velocity_unavailable_reason": unavailable_reason,
            }
        )
        if reference_policy is not None:
            targets[-1]["velocity_reference_evidence"] = reference_evidence[idx]

    result = {"targets": targets}
    if reference_policy is not None:
        result["velocity_reference_policy"] = reference_policy
    return result


def collect_fast_time_samples(
    sensing_data: dict,
    fast_time_samples: list,
    sample_times_s: Optional[list[float]] = None,
) -> None:
    """Collect canonical typed delay-domain rows for optional summaries.

    Product descriptors must already be materialized. Exact producer-authored
    time coordinates are retained, and overlapping slow-time windows are
    de-duplicated only when their repeated rows agree exactly.
    """

    if sample_times_s is None:
        raise ValueError("Sensing summary delay-domain collection requires exact timestamps")
    product = extract_delay_domain_samples(sensing_data)
    if product is None:
        return
    append_delay_domain_samples(product, fast_time_samples, sample_times_s)


def compute_frame_association(
    gt_entry: dict,
    *,
    range_gate_m: float = 10.0,
    velocity_gate_m_s: float = 5.0,
) -> dict:
    """Compute TP/FP/FN association for one frame's detections vs GT.

    Uses :func:`shared.sensing.evaluation.associate_detections_to_gt` if
    available, otherwise falls back to a simple nearest-neighbor heuristic.

    Returns dict with keys: tp, fp, fn, range_rmse, velocity_rmse. RMSE
    values are ``None`` when there are no matched pairs.
    Association gates are configurable via ``range_gate_m`` and
    ``velocity_gate_m_s``.
    """
    import numpy as np

    gt_ranges = np.array(gt_entry.get("target_ranges", []), dtype=np.float64)
    gt_speeds = np.array(gt_entry.get("target_speeds", []), dtype=np.float64)
    det_ranges = np.array(gt_entry.get("det_ranges", []), dtype=np.float64)
    det_speeds = np.array(gt_entry.get("det_velocities", []), dtype=np.float64)

    n_gt = len(gt_ranges)
    n_det = len(det_ranges)

    new_policy = (
        gt_entry.get("velocity_reference_policy") == INSTANTANEOUS_VELOCITY_REFERENCE_POLICY
    )
    if new_policy:
        if not gt_entry.get("evaluation_reference_valid"):
            return {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "range_rmse": None,
                "velocity_rmse": None,
                "evaluated": False,
                "skipped_reason": "truth_result_epoch_mismatch",
                "truth_targets_skipped": n_gt,
                "detections_skipped": n_det,
                "evaluation_reference_protocol": "instantaneous_velocity_support_v1",
            }
        evidence = gt_entry.get("target_velocity_reference_evidence")
        if not isinstance(evidence, list) or len(evidence) != n_gt:
            raise ValueError(
                "Instantaneous evaluation requires aligned velocity reference evidence"
            )
        from shared.sensing.evaluation import associate_detections_to_gt

        result = associate_detections_to_gt(
            gt_ranges,
            gt_speeds,
            det_ranges,
            det_speeds,
            range_gate=range_gate_m,
            velocity_gate=velocity_gate_m_s,
            gt_velocity_valid=[
                row["instantaneous_eligible"] and bool(valid)
                for row, valid in zip(evidence, gt_entry["target_velocity_valid"], strict=True)
            ],
        )
        return {
            "tp": result.true_positives,
            "fp": result.false_positives,
            "fn": result.false_negatives,
            "range_rmse": result.range_rmse,
            "velocity_rmse": result.velocity_rmse,
            "velocity_matched_pair_count": result.velocity_matched_pair_count,
            "velocity_excluded_matched_pair_count": result.velocity_excluded_matched_pair_count,
            "velocity_unavailable_opportunity_count": result.velocity_unavailable_opportunity_count,
            "evaluation_reference_protocol": result.evaluation_reference_protocol,
        }

    velocity_valid = gt_entry.get("target_velocity_valid")
    if n_gt > 0 and (
        not isinstance(velocity_valid, list)
        or len(velocity_valid) != n_gt
        or not all(isinstance(value, (bool, np.bool_)) for value in velocity_valid)
        or not all(bool(value) for value in velocity_valid)
    ):
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "range_rmse": None,
            "velocity_rmse": None,
            "evaluated": False,
            "skipped_reason": "truth_velocity_unavailable",
            "truth_targets_skipped": n_gt,
            "detections_skipped": n_det,
        }

    if n_gt == 0 and n_det == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "range_rmse": None, "velocity_rmse": None}

    try:
        from shared.sensing.evaluation import associate_detections_to_gt

        result = associate_detections_to_gt(
            gt_ranges,
            gt_speeds,
            det_ranges,
            det_speeds,
            range_gate=range_gate_m,
            velocity_gate=velocity_gate_m_s,
        )
        return {
            "tp": result.true_positives,
            "fp": result.false_positives,
            "fn": result.false_negatives,
            "range_rmse": result.range_rmse,
            "velocity_rmse": result.velocity_rmse,
        }
    except ImportError:
        pass

    # Fallback: simple gated nearest-neighbor
    matched_gt = set()
    matched_det = set()
    range_errors = []
    vel_errors = []
    for di in range(n_det):
        best_gi, best_dist = -1, float("inf")
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            range_dist = abs(det_ranges[di] - gt_ranges[gi])
            velocity_dist = abs(det_speeds[di] - gt_speeds[gi])
            if (
                range_dist < best_dist
                and range_dist < range_gate_m
                and velocity_dist < velocity_gate_m_s
            ):
                best_dist = range_dist
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched_det.add(di)
            range_errors.append(det_ranges[di] - gt_ranges[best_gi])
            vel_errors.append(det_speeds[di] - gt_speeds[best_gi])

    tp = len(matched_gt)
    fp = n_det - tp
    fn = n_gt - tp
    r_rmse = float(np.sqrt(np.mean(np.array(range_errors) ** 2))) if range_errors else None
    v_rmse = float(np.sqrt(np.mean(np.array(vel_errors) ** 2))) if vel_errors else None
    return {"tp": tp, "fp": fp, "fn": fn, "range_rmse": r_rmse, "velocity_rmse": v_rmse}
