"""Display/evaluation references selected from recorded motion-model evidence.

Canonical truth remains untouched. Explicit relative-kinematics evidence binds
velocity scoring to the result epoch. Frozen-pose root translation can provide
a distinct bulk reference. Undeclared recordings retain their recorded policy.
"""

import math
from collections.abc import Mapping

import numpy as np

from shared.sensing.evaluation import (
    INSTANTANEOUS_REFERENCE_PROTOCOL,
    LEGACY_REFERENCE_PROTOCOL,
)
from shared.sensing.gt import extract_stored_gt_entry
from shared.sensing.signal_processing import bistatic_path_closing_rate_m_s
from shared.sensing.targets import (
    INSTANTANEOUS_VELOCITY_REFERENCE_POLICY,
    validate_velocity_reference_evidence,
)

SIGNAL_TRUTH_VERSION = "signal-truth-v3-relative-kinematics"
RECORDED_REFERENCE = "Recorded target-center velocity"
ROOT_REFERENCE = "Root projection (fixed radar)"
INSTANTANEOUS_REFERENCE = "Instantaneous relative velocity"
ROOT_RELATIVE_REFERENCE = "Instantaneous root relative velocity"
INTERVAL_REFERENCE = "Recorded interval mean; instantaneous velocity unavailable"


def evaluation_velocity_mask(truth):
    """Return a strict support mask only for explicitly versioned truth.

    Legacy consumers keep their existing all-or-nothing range/velocity behavior.
    New-policy unavailable velocities retain range-only evaluation opportunities.
    """
    if truth.get("evaluation_reference_protocol") == INSTANTANEOUS_REFERENCE_PROTOCOL:
        values = truth.get("target_velocity_valid")
    elif "target_velocity_reference_evidence" in truth:
        values = truth.get("target_velocity_instantaneous_eligible")
    else:
        return None
    if (
        not isinstance(values, list)
        or len(values) != len(truth["target_ids"])
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError("Instantaneous evaluation requires aligned explicit velocity validity")
    return values


def extract_recorded_evaluation_truth(sensing, frame=None):
    """Select new-policy result references while preserving old loader semantics.

    Incomplete acquisitions retain raw truth for processing. A lifecycle fallback
    at a different epoch remains explicit unavailable evaluation, not empty truth.
    """
    raw = extract_stored_gt_entry(sensing)
    if raw is None or evaluation_velocity_mask(raw) is None:
        return raw
    temporal = sensing.get("temporal") or {}
    result_epoch = (temporal.get("result") or {}).get("reference_time_s")
    if result_epoch is None:
        return raw
    if (temporal.get("truth") or {}).get("reference_time_s") != result_epoch:
        raw["_evaluation_unavailable_reason"] = "Truth and result reference epochs differ"
        return raw
    return extract_evaluation_truth(sensing, frame)


def _root_state(target, declaration):
    state = target.get("animated_asset")
    if not isinstance(state, Mapping) or any(
        not state.get(key) or state.get(key) != declaration.get(key)
        for key in ("binding_id", "asset_revision_sha256")
    ):
        raise ValueError("Root velocity evidence does not match the animated asset")
    return state


def _frozen_root_declaration(sensing, target_id):
    acquisition = (sensing.get("config_resolved") or {}).get("acquisition") or {}
    if acquisition.get("sample_source") != "coherent_one_solve":
        return None
    for row in (sensing.get("target_catalog") or {}).get("targets", []):
        if row.get("target_id") == target_id:
            declaration = row.get("animated_asset") or {}
            if (
                declaration.get("rt_velocity_source") == "piecewise_linear_root_translation"
                and declaration.get("coherent_geometry_policy") == "frozen_at_solve_epoch"
            ):
                return declaration
    return None


def _unavailable(truth, index, reason):
    truth["target_speeds"][index] = None
    truth["target_velocity_valid"][index] = False
    truth["target_velocity_sources"][index] = "unavailable"
    truth["target_velocity_reference_dt_s"][index] = None
    truth["target_velocity_unavailable_reasons"][index] = reason


def _instantaneous_truth(sensing, truth):
    temporal = sensing.get("temporal") or {}
    epoch = (temporal.get("truth") or {}).get("reference_time_s")
    result_epoch = (temporal.get("result") or {}).get("reference_time_s")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, (int, float))
        or not math.isfinite(epoch)
        or isinstance(result_epoch, bool)
        or not isinstance(result_epoch, (int, float))
        or not math.isfinite(result_epoch)
        or epoch != result_epoch
    ):
        raise ValueError("Instantaneous truth must match the exact result reference epoch")
    pair_id = (temporal.get("acquisition") or {}).get("pair_id")
    evidence = truth.get("target_velocity_reference_evidence", [])
    if len(evidence) != len(truth["target_ids"]):
        raise ValueError("Instantaneous truth evidence must align with every active target")
    targets = {row["target_id"]: row for row in sensing["truth"]["targets"]}
    truth["evaluation_reference_protocol"] = INSTANTANEOUS_REFERENCE_PROTOCOL
    truth["velocity_reference_policy"] = INSTANTANEOUS_VELOCITY_REFERENCE_POLICY
    truth["recorded_target_velocity_sources"] = list(truth["target_velocity_sources"])
    truth["recorded_target_velocity_valid"] = list(truth["target_velocity_valid"])
    truth["recorded_target_velocity_reference_dt_s"] = list(truth["target_velocity_reference_dt_s"])
    for i, (target_id, item) in enumerate(zip(truth["target_ids"], evidence, strict=True)):
        errors = validate_velocity_reference_evidence(
            item,
            pair_id=pair_id,
            target_id=target_id,
            evaluation_epoch_s=epoch,
            velocity_source=truth["target_velocity_sources"][i],
            velocity_valid=truth["target_velocity_valid"][i],
        )
        if errors:
            raise ValueError("; ".join(errors))
        kind = item["support"]["kind"]
        truth["target_velocity_references"][i] = INSTANTANEOUS_REFERENCE
        if kind != "instantaneous":
            if kind == "interval_mean":
                truth["target_velocity_references"][i] = INTERVAL_REFERENCE
            _unavailable(truth, i, "Instantaneous velocity unavailable: " + item["fallback_reason"])
            continue
        value = 0.5 * bistatic_path_closing_rate_m_s(
            item["tx_position_m"],
            item["rx_position_m"],
            item["target_position_m"],
            item["target_velocity_m_s"],
            tx_velocity_m_s=item["tx_velocity_m_s"],
            rx_velocity_m_s=item["rx_velocity_m_s"],
        )
        if not math.isclose(
            value,
            truth["recorded_target_speeds"][i],
            rel_tol=32 * np.finfo(float).eps,
            abs_tol=32 * np.finfo(float).eps,
        ):
            raise ValueError(
                "Recorded instantaneous velocity disagrees with its kinematic evidence"
            )
        declaration = _frozen_root_declaration(sensing, target_id)
        if declaration is not None:
            truth["target_velocity_references"][i] = ROOT_RELATIVE_REFERENCE
            try:
                state = _root_state(targets[target_id], declaration)
                value = 0.5 * bistatic_path_closing_rate_m_s(
                    item["tx_position_m"],
                    item["rx_position_m"],
                    item["target_position_m"],
                    state.get("root_velocity_m_s"),
                    tx_velocity_m_s=item["tx_velocity_m_s"],
                    rx_velocity_m_s=item["rx_velocity_m_s"],
                )
            except (ValueError, TypeError, KeyError) as exc:
                _unavailable(truth, i, str(exc))
                continue
            truth["target_velocity_sources"][i] = "recorded_root_relative_velocity_projection"
        truth["target_speeds"][i] = value
        truth["target_velocity_valid"][i] = True
        truth["target_velocity_reference_dt_s"][i] = None
        truth["target_velocity_unavailable_reasons"][i] = None
    return truth


def _pair_positions(sensing, frame):
    """Read the explicitly bound pair's poses, never the first available radar."""
    pairs = sensing.get("pair_axis")
    if not isinstance(pairs, list) or len(pairs) != 1 or frame is None:
        raise ValueError("Root velocity reference requires the recorded radar pair pose")
    pair = pairs[0]
    acquisition = sensing["temporal"]["acquisition"]
    if pair.get("pair_index") != 0 or any(
        pair.get(key) != acquisition.get(key) for key in ("pair_id", "tx_sensor_id", "rx_sensor_id")
    ):
        raise ValueError("Root velocity reference radar pair does not match the acquisition")
    index = pair.get("frame_pair_index")
    if type(index) is not int or index < 0:
        raise ValueError("Root velocity reference requires an exact frame_pair_index")
    positions = []
    for key in ("tx_positions", "rx_positions"):
        array = np.asarray(frame.get(key), dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or index >= len(array):
            raise ValueError(f"Root velocity reference requires the bound {key} row")
        positions.append(array[index])
    return positions


def extract_evaluation_truth(sensing, frame=None):
    """Return fresh numeric truth with explicit reference labels and diagnostics.

    Versioned references use exact saved endpoint poses/velocities. Frozen-root
    projection additionally requires the declared coherent motion model. The
    fixed-sensor root convention remains explicit for undeclared recordings.
    """
    truth = extract_stored_gt_entry(sensing)
    if truth is None:
        return None
    truth["recorded_target_speeds"] = list(truth["target_speeds"])
    truth["target_velocity_references"] = [RECORDED_REFERENCE] * len(truth["target_ids"])
    truth["evaluation_reference_protocol"] = LEGACY_REFERENCE_PROTOCOL
    policy = sensing["truth"].get("velocity_reference_policy")
    if policy == INSTANTANEOUS_VELOCITY_REFERENCE_POLICY or truth.get(
        "target_velocity_reference_evidence"
    ):
        return _instantaneous_truth(sensing, truth)
    catalog = {
        row["target_id"]: row for row in (sensing.get("target_catalog") or {}).get("targets", [])
    }
    targets = {row["target_id"]: row for row in sensing["truth"]["targets"]}
    acquisition = (sensing.get("config_resolved") or {}).get("acquisition") or {}
    if acquisition.get("sample_source") != "coherent_one_solve":
        return truth
    for i, tid in enumerate(truth["target_ids"]):
        declaration = catalog.get(tid, {}).get("animated_asset") or {}
        if (
            declaration.get("rt_velocity_source") != "piecewise_linear_root_translation"
            or declaration.get("coherent_geometry_policy") != "frozen_at_solve_epoch"
        ):
            continue
        truth["target_velocity_references"][i] = ROOT_REFERENCE
        try:
            state = targets[tid].get("animated_asset")
            if not isinstance(state, Mapping) or any(
                not state.get(key) or state.get(key) != declaration.get(key)
                for key in ("binding_id", "asset_revision_sha256")
            ):
                raise ValueError("Root velocity evidence does not match the animated asset")
            tx, rx = _pair_positions(sensing, frame)
            value = 0.5 * bistatic_path_closing_rate_m_s(
                tx,
                rx,
                targets[tid]["position_m"],
                state.get("root_velocity_m_s"),
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            truth["target_speeds"][i] = None
            truth["target_velocity_valid"][i] = False
            truth["target_velocity_sources"][i] = "unavailable"
            truth["target_velocity_reference_dt_s"][i] = None
            truth["target_velocity_unavailable_reasons"][i] = str(exc)
            continue
        truth["target_speeds"][i] = value
        truth["target_velocity_valid"][i] = True
        truth["target_velocity_sources"][i] = "recorded_root_velocity_projection"
        truth["target_velocity_reference_dt_s"][i] = None
        truth["target_velocity_unavailable_reasons"][i] = None
    return truth
