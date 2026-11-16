"""Identity-keyed extraction of canonical schema-v4 sensing truth.

The sensing compatibility break intentionally removes precedence among legacy
``gt*`` aliases. Native consumers read one complete ``truth`` snapshot, join
its pair observables by stable ID, and retain target IDs beside numeric values.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np

from shared.sensing.targets import (
    INSTANTANEOUS_VELOCITY_REFERENCE_POLICY,
    validate_velocity_reference_evidence,
)

GT_CHANNEL_PRECEDENCE = ("truth",)


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def extract_stored_gt_entry(
    sensing_block: Mapping[str, Any],
    expected_targets: Optional[int] = None,
) -> Optional[dict[str, list[Any]]]:
    """Extract active truth observables without positional or alias fallback.

    The output keeps ``target_ids`` aligned with the convenience numeric lists
    used by analysis code. Alignment is created only after exact target and
    pair identity have been validated by the schema-v4 contract.
    """

    if not isinstance(sensing_block, Mapping):
        return None
    truth = sensing_block.get("truth")
    if truth is None:
        return None
    if not isinstance(truth, Mapping):
        raise ValueError("sensing.truth must be a mapping")
    new_policy = "velocity_reference_policy" in truth
    if new_policy and truth["velocity_reference_policy"] != INSTANTANEOUS_VELOCITY_REFERENCE_POLICY:
        raise ValueError("sensing.truth.velocity_reference_policy is invalid")
    targets = truth.get("targets")
    if not isinstance(targets, list):
        raise ValueError("sensing.truth.targets must be a list")

    temporal = sensing_block.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    pair_id = acquisition.get("pair_id") if isinstance(acquisition, Mapping) else None
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("Canonical sensing truth requires temporal acquisition pair_id")
    config = sensing_block.get("config_resolved")
    if not isinstance(config, Mapping):
        raise ValueError(
            "Canonical sensing truth requires a complete config_resolved mapping; "
            "config is not a fallback"
        )
    range_mode = config.get("range_mode")
    if range_mode not in {"monostatic", "bistatic"}:
        raise ValueError(
            "Canonical sensing truth requires config_resolved.range_mode to be exactly "
            "'monostatic' or 'bistatic'"
        )
    monostatic = range_mode == "monostatic"

    target_ids: list[str] = []
    target_ranges: list[float] = []
    target_speeds: list[float | None] = []
    target_velocity_valid: list[bool] = []
    target_velocity_sources: list[str] = []
    target_velocity_reference_dt_s: list[float | None] = []
    target_velocity_unavailable_reasons: list[str | None] = []
    target_velocity_reference_evidence: list[dict[str, Any] | None] = []
    target_velocity_instantaneous_eligible: list[bool] = []
    has_evidence = new_policy
    for target_index, target in enumerate(targets):
        path = f"sensing.truth.targets[{target_index}]"
        if not isinstance(target, Mapping):
            raise ValueError(f"{path} must be a mapping")
        if target.get("lifecycle") != "active" or target.get("valid") is not True:
            continue
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"{path}.target_id must be a stable non-empty string")
        observables = target.get("pair_observables")
        if not isinstance(observables, list):
            raise ValueError(f"{path}.pair_observables must be a list")
        matches = [
            observable
            for observable in observables
            if isinstance(observable, Mapping) and observable.get("pair_id") == pair_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{path} must contain exactly one observable for acquisition pair {pair_id!r}"
            )
        observable = matches[0]
        evidence = observable.get("velocity_reference_evidence")
        if "velocity_reference_evidence" in observable and not new_policy:
            raise ValueError("velocity_reference_evidence requires the truth policy")
        if new_policy or "velocity_reference_evidence" in observable:
            truth_temporal = temporal.get("truth")
            if not isinstance(truth_temporal, Mapping):
                raise ValueError("Versioned velocity evidence requires exact temporal.truth")
            epoch = _finite_float(
                truth_temporal.get("reference_time_s"), path="temporal.truth.reference_time_s"
            )
            errors = validate_velocity_reference_evidence(
                evidence,
                pair_id=pair_id,
                target_id=target_id,
                evaluation_epoch_s=epoch,
                velocity_source=observable.get("velocity_derivation_source"),
                velocity_valid=observable.get("velocity_valid"),
                target_position_m=target.get("position_m"),
                path=f"{path}.velocity_reference_evidence",
            )
            if errors:
                raise ValueError("; ".join(errors))
            has_evidence = True
        target_velocity_reference_evidence.append(deepcopy(evidence))
        target_velocity_instantaneous_eligible.append(
            isinstance(evidence, Mapping) and evidence.get("instantaneous_eligible") is True
        )
        range_field = "monostatic_equivalent_range_m" if monostatic else "bistatic_path_length_m"
        target_ids.append(target_id)
        target_range = _finite_float(observable.get(range_field), path=f"{path}.{range_field}")
        if target_range < 0.0:
            raise ValueError(f"{path}.{range_field} must be non-negative")
        target_ranges.append(target_range)
        velocity_valid = observable.get("velocity_valid")
        if not isinstance(velocity_valid, bool):
            raise ValueError(f"{path}.velocity_valid must be an explicit bool")
        velocity_source = observable.get("velocity_derivation_source")
        if velocity_source not in {
            "exact_epoch_finite_difference",
            "analytic_velocity_projection",
            "unavailable",
        }:
            raise ValueError(f"{path}.velocity_derivation_source is invalid")
        reference_dt = observable.get("velocity_reference_dt_s")
        if velocity_valid:
            if velocity_source == "unavailable":
                raise ValueError(f"{path} valid velocity cannot use source 'unavailable'")
            target_speeds.append(
                _finite_float(
                    observable.get("monostatic_equivalent_velocity_m_s"),
                    path=f"{path}.monostatic_equivalent_velocity_m_s",
                )
            )
            if velocity_source == "exact_epoch_finite_difference":
                reference_dt = _finite_float(
                    reference_dt,
                    path=f"{path}.velocity_reference_dt_s",
                )
                if reference_dt <= 0.0:
                    raise ValueError(f"{path}.velocity_reference_dt_s must be positive")
                if isinstance(evidence, Mapping):
                    support = evidence["support"]
                    first, last = support["start_time_s"], support["end_time_s"]
                    tolerance = 8 * float(np.spacing(max(abs(first), abs(last))))
                    if abs(reference_dt - (last - first)) > tolerance:
                        raise ValueError(
                            f"{path}.velocity_reference_dt_s disagrees with interval support"
                        )
            elif reference_dt is not None:
                raise ValueError(
                    f"{path}.velocity_reference_dt_s must be null for analytic projection"
                )
            unavailable_reason = None
        else:
            if velocity_source != "unavailable":
                raise ValueError(f"{path} invalid velocity must use source 'unavailable'")
            if observable.get("monostatic_equivalent_velocity_m_s") is not None:
                raise ValueError(
                    f"{path}.monostatic_equivalent_velocity_m_s must be null when unavailable"
                )
            if reference_dt is not None:
                raise ValueError(
                    f"{path}.velocity_reference_dt_s must be null when velocity is unavailable"
                )
            unavailable_reason = observable.get("undefined_reason")
            if not isinstance(unavailable_reason, str) or not unavailable_reason:
                raise ValueError(f"{path} unavailable velocity requires undefined_reason")
            target_speeds.append(None)
        target_velocity_valid.append(velocity_valid)
        target_velocity_sources.append(velocity_source)
        target_velocity_reference_dt_s.append(reference_dt)
        target_velocity_unavailable_reasons.append(unavailable_reason)

    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Canonical sensing truth active target_ids must be unique")

    if expected_targets is not None and len(target_ids) != int(expected_targets):
        raise ValueError(
            "Canonical sensing truth active-target count disagrees with the generic target "
            f"projection ({len(target_ids)} != {int(expected_targets)})"
        )
    result: dict[str, list[Any]] = {
        "target_ids": target_ids,
        "target_ranges": target_ranges,
        "target_speeds": target_speeds,
        "target_velocity_valid": target_velocity_valid,
        "target_velocity_sources": target_velocity_sources,
        "target_velocity_reference_dt_s": target_velocity_reference_dt_s,
        "target_velocity_unavailable_reasons": target_velocity_unavailable_reasons,
    }
    if has_evidence:
        result["target_velocity_reference_evidence"] = target_velocity_reference_evidence
        result["target_velocity_instantaneous_eligible"] = target_velocity_instantaneous_eligible
    return result


def select_gt_channel(sensing_block: Mapping[str, Any]) -> Optional[str]:
    """Return ``truth`` when the canonical schema-v4 channel is present."""

    if not isinstance(sensing_block, Mapping):
        return None
    return "truth" if isinstance(sensing_block.get("truth"), Mapping) else None
