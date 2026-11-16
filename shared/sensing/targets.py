"""Identity-keyed sensing target catalog, dynamic state, and truth contracts."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import numpy as np

from shared.frames.json_codec import dumps_frame_json

TARGET_CONTRACT_VERSION = 1
TARGET_LIFECYCLES = frozenset({"inactive", "active", "removed"})
INSTANTANEOUS_VELOCITY_REFERENCE_POLICY = "instantaneous_relative_kinematics_v1"

_CATALOG_REVISION_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PAIR_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_TARGET_ID_RE = re.compile(r"^target_\d+$")


def _content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(dumps_frame_json(value).encode("utf-8")).hexdigest()


def _uuid_form(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def _semantic_target_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and not any(unicodedata.category(character).startswith("C") for character in value)
        and not _uuid_form(value)
        and _FALLBACK_TARGET_ID_RE.fullmatch(value) is None
    )


def _contains_nonfinite_number(value: Any) -> bool:
    """Return whether nested catalog metadata contains NaN or infinity."""

    if isinstance(value, Mapping):
        return any(_contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_number(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)))
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    if isinstance(value, np.generic):
        return _contains_nonfinite_number(value.item())
    return False


def build_target_catalog(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a content-addressed, order-independent static target catalog.

    ``target_id`` is authored semantic text.  It is neither an occurrence UUID
    nor an indexed fallback.  Catalog order is canonicalized only for stable
    serialization; consumers always join by exact target ID.
    """

    normalized = [dict(target) for target in targets]
    normalized.sort(key=lambda target: str(target.get("target_id", "")))
    body = {
        "contract_version": TARGET_CONTRACT_VERSION,
        "targets": normalized,
    }
    catalog = dict(body)
    catalog["catalog_revision"] = _content_hash(body)
    errors = validate_target_catalog(catalog)
    if errors:
        raise ValueError("Invalid sensing target catalog: " + "; ".join(errors))
    return catalog


def validate_target_catalog(catalog: Any) -> list[str]:
    """Return static-catalog contract violations."""

    if not isinstance(catalog, Mapping):
        return ["sensing.target_catalog must be a mapping"]
    errors: list[str] = []
    if catalog.get("contract_version") != TARGET_CONTRACT_VERSION:
        errors.append(
            f"sensing.target_catalog.contract_version must equal {TARGET_CONTRACT_VERSION}"
        )
    targets = catalog.get("targets")
    if not isinstance(targets, list):
        return errors + ["sensing.target_catalog.targets must be a list"]

    seen: set[str] = set()
    previous_id: str | None = None
    for index, target in enumerate(targets):
        path = f"sensing.target_catalog.targets[{index}]"
        if not isinstance(target, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        target_id = target.get("target_id")
        if not _semantic_target_id(target_id):
            errors.append(f"{path}.target_id must be exact non-empty semantic text, not target_N")
            continue
        if target_id in seen:
            errors.append(f"{path}.target_id duplicates {target_id!r}")
        seen.add(target_id)
        if previous_id is not None and target_id <= previous_id:
            errors.append("sensing.target_catalog.targets must be sorted by exact target_id")
        previous_id = target_id
        display_name = target.get("display_name")
        if not isinstance(display_name, str) or not display_name:
            errors.append(f"{path}.display_name must be a non-empty string")
        identity_source = target.get("identity_source")
        if identity_source is not None and (
            not isinstance(identity_source, str) or not identity_source
        ):
            errors.append(f"{path}.identity_source must be null or a non-empty string")
        animated = target.get("animated_asset")
        if animated is not None:
            if not isinstance(animated, Mapping):
                errors.append(f"{path}.animated_asset must be a mapping")
            else:
                for key in ("asset_revision_sha256", "binding_id"):
                    digest = animated.get(key)
                    if not isinstance(digest, str) or _PAIR_ID_RE.fullmatch(digest) is None:
                        errors.append(f"{path}.animated_asset.{key} must be a SHA-256 digest")
                context = animated.get("binding_context")
                if not isinstance(context, Mapping) or context.get("target_id") != target_id:
                    errors.append(
                        f"{path}.animated_asset.binding_context.target_id must match target_id"
                    )
                elif (
                    not _finite_number_value(context.get("playback_rate"))
                    or context["playback_rate"] <= 0
                    or not _finite_number_value(context.get("playback_offset_s"))
                    or context.get("retiming_policy")
                    not in {"affine_scenario_to_motion", "explicit_per_sample"}
                    or any(
                        not _semantic_target_id(context.get(key))
                        for key in ("scenario_clock_id", "motion_clock_id")
                    )
                ):
                    errors.append(
                        f"{path}.animated_asset.binding_context requires explicit finite playback clocks"
                    )
        if _contains_nonfinite_number(target):
            errors.append(f"{path} static metadata must not contain NaN or infinity")

    revision = catalog.get("catalog_revision")
    if not isinstance(revision, str) or _CATALOG_REVISION_RE.fullmatch(revision) is None:
        errors.append("sensing.target_catalog.catalog_revision must be a sha256:-prefixed digest")
    else:
        try:
            expected = _content_hash(
                {
                    "contract_version": catalog.get("contract_version"),
                    "targets": targets,
                }
            )
        except (TypeError, ValueError):
            errors.append("sensing.target_catalog must contain serializable static metadata")
        else:
            if revision != expected:
                errors.append("sensing.target_catalog.catalog_revision does not match its content")
    return errors


def build_target_state(
    catalog: Mapping[str, Any],
    *,
    scene_frame_id: int | str,
    reference_time_s: float,
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one order-independent dynamic target-state snapshot."""

    state = {
        "contract_version": TARGET_CONTRACT_VERSION,
        "scene_frame_id": scene_frame_id,
        "reference_time_s": float(reference_time_s),
        "catalog_revision": catalog.get("catalog_revision"),
        "targets": [dict(target) for target in targets],
    }
    errors = validate_target_state(catalog, state)
    if errors:
        raise ValueError("Invalid sensing target state: " + "; ".join(errors))
    return state


def _finite_vector(value: Any, *, size: int = 3) -> bool:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return False
    array = np.asarray(value)
    return (
        array.shape == (size,)
        and np.issubdtype(array.dtype, np.number)
        and bool(np.all(np.isfinite(array)))
    )


def _validate_dynamic_target(
    target: Mapping[str, Any],
    *,
    path: str,
    known_ids: set[str],
    seen: set[str],
) -> list[str]:
    """Validate identity, lifecycle, and kinematics for one snapshot row."""

    errors: list[str] = []
    target_id = target.get("target_id")
    if not _semantic_target_id(target_id):
        return [f"{path}.target_id must be exact non-UUID semantic text"]
    if target_id not in known_ids:
        errors.append(f"{path}.target_id {target_id!r} is absent from target_catalog")
    if target_id in seen:
        errors.append(f"{path}.target_id duplicates {target_id!r}")
    seen.add(target_id)

    lifecycle = target.get("lifecycle")
    if lifecycle not in TARGET_LIFECYCLES:
        errors.append(f"{path}.lifecycle must be active, inactive, or removed")
    valid = target.get("valid")
    if not isinstance(valid, bool):
        return errors + [f"{path}.valid must be a bool"]

    position = target.get("position_m")
    orientation = target.get("orientation_rad")
    velocity = target.get("velocity_m_s")
    invalid_reason = target.get("invalid_reason")
    if lifecycle == "active" and valid:
        if not _finite_vector(position):
            errors.append(f"{path}.position_m must be a finite xyz vector when active/valid")
        for field, value in (
            ("orientation_rad", orientation),
            ("velocity_m_s", velocity),
        ):
            if value is not None and not _finite_vector(value):
                errors.append(f"{path}.{field} must be null or a finite xyz vector")
        if invalid_reason is not None:
            errors.append(f"{path}.invalid_reason must be null when active/valid")
    else:
        for field, value in (
            ("position_m", position),
            ("orientation_rad", orientation),
            ("velocity_m_s", velocity),
        ):
            if value is not None:
                errors.append(f"{path}.{field} must be null unless active and valid")
        if lifecycle == "active":
            if not isinstance(invalid_reason, str) or not invalid_reason:
                errors.append(
                    f"{path}.invalid_reason must be non-empty when active state is invalid"
                )
        elif invalid_reason is not None:
            errors.append(f"{path}.invalid_reason must be null for {lifecycle} state")
    if lifecycle in {"inactive", "removed"} and valid:
        errors.append(f"{path}.valid must be false when lifecycle is {lifecycle}")
    return errors


def validate_target_state(catalog: Any, state: Any) -> list[str]:
    """Return state violations, including unknown IDs and lifecycle misuse."""

    catalog_errors = validate_target_catalog(catalog)
    if catalog_errors:
        return catalog_errors
    if not isinstance(state, Mapping):
        return ["sensing.target_state must be a mapping"]
    errors: list[str] = []
    if state.get("contract_version") != TARGET_CONTRACT_VERSION:
        errors.append(f"sensing.target_state.contract_version must equal {TARGET_CONTRACT_VERSION}")
    if state.get("catalog_revision") != catalog.get("catalog_revision"):
        errors.append("sensing.target_state.catalog_revision must match target_catalog")
    scene_frame_id = state.get("scene_frame_id")
    if not (
        isinstance(scene_frame_id, int)
        and not isinstance(scene_frame_id, bool)
        or isinstance(scene_frame_id, str)
        and bool(scene_frame_id)
    ):
        errors.append("sensing.target_state.scene_frame_id must be an integer or string")
    reference_time_s = state.get("reference_time_s")
    if (
        not isinstance(reference_time_s, (int, float))
        or isinstance(reference_time_s, bool)
        or not math.isfinite(float(reference_time_s))
    ):
        errors.append("sensing.target_state.reference_time_s must be finite")

    catalog_targets = catalog.get("targets", [])
    known_ids = {
        target.get("target_id") for target in catalog_targets if isinstance(target, Mapping)
    }
    targets = state.get("targets")
    if not isinstance(targets, list):
        return errors + ["sensing.target_state.targets must be a list"]
    seen: set[str] = set()
    for index, target in enumerate(targets):
        path = f"sensing.target_state.targets[{index}]"
        if not isinstance(target, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        errors.extend(
            _validate_dynamic_target(
                target,
                path=path,
                known_ids=known_ids,
                seen=seen,
            )
        )
        catalog_row = next(
            (row for row in catalog_targets if row.get("target_id") == target.get("target_id")), {}
        )
        static = catalog_row.get("animated_asset")
        animated = target.get("animated_asset")
        if animated is not None or (
            static is not None and target.get("lifecycle") == "active" and target.get("valid")
        ):
            errors.extend(_validate_animated_state(animated, static, path=f"{path}.animated_asset"))
    missing_ids = known_ids - seen
    if missing_ids:
        errors.append(
            "sensing.target_state.targets must contain every catalog target; "
            f"missing {sorted(missing_ids)!r}"
        )
    return errors


def _validate_animated_state(animated: Any, static: Any, *, path: str) -> list[str]:
    """Validate source-solve geometry evidence without conflating truth epochs."""
    if not isinstance(animated, Mapping) or not isinstance(static, Mapping):
        return [f"{path} requires matching catalog metadata and complete dynamic evidence"]
    errors = []
    for key in ("asset_revision_sha256", "binding_id"):
        if animated.get(key) != static.get(key):
            errors.append(f"{path}.{key} must match its catalog")
    if _contains_nonfinite_number(animated):
        errors.append(f"{path} must not contain NaN or infinity")
    for key in ("rt_mesh_center_m", "root_velocity_m_s"):
        if not _finite_vector(animated.get(key)):
            errors.append(f"{path}.{key} must be a finite xyz vector")
    step = animated.get("geometry_scene_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        errors.append(f"{path}.geometry_scene_step must be a nonnegative integer")
    if animated.get("geometry_evidence_policy") != "source_solve_sample_not_interpolated":
        errors.append(f"{path}.geometry_evidence_policy must identify a source solve sample")
    sample = animated.get("sample")
    if not isinstance(sample, Mapping):
        return errors + [f"{path}.sample requires complete binding sample evidence"]
    for key in ("sample_id", "pose_sample_id"):
        if not _semantic_target_id(sample.get(key)):
            errors.append(f"{path}.sample.{key} requires exact nonempty semantic identity")
    for key in ("scenario_time_s", "motion_time_s"):
        if not _finite_number_value(sample.get(key)):
            errors.append(f"{path}.sample.{key} must be finite")
    if sample.get("pulse_time_s") is not None and not _finite_number_value(
        sample.get("pulse_time_s")
    ):
        errors.append(f"{path}.sample.pulse_time_s must be finite or null")
    for key in ("vertices_sha256", "normals_sha256"):
        if not isinstance(sample.get(key), str) or _PAIR_ID_RE.fullmatch(sample[key]) is None:
            errors.append(f"{path}.sample.{key} must be a SHA-256 digest")
    evaluated = sample.get("evaluated_motion")
    if not isinstance(evaluated, Mapping) or any(
        not _finite_number_value(evaluated.get(key)) for key in ("pose_time_s", "root_time_s")
    ):
        errors.append(f"{path}.sample.evaluated_motion requires finite pose/root times")
    if isinstance(evaluated, Mapping):
        for key in ("pose_cycle", "root_cycle"):
            cycle = evaluated.get(key)
            if isinstance(cycle, bool) or not isinstance(cycle, int):
                errors.append(f"{path}.sample.evaluated_motion.{key} must be an integer")
    if sample.get("validity") != {"valid": True, "reason": None}:
        errors.append(f"{path}.sample.validity must identify a valid bound sample")
    root = sample.get("root")
    if not isinstance(root, Mapping):
        errors.append(f"{path}.sample.root requires resolved root evidence")
    else:
        for key in ("lower_sample_id", "upper_sample_id"):
            if not _semantic_target_id(root.get(key)):
                errors.append(f"{path}.sample.root.{key} requires exact nonempty identity")
        fraction = root.get("fraction")
        if not _finite_number_value(fraction) or not 0 <= fraction <= 1:
            errors.append(f"{path}.sample.root.fraction must lie in [0, 1]")
        try:
            affine = np.asarray(root.get("resolved_affine_4x4"), dtype=np.float64)
            affine_valid = affine.shape == (4, 4) and np.all(np.isfinite(affine))
        except (TypeError, ValueError):
            affine_valid = False
        if not affine_valid:
            errors.append(f"{path}.sample.root.resolved_affine_4x4 must be finite 4x4")
        digest = root.get("resolved_affine_sha256")
        if not isinstance(digest, str) or _PAIR_ID_RE.fullmatch(digest) is None:
            errors.append(f"{path}.sample.root.resolved_affine_sha256 must be a SHA-256 digest")
        elif affine_valid:
            header = b'{"dtype":"<f8","shape":[4,4]}'
            expected_digest = hashlib.sha256(
                header + b"\0" + np.ascontiguousarray(affine, dtype="<f8").tobytes(order="C")
            ).hexdigest()
            if digest != expected_digest:
                errors.append(
                    f"{path}.sample.root.resolved_affine_sha256 does not match its matrix"
                )
    context = static.get("binding_context", {})
    if context.get("retiming_policy") == "affine_scenario_to_motion" and all(
        _finite_number_value(number)
        for number in (
            sample.get("scenario_time_s"),
            sample.get("motion_time_s"),
            context.get("playback_rate"),
            context.get("playback_offset_s"),
        )
    ):
        expected = (
            context["playback_rate"] * sample["scenario_time_s"] + context["playback_offset_s"]
        )
        if not math.isclose(sample["motion_time_s"], expected, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"{path}.sample.motion_time_s does not match affine clock binding")
    return errors


def _finite_number_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_target_lifecycle_sequence(
    catalog: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate per-ID lifecycle continuity across ordered snapshots."""

    errors: list[str] = []
    removed: set[str] = set()
    prior_time: float | None = None
    for state_index, state in enumerate(states):
        errors.extend(
            f"states[{state_index}]: {error}" for error in validate_target_state(catalog, state)
        )
        time_value = state.get("reference_time_s") if isinstance(state, Mapping) else None
        if isinstance(time_value, (int, float)) and not isinstance(time_value, bool):
            current_time = float(time_value)
            if prior_time is not None and current_time <= prior_time:
                errors.append(f"states[{state_index}]: reference_time_s must increase")
            prior_time = current_time
        targets = state.get("targets", []) if isinstance(state, Mapping) else []
        for target in targets if isinstance(targets, list) else []:
            if not isinstance(target, Mapping):
                continue
            target_id = target.get("target_id")
            lifecycle = target.get("lifecycle")
            if isinstance(target_id, str) and target_id in removed and lifecycle != "removed":
                errors.append(
                    f"states[{state_index}]: target {target_id!r} reappears after removed"
                )
            if isinstance(target_id, str) and lifecycle == "removed":
                removed.add(target_id)
    return errors


def build_target_truth(
    catalog: Mapping[str, Any],
    *,
    targets: Sequence[Mapping[str, Any]],
    velocity_reference_policy: str | None = None,
) -> dict[str, Any]:
    """Build canonical evaluation-only truth joined by semantic target ID."""

    truth = {
        "contract_version": TARGET_CONTRACT_VERSION,
        "catalog_revision": catalog.get("catalog_revision"),
        "targets": [dict(target) for target in targets],
    }
    if velocity_reference_policy is not None:
        truth["velocity_reference_policy"] = velocity_reference_policy
    errors = validate_target_truth(catalog, truth)
    if errors:
        raise ValueError("Invalid sensing target truth: " + "; ".join(errors))
    return truth


def validate_velocity_reference_evidence(
    evidence: Any,
    *,
    pair_id: str,
    target_id: str,
    evaluation_epoch_s: float | None = None,
    velocity_source: str | None = None,
    velocity_valid: bool | None = None,
    target_position_m: Any = None,
    path: str = "velocity_reference_evidence",
) -> list[str]:
    """Validate optional versioned relative-kinematics truth evidence.

    Interval fallback remains a valid recorded observable, with its own temporal
    support; it is never eligible as an instantaneous comparison. Outer temporal
    consumers additionally bind the evidence to their exact truth/result epoch.
    """
    if not isinstance(evidence, Mapping):
        return [f"{path} must be a mapping"]
    errors: list[str] = []
    if type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
        errors.append(f"{path}.schema_version must equal 1")
    if evidence.get("policy") != INSTANTANEOUS_VELOCITY_REFERENCE_POLICY:
        errors.append(f"{path}.policy must be {INSTANTANEOUS_VELOCITY_REFERENCE_POLICY}")
    if evidence.get("pair_id") != pair_id:
        errors.append(f"{path}.pair_id must match its exact pair")
    if evidence.get("target_id") != target_id:
        errors.append(f"{path}.target_id must match its exact target")
    epoch = evidence.get("evaluation_epoch_s")
    if not _finite_number_value(epoch):
        errors.append(f"{path}.evaluation_epoch_s must be finite")
    elif evaluation_epoch_s is not None and epoch != evaluation_epoch_s:
        errors.append(f"{path}.evaluation_epoch_s must match the truth epoch")
    for key in ("tx_position_m", "rx_position_m", "target_position_m"):
        if not _finite_vector(evidence.get(key)):
            errors.append(f"{path}.{key} must be a finite three-vector")
    if (
        target_position_m is not None
        and _finite_vector(target_position_m)
        and _finite_vector(evidence.get("target_position_m"))
        and not np.array_equal(target_position_m, evidence["target_position_m"])
    ):
        errors.append(f"{path}.target_position_m must match its exact target state")
    velocities = ("tx_velocity_m_s", "rx_velocity_m_s", "target_velocity_m_s")
    for key in velocities:
        if key not in evidence or (evidence[key] is not None and not _finite_vector(evidence[key])):
            errors.append(f"{path}.{key} must be null or a finite three-vector")
    eligible = evidence.get("instantaneous_eligible")
    if type(eligible) is not bool:
        errors.append(f"{path}.instantaneous_eligible must be boolean")
    source = evidence.get("source")
    if velocity_source is not None and source != velocity_source:
        errors.append(f"{path}.source must match the recorded derivation")
    support = evidence.get("support")
    if not isinstance(support, Mapping):
        return errors + [f"{path}.support must be a mapping"]
    kind = support.get("kind")
    expected_source = {
        "instantaneous": "analytic_velocity_projection",
        "interval_mean": "exact_epoch_finite_difference",
        "unavailable": "unavailable",
    }.get(kind)
    if expected_source is None:
        return errors + [f"{path}.support.kind is invalid"]
    if source != expected_source:
        errors.append(f"{path}.source disagrees with temporal support")
    if eligible is not (kind == "instantaneous"):
        errors.append(f"{path}.instantaneous_eligible disagrees with temporal support")
    if velocity_valid is not None and velocity_valid is not (kind != "unavailable"):
        errors.append(f"{path}.support disagrees with recorded velocity validity")
    reason = evidence.get("fallback_reason")
    if kind == "instantaneous":
        if any(evidence.get(key) is None for key in velocities):
            errors.append(f"{path} instantaneous support requires all endpoint velocities")
        if reason is not None:
            errors.append(f"{path}.fallback_reason must be null for instantaneous support")
        if not _finite_number_value(epoch) or any(
            not _finite_number_value(support.get(key)) or support[key] != epoch
            for key in ("reference_time_s", "start_time_s", "end_time_s")
        ):
            errors.append(f"{path}.support must identify the exact instantaneous epoch")
    else:
        if not isinstance(reason, str) or not reason:
            errors.append(f"{path}.fallback_reason must explain noninstantaneous support")
        if kind == "unavailable":
            if any(
                key not in support or support[key] is not None
                for key in ("reference_time_s", "start_time_s", "end_time_s")
            ):
                errors.append(f"{path}.support unavailable times must be null")
        else:
            first, last, midpoint = (
                support.get("start_time_s"),
                support.get("end_time_s"),
                support.get("reference_time_s"),
            )
            if not all(_finite_number_value(x) for x in (first, last, midpoint)):
                errors.append(f"{path}.support interval times must be finite")
            elif first >= last or last != epoch or midpoint != (first + last) / 2.0:
                errors.append(f"{path}.support must identify the prior interval and midpoint")
    return errors


def validate_target_truth(catalog: Any, truth: Any) -> list[str]:
    """Return canonical complete, identity-keyed target-truth violations."""

    catalog_errors = validate_target_catalog(catalog)
    if catalog_errors:
        return catalog_errors
    if not isinstance(truth, Mapping):
        return ["sensing.truth must be a mapping"]
    errors: list[str] = []
    new_policy = "velocity_reference_policy" in truth
    if new_policy and truth["velocity_reference_policy"] != INSTANTANEOUS_VELOCITY_REFERENCE_POLICY:
        errors.append("sensing.truth.velocity_reference_policy is invalid")
    if truth.get("contract_version") != TARGET_CONTRACT_VERSION:
        errors.append(f"sensing.truth.contract_version must equal {TARGET_CONTRACT_VERSION}")
    if truth.get("catalog_revision") != catalog.get("catalog_revision"):
        errors.append("sensing.truth.catalog_revision must match target_catalog")
    known_ids = {
        target.get("target_id")
        for target in catalog.get("targets", [])
        if isinstance(target, Mapping)
    }
    targets = truth.get("targets")
    if not isinstance(targets, list):
        return errors + ["sensing.truth.targets must be a list"]
    seen: set[str] = set()
    for index, target in enumerate(targets):
        path = f"sensing.truth.targets[{index}]"
        if not isinstance(target, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        errors.extend(
            _validate_dynamic_target(
                target,
                path=path,
                known_ids=known_ids,
                seen=seen,
            )
        )
        pair_observables = target.get("pair_observables")
        if not isinstance(pair_observables, list):
            errors.append(f"{path}.pair_observables must be a list")
            continue
        if not (target.get("lifecycle") == "active" and target.get("valid") is True):
            if pair_observables:
                errors.append(
                    f"{path}.pair_observables must be empty unless target truth is active/valid"
                )
            continue
        seen_pairs: set[str] = set()
        for pair_index, observable in enumerate(pair_observables):
            observable_path = f"{path}.pair_observables[{pair_index}]"
            if not isinstance(observable, Mapping):
                errors.append(f"{observable_path} must be a mapping")
                continue
            pair_id = observable.get("pair_id")
            if not isinstance(pair_id, str) or _PAIR_ID_RE.fullmatch(pair_id) is None:
                errors.append(f"{observable_path}.pair_id must be a canonical SHA-256 ID")
            elif pair_id in seen_pairs:
                errors.append(f"{observable_path}.pair_id duplicates {pair_id!r}")
            else:
                seen_pairs.add(pair_id)
            if "velocity_reference_evidence" in observable and not new_policy:
                errors.append(
                    f"{observable_path}.velocity_reference_evidence requires the truth policy"
                )
            if new_policy or "velocity_reference_evidence" in observable:
                errors.extend(
                    validate_velocity_reference_evidence(
                        observable.get("velocity_reference_evidence"),
                        pair_id=pair_id,
                        target_id=target.get("target_id"),
                        velocity_source=observable.get("velocity_derivation_source"),
                        velocity_valid=observable.get("velocity_valid"),
                        target_position_m=target.get("position_m"),
                        path=f"{observable_path}.velocity_reference_evidence",
                    )
                )
            numeric_fields = (
                "bistatic_path_length_m",
                "monostatic_equivalent_range_m",
                "path_closing_rate_m_s",
                "monostatic_equivalent_velocity_m_s",
                "bistatic_bisector_velocity_m_s",
            )
            angular_keys = (
                "receiver_azimuth_deg",
                "receiver_elevation_deg",
                "angular_reference_frame",
            )
            if any(key in observable for key in angular_keys):
                if observable.get("angular_reference_frame") != "rx_array_local":
                    errors.append(
                        f"{observable_path}.angular_reference_frame must be rx_array_local"
                    )
                for key, limit in (("receiver_azimuth_deg", 180), ("receiver_elevation_deg", 90)):
                    value = observable.get(key)
                    if not _finite_number_value(value) or abs(value) > limit:
                        errors.append(
                            f"{observable_path}.{key} must be finite within +/-{limit} degrees"
                        )
            any_undefined = False
            for field in numeric_fields:
                value = observable.get(field)
                if value is None:
                    any_undefined = True
                elif (
                    not isinstance(value, (int, float, np.integer, np.floating))
                    or isinstance(value, (bool, np.bool_))
                    or not math.isfinite(float(value))
                ):
                    errors.append(f"{observable_path}.{field} must be null or finite")
            undefined_reason = observable.get("undefined_reason")
            if any_undefined:
                if not isinstance(undefined_reason, str) or not undefined_reason:
                    errors.append(
                        f"{observable_path}.undefined_reason must explain null observables"
                    )
            elif undefined_reason is not None:
                errors.append(
                    f"{observable_path}.undefined_reason must be null when observables are finite"
                )
    missing_ids = known_ids - seen
    if missing_ids:
        errors.append(
            "sensing.truth.targets must contain every catalog target; "
            f"missing {sorted(missing_ids)!r}"
        )
    return errors


def target_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index target rows by exact ID, rejecting positional/fallback identity."""

    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target_id = row.get("target_id") if isinstance(row, Mapping) else None
        if not _semantic_target_id(target_id):
            raise ValueError("Target rows require exact semantic target_id values")
        if target_id in indexed:
            raise ValueError(f"Duplicate target_id {target_id!r}")
        indexed[target_id] = row
    return indexed


__all__ = [
    "INSTANTANEOUS_VELOCITY_REFERENCE_POLICY",
    "TARGET_CONTRACT_VERSION",
    "TARGET_LIFECYCLES",
    "build_target_catalog",
    "build_target_state",
    "build_target_truth",
    "target_rows_by_id",
    "validate_target_catalog",
    "validate_target_lifecycle_sequence",
    "validate_target_state",
    "validate_target_truth",
    "validate_velocity_reference_evidence",
]
