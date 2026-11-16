"""Pure, deterministic replay contracts for native sensing schema-v4 results.

This module deliberately sits above the generic frame/HDF5 container.  It
turns one fully validated native sensing result and explicit scene context into
an immutable replay event that can be indexed, exported, and compared without
running ray tracing or processing again.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal
from uuid import UUID

import numpy as np

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    assert_sensing_contract,
    canonical_sensing_hash,
)
from shared.frames.json_codec import dumps_frame_json

EvidenceLayer = Literal["estimate", "truth", "oracle"]
CheckStatus = Literal["pass", "warning", "fail"]
ReplayDisposition = Literal["ready", "ready_with_warnings", "blocked"]
FrameId = int | str

REPLAY_MANIFEST_VERSION = 1
PIN_POLICY_EXACT = "exact_scene_binding"
PIN_POLICY_ALLOW_STALE = "allow_pinned_stale"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = {"__replay_missing__": True}


def _normalized_json(
    value: Any,
    *,
    path: str,
    nonfinite_to_none: bool = False,
) -> Any:
    """Return deterministic frame JSON with an explicit non-finite policy.

    Identity, timing, recipe, product, and pose callers retain the strict
    default.  Optional result evidence may opt into the public export policy
    that represents NaN and either infinity as JSON ``null``.  NumPy arrays
    are expanded before that substitution so non-finite elements cannot hide
    inside the frame codec's binary ndarray representation.
    """

    def canonicalize_nonfinite(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): canonicalize_nonfinite(child) for key, child in item.items()}
        if isinstance(item, np.ndarray):
            return canonicalize_nonfinite(item.tolist())
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return None if not bool(np.isfinite(item)) else float(item)
        if isinstance(item, np.complexfloating):
            return None if not bool(np.isfinite(item)) else complex(item)
        if isinstance(item, np.generic):
            return canonicalize_nonfinite(item.item())
        if isinstance(item, (list, tuple)):
            return [canonicalize_nonfinite(child) for child in item]
        if isinstance(item, complex):
            if not math.isfinite(item.real) or not math.isfinite(item.imag):
                return None
            return item
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    def reject_source_nonfinite(item: Any, item_path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_source_nonfinite(child, f"{item_path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                reject_source_nonfinite(child, f"{item_path}[{index}]")
        elif isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.number):
            if not bool(np.all(np.isfinite(item))):
                raise ValueError(f"{item_path} must contain only finite numbers")
        elif isinstance(item, (float, complex, np.floating, np.complexfloating)):
            if not bool(np.isfinite(item)):
                raise ValueError(f"{item_path} must be finite")

    source = canonicalize_nonfinite(value) if nonfinite_to_none else value
    reject_source_nonfinite(source, path)
    try:
        normalized = json.loads(dumps_frame_json(source))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be deterministic frame-JSON data") from exc

    def reject_nonfinite(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                reject_nonfinite(child, f"{item_path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_nonfinite(child, f"{item_path}[{index}]")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{item_path} must be finite")

    reject_nonfinite(normalized, path)
    return normalized


@dataclass(frozen=True, slots=True)
class FrozenMap(Mapping[str, Any]):
    """Small recursively immutable, hashable mapping used by replay records."""

    entries: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        previous: str | None = None
        for key, _value in self.entries:
            if not isinstance(key, str):
                raise TypeError("FrozenMap keys must be strings")
            if previous is not None and key <= previous:
                raise ValueError("FrozenMap entries must have unique sorted keys")
            previous = key

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible copy."""

        return {key: _thaw(value) for key, value in self.entries}


def _freeze_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenMap(
            tuple(
                (str(key), _freeze_normalized(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


def freeze_replay_json(value: Any, *, path: str = "value") -> Any:
    """Freeze shared-codec JSON data for safe use in immutable replay records."""

    if isinstance(value, FrozenMap):
        return value
    return _freeze_normalized(_normalized_json(value, path=path))


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _frame_id(value: Any, *, path: str) -> FrameId:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{path} must be an integer or non-empty string")


def _nonnegative_int(value: Any, *, path: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or int(value) < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return int(value)


def _uuid(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a UUID string")
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{path} must be a UUID string") from exc
    return value


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _frame_token(value: FrameId) -> tuple[str, str]:
    return ("int", str(value)) if isinstance(value, int) else ("str", value)


def _finite_vector3(value: Any, *, path: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite xyz vector") from exc
    if array.shape != (3,) or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{path} must be a finite xyz vector")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True, slots=True)
class ReplayEventKey:
    """Exact identity of one processed result occurrence."""

    clock_id: str
    acquisition_id: str
    processing_run_id: str
    pair_id: str
    reference_time_s: float

    def __post_init__(self) -> None:
        _uuid(self.clock_id, path="clock_id")
        _uuid(self.acquisition_id, path="acquisition_id")
        _uuid(self.processing_run_id, path="processing_run_id")
        _sha256(self.pair_id, path="pair_id")
        object.__setattr__(
            self,
            "reference_time_s",
            _finite_float(self.reference_time_s, path="reference_time_s"),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "acquisition_id": self.acquisition_id,
            "clock_id": self.clock_id,
            "pair_id": self.pair_id,
            "processing_run_id": self.processing_run_id,
            "reference_time_s": self.reference_time_s,
        }


@dataclass(frozen=True, slots=True)
class ReplaySceneContext:
    """Caller-supplied storage, bound-scene, and current-scene context."""

    storage_frame_id: FrameId
    bound_scene_frame_id: FrameId
    current_scene_frame_id: FrameId
    bound_scene_time_s: float
    current_scene_time_s: float
    pinned: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_frame_id",
            _frame_id(self.storage_frame_id, path="storage_frame_id"),
        )
        object.__setattr__(
            self,
            "bound_scene_frame_id",
            _frame_id(self.bound_scene_frame_id, path="bound_scene_frame_id"),
        )
        object.__setattr__(
            self,
            "current_scene_frame_id",
            _frame_id(self.current_scene_frame_id, path="current_scene_frame_id"),
        )
        object.__setattr__(
            self,
            "bound_scene_time_s",
            _finite_float(self.bound_scene_time_s, path="bound_scene_time_s"),
        )
        object.__setattr__(
            self,
            "current_scene_time_s",
            _finite_float(self.current_scene_time_s, path="current_scene_time_s"),
        )
        if not isinstance(self.pinned, bool):
            raise ValueError("pinned must be a bool")
        if self.bound_scene_frame_id == self.current_scene_frame_id and not math.isclose(
            self.bound_scene_time_s,
            self.current_scene_time_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("one scene_frame_id cannot have contradictory scene times")


@dataclass(frozen=True, slots=True)
class ReplaySceneBinding:
    """Resolved scene relationship, including explicit fresh/stale state."""

    storage_frame_id: FrameId
    bound_scene_frame_id: FrameId
    current_scene_frame_id: FrameId
    bound_scene_time_s: float
    current_scene_time_s: float
    current_scene_result_offset_s: float
    stale_age_s: float
    pinned: bool
    state: Literal["fresh", "stale"]
    fresh: bool
    stale: bool

    def to_manifest(self) -> dict[str, Any]:
        return {
            "bound_scene_frame_id": self.bound_scene_frame_id,
            "bound_scene_time_s": self.bound_scene_time_s,
            "current_scene_frame_id": self.current_scene_frame_id,
            "current_scene_result_offset_s": self.current_scene_result_offset_s,
            "current_scene_time_s": self.current_scene_time_s,
            "fresh": self.fresh,
            "pinned": self.pinned,
            "stale": self.stale,
            "stale_age_s": self.stale_age_s,
            "state": self.state,
            "storage_frame_id": self.storage_frame_id,
        }


@dataclass(frozen=True, slots=True)
class ReplayPairPose:
    """Exact typed sensing pair and its complete bound-scene world pose."""

    pair_id: str
    pair_index: int
    frame_pair_index: int
    tx_sensor_id: str
    rx_sensor_id: str
    tx_device_index: int
    rx_device_index: int
    tx_position_m: tuple[float, float, float]
    tx_orientation_yaw_pitch_roll_rad: tuple[float, float, float]
    rx_position_m: tuple[float, float, float]
    rx_orientation_yaw_pitch_roll_rad: tuple[float, float, float]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "frame_pair_index": self.frame_pair_index,
            "pair_id": self.pair_id,
            "pair_index": self.pair_index,
            "rx": {
                "device_index": self.rx_device_index,
                "orientation_yaw_pitch_roll_rad": list(self.rx_orientation_yaw_pitch_roll_rad),
                "position_m": list(self.rx_position_m),
                "sensor_id": self.rx_sensor_id,
            },
            "tx": {
                "device_index": self.tx_device_index,
                "orientation_yaw_pitch_roll_rad": list(self.tx_orientation_yaw_pitch_roll_rad),
                "position_m": list(self.tx_position_m),
                "sensor_id": self.tx_sensor_id,
            },
        }


@dataclass(frozen=True, slots=True)
class ReplayPulse:
    """One exact ordered acquisition slot and its source occurrence."""

    sample_index: int
    sample_time_s: float
    valid: bool
    phase_coherent: bool
    source_generation_id: str
    source_scene_frame_id: FrameId
    sample_offset: int
    source_frame_set_id: str | None
    solve_id: str | None
    source_reference_id: str
    record: FrozenMap

    def to_manifest(self) -> dict[str, Any]:
        return {
            "phase_coherent": self.phase_coherent,
            "record": self.record.to_dict(),
            "sample_index": self.sample_index,
            "sample_offset": self.sample_offset,
            "sample_time_s": self.sample_time_s,
            "solve_id": self.solve_id,
            "source_frame_set_id": self.source_frame_set_id,
            "source_generation_id": self.source_generation_id,
            "source_reference_id": self.source_reference_id,
            "source_scene_frame_id": self.source_scene_frame_id,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class ReplayProductReference:
    """Typed, lineage-complete reference to one sensing product."""

    product_id: str
    data_key: str
    kind: str
    variant: str
    representation: str
    unit: str
    scale_semantics: str
    axes: tuple[FrozenMap, ...]
    lineage: FrozenMap
    time_span: FrozenMap
    evidence_layer: EvidenceLayer

    def to_manifest(self) -> dict[str, Any]:
        return {
            "axes": [axis.to_dict() for axis in self.axes],
            "data_key": self.data_key,
            "evidence_layer": self.evidence_layer,
            "kind": self.kind,
            "lineage": self.lineage.to_dict(),
            "product_id": self.product_id,
            "representation": self.representation,
            "scale_semantics": self.scale_semantics,
            "time_span": self.time_span.to_dict(),
            "unit": self.unit,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class ReplayRecipeSnapshot:
    """Immutable processing recipe plus its execution-plan context."""

    processing_run_id: str
    processing_recipe_id: str
    canonical_hash: str
    parameters: FrozenMap
    processing_plan: FrozenMap

    def to_manifest(self) -> dict[str, Any]:
        return {
            "canonical_hash": self.canonical_hash,
            "parameters": self.parameters.to_dict(),
            "processing_plan": self.processing_plan.to_dict(),
            "processing_recipe_id": self.processing_recipe_id,
            "processing_run_id": self.processing_run_id,
        }


@dataclass(frozen=True, slots=True)
class ReplayEvidenceReference:
    """Deterministic reference to an estimate, truth row, or oracle result."""

    reference_id: str
    kind: Literal["detection", "track", "truth"]
    layer: EvidenceLayer
    identity: str
    ordinal: int
    payload_hash: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kind": self.kind,
            "layer": self.layer,
            "ordinal": self.ordinal,
            "payload_hash": self.payload_hash,
            "reference_id": self.reference_id,
        }


@dataclass(frozen=True, slots=True)
class ReplayPreflightCheck:
    """One actionable replay credibility check; no aggregate numeric score."""

    check_id: str
    status: CheckStatus
    message: str
    action: str
    metrics: FrozenMap

    def to_manifest(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "check_id": self.check_id,
            "message": self.message,
            "metrics": self.metrics.to_dict(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ReplayCredibilityPreflight:
    """Actionable checks and categorical disposition for one replay event."""

    disposition: ReplayDisposition
    checks: tuple[ReplayPreflightCheck, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "checks": [check.to_manifest() for check in self.checks],
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """Immutable canonical sensing replay event."""

    key: ReplayEventKey
    scene_binding: ReplaySceneBinding
    pair_pose: ReplayPairPose
    pulses: tuple[ReplayPulse, ...]
    source_samples_id: str
    recipe: ReplayRecipeSnapshot
    products: tuple[ReplayProductReference, ...]
    root_source_product_ids: tuple[str, ...]
    evidence: tuple[ReplayEvidenceReference, ...]
    preflight: ReplayCredibilityPreflight

    @property
    def detections(self) -> tuple[ReplayEvidenceReference, ...]:
        return tuple(reference for reference in self.evidence if reference.kind == "detection")

    @property
    def tracks(self) -> tuple[ReplayEvidenceReference, ...]:
        return tuple(reference for reference in self.evidence if reference.kind == "track")

    @property
    def truth(self) -> tuple[ReplayEvidenceReference, ...]:
        return tuple(reference for reference in self.evidence if reference.layer == "truth")

    @property
    def estimates(self) -> tuple[ReplayEvidenceReference, ...]:
        return tuple(reference for reference in self.evidence if reference.layer == "estimate")

    @property
    def oracle(self) -> tuple[ReplayEvidenceReference, ...]:
        return tuple(reference for reference in self.evidence if reference.layer == "oracle")

    @property
    def estimate_reference_id(self) -> str:
        """Truth-independent identity of estimate-layer products and records."""

        return str(
            canonical_sensing_hash(
                {
                    "evidence": [
                        reference.to_manifest()
                        for reference in self.evidence
                        if reference.layer == "estimate"
                    ],
                    "products": [
                        product.to_manifest()
                        for product in self.products
                        if product.evidence_layer == "estimate"
                    ],
                }
            )
        )

    def manifest_payload(self) -> dict[str, Any]:
        """Return the complete deterministic replay/view manifest body.

        This payload deliberately retains storage and current-view context for
        auditability.  Those location and presentation fields are excluded
        from :attr:`manifest_id`; use :attr:`view_manifest_id` when the exact
        serialized view occurrence itself needs an identity.
        """

        return {
            "event_key": self.key.to_manifest(),
            "evidence": [reference.to_manifest() for reference in self.evidence],
            "preflight": self.preflight.to_manifest(),
            "products": [product.to_manifest() for product in self.products],
            "pair_pose": self.pair_pose.to_manifest(),
            "pulse_timeline": [pulse.to_manifest() for pulse in self.pulses],
            "recipe": self.recipe.to_manifest(),
            "replay_manifest_version": REPLAY_MANIFEST_VERSION,
            "root_source_product_ids": list(self.root_source_product_ids),
            "scene_binding": self.scene_binding.to_manifest(),
            "source_samples_id": self.source_samples_id,
        }

    def scientific_manifest_payload(self) -> dict[str, Any]:
        """Return the storage- and presentation-invariant event content.

        Storage rows and bound/current scene identifiers are locations, while
        pin state and current scene time are view state.  The exact bound pose
        remains scientific content through ``pair_pose``.  Scene alignment is
        still preserved in :meth:`manifest_payload` and the exported replay
        manifest, but it cannot change the canonical event fingerprint or
        detector-stage cache key while a user seeks with one pinned result.
        """

        return {
            "event_key": self.key.to_manifest(),
            "evidence": [reference.to_manifest() for reference in self.evidence],
            "products": [product.to_manifest() for product in self.products],
            "pair_pose": self.pair_pose.to_manifest(),
            "pulse_timeline": [pulse.to_manifest() for pulse in self.pulses],
            "recipe": self.recipe.to_manifest(),
            "replay_manifest_version": REPLAY_MANIFEST_VERSION,
            "root_source_product_ids": list(self.root_source_product_ids),
            "source_samples_id": self.source_samples_id,
        }

    @property
    def manifest_id(self) -> str:
        """Return the canonical scientific event/content identity."""

        return str(canonical_sensing_hash(self.scientific_manifest_payload()))

    @property
    def view_manifest_id(self) -> str:
        """Return the identity of the complete scene/view-bound manifest."""

        return str(canonical_sensing_hash(self.manifest_payload()))

    def to_manifest(self) -> dict[str, Any]:
        manifest = self.manifest_payload()
        manifest["manifest_id"] = self.manifest_id
        manifest["view_manifest_id"] = self.view_manifest_id
        return manifest

    def to_json(self) -> str:
        return json.dumps(
            self.to_manifest(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    manifest_json = to_json


def _extract_sensing(frame: Any) -> tuple[Mapping[str, Any], Any, Any]:
    if isinstance(frame, Mapping):
        if "sensing" in frame:
            sensing = frame.get("sensing")
            storage_id = frame.get("frame_index")
        elif frame.get("schema_version") == SENSING_SCHEMA_VERSION:
            sensing = frame
            storage_id = None
        else:
            raise ValueError("frame must contain a native sensing schema-v4 payload")
    else:
        sensing = getattr(frame, "sensing", None)
        storage_id = getattr(frame, "frame_index", None)
    if not isinstance(sensing, Mapping):
        raise ValueError("frame.sensing must be a mapping")
    return sensing, storage_id, frame


def _validate_loaded_physical_axes(sensing: Mapping[str, Any]) -> None:
    """Reject non-finite loaded dense-axis coordinates used by replay products."""
    products = sensing.get("products")
    if not isinstance(products, Mapping):
        return
    checked: set[str] = set()
    for data_key, descriptor in products.items():
        if not isinstance(descriptor, Mapping):
            continue
        axes = descriptor.get("axes")
        if not isinstance(axes, list):
            continue
        for axis_index, axis in enumerate(axes):
            if not isinstance(axis, Mapping) or axis.get("coordinate_kind") != "dense":
                continue
            raw_keys = [axis.get("coordinate_key")]
            alternate_keys = axis.get("alternate_coordinate_keys")
            if isinstance(alternate_keys, list):
                raw_keys.extend(alternate_keys)
            for raw_key in raw_keys:
                if not isinstance(raw_key, str) or raw_key in checked or raw_key not in sensing:
                    continue
                checked.add(raw_key)
                values = sensing[raw_key]
                try:
                    array = np.asarray(values)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"sensing.{raw_key} must contain finite real axis coordinates"
                    ) from exc
                if (
                    array.dtype.kind not in "iuf"
                    or array.dtype.kind == "b"
                    or not bool(np.all(np.isfinite(array)))
                ):
                    raise ValueError(
                        f"sensing.{raw_key} must contain finite real axis coordinates "
                        f"for products[{data_key!r}].axes[{axis_index}]"
                    )


def _frame_value(frame: Any, key: str) -> Any:
    return frame.get(key) if isinstance(frame, Mapping) else getattr(frame, key, None)


def _pose_row(frame: Any, key: str, index: int) -> tuple[float, float, float]:
    values = _frame_value(frame, key)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame.{key} must be a finite [frame_pair, xyz] array") from exc
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"frame.{key} must be a finite [frame_pair, xyz] array")
    if index >= array.shape[0]:
        raise ValueError(f"pair_axis.frame_pair_index is outside frame.{key}")
    return _finite_vector3(array[index], path=f"frame.{key}[{index}]")


def _build_pair_pose(
    frame: Any,
    sensing: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> ReplayPairPose:
    pair_axis = sensing.get("pair_axis")
    if not isinstance(pair_axis, list) or len(pair_axis) != 1:
        raise ValueError("canonical replay requires exactly one typed pair_axis row")
    pair = pair_axis[0]
    if not isinstance(pair, Mapping):
        raise ValueError("sensing.pair_axis[0] must be a mapping")
    pair_id = _sha256(pair.get("pair_id"), path="sensing.pair_axis[0].pair_id")
    tx_sensor_id = _nonempty(pair.get("tx_sensor_id"), path="sensing.pair_axis[0].tx_sensor_id")
    rx_sensor_id = _nonempty(pair.get("rx_sensor_id"), path="sensing.pair_axis[0].rx_sensor_id")
    for field, pair_value in (
        ("pair_id", pair_id),
        ("tx_sensor_id", tx_sensor_id),
        ("rx_sensor_id", rx_sensor_id),
    ):
        if pair_value != acquisition.get(field):
            raise ValueError(f"sensing.pair_axis[0].{field} must match temporal acquisition")
    pair_index = _nonnegative_int(pair.get("pair_index"), path="pair_axis[0].pair_index")
    if pair_index != 0:
        raise ValueError("schema-v4 replay requires pair_axis[0].pair_index equal 0")
    frame_pair_index = _nonnegative_int(
        pair.get("frame_pair_index"), path="pair_axis[0].frame_pair_index"
    )
    tx_device_index = _nonnegative_int(
        pair.get("tx_device_index"), path="pair_axis[0].tx_device_index"
    )
    rx_device_index = _nonnegative_int(
        pair.get("rx_device_index"), path="pair_axis[0].rx_device_index"
    )
    return ReplayPairPose(
        pair_id=pair_id,
        pair_index=pair_index,
        frame_pair_index=frame_pair_index,
        tx_sensor_id=tx_sensor_id,
        rx_sensor_id=rx_sensor_id,
        tx_device_index=tx_device_index,
        rx_device_index=rx_device_index,
        tx_position_m=_pose_row(frame, "tx_positions", frame_pair_index),
        tx_orientation_yaw_pitch_roll_rad=_pose_row(frame, "tx_orientations", frame_pair_index),
        rx_position_m=_pose_row(frame, "rx_positions", frame_pair_index),
        rx_orientation_yaw_pitch_roll_rad=_pose_row(frame, "rx_orientations", frame_pair_index),
    )


def _mapping(parent: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping")
    return value


def _resolve_context(
    scene_context: ReplaySceneContext | None,
    *,
    storage_frame_id: FrameId | None,
    bound_scene_frame_id: FrameId | None,
    current_scene_frame_id: FrameId | None,
    bound_scene_time_s: float | None,
    current_scene_time_s: float | None,
    pinned: bool,
) -> ReplaySceneContext:
    explicit = (
        storage_frame_id,
        bound_scene_frame_id,
        current_scene_frame_id,
        bound_scene_time_s,
        current_scene_time_s,
    )
    if scene_context is not None:
        if any(value is not None for value in explicit) or pinned:
            raise ValueError("scene_context cannot be combined with individual scene arguments")
        if not isinstance(scene_context, ReplaySceneContext):
            raise TypeError("scene_context must be a ReplaySceneContext")
        return scene_context
    if any(value is None for value in explicit):
        raise ValueError(
            "storage, bound/current scene IDs, and bound/current scene times are required"
        )
    return ReplaySceneContext(
        storage_frame_id=storage_frame_id,  # type: ignore[arg-type]
        bound_scene_frame_id=bound_scene_frame_id,  # type: ignore[arg-type]
        current_scene_frame_id=current_scene_frame_id,  # type: ignore[arg-type]
        bound_scene_time_s=bound_scene_time_s,  # type: ignore[arg-type]
        current_scene_time_s=current_scene_time_s,  # type: ignore[arg-type]
        pinned=pinned,
    )


def _build_scene_binding(
    context: ReplaySceneContext,
    *,
    reference_time_s: float,
) -> ReplaySceneBinding:
    fresh = context.bound_scene_frame_id == context.current_scene_frame_id
    offset = context.current_scene_time_s - reference_time_s
    return ReplaySceneBinding(
        storage_frame_id=context.storage_frame_id,
        bound_scene_frame_id=context.bound_scene_frame_id,
        current_scene_frame_id=context.current_scene_frame_id,
        bound_scene_time_s=context.bound_scene_time_s,
        current_scene_time_s=context.current_scene_time_s,
        current_scene_result_offset_s=offset,
        stale_age_s=max(0.0, offset) if not fresh else 0.0,
        pinned=context.pinned,
        state="fresh" if fresh else "stale",
        fresh=fresh,
        stale=not fresh,
    )


def _build_pulses(samples: Sequence[Any]) -> tuple[ReplayPulse, ...]:
    pulses: list[ReplayPulse] = []
    for position, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"sensing.temporal.acquisition.samples[{position}] must be a mapping")
        source = raw_sample.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(
                f"complete acquisition sample {position} requires an exact source mapping"
            )
        sample_index = raw_sample.get("sample_index")
        sample_offset = source.get("sample_offset")
        if not isinstance(sample_index, Integral) or isinstance(sample_index, (bool, np.bool_)):
            raise ValueError(f"sample {position} has no exact integer sample_index")
        if not isinstance(sample_offset, Integral) or isinstance(sample_offset, (bool, np.bool_)):
            raise ValueError(f"sample {position} has no exact integer sample_offset")
        sample_time_s = _finite_float(
            raw_sample.get("sample_time_s"), path=f"samples[{position}].sample_time_s"
        )
        source_record = {
            "sample_index": int(sample_index),
            "sample_time_s": sample_time_s,
            "source": dict(source),
        }
        reference_id = canonical_sensing_hash(source_record)
        record = freeze_replay_json(raw_sample, path=f"samples[{position}]")
        if not isinstance(record, FrozenMap):
            raise AssertionError("sample mapping did not freeze as a mapping")
        pulses.append(
            ReplayPulse(
                sample_index=int(sample_index),
                sample_time_s=sample_time_s,
                valid=raw_sample.get("valid") is True,
                phase_coherent=raw_sample.get("phase_coherent") is True,
                source_generation_id=_uuid(
                    source.get("source_generation_id"),
                    path=f"samples[{position}].source.source_generation_id",
                ),
                source_scene_frame_id=_frame_id(
                    source.get("scene_frame_id"),
                    path=f"samples[{position}].source.scene_frame_id",
                ),
                sample_offset=int(sample_offset),
                source_frame_set_id=(
                    _nonempty(
                        source.get("source_frame_set_id"),
                        path=f"samples[{position}].source.source_frame_set_id",
                    )
                    if "source_frame_set_id" in source
                    else None
                ),
                solve_id=(
                    _nonempty(
                        source.get("solve_id"),
                        path=f"samples[{position}].source.solve_id",
                    )
                    if "solve_id" in source
                    else None
                ),
                source_reference_id=reference_id,
                record=record,
            )
        )
    return tuple(pulses)


def _result_layer(sensing: Mapping[str, Any], config: Mapping[str, Any]) -> EvidenceLayer:
    processing = config.get("processing")
    candidates = [sensing.get("processing_strategy"), config.get("processing_mode")]
    if isinstance(processing, Mapping):
        candidates.extend((processing.get("mode"), processing.get("strategy")))
    source_processing = config.get("source_processing")
    if isinstance(processing, Mapping) and processing.get("mode") == "offline_reprocess":
        if isinstance(source_processing, Mapping):
            candidates.extend((source_processing.get("mode"), source_processing.get("strategy")))
    return "oracle" if any("oracle" in str(value).lower() for value in candidates) else "estimate"


def _product_layer(descriptor: Mapping[str, Any], default: EvidenceLayer) -> EvidenceLayer:
    role = str(descriptor.get("evidence_role", "")).lower()
    if "truth" in role:
        return "truth"
    if "oracle" in role:
        return "oracle"
    return default


def _build_products(
    products: Mapping[str, Any],
    *,
    default_layer: EvidenceLayer,
) -> tuple[ReplayProductReference, ...]:
    references: list[ReplayProductReference] = []
    for data_key in sorted(products):
        descriptor = products[data_key]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"sensing.products[{data_key!r}] must be a mapping")
        axes_value = descriptor.get("axes")
        if not isinstance(axes_value, list):
            raise ValueError(f"sensing.products[{data_key!r}].axes must be a list")
        axes: list[FrozenMap] = []
        for axis_index, axis in enumerate(axes_value):
            frozen = freeze_replay_json(
                axis, path=f"sensing.products[{data_key!r}].axes[{axis_index}]"
            )
            if not isinstance(frozen, FrozenMap):
                raise ValueError(f"product {data_key!r} axis {axis_index} must be a mapping")
            axes.append(frozen)
        lineage = freeze_replay_json(
            descriptor.get("lineage"), path=f"sensing.products[{data_key!r}].lineage"
        )
        time_span = freeze_replay_json(
            descriptor.get("time_span"), path=f"sensing.products[{data_key!r}].time_span"
        )
        if not isinstance(lineage, FrozenMap) or not isinstance(time_span, FrozenMap):
            raise ValueError(f"product {data_key!r} requires mapping lineage and time_span")
        references.append(
            ReplayProductReference(
                product_id=_sha256(
                    descriptor.get("product_id"), path=f"products[{data_key!r}].product_id"
                ),
                data_key=_nonempty(data_key, path="product data_key"),
                kind=_nonempty(descriptor.get("kind"), path=f"products[{data_key!r}].kind"),
                variant=_nonempty(
                    descriptor.get("variant"), path=f"products[{data_key!r}].variant"
                ),
                representation=_nonempty(
                    descriptor.get("representation"),
                    path=f"products[{data_key!r}].representation",
                ),
                unit=_nonempty(descriptor.get("unit"), path=f"products[{data_key!r}].unit"),
                scale_semantics=_nonempty(
                    descriptor.get("scale_semantics"),
                    path=f"products[{data_key!r}].scale_semantics",
                ),
                axes=tuple(axes),
                lineage=lineage,
                time_span=time_span,
                evidence_layer=_product_layer(descriptor, default_layer),
            )
        )
    return tuple(references)


def _root_source_product_ids(products: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve deterministic acquisition-root/external product ancestry."""
    graph: dict[str, tuple[str, ...]] = {}
    for data_key, descriptor in products.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"sensing.products[{data_key!r}] must be a mapping")
        product_id = _sha256(
            descriptor.get("product_id"), path=f"sensing.products[{data_key!r}].product_id"
        )
        lineage = descriptor.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError(f"sensing.products[{data_key!r}].lineage must be a mapping")
        raw_parents = lineage.get("source_product_ids")
        if not isinstance(raw_parents, list):
            raise ValueError(
                f"sensing.products[{data_key!r}].lineage.source_product_ids must be a list"
            )
        parents = tuple(
            _sha256(parent, path=f"sensing.products[{data_key!r}].source_product_ids")
            for parent in raw_parents
        )
        graph[product_id] = parents

    memo: dict[str, frozenset[str]] = {}

    def roots(product_id: str, visiting: frozenset[str]) -> frozenset[str]:
        if product_id in memo:
            return memo[product_id]
        if product_id in visiting:
            raise ValueError("sensing product lineage contains a cycle")
        parents = graph.get(product_id)
        if parents is None or not parents:
            resolved = frozenset({product_id})
        else:
            resolved = frozenset().union(
                *(roots(parent, visiting | {product_id}) for parent in parents)
            )
        memo[product_id] = resolved
        return resolved

    root_ids = frozenset().union(*(roots(product_id, frozenset()) for product_id in graph))
    return tuple(sorted(root_ids))


def _explicit_layer(
    record: Mapping[str, Any], inferred: EvidenceLayer, *, path: str
) -> EvidenceLayer:
    if "evidence_layer" not in record:
        return inferred
    explicit = record.get("evidence_layer")
    if explicit not in {"estimate", "truth", "oracle"}:
        raise ValueError(f"{path}.evidence_layer must be estimate, truth, or oracle")
    if explicit != inferred:
        raise ValueError(
            f"{path}.evidence_layer={explicit!r} contradicts inferred layer {inferred!r}"
        )
    return inferred


def _evidence_identity(value: Any, *, path: str, fallback: str) -> str:
    """Return a stable evidence identity without accepting non-finite IDs."""
    if value is None:
        return fallback
    if isinstance(value, (complex, np.complexfloating)):
        if not bool(np.isfinite(value)):
            raise ValueError(f"{path} must not be non-finite")
    elif isinstance(value, (float, np.floating)) and not bool(np.isfinite(value)):
        raise ValueError(f"{path} must not be non-finite")
    return str(value)


def _reference_records(
    records: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    kind: Literal["detection", "track", "truth"],
    layer: EvidenceLayer,
) -> list[ReplayEvidenceReference]:
    prepared: list[tuple[str, str, EvidenceLayer]] = []
    for path, record in records:
        payload = _normalized_json(record, path=path, nonfinite_to_none=True)
        payload_hash = canonical_sensing_hash(payload)
        if kind == "truth":
            identity_field = "target_id"
            raw_identity = record.get("target_id")
        elif kind == "track":
            identity_field = "track_id"
            raw_identity = record.get("track_id")
        else:
            identity_field = "detection_id"
            raw_identity = record.get("detection_id")
        identity = _evidence_identity(
            raw_identity,
            path=f"{path}.{identity_field}",
            fallback=payload_hash,
        )
        prepared.append((identity, payload_hash, _explicit_layer(record, layer, path=path)))
    prepared.sort()

    result: list[ReplayEvidenceReference] = []
    occurrence_by_identity: dict[tuple[str, str], int] = {}
    for identity, payload_hash, record_layer in prepared:
        occurrence_key = (identity, payload_hash)
        ordinal = occurrence_by_identity.get(occurrence_key, 0)
        occurrence_by_identity[occurrence_key] = ordinal + 1
        reference_id = canonical_sensing_hash(
            {
                "identity": identity,
                "kind": kind,
                "layer": record_layer,
                "ordinal": ordinal,
                "payload_hash": payload_hash,
            }
        )
        result.append(
            ReplayEvidenceReference(
                reference_id=reference_id,
                kind=kind,
                layer=record_layer,
                identity=identity,
                ordinal=ordinal,
                payload_hash=payload_hash,
            )
        )
    return result


def _track_records(
    sensing: Mapping[str, Any], detections: list[Any]
) -> list[tuple[str, Mapping[str, Any]]]:
    raw_tracks = sensing.get("tracks")
    if raw_tracks is not None:
        if isinstance(raw_tracks, Mapping):
            records: list[tuple[str, Mapping[str, Any]]] = []
            for track_id, raw_track in raw_tracks.items():
                if not isinstance(raw_track, Mapping):
                    raise ValueError(f"sensing.tracks[{track_id!r}] must be a mapping")
                record = dict(raw_track)
                if "track_id" in record and str(record["track_id"]) != str(track_id):
                    raise ValueError(f"sensing.tracks[{track_id!r}] has contradictory track_id")
                record["track_id"] = track_id
                records.append((f"sensing.tracks[{track_id!r}]", record))
            return records
        if isinstance(raw_tracks, list):
            records = []
            for index, track in enumerate(raw_tracks):
                if not isinstance(track, Mapping):
                    raise ValueError(f"sensing.tracks[{index}] must be a mapping")
                records.append((f"sensing.tracks[{index}]", track))
            return records
        raise ValueError("sensing.tracks must be a mapping or list when present")

    grouped: dict[str, list[str]] = {}
    for index, raw_detection in enumerate(detections):
        if not isinstance(raw_detection, Mapping):
            raise ValueError(f"sensing.detections[{index}] must be a mapping")
        if raw_detection.get("track_id") is None:
            continue
        track_id = _evidence_identity(
            raw_detection["track_id"],
            path=f"sensing.detections[{index}].track_id",
            fallback="",
        )
        grouped.setdefault(track_id, []).append(
            canonical_sensing_hash(
                _normalized_json(
                    raw_detection,
                    path=f"detections[{index}]",
                    nonfinite_to_none=True,
                )
            )
        )
    return [
        (
            f"sensing.detections.track_id[{track_id!r}]",
            {"detection_payload_hashes": sorted(payload_hashes), "track_id": track_id},
        )
        for track_id, payload_hashes in sorted(grouped.items())
    ]


def _build_evidence(
    sensing: Mapping[str, Any],
    *,
    result_layer: EvidenceLayer,
) -> tuple[ReplayEvidenceReference, ...]:
    raw_detections = sensing.get("detections", [])
    if not isinstance(raw_detections, list):
        raise ValueError("sensing.detections must be a list when present")
    detection_records: list[tuple[str, Mapping[str, Any]]] = []
    for index, detection in enumerate(raw_detections):
        if not isinstance(detection, Mapping):
            raise ValueError(f"sensing.detections[{index}] must be a mapping")
        detection_records.append((f"sensing.detections[{index}]", detection))

    references = _reference_records(
        detection_records,
        kind="detection",
        layer=result_layer,
    )
    references.extend(
        _reference_records(
            _track_records(sensing, raw_detections),
            kind="track",
            layer=result_layer,
        )
    )

    raw_truth = sensing.get("truth")
    if raw_truth is not None:
        if not isinstance(raw_truth, Mapping) or not isinstance(raw_truth.get("targets"), list):
            raise ValueError("sensing.truth.targets must be a list")
        truth_records: list[tuple[str, Mapping[str, Any]]] = []
        for index, target in enumerate(raw_truth["targets"]):
            if not isinstance(target, Mapping):
                raise ValueError(f"sensing.truth.targets[{index}] must be a mapping")
            truth_records.append((f"sensing.truth.targets[{index}]", target))
        references.extend(_reference_records(truth_records, kind="truth", layer="truth"))

    references.sort(
        key=lambda reference: (
            reference.layer,
            reference.kind,
            reference.identity,
            reference.payload_hash,
            reference.ordinal,
        )
    )
    return tuple(references)


def _check(
    check_id: str,
    status: CheckStatus,
    message: str,
    action: str,
    metrics: Mapping[str, Any],
) -> ReplayPreflightCheck:
    frozen = freeze_replay_json(metrics, path=f"preflight.{check_id}.metrics")
    if not isinstance(frozen, FrozenMap):
        raise AssertionError("preflight metrics did not freeze as a mapping")
    return ReplayPreflightCheck(check_id, status, message, action, frozen)


def _scene_alignment_check(scene_binding: ReplaySceneBinding) -> ReplayPreflightCheck:
    return _check(
        "scene_alignment",
        "pass" if scene_binding.fresh else "warning",
        (
            "The sensing result is bound to the current scene."
            if scene_binding.fresh
            else "The sensing result is bound to a different scene than the current view."
        ),
        (
            "No action required."
            if scene_binding.fresh
            else "Seek to the bound scene or explicitly use the pinned-stale selection policy."
        ),
        {
            "bound_scene_frame_id": scene_binding.bound_scene_frame_id,
            "current_scene_frame_id": scene_binding.current_scene_frame_id,
            "current_scene_result_offset_s": scene_binding.current_scene_result_offset_s,
            "pinned": scene_binding.pinned,
            "stale_age_s": scene_binding.stale_age_s,
        },
    )


def _summarize_preflight(checks: Iterable[ReplayPreflightCheck]) -> ReplayCredibilityPreflight:
    ordered = tuple(sorted(checks, key=lambda item: item.check_id))
    statuses = {check.status for check in ordered}
    disposition: ReplayDisposition
    if "fail" in statuses:
        disposition = "blocked"
    elif "warning" in statuses:
        disposition = "ready_with_warnings"
    else:
        disposition = "ready"
    return ReplayCredibilityPreflight(disposition, ordered)


def _build_preflight(
    *,
    acquisition: Mapping[str, Any],
    config: Mapping[str, Any],
    scene_binding: ReplaySceneBinding,
    pair_pose: ReplayPairPose,
    pulses: tuple[ReplayPulse, ...],
    products: tuple[ReplayProductReference, ...],
    root_source_product_ids: tuple[str, ...],
    evidence: tuple[ReplayEvidenceReference, ...],
) -> ReplayCredibilityPreflight:
    checks: list[ReplayPreflightCheck] = [
        _check(
            "native_schema_contract",
            "pass",
            "Native sensing schema-v4 validation passed.",
            "No action required.",
            {"schema_version": SENSING_SCHEMA_VERSION},
        ),
        _check(
            "acquisition_coverage",
            "pass",
            "Every expected pulse has an observed, valid source sample.",
            "No action required.",
            {
                "expected_sample_count": acquisition.get("expected_sample_count"),
                "observed_sample_count": acquisition.get("observed_sample_count"),
                "valid_sample_count": acquisition.get("valid_sample_count"),
                "start_time_s": acquisition.get("start_time_s"),
                "end_time_s": acquisition.get("end_time_s"),
            },
        ),
    ]

    processing = config.get("processing")
    mode = str(processing.get("mode", "")) if isinstance(processing, Mapping) else ""
    coherent_required = "coherent" in mode and "oracle" not in mode
    coherent_count = sum(pulse.phase_coherent for pulse in pulses)
    coherent_status: CheckStatus = (
        "fail" if coherent_required and coherent_count != len(pulses) else "pass"
    )
    checks.append(
        _check(
            "phase_coherence",
            coherent_status,
            (
                "The processing mode's phase-coherence requirement is satisfied."
                if coherent_status == "pass"
                else "A coherent processing mode contains non-coherent pulse slots."
            ),
            (
                "No action required."
                if coherent_status == "pass"
                else "Regenerate the acquisition with coherent sources or choose a non-coherent recipe."
            ),
            {
                "coherent_required": coherent_required,
                "phase_coherent_sample_count": coherent_count,
                "sample_count": len(pulses),
            },
        )
    )

    checks.append(_scene_alignment_check(scene_binding))

    checks.append(
        _check(
            "bound_pair_pose",
            "pass",
            "The typed sensing pair has complete bound-scene TX/RX poses.",
            "No action required.",
            {
                "frame_pair_index": pair_pose.frame_pair_index,
                "pair_id": pair_pose.pair_id,
                "rx_sensor_id": pair_pose.rx_sensor_id,
                "tx_sensor_id": pair_pose.tx_sensor_id,
            },
        )
    )

    checks.append(
        _check(
            "typed_products",
            "pass" if products else "warning",
            (
                "All replay products have typed references and immutable lineage."
                if products
                else "The result contains no typed numeric products."
            ),
            (
                "No action required."
                if products
                else "Inspect detection metadata or regenerate products."
            ),
            {
                "estimate_product_count": sum(
                    product.evidence_layer == "estimate" for product in products
                ),
                "oracle_product_count": sum(
                    product.evidence_layer == "oracle" for product in products
                ),
                "product_count": len(products),
                "root_source_product_count": len(root_source_product_ids),
                "truth_product_count": sum(
                    product.evidence_layer == "truth" for product in products
                ),
            },
        )
    )

    sampling_frequency = _finite_float(
        config.get("sampling_frequency"), path="config_resolved.sampling_frequency"
    )
    prf_hz = _finite_float(config.get("prf_hz"), path="config_resolved.prf_hz")
    carrier_hz = _finite_float(
        config.get("carrier_freq_hz"), path="config_resolved.carrier_freq_hz"
    )
    speed_of_light_m_s = 299_792_458.0
    cpi_duration_s = pulses[-1].sample_time_s - pulses[0].sample_time_s
    monostatic = config.get("range_mode") == "monostatic"
    response = config.get("range_response") or {}
    response_model = response.get("model", "point_bin_floor")
    bandwidth_hz = _finite_float(
        config.get("bandwidth", sampling_frequency), path="config_resolved.bandwidth"
    )
    # CFR ambiguity is set by the unpadded frequency spacing, not PRF or
    # IFFT padding. Point-bin projection does not model periodic delay aliasing.
    delay_period_s = (
        float(response["num_frequency_bins"]) / bandwidth_hz
        if response_model == "cfr_ifft"
        else None
    )
    range_factor = 2.0 if monostatic else 1.0
    sampled_doppler = config.get("acquisition", {}).get("sample_source") != "analytic_path_snapshot"
    checks.append(
        _check(
            "declared_resolution_envelope",
            "pass",
            "Declared observation-grid and acquisition scales are available for credibility review.",
            "Compare these declared limits with the scene and expected target dynamics.",
            {
                "carrier_freq_hz": carrier_hz,
                "cpi_duration_s": cpi_duration_s,
                "monostatic_range_resolution_m": (
                    speed_of_light_m_s / (2.0 * bandwidth_hz) if monostatic else None
                ),
                "range_response_model": response_model,
                "delay_ambiguity_period_s": delay_period_s,
                "modeled_unambiguous_range_m": (
                    speed_of_light_m_s * delay_period_s / range_factor
                    if delay_period_s is not None
                    else None
                ),
                "monostatic_unambiguous_range_m": (
                    speed_of_light_m_s * delay_period_s / 2.0
                    if monostatic and delay_period_s is not None
                    else None
                ),
                "theoretical_prf_monostatic_range_bound_m": (
                    speed_of_light_m_s / (2.0 * prf_hz) if monostatic else None
                ),
                "range_ambiguity_policy": (
                    "frequency_grid_delay_period"
                    if delay_period_s is not None
                    else "point_bin_projection_no_periodic_delay_model"
                ),
                "sampled_doppler": sampled_doppler,
                "monostatic_unambiguous_velocity_m_s": (
                    speed_of_light_m_s * prf_hz / (4.0 * carrier_hz)
                    if monostatic and sampled_doppler
                    else None
                ),
                "prf_hz": prf_hz,
                "range_mode": config.get("range_mode"),
                "sampling_frequency_hz": sampling_frequency,
            },
        )
    )
    checks.append(
        _check(
            "evidence_layer_separation",
            "pass",
            "Estimate, truth, and oracle references are explicitly separated.",
            "Keep truth and oracle layers diagnostic when interpreting estimates.",
            {
                "estimate_reference_count": sum(item.layer == "estimate" for item in evidence),
                "oracle_reference_count": sum(item.layer == "oracle" for item in evidence),
                "truth_reference_count": sum(item.layer == "truth" for item in evidence),
            },
        )
    )

    return _summarize_preflight(checks)


def build_replay_event(
    frame: Any,
    scene_context: ReplaySceneContext | None = None,
    *,
    storage_frame_id: FrameId | None = None,
    bound_scene_frame_id: FrameId | None = None,
    current_scene_frame_id: FrameId | None = None,
    bound_scene_time_s: float | None = None,
    current_scene_time_s: float | None = None,
    pinned: bool = False,
    require_product_data: bool = True,
) -> ReplayEvent:
    """Build one strict replay event from a complete native schema-v4 frame.

    Existing schema validators run first.  Replay-specific cross-checks then
    reject incomplete acquisitions, recipe/hash mismatches, ambiguous scene
    bindings, and non-deterministic metadata. ``require_product_data=False``
    admits a selective projection only when its complete descriptors remain
    valid; any loaded product bytes are still checked against their descriptor.
    """

    if not isinstance(require_product_data, bool):
        raise TypeError("require_product_data must be a bool")
    sensing, frame_storage_id, replay_frame = _extract_sensing(frame)
    assert_sensing_contract(
        {"sensing": sensing},
        require_schema_version=True,
        require_described_products=require_product_data,
    )
    _validate_loaded_physical_axes(sensing)
    if sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        raise ValueError(f"replay requires sensing schema version {SENSING_SCHEMA_VERSION}")

    context = _resolve_context(
        scene_context,
        storage_frame_id=storage_frame_id,
        bound_scene_frame_id=bound_scene_frame_id,
        current_scene_frame_id=current_scene_frame_id,
        bound_scene_time_s=bound_scene_time_s,
        current_scene_time_s=current_scene_time_s,
        pinned=pinned,
    )
    if (
        frame_storage_id is not None
        and _frame_id(frame_storage_id, path="frame.frame_index") != context.storage_frame_id
    ):
        raise ValueError("scene_context.storage_frame_id must match frame.frame_index")

    temporal = _mapping(sensing, "temporal", path="sensing")
    clock = _mapping(temporal, "clock", path="sensing.temporal")
    acquisition = _mapping(temporal, "acquisition", path="sensing.temporal")
    result = _mapping(temporal, "result", path="sensing.temporal")
    display = _mapping(temporal, "display_binding", path="sensing.temporal")
    config = _mapping(sensing, "config_resolved", path="sensing")
    recipe_parameters = _mapping(config, "processing_recipe", path="sensing.config_resolved")
    processing_plan = _mapping(config, "processing", path="sensing.config_resolved")
    products_mapping = _mapping(sensing, "products", path="sensing")

    if acquisition.get("completeness") != "complete":
        raise ValueError("canonical replay requires acquisition.completeness='complete'")
    display_frame_id = display.get("scene_frame_id")
    if display_frame_id is None:
        raise ValueError("canonical replay requires a non-null display-bound scene_frame_id")
    if (
        _frame_id(display_frame_id, path="sensing.temporal.display_binding.scene_frame_id")
        != context.bound_scene_frame_id
    ):
        raise ValueError(
            "scene_context.bound_scene_frame_id must match temporal display_binding.scene_frame_id"
        )

    recipe_id = _sha256(
        result.get("processing_recipe_id"),
        path="sensing.temporal.result.processing_recipe_id",
    )
    recipe_hash = canonical_sensing_hash(recipe_parameters)
    if recipe_id != recipe_hash:
        raise ValueError(
            "processing_recipe_id must equal canonical_sensing_hash("
            "config_resolved.processing_recipe)"
        )

    key = ReplayEventKey(
        clock_id=_uuid(clock.get("clock_id"), path="sensing.temporal.clock.clock_id"),
        acquisition_id=_uuid(
            acquisition.get("acquisition_id"),
            path="sensing.temporal.acquisition.acquisition_id",
        ),
        processing_run_id=_uuid(
            result.get("processing_run_id"),
            path="sensing.temporal.result.processing_run_id",
        ),
        pair_id=_sha256(acquisition.get("pair_id"), path="sensing.temporal.acquisition.pair_id"),
        reference_time_s=_finite_float(
            result.get("reference_time_s"), path="sensing.temporal.result.reference_time_s"
        ),
    )
    scene_binding = _build_scene_binding(context, reference_time_s=key.reference_time_s)
    pair_pose = _build_pair_pose(replay_frame, sensing, acquisition)
    if pair_pose.pair_id != key.pair_id:
        raise ValueError("replay pair pose pair_id must match ReplayEventKey.pair_id")
    raw_samples = acquisition.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("canonical replay requires a non-empty acquisition pulse timeline")
    pulses = _build_pulses(raw_samples)
    if not all(pulse.valid for pulse in pulses):
        raise ValueError("canonical replay requires every acquisition pulse to be valid")
    source_samples_id = canonical_sensing_hash([pulse.record.to_dict() for pulse in pulses])

    frozen_recipe = freeze_replay_json(
        recipe_parameters, path="sensing.config_resolved.processing_recipe"
    )
    frozen_processing = freeze_replay_json(
        processing_plan, path="sensing.config_resolved.processing"
    )
    if not isinstance(frozen_recipe, FrozenMap) or not isinstance(frozen_processing, FrozenMap):
        raise AssertionError("recipe mappings did not freeze as mappings")
    recipe = ReplayRecipeSnapshot(
        processing_run_id=key.processing_run_id,
        processing_recipe_id=recipe_id,
        canonical_hash=recipe_hash,
        parameters=frozen_recipe,
        processing_plan=frozen_processing,
    )
    result_layer = _result_layer(sensing, config)
    products = _build_products(products_mapping, default_layer=result_layer)
    root_source_product_ids = _root_source_product_ids(products_mapping)
    evidence = _build_evidence(sensing, result_layer=result_layer)
    preflight = _build_preflight(
        acquisition=acquisition,
        config=config,
        scene_binding=scene_binding,
        pair_pose=pair_pose,
        pulses=pulses,
        products=products,
        root_source_product_ids=root_source_product_ids,
        evidence=evidence,
    )
    return ReplayEvent(
        key=key,
        scene_binding=scene_binding,
        pair_pose=pair_pose,
        pulses=pulses,
        source_samples_id=source_samples_id,
        recipe=recipe,
        products=products,
        root_source_product_ids=root_source_product_ids,
        evidence=evidence,
        preflight=preflight,
    )


def rebind_replay_event(event: ReplayEvent, scene_context: ReplaySceneContext) -> ReplayEvent:
    """Refresh only view state on an already validated immutable replay event.

    The caller must retain the same acquisition, products, configuration, and
    bound pair pose. Storage and bound-scene identity/time cannot change here;
    those changes require building and validating a new event from its source.
    Current scene ID/time and pinning are validated by ``ReplaySceneContext``.
    Both scene binding and its credibility check use the canonical builders,
    preserving exact fresh-build manifests without repeating scientific work.
    """
    if not isinstance(event, ReplayEvent) or not isinstance(scene_context, ReplaySceneContext):
        raise TypeError("rebind requires a ReplayEvent and ReplaySceneContext")
    original = event.scene_binding
    for field_name in ("storage_frame_id", "bound_scene_frame_id", "bound_scene_time_s"):
        previous = getattr(original, field_name)
        current = getattr(scene_context, field_name)
        if (
            type(previous) is not type(current)
            or previous != current
            or (isinstance(previous, float) and previous.hex() != current.hex())
        ):
            raise ValueError(f"rebind requires unchanged {field_name}")
    binding = _build_scene_binding(scene_context, reference_time_s=event.key.reference_time_s)
    if (
        binding == original
        and binding.current_scene_time_s.hex() == original.current_scene_time_s.hex()
    ):
        return event
    if sum(check.check_id == "scene_alignment" for check in event.preflight.checks) != 1:
        raise ValueError("rebind requires exactly one canonical scene_alignment check")
    preflight = _summarize_preflight(
        _scene_alignment_check(binding) if check.check_id == "scene_alignment" else check
        for check in event.preflight.checks
    )
    return replace(event, scene_binding=binding, preflight=preflight)


def _event_occurrence_token(event: ReplayEvent) -> tuple[Any, ...]:
    binding = event.scene_binding
    return (
        event.key,
        _frame_token(binding.storage_frame_id),
        _frame_token(binding.bound_scene_frame_id),
        _frame_token(binding.current_scene_frame_id),
        binding.pinned,
    )


def _event_order_key(event: ReplayEvent) -> tuple[Any, ...]:
    return (
        _frame_token(event.scene_binding.current_scene_frame_id),
        _frame_token(event.scene_binding.bound_scene_frame_id),
        event.key.reference_time_s,
        event.key.clock_id,
        event.key.acquisition_id,
        event.key.processing_run_id,
        event.key.pair_id,
        _frame_token(event.scene_binding.storage_frame_id),
        event.view_manifest_id,
    )


def _pin_policy(value: str) -> str:
    aliases = {
        "exact": PIN_POLICY_EXACT,
        "exact_scene_binding": PIN_POLICY_EXACT,
        "allow_stale": PIN_POLICY_ALLOW_STALE,
        "pinned": PIN_POLICY_ALLOW_STALE,
        "allow_pinned_stale": PIN_POLICY_ALLOW_STALE,
    }
    try:
        return aliases[value]
    except (KeyError, TypeError) as exc:
        raise ValueError("pin_policy must be exact_scene_binding or allow_pinned_stale") from exc


@dataclass(frozen=True, slots=True)
class ReplayIndex:
    """Order-invariant replay index with idempotent duplicate insertion."""

    events: tuple[ReplayEvent, ...]

    def __post_init__(self) -> None:
        canonical: dict[tuple[Any, ...], ReplayEvent] = {}
        for event in self.events:
            if not isinstance(event, ReplayEvent):
                raise TypeError("ReplayIndex accepts ReplayEvent values only")
            token = _event_occurrence_token(event)
            previous = canonical.get(token)
            if previous is None:
                canonical[token] = event
            elif previous.view_manifest_id != event.view_manifest_id:
                raise ValueError("conflicting replay events share one exact event/scene occurrence")
        object.__setattr__(self, "events", tuple(sorted(canonical.values(), key=_event_order_key)))

    def find(
        self,
        *,
        current_scene_frame_id: FrameId,
        pin_policy: str = PIN_POLICY_EXACT,
        storage_frame_id: FrameId | None = None,
        bound_scene_frame_id: FrameId | None = None,
        bound_scene_time_s: float | None = None,
        current_scene_time_s: float | None = None,
        clock_id: str | None = None,
        acquisition_id: str | None = None,
        processing_run_id: str | None = None,
        pair_id: str | None = None,
    ) -> ReplayEvent | None:
        """Return the deterministic best exact/pinned event, or ``None``."""

        current_id = _frame_id(current_scene_frame_id, path="current_scene_frame_id")
        policy = _pin_policy(pin_policy)
        storage_id = (
            _frame_id(storage_frame_id, path="storage_frame_id")
            if storage_frame_id is not None
            else None
        )
        bound_id = (
            _frame_id(bound_scene_frame_id, path="bound_scene_frame_id")
            if bound_scene_frame_id is not None
            else None
        )
        bound_time = (
            _finite_float(bound_scene_time_s, path="bound_scene_time_s")
            if bound_scene_time_s is not None
            else None
        )
        current_time = (
            _finite_float(current_scene_time_s, path="current_scene_time_s")
            if current_scene_time_s is not None
            else None
        )
        candidates: list[ReplayEvent] = []
        for event in self.events:
            binding = event.scene_binding
            if binding.current_scene_frame_id != current_id:
                continue
            if storage_id is not None and binding.storage_frame_id != storage_id:
                continue
            if bound_id is not None and binding.bound_scene_frame_id != bound_id:
                continue
            if bound_time is not None and not math.isclose(
                binding.bound_scene_time_s,
                bound_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            if current_time is not None and not math.isclose(
                binding.current_scene_time_s,
                current_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            if clock_id is not None and event.key.clock_id != clock_id:
                continue
            if acquisition_id is not None and event.key.acquisition_id != acquisition_id:
                continue
            if processing_run_id is not None and event.key.processing_run_id != processing_run_id:
                continue
            if pair_id is not None and event.key.pair_id != pair_id:
                continue
            exact = binding.bound_scene_frame_id == current_id
            if policy == PIN_POLICY_EXACT and not exact:
                continue
            if policy == PIN_POLICY_ALLOW_STALE and not (
                exact or (binding.stale and binding.pinned)
            ):
                continue
            candidates.append(event)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda event: (
                event.scene_binding.fresh,
                event.key.reference_time_s,
                event.key.processing_run_id,
                event.key.acquisition_id,
                _frame_token(event.scene_binding.storage_frame_id),
                event.view_manifest_id,
            ),
        )

    def select(self, **criteria: Any) -> ReplayEvent:
        """Strict counterpart to :meth:`find`; fail if no exact policy match exists."""

        event = self.find(**criteria)
        if event is None:
            raise LookupError("no replay event satisfies the exact scene binding and pin policy")
        return event

    select_by_scene_binding = select


def build_replay_index(events: Iterable[ReplayEvent]) -> ReplayIndex:
    return ReplayIndex(tuple(events))


def select_replay_event(
    events: ReplayIndex | Iterable[ReplayEvent], **criteria: Any
) -> ReplayEvent:
    index = events if isinstance(events, ReplayIndex) else build_replay_index(events)
    return index.select(**criteria)


@dataclass(frozen=True, slots=True)
class ReplayValueDiff:
    path: str
    left: Any
    right: Any

    def to_manifest(self) -> dict[str, Any]:
        return {"left": _thaw(self.left), "path": self.path, "right": _thaw(self.right)}


@dataclass(frozen=True, slots=True)
class ReplayReferenceDiff:
    kind: str
    layer: EvidenceLayer
    shared: tuple[str, ...]
    left_only: tuple[str, ...]
    right_only: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "left_only": list(self.left_only),
            "right_only": list(self.right_only),
            "shared": list(self.shared),
        }


@dataclass(frozen=True, slots=True)
class ReplayProductDiff:
    data_key: str
    kind: str
    variant: str
    status: Literal["added", "removed", "changed", "unchanged"]
    left_product_id: str | None
    right_product_id: str | None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "data_key": self.data_key,
            "kind": self.kind,
            "left_product_id": self.left_product_id,
            "right_product_id": self.right_product_id,
            "status": self.status,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class ReplayABComparison:
    """Deterministic same-source comparison between two different recipes."""

    left_key: ReplayEventKey
    right_key: ReplayEventKey
    source_samples_id: str
    root_source_product_ids: tuple[str, ...]
    parameter_diff: tuple[ReplayValueDiff, ...]
    result_diff: tuple[ReplayReferenceDiff, ...]
    truth_diff: tuple[ReplayReferenceDiff, ...]
    product_diff: tuple[ReplayProductDiff, ...]

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "left_event_key": self.left_key.to_manifest(),
            "parameter_diff": [item.to_manifest() for item in self.parameter_diff],
            "product_diff": [item.to_manifest() for item in self.product_diff],
            "result_diff": [item.to_manifest() for item in self.result_diff],
            "right_event_key": self.right_key.to_manifest(),
            "root_source_product_ids": list(self.root_source_product_ids),
            "source_samples_id": self.source_samples_id,
            "truth_diff": [item.to_manifest() for item in self.truth_diff],
        }

    @property
    def comparison_id(self) -> str:
        return str(canonical_sensing_hash(self.manifest_payload()))

    def to_manifest(self) -> dict[str, Any]:
        manifest = self.manifest_payload()
        manifest["comparison_id"] = self.comparison_id
        return manifest

    def to_json(self) -> str:
        return json.dumps(
            self.to_manifest(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )


def _value_diff(left: Any, right: Any, *, path: str) -> list[ReplayValueDiff]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[ReplayValueDiff] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                differences.append(
                    ReplayValueDiff(
                        child_path,
                        freeze_replay_json(_MISSING),
                        freeze_replay_json(right[key], path=child_path),
                    )
                )
            elif key not in right:
                differences.append(
                    ReplayValueDiff(
                        child_path,
                        freeze_replay_json(left[key], path=child_path),
                        freeze_replay_json(_MISSING),
                    )
                )
            else:
                differences.extend(_value_diff(left[key], right[key], path=child_path))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        differences = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}[{index}]"
            if index >= len(left):
                differences.append(
                    ReplayValueDiff(
                        child_path,
                        freeze_replay_json(_MISSING),
                        freeze_replay_json(right[index], path=child_path),
                    )
                )
            elif index >= len(right):
                differences.append(
                    ReplayValueDiff(
                        child_path,
                        freeze_replay_json(left[index], path=child_path),
                        freeze_replay_json(_MISSING),
                    )
                )
            else:
                differences.extend(_value_diff(left[index], right[index], path=child_path))
        return differences
    if left == right:
        return []
    return [
        ReplayValueDiff(
            path,
            freeze_replay_json(left, path=f"{path}.left"),
            freeze_replay_json(right, path=f"{path}.right"),
        )
    ]


def _reference_diffs(
    left: ReplayEvent,
    right: ReplayEvent,
    *,
    layers: set[EvidenceLayer],
) -> tuple[ReplayReferenceDiff, ...]:
    groups = sorted(
        {
            (reference.kind, reference.layer)
            for event in (left, right)
            for reference in event.evidence
            if reference.layer in layers
        }
    )
    differences: list[ReplayReferenceDiff] = []
    for kind, layer in groups:
        left_ids = {
            reference.reference_id
            for reference in left.evidence
            if reference.kind == kind and reference.layer == layer
        }
        right_ids = {
            reference.reference_id
            for reference in right.evidence
            if reference.kind == kind and reference.layer == layer
        }
        differences.append(
            ReplayReferenceDiff(
                kind=kind,
                layer=layer,
                shared=tuple(sorted(left_ids & right_ids)),
                left_only=tuple(sorted(left_ids - right_ids)),
                right_only=tuple(sorted(right_ids - left_ids)),
            )
        )
    return tuple(differences)


def _product_diffs(left: ReplayEvent, right: ReplayEvent) -> tuple[ReplayProductDiff, ...]:
    def index(event: ReplayEvent) -> dict[tuple[str, str, str], ReplayProductReference]:
        return {
            (product.data_key, product.kind, product.variant): product for product in event.products
        }

    left_products = index(left)
    right_products = index(right)
    differences: list[ReplayProductDiff] = []
    for key in sorted(set(left_products) | set(right_products)):
        left_product = left_products.get(key)
        right_product = right_products.get(key)
        status: Literal["added", "removed", "changed", "unchanged"]
        if left_product is None:
            status = "added"
        elif right_product is None:
            status = "removed"
        elif left_product.product_id == right_product.product_id:
            status = "unchanged"
        else:
            status = "changed"
        differences.append(
            ReplayProductDiff(
                data_key=key[0],
                kind=key[1],
                variant=key[2],
                status=status,
                left_product_id=left_product.product_id if left_product else None,
                right_product_id=right_product.product_id if right_product else None,
            )
        )
    return tuple(differences)


def compare_replay_events(first: ReplayEvent, second: ReplayEvent) -> ReplayABComparison:
    """Compare two runs only when their acquisition sources are exactly equal."""

    if not isinstance(first, ReplayEvent) or not isinstance(second, ReplayEvent):
        raise TypeError("A/B comparison requires two ReplayEvent values")
    for field_name in ("clock_id", "acquisition_id", "pair_id"):
        if getattr(first.key, field_name) != getattr(second.key, field_name):
            raise ValueError(f"A/B comparison requires the same {field_name}")
    if first.source_samples_id != second.source_samples_id:
        raise ValueError("A/B comparison requires exactly the same ordered source samples")
    if not first.root_source_product_ids or not second.root_source_product_ids:
        raise ValueError("A/B comparison requires root source-product identities")
    if first.root_source_product_ids != second.root_source_product_ids:
        raise ValueError("A/B comparison requires the same root source-product identities")
    if first.key.processing_run_id == second.key.processing_run_id:
        raise ValueError("A/B comparison requires different processing runs")
    if first.recipe.canonical_hash == second.recipe.canonical_hash:
        raise ValueError("A/B comparison requires different processing recipes")

    left, right = sorted(
        (first, second),
        key=lambda event: (
            event.recipe.canonical_hash,
            event.key.processing_run_id,
            event.manifest_id,
        ),
    )
    parameter_diff = _value_diff(
        {
            "processing_plan": left.recipe.processing_plan,
            "recipe": left.recipe.parameters,
        },
        {
            "processing_plan": right.recipe.processing_plan,
            "recipe": right.recipe.parameters,
        },
        path="parameters",
    )
    return ReplayABComparison(
        left_key=left.key,
        right_key=right.key,
        source_samples_id=left.source_samples_id,
        root_source_product_ids=left.root_source_product_ids,
        parameter_diff=tuple(parameter_diff),
        result_diff=_reference_diffs(left, right, layers={"estimate", "oracle"}),
        truth_diff=_reference_diffs(left, right, layers={"truth"}),
        product_diff=_product_diffs(left, right),
    )


compare_replay_ab = compare_replay_events


__all__ = [
    "PIN_POLICY_ALLOW_STALE",
    "PIN_POLICY_EXACT",
    "REPLAY_MANIFEST_VERSION",
    "FrozenMap",
    "ReplayABComparison",
    "ReplayCredibilityPreflight",
    "ReplayEvent",
    "ReplayEventKey",
    "ReplayEvidenceReference",
    "ReplayIndex",
    "ReplayPreflightCheck",
    "ReplayPairPose",
    "ReplayProductDiff",
    "ReplayProductReference",
    "ReplayPulse",
    "ReplayRecipeSnapshot",
    "ReplayReferenceDiff",
    "ReplaySceneBinding",
    "ReplaySceneContext",
    "ReplayValueDiff",
    "build_replay_event",
    "build_replay_index",
    "compare_replay_ab",
    "compare_replay_events",
    "freeze_replay_json",
    "rebind_replay_event",
    "select_replay_event",
]
