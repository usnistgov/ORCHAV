"""Typed boundary for already-calibrated sensing measurements.

This module deliberately starts *after* calibration.  It loads a small JSON
manifest plus numeric ``.npy`` or ``.npz`` products, validates their scientific
identity and provenance, and compares them with simulation products through an
explicit acquisition/alignment map.  It never resolves a calibration-record
reference or imports calibration implementation code.

The v1 contract is intentionally strict: acquisition identities are stable
and non-ordinal, timestamps are ordered, every numeric product is
content-addressed, and missing uncertainty or calibration metadata is
represented explicitly rather than inferred from absence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import UUID

import numpy as np

MEASUREMENT_CONTRACT_VERSION = 1
MEASUREMENT_COMPARISON_CONTRACT_VERSION = 1
MEASUREMENT_ALIGNMENT_CONTRACT_VERSION = 1
INDEPENDENT_VALUE_UNCERTAINTY_POLICY = "mutually_uncorrelated_product_value_standard_uncertainties"
RESIDUAL_DEFINITION = "axis_aligned_measurement_minus_simulation"
AGGREGATE_WEIGHTING_POLICY = "equal_weight_per_residual_element_across_selected_products"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_ORDINAL_ID_RE = re.compile(r"^(?:(?:frame|sample|pulse|cpi|index|idx)[-_ :]*|)\d+$", re.I)
_SOURCE_KINDS = frozenset({"measurement", "simulation"})
_DATASET_ROLES = frozenset({"calibration", "tuning", "evaluation", "demonstration"})
_REDISTRIBUTION_VALUES = frozenset({"permitted", "prohibited", "user_supplied_only"})
_PRIVACY_VALUES = frozenset({"none", "deidentified", "restricted"})
_MISSING_REASONS = frozenset(
    {"not_applicable", "not_measured", "not_provided", "redacted", "unknown"}
)
_REPRESENTATION_UNITS = {
    "complex_voltage": "1",
    "log_power_ratio_db": "dB",
    "power_linear": "1",
}
_PORTABLE_NUMERIC_DTYPES = frozenset({("f", 4), ("f", 8), ("c", 8), ("c", 16)})
_SIGN_REVERSAL_AXES = {
    (
        "doppler_frequency_positive_approaching",
        "doppler_frequency_positive_receding",
    ): "doppler_frequency",
    (
        "doppler_frequency_positive_receding",
        "doppler_frequency_positive_approaching",
    ): "doppler_frequency",
    (
        "radial_velocity_positive_approaching",
        "radial_velocity_positive_receding",
    ): "radial_velocity",
    (
        "radial_velocity_positive_receding",
        "radial_velocity_positive_approaching",
    ): "radial_velocity",
}
_FORBIDDEN_PATH_PARTS = frozenset(
    {
        "_internal",
        "_private",
        "calibration",
        "internal",
        "private",
        "restricted",
        "sensingmeasurements",
    }
)


class MeasurementContractError(ValueError):
    """Raised when an already-calibrated measurement contract is invalid."""


def canonical_measurement_sha256(value: Any, *, domain: str) -> str:
    """Hash JSON-compatible *value* with an explicit measurement domain."""

    if not isinstance(domain, str) or not domain.strip():
        raise MeasurementContractError("domain must be a non-empty string")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MeasurementContractError("measurement hash input must be finite JSON data") from exc
    digest = hashlib.sha256()
    digest.update(f"orchav.sensing.measurement.{domain}.v1\0".encode("ascii"))
    digest.update(encoded)
    return digest.hexdigest()


def _exact_mapping(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeasurementContractError(f"{path} must be a mapping")
    keys = set(value)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required - optional)
    if missing:
        raise MeasurementContractError(f"{path} is missing required fields {missing!r}")
    if unexpected:
        raise MeasurementContractError(f"{path} has unexpected fields {unexpected!r}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MeasurementContractError(f"measurement JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MeasurementContractError(f"{path} must be a non-empty trimmed string")
    return value


def _stable_identity(
    value: Any,
    *,
    path: str,
    allow_bare_sha256: bool = False,
) -> str:
    candidate = _string(value, path=path)
    try:
        if str(UUID(candidate)) == candidate:
            return candidate
    except (ValueError, AttributeError):
        pass
    if candidate.startswith("sha256:") and _SHA256_RE.fullmatch(candidate[7:]):
        return candidate
    if allow_bare_sha256 and _SHA256_RE.fullmatch(candidate):
        return candidate
    if ":" in candidate and not _ORDINAL_ID_RE.fullmatch(candidate.rsplit(":", 1)[-1]):
        return candidate
    raise MeasurementContractError(
        f"{path} must be a canonical UUID, sha256 identity, or non-ordinal namespaced ID"
    )


def _sha256(value: Any, *, path: str) -> str:
    candidate = _string(value, path=path)
    if _SHA256_RE.fullmatch(candidate) is None:
        raise MeasurementContractError(f"{path} must be a lowercase SHA-256 digest")
    return candidate


def _finite(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementContractError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise MeasurementContractError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise MeasurementContractError(f"{path} must be a finite number")
    return result


def _stable_sample_id(value: Any, *, path: str) -> str:
    candidate = _string(value, path=path)
    if _ORDINAL_ID_RE.fullmatch(candidate):
        raise MeasurementContractError(f"{path} must be a stable source identity, not an ordinal")
    try:
        if str(UUID(candidate)) == candidate:
            return candidate
    except ValueError:
        pass
    if ":" not in candidate:
        raise MeasurementContractError(
            f"{path} must be a canonical UUID or namespaced stable identity"
        )
    if _ORDINAL_ID_RE.fullmatch(candidate.rsplit(":", 1)[-1]):
        raise MeasurementContractError(f"{path} must not hide an ordinal behind a namespace")
    return candidate


def _opaque_reference(value: Any, *, path: str) -> str:
    candidate = _string(value, path=path)
    pure = PurePosixPath(candidate)
    if (
        candidate.startswith(("/", "\\"))
        or "\\" in candidate
        or candidate.lower().startswith("file:")
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise MeasurementContractError(f"{path} must be an opaque ID, not a filesystem path")
    if ":" not in candidate:
        raise MeasurementContractError(f"{path} must be a namespaced opaque ID")
    return candidate


def _safe_product_path(value: Any, *, root: Path, path: str) -> tuple[str, Path]:
    raw = _string(value, path=path)
    if "\\" in raw or ":" in raw:
        raise MeasurementContractError(f"{path} must be a portable relative POSIX path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts:
        raise MeasurementContractError(f"{path} must be a relative path")
    if any(
        part in {"", ".", ".."} or part.startswith(".") or part.casefold() in _FORBIDDEN_PATH_PARTS
        for part in relative.parts
    ):
        raise MeasurementContractError(f"{path} contains a private or unsafe path component")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise MeasurementContractError(f"{path} must not traverse a symlink")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise MeasurementContractError(f"{path} does not resolve inside the manifest root") from exc
    if candidate.suffix not in {".npy", ".npz"}:
        raise MeasurementContractError(f"{path} must reference a .npy or .npz product")
    if not candidate.is_file():
        raise MeasurementContractError(f"{path} must reference a regular file")
    return raw, candidate


def _readonly_array(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    # Back the exposed view with immutable ``bytes``.  Merely clearing NumPy's
    # writable flag is reversible when an array owns its storage, which would
    # let a caller mutate values after their identities were authenticated.
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _stable_rms(magnitudes: np.ndarray, *, path: str) -> float:
    """Return a finite RMS without squaring values at their original scale."""

    values = np.asarray(magnitudes, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise MeasurementContractError(f"{path} requires non-empty finite magnitudes")
    scale = float(np.max(values))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = scale * float(np.sqrt(np.mean(np.square(values / scale))))
    if not math.isfinite(result):
        raise MeasurementContractError(f"{path} is not representable as a finite float")
    return result


def _stable_complex_mean(values: np.ndarray, *, path: str) -> complex:
    """Return a finite complex mean without overflowing the accumulation."""

    array = np.asarray(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise MeasurementContractError(f"{path} requires non-empty finite values")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0 + 0.0j
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = complex(np.mean(array / scale)) * scale
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise MeasurementContractError(f"{path} is not representable as a finite complex value")
    return result


@dataclass(frozen=True)
class MissingField:
    """An explicit reason why a contract field has no value."""

    reason: str


@dataclass(frozen=True)
class StandardUncertainty:
    """Scalar standard uncertainty, or an explicit missing-value reason."""

    unit: str
    standard_deviation: float | None
    missing: MissingField | None = None

    @property
    def is_known(self) -> bool:
        return self.missing is None


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    source_kind: Literal["measurement", "simulation"]
    role: str
    split_id: str
    split_sha256: str


@dataclass(frozen=True)
class AcquisitionSample:
    """One ordered source sample, including an explicitly missing slot."""

    acquisition_id: str
    sample_id: str
    timestamp_s: float
    valid: bool
    missing: MissingField | None = None


@dataclass(frozen=True)
class SensorPair:
    pair_id: str
    tx_sensor_id: str
    rx_sensor_id: str
    topology: str


@dataclass(frozen=True)
class GovernanceRecord:
    """Minimum provenance, redistribution, privacy, and release authority."""

    provenance_id: str
    license_spdx: str
    redistribution: str
    privacy: str
    authority_id: str


@dataclass(frozen=True)
class ClockReference:
    clock_id: str
    origin: str
    unit: Literal["s"] = "s"


@dataclass(frozen=True)
class CalibrationRecord:
    """Opaque calibration provenance; no referenced content is resolved here."""

    record_id: str | None
    sha256: str | None
    scope: CalibrationScope | None = None
    missing: MissingField | None = None

    @property
    def is_known(self) -> bool:
        return self.missing is None


@dataclass(frozen=True)
class ProcessingLineage:
    processing_run_id: str
    processor: str
    processor_version: str
    recipe_sha256: str
    source_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationScope:
    """Exact products/acquisitions to which an opaque calibration record applies."""

    pair_ids: tuple[str, ...]
    acquisition_ids: tuple[str, ...]
    observable_variants: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReferenceTransform:
    transform_id: str
    from_frame: str
    to_frame: str
    matrix: np.ndarray = field(repr=False, compare=False)
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _readonly_array(self.matrix, dtype=np.float64))


@dataclass(frozen=True)
class ReferenceChain:
    source_frame: str
    target_frame: str
    transforms: tuple[ReferenceTransform, ...]
    sha256: str


@dataclass(frozen=True)
class MeasurementAxis:
    name: str
    unit: str
    reference_frame: str
    coordinates: np.ndarray = field(repr=False, compare=False)
    uncertainty: StandardUncertainty

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinates",
            _readonly_array(self.coordinates, dtype=np.float64),
        )


@dataclass(frozen=True)
class MeasurementProduct:
    product_id: str
    acquisition_id: str
    pair_id: str
    sample_id: str
    timestamp_s: float
    observable: str
    variant: str
    representation: str
    unit: str
    sign_convention: str
    axes: tuple[MeasurementAxis, ...]
    values: np.ndarray = field(repr=False, compare=False)
    uncertainty: StandardUncertainty
    data_path: str
    data_sha256: str
    product_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_array(self.values))


@dataclass(frozen=True)
class MeasurementManifest:
    contract_version: int
    dataset: DatasetIdentity
    governance: GovernanceRecord
    acquisitions: tuple[AcquisitionSample, ...]
    sensor_topology: tuple[SensorPair, ...]
    clock: ClockReference
    calibration_record: CalibrationRecord
    reference_chain: ReferenceChain
    processing_lineage: ProcessingLineage
    products: tuple[MeasurementProduct, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class AcquisitionPair:
    """Explicit stable-identity mapping; positional matching is never supported."""

    sample_id: str
    measurement_acquisition_id: str
    simulation_acquisition_id: str

    def __post_init__(self) -> None:
        _stable_sample_id(self.sample_id, path="alignment.sample_id")
        _stable_identity(
            self.measurement_acquisition_id,
            path="alignment.measurement_acquisition_id",
        )
        _stable_identity(
            self.simulation_acquisition_id,
            path="alignment.simulation_acquisition_id",
        )


@dataclass(frozen=True)
class AxisAlignment:
    name: str
    unit: str
    measurement_to_simulation_offset: float
    measurement_to_simulation_scale: int = 1
    absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        _string(self.name, path="axis_alignment.name")
        _string(self.unit, path="axis_alignment.unit")
        _finite(
            self.measurement_to_simulation_offset,
            path="axis_alignment.measurement_to_simulation_offset",
        )
        if isinstance(
            self.measurement_to_simulation_scale, bool
        ) or self.measurement_to_simulation_scale not in {-1, 1}:
            raise MeasurementContractError("axis alignment scale must be exactly -1 or 1")
        tolerance = _finite(self.absolute_tolerance, path="axis_alignment.absolute_tolerance")
        if tolerance < 0.0:
            raise MeasurementContractError("axis alignment tolerance cannot be negative")


@dataclass(frozen=True)
class MeasurementAlignment:
    acquisition_pairs: tuple[AcquisitionPair, ...]
    axis_alignments: tuple[AxisAlignment, ...]
    measurement_to_simulation_time_offset_s: float
    timestamp_tolerance_s: float = 1e-12

    def __post_init__(self) -> None:
        if not self.acquisition_pairs:
            raise MeasurementContractError("alignment requires explicit acquisition pairs")
        sample_ids = [item.sample_id for item in self.acquisition_pairs]
        measurement_ids = [item.measurement_acquisition_id for item in self.acquisition_pairs]
        simulation_ids = [item.simulation_acquisition_id for item in self.acquisition_pairs]
        for values, label in (
            (sample_ids, "sample IDs"),
            (measurement_ids, "measurement acquisition IDs"),
            (simulation_ids, "simulation acquisition IDs"),
        ):
            if len(values) != len(set(values)):
                raise MeasurementContractError(f"alignment has duplicate {label}")
        names = [axis.name for axis in self.axis_alignments]
        if len(names) != len(set(names)):
            raise MeasurementContractError("alignment has duplicate axis names")
        _finite(
            self.measurement_to_simulation_time_offset_s,
            path="alignment.measurement_to_simulation_time_offset_s",
        )
        tolerance = _finite(
            self.timestamp_tolerance_s,
            path="alignment.timestamp_tolerance_s",
        )
        if tolerance < 0.0:
            raise MeasurementContractError("timestamp tolerance cannot be negative")


@dataclass(frozen=True)
class MeasurementAlignmentRecipe:
    """Versioned, content-addressed alignment bound to exact input manifests."""

    contract_version: int
    alignment_id: str
    alignment_sha256: str
    measurement_dataset_id: str
    simulation_dataset_id: str
    measurement_manifest_sha256: str
    simulation_manifest_sha256: str
    split_sha256: str
    observable: str
    variant: str
    uncertainty_combination_policy: str
    alignment: MeasurementAlignment


@dataclass(frozen=True)
class ProductResidual:
    sample_id: str
    measurement_acquisition_id: str
    simulation_acquisition_id: str
    simulation_timestamp_s: float
    timestamp_residual_s: float
    residual: np.ndarray = field(repr=False, compare=False)
    rmse: float
    max_absolute_error: float
    mean_residual: complex
    combined_standard_uncertainty: StandardUncertainty
    normalized_rmse: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual", _readonly_array(self.residual))


@dataclass(frozen=True)
class MeasurementComparison:
    contract_version: int
    comparison_id: str
    alignment_id: str
    alignment_sha256: str
    measurement_dataset_id: str
    simulation_dataset_id: str
    measurement_governance: GovernanceRecord
    simulation_governance: GovernanceRecord
    dataset_role: str
    split_id: str
    split_sha256: str
    observable: str
    variant: str
    representation: str
    unit: str
    sign_convention: str
    uncertainty_combination_policy: str
    residual_definition: str
    aggregate_weighting_policy: str
    reference_chain_sha256: str
    residuals: tuple[ProductResidual, ...]
    aggregate_rmse: float
    aggregate_max_absolute_error: float
    aggregate_mean_residual: complex
    aggregate_normalized_rmse: float | None


def _parse_uncertainty(value: Any, *, path: str, expected_unit: str) -> StandardUncertainty:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"status", "unit"}),
        optional=frozenset({"reason", "standard_deviation"}),
    )
    status = _string(mapping["status"], path=f"{path}.status")
    unit = _string(mapping["unit"], path=f"{path}.unit")
    if unit != expected_unit:
        raise MeasurementContractError(
            f"{path}.unit {unit!r} does not match expected unit {expected_unit!r}"
        )
    if status == "known":
        if set(mapping) != {"status", "unit", "standard_deviation"}:
            raise MeasurementContractError(
                f"{path} known uncertainty requires only standard_deviation"
            )
        deviation = _finite(
            mapping["standard_deviation"],
            path=f"{path}.standard_deviation",
        )
        if deviation < 0.0:
            raise MeasurementContractError(f"{path}.standard_deviation cannot be negative")
        return StandardUncertainty(unit=unit, standard_deviation=deviation)
    if status == "missing":
        if set(mapping) != {"status", "unit", "reason"}:
            raise MeasurementContractError(f"{path} missing uncertainty requires only reason")
        reason = _string(mapping["reason"], path=f"{path}.reason")
        if reason not in _MISSING_REASONS:
            raise MeasurementContractError(f"{path}.reason is not a supported missing reason")
        return StandardUncertainty(
            unit=unit,
            standard_deviation=None,
            missing=MissingField(reason),
        )
    raise MeasurementContractError(f"{path}.status must be 'known' or 'missing'")


def _parse_calibration_record(value: Any, *, path: str) -> CalibrationRecord:
    if not isinstance(value, Mapping):
        raise MeasurementContractError(f"{path} must be a mapping")
    status = _string(value.get("status"), path=f"{path}.status")
    if status == "known":
        mapping = _exact_mapping(
            value,
            path=path,
            required=frozenset({"status", "record_id", "sha256", "scope"}),
        )
        scope_raw = _exact_mapping(
            mapping["scope"],
            path=f"{path}.scope",
            required=frozenset({"pair_ids", "acquisition_ids", "observable_variants"}),
        )
        raw_pair_ids = scope_raw["pair_ids"]
        raw_acquisition_ids = scope_raw["acquisition_ids"]
        raw_observables = scope_raw["observable_variants"]
        if not isinstance(raw_pair_ids, list) or not raw_pair_ids:
            raise MeasurementContractError(f"{path}.scope.pair_ids must be a non-empty list")
        if not isinstance(raw_acquisition_ids, list) or not raw_acquisition_ids:
            raise MeasurementContractError(f"{path}.scope.acquisition_ids must be a non-empty list")
        if not isinstance(raw_observables, list) or not raw_observables:
            raise MeasurementContractError(
                f"{path}.scope.observable_variants must be a non-empty list"
            )
        pair_ids = tuple(
            _stable_identity(
                item,
                path=f"{path}.scope.pair_ids[{index}]",
                allow_bare_sha256=True,
            )
            for index, item in enumerate(raw_pair_ids)
        )
        acquisition_ids = tuple(
            _stable_identity(item, path=f"{path}.scope.acquisition_ids[{index}]")
            for index, item in enumerate(raw_acquisition_ids)
        )
        observable_variants: list[tuple[str, str]] = []
        for index, item in enumerate(raw_observables):
            observable_raw = _exact_mapping(
                item,
                path=f"{path}.scope.observable_variants[{index}]",
                required=frozenset({"observable", "variant"}),
            )
            observable_variants.append(
                (
                    _string(
                        observable_raw["observable"],
                        path=f"{path}.scope.observable_variants[{index}].observable",
                    ),
                    _string(
                        observable_raw["variant"],
                        path=f"{path}.scope.observable_variants[{index}].variant",
                    ),
                )
            )
        for values, label in (
            (pair_ids, "pair_ids"),
            (acquisition_ids, "acquisition_ids"),
            (tuple(observable_variants), "observable_variants"),
        ):
            if len(values) != len(set(values)):
                raise MeasurementContractError(f"{path}.scope.{label} contains duplicates")
        return CalibrationRecord(
            record_id=_opaque_reference(mapping["record_id"], path=f"{path}.record_id"),
            sha256=_sha256(mapping["sha256"], path=f"{path}.sha256"),
            scope=CalibrationScope(pair_ids, acquisition_ids, tuple(observable_variants)),
        )
    if status == "missing":
        mapping = _exact_mapping(
            value,
            path=path,
            required=frozenset({"status", "reason"}),
        )
        reason = _string(mapping["reason"], path=f"{path}.reason")
        if reason not in _MISSING_REASONS:
            raise MeasurementContractError(f"{path}.reason is not a supported missing reason")
        return CalibrationRecord(record_id=None, sha256=None, missing=MissingField(reason))
    raise MeasurementContractError(f"{path}.status must be 'known' or 'missing'")


def _parse_transform(value: Any, *, path: str) -> ReferenceTransform:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"transform_id", "from_frame", "to_frame", "matrix", "sha256"}),
    )
    expected_hash = canonical_measurement_sha256(
        {key: item for key, item in mapping.items() if key != "sha256"},
        domain="reference_transform",
    )
    declared_hash = _sha256(mapping["sha256"], path=f"{path}.sha256")
    if declared_hash != expected_hash:
        raise MeasurementContractError(f"{path}.sha256 does not match transform metadata")
    try:
        matrix = np.asarray(mapping["matrix"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MeasurementContractError(f"{path}.matrix must be a finite 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise MeasurementContractError(f"{path}.matrix must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
        raise MeasurementContractError(f"{path}.matrix must be an affine transform")
    rotation = matrix[:3, :3]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            gram = rotation.T @ rotation
            determinant = float(np.linalg.det(rotation))
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise MeasurementContractError(
            f"{path}.matrix rotation must have finite orthonormal geometry"
        ) from exc
    if not np.all(np.isfinite(gram)) or not math.isfinite(determinant):
        raise MeasurementContractError(
            f"{path}.matrix rotation must have finite orthonormal geometry"
        )
    if not np.allclose(gram, np.eye(3), atol=1e-9, rtol=0.0):
        raise MeasurementContractError(f"{path}.matrix rotation must be orthonormal")
    if not math.isclose(determinant, 1.0, abs_tol=1e-9):
        raise MeasurementContractError(f"{path}.matrix must preserve handedness")
    return ReferenceTransform(
        transform_id=_stable_identity(mapping["transform_id"], path=f"{path}.transform_id"),
        from_frame=_string(mapping["from_frame"], path=f"{path}.from_frame"),
        to_frame=_string(mapping["to_frame"], path=f"{path}.to_frame"),
        matrix=matrix,
        sha256=declared_hash,
    )


def _parse_reference_chain(value: Any, *, path: str) -> ReferenceChain:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"source_frame", "target_frame", "transforms", "sha256"}),
    )
    expected_hash = canonical_measurement_sha256(
        {key: item for key, item in mapping.items() if key != "sha256"},
        domain="reference_chain",
    )
    declared_hash = _sha256(mapping["sha256"], path=f"{path}.sha256")
    if declared_hash != expected_hash:
        raise MeasurementContractError(f"{path}.sha256 does not match reference metadata")
    source = _string(mapping["source_frame"], path=f"{path}.source_frame")
    target = _string(mapping["target_frame"], path=f"{path}.target_frame")
    raw_transforms = mapping["transforms"]
    if not isinstance(raw_transforms, list):
        raise MeasurementContractError(f"{path}.transforms must be a list")
    transforms = tuple(
        _parse_transform(item, path=f"{path}.transforms[{index}]")
        for index, item in enumerate(raw_transforms)
    )
    if transforms:
        if transforms[0].from_frame != source or transforms[-1].to_frame != target:
            raise MeasurementContractError(f"{path}.transforms do not bind chain endpoints")
        for previous, following in zip(transforms, transforms[1:]):
            if previous.to_frame != following.from_frame:
                raise MeasurementContractError(f"{path}.transforms are not a continuous chain")
    elif source != target:
        raise MeasurementContractError(
            f"{path} requires transforms when source_frame differs from target_frame"
        )
    transform_ids = [item.transform_id for item in transforms]
    if len(transform_ids) != len(set(transform_ids)):
        raise MeasurementContractError(f"{path}.transforms contain duplicate identities")
    return ReferenceChain(source, target, transforms, declared_hash)


def _parse_axis(value: Any, *, path: str, expected_size: int) -> MeasurementAxis:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"name", "unit", "reference_frame", "coordinates", "uncertainty"}),
    )
    name = _string(mapping["name"], path=f"{path}.name")
    unit = _string(mapping["unit"], path=f"{path}.unit")
    try:
        coordinates = np.asarray(mapping["coordinates"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MeasurementContractError(
            f"{path}.coordinates must contain {expected_size} finite values"
        ) from exc
    if coordinates.shape != (expected_size,) or not np.all(np.isfinite(coordinates)):
        raise MeasurementContractError(
            f"{path}.coordinates must contain {expected_size} finite values"
        )
    if expected_size > 1 and np.any(coordinates[1:] <= coordinates[:-1]):
        raise MeasurementContractError(f"{path}.coordinates must be strictly increasing")
    return MeasurementAxis(
        name=name,
        unit=unit,
        reference_frame=_string(
            mapping["reference_frame"],
            path=f"{path}.reference_frame",
        ),
        coordinates=coordinates,
        uncertainty=_parse_uncertainty(
            mapping["uncertainty"],
            path=f"{path}.uncertainty",
            expected_unit=unit,
        ),
    )


def _load_numeric_product(
    value: Any,
    *,
    root: Path,
    path: str,
) -> tuple[str, str, np.ndarray]:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"path", "sha256", "dtype", "shape"}),
        optional=frozenset({"array_key"}),
    )
    relative_path, source = _safe_product_path(mapping["path"], root=root, path=f"{path}.path")
    declared_hash = _sha256(mapping["sha256"], path=f"{path}.sha256")
    try:
        authenticated_bytes = source.read_bytes()
    except OSError as exc:
        raise MeasurementContractError(f"{path} could not be read safely") from exc
    if hashlib.sha256(authenticated_bytes).hexdigest() != declared_hash:
        raise MeasurementContractError(f"{path}.sha256 does not match product bytes")
    authenticated_stream = io.BytesIO(authenticated_bytes)
    try:
        if source.suffix == ".npy":
            if "array_key" in mapping:
                raise MeasurementContractError(f"{path}.array_key is invalid for .npy")
            loaded = np.load(authenticated_stream, allow_pickle=False)
        else:
            key = _string(mapping.get("array_key"), path=f"{path}.array_key")
            with np.load(authenticated_stream, allow_pickle=False) as archive:
                if archive.files != [key]:
                    raise MeasurementContractError(
                        f"{path} .npz archive must contain exactly the declared array_key"
                    )
                loaded = np.array(archive[key], copy=True)
    except (OSError, ValueError) as exc:
        if isinstance(exc, MeasurementContractError):
            raise
        raise MeasurementContractError(f"{path} could not be loaded safely") from exc
    array = np.asarray(loaded)
    if array.dtype.hasobject:
        raise MeasurementContractError(f"{path} cannot contain object data")
    if array.size == 0:
        raise MeasurementContractError(f"{path} cannot contain an empty numeric product")
    declared_dtype = _string(mapping["dtype"], path=f"{path}.dtype")
    if array.dtype.str != declared_dtype:
        raise MeasurementContractError(
            f"{path}.dtype {declared_dtype!r} does not match {array.dtype.str!r}"
        )
    raw_shape = mapping["shape"]
    if not isinstance(raw_shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in raw_shape
    ):
        raise MeasurementContractError(f"{path}.shape must be a list of nonnegative integers")
    if tuple(raw_shape) != array.shape:
        raise MeasurementContractError(f"{path}.shape does not match numeric data")
    if (array.dtype.kind, array.dtype.itemsize) not in _PORTABLE_NUMERIC_DTYPES:
        raise MeasurementContractError(
            f"{path} dtype must be float32, float64, complex64, or complex128; "
            "either byte order is admitted and retained in the authenticated metadata"
        )
    if not np.all(np.isfinite(array)):
        raise MeasurementContractError(f"{path} numeric data must be finite")
    return relative_path, declared_hash, array


def _parse_acquisition(value: Any, *, path: str) -> AcquisitionSample:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"acquisition_id", "sample_id", "timestamp_s", "valid"}),
        optional=frozenset({"missing_reason"}),
    )
    valid = mapping["valid"]
    if not isinstance(valid, bool):
        raise MeasurementContractError(f"{path}.valid must be a boolean")
    if valid and "missing_reason" in mapping:
        raise MeasurementContractError(f"{path} valid samples cannot declare missing_reason")
    missing: MissingField | None = None
    if not valid:
        if "missing_reason" not in mapping:
            raise MeasurementContractError(
                f"{path} invalid samples require an explicit missing_reason"
            )
        reason = _string(mapping["missing_reason"], path=f"{path}.missing_reason")
        if reason not in _MISSING_REASONS:
            raise MeasurementContractError(f"{path}.missing_reason is unsupported")
        missing = MissingField(reason)
    return AcquisitionSample(
        acquisition_id=_stable_identity(
            mapping["acquisition_id"],
            path=f"{path}.acquisition_id",
        ),
        sample_id=_stable_sample_id(mapping["sample_id"], path=f"{path}.sample_id"),
        timestamp_s=_finite(mapping["timestamp_s"], path=f"{path}.timestamp_s"),
        valid=valid,
        missing=missing,
    )


def _parse_sensor_pair(value: Any, *, path: str) -> SensorPair:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset({"pair_id", "tx_sensor_id", "rx_sensor_id", "topology"}),
    )
    return SensorPair(
        pair_id=_stable_identity(
            mapping["pair_id"],
            path=f"{path}.pair_id",
            allow_bare_sha256=True,
        ),
        tx_sensor_id=_stable_identity(
            mapping["tx_sensor_id"],
            path=f"{path}.tx_sensor_id",
        ),
        rx_sensor_id=_stable_identity(
            mapping["rx_sensor_id"],
            path=f"{path}.rx_sensor_id",
        ),
        topology=_string(mapping["topology"], path=f"{path}.topology"),
    )


def _parse_product(
    value: Any,
    *,
    root: Path,
    path: str,
    acquisitions: Mapping[str, AcquisitionSample],
    pair_ids: frozenset[str],
) -> MeasurementProduct:
    mapping = _exact_mapping(
        value,
        path=path,
        required=frozenset(
            {
                "product_id",
                "acquisition_id",
                "pair_id",
                "observable",
                "variant",
                "representation",
                "unit",
                "sign_convention",
                "axes",
                "data",
                "uncertainty",
                "product_sha256",
            }
        ),
    )
    expected_hash = canonical_measurement_sha256(
        {key: item for key, item in mapping.items() if key != "product_sha256"},
        domain="product",
    )
    declared_hash = _sha256(mapping["product_sha256"], path=f"{path}.product_sha256")
    if declared_hash != expected_hash:
        raise MeasurementContractError(f"{path}.product_sha256 does not match product metadata")
    representation = _string(mapping["representation"], path=f"{path}.representation")
    if representation not in _REPRESENTATION_UNITS:
        raise MeasurementContractError(f"{path}.representation is not supported")
    unit = _string(mapping["unit"], path=f"{path}.unit")
    if unit != _REPRESENTATION_UNITS[representation]:
        raise MeasurementContractError(
            f"{path}.unit does not match representation {representation!r}"
        )
    relative_path, data_hash, values = _load_numeric_product(
        mapping["data"],
        root=root,
        path=f"{path}.data",
    )
    if representation == "complex_voltage":
        if not np.issubdtype(values.dtype, np.complexfloating):
            raise MeasurementContractError(f"{path}.data must be complex floating data")
    elif not np.issubdtype(values.dtype, np.floating):
        raise MeasurementContractError(f"{path}.data must be real floating data")
    elif representation == "power_linear" and np.any(values < 0.0):
        raise MeasurementContractError(f"{path}.data power_linear values must be nonnegative")
    raw_axes = mapping["axes"]
    if not isinstance(raw_axes, list) or len(raw_axes) != values.ndim:
        raise MeasurementContractError(f"{path}.axes must describe every numeric dimension")
    axes = tuple(
        _parse_axis(axis, path=f"{path}.axes[{index}]", expected_size=values.shape[index])
        for index, axis in enumerate(raw_axes)
    )
    axis_names = [axis.name for axis in axes]
    if len(axis_names) != len(set(axis_names)):
        raise MeasurementContractError(f"{path}.axes contain duplicate names")
    acquisition_id = _stable_identity(
        mapping["acquisition_id"],
        path=f"{path}.acquisition_id",
    )
    acquisition = acquisitions.get(acquisition_id)
    if acquisition is None:
        raise MeasurementContractError(f"{path}.acquisition_id is absent from acquisitions")
    if not acquisition.valid:
        raise MeasurementContractError(f"{path} cannot bind an invalid acquisition slot")
    pair_id = _stable_identity(
        mapping["pair_id"],
        path=f"{path}.pair_id",
        allow_bare_sha256=True,
    )
    if pair_id not in pair_ids:
        raise MeasurementContractError(f"{path}.pair_id is absent from sensor_topology")
    return MeasurementProduct(
        product_id=_stable_identity(
            mapping["product_id"],
            path=f"{path}.product_id",
            allow_bare_sha256=True,
        ),
        acquisition_id=acquisition_id,
        pair_id=pair_id,
        sample_id=acquisition.sample_id,
        timestamp_s=acquisition.timestamp_s,
        observable=_string(mapping["observable"], path=f"{path}.observable"),
        variant=_string(mapping["variant"], path=f"{path}.variant"),
        representation=representation,
        unit=unit,
        sign_convention=_string(
            mapping["sign_convention"],
            path=f"{path}.sign_convention",
        ),
        axes=axes,
        values=values,
        uncertainty=_parse_uncertainty(
            mapping["uncertainty"],
            path=f"{path}.uncertainty",
            expected_unit=unit,
        ),
        data_path=relative_path,
        data_sha256=data_hash,
        product_sha256=declared_hash,
    )


def load_measurement_manifest(path: str | Path) -> MeasurementManifest:
    """Load and fully validate one already-calibrated measurement manifest."""

    manifest_path = Path(path)
    try:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise MeasurementContractError("measurement manifest must be a regular file")
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementContractError("measurement manifest is not valid UTF-8 JSON") from exc
    mapping = _exact_mapping(
        raw,
        path="manifest",
        required=frozenset(
            {
                "contract_version",
                "dataset",
                "clock",
                "governance",
                "acquisitions",
                "sensor_topology",
                "calibration_record",
                "reference_chain",
                "processing_lineage",
                "products",
                "manifest_sha256",
            }
        ),
    )
    version = mapping["contract_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != MEASUREMENT_CONTRACT_VERSION
    ):
        raise MeasurementContractError(
            f"manifest.contract_version must be {MEASUREMENT_CONTRACT_VERSION}"
        )
    expected_manifest_hash = canonical_measurement_sha256(
        {key: item for key, item in mapping.items() if key != "manifest_sha256"},
        domain="manifest",
    )
    manifest_hash = _sha256(mapping["manifest_sha256"], path="manifest.manifest_sha256")
    if manifest_hash != expected_manifest_hash:
        raise MeasurementContractError("manifest.manifest_sha256 does not match metadata")

    dataset_raw = _exact_mapping(
        mapping["dataset"],
        path="manifest.dataset",
        required=frozenset({"dataset_id", "source_kind", "role", "split_id", "split_sha256"}),
    )
    source_kind = _string(dataset_raw["source_kind"], path="manifest.dataset.source_kind")
    if source_kind not in _SOURCE_KINDS:
        raise MeasurementContractError("manifest.dataset.source_kind is unsupported")
    role = _string(dataset_raw["role"], path="manifest.dataset.role")
    if role not in _DATASET_ROLES:
        raise MeasurementContractError("manifest.dataset.role is unsupported")
    dataset = DatasetIdentity(
        dataset_id=_stable_identity(dataset_raw["dataset_id"], path="manifest.dataset.dataset_id"),
        source_kind=source_kind,  # type: ignore[arg-type]
        role=role,
        split_id=_string(dataset_raw["split_id"], path="manifest.dataset.split_id"),
        split_sha256=_sha256(
            dataset_raw["split_sha256"],
            path="manifest.dataset.split_sha256",
        ),
    )

    governance_raw = _exact_mapping(
        mapping["governance"],
        path="manifest.governance",
        required=frozenset(
            {
                "provenance_id",
                "license_spdx",
                "redistribution",
                "privacy",
                "authority_id",
            }
        ),
    )
    license_spdx = _string(
        governance_raw["license_spdx"],
        path="manifest.governance.license_spdx",
    )
    if _SPDX_RE.fullmatch(license_spdx) is None:
        raise MeasurementContractError(
            "manifest.governance.license_spdx must be a single SPDX license identifier"
        )
    redistribution = _string(
        governance_raw["redistribution"],
        path="manifest.governance.redistribution",
    )
    if redistribution not in _REDISTRIBUTION_VALUES:
        raise MeasurementContractError("manifest.governance.redistribution is unsupported")
    privacy = _string(
        governance_raw["privacy"],
        path="manifest.governance.privacy",
    )
    if privacy not in _PRIVACY_VALUES:
        raise MeasurementContractError("manifest.governance.privacy is unsupported")
    governance = GovernanceRecord(
        provenance_id=_stable_identity(
            governance_raw["provenance_id"],
            path="manifest.governance.provenance_id",
        ),
        license_spdx=license_spdx,
        redistribution=redistribution,
        privacy=privacy,
        authority_id=_opaque_reference(
            governance_raw["authority_id"],
            path="manifest.governance.authority_id",
        ),
    )

    raw_acquisitions = mapping["acquisitions"]
    if not isinstance(raw_acquisitions, list) or not raw_acquisitions:
        raise MeasurementContractError("manifest.acquisitions must be a non-empty list")
    acquisitions = tuple(
        _parse_acquisition(item, path=f"manifest.acquisitions[{index}]")
        for index, item in enumerate(raw_acquisitions)
    )
    for values, label in (
        ([item.acquisition_id for item in acquisitions], "acquisition IDs"),
        ([item.sample_id for item in acquisitions], "sample IDs"),
    ):
        if len(values) != len(set(values)):
            raise MeasurementContractError(f"manifest.acquisitions contain duplicate {label}")
    acquisition_timestamps = [item.timestamp_s for item in acquisitions]
    if any(
        current <= previous
        for previous, current in zip(acquisition_timestamps, acquisition_timestamps[1:])
    ):
        raise MeasurementContractError(
            "manifest.acquisitions timestamps must be strictly increasing in source order"
        )
    acquisitions_by_id = {item.acquisition_id: item for item in acquisitions}

    raw_topology = mapping["sensor_topology"]
    if not isinstance(raw_topology, list) or not raw_topology:
        raise MeasurementContractError("manifest.sensor_topology must be a non-empty list")
    sensor_topology = tuple(
        _parse_sensor_pair(item, path=f"manifest.sensor_topology[{index}]")
        for index, item in enumerate(raw_topology)
    )
    pair_ids = [item.pair_id for item in sensor_topology]
    ordered_sensor_pairs = [(item.tx_sensor_id, item.rx_sensor_id) for item in sensor_topology]
    if len(pair_ids) != len(set(pair_ids)):
        raise MeasurementContractError("manifest.sensor_topology contains duplicate pair IDs")
    if len(ordered_sensor_pairs) != len(set(ordered_sensor_pairs)):
        raise MeasurementContractError(
            "manifest.sensor_topology contains duplicate ordered sensor pairs"
        )

    clock_raw = _exact_mapping(
        mapping["clock"],
        path="manifest.clock",
        required=frozenset({"clock_id", "origin", "unit"}),
    )
    if clock_raw["unit"] != "s":
        raise MeasurementContractError("manifest.clock.unit must be 's'")
    clock = ClockReference(
        clock_id=_string(clock_raw["clock_id"], path="manifest.clock.clock_id"),
        origin=_string(clock_raw["origin"], path="manifest.clock.origin"),
    )

    lineage_raw = _exact_mapping(
        mapping["processing_lineage"],
        path="manifest.processing_lineage",
        required=frozenset(
            {
                "processing_run_id",
                "processor",
                "processor_version",
                "recipe_sha256",
                "source_sha256",
            }
        ),
    )
    raw_source_hashes = lineage_raw["source_sha256"]
    if not isinstance(raw_source_hashes, list) or not raw_source_hashes:
        raise MeasurementContractError(
            "manifest.processing_lineage.source_sha256 must be a non-empty list"
        )
    source_hashes = tuple(
        _sha256(item, path=f"manifest.processing_lineage.source_sha256[{index}]")
        for index, item in enumerate(raw_source_hashes)
    )
    if len(source_hashes) != len(set(source_hashes)):
        raise MeasurementContractError(
            "manifest.processing_lineage.source_sha256 contains duplicates"
        )
    lineage = ProcessingLineage(
        processing_run_id=_stable_identity(
            lineage_raw["processing_run_id"],
            path="manifest.processing_lineage.processing_run_id",
        ),
        processor=_string(
            lineage_raw["processor"],
            path="manifest.processing_lineage.processor",
        ),
        processor_version=_string(
            lineage_raw["processor_version"],
            path="manifest.processing_lineage.processor_version",
        ),
        recipe_sha256=_sha256(
            lineage_raw["recipe_sha256"],
            path="manifest.processing_lineage.recipe_sha256",
        ),
        source_sha256=source_hashes,
    )

    raw_products = mapping["products"]
    if not isinstance(raw_products, list) or not raw_products:
        raise MeasurementContractError("manifest.products must be a non-empty list")
    products = tuple(
        _parse_product(
            item,
            root=manifest_path.parent,
            path=f"manifest.products[{index}]",
            acquisitions=acquisitions_by_id,
            pair_ids=frozenset(pair_ids),
        )
        for index, item in enumerate(raw_products)
    )
    product_ids = [product.product_id for product in products]
    if len(product_ids) != len(set(product_ids)):
        raise MeasurementContractError("manifest.products contain duplicate product IDs")
    product_occurrences = [
        (product.acquisition_id, product.pair_id, product.observable, product.variant)
        for product in products
    ]
    if len(product_occurrences) != len(set(product_occurrences)):
        raise MeasurementContractError(
            "manifest.products contain a duplicate acquisition ID for the same "
            "observable/variant"
        )
    expected_split_hash = canonical_measurement_sha256(
        {
            "split_id": dataset.split_id,
            "sample_ids": [item.sample_id for item in acquisitions],
        },
        domain="dataset_split",
    )
    if dataset.split_sha256 != expected_split_hash:
        raise MeasurementContractError(
            "manifest.dataset.split_sha256 does not match ordered source sample identities"
        )

    calibration = _parse_calibration_record(
        mapping["calibration_record"],
        path="manifest.calibration_record",
    )
    if source_kind == "measurement" and not calibration.is_known:
        raise MeasurementContractError(
            "measurement manifests require an opaque calibration-record ID and hash"
        )
    if source_kind == "simulation" and calibration.is_known:
        raise MeasurementContractError(
            "simulation manifests must mark calibration_record as not_applicable"
        )
    if (
        source_kind == "simulation"
        and calibration.missing is not None
        and calibration.missing.reason != "not_applicable"
    ):
        raise MeasurementContractError(
            "simulation calibration_record missing reason must be not_applicable"
        )
    if calibration.is_known:
        assert calibration.scope is not None
        for product in products:
            if (
                product.pair_id not in calibration.scope.pair_ids
                or product.acquisition_id not in calibration.scope.acquisition_ids
                or (product.observable, product.variant)
                not in calibration.scope.observable_variants
            ):
                raise MeasurementContractError(
                    "manifest.calibration_record.scope does not cover every measurement product"
                )

    return MeasurementManifest(
        contract_version=version,
        dataset=dataset,
        governance=governance,
        acquisitions=acquisitions,
        sensor_topology=sensor_topology,
        clock=clock,
        calibration_record=calibration,
        reference_chain=_parse_reference_chain(
            mapping["reference_chain"],
            path="manifest.reference_chain",
        ),
        processing_lineage=lineage,
        products=products,
        manifest_sha256=manifest_hash,
    )


def _selected_products(
    manifest: MeasurementManifest,
    *,
    observable: str,
    variant: str,
) -> dict[str, MeasurementProduct]:
    selected: dict[str, MeasurementProduct] = {}
    for product in manifest.products:
        if product.observable != observable or product.variant != variant:
            continue
        if product.acquisition_id in selected:
            raise MeasurementContractError(
                "V1 comparison supports exactly one sensor-pair product per acquisition"
            )
        selected[product.acquisition_id] = product
    if not selected:
        raise MeasurementContractError(
            f"dataset {manifest.dataset.dataset_id} has no {observable!r}/{variant!r} products"
        )
    return selected


def load_measurement_alignment_recipe(
    value: Mapping[str, Any] | str | Path,
) -> MeasurementAlignmentRecipe:
    """Load a strict, versioned alignment recipe from JSON data or a JSON file."""

    if isinstance(value, (str, Path)):
        recipe_path = Path(value)
        try:
            if not recipe_path.is_file() or recipe_path.is_symlink():
                raise MeasurementContractError("alignment recipe must be a regular file")
            raw = json.loads(
                recipe_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementContractError("alignment recipe is not valid UTF-8 JSON") from exc
    else:
        raw = dict(value)
    mapping = _exact_mapping(
        raw,
        path="alignment_recipe",
        required=frozenset(
            {
                "contract",
                "contract_version",
                "alignment_id",
                "alignment_sha256",
                "measurement_dataset_id",
                "simulation_dataset_id",
                "measurement_manifest_sha256",
                "simulation_manifest_sha256",
                "split_sha256",
                "observable",
                "variant",
                "uncertainty_combination_policy",
                "acquisition_pairs",
                "axis_alignments",
                "measurement_to_simulation_time_offset_s",
                "timestamp_tolerance_s",
            }
        ),
    )
    if mapping["contract"] != "orchav.sensing.measurement_alignment":
        raise MeasurementContractError("alignment_recipe.contract is unsupported")
    version = mapping["contract_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != MEASUREMENT_ALIGNMENT_CONTRACT_VERSION
    ):
        raise MeasurementContractError(
            "alignment_recipe.contract_version must be " f"{MEASUREMENT_ALIGNMENT_CONTRACT_VERSION}"
        )
    body = {
        key: item
        for key, item in mapping.items()
        if key not in {"alignment_id", "alignment_sha256"}
    }
    expected_hash = canonical_measurement_sha256(body, domain="alignment_recipe")
    declared_hash = _sha256(
        mapping["alignment_sha256"],
        path="alignment_recipe.alignment_sha256",
    )
    if declared_hash != expected_hash:
        raise MeasurementContractError(
            "alignment_recipe.alignment_sha256 does not match recipe content"
        )
    if mapping["alignment_id"] != f"sha256:{declared_hash}":
        raise MeasurementContractError(
            "alignment_recipe.alignment_id must be sha256:<alignment_sha256>"
        )
    uncertainty_policy = _string(
        mapping["uncertainty_combination_policy"],
        path="alignment_recipe.uncertainty_combination_policy",
    )
    if uncertainty_policy != INDEPENDENT_VALUE_UNCERTAINTY_POLICY:
        raise MeasurementContractError(
            "alignment_recipe.uncertainty_combination_policy must explicitly declare "
            f"{INDEPENDENT_VALUE_UNCERTAINTY_POLICY!r}"
        )
    raw_pairs = mapping["acquisition_pairs"]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise MeasurementContractError(
            "alignment_recipe.acquisition_pairs must be a non-empty list"
        )
    pairs: list[AcquisitionPair] = []
    for index, item in enumerate(raw_pairs):
        pair = _exact_mapping(
            item,
            path=f"alignment_recipe.acquisition_pairs[{index}]",
            required=frozenset(
                {
                    "sample_id",
                    "measurement_acquisition_id",
                    "simulation_acquisition_id",
                }
            ),
        )
        pairs.append(
            AcquisitionPair(
                sample_id=pair["sample_id"],
                measurement_acquisition_id=pair["measurement_acquisition_id"],
                simulation_acquisition_id=pair["simulation_acquisition_id"],
            )
        )
    raw_axes = mapping["axis_alignments"]
    if not isinstance(raw_axes, list) or not raw_axes:
        raise MeasurementContractError("alignment_recipe.axis_alignments must be a non-empty list")
    axes: list[AxisAlignment] = []
    for index, item in enumerate(raw_axes):
        axis = _exact_mapping(
            item,
            path=f"alignment_recipe.axis_alignments[{index}]",
            required=frozenset(
                {
                    "name",
                    "unit",
                    "measurement_to_simulation_scale",
                    "measurement_to_simulation_offset",
                    "absolute_tolerance",
                }
            ),
        )
        axes.append(
            AxisAlignment(
                name=axis["name"],
                unit=axis["unit"],
                measurement_to_simulation_offset=axis["measurement_to_simulation_offset"],
                measurement_to_simulation_scale=axis["measurement_to_simulation_scale"],
                absolute_tolerance=axis["absolute_tolerance"],
            )
        )
    return MeasurementAlignmentRecipe(
        contract_version=version,
        alignment_id=mapping["alignment_id"],
        alignment_sha256=declared_hash,
        measurement_dataset_id=_stable_identity(
            mapping["measurement_dataset_id"],
            path="alignment_recipe.measurement_dataset_id",
        ),
        simulation_dataset_id=_stable_identity(
            mapping["simulation_dataset_id"],
            path="alignment_recipe.simulation_dataset_id",
        ),
        measurement_manifest_sha256=_sha256(
            mapping["measurement_manifest_sha256"],
            path="alignment_recipe.measurement_manifest_sha256",
        ),
        simulation_manifest_sha256=_sha256(
            mapping["simulation_manifest_sha256"],
            path="alignment_recipe.simulation_manifest_sha256",
        ),
        split_sha256=_sha256(
            mapping["split_sha256"],
            path="alignment_recipe.split_sha256",
        ),
        observable=_string(mapping["observable"], path="alignment_recipe.observable"),
        variant=_string(mapping["variant"], path="alignment_recipe.variant"),
        uncertainty_combination_policy=uncertainty_policy,
        alignment=MeasurementAlignment(
            acquisition_pairs=tuple(pairs),
            axis_alignments=tuple(axes),
            measurement_to_simulation_time_offset_s=mapping[
                "measurement_to_simulation_time_offset_s"
            ],
            timestamp_tolerance_s=mapping["timestamp_tolerance_s"],
        ),
    )


def alignment_recipe_mapping(recipe: MeasurementAlignmentRecipe) -> dict[str, Any]:
    """Return the canonical JSON-compatible mapping for an alignment recipe."""

    return {
        "contract": "orchav.sensing.measurement_alignment",
        "contract_version": recipe.contract_version,
        "alignment_id": recipe.alignment_id,
        "alignment_sha256": recipe.alignment_sha256,
        "measurement_dataset_id": recipe.measurement_dataset_id,
        "simulation_dataset_id": recipe.simulation_dataset_id,
        "measurement_manifest_sha256": recipe.measurement_manifest_sha256,
        "simulation_manifest_sha256": recipe.simulation_manifest_sha256,
        "split_sha256": recipe.split_sha256,
        "observable": recipe.observable,
        "variant": recipe.variant,
        "uncertainty_combination_policy": recipe.uncertainty_combination_policy,
        "acquisition_pairs": [
            {
                "sample_id": pair.sample_id,
                "measurement_acquisition_id": pair.measurement_acquisition_id,
                "simulation_acquisition_id": pair.simulation_acquisition_id,
            }
            for pair in recipe.alignment.acquisition_pairs
        ],
        "axis_alignments": [
            {
                "name": axis.name,
                "unit": axis.unit,
                "measurement_to_simulation_scale": (axis.measurement_to_simulation_scale),
                "measurement_to_simulation_offset": (axis.measurement_to_simulation_offset),
                "absolute_tolerance": axis.absolute_tolerance,
            }
            for axis in recipe.alignment.axis_alignments
        ],
        "measurement_to_simulation_time_offset_s": (
            recipe.alignment.measurement_to_simulation_time_offset_s
        ),
        "timestamp_tolerance_s": recipe.alignment.timestamp_tolerance_s,
    }


def _comparison_preflight(
    measurement: MeasurementManifest,
    simulation: MeasurementManifest,
    recipe: MeasurementAlignmentRecipe,
) -> tuple[
    list[tuple[AcquisitionPair, MeasurementProduct, MeasurementProduct]],
    dict[str, AxisAlignment],
]:
    alignment = recipe.alignment
    observable = recipe.observable
    variant = recipe.variant
    if (
        recipe.measurement_dataset_id != measurement.dataset.dataset_id
        or recipe.simulation_dataset_id != simulation.dataset.dataset_id
        or recipe.measurement_manifest_sha256 != measurement.manifest_sha256
        or recipe.simulation_manifest_sha256 != simulation.manifest_sha256
        or recipe.split_sha256 != measurement.dataset.split_sha256
    ):
        raise MeasurementContractError("alignment recipe is not bound to the exact input manifests")
    if measurement.dataset.source_kind != "measurement":
        raise MeasurementContractError("left comparison input must be a measurement dataset")
    if simulation.dataset.source_kind != "simulation":
        raise MeasurementContractError("right comparison input must be a simulation dataset")
    if measurement.dataset.dataset_id == simulation.dataset.dataset_id:
        raise MeasurementContractError("measurement and simulation dataset IDs must differ")
    if measurement.dataset.role != "evaluation" or simulation.dataset.role != "evaluation":
        raise MeasurementContractError(
            "comparison requires evaluation-role datasets; calibration/tuning leakage is rejected"
        )
    if (
        measurement.dataset.split_id != simulation.dataset.split_id
        or measurement.dataset.split_sha256 != simulation.dataset.split_sha256
    ):
        raise MeasurementContractError("comparison datasets do not describe the same frozen split")
    if measurement.reference_chain.sha256 != simulation.reference_chain.sha256:
        raise MeasurementContractError("comparison reference transform chains do not match")
    if measurement.sensor_topology != simulation.sensor_topology:
        raise MeasurementContractError("comparison sensor-pair topology does not match")
    if len(measurement.sensor_topology) != 1:
        raise MeasurementContractError("V1 comparison supports exactly one sensor pair")
    if measurement.clock.unit != "s" or simulation.clock.unit != "s":
        raise MeasurementContractError("comparison clocks must use seconds")

    measured = _selected_products(measurement, observable=observable, variant=variant)
    simulated = _selected_products(simulation, observable=observable, variant=variant)
    if {pair.measurement_acquisition_id for pair in alignment.acquisition_pairs} != set(measured):
        raise MeasurementContractError(
            "alignment must cover every selected measurement acquisition exactly once"
        )
    if {pair.simulation_acquisition_id for pair in alignment.acquisition_pairs} != set(simulated):
        raise MeasurementContractError(
            "alignment must cover every selected simulation acquisition exactly once; "
            "an observable/variant mismatch may have removed a product"
        )
    axis_alignments = {axis.name: axis for axis in alignment.axis_alignments}
    resolved: list[tuple[AcquisitionPair, MeasurementProduct, MeasurementProduct]] = []
    for pair in alignment.acquisition_pairs:
        left = measured[pair.measurement_acquisition_id]
        right = simulated[pair.simulation_acquisition_id]
        if left.sample_id != pair.sample_id or right.sample_id != pair.sample_id:
            raise MeasurementContractError("alignment sample identity does not match both products")
        aligned_time = left.timestamp_s + alignment.measurement_to_simulation_time_offset_s
        if not math.isclose(
            aligned_time,
            right.timestamp_s,
            abs_tol=alignment.timestamp_tolerance_s,
            rel_tol=0.0,
        ):
            raise MeasurementContractError("aligned acquisition timestamps do not match")
        for field_name in ("pair_id", "observable", "variant", "representation", "unit"):
            if getattr(left, field_name) != getattr(right, field_name):
                raise MeasurementContractError(
                    f"comparison product {field_name} values do not match"
                )
        if len(left.axes) != len(right.axes):
            raise MeasurementContractError("comparison product ranks do not match")
        if left.values.shape != right.values.shape:
            raise MeasurementContractError("comparison product shapes do not match")
        if set(axis_alignments) != {axis.name for axis in left.axes}:
            raise MeasurementContractError(
                "alignment must name every selected product axis exactly once"
            )
        negative_sign_axes = {
            axis.name
            for axis in axis_alignments.values()
            if axis.measurement_to_simulation_scale == -1
            and axis.name in {"doppler_frequency", "radial_velocity"}
        }
        if left.sign_convention == right.sign_convention:
            if negative_sign_axes:
                raise MeasurementContractError(
                    "a negative Doppler/velocity axis transform requires explicit opposite "
                    "sign_convention values"
                )
        else:
            required_axis = _SIGN_REVERSAL_AXES.get((left.sign_convention, right.sign_convention))
            if required_axis is None or required_axis not in negative_sign_axes:
                raise MeasurementContractError(
                    "comparison product sign_convention values are not an admitted "
                    "approaching/receding pair with the matching negative axis transform"
                )
        for left_axis, right_axis in zip(left.axes, right.axes):
            if (
                left_axis.name != right_axis.name
                or left_axis.unit != right_axis.unit
                or left_axis.reference_frame != right_axis.reference_frame
            ):
                raise MeasurementContractError(
                    "comparison axis name, unit, or reference frame does not match"
                )
            axis_alignment = axis_alignments[left_axis.name]
            if axis_alignment.unit != left_axis.unit:
                raise MeasurementContractError("axis alignment unit does not match product axis")
            aligned_coordinates = (
                axis_alignment.measurement_to_simulation_scale * left_axis.coordinates
                + axis_alignment.measurement_to_simulation_offset
            )
            if axis_alignment.measurement_to_simulation_scale == -1:
                aligned_coordinates = aligned_coordinates[::-1]
            if not np.allclose(
                aligned_coordinates,
                right_axis.coordinates,
                atol=axis_alignment.absolute_tolerance,
                rtol=0.0,
            ):
                raise MeasurementContractError(
                    f"aligned coordinates for axis {left_axis.name!r} do not match"
                )
        resolved.append((pair, left, right))
    product_contracts = {
        (left.representation, left.unit, left.sign_convention) for _, left, _ in resolved
    }
    if len(product_contracts) != 1:
        raise MeasurementContractError(
            "comparison products must share one representation, unit, and sign convention"
        )
    if next(iter(product_contracts))[0] == "complex_voltage":
        raise MeasurementContractError(
            "complex_voltage comparison requires an explicit hashed phase-reference, "
            "coherence, and alignment contract; V1 can inventory but not compare it"
        )
    axis_layouts = {
        tuple((axis.name, axis.unit, axis.reference_frame) for axis in left.axes)
        for _, left, _ in resolved
    }
    if len(axis_layouts) != 1:
        raise MeasurementContractError(
            "comparison products must share one ordered axis/reference-frame layout"
        )
    return resolved, axis_alignments


def compare_measurement_to_simulation(
    measurement: MeasurementManifest,
    simulation: MeasurementManifest,
    recipe: MeasurementAlignmentRecipe,
) -> MeasurementComparison:
    """Compare aligned products without ordinal, resampling, or calibration fallback."""

    recipe = load_measurement_alignment_recipe(alignment_recipe_mapping(recipe))
    observable = recipe.observable
    variant = recipe.variant
    alignment = recipe.alignment
    resolved, axis_alignments = _comparison_preflight(
        measurement,
        simulation,
        recipe,
    )
    residuals: list[ProductResidual] = []
    flat_residuals: list[np.ndarray] = []
    normalized_magnitudes: list[np.ndarray] = []
    for pair, left, right in resolved:
        aligned_measurement_values = left.values
        for axis_index, axis in enumerate(left.axes):
            if axis_alignments[axis.name].measurement_to_simulation_scale == -1:
                aligned_measurement_values = np.flip(
                    aligned_measurement_values,
                    axis=axis_index,
                )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                residual = aligned_measurement_values - right.values
                absolute = np.abs(residual)
        except FloatingPointError as exc:
            raise MeasurementContractError(
                "derived residual values are not representable as finite numbers"
            ) from exc
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(absolute)):
            raise MeasurementContractError(
                "derived residual values are not representable as finite numbers"
            )
        rmse = _stable_rms(absolute, path="product residual RMSE")
        combined_uncertainty: StandardUncertainty
        normalized_rmse: float | None
        if left.uncertainty.is_known and right.uncertainty.is_known:
            assert left.uncertainty.standard_deviation is not None
            assert right.uncertainty.standard_deviation is not None
            sigma = math.hypot(
                left.uncertainty.standard_deviation,
                right.uncertainty.standard_deviation,
            )
            if not math.isfinite(sigma):
                raise MeasurementContractError(
                    "combined product-value standard uncertainty is not representable "
                    "as a finite float"
                )
            combined_uncertainty = StandardUncertainty(left.unit, sigma)
            if sigma == 0.0 and rmse != 0.0:
                raise MeasurementContractError(
                    "nonzero residual cannot be normalized by zero combined uncertainty"
                )
            normalized_rmse = rmse / sigma if sigma > 0.0 else 0.0
            if not math.isfinite(normalized_rmse):
                raise MeasurementContractError(
                    "normalized residual is not representable as a finite float"
                )
            if sigma > 0.0:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    normalized = absolute / sigma
                if not np.all(np.isfinite(normalized)):
                    raise MeasurementContractError(
                        "normalized residual values are not representable as finite numbers"
                    )
                normalized_magnitudes.append(normalized)
            else:
                normalized_magnitudes.append(np.zeros_like(absolute, dtype=np.float64))
        else:
            combined_uncertainty = StandardUncertainty(
                left.unit,
                None,
                MissingField("not_provided"),
            )
            normalized_rmse = None
        flat_residuals.append(residual.reshape(-1))
        residuals.append(
            ProductResidual(
                sample_id=pair.sample_id,
                measurement_acquisition_id=left.acquisition_id,
                simulation_acquisition_id=right.acquisition_id,
                simulation_timestamp_s=right.timestamp_s,
                timestamp_residual_s=(
                    left.timestamp_s
                    + alignment.measurement_to_simulation_time_offset_s
                    - right.timestamp_s
                ),
                residual=residual,
                rmse=rmse,
                max_absolute_error=float(np.max(absolute)),
                mean_residual=_stable_complex_mean(
                    residual,
                    path="product mean residual",
                ),
                combined_standard_uncertainty=combined_uncertainty,
                normalized_rmse=normalized_rmse,
            )
        )
    flat = np.concatenate(flat_residuals)
    absolute = np.abs(flat)
    aggregate_normalized = (
        _stable_rms(
            np.concatenate([item.reshape(-1) for item in normalized_magnitudes]),
            path="aggregate normalized RMSE",
        )
        if normalized_magnitudes and len(normalized_magnitudes) == len(residuals)
        else None
    )
    comparison_identity = {
        "measurement_manifest_sha256": measurement.manifest_sha256,
        "simulation_manifest_sha256": simulation.manifest_sha256,
        "alignment_id": recipe.alignment_id,
        "observable": observable,
        "variant": variant,
        "uncertainty_combination_policy": recipe.uncertainty_combination_policy,
        "residual_definition": RESIDUAL_DEFINITION,
        "aggregate_weighting_policy": AGGREGATE_WEIGHTING_POLICY,
        "acquisition_pairs": [
            {
                "sample_id": pair.sample_id,
                "measurement_acquisition_id": pair.measurement_acquisition_id,
                "simulation_acquisition_id": pair.simulation_acquisition_id,
            }
            for pair in alignment.acquisition_pairs
        ],
        "axis_alignments": [
            {
                "name": axis.name,
                "unit": axis.unit,
                "measurement_to_simulation_offset": axis.measurement_to_simulation_offset,
                "measurement_to_simulation_scale": axis.measurement_to_simulation_scale,
                "absolute_tolerance": axis.absolute_tolerance,
            }
            for axis in alignment.axis_alignments
        ],
        "measurement_to_simulation_time_offset_s": (
            alignment.measurement_to_simulation_time_offset_s
        ),
        "timestamp_tolerance_s": alignment.timestamp_tolerance_s,
    }
    first_left = resolved[0][1]
    return MeasurementComparison(
        contract_version=MEASUREMENT_COMPARISON_CONTRACT_VERSION,
        comparison_id=canonical_measurement_sha256(
            comparison_identity,
            domain="comparison",
        ),
        alignment_id=recipe.alignment_id,
        alignment_sha256=recipe.alignment_sha256,
        measurement_dataset_id=measurement.dataset.dataset_id,
        simulation_dataset_id=simulation.dataset.dataset_id,
        measurement_governance=measurement.governance,
        simulation_governance=simulation.governance,
        dataset_role="evaluation",
        split_id=measurement.dataset.split_id,
        split_sha256=measurement.dataset.split_sha256,
        observable=observable,
        variant=variant,
        representation=first_left.representation,
        unit=first_left.unit,
        sign_convention=resolved[0][2].sign_convention,
        uncertainty_combination_policy=recipe.uncertainty_combination_policy,
        residual_definition=RESIDUAL_DEFINITION,
        aggregate_weighting_policy=AGGREGATE_WEIGHTING_POLICY,
        reference_chain_sha256=measurement.reference_chain.sha256,
        residuals=tuple(residuals),
        aggregate_rmse=_stable_rms(absolute, path="aggregate residual RMSE"),
        aggregate_max_absolute_error=float(np.max(absolute)),
        aggregate_mean_residual=_stable_complex_mean(
            flat,
            path="aggregate mean residual",
        ),
        aggregate_normalized_rmse=aggregate_normalized,
    )


def _uncertainty_mapping(value: StandardUncertainty) -> dict[str, Any]:
    if value.is_known:
        return {
            "status": "known",
            "unit": value.unit,
            "standard_deviation": value.standard_deviation,
        }
    assert value.missing is not None
    return {"status": "missing", "unit": value.unit, "reason": value.missing.reason}


def _numeric_payload(value: np.ndarray) -> dict[str, Any]:
    if np.iscomplexobj(value):
        return {
            "encoding": "complex_cartesian",
            "real": np.asarray(value.real, dtype=np.float64).tolist(),
            "imag": np.asarray(value.imag, dtype=np.float64).tolist(),
        }
    return {
        "encoding": "real",
        "values": np.asarray(value, dtype=np.float64).tolist(),
    }


def comparison_export_mapping(comparison: MeasurementComparison) -> dict[str, Any]:
    """Return a deterministic JSON-compatible comparison/result mapping."""

    for governance, label in (
        (comparison.measurement_governance, "measurement"),
        (comparison.simulation_governance, "simulation"),
    ):
        if governance.redistribution != "permitted" or governance.privacy == "restricted":
            raise MeasurementContractError(
                f"{label} governance does not authorize comparison-product export"
            )

    def governance_mapping(value: GovernanceRecord) -> dict[str, str]:
        return {
            "provenance_id": value.provenance_id,
            "license_spdx": value.license_spdx,
            "redistribution": value.redistribution,
            "privacy": value.privacy,
            "authority_id": value.authority_id,
        }

    result: dict[str, Any] = {
        "contract": "orchav.sensing.measurement_comparison",
        "contract_version": comparison.contract_version,
        "comparison_id": comparison.comparison_id,
        "alignment_id": comparison.alignment_id,
        "alignment_sha256": comparison.alignment_sha256,
        "inputs": {
            "measurement_dataset_id": comparison.measurement_dataset_id,
            "simulation_dataset_id": comparison.simulation_dataset_id,
            "dataset_role": comparison.dataset_role,
            "split_id": comparison.split_id,
            "split_sha256": comparison.split_sha256,
            "reference_chain_sha256": comparison.reference_chain_sha256,
            "measurement_governance": governance_mapping(comparison.measurement_governance),
            "simulation_governance": governance_mapping(comparison.simulation_governance),
        },
        "product_contract": {
            "observable": comparison.observable,
            "variant": comparison.variant,
            "representation": comparison.representation,
            "unit": comparison.unit,
            "sign_convention": comparison.sign_convention,
            "residual_definition": comparison.residual_definition,
        },
        "aggregate_contract": {
            "weighting_policy": comparison.aggregate_weighting_policy,
        },
        "uncertainty_contract": {
            "value_combination_policy": comparison.uncertainty_combination_policy,
            "axis_uncertainty_use": (
                "retained_for_axis_contract_validation_not_propagated_into_value_residuals"
            ),
        },
        "aggregate": {
            "rmse": comparison.aggregate_rmse,
            "max_absolute_error": comparison.aggregate_max_absolute_error,
            "mean_residual": {
                "real": comparison.aggregate_mean_residual.real,
                "imag": comparison.aggregate_mean_residual.imag,
            },
            "normalized_rmse": comparison.aggregate_normalized_rmse,
        },
        "residuals": [
            {
                "sample_id": residual.sample_id,
                "measurement_acquisition_id": residual.measurement_acquisition_id,
                "simulation_acquisition_id": residual.simulation_acquisition_id,
                "simulation_timestamp_s": residual.simulation_timestamp_s,
                "timestamp_residual_s": residual.timestamp_residual_s,
                "residual": _numeric_payload(residual.residual),
                "metrics": {
                    "rmse": residual.rmse,
                    "max_absolute_error": residual.max_absolute_error,
                    "mean_residual": {
                        "real": residual.mean_residual.real,
                        "imag": residual.mean_residual.imag,
                    },
                    "normalized_rmse": residual.normalized_rmse,
                },
                "combined_standard_uncertainty": _uncertainty_mapping(
                    residual.combined_standard_uncertainty
                ),
            }
            for residual in comparison.residuals
        ],
    }
    result["export_sha256"] = canonical_measurement_sha256(result, domain="comparison_export")
    return result


__all__ = [
    "AcquisitionPair",
    "AcquisitionSample",
    "AGGREGATE_WEIGHTING_POLICY",
    "AxisAlignment",
    "CalibrationRecord",
    "CalibrationScope",
    "ClockReference",
    "DatasetIdentity",
    "GovernanceRecord",
    "INDEPENDENT_VALUE_UNCERTAINTY_POLICY",
    "MEASUREMENT_COMPARISON_CONTRACT_VERSION",
    "MEASUREMENT_CONTRACT_VERSION",
    "MEASUREMENT_ALIGNMENT_CONTRACT_VERSION",
    "MeasurementAlignment",
    "MeasurementAlignmentRecipe",
    "MeasurementAxis",
    "MeasurementComparison",
    "MeasurementContractError",
    "MeasurementManifest",
    "MeasurementProduct",
    "MissingField",
    "ProcessingLineage",
    "ProductResidual",
    "ReferenceChain",
    "ReferenceTransform",
    "RESIDUAL_DEFINITION",
    "SensorPair",
    "StandardUncertainty",
    "canonical_measurement_sha256",
    "compare_measurement_to_simulation",
    "comparison_export_mapping",
    "alignment_recipe_mapping",
    "load_measurement_alignment_recipe",
    "load_measurement_manifest",
]
