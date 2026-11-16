"""Lossless, source-neutral interchange for live sensing channel evidence.

The public frame format intentionally does not retain every value needed to
reconstruct a phase-coherent waveform.  This module defines a small JSON
contract for evidence captured directly from a live Sionna ``Paths.cir`` call.
It is not an ordinary-frame format and it is not a 5GNRad input file.  A peer
adapter may consume it only after applying the explicitly recorded convention
mapping.

The initial contract is deliberately narrow: one SISO pair, one or more paths,
and a finite slow-time sequence.  That is sufficient for the static,
approaching, and receding single-path comparisons used by the v0.2 audit while
avoiding an unsupported general MIMO interoperability claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_NAME = "orchav.sensing.live_cir_interchange"
SCHEMA_VERSION = 1
SPEED_OF_LIGHT_M_S = 299_792_458.0
ORCHAV_DOPPLER_CONVENTION = "positive_approaching"
PEER_DOPPLER_CONVENTION = "positive_receding"
_PATH_ROLES = frozenset({"target", "background", "control"})
_REQUIRED_LOSSY_FIELDS = frozenset(
    {
        "full_mimo_tensor",
        "polarization_components",
        "interaction_vertices",
        "material_interaction_chain",
    }
)


class SensingInterchangeError(ValueError):
    """Raised when live-CIR interchange evidence is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, finite-only JSON bytes for identity and publication."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SensingInterchangeError(f"interchange content is not finite JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _content_sha256(record: Mapping[str, Any]) -> str:
    content = dict(record)
    content.pop("interchange_id", None)
    content.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _finite_vector(value: Sequence[float], *, name: str, length: int | None = None) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SensingInterchangeError(f"{name} must be a finite numeric vector") from exc
    if array.ndim != 1 or (length is not None and array.size != length):
        expected = f" length {length}" if length is not None else " one-dimensional"
        raise SensingInterchangeError(f"{name} must be{expected}")
    if not np.all(np.isfinite(array)):
        raise SensingInterchangeError(f"{name} must contain only finite values")
    return [float(item) for item in array]


def _finite_three_vector(value: Sequence[float], *, name: str) -> list[float]:
    return _finite_vector(value, name=name, length=3)


def _is_json_number(value: Any) -> bool:
    return type(value) in {int, float}


def _require_json_number_list(value: Any, *, name: str, length: int | None = None) -> None:
    if (
        not isinstance(value, list)
        or (length is not None and len(value) != length)
        or any(not _is_json_number(item) for item in value)
    ):
        expected = f" of length {length}" if length is not None else ""
        raise SensingInterchangeError(f"{name} must be a JSON-number list{expected}")


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    flattened = np.asarray(values, dtype=np.complex128).reshape(-1)
    if flattened.size < 2 or not np.all(np.isfinite(flattened)):
        raise SensingInterchangeError("each path requires at least two finite complex CIR samples")
    return [[float(item.real), float(item.imag)] for item in flattened]


def complex_pairs_to_array(value: Sequence[Sequence[float]]) -> np.ndarray:
    """Decode a JSON ``[real, imag]`` sequence without allowing object arrays."""

    if not isinstance(value, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(component) not in {int, float} for component in pair)
        for pair in value
    ):
        raise SensingInterchangeError("complex samples must be finite numeric pairs")
    try:
        pairs = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SensingInterchangeError("complex samples must be finite numeric pairs") from exc
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] < 2:
        raise SensingInterchangeError("complex samples must have shape [slow_time, 2]")
    if not np.all(np.isfinite(pairs)):
        raise SensingInterchangeError("complex samples must be finite")
    result = pairs[:, 0] + 1j * pairs[:, 1]
    result.setflags(write=False)
    return result


def _validate_geometry(geometry: Mapping[str, Any]) -> dict[str, list[float]]:
    required = (
        "tx_position_m",
        "rx_position_m",
        "tx_velocity_m_s",
        "rx_velocity_m_s",
        "target_reference_position_m",
        "target_velocity_m_s",
    )
    if not isinstance(geometry, Mapping):
        raise SensingInterchangeError("geometry must be a mapping")
    if set(geometry) != set(required):
        missing = sorted(set(required) - set(geometry))
        extra = sorted(set(geometry) - set(required))
        raise SensingInterchangeError(
            f"geometry fields do not match the v1 contract; missing={missing}, extra={extra}"
        )
    return {
        name: _finite_three_vector(geometry[name], name=f"geometry.{name}") for name in required
    }


def build_live_cir_interchange(
    *,
    case_name: str,
    carrier_frequency_hz: float,
    sample_times_s: Sequence[float],
    cir_by_path: np.ndarray,
    delays_s: Sequence[float],
    doppler_hz: Sequence[float],
    path_roles: Sequence[str],
    path_ids: Sequence[str],
    target_ids: Sequence[str | None],
    geometry: Mapping[str, Any],
    provenance: Mapping[str, Any],
    angles_rad: Sequence[Mapping[str, float] | None] | None = None,
    lossy_fields: Sequence[str] = tuple(sorted(_REQUIRED_LOSSY_FIELDS)),
) -> dict[str, Any]:
    """Build and validate one lossless SISO live-CIR interchange record.

    ``cir_by_path`` has shape ``[path, slow_time]``.  The caller must provide a
    role and stable identifier for every exported path so target and background
    contributions cannot be silently conflated.
    """

    if not isinstance(case_name, str) or not case_name.strip():
        raise SensingInterchangeError("case_name must be a non-empty string")
    carrier = float(carrier_frequency_hz)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise SensingInterchangeError("carrier_frequency_hz must be finite and positive")

    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise SensingInterchangeError("sample_times_s must be a finite vector with length >= 2")
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        raise SensingInterchangeError("sample_times_s must be strictly increasing")
    nominal_dt = float(np.median(deltas))
    if not np.allclose(deltas, nominal_dt, rtol=1.0e-10, atol=1.0e-15):
        raise SensingInterchangeError("v1 requires a uniformly sampled slow-time clock")

    coefficients = np.asarray(cir_by_path, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[1] != times.size:
        raise SensingInterchangeError("cir_by_path must have shape [path, len(sample_times_s)]")
    if coefficients.shape[0] < 1 or not np.all(np.isfinite(coefficients)):
        raise SensingInterchangeError("cir_by_path must contain at least one finite path")
    path_count = coefficients.shape[0]

    delays = np.asarray(delays_s, dtype=np.float64)
    dopplers = np.asarray(doppler_hz, dtype=np.float64)
    parallel_lengths = {
        "delays_s": delays.size,
        "doppler_hz": dopplers.size,
        "path_roles": len(path_roles),
        "path_ids": len(path_ids),
        "target_ids": len(target_ids),
    }
    if any(length != path_count for length in parallel_lengths.values()):
        raise SensingInterchangeError(
            f"per-path fields must all have length {path_count}: {parallel_lengths}"
        )
    if delays.ndim != 1 or dopplers.ndim != 1:
        raise SensingInterchangeError("delays_s and doppler_hz must be one-dimensional")
    if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise SensingInterchangeError("delays_s must be finite and nonnegative")
    if not np.all(np.isfinite(dopplers)):
        raise SensingInterchangeError("doppler_hz must be finite")
    if any(role not in _PATH_ROLES for role in path_roles):
        raise SensingInterchangeError(f"path_roles must be selected from {sorted(_PATH_ROLES)}")
    if len(set(path_ids)) != path_count or any(not str(item).strip() for item in path_ids):
        raise SensingInterchangeError("path_ids must be unique non-empty strings")
    for role, target_id in zip(path_roles, target_ids, strict=True):
        if role == "target" and (not isinstance(target_id, str) or not target_id.strip()):
            raise SensingInterchangeError("target paths require a non-empty target_id")
        if role != "target" and target_id is not None:
            raise SensingInterchangeError("only target paths may carry target_id")

    angle_values = list(angles_rad) if angles_rad is not None else [None] * path_count
    if len(angle_values) != path_count:
        raise SensingInterchangeError("angles_rad must have one entry per path")
    encoded_angles: list[dict[str, float] | None] = []
    expected_angles = {"departure_azimuth", "departure_zenith", "arrival_azimuth", "arrival_zenith"}
    for index, angles in enumerate(angle_values):
        if angles is None:
            encoded_angles.append(None)
            continue
        if set(angles) != expected_angles:
            raise SensingInterchangeError(
                f"angles_rad[{index}] must contain {sorted(expected_angles)}"
            )
        checked = {name: float(angles[name]) for name in sorted(expected_angles)}
        if not all(math.isfinite(value) for value in checked.values()):
            raise SensingInterchangeError(f"angles_rad[{index}] must be finite")
        encoded_angles.append(checked)

    omitted = sorted(set(lossy_fields))
    if not _REQUIRED_LOSSY_FIELDS.issubset(omitted):
        missing = sorted(_REQUIRED_LOSSY_FIELDS - set(omitted))
        raise SensingInterchangeError(
            f"lossy_fields must explicitly name omitted v1 data: {missing}"
        )
    if not isinstance(provenance, Mapping) or not provenance:
        raise SensingInterchangeError("provenance must be a non-empty mapping")

    paths = []
    for index in range(path_count):
        paths.append(
            {
                "path_id": str(path_ids[index]),
                "interaction_role": str(path_roles[index]),
                "target_id": target_ids[index],
                "delay_s": float(delays[index]),
                "doppler": {
                    "orchav_hz": float(dopplers[index]),
                    "positive_receding_peer_hz": float(-dopplers[index]),
                },
                "angles_rad": encoded_angles[index],
                "coefficient_samples_ri": _complex_pairs(coefficients[index]),
            }
        )

    record: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "topology": "siso",
            "capture_boundary": "live_sionna_paths_cir_before_freezing",
            "ordinary_frame_replay_supported": False,
            "case_name": case_name,
        },
        "clock": {
            "carrier_frequency_hz": carrier,
            "sample_rate_hz": float(1.0 / nominal_dt),
            "sample_times_s": [float(item) for item in times],
            "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
        },
        "conventions": {
            "complex_slow_time_phase": "exp(+j*2*pi*doppler_hz*time_s)",
            "orchav_doppler_sign": ORCHAV_DOPPLER_CONVENTION,
            "positive_receding_peer_doppler_sign": PEER_DOPPLER_CONVENTION,
            "peer_adapter": "positive_receding_peer_hz = -orchav_hz",
            "delay_unit": "s",
            "angle_unit": "rad",
            "sionna_angle_names": "azimuth_and_zenith",
        },
        "geometry": _validate_geometry(geometry),
        "provenance": dict(provenance),
        "paths": paths,
        "lossy_fields": omitted,
    }
    digest = _content_sha256(record)
    record["content_sha256"] = digest
    record["interchange_id"] = f"sha256:{digest}"
    validate_live_cir_interchange(record)
    return record


def validate_live_cir_interchange(record: Mapping[str, Any]) -> None:
    """Validate schema, identity, clock, provenance, and every complex path."""

    if not isinstance(record, Mapping):
        raise SensingInterchangeError("interchange record must be a mapping")
    required_top_level = {
        "schema",
        "schema_version",
        "scope",
        "clock",
        "conventions",
        "geometry",
        "provenance",
        "paths",
        "lossy_fields",
        "content_sha256",
        "interchange_id",
    }
    if set(record) != required_top_level:
        raise SensingInterchangeError("live-CIR top-level fields do not match the v1 contract")
    if (
        record.get("schema") != SCHEMA_NAME
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != SCHEMA_VERSION
    ):
        raise SensingInterchangeError("unsupported live-CIR interchange schema")
    digest = record.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SensingInterchangeError("content_sha256 must be a lowercase SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise SensingInterchangeError("content_sha256 is not hexadecimal") from exc
    if digest != digest.lower() or record.get("interchange_id") != f"sha256:{digest}":
        raise SensingInterchangeError("interchange identity fields disagree")
    if _content_sha256(record) != digest:
        raise SensingInterchangeError("interchange content hash mismatch")

    scope = record.get("scope")
    if not isinstance(scope, Mapping) or scope.get("topology") != "siso":
        raise SensingInterchangeError("v1 supports only an explicit SISO scope")
    if set(scope) != {
        "topology",
        "capture_boundary",
        "ordinary_frame_replay_supported",
        "case_name",
    }:
        raise SensingInterchangeError("scope fields do not match the v1 contract")
    if scope.get("capture_boundary") != "live_sionna_paths_cir_before_freezing":
        raise SensingInterchangeError("interchange must originate at the live Paths.cir boundary")
    if scope.get("ordinary_frame_replay_supported") is not False:
        raise SensingInterchangeError("ordinary frame replay must remain explicitly unsupported")
    case_name = scope.get("case_name")
    if not isinstance(case_name, str) or not case_name.strip():
        raise SensingInterchangeError("scope.case_name must be a non-empty string")

    clock = record.get("clock")
    if not isinstance(clock, Mapping):
        raise SensingInterchangeError("clock must be a mapping")
    if set(clock) != {
        "carrier_frequency_hz",
        "sample_rate_hz",
        "sample_times_s",
        "speed_of_light_m_s",
    }:
        raise SensingInterchangeError("clock fields do not match the v1 contract")
    if any(
        not _is_json_number(clock.get(field))
        for field in ("carrier_frequency_hz", "sample_rate_hz", "speed_of_light_m_s")
    ):
        raise SensingInterchangeError("clock scalar fields must be JSON numbers")
    _require_json_number_list(clock.get("sample_times_s"), name="clock.sample_times_s")
    try:
        carrier = float(clock.get("carrier_frequency_hz"))
        times = np.asarray(clock.get("sample_times_s"), dtype=np.float64)
        sample_rate = float(clock.get("sample_rate_hz"))
        speed_of_light = float(clock.get("speed_of_light_m_s"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SensingInterchangeError("clock values must be finite numeric values") from exc
    if (
        isinstance(clock.get("carrier_frequency_hz"), bool)
        or not math.isfinite(carrier)
        or carrier <= 0
    ):
        raise SensingInterchangeError("clock carrier_frequency_hz must be finite and positive")
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise SensingInterchangeError("clock sample_times_s is invalid")
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        raise SensingInterchangeError("clock sample_times_s must be increasing")
    if (
        isinstance(clock.get("sample_rate_hz"), bool)
        or not math.isfinite(sample_rate)
        or sample_rate <= 0.0
    ):
        raise SensingInterchangeError("clock sample_rate_hz must be finite and positive")
    if not np.allclose(deltas, 1.0 / sample_rate, rtol=1.0e-10, atol=1.0e-15):
        raise SensingInterchangeError("clock sample rate disagrees with sample_times_s")
    if isinstance(clock.get("speed_of_light_m_s"), bool) or speed_of_light != SPEED_OF_LIGHT_M_S:
        raise SensingInterchangeError("v1 records the ORCHAV SI speed of light exactly")

    conventions = record.get("conventions")
    if not isinstance(conventions, Mapping):
        raise SensingInterchangeError("conventions must be a mapping")
    expected_conventions = {
        "complex_slow_time_phase": "exp(+j*2*pi*doppler_hz*time_s)",
        "orchav_doppler_sign": ORCHAV_DOPPLER_CONVENTION,
        "positive_receding_peer_doppler_sign": PEER_DOPPLER_CONVENTION,
        "peer_adapter": "positive_receding_peer_hz = -orchav_hz",
        "delay_unit": "s",
        "angle_unit": "rad",
        "sionna_angle_names": "azimuth_and_zenith",
    }
    if dict(conventions) != expected_conventions:
        raise SensingInterchangeError("conventions do not match the complete v1 contract")

    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        raise SensingInterchangeError("geometry must be a mapping")
    for name, vector in geometry.items():
        _require_json_number_list(vector, name=f"geometry.{name}", length=3)
    _validate_geometry(geometry)
    if not isinstance(record.get("provenance"), Mapping) or not record["provenance"]:
        raise SensingInterchangeError("provenance must be present")
    lossy_fields = record.get("lossy_fields")
    if (
        not isinstance(lossy_fields, list)
        or any(not isinstance(item, str) or not item for item in lossy_fields)
        or len(lossy_fields) != len(set(lossy_fields))
        or not _REQUIRED_LOSSY_FIELDS.issubset(lossy_fields)
    ):
        raise SensingInterchangeError("required lossy-field disclosure is incomplete")

    paths = record.get("paths")
    if not isinstance(paths, list) or not paths:
        raise SensingInterchangeError("at least one path is required")
    path_ids: set[str] = set()
    for index, path in enumerate(paths):
        if not isinstance(path, Mapping):
            raise SensingInterchangeError(f"paths[{index}] must be a mapping")
        if set(path) != {
            "path_id",
            "interaction_role",
            "target_id",
            "delay_s",
            "doppler",
            "angles_rad",
            "coefficient_samples_ri",
        }:
            raise SensingInterchangeError(f"paths[{index}] fields do not match the v1 contract")
        path_id = path.get("path_id")
        if not isinstance(path_id, str) or not path_id.strip() or path_id in path_ids:
            raise SensingInterchangeError("path IDs must be unique non-empty strings")
        path_ids.add(path_id)
        role = path.get("interaction_role")
        target_id = path.get("target_id")
        if role not in _PATH_ROLES:
            raise SensingInterchangeError(f"paths[{index}] has an invalid interaction role")
        if role == "target" and (not isinstance(target_id, str) or not target_id.strip()):
            raise SensingInterchangeError("target path is missing target_id")
        if role != "target" and target_id is not None:
            raise SensingInterchangeError("non-target path unexpectedly has target_id")
        if not _is_json_number(path.get("delay_s")):
            raise SensingInterchangeError("path delay must be a JSON number")
        delay = float(path["delay_s"])
        if not math.isfinite(delay) or delay < 0.0:
            raise SensingInterchangeError("path delay must be finite and nonnegative")
        doppler = path.get("doppler")
        if not isinstance(doppler, Mapping) or set(doppler) != {
            "orchav_hz",
            "positive_receding_peer_hz",
        }:
            raise SensingInterchangeError("path Doppler mapping is missing")
        if any(
            not _is_json_number(doppler.get(field))
            for field in ("orchav_hz", "positive_receding_peer_hz")
        ):
            raise SensingInterchangeError("path Doppler values must be JSON numbers")
        orchav_hz = float(doppler["orchav_hz"])
        peer_hz = float(doppler["positive_receding_peer_hz"])
        if not math.isfinite(orchav_hz) or not math.isfinite(peer_hz) or peer_hz != -orchav_hz:
            raise SensingInterchangeError("path Doppler signs are not exact opposites")
        angles = path.get("angles_rad")
        if angles is not None:
            expected_angles = {
                "departure_azimuth",
                "departure_zenith",
                "arrival_azimuth",
                "arrival_zenith",
            }
            if not isinstance(angles, Mapping) or set(angles) != expected_angles:
                raise SensingInterchangeError(f"paths[{index}] angles do not match v1")
            if any(not _is_json_number(angles[name]) for name in expected_angles):
                raise SensingInterchangeError(f"paths[{index}] angles must be JSON numbers")
            angle_values = [float(angles[name]) for name in expected_angles]
            if not all(math.isfinite(value) for value in angle_values):
                raise SensingInterchangeError(f"paths[{index}] angles must be finite")
        samples = complex_pairs_to_array(path.get("coefficient_samples_ri", []))
        if samples.size != times.size:
            raise SensingInterchangeError("path sample count disagrees with the clock")


def write_live_cir_interchange(path: Path, record: Mapping[str, Any]) -> Path:
    """Publish one record atomically without replacing an existing destination."""

    validate_live_cir_interchange(record)
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite interchange: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite interchange: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_live_cir_interchange(path: Path) -> dict[str, Any]:
    """Load and authenticate a live-CIR interchange JSON file."""

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensingInterchangeError(f"could not load interchange {path}: {exc}") from exc
    validate_live_cir_interchange(record)
    return record


def adapt_slow_time_to_positive_receding(samples: Sequence[complex]) -> np.ndarray:
    """Reverse phase evolution while preserving magnitude and the initial sample.

    ORCHAV/Sionna complex slow time and the inspected peer fixture both use
    ``exp(+j 2 pi f t)``, but their physical Doppler signs are opposite.  This
    adapter changes the tone slope exactly once; it is not a unit conversion.
    """

    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise SensingInterchangeError("slow-time samples must be finite with length >= 2")
    if abs(values[0]) <= np.finfo(np.float64).tiny:
        raise SensingInterchangeError("the initial slow-time sample must be nonzero")
    adapted = values[0] * np.conj(values / values[0])
    adapted.setflags(write=False)
    return adapted


def _unitary_doppler_stages(samples: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    sample_count = samples.size
    raw = np.fft.fftshift(np.fft.fft(samples)) / math.sqrt(sample_count)
    mean_removed_samples = samples - np.mean(samples)
    mean_removed = np.fft.fftshift(np.fft.fft(mean_removed_samples)) / math.sqrt(sample_count)
    axis = np.fft.fftshift(np.fft.fftfreq(sample_count, d=1.0 / sample_rate_hz))

    def encode(values: np.ndarray) -> list[list[float]]:
        return [[float(item.real), float(item.imag)] for item in values]

    return {
        "frequency_axis_hz": [float(item) for item in axis],
        "slow_time_ri": encode(samples),
        "raw_doppler_ri": encode(raw),
        "mean_removed_slow_time_ri": encode(mean_removed_samples),
        "common_mean_removed_doppler_ri": encode(mean_removed),
        "raw_peak_bin": int(np.argmax(np.abs(raw) ** 2)),
        "raw_peak_hz": float(axis[int(np.argmax(np.abs(raw) ** 2))]),
        "mean_removed_peak_bin": int(np.argmax(np.abs(mean_removed) ** 2)),
        "mean_removed_peak_hz": float(axis[int(np.argmax(np.abs(mean_removed) ** 2))]),
        "raw_energy": float(np.sum(np.abs(raw) ** 2)),
        "mean_removed_energy": float(np.sum(np.abs(mean_removed) ** 2)),
    }


def _single_path_stage_checks(
    orchav: Mapping[str, Any],
    peer: Mapping[str, Any],
    *,
    expected_orchav: float,
    expected_peer: float,
    bin_spacing: float,
) -> dict[str, bool]:
    is_static = abs(expected_orchav) <= bin_spacing * 1.0e-9
    checks = {
        "explicit_sign_adapter": expected_peer == -expected_orchav,
        "raw_peak_matches_orchav": abs(orchav["raw_peak_hz"] - expected_orchav)
        <= 0.5 * bin_spacing + 1.0e-12,
        "raw_peak_matches_positive_receding_peer": abs(peer["raw_peak_hz"] - expected_peer)
        <= 0.5 * bin_spacing + 1.0e-12,
        "raw_energy_preserved_by_adapter": math.isclose(
            orchav["raw_energy"], peer["raw_energy"], rel_tol=1.0e-12, abs_tol=1.0e-300
        ),
        "mean_removed_energy_preserved_by_adapter": math.isclose(
            orchav["mean_removed_energy"],
            peer["mean_removed_energy"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-300,
        ),
    }
    if is_static:
        checks["static_raw_dc_is_visible"] = (
            orchav["raw_energy"] > 0.0 and orchav["raw_peak_hz"] == 0.0
        )
        checks["static_common_mean_removal_is_zero"] = orchav["mean_removed_energy"] <= max(
            orchav["raw_energy"] * 1.0e-24, 1.0e-300
        )
    else:
        checks["mean_removed_peak_matches_orchav"] = (
            abs(orchav["mean_removed_peak_hz"] - expected_orchav) <= 0.5 * bin_spacing + 1.0e-12
        )
        checks["mean_removed_peak_matches_positive_receding_peer"] = (
            abs(peer["mean_removed_peak_hz"] - expected_peer) <= 0.5 * bin_spacing + 1.0e-12
        )
    return checks


def build_single_path_stage_evidence(
    record: Mapping[str, Any], *, path_index: int = 0
) -> dict[str, Any]:
    """Expose raw and common-MTI stages before any detector or link-budget model."""

    validate_live_cir_interchange(record)
    paths = record["paths"]
    if path_index < 0 or path_index >= len(paths):
        raise SensingInterchangeError("path_index is outside the interchange path list")
    path = paths[path_index]
    orchav_samples = complex_pairs_to_array(path["coefficient_samples_ri"])
    peer_samples = adapt_slow_time_to_positive_receding(orchav_samples)
    sample_rate = float(record["clock"]["sample_rate_hz"])
    orchav = _unitary_doppler_stages(orchav_samples, sample_rate)
    peer = _unitary_doppler_stages(peer_samples, sample_rate)
    peer_axis = np.asarray(peer["frequency_axis_hz"], dtype=np.float64)
    expected_orchav = float(path["doppler"]["orchav_hz"])
    expected_peer = float(path["doppler"]["positive_receding_peer_hz"])
    bin_spacing = sample_rate / orchav_samples.size
    checks = _single_path_stage_checks(
        orchav,
        peer,
        expected_orchav=expected_orchav,
        expected_peer=expected_peer,
        bin_spacing=bin_spacing,
    )
    evidence: dict[str, Any] = {
        "schema": "orchav.sensing.single_path_stage_evidence",
        "schema_version": 1,
        "interchange_id": record["interchange_id"],
        "case_name": record["scope"]["case_name"],
        "path_id": path["path_id"],
        "scope": {
            "pre_detection_only": True,
            "waveform_link_budget_parity_claimed": False,
            "detector_parity_claimed": False,
            "matlab_execution_claimed": False,
        },
        "expected": {
            "orchav_doppler_hz": expected_orchav,
            "positive_receding_peer_doppler_hz": expected_peer,
            "bin_spacing_hz": float(bin_spacing),
        },
        "orchav_convention_stages": orchav,
        "positive_receding_peer_reference_stages": peer,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "peer_frequency_axis_hz": [float(item) for item in peer_axis],
    }
    digest = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    evidence["content_sha256"] = digest
    evidence["stage_evidence_id"] = f"sha256:{digest}"
    validate_single_path_stage_evidence(evidence)
    return evidence


def _stage_evidence_sha256(record: Mapping[str, Any]) -> str:
    content = dict(record)
    content.pop("content_sha256", None)
    content.pop("stage_evidence_id", None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _finite_json_vector(value: Any, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or not value
        or any(type(component) not in {int, float} for component in value)
    ):
        raise SensingInterchangeError(f"{name} must be a non-empty finite numeric list")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise SensingInterchangeError(f"{name} must be a non-empty finite numeric list")
    return array


def _validate_doppler_stage_block(
    stage: Any,
    *,
    name: str,
    sample_rate_hz: float,
    expected_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    expected_fields = {
        "frequency_axis_hz",
        "slow_time_ri",
        "raw_doppler_ri",
        "mean_removed_slow_time_ri",
        "common_mean_removed_doppler_ri",
        "raw_peak_bin",
        "raw_peak_hz",
        "mean_removed_peak_bin",
        "mean_removed_peak_hz",
        "raw_energy",
        "mean_removed_energy",
    }
    if not isinstance(stage, Mapping) or set(stage) != expected_fields:
        raise SensingInterchangeError(f"{name} fields do not match the v1 contract")
    slow_time = complex_pairs_to_array(stage["slow_time_ri"])
    count = slow_time.size
    if expected_count is not None and count != expected_count:
        raise SensingInterchangeError("stage-evidence sample counts disagree")
    axis = _finite_json_vector(stage["frequency_axis_hz"], name=f"{name}.frequency_axis_hz")
    if axis.size != count:
        raise SensingInterchangeError("stage-evidence frequency and slow-time lengths disagree")
    recomputed = _unitary_doppler_stages(slow_time, sample_rate_hz)
    if not np.allclose(
        axis,
        np.asarray(recomputed["frequency_axis_hz"], dtype=np.float64),
        rtol=1.0e-13,
        atol=1.0e-13,
    ):
        raise SensingInterchangeError(f"{name} frequency axis is inconsistent")
    for field in (
        "raw_doppler_ri",
        "mean_removed_slow_time_ri",
        "common_mean_removed_doppler_ri",
    ):
        actual = complex_pairs_to_array(stage[field])
        expected = complex_pairs_to_array(recomputed[field])
        if actual.size != count or not np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-300):
            raise SensingInterchangeError(f"{name}.{field} is inconsistent with slow time")
    for field in ("raw_peak_bin", "mean_removed_peak_bin"):
        if type(stage[field]) is not int or stage[field] != recomputed[field]:
            raise SensingInterchangeError(f"{name}.{field} is inconsistent")
    for field in ("raw_peak_hz", "mean_removed_peak_hz", "raw_energy", "mean_removed_energy"):
        value = stage[field]
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value),
                float(recomputed[field]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-300,
            )
        ):
            raise SensingInterchangeError(f"{name}.{field} is inconsistent")
    return slow_time, axis


def validate_single_path_stage_evidence(record: Mapping[str, Any]) -> None:
    """Authenticate one derived pre-detection stage-evidence record."""

    if not isinstance(record, Mapping):
        raise SensingInterchangeError("stage evidence must be a mapping")
    expected_fields = {
        "schema",
        "schema_version",
        "interchange_id",
        "case_name",
        "path_id",
        "scope",
        "expected",
        "orchav_convention_stages",
        "positive_receding_peer_reference_stages",
        "checks",
        "passed",
        "peer_frequency_axis_hz",
        "content_sha256",
        "stage_evidence_id",
    }
    if set(record) != expected_fields:
        raise SensingInterchangeError("stage-evidence fields do not match the v1 contract")
    if record.get("schema") != "orchav.sensing.single_path_stage_evidence" or (
        type(record.get("schema_version")) is not int or record.get("schema_version") != 1
    ):
        raise SensingInterchangeError("unsupported single-path stage-evidence schema")
    digest = record.get("content_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
        or record.get("stage_evidence_id") != f"sha256:{digest}"
    ):
        raise SensingInterchangeError("stage-evidence identity fields disagree")
    if _stage_evidence_sha256(record) != digest:
        raise SensingInterchangeError("stage-evidence content hash mismatch")
    interchange_id = record.get("interchange_id")
    if (
        not isinstance(interchange_id, str)
        or len(interchange_id) != 71
        or not interchange_id.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in interchange_id[7:])
    ):
        raise SensingInterchangeError("stage evidence must reference an authenticated interchange")
    if any(
        not isinstance(record.get(field), str) or not record[field].strip()
        for field in ("case_name", "path_id")
    ):
        raise SensingInterchangeError("stage evidence requires non-empty case and path IDs")
    expected_scope = {
        "pre_detection_only": True,
        "waveform_link_budget_parity_claimed": False,
        "detector_parity_claimed": False,
        "matlab_execution_claimed": False,
    }
    if record.get("scope") != expected_scope:
        raise SensingInterchangeError("stage-evidence scope exceeds the v1 pre-detection boundary")
    expected = record.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {
        "orchav_doppler_hz",
        "positive_receding_peer_doppler_hz",
        "bin_spacing_hz",
    }:
        raise SensingInterchangeError("stage-evidence expected fields do not match v1")
    if any(type(expected[field]) not in {int, float} for field in expected):
        raise SensingInterchangeError("stage-evidence expectations must be numeric")
    expected_orchav = float(expected["orchav_doppler_hz"])
    expected_peer = float(expected["positive_receding_peer_doppler_hz"])
    bin_spacing = float(expected["bin_spacing_hz"])
    if (
        not all(math.isfinite(value) for value in (expected_orchav, expected_peer, bin_spacing))
        or bin_spacing <= 0.0
        or expected_peer != -expected_orchav
    ):
        raise SensingInterchangeError("stage-evidence Doppler expectations are inconsistent")
    orchav_stage = record.get("orchav_convention_stages")
    if not isinstance(orchav_stage, Mapping) or not isinstance(
        orchav_stage.get("slow_time_ri"), list
    ):
        raise SensingInterchangeError("ORCHAV stage block is invalid")
    sample_count = len(orchav_stage["slow_time_ri"])
    sample_rate_hz = bin_spacing * sample_count
    orchav_samples, orchav_axis = _validate_doppler_stage_block(
        orchav_stage,
        name="orchav_convention_stages",
        sample_rate_hz=sample_rate_hz,
    )
    peer_stage = record.get("positive_receding_peer_reference_stages")
    peer_samples, peer_axis = _validate_doppler_stage_block(
        peer_stage,
        name="positive_receding_peer_reference_stages",
        sample_rate_hz=sample_rate_hz,
        expected_count=sample_count,
    )
    if not np.allclose(
        peer_samples,
        adapt_slow_time_to_positive_receding(orchav_samples),
        rtol=1.0e-12,
        atol=1.0e-300,
    ):
        raise SensingInterchangeError("peer slow time does not apply the declared sign adapter")
    peer_frequency_axis = _finite_json_vector(
        record.get("peer_frequency_axis_hz"), name="peer_frequency_axis_hz"
    )
    if (
        peer_frequency_axis.size != peer_axis.size
        or not np.array_equal(peer_frequency_axis, peer_axis)
        or not np.array_equal(orchav_axis, peer_axis)
    ):
        raise SensingInterchangeError("stage-evidence frequency axes disagree")
    checks = record.get("checks")
    expected_checks = _single_path_stage_checks(
        orchav_stage,
        peer_stage,
        expected_orchav=expected_orchav,
        expected_peer=expected_peer,
        bin_spacing=bin_spacing,
    )
    if (
        not isinstance(checks, Mapping)
        or dict(checks) != expected_checks
        or any(type(value) is not bool for value in checks.values())
        or type(record.get("passed")) is not bool
        or record["passed"] != bool(all(checks.values()))
    ):
        raise SensingInterchangeError("stage-evidence check verdict is inconsistent")


def write_single_path_stage_evidence(path: Path, record: Mapping[str, Any]) -> Path:
    """Publish authenticated stage evidence atomically without replacing a file."""

    validate_single_path_stage_evidence(record)
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite stage evidence: {destination}")
    payload = canonical_json_bytes(record)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite stage evidence: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_single_path_stage_evidence(path: Path) -> dict[str, Any]:
    """Load and authenticate a derived pre-detection stage-evidence record."""

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensingInterchangeError(f"could not load stage evidence {path}: {exc}") from exc
    validate_single_path_stage_evidence(record)
    return record
