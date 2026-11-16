"""Shared detection trackers for optional sensing payloads.

Trackers consume timestamped detection dictionaries and return filtered
detections plus metadata. ``MofNTracker`` provides simple multi-update
confirmation, while Kalman variants smooth range/velocity or Cartesian
coordinates. All trackers implement :class:`TrackerInterface` so callers can
choose a tracker by configuration.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from shared.sensing.tracking_inspection import TrackingDecisionRecorder


@dataclass
class TrackerConfig:
    """Shared association and confirmation gates for sensing trackers.

    All association gates are strictly positive physical scales. Initial
    confirmation requires at least ``m`` hits in the last ``n`` calls for every
    tracker. In default ``latched`` mode, confirmation persists until retirement;
    ``continuous`` mode rechecks the window on every call. ``min_track_lifetime``
    separately counts cumulative hits. JPDA
    counts a hit when its marginal missed-detection probability is <= 0.5.

    Association costs divide residuals by these scales before combining them, so metres,
    metres per second, and degrees are never added directly.
    """

    m: int = 2
    n: int = 3
    max_missed_frames: int = 1
    range_gate_m: float = 0.6
    velocity_gate_m_s: float = 0.3
    azimuth_gate_deg: float | None = None
    azimuth_cost_scale_deg: float | None = None
    output_tentative: bool = False
    min_track_lifetime: int = 0  # Minimum hits before a track can appear in output (0 = use m)
    xy_gate_m: float = 2.0  # Euclidean distance gate in x,y plane (Cartesian tracker)
    # range_dot = -range_rate_scale * velocity_m_s; 2 for total bistatic path.
    range_rate_scale: float = field(default=1.0, kw_only=True)
    record_diagnostics: bool = field(default=False, kw_only=True)
    confirmation_mode: str = field(default="latched", kw_only=True)
    include_z: bool = False  # Enable 3D tracking (6-state [x,y,z,vx,vy,vz])

    def __post_init__(self) -> None:
        """Reject ambiguous or non-physical tracker parameters."""
        _validate_tracker_config(self)


def _confirmation_state(previous: bool, history: deque[int], config: TrackerConfig) -> bool:
    """Apply the initial recent-hit gate and selected confirmation lifecycle."""
    return (config.confirmation_mode == "latched" and previous) or sum(history) >= config.m


class TrackerInterface(Protocol):
    """Protocol for sensing target trackers."""

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Filter detections through the tracker.

        ``update_time_s`` is a physical timestamp on the sensing scenario
        clock. Successive calls must use finite, strictly increasing values.

        Returns:
            Tuple of (filtered detections, tracker metadata dict).
        """
        ...


def _validated_update_time(
    previous_time_s: float | None,
    update_time_s: float,
) -> tuple[float, float | None]:
    """Validate an update timestamp and return it with its elapsed interval."""
    if isinstance(update_time_s, (bool, np.bool_)):
        raise ValueError("update_time_s must be a finite number")
    try:
        timestamp_s = float(update_time_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("update_time_s must be a finite number") from exc

    if not np.isfinite(timestamp_s):
        raise ValueError("update_time_s must be a finite number")
    if previous_time_s is None:
        return timestamp_s, None

    with np.errstate(over="ignore", invalid="ignore"):
        dt_s = timestamp_s - previous_time_s
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(
            "update_time_s must be strictly increasing with a finite elapsed interval "
            f"(previous={previous_time_s}, current={timestamp_s})"
        )
    return timestamp_s, dt_s


def _validate_detection_inputs(
    detections: Any,
    *,
    required_fields: tuple[str, ...],
    require_azimuth: bool = False,
) -> None:
    """Fail closed on incomplete or non-finite physical measurements.

    Trackers never infer absent range, radial velocity, Cartesian position, or
    configured angular measurements. Validation happens before prediction or
    lifecycle state changes so a rejected call can be corrected at the same
    physical timestamp.
    """
    if not isinstance(detections, list):
        raise ValueError("detections must be a list of mappings")
    for detection_index, detection in enumerate(detections):
        path = f"detections[{detection_index}]"
        if not isinstance(detection, Mapping):
            raise ValueError(f"{path} must be a mapping")
        fields = list(required_fields)
        if require_azimuth:
            fields.append("aoa_az_deg")
        elif "aoa_az_deg" in detection:
            # An optional angle does not participate without an angular gate,
            # but retaining a malformed value would poison a later identity
            # comparison and its persisted detection metadata.
            fields.append("aoa_az_deg")
        for field_name in fields:
            if field_name not in detection:
                raise ValueError(f"{path}.{field_name} must be explicit and finite")
            raw_value = detection[field_name]
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError(f"{path}.{field_name} must be explicit and finite")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}.{field_name} must be explicit and finite") from exc
            if not np.isfinite(value):
                raise ValueError(f"{path}.{field_name} must be explicit and finite")
            if field_name == "range_m" and value < 0.0:
                raise ValueError(f"{path}.range_m must be finite and non-negative")


def _validated_positive_float(value: Any, name: str) -> float:
    """Return a finite, strictly positive floating-point model parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and strictly positive")
    try:
        validated = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and strictly positive") from exc
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return validated


def _validated_probability_open(value: Any, name: str) -> float:
    """Return a finite probability strictly between zero and one."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    try:
        validated = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(validated) or not 0.0 < validated < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return validated


def _validated_nonnegative_integer(value: Any, name: str) -> int:
    """Return an integer greater than or equal to zero without coercing floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    validated = int(value)
    if validated < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return validated


def _validate_tracker_config(config: TrackerConfig) -> None:
    """Validate the shared tracker contract once, at construction time."""
    if config.confirmation_mode not in {"latched", "continuous"}:
        raise ValueError("confirmation_mode must be latched or continuous")
    m = _validated_nonnegative_integer(config.m, "m")
    n = _validated_nonnegative_integer(config.n, "n")
    if m == 0 or n == 0:
        raise ValueError("m and n must be positive integers")
    if m > n:
        raise ValueError("m must be less than or equal to n")

    max_missed_frames = _validated_nonnegative_integer(
        config.max_missed_frames, "max_missed_frames"
    )
    min_track_lifetime = _validated_nonnegative_integer(
        config.min_track_lifetime, "min_track_lifetime"
    )
    range_gate_m = _validated_positive_float(config.range_gate_m, "range_gate_m")
    velocity_gate_m_s = _validated_positive_float(config.velocity_gate_m_s, "velocity_gate_m_s")
    xy_gate_m = _validated_positive_float(config.xy_gate_m, "xy_gate_m")
    azimuth_gate_deg = None
    if config.azimuth_gate_deg is not None:
        azimuth_gate_deg = _validated_positive_float(config.azimuth_gate_deg, "azimuth_gate_deg")
    azimuth_cost_scale_deg = None
    if config.azimuth_cost_scale_deg is not None:
        azimuth_cost_scale_deg = _validated_positive_float(
            config.azimuth_cost_scale_deg, "azimuth_cost_scale_deg"
        )
    if not isinstance(config.output_tentative, (bool, np.bool_)):
        raise ValueError("output_tentative must be boolean")
    if not isinstance(config.record_diagnostics, (bool, np.bool_)):
        raise ValueError("record_diagnostics must be boolean")
    if not isinstance(config.include_z, (bool, np.bool_)):
        raise ValueError("include_z must be boolean")

    config.range_rate_scale = _validated_positive_float(config.range_rate_scale, "range_rate_scale")
    config.m = m
    config.n = n
    config.max_missed_frames = max_missed_frames
    config.min_track_lifetime = min_track_lifetime
    config.range_gate_m = range_gate_m
    config.velocity_gate_m_s = velocity_gate_m_s
    config.xy_gate_m = xy_gate_m
    config.azimuth_gate_deg = azimuth_gate_deg
    config.azimuth_cost_scale_deg = azimuth_cost_scale_deg
    config.output_tentative = bool(config.output_tentative)
    config.include_z = bool(config.include_z)


def _gate_normalized_cost(*residual_and_scale: tuple[float, float]) -> float:
    """Combine heterogeneous residuals into a dimensionless Euclidean cost."""
    return float(
        np.sqrt(
            sum((float(residual) / float(scale)) ** 2 for residual, scale in residual_and_scale)
        )
    )


def _radial_prediction_matrices(
    dt_s: float,
    acceleration_psd_m2_s3: float,
    range_rate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build radial constant-velocity ``F`` and white-acceleration ``Q``.

    The state units are ``[m, m/s]`` and radial velocity follows ORCHAV's
    positive-approaching convention. Range therefore evolves as
    ``range_next = range - range_rate_scale * velocity * dt``. The scale is
    one for monostatic range and two for total bistatic path length, where
    velocity is the Doppler-equivalent ``lambda * f_D / 2``. Measurement
    variances and association gates remain in the reported coordinates.
    Acceleration PSD describes noise in the reported velocity derivative,
    not the derivative of total path rate. ``F`` maps a state over ``dt_s``
    and is dimensionally mixed by construction (its upper-right entry has
    units of seconds). ``acceleration_psd_m2_s3`` is a continuous
    white-acceleration spectral density in m^2/s^3, yielding ``Q`` entries
    with the matching state-covariance units
    ``[[m^2, m^2/s], [m^2/s, m^2/s^2]]``.
    """
    dt_s = _validated_positive_float(dt_s, "dt_s")
    acceleration_psd_m2_s3 = _validated_acceleration_psd(acceleration_psd_m2_s3)
    scale = _validated_positive_float(range_rate_scale, "range_rate_scale")
    with np.errstate(over="ignore", invalid="ignore"):
        dt_sq = dt_s * dt_s
        transition = np.array([[1.0, -scale * dt_s], [0.0, 1.0]])
        process_covariance = acceleration_psd_m2_s3 * np.array(
            [
                [scale * scale * dt_sq * dt_s / 3.0, -scale * dt_sq / 2.0],
                [-scale * dt_sq / 2.0, dt_s],
            ]
        )
    if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(process_covariance)):
        raise ValueError("radial prediction F/Q are outside numeric range")
    return transition, process_covariance


def _cartesian_prediction_matrices(
    dt_s: float,
    acceleration_psd_m2_s3: float,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Cartesian constant-velocity matrices with one ``Q`` block per axis.

    Positions precede velocities in the state. Each per-axis covariance block
    has units ``[[m^2, m^2/s], [m^2/s, m^2/s^2]]`` and is independent of the
    other axes.
    """
    dt_s = _validated_positive_float(dt_s, "dt_s")
    acceleration_psd_m2_s3 = _validated_acceleration_psd(acceleration_psd_m2_s3)
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    state_dim = 2 * dimensions
    transition = np.eye(state_dim)
    process_covariance = np.zeros((state_dim, state_dim))
    with np.errstate(over="ignore", invalid="ignore"):
        dt_sq = dt_s * dt_s
        q_position = acceleration_psd_m2_s3 * dt_sq * dt_s / 3.0
        q_cross = acceleration_psd_m2_s3 * dt_sq / 2.0
        q_velocity = acceleration_psd_m2_s3 * dt_s

    for axis in range(dimensions):
        velocity_axis = dimensions + axis
        transition[axis, velocity_axis] = dt_s
        process_covariance[axis, axis] = q_position
        process_covariance[axis, velocity_axis] = q_cross
        process_covariance[velocity_axis, axis] = q_cross
        process_covariance[velocity_axis, velocity_axis] = q_velocity
    if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(process_covariance)):
        raise ValueError("Cartesian prediction F/Q are outside numeric range")
    return transition, process_covariance


def _validated_track_predictions(
    tracks: Mapping[int, Any],
    transition: np.ndarray,
    process_covariance: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Precompute finite predictions before mutating tracker lifecycle state."""

    predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for track_id, track in tracks.items():
            predicted_state = transition @ track.x
            predicted_covariance = transition @ track.P @ transition.T + process_covariance
            if not np.all(np.isfinite(predicted_state)) or not np.all(
                np.isfinite(predicted_covariance)
            ):
                raise ValueError("tracker prediction state/covariance is outside numeric range")
            predictions[int(track_id)] = (predicted_state, predicted_covariance)
    return predictions


def _validated_acceleration_psd(value: float) -> float:
    """Return a finite, non-negative acceleration power spectral density."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("acceleration_psd_m2_s3 must be finite and non-negative")
    try:
        acceleration_psd = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("acceleration_psd_m2_s3 must be finite and non-negative") from exc
    if not np.isfinite(acceleration_psd) or acceleration_psd < 0.0:
        raise ValueError("acceleration_psd_m2_s3 must be finite and non-negative")
    return acceleration_psd


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    track_id: int
    range_m: float
    velocity_m_s: float
    hit_history: deque[int]
    hits: int = 1  # Cumulative associations; initial confirmation uses recent hits.
    misses: int = 0
    confirmed: bool = False
    last_detection: dict[str, Any] = field(default_factory=dict)


def _azimuth_difference_deg(lhs: Any, rhs: Any) -> float | None:
    try:
        lhs_value = float(lhs)
        rhs_value = float(rhs)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lhs_value) or not np.isfinite(rhs_value):
        return None
    return abs(((lhs_value - rhs_value + 180.0) % 360.0) - 180.0)


# ---------------------------------------------------------------------------
# M-of-N tracker
# ---------------------------------------------------------------------------


class MofNTracker:
    """M-of-N multi-frame confirmation tracker.

    A detection must appear in at least *M* of the last *N* frames to be
    emitted as confirmed.  Uses greedy nearest-neighbor association with
    range and velocity gates.

    Args:
        config: Tracker configuration.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._last_update_time_s: float | None = None

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply M-of-N confirmation over detection tracks."""
        cfg = self._config
        _validate_detection_inputs(
            detections,
            required_fields=("range_m", "velocity_m_s"),
            require_azimuth=(
                cfg.azimuth_gate_deg is not None or cfg.azimuth_cost_scale_deg is not None
            ),
        )
        timestamp_s, dt_s = _validated_update_time(self._last_update_time_s, update_time_s)
        self._last_update_time_s = timestamp_s

        recorder = (
            TrackingDecisionRecorder(
                self._tracks,
                detections,
                timestamp_s,
                ["range_m", "closing_velocity_m_s"],
                prediction_kind="last_observation",
            )
            if cfg.record_diagnostics
            else None
        )
        track_to_det: dict[int, int] = {}
        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(detections):
                cost = self._track_cost(track, det)
                if recorder is not None:
                    recorder.candidate(track_id, det_idx, cost)
                if cost is None:
                    continue
                candidates.append((cost, track_id, det_idx))

        for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0]):
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            track_to_det[track_id] = det_idx
            det_to_track[det_idx] = track_id

        for track_id, track in list(self._tracks.items()):
            det_idx = track_to_det.get(track_id)
            if det_idx is None:
                track.misses += 1
                track.hit_history.append(0)
            else:
                det = detections[det_idx]
                track.range_m = float(det["range_m"])
                track.velocity_m_s = float(det["velocity_m_s"])
                track.last_detection = dict(det)
                track.misses = 0
                track.hit_history.append(1)
                track.hits += 1
            track.confirmed = _confirmation_state(track.confirmed, track.hit_history, cfg)
            if track.misses > cfg.max_missed_frames:
                del self._tracks[track_id]

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            history: deque[int] = deque([1], maxlen=cfg.n)
            track = _TrackState(
                track_id=track_id,
                range_m=float(det["range_m"]),
                velocity_m_s=float(det["velocity_m_s"]),
                hit_history=history,
                misses=0,
                confirmed=(cfg.m <= 1),
                last_detection=dict(det),
            )
            self._tracks[track_id] = track
            det_to_track[det_idx] = track_id

        filtered: list[dict[str, Any]] = []
        for det_idx, det in enumerate(detections):
            track_id = det_to_track.get(det_idx)
            if track_id is None:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            out = dict(det)
            out["track_id"] = int(track_id)
            out["track_confirmed"] = bool(track.confirmed)
            min_life = cfg.min_track_lifetime if cfg.min_track_lifetime > 0 else cfg.m
            if (track.confirmed and track.hits >= min_life) or cfg.output_tentative:
                filtered.append(out)

        return filtered, {
            **(
                {"decisions": recorder.finish(self._tracks, det_to_track, filtered)}
                if recorder is not None
                else {}
            ),
            "enabled": True,
            "confirmation_mode": cfg.confirmation_mode,
            "m": cfg.m,
            "n": cfg.n,
            "max_missed_frames": cfg.max_missed_frames,
            "association_range_gate_m": cfg.range_gate_m,
            "association_velocity_gate_m_s": cfg.velocity_gate_m_s,
            "association_azimuth_gate_deg": cfg.azimuth_gate_deg,
            "association_azimuth_cost_scale_deg": cfg.azimuth_cost_scale_deg,
            "raw": len(detections),
            "kept": len(filtered),
            "active_tracks": len(self._tracks),
            "confirmed_tracks": sum(1 for t in self._tracks.values() if t.confirmed),
            "output_tentative": cfg.output_tentative,
            "update_time_s": timestamp_s,
            "prediction_dt_s": dt_s,
        }

    def _track_cost(self, track: _TrackState, detection: dict[str, Any]) -> float | None:
        cfg = self._config
        dr = abs(float(detection["range_m"]) - track.range_m)
        dv = abs(float(detection["velocity_m_s"]) - track.velocity_m_s)
        azimuth_diff = _azimuth_difference_deg(
            detection.get("aoa_az_deg"),
            track.last_detection.get("aoa_az_deg"),
        )

        if cfg.range_gate_m > 0.0 and dr > cfg.range_gate_m:
            return None
        if cfg.velocity_gate_m_s > 0.0 and dv > cfg.velocity_gate_m_s:
            return None
        if (
            cfg.azimuth_gate_deg is not None
            and cfg.azimuth_gate_deg > 0.0
            and azimuth_diff is not None
            and azimuth_diff > cfg.azimuth_gate_deg
        ):
            return None
        components = [(dr, cfg.range_gate_m), (dv, cfg.velocity_gate_m_s)]
        azimuth_scale = cfg.azimuth_cost_scale_deg
        if azimuth_scale is None:
            azimuth_scale = cfg.azimuth_gate_deg
        if azimuth_scale is not None and azimuth_diff is not None:
            components.append((azimuth_diff, azimuth_scale))
        return _gate_normalized_cost(*components)


# ---------------------------------------------------------------------------
# Kalman tracker
# ---------------------------------------------------------------------------


@dataclass
class _KalmanTrackState:
    track_id: int
    x: np.ndarray  # state [range, velocity]
    P: np.ndarray  # covariance 2x2
    misses: int = 0
    hit_history: deque[int] = field(default_factory=deque)
    hits: int = 1
    confirmed: bool = False
    last_detection: dict[str, Any] = field(default_factory=dict)


class KalmanTracker:
    """Constant-velocity Kalman tracker with Hungarian association.

    State vector: ``[range_m, velocity_m_s]``, with positive velocity meaning
    approach. Prediction therefore decreases range for a positive velocity.
    It uses the elapsed physical time between successive updates and a
    continuous-white-acceleration process model. Association uses
    :func:`scipy.optimize.linear_sum_assignment` (Hungarian algorithm).

    Args:
        config: Tracker configuration (same as M-of-N).
        acceleration_psd_m2_s3: Noise PSD of the reported velocity derivative.
            For bistatic equivalent velocity this is one quarter of path-acceleration PSD.
        measurement_noise_range: Measurement noise variance for range (m^2).
        measurement_noise_velocity: Measurement noise variance for velocity (m/s)^2.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        acceleration_psd_m2_s3: float = 0.5,
        measurement_noise_range: float = 0.25,
        measurement_noise_velocity: float = 0.1,
    ) -> None:
        self._config = config or TrackerConfig()
        self._tracks: dict[int, _KalmanTrackState] = {}
        self._next_track_id = 1
        self._last_update_time_s: float | None = None
        self._acceleration_psd_m2_s3 = _validated_acceleration_psd(acceleration_psd_m2_s3)
        self._F: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        measurement_noise_range = _validated_positive_float(
            measurement_noise_range, "measurement_noise_range"
        )
        measurement_noise_velocity = _validated_positive_float(
            measurement_noise_velocity, "measurement_noise_velocity"
        )
        # Dimensionless observation matrix: observe [range_m, velocity_m_s].
        self._H = np.eye(2)
        # R diagonal units are [m^2, (m/s)^2].
        self._R = np.diag([measurement_noise_range, measurement_noise_velocity])

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Predict, associate, update, and manage track lifecycle."""
        cfg = self._config
        _validate_detection_inputs(
            detections,
            required_fields=("range_m", "velocity_m_s"),
            require_azimuth=(
                cfg.azimuth_gate_deg is not None or cfg.azimuth_cost_scale_deg is not None
            ),
        )
        timestamp_s, dt_s = _validated_update_time(self._last_update_time_s, update_time_s)

        # 1. Predict all existing tracks
        predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        transition: np.ndarray | None = None
        process_covariance: np.ndarray | None = None
        if dt_s is not None:
            transition, process_covariance = _radial_prediction_matrices(
                dt_s,
                self._acceleration_psd_m2_s3,
                cfg.range_rate_scale,
            )
            predictions = _validated_track_predictions(
                self._tracks,
                transition,
                process_covariance,
            )
        self._last_update_time_s = timestamp_s
        if transition is not None and process_covariance is not None:
            self._F, self._Q = transition, process_covariance
            for track_id, (predicted_state, predicted_covariance) in predictions.items():
                track = self._tracks[track_id]
                track.x = predicted_state
                track.P = predicted_covariance

        # 2. Build cost matrix and associate via Hungarian algorithm
        recorder = (
            TrackingDecisionRecorder(
                self._tracks, detections, timestamp_s, ["range_m", "closing_velocity_m_s"]
            )
            if cfg.record_diagnostics
            else None
        )
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        matched_track_to_det: dict[int, int] = {}
        matched_det_to_track: dict[int, int] = {}

        if n_tracks > 0 and n_dets > 0:
            cost_matrix = np.full((n_tracks, n_dets), 1e9)
            for ti, tid in enumerate(track_ids):
                track = self._tracks[tid]
                for di, det in enumerate(detections):
                    dr = abs(float(det["range_m"]) - float(track.x[0]))
                    dv = abs(float(det["velocity_m_s"]) - float(track.x[1]))
                    azimuth_diff = _azimuth_difference_deg(
                        det.get("aoa_az_deg"),
                        track.last_detection.get("aoa_az_deg"),
                    )
                    if cfg.range_gate_m > 0.0 and dr > cfg.range_gate_m:
                        continue
                    if cfg.velocity_gate_m_s > 0.0 and dv > cfg.velocity_gate_m_s:
                        continue
                    if (
                        cfg.azimuth_gate_deg is not None
                        and cfg.azimuth_gate_deg > 0.0
                        and azimuth_diff is not None
                        and azimuth_diff > cfg.azimuth_gate_deg
                    ):
                        continue
                    components = [(dr, cfg.range_gate_m), (dv, cfg.velocity_gate_m_s)]
                    azimuth_scale = cfg.azimuth_cost_scale_deg
                    if azimuth_scale is None:
                        azimuth_scale = cfg.azimuth_gate_deg
                    if azimuth_scale is not None and azimuth_diff is not None:
                        components.append((azimuth_diff, azimuth_scale))
                    cost_matrix[ti, di] = _gate_normalized_cost(*components)

            if recorder is not None:
                recorder.matrix(track_ids, cost_matrix)
            row_ind, col_ind = _linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e8:
                    matched_track_to_det[track_ids[r]] = int(c)
                    matched_det_to_track[int(c)] = track_ids[r]

        # 3. Update matched tracks
        for tid, di in matched_track_to_det.items():
            track = self._tracks[tid]
            det = detections[di]
            z = np.array([float(det["range_m"]), float(det["velocity_m_s"])])
            y = z - self._H @ track.x  # innovation
            S = self._H @ track.P @ self._H.T + self._R
            K = track.P @ self._H.T @ np.linalg.inv(S)
            track.x = track.x + K @ y
            identity_minus_kh = np.eye(2) - K @ self._H
            track.P = identity_minus_kh @ track.P @ identity_minus_kh.T + K @ self._R @ K.T
            track.misses = 0
            track.hits += 1
            track.hit_history.append(1)
            track.confirmed = _confirmation_state(track.confirmed, track.hit_history, cfg)
            track.last_detection = dict(det)

        # 4. Handle unmatched tracks (coast)
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_to_det:
                self._tracks[tid].misses += 1
                self._tracks[tid].hit_history.append(0)
                self._tracks[tid].confirmed = _confirmation_state(
                    self._tracks[tid].confirmed, self._tracks[tid].hit_history, cfg
                )
                if self._tracks[tid].misses > cfg.max_missed_frames:
                    del self._tracks[tid]

        # 5. Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_to_track:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            x0 = np.array([float(det["range_m"]), float(det["velocity_m_s"])])
            self._tracks[tid] = _KalmanTrackState(
                track_id=tid,
                hit_history=deque([1], maxlen=cfg.n),
                x=x0,
                P=self._R.copy() * 4.0,  # initial uncertainty
                confirmed=(cfg.m <= 1),
                last_detection=dict(det),
            )
            matched_det_to_track[di] = tid

        # 6. Build output
        filtered: list[dict[str, Any]] = []
        for di, det in enumerate(detections):
            tid = matched_det_to_track.get(di)
            if tid is None:
                continue
            track = self._tracks.get(tid)
            if track is None:
                continue
            out = dict(det)
            out["track_id"] = int(tid)
            out["track_confirmed"] = bool(track.confirmed)
            # Use Kalman-smoothed state for output
            out["track_range_m"] = float(track.x[0])
            out["track_velocity_m_s"] = float(track.x[1])
            min_life = cfg.min_track_lifetime if cfg.min_track_lifetime > 0 else cfg.m
            if (track.confirmed and track.hits >= min_life) or cfg.output_tentative:
                filtered.append(out)

        return filtered, {
            **(
                {"decisions": recorder.finish(self._tracks, matched_det_to_track, filtered)}
                if recorder is not None
                else {}
            ),
            "enabled": True,
            "tracker_type": "kalman",
            "range_rate_scale": cfg.range_rate_scale,
            "range_rate_convention": "range_dot=-range_rate_scale*velocity_m_s",
            "acceleration_psd_coordinate": "reported_velocity_derivative",
            "confirmation_mode": cfg.confirmation_mode,
            "m": cfg.m,
            "n": cfg.n,
            "max_missed_frames": cfg.max_missed_frames,
            "association_range_gate_m": cfg.range_gate_m,
            "association_velocity_gate_m_s": cfg.velocity_gate_m_s,
            "association_azimuth_gate_deg": cfg.azimuth_gate_deg,
            "association_azimuth_cost_scale_deg": cfg.azimuth_cost_scale_deg,
            "acceleration_psd_m2_s3": self._acceleration_psd_m2_s3,
            "raw": len(detections),
            "kept": len(filtered),
            "active_tracks": len(self._tracks),
            "confirmed_tracks": sum(1 for t in self._tracks.values() if t.confirmed),
            "output_tentative": cfg.output_tentative,
            "update_time_s": timestamp_s,
            "prediction_dt_s": dt_s,
        }


# ---------------------------------------------------------------------------
# Cartesian Kalman tracker
# ---------------------------------------------------------------------------


@dataclass
class _CartesianTrackState:
    track_id: int
    x: np.ndarray  # state [x, y, vx, vy]
    P: np.ndarray  # covariance 4x4
    misses: int = 0
    hit_history: deque[int] = field(default_factory=deque)
    hits: int = 1
    confirmed: bool = False
    last_detection: dict[str, Any] = field(default_factory=dict)


class CartesianKalmanTracker:
    """Constant-velocity Kalman tracker in 2D or 3D Cartesian coordinates.

    When ``include_z=False`` (default), the state vector is ``[x, y, vx, vy]``
    with measurements ``[x_m, y_m]``.  When ``include_z=True``, the state
    vector is ``[x, y, z, vx, vy, vz]`` with measurements ``[x_m, y_m, z_m]``.

    Association uses :func:`scipy.optimize.linear_sum_assignment` with
    Euclidean distance gate.

    Args:
        config: Tracker configuration.
        include_z: Enable 3D tracking with elevation (6-state model).
        acceleration_psd_m2_s3: Per-axis acceleration-noise power spectral density.
        measurement_noise_xy: Measurement position variance for every observed
            Cartesian axis (m^2).
        initial_velocity_variance_m2_s2: Initial per-axis velocity variance
            ((m/s)^2). This is independent of the acceleration PSD because
            the two quantities have different units.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        include_z: bool = False,
        acceleration_psd_m2_s3: float = 1.0,
        measurement_noise_xy: float = 0.5,
        initial_velocity_variance_m2_s2: float = 4.0,
    ) -> None:
        self._config = config or TrackerConfig()
        self._tracks: dict[int, _CartesianTrackState] = {}
        self._next_track_id = 1
        self._last_update_time_s: float | None = None
        self._acceleration_psd_m2_s3 = _validated_acceleration_psd(acceleration_psd_m2_s3)
        measurement_noise_xy = _validated_positive_float(
            measurement_noise_xy, "measurement_noise_xy"
        )
        self._initial_velocity_variance_m2_s2 = _validated_positive_float(
            initial_velocity_variance_m2_s2,
            "initial_velocity_variance_m2_s2",
        )
        if not isinstance(include_z, (bool, np.bool_)):
            raise ValueError("include_z must be boolean")
        self._include_z = include_z
        self._state_dim = 6 if include_z else 4
        self._meas_dim = 3 if include_z else 2
        self._F: np.ndarray | None = None
        self._Q: np.ndarray | None = None

        if include_z:
            # Dimensionless H observes [x_m, y_m, z_m].
            self._H = np.zeros((3, 6))
            self._H[0, 0] = 1.0
            self._H[1, 1] = 1.0
            self._H[2, 2] = 1.0
            # R is a position covariance; every diagonal entry has units m^2.
            self._R = np.diag([measurement_noise_xy, measurement_noise_xy, measurement_noise_xy])
        else:
            # Dimensionless H observes [x_m, y_m] only.
            self._H = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
            # R is a position covariance; every diagonal entry has units m^2.
            self._R = np.diag([measurement_noise_xy, measurement_noise_xy])

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Predict, associate, update, and manage track lifecycle."""
        cfg = self._config
        if isinstance(detections, list) and any(
            isinstance(det, Mapping) and ("x_m" not in det or "y_m" not in det)
            for det in detections
        ):
            raise ValueError(
                "Cartesian tracking requires localized receiver-frame x_m/y_m; "
                "bistatic total path length and bearing alone are not localized. "
                "Use a radial tracker or supply independently localized coordinates."
            )
        required_fields = ["range_m", "velocity_m_s", "x_m", "y_m"]
        if self._include_z:
            required_fields.append("z_m")
        _validate_detection_inputs(
            detections,
            required_fields=tuple(required_fields),
        )
        timestamp_s, dt_s = _validated_update_time(self._last_update_time_s, update_time_s)

        # 1. Predict all existing tracks
        predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        transition: np.ndarray | None = None
        process_covariance: np.ndarray | None = None
        if dt_s is not None:
            dimensions = 3 if self._include_z else 2
            transition, process_covariance = _cartesian_prediction_matrices(
                dt_s,
                self._acceleration_psd_m2_s3,
                dimensions,
            )
            predictions = _validated_track_predictions(
                self._tracks,
                transition,
                process_covariance,
            )
        self._last_update_time_s = timestamp_s
        if transition is not None and process_covariance is not None:
            self._F, self._Q = transition, process_covariance
            for track_id, (predicted_state, predicted_covariance) in predictions.items():
                track = self._tracks[track_id]
                track.x = predicted_state
                track.P = predicted_covariance

        # 2. Build cost matrix and associate via Hungarian algorithm. The
        #    call-boundary validation above guarantees finite coordinates.
        recorder = (
            TrackingDecisionRecorder(
                self._tracks,
                detections,
                timestamp_s,
                (
                    ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"]
                    if self._include_z
                    else ["x_m", "y_m", "vx_m_s", "vy_m_s"]
                ),
            )
            if cfg.record_diagnostics
            else None
        )
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)

        valid_dets: list[tuple[int, dict[str, Any], float, float, float]] = []
        for di, det in enumerate(detections):
            xf = float(det["x_m"])
            yf = float(det["y_m"])
            zf = float(det["z_m"]) if self._include_z else 0.0
            valid_dets.append((di, det, xf, yf, zf))
        n_valid = len(valid_dets)

        matched_track_to_det: dict[int, int] = {}
        matched_det_to_track: dict[int, int] = {}

        if n_tracks > 0 and n_valid > 0:
            cost_matrix = np.full((n_tracks, n_valid), 1e9)
            for ti, tid in enumerate(track_ids):
                track = self._tracks[tid]
                pred_x, pred_y = float(track.x[0]), float(track.x[1])
                pred_z = float(track.x[2]) if self._include_z else 0.0
                for vi, (di, det, xf, yf, zf) in enumerate(valid_dets):
                    if self._include_z:
                        dist = float(
                            np.sqrt((xf - pred_x) ** 2 + (yf - pred_y) ** 2 + (zf - pred_z) ** 2)
                        )
                    else:
                        dist = float(np.hypot(xf - pred_x, yf - pred_y))
                    if dist > cfg.xy_gate_m:
                        continue
                    dv = abs(
                        float(det["velocity_m_s"]) - float(track.last_detection["velocity_m_s"])
                    )
                    if dv > cfg.velocity_gate_m_s:
                        continue
                    cost_matrix[ti, vi] = _gate_normalized_cost(
                        (dist, cfg.xy_gate_m),
                        (dv, cfg.velocity_gate_m_s),
                    )

            if recorder is not None:
                recorder.matrix(track_ids, cost_matrix)
            row_ind, col_ind = _linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e8:
                    orig_di = valid_dets[c][0]
                    matched_track_to_det[track_ids[r]] = orig_di
                    matched_det_to_track[orig_di] = track_ids[r]

        # 3. Update matched tracks
        for tid, di in matched_track_to_det.items():
            track = self._tracks[tid]
            det = detections[di]
            if self._include_z:
                z = np.array([float(det["x_m"]), float(det["y_m"]), float(det["z_m"])])
            else:
                z = np.array([float(det["x_m"]), float(det["y_m"])])
            y = z - self._H @ track.x  # innovation
            S = self._H @ track.P @ self._H.T + self._R
            K = track.P @ self._H.T @ np.linalg.inv(S)
            track.x = track.x + K @ y
            identity_minus_kh = np.eye(self._state_dim) - K @ self._H
            track.P = identity_minus_kh @ track.P @ identity_minus_kh.T + K @ self._R @ K.T
            track.misses = 0
            track.hits += 1
            track.hit_history.append(1)
            track.confirmed = _confirmation_state(track.confirmed, track.hit_history, cfg)
            track.last_detection = dict(det)

        # 4. Handle unmatched tracks (coast)
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_to_det:
                self._tracks[tid].misses += 1
                self._tracks[tid].hit_history.append(0)
                self._tracks[tid].confirmed = _confirmation_state(
                    self._tracks[tid].confirmed, self._tracks[tid].hit_history, cfg
                )
                if self._tracks[tid].misses > cfg.max_missed_frames:
                    del self._tracks[tid]

        # 5. Create new tracks for unmatched detections (valid position only)
        for di, det, xf, yf, zf in valid_dets:
            if di in matched_det_to_track:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            if self._include_z:
                x0 = np.array([xf, yf, zf, 0.0, 0.0, 0.0])
                p0 = np.diag(
                    [
                        self._R[0, 0] * 4.0,
                        self._R[1, 1] * 4.0,
                        self._R[2, 2] * 4.0,
                        self._initial_velocity_variance_m2_s2,
                        self._initial_velocity_variance_m2_s2,
                        self._initial_velocity_variance_m2_s2,
                    ]
                )
            else:
                x0 = np.array([xf, yf, 0.0, 0.0])
                p0 = np.diag(
                    [
                        self._R[0, 0] * 4.0,
                        self._R[1, 1] * 4.0,
                        self._initial_velocity_variance_m2_s2,
                        self._initial_velocity_variance_m2_s2,
                    ]
                )
            self._tracks[tid] = _CartesianTrackState(
                track_id=tid,
                hit_history=deque([1], maxlen=cfg.n),
                x=x0,
                P=p0,
                confirmed=(cfg.m <= 1),
                last_detection=dict(det),
            )
            matched_det_to_track[di] = tid

        # 6. Build output
        filtered: list[dict[str, Any]] = []
        for di, det in enumerate(detections):
            tid = matched_det_to_track.get(di)
            if tid is None:
                continue
            track = self._tracks.get(tid)
            if track is None:
                continue
            out = dict(det)
            out["track_id"] = int(tid)
            out["track_confirmed"] = bool(track.confirmed)
            # Cartesian state output
            out["track_x_m"] = float(track.x[0])
            out["track_y_m"] = float(track.x[1])
            if self._include_z:
                out["track_z_m"] = float(track.x[2])
                out["track_vx_m_s"] = float(track.x[3])
                out["track_vy_m_s"] = float(track.x[4])
                out["track_vz_m_s"] = float(track.x[5])
            else:
                out["track_vx_m_s"] = float(track.x[2])
                out["track_vy_m_s"] = float(track.x[3])
            # Report horizontal Cartesian range and the explicit raw Doppler
            # velocity; projecting noisy [vx, vy] would change its semantics.
            out["track_range_m"] = float(np.hypot(track.x[0], track.x[1]))
            out["track_velocity_m_s"] = float(det["velocity_m_s"])
            min_life = cfg.min_track_lifetime if cfg.min_track_lifetime > 0 else cfg.m
            if (track.confirmed and track.hits >= min_life) or cfg.output_tentative:
                filtered.append(out)

        tracker_type = "cartesian3d" if self._include_z else "cartesian"
        return filtered, {
            **(
                {"decisions": recorder.finish(self._tracks, matched_det_to_track, filtered)}
                if recorder is not None
                else {}
            ),
            "enabled": True,
            "tracker_type": tracker_type,
            "confirmation_mode": cfg.confirmation_mode,
            "m": cfg.m,
            "n": cfg.n,
            "max_missed_frames": cfg.max_missed_frames,
            "association_xy_gate_m": cfg.xy_gate_m,
            "association_velocity_gate_m_s": cfg.velocity_gate_m_s,
            "acceleration_psd_m2_s3": self._acceleration_psd_m2_s3,
            "initial_velocity_variance_m2_s2": self._initial_velocity_variance_m2_s2,
            "raw": len(detections),
            "kept": len(filtered),
            "active_tracks": len(self._tracks),
            "confirmed_tracks": sum(1 for t in self._tracks.values() if t.confirmed),
            "output_tentative": cfg.output_tentative,
            "update_time_s": timestamp_s,
            "prediction_dt_s": dt_s,
        }


# ---------------------------------------------------------------------------
# JPDAF tracker
# ---------------------------------------------------------------------------


@dataclass
class _JPDATrackState:
    track_id: int
    x: np.ndarray  # state [range, velocity]
    P: np.ndarray  # covariance 2x2
    misses: int = 0
    hit_history: deque[int] = field(default_factory=deque)
    hits: int = 1
    confirmed: bool = False
    last_detection: dict[str, Any] = field(default_factory=dict)


class JointAssociationLimitError(RuntimeError):
    """Exact JPDA search exceeded its explicit computational budget."""


def _joint_association_marginals(
    log_weights: dict[int, dict[int, float]],
    log_miss: float,
    *,
    max_nodes: int = 100_000,
) -> tuple[dict[int, dict[int | None, float]], int]:
    """Sum legal joint events with no reused detection, in log space.

    Each event has product weight: missed=(1-Pd), assigned=Pd*pdf/clutter.
    Independent gate components factor exactly. The node budget includes
    partial events, bounding work even when there are few complete events.
    """
    remaining = set(log_weights)
    result: dict[int, dict[int | None, float]] = {}
    nodes = 0
    while remaining:
        cluster = {min(remaining)}
        detections = set(log_weights[next(iter(cluster))])
        while True:
            linked = {
                tid for tid in remaining - cluster if detections.intersection(log_weights[tid])
            }
            if not linked:
                break
            cluster |= linked
            for tid in linked:
                detections.update(log_weights[tid])
        remaining -= cluster
        tids = sorted(cluster, key=lambda tid: (len(log_weights[tid]), tid))
        accum = {tid: {None: -np.inf, **{di: -np.inf for di in log_weights[tid]}} for tid in tids}
        choices: list[int | None] = []
        used: set[int] = set()
        norm = -np.inf

        # Explicit DFS stack avoids Python recursion limits for large sparse
        # components. A frame stores depth, accumulated log weight and options.
        stack = [(0, 0.0, iter([(None, log_miss), *log_weights[tids[0]].items()]))]
        nodes += 1
        while stack:
            if nodes > max_nodes:
                raise JointAssociationLimitError(
                    f"Exact JPDA exceeded max_joint_nodes={max_nodes}; tighten gates or explicitly raise the budget"
                )
            k, log_weight, options = stack[-1]
            try:
                choice, option_weight = next(options)
            except StopIteration:
                stack.pop()
                if choices:
                    previous_choice = choices.pop()
                    if previous_choice is not None:
                        used.remove(previous_choice)
                continue
            if choice is not None and choice in used:
                continue
            nodes += 1
            if nodes > max_nodes:
                raise JointAssociationLimitError(
                    f"Exact JPDA exceeded max_joint_nodes={max_nodes}; tighten gates or explicitly raise the budget"
                )
            choices.append(choice)
            if choice is not None:
                used.add(choice)
            child_weight = log_weight + option_weight
            if k + 1 == len(tids):
                norm = np.logaddexp(norm, child_weight)
                for tid, selected in zip(tids, choices):
                    accum[tid][selected] = np.logaddexp(accum[tid][selected], child_weight)
                choices.pop()
                if choice is not None:
                    used.remove(choice)
            else:
                stack.append(
                    (
                        k + 1,
                        child_weight,
                        iter([(None, log_miss), *log_weights[tids[k + 1]].items()]),
                    )
                )
        result.update(
            {
                tid: {di: float(np.exp(lw - norm)) for di, lw in values.items()}
                for tid, values in accum.items()
            }
        )
    return result, nodes


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rectangular assignment, including without optional SciPy."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _linear_sum_assignment_numpy(cost)
    return linear_sum_assignment(cost)


def _linear_sum_assignment_numpy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shortest augmenting path Hungarian algorithm, O(min(n,m)^2 max(n,m))."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.all(np.isfinite(cost)):
        raise ValueError("assignment cost must be a finite matrix")
    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T if transposed else cost
    n, m = a.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    columns = np.flatnonzero(p[1:])
    rows = p[columns + 1] - 1
    if transposed:
        rows, columns = columns, rows
    order = np.argsort(rows)
    return rows[order], columns[order]


def _maximum_weight_one_to_one_assignments(
    track_betas: dict[int, dict[int | None, float]],
    track_ids: list[int],
    detection_count: int,
) -> dict[int, int]:
    """Resolve marginal JPDA weights into one-to-one reporting assignments.

    JPDA state updates intentionally use all gated detections. Published
    detection-to-track identities cannot do that: each detection and each
    track must occur at most once. This maximum-weight bipartite assignment
    supplies that separate hard reporting view.
    """
    if not track_ids or detection_count == 0:
        return {}

    weights = np.zeros((len(track_ids), detection_count), dtype=float)
    for row, track_id in enumerate(track_ids):
        for detection_index, beta in track_betas.get(track_id, {}).items():
            if detection_index is not None:
                weights[row, detection_index] = max(0.0, float(beta))

    rows, columns = _linear_sum_assignment(-weights)
    return {
        int(column): int(track_ids[row])
        for row, column in zip(rows, columns)
        if weights[row, column] > 0.0
    }


class JPDAFTracker:
    """Joint Probabilistic Data Association Filter tracker.

    Radial velocity follows the same positive-approaching convention as the
    sensing products. Unlike the nearest-neighbor :class:`KalmanTracker`, JPDAF computes
    probabilistic association weights for ALL feasible track-detection
    pairs.  Each detection contributes to every gated track's state update
    weighted by its likelihood.  This handles:

    - **Crossing targets**: probabilistic handling of ambiguous associations;
      exact JPDA can still coalesce and does not guarantee identity preservation
    - **Ghost/clutter**: false alarm hypothesis absorbs unlikely detections
    - **Missed detections**: track coasts with probability-weighted uncertainty

    Enumerates feasible joint events within connected ambiguity clusters in
    log space. Detection exclusivity is enforced before marginalization.
    Worst-case cost is exponential; ``max_joint_nodes`` bounds search work
    per frame and raises ``JointAssociationLimitError`` without advancing state.
    ``pd`` is the effective detection probability inside the configured gates
    (equivalent to Stone Soup PDAHypothesiser with prob_gate=1).
    Persistent unclaimed observations passing the separation policy may initiate in-gate births
    after ``birth_confirmation_frames`` consecutive frames (default three).
    ``birth_separation_chi2`` (default 1.0) rejects candidate observations
    within that squared R-normalized distance of a claimed report. This is
    a configurable duplicate-suppression policy, not physical resolution;
    R represents measurement uncertainty. Zero blocks only exact duplicates.

    Args:
        config: Tracker configuration (m, n, max_missed_frames, gates).
        acceleration_psd_m2_s3: Noise PSD of the reported velocity derivative.
            For bistatic equivalent velocity this is one quarter of path-acceleration PSD.
        measurement_noise_range: Measurement noise variance for range (m^2).
        measurement_noise_velocity: Measurement noise variance for velocity (m/s)^2.
        pd: Probability of detection strictly between zero and one. Prior weight for the
            "target detected" vs "missed" hypothesis.
        beta_fa: False alarm spatial density.  Controls how readily
            detections are attributed to clutter vs tracks.
        gate_threshold: Chi-squared Mahalanobis gate threshold.
            9.21 = 99% for 2 DOF.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        acceleration_psd_m2_s3: float = 0.5,
        measurement_noise_range: float = 0.25,
        measurement_noise_velocity: float = 0.1,
        pd: float = 0.9,
        beta_fa: float = 0.01,
        gate_threshold: float = 9.21,
        max_joint_nodes: int = 100_000,
        birth_confirmation_frames: int = 3,
        birth_separation_chi2: float = 1.0,
    ) -> None:
        if isinstance(birth_separation_chi2, (bool, np.bool_)):
            raise ValueError("birth_separation_chi2 must be finite and non-negative")
        try:
            self._birth_separation_chi2 = float(birth_separation_chi2)
        except (TypeError, ValueError) as exc:
            raise ValueError("birth_separation_chi2 must be finite and non-negative") from exc
        if not np.isfinite(self._birth_separation_chi2) or self._birth_separation_chi2 < 0.0:
            raise ValueError("birth_separation_chi2 must be finite and non-negative")
        self._max_joint_nodes = _validated_nonnegative_integer(max_joint_nodes, "max_joint_nodes")
        self._birth_confirmation_frames = _validated_nonnegative_integer(
            birth_confirmation_frames, "birth_confirmation_frames"
        )
        if self._max_joint_nodes < 1 or self._birth_confirmation_frames < 2:
            raise ValueError("max_joint_nodes must be positive and birth_confirmation_frames >= 2")
        self._birth_candidates: list[tuple[np.ndarray, int]] = []
        self._config = config or TrackerConfig()
        self._tracks: dict[int, _JPDATrackState] = {}
        self._next_track_id = 1
        self._last_update_time_s: float | None = None
        self._acceleration_psd_m2_s3 = _validated_acceleration_psd(acceleration_psd_m2_s3)
        self._F: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        measurement_noise_range = _validated_positive_float(
            measurement_noise_range, "measurement_noise_range"
        )
        measurement_noise_velocity = _validated_positive_float(
            measurement_noise_velocity, "measurement_noise_velocity"
        )
        pd = _validated_probability_open(pd, "pd")
        beta_fa = _validated_positive_float(beta_fa, "beta_fa")
        gate_threshold = _validated_positive_float(gate_threshold, "gate_threshold")
        # Dimensionless H observes [range_m, velocity_m_s].
        self._H = np.eye(2)
        # R diagonal units are [m^2, (m/s)^2].
        self._R = np.diag([measurement_noise_range, measurement_noise_velocity])
        self._pd = pd
        self._beta_fa = beta_fa
        self._gate_threshold = gate_threshold

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Predict, gate, compute JPDA weights, update, and manage tracks."""
        cfg = self._config
        _validate_detection_inputs(
            detections,
            required_fields=("range_m", "velocity_m_s"),
            require_azimuth=(
                cfg.azimuth_gate_deg is not None or cfg.azimuth_cost_scale_deg is not None
            ),
        )
        timestamp_s, dt_s = _validated_update_time(self._last_update_time_s, update_time_s)

        # 1. Predict all existing tracks
        predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        transition: np.ndarray | None = None
        process_covariance: np.ndarray | None = None
        if dt_s is not None:
            transition, process_covariance = _radial_prediction_matrices(
                dt_s,
                self._acceleration_psd_m2_s3,
                cfg.range_rate_scale,
            )
            predictions = _validated_track_predictions(
                self._tracks,
                transition,
                process_covariance,
            )
        previous_time = self._last_update_time_s
        previous_fq = (self._F, self._Q)
        previous_states = {tid: (t.x, t.P) for tid, t in self._tracks.items()}
        self._last_update_time_s = timestamp_s
        if transition is not None and process_covariance is not None:
            self._F, self._Q = transition, process_covariance
            for track_id, (predicted_state, predicted_covariance) in predictions.items():
                track = self._tracks[track_id]
                track.x = predicted_state
                track.P = predicted_covariance

        # 2. Gate: compute Mahalanobis distance for each track-detection pair
        recorder = (
            TrackingDecisionRecorder(
                self._tracks, detections, timestamp_s, ["range_m", "closing_velocity_m_s"]
            )
            if cfg.record_diagnostics
            else None
        )
        track_ids = list(self._tracks.keys())
        n_dets = len(detections)

        # gated[tid] = [(det_idx, innovation, mahal_dist_sq, log_likelihood)]
        gated: dict[int, list[tuple[int, np.ndarray, float, float]]] = {
            tid: [] for tid in track_ids
        }

        for tid in track_ids:
            track = self._tracks[tid]
            S = self._H @ track.P @ self._H.T + self._R
            S_inv = np.linalg.inv(S)
            log_det_S = float(np.log(max(np.linalg.det(S), 1e-30)))

            for di, det in enumerate(detections):
                z = np.array([float(det["range_m"]), float(det["velocity_m_s"])])
                nu = z - self._H @ track.x
                d2 = float(nu @ S_inv @ nu)
                if abs(float(nu[0])) > cfg.range_gate_m:
                    continue
                if abs(float(nu[1])) > cfg.velocity_gate_m_s:
                    continue
                azimuth_diff = _azimuth_difference_deg(
                    det.get("aoa_az_deg"),
                    track.last_detection.get("aoa_az_deg"),
                )

                if d2 >= self._gate_threshold:
                    continue
                if (
                    cfg.azimuth_gate_deg is not None
                    and cfg.azimuth_gate_deg > 0.0
                    and azimuth_diff is not None
                    and azimuth_diff > cfg.azimuth_gate_deg
                ):
                    continue

                log_lik = -0.5 * (2 * np.log(2 * np.pi) + log_det_S + d2)
                azimuth_scale = cfg.azimuth_cost_scale_deg
                if azimuth_scale is None or azimuth_scale <= 0.0:
                    azimuth_scale = cfg.azimuth_gate_deg
                if azimuth_scale is not None and azimuth_scale > 0.0 and azimuth_diff is not None:
                    log_lik -= 0.5 * (azimuth_diff / azimuth_scale) ** 2
                gated[tid].append((di, nu, d2, log_lik))

        # Exact exclusion-consistent joint hypotheses, factored by connected gates.
        # Fail explicitly at the configured search budget; never substitute PDA.
        try:
            track_betas, hypothesis_nodes = _joint_association_marginals(
                {
                    tid: {
                        di: np.log(self._pd) + ll - np.log(self._beta_fa)
                        for di, _nu, _d2, ll in gated[tid]
                    }
                    for tid in track_ids
                },
                np.log1p(-self._pd),
                max_nodes=self._max_joint_nodes,
            )
        except BaseException:
            # Preserve transactional prediction even on unexpected search failure
            # or cancellation; reraising keeps the original failure visible.
            self._last_update_time_s = previous_time
            self._F, self._Q = previous_fq
            for tid, (x, p) in previous_states.items():
                self._tracks[tid].x, self._tracks[tid].P = x, p
            raise

        if recorder is not None:
            for tid in track_ids:
                accepted = {di: d2 for di, _, d2, _ in gated[tid]}
                for di in range(n_dets):
                    recorder.candidate(
                        tid, di, accepted.get(di), probability=track_betas[tid].get(di, 0.0)
                    )

        # 4. JPDA update
        for tid in track_ids:
            track = self._tracks[tid]
            betas = track_betas[tid]
            beta_i0 = betas[None]

            S = self._H @ track.P @ self._H.T + self._R
            S_inv = np.linalg.inv(S)
            K = track.P @ self._H.T @ S_inv

            # Combined innovation
            nu_combined = np.zeros(2)
            for di, nu, _d2, _log_lik in gated[tid]:
                beta_ij = betas.get(di, 0.0)
                nu_combined += beta_ij * nu

            # State update
            track.x = track.x + K @ nu_combined

            # Joseph-form conditional covariance preserves symmetry and
            # positive semidefiniteness under finite precision. The JPDA
            # mixture then adds missed-detection and innovation-spread terms.
            identity_minus_kh = np.eye(2) - K @ self._H
            P_kf = identity_minus_kh @ track.P @ identity_minus_kh.T + K @ self._R @ K.T
            P_tilde = np.zeros((2, 2))
            for di, nu, _d2, _log_lik in gated[tid]:
                beta_ij = betas.get(di, 0.0)
                P_tilde += beta_ij * np.outer(nu, nu)
            P_tilde -= np.outer(nu_combined, nu_combined)
            P_spread = K @ P_tilde @ K.T

            track.P = beta_i0 * track.P + (1.0 - beta_i0) * P_kf + P_spread
            # Symmetrize for numerical stability
            track.P = 0.5 * (track.P + track.P.T)

        # A hard, one-to-one assignment is used only for reporting and stored
        # detection attributes. State updates above remain probabilistic and
        # can use every gated measurement.
        det_to_track = _maximum_weight_one_to_one_assignments(
            track_betas,
            track_ids,
            n_dets,
        )
        for di, tid in det_to_track.items():
            self._tracks[tid].last_detection = dict(detections[di])

        # 5. Track management
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            beta_i0 = track_betas.get(tid, {}).get(None, 1.0)

            if beta_i0 > 0.5:
                track.misses += 1
            else:
                track.misses = 0
                track.hits += 1

            track.hit_history.append(int(beta_i0 <= 0.5))
            track.confirmed = _confirmation_state(track.confirmed, track.hit_history, cfg)

            if track.misses > cfg.max_missed_frames:
                del self._tracks[tid]

        # Retirement must release its reporting assignment before birth handling.
        det_to_track = {di: tid for di, tid in det_to_track.items() if tid in self._tracks}
        # Ungated observations start immediately, preserving the existing contract.
        # In-gate surplus observations must persist and pass the configured
        # duplicate-suppression policy. R is uncertainty, not physical resolution.
        # This conservative birth heuristic is separate from joint association.
        next_candidates: list[tuple[np.ndarray, int]] = []
        consumed_candidates: set[int] = set()
        for di, det in enumerate(detections):
            if di in det_to_track:
                continue
            z = np.array([float(det["range_m"]), float(det["velocity_m_s"])])
            live_gated = any(di in track_betas[tid] for tid in track_ids if tid in self._tracks)
            if live_gated:
                claimed_residuals = [
                    z - np.array([detections[d]["range_m"], detections[d]["velocity_m_s"]])
                    for d in det_to_track
                ]
                if any(
                    float(residual @ np.linalg.solve(self._R, residual))
                    <= self._birth_separation_chi2
                    for residual in claimed_residuals
                ):
                    continue
                matches = []
                for ci, (prior, count) in enumerate(self._birth_candidates):
                    predicted = self._F @ prior if self._F is not None else prior
                    residual = np.abs(z - predicted)
                    if (
                        ci not in consumed_candidates
                        and residual[0] <= cfg.range_gate_m
                        and residual[1] <= cfg.velocity_gate_m_s
                    ):
                        cost = _gate_normalized_cost(
                            (residual[0], cfg.range_gate_m),
                            (residual[1], cfg.velocity_gate_m_s),
                        )
                        matches.append((cost, ci, count))
                count = 1
                if matches:
                    _, ci, old_count = min(matches)
                    consumed_candidates.add(ci)
                    count = old_count + 1
                if count < self._birth_confirmation_frames:
                    next_candidates.append((z, count))
                    continue
            tid = self._next_track_id
            self._next_track_id += 1
            x0 = np.array([float(det["range_m"]), float(det["velocity_m_s"])])
            self._tracks[tid] = _JPDATrackState(
                track_id=tid,
                hit_history=deque([1], maxlen=cfg.n),
                x=x0,
                P=self._R.copy() * 4.0,
                confirmed=(cfg.m <= 1),
                last_detection=dict(det),
            )
            track_betas[tid] = {di: 1.0, None: 0.0}
            gated[tid] = [(di, np.zeros(2), 0.0, 0.0)]
            det_to_track[di] = tid

        self._birth_candidates = next_candidates

        # 7. Build output from the one-to-one reporting assignment.
        filtered: list[dict[str, Any]] = []
        for di, det in enumerate(detections):
            if di not in det_to_track:
                continue
            tid = det_to_track[di]
            track = self._tracks.get(tid)
            if track is None:
                continue
            out = dict(det)
            out["track_id"] = int(tid)
            out["track_confirmed"] = bool(track.confirmed)
            out["track_range_m"] = float(track.x[0])
            out["track_velocity_m_s"] = float(track.x[1])
            min_life = cfg.min_track_lifetime if cfg.min_track_lifetime > 0 else cfg.m
            if (track.confirmed and track.hits >= min_life) or cfg.output_tentative:
                filtered.append(out)

        return filtered, {
            **(
                {
                    "decisions": recorder.finish(
                        self._tracks,
                        det_to_track,
                        filtered,
                        missed_probabilities={
                            tid: values.get(None, 1.0) for tid, values in track_betas.items()
                        },
                    )
                }
                if recorder is not None
                else {}
            ),
            "enabled": True,
            "tracker_type": "jpdaf",
            "range_rate_scale": cfg.range_rate_scale,
            "range_rate_convention": "range_dot=-range_rate_scale*velocity_m_s",
            "acceleration_psd_coordinate": "reported_velocity_derivative",
            "association_method": "exact_joint_cluster_enumeration",
            "joint_search_nodes": hypothesis_nodes,
            "max_joint_nodes": self._max_joint_nodes,
            "pending_birth_candidates": len(self._birth_candidates),
            "birth_confirmation_frames": self._birth_confirmation_frames,
            "birth_separation_chi2": self._birth_separation_chi2,
            "confirmation_mode": cfg.confirmation_mode,
            "m": cfg.m,
            "n": cfg.n,
            "max_missed_frames": cfg.max_missed_frames,
            "gate_threshold": self._gate_threshold,
            "association_range_gate_m": cfg.range_gate_m,
            "association_velocity_gate_m_s": cfg.velocity_gate_m_s,
            "association_azimuth_gate_deg": cfg.azimuth_gate_deg,
            "association_azimuth_cost_scale_deg": cfg.azimuth_cost_scale_deg,
            "acceleration_psd_m2_s3": self._acceleration_psd_m2_s3,
            "pd": self._pd,
            "beta_fa": self._beta_fa,
            "raw": len(detections),
            "kept": len(filtered),
            "active_tracks": len(self._tracks),
            "confirmed_tracks": sum(1 for t in self._tracks.values() if t.confirmed),
            "output_tentative": cfg.output_tentative,
            "update_time_s": timestamp_s,
            "prediction_dt_s": dt_s,
        }


# ---------------------------------------------------------------------------
# Track-correlation ghost pruning (batch, post-tracking)
# ---------------------------------------------------------------------------


def _identify_contained_ghost_tracks(
    tracks: dict[int, dict[str, list]],
    *,
    range_ratio_threshold: float = 1.3,
    containment_ratio: float = 0.8,
    velocity_gate_m_s: float = 5.0,
) -> set[int]:
    """Identify candidate duplicate tracks using temporal containment.

    Some multipath duplicates have lifespans contained within a closer track.
    This heuristic flags Track B when most of its observed frame IDs also
    belong to Track A, B is at significantly longer range, and both have similar mean
    velocities. Those conditions do not prove path provenance and can also
    hold for distinct physical targets.

    Args:
        tracks: Per-track histories with ``frames`` and ``ranges`` lists.
        range_ratio_threshold: B's mean range must exceed A's by this factor.
        containment_ratio: Fraction of B's frame IDs that must also occur in
            A's observations to count as contained.
        velocity_gate_m_s: Only compare tracks whose mean velocities are
            within this gate.

    Returns:
        Set of track IDs flagged as candidates by containment analysis.
    """
    ghost_ids: set[int] = set()
    track_ids = sorted(tracks.keys())

    for i, tid_a in enumerate(track_ids):
        ta = tracks[tid_a]
        fa_set = set(ta["frames"])
        mean_ra = float(np.mean(ta["ranges"]))
        mean_va = float(np.mean(ta["velocities"]))

        for j, tid_b in enumerate(track_ids):
            if i == j or tid_b in ghost_ids:
                continue
            tb = tracks[tid_b]
            mean_rb = float(np.mean(tb["ranges"]))
            mean_vb = float(np.mean(tb["velocities"]))

            # Candidate-pair assumption: mean velocities are similar.
            if abs(mean_va - mean_vb) > velocity_gate_m_s:
                continue

            # Only consider B if it is at significantly longer range.
            if mean_rb <= mean_ra * range_ratio_threshold:
                continue

            # Require co-observed frame IDs, not merely times within A's span.
            fb_in_fa = sum(1 for f in tb["frames"] if f in fa_set)
            if len(tb["frames"]) > 0 and fb_in_fa >= containment_ratio * len(tb["frames"]):
                ghost_ids.add(tid_b)

    return ghost_ids


def prune_correlated_ghost_tracks(
    all_results: list[dict[str, Any]],
    *,
    correlation_threshold: float = 0.85,
    min_overlap_frames: int = 5,
    velocity_gate_m_s: float = 5.0,
    enable_coexistence: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Heuristically prune farther tracks with correlated histories.

    Two pruning passes:

    1. **Correlation pass**: For each pair of tracks with similar mean
       velocity, compute Pearson rho on range AND velocity histories.
       If either exceeds the threshold, prune the farther-range track.
    2. **Co-existence pass** (optional): If most of Track B's frame IDs are
       also observed by Track A and B is at significantly longer range,
       B is flagged as a possible multipath duplicate of A.

    A multipath duplicate can maintain a correlated history and a range offset
    from its source target. The algorithm therefore prunes the farther track
    among correlated candidates. It does not prove that the nearer track is a
    direct return or that the farther track is not a distinct physical target.

    Args:
        all_results: List of per-frame result dicts with ``detections``.
        correlation_threshold: Pearson rho above which two tracks are
            considered correlated (default 0.85).
        min_overlap_frames: Minimum overlapping frames required to compute
            correlation (default 5).
        velocity_gate_m_s: Only compare tracks whose mean velocities are
            within this gate (default 5.0).
        enable_coexistence: Run co-existence analysis as a second pruning
            pass (default True).

    Returns:
        Modified results with candidate tracks removed, plus metadata.
    """
    # 1. Build per-track range/velocity histories
    tracks: dict[int, dict[str, list]] = {}
    for result in all_results:
        frame_idx = int(result.get("_frame_index", 0))
        for det in result.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            if tid not in tracks:
                tracks[tid] = {"frames": [], "ranges": [], "velocities": []}
            tracks[tid]["frames"].append(frame_idx)
            tracks[tid]["ranges"].append(float(det.get("track_range_m", det.get("range_m", 0))))
            tracks[tid]["velocities"].append(
                float(det.get("track_velocity_m_s", det.get("velocity_m_s", 0)))
            )

    if len(tracks) < 2:
        return all_results, {"enabled": True, "ghost_tracks": [], "pruned": 0}

    # 2. Find correlated pairs (range AND velocity correlation)
    ghost_track_ids: set[int] = set()
    track_ids = sorted(tracks.keys())

    for i in range(len(track_ids)):
        tid_a = track_ids[i]
        if tid_a in ghost_track_ids:
            continue
        ta = tracks[tid_a]
        ra = np.array(ta["ranges"])
        fa = np.array(ta["frames"])
        va = np.array(ta["velocities"])
        mean_va = float(np.mean(va))

        for j in range(i + 1, len(track_ids)):
            tid_b = track_ids[j]
            if tid_b in ghost_track_ids:
                continue
            tb = tracks[tid_b]
            mean_vb = float(np.mean(tb["velocities"]))

            # Velocity proximity check
            if abs(mean_va - mean_vb) > velocity_gate_m_s:
                continue

            rb = np.array(tb["ranges"])
            fb = np.array(tb["frames"])
            vb = np.array(tb["velocities"])

            # Find overlapping frames
            common_frames = np.intersect1d(fa, fb)
            if len(common_frames) < min_overlap_frames:
                continue

            # Extract values at common frames
            fa_idx = np.searchsorted(fa, common_frames)
            fb_idx = np.searchsorted(fb, common_frames)
            valid = (fa_idx < len(ra)) & (fb_idx < len(rb))
            if np.sum(valid) < min_overlap_frames:
                continue
            ra_common = ra[fa_idx[valid]]
            rb_common = rb[fb_idx[valid]]

            # Range Pearson correlation
            range_rho = 0.0
            if np.std(ra_common) > 1e-6 and np.std(rb_common) > 1e-6:
                range_rho = float(np.corrcoef(ra_common, rb_common)[0, 1])

            # Velocity Pearson correlation
            va_common = va[fa_idx[valid]]
            vb_common = vb[fb_idx[valid]]
            vel_rho = 0.0
            if np.std(va_common) > 1e-6 and np.std(vb_common) > 1e-6:
                vel_rho = float(np.corrcoef(va_common, vb_common)[0, 1])

            if range_rho > correlation_threshold or vel_rho > correlation_threshold:
                # Heuristically prune the farther-range candidate.
                mean_ra = float(np.mean(ra))
                mean_rb = float(np.mean(rb))
                ghost_tid = tid_b if mean_rb > mean_ra else tid_a
                ghost_track_ids.add(ghost_tid)

    # 3. Co-existence analysis: flag candidates that correlation missed.
    coexistence_ghosts: set[int] = set()
    if enable_coexistence:
        # Only check tracks not already pruned
        remaining = {tid: t for tid, t in tracks.items() if tid not in ghost_track_ids}
        if len(remaining) >= 2:
            coexistence_ghosts = _identify_contained_ghost_tracks(
                remaining, velocity_gate_m_s=velocity_gate_m_s
            )
            ghost_track_ids |= coexistence_ghosts

    # 4. Remove ghost track detections from results
    pruned_count = 0
    if ghost_track_ids:
        for result in all_results:
            original = result.get("detections", [])
            filtered = [d for d in original if d.get("track_id") not in ghost_track_ids]
            pruned_count += len(original) - len(filtered)
            result["detections"] = filtered

    return all_results, {
        "enabled": True,
        "correlation_threshold": correlation_threshold,
        "velocity_gate_m_s": velocity_gate_m_s,
        "min_overlap_frames": min_overlap_frames,
        "ghost_tracks": sorted(ghost_track_ids),
        "coexistence_ghosts": sorted(coexistence_ghosts),
        "pruned": pruned_count,
    }
