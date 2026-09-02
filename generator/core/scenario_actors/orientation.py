"""Pure quaternion orientation evaluation for prepared actors."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import SupportsFloat, SupportsIndex

import numpy as np

from ._adapters import discriminator, numeric_pair, sequence, value, vec3
from .errors import PosePreparationError
from .quaternion import Quaternion
from .types import PreparedMobility, PreparedOrientation, Timeline

_VECTOR_EPSILON = 1e-12


def prepare_orientation(
    spec: object | None,
    timeline: Timeline,
    mobility: PreparedMobility,
    *,
    references: Mapping[str, PreparedMobility] | None = None,
    path: str = "orientation",
) -> PreparedOrientation:
    """Evaluate one orientation spec against already prepared actor positions."""

    orientation_type = "fixed" if spec is None else discriminator(spec)
    try:
        if orientation_type == "fixed":
            quaternions = _fixed(spec, timeline)
        elif orientation_type == "keyframes":
            quaternions = _keyframes(spec, timeline, path=path)
        elif orientation_type == "align_motion":
            quaternions = _align_motion(spec, timeline, mobility, path=path)
        elif orientation_type == "look_at":
            quaternions = _look_at(
                spec,
                timeline,
                mobility,
                references=references or {},
                path=path,
            )
        elif orientation_type == "spin":
            quaternions = _spin(spec, timeline, path=path)
        elif orientation_type == "random":
            quaternions = _random(spec, timeline, path=path)
        else:
            raise PosePreparationError(
                "unsupported_orientation",
                f"{path}.type",
                f"unsupported orientation type {orientation_type!r}",
            )
    except PosePreparationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PosePreparationError("invalid_orientation", path, str(exc)) from exc
    if len(quaternions) != timeline.steps:
        raise PosePreparationError(
            "sample_count_mismatch",
            path,
            f"evaluator returned {len(quaternions)} samples for {timeline.steps} timeline steps",
        )
    return PreparedOrientation(tuple(quaternions))


def apply_asset_alignment(
    orientation: PreparedOrientation,
    alignment: Quaternion | object,
    *,
    path: str = "asset.alignment",
) -> PreparedOrientation:
    """Compose target asset-front alignment exactly once at the asset boundary.

    Sequence inputs are interpreted as yaw/pitch/roll degrees. Alignment is a
    local rotation and is therefore right-multiplied onto authored orientation.
    """

    if orientation.asset_alignment_applied:
        raise PosePreparationError(
            "asset_alignment_already_applied",
            path,
            "asset-front alignment may only be composed once",
        )
    if isinstance(alignment, Quaternion):
        alignment_quaternion = alignment
    else:
        yaw, pitch, roll = vec3(alignment, name=path)
        alignment_quaternion = Quaternion.from_euler_deg(yaw, pitch, roll)
    return PreparedOrientation(
        tuple(quaternion * alignment_quaternion for quaternion in orientation.quaternions),
        asset_alignment_applied=True,
    )


def _fixed(spec: object | None, timeline: Timeline) -> list[Quaternion]:
    yaw = float(value(spec, "yaw_deg", default=0.0)) if spec is not None else 0.0
    pitch = float(value(spec, "pitch_deg", default=0.0)) if spec is not None else 0.0
    roll = float(value(spec, "roll_deg", default=0.0)) if spec is not None else 0.0
    quaternion = Quaternion.from_euler_deg(yaw, pitch, roll)
    return [quaternion] * timeline.steps


def _keyframes(spec: object, timeline: Timeline, *, path: str) -> list[Quaternion]:
    raw_keyframes = sequence(value(spec, "keyframes"), name=f"{path}.keyframes")
    if len(raw_keyframes) < 2:
        raise ValueError(f"{path}.keyframes must contain at least two entries")
    times: list[float] = []
    rotations: list[Quaternion] = []
    for index, keyframe in enumerate(raw_keyframes):
        time_s = float(value(keyframe, "time_s"))
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError(f"{path}.keyframes[{index}].time_s is invalid")
        if times and time_s <= times[-1]:
            raise ValueError(f"{path}.keyframes times must be strictly increasing")
        times.append(time_s)
        rotations.append(
            Quaternion.from_euler_deg(
                float(value(keyframe, "yaw_deg", default=0.0)),
                float(value(keyframe, "pitch_deg", default=0.0)),
                float(value(keyframe, "roll_deg", default=0.0)),
            )
        )
    result: list[Quaternion] = []
    for timestamp in timeline.timestamps_s:
        if timestamp <= times[0]:
            result.append(rotations[0])
            continue
        if timestamp >= times[-1]:
            result.append(rotations[-1])
            continue
        upper = int(np.searchsorted(times, timestamp, side="right"))
        lower = upper - 1
        fraction = (timestamp - times[lower]) / (times[upper] - times[lower])
        result.append(rotations[lower].slerp(rotations[upper], fraction))
    return result


def _align_motion(
    spec: object,
    timeline: Timeline,
    mobility: PreparedMobility,
    *,
    path: str,
) -> list[Quaternion]:
    if not mobility.has_physical_velocity:
        raise PosePreparationError(
            "orientation_requires_physical_path",
            path,
            "align_motion requires mobility with physical velocity",
        )
    allow_pitch = bool(value(spec, "allow_pitch", default=True))
    yaw_offset = float(value(spec, "yaw_offset_deg", default=0.0))
    pitch_offset = float(value(spec, "pitch_offset_deg", default=0.0))
    roll_offset = float(value(spec, "roll_offset_deg", default=0.0))
    smoothing_time = float(value(spec, "smoothing_time_s", default=0.0))
    max_yaw_rate = _optional_positive_float(
        value(spec, "max_yaw_rate_deg_s", default=None),
        f"{path}.max_yaw_rate_deg_s",
    )
    max_pitch_rate = _optional_positive_float(
        value(spec, "max_pitch_rate_deg_s", default=None),
        f"{path}.max_pitch_rate_deg_s",
    )
    if not math.isfinite(smoothing_time) or smoothing_time < 0.0:
        raise ValueError(f"{path}.smoothing_time_s must be finite and non-negative")

    raw: list[tuple[float, float, float]] = []
    for forward in mobility.forward_vectors:
        direction_yaw, direction_pitch = _yaw_pitch_for_forward_axis(forward)
        yaw = direction_yaw + yaw_offset
        pitch = direction_pitch + pitch_offset if allow_pitch else pitch_offset
        raw.append((yaw, pitch, roll_offset))
    return _smooth_and_limit(raw, timeline, smoothing_time, max_yaw_rate, max_pitch_rate)


def _look_at(
    spec: object,
    timeline: Timeline,
    mobility: PreparedMobility,
    *,
    references: Mapping[str, PreparedMobility],
    path: str,
) -> list[Quaternion]:
    actor_name = value(spec, "actor", default=None)
    point = value(spec, "point_m", default=None)
    if (actor_name is None) == (point is None):
        raise ValueError(f"{path} must define exactly one of actor or point_m")
    if actor_name is not None:
        actor_name = str(actor_name)
        target = references.get(actor_name)
        if target is None:
            raise PosePreparationError(
                "missing_actor_reference",
                f"{path}.actor",
                f"actor {actor_name!r} does not exist",
            )
        target_positions = target.positions_m
    else:
        target_point = vec3(point, name=f"{path}.point_m")
        target_positions = (target_point,) * timeline.steps

    yaw_offset = float(value(spec, "yaw_offset_deg", default=0.0))
    pitch_offset = float(value(spec, "pitch_offset_deg", default=0.0))
    roll_offset = float(value(spec, "roll_offset_deg", default=0.0))
    allow_pitch = bool(value(spec, "allow_pitch", default=True))
    smoothing_time = float(value(spec, "smoothing_time_s", default=0.0))
    max_yaw_rate = _optional_positive_float(
        value(spec, "max_yaw_rate_deg_s", default=None),
        f"{path}.max_yaw_rate_deg_s",
    )
    max_pitch_rate = _optional_positive_float(
        value(spec, "max_pitch_rate_deg_s", default=None),
        f"{path}.max_pitch_rate_deg_s",
    )
    if not math.isfinite(smoothing_time) or smoothing_time < 0.0:
        raise ValueError(f"{path}.smoothing_time_s must be finite and non-negative")
    yaw_limits_raw = value(spec, "yaw_limits_deg", default=None)
    pitch_limits_raw = value(spec, "pitch_limits_deg", default=None)
    yaw_limits = (
        numeric_pair(yaw_limits_raw, name=f"{path}.yaw_limits_deg")
        if yaw_limits_raw is not None
        else None
    )
    pitch_limits = (
        numeric_pair(pitch_limits_raw, name=f"{path}.pitch_limits_deg")
        if pitch_limits_raw is not None
        else None
    )
    _validate_limits(yaw_limits, f"{path}.yaw_limits_deg")
    _validate_limits(pitch_limits, f"{path}.pitch_limits_deg")

    raw_angles: list[tuple[float, float, float]] = []
    previous: tuple[float, float, float] | None = None
    for owner_position, target_position in zip(mobility.positions_m, target_positions):
        delta = np.asarray(target_position, dtype=np.float64) - np.asarray(
            owner_position, dtype=np.float64
        )
        norm = float(np.linalg.norm(delta))
        if norm <= _VECTOR_EPSILON:
            if previous is not None:
                angles = previous
            else:
                yaw = yaw_offset
                pitch = pitch_offset
                if yaw_limits is not None:
                    yaw = min(max(yaw, yaw_limits[0]), yaw_limits[1])
                if pitch_limits is not None:
                    pitch = min(max(pitch, pitch_limits[0]), pitch_limits[1])
                angles = (yaw, pitch, roll_offset)
        else:
            relative_yaw, relative_pitch = _yaw_pitch_for_forward_axis(delta)
            if not allow_pitch:
                relative_pitch = 0.0
            yaw = relative_yaw + yaw_offset
            pitch = relative_pitch + pitch_offset
            if yaw_limits is not None:
                yaw = min(max(yaw, yaw_limits[0]), yaw_limits[1])
            if pitch_limits is not None:
                pitch = min(max(pitch, pitch_limits[0]), pitch_limits[1])
            angles = (yaw, pitch, roll_offset)
        previous = angles
        raw_angles.append(angles)
    return _smooth_and_limit(
        raw_angles,
        timeline,
        smoothing_time,
        max_yaw_rate,
        max_pitch_rate,
    )


def _spin(spec: object, timeline: Timeline, *, path: str) -> list[Quaternion]:
    axis_raw = value(spec, "axis")
    axis = getattr(axis_raw, "value", axis_raw)
    axis = str(axis).lower()
    if axis not in ("yaw", "pitch", "roll"):
        raise ValueError(f"{path}.axis must be yaw, pitch, or roll")
    rate = float(value(spec, "rate_deg_s"))
    if not math.isfinite(rate):
        raise ValueError(f"{path}.rate_deg_s must be finite")
    initial = {
        "yaw": float(value(spec, "yaw_deg", default=0.0)),
        "pitch": float(value(spec, "pitch_deg", default=0.0)),
        "roll": float(value(spec, "roll_deg", default=0.0)),
    }
    result = []
    for timestamp in timeline.timestamps_s:
        angles = dict(initial)
        angles[axis] += rate * timestamp
        result.append(Quaternion.from_euler_deg(angles["yaw"], angles["pitch"], angles["roll"]))
    return result


def _random(spec: object, timeline: Timeline, *, path: str) -> list[Quaternion]:
    seed = int(value(spec, "seed"))
    yaw_range = numeric_pair(value(spec, "yaw_range_deg"), name=f"{path}.yaw_range_deg")
    pitch_range = numeric_pair(value(spec, "pitch_range_deg"), name=f"{path}.pitch_range_deg")
    roll_range = numeric_pair(value(spec, "roll_range_deg"), name=f"{path}.roll_range_deg")
    for bounds, name in (
        (yaw_range, "yaw_range_deg"),
        (pitch_range, "pitch_range_deg"),
        (roll_range, "roll_range_deg"),
    ):
        _validate_limits(bounds, f"{path}.{name}")
    update_interval_raw = value(spec, "update_interval_s", default=None)
    update_interval = float(update_interval_raw) if update_interval_raw is not None else None
    if update_interval is not None and (
        not math.isfinite(update_interval) or update_interval <= 0.0
    ):
        raise ValueError(f"{path}.update_interval_s must be finite and positive")
    result = []
    for timestamp in timeline.timestamps_s:
        sample_time = (
            math.floor((timestamp + _VECTOR_EPSILON) / update_interval) * update_interval
            if update_interval is not None
            else timestamp
        )
        samples = _keyed_uniforms(seed, sample_time, 3)
        angles = tuple(
            low + sample * (high - low)
            for sample, (low, high) in zip(samples, (yaw_range, pitch_range, roll_range))
        )
        result.append(Quaternion.from_euler_deg(*angles))
    return result


def _smooth_and_limit(
    raw_angles: list[tuple[float, float, float]],
    timeline: Timeline,
    smoothing_time_s: float,
    max_yaw_rate_deg_s: float | None,
    max_pitch_rate_deg_s: float | None,
) -> list[Quaternion]:
    if not raw_angles:
        return []
    result = [Quaternion.from_euler_deg(*raw_angles[0])]
    previous_yaw, previous_pitch, previous_roll = raw_angles[0]
    for index in range(1, len(raw_angles)):
        raw_yaw, raw_pitch, raw_roll = raw_angles[index]
        delta_time = timeline.timestamps_s[index] - timeline.timestamps_s[index - 1]
        alpha = 1.0 if smoothing_time_s <= 0.0 else 1.0 - math.exp(-delta_time / smoothing_time_s)
        yaw_delta = _shortest_angle_delta(previous_yaw, raw_yaw) * alpha
        pitch_delta = (raw_pitch - previous_pitch) * alpha
        if max_yaw_rate_deg_s is not None:
            limit = max_yaw_rate_deg_s * delta_time
            yaw_delta = min(max(yaw_delta, -limit), limit)
        if max_pitch_rate_deg_s is not None:
            limit = max_pitch_rate_deg_s * delta_time
            pitch_delta = min(max(pitch_delta, -limit), limit)
        previous_yaw = previous_yaw + yaw_delta
        previous_pitch = previous_pitch + pitch_delta
        previous_roll = raw_roll
        result.append(Quaternion.from_euler_deg(previous_yaw, previous_pitch, previous_roll))
    return result


def _shortest_angle_delta(start_deg: float, end_deg: float) -> float:
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def _yaw_pitch_for_forward_axis(direction: object) -> tuple[float, float]:
    """Return right-handed Z/Y/X angles that align local +X with ``direction``."""

    components = np.asarray(direction, dtype=np.float64).reshape(-1)
    if components.size != 3 or not np.all(np.isfinite(components)):
        raise ValueError("direction must contain exactly three finite coordinates")
    x, y, z = (float(component) for component in components)
    horizontal = math.hypot(x, y)
    yaw = math.degrees(math.atan2(y, x))
    pitch = -math.degrees(math.atan2(z, horizontal))
    if pitch == 0.0:
        pitch = 0.0
    return yaw, pitch


def _optional_positive_float(raw: object, name: str) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, (str, bytes, SupportsFloat, SupportsIndex)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(raw)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _validate_limits(bounds: tuple[float, float] | None, name: str) -> None:
    if bounds is not None and bounds[0] > bounds[1]:
        raise ValueError(f"{name} minimum must not exceed maximum")


def _keyed_uniforms(seed: int, timestamp: float, count: int) -> tuple[float, ...]:
    key = f"random_orientation:{int(seed)}:{float(timestamp).hex()}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    return tuple(
        int.from_bytes(digest[index * 8 : index * 8 + 8], "big") / float(2**64)
        for index in range(count)
    )
