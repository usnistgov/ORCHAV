"""Truth-independent measurement fusion for synchronized monostatic sensors.

The reference processor tracks world position/velocity with a constant-velocity
extended Kalman filter (EKF). Each stationary sensor reports range, receiver-local
azimuth/elevation and positive-closing radial velocity. Sensor updates are sequential
in stable ID order, with at most one report per sensor per track per epoch. This is
detection-level fusion, not coherent channel combining or track-to-track fusion.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

COMPONENTS = ("range_m", "azimuth_rad", "elevation_rad", "closing_velocity_m_s")


def validated_covariance(value, dimension):
    """Require a finite symmetric positive-definite covariance in component order."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("Measurement covariance has invalid shape or nonfinite entries")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("Measurement covariance must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Measurement covariance must be positive definite") from exc
    return matrix


def rotation_matrix(orientation_rad) -> np.ndarray:
    """World-from-array rotation, yaw/pitch/roll in radians (Rz Ry Rx)."""
    yaw, pitch, roll = orientation_rad
    cy, sy, cp, sp, cr, sr = (
        np.cos(yaw),
        np.sin(yaw),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(roll),
        np.sin(roll),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass(frozen=True)
class Sensor:
    """Known stationary monostatic sensor pose in the common world frame."""

    sensor_id: str
    position_m: tuple[float, float, float]
    orientation_rad: tuple[float, float, float]
    measurement_covariance: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self):
        if not self.sensor_id or any(
            np.shape(v) != (3,) or not np.all(np.isfinite(v))
            for v in (self.position_m, self.orientation_rad)
        ):
            raise ValueError("Sensor needs an ID and finite 3-D position/orientation")

        if self.measurement_covariance is not None:
            validated_covariance(self.measurement_covariance, 4)

    @property
    def rotation(self):
        return rotation_matrix(self.orientation_rad)


@dataclass(frozen=True)
class Observation:
    """Native detection with explicit usable components; absent velocity is never zero.

    Covariance uses ``components`` order and SI units (angles in radians).
    Quality reasons describe a policy, not calibrated detection probabilities.
    """

    sensor_id: str
    detection_id: str
    time_s: float
    range_m: float | None
    azimuth_rad: float | None
    elevation_rad: float | None
    closing_velocity_m_s: float | None
    components: tuple[str, ...] = COMPONENTS
    covariance: tuple[tuple[float, ...], ...] | None = None
    quality_reasons: tuple[str, ...] = ("quality_unknown_declared_fallback",)
    native_source: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.sensor_id or not self.detection_id or not np.isfinite(self.time_s):
            raise ValueError("Detection needs IDs and finite epoch")
        if len(set(self.components)) != len(self.components) or any(
            c not in COMPONENTS for c in self.components
        ):
            raise ValueError("Detection needs distinct known usable components")
        for component in COMPONENTS:
            value = getattr(self, component)
            if value is None and component not in self.components:
                continue
            if value is None or not np.isfinite(value):
                raise ValueError("Detection components must have finite values")
        if self.range_m is not None and self.range_m <= 0:
            raise ValueError("Detection needs positive range")
        if self.azimuth_rad is not None and abs(self.azimuth_rad) > np.pi:
            raise ValueError("Invalid azimuth")
        if self.elevation_rad is not None and abs(self.elevation_rad) >= np.pi / 2:
            raise ValueError("Invalid elevation")
        if self.covariance is not None:
            validated_covariance(self.covariance, len(self.components))

    @property
    def indices(self):
        return [COMPONENTS.index(c) for c in self.components]

    @property
    def vector(self):
        return np.array([getattr(self, c) for c in self.components])

    @property
    def has_position(self):
        return set(COMPONENTS[:3]).issubset(self.components)

    def world_position(self, sensor: Sensor):
        if not self.has_position:
            raise ValueError("World position requires range, azimuth and elevation")
        a, e = self.azimuth_rad, self.elevation_rad
        direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return np.asarray(sensor.position_m) + sensor.rotation @ (self.range_m * direction)


@dataclass(frozen=True)
class FusionConfig:
    """Explicit model uncertainties and gates; not calibrated confidence bounds."""

    range_std_m: float = 0.2
    azimuth_std_deg: float = 2.0
    elevation_std_deg: float = 3.0
    closing_velocity_std_m_s: float = 0.10
    acceleration_psd_m2_s3: float = 0.1
    position_gate_m: float = 1.0
    innovation_gate: float = 25.0
    initial_velocity_std_m_s: float = 5.0
    birth_exclusion_m: float = 0.6
    birth_velocity_gate_m_s: float = 0.2
    confirmation_epochs: int = 3
    max_missed_epochs: int = 3
    confirmation_mode: str = "rolling"
    confirmation_window: int = 5
    minimum_recent_sensors: int = 1
    gate_mode: str = "probability"
    quality_policy: str = "dc_boundary"
    dc_boundary_width_bins: int = 1
    birth_policy: str = "measurement_consistency"

    def __post_init__(self):
        scalars = [
            self.range_std_m,
            self.azimuth_std_deg,
            self.elevation_std_deg,
            self.closing_velocity_std_m_s,
            self.acceleration_psd_m2_s3,
            self.position_gate_m,
            self.innovation_gate,
            self.initial_velocity_std_m_s,
            self.birth_exclusion_m,
            self.birth_velocity_gate_m_s,
        ]
        if not np.all(np.isfinite(scalars)) or min(scalars) <= 0:
            raise ValueError("Fusion uncertainties and gates must be positive and finite")
        if (
            any(type(v) is not int for v in (self.confirmation_epochs, self.max_missed_epochs))
            or self.confirmation_epochs < 1
            or self.max_missed_epochs < 0
        ):
            raise ValueError(
                "Confirmation/miss counts must be nonnegative integers (confirmation >= 1)"
            )

        if self.confirmation_mode not in {"rolling", "legacy_cumulative"}:
            raise ValueError("Unknown confirmation mode")
        if (
            type(self.confirmation_window) is not int
            or self.confirmation_window < 1
            or (
                self.confirmation_mode == "rolling"
                and self.confirmation_window < self.confirmation_epochs
            )
        ):
            raise ValueError("Rolling confirmation window must cover required hits")
        if type(self.minimum_recent_sensors) is not int or self.minimum_recent_sensors < 1:
            raise ValueError("minimum_recent_sensors must be a positive integer")
        if self.gate_mode not in {"probability", "legacy_numeric"}:
            raise ValueError("Unknown gate mode")
        if self.quality_policy not in {"dc_boundary", "legacy"}:
            raise ValueError("Unknown quality policy")
        if type(self.dc_boundary_width_bins) is not int or self.dc_boundary_width_bins < 0:
            raise ValueError("DC boundary width must be a nonnegative integer")
        if self.birth_policy not in {"measurement_consistency", "legacy_distance_velocity"}:
            raise ValueError("Unknown birth policy")

    def gate(self, dimensions):
        """Preserve the full gate's chi-square probability for reduced reports."""
        if self.gate_mode == "legacy_numeric" or dimensions == 4:
            return self.innovation_gate
        return float(chi2.isf(chi2.sf(self.innovation_gate, 4), dimensions))

    @property
    def measurement_covariance(self):
        return np.diag(
            np.square(
                [
                    self.range_std_m,
                    np.radians(self.azimuth_std_deg),
                    np.radians(self.elevation_std_deg),
                    self.closing_velocity_std_m_s,
                ]
            )
        )


class GeometryRejection(ValueError):
    """Expected singular geometry, distinct from malformed input or numerical failure."""


def measurement_model(
    state: np.ndarray, sensor: Sensor, components=COMPONENTS
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted [range, azimuth, elevation, closing speed] and analytic Jacobian.

    State order is [world x,y,z,vx,vy,vz]. For a stationary monostatic sensor,
    closing speed is -u dot v, where u points from sensor to target.
    """
    delta = state[:3] - sensor.position_m
    distance = np.linalg.norm(delta)
    local = sensor.rotation.T @ delta
    x, y, z = local
    horizontal = np.hypot(x, y)
    if distance < 1e-6 or (horizontal < 1e-6 and any(c in components for c in COMPONENTS[1:3])):
        raise GeometryRejection("singular_sensor_or_polar_axis")
    u = delta / distance
    velocity = state[3:]
    predicted = np.array([distance, np.arctan2(y, x), np.arctan2(z, horizontal), -u @ velocity])
    jac = np.zeros((4, 6))
    jac[0, :3] = u
    if horizontal >= 1e-6:
        jac[1, :3] = np.array([-y, x, 0]) / horizontal**2 @ sensor.rotation.T
        jac[2, :3] = (
            np.array([-x * z / horizontal, -y * z / horizontal, horizontal])
            / distance**2
            @ sensor.rotation.T
        )
    jac[3, :3] = -(velocity - u * (u @ velocity)) / distance
    jac[3, 3:] = -u
    indices = [COMPONENTS.index(c) for c in components]
    return predicted[indices], jac[indices]


def gated_assignment(costs: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """Maximum valid cardinality then minimum cost, with explicit unmatched dummies."""
    rows, cols = costs.shape
    if not rows or not cols:
        return []
    unmatched = (min(rows, cols) + 1) * (gate + 1)
    augmented = np.full((rows, cols + rows), unmatched, dtype=float)
    augmented[:, :cols] = np.where(np.isfinite(costs) & (costs <= gate), costs, unmatched * 2)
    rr, cc = linear_sum_assignment(augmented)
    return [(int(r), int(c)) for r, c in zip(rr, cc) if c < cols and augmented[r, c] < unmatched]


@dataclass
class _Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    hit_epochs: int = 0
    missed_epochs: int = 0
    sources: list[dict] = field(default_factory=list)
    recent_hits: list[bool] = field(default_factory=list)
    recent_support: list[list[str]] = field(default_factory=list)
    confirmed: bool = False


class MonostaticFusion:
    """Process complete synchronized epochs, including empty detections.

    Input has no truth fields. Confirmation counts distinct epochs, not sensor
    updates. Confirmed tracks coast for max_missed_epochs and are explicitly
    marked predicted when no sensor updated them. Sensor order is deterministic;
    the baseline does not implement joint multi-hypothesis association.
    """

    def __init__(self, sensors: list[Sensor], config: FusionConfig | None = None):
        if not sensors or len({s.sensor_id for s in sensors}) != len(sensors):
            raise ValueError("Require distinct sensor IDs")
        self.sensors = {s.sensor_id: s for s in sensors}
        self.config = config or FusionConfig()
        self.tracks: list[_Track] = []
        self.time_s: float | None = None
        self.next_id = 1

    def _measurement_covariance(self, observation, sensor):
        if observation.covariance is not None:
            return validated_covariance(observation.covariance, len(observation.components))
        full = (
            self.config.measurement_covariance
            if sensor.measurement_covariance is None
            else validated_covariance(sensor.measurement_covariance, 4)
        )
        return full[np.ix_(observation.indices, observation.indices)]

    def _innovation(self, track, observation, sensor):
        predicted, jac = measurement_model(track.state, sensor, observation.components)
        residual = observation.vector - predicted
        if "azimuth_rad" in observation.components:
            i = observation.components.index("azimuth_rad")
            residual[i] = (residual[i] + np.pi) % (2 * np.pi) - np.pi
        covariance = jac @ track.covariance @ jac.T + self._measurement_covariance(
            observation, sensor
        )
        return residual, jac, covariance

    def _update(self, track, observation, sensor, *, birth=False):
        residual, jac, innovation_cov = self._innovation(track, observation, sensor)
        gain = np.linalg.solve(innovation_cov, jac @ track.covariance).T
        track.state += gain @ residual
        remainder = np.eye(6) - gain @ jac
        track.covariance = (
            remainder @ track.covariance @ remainder.T
            + gain @ self._measurement_covariance(observation, sensor) @ gain.T
        )
        track.covariance = (track.covariance + track.covariance.T) / 2
        track.sources.append(
            {
                "sensor_id": sensor.sensor_id,
                "detection_id": observation.detection_id,
                "birth": birth,
                "components": list(observation.components),
                "quality_reasons": list(observation.quality_reasons),
                "native_source": deepcopy(observation.native_source),
                "residual": residual.tolist(),
                "measurement_covariance": self._measurement_covariance(
                    observation, sensor
                ).tolist(),
                "innovation_covariance": innovation_cov.tolist(),
                "gate": self.config.gate(len(observation.components)),
                "innovation_squared": float(residual @ np.linalg.solve(innovation_cov, residual)),
            }
        )

    def step(
        self, time_s: float, observations: list[Observation], *, available_sensors=None
    ) -> dict:
        """Publish an epoch only after every update and diagnostic succeeds."""
        working = copy(self)
        # Previous-epoch sources are discarded by prediction. Copy only state
        # that can survive into this epoch, keeping publication transactional.
        working.tracks = [
            replace(
                t,
                state=t.state.copy(),
                covariance=t.covariance.copy(),
                sources=[],
                recent_hits=list(t.recent_hits),
                recent_support=[list(support) for support in t.recent_support],
            )
            for t in self.tracks
        ]
        result = working._step(time_s, observations, available_sensors=available_sensors)
        self.tracks, self.next_id, self.time_s = working.tracks, working.next_id, working.time_s
        return result

    def _step(
        self, time_s: float, observations: list[Observation], *, available_sensors=None
    ) -> dict:
        if not np.isfinite(time_s) or (self.time_s is not None and time_s <= self.time_s):
            raise ValueError("Epoch times must be finite and strictly increasing")
        available = set(self.sensors) if available_sensors is None else set(available_sensors)
        if not available.issubset(self.sensors) or any(
            o.sensor_id not in available for o in observations
        ):
            raise ValueError("Detections cannot belong to missing/unknown sources")
        identities = {(o.sensor_id, o.detection_id) for o in observations}
        if len(identities) != len(observations) or any(
            o.sensor_id not in self.sensors or abs(o.time_s - time_s) > 1e-9 for o in observations
        ):
            raise ValueError(
                "Detections must have unique IDs, known sensors and the exact epoch time"
            )
        dt = 0.0 if self.time_s is None else time_s - self.time_s
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        process_cov = self.config.acceleration_psd_m2_s3 * np.block(
            [
                [np.eye(3) * dt**3 / 3, np.eye(3) * dt**2 / 2],
                [np.eye(3) * dt**2 / 2, np.eye(3) * dt],
            ]
        )
        for track in self.tracks:
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + process_cov
            track.sources = []
        suppressed_births = []
        geometry_rejections = []
        decisions = []
        unsupported_births = [
            {
                "sensor_id": o.sensor_id,
                "detection_id": o.detection_id,
                "reason": "no_usable_components",
            }
            for o in observations
            if not o.components
        ]
        for sensor_id, sensor in sorted(self.sensors.items()):
            reports = sorted(
                (o for o in observations if o.sensor_id == sensor_id and o.components),
                key=lambda o: o.detection_id,
            )
            costs = np.full((len(self.tracks), len(reports)), np.inf)
            for i, track in enumerate(self.tracks):
                for j, observation in enumerate(reports):
                    if (
                        observation.has_position
                        and np.linalg.norm(track.state[:3] - observation.world_position(sensor))
                        > self.config.position_gate_m
                    ):
                        decisions.append(
                            {
                                "sensor_id": sensor_id,
                                "detection_id": observation.detection_id,
                                "track_id": track.track_id,
                                "outcome": "position_gate_rejection",
                            }
                        )
                        continue
                    try:
                        residual, _, covariance = self._innovation(track, observation, sensor)
                    except GeometryRejection as exc:
                        geometry_rejections.append(
                            {
                                "sensor_id": sensor_id,
                                "detection_id": observation.detection_id,
                                "track_id": track.track_id,
                                "reason": str(exc),
                            }
                        )
                        continue
                    squared = float(residual @ np.linalg.solve(covariance, residual))
                    gate = self.config.gate(len(observation.components))
                    costs[i, j] = squared / gate
                    decisions.append(
                        {
                            "sensor_id": sensor_id,
                            "detection_id": observation.detection_id,
                            "track_id": track.track_id,
                            "innovation_squared": squared,
                            "gate": gate,
                            "components": list(observation.components),
                            "outcome": (
                                "candidate" if squared <= gate else "innovation_gate_rejection"
                            ),
                        }
                    )
            matches = gated_assignment(costs, 1.0)
            matched_ids = {(self.tracks[i].track_id, reports[j].detection_id) for i, j in matches}
            for decision in decisions:
                if decision["sensor_id"] == sensor_id and decision["outcome"] == "candidate":
                    decision["outcome"] = (
                        "associated"
                        if (decision["track_id"], decision["detection_id"]) in matched_ids
                        else "assignment_not_selected"
                    )
            for i, j in matches:
                self._update(self.tracks[i], reports[j], sensor)
            used = {j for _, j in matches}
            for j, observation in enumerate(reports):
                if j in used:
                    continue
                if not observation.has_position:
                    unsupported_births.append(
                        {
                            "sensor_id": sensor_id,
                            "detection_id": observation.detection_id,
                            "reason": "insufficient_spatial_components",
                        }
                    )
                    continue
                # An extended object can supply neighbouring peaks. Do not
                # create another point track near one already measured during
                # this epoch with compatible radial motion. No GT is consulted.
                neighbours = []
                for track in self.tracks:
                    if not track.sources:
                        continue
                    distance = np.linalg.norm(track.state[:3] - observation.world_position(sensor))
                    if distance > self.config.birth_exclusion_m:
                        continue
                    try:
                        predicted, _ = measurement_model(
                            track.state, sensor, observation.components
                        )
                    except GeometryRejection as exc:
                        geometry_rejections.append(
                            {
                                "sensor_id": sensor_id,
                                "detection_id": observation.detection_id,
                                "track_id": track.track_id,
                                "reason": str(exc),
                            }
                        )
                        continue
                    radial_ok = True
                    if "closing_velocity_m_s" in observation.components:
                        vi = observation.components.index("closing_velocity_m_s")
                        radial_ok = (
                            abs(predicted[vi] - observation.closing_velocity_m_s)
                            <= self.config.birth_velocity_gate_m_s
                        )
                    consistent = True
                    if self.config.birth_policy == "measurement_consistency":
                        residual, _, covariance = self._innovation(track, observation, sensor)
                        consistent = residual @ np.linalg.solve(
                            covariance, residual
                        ) <= self.config.gate(len(observation.components))
                    if radial_ok and consistent:
                        neighbours.append((distance, track.track_id))
                if neighbours:
                    suppressed_births.append(
                        {
                            "sensor_id": sensor_id,
                            "detection_id": observation.detection_id,
                            "near_track_id": min(neighbours)[1],
                            "reason": self.config.birth_policy,
                            "components": list(observation.components),
                        }
                    )
                    continue
                # A report seeds the linearization point, then is assimilated
                # once from a diffuse position prior and broad zero-mean velocity.
                track = _Track(
                    self.next_id,
                    np.r_[observation.world_position(sensor), np.zeros(3)],
                    np.diag([1e6] * 3 + [self.config.initial_velocity_std_m_s**2] * 3),
                )
                try:
                    self._update(track, observation, sensor, birth=True)
                except GeometryRejection as exc:
                    # This expected geometry failure rejects this report only.
                    # Unexpected failures still escape the transactional step.
                    geometry_rejections.append(
                        {
                            "sensor_id": sensor_id,
                            "detection_id": observation.detection_id,
                            "track_id": None,
                            "stage": "birth",
                            "reason": str(exc),
                        }
                    )
                    continue
                self.next_id += 1
                self.tracks.append(track)
        for track in self.tracks:
            support = sorted({s["sensor_id"] for s in track.sources})
            track.recent_hits = (track.recent_hits + [bool(support)])[
                -self.config.confirmation_window :
            ]
            track.recent_support = (track.recent_support + [support])[
                -self.config.confirmation_window :
            ]
            if track.sources:
                track.hit_epochs += 1
                track.missed_epochs = 0
            else:
                track.missed_epochs += 1
            hits = (
                track.hit_epochs
                if self.config.confirmation_mode == "legacy_cumulative"
                else sum(track.recent_hits)
            )
            sensors = {s for support in track.recent_support for s in support}
            track.confirmed |= (
                hits >= self.config.confirmation_epochs
                and len(sensors) >= self.config.minimum_recent_sensors
            )
        retired = [
            t.track_id for t in self.tracks if t.missed_epochs > self.config.max_missed_epochs
        ]
        self.tracks = [t for t in self.tracks if t.missed_epochs <= self.config.max_missed_epochs]
        self.time_s = float(time_s)
        return {
            "time_s": float(time_s),
            "geometry_rejections": geometry_rejections,
            "decisions": decisions,
            "unsupported_births": unsupported_births,
            "source_status": {
                n: ("available" if n in available else "missing") for n in sorted(self.sensors)
            },
            "suppressed_births": suppressed_births,
            "retired_track_ids": retired,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "position_m": t.state[:3].tolist(),
                    "velocity_m_s": t.state[3:].tolist(),
                    "covariance": t.covariance.tolist(),
                    "confirmed": t.confirmed,
                    "recent_hits": list(t.recent_hits),
                    "current_sensor_support": sorted({s["sensor_id"] for s in t.sources}),
                    "recent_sensor_support": sorted(
                        {s for support in t.recent_support for s in support}
                    ),
                    "predicted_only": not bool(t.sources),
                    "hit_epochs": t.hit_epochs,
                    "missed_epochs": t.missed_epochs,
                    "sources": list(t.sources),
                }
                for t in self.tracks
            ],
        }
