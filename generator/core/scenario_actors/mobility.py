"""Pure mobility evaluation for actor and group specifications."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from ._adapters import discriminator, numeric_pair, sequence, value, vec3
from .errors import PosePreparationError
from .types import Position3, PreparedMobility, Timeline

_DISTANCE_EPSILON = 1e-12
_MAX_RANDOM_WAYPOINT_DESTINATION_ATTEMPTS = 128


@dataclass(frozen=True, slots=True)
class _Traversal:
    type: str
    speed_mps: float | None = None
    after_end: str = "hold"


def prepare_mobility(
    spec: object,
    timeline: Timeline,
    *,
    path: str = "mobility",
) -> PreparedMobility:
    """Evaluate one shared-schema mobility spec into exact-length frozen samples."""

    mobility_type = discriminator(spec)
    evaluators: dict[str, Callable[[object, Timeline, str], tuple[list[Position3], bool]]] = {
        "stationary": _stationary,
        "linear": _linear,
        "waypoint": _waypoint,
        "circular": _circular,
        "survey": _survey,
        "grid_scan": _grid_scan,
        "oscillating": _oscillating,
        "pendulum": _pendulum,
        "figure8": _figure8,
        "spiral": _spiral,
        "random_sampling": _random_sampling,
        "sampled": _sampled,
        "gauss_markov": _gauss_markov,
        "random_waypoint": _random_waypoint,
        "manhattan_grid": _manhattan_grid,
    }
    evaluator = evaluators.get(mobility_type)
    if evaluator is None:
        if mobility_type == "group_member":
            raise PosePreparationError(
                "structural_mobility_requires_scenario",
                path,
                "group_member mobility must be resolved by prepare_scenario()",
            )
        raise PosePreparationError(
            "unsupported_mobility",
            f"{path}.type",
            f"unsupported mobility type {mobility_type!r}",
        )

    try:
        positions, physical_velocity = evaluator(spec, timeline, path)
    except PosePreparationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PosePreparationError("invalid_mobility", path, str(exc)) from exc
    return _finalize(positions, timeline, physical_velocity=physical_velocity, path=path)


def derive_group_member_mobility(
    group: PreparedMobility,
    offset: object,
    timeline: Timeline,
    *,
    deviation: object | None = None,
    seed: int | None = None,
    path: str = "mobility",
) -> PreparedMobility:
    """Apply one actor's right/forward/up offset to a prepared shared group path.

    Physical paths orient offsets from their resolved heading. Nonphysical
    paths use fixed world axes with +X forward, -Y right, and +Z up.
    """

    right_m, forward_m, up_m = vec3(offset, name=f"{path}.offset_m")
    positions: list[Position3] = []
    deviations = _deviation_samples(deviation, seed, timeline, path=path)
    for index, (origin, forward) in enumerate(zip(group.positions_m, group.forward_vectors)):
        right_vec, forward_vec, local_up = _group_basis(
            forward,
            physical_velocity=group.has_physical_velocity,
        )
        deviation_right, deviation_forward, deviation_up = deviations[index]
        position = (
            np.asarray(origin, dtype=np.float64)
            + (right_m + deviation_right) * right_vec
            + (forward_m + deviation_forward) * forward_vec
            + (up_m + deviation_up) * local_up
        )
        positions.append(_as_position(position))
    return _finalize(
        positions,
        timeline,
        physical_velocity=group.has_physical_velocity,
        path=path,
    )


def group_offset_from_world_position(
    group: PreparedMobility,
    world_position: object,
    *,
    step: int = 0,
) -> Position3:
    """Project a world position into a prepared group's local offset axes."""

    index = int(step)
    if index < 0 or index >= group.steps:
        raise IndexError(f"group sample index is out of range: {index}")
    position: npt.NDArray[np.float64] = np.asarray(
        vec3(world_position, name="world_position"),
        dtype=np.float64,
    )
    origin: npt.NDArray[np.float64] = np.asarray(
        group.positions_m[index],
        dtype=np.float64,
    )
    right, forward, up = _group_basis(
        group.forward_vectors[index],
        physical_velocity=group.has_physical_velocity,
    )
    delta = position - origin
    return (
        float(np.dot(delta, right)),
        float(np.dot(delta, forward)),
        float(np.dot(delta, up)),
    )


def _group_basis(
    forward: object,
    *,
    physical_velocity: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    forward_vec = np.asarray(forward, dtype=np.float64)
    if not physical_velocity:
        forward_vec = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    right_vec = np.cross(forward_vec, world_up)
    right_norm = float(np.linalg.norm(right_vec))
    if right_norm <= _DISTANCE_EPSILON:
        right_vec = np.array((0.0, -1.0, 0.0), dtype=np.float64)
    else:
        right_vec /= right_norm
    local_up = np.cross(right_vec, forward_vec)
    local_up_norm = float(np.linalg.norm(local_up))
    if local_up_norm <= _DISTANCE_EPSILON:
        local_up = world_up
    else:
        local_up /= local_up_norm
    return right_vec, forward_vec, local_up


def _stationary(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    position = vec3(value(spec, "position_m"), name=f"{path}.position_m")
    return [position] * timeline.steps, False


def _linear(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    start = vec3(value(spec, "start_m"), name=f"{path}.start_m")
    end = vec3(value(spec, "end_m"), name=f"{path}.end_m")
    points = (start, end)
    length = _distance(start, end)
    distances = _traversal_distances(length, spec, timeline, path=path)
    return _sample_polyline(points, distances), True


def _waypoint(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    raw_points = sequence(value(spec, "points_m"), name=f"{path}.points_m")
    points = tuple(
        vec3(point, name=f"{path}.points_m[{index}]") for index, point in enumerate(raw_points)
    )
    if len(points) < 2:
        raise ValueError(f"{path}.points_m must contain at least two points")
    interpolation = str(value(spec, "interpolation", default="linear")).lower()
    if interpolation == "catmull_rom":
        points = _catmull_rom_polyline(points)
    elif interpolation != "linear":
        raise ValueError(f"{path}.interpolation must be linear or catmull_rom")
    length = _polyline_length(points)
    distances = _traversal_distances(length, spec, timeline, path=path)
    return _sample_polyline(points, distances), True


def _circular(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    center = vec3(value(spec, "center_m"), name=f"{path}.center_m")
    radius = float(value(spec, "radius_m"))
    start_angle_deg = float(value(spec, "start_angle_deg", default=0.0))
    turns = float(value(spec, "turns", default=1.0))
    clockwise = bool(value(spec, "clockwise", default=False))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{path}.radius_m must be finite and positive")
    if not math.isfinite(turns) or turns <= 0.0:
        raise ValueError(f"{path}.turns must be finite and positive")
    arc_radians = 2.0 * math.pi * turns
    length = radius * arc_radians
    distances = _traversal_distances(length, spec, timeline, path=path)
    direction = -1.0 if clockwise else 1.0
    start_radians = math.radians(start_angle_deg)
    positions: list[Position3] = []
    for distance in distances:
        angle = start_radians + direction * distance / radius
        positions.append(
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                center[2],
            )
        )
    return positions, True


def _survey(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    """Prepare a rotated lawnmower survey from its local rectangle."""

    origin = vec3(value(spec, "origin_m"), name=f"{path}.origin_m")
    width = float(value(spec, "width_m"))
    height = float(value(spec, "height_m"))
    lane_spacing = float(value(spec, "row_spacing_m"))
    heading = math.radians(float(value(spec, "heading_deg", default=0.0)))
    if lane_spacing <= 0.0 or not math.isfinite(lane_spacing):
        raise ValueError(f"{path}.row_spacing_m must be finite and positive")
    if (
        width <= 0.0
        or height <= 0.0
        or not all(math.isfinite(item) for item in (width, height, heading))
    ):
        raise ValueError(f"{path} survey dimensions and heading are invalid")
    lane_count = max(2, int(math.ceil(height / lane_spacing)) + 1)
    local_rows = np.linspace(0.0, height, lane_count)
    local_points: list[tuple[float, float]] = []
    for index, y in enumerate(local_rows):
        xs = (0.0, width) if index % 2 == 0 else (width, 0.0)
        local_points.extend((float(x), float(y)) for x in xs)
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)
    points = [
        (
            origin[0] + local_x * cos_heading - local_y * sin_heading,
            origin[1] + local_x * sin_heading + local_y * cos_heading,
            origin[2],
        )
        for local_x, local_y in local_points
    ]
    distances = _traversal_distances(_polyline_length(points), spec, timeline, path=path)
    return _sample_polyline(points, distances), True


def _grid_scan(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    """Prepare an interpolated three-dimensional snake or raster scan."""

    x_bounds = numeric_pair(value(spec, "x_bounds_m"), name=f"{path}.x_bounds_m")
    y_bounds = numeric_pair(value(spec, "y_bounds_m"), name=f"{path}.y_bounds_m")
    z_bounds = numeric_pair(value(spec, "z_bounds_m"), name=f"{path}.z_bounds_m")
    x_steps = int(value(spec, "x_steps"))
    y_steps = int(value(spec, "y_steps"))
    z_steps = int(value(spec, "z_steps"))
    if min(x_steps, y_steps, z_steps) < 1:
        raise ValueError(f"{path} grid step counts must be positive")
    x_values = tuple(float(item) for item in np.linspace(*x_bounds, x_steps))
    y_values = tuple(float(item) for item in np.linspace(*y_bounds, y_steps))
    z_values = tuple(float(item) for item in np.linspace(*z_bounds, z_steps))
    corner = str(value(spec, "start_corner", default="bottom_left"))
    if "right" in corner:
        x_values = tuple(reversed(x_values))
    if "top" in corner:
        y_values = tuple(reversed(y_values))
    pattern = str(value(spec, "traversal_pattern", default="snake"))
    points: list[Position3] = []
    row_index = 0
    for layer_index, z in enumerate(z_values):
        layer_y = (
            y_values if layer_index % 2 == 0 or pattern == "raster" else tuple(reversed(y_values))
        )
        for y in layer_y:
            row_x = x_values
            if pattern == "snake" and row_index % 2 == 1:
                row_x = tuple(reversed(row_x))
            points.extend((x, y, z) for x in row_x)
            row_index += 1
    interpolation = str(value(spec, "interpolation", default="linear"))
    if interpolation == "catmull_rom" and len(points) >= 2:
        points = list(_catmull_rom_polyline(tuple(points)))
    distances = _traversal_distances(_polyline_length(points), spec, timeline, path=path)
    return _sample_polyline(points, distances), True


def _oscillating(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    center = vec3(value(spec, "center_m"), name=f"{path}.center_m")
    amplitude = float(value(spec, "amplitude_m"))
    frequency = float(value(spec, "frequency_hz"))
    phase = math.radians(float(value(spec, "phase_deg", default=0.0)))
    axis = np.asarray(vec3(value(spec, "axis", default=(1.0, 0.0, 0.0)), name=f"{path}.axis"))
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _DISTANCE_EPSILON:
        raise ValueError(f"{path}.axis must be non-zero")
    if (
        amplitude < 0.0
        or frequency < 0.0
        or not all(math.isfinite(v) for v in (amplitude, frequency))
    ):
        raise ValueError(f"{path} amplitude and frequency must be finite and non-negative")
    axis /= axis_norm
    center_array = np.asarray(center)
    return [
        _as_position(
            center_array + amplitude * math.sin(2.0 * math.pi * frequency * time + phase) * axis
        )
        for time in timeline.timestamps_s
    ], True


def _pendulum(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    pivot = vec3(value(spec, "pivot_m"), name=f"{path}.pivot_m")
    length = float(value(spec, "length_m"))
    max_angle = math.radians(float(value(spec, "max_angle_deg")))
    frequency = float(value(spec, "frequency_hz"))
    plane = str(value(spec, "plane", default="xz")).lower()
    if (
        length <= 0.0
        or frequency < 0.0
        or not all(math.isfinite(v) for v in (length, max_angle, frequency))
    ):
        raise ValueError(f"{path} length, angle, and frequency are invalid")
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"{path}.plane must be xy, xz, or yz")
    positions: list[Position3] = []
    for time in timeline.timestamps_s:
        phase = math.radians(float(value(spec, "phase_deg", default=0.0)))
        angle = max_angle * math.sin(2.0 * math.pi * frequency * time + phase)
        if plane == "xz":
            point = (
                pivot[0] + length * math.sin(angle),
                pivot[1],
                pivot[2] - length * math.cos(angle),
            )
        elif plane == "yz":
            point = (
                pivot[0],
                pivot[1] + length * math.sin(angle),
                pivot[2] - length * math.cos(angle),
            )
        else:
            point = (
                pivot[0] + length * math.sin(angle),
                pivot[1] + length * math.cos(angle),
                pivot[2],
            )
        positions.append(point)
    return positions, True


def _figure8(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    center = vec3(value(spec, "center_m"), name=f"{path}.center_m")
    size = float(value(spec, "size_m"))
    plane = str(value(spec, "plane", default="xy"))
    turns = float(value(spec, "turns", default=1.0))
    if size <= 0.0 or turns <= 0.0 or plane not in ("xy", "xz", "yz"):
        raise ValueError(f"{path} figure8 fields are invalid")
    dense: list[Position3] = []
    for parameter in np.linspace(0.0, 2.0 * math.pi * turns, max(129, int(128 * turns))):
        denominator = 1.0 + math.sin(parameter) ** 2
        primary = size * math.cos(parameter) / denominator
        secondary = size * math.sin(parameter) * math.cos(parameter) / denominator
        if plane == "xy":
            dense.append((center[0] + primary, center[1] + secondary, center[2]))
        elif plane == "xz":
            dense.append((center[0] + primary, center[1], center[2] + secondary))
        else:
            dense.append((center[0], center[1] + primary, center[2] + secondary))
    distances = _traversal_distances(_polyline_length(dense), spec, timeline, path=path)
    return _sample_polyline(dense, distances), True


def _spiral(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    center = vec3(value(spec, "center_m"), name=f"{path}.center_m")
    radius = float(value(spec, "radius_m"))
    start_altitude = float(value(spec, "start_altitude_m"))
    end_altitude = float(value(spec, "end_altitude_m"))
    turns = float(value(spec, "turns"))
    start_angle = math.radians(float(value(spec, "start_angle_deg", default=0.0)))
    direction = -1.0 if bool(value(spec, "clockwise", default=False)) else 1.0
    if radius <= 0.0 or turns <= 0.0:
        raise ValueError(f"{path} radius and turns must be positive")
    parameters = np.linspace(0.0, 1.0, max(129, int(128 * turns)))
    dense: list[Position3] = []
    for fraction in parameters:
        angle = start_angle + direction * 2.0 * math.pi * turns * float(fraction)
        dense.append(
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                start_altitude + (end_altitude - start_altitude) * float(fraction),
            )
        )
    distances = _traversal_distances(_polyline_length(dense), spec, timeline, path=path)
    return _sample_polyline(dense, distances), True


def _random_sampling(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    x_bounds = numeric_pair(value(spec, "x_bounds_m"), name=f"{path}.x_bounds_m")
    y_bounds = numeric_pair(value(spec, "y_bounds_m"), name=f"{path}.y_bounds_m")
    z_bounds = numeric_pair(value(spec, "z_bounds_m"), name=f"{path}.z_bounds_m")
    minimum = (x_bounds[0], y_bounds[0], z_bounds[0])
    maximum = (x_bounds[1], y_bounds[1], z_bounds[1])
    initial_raw = value(spec, "initial_position_m", default=None)
    initial = (
        vec3(initial_raw, name=f"{path}.initial_position_m") if initial_raw is not None else None
    )
    seed = int(value(spec, "seed"))
    if any(high < low for low, high in zip(minimum, maximum)):
        raise ValueError(f"{path} bounds must have min <= max on every axis")
    if initial is not None and any(
        component < low or component > high
        for component, low, high in zip(initial, minimum, maximum)
    ):
        raise ValueError(f"{path}.initial_position_m must lie within configured bounds")
    sampling = str(value(spec, "sampling", default="uniform"))
    min_distance_raw = value(spec, "min_distance_m", default=None)
    min_distance = float(min_distance_raw) if min_distance_raw is not None else None
    positions: list[Position3] = []
    for index, timestamp in enumerate(timeline.timestamps_s):
        if index == 0 and initial is not None:
            positions.append(initial)
            continue
        candidate: Position3 | None = None
        for attempt in range(256):
            samples = _keyed_uniforms(seed + attempt, timestamp, 3, namespace="random_sampling")
            candidate = (
                minimum[0] + samples[0] * (maximum[0] - minimum[0]),
                minimum[1] + samples[1] * (maximum[1] - minimum[1]),
                minimum[2] + samples[2] * (maximum[2] - minimum[2]),
            )
            if sampling != "poisson_disk":
                break
            if min_distance is None:
                raise ValueError(f"{path}.min_distance_m is required for poisson_disk sampling")
            if all(_distance(candidate, previous) >= min_distance for previous in positions):
                break
        else:
            raise ValueError(f"{path} could not satisfy min_distance_m within configured bounds")
        assert candidate is not None
        positions.append(candidate)
    # Independent samples are spatial observations, not a physically continuous
    # path. Zero velocities and a false capability flag prevent false Doppler.
    return positions, False


def _sampled(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    positions_path = f"{path}.positions_m"
    positions = [
        vec3(position, name=f"{positions_path}[{index}]")
        for index, position in enumerate(sequence(value(spec, "positions_m"), name=positions_path))
    ]
    return positions, True


def _gauss_markov(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    initial = vec3(value(spec, "initial_position_m"), name=f"{path}.initial_position_m")
    bounds = _bounds3(spec, path)
    alpha = float(value(spec, "alpha"))
    mean_speed = float(value(spec, "mean_speed_mps"))
    mean_direction = math.radians(float(value(spec, "mean_direction_deg", default=0.0)))
    speed_std = float(value(spec, "speed_std_mps", default=0.0))
    direction_std = math.radians(float(value(spec, "direction_std_deg", default=0.0)))
    seed = int(value(spec, "seed"))
    if timeline.duration_s <= 0.0:
        return [initial] * timeline.steps, True
    # A fixed internal time grid makes results invariant to output sample count.
    integration_dt = min(0.05, timeline.duration_s)
    internal_times = [
        float(timestamp) for timestamp in np.arange(0.0, timeline.duration_s, integration_dt)
    ] + [timeline.duration_s]
    rng = np.random.default_rng(seed)
    positions: list[npt.NDArray[np.float64]] = [np.asarray(initial, dtype=np.float64)]
    speed = mean_speed
    direction = mean_direction
    for previous_time, current_time in zip(internal_times, internal_times[1:]):
        dt = current_time - previous_time
        correlation = alpha**dt if alpha > 0.0 else 0.0
        noise_scale = math.sqrt(max(0.0, 1.0 - correlation * correlation))
        speed = max(
            0.0,
            correlation * speed
            + (1.0 - correlation) * mean_speed
            + noise_scale * rng.normal(0.0, speed_std),
        )
        direction = (
            correlation * direction
            + (1.0 - correlation) * mean_direction
            + noise_scale * rng.normal(0.0, direction_std)
        )
        next_position = positions[-1] + np.array(
            (speed * math.cos(direction) * dt, speed * math.sin(direction) * dt, 0.0)
        )
        next_position = np.asarray(
            [
                min(max(component, axis_bounds[0]), axis_bounds[1])
                for component, axis_bounds in zip(next_position, bounds)
            ]
        )
        positions.append(next_position)
    return _sample_time_series(internal_times, positions, timeline.timestamps_s), True


def _random_waypoint(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    initial = vec3(value(spec, "initial_position_m"), name=f"{path}.initial_position_m")
    bounds = _bounds3(spec, path)
    speed_range = numeric_pair(value(spec, "speed_range_mps"), name=f"{path}.speed_range_mps")
    pause_range = numeric_pair(
        value(spec, "pause_range_s", default=(0.0, 0.0)), name=f"{path}.pause_range_s"
    )
    seed = int(value(spec, "seed"))
    events_t, events_p = _random_waypoint_events(
        initial, bounds, speed_range, pause_range, seed, timeline.duration_s
    )
    return _sample_time_series(events_t, events_p, timeline.timestamps_s), True


def _manhattan_grid(
    spec: object,
    timeline: Timeline,
    path: str,
) -> tuple[list[Position3], bool]:
    origin_xy = numeric_pair(value(spec, "origin_xy_m"), name=f"{path}.origin_xy_m")
    block_size = float(value(spec, "block_size_m"))
    width = int(value(spec, "grid_width"))
    height = int(value(spec, "grid_height"))
    altitude = float(value(spec, "altitude_m"))
    turn_probability = float(value(spec, "turn_probability", default=0.5))
    speed_range = numeric_pair(value(spec, "speed_range_mps"), name=f"{path}.speed_range_mps")
    pause_range = numeric_pair(
        value(spec, "pause_range_s", default=(0.0, 0.0)), name=f"{path}.pause_range_s"
    )
    seed = int(value(spec, "seed"))
    if min(width, height) < 1 or block_size <= 0.0:
        raise ValueError(f"{path} grid dimensions and block size must be positive")
    rng = np.random.default_rng(seed)
    node = (0, 0)
    direction = (1, 0) if width > 1 else (0, 1)
    event_times = [0.0]
    event_positions: list[np.ndarray] = [np.asarray((*origin_xy, altitude))]
    while event_times[-1] < timeline.duration_s:
        neighbors = [
            (dx, dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= node[0] + dx < width and 0 <= node[1] + dy < height
        ]
        if not neighbors:
            break
        if direction not in neighbors or rng.random() < turn_probability:
            direction = neighbors[int(rng.integers(0, len(neighbors)))]
        node = (node[0] + direction[0], node[1] + direction[1])
        speed = float(rng.uniform(*speed_range))
        arrival = event_times[-1] + block_size / speed
        event_times.append(arrival)
        event_positions.append(
            np.asarray(
                (
                    origin_xy[0] + node[0] * block_size,
                    origin_xy[1] + node[1] * block_size,
                    altitude,
                )
            )
        )
        pause = float(rng.uniform(*pause_range))
        if pause > 0.0:
            event_times.append(arrival + pause)
            event_positions.append(event_positions[-1].copy())
    return _sample_time_series(event_times, event_positions, timeline.timestamps_s), True


def _parse_traversal(spec: object, *, path: str) -> _Traversal:
    traversal = value(spec, "traversal", default=None)
    if traversal is None:
        return _Traversal("fit_duration")
    traversal_type = discriminator(traversal)
    if traversal_type == "fit_duration":
        return _Traversal(traversal_type)
    if traversal_type != "constant_speed":
        raise ValueError(f"{path}.traversal.type is unsupported")
    speed = float(value(traversal, "speed_mps"))
    after_end_raw = value(traversal, "after_end", default="hold")
    after_end = getattr(after_end_raw, "value", after_end_raw)
    after_end = str(after_end).lower()
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"{path}.traversal.speed_mps must be finite and positive")
    if after_end not in ("hold", "loop", "ping_pong"):
        raise ValueError(f"{path}.traversal.after_end is invalid")
    return _Traversal(traversal_type, speed, after_end)


def _traversal_distances(
    length: float,
    spec: object,
    timeline: Timeline,
    *,
    path: str,
) -> tuple[float, ...]:
    if not math.isfinite(length) or length < 0.0:
        raise ValueError(f"{path} path length is invalid")
    if length <= _DISTANCE_EPSILON:
        return (0.0,) * timeline.steps
    _require_moving_timeline(timeline, path=path)
    traversal = _parse_traversal(spec, path=path)
    if traversal.type == "fit_duration":
        return tuple(
            length * timestamp / timeline.duration_s for timestamp in timeline.timestamps_s
        )
    assert traversal.speed_mps is not None
    return tuple(
        _apply_after_end(traversal.speed_mps * timestamp, length, traversal.after_end)
        for timestamp in timeline.timestamps_s
    )


def _apply_after_end(distance: float, length: float, mode: str) -> float:
    if mode == "hold":
        return min(distance, length)
    if mode == "loop":
        return distance % length
    period = 2.0 * length
    phase = distance % period
    return phase if phase <= length else period - phase


def _sample_polyline(
    points: tuple[Position3, ...] | list[Position3],
    distances: tuple[float, ...],
) -> list[Position3]:
    if not points:
        raise ValueError("polyline must contain at least one point")
    if len(points) == 1:
        return [points[0]] * len(distances)
    point_array: npt.NDArray[np.float64] = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(point_array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= _DISTANCE_EPSILON:
        return [_as_position(point_array[0])] * len(distances)
    positions: list[Position3] = []
    for distance in distances:
        clamped = min(max(float(distance), 0.0), total)
        segment = min(int(np.searchsorted(cumulative, clamped, side="right") - 1), len(points) - 2)
        while segment < len(segment_lengths) - 1 and segment_lengths[segment] <= _DISTANCE_EPSILON:
            segment += 1
        length = float(segment_lengths[segment])
        if length <= _DISTANCE_EPSILON:
            positions.append(_as_position(point_array[segment]))
            continue
        fraction = (clamped - float(cumulative[segment])) / length
        positions.append(
            _as_position(
                point_array[segment] + fraction * (point_array[segment + 1] - point_array[segment])
            )
        )
    return positions


def _catmull_rom_polyline(points: tuple[Position3, ...]) -> tuple[Position3, ...]:
    dense: list[Position3] = []
    samples_per_segment = 64
    arrays: list[npt.NDArray[np.float64]] = [
        np.asarray(point, dtype=np.float64) for point in points
    ]
    for segment in range(len(arrays) - 1):
        p0 = arrays[max(segment - 1, 0)]
        p1 = arrays[segment]
        p2 = arrays[segment + 1]
        p3 = arrays[min(segment + 2, len(arrays) - 1)]
        for sample in range(samples_per_segment):
            t = sample / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            dense.append(_as_position(point))
    dense.append(points[-1])
    return tuple(dense)


def _finalize(
    positions: list[Position3] | tuple[Position3, ...],
    timeline: Timeline,
    *,
    physical_velocity: bool,
    path: str,
) -> PreparedMobility:
    if len(positions) != timeline.steps:
        raise PosePreparationError(
            "sample_count_mismatch",
            path,
            f"evaluator returned {len(positions)} samples for {timeline.steps} timeline steps",
        )
    normalized = tuple(_as_position(position) for position in positions)
    positions_move = _positions_move(normalized)
    if positions_move:
        _require_moving_timeline(timeline, path=path)
    has_physical_velocity = physical_velocity and positions_move
    velocities = (
        _velocities(normalized, timeline)
        if has_physical_velocity
        else ((0.0, 0.0, 0.0),) * timeline.steps
    )
    forwards = _stable_forward_vectors(normalized)
    return PreparedMobility(
        normalized,
        velocities,
        forwards,
        has_physical_velocity,
    )


def prepare_sampled_mobility(
    positions: list[Position3] | tuple[Position3, ...],
    timeline: Timeline,
    *,
    physical_velocity: bool = True,
    path: str = "mobility",
) -> PreparedMobility:
    """Canonicalize an externally sampled trajectory for orientation evaluation.

    Scripted generator and measurement adapters already own their position
    sampling. This boundary derives velocities and stable forward vectors with
    the same rules as schema-backed mobility models, including treating a
    degenerate constant trajectory as having no physical velocity.
    """

    return _finalize(
        positions,
        timeline,
        physical_velocity=physical_velocity,
        path=path,
    )


def _velocities(
    positions: tuple[Position3, ...],
    timeline: Timeline,
) -> tuple[Position3, ...]:
    if len(positions) == 1 or timeline.duration_s <= 0.0:
        return ((0.0, 0.0, 0.0),) * len(positions)
    edge_order: Literal[1, 2] = 2 if len(positions) >= 3 else 1
    gradients: list[npt.NDArray[np.float64]] = []
    for axis in range(3):
        components = [position[axis] for position in positions]
        gradient = np.gradient(
            components,
            timeline.timestamps_s,
            edge_order=edge_order,
        )
        gradients.append(np.asarray(gradient, dtype=np.float64))
    return tuple(
        (float(gradients[0][index]), float(gradients[1][index]), float(gradients[2][index]))
        for index in range(len(positions))
    )


def _stable_forward_vectors(positions: tuple[Position3, ...]) -> tuple[Position3, ...]:
    if len(positions) == 1:
        return ((1.0, 0.0, 0.0),)
    array: npt.NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    first_valid: np.ndarray | None = None
    for delta in np.diff(array, axis=0):
        norm = float(np.linalg.norm(delta))
        if norm > _DISTANCE_EPSILON:
            first_valid = delta / norm
            break
    forward = first_valid if first_valid is not None else np.array((1.0, 0.0, 0.0))
    result: list[Position3] = []
    for index in range(len(array)):
        if index < len(array) - 1:
            delta = array[index + 1] - array[index]
            norm = float(np.linalg.norm(delta))
            if norm > _DISTANCE_EPSILON:
                forward = delta / norm
        result.append(_as_position(forward))
    return tuple(result)


def _deviation_samples(
    deviation: object | None,
    seed: int | None,
    timeline: Timeline,
    *,
    path: str,
) -> tuple[Position3, ...]:
    if deviation is None:
        return ((0.0, 0.0, 0.0),) * timeline.steps
    if seed is None:
        raise PosePreparationError(
            "missing_seed",
            f"{path}.deviation",
            "group deviation requires an explicit seed",
        )
    if isinstance(deviation, (int, float)):
        bounds = (float(deviation),) * 3
    else:
        bounds = vec3(deviation, name=f"{path}.deviation")
    if any(bound < 0.0 for bound in bounds):
        raise ValueError(f"{path}.deviation bounds must be non-negative")
    samples = (
        _keyed_uniforms(seed, timestamp, 3, namespace="group_deviation")
        for timestamp in timeline.timestamps_s
    )
    return tuple(
        (
            (sample[0] * 2.0 - 1.0) * bounds[0],
            (sample[1] * 2.0 - 1.0) * bounds[1],
            (sample[2] * 2.0 - 1.0) * bounds[2],
        )
        for sample in samples
    )


def _keyed_uniforms(
    seed: int, timestamp: float, count: int, *, namespace: str
) -> tuple[float, ...]:
    key = f"{namespace}:{int(seed)}:{float(timestamp).hex()}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    values = []
    for index in range(count):
        offset = index * 8
        raw = int.from_bytes(digest[offset : offset + 8], "big")
        values.append(raw / float(2**64))
    return tuple(values)


def _bounds3(
    spec: object,
    path: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds = (
        numeric_pair(value(spec, "x_bounds_m"), name=f"{path}.x_bounds_m"),
        numeric_pair(value(spec, "y_bounds_m"), name=f"{path}.y_bounds_m"),
        numeric_pair(value(spec, "z_bounds_m"), name=f"{path}.z_bounds_m"),
    )
    if any(high < low for low, high in bounds):
        raise ValueError(f"{path} bounds must be ordered")
    return bounds


def _random_waypoint_events(
    initial: Position3,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    speed_range: tuple[float, float],
    pause_range: tuple[float, float],
    seed: int,
    duration_s: float,
) -> tuple[list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    event_times = [0.0]
    event_positions: list[npt.NDArray[np.float64]] = [np.asarray(initial, dtype=np.float64)]
    while event_times[-1] < duration_s:
        destination: npt.NDArray[np.float64] | None = None
        distance = 0.0
        for _ in range(_MAX_RANDOM_WAYPOINT_DESTINATION_ATTEMPTS):
            candidate = np.asarray([rng.uniform(low, high) for low, high in bounds])
            candidate_distance = float(np.linalg.norm(candidate - event_positions[-1]))
            if candidate_distance > _DISTANCE_EPSILON:
                destination = candidate
                distance = candidate_distance
                break
        if destination is None:
            raise ValueError("random waypoint bounds could not produce a distinct destination")
        speed = float(rng.uniform(*speed_range))
        if speed <= 0.0:
            raise ValueError("random waypoint speeds must be positive")
        arrival = event_times[-1] + distance / speed
        if not math.isfinite(arrival) or arrival <= event_times[-1]:
            raise ValueError("random waypoint travel time must advance the timeline")
        event_times.append(arrival)
        event_positions.append(destination)
        pause = float(rng.uniform(*pause_range))
        if pause < 0.0:
            raise ValueError("random waypoint pauses must be non-negative")
        if pause > 0.0:
            event_times.append(arrival + pause)
            event_positions.append(destination.copy())
    return event_times, event_positions


def _sample_time_series(
    event_times: list[float],
    event_positions: list[np.ndarray],
    sample_times: tuple[float, ...],
) -> list[Position3]:
    if len(event_times) != len(event_positions) or not event_times:
        raise ValueError("time-series events are invalid")
    result: list[Position3] = []
    for timestamp in sample_times:
        if timestamp <= event_times[0] or len(event_times) == 1:
            result.append(_as_position(event_positions[0]))
            continue
        if timestamp >= event_times[-1]:
            result.append(_as_position(event_positions[-1]))
            continue
        upper = int(np.searchsorted(event_times, timestamp, side="right"))
        lower = upper - 1
        interval = event_times[upper] - event_times[lower]
        if interval <= _DISTANCE_EPSILON:
            result.append(_as_position(event_positions[upper]))
            continue
        fraction = (timestamp - event_times[lower]) / interval
        result.append(
            _as_position(
                event_positions[lower]
                + fraction * (event_positions[upper] - event_positions[lower])
            )
        )
    return result


def _polyline_length(points: tuple[Position3, ...] | list[Position3]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1).sum()
    )


def _distance(first: Position3, second: Position3) -> float:
    return float(
        np.linalg.norm(np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64))
    )


def _positions_move(positions: tuple[Position3, ...]) -> bool:
    if len(positions) < 2:
        return False
    first: npt.NDArray[np.float64] = np.asarray(positions[0], dtype=np.float64)
    return any(
        float(np.linalg.norm(np.asarray(position) - first)) > _DISTANCE_EPSILON
        for position in positions[1:]
    )


def _require_moving_timeline(timeline: Timeline, *, path: str) -> None:
    if timeline.steps < 2:
        raise PosePreparationError(
            "moving_timeline_too_short",
            path,
            "moving mobility requires at least two timeline steps",
        )
    if timeline.duration_s <= 0.0:
        raise PosePreparationError(
            "moving_timeline_zero_duration",
            path,
            "moving mobility requires positive timeline duration_s",
        )


def _as_position(value_to_convert: object) -> Position3:
    array = np.asarray(value_to_convert, dtype=np.float64).reshape(-1)
    if array.size != 3 or not np.isfinite(array).all():
        raise ValueError("position evaluator produced a non-finite three-vector")
    return (float(array[0]), float(array[1]), float(array[2]))
