"""Immutable actor, mobility, orientation, group, and timeline specifications.

A world position belongs to an actor's mobility specification, or to the
mobility specification of the optional group referenced by a ``group_member``
actor.  The models are renderer-neutral and contain no session identifiers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCENARIO_SCHEMA_VERSION = 2
# Seeds use the non-negative signed 32-bit range for portable, deterministic
# serialization across supported random-number runtimes.
MAX_RANDOM_SEED = 2_147_483_647


def _finite(value: float) -> float:
    """Reject NaN and infinities, including inside vectors and ranges."""
    import math

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return number


FiniteFloat: TypeAlias = Annotated[float, BeforeValidator(_finite)]
Vector2: TypeAlias = tuple[FiniteFloat, FiniteFloat]
Vector3: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Range2: TypeAlias = tuple[FiniteFloat, FiniteFloat]
RandomSeed: TypeAlias = Annotated[
    int,
    Field(
        ge=0,
        le=MAX_RANDOM_SEED,
        description="Portable non-negative signed 32-bit random seed.",
    ),
]


class FrozenStrictModel(BaseModel):
    """Pydantic base for immutable, typo-rejecting scenario values."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def _non_empty(value: str, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must be a non-empty string")
    return normalized


def _validate_ordered_range(value: Range2, *, label: str) -> Range2:
    if value[0] > value[1]:
        raise ValueError(f"{label} minimum must not exceed its maximum")
    return value


class ActorRole(str, Enum):
    """Actor role inferred from its enclosing ``actors`` subsection."""

    TX = "tx"
    RX = "rx"
    TARGET = "target"


# ---------------------------------------------------------------------------
# Timeline and path traversal
# ---------------------------------------------------------------------------
class TimelineSpec(FrozenStrictModel):
    """Endpoint-inclusive scenario timeline."""

    steps: int = Field(ge=1)
    duration_s: FiniteFloat = Field(ge=0.0)


class FitDurationTraversalSpec(FrozenStrictModel):
    """Traverse the complete authored path over the scenario duration."""

    type: Literal["fit_duration"] = "fit_duration"


class ConstantSpeedTraversalSpec(FrozenStrictModel):
    """Traverse a path at a physical speed and define end-of-path behavior."""

    type: Literal["constant_speed"] = "constant_speed"
    speed_mps: FiniteFloat = Field(gt=0.0)
    after_end: Literal["hold", "loop", "ping_pong"] = "hold"


TraversalSpec: TypeAlias = Annotated[
    FitDurationTraversalSpec | ConstantSpeedTraversalSpec,
    Field(discriminator="type"),
]


def _fit_duration() -> FitDurationTraversalSpec:
    return FitDurationTraversalSpec()


# ---------------------------------------------------------------------------
# Mobility
# ---------------------------------------------------------------------------
class StationaryMobilitySpec(FrozenStrictModel):
    type: Literal["stationary"] = "stationary"
    position_m: Vector3


class LinearMobilitySpec(FrozenStrictModel):
    type: Literal["linear"] = "linear"
    start_m: Vector3
    end_m: Vector3
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class WaypointMobilitySpec(FrozenStrictModel):
    type: Literal["waypoint"] = "waypoint"
    points_m: tuple[Vector3, ...] = Field(min_length=2)
    interpolation: Literal["linear", "catmull_rom"] = "linear"
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class SampledMobilitySpec(FrozenStrictModel):
    """Physical positions already sampled at each scenario timeline step."""

    type: Literal["sampled"] = "sampled"
    positions_m: tuple[Vector3, ...] = Field(min_length=1)


class CircularMobilitySpec(FrozenStrictModel):
    type: Literal["circular"] = "circular"
    center_m: Vector3
    radius_m: FiniteFloat = Field(gt=0.0)
    start_angle_deg: FiniteFloat = 0.0
    clockwise: bool = False
    turns: FiniteFloat = Field(default=1.0, gt=0.0)
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class SurveyMobilitySpec(FrozenStrictModel):
    """Lawnmower survey path in a rotated horizontal rectangle."""

    type: Literal["survey"] = "survey"
    origin_m: Vector3
    width_m: FiniteFloat = Field(gt=0.0)
    height_m: FiniteFloat = Field(gt=0.0)
    row_spacing_m: FiniteFloat = Field(gt=0.0)
    heading_deg: FiniteFloat = 0.0
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class GridScanMobilitySpec(FrozenStrictModel):
    type: Literal["grid_scan"] = "grid_scan"
    x_bounds_m: Range2
    y_bounds_m: Range2
    z_bounds_m: Range2
    x_steps: int = Field(ge=1)
    y_steps: int = Field(ge=1)
    z_steps: int = Field(ge=1)
    traversal_pattern: Literal["snake", "raster"] = "snake"
    start_corner: Literal["bottom_left", "bottom_right", "top_left", "top_right"] = "bottom_left"
    interpolation: Literal["linear", "catmull_rom"] = "linear"
    traversal: TraversalSpec = Field(default_factory=_fit_duration)

    @field_validator("x_bounds_m", "y_bounds_m", "z_bounds_m")
    @classmethod
    def validate_bounds(cls, value: Range2, info) -> Range2:
        return _validate_ordered_range(value, label=info.field_name)


class OscillatingMobilitySpec(FrozenStrictModel):
    type: Literal["oscillating"] = "oscillating"
    center_m: Vector3
    axis: Vector3
    amplitude_m: FiniteFloat = Field(ge=0.0)
    frequency_hz: FiniteFloat = Field(gt=0.0)
    phase_deg: FiniteFloat = 0.0

    @field_validator("axis")
    @classmethod
    def validate_axis(cls, value: Vector3) -> Vector3:
        if all(component == 0.0 for component in value):
            raise ValueError("axis must be non-zero")
        return value


class PendulumMobilitySpec(FrozenStrictModel):
    type: Literal["pendulum"] = "pendulum"
    pivot_m: Vector3
    length_m: FiniteFloat = Field(gt=0.0)
    max_angle_deg: FiniteFloat = Field(gt=0.0, lt=180.0)
    frequency_hz: FiniteFloat = Field(gt=0.0)
    plane: Literal["xy", "xz", "yz"] = "xz"
    phase_deg: FiniteFloat = 0.0


class Figure8MobilitySpec(FrozenStrictModel):
    type: Literal["figure8"] = "figure8"
    center_m: Vector3
    size_m: FiniteFloat = Field(gt=0.0)
    plane: Literal["xy", "xz", "yz"] = "xy"
    turns: FiniteFloat = Field(default=1.0, gt=0.0)
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class SpiralMobilitySpec(FrozenStrictModel):
    type: Literal["spiral"] = "spiral"
    center_m: Vector3
    radius_m: FiniteFloat = Field(gt=0.0)
    start_altitude_m: FiniteFloat
    end_altitude_m: FiniteFloat
    turns: FiniteFloat = Field(gt=0.0)
    start_angle_deg: FiniteFloat = 0.0
    clockwise: bool = False
    traversal: TraversalSpec = Field(default_factory=_fit_duration)


class RandomSamplingMobilitySpec(FrozenStrictModel):
    """Independent spatial samples; this model has no physical tangent."""

    type: Literal["random_sampling"] = "random_sampling"
    x_bounds_m: Range2
    y_bounds_m: Range2
    z_bounds_m: Range2
    initial_position_m: Vector3 | None = Field(
        default=None,
        description=(
            "Optional exact first observation. Later positions remain seeded independent samples."
        ),
    )
    seed: RandomSeed
    sampling: Literal["uniform", "poisson_disk"] = "uniform"
    min_distance_m: FiniteFloat | None = Field(default=None, gt=0.0)

    @field_validator("x_bounds_m", "y_bounds_m", "z_bounds_m")
    @classmethod
    def validate_bounds(cls, value: Range2, info) -> Range2:
        return _validate_ordered_range(value, label=info.field_name)

    @model_validator(mode="after")
    def validate_sampling(self) -> "RandomSamplingMobilitySpec":
        if self.sampling == "poisson_disk" and self.min_distance_m is None:
            raise ValueError("min_distance_m is required for poisson_disk sampling")
        if self.sampling == "uniform" and self.min_distance_m is not None:
            raise ValueError("min_distance_m is only valid for poisson_disk sampling")
        if self.initial_position_m is not None and any(
            component < bounds[0] or component > bounds[1]
            for component, bounds in zip(
                self.initial_position_m,
                (self.x_bounds_m, self.y_bounds_m, self.z_bounds_m),
            )
        ):
            raise ValueError("initial_position_m must lie within random_sampling bounds")
        return self


class GaussMarkovMobilitySpec(FrozenStrictModel):
    type: Literal["gauss_markov"] = "gauss_markov"
    initial_position_m: Vector3
    x_bounds_m: Range2
    y_bounds_m: Range2
    z_bounds_m: Range2
    alpha: FiniteFloat = Field(ge=0.0, le=1.0)
    mean_speed_mps: FiniteFloat = Field(ge=0.0)
    mean_direction_deg: FiniteFloat = 0.0
    speed_std_mps: FiniteFloat = Field(default=0.0, ge=0.0)
    direction_std_deg: FiniteFloat = Field(default=0.0, ge=0.0)
    seed: RandomSeed

    @field_validator("x_bounds_m", "y_bounds_m", "z_bounds_m")
    @classmethod
    def validate_bounds(cls, value: Range2, info) -> Range2:
        return _validate_ordered_range(value, label=info.field_name)


class RandomWaypointMobilitySpec(FrozenStrictModel):
    type: Literal["random_waypoint"] = "random_waypoint"
    initial_position_m: Vector3
    x_bounds_m: Range2
    y_bounds_m: Range2
    z_bounds_m: Range2
    speed_range_mps: Range2
    pause_range_s: Range2 = (0.0, 0.0)
    seed: RandomSeed

    @field_validator("x_bounds_m", "y_bounds_m", "z_bounds_m")
    @classmethod
    def validate_bounds(cls, value: Range2, info) -> Range2:
        return _validate_ordered_range(value, label=info.field_name)

    @field_validator("speed_range_mps")
    @classmethod
    def validate_speed_range(cls, value: Range2) -> Range2:
        _validate_ordered_range(value, label="speed_range_mps")
        if value[0] <= 0.0:
            raise ValueError("speed_range_mps values must be positive")
        return value

    @field_validator("pause_range_s")
    @classmethod
    def validate_pause_range(cls, value: Range2) -> Range2:
        _validate_ordered_range(value, label="pause_range_s")
        if value[0] < 0.0:
            raise ValueError("pause_range_s values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_spatial_extent(self) -> "RandomWaypointMobilitySpec":
        bounds = (self.x_bounds_m, self.y_bounds_m, self.z_bounds_m)
        if all(low == high for low, high in bounds):
            raise ValueError("random_waypoint bounds must span a nonzero spatial extent")
        return self


class ManhattanGridMobilitySpec(FrozenStrictModel):
    """Rectangular grid anchored in XY at an absolute world altitude."""

    type: Literal["manhattan_grid"] = "manhattan_grid"
    origin_xy_m: Vector2
    block_size_m: FiniteFloat = Field(gt=0.0)
    grid_width: int = Field(ge=1)
    grid_height: int = Field(ge=1)
    altitude_m: FiniteFloat = Field(description="Absolute world altitude in meters.")
    turn_probability: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)
    speed_range_mps: Range2
    pause_range_s: Range2 = (0.0, 0.0)
    seed: RandomSeed

    @field_validator("speed_range_mps")
    @classmethod
    def validate_speed_range(cls, value: Range2) -> Range2:
        _validate_ordered_range(value, label="speed_range_mps")
        if value[0] <= 0.0:
            raise ValueError("speed_range_mps values must be positive")
        return value

    @field_validator("pause_range_s")
    @classmethod
    def validate_pause_range(cls, value: Range2) -> Range2:
        _validate_ordered_range(value, label="pause_range_s")
        if value[0] < 0.0:
            raise ValueError("pause_range_s values must be non-negative")
        return value


class NetworkRouteMobilitySpec(FrozenStrictModel):
    """Motion prepared offline from a cached spatial-network graph."""

    type: Literal["network_route"] = "network_route"
    travel_mode: Literal["pedestrian", "bike", "car", "drone"] = "pedestrian"
    route: Literal["random_walk", "shortest_path"] = "shortest_path"
    altitude_m: FiniteFloat = Field(
        default=0.0,
        description="World-space Z coordinate paired with the graph's local XY coordinates.",
    )
    seed: RandomSeed | None = None
    graph_path: str | None = Field(
        default=None,
        description=(
            "Cached GraphML, XML, or node-link JSON path relative to the scenario. "
            "Defaults to street_network.graphml."
        ),
    )
    start_node: str | int | None = None
    end_node: str | int | None = None
    traversal: TraversalSpec = Field(default_factory=_fit_duration)

    @field_validator("graph_path")
    @classmethod
    def validate_graph_path(cls, value: str | None) -> str | None:
        return None if value is None else _non_empty(value, label="graph_path")

    @model_validator(mode="after")
    def validate_route(self) -> "NetworkRouteMobilitySpec":
        has_start = self.start_node is not None
        has_end = self.end_node is not None
        if has_start != has_end:
            raise ValueError("start_node and end_node must be provided together")
        if self.route == "random_walk" and self.seed is None:
            raise ValueError("seed is required for a random_walk network route")
        if self.route == "random_walk" and has_start:
            raise ValueError("start_node and end_node are only valid for shortest_path routes")
        return self


class MeshSequenceMobilitySpec(FrozenStrictModel):
    """Position samples loaded from a renderer-neutral external resource."""

    type: Literal["mesh_sequence"] = "mesh_sequence"
    positions_path: str
    position_key: str = "positions"
    interpolation: Literal["linear", "step"] = "linear"
    traversal: TraversalSpec = Field(default_factory=_fit_duration)

    @field_validator("positions_path", "position_key")
    @classmethod
    def validate_resource_string(cls, value: str, info) -> str:
        return _non_empty(value, label=info.field_name)


class GroupOffsetSpec(FrozenStrictModel):
    right: FiniteFloat = 0.0
    forward: FiniteFloat = 0.0
    up: FiniteFloat = 0.0


class GroupMemberMobilitySpec(FrozenStrictModel):
    type: Literal["group_member"] = "group_member"
    group: str
    offset_m: GroupOffsetSpec = Field(default_factory=GroupOffsetSpec)

    @field_validator("group")
    @classmethod
    def validate_group_name(cls, value: str) -> str:
        return _non_empty(value, label="group")


StandaloneMobilitySpec: TypeAlias = Annotated[
    StationaryMobilitySpec
    | LinearMobilitySpec
    | WaypointMobilitySpec
    | SampledMobilitySpec
    | CircularMobilitySpec
    | SurveyMobilitySpec
    | GridScanMobilitySpec
    | OscillatingMobilitySpec
    | PendulumMobilitySpec
    | Figure8MobilitySpec
    | SpiralMobilitySpec
    | RandomSamplingMobilitySpec
    | GaussMarkovMobilitySpec
    | RandomWaypointMobilitySpec
    | ManhattanGridMobilitySpec
    | NetworkRouteMobilitySpec
    | MeshSequenceMobilitySpec,
    Field(discriminator="type"),
]

MobilitySpec: TypeAlias = Annotated[
    StandaloneMobilitySpec | GroupMemberMobilitySpec,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------
class FixedOrientationSpec(FrozenStrictModel):
    type: Literal["fixed"] = "fixed"
    yaw_deg: FiniteFloat = 0.0
    pitch_deg: FiniteFloat = 0.0
    roll_deg: FiniteFloat = 0.0


class OrientationKeyframeSpec(FrozenStrictModel):
    time_s: FiniteFloat = Field(ge=0.0)
    yaw_deg: FiniteFloat = 0.0
    pitch_deg: FiniteFloat = 0.0
    roll_deg: FiniteFloat = 0.0


class KeyframesOrientationSpec(FrozenStrictModel):
    type: Literal["keyframes"] = "keyframes"
    keyframes: tuple[OrientationKeyframeSpec, ...] = Field(min_length=2)

    @field_validator("keyframes")
    @classmethod
    def validate_keyframe_times(
        cls, value: tuple[OrientationKeyframeSpec, ...]
    ) -> tuple[OrientationKeyframeSpec, ...]:
        times = [keyframe.time_s for keyframe in value]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError("keyframe time_s values must be strictly increasing")
        return value


class AlignMotionOrientationSpec(FrozenStrictModel):
    type: Literal["align_motion"] = "align_motion"
    allow_pitch: bool = True
    smoothing_time_s: FiniteFloat = Field(default=0.0, ge=0.0)
    yaw_offset_deg: FiniteFloat = 0.0
    pitch_offset_deg: FiniteFloat = 0.0
    roll_offset_deg: FiniteFloat = 0.0
    max_yaw_rate_deg_s: FiniteFloat | None = Field(default=None, gt=0.0)
    max_pitch_rate_deg_s: FiniteFloat | None = Field(default=None, gt=0.0)


class LookAtOrientationSpec(FrozenStrictModel):
    type: Literal["look_at"] = "look_at"
    actor: str | None = None
    point_m: Vector3 | None = None
    allow_pitch: bool = True
    smoothing_time_s: FiniteFloat = Field(default=0.0, ge=0.0)
    max_yaw_rate_deg_s: FiniteFloat | None = Field(default=None, gt=0.0)
    max_pitch_rate_deg_s: FiniteFloat | None = Field(default=None, gt=0.0)
    yaw_offset_deg: FiniteFloat = 0.0
    pitch_offset_deg: FiniteFloat = 0.0
    roll_offset_deg: FiniteFloat = 0.0
    yaw_limits_deg: Range2 | None = None
    pitch_limits_deg: Range2 | None = None

    @field_validator("actor")
    @classmethod
    def validate_actor_name(cls, value: str | None) -> str | None:
        return None if value is None else _non_empty(value, label="actor")

    @field_validator("yaw_limits_deg", "pitch_limits_deg")
    @classmethod
    def validate_limits(cls, value: Range2 | None, info) -> Range2 | None:
        if value is None:
            return None
        return _validate_ordered_range(value, label=info.field_name)

    @model_validator(mode="after")
    def validate_target(self) -> "LookAtOrientationSpec":
        if (self.actor is None) == (self.point_m is None):
            raise ValueError("look_at requires exactly one of actor or point_m")
        return self


class SpinOrientationSpec(FrozenStrictModel):
    type: Literal["spin"] = "spin"
    axis: Literal["yaw", "pitch", "roll"] = "yaw"
    rate_deg_s: FiniteFloat
    yaw_deg: FiniteFloat = 0.0
    pitch_deg: FiniteFloat = 0.0
    roll_deg: FiniteFloat = 0.0


class RandomOrientationSpec(FrozenStrictModel):
    type: Literal["random"] = "random"
    seed: RandomSeed
    yaw_range_deg: Range2 = (-180.0, 180.0)
    pitch_range_deg: Range2 = (-90.0, 90.0)
    roll_range_deg: Range2 = (-180.0, 180.0)
    update_interval_s: FiniteFloat | None = Field(default=None, gt=0.0)

    @field_validator("yaw_range_deg", "pitch_range_deg", "roll_range_deg")
    @classmethod
    def validate_ranges(cls, value: Range2, info) -> Range2:
        return _validate_ordered_range(value, label=info.field_name)


OrientationSpec: TypeAlias = Annotated[
    FixedOrientationSpec
    | KeyframesOrientationSpec
    | AlignMotionOrientationSpec
    | LookAtOrientationSpec
    | SpinOrientationSpec
    | RandomOrientationSpec,
    Field(discriminator="type"),
]


def _fixed_orientation() -> FixedOrientationSpec:
    return FixedOrientationSpec()


# ---------------------------------------------------------------------------
# Target assets, actors, and groups
# ---------------------------------------------------------------------------
class CatalogAssetSpec(FrozenStrictModel):
    source: Literal["catalog"] = "catalog"
    id: str
    material_type: str = "glass"
    scale: FiniteFloat = Field(default=1.0, gt=0.0)
    switch_meshes: bool = True
    mesh_end_behavior: Literal["loop", "hold_last"] = "loop"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        asset_id = _non_empty(value, label="id")
        segments = asset_id.split("/")
        if (
            "\\" in asset_id
            or PurePosixPath(asset_id).is_absolute()
            or PureWindowsPath(asset_id).is_absolute()
            or any(segment in {"", ".", ".."} for segment in segments)
        ):
            raise ValueError("catalog id must be a relative forward-slash path")
        return asset_id

    @field_validator("material_type")
    @classmethod
    def validate_material_type(cls, value: str) -> str:
        return _non_empty(value, label="material_type")


class FileAssetSpec(FrozenStrictModel):
    source: Literal["file"] = "file"
    path: str
    material_type: str = "glass"
    scale: FiniteFloat = Field(default=1.0, gt=0.0)

    @field_validator("path", "material_type")
    @classmethod
    def validate_string(cls, value: str, info) -> str:
        return _non_empty(value, label=info.field_name)


class DirectoryAssetSpec(FrozenStrictModel):
    source: Literal["directory"] = "directory"
    path: str
    pattern: str = "*.ply"
    material_type: str = "glass"
    scale: FiniteFloat = Field(default=1.0, gt=0.0)
    switch_meshes: bool = True
    start_index: int = Field(default=0, ge=0)
    frame_stride: int = Field(default=1, ge=1)
    mesh_end_behavior: Literal["loop", "hold_last"] = "loop"

    @field_validator("path", "pattern", "material_type")
    @classmethod
    def validate_string(cls, value: str, info) -> str:
        return _non_empty(value, label=info.field_name)


TargetAssetSpec: TypeAlias = Annotated[
    CatalogAssetSpec | FileAssetSpec | DirectoryAssetSpec,
    Field(discriminator="source"),
]


class ActorSpec(FrozenStrictModel):
    """Common serialized actor fields; role is supplied by the container."""

    role: ClassVar[ActorRole]
    name: str
    mobility: MobilitySpec
    orientation: OrientationSpec = Field(default_factory=_fixed_orientation)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _non_empty(value, label="actor name")


class TxActorSpec(ActorSpec):
    role: ClassVar[ActorRole] = ActorRole.TX
    power_dbm: FiniteFloat | None = None


class RxActorSpec(ActorSpec):
    role: ClassVar[ActorRole] = ActorRole.RX


class TargetActorSpec(ActorSpec):
    role: ClassVar[ActorRole] = ActorRole.TARGET
    asset: TargetAssetSpec


class ActorsSpec(FrozenStrictModel):
    tx: tuple[TxActorSpec, ...] = ()
    rx: tuple[RxActorSpec, ...] = ()
    targets: tuple[TargetActorSpec, ...] = ()

    def all(self) -> tuple[ActorSpec, ...]:
        """Return all actors in stable role order."""
        return (*self.tx, *self.rx, *self.targets)


class GroupDeviationSpec(FrozenStrictModel):
    max_right_m: FiniteFloat = Field(default=0.0, ge=0.0)
    max_forward_m: FiniteFloat = Field(default=0.0, ge=0.0)
    max_up_m: FiniteFloat = Field(default=0.0, ge=0.0)
    seed: RandomSeed

    @model_validator(mode="after")
    def validate_nonzero_bound(self) -> "GroupDeviationSpec":
        if self.max_right_m == self.max_forward_m == self.max_up_m == 0.0:
            raise ValueError("group deviation must have at least one positive bound")
        return self


class GroupSpec(FrozenStrictModel):
    name: str
    mobility: StandaloneMobilitySpec
    deviation: GroupDeviationSpec | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _non_empty(value, label="group name")


__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "MAX_RANDOM_SEED",
    "ActorRole",
    "ActorSpec",
    "ActorsSpec",
    "AlignMotionOrientationSpec",
    "CatalogAssetSpec",
    "CircularMobilitySpec",
    "ConstantSpeedTraversalSpec",
    "DirectoryAssetSpec",
    "Figure8MobilitySpec",
    "FileAssetSpec",
    "FitDurationTraversalSpec",
    "FixedOrientationSpec",
    "GaussMarkovMobilitySpec",
    "GridScanMobilitySpec",
    "GroupDeviationSpec",
    "GroupMemberMobilitySpec",
    "GroupOffsetSpec",
    "GroupSpec",
    "KeyframesOrientationSpec",
    "LinearMobilitySpec",
    "LookAtOrientationSpec",
    "ManhattanGridMobilitySpec",
    "MeshSequenceMobilitySpec",
    "MobilitySpec",
    "NetworkRouteMobilitySpec",
    "OrientationKeyframeSpec",
    "OrientationSpec",
    "OscillatingMobilitySpec",
    "PendulumMobilitySpec",
    "RandomOrientationSpec",
    "RandomSeed",
    "RandomSamplingMobilitySpec",
    "RandomWaypointMobilitySpec",
    "RxActorSpec",
    "SampledMobilitySpec",
    "SpinOrientationSpec",
    "SpiralMobilitySpec",
    "StandaloneMobilitySpec",
    "StationaryMobilitySpec",
    "SurveyMobilitySpec",
    "TargetActorSpec",
    "TargetAssetSpec",
    "TimelineSpec",
    "TraversalSpec",
    "TxActorSpec",
    "WaypointMobilitySpec",
]
