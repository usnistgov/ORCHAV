"""Mobility metadata and position-preserving authoring helpers.

The shared actor schema owns every serialized mobility field.  This module
adds only authoring metadata and pure transformations over those immutable
Pydantic values; it deliberately has no Qt, renderer, persistence, or
workspace dependencies.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from pydantic import TypeAdapter

from shared.scenarios.actors import (
    ActorRole,
    CircularMobilitySpec,
    Figure8MobilitySpec,
    FitDurationTraversalSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupMemberMobilitySpec,
    LinearMobilitySpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    MobilitySpec,
    NetworkRouteMobilitySpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    Vector3,
    WaypointMobilitySpec,
)

from .model_capabilities import SUPPORTED_AUTHORING_MOBILITY_TYPES

MobilityModel: TypeAlias = MobilitySpec
MobilityModelType: TypeAlias = (
    type[StationaryMobilitySpec]
    | type[LinearMobilitySpec]
    | type[WaypointMobilitySpec]
    | type[CircularMobilitySpec]
    | type[SurveyMobilitySpec]
    | type[GridScanMobilitySpec]
    | type[OscillatingMobilitySpec]
    | type[PendulumMobilitySpec]
    | type[Figure8MobilitySpec]
    | type[SpiralMobilitySpec]
    | type[RandomSamplingMobilitySpec]
    | type[GaussMarkovMobilitySpec]
    | type[RandomWaypointMobilitySpec]
    | type[ManhattanGridMobilitySpec]
    | type[NetworkRouteMobilitySpec]
    | type[MeshSequenceMobilitySpec]
    | type[GroupMemberMobilitySpec]
)


class MobilityKind(str, Enum):
    """Canonical mobility discriminators."""

    STATIONARY = "stationary"
    LINEAR = "linear"
    WAYPOINT = "waypoint"
    CIRCULAR = "circular"
    SURVEY = "survey"
    GRID_SCAN = "grid_scan"
    OSCILLATING = "oscillating"
    PENDULUM = "pendulum"
    FIGURE8 = "figure8"
    SPIRAL = "spiral"
    RANDOM_SAMPLING = "random_sampling"
    GAUSS_MARKOV = "gauss_markov"
    RANDOM_WAYPOINT = "random_waypoint"
    MANHATTAN_GRID = "manhattan_grid"
    NETWORK_ROUTE = "network_route"
    MESH_SEQUENCE = "mesh_sequence"
    GROUP_MEMBER = "group_member"


class MobilityFamily(str, Enum):
    """Editor-facing families that share spatial interaction semantics."""

    FIXED = "fixed"
    CONTROL_POINT_PATH = "control_point_path"
    PARAMETRIC = "parametric"
    SPATIAL_SAMPLING = "spatial_sampling"
    STOCHASTIC = "stochastic"
    MAP_AWARE = "map_aware"
    RESOURCE_BACKED = "resource_backed"
    RELATIONAL = "relational"


class MobilityContext(str, Enum):
    """External state required to create or translate a mobility model."""

    NONE = "none"
    GROUP = "group"
    NETWORK_GRAPH = "network_graph"
    POSITION_SEQUENCE = "position_sequence"


_ALL_ACTOR_ROLES = frozenset(ActorRole)
_TARGET_ONLY = frozenset((ActorRole.TARGET,))


@dataclass(frozen=True, slots=True)
class MobilityModelDescriptor:
    """Immutable metadata for one canonical mobility discriminator."""

    kind: MobilityKind
    model_type: MobilityModelType
    label: str
    family: MobilityFamily
    context: MobilityContext = MobilityContext.NONE
    allowed_roles: frozenset[ActorRole] = _ALL_ACTOR_ROLES

    def __post_init__(self) -> None:
        schema_kind = self.model_type.model_fields["type"].default
        if schema_kind != self.kind.value:
            raise ValueError(
                f"{self.model_type.__name__} uses {schema_kind!r}, not {self.kind.value!r}"
            )
        if not self.label.strip():
            raise ValueError("mobility model labels must not be empty")
        if not self.allowed_roles:
            raise ValueError("mobility models must allow at least one actor role")

    @property
    def requires_context(self) -> bool:
        """Return whether construction needs state outside an actor mobility."""

        return self.context is not MobilityContext.NONE

    @property
    def required_context(self) -> MobilityContext | None:
        """Return the required context, or ``None`` for self-contained models."""

        return self.context if self.requires_context else None

    @property
    def roles(self) -> frozenset[ActorRole]:
        """Return the schema roles for which this model is valid."""

        return self.allowed_roles

    def allows_role(self, role: ActorRole | str) -> bool:
        """Return whether an actor role may use this mobility model."""

        return ActorRole(role) in self.allowed_roles


class MobilityNeedsContextError(ValueError):
    """Raised when a pure mobility edit cannot resolve required external state."""

    def __init__(
        self,
        kind: MobilityKind | str,
        context: MobilityContext,
        *,
        operation: str,
    ) -> None:
        normalized_kind = _normalize_kind(kind)
        if context is MobilityContext.NONE:
            raise ValueError("a needs-context error requires non-empty context")
        self.kind = normalized_kind
        self.context = context
        self.required_context = context
        self.operation = str(operation)
        super().__init__(
            f"{self.operation} {normalized_kind.value!r} mobility requires "
            f"{context.value} context"
        )


def _descriptor(
    kind: MobilityKind,
    model_type: MobilityModelType,
    label: str,
    family: MobilityFamily,
    *,
    context: MobilityContext = MobilityContext.NONE,
    allowed_roles: frozenset[ActorRole] = _ALL_ACTOR_ROLES,
) -> MobilityModelDescriptor:
    return MobilityModelDescriptor(
        kind=kind,
        model_type=model_type,
        label=label,
        family=family,
        context=context,
        allowed_roles=allowed_roles,
    )


MOBILITY_MODELS: Mapping[MobilityKind, MobilityModelDescriptor] = MappingProxyType(
    {
        MobilityKind.STATIONARY: _descriptor(
            MobilityKind.STATIONARY,
            StationaryMobilitySpec,
            "Stationary",
            MobilityFamily.FIXED,
        ),
        MobilityKind.LINEAR: _descriptor(
            MobilityKind.LINEAR,
            LinearMobilitySpec,
            "Linear",
            MobilityFamily.CONTROL_POINT_PATH,
        ),
        MobilityKind.WAYPOINT: _descriptor(
            MobilityKind.WAYPOINT,
            WaypointMobilitySpec,
            "Waypoint",
            MobilityFamily.CONTROL_POINT_PATH,
        ),
        MobilityKind.CIRCULAR: _descriptor(
            MobilityKind.CIRCULAR,
            CircularMobilitySpec,
            "Circular",
            MobilityFamily.PARAMETRIC,
        ),
        MobilityKind.SURVEY: _descriptor(
            MobilityKind.SURVEY,
            SurveyMobilitySpec,
            "Survey",
            MobilityFamily.CONTROL_POINT_PATH,
        ),
        MobilityKind.GRID_SCAN: _descriptor(
            MobilityKind.GRID_SCAN,
            GridScanMobilitySpec,
            "Grid Scan",
            MobilityFamily.SPATIAL_SAMPLING,
        ),
        MobilityKind.OSCILLATING: _descriptor(
            MobilityKind.OSCILLATING,
            OscillatingMobilitySpec,
            "Oscillating",
            MobilityFamily.PARAMETRIC,
        ),
        MobilityKind.PENDULUM: _descriptor(
            MobilityKind.PENDULUM,
            PendulumMobilitySpec,
            "Pendulum",
            MobilityFamily.PARAMETRIC,
        ),
        MobilityKind.FIGURE8: _descriptor(
            MobilityKind.FIGURE8,
            Figure8MobilitySpec,
            "Figure Eight",
            MobilityFamily.PARAMETRIC,
        ),
        MobilityKind.SPIRAL: _descriptor(
            MobilityKind.SPIRAL,
            SpiralMobilitySpec,
            "Spiral",
            MobilityFamily.PARAMETRIC,
        ),
        MobilityKind.RANDOM_SAMPLING: _descriptor(
            MobilityKind.RANDOM_SAMPLING,
            RandomSamplingMobilitySpec,
            "Random Sampling",
            MobilityFamily.SPATIAL_SAMPLING,
        ),
        MobilityKind.GAUSS_MARKOV: _descriptor(
            MobilityKind.GAUSS_MARKOV,
            GaussMarkovMobilitySpec,
            "Gauss-Markov",
            MobilityFamily.STOCHASTIC,
        ),
        MobilityKind.RANDOM_WAYPOINT: _descriptor(
            MobilityKind.RANDOM_WAYPOINT,
            RandomWaypointMobilitySpec,
            "Random Waypoint",
            MobilityFamily.STOCHASTIC,
        ),
        MobilityKind.MANHATTAN_GRID: _descriptor(
            MobilityKind.MANHATTAN_GRID,
            ManhattanGridMobilitySpec,
            "Manhattan Grid",
            MobilityFamily.STOCHASTIC,
        ),
        MobilityKind.NETWORK_ROUTE: _descriptor(
            MobilityKind.NETWORK_ROUTE,
            NetworkRouteMobilitySpec,
            "Network Route",
            MobilityFamily.MAP_AWARE,
            context=MobilityContext.NETWORK_GRAPH,
        ),
        MobilityKind.MESH_SEQUENCE: _descriptor(
            MobilityKind.MESH_SEQUENCE,
            MeshSequenceMobilitySpec,
            "Mesh Sequence",
            MobilityFamily.RESOURCE_BACKED,
            context=MobilityContext.POSITION_SEQUENCE,
            allowed_roles=_TARGET_ONLY,
        ),
        MobilityKind.GROUP_MEMBER: _descriptor(
            MobilityKind.GROUP_MEMBER,
            GroupMemberMobilitySpec,
            "Group Formation Member",
            MobilityFamily.RELATIONAL,
            context=MobilityContext.GROUP,
        ),
    }
)

MOBILITY_MODEL_TYPES: Mapping[MobilityKind, MobilityModelType] = MappingProxyType(
    {kind: descriptor.model_type for kind, descriptor in MOBILITY_MODELS.items()}
)
MOBILITY_MODEL_LABELS: Mapping[MobilityKind, str] = MappingProxyType(
    {kind: descriptor.label for kind, descriptor in MOBILITY_MODELS.items()}
)


_registry_discriminators = frozenset(kind.value for kind in MOBILITY_MODELS)
if _registry_discriminators != SUPPORTED_AUTHORING_MOBILITY_TYPES:
    raise RuntimeError(
        "editable mobility registry does not match the authoring capability manifest "
        f"(registry={sorted(_registry_discriminators)}, "
        f"editable={sorted(SUPPORTED_AUTHORING_MOBILITY_TYPES)})"
    )


_VECTOR3_ADAPTER = TypeAdapter(Vector3)
_DEFAULT_EXTENT_M = 10.0
_DEFAULT_RADIUS_M = 5.0
_DEFAULT_GRID_VERTICAL_EXTENT_M = 2.0
_DEFAULT_RANDOM_WAYPOINT_HALF_EXTENT_M = 2.0
_DEFAULT_RANDOM_WAYPOINT_SPEED_RANGE_MPS = (4.0, 6.0)
_DEFAULT_MANHATTAN_BLOCK_M = 2.0


def _normalize_kind(kind: MobilityKind | str) -> MobilityKind:
    if isinstance(kind, MobilityKind):
        return kind
    normalized = str(kind).strip().lower()
    try:
        return MobilityKind(normalized)
    except ValueError as exc:
        raise ValueError(f"unsupported mobility kind: {kind!r}") from exc


def _vector3(value: object) -> Vector3:
    return cast(Vector3, _VECTOR3_ADAPTER.validate_python(value))


def _duration(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("duration_s must be finite and non-negative")
    try:
        duration = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_s must be finite and non-negative") from exc
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    return duration


def _seed(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        return int(operator.index(cast(Any, value)))
    except TypeError as exc:
        raise ValueError("seed must be an integer") from exc


def _full_traversal() -> FitDurationTraversalSpec:
    """Return the canonical traversal for newly authored complete paths."""

    return FitDurationTraversalSpec()


def mobility_kind(mobility: MobilityModel) -> MobilityKind:
    """Return the typed canonical discriminator for a mobility value."""

    return _normalize_kind(mobility.type)


def default_mobility(
    kind: MobilityKind | str,
    anchor: Vector3,
    duration_s: float,
    *,
    seed: int = 0,
) -> MobilityModel:
    """Create a canonical mobility whose time-zero position is ``anchor``.

    Relationship and resource-backed models cannot be fabricated from a world
    position alone and raise :class:`MobilityNeedsContextError`.
    """

    normalized_kind = _normalize_kind(kind)
    descriptor = MOBILITY_MODELS[normalized_kind]
    if descriptor.requires_context:
        assert descriptor.required_context is not None
        raise MobilityNeedsContextError(
            normalized_kind,
            descriptor.required_context,
            operation="creating",
        )

    x, y, z = _vector3(anchor)
    duration = _duration(duration_s)
    normalized_seed = _seed(seed)
    cycle_frequency_hz = 1.0 / duration if duration > 0.0 else 1.0
    traversal = _full_traversal()

    if normalized_kind is MobilityKind.STATIONARY:
        return StationaryMobilitySpec(position_m=(x, y, z))
    if normalized_kind is MobilityKind.LINEAR:
        return LinearMobilitySpec(
            start_m=(x, y, z),
            end_m=(x + _DEFAULT_EXTENT_M, y, z),
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.WAYPOINT:
        return WaypointMobilitySpec(
            points_m=(
                (x, y, z),
                (x + _DEFAULT_EXTENT_M / 2.0, y, z),
                (
                    x + _DEFAULT_EXTENT_M / 2.0,
                    y + _DEFAULT_EXTENT_M / 2.0,
                    z,
                ),
            ),
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.CIRCULAR:
        return CircularMobilitySpec(
            center_m=(x - _DEFAULT_RADIUS_M, y, z),
            radius_m=_DEFAULT_RADIUS_M,
            start_angle_deg=0.0,
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.SURVEY:
        return SurveyMobilitySpec(
            origin_m=(x, y, z),
            width_m=_DEFAULT_EXTENT_M,
            height_m=_DEFAULT_EXTENT_M,
            row_spacing_m=_DEFAULT_EXTENT_M / 4.0,
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.GRID_SCAN:
        return GridScanMobilitySpec(
            x_bounds_m=(x, x + _DEFAULT_EXTENT_M),
            y_bounds_m=(y, y + _DEFAULT_EXTENT_M),
            z_bounds_m=(z, z + _DEFAULT_GRID_VERTICAL_EXTENT_M),
            x_steps=3,
            y_steps=3,
            z_steps=2,
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.OSCILLATING:
        return OscillatingMobilitySpec(
            center_m=(x, y, z),
            axis=(1.0, 0.0, 0.0),
            amplitude_m=_DEFAULT_RADIUS_M,
            frequency_hz=cycle_frequency_hz,
            phase_deg=0.0,
        )
    if normalized_kind is MobilityKind.PENDULUM:
        return PendulumMobilitySpec(
            pivot_m=(x, y, z + _DEFAULT_RADIUS_M),
            length_m=_DEFAULT_RADIUS_M,
            max_angle_deg=30.0,
            frequency_hz=cycle_frequency_hz,
            plane="xz",
            phase_deg=0.0,
        )
    if normalized_kind is MobilityKind.FIGURE8:
        return Figure8MobilitySpec(
            center_m=(x - _DEFAULT_RADIUS_M, y, z),
            size_m=_DEFAULT_RADIUS_M,
            plane="xy",
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.SPIRAL:
        return SpiralMobilitySpec(
            center_m=(x - _DEFAULT_RADIUS_M, y, z),
            radius_m=_DEFAULT_RADIUS_M,
            start_altitude_m=z,
            end_altitude_m=z + _DEFAULT_EXTENT_M,
            turns=1.0,
            start_angle_deg=0.0,
            traversal=traversal,
        )
    if normalized_kind is MobilityKind.RANDOM_SAMPLING:
        return RandomSamplingMobilitySpec(
            x_bounds_m=(x - _DEFAULT_EXTENT_M, x + _DEFAULT_EXTENT_M),
            y_bounds_m=(y - _DEFAULT_EXTENT_M, y + _DEFAULT_EXTENT_M),
            z_bounds_m=(z, z),
            initial_position_m=(x, y, z),
            seed=normalized_seed,
        )
    if normalized_kind is MobilityKind.GAUSS_MARKOV:
        return GaussMarkovMobilitySpec(
            initial_position_m=(x, y, z),
            x_bounds_m=(x - _DEFAULT_EXTENT_M, x + _DEFAULT_EXTENT_M),
            y_bounds_m=(y - _DEFAULT_EXTENT_M, y + _DEFAULT_EXTENT_M),
            z_bounds_m=(z, z),
            alpha=0.65,
            mean_speed_mps=2.0,
            speed_std_mps=0.4,
            direction_std_deg=40.0,
            seed=normalized_seed,
        )
    if normalized_kind is MobilityKind.RANDOM_WAYPOINT:
        return RandomWaypointMobilitySpec(
            initial_position_m=(x, y, z),
            x_bounds_m=(
                x - _DEFAULT_RANDOM_WAYPOINT_HALF_EXTENT_M,
                x + _DEFAULT_RANDOM_WAYPOINT_HALF_EXTENT_M,
            ),
            y_bounds_m=(
                y - _DEFAULT_RANDOM_WAYPOINT_HALF_EXTENT_M,
                y + _DEFAULT_RANDOM_WAYPOINT_HALF_EXTENT_M,
            ),
            z_bounds_m=(z, z),
            speed_range_mps=_DEFAULT_RANDOM_WAYPOINT_SPEED_RANGE_MPS,
            seed=normalized_seed,
        )
    if normalized_kind is MobilityKind.MANHATTAN_GRID:
        return ManhattanGridMobilitySpec(
            origin_xy_m=(x, y),
            block_size_m=_DEFAULT_MANHATTAN_BLOCK_M,
            grid_width=3,
            grid_height=3,
            altitude_m=z,
            speed_range_mps=(1.0, 3.0),
            seed=normalized_seed,
        )

    raise RuntimeError(f"no context-free default registered for {normalized_kind.value!r}")


def _shift(point: Vector3, offset: Vector3) -> Vector3:
    return (
        point[0] + offset[0],
        point[1] + offset[1],
        point[2] + offset[2],
    )


def _shift_range(bounds: tuple[float, float], delta: float) -> tuple[float, float]:
    return bounds[0] + delta, bounds[1] + delta


def translate_mobility(mobility: MobilityModel, offset: Vector3) -> MobilityModel:
    """Translate every absolute spatial field by one world-space offset.

    The complete Pydantic model is copied, so traversal, interpolation,
    stochastic parameters, and every other non-spatial field remain intact.
    Models whose world positions live in a group or external resource require
    resolution context and raise :class:`MobilityNeedsContextError`.
    """

    kind = mobility_kind(mobility)
    descriptor = MOBILITY_MODELS[kind]
    if descriptor.requires_context:
        assert descriptor.required_context is not None
        raise MobilityNeedsContextError(
            kind,
            descriptor.required_context,
            operation="translating",
        )

    delta = _vector3(offset)
    dx, dy, dz = delta
    updates: dict[str, object]
    if isinstance(mobility, StationaryMobilitySpec):
        updates = {"position_m": _shift(mobility.position_m, delta)}
    elif isinstance(mobility, LinearMobilitySpec):
        updates = {
            "start_m": _shift(mobility.start_m, delta),
            "end_m": _shift(mobility.end_m, delta),
        }
    elif isinstance(mobility, WaypointMobilitySpec):
        updates = {"points_m": tuple(_shift(point, delta) for point in mobility.points_m)}
    elif isinstance(mobility, CircularMobilitySpec):
        updates = {"center_m": _shift(mobility.center_m, delta)}
    elif isinstance(mobility, SurveyMobilitySpec):
        updates = {"origin_m": _shift(mobility.origin_m, delta)}
    elif isinstance(mobility, GridScanMobilitySpec):
        updates = {
            "x_bounds_m": _shift_range(mobility.x_bounds_m, dx),
            "y_bounds_m": _shift_range(mobility.y_bounds_m, dy),
            "z_bounds_m": _shift_range(mobility.z_bounds_m, dz),
        }
    elif isinstance(mobility, OscillatingMobilitySpec):
        updates = {"center_m": _shift(mobility.center_m, delta)}
    elif isinstance(mobility, PendulumMobilitySpec):
        updates = {"pivot_m": _shift(mobility.pivot_m, delta)}
    elif isinstance(mobility, Figure8MobilitySpec):
        updates = {"center_m": _shift(mobility.center_m, delta)}
    elif isinstance(mobility, SpiralMobilitySpec):
        updates = {
            "center_m": _shift(mobility.center_m, delta),
            "start_altitude_m": mobility.start_altitude_m + dz,
            "end_altitude_m": mobility.end_altitude_m + dz,
        }
    elif isinstance(mobility, RandomSamplingMobilitySpec):
        updates = {
            "x_bounds_m": _shift_range(mobility.x_bounds_m, dx),
            "y_bounds_m": _shift_range(mobility.y_bounds_m, dy),
            "z_bounds_m": _shift_range(mobility.z_bounds_m, dz),
            "initial_position_m": (
                _shift(mobility.initial_position_m, delta)
                if mobility.initial_position_m is not None
                else None
            ),
        }
    elif isinstance(mobility, GaussMarkovMobilitySpec):
        updates = {
            "initial_position_m": _shift(mobility.initial_position_m, delta),
            "x_bounds_m": _shift_range(mobility.x_bounds_m, dx),
            "y_bounds_m": _shift_range(mobility.y_bounds_m, dy),
            "z_bounds_m": _shift_range(mobility.z_bounds_m, dz),
        }
    elif isinstance(mobility, RandomWaypointMobilitySpec):
        updates = {
            "initial_position_m": _shift(mobility.initial_position_m, delta),
            "x_bounds_m": _shift_range(mobility.x_bounds_m, dx),
            "y_bounds_m": _shift_range(mobility.y_bounds_m, dy),
            "z_bounds_m": _shift_range(mobility.z_bounds_m, dz),
        }
    elif isinstance(mobility, ManhattanGridMobilitySpec):
        updates = {
            "origin_xy_m": (
                mobility.origin_xy_m[0] + dx,
                mobility.origin_xy_m[1] + dy,
            ),
            "altitude_m": mobility.altitude_m + dz,
        }
    else:  # Registry completeness makes this a defensive programming error.
        raise RuntimeError(f"no translation registered for {kind.value!r}")

    return cast(MobilityModel, mobility.model_copy(update=updates))


def convert_mobility(
    mobility: MobilityModel,
    target_kind: MobilityKind | str,
    anchor: Vector3,
    *,
    duration_s: float,
    seed: int = 0,
) -> MobilityModel:
    """Convert to a canonical model while preserving the visible position.

    Conversion to the current kind is identity-preserving.  Every other
    context-free target is rebuilt with ``anchor`` as its canonical time-zero
    sample.  Group and resource targets report their typed context requirement.
    """

    normalized_target = _normalize_kind(target_kind)
    if mobility_kind(mobility) is normalized_target:
        return mobility
    return default_mobility(
        normalized_target,
        anchor,
        duration_s,
        seed=seed,
    )


__all__ = [
    "MOBILITY_MODELS",
    "MOBILITY_MODEL_LABELS",
    "MOBILITY_MODEL_TYPES",
    "MobilityContext",
    "MobilityFamily",
    "MobilityKind",
    "MobilityModel",
    "MobilityModelDescriptor",
    "MobilityModelType",
    "MobilityNeedsContextError",
    "convert_mobility",
    "default_mobility",
    "mobility_kind",
    "translate_mobility",
]
