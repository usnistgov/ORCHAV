"""Declarative direct-manipulation controls for authored mobility models.

The control rig is renderer and UI neutral. It describes the
semantic handles and decorative guides for a mobility value, then applies a
drag through a named domain operation.  A viewport can therefore render and
prioritize controls without owning mobility mathematics. Model adapters extend
the same interaction contract without adding model-specific viewport modes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from .domain import Circular, Linear, Mobility, Stationary, Vector3, Waypoint, vector3

_STABLE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class MobilityControlConstraint(str, Enum):
    """World-space constraint expected while dragging a semantic handle."""

    FREE = "free"
    PLANE = "plane"
    RADIAL = "radial"
    ANGULAR = "angular"


class MobilityControlOperation(str, Enum):
    """Domain update performed by one semantic handle."""

    SET_POSITION = "set_position"
    SET_START = "set_start"
    SET_END = "set_end"
    SET_WAYPOINT = "set_waypoint"
    SET_CENTER = "set_center"
    SET_RADIUS = "set_radius"
    SET_START_ANGLE = "set_start_angle"


class MobilityControlGlyph(str, Enum):
    """Renderer-neutral glyph hint for distinguishing control semantics."""

    DISC = "disc"
    DIAMOND = "diamond"
    SQUARE = "square"
    CROSS = "cross"
    RING = "ring"
    ARROW = "arrow"


class MobilityGuideStyle(str, Enum):
    """Presentation hint for a non-interactive control-rig guide."""

    SEGMENT = "segment"
    RADIUS = "radius"
    ANGLE = "angle"


def _stable_key(value: object, *, subject: str) -> str:
    key = str(value)
    if not _STABLE_KEY_PATTERN.fullmatch(key):
        raise ValueError(
            f"{subject} key must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )
    return key


@dataclass(frozen=True, slots=True)
class MobilityControlDescriptor:
    """One stable semantic control rendered above decorative trajectory data."""

    key: str
    position: Vector3
    label: str
    tooltip: str
    operation: MobilityControlOperation
    constraint: MobilityControlConstraint = MobilityControlConstraint.FREE
    glyph: MobilityControlGlyph = MobilityControlGlyph.DISC
    point_size: float = 13.0
    ordinal: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _stable_key(self.key, subject="control"))
        object.__setattr__(self, "position", vector3(self.position))
        object.__setattr__(self, "operation", MobilityControlOperation(self.operation))
        object.__setattr__(self, "constraint", MobilityControlConstraint(self.constraint))
        object.__setattr__(self, "glyph", MobilityControlGlyph(self.glyph))
        object.__setattr__(self, "point_size", float(self.point_size))
        if not self.label.strip():
            raise ValueError("control label must not be empty")
        if not self.tooltip.strip():
            raise ValueError("control tooltip must not be empty")
        if not all(math.isfinite(component) for component in self.position):
            raise ValueError("control position must contain only finite values")
        if not math.isfinite(self.point_size) or self.point_size <= 0.0:
            raise ValueError("control point size must be finite and positive")
        if self.operation is MobilityControlOperation.SET_WAYPOINT:
            if (
                self.ordinal is None
                or isinstance(self.ordinal, bool)
                or not isinstance(self.ordinal, int)
                or self.ordinal < 0
            ):
                raise ValueError("waypoint controls require a non-negative ordinal")
        elif self.ordinal is not None:
            raise ValueError("only waypoint controls may have an ordinal")


@dataclass(frozen=True, slots=True)
class MobilityControlGuide:
    """One decorative, non-interactive guide between semantic controls."""

    key: str
    points: tuple[Vector3, ...]
    style: MobilityGuideStyle = MobilityGuideStyle.SEGMENT
    line_width: float = 1.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _stable_key(self.key, subject="guide"))
        object.__setattr__(self, "points", tuple(vector3(point) for point in self.points))
        object.__setattr__(self, "style", MobilityGuideStyle(self.style))
        object.__setattr__(self, "line_width", float(self.line_width))
        if len(self.points) < 2:
            raise ValueError("a mobility guide requires at least two points")
        if not all(math.isfinite(component) for point in self.points for component in point):
            raise ValueError("guide points must contain only finite values")
        if not math.isfinite(self.line_width) or self.line_width <= 0.0:
            raise ValueError("guide line width must be finite and positive")


@dataclass(frozen=True, slots=True)
class MobilityControlRig:
    """Complete declarative controls and guides for one mobility value."""

    mobility_kind: str
    controls: tuple[MobilityControlDescriptor, ...]
    guides: tuple[MobilityControlGuide, ...] = ()

    def __post_init__(self) -> None:
        kind = str(self.mobility_kind).strip()
        if not kind:
            raise ValueError("mobility kind must not be empty")
        object.__setattr__(self, "mobility_kind", kind)
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "guides", tuple(self.guides))
        control_keys = tuple(control.key for control in self.controls)
        guide_keys = tuple(guide.key for guide in self.guides)
        if len(control_keys) != len(set(control_keys)):
            raise ValueError("mobility control keys must be unique")
        if len(guide_keys) != len(set(guide_keys)):
            raise ValueError("mobility guide keys must be unique")

    def control(self, key: str) -> MobilityControlDescriptor:
        """Return the control with *key* or raise a precise lookup error."""

        normalized = _stable_key(key, subject="control")
        for control in self.controls:
            if control.key == normalized:
                return control
        raise KeyError(f"unknown {self.mobility_kind} mobility control: {normalized}")


DescribeRig: TypeAlias = Callable[[Mobility], MobilityControlRig]
UpdateRig: TypeAlias = Callable[[Mobility, MobilityControlDescriptor, Vector3], Mobility]


@dataclass(frozen=True, slots=True)
class MobilityControlRigAdapter:
    """Descriptor and updater pair registered for one mobility domain type."""

    mobility_type: type[object]
    describe: DescribeRig
    update: UpdateRig


def _validated_update(mobility: Mobility, **changes: object) -> Mobility:
    values = mobility.model_dump(mode="python")
    values.update(changes)
    return cast(Mobility, cast(Any, type(mobility)).model_validate(values))


def _stationary_rig(mobility: Mobility) -> MobilityControlRig:
    if not isinstance(mobility, Stationary):
        raise TypeError("stationary adapter received another mobility type")
    return MobilityControlRig(
        mobility.type,
        (
            MobilityControlDescriptor(
                key="position",
                position=mobility.position_m,
                label="Position",
                tooltip="Drag to move the stationary position.",
                operation=MobilityControlOperation.SET_POSITION,
                glyph=MobilityControlGlyph.DISC,
                point_size=14.0,
            ),
        ),
    )


def _linear_rig(mobility: Mobility) -> MobilityControlRig:
    if not isinstance(mobility, Linear):
        raise TypeError("linear adapter received another mobility type")
    return MobilityControlRig(
        mobility.type,
        (
            MobilityControlDescriptor(
                key="start",
                position=mobility.start_m,
                label="Start point",
                tooltip="Drag to change where the linear motion starts.",
                operation=MobilityControlOperation.SET_START,
                glyph=MobilityControlGlyph.DIAMOND,
                point_size=15.0,
            ),
            MobilityControlDescriptor(
                key="end",
                position=mobility.end_m,
                label="Arrival point",
                tooltip="Drag to change where the linear motion ends.",
                operation=MobilityControlOperation.SET_END,
                glyph=MobilityControlGlyph.SQUARE,
                point_size=15.0,
            ),
        ),
        (
            MobilityControlGuide(
                key="segment",
                points=(mobility.start_m, mobility.end_m),
            ),
        ),
    )


def _waypoint_rig(mobility: Mobility) -> MobilityControlRig:
    if not isinstance(mobility, Waypoint):
        raise TypeError("waypoint adapter received another mobility type")
    controls = tuple(
        MobilityControlDescriptor(
            key=f"waypoint_{index}",
            position=point,
            label=f"Waypoint {index + 1}",
            tooltip=f"Drag to move waypoint {index + 1}.",
            operation=MobilityControlOperation.SET_WAYPOINT,
            glyph=MobilityControlGlyph.DIAMOND,
            point_size=14.0,
            ordinal=index,
        )
        for index, point in enumerate(mobility.points_m)
    )
    guides = tuple(
        MobilityControlGuide(
            key=f"segment_{index}",
            points=(start, end),
        )
        for index, (start, end) in enumerate(zip(mobility.points_m, mobility.points_m[1:]))
    )
    return MobilityControlRig(mobility.type, controls, guides)


def _circular_rig(mobility: Mobility) -> MobilityControlRig:
    if not isinstance(mobility, Circular):
        raise TypeError("circular adapter received another mobility type")
    center_x, center_y, center_z = mobility.center_m
    start_angle = math.radians(mobility.start_angle_deg)
    angle_position = (
        center_x + mobility.radius_m * math.cos(start_angle),
        center_y + mobility.radius_m * math.sin(start_angle),
        center_z,
    )
    radius_angle = start_angle + math.pi / 2.0
    radius_position = (
        center_x + mobility.radius_m * math.cos(radius_angle),
        center_y + mobility.radius_m * math.sin(radius_angle),
        center_z,
    )
    return MobilityControlRig(
        mobility.type,
        (
            MobilityControlDescriptor(
                key="center",
                position=mobility.center_m,
                label="Circle center",
                tooltip="Drag to translate the circle without changing its radius or angle.",
                operation=MobilityControlOperation.SET_CENTER,
                constraint=MobilityControlConstraint.PLANE,
                glyph=MobilityControlGlyph.CROSS,
                point_size=15.0,
            ),
            MobilityControlDescriptor(
                key="radius",
                position=radius_position,
                label="Radius",
                tooltip="Drag toward or away from the center to change only the radius.",
                operation=MobilityControlOperation.SET_RADIUS,
                constraint=MobilityControlConstraint.RADIAL,
                glyph=MobilityControlGlyph.RING,
                point_size=16.0,
            ),
            MobilityControlDescriptor(
                key="start_angle",
                position=angle_position,
                label="Start angle",
                tooltip="Drag around the center to change only the start angle.",
                operation=MobilityControlOperation.SET_START_ANGLE,
                constraint=MobilityControlConstraint.ANGULAR,
                glyph=MobilityControlGlyph.ARROW,
                point_size=16.0,
            ),
        ),
        (
            MobilityControlGuide(
                key="radius_guide",
                points=(mobility.center_m, radius_position),
                style=MobilityGuideStyle.RADIUS,
            ),
            MobilityControlGuide(
                key="start_angle_guide",
                points=(mobility.center_m, angle_position),
                style=MobilityGuideStyle.ANGLE,
            ),
        ),
    )


def _update_stationary(
    mobility: Mobility,
    control: MobilityControlDescriptor,
    position: Vector3,
) -> Mobility:
    if not isinstance(mobility, Stationary):
        raise TypeError("stationary adapter received another mobility type")
    if control.operation is not MobilityControlOperation.SET_POSITION:
        raise ValueError(f"{control.key} is not a stationary mobility control")
    return Stationary(position_m=vector3(position))


def _update_linear(
    mobility: Mobility,
    control: MobilityControlDescriptor,
    position: Vector3,
) -> Mobility:
    if not isinstance(mobility, Linear):
        raise TypeError("linear adapter received another mobility type")
    if control.operation is MobilityControlOperation.SET_START:
        return _validated_update(mobility, start_m=vector3(position))
    if control.operation is MobilityControlOperation.SET_END:
        return _validated_update(mobility, end_m=vector3(position))
    raise ValueError(f"{control.key} is not a linear mobility control")


def _update_waypoint(
    mobility: Mobility,
    control: MobilityControlDescriptor,
    position: Vector3,
) -> Mobility:
    if not isinstance(mobility, Waypoint):
        raise TypeError("waypoint adapter received another mobility type")
    if control.operation is not MobilityControlOperation.SET_WAYPOINT or control.ordinal is None:
        raise ValueError(f"{control.key} is not a waypoint mobility control")
    if control.ordinal >= len(mobility.points_m):
        raise ValueError(f"unknown waypoint control ordinal: {control.ordinal}")
    points = list(mobility.points_m)
    points[control.ordinal] = vector3(position)
    return _validated_update(mobility, points_m=tuple(points))


def _update_circular(
    mobility: Mobility,
    control: MobilityControlDescriptor,
    position: Vector3,
) -> Mobility:
    if not isinstance(mobility, Circular):
        raise TypeError("circular adapter received another mobility type")
    if control.operation is MobilityControlOperation.SET_CENTER:
        return _validated_update(mobility, center_m=vector3(position))
    dx = position[0] - mobility.center_m[0]
    dy = position[1] - mobility.center_m[1]
    if control.operation is MobilityControlOperation.SET_RADIUS:
        axis_x = control.position[0] - mobility.center_m[0]
        axis_y = control.position[1] - mobility.center_m[1]
        axis_length = math.hypot(axis_x, axis_y)
        if axis_length <= 0.0:
            raise ValueError("radius control axis cannot coincide with the circle center")
        radius = (dx * axis_x + dy * axis_y) / axis_length
        if radius <= 0.0:
            raise ValueError("radius control cannot coincide with or pass beyond the circle center")
        return _validated_update(mobility, radius_m=radius)
    if control.operation is MobilityControlOperation.SET_START_ANGLE:
        if dx == 0.0 and dy == 0.0:
            raise ValueError("start-angle control cannot coincide with the circle center")
        return _validated_update(
            mobility,
            start_angle_deg=math.degrees(math.atan2(dy, dx)),
        )
    raise ValueError(f"{control.key} is not a circular mobility control")


_ADAPTERS: dict[type[object], MobilityControlRigAdapter] = {
    Stationary: MobilityControlRigAdapter(Stationary, _stationary_rig, _update_stationary),
    Linear: MobilityControlRigAdapter(Linear, _linear_rig, _update_linear),
    Waypoint: MobilityControlRigAdapter(Waypoint, _waypoint_rig, _update_waypoint),
    Circular: MobilityControlRigAdapter(Circular, _circular_rig, _update_circular),
}

MOBILITY_CONTROL_RIG_ADAPTERS: Mapping[type[object], MobilityControlRigAdapter] = MappingProxyType(
    _ADAPTERS
)


def _adapter_for(mobility: Mobility) -> MobilityControlRigAdapter:
    try:
        return MOBILITY_CONTROL_RIG_ADAPTERS[type(mobility)]
    except KeyError as exc:
        raise TypeError(f"unsupported mobility type: {type(mobility).__name__}") from exc


def mobility_control_rig(mobility: Mobility) -> MobilityControlRig:
    """Describe stable semantic handles and decorative guides for *mobility*."""

    return _adapter_for(mobility).describe(mobility)


def update_mobility_from_rig_control(
    mobility: Mobility,
    control: MobilityControlDescriptor | str,
    world_position: Vector3,
) -> Mobility:
    """Apply one named control drag without coupling unrelated parameters."""

    rig = mobility_control_rig(mobility)
    current_control = rig.control(
        control.key if isinstance(control, MobilityControlDescriptor) else control
    )
    return _adapter_for(mobility).update(mobility, current_control, vector3(world_position))
