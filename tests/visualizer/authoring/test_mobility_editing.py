"""Renderer-neutral mobility control and conversion behavior."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from shared.scenarios.actors import (
    CircularMobilitySpec,
    LinearMobilitySpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring.domain import (
    MobilityControl,
    MobilityControlKind,
    convert_mobility,
    mobility_controls,
    update_mobility_control,
)
from visualizer.src.authoring.mobility_control_rig import (
    MOBILITY_CONTROL_RIG_ADAPTERS,
    MobilityControlConstraint,
    MobilityControlDescriptor,
    MobilityControlGlyph,
    MobilityControlOperation,
    MobilityControlRig,
    mobility_control_rig,
    update_mobility_from_rig_control,
)


@pytest.mark.parametrize(
    ("mobility", "expected"),
    (
        (
            StationaryMobilitySpec(position_m=(1, 2, 3)),
            ((MobilityControlKind.POSITION, (1.0, 2.0, 3.0), None),),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            (
                (MobilityControlKind.START, (1.0, 2.0, 3.0), None),
                (MobilityControlKind.END, (4.0, 5.0, 6.0), None),
            ),
        ),
        (
            WaypointMobilitySpec(points_m=((1, 2, 3), (4, 5, 6))),
            (
                (MobilityControlKind.WAYPOINT, (1.0, 2.0, 3.0), 0),
                (MobilityControlKind.WAYPOINT, (4.0, 5.0, 6.0), 1),
            ),
        ),
        (
            CircularMobilitySpec(
                center_m=(1, 2, 3),
                radius_m=2.0,
                start_angle_deg=90.0,
            ),
            (
                (MobilityControlKind.CENTER, (1.0, 2.0, 3.0), None),
                (MobilityControlKind.RIM, (1.0, 4.0, 3.0), None),
            ),
        ),
    ),
)
def test_mobility_controls_describe_semantic_handles(mobility, expected) -> None:
    controls = mobility_controls(mobility)

    assert tuple(control.kind for control in controls) == tuple(item[0] for item in expected)
    assert tuple(control.ordinal for control in controls) == tuple(item[2] for item in expected)
    for control, item in zip(controls, expected):
        assert control.position == pytest.approx(item[1])


@pytest.mark.parametrize(
    ("mobility", "control_index", "position", "expected"),
    (
        (
            StationaryMobilitySpec(position_m=(1, 2, 3)),
            0,
            (7, 8, 9),
            StationaryMobilitySpec(position_m=(7, 8, 9)),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            0,
            (7, 8, 9),
            LinearMobilitySpec(start_m=(7, 8, 9), end_m=(4, 5, 6)),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            1,
            (7, 8, 9),
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(7, 8, 9)),
        ),
        (
            WaypointMobilitySpec(points_m=((1, 2, 3), (4, 5, 6))),
            1,
            (7, 8, 9),
            WaypointMobilitySpec(points_m=((1, 2, 3), (7, 8, 9))),
        ),
        (
            CircularMobilitySpec(
                center_m=(1, 2, 3),
                radius_m=2.0,
                start_angle_deg=30.0,
                clockwise=False,
            ),
            0,
            (7, 8, 9),
            CircularMobilitySpec(
                center_m=(7, 8, 9),
                radius_m=2.0,
                start_angle_deg=30.0,
                clockwise=False,
            ),
        ),
    ),
)
def test_update_mobility_control_changes_only_the_referenced_control(
    mobility, control_index, position, expected
) -> None:
    original = mobility
    updated = update_mobility_control(
        mobility,
        mobility_controls(mobility)[control_index],
        position,
    )

    assert updated == expected
    assert mobility is original
    assert mobility != updated


def test_dragging_circular_rim_updates_radius_and_start_angle() -> None:
    mobility = CircularMobilitySpec(
        center_m=(1, 2, 3),
        radius_m=2.0,
        start_angle_deg=30.0,
        clockwise=False,
    )

    updated = update_mobility_control(
        mobility,
        mobility_controls(mobility)[1],
        (4, 6, 99),
    )

    assert isinstance(updated, CircularMobilitySpec)
    assert updated.center_m == mobility.center_m
    assert updated.radius_m == pytest.approx(5.0)
    assert updated.start_angle_deg == pytest.approx(math.degrees(math.atan2(4.0, 3.0)))
    assert updated.clockwise is False


@pytest.mark.parametrize(
    ("mobility", "kind", "expected"),
    (
        (
            LinearMobilitySpec(start_m=(0, 0, 0), end_m=(1, 0, 0)),
            "stationary",
            StationaryMobilitySpec(position_m=(5, 5, 5)),
        ),
        (
            StationaryMobilitySpec(position_m=(0, 0, 0)),
            "linear",
            LinearMobilitySpec(start_m=(5, 5, 5), end_m=(15, 5, 5)),
        ),
        (
            StationaryMobilitySpec(position_m=(0, 0, 0)),
            "waypoint",
            WaypointMobilitySpec(points_m=((5, 5, 5), (10, 5, 5), (10, 10, 5))),
        ),
        (
            StationaryMobilitySpec(position_m=(0, 0, 0)),
            "circular",
            CircularMobilitySpec(
                center_m=(0, 5, 5),
                radius_m=5.0,
                start_angle_deg=0.0,
            ),
        ),
    ),
)
def test_convert_mobility_uses_position_preserving_defaults(mobility, kind, expected) -> None:
    converted = convert_mobility(mobility, kind, (5, 5, 5))

    assert converted == expected


def test_convert_mobility_keeps_same_kind_instance_and_circular_start_anchor() -> None:
    circular = CircularMobilitySpec(
        center_m=(1, 2, 3),
        radius_m=7.0,
        start_angle_deg=15.0,
        clockwise=False,
    )

    unchanged = convert_mobility(circular, "circular", (5, 5, 5))
    converted = convert_mobility(
        StationaryMobilitySpec(position_m=(0, 0, 0)),
        "circular",
        (5, 5, 5),
    )

    assert unchanged is circular
    assert mobility_controls(converted)[1].position == pytest.approx((5, 5, 5))


def test_controls_are_immutable_and_reject_invalid_ordinals() -> None:
    control = MobilityControl(MobilityControlKind.POSITION, (1, 2, 3))

    with pytest.raises(FrozenInstanceError):
        control.position = (4, 5, 6)  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-negative ordinal"):
        MobilityControl(MobilityControlKind.WAYPOINT, (1, 2, 3))
    with pytest.raises(ValueError, match="only waypoint"):
        MobilityControl(MobilityControlKind.START, (1, 2, 3), ordinal=0)


def test_update_rejects_controls_for_other_mobility_and_stale_waypoint_indices() -> None:
    with pytest.raises(ValueError, match="not a stationary"):
        update_mobility_control(
            StationaryMobilitySpec(position_m=(0, 0, 0)),
            MobilityControl(MobilityControlKind.START, (0, 0, 0)),
            (1, 1, 1),
        )
    with pytest.raises(ValueError, match="unknown waypoint control ordinal"):
        update_mobility_control(
            WaypointMobilitySpec(points_m=((0, 0, 0), (1, 0, 0))),
            MobilityControl(MobilityControlKind.WAYPOINT, (0, 0, 0), ordinal=3),
            (1, 1, 1),
        )
    with pytest.raises(ValueError, match="unsupported mobility kind"):
        convert_mobility(
            StationaryMobilitySpec(position_m=(0, 0, 0)),
            "teleport",
            (0, 0, 0),
        )


@pytest.mark.parametrize(
    ("mobility", "keys", "constraints", "guide_keys"),
    (
        (
            StationaryMobilitySpec(position_m=(1, 2, 3)),
            ("position",),
            (MobilityControlConstraint.FREE,),
            (),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            ("start", "end"),
            (MobilityControlConstraint.FREE, MobilityControlConstraint.FREE),
            ("segment",),
        ),
        (
            WaypointMobilitySpec(points_m=((1, 2, 3), (4, 5, 6), (7, 8, 9))),
            ("waypoint_0", "waypoint_1", "waypoint_2"),
            (
                MobilityControlConstraint.FREE,
                MobilityControlConstraint.FREE,
                MobilityControlConstraint.FREE,
            ),
            ("segment_0", "segment_1"),
        ),
        (
            CircularMobilitySpec(
                center_m=(1, 2, 3),
                radius_m=2.0,
                start_angle_deg=90.0,
            ),
            ("center", "radius", "start_angle"),
            (
                MobilityControlConstraint.PLANE,
                MobilityControlConstraint.RADIAL,
                MobilityControlConstraint.ANGULAR,
            ),
            ("radius_guide", "start_angle_guide"),
        ),
    ),
)
def test_control_rig_declares_stable_semantic_handles_and_decorative_guides(
    mobility,
    keys,
    constraints,
    guide_keys,
) -> None:
    rig = mobility_control_rig(mobility)

    assert rig.mobility_kind == mobility.type
    assert tuple(control.key for control in rig.controls) == keys
    assert tuple(control.constraint for control in rig.controls) == constraints
    assert tuple(guide.key for guide in rig.guides) == guide_keys
    assert all(control.label and control.tooltip for control in rig.controls)
    assert all(control.point_size > 0.0 for control in rig.controls)
    assert all(isinstance(control.glyph, MobilityControlGlyph) for control in rig.controls)


@pytest.mark.parametrize(
    ("mobility", "key", "position", "expected"),
    (
        (
            StationaryMobilitySpec(position_m=(1, 2, 3)),
            "position",
            (7, 8, 9),
            StationaryMobilitySpec(position_m=(7, 8, 9)),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            "start",
            (7, 8, 9),
            LinearMobilitySpec(start_m=(7, 8, 9), end_m=(4, 5, 6)),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            "end",
            (7, 8, 9),
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(7, 8, 9)),
        ),
        (
            WaypointMobilitySpec(points_m=((1, 2, 3), (4, 5, 6), (7, 8, 9))),
            "waypoint_1",
            (10, 11, 12),
            WaypointMobilitySpec(points_m=((1, 2, 3), (10, 11, 12), (7, 8, 9))),
        ),
    ),
)
def test_control_rig_updates_only_the_named_semantic_value(
    mobility,
    key,
    position,
    expected,
) -> None:
    assert update_mobility_from_rig_control(mobility, key, position) == expected


def test_circular_radius_and_start_angle_controls_are_independent() -> None:
    mobility = CircularMobilitySpec(
        center_m=(1, 2, 3),
        radius_m=2.0,
        start_angle_deg=15.0,
        clockwise=False,
    )
    rig = mobility_control_rig(mobility)

    radius_control = rig.control("radius")
    angle_control = rig.control("start_angle")
    assert radius_control.position != pytest.approx(angle_control.position)
    assert radius_control.operation is MobilityControlOperation.SET_RADIUS
    assert angle_control.operation is MobilityControlOperation.SET_START_ANGLE

    radius_axis = (
        (radius_control.position[0] - mobility.center_m[0]) / mobility.radius_m,
        (radius_control.position[1] - mobility.center_m[1]) / mobility.radius_m,
    )
    radius_position = (
        mobility.center_m[0] + radius_axis[0] * 5.0,
        mobility.center_m[1] + radius_axis[1] * 5.0,
        99.0,
    )
    resized = update_mobility_from_rig_control(mobility, radius_control, radius_position)
    assert isinstance(resized, CircularMobilitySpec)
    assert resized.center_m == mobility.center_m
    assert resized.radius_m == pytest.approx(5.0)
    assert resized.start_angle_deg == pytest.approx(mobility.start_angle_deg)
    assert resized.clockwise is False

    tangential_drag = (
        radius_control.position[0] - radius_axis[1] * 20.0,
        radius_control.position[1] + radius_axis[0] * 20.0,
        radius_control.position[2],
    )
    tangential = update_mobility_from_rig_control(
        mobility,
        radius_control,
        tangential_drag,
    )
    assert isinstance(tangential, CircularMobilitySpec)
    assert tangential.radius_m == pytest.approx(mobility.radius_m)

    rotated = update_mobility_from_rig_control(mobility, angle_control, (1, -8, 99))
    assert isinstance(rotated, CircularMobilitySpec)
    assert rotated.center_m == mobility.center_m
    assert rotated.radius_m == pytest.approx(mobility.radius_m)
    assert rotated.start_angle_deg == pytest.approx(-90.0)
    assert rotated.clockwise is False


def test_control_keys_stay_stable_when_mobility_values_change() -> None:
    before = mobility_control_rig(CircularMobilitySpec(center_m=(0, 0, 0), radius_m=2.0))
    after = mobility_control_rig(
        CircularMobilitySpec(
            center_m=(10, 20, 30),
            radius_m=8.0,
            start_angle_deg=85.0,
            clockwise=False,
        )
    )

    assert tuple(control.key for control in before.controls) == tuple(
        control.key for control in after.controls
    )
    assert tuple(MOBILITY_CONTROL_RIG_ADAPTERS) == (
        StationaryMobilitySpec,
        LinearMobilitySpec,
        WaypointMobilitySpec,
        CircularMobilitySpec,
    )


def test_control_rig_values_are_immutable_and_validate_semantics() -> None:
    control = MobilityControlDescriptor(
        key="position",
        position=(1, 2, 3),
        label="Position",
        tooltip="Move the position.",
        operation=MobilityControlOperation.SET_POSITION,
    )

    with pytest.raises(FrozenInstanceError):
        control.position = (4, 5, 6)  # type: ignore[misc]
    with pytest.raises(ValueError, match="control key"):
        MobilityControlDescriptor(
            key="Start point",
            position=(1, 2, 3),
            label="Start",
            tooltip="Move it.",
            operation=MobilityControlOperation.SET_START,
        )
    with pytest.raises(ValueError, match="keys must be unique"):
        MobilityControlRig("linear", (control, control))
    with pytest.raises(KeyError, match="unknown linear mobility control"):
        mobility_control_rig(LinearMobilitySpec(start_m=(0, 0, 0), end_m=(1, 0, 0))).control(
            "position"
        )
    with pytest.raises(ValueError, match="cannot coincide"):
        update_mobility_from_rig_control(
            CircularMobilitySpec(center_m=(1, 2, 3), radius_m=2.0),
            "radius",
            (1, 2, 99),
        )
