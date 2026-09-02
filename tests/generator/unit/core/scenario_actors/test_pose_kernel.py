"""Focused scientific-contract tests for actor pose preparation."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from generator.core.scenario_actors import (
    PosePreparationError,
    Quaternion,
    Timeline,
    apply_asset_alignment,
    prepare_mobility,
    prepare_orientation,
    prepare_scenario,
)
from generator.core.scenario_actors.runtime import PreparedMobilityAdapter
from shared.scenarios.actors import (
    ActorsSpec,
    AlignMotionOrientationSpec,
    CatalogAssetSpec,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    Figure8MobilitySpec,
    FixedOrientationSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    GroupSpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    ManhattanGridMobilitySpec,
    OrientationKeyframeSpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    RxActorSpec,
    SampledMobilitySpec,
    SpinOrientationSpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    TargetActorSpec,
    TimelineSpec,
    TxActorSpec,
    WaypointMobilitySpec,
)
from shared.scenarios.model import ScenarioModel

pytestmark = pytest.mark.unit


def _angles_close(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
    *,
    atol: float = 1e-8,
) -> None:
    for actual_angle, expected_angle in zip(actual, expected):
        delta = (actual_angle - expected_angle + 180.0) % 360.0 - 180.0
        assert abs(delta) <= atol


def _forward_axis(orientation_deg: tuple[float, float, float]) -> np.ndarray:
    yaw, pitch, _roll = np.radians(orientation_deg)
    return np.asarray(
        (
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
        )
    )


def test_timeline_is_endpoint_inclusive_and_frozen() -> None:
    timeline = Timeline(steps=5, duration_s=2.0)

    assert timeline.timestamps_s == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert Timeline(steps=1, duration_s=10.0).timestamps_s == (0.0,)
    with pytest.raises(FrozenInstanceError):
        timeline.steps = 2  # type: ignore[misc]


def test_linear_fit_duration_has_exact_samples_and_physical_velocity() -> None:
    mobility = prepare_mobility(
        LinearMobilitySpec(start_m=(0, 0, 0), end_m=(4, 0, 0)),
        Timeline(steps=5, duration_s=2.0),
    )

    assert mobility.positions_m == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    )
    assert mobility.velocities_mps == ((2.0, 0.0, 0.0),) * 5
    assert mobility.forward_vectors == ((1.0, 0.0, 0.0),) * 5


@pytest.mark.parametrize(
    ("after_end", "expected_x"),
    [
        ("hold", (0.0, 1.0, 2.0, 2.0, 2.0)),
        ("loop", (0.0, 1.0, 0.0, 1.0, 0.0)),
        ("ping_pong", (0.0, 1.0, 2.0, 1.0, 0.0)),
    ],
)
def test_constant_speed_traversal_controls_positions(
    after_end: str,
    expected_x: tuple[float, ...],
) -> None:
    mobility = prepare_mobility(
        LinearMobilitySpec(
            start_m=(0, 0, 0),
            end_m=(2, 0, 0),
            traversal=ConstantSpeedTraversalSpec(
                speed_mps=1.0,
                after_end=after_end,
            ),
        ),
        Timeline(steps=5, duration_s=4.0),
    )

    assert tuple(position[0] for position in mobility.positions_m) == expected_x


def test_waypoint_uses_arc_length_and_preserves_turn_heading() -> None:
    mobility = prepare_mobility(
        WaypointMobilitySpec(
            points_m=((0, 0, 0), (2, 0, 0), (2, 2, 0)),
        ),
        Timeline(steps=3, duration_s=2.0),
    )

    assert mobility.positions_m == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0),
    )
    assert mobility.forward_vectors == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    )


def test_circular_angles_are_degrees_and_endpoint_inclusive() -> None:
    mobility = prepare_mobility(
        CircularMobilitySpec(
            center_m=(1, 2, 3),
            radius_m=2,
            start_angle_deg=90,
            clockwise=False,
        ),
        Timeline(steps=5, duration_s=4.0),
    )

    np.testing.assert_allclose(mobility.positions_m[0], (1, 4, 3), atol=1e-12)
    np.testing.assert_allclose(mobility.positions_m[-1], (1, 4, 3), atol=1e-12)


def test_moving_path_rejects_degenerate_timeline() -> None:
    with pytest.raises(PosePreparationError, match="at least two"):
        prepare_mobility(
            LinearMobilitySpec(start_m=(0, 0, 0), end_m=(1, 0, 0)),
            Timeline(steps=1, duration_s=1.0),
        )
    with pytest.raises(PosePreparationError, match="positive timeline"):
        prepare_mobility(
            LinearMobilitySpec(start_m=(0, 0, 0), end_m=(1, 0, 0)),
            Timeline(steps=2, duration_s=0.0),
        )


def test_discrete_sampling_is_seeded_and_has_no_physical_velocity() -> None:
    spec = RandomSamplingMobilitySpec(
        x_bounds_m=(-1, 1),
        y_bounds_m=(-2, 2),
        z_bounds_m=(3, 3),
        seed=42,
    )
    coarse = prepare_mobility(spec, Timeline(steps=3, duration_s=2.0))
    fine = prepare_mobility(spec, Timeline(steps=5, duration_s=2.0))

    assert coarse.positions_m == tuple(fine.positions_m[index] for index in (0, 2, 4))
    assert coarse.velocities_mps == ((0.0, 0.0, 0.0),) * 3
    assert not coarse.has_physical_velocity
    assert not PreparedMobilityAdapter(
        coarse,
        timeline_steps=3,
        duration_s=2.0,
    ).has_physical_motion
    with pytest.raises(PosePreparationError, match="physical velocity"):
        prepare_orientation(
            AlignMotionOrientationSpec(),
            Timeline(steps=3, duration_s=2.0),
            coarse,
        )


def test_discrete_sampling_can_pin_only_the_initial_observation() -> None:
    unanchored = RandomSamplingMobilitySpec(
        x_bounds_m=(-10, 10),
        y_bounds_m=(-10, 10),
        z_bounds_m=(3, 3),
        seed=42,
    )
    anchored = unanchored.model_copy(update={"initial_position_m": (4.0, -2.0, 3.0)})
    timeline = Timeline(steps=5, duration_s=2.0)

    unanchored_prepared = prepare_mobility(unanchored, timeline)
    anchored_prepared = prepare_mobility(anchored, timeline)

    assert anchored_prepared.positions_m[0] == (4.0, -2.0, 3.0)
    assert anchored_prepared.positions_m[1:] == unanchored_prepared.positions_m[1:]
    assert len(set(anchored_prepared.positions_m)) == timeline.steps
    assert not anchored_prepared.has_physical_velocity

    outside_bounds = unanchored.model_copy(update={"initial_position_m": (11.0, -2.0, 3.0)})
    with pytest.raises(PosePreparationError, match="initial_position_m.*bounds"):
        prepare_mobility(outside_bounds, timeline)


def test_stationary_has_no_physical_velocity_and_cannot_align_motion() -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    mobility = prepare_mobility(
        StationaryMobilitySpec(position_m=(1, 2, 3)),
        timeline,
    )

    assert not mobility.has_physical_velocity
    assert mobility.velocities_mps == ((0.0, 0.0, 0.0),) * timeline.steps
    with pytest.raises(PosePreparationError, match="physical velocity"):
        prepare_orientation(AlignMotionOrientationSpec(), timeline, mobility)


@pytest.mark.parametrize(
    "spec",
    (
        LinearMobilitySpec(start_m=(1, 2, 3), end_m=(1, 2, 3)),
        OscillatingMobilitySpec(
            center_m=(1, 2, 3),
            axis=(1, 0, 0),
            amplitude_m=0.0,
            frequency_hz=1.0,
        ),
    ),
    ids=("zero-length-linear", "zero-amplitude-oscillating"),
)
def test_align_motion_uses_actual_prepared_motion_not_model_type(spec) -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    mobility = prepare_mobility(spec, timeline)

    assert len(set(mobility.positions_m)) == 1
    assert not mobility.has_physical_velocity
    assert mobility.velocities_mps == ((0.0, 0.0, 0.0),) * timeline.steps
    with pytest.raises(PosePreparationError) as error:
        prepare_orientation(
            AlignMotionOrientationSpec(),
            timeline,
            mobility,
            path="actors.rx[0].orientation",
        )
    assert error.value.issue.code == "orientation_requires_physical_path"
    assert error.value.issue.path == "actors.rx[0].orientation"


def test_random_waypoint_runtime_rejects_degenerate_bounds() -> None:
    with pytest.raises(PosePreparationError, match="distinct destination"):
        prepare_mobility(
            {
                "type": "random_waypoint",
                "initial_position_m": (1.0, 2.0, 3.0),
                "x_bounds_m": (1.0, 1.0),
                "y_bounds_m": (2.0, 2.0),
                "z_bounds_m": (3.0, 3.0),
                "speed_range_mps": (1.0, 2.0),
                "seed": 8,
            },
            Timeline(steps=3, duration_s=2.0),
        )


def test_quaternion_slerp_takes_shortest_path() -> None:
    start = Quaternion.from_euler_deg(170, 0, 0)
    end = Quaternion.from_euler_deg(-170, 0, 0)

    midpoint = start.slerp(end, 0.5).to_euler_deg()

    _angles_close(midpoint, (180, 0, 0))


def test_keyframe_orientation_uses_quaternion_slerp() -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    mobility = prepare_mobility(
        StationaryMobilitySpec(position_m=(0, 0, 0)),
        timeline,
    )
    orientation = prepare_orientation(
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0, yaw_deg=170),
                OrientationKeyframeSpec(time_s=2, yaw_deg=-170),
            )
        ),
        timeline,
        mobility,
    )

    _angles_close(orientation.euler_deg[1], (180, 0, 0))


@pytest.mark.parametrize(
    ("end_m", "expected_pitch_deg"),
    [
        ((0.0, 0.0, 2.0), -90.0),
        ((0.0, 0.0, -2.0), 90.0),
        ((2.0, -3.0, 4.0), None),
        ((2.0, -3.0, -4.0), None),
    ],
)
def test_align_motion_points_forward_axis_along_three_dimensional_path(
    end_m: tuple[float, float, float],
    expected_pitch_deg: float | None,
) -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    mobility = prepare_mobility(
        LinearMobilitySpec(start_m=(0, 0, 0), end_m=end_m),
        timeline,
    )
    orientation = prepare_orientation(
        AlignMotionOrientationSpec(allow_pitch=True),
        timeline,
        mobility,
    )

    expected_forward = np.asarray(end_m, dtype=np.float64)
    expected_forward /= np.linalg.norm(expected_forward)
    for angles in orientation.euler_deg:
        np.testing.assert_allclose(_forward_axis(angles), expected_forward, atol=1e-8)
        if expected_pitch_deg is not None:
            _angles_close(angles, (0, expected_pitch_deg, 0), atol=1e-7)


def test_pitch_enabled_look_at_tracks_moving_actor_in_three_dimensions() -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    owner = prepare_mobility(
        LinearMobilitySpec(start_m=(-1, 0, 1), end_m=(1, 0, -1)),
        timeline,
    )
    target = prepare_mobility(
        LinearMobilitySpec(start_m=(2, 1, 4), end_m=(-2, 1, -4)),
        timeline,
    )
    orientation = prepare_orientation(
        LookAtOrientationSpec(actor="target"),
        timeline,
        owner,
        references={"target": target},
    )

    for owner_position, target_position, angles in zip(
        owner.positions_m,
        target.positions_m,
        orientation.euler_deg,
    ):
        expected_forward = np.asarray(target_position) - np.asarray(owner_position)
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(_forward_axis(angles), expected_forward, atol=1e-8)


@pytest.mark.parametrize(
    ("target_point", "pitch_limits", "expected_pitch"),
    [
        ((1.0, 0.0, 10.0), (-20.0, 70.0), -20.0),
        ((1.0, 0.0, -10.0), (-70.0, 20.0), 20.0),
    ],
)
def test_look_at_pitch_limits_follow_forward_axis_convention(
    target_point: tuple[float, float, float],
    pitch_limits: tuple[float, float],
    expected_pitch: float,
) -> None:
    timeline = Timeline(steps=1, duration_s=0.0)
    owner = prepare_mobility(StationaryMobilitySpec(position_m=(0, 0, 0)), timeline)

    orientation = prepare_orientation(
        LookAtOrientationSpec(point_m=target_point, pitch_limits_deg=pitch_limits),
        timeline,
        owner,
    )

    _angles_close(orientation.euler_deg[0], (0.0, expected_pitch, 0.0))
    assert math.copysign(1.0, _forward_axis(orientation.euler_deg[0])[2]) == math.copysign(
        1.0, target_point[2]
    )


def test_look_at_offsets_are_applied_before_angle_limits() -> None:
    timeline = Timeline(steps=1, duration_s=0.0)
    owner = prepare_mobility(StationaryMobilitySpec(position_m=(0, 0, 0)), timeline)

    orientation = prepare_orientation(
        LookAtOrientationSpec(
            point_m=(1.0, 1.0, 1.0),
            yaw_offset_deg=30.0,
            pitch_offset_deg=-20.0,
            yaw_limits_deg=(-10.0, 50.0),
            pitch_limits_deg=(-30.0, 30.0),
        ),
        timeline,
        owner,
    )

    _angles_close(orientation.euler_deg[0], (50.0, -30.0, 0.0))


def test_look_at_can_track_yaw_while_pitch_is_fixed() -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    owner = prepare_mobility(StationaryMobilitySpec(position_m=(0, 0, 0)), timeline)
    target = prepare_mobility(
        LinearMobilitySpec(start_m=(1, 0, 10), end_m=(0, 1, 20)),
        timeline,
    )
    orientation = prepare_orientation(
        LookAtOrientationSpec(actor="target", allow_pitch=False, pitch_offset_deg=7),
        timeline,
        owner,
        references={"target": target},
    )

    assert [pytest.approx(item[1]) for item in orientation.euler_deg] == [7, 7, 7]
    _angles_close(orientation.euler_deg[0], (0, 7, 0))
    _angles_close(orientation.euler_deg[-1], (90, 7, 0))


def test_coincident_actor_and_point_look_at_share_hold_semantics() -> None:
    timeline = Timeline(steps=3, duration_s=2.0)
    owner = prepare_mobility(
        LinearMobilitySpec(start_m=(0, 0, 0), end_m=(2, 0, 0)),
        timeline,
    )
    target = prepare_mobility(
        LinearMobilitySpec(start_m=(0, 1, 0), end_m=(2, 0, 0)),
        timeline,
    )
    actor_orientation = prepare_orientation(
        LookAtOrientationSpec(
            actor="target",
            yaw_offset_deg=10.0,
            pitch_offset_deg=2.0,
            roll_offset_deg=3.0,
        ),
        timeline,
        owner,
        references={"target": target},
    )
    point_orientation = prepare_orientation(
        LookAtOrientationSpec(
            point_m=(2, 0, 0),
            yaw_offset_deg=10.0,
            pitch_offset_deg=2.0,
            roll_offset_deg=3.0,
        ),
        timeline,
        owner,
    )

    _angles_close(actor_orientation.euler_deg[-1], actor_orientation.euler_deg[-2])
    _angles_close(point_orientation.euler_deg[-1], point_orientation.euler_deg[-2])


def test_spin_and_random_orientation_are_exact_and_deterministic() -> None:
    coarse_timeline = Timeline(steps=3, duration_s=2.0)
    fine_timeline = Timeline(steps=5, duration_s=2.0)
    coarse_mobility = prepare_mobility(
        StationaryMobilitySpec(position_m=(0, 0, 0)),
        coarse_timeline,
    )
    fine_mobility = prepare_mobility(
        StationaryMobilitySpec(position_m=(0, 0, 0)),
        fine_timeline,
    )
    spin = prepare_orientation(
        SpinOrientationSpec(axis="yaw", rate_deg_s=45, yaw_deg=10),
        coarse_timeline,
        coarse_mobility,
    )
    random_spec = RandomOrientationSpec(seed=7)
    coarse_random = prepare_orientation(
        random_spec,
        coarse_timeline,
        coarse_mobility,
    )
    fine_random = prepare_orientation(random_spec, fine_timeline, fine_mobility)

    for actual, expected in zip(spin.euler_deg, ((10, 0, 0), (55, 0, 0), (100, 0, 0))):
        _angles_close(actual, expected)
    assert coarse_random.quaternions == tuple(fine_random.quaternions[index] for index in (0, 2, 4))


def test_asset_alignment_composes_once() -> None:
    timeline = Timeline(steps=2, duration_s=1.0)
    mobility = prepare_mobility(StationaryMobilitySpec(position_m=(0, 0, 0)), timeline)
    authored = prepare_orientation(FixedOrientationSpec(yaw_deg=10), timeline, mobility)

    aligned = apply_asset_alignment(authored, (20, 0, 0))

    _angles_close(aligned.euler_deg[0], (30, 0, 0))
    assert aligned.asset_alignment_applied
    with pytest.raises(PosePreparationError, match="only be composed once"):
        apply_asset_alignment(aligned, (20, 0, 0))


def test_prepare_scenario_supports_standalone_actors_without_groups() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            tx=(
                TxActorSpec(
                    name="base",
                    mobility=StationaryMobilitySpec(position_m=(0, 0, 5)),
                ),
            ),
            rx=(
                RxActorSpec(
                    name="mobile",
                    mobility=LinearMobilitySpec(start_m=(0, 0, 0), end_m=(2, 0, 0)),
                    orientation=AlignMotionOrientationSpec(),
                ),
            ),
        ),
    )

    prepared = prepare_scenario(scenario)

    assert prepared.groups == ()
    assert prepared.actor("base").role == "tx"
    assert prepared.actor("mobile").positions_m[-1] == (2.0, 0.0, 0.0)


def test_prepare_mobility_accepts_exact_sampled_positions_directly() -> None:
    positions = ((0, 0, 0), (1, 0, 0), (4, 0, 0), (9, 0, 0))
    timeline = Timeline(steps=4, duration_s=3.0)

    prepared = prepare_mobility(
        SampledMobilitySpec(positions_m=positions),
        timeline,
    )

    assert prepared.positions_m == tuple(
        tuple(float(component) for component in position) for position in positions
    )
    assert prepared.has_physical_velocity
    assert prepared.velocities_mps == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
    )
    assert prepared.forward_vectors == ((1.0, 0.0, 0.0),) * timeline.steps
    assert PreparedMobilityAdapter(
        prepared,
        timeline_steps=timeline.steps,
        duration_s=timeline.duration_s,
    ).has_physical_motion


def test_prepare_mobility_rejects_sampled_count_mismatch_directly() -> None:
    with pytest.raises(PosePreparationError) as error:
        prepare_mobility(
            SampledMobilitySpec(
                positions_m=((0, 0, 0), (1, 0, 0), (4, 0, 0)),
            ),
            Timeline(steps=4, duration_s=3.0),
            path="actors.rx[0].mobility",
        )

    assert error.value.issue.code == "sample_count_mismatch"
    assert error.value.issue.path == "actors.rx[0].mobility"
    assert error.value.issue.message == "evaluator returned 3 samples for 4 timeline steps"


def test_sampled_mobility_preserves_irregular_timeline_samples_exactly() -> None:
    positions = ((0, 0, 0), (1, 0, 0), (4, 0, 0), (9, 0, 0))
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=4, duration_s=3),
        actors=ActorsSpec(
            rx=(
                RxActorSpec(
                    name="measured",
                    mobility=SampledMobilitySpec(positions_m=positions),
                ),
            ),
        ),
    )

    prepared = prepare_scenario(scenario).actor("measured").mobility

    assert prepared.positions_m == tuple(
        tuple(float(component) for component in position) for position in positions
    )


def test_sampled_mobility_derives_physical_velocity_from_timeline() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=4, duration_s=3),
        actors=ActorsSpec(
            rx=(
                RxActorSpec(
                    name="measured",
                    mobility=SampledMobilitySpec(
                        positions_m=((0, 0, 0), (1, 0, 0), (4, 0, 0), (9, 0, 0))
                    ),
                ),
            ),
        ),
    )

    prepared = prepare_scenario(scenario).actor("measured").mobility

    assert prepared.has_physical_velocity
    assert prepared.velocities_mps == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
    )


def test_sampled_mobility_rejects_a_count_different_from_timeline() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=4, duration_s=3),
        actors=ActorsSpec(
            rx=(
                RxActorSpec(
                    name="measured",
                    mobility=SampledMobilitySpec(positions_m=((0, 0, 0), (1, 0, 0), (4, 0, 0))),
                ),
            ),
        ),
    )

    with pytest.raises(PosePreparationError) as error:
        prepare_scenario(scenario)

    assert error.value.issue.code == "sample_count_mismatch"
    assert error.value.issue.path == "actors.rx[0].mobility"


def test_prepare_scenario_resolves_optional_group_offsets_once() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        groups=(
            GroupSpec(
                name="convoy",
                mobility=LinearMobilitySpec(start_m=(0, 0, 0), end_m=(2, 0, 0)),
            ),
        ),
        actors=ActorsSpec(
            tx=(
                TxActorSpec(
                    name="lead",
                    mobility=GroupMemberMobilitySpec(group="convoy"),
                ),
            ),
            rx=(
                RxActorSpec(
                    name="right",
                    mobility=GroupMemberMobilitySpec(
                        group="convoy",
                        offset_m=GroupOffsetSpec(right=2, forward=-1, up=1),
                    ),
                ),
            ),
        ),
    )

    prepared = prepare_scenario(scenario)

    assert len(prepared.groups) == 1
    assert prepared.actor("lead").positions_m == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )
    assert prepared.actor("right").positions_m == (
        (-1.0, -2.0, 1.0),
        (0.0, -2.0, 1.0),
        (1.0, -2.0, 1.0),
    )


@pytest.mark.parametrize(
    "group_mobility",
    [
        StationaryMobilitySpec(position_m=(10, 20, 30)),
        RandomSamplingMobilitySpec(
            x_bounds_m=(8, 12),
            y_bounds_m=(18, 22),
            z_bounds_m=(30, 30),
            seed=17,
        ),
    ],
)
def test_nonphysical_group_offsets_use_world_axes(
    group_mobility: StationaryMobilitySpec | RandomSamplingMobilitySpec,
) -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        groups=(GroupSpec(name="pair", mobility=group_mobility),),
        actors=ActorsSpec(
            tx=(
                TxActorSpec(
                    name="origin",
                    mobility=GroupMemberMobilitySpec(group="pair"),
                ),
            ),
            rx=(
                RxActorSpec(
                    name="offset",
                    mobility=GroupMemberMobilitySpec(
                        group="pair",
                        offset_m=GroupOffsetSpec(right=2, forward=3, up=4),
                    ),
                ),
            ),
        ),
    )

    prepared = prepare_scenario(scenario)
    group = prepared.groups[0].mobility
    origin = prepared.actor("origin").mobility
    offset = prepared.actor("offset").mobility

    assert not group.has_physical_velocity
    assert not origin.has_physical_velocity
    assert not offset.has_physical_velocity
    assert offset.velocities_mps == ((0.0, 0.0, 0.0),) * 3
    np.testing.assert_allclose(
        np.asarray(offset.positions_m) - np.asarray(group.positions_m),
        np.tile((3.0, -2.0, 4.0), (3, 1)),
    )


def test_group_deviation_is_seeded_per_member() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        groups=(
            GroupSpec(
                name="pair",
                mobility=LinearMobilitySpec(start_m=(0, 0, 0), end_m=(2, 0, 0)),
                deviation=GroupDeviationSpec(max_right_m=0.2, seed=123),
            ),
        ),
        actors=ActorsSpec(
            tx=(TxActorSpec(name="one", mobility=GroupMemberMobilitySpec(group="pair")),),
            rx=(RxActorSpec(name="two", mobility=GroupMemberMobilitySpec(group="pair")),),
        ),
    )

    first = prepare_scenario(scenario)
    second = prepare_scenario(scenario)

    assert first.actor("one").positions_m == second.actor("one").positions_m
    assert first.actor("one").positions_m != first.actor("two").positions_m


def test_target_asset_alignment_is_applied_at_scenario_boundary() -> None:
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=1, duration_s=0),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="car",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=StationaryMobilitySpec(position_m=(0, 0, 0)),
                    orientation=FixedOrientationSpec(yaw_deg=15),
                ),
            )
        ),
    )

    prepared = prepare_scenario(scenario, asset_alignments={"car": (90, 0, 0)})

    _angles_close(prepared.actor("car").orientation.euler_deg[0], (105, 0, 0))


@pytest.mark.parametrize(
    "spec",
    [
        SurveyMobilitySpec(
            origin_m=(0, 0, 1),
            width_m=10,
            height_m=4,
            row_spacing_m=2,
            heading_deg=30,
        ),
        GridScanMobilitySpec(
            x_bounds_m=(0, 2),
            y_bounds_m=(0, 2),
            z_bounds_m=(1, 2),
            x_steps=2,
            y_steps=2,
            z_steps=2,
        ),
        OscillatingMobilitySpec(
            center_m=(0, 0, 0),
            axis=(1, 0, 0),
            amplitude_m=0.1,
            frequency_hz=1,
        ),
        PendulumMobilitySpec(
            pivot_m=(0, 0, 2),
            length_m=1,
            max_angle_deg=20,
            frequency_hz=1,
        ),
        Figure8MobilitySpec(center_m=(0, 0, 1), size_m=2),
        SpiralMobilitySpec(
            center_m=(0, 0, 0),
            radius_m=2,
            start_altitude_m=1,
            end_altitude_m=3,
            turns=2,
        ),
        GaussMarkovMobilitySpec(
            initial_position_m=(0, 0, 1),
            x_bounds_m=(-10, 10),
            y_bounds_m=(-10, 10),
            z_bounds_m=(1, 1),
            alpha=0.8,
            mean_speed_mps=1,
            seed=5,
        ),
        RandomWaypointMobilitySpec(
            initial_position_m=(0, 0, 1),
            x_bounds_m=(-2, 2),
            y_bounds_m=(-2, 2),
            z_bounds_m=(1, 1),
            speed_range_mps=(0.5, 1),
            seed=5,
        ),
        ManhattanGridMobilitySpec(
            origin_xy_m=(0, 0),
            block_size_m=2,
            grid_width=3,
            grid_height=2,
            altitude_m=1,
            speed_range_mps=(0.5, 1),
            seed=5,
        ),
    ],
)
def test_advanced_models_return_exact_finite_samples(spec: object) -> None:
    mobility = prepare_mobility(spec, Timeline(steps=7, duration_s=3.0))

    assert len(mobility.positions_m) == 7
    assert all(math.isfinite(component) for point in mobility.positions_m for component in point)


def test_manhattan_uses_xy_origin_and_absolute_altitude() -> None:
    mobility = prepare_mobility(
        ManhattanGridMobilitySpec(
            origin_xy_m=(10, 20),
            block_size_m=2,
            grid_width=2,
            grid_height=2,
            altitude_m=5,
            speed_range_mps=(1, 1),
            seed=9,
        ),
        Timeline(steps=3, duration_s=2),
    )

    assert {position[2] for position in mobility.positions_m} == {5.0}
