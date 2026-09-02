"""Schema-direct mobility registry and transformation tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from types import MappingProxyType

import pytest
from pydantic import TypeAdapter

from generator.core.scenario_actors import prepare_scenario
from generator.core.scenario_actors.mobility import prepare_mobility
from generator.core.scenario_actors.types import Timeline
from shared.scenarios.actors import (
    ActorRole,
    ConstantSpeedTraversalSpec,
    FitDurationTraversalSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupMemberMobilitySpec,
    LinearMobilitySpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    MobilitySpec,
    NetworkRouteMobilitySpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from shared.scenarios.yaml import validate_scenario_data
from visualizer.src.authoring.compiler import canonical_scenario_mapping
from visualizer.src.authoring.domain import (
    AuthoringActor,
    AuthoringScenario,
    SceneReference,
)
from visualizer.src.authoring.mobility_models import (
    MOBILITY_MODEL_LABELS,
    MOBILITY_MODEL_TYPES,
    MOBILITY_MODELS,
    MobilityContext,
    MobilityFamily,
    MobilityKind,
    MobilityNeedsContextError,
    convert_mobility,
    default_mobility,
    mobility_kind,
    translate_mobility,
)
from visualizer.src.authoring.model_capabilities import (
    SUPPORTED_AUTHORING_MOBILITY_TYPES,
)

ANCHOR = (3.0, -2.0, 7.0)
DURATION_S = 4.0
SEED = 314159

CONTEXT_FREE_KINDS = tuple(
    kind for kind, descriptor in MOBILITY_MODELS.items() if not descriptor.requires_context
)
CONTEXT_KINDS = {
    MobilityKind.GROUP_MEMBER: MobilityContext.GROUP,
    MobilityKind.NETWORK_ROUTE: MobilityContext.NETWORK_GRAPH,
    MobilityKind.MESH_SEQUENCE: MobilityContext.POSITION_SEQUENCE,
}
TRAVERSABLE_KINDS = {
    MobilityKind.LINEAR,
    MobilityKind.WAYPOINT,
    MobilityKind.CIRCULAR,
    MobilityKind.SURVEY,
    MobilityKind.GRID_SCAN,
    MobilityKind.FIGURE8,
    MobilityKind.SPIRAL,
}


def _schema_discriminators() -> set[str]:
    discriminator = TypeAdapter(MobilitySpec).json_schema()["discriminator"]
    return set(discriminator["mapping"])


def _assert_vec3_close(actual: Iterable[float], expected: Iterable[float]) -> None:
    assert tuple(actual) == pytest.approx(tuple(expected), abs=1e-9)


def _prepared_positions(kind: MobilityKind):
    model = default_mobility(kind, ANCHOR, DURATION_S, seed=SEED)
    prepared = prepare_mobility(model, Timeline(steps=9, duration_s=DURATION_S))
    return model, prepared.positions_m


def _distance_from_line(point, start, end) -> float:
    direction = tuple(end[axis] - start[axis] for axis in range(3))
    offset = tuple(point[axis] - start[axis] for axis in range(3))
    direction_norm = math.sqrt(sum(component**2 for component in direction))
    assert direction_norm > 0.0
    cross = (
        offset[1] * direction[2] - offset[2] * direction[1],
        offset[2] * direction[0] - offset[0] * direction[2],
        offset[0] * direction[1] - offset[1] * direction[0],
    )
    return math.sqrt(sum(component**2 for component in cross)) / direction_norm


def test_registry_is_immutable_typed_and_complete() -> None:
    assert isinstance(MOBILITY_MODELS, MappingProxyType)
    assert isinstance(MOBILITY_MODEL_TYPES, MappingProxyType)
    assert isinstance(MOBILITY_MODEL_LABELS, MappingProxyType)
    registry_types = {kind.value for kind in MOBILITY_MODELS}
    assert registry_types == SUPPORTED_AUTHORING_MOBILITY_TYPES
    assert _schema_discriminators() - registry_types == {"sampled"}
    assert len(MOBILITY_MODELS) == 17

    for kind, descriptor in MOBILITY_MODELS.items():
        assert descriptor.kind is kind
        assert descriptor.model_type is MOBILITY_MODEL_TYPES[kind]
        assert descriptor.label == MOBILITY_MODEL_LABELS[kind]
        assert descriptor.label
        assert isinstance(descriptor.family, MobilityFamily)
        assert descriptor.model_type.model_fields["type"].default == kind.value
        assert descriptor.allowed_roles

    with pytest.raises(TypeError):
        MOBILITY_MODELS[MobilityKind.STATIONARY] = MOBILITY_MODELS[  # type: ignore[index]
            MobilityKind.LINEAR
        ]


def test_registry_declares_context_and_actor_role_constraints() -> None:
    for kind, required_context in CONTEXT_KINDS.items():
        descriptor = MOBILITY_MODELS[kind]
        assert descriptor.context is required_context
        assert descriptor.required_context is required_context
        assert descriptor.requires_context

    for kind in CONTEXT_FREE_KINDS:
        descriptor = MOBILITY_MODELS[kind]
        assert descriptor.context is MobilityContext.NONE
        assert descriptor.required_context is None

    mesh_sequence = MOBILITY_MODELS[MobilityKind.MESH_SEQUENCE]
    assert mesh_sequence.allowed_roles == {ActorRole.TARGET}
    assert mesh_sequence.allows_role(ActorRole.TARGET)
    assert not mesh_sequence.allows_role(ActorRole.TX)
    for kind, descriptor in MOBILITY_MODELS.items():
        if kind is not MobilityKind.MESH_SEQUENCE:
            assert descriptor.allowed_roles == set(ActorRole)


@pytest.mark.parametrize("kind", CONTEXT_FREE_KINDS, ids=lambda kind: kind.value)
def test_defaults_use_shared_specs_and_start_at_the_visible_anchor(kind: MobilityKind) -> None:
    model, positions = _prepared_positions(kind)

    assert isinstance(model, MOBILITY_MODEL_TYPES[kind])
    assert mobility_kind(model) is kind
    _assert_vec3_close(positions[0], ANCHOR)

    if kind in TRAVERSABLE_KINDS:
        assert isinstance(model.traversal, FitDurationTraversalSpec)


def test_stochastic_defaults_are_repeatable_and_use_the_requested_seed() -> None:
    seeded_kinds = (
        MobilityKind.RANDOM_SAMPLING,
        MobilityKind.GAUSS_MARKOV,
        MobilityKind.RANDOM_WAYPOINT,
        MobilityKind.MANHATTAN_GRID,
    )
    timeline = Timeline(steps=11, duration_s=DURATION_S)

    for kind in seeded_kinds:
        first = default_mobility(kind, ANCHOR, DURATION_S, seed=SEED)
        second = default_mobility(kind, ANCHOR, DURATION_S, seed=SEED)
        assert first == second
        assert first.seed == SEED
        assert prepare_mobility(first, timeline) == prepare_mobility(second, timeline)


def test_spatial_and_stochastic_defaults_prepare_visible_motion() -> None:
    timeline = Timeline(steps=31, duration_s=DURATION_S)
    random_sampling_model = default_mobility(
        MobilityKind.RANDOM_SAMPLING,
        ANCHOR,
        DURATION_S,
        seed=SEED,
    )
    gauss_markov_model = default_mobility(
        MobilityKind.GAUSS_MARKOV,
        ANCHOR,
        DURATION_S,
        seed=SEED,
    )
    manhattan_grid_model = default_mobility(
        MobilityKind.MANHATTAN_GRID,
        ANCHOR,
        DURATION_S,
        seed=SEED,
    )
    random_sampling = prepare_mobility(random_sampling_model, timeline)
    gauss_markov = prepare_mobility(gauss_markov_model, timeline)
    manhattan_grid = prepare_mobility(manhattan_grid_model, timeline)

    assert isinstance(random_sampling_model, RandomSamplingMobilitySpec)
    assert isinstance(gauss_markov_model, GaussMarkovMobilitySpec)
    assert isinstance(manhattan_grid_model, ManhattanGridMobilitySpec)
    assert random_sampling_model.initial_position_m == ANCHOR
    assert random_sampling_model.x_bounds_m == (ANCHOR[0] - 10.0, ANCHOR[0] + 10.0)
    assert len(set(random_sampling.positions_m)) > 1
    assert gauss_markov_model.alpha == pytest.approx(0.65)
    assert gauss_markov_model.mean_speed_mps == pytest.approx(2.0)
    assert gauss_markov_model.speed_std_mps == pytest.approx(0.4)
    assert gauss_markov_model.direction_std_deg == pytest.approx(40.0)
    assert len(set(gauss_markov.positions_m)) > 1
    assert any(abs(point[1] - ANCHOR[1]) > 1e-6 for point in gauss_markov.positions_m)
    assert (
        max(
            _distance_from_line(
                point,
                gauss_markov.positions_m[0],
                gauss_markov.positions_m[-1],
            )
            for point in gauss_markov.positions_m[1:-1]
        )
        > 0.5
    )
    assert manhattan_grid_model.block_size_m == 2.0
    assert len(set(manhattan_grid.positions_m)) > 1
    assert any(
        abs(point[0] - ANCHOR[0]) >= manhattan_grid_model.block_size_m
        or abs(point[1] - ANCHOR[1]) >= manhattan_grid_model.block_size_m
        for point in manhattan_grid.positions_m
    )


def test_path_defaults_prepare_visible_multiaxis_shapes() -> None:
    timeline = Timeline(steps=30, duration_s=3.0)
    waypoint_model = default_mobility(
        MobilityKind.WAYPOINT,
        ANCHOR,
        timeline.duration_s,
        seed=SEED,
    )
    grid_scan_model = default_mobility(
        MobilityKind.GRID_SCAN,
        ANCHOR,
        timeline.duration_s,
        seed=SEED,
    )
    random_waypoint_model = default_mobility(
        MobilityKind.RANDOM_WAYPOINT,
        ANCHOR,
        timeline.duration_s,
        seed=SEED,
    )

    assert isinstance(waypoint_model, WaypointMobilitySpec)
    assert len(waypoint_model.points_m) == 3
    first, corner, last = waypoint_model.points_m
    _assert_vec3_close(first, ANCHOR)
    assert _distance_from_line(corner, first, last) > 1.0

    assert isinstance(grid_scan_model, GridScanMobilitySpec)
    assert grid_scan_model.z_steps == 2
    assert grid_scan_model.z_bounds_m[1] - grid_scan_model.z_bounds_m[0] > 1.0
    grid_scan = prepare_mobility(grid_scan_model, timeline)
    assert (
        max(point[2] for point in grid_scan.positions_m)
        - min(point[2] for point in grid_scan.positions_m)
        > 1.0
    )

    assert isinstance(random_waypoint_model, RandomWaypointMobilitySpec)
    assert random_waypoint_model.x_bounds_m == (ANCHOR[0] - 2.0, ANCHOR[0] + 2.0)
    assert random_waypoint_model.y_bounds_m == (ANCHOR[1] - 2.0, ANCHOR[1] + 2.0)
    assert random_waypoint_model.speed_range_mps == (4.0, 6.0)
    random_waypoint = prepare_mobility(random_waypoint_model, timeline)
    start = random_waypoint.positions_m[0]
    end = random_waypoint.positions_m[-1]
    assert (
        max(_distance_from_line(point, start, end) for point in random_waypoint.positions_m[1:-1])
        > 0.5
    )


def test_constant_speed_traversal_keeps_nested_discriminator_through_generator() -> None:
    tx = AuthoringActor.create(ActorRole.TX, "TX")
    rx = AuthoringActor.create(ActorRole.RX, "RX").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 0.0),
            end_m=(10.0, 0.0, 0.0),
            traversal=ConstantSpeedTraversalSpec(
                speed_mps=2.0,
                after_end="hold",
            ),
        )
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx, rx),
    )

    mapping = canonical_scenario_mapping(scenario)
    traversal = mapping["actors"]["rx"][0]["mobility"]["traversal"]
    prepared = prepare_scenario(
        validate_scenario_data(mapping),
        base_dir=".",
    )

    assert traversal == {"type": "constant_speed", "speed_mps": 2.0}
    assert prepared.actor("RX").positions_m[0] == (0.0, 0.0, 0.0)
    assert prepared.actor("RX").positions_m[-1] == pytest.approx((6.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("kind", "context"),
    tuple(CONTEXT_KINDS.items()),
    ids=lambda value: value.value,
)
def test_context_dependent_defaults_report_typed_requirement(
    kind: MobilityKind,
    context: MobilityContext,
) -> None:
    with pytest.raises(MobilityNeedsContextError) as caught:
        default_mobility(kind, ANCHOR, DURATION_S, seed=SEED)

    assert caught.value.kind is kind
    assert caught.value.context is context
    assert caught.value.required_context is context
    assert caught.value.operation == "creating"


@pytest.mark.parametrize("kind", CONTEXT_FREE_KINDS, ids=lambda kind: kind.value)
def test_translation_offsets_the_complete_canonical_trajectory(kind: MobilityKind) -> None:
    offset = (2.5, -1.25, 4.0)
    model = default_mobility(kind, ANCHOR, DURATION_S, seed=SEED)
    translated = translate_mobility(model, offset)
    timeline = Timeline(steps=13, duration_s=DURATION_S)
    before = prepare_mobility(model, timeline).positions_m
    after = prepare_mobility(translated, timeline).positions_m

    assert mobility_kind(translated) is kind
    for original, shifted in zip(before, after):
        _assert_vec3_close(
            shifted,
            (
                original[0] + offset[0],
                original[1] + offset[1],
                original[2] + offset[2],
            ),
        )


def test_translation_preserves_non_spatial_fields_and_full_traversal() -> None:
    traversal = ConstantSpeedTraversalSpec(
        speed_mps=2.5,
        after_end="ping_pong",
    )
    model = MOBILITY_MODEL_TYPES[MobilityKind.LINEAR](
        start_m=(0.0, 0.0, 0.0),
        end_m=(10.0, 0.0, 0.0),
        traversal=traversal,
    )

    translated = translate_mobility(model, (1.0, 2.0, 3.0))

    assert translated.start_m == (1.0, 2.0, 3.0)
    assert translated.end_m == (11.0, 2.0, 3.0)
    assert translated.traversal == traversal


@pytest.mark.parametrize(
    ("model", "context"),
    (
        (
            GroupMemberMobilitySpec(group="66f83594-f041-410f-8fa6-ad06d28bf80b"),
            MobilityContext.GROUP,
        ),
        (
            NetworkRouteMobilitySpec(graph_path="network.graphml"),
            MobilityContext.NETWORK_GRAPH,
        ),
        (
            MeshSequenceMobilitySpec(positions_path="positions.npy"),
            MobilityContext.POSITION_SEQUENCE,
        ),
    ),
    ids=("group_member", "network_route", "mesh_sequence"),
)
def test_context_dependent_translation_reports_typed_requirement(model, context) -> None:
    with pytest.raises(MobilityNeedsContextError) as caught:
        translate_mobility(model, (1.0, 2.0, 3.0))

    assert caught.value.kind is mobility_kind(model)
    assert caught.value.context is context
    assert caught.value.operation == "translating"


@pytest.mark.parametrize("target_kind", CONTEXT_FREE_KINDS, ids=lambda kind: kind.value)
def test_conversion_preserves_visible_anchor_for_every_context_free_target(
    target_kind: MobilityKind,
) -> None:
    source = default_mobility(
        (
            MobilityKind.LINEAR
            if target_kind is MobilityKind.STATIONARY
            else MobilityKind.STATIONARY
        ),
        (-100.0, -100.0, -100.0),
        DURATION_S,
        seed=SEED,
    )
    converted = convert_mobility(
        source,
        target_kind,
        ANCHOR,
        duration_s=DURATION_S,
        seed=SEED,
    )
    prepared = prepare_mobility(converted, Timeline(steps=7, duration_s=DURATION_S))

    _assert_vec3_close(prepared.positions_m[0], ANCHOR)


def test_same_kind_conversion_preserves_the_exact_model() -> None:
    source = default_mobility(MobilityKind.LINEAR, ANCHOR, DURATION_S, seed=SEED)

    converted = convert_mobility(
        source,
        " linear ",
        (999.0, 999.0, 999.0),
        duration_s=20.0,
        seed=1,
    )

    assert converted is source


@pytest.mark.parametrize(
    ("kind", "context"),
    tuple(CONTEXT_KINDS.items()),
    ids=lambda value: value.value,
)
def test_conversion_to_context_model_reports_typed_requirement(
    kind: MobilityKind,
    context: MobilityContext,
) -> None:
    source = StationaryMobilitySpec(position_m=ANCHOR)

    with pytest.raises(MobilityNeedsContextError) as caught:
        convert_mobility(
            source,
            kind,
            ANCHOR,
            duration_s=DURATION_S,
            seed=SEED,
        )

    assert caught.value.kind is kind
    assert caught.value.context is context


def test_unknown_kinds_and_invalid_scalar_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported mobility kind"):
        default_mobility("teleport", ANCHOR, DURATION_S)
    with pytest.raises(ValueError, match="duration_s"):
        default_mobility(MobilityKind.LINEAR, ANCHOR, float("nan"))
    with pytest.raises(ValueError, match="seed"):
        default_mobility(MobilityKind.RANDOM_WAYPOINT, ANCHOR, DURATION_S, seed=1.5)  # type: ignore[arg-type]
