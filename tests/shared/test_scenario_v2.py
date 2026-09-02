"""Focused contract tests for the actor scenario model."""

from __future__ import annotations

from math import inf, nan

import pytest
import yaml
from pydantic import ValidationError

from shared.scenarios import (
    ActorRole,
    ConstantSpeedTraversalSpec,
    FixedOrientationSpec,
    ScenarioModel,
    load_scenario,
    validate_scenario_data,
)


def _stationary(position=(0.0, 0.0, 0.0)) -> dict:
    return {"type": "stationary", "position_m": list(position)}


def _scenario(**overrides) -> dict:
    data = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0.0},
    }
    data.update(overrides)
    return data


def _moving_scenario(**overrides) -> dict:
    data = _scenario(
        timeline={"steps": 5, "duration_s": 2.0},
        actors={
            "rx": [
                {
                    "name": "walker",
                    "mobility": {
                        "type": "linear",
                        "start_m": [0.0, 0.0, 0.0],
                        "end_m": [2.0, 0.0, 0.0],
                    },
                }
            ]
        },
    )
    data.update(overrides)
    return data


def test_schema_v2_requires_explicit_version_and_timeline() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        validate_scenario_data({"schema_version": 1, "timeline": {"steps": 1, "duration_s": 0}})
    with pytest.raises(ValueError, match="expected 2"):
        validate_scenario_data({"timeline": {"steps": 1, "duration_s": 0}})
    with pytest.raises(ValueError, match="timeline"):
        validate_scenario_data({"schema_version": 2})


def test_actors_and_groups_are_optional_for_scripted_scenarios() -> None:
    scenario = validate_scenario_data(_scenario())

    assert scenario.actors.all() == ()
    assert scenario.groups == ()


def test_standalone_actor_does_not_require_a_group() -> None:
    scenario = validate_scenario_data(
        _scenario(
            actors={
                "tx": [{"name": "base", "mobility": _stationary((1, 2, 3))}],
                "rx": [{"name": "user", "mobility": _stationary((4, 5, 6))}],
            }
        )
    )

    assert scenario.groups == ()
    assert scenario.actors.tx[0].role is ActorRole.TX
    assert scenario.actors.rx[0].role is ActorRole.RX
    assert "role" not in scenario.actors.tx[0].model_dump()


def test_orientation_omission_means_fixed_zero_and_models_are_frozen() -> None:
    scenario = validate_scenario_data(
        _scenario(actors={"tx": [{"name": "base", "mobility": _stationary()}]})
    )
    actor = scenario.actors.tx[0]

    assert actor.orientation == FixedOrientationSpec()
    with pytest.raises(ValidationError):
        actor.name = "changed"
    with pytest.raises(ValidationError):
        scenario.timeline.steps = 2


def test_group_membership_is_explicit_and_group_path_is_shared() -> None:
    scenario = validate_scenario_data(
        _scenario(
            timeline={"steps": 3, "duration_s": 2.0},
            groups=[
                {
                    "name": "convoy",
                    "mobility": {
                        "type": "waypoint",
                        "points_m": [[0, 0, 1.5], [10, 0, 1.5]],
                    },
                    "deviation": {"max_right_m": 0.2, "seed": 7},
                }
            ],
            actors={
                "rx": [
                    {
                        "name": "lead",
                        "mobility": {"type": "group_member", "group": "convoy"},
                    }
                ],
                "targets": [
                    {
                        "name": "escort",
                        "asset": {"source": "catalog", "id": "car"},
                        "mobility": {
                            "type": "group_member",
                            "group": "convoy",
                            "offset_m": {"right": 3, "forward": -5, "up": 0},
                        },
                    }
                ],
            },
        )
    )

    assert scenario.groups[0].name == "convoy"
    assert scenario.actors.targets[0].mobility.offset_m.right == 3.0


@pytest.mark.parametrize(
    ("groups", "actors", "match"),
    [
        (
            [],
            {"rx": [{"name": "member", "mobility": {"type": "group_member", "group": "x"}}]},
            "missing group",
        ),
        (
            [{"name": "x", "mobility": _stationary()}],
            {"rx": [{"name": "member", "mobility": {"type": "group_member", "group": "x"}}]},
            "at least two",
        ),
        (
            [
                {"name": "x", "mobility": _stationary()},
                {"name": "x", "mobility": _stationary()},
            ],
            {},
            "group names must be unique",
        ),
    ],
)
def test_invalid_group_graph_is_rejected(groups: list, actors: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_scenario_data(_scenario(groups=groups, actors=actors))


def test_actor_names_are_globally_unique_across_roles() -> None:
    with pytest.raises(ValueError, match="globally unique"):
        validate_scenario_data(
            _scenario(
                actors={
                    "tx": [{"name": "same", "mobility": _stationary()}],
                    "rx": [{"name": "same", "mobility": _stationary()}],
                }
            )
        )


@pytest.mark.parametrize("forbidden", ["id", "role", "position"])
def test_actor_rejects_ids_roles_and_actor_level_positions(forbidden: str) -> None:
    actor = {"name": "a", "mobility": _stationary(), forbidden: "forbidden"}
    with pytest.raises(ValueError, match=forbidden):
        validate_scenario_data(_scenario(actors={"tx": [actor]}))


def test_mobility_and_orientation_are_strict_discriminated_unions() -> None:
    bad_documents = [
        _scenario(
            actors={"tx": [{"name": "a", "mobility": {"type": "drone", "position_m": [0, 0, 0]}}]}
        ),
        _scenario(
            actors={
                "tx": [
                    {
                        "name": "a",
                        "mobility": {"type": "stationary", "position_m": [0, 0, 0], "radius_m": 1},
                    }
                ]
            }
        ),
        _scenario(
            actors={"tx": [{"name": "a", "mobility": _stationary(), "orientation": [0, 0, 0]}]}
        ),
        _scenario(
            actors={
                "tx": [{"name": "a", "mobility": _stationary(), "orientation": {"type": "forward"}}]
            }
        ),
    ]

    for data in bad_documents:
        with pytest.raises(ValueError):
            validate_scenario_data(data)


def test_random_waypoint_requires_nonzero_spatial_extent() -> None:
    data = _moving_scenario()
    data["actors"]["rx"][0]["mobility"] = {
        "type": "random_waypoint",
        "initial_position_m": [1.0, 2.0, 3.0],
        "x_bounds_m": [1.0, 1.0],
        "y_bounds_m": [2.0, 2.0],
        "z_bounds_m": [3.0, 3.0],
        "speed_range_mps": [1.0, 2.0],
        "seed": 7,
    }

    with pytest.raises(ValueError, match="nonzero spatial extent"):
        validate_scenario_data(data)


def test_random_sampling_initial_position_is_optional_and_bounded() -> None:
    mobility = {
        "type": "random_sampling",
        "x_bounds_m": [-10.0, 10.0],
        "y_bounds_m": [-5.0, 5.0],
        "z_bounds_m": [2.0, 2.0],
        "seed": 7,
    }
    data = _moving_scenario()
    data["actors"]["rx"][0]["mobility"] = mobility

    unanchored = validate_scenario_data(data)
    assert unanchored.actors.rx[0].mobility.initial_position_m is None

    mobility["initial_position_m"] = [4.0, -3.0, 2.0]
    anchored = validate_scenario_data(data)
    assert anchored.actors.rx[0].mobility.initial_position_m == (4.0, -3.0, 2.0)

    mobility["initial_position_m"] = [11.0, -3.0, 2.0]
    with pytest.raises(ValueError, match="initial_position_m.*bounds"):
        validate_scenario_data(data)


def test_manhattan_requires_xy_origin_without_an_origin_alias() -> None:
    mobility = {
        "type": "manhattan_grid",
        "origin_xy_m": [10.0, 20.0],
        "block_size_m": 5.0,
        "grid_width": 2,
        "grid_height": 2,
        "altitude_m": 30.0,
        "speed_range_mps": [1.0, 2.0],
        "seed": 4,
    }
    scenario = validate_scenario_data(
        _scenario(
            timeline={"steps": 3, "duration_s": 2.0},
            actors={"rx": [{"name": "walker", "mobility": mobility}]},
        )
    )
    assert scenario.actors.rx[0].mobility.origin_xy_m == (10.0, 20.0)
    assert scenario.actors.rx[0].mobility.altitude_m == 30.0

    mobility["origin_m"] = [10.0, 20.0, 30.0]
    del mobility["origin_xy_m"]
    with pytest.raises(ValueError, match="origin_xy_m|origin_m"):
        validate_scenario_data(
            _scenario(
                timeline={"steps": 3, "duration_s": 2.0},
                actors={"rx": [{"name": "walker", "mobility": mobility}]},
            )
        )


@pytest.mark.parametrize(
    "mobility",
    [
        _stationary(),
        {
            "type": "random_sampling",
            "x_bounds_m": [-1.0, 1.0],
            "y_bounds_m": [-1.0, 1.0],
            "z_bounds_m": [0.0, 0.0],
            "seed": 5,
        },
    ],
)
def test_align_motion_requires_standalone_physical_velocity(mobility: dict) -> None:
    with pytest.raises(ValueError, match="physical velocity"):
        validate_scenario_data(
            _scenario(
                timeline={"steps": 3, "duration_s": 2.0},
                actors={
                    "rx": [
                        {
                            "name": "observer",
                            "mobility": mobility,
                            "orientation": {"type": "align_motion"},
                        }
                    ]
                },
            )
        )


@pytest.mark.parametrize(
    "group_mobility",
    [
        _stationary(),
        {
            "type": "random_sampling",
            "x_bounds_m": [-1.0, 1.0],
            "y_bounds_m": [-1.0, 1.0],
            "z_bounds_m": [0.0, 0.0],
            "seed": 6,
        },
    ],
)
def test_align_motion_requires_group_physical_velocity(group_mobility: dict) -> None:
    member = {"type": "group_member", "group": "pair"}
    with pytest.raises(ValueError, match="physical velocity"):
        validate_scenario_data(
            _scenario(
                timeline={"steps": 3, "duration_s": 2.0},
                groups=[{"name": "pair", "mobility": group_mobility}],
                actors={
                    "tx": [
                        {
                            "name": "one",
                            "mobility": member,
                            "orientation": {"type": "align_motion"},
                        }
                    ],
                    "rx": [{"name": "two", "mobility": member}],
                },
            )
        )


def test_constant_speed_requires_positive_speed_and_defaults_to_hold() -> None:
    scenario = validate_scenario_data(
        _moving_scenario(
            actors={
                "rx": [
                    {
                        "name": "walker",
                        "mobility": {
                            "type": "linear",
                            "start_m": [0, 0, 0],
                            "end_m": [2, 0, 0],
                            "traversal": {"type": "constant_speed", "speed_mps": 1.0},
                        },
                    }
                ]
            }
        )
    )
    traversal = scenario.actors.rx[0].mobility.traversal
    assert isinstance(traversal, ConstantSpeedTraversalSpec)
    assert traversal.after_end == "hold"

    data = _moving_scenario()
    data["actors"]["rx"][0]["mobility"]["traversal"] = {"type": "constant_speed"}
    with pytest.raises(ValueError, match="speed_mps"):
        validate_scenario_data(data)


@pytest.mark.parametrize("steps,duration", [(1, 1.0), (2, 0.0)])
def test_moving_scenario_timeline_constraints(steps: int, duration: float) -> None:
    with pytest.raises(ValueError, match="moving scenarios"):
        validate_scenario_data(_moving_scenario(timeline={"steps": steps, "duration_s": duration}))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["timeline"].update(duration_s=nan),
        lambda data: data["actors"]["rx"][0]["mobility"].update(end_m=[inf, 0, 0]),
        lambda data: data["actors"]["rx"][0].update(orientation={"type": "fixed", "yaw_deg": nan}),
    ],
)
def test_nonfinite_values_are_rejected(mutate) -> None:
    data = _moving_scenario()
    mutate(data)
    with pytest.raises(ValueError, match="finite"):
        validate_scenario_data(data)


def test_look_at_actor_reference_must_exist_and_not_be_self() -> None:
    for target in ("missing", "watcher"):
        with pytest.raises(ValueError, match="look_at"):
            validate_scenario_data(
                _scenario(
                    actors={
                        "rx": [
                            {
                                "name": "watcher",
                                "mobility": _stationary(),
                                "orientation": {"type": "look_at", "actor": target},
                            }
                        ]
                    }
                )
            )


def test_target_asset_is_required_and_mesh_sequence_is_target_only() -> None:
    with pytest.raises(ValueError, match="asset"):
        validate_scenario_data(
            _scenario(actors={"targets": [{"name": "cube", "mobility": _stationary()}]})
        )

    with pytest.raises(ValueError, match="target actors"):
        validate_scenario_data(
            _scenario(
                timeline={"steps": 2, "duration_s": 1.0},
                actors={
                    "rx": [
                        {
                            "name": "bad",
                            "mobility": {
                                "type": "mesh_sequence",
                                "positions_path": "positions.h5",
                            },
                        }
                    ]
                },
            )
        )


@pytest.mark.parametrize(
    "asset",
    [
        {"source": "catalog", "id": "human/walking"},
        {"source": "directory", "path": "targets/human"},
    ],
    ids=("catalog", "directory"),
)
def test_target_mesh_end_behavior_defaults_to_loop_and_parses_hold_last(asset: dict) -> None:
    actor = {
        "name": "human",
        "asset": asset,
        "mobility": _stationary(),
    }

    default_scenario = validate_scenario_data(_scenario(actors={"targets": [actor]}))
    assert default_scenario.actors.targets[0].asset.mesh_end_behavior == "loop"

    actor["asset"] = {**asset, "mesh_end_behavior": "hold_last"}
    hold_scenario = validate_scenario_data(_scenario(actors={"targets": [actor]}))
    assert hold_scenario.actors.targets[0].asset.mesh_end_behavior == "hold_last"

    actor["asset"] = {**asset, "mesh_end_behavior": "reverse"}
    with pytest.raises(ValueError, match="mesh_end_behavior"):
        validate_scenario_data(_scenario(actors={"targets": [actor]}))


@pytest.mark.parametrize("asset_id", ["../cube", "cars/../cube", "/cube", "C:/cube", "cars\\cube"])
def test_catalog_asset_id_cannot_escape_the_catalog(asset_id: str) -> None:
    with pytest.raises(ValueError, match="catalog id"):
        validate_scenario_data(
            _scenario(
                actors={
                    "targets": [
                        {
                            "name": "unsafe",
                            "asset": {"source": "catalog", "id": asset_id},
                            "mobility": _stationary(),
                        }
                    ]
                }
            )
        )


def test_retired_pose_roots_and_raytracing_timeline_are_rejected() -> None:
    for retired_field in ("devices", "targets"):
        with pytest.raises(ValueError, match=retired_field):
            validate_scenario_data(
                _scenario(**{retired_field: [] if retired_field == "targets" else {}})
            )

    with pytest.raises(ValueError, match="steps"):
        validate_scenario_data(_scenario(raytracing={"steps": 5}))


def test_load_scenario_returns_typed_model(tmp_path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(_scenario()), encoding="utf-8")

    scenario = load_scenario(path)

    assert isinstance(scenario, ScenarioModel)
    assert scenario.schema_version == 2
