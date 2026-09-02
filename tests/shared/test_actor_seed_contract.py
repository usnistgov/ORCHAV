"""Canonical random-seed bounds shared by actor schema families."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.scenarios.actors import (
    MAX_RANDOM_SEED,
    GaussMarkovMobilitySpec,
    GroupDeviationSpec,
    ManhattanGridMobilitySpec,
    NetworkRouteMobilitySpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
)

SEEDED_MODELS = (
    RandomSamplingMobilitySpec(
        x_bounds_m=(-1.0, 1.0),
        y_bounds_m=(-1.0, 1.0),
        z_bounds_m=(0.0, 1.0),
        seed=0,
    ),
    GaussMarkovMobilitySpec(
        initial_position_m=(0.0, 0.0, 0.0),
        x_bounds_m=(-1.0, 1.0),
        y_bounds_m=(-1.0, 1.0),
        z_bounds_m=(0.0, 0.0),
        alpha=0.8,
        mean_speed_mps=1.0,
        seed=0,
    ),
    RandomWaypointMobilitySpec(
        initial_position_m=(0.0, 0.0, 0.0),
        x_bounds_m=(-1.0, 1.0),
        y_bounds_m=(-1.0, 1.0),
        z_bounds_m=(0.0, 0.0),
        speed_range_mps=(1.0, 2.0),
        seed=0,
    ),
    ManhattanGridMobilitySpec(
        origin_xy_m=(0.0, 0.0),
        block_size_m=10.0,
        grid_width=2,
        grid_height=2,
        altitude_m=0.0,
        speed_range_mps=(1.0, 2.0),
        seed=0,
    ),
    NetworkRouteMobilitySpec(seed=0),
    RandomOrientationSpec(seed=0),
    GroupDeviationSpec(max_right_m=1.0, seed=0),
)


@pytest.mark.parametrize("model", SEEDED_MODELS, ids=lambda model: model.__class__.__name__)
def test_all_canonical_seed_fields_share_the_supported_seed_range(model) -> None:
    values = model.model_dump(mode="python")

    for seed in (0, MAX_RANDOM_SEED):
        values["seed"] = seed
        assert type(model).model_validate(values).seed == seed

    for seed in (-1, MAX_RANDOM_SEED + 1):
        values["seed"] = seed
        with pytest.raises(ValidationError):
            type(model).model_validate(values)


def test_shortest_path_network_route_still_allows_an_omitted_seed() -> None:
    mobility = NetworkRouteMobilitySpec(
        route="shortest_path",
        seed=None,
        start_node="A",
        end_node="B",
    )

    assert mobility.seed is None
