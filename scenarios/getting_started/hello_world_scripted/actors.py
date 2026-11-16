"""Calculated actors shared by generation and Local What-if Preview."""

import numpy as np

from shared.scenarios import ScenarioConfiguration
from shared.scenarios.actors import (
    ActorsSpec,
    LookAtOrientationSpec,
    RxActorSpec,
    StationaryMobilitySpec,
    TxActorSpec,
    WaypointMobilitySpec,
)

TX_POSITION = (-114.0, 37.0, 30.0)
RX_START_POSITION = (-27.18, -12.61, 1.5)


def _arc_sweep_points(
    steps: int,
    *,
    start_m: tuple[float, float, float] = RX_START_POSITION,
    center_xy_m: tuple[float, float] = TX_POSITION[:2],
    sweep_deg: float = 80.0,
    radial_bow_m: float = 18.0,
) -> tuple[tuple[float, float, float], ...]:
    """Sample the curved RX sweep at every scenario timeline step."""
    if steps < 2:
        raise ValueError("The scripted curved sweep requires at least two timeline steps")

    origin = np.asarray(start_m, dtype=np.float64)
    center_xy = np.asarray(center_xy_m, dtype=np.float64)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    eased_t = t * t * (3.0 - 2.0 * t)

    offset = origin[:2] - center_xy
    start_radius = max(float(np.linalg.norm(offset)), 1.0)
    start_angle = float(np.arctan2(offset[1], offset[0]))
    angle = start_angle + np.deg2rad(float(sweep_deg)) * eased_t
    radius = start_radius + float(radial_bow_m) * np.sin(np.pi * eased_t)

    x = center_xy[0] + radius * np.cos(angle)
    y = center_xy[1] + radius * np.sin(angle)
    z = np.full_like(x, origin[2])
    return tuple((float(px), float(py), float(pz)) for px, py, pz in zip(x, y, z))


def build_actors(scenario: ScenarioConfiguration) -> ActorsSpec:
    """Build the curved RX sweep and its look-at orientation from the timeline."""
    return ActorsSpec(
        tx=(
            TxActorSpec(
                name="TX1",
                mobility=StationaryMobilitySpec(position_m=TX_POSITION),
            ),
        ),
        rx=(
            RxActorSpec(
                name="RX1",
                mobility=WaypointMobilitySpec(
                    points_m=_arc_sweep_points(scenario.timeline.steps),
                    interpolation="linear",
                ),
                orientation=LookAtOrientationSpec(actor="TX1"),
            ),
        ),
    )
