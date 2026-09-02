#!/usr/bin/env python3
"""
Hello World Scripted - Custom Mobility Example
==============================================

Demonstrates the Python-scripted scenario path with a trajectory computed from
an arbitrary equation:

- 1 stationary TX (transmitter) using the default fixed orientation
- 1 RX following a custom curved sweep trajectory while looking at the TX
- No targets

This example keeps the same Etoile scene as the default Hello World scenario,
but defines RX mobility in Python so the trajectory can be computed from an
arbitrary equation.

Workflow:
1. Load scenario configuration from YAML
2. Compute the RX path in Python
3. Build immutable actor specifications
4. Run the ray tracing pipeline
5. Output frames to HDF5 files

Usage:
    python generate.py
    # Or from project root:
    python scenarios/getting_started/hello_world_scripted/generate.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Direct scripts must prefer the checkout containing the script over an
# unrelated editable ORCHAV installation in the active environment.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_project_root_entry = str(PROJECT_ROOT)
if _project_root_entry in sys.path:
    sys.path.remove(_project_root_entry)
sys.path.insert(0, _project_root_entry)

from generator import (
    build_simulation_config,
    perform_pipeline,
)
from shared.scenarios import load_scenario_configuration
from shared.scenarios.actors import (
    ActorsSpec,
    LookAtOrientationSpec,
    RxActorSpec,
    StationaryMobilitySpec,
    TxActorSpec,
    WaypointMobilitySpec,
)

logger = logging.getLogger(__name__)

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


def main() -> None:
    """Run the scripted hello world example."""
    scenario_path = Path(__file__).parent
    scenario = load_scenario_configuration(scenario_path, project_root=PROJECT_ROOT)
    simulation_config = build_simulation_config(scenario)

    actors = ActorsSpec(
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

    output_file = perform_pipeline(
        simulation_config=simulation_config,
        scenario_configuration=scenario,
        actors=actors,
    )

    logger.info("Output: %s", output_file)


if __name__ == "__main__":
    main()
