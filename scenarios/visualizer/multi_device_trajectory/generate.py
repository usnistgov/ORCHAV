#!/usr/bin/env python3
"""Generate the Etoile multi-device trajectory visualizer scenario.

The scenario demonstrates:
- 1 rooftop transmitter near the Arc de Triomphe
- 3 receivers:
    - descending airborne receiver around the pedestrian
    - high-altitude airborne receiver sweeping beside the pedestrian path
    - ground patrol receiver moving parallel to the pedestrian
- 1 moving mesh target used as a visual reference and scatterer

Usage:
    python generate.py
"""

import logging
import math
import sys
from pathlib import Path

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
    DirectoryAssetSpec,
    FixedOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    RxActorSpec,
    StationaryMobilitySpec,
    TargetActorSpec,
    TxActorSpec,
    WaypointMobilitySpec,
)

logger = logging.getLogger(__name__)

# Pedestrian path along an Etoile avenue.
PEDESTRIAN_START = (-21.858, -21.565, 1.6)
PEDESTRIAN_END = (135.386, -98.332, 1.6)

# Direction vector of the pedestrian path (normalised)
_dx = PEDESTRIAN_END[0] - PEDESTRIAN_START[0]
_dy = PEDESTRIAN_END[1] - PEDESTRIAN_START[1]
_path_len = math.hypot(_dx, _dy)
_ux, _uy = _dx / _path_len, _dy / _path_len  # unit along path
_nx, _ny = -_uy, _ux  # unit normal (left of path)

# Airborne receiver 1: descending spiral around the pedestrian.
DRONE1_RADIUS = 25.0
DRONE1_ALT_START = 35.0
DRONE1_ALT_END = 3.0
DRONE1_ORBITS = 2.0

# Airborne receiver 2: high-altitude waypoint sweep offset from the path.
DRONE2_ALTITUDE = 40.0
DRONE2_OFFSET = 30.0  # metres to the right of the pedestrian path
LOOK_AT_SMOOTHING_TIME_S = 0.1


def _descending_spiral_waypoints(n_waypoints: int = 17) -> list:
    """Waypoints for a descending spiral centred on the pedestrian path."""
    waypoints = []
    for i in range(n_waypoints):
        t = i / (n_waypoints - 1)
        cx = PEDESTRIAN_START[0] + t * (PEDESTRIAN_END[0] - PEDESTRIAN_START[0])
        cy = PEDESTRIAN_START[1] + t * (PEDESTRIAN_END[1] - PEDESTRIAN_START[1])
        angle = 2.0 * math.pi * DRONE1_ORBITS * t
        alt = DRONE1_ALT_START + t * (DRONE1_ALT_END - DRONE1_ALT_START)
        x = cx + DRONE1_RADIUS * math.cos(angle)
        y = cy + DRONE1_RADIUS * math.sin(angle)
        waypoints.append((x, y, alt))
    return waypoints


def _high_sweep_waypoints() -> list:
    """Waypoints for a high-altitude sweep offset to the right of the path."""
    fractions = [0.0, 0.33, 0.66, 1.0]
    waypoints = []
    for t in fractions:
        cx = PEDESTRIAN_START[0] + t * _dx
        cy = PEDESTRIAN_START[1] + t * _dy
        # Offset to the right of the path direction
        x = cx - _nx * DRONE2_OFFSET
        y = cy - _ny * DRONE2_OFFSET
        waypoints.append((x, y, DRONE2_ALTITUDE))
    return waypoints


def _patrol_endpoints() -> tuple:
    """Start/end for a ground patrol parallel to the pedestrian, offset left."""
    offset = 15.0  # metres to the left of the pedestrian path
    sx = PEDESTRIAN_START[0] + _nx * offset
    sy = PEDESTRIAN_START[1] + _ny * offset
    ex = PEDESTRIAN_END[0] + _nx * offset
    ey = PEDESTRIAN_END[1] + _ny * offset
    return (sx, sy, 1.5), (ex, ey, 1.5)


def main() -> None:
    """Run the multi-device trajectory scenario."""
    scenario_path = Path(__file__).parent
    scenario = load_scenario_configuration(scenario_path, project_root=PROJECT_ROOT)
    simulation_config = build_simulation_config(scenario)

    patrol_start, patrol_end = _patrol_endpoints()
    patrol_yaw = math.degrees(math.atan2(_uy, _ux))

    actors = ActorsSpec(
        tx=(
            TxActorSpec(
                name="gNB",
                mobility=StationaryMobilitySpec(position_m=(166.171, -64.377, 13.489)),
                orientation=FixedOrientationSpec(pitch_deg=-30.0),
            ),
        ),
        rx=(
            RxActorSpec(
                name="Drone_Tracker",
                mobility=WaypointMobilitySpec(points_m=tuple(_descending_spiral_waypoints())),
                orientation=LookAtOrientationSpec(
                    actor="Pedestrian",
                    smoothing_time_s=LOOK_AT_SMOOTHING_TIME_S,
                ),
            ),
            RxActorSpec(
                name="Drone_Overwatch",
                mobility=WaypointMobilitySpec(points_m=tuple(_high_sweep_waypoints())),
                orientation=LookAtOrientationSpec(
                    actor="Pedestrian",
                    smoothing_time_s=LOOK_AT_SMOOTHING_TIME_S,
                ),
            ),
            RxActorSpec(
                name="Patrol",
                mobility=LinearMobilitySpec(start_m=patrol_start, end_m=patrol_end),
                orientation=FixedOrientationSpec(yaw_deg=patrol_yaw),
            ),
        ),
        targets=(
            TargetActorSpec(
                name="Pedestrian",
                asset=DirectoryAssetSpec(
                    path="libraries/targets/nist_human_walking",
                    pattern="fitted_Image_Psm_01_000*.ply",
                    material_type="glass",
                    scale=1.0,
                ),
                mobility=LinearMobilitySpec(
                    start_m=PEDESTRIAN_START,
                    end_m=PEDESTRIAN_END,
                ),
                orientation=FixedOrientationSpec(yaw_deg=180.0),
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
