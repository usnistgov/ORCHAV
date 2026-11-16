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
but defines RX mobility in actors.py so generation and preview share the
same calculated trajectory.

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
from shared.scenarios.scripted import load_scripted_actors

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the scripted hello world example."""
    scenario_path = Path(__file__).parent
    scenario = load_scenario_configuration(scenario_path, project_root=PROJECT_ROOT)
    simulation_config = build_simulation_config(scenario)

    actors = load_scripted_actors(scenario)

    output_file = perform_pipeline(
        simulation_config=simulation_config,
        scenario_configuration=scenario,
        actors=actors,
    )

    logger.info("Output: %s", output_file)


if __name__ == "__main__":
    main()
