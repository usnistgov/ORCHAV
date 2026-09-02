"""
Synthetic test fixtures for visualizer tests.

Provides portable, minimal test data that doesn't depend on hardcoded paths.
All fixtures are designed to be < 1MB for fast test execution.
"""

from typing import Any, Dict

import numpy as np


def create_synthetic_frame_data(
    num_tx: int = 2,
    num_rx: int = 2,
    num_targets: int = 1,
    num_tx_rx_pairs: int = 4,
    num_reflections: int = 3,
    include_orientations: bool = True,
    include_coverage: bool = False,
) -> Dict[str, Any]:
    """
    Create synthetic frame data for testing.

    Args:
        num_tx: Number of transmitters
        num_rx: Number of receivers
        num_targets: Number of targets
        num_tx_rx_pairs: Number of TX-RX path pairs
        num_reflections: Max reflection order
        include_orientations: Include orientation data
        include_coverage: Include coverage map data

    Returns:
        Dictionary with frame data matching expected format
    """
    # TX positions (random but deterministic)
    np.random.seed(42)
    tx_positions = np.random.uniform(-10, 10, (num_tx, 3))
    rx_positions = np.random.uniform(-10, 10, (num_rx, 3))

    # Target positions
    target_positions = np.random.uniform(-5, 5, (num_targets, 3))
    target_names = [f"target_{i+1}" for i in range(num_targets)]

    # MPC data - simplified
    # Create path vertices for each TX-RX pair
    all_vertices = []
    all_interactions = []
    all_delays = []
    all_path_losses = []

    for pair_idx in range(num_tx_rx_pairs):
        # Create some paths for this pair (e.g., 5-10 paths)
        num_paths = np.random.randint(5, 11)
        pair_vertices = []
        pair_interactions = []
        pair_delays = []
        pair_path_losses = []

        for path_idx in range(num_paths):
            # Number of bounces for this path (0 = LoS, 1-3 = reflections)
            num_bounces = np.random.randint(0, num_reflections + 1)

            # Create vertices (TX -> bounces -> RX)
            path_vertices = np.random.uniform(-10, 10, (num_bounces + 2, 3))
            pair_vertices.append(path_vertices)

            # Interaction types (0=LoS, 1=Specular, 2=Diffuse)
            interactions = np.random.randint(0, 3, num_bounces + 1)
            pair_interactions.append(interactions)

            # Delays and path losses
            pair_delays.append(np.random.uniform(0, 100))
            pair_path_losses.append(np.random.uniform(30, 80))

        all_vertices.append(pair_vertices)
        all_interactions.append(pair_interactions)
        all_delays.append(pair_delays)
        all_path_losses.append(pair_path_losses)

    frame_data = {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "tx_names": [f"tx_{i+1}" for i in range(num_tx)],
        "rx_names": [f"rx_{i+1}" for i in range(num_rx)],
        "target_positions": target_positions,
        "target_names": target_names,
        "targets_metadata": [
            {
                "name": name,
                "current_position": pos.tolist(),
                "mesh_path": f"/fake/path/to/{name}.obj",
                "scale": 1.0,
                "orientation": [0.0, 0.0, 0.0],
            }
            for name, pos in zip(target_names, target_positions)
        ],
        # MPC data
        "all_padded_vertices": all_vertices,
        "all_padded_interactions": all_interactions,
        "all_delays": all_delays,
        "all_path_losses": all_path_losses,
        "tx_rx_pairs": [(i % num_tx, i % num_rx) for i in range(num_tx_rx_pairs)],
    }

    if include_orientations:
        frame_data["tx_orientations"] = np.random.uniform(-np.pi, np.pi, (num_tx, 3))
        frame_data["rx_orientations"] = np.random.uniform(-np.pi, np.pi, (num_rx, 3))
        frame_data["target_orientations"] = np.random.uniform(-np.pi, np.pi, (num_targets, 3))

    if include_coverage:
        # Simplified coverage map (10x10 grid)
        frame_data["coverage_data"] = {
            "grid_x": np.linspace(-10, 10, 10),
            "grid_y": np.linspace(-10, 10, 10),
            "values": np.random.uniform(-80, -40, (10, 10)),
            "height_levels": [0.0, 1.5, 3.0],
        }

    return frame_data


def create_minimal_xml_scene() -> str:
    """Create a minimal XML scene for testing."""
    return """<?xml version="1.0" encoding="utf-8"?>
<scene version="0.6.0">
    <shape type="obj" id="building_1">
        <string name="filename" value="building1.obj"/>
        <transform name="to_world">
            <scale value="1.0"/>
            <translate x="0" y="0" z="0"/>
        </transform>
        <bsdf type="diffuse" id="mat_concrete">
            <rgb name="reflectance" value="0.5 0.5 0.5"/>
        </bsdf>
    </shape>

    <shape type="obj" id="building_2">
        <string name="filename" value="building2.obj"/>
        <transform name="to_world">
            <scale value="1.0"/>
            <translate x="5" y="5" z="0"/>
        </transform>
        <bsdf type="diffuse" id="mat_concrete"/>
    </shape>
</scene>"""


def create_synthetic_coverage_map(grid_size: int = 10) -> Dict[str, Any]:
    """
    Create synthetic coverage map data for testing.

    Args:
        grid_size: Size of grid (grid_size x grid_size)

    Returns:
        Dictionary with coverage map data
    """
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)

    # Create 3 height levels with different values
    height_levels = [0.0, 1.5, 3.0]
    coverage_values = {}

    for height in height_levels:
        # Simulate some pattern (e.g., distance-based falloff)
        xx, yy = np.meshgrid(x, y)
        distance = np.sqrt(xx**2 + yy**2)
        values = -40 - 2 * distance + np.random.normal(0, 3, (grid_size, grid_size))
        coverage_values[height] = values

    return {
        "grid_x": x,
        "grid_y": y,
        "height_levels": height_levels,
        "coverage_values": coverage_values,
        "metadata": {
            "frequency_ghz": 3.5,
            "tx_power_dbm": 30,
            "scenario": "urban",
        },
    }
