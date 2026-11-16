"""Frame-free sensing overviews seed the ordinary scene radio labels."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.sensing.volume_scene import initialize_scene_radios


def scene_only_visualizer():
    tx = SimpleNamespace(mobility=SimpleNamespace(type="stationary", position_m=(-6, 0, 2)))
    rx = SimpleNamespace(mobility=SimpleNamespace(type="stationary", position_m=(1, 2, 3)))
    return SimpleNamespace(
        _scene_only_mode=True,
        scenario_config=SimpleNamespace(actors=SimpleNamespace(tx=[tx], rx=[rx])),
        node_service=Mock(),
        scene_service=Mock(),
        ui_manager=Mock(),
    )


def test_volume_radios_use_authored_positions_and_existing_label_service():
    viz = scene_only_visualizer()
    initialize_scene_radios(viz)
    assert (viz.num_tx, viz.num_rx) == (1, 1)
    viz.node_service.create_tx_rx_markers.assert_called_once_with()
    viz.node_service.update_available_tx_rx_from_frame.assert_called_once_with(1, 1)
    tx, rx = viz.node_service.update_tx_rx_positions.call_args.args
    np.testing.assert_array_equal(tx, [[-6, 0, 2]])
    np.testing.assert_array_equal(rx, [[1, 2, 3]])
    viz.ui_manager.set_panel_visible.assert_called_once_with("nodes", True)
    viz.scene_service.remove_empty_scene_anchor.assert_called_once_with()


@pytest.mark.parametrize("case", ["frames", "moving", "no_radios"])
def test_overview_never_replaces_frame_poses_or_invents_moving_positions(case):
    viz = scene_only_visualizer()
    if case == "frames":
        viz._scene_only_mode = False
    elif case == "moving":
        viz.scenario_config.actors.tx[0].mobility.type = "linear"
    else:
        viz.scenario_config.actors.tx = []
    initialize_scene_radios(viz)
    assert not viz.node_service.mock_calls
    assert not viz.ui_manager.mock_calls
