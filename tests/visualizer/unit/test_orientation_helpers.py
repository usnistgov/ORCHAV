"""Regression tests for orientation helper robustness with numpy-backed state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tests.visualizer.fixtures.mock_factories import make_mock_visualizer
from visualizer.src.model import RenderObjectState
from visualizer.src.scene import orientation_helpers
from visualizer.src.services.object_identity import (
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from visualizer.src.types.render_payloads import OrientationFramePayload


def test_create_orientation_frames_handles_numpy_current_positions():
    """create_orientation_frames should not use truthiness on numpy arrays."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
    mock_viz.current_tx_positions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    mock_viz.current_rx_positions = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64)
    mock_viz.cache_service = Mock()
    mock_viz.cache_service.get_frame.return_value = {
        "tx_orientations": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        "rx_orientations": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        "targets_metadata": [],
        "num_tx": 1,
    }
    with (
        patch.object(orientation_helpers, "update_tx_orientation_frames") as update_tx,
        patch.object(orientation_helpers, "update_rx_orientation_frames") as update_rx,
        patch.object(orientation_helpers, "update_target_orientation_frames") as update_target,
    ):
        orientation_helpers.create_orientation_frames(mock_viz, 0)

    update_tx.assert_called_once()
    update_rx.assert_called_once()
    update_target.assert_called_once()


def test_visualizer_orientation_normalization_handles_numpy_current_positions():
    """Node orientation normalization should accept numpy position arrays."""
    dummy = SimpleNamespace(
        current_tx_positions=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64),
        current_rx_positions=np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64),
        tx_markers=[object(), object()],
        rx_markers=[object()],
    )

    tx_norm, rx_norm = orientation_helpers.normalize_node_orientation_lists(
        dummy,
        tx_values=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        rx_values=[[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]],
        frame_data={"num_tx": 2},
    )

    assert len(tx_norm) == 2
    assert len(rx_norm) == 1


def test_update_tx_orientation_frames_uses_neutral_render_handles():
    """TX orientation frames are stable render handles, not native coordinate frames."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mock_viz.tx_orientation_frames = []
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]

    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    frame = mock_viz.tx_orientation_frames[0]
    assert isinstance(frame, RenderObjectState)
    assert frame.id == make_node_geometry_name("tx", 0, "orientation_frame")
    np.testing.assert_allclose(frame.world_transform.translation, [1.0, 2.0, 3.0])
    assert mock_viz.renderer.ensure_object.call_args.args[0].id == frame.id
    mock_viz.renderer.create_coordinate_frame.assert_not_called()
    mock_viz.renderer.update_geometry_in_visualizer.assert_not_called()


def test_update_tx_orientation_frames_ensures_one_complete_effective_snapshot():
    """The frame hot path does not follow an ensure with transform or visibility calls."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mock_viz.tx_orientation_frames = []
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
    mock_viz.node_service = SimpleNamespace(orientation_frame_visible=Mock(return_value=True))

    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])
    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    assert mock_viz.renderer.ensure_object.call_count == 2
    assert all(
        call.args[0].visible is True for call in mock_viz.renderer.ensure_object.call_args_list
    )
    mock_viz.renderer.set_transform.assert_not_called()
    mock_viz.renderer.set_visible.assert_not_called()


def test_failed_orientation_ensure_does_not_fall_through_to_partial_updates():
    """A failed complete snapshot is not masked by backend-shaped setters."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mock_viz.tx_orientation_frames = []
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
    mock_viz.node_service = SimpleNamespace(orientation_frame_visible=Mock(return_value=True))
    mock_viz.renderer.ensure_object = Mock(return_value=False)

    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    mock_viz.renderer.ensure_object.assert_called_once()
    frame_id = make_node_geometry_name("tx", 0, "orientation_frame")
    assert frame_id in mock_viz._pending_orientation_frame_syncs
    mock_viz.renderer.set_transform.assert_not_called()
    mock_viz.renderer.set_visible.assert_not_called()

    mock_viz.renderer.ensure_object.return_value = True
    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    assert frame_id not in mock_viz._pending_orientation_frame_syncs


def test_failed_orientation_removal_remains_pending_until_retry_succeeds():
    mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    orientations = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
    orientation_helpers.update_tx_orientation_frames(mock_viz, orientations)
    stale_id = make_node_geometry_name("tx", 1, "orientation_frame")
    failed_once = False

    def _remove(render_id):
        nonlocal failed_once
        if render_id == stale_id and not failed_once:
            failed_once = True
            return False
        return True

    mock_viz.renderer.remove_object = Mock(side_effect=_remove)

    orientation_helpers.update_tx_orientation_frames(mock_viz, orientations[:1])

    assert len(mock_viz.tx_orientation_frames) == 1
    assert stale_id in mock_viz._pending_orientation_frame_removals

    orientation_helpers.update_tx_orientation_frames(mock_viz, orientations[:1])

    assert stale_id not in mock_viz._pending_orientation_frame_removals
    assert mock_viz.renderer.remove_object.call_args_list[-1].args == (stale_id,)


def test_update_tx_orientation_frames_records_benchmark_metrics():
    """Benchmark mode should expose orientation-frame sync counters."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mock_viz.pipeline = SimpleNamespace(benchmark_recorder=object())
    mock_viz.tx_orientation_frames = []
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]

    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    metrics = mock_viz._orientation_frame_breakdown
    assert metrics["orientation_frame_sync_count"] == 1.0
    assert metrics["tx_orientation_frame_sync_count"] == 1.0
    assert metrics["orientation_frame_sync_ms"] >= 0.0


def test_update_tx_orientation_frames_refreshes_existing_handle_size():
    """Changing the orientation scale updates the stable handle payload in place."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mock_viz.tx_orientation_frames = []
    mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
    mock_viz.orientation_scale = 2.0

    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])
    frame = mock_viz.tx_orientation_frames[0]
    first_payload = frame.payload

    mock_viz.orientation_scale = 5.0
    orientation_helpers.update_tx_orientation_frames(mock_viz, [[0.0, 0.0, 0.0]])

    assert mock_viz.tx_orientation_frames[0] is frame
    assert frame.payload is not first_payload
    assert isinstance(frame.payload, OrientationFramePayload)
    assert frame.metadata["size"] == pytest.approx(5.0)
    assert frame.payload.size == pytest.approx(5.0)
    assert frame.payload.thickness == pytest.approx(frame.metadata["thickness"])
    assert frame.metadata["pickable"] is False


def test_selected_nonzero_tx_keeps_canonical_orientation_identity() -> None:
    """Selection changes visibility, not the index space of node frame data."""
    mock_viz = make_mock_visualizer(tx_count=3, rx_count=0)
    mock_viz.app_state.selected_tx = 2
    mock_viz.current_tx_positions = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    orientations = [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    orientation_helpers.update_tx_orientation_frames(mock_viz, orientations)

    selected_frame = mock_viz.tx_orientation_frames[2]
    assert selected_frame.id == make_node_geometry_name("tx", 2, "orientation_frame")
    np.testing.assert_allclose(selected_frame.world_transform.translation, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(
        selected_frame.world_transform.matrix,
        orientation_helpers.create_orientation_transform(
            [3.0, 0.0, 0.0],
            *orientations[2],
        ),
    )


def test_update_target_orientation_frames_uses_target_render_id():
    """Target orientation frames are keyed beside the target entry identity."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    mesh = RenderObjectState(
        id="target:walker::mesh",
        payload=mock_viz.tx_markers[0].payload,
    )
    mock_viz.target_entries = [
        {
            "name": "Walker",
            "target_name": "Walker",
            "mesh": mesh,
            "position": [4.0, 5.0, 6.0],
        }
    ]

    orientation_helpers.update_target_orientation_frames(
        mock_viz,
        {"Walker": [0.0, 0.0, 0.0]},
    )

    frame = mock_viz.target_orientation_frames[0]
    assert isinstance(frame, RenderObjectState)
    assert frame.id == make_target_entry_geometry_name(
        mock_viz.target_entries[0],
        "orientation_frame",
    )
    np.testing.assert_allclose(frame.world_transform.translation, [4.0, 5.0, 6.0])


def test_target_orientation_frames_match_names_when_metadata_order_differs() -> None:
    """Target frame identity follows target names instead of metadata position."""
    mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
    payload = mock_viz.tx_markers[0].payload
    walker = {
        "name": "Walker",
        "target_name": "Walker",
        "mesh": RenderObjectState(id="target:walker::mesh", payload=payload),
        "position": [1.0, 2.0, 3.0],
    }
    car = {
        "name": "Car",
        "target_name": "Car",
        "mesh": RenderObjectState(id="target:car::mesh", payload=payload),
        "position": [4.0, 5.0, 6.0],
    }
    mock_viz.target_entries = [walker, car]
    orientations = {
        "Car": [0.7, 0.8, 0.9],
        "Walker": [0.1, 0.2, 0.3],
    }

    orientation_helpers.update_target_orientation_frames(mock_viz, orientations)

    walker_frame, car_frame = mock_viz.target_orientation_frames
    assert walker_frame.id == make_target_entry_geometry_name(walker, "orientation_frame")
    assert car_frame.id == make_target_entry_geometry_name(car, "orientation_frame")
    np.testing.assert_allclose(
        walker_frame.world_transform.matrix,
        orientation_helpers.create_orientation_transform(
            walker["position"],
            *orientations["Walker"],
        ),
    )
    np.testing.assert_allclose(
        car_frame.world_transform.matrix,
        orientation_helpers.create_orientation_transform(
            car["position"],
            *orientations["Car"],
        ),
    )
