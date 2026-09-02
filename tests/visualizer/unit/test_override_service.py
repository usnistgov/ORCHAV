"""Tests for runtime frame overrides and panel synchronization."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.types import StandardMPCFrame
from tests.visualizer.fixtures.mock_factories import make_mock_visualizer
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.services.cache_service import CacheInvalidationScope
from visualizer.src.services.override_service import OverrideService


def _override_frame() -> StandardMPCFrame:
    """Build one canonical override result with nonzero node orientations."""
    return standard_mpc_frame_from_pair_data(
        frame_index=3,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        tx_positions=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
        rx_positions=np.array([[2.0, 2.0, 2.0]], dtype=np.float64),
        tx_orientations=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        rx_orientations=np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        tx_names=("TX1",),
        rx_names=("RX1",),
        vertices_by_pair=[np.array([[[1.5, 1.5, 1.5]]], dtype=np.float32)],
        interactions_by_pair=[np.array([[1]], dtype=np.uint8)],
        path_lengths_by_pair=[np.array([1], dtype=np.int64)],
    )


class TestOverrideService:
    """Test OverrideService functionality."""

    def test_apply_position_overrides_caches_visual_payload_and_updates_node_state(self):
        """The override result reaches redraw as one complete visual payload."""
        mock_viz = make_mock_visualizer()
        mock_viz.app_state.step = 3
        mock_viz.animation_step = 3
        mock_viz.panel_manager = None
        mock_viz.sensing_panel = None
        mock_viz.extension_services = {}
        mock_viz.app_state.extension_state = {}
        mock_viz.mpc_core = SimpleNamespace(canon_points_dtype=np.dtype(np.float32))

        mock_viz.frame_source = Mock()
        updated_frame = _override_frame()
        mock_viz.frame_source.load_frame_with_overrides = Mock(return_value=updated_frame)
        mock_viz.cache_service = Mock()
        mock_viz.node_service = Mock()
        mock_panel = Mock()
        mock_viz.ui_manager.panels = {"overrides": mock_panel}

        service = OverrideService(mock_viz)
        overrides = [{"name": "TX1", "position": [1, 1, 1]}]

        service.apply_position_overrides(overrides)

        mock_viz.frame_source.load_frame_with_overrides.assert_called_once_with(3, overrides)
        mock_viz.cache_service.store_frame.assert_called_once()
        cache_step, cached_payload = mock_viz.cache_service.store_frame.call_args.args
        assert cache_step == 3
        assert mock_viz.cache_service.store_frame.call_args.kwargs == {"source": "override"}
        assert isinstance(cached_payload, dict)
        assert isinstance(cached_payload["canonical_data"], CanonicalStepData)
        np.testing.assert_allclose(cached_payload["tx_orientations"], [[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(cached_payload["rx_orientations"], [[0.4, 0.5, 0.6]])
        mock_viz.cache_service.invalidate.assert_called_with(
            CacheInvalidationScope.MPC_RENDER_SETTINGS,
            reason="position_override",
        )
        mock_viz.cache_service.invalidate_canonical_step.assert_called_with(
            mock_viz.app_state.step,
            reason="position_override",
        )

        assert mock_viz.current_tx_positions == [[1.0, 1.0, 1.0]]
        assert mock_viz.current_rx_positions == [[2.0, 2.0, 2.0]]
        np.testing.assert_allclose(mock_viz.current_tx_orientations, [[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(mock_viz.current_rx_orientations, [[0.4, 0.5, 0.6]])
        mock_viz.node_service.update_tx_rx_positions.assert_called_once_with(
            [[1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0]],
        )
        mock_panel.set_busy.assert_called_with(False)
        mock_panel.update_objects.assert_called()
        assert mock_viz.force_update_next_frame is True
        mock_viz.schedule_update.assert_called()

    def test_apply_position_overrides_handles_failure(self):
        """Test handling of override computation failure."""
        mock_viz = make_mock_visualizer()

        mock_viz.frame_source = Mock()
        mock_viz.frame_source.load_frame_with_overrides = Mock(return_value=None)
        mock_panel = Mock()
        mock_viz.ui_manager.panels = {"overrides": mock_panel}

        service = OverrideService(mock_viz)
        service.apply_position_overrides([{}])

        mock_panel.set_status_text.assert_called_with("Override computation failed.", error=True)
        assert not mock_viz.schedule_update.called

    def test_update_override_panel_with_frame_data(self):
        """Test populating override panel from frame data."""
        mock_viz = make_mock_visualizer()
        mock_panel = Mock()
        mock_viz.ui_manager.panels = {"overrides": mock_panel}

        service = OverrideService(mock_viz)

        service.update_override_panel_with_frame_data(_override_frame())

        mock_panel.update_objects.assert_called()
        call_args = mock_panel.update_objects.call_args[0][0]

        assert "TX1" in call_args
        assert call_args["TX1"]["position"] == [1.0, 1.0, 1.0]
        assert "RX1" in call_args
        assert call_args["RX1"]["position"] == [2.0, 2.0, 2.0]

    def test_extract_objects_from_frame_uses_canonical_orientation_state(self):
        """Canonical node orientations populate the editable panel rows."""
        mock_viz = make_mock_visualizer()
        service = OverrideService(mock_viz)

        objects = service._extract_objects_from_frame(_override_frame())

        assert "TX1" in objects
        assert objects["TX1"]["position"] == [1.0, 1.0, 1.0]
        assert objects["TX1"]["orientation"] == [0.1, 0.2, 0.3]
        assert objects["RX1"]["orientation"] == [0.4, 0.5, 0.6]
