"""
Test suite for NodeService.

Tests TX/RX marker management, orientation handling, target processing,
and node visibility/coloring logic.
"""

from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

pytestmark = pytest.mark.usefixtures("mock_node_service_label")

# Import test fixtures
from tests.visualizer.fixtures.mock_factories import (
    make_mock_visualizer,
)
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.model import (
    RenderObjectState,
    make_text_label_state,
    render_state_points,
)

# Import the service under test
from visualizer.src.services.cache_service import CacheService
from visualizer.src.services.node_service import NodeService
from visualizer.src.services.object_identity import (
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from visualizer.src.types.render_payloads import MeshPayload, TextLabelPayload


class _ColorGeometry:
    def __init__(self, center=(0.0, 0.0, 0.0), color=(1.0, 1.0, 1.0)):
        self._center = np.asarray(center, dtype=float)
        self.vertex_colors = np.asarray([color], dtype=float)

    def get_center(self):
        return self._center.copy()

    def paint_uniform_color(self, color):
        self.vertex_colors = np.asarray([color], dtype=float)

    def translate(self, delta):
        self._center += np.asarray(delta, dtype=float)


def _payload_radius(payload) -> float:
    vertices = np.asarray(payload.vertices, dtype=float)
    return float(np.max(np.linalg.norm(vertices, axis=1)))


def _payload_extent(payload) -> np.ndarray:
    vertices = np.asarray(payload.vertices, dtype=float)
    return vertices.max(axis=0) - vertices.min(axis=0)


def _target_mesh(center=(0.0, 0.0, 0.0), name: str = "target_1") -> RenderObjectState:
    center_arr = np.asarray(center, dtype=np.float64)
    offsets = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return RenderObjectState(
        id=f"target:{name}::mesh",
        payload=MeshPayload(
            vertices=center_arr + offsets,
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _label_state(
    label_id: str,
    *,
    text: str = "Label",
    color=(0.8, 0.8, 0.8),
    position=(0.0, 0.0, 2.0),
    visible: bool = True,
) -> RenderObjectState:
    return make_text_label_state(
        label_id,
        text,
        color,
        position=position,
        visible=visible,
    )


def _node_label_state(
    kind: str,
    index: int,
    *,
    position=(0.0, 0.0, 2.0),
    visible: bool = True,
) -> RenderObjectState:
    is_tx = str(kind).lower() == "tx"
    return _label_state(
        make_node_geometry_name(kind, index, "label"),
        text=f"{'TX' if is_tx else 'RX'}{index + 1}",
        color=[1.0, 0.0, 0.0] if is_tx else [0.0, 0.0, 1.0],
        position=position,
        visible=visible,
    )


def _ensured_object(mock_viz, object_id: str):
    matches = _ensured_objects(mock_viz, object_id)
    if matches:
        return matches[-1]
    raise AssertionError(f"render object {object_id!r} was not ensured")


def _ensured_objects(mock_viz, object_id: str):
    return [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == object_id
    ]


def _assert_label_ensured(mock_viz, label_id: str, label) -> None:
    assert isinstance(label, RenderObjectState)
    ensured = _ensured_object(mock_viz, label_id)
    assert isinstance(ensured.payload, TextLabelPayload)
    assert ensured.payload is label.payload


class TestNodeServiceInit:
    """Test NodeService initialization."""

    def test_init_rejects_missing_application_target_owner(self):
        mock_viz = make_mock_visualizer()
        del mock_viz.target_service

        with pytest.raises(
            ValueError,
            match="application-composed TargetService owner",
        ):
            NodeService(mock_viz)


class TestTxRxMarkerManagement:
    """Test TX/RX marker creation and management."""

    def test_create_tx_rx_markers_creates_correct_count(self):
        """Test that create_tx_rx_markers creates the right number of markers."""
        mock_viz = make_mock_visualizer(tx_count=3, rx_count=2)
        mock_viz.num_tx = 3
        mock_viz.num_rx = 2
        mock_viz.tx_markers = []
        mock_viz.rx_markers = []
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []

        service = NodeService(mock_viz)
        service.create_tx_rx_markers()

        assert len(mock_viz.tx_markers) == 3
        assert len(mock_viz.rx_markers) == 2
        assert len(mock_viz.tx_labels) == 3
        assert len(mock_viz.rx_labels) == 2
        assert all(isinstance(marker, RenderObjectState) for marker in mock_viz.tx_markers)
        assert mock_viz.tx_markers[0].id == make_node_geometry_name("tx", 0, "marker")

    def test_create_tx_rx_markers_removes_old_entities(self):
        """Test that old marker IDs are removed before creating new ones."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=2)
        mock_viz.num_tx = 2
        mock_viz.num_rx = 2

        old_tx_marker = Mock()
        old_rx_marker = Mock()
        mock_viz.tx_markers = [old_tx_marker]
        mock_viz.rx_markers = [old_rx_marker]
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.rx_labels = [_node_label_state("rx", 0)]

        service = NodeService(mock_viz)
        service.create_tx_rx_markers()

        mock_viz.renderer.remove_object.assert_any_call(make_node_geometry_name("tx", 0, "marker"))
        mock_viz.renderer.remove_object.assert_any_call(make_node_geometry_name("tx", 0, "label"))
        mock_viz.renderer.remove_object.assert_any_call(make_node_geometry_name("rx", 0, "marker"))
        mock_viz.renderer.remove_object.assert_any_call(make_node_geometry_name("rx", 0, "label"))

    def test_node_marker_handle_supports_box_payload(self):
        """Marker payloads can be primitives other than spheres."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.node_marker_config = {"tx": {"shape": "box", "center": True}}
        service = NodeService(mock_viz)

        handle = service._node_marker_handle("tx", 0, size=0.4)

        assert handle.id == make_node_geometry_name("tx", 0, "marker")
        assert handle.metadata["shape"] == "box"
        points = render_state_points(handle)
        np.testing.assert_allclose(points.min(axis=0), [-0.4, -0.4, -0.4])
        np.testing.assert_allclose(points.max(axis=0), [0.4, 0.4, 0.4])

    def test_node_marker_handle_supports_centered_custom_mesh(self, tmp_path):
        """Custom marker meshes are geometry-only; node coloring owns the color."""
        mesh_path = tmp_path / "antenna.obj"
        mesh_path.write_text(
            "\n".join(
                [
                    "v 10 0 0 0.2 0.3 0.4",
                    "v 11 0 0 0.2 0.3 0.4",
                    "v 10 1 0 0.2 0.3 0.4",
                    "f 1 2 3",
                ]
            ),
            encoding="utf-8",
        )
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.node_marker_config = {
            "tx": {
                "shape": "mesh",
                "mesh_path": mesh_path,
                "scale": 2.0,
                "center": True,
            }
        }
        service = NodeService(mock_viz)

        handle = service._node_marker_handle("tx", 0, size=0.4)

        assert handle.metadata["shape"] == "mesh"
        assert handle.payload.vertex_colors is None
        points = render_state_points(handle)
        np.testing.assert_allclose(points.min(axis=0), [-0.4, -0.4, 0.0])
        np.testing.assert_allclose(points.max(axis=0), [0.4, 0.4, 0.0])

    def test_node_anchor_uses_neutral_marker_without_querying_renderer(self):
        """Missing frame positions fall back only to application-owned marker state."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.current_tx_positions = []
        mock_viz.renderer.get_named_position = Mock(return_value=np.asarray([9.0, 8.0, 7.0]))
        service = NodeService(mock_viz)
        marker = service._node_marker_handle(
            "tx",
            0,
            position=np.asarray([1.0, 2.0, 3.0]),
        )

        np.testing.assert_allclose(
            service._resolve_node_anchor("tx", 0, marker),
            [1.0, 2.0, 3.0],
        )
        mock_viz.renderer.get_named_position.assert_not_called()

    def test_node_color_helpers_ignore_native_geometry_fallbacks(self):
        """Node marker color policy should not mutate renderer-native objects."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        service = NodeService(mock_viz)
        geometry = _ColorGeometry(color=(0.2, 0.3, 0.4))

        assert service._uniform_color(geometry, default=[1.0, 0.0, 0.0]) == [
            1.0,
            0.0,
            0.0,
        ]
        service._paint_geometry_color(geometry, [0.9, 0.8, 0.7])
        np.testing.assert_allclose(geometry.vertex_colors[0], [0.2, 0.3, 0.4])

    def test_ensure_tx_rx_markers_created_trims_excess_nodes(self):
        """Ensure stale TX/RX marker and label placeholders are removed."""
        mock_viz = make_mock_visualizer(tx_count=3, rx_count=3)
        service = NodeService(mock_viz)

        service.ensure_tx_rx_markers_created(1, 1)

        assert len(mock_viz.tx_markers) == 1
        assert len(mock_viz.rx_markers) == 1
        assert len(mock_viz.tx_labels) == 1
        assert len(mock_viz.rx_labels) == 1

    def test_ensure_tx_rx_markers_created_builds_missing_labels_when_hidden(self):
        """Labels should be created even when global label visibility is currently off."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.app_state.show_labels = False
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        service = NodeService(mock_viz)

        service.ensure_tx_rx_markers_created(1, 1)

        assert len(mock_viz.tx_labels) == 1
        assert len(mock_viz.rx_labels) == 1

    def test_ensure_tx_rx_markers_created_adds_labels_for_existing_markers(self):
        """Missing labels must be inserted for existing marker state."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        service = NodeService(mock_viz)

        service.ensure_tx_rx_markers_created(1, 1)

        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("tx", 0, "label"),
            mock_viz.tx_labels[0],
        )
        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("rx", 0, "label"),
            mock_viz.rx_labels[0],
        )

    def test_ensure_tx_rx_markers_created_syncs_without_open3d_vis(self):
        """Pygfx uses renderer entities even though visualizer.vis is None."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.vis_initialized = False
        mock_viz.vis = None
        mock_viz.renderer._initialized = True
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]
        service = NodeService(mock_viz)

        service.ensure_tx_rx_markers_created(1, 1)

        tx_object = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        rx_object = _ensured_object(mock_viz, make_node_geometry_name("rx", 0, "marker"))
        assert tx_object.visible is True
        assert rx_object.visible is True
        np.testing.assert_allclose(tx_object.transform.translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rx_object.transform.translation, [4.0, 5.0, 6.0])

    def test_ensure_tx_rx_markers_created_requires_common_object_surface(self):
        """Persistent node entities do not fall back to backend-shaped APIs."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.vis_initialized = False
        mock_viz.vis = None
        mock_viz.renderer._initialized = True
        del mock_viz.renderer.ensure_object
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]
        service = NodeService(mock_viz)

        service.ensure_tx_rx_markers_created(1, 1)

        mock_viz.renderer.ensure_named_geometry.assert_not_called()


class TestTxRxPositionUpdates:
    """Test TX/RX position update logic."""

    def test_update_tx_rx_positions_stores_positions(self):
        """Test that positions are stored in visualizer."""
        mock_viz = make_mock_visualizer()
        service = NodeService(mock_viz)

        tx_pos = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        rx_pos = [[7.0, 8.0, 9.0]]

        service.update_tx_rx_positions(tx_pos, rx_pos)

        assert mock_viz.current_tx_positions == tx_pos
        assert mock_viz.current_rx_positions == rx_pos


class TestNodeColoring:
    """Test node coloring logic."""

    def test_update_individual_coloring_legend_uses_lightweight_labels(self):
        """Build individual TX/RX legend entries without native widgets."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.node_coloring_mode = "individual"
        service = NodeService(mock_viz)

        service.apply_node_coloring()
        service.update_node_coloring_legend()

        labels = [call.args[0] for call in mock_viz.tx_rx_legend_layout.addWidget.call_args_list]
        assert [label.text() for label in labels] == ["TX1", "RX1"]
        assert labels[0].styleSheet().startswith("color: #e52d2d;")
        assert labels[1].styleSheet().startswith("color: #2de5e5;")

    def test_apply_node_coloring_per_type_mode(self):
        """Test per-type coloring (red TX, blue RX)."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=2)
        mock_viz.node_coloring_mode = "per_type"
        tx_markers = list(mock_viz.tx_markers)
        rx_markers = list(mock_viz.rx_markers)

        service = NodeService(mock_viz)
        service.apply_node_coloring()

        # Verify TX markers are red
        for marker in tx_markers:
            np.testing.assert_allclose(marker.material.base_color, [1.0, 0.0, 0.0, 1.0])

        # Verify RX markers are blue (consistent with marker creation)
        for marker in rx_markers:
            np.testing.assert_allclose(marker.material.base_color, [0.0, 0.0, 1.0, 1.0])

    def test_apply_node_coloring_individual_mode(self):
        """Test individual coloring mode generates distinct colors."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=2)
        mock_viz.node_coloring_mode = "individual"
        markers = list(mock_viz.tx_markers + mock_viz.rx_markers)

        service = NodeService(mock_viz)
        service.apply_node_coloring()

        # Verify colors were generated
        assert len(mock_viz.individual_node_colors) == 4  # 2 TX + 2 RX

        # Verify each marker got painted with unique color
        for marker in markers:
            color = np.asarray(marker.material.base_color[:3], dtype=float)
            assert np.any(color > 0.0)

    def test_apply_node_coloring_updates_renderer(self):
        """Test that coloring updates the renderer."""
        mock_viz = make_mock_visualizer()
        mock_viz.vis_initialized = True

        service = NodeService(mock_viz)
        service.apply_node_coloring()

        # Verify renderer was updated (via renderer, not vis directly)
        mock_viz.renderer.update_renderer.assert_called()

    def test_apply_node_coloring_updates_labels(self):
        """Test that labels are colored to match markers."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=2)
        mock_viz.node_coloring_mode = "per_type"
        mock_viz.tx_labels = [_node_label_state("tx", index) for index in range(2)]
        mock_viz.rx_labels = [_node_label_state("rx", index) for index in range(2)]

        service = NodeService(mock_viz)
        service.apply_node_coloring()

        # Verify labels match marker colors (TX=red, RX=blue)
        for label in mock_viz.tx_labels:
            assert label.material.base_color[:3] == (1.0, 0.0, 0.0)
        for label in mock_viz.rx_labels:
            assert label.material.base_color[:3] == (0.0, 0.0, 1.0)

    def test_apply_node_coloring_syncs_existing_marker_handles(self):
        """Entity sync publishes current marker handle colors."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.app_state.show_labels = False
        mock_viz.renderer.get_named_position = Mock(return_value=None)
        mock_viz.renderer.set_named_transform = Mock()
        mock_viz.renderer.invalidate_geometry_payload_cache = Mock()

        service = NodeService(mock_viz)

        mock_viz.node_coloring_mode = "individual"
        service.apply_node_coloring()
        mock_viz.node_coloring_mode = "per_type"
        service.apply_node_coloring()

        tx_object = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        rx_object = _ensured_object(mock_viz, make_node_geometry_name("rx", 0, "marker"))
        np.testing.assert_allclose(
            tx_object.material_payload.base_color,
            [1.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            rx_object.material_payload.base_color,
            [0.0, 0.0, 1.0, 1.0],
        )
        assert isinstance(mock_viz.tx_markers[0], RenderObjectState)
        assert isinstance(mock_viz.rx_markers[0], RenderObjectState)

    def test_entity_sync_repositions_named_labels(self):
        """Entity sync keeps labels anchored to node positions."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        service = NodeService(mock_viz)
        mock_viz.tx_markers = [
            service._node_marker_handle("tx", 0, visible=True, position=[0.0, 0.0, 0.0])
        ]
        mock_viz.rx_markers = []
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = [[4.0, 5.0, 6.0]]
        mock_viz.app_state.show_labels = True
        mock_viz.label_offset_x = 1.5
        mock_viz.label_offset_y = -0.5
        mock_viz.label_offset_z = 1.0
        mock_viz.renderer.get_named_position = Mock(return_value=np.asarray([4.0, 5.0, 6.0]))
        mock_viz.renderer.set_named_transform = Mock()

        service.update_tx_rx_visibility()

        label_name = make_node_geometry_name("tx", 0, "label")
        ensured_label = _ensured_object(mock_viz, label_name)
        np.testing.assert_allclose(
            ensured_label.transform.translation,
            np.asarray([5.5, 4.5, 7.0]),
        )
        assert ensured_label.metadata["layout_anchor"] == (4.0, 5.0, 6.0)
        assert ensured_label.metadata["layout_offset"] == (1.5, -0.5, 1.0)

    def test_entity_sync_keeps_neutral_text_label_payload(self):
        """Entity sync preserves text intent without backend-native geometry."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        service = NodeService(mock_viz)
        mock_viz.tx_markers = [
            service._node_marker_handle("tx", 0, visible=True, position=[0.0, 0.0, 0.0])
        ]
        mock_viz.rx_markers = []
        mock_viz.tx_labels = [
            _label_state(
                make_node_geometry_name("tx", 0, "label"),
                text="Custom TX",
                color=(0.9, 0.1, 0.2),
            )
        ]
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = [[4.0, 5.0, 6.0]]
        mock_viz.app_state.show_labels = True
        mock_viz.renderer.get_named_position = Mock(return_value=np.asarray([4.0, 5.0, 6.0]))
        mock_viz.renderer.set_named_transform = Mock()

        service.update_tx_rx_visibility()

        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("tx", 0, "label"),
            mock_viz.tx_labels[0],
        )
        ensured = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        assert isinstance(ensured.payload, TextLabelPayload)
        assert ensured.payload.text == "Custom TX"
        assert ensured.material_payload.base_color[:3] == (0.9, 0.1, 0.2)

    def test_apply_node_coloring_syncs_from_current_positions(self):
        """Node recoloring syncs marker handles from authoritative frame state."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.rx_markers = []
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = np.asarray([[4.0, 5.0, 6.0]], dtype=float)
        mock_viz.current_rx_positions = np.empty((0, 3), dtype=float)
        mock_viz.label_font_size = 0.3
        mock_viz.tx_marker_size = 0.7
        mock_viz.renderer.request_redraw = Mock()

        service = NodeService(mock_viz)
        service.apply_node_coloring()

        assert isinstance(mock_viz.tx_markers[0], RenderObjectState)
        assert mock_viz.tx_markers[0].id == make_node_geometry_name("tx", 0, "marker")
        ensure_call = mock_viz.renderer.ensure_object.call_args
        assert ensure_call is not None
        render_object = ensure_call.args[0]
        assert render_object.id == make_node_geometry_name("tx", 0, "marker")
        np.testing.assert_allclose(render_object.transform.translation, [4.0, 5.0, 6.0])
        assert render_object.visible is True
        assert all(
            call.args[0] != make_node_geometry_name("tx", 0, "marker")
            for call in mock_viz.renderer.set_visible.call_args_list
        )


class TestOrientationFrames:
    """Test orientation frame creation and updates."""

    def test_apply_orientation_to_marker_updates_neutral_handle_transform_only(self):
        """Marker orientation should update render state, not rewrite payload vertices."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        service = NodeService(mock_viz)
        marker = service._node_marker_handle("tx", 0, position=[1.0, 2.0, 3.0])
        original_points = render_state_points(marker).copy()

        service.apply_orientation_to_marker(marker, [np.pi / 2.0, 0.0, 0.0], "TX1")

        np.testing.assert_allclose(render_state_points(marker), original_points)
        np.testing.assert_allclose(marker.world_transform.matrix[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            marker.world_transform.matrix[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_apply_orientation_to_marker_does_not_mutate_native_marker(self):
        """Native marker objects are no longer part of the node orientation contract."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        service = NodeService(mock_viz)
        marker = Mock()
        marker.transform = Mock()
        marker.translate = Mock()
        marker.get_center = Mock(return_value=np.asarray([1.0, 2.0, 3.0]))

        service.apply_orientation_to_marker(marker, [np.pi / 2.0, 0.0, 0.0], "TX1")

        marker.transform.assert_not_called()
        marker.translate.assert_not_called()
        marker.get_center.assert_not_called()

    @pytest.mark.parametrize(
        ("service_method", "helper_name", "arguments"),
        [
            ("create_orientation_frames", "helpers_create_orientation_frames", (5,)),
            (
                "update_tx_orientation_frames",
                "helpers_update_tx_orientation_frames",
                ([[0.1, 0.2, 0.3]],),
            ),
            (
                "update_rx_orientation_frames",
                "helpers_update_rx_orientation_frames",
                ([[0.1, 0.2, 0.3]],),
            ),
            (
                "update_target_orientation_frames",
                "helpers_update_target_orientation_frames",
                ({"target": [0.1, 0.2, 0.3]},),
            ),
        ],
    )
    def test_orientation_helpers_run_inside_one_renderer_batch(
        self,
        service_method,
        helper_name,
        arguments,
    ):
        mock_viz = make_mock_visualizer()
        service = NodeService(mock_viz)
        batch_active = False
        batch_events = []

        @contextmanager
        def _tracked_batch():
            nonlocal batch_active
            batch_active = True
            batch_events.append("enter")
            try:
                yield
            finally:
                batch_active = False
                batch_events.append("exit")

        def _assert_batch_active(*_args):
            assert batch_active is True

        mock_viz.renderer.batch_updates = _tracked_batch
        with patch(
            f"visualizer.src.services.node_service.{helper_name}",
            side_effect=_assert_batch_active,
        ):
            getattr(service, service_method)(*arguments)

        assert batch_events == ["enter", "exit"]


class TestOrientationVisibility:
    """Test orientation frame visibility management."""

    def test_update_orientation_visibility_shows_tx_frames_when_enabled(self):
        """Test TX orientation frames are shown when checkbox is enabled."""
        mock_viz = make_mock_visualizer(tx_count=2)
        mock_viz.show_tx_orientation = True

        service = NodeService(mock_viz)
        assert service.update_orientation_visibility() is True

        assert (
            _ensured_object(
                mock_viz,
                make_node_geometry_name("tx", 0, "orientation_frame"),
            ).visible
            is True
        )
        assert (
            _ensured_object(
                mock_viz,
                make_node_geometry_name("tx", 1, "orientation_frame"),
            ).visible
            is True
        )

    def test_update_orientation_visibility_hides_frames_when_disabled(self):
        """Test orientation frames are hidden when checkbox is disabled."""
        mock_viz = make_mock_visualizer(tx_count=2)
        mock_viz.show_tx_orientation = False

        service = NodeService(mock_viz)
        assert service.update_orientation_visibility() is True

        assert (
            _ensured_object(
                mock_viz,
                make_node_geometry_name("tx", 0, "orientation_frame"),
            ).visible
            is False
        )
        assert (
            _ensured_object(
                mock_viz,
                make_node_geometry_name("tx", 1, "orientation_frame"),
            ).visible
            is False
        )

    def test_orientation_visibility_helpers_run_inside_one_renderer_batch(self):
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
        service = NodeService(mock_viz)
        batch_active = False
        batch_events = []

        @contextmanager
        def _tracked_batch():
            nonlocal batch_active
            batch_active = True
            batch_events.append("enter")
            try:
                yield
            finally:
                batch_active = False
                batch_events.append("exit")

        def _sync(_viz, _frame, _visible):
            assert batch_active is True
            return True

        mock_viz.renderer.batch_updates = _tracked_batch
        with patch(
            "visualizer.src.services.node_service.helpers_sync_orientation_frame_visibility",
            side_effect=_sync,
        ):
            assert service.update_orientation_visibility() is True

        assert batch_events == ["enter", "exit"]

    def test_failed_orientation_visibility_snapshot_is_retried(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.show_tx_orientation = True
        frame_id = make_node_geometry_name("tx", 0, "orientation_frame")
        failed_once = False

        def _ensure(render_object):
            nonlocal failed_once
            if render_object.id == frame_id and not failed_once:
                failed_once = True
                return False
            return True

        mock_viz.renderer.ensure_object = Mock(side_effect=_ensure)
        service = NodeService(mock_viz)

        assert service.update_orientation_visibility() is False
        assert frame_id in mock_viz._pending_orientation_frame_syncs

        assert service.update_orientation_visibility() is True
        assert frame_id not in mock_viz._pending_orientation_frame_syncs

    def test_orientation_visibility_policy_combines_selection_and_pov(self):
        """Frame snapshots and UI events share one effective-visibility policy."""
        mock_viz = make_mock_visualizer(tx_count=2)
        mock_viz.show_tx_orientation = True
        mock_viz.app_state.selected_tx = 1
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("tx", 1)
        mock_viz.tx_entries = [{"visible": True}, {"visible": True}]
        service = NodeService(mock_viz)

        assert service.orientation_frame_visible("tx", 0) is False
        assert service.orientation_frame_visible("tx", 1) is False

        mock_viz.app_state.pov_hidden_node = None
        assert service.orientation_frame_visible("tx", 0) is False
        assert service.orientation_frame_visible("tx", 1) is True

        mock_viz.tx_entries[1]["visible"] = False
        assert service.orientation_frame_visible("tx", 1) is False

    def test_target_orientation_visibility_inherits_parent_runtime_state(self):
        """Target frames cannot outlive a hidden or frame-absent target mesh."""
        mock_viz = make_mock_visualizer()
        mock_viz.show_target_orientation = True
        mock_viz.target_entries = [
            {
                "entry_type": "target",
                "visible": False,
                "_frame_visible": True,
            }
        ]
        service = NodeService(mock_viz)

        assert service.orientation_frame_visible("target", 0) is False

        mock_viz.target_entries[0]["visible"] = True
        mock_viz.target_entries[0]["_frame_visible"] = False
        assert service.orientation_frame_visible("target", 0) is False

        mock_viz.target_entries[0]["_frame_visible"] = True
        assert service.orientation_frame_visible("target", 0) is True


class TestPovSemanticRefresh:
    """POV transitions republish complete snapshots from semantic owners."""

    @staticmethod
    def _prepare_node_state(mock_viz, service: NodeService) -> None:
        mock_viz.app_state.camera_mode = "overview"
        mock_viz.app_state.pov_hidden_node = None
        mock_viz.app_state.selected_tx = "all"
        mock_viz.app_state.selected_rx = "all"
        mock_viz.app_state.show_labels = True
        service.refresh_comm_node_entries()

    def test_paused_pov_hides_marker_and_label_in_one_semantic_refresh(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=1)
        mock_viz.rx_labels = [_node_label_state("rx", 0)]
        service = NodeService(mock_viz)
        self._prepare_node_state(mock_viz, service)
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("rx", 0)

        service.sync_pov_entity_visibility([("rx", 0)])

        marker = _ensured_object(mock_viz, make_node_geometry_name("rx", 0, "marker"))
        label = _ensured_object(mock_viz, make_node_geometry_name("rx", 0, "label"))
        assert marker.visible is False
        assert label.visible is False
        assert mock_viz.rx_markers[0].visible is True
        assert mock_viz.rx_labels[0].visible is True

    def test_target_missing_at_pov_restore_stays_hidden_with_its_label(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mesh = _target_mesh([1.0, 2.0, 3.0], name="walker")
        target_entry = {
            "name": "Walker",
            "target_name": "walker",
            "node_index": 0,
            "entry_type": "target",
            "mesh": mesh,
            "position": [1.0, 2.0, 3.0],
            "visible": True,
            "show_label": True,
            "_frame_visible": True,
        }
        mock_viz.target_entries = [target_entry]
        mock_viz.target_labels = [
            _label_state(make_target_entry_geometry_name(target_entry, "label"))
        ]
        orientation = RenderObjectState(
            id=make_target_entry_geometry_name(target_entry, "orientation_frame"),
            payload=mesh.payload,
        )
        mock_viz.target_orientation_frames = [orientation]
        mock_viz.show_target_orientation = True
        mock_viz.target_outlines_enabled = True
        mock_viz.outline_color = [0.05, 0.05, 0.05]
        mock_viz.app_state.camera_mode = "overview"
        mock_viz.app_state.pov_hidden_node = None
        service = NodeService(mock_viz)

        # Seed the persistent outline while the target is still present.
        assert service.target_service.sync_target_entry_snapshot(target_entry)
        outline = target_entry["outline_geometry"]
        assert _ensured_object(mock_viz, outline.id).visible is True

        # The target disappears from the frame while POV is active. Leaving
        # POV must not restore any child snapshot past the missing parent.
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("target", 0)
        target_entry["_frame_visible"] = False
        mock_viz.app_state.camera_mode = "overview"
        mock_viz.app_state.pov_hidden_node = None

        service.sync_pov_entity_visibility([("target", 0)])

        target = _ensured_object(mock_viz, make_target_entry_geometry_name(target_entry, "mesh"))
        label = _ensured_object(mock_viz, make_target_entry_geometry_name(target_entry, "label"))
        outline_snapshot = _ensured_object(mock_viz, outline.id)
        assert target.visible is False
        assert label.visible is False
        assert outline_snapshot.visible is False
        assert _ensured_object(mock_viz, orientation.id).visible is False
        assert target_entry["visible"] is True
        assert mesh.visible is True

    def test_selection_change_during_pov_is_resolved_on_restore(self):
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0), _node_label_state("tx", 1)]
        service = NodeService(mock_viz)
        self._prepare_node_state(mock_viz, service)
        mock_viz.app_state.camera_mode = "overview"
        mock_viz.app_state.selected_tx = 1
        mock_viz.app_state.pov_hidden_node = None

        service.sync_pov_entity_visibility([("tx", 0)])

        first = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        second = _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "marker"))
        first_label = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        second_label = _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "label"))
        assert first.visible is False
        assert first_label.visible is False
        assert second.visible is True
        assert second_label.visible is True

    def test_semantic_visibility_change_during_pov_is_not_overwritten(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        service = NodeService(mock_viz)
        self._prepare_node_state(mock_viz, service)
        mock_viz.tx_entries[0]["visible"] = False
        mock_viz.app_state.camera_mode = "overview"
        mock_viz.app_state.pov_hidden_node = None

        service.sync_pov_entity_visibility([("tx", 0)])

        marker = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        label = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        assert marker.visible is False
        assert label.visible is False
        assert mock_viz.tx_entries[0]["visible"] is False

    def test_pov_snapshot_owners_share_one_renderer_batch(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.target_entries = [{"entry_type": "target"}]
        service = NodeService(mock_viz)
        batch_depth = 0
        batch_events = []

        @contextmanager
        def _tracked_batch():
            nonlocal batch_depth
            batch_depth += 1
            batch_events.append("enter")
            try:
                yield
            finally:
                batch_events.append("exit")
                batch_depth -= 1

        def _assert_batched(*_args, **_kwargs):
            assert batch_depth == 1
            return True

        mock_viz.renderer.batch_updates = _tracked_batch
        mock_viz.renderer.request_redraw = Mock(side_effect=_assert_batched)
        service._sync_tx_rx_visual_entities = Mock(side_effect=_assert_batched)
        service.target_service.sync_target_entry_snapshot = Mock(side_effect=_assert_batched)
        service.update_orientation_visibility = Mock(side_effect=_assert_batched)

        assert service.sync_pov_entity_visibility((("tx", 0), ("target", 0))) is True

        assert batch_events == ["enter", "exit"]
        mock_viz.renderer.update_renderer.assert_not_called()

    @pytest.mark.parametrize("failed_component", ["node", "target", "orientation"])
    def test_pov_sync_propagates_each_owner_failure_and_identical_retry_converges(
        self,
        failed_component,
    ):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.target_entries = [{"entry_type": "target"}]
        service = NodeService(mock_viz)
        outcomes = iter((False, True))

        node_sync = Mock(return_value=True)
        target_sync = Mock(return_value=True)
        orientation_sync = Mock(return_value=True)
        failing_sync = Mock(side_effect=lambda *_args, **_kwargs: next(outcomes))
        if failed_component == "node":
            node_sync = failing_sync
        elif failed_component == "target":
            target_sync = failing_sync
        else:
            orientation_sync = failing_sync

        service._sync_tx_rx_visual_entities = node_sync
        service.target_service.sync_target_entry_snapshot = target_sync
        service.update_orientation_visibility = orientation_sync
        refs = (("tx", 0), ("target", 0))

        assert service.sync_pov_entity_visibility(refs) is False
        assert service.sync_pov_entity_visibility(refs) is True
        assert failing_sync.call_count == 2
        assert node_sync.call_count == 2
        assert target_sync.call_count == 2
        assert orientation_sync.call_count == 2


class TestTxRxDiscovery:
    """Test TX/RX node discovery logic."""

    def test_discover_skips_when_no_frame_source(self):
        """Test that discovery is skipped when frame source is not available."""
        mock_viz = make_mock_visualizer()
        mock_viz.frame_source = None
        mock_viz.tx_rx_data_loaded = False

        service = NodeService(mock_viz)
        service.discover_available_tx_rx()

        # Should exit early without completing discovery
        assert mock_viz.tx_rx_data_loaded is False

    def test_discover_uses_cached_data_when_available(self):
        """Test that discovery uses cached preload data when available."""
        mock_viz = make_mock_visualizer()

        # Setup frame source
        mock_viz.frame_source = Mock()
        mock_viz.frame_source.has_frame = Mock(return_value=True)

        # Setup cached data
        mock_viz.cache_service = CacheService(mock_viz)
        mock_viz.cache_service.store_frame(
            0,
            {"tx_positions": [[0, 0, 0], [1, 1, 1]], "rx_positions": [[2, 2, 2]]},
            source="preload",
        )

        # Setup MPCCore
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core.discover_tx_rx = Mock(return_value=([0, 1], [0]))

        # Setup progress
        mock_viz.progress = Mock()
        mock_viz.progress.note = Mock()

        service = NodeService(mock_viz)

        with patch.object(service, "populate_tx_rx_selections"):
            service.discover_available_tx_rx()

        # Verify cached data was used
        assert mock_viz.available_tx == [0, 1]
        assert mock_viz.available_rx == [0]
        mock_viz.mpc_core.discover_tx_rx.assert_called_once()

    def test_discover_falls_back_to_provider_when_no_cache(self):
        """Test that discovery loads from provider when no cached data."""
        mock_viz = make_mock_visualizer()

        # Setup frame source
        mock_viz.frame_source = Mock()
        mock_viz.frame_source.has_frame = Mock(return_value=True)
        mock_viz.frame_source.load_frame = Mock(return_value=build_standard_mpc_frame())

        # No cached data
        mock_viz.cache_service = CacheService(mock_viz)
        mock_viz.animation_service = None

        # Setup MPCCore
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core.canon_points_dtype = np.float32
        mock_viz.mpc_core.discover_tx_rx = Mock(return_value=([0, 1], [0, 1]))

        # Setup progress
        mock_viz.progress = Mock()
        mock_viz.progress.note = Mock()

        service = NodeService(mock_viz)

        with patch.object(service, "populate_tx_rx_selections"):
            service.discover_available_tx_rx()

        # Verify provider was used
        mock_viz.frame_source.load_frame.assert_called_once_with(0)
        loaded_payload = mock_viz.mpc_core.discover_tx_rx.call_args.args[0]
        assert isinstance(loaded_payload, dict)
        assert loaded_payload["num_tx"] == 2
        assert loaded_payload["num_rx"] == 2
        assert "canonical_data" in loaded_payload
        assert mock_viz.available_tx == [0, 1]
        assert mock_viz.available_rx == [0, 1]

    def test_discover_uses_animation_service_cache_prime_before_provider(self):
        """Cache priming should avoid a second direct provider load on startup."""
        mock_viz = make_mock_visualizer()
        mock_viz.frame_source = Mock()
        mock_viz.frame_source.has_frame = Mock(return_value=True)
        mock_viz.frame_source.load_frame = Mock()
        mock_viz.cache_service = CacheService(mock_viz)
        mock_viz.animation_service.ensure_step_cached = Mock(
            return_value={"tx_positions": [[0, 0, 0]], "rx_positions": [[1, 1, 1]]}
        )
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core.discover_tx_rx = Mock(return_value=([0], [0]))
        mock_viz.progress = Mock()
        mock_viz.progress.note = Mock()

        service = NodeService(mock_viz)

        with patch.object(service, "populate_tx_rx_selections"):
            service.discover_available_tx_rx()

        mock_viz.animation_service.ensure_step_cached.assert_called_once_with(0)
        mock_viz.frame_source.load_frame.assert_not_called()
        assert mock_viz.available_tx == [0]
        assert mock_viz.available_rx == [0]


class TestLabelOffsets:
    """Test label offset application."""

    def test_apply_label_offsets_updates_positions(self):
        """Test that label offsets are applied through entity label transforms."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=1)
        mock_viz.label_offset_x = 1.0
        mock_viz.label_offset_y = 2.0
        mock_viz.label_offset_z = 3.0

        # Add TX/RX positions
        mock_viz.current_tx_positions = [[0, 0, 0], [5, 0, 0]]
        mock_viz.current_rx_positions = [[10, 0, 0]]
        mock_viz.tx_labels = [
            _node_label_state("tx", 0),
            _node_label_state("tx", 1),
        ]
        mock_viz.rx_labels = [_node_label_state("rx", 0)]

        service = NodeService(mock_viz)
        service.apply_label_offsets()

        expected_positions = {
            make_node_geometry_name("tx", 0, "label"): [1.0, 2.0, 3.0],
            make_node_geometry_name("tx", 1, "label"): [6.0, 2.0, 3.0],
            make_node_geometry_name("rx", 0, "label"): [11.0, 2.0, 3.0],
        }
        for label_name, expected_position in expected_positions.items():
            ensured = _ensured_object(mock_viz, label_name)
            assert isinstance(ensured.payload, TextLabelPayload)
            np.testing.assert_allclose(ensured.transform.translation, expected_position)
            assert ensured.metadata["layout_offset"] == (1.0, 2.0, 3.0)

    def test_apply_label_offsets_keeps_unselected_label_hidden(self):
        """Hidden labels remain declarative objects with false visibility."""
        mock_viz = make_mock_visualizer(tx_count=2)
        mock_viz.label_offset_x = 1.0
        mock_viz.label_offset_y = 0.0
        mock_viz.label_offset_z = 0.0
        mock_viz.app_state.selected_tx = 0

        # Add positions
        mock_viz.current_tx_positions = [[0, 0, 0], [5, 0, 0]]
        mock_viz.tx_labels = [
            _node_label_state("tx", 0),
            _node_label_state("tx", 1),
        ]

        service = NodeService(mock_viz)
        service.apply_label_offsets()

        visible_label = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        hidden_label = _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "label"))
        assert visible_label.visible is True
        assert hidden_label.visible is False

    def test_apply_label_offsets_keeps_pov_target_label_hidden(self):
        """Label offset refreshes must not re-show the target used as the POV camera."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.label_offset_x = 1.0
        mock_viz.label_offset_y = 2.0
        mock_viz.label_offset_z = 3.0
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("target", 0)
        mock_viz.app_state.show_target_labels = True
        target_entry = {
            "target_name": "target_1",
            "position": np.array([1.0, 2.0, 3.0]),
            "show_label": True,
        }
        mock_viz.target_entries = [target_entry]
        label_name = make_target_entry_geometry_name(target_entry, "label")
        mock_viz.target_labels = [_label_state(label_name, text="Target 1")]

        service = NodeService(mock_viz)
        service.apply_label_offsets()

        ensured = _ensured_object(mock_viz, label_name)
        assert ensured.visible is False
        np.testing.assert_allclose(ensured.transform.translation, [2.0, 4.0, 6.0])


class TestCommNodeEntries:
    """Test communication node entry management."""

    def test_refresh_comm_node_entries_creates_entries(self):
        """Test that entries are created for TX/RX nodes."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=1)

        # Setup positions and orientations
        mock_viz.current_tx_positions = [[0, 0, 0], [1, 1, 1]]
        mock_viz.current_rx_positions = [[2, 2, 2]]
        mock_viz.current_tx_orientations = [[0, 0, 0], [0, 0, 0]]
        mock_viz.current_rx_orientations = [[0, 0, 0]]

        # Initialize empty entries
        mock_viz.tx_entries = []
        mock_viz.rx_entries = []

        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()

        assert len(mock_viz.tx_entries) == 2
        assert len(mock_viz.rx_entries) == 1
        assert mock_viz.tx_entries[0]["name"] == "TX1"
        assert mock_viz.rx_entries[0]["name"] == "RX1"

    def test_refresh_comm_node_entries_uses_device_names_when_enabled(self):
        """Device-name mode uses scenario/frame names for TX/RX object entries."""
        from visualizer.src.state import create_initial_state

        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.current_tx_positions = [[0, 0, 0]]
        mock_viz.current_rx_positions = [[2, 2, 2]]
        mock_viz.current_tx_orientations = [[0, 0, 0]]
        mock_viz.current_rx_orientations = [[0, 0, 0]]
        mock_viz.tx_entries = []
        mock_viz.rx_entries = []
        mock_viz.app_state = create_initial_state(
            node_label_mode="name",
            tx_device_names=("MainTransmitter",),
            rx_device_names=("WalkingReceiver",),
        )

        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()

        assert mock_viz.tx_entries[0]["name"] == "MainTransmitter"
        assert mock_viz.rx_entries[0]["name"] == "WalkingReceiver"

    def test_refresh_comm_node_entries_updates_existing(self):
        """Test that existing entries are updated rather than replaced."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)

        # Setup initial entry
        initial_entry = {"name": "TX1", "position": [0, 0, 0]}
        mock_viz.tx_entries = [initial_entry]
        mock_viz.rx_entries = []

        # Update position
        mock_viz.current_tx_positions = [[1, 1, 1]]
        mock_viz.current_tx_orientations = [[0, 0, 0]]

        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()

        # Should still be the same list object, but content updated
        assert len(mock_viz.tx_entries) == 1
        assert mock_viz.tx_entries[0]["position"] == [1.0, 1.0, 1.0]
        assert mock_viz.tx_entries[0] is initial_entry  # Same dictionary object updated

    def test_refresh_comm_node_entries_preserves_visibility_and_label_intent(self):
        """Frame hydration must not overwrite session/object-panel intent."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_tx_orientations = [[0.0, 0.0, 0.0]]
        mock_viz.tx_entries = [{"visible": False, "show_label": False}]
        mock_viz.rx_entries = []

        NodeService(mock_viz).refresh_comm_node_entries()

        assert mock_viz.tx_entries[0]["visible"] is False
        assert mock_viz.tx_entries[0]["show_label"] is False


class TestPopulateTxRxSelections:
    """Test TX/RX dropdown population."""

    def test_populate_tx_rx_selections_adds_items(self, qapp):
        """Test that dropdowns are populated with available nodes."""
        mock_viz = make_mock_visualizer()
        mock_viz.available_tx = [0, 1, 2]
        mock_viz.available_rx = [0]

        mock_viz.tx_dropdown = QComboBox()
        mock_viz.rx_dropdown = QComboBox()

        service = NodeService(mock_viz)
        service.populate_tx_rx_selections()

        # Verify TX items (All + 3 TXs = 4 items)
        assert mock_viz.tx_dropdown.count() == 4
        # Verify RX items (All + 1 RX = 2 items)
        assert mock_viz.rx_dropdown.count() == 2

    def test_populate_tx_rx_selections_preserves_selection(self, qapp):
        """Test that previous selection is restored."""
        mock_viz = make_mock_visualizer()
        mock_viz.available_tx = [0, 1]
        mock_viz.available_rx = [0]

        mock_viz.tx_dropdown = QComboBox()
        mock_viz.rx_dropdown = QComboBox()

        # Set previous selection in the authoritative snapshot.
        mock_viz.app_state.selected_tx = 1

        service = NodeService(mock_viz)
        service.populate_tx_rx_selections(preserve_selection=True)

        # Verify selection was restored
        assert mock_viz.tx_dropdown.currentData() == 1

    def test_populate_tx_rx_selections_updates_app_state(self, qapp):
        """TX/RX selector updates should route through set_state when available."""
        from visualizer.src.state import create_initial_state, update_state

        mock_viz = make_mock_visualizer()
        mock_viz.available_tx = [0, 1]
        mock_viz.available_rx = [0, 1]
        mock_viz.tx_dropdown = QComboBox()
        mock_viz.rx_dropdown = QComboBox()
        mock_viz.app_state = create_initial_state(selected_tx=1, selected_rx=1)

        def _set_state(**changes):
            mock_viz.app_state = update_state(mock_viz.app_state, **changes)

        mock_viz.set_state = Mock(side_effect=_set_state)
        service = NodeService(mock_viz)
        service.populate_tx_rx_selections(preserve_selection=True)

        mock_viz.set_state.assert_called()
        assert mock_viz.app_state.selected_tx == 1
        assert mock_viz.app_state.selected_rx == 1

    def test_populate_tx_rx_selections_uses_device_names_when_enabled(self, qapp):
        """The TX/RX dropdowns show scenario names in device-name mode."""
        from visualizer.src.state import create_initial_state

        mock_viz = make_mock_visualizer()
        mock_viz.available_tx = [0]
        mock_viz.available_rx = [0]
        mock_viz.tx_dropdown = QComboBox()
        mock_viz.rx_dropdown = QComboBox()
        mock_viz.app_state = create_initial_state(
            node_label_mode="name",
            tx_device_names=("MainTransmitter",),
            rx_device_names=("WalkingReceiver",),
        )

        service = NodeService(mock_viz)
        service.populate_tx_rx_selections()

        assert mock_viz.tx_dropdown.itemText(1) == "MainTransmitter"
        assert mock_viz.tx_dropdown.itemData(1) == 0
        assert mock_viz.rx_dropdown.itemText(1) == "WalkingReceiver"
        assert mock_viz.rx_dropdown.itemData(1) == 0


class TestObjectPanelLabelAndIdentity:
    """Object-panel driven node behavior should only affect node labels."""

    def test_set_node_label_visibility_affects_only_label_geometry(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.renderer.set_geometry_visible = Mock()
        mock_viz.renderer.set_geometry_transform_fast = Mock()
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        mock_viz.renderer.get_named_position = Mock(return_value=[0.0, 0.0, 0.0])

        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()
        tx_entry = mock_viz.tx_entries[0]

        service.set_node_label_visibility(tx_entry, False)

        assert mock_viz.tx_entries[0]["show_label"] is False
        render_object = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        assert render_object.visible is True
        label_object = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        assert label_object.visible is False
        mock_viz.renderer.request_redraw.assert_called_once()
        mock_viz.renderer.update_renderer.assert_not_called()

    def test_update_tx_rx_visibility_hides_unanchored_nodes(self):
        """Open3D/pygfx visibility path must hide markers that have no resolved anchor."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.tx_labels = [
            _node_label_state("tx", 0),
            _node_label_state("tx", 1),
        ]
        mock_viz.app_state.selected_tx = "all"
        mock_viz.app_state.selected_rx = "all"
        mock_viz.app_state.show_labels = True
        mock_viz.renderer.set_geometry_visible = Mock()
        mock_viz.renderer.set_geometry_transform_fast = Mock()
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        mock_viz.renderer.get_named_position = Mock(return_value=None)

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        assert _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker")).visible is True
        assert (
            _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "marker")).visible is False
        )
        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("tx", 0, "label"),
            mock_viz.tx_labels[0],
        )
        hidden_label = _ensured_object(
            mock_viz,
            make_node_geometry_name("tx", 1, "label"),
        )
        assert hidden_label.visible is False

    def test_empty_frame_hides_nodes_without_discarding_entry_intent(self) -> None:
        """Inventory identity and user flags survive a temporary anchorless frame."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
        mock_viz.available_tx = [0, 1]
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_viz.current_tx_orientations = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
        mock_viz.tx_labels = [
            _node_label_state("tx", 0),
            _node_label_state("tx", 1),
        ]
        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()
        first_entry, second_entry = mock_viz.tx_entries
        second_entry["visible"] = False
        second_entry["show_label"] = False

        service.update_tx_rx_positions(
            np.empty((0, 3)),
            np.empty((0, 3)),
        )

        assert mock_viz.current_tx_positions.shape == (0, 3)
        assert mock_viz.tx_entries == [first_entry, second_entry]
        assert mock_viz.tx_entries[0] is first_entry
        assert mock_viz.tx_entries[1] is second_entry
        assert second_entry["visible"] is False
        assert second_entry["show_label"] is False
        assert first_entry["_frame_position_valid"] is False
        assert second_entry["_frame_position_valid"] is False
        assert (
            _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker")).visible is False
        )
        assert (
            _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "marker")).visible is False
        )

    def test_update_tx_rx_visibility_respects_per_entry_visibility(self):
        """Canonical node entry visibility gates marker and label snapshots."""
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.tx_entries = [{"visible": False, "show_label": True}]

        NodeService(mock_viz).update_tx_rx_visibility()

        marker = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        label = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "label"))
        assert marker.visible is False
        assert label.visible is False

    def test_effective_node_visibility_preserves_semantic_handle_state(self):
        """Selection and POV policy belong only to renderer snapshots."""
        mock_viz = make_mock_visualizer(tx_count=2, rx_count=0)
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_viz.tx_labels = [
            _node_label_state("tx", 0),
            _node_label_state("tx", 1),
        ]
        mock_viz.app_state.selected_tx = 0

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        hidden_marker = mock_viz.tx_markers[1]
        hidden_label = mock_viz.tx_labels[1]
        assert hidden_marker.visible is True
        assert hidden_label.visible is True
        assert (
            _ensured_object(mock_viz, make_node_geometry_name("tx", 1, "marker")).visible is False
        )
        assert _ensured_object(mock_viz, hidden_label.id).visible is False

    def test_pov_hides_rx_snapshot_without_mutating_marker_intent(self):
        """The Open3D/pygfx object contract receives the same POV-hidden RX."""
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=1)
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]
        mock_viz.rx_labels = []
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("rx", 0)

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        marker = mock_viz.rx_markers[0]
        assert marker.visible is True
        assert _ensured_object(mock_viz, marker.id).visible is False

    def test_refresh_entries_preserves_ids_across_display_rename(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()

        before_id = mock_viz.tx_entries[0]["object_id"]
        before_key = mock_viz.tx_entries[0]["object_key"]

        mock_viz.app_state.tx_labels = ("Renamed TX",)
        service.refresh_comm_node_entries()

        assert mock_viz.tx_entries[0]["name"] == "Renamed TX"
        assert mock_viz.tx_entries[0]["object_id"] == before_id
        assert mock_viz.tx_entries[0]["object_key"] == before_key
        assert mock_viz.tx_entries[0]["label_geometry_name"] == make_node_geometry_name(
            "tx", 0, "label"
        )

    def test_recreate_tx_rx_labels_positions_new_labels_before_renderer_registration(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.label_offset_x = 1.5
        mock_viz.label_offset_y = -0.5
        mock_viz.label_offset_z = 1.0

        service = NodeService(mock_viz)
        service._update_tx_rx_visibility = Mock()

        service.recreate_tx_rx_labels(0.6)

        assert len(mock_viz.tx_labels) == 1
        assert len(mock_viz.rx_labels) == 1
        assert isinstance(mock_viz.tx_labels[0].payload, TextLabelPayload)
        assert mock_viz.tx_labels[0].payload.font_size == pytest.approx(0.6)
        np.testing.assert_allclose(
            mock_viz.tx_labels[0].world_transform.translation,
            [1.5, -0.5, 2.5],
        )
        np.testing.assert_allclose(
            mock_viz.rx_labels[0].world_transform.translation,
            [1.5, 9.5, 2.5],
        )
        mock_viz.renderer.translate_geometry.assert_not_called()
        service._update_tx_rx_visibility.assert_called_once()

    def test_recreate_target_labels_positions_new_labels_before_renderer_registration(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.label_offset_x = 1.0
        mock_viz.label_offset_y = 2.0
        mock_viz.label_offset_z = 3.0
        mesh = _target_mesh([4.0, 5.0, 6.0], name="car_1")
        mock_viz.target_entries = [
            {
                "entry_type": "target",
                "name": "Car 1",
                "mesh": mesh,
                "show_label": True,
                "target_name": "car_1",
            }
        ]
        service = NodeService(mock_viz)
        service.update_target_label_visibility = Mock()

        service.recreate_target_labels(0.6)

        assert len(mock_viz.target_labels) == 1
        assert isinstance(mock_viz.target_labels[0].payload, TextLabelPayload)
        assert mock_viz.target_labels[0].payload.font_size == pytest.approx(0.6)
        np.testing.assert_allclose(
            mock_viz.target_labels[0].world_transform.translation,
            [5.0, 7.0, 9.0],
        )
        mock_viz.renderer.translate_geometry.assert_not_called()
        service.update_target_label_visibility.assert_called_once()

    def test_update_tx_rx_visibility_registers_missing_visible_neutral_label(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.renderer.set_geometry_visible = Mock()
        mock_viz.renderer.set_geometry_transform_fast = Mock()
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)

        service = NodeService(mock_viz)
        service.refresh_comm_node_entries()
        service.update_tx_rx_visibility()

        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("tx", 0, "label"),
            mock_viz.tx_labels[0],
        )

    def test_failed_node_label_sync_stays_pending_until_identical_retry_succeeds(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        label_id = make_node_geometry_name("tx", 0, "label")
        failed_once = False

        def _ensure(render_object):
            nonlocal failed_once
            if render_object.id == label_id and not failed_once:
                failed_once = True
                return False
            return True

        mock_viz.renderer.ensure_object = Mock(side_effect=_ensure)
        service = NodeService(mock_viz)

        assert service._sync_tx_rx_visual_entities() is False
        assert service._pending_node_entity_ids == {"node:tx_0"}

        assert service.retry_pending_node_syncs() is True
        assert not service._pending_node_entity_ids
        label_calls = [
            call
            for call in mock_viz.renderer.ensure_object.call_args_list
            if call.args[0].id == label_id
        ]
        assert len(label_calls) == 2

    def test_failed_node_removal_stays_pending_until_identical_retry_succeeds(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        label_id = make_node_geometry_name("tx", 0, "label")
        failed_once = False

        def _remove(object_id):
            nonlocal failed_once
            if object_id == label_id and not failed_once:
                failed_once = True
                return False
            return True

        mock_viz.renderer.remove_object = Mock(side_effect=_remove)
        service = NodeService(mock_viz)

        assert service._remove_node_marker_entity("tx", 0) is False
        assert service._pending_node_removals == {("tx", 0)}

        assert service.retry_pending_node_syncs() is True
        assert not service._pending_node_removals
        assert mock_viz.renderer.remove_object.call_count == 4

    def test_update_tx_rx_visibility_ensures_missing_visible_rx_handle_in_pygfx_path(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)
        mock_viz.renderer.is_named_visible = Mock(return_value=None)
        mock_viz.renderer.get_named_position = Mock(return_value=None)
        mock_viz.renderer.ensure_object = Mock(return_value=True)
        mock_viz.renderer.set_transform = Mock(return_value=True)
        mock_viz.renderer.set_visible = Mock(return_value=True)
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]

        service = NodeService(mock_viz)
        mock_viz.tx_markers = [
            service._node_marker_handle("tx", 0, visible=True, position=[0.0, 0.0, 0.0])
        ]
        mock_viz.rx_markers = [
            service._node_marker_handle("rx", 0, visible=True, position=[0.0, 0.0, 0.0])
        ]

        service.update_tx_rx_visibility()

        ensured_ids = [call.args[0].id for call in mock_viz.renderer.ensure_object.call_args_list]
        assert make_node_geometry_name("rx", 0, "marker") in ensured_ids
        rx_call = next(
            call
            for call in mock_viz.renderer.ensure_object.call_args_list
            if call.args[0].id == make_node_geometry_name("rx", 0, "marker")
        )
        assert rx_call.args[0].visible is True
        np.testing.assert_allclose(rx_call.args[0].transform.translation, [4.0, 5.0, 6.0])
        assert all(
            call.args[0] != make_node_geometry_name("rx", 0, "marker")
            for call in mock_viz.renderer.set_visible.call_args_list
        )

    def test_update_tx_rx_visibility_applies_orientation_to_marker_transform(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=1)
        mock_viz.renderer.ensure_object = Mock(return_value=True)
        mock_viz.renderer.set_transform = Mock(return_value=True)
        mock_viz.renderer.set_visible = Mock(return_value=True)
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]
        mock_viz.current_tx_orientations = [[np.pi / 2.0, 0.0, 0.0]]
        mock_viz.current_rx_orientations = [[0.0, 0.0, 0.0]]

        service = NodeService(mock_viz)
        mock_viz.tx_markers = [
            service._node_marker_handle("tx", 0, visible=True, position=[1.0, 2.0, 3.0])
        ]
        mock_viz.rx_markers = [
            service._node_marker_handle("rx", 0, visible=True, position=[4.0, 5.0, 6.0])
        ]

        service.update_tx_rx_visibility()

        tx_object = _ensured_object(mock_viz, make_node_geometry_name("tx", 0, "marker"))
        np.testing.assert_allclose(tx_object.transform.translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            tx_object.transform.matrix[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_update_tx_marker_sizes_updates_pygfx_handle_with_new_radius(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        mock_viz.renderer.is_named_visible = Mock(return_value=True)
        mock_viz.renderer.get_named_position = Mock(return_value=np.asarray([1.0, 2.0, 3.0]))
        mock_viz.renderer.ensure_object = Mock(return_value=True)
        mock_viz.renderer.remove_object = Mock(return_value=True)
        mock_viz.renderer.set_transform = Mock(return_value=True)
        mock_viz.renderer.set_visible = Mock(return_value=True)
        mock_viz.renderer.set_material = Mock(return_value=True)
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.current_rx_positions = []
        mock_viz.tx_marker_size = 0.8

        service = NodeService(mock_viz)
        old_marker = service._node_marker_handle(
            "tx",
            0,
            size=0.3,
            color=[1.0, 0.0, 0.0],
            visible=True,
            position=[1.0, 2.0, 3.0],
        )
        mock_viz.tx_markers = [old_marker]

        service.update_tx_marker_sizes()

        mock_viz.renderer.remove_object.assert_not_called()
        tx_objects = [
            call.args[0]
            for call in mock_viz.renderer.ensure_object.call_args_list
            if call.args[0].id == make_node_geometry_name("tx", 0, "marker")
        ]
        assert tx_objects
        assert any(_payload_radius(obj.payload) == pytest.approx(0.8) for obj in tx_objects)
        assert any(obj.visible is True for obj in tx_objects)
        np.testing.assert_allclose(tx_objects[-1].transform.translation, [1.0, 2.0, 3.0])
        assert all(
            call.args[0] != make_node_geometry_name("tx", 0, "marker")
            for call in mock_viz.renderer.set_visible.call_args_list
        )

    def test_update_rx_marker_sizes_updates_custom_mesh_payload_scale(self, tmp_path):
        mesh_path = tmp_path / "receiver_marker.obj"
        mesh_path.write_text(
            "\n".join(
                [
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "f 1 2 3",
                ]
            ),
            encoding="utf-8",
        )
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=1)
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        mock_viz.renderer.is_named_visible = Mock(return_value=True)
        mock_viz.renderer.get_named_position = Mock(return_value=np.asarray([4.0, 5.0, 6.0]))
        mock_viz.renderer.ensure_object = Mock(return_value=True)
        mock_viz.renderer.remove_object = Mock(return_value=True)
        mock_viz.renderer.set_transform = Mock(return_value=True)
        mock_viz.renderer.set_visible = Mock(return_value=True)
        mock_viz.renderer.set_material = Mock(return_value=True)
        mock_viz.node_marker_config = {
            "rx": {
                "shape": "mesh",
                "mesh_path": mesh_path,
                "scale": 2.0,
                "center": True,
            }
        }
        mock_viz.tx_labels = []
        mock_viz.rx_labels = []
        mock_viz.current_tx_positions = []
        mock_viz.current_rx_positions = [[4.0, 5.0, 6.0]]
        mock_viz.rx_marker_size = 0.9

        service = NodeService(mock_viz)
        old_marker = service._node_marker_handle(
            "rx",
            0,
            size=0.3,
            color=[0.0, 0.0, 1.0],
            visible=True,
            position=[4.0, 5.0, 6.0],
        )
        mock_viz.rx_markers = [old_marker]

        service.update_rx_marker_sizes()

        rx_objects = [
            call.args[0]
            for call in mock_viz.renderer.ensure_object.call_args_list
            if call.args[0].id == make_node_geometry_name("rx", 0, "marker")
        ]
        assert rx_objects
        np.testing.assert_allclose(_payload_extent(rx_objects[-1].payload), [1.8, 1.8, 0.0])
        np.testing.assert_allclose(rx_objects[-1].transform.translation, [4.0, 5.0, 6.0])

    def test_update_tx_rx_visibility_registers_visible_label_via_entity_sync(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        mock_viz.tx_labels = [_node_label_state("tx", 0)]
        mock_viz.app_state.show_labels = True
        mock_viz.renderer.set_geometry_visible = Mock()
        mock_viz.renderer.set_geometry_transform_fast = Mock()
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        label_name = make_node_geometry_name("tx", 0, "label")
        _assert_label_ensured(mock_viz, label_name, mock_viz.tx_labels[0])

    def test_repeated_tx_rx_visibility_sync_keeps_stable_label_contract(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        label = _node_label_state("tx", 0)
        mock_viz.tx_labels = [label]
        mock_viz.app_state.show_labels = True
        service = NodeService(mock_viz)

        service.update_tx_rx_visibility()
        service.update_tx_rx_visibility()

        label_objects = _ensured_objects(mock_viz, label.id)
        assert len(label_objects) == 2
        assert all(obj.payload is label.payload for obj in label_objects)
        assert list(name for name in mock_viz._named_objects if name == label.id) == [label.id]

    def test_update_tx_rx_visibility_ensures_hidden_neutral_label(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        label = _node_label_state("tx", 0)
        mock_viz.tx_labels = [label]
        mock_viz.app_state.show_labels = False
        mock_viz.renderer.set_geometry_visible = Mock()
        mock_viz.renderer.set_geometry_transform_fast = Mock()
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        ensured = _ensured_object(mock_viz, label.id)
        assert label.visible is True
        assert ensured.visible is False

    def test_update_tx_rx_visibility_keeps_label_payload_backend_neutral(self):
        mock_viz = make_mock_visualizer(tx_count=1, rx_count=0)
        label = _node_label_state("tx", 0)
        mock_viz.tx_labels = [label]
        mock_viz.current_tx_positions = [[1.0, 2.0, 3.0]]
        mock_viz.app_state.show_labels = True
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)

        service = NodeService(mock_viz)
        service.update_tx_rx_visibility()

        _assert_label_ensured(
            mock_viz,
            make_node_geometry_name("tx", 0, "label"),
            label,
        )
        assert isinstance(label.payload, TextLabelPayload)

    def test_update_target_label_visibility_hides_individual_target_label(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        target_entry = {
            "entry_type": "target",
            "name": "Car 1",
            "mesh": Mock(),
            "show_label": False,
            "target_name": "car_1",
        }
        mock_viz.target_entries = [target_entry]
        label_name = make_target_entry_geometry_name(target_entry, "label")
        mock_viz.target_labels = [_label_state(label_name, text="Car 1")]

        service = NodeService(mock_viz)
        service.update_target_label_visibility()

        assert _ensured_object(mock_viz, label_name).visible is False

    def test_update_target_label_visibility_registers_visible_neutral_label(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)
        mock_viz.app_state.show_target_labels = True
        target_entry = {
            "entry_type": "target",
            "name": "Car 1",
            "mesh": Mock(),
            "show_label": True,
            "target_name": "car_1",
        }
        mock_viz.target_entries = [target_entry]
        label_name = make_target_entry_geometry_name(target_entry, "label")
        mock_viz.target_labels = [_label_state(label_name, text="Car 1")]

        service = NodeService(mock_viz)
        service.update_target_label_visibility()

        ensured = _ensured_object(mock_viz, label_name)
        assert isinstance(ensured.payload, TextLabelPayload)
        assert ensured.visible is True

    def test_update_target_label_visibility_resolves_sparse_labels_by_id(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.app_state.show_target_labels = True
        missing_entry = {
            "entry_type": "target",
            "name": "Missing",
            "mesh": None,
            "show_label": True,
            "target_name": "missing",
        }
        valid_entry = {
            "entry_type": "target",
            "name": "Car 1",
            "mesh": Mock(),
            "show_label": True,
            "target_name": "car_1",
        }
        mock_viz.target_entries = [missing_entry, valid_entry]
        label_name = make_target_entry_geometry_name(valid_entry, "label")
        mock_viz.target_labels = [_label_state(label_name, text="Car 1")]

        NodeService(mock_viz).update_target_label_visibility()

        assert _ensured_object(mock_viz, label_name).visible is True

    def test_update_target_label_visibility_hides_pov_target_label(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.renderer.has_named_geometry = Mock(return_value=True)
        mock_viz.app_state.camera_mode = "pov"
        mock_viz.app_state.pov_hidden_node = ("target", 0)
        mock_viz.app_state.show_target_labels = True
        target_entry = {
            "entry_type": "target",
            "name": "Car 1",
            "mesh": Mock(),
            "show_label": True,
            "target_name": "car_1",
        }
        mock_viz.target_entries = [target_entry]
        label_name = make_target_entry_geometry_name(target_entry, "label")
        mock_viz.target_labels = [_label_state(label_name, text="Car 1")]

        service = NodeService(mock_viz)
        service.update_target_label_visibility()

        assert _ensured_object(mock_viz, label_name).visible is False

    def test_update_target_label_visibility_ensures_hidden_neutral_label(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.renderer.has_named_geometry = Mock(return_value=False)
        mock_viz.app_state.show_target_labels = False
        target_entry = {
            "entry_type": "target",
            "name": "Car 1",
            "mesh": Mock(),
            "show_label": True,
            "target_name": "car_1",
        }
        mock_viz.target_entries = [target_entry]
        label = _label_state(
            make_target_entry_geometry_name(target_entry, "label"),
            text="Car 1",
        )
        mock_viz.target_labels = [label]

        service = NodeService(mock_viz)
        service.update_target_label_visibility()

        assert label.visible is True
        assert _ensured_object(mock_viz, label.id).visible is False

    def test_target_label_visibility_uses_one_renderer_batch(self):
        mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
        mock_viz.app_state.show_target_labels = True
        entries = [
            {
                "entry_type": "target",
                "name": f"Car {index + 1}",
                "target_name": f"car_{index + 1}",
                "mesh": Mock(),
                "show_label": True,
            }
            for index in range(2)
        ]
        mock_viz.target_entries = entries
        mock_viz.target_labels = [
            _label_state(make_target_entry_geometry_name(entry, "label")) for entry in entries
        ]
        service = NodeService(mock_viz)
        batch_depth = 0
        batch_events = []

        @contextmanager
        def _tracked_batch():
            nonlocal batch_depth
            batch_depth += 1
            batch_events.append("enter")
            try:
                yield
            finally:
                batch_events.append("exit")
                batch_depth -= 1

        def _assert_batched(*_args, **_kwargs):
            assert batch_depth == 1
            return True

        mock_viz.renderer.batch_updates = _tracked_batch
        mock_viz.renderer.request_redraw = Mock(side_effect=_assert_batched)
        service._node_render_sync.sync_label = Mock(side_effect=_assert_batched)

        service.update_target_label_visibility()

        assert service._node_render_sync.sync_label.call_count == 2
        assert batch_events == ["enter", "exit"]
        mock_viz.renderer.update_renderer.assert_not_called()


# Add more test classes as needed:
# - TestCommNodeEntries (for _refresh_comm_node_entries)
# - TestProcessPositionAndTargetData (complex integration test)
# - TestPopulateTxRxSelections (dropdown population)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
