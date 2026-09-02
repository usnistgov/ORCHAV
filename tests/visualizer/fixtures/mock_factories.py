"""
Mock factory functions for creating test doubles.

Provides reusable mocks for visualizer components with sensible defaults.
"""

from contextlib import contextmanager
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np

from visualizer.src.model import RenderObjectState, Transform
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.orientation_frame_payloads import make_orientation_frame_handle
from visualizer.src.services.object_identity import make_node_geometry_name
from visualizer.src.services.target_service import TargetService
from visualizer.src.types.render_payloads import MeshPayload


@contextmanager
def _mock_batch_updates():
    """Mock batch_updates context manager for test renderers."""
    yield


class RecordingBatchRenderer:
    """Record semantic batches and present only changed object snapshots."""

    capabilities = RendererCapabilities(static_mesh_batching=True)

    def __init__(self) -> None:
        self.batch_depth = 0
        self.outer_batch_count = 0
        self.update_renderer_calls = 0
        self.operation_depths: list[tuple[str, int]] = []
        self._snapshots: dict[str, tuple] = {}
        self._batch_dirty = False

    @staticmethod
    def _snapshot_signature(render_object) -> tuple:
        return (
            id(render_object.payload),
            repr(render_object.material_payload),
            render_object.transform_matrix.tobytes(),
            bool(render_object.visible),
            bool(render_object.is_edge),
            repr(render_object.metadata),
        )

    @contextmanager
    def batch_updates(self):
        outermost = self.batch_depth == 0
        if outermost:
            self.outer_batch_count += 1
        self.batch_depth += 1
        try:
            yield
        finally:
            self.batch_depth -= 1
            if outermost and self._batch_dirty:
                self._batch_dirty = False
                self.update_renderer()

    def ensure_object(self, render_object) -> bool:
        if self.batch_depth <= 0:
            raise AssertionError(f"{render_object.id} synchronized outside a renderer batch")
        self.operation_depths.append((render_object.id, self.batch_depth))
        signature = self._snapshot_signature(render_object)
        if self._snapshots.get(render_object.id) != signature:
            self._snapshots[render_object.id] = signature
            self._batch_dirty = True
        return True

    def update_renderer(self) -> None:
        self.update_renderer_calls += 1


def _make_marker_handle(kind: str, index: int, position: list[float]) -> RenderObjectState:
    vertices = np.asarray(
        [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    return RenderObjectState(
        id=make_node_geometry_name(kind, index, "marker"),
        payload=MeshPayload(vertices=vertices, triangles=triangles),
        world_transform=Transform.from_translation(position),
    )


def make_mock_visualizer(
    tx_count: int = 2,
    rx_count: int = 2,
    has_coverage: bool = False,
    vis_initialized: bool = True,
    has_services: bool = True,
) -> Mock:
    """
    Create a mock visualizer with standard configuration.

    Args:
        tx_count: Number of TX markers/positions
        rx_count: Number of RX markers/positions
        has_coverage: Whether to include coverage service
        vis_initialized: Whether vis is initialized
        has_services: Whether to include service mocks

    Returns:
        Mock visualizer object
    """
    mock_viz = Mock()

    # Basic state
    mock_viz.vis_initialized = vis_initialized
    mock_viz.num_tx = tx_count
    mock_viz.num_rx = rx_count

    # TX/RX markers and labels
    mock_viz.tx_markers = []
    for i in range(tx_count):
        mock_viz.tx_markers.append(_make_marker_handle("tx", i, [i * 5.0, 0.0, 1.5]))

    mock_viz.rx_markers = []
    for i in range(rx_count):
        mock_viz.rx_markers.append(_make_marker_handle("rx", i, [i * 5.0, 10.0, 1.5]))

    mock_viz.tx_labels = []
    for i in range(tx_count):
        label = Mock()
        label.get_center = Mock(return_value=np.array([i * 5.0, 0.0, 2.0]))
        label.translate = Mock()
        label.paint_uniform_color = Mock()
        mock_viz.tx_labels.append(label)

    mock_viz.rx_labels = []
    for i in range(rx_count):
        label = Mock()
        label.get_center = Mock(return_value=np.array([i * 5.0, 10.0, 2.0]))
        label.translate = Mock()
        label.paint_uniform_color = Mock()
        mock_viz.rx_labels.append(label)

    # Positions
    mock_viz.current_tx_positions = [[i * 5.0, 0.0, 1.5] for i in range(tx_count)]
    mock_viz.current_rx_positions = [[i * 5.0, 10.0, 1.5] for i in range(rx_count)]

    # Orientations (default to empty or zeros)
    mock_viz.current_tx_orientations = [[0.0, 0.0, 0.0] for _ in range(tx_count)]
    mock_viz.current_rx_orientations = [[0.0, 0.0, 0.0] for _ in range(rx_count)]

    # Orientation frames
    mock_viz.tx_orientation_frames = [
        make_orientation_frame_handle(
            make_node_geometry_name("tx", i, "orientation_frame"),
            size=3.0,
            visible=False,
        )
        for i in range(tx_count)
    ]
    mock_viz.rx_orientation_frames = [
        make_orientation_frame_handle(
            make_node_geometry_name("rx", i, "orientation_frame"),
            size=3.0,
            visible=False,
        )
        for i in range(rx_count)
    ]
    mock_viz.target_orientation_frames = []

    mock_viz.selected_objects = set()

    # Runtime-only orientation state
    mock_viz.show_tx_orientation = False
    mock_viz.show_rx_orientation = False
    mock_viz.show_target_orientation = False
    mock_viz.orientation_scale = 3.0

    # Label offsets
    mock_viz.label_offset_x = 0.0
    mock_viz.label_offset_y = 0.0
    mock_viz.label_offset_z = 0.5
    mock_viz.label_font_size = 0.5
    mock_viz.tx_marker_size = 0.3
    mock_viz.rx_marker_size = 0.3
    mock_viz.node_marker_config = {"default": {"shape": "sphere", "center": True}}

    # Node coloring
    mock_viz.node_coloring_mode = "per_type"
    mock_viz.individual_node_colors = []

    # Entries for object panel
    mock_viz.tx_entries = []
    mock_viz.rx_entries = []
    mock_viz.target_entries = []
    mock_viz.mesh_entries = []

    # Open3D visualizer mock
    if vis_initialized:
        mock_viz.vis = Mock()
        mock_viz.vis.add_geometry = Mock()
        mock_viz.vis.remove_geometry = Mock()
        mock_viz.vis.update_geometry = Mock()
        mock_viz.vis.update_renderer = Mock()
        mock_viz.vis.poll_events = Mock()
    else:
        mock_viz.vis = None

    # Renderer mock
    mock_viz.renderer = Mock()
    mock_viz.renderer.capabilities = RendererCapabilities()
    _default_sphere = Mock()
    _default_sphere.get_center = Mock(return_value=np.array([0.0, 0.0, 0.0]))
    mock_viz.renderer.create_sphere = Mock(return_value=_default_sphere)
    mock_viz.renderer.translate_geometry = Mock()
    mock_viz.renderer.batch_updates = _mock_batch_updates  # Support batch updates
    mock_viz.renderer.get_named_position = Mock(return_value=None)
    mock_viz.renderer.renderer_type = "mock"
    mock_viz.renderer.begin_frame_update = Mock()
    mock_viz.renderer.end_frame_update = Mock(return_value=True)
    mock_viz._named_objects = {}
    mock_viz._named_visibility = {}

    def _ensure_object(render_object):
        mock_viz._named_objects[render_object.id] = render_object
        mock_viz._named_visibility[render_object.id] = bool(render_object.visible)
        return True

    def _remove_object(object_id):
        mock_viz._named_objects.pop(object_id, None)
        mock_viz._named_visibility.pop(object_id, None)
        return True

    def _set_visible(object_id, visible):
        mock_viz._named_visibility[str(object_id)] = bool(visible)
        return True

    def _has_named_geometry(object_id):
        return str(object_id) in mock_viz._named_objects

    def _is_named_visible(object_id):
        return mock_viz._named_visibility.get(str(object_id))

    def _add_or_update_named_geometry(name, geometry, **_kwargs):
        mock_viz._named_objects[str(name)] = geometry
        mock_viz._named_visibility.setdefault(str(name), True)
        return True

    mock_viz.renderer.ensure_object = Mock(side_effect=_ensure_object)
    mock_viz.renderer.remove_object = Mock(side_effect=_remove_object)
    mock_viz.renderer.set_visible = Mock(side_effect=_set_visible)
    mock_viz.renderer.set_transform = Mock(return_value=True)
    mock_viz.renderer.has_named_geometry = Mock(side_effect=_has_named_geometry)
    mock_viz.renderer.get_named_geometry_names = Mock(
        side_effect=lambda: tuple(sorted(mock_viz._named_objects))
    )
    mock_viz.renderer.is_named_visible = Mock(side_effect=_is_named_visible)
    mock_viz.renderer.ensure_named_geometry = Mock(side_effect=_add_or_update_named_geometry)
    mock_viz.renderer.add_or_update_named_geometry = Mock(side_effect=_add_or_update_named_geometry)
    mock_viz.renderer.remove_named_geometry = Mock(side_effect=_remove_object)
    mock_viz.renderer.set_named_visibility = Mock(side_effect=_set_visible)

    # Services
    if has_services:
        mock_viz.animation_service = Mock()
        mock_viz.material_mode_service = Mock()
        mock_viz.material_mode_command_service = Mock()
        mock_viz.material_entry_edit_service = Mock()
        mock_viz.scene_service = Mock()
        if has_coverage:
            mock_viz.coverage_service = Mock()

    # UI Manager
    mock_viz.ui_manager = Mock()
    mock_viz.ui_manager.panels = {}

    # Target geometry labels (Open3D text meshes, not state labels)
    mock_viz.target_labels = []

    # App state (authoritative selection, visibility, POV, and custom labels)
    mock_viz.app_state = Mock()
    mock_viz.app_state.selected_tx = "all"
    mock_viz.app_state.selected_rx = "all"
    mock_viz.app_state.show_labels = True
    mock_viz.app_state.pov_hidden_node = None  # No POV node hidden by default
    mock_viz.app_state.node_label_mode = "role"
    mock_viz.app_state.tx_labels = ()
    mock_viz.app_state.rx_labels = ()
    mock_viz.app_state.tx_device_names = ()
    mock_viz.app_state.rx_device_names = ()
    mock_viz.app_state.target_labels = ()
    mock_viz.app_state.show_target_labels = True

    # Mirror application composition: NodeService consumes this existing
    # target owner and must never construct a second one implicitly.
    mock_viz.target_service = TargetService(mock_viz)

    # Legend labels (for node coloring)
    mock_viz.tx_legend_label = Mock()
    mock_viz.rx_legend_label = Mock()
    mock_viz.tx_rx_legend_layout = Mock()
    mock_viz.tx_rx_legend_layout.count = Mock(return_value=0)
    mock_viz.tx_rx_legend_layout.itemAt = Mock(return_value=None)
    mock_viz.tx_rx_legend_layout.addWidget = Mock()

    return mock_viz


def make_mock_frame_data(
    num_tx: int = 2,
    num_rx: int = 2,
    num_targets: int = 1,
    num_tx_rx_pairs: int = 4,
    has_orientations: bool = True,
) -> Dict[str, Any]:
    """
    Generate synthetic frame data for testing.

    Args:
        num_tx: Number of transmitters
        num_rx: Number of receivers
        num_targets: Number of targets
        num_tx_rx_pairs: Number of TX-RX pairs
        has_orientations: Include orientation data

    Returns:
        Dictionary with frame data
    """
    from tests.visualizer.fixtures.synthetic_data import create_synthetic_frame_data

    return create_synthetic_frame_data(
        num_tx=num_tx,
        num_rx=num_rx,
        num_targets=num_targets,
        num_tx_rx_pairs=num_tx_rx_pairs,
        include_orientations=has_orientations,
    )


def make_mock_view_model(
    num_paths: int = 10,
    color_mode: str = "delay",
) -> Mock:
    """
    Create a minimal ViewModel mock.

    Args:
        num_paths: Number of MPC paths
        color_mode: Color mode (delay, path_loss, material)

    Returns:
        Mock ViewModel
    """
    mock_vm = Mock()

    # MPC data
    mock_vm.mpc_points = np.random.uniform(-10, 10, (num_paths * 2, 3))
    mock_vm.mpc_colors = np.random.uniform(0, 1, (num_paths * 2, 3))
    mock_vm.mpc_segments = np.array([[i * 2, i * 2 + 1] for i in range(num_paths)])

    # Metadata
    mock_vm.color_mode = color_mode
    mock_vm.delay_range = [0.0, 100.0]
    mock_vm.path_loss_range = [30.0, 80.0]

    # TX/RX data
    mock_vm.tx_positions = [[i * 5.0, 0.0, 1.5] for i in range(2)]
    mock_vm.rx_positions = [[i * 5.0, 10.0, 1.5] for i in range(2)]

    return mock_vm


def make_mock_open3d_geometry(geom_type: str = "mesh") -> Mock:
    """
    Create a mock Open3D geometry object.

    Args:
        geom_type: Type of geometry (mesh, sphere, lineset, pcd)

    Returns:
        Mock geometry object
    """
    mock_geom = Mock()

    # Common methods
    mock_geom.get_center = Mock(return_value=np.array([0.0, 0.0, 0.0]))
    mock_geom.translate = Mock()
    mock_geom.rotate = Mock()
    mock_geom.scale = Mock()
    mock_geom.transform = Mock()
    mock_geom.paint_uniform_color = Mock()
    mock_geom.compute_vertex_normals = Mock()

    # Properties
    if geom_type == "mesh":
        mock_geom.vertices = Mock()
        mock_geom.triangles = Mock()
        mock_geom.vertex_normals = Mock()
        mock_geom.vertex_colors = Mock()
        mock_geom.has_vertex_normals = Mock(return_value=True)
    elif geom_type == "lineset":
        mock_geom.points = Mock()
        mock_geom.lines = Mock()
        mock_geom.colors = Mock()
    elif geom_type == "pcd":
        mock_geom.points = Mock()
        mock_geom.colors = Mock()
        mock_geom.normals = Mock()

    # Visibility (if supported)
    mock_geom.set_visible = Mock()

    return mock_geom
