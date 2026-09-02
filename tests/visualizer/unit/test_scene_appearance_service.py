"""Tests for SceneAppearanceService."""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.model import RenderObjectState
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.defaults import (
    DEFAULT_SCENE_BACKGROUND_COLOR,
    DEFAULT_SCENE_BACKGROUND_PRESET,
)
from visualizer.src.services.scene_appearance_service import SceneAppearanceService
from visualizer.src.types.render_payloads import MeshPayload


class MockRenderer:
    """Mock renderer for testing."""

    def __init__(self):
        self.background_color = [0.0, 0.0, 0.0]
        self.geometry_updates = []
        self.added_geometries = []
        self.removed_geometries = []
        self.capabilities = RendererCapabilities(transparency=True)
        self.transparency_updates = []
        self.material_updates = []
        self.render_updates = 0

    def set_background_color(self, color: List[float]) -> None:
        self.background_color = list(color)

    def update_geometry_in_visualizer(self, mesh) -> None:
        self.geometry_updates.append(mesh)

    def add_geometry_to_visualizer(self, geometry, reset_bounding_box: bool = False) -> None:
        self.added_geometries.append(geometry)

    def remove_geometry_from_visualizer(self, geometry, reset_bounding_box: bool = False) -> None:
        self.removed_geometries.append(geometry)

    def update_renderer(self) -> None:
        self.render_updates += 1

    def set_geometry_transparency(
        self,
        mesh,
        alpha: float,
        color: List[float],
        *,
        defer_redraw: bool,
        roughness: float,
        metallic: float,
        reflectance: float,
    ) -> None:
        self.transparency_updates.append(
            {
                "mesh": mesh,
                "alpha": alpha,
                "color": color,
                "defer_redraw": defer_redraw,
                "roughness": roughness,
                "metallic": metallic,
                "reflectance": reflectance,
            }
        )

    def set_named_material(self, name: str, material: Dict[str, Any]) -> bool:
        self.material_updates.append({"name": name, "material": material})
        return True


class MockVisualizer:
    """Mock visualizer for testing."""

    def __init__(self):
        self.renderer = MockRenderer()
        self.current_background_color = list(DEFAULT_SCENE_BACKGROUND_COLOR)
        self.current_background_preset = DEFAULT_SCENE_BACKGROUND_PRESET
        self.mesh_entries: List[Dict[str, Any]] = []
        self.target_entries: List[Dict[str, Any]] = []
        self.vis = SimpleNamespace()
        self.vis_initialized = True
        self.outline_color = [0.0, 0.0, 0.0]
        self.outlines_enabled = False
        self.current_building_alpha = 1.0
        self.current_target_alpha = 1.0
        self.target_service = SimpleNamespace(
            refresh_target_entry_material=Mock(return_value=True),
        )


@pytest.fixture
def mock_visualizer() -> MockVisualizer:
    return MockVisualizer()


@pytest.fixture
def service(mock_visualizer: MockVisualizer) -> SceneAppearanceService:
    return SceneAppearanceService(mock_visualizer)


# =============================================================================
# Color Name Detection Tests
# =============================================================================


def test_get_color_name_black(service: SceneAppearanceService) -> None:
    """Verify black color is correctly identified."""
    assert service.get_color_name([0.0, 0.0, 0.0]) == "Black"
    assert service.get_color_name([0.05, 0.05, 0.05]) == "Black"


def test_get_color_name_white(service: SceneAppearanceService) -> None:
    """Verify white color is correctly identified."""
    assert service.get_color_name([1.0, 1.0, 1.0]) == "White"
    assert service.get_color_name([0.95, 0.95, 0.95]) == "White"


def test_get_color_name_grays(service: SceneAppearanceService) -> None:
    """Verify gray shades are correctly identified."""
    assert service.get_color_name([0.2, 0.2, 0.2]) == "Dark Gray"
    assert service.get_color_name([0.5, 0.5, 0.5]) == "Gray"
    assert service.get_color_name([0.7, 0.7, 0.7]) == "Light Gray"


def test_get_color_name_custom(service: SceneAppearanceService) -> None:
    """Verify non-gray colors are identified as Custom."""
    assert service.get_color_name([1.0, 0.0, 0.0]) == "Custom"  # Red
    assert service.get_color_name([0.0, 1.0, 0.0]) == "Custom"  # Green
    assert service.get_color_name([0.0, 0.0, 1.0]) == "Custom"  # Blue


# =============================================================================
# Background Color Tests
# =============================================================================


def test_set_background_color(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify set_background_color updates renderer and visualizer state."""
    mock_visualizer._set_status_message = Mock()

    service.set_background_color([0.5, 0.5, 0.5])

    assert mock_visualizer.renderer.background_color == [0.5, 0.5, 0.5]
    assert mock_visualizer.current_background_color == [0.5, 0.5, 0.5]
    mock_visualizer._set_status_message.assert_called_once_with(
        "Background: Gray RGB(0.5, 0.5, 0.5)",
        5000,
    )


def test_set_background_color_no_renderer(mock_visualizer: MockVisualizer) -> None:
    """Verify set_background_color handles missing renderer gracefully."""
    mock_visualizer.renderer = None
    service = SceneAppearanceService(mock_visualizer)

    # Should not raise
    service.set_background_color([0.5, 0.5, 0.5])


def test_ensure_light_gray_background(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify ensure_light_gray_background sets the default color."""
    mock_visualizer.current_background_color = [0.0, 0.0, 0.0]

    service.ensure_light_gray_background()

    assert mock_visualizer.current_background_color == [0.2, 0.2, 0.2]
    assert mock_visualizer.renderer.background_color == [0.2, 0.2, 0.2]


def test_reset_to_default_background(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify reset_to_default_background resets to the default dark gray."""
    mock_visualizer.current_background_color = [0.0, 0.0, 0.0]
    mock_visualizer.current_background_preset = "Dark"

    service.reset_to_default_background()

    assert mock_visualizer.current_background_preset == "Dark Gray"
    assert mock_visualizer.current_background_color == [0.2, 0.2, 0.2]


def test_set_background_preset(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify set_background_preset sets both preset name and color."""
    service.set_background_preset("Night", [0.1, 0.1, 0.15])

    assert mock_visualizer.current_background_preset == "Night"
    assert mock_visualizer.current_background_color == [0.1, 0.1, 0.15]


# =============================================================================
# Outline Tests
# =============================================================================


def test_set_edge_visibility(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify set_edge_visibility updates outline state."""
    mock_visualizer.outlines_enabled = False

    service.set_edge_visibility(True)

    assert mock_visualizer.outlines_enabled is True


def test_set_edge_visibility_no_mesh_entries(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    """Verify set_edge_visibility handles empty mesh entries."""
    mock_visualizer.mesh_entries = []

    # Should not raise
    service.set_edge_visibility(True)
    assert mock_visualizer.outlines_enabled is True


# =============================================================================
# Transparency Tests
# =============================================================================


def test_set_building_transparency_updates_scene_meshes(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    mesh = object()
    refresh_all = Mock(return_value=True)
    mock_visualizer.scene_service = SimpleNamespace(
        refresh_all_scene_materials=refresh_all,
    )
    mock_visualizer.mesh_entries = [
        {
            "mesh": mesh,
            "color": [0.2, 0.3, 0.4],
            "pbr_properties": {"roughness": 0.8, "metallic": 0.1, "reflectance": 0.6},
        }
    ]

    service.set_building_transparency(0.55)

    assert mock_visualizer.current_building_alpha == 0.55
    refresh_all.assert_called_once_with()
    assert mock_visualizer.renderer.transparency_updates == []
    assert mock_visualizer.renderer.material_updates == []
    assert mock_visualizer.renderer.render_updates == 1


def test_scene_transparency_delegates_to_object_appearance_service() -> None:
    mesh = RenderObjectState(
        id="scene:wall::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        ),
    )
    renderer = SimpleNamespace()
    refresh_material = Mock(return_value=True)
    entry = {
        "entry_type": "mesh",
        "mesh": mesh,
        "visible": True,
        "color": [0.2, 0.3, 0.4],
        "pbr_properties": {"roughness": 0.8},
    }
    service = SceneAppearanceService(
        SimpleNamespace(
            renderer=renderer,
            mesh_entries=[entry],
            target_entries=[],
            current_building_alpha=0.35,
            object_appearance_service=SimpleNamespace(refresh_entry_material=refresh_material),
        )
    )

    assert service._set_entry_transparency(entry, 0.35)

    refresh_material.assert_called_once_with(entry, update_renderer=False)
    assert mesh.material.base_color == (1.0, 1.0, 1.0, 1.0)


def test_target_transparency_delegates_to_object_appearance_service() -> None:
    mesh = RenderObjectState(
        id="target:walker::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        ),
    )
    renderer = SimpleNamespace()
    refresh_material = Mock(return_value=True)
    entry = {
        "entry_type": "target",
        "mesh": mesh,
        "visible": True,
        "_frame_visible": False,
        "color": [0.4, 0.5, 0.6],
    }
    service = SceneAppearanceService(
        SimpleNamespace(
            renderer=renderer,
            mesh_entries=[],
            target_entries=[entry],
            current_target_alpha=0.45,
            object_appearance_service=SimpleNamespace(refresh_entry_material=refresh_material),
        )
    )

    assert service._set_entry_transparency(entry, 0.45)

    refresh_material.assert_called_once_with(entry, update_renderer=False)
    assert mesh.visible is True


def test_set_target_transparency_respects_renderer_capability(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    mock_visualizer.renderer.capabilities = RendererCapabilities()
    mock_visualizer.target_entries = [{"mesh": object()}]

    service.set_target_transparency(0.4)

    assert mock_visualizer.current_target_alpha == 1.0
    assert mock_visualizer.renderer.transparency_updates == []


def test_set_building_transparency_refreshes_all_scene_owners_once(
    service: SceneAppearanceService, mock_visualizer: MockVisualizer
) -> None:
    mesh_a = object()
    mesh_b = object()
    unmerged = object()
    mock_visualizer.mesh_entries = [
        {"mesh": mesh_a},
        {"mesh": mesh_b},
        {"mesh": unmerged, "color": [0.1, 0.2, 0.3]},
    ]
    refresh_all = Mock(return_value=True)
    mock_visualizer.scene_service = SimpleNamespace(
        refresh_all_scene_materials=refresh_all,
    )

    service.set_building_transparency(0.25)

    refresh_all.assert_called_once_with()
    assert mock_visualizer.renderer.transparency_updates == []
    assert mock_visualizer.renderer.material_updates == []
    assert mock_visualizer.renderer.render_updates == 1
