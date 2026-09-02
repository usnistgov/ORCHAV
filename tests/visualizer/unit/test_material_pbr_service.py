"""Tests for MaterialPBRService."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock, call

import numpy as np
import pytest

from visualizer.src.materials.appearance import (
    VisualMaterialBinding,
    VisualMaterialSource,
)
from visualizer.src.model import RenderObjectState
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.types.render_payloads import MeshPayload, SurfaceColorSource


class MockRenderer:
    """Mock renderer for testing."""

    def __init__(self):
        self.renderer_type = "mock"
        self.capabilities = RendererCapabilities()
        self.updated_geometries = []

    def update_geometry_in_visualizer(self, mesh) -> None:
        self.updated_geometries.append(mesh)

    def update_renderer(self) -> None:
        pass

    def ensure_object(self, obj: Any) -> bool:
        self.updated_geometries.append(obj)
        return True


class MockPBRRenderer(MockRenderer):
    """Mock PBR renderer that records named material updates."""

    def __init__(self):
        super().__init__()
        self.renderer_type = "open3d"
        self.capabilities = RendererCapabilities(pbr=True)
        self.material_updates = []
        self.object_material_updates = []
        self.added_geometries = []

    def set_material(self, name: str, material: Any) -> bool:
        self.object_material_updates.append((name, material))
        return True

    def set_named_material(self, name: str, material: Dict[str, Any]) -> bool:
        self.material_updates.append((name, material))
        return True

    def ensure_named_geometry(
        self,
        name: str,
        geometry: Any,
        material: Dict[str, Any] | None = None,
        visible: bool | None = None,
        **_kwargs: Any,
    ) -> bool:
        self.added_geometries.append((name, geometry, material, visible))
        return True


def _mesh_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def _scene_state(stable_id: str) -> RenderObjectState:
    return RenderObjectState(
        id=f"scene:{stable_id}::mesh",
        payload=_mesh_payload(),
        metadata={"type": "scene_mesh"},
    )


def _target_state(stable_id: str) -> RenderObjectState:
    return RenderObjectState(
        id=f"target:{stable_id}::mesh",
        payload=_mesh_payload(),
        metadata={"type": "target_mesh"},
    )


class MockVisualizer:
    """Mock visualizer for testing."""

    def __init__(self):
        self.renderer = MockRenderer()
        self.mesh_entries: List[Dict[str, Any]] = []
        self.target_entries: List[Dict[str, Any]] = []
        self.scene_service = SimpleNamespace(
            render_scene=Mock(),
        )
        self.target_service = SimpleNamespace(
            refresh_target_entry_material=Mock(return_value=True),
        )
        self.object_appearance_service = SimpleNamespace(
            refresh_entry_appearance_batch=Mock(return_value=True),
        )


@pytest.fixture
def mock_visualizer() -> MockVisualizer:
    return MockVisualizer()


@pytest.fixture
def service(
    tmp_path: Path, mock_visualizer: MockVisualizer, monkeypatch: pytest.MonkeyPatch
) -> MaterialPBRService:
    """Create a MaterialPBRService with a temporary preset directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return MaterialPBRService(mock_visualizer)


# =============================================================================
# Built-in Presets Tests
# =============================================================================


def test_built_in_presets_exist() -> None:
    """Verify built-in presets are defined."""
    assert len(MaterialPBRService.PRESETS) > 0


def test_built_in_presets_have_required_properties() -> None:
    """Verify all built-in presets have required PBR properties."""
    required_props = ["roughness", "metallic", "reflectance", "alpha"]

    for name, preset in MaterialPBRService.PRESETS.items():
        for prop in required_props:
            assert prop in preset, f"Preset '{name}' missing property '{prop}'"


def test_mirror_preset_has_low_roughness() -> None:
    """Verify 'Mirror/Polished' preset has very low roughness."""
    preset = MaterialPBRService.PRESETS.get("Mirror/Polished")
    assert preset is not None
    assert preset["roughness"] < 0.1


def test_matte_preset_has_high_roughness() -> None:
    """Verify 'Matte' preset has high roughness."""
    preset = MaterialPBRService.PRESETS.get("Matte")
    assert preset is not None
    assert preset["roughness"] > 0.8


def test_metal_presets_have_high_metallic() -> None:
    """Verify metal presets have metallic=1.0."""
    metal_presets = ["Polished Metal", "Brushed Metal", "Gold", "Copper", "Silver"]

    for preset_name in metal_presets:
        preset = MaterialPBRService.PRESETS.get(preset_name)
        assert preset is not None, f"Missing metal preset: {preset_name}"
        assert preset["metallic"] >= 0.9, f"{preset_name} should have high metallic"


def test_glass_presets_have_low_alpha() -> None:
    """Verify glass presets have low alpha (transparency)."""
    glass_presets = ["Clear Glass", "Frosted Glass", "Tinted Glass"]

    for preset_name in glass_presets:
        preset = MaterialPBRService.PRESETS.get(preset_name)
        assert preset is not None, f"Missing glass preset: {preset_name}"
        assert preset["alpha"] < 0.5, f"{preset_name} should have low alpha"


# =============================================================================
# Property Override Tests
# =============================================================================


def test_set_property_creates_override(service: MaterialPBRService) -> None:
    """Verify set_property creates an override entry."""
    service.set_property("concrete", "roughness", 0.9)

    assert "concrete" in service.overrides
    assert service.overrides["concrete"]["roughness"] == 0.9


def test_set_property_updates_existing_override(service: MaterialPBRService) -> None:
    """Verify set_property updates existing override."""
    service.set_property("concrete", "roughness", 0.5)
    service.set_property("concrete", "roughness", 0.8)

    assert service.overrides["concrete"]["roughness"] == 0.8


def test_set_property_supports_color(service: MaterialPBRService) -> None:
    """Verify set_property supports color property as list."""
    service.set_property("concrete", "color", [1.0, 0.5, 0.0])

    assert service.overrides["concrete"]["color"] == [1.0, 0.5, 0.0]


def test_get_effective_properties_strips_texture_maps_when_textures_disabled(
    service: MaterialPBRService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Texture-disabled launches should keep color editable for textured materials."""
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    service.overrides["concrete"] = {
        "color": [0.1, 0.2, 0.3],
        "texture_path": "/tmp/override_albedo.png",
    }

    props = service.get_effective_properties(
        "concrete",
        {
            "color": [0.5, 0.5, 0.5],
            "texture_path": "/tmp/albedo.png",
            "normal_map_path": "/tmp/normal.png",
            "roughness_map_path": "/tmp/roughness.png",
            "ao_map_path": "/tmp/ao.png",
            "metallic_map_path": "/tmp/metallic.png",
            "normal_map_strength": 1.5,
        },
    )

    assert props["color"] == [0.1, 0.2, 0.3]
    assert props["texture_path"] is None
    assert props["normal_map_path"] is None
    assert props["roughness_map_path"] is None
    assert props["ao_map_path"] is None
    assert props["metallic_map_path"] is None
    assert props["normal_map_strength"] == 1.5


def test_texture_disabled_entry_color_edit_updates_fallback_without_losing_albedo(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inactive authored albedo must not block editing its uniform fallback."""
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    albedo = tmp_path / "authored_albedo.png"
    albedo.write_bytes(b"texture")
    entry = {
        "name": "Textured wall",
        "entry_type": "mesh",
        "mesh": _scene_state("textured_wall"),
        "material_type": "wood",
        "pbr_properties": {
            "color": [0.6, 0.5, 0.4],
            "texture_path": str(albedo),
            "roughness": 0.7,
        },
    }
    mock_visualizer.mesh_entries = [entry]

    assert service.set_property("wood", "color", [0.1, 0.4, 0.8]) is True

    fallback = service.resolve_entry_material(entry)
    assert fallback.properties["color"] == (0.1, 0.4, 0.8)
    assert fallback.properties["texture_path"] is None
    binding = service.get_visual_binding(entry)
    assert binding.source is VisualMaterialSource.MANUAL
    assert binding.overrides["texture_path"] == str(albedo)

    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    textured = service.resolve_entry_material(entry)
    assert textured.texture_policy.active_albedo_path == str(albedo)
    assert textured.texture_policy.color_editable is False
    assert textured.payload.base_color[:3] == (1.0, 1.0, 1.0)


def test_missing_albedo_entry_accepts_color_without_discarding_declared_path(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A declared but inactive albedo must behave as a uniform fallback."""
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    missing_albedo = tmp_path / "missing_albedo.png"
    entry = {
        "name": "Missing texture wall",
        "entry_type": "mesh",
        "mesh": _scene_state("missing_texture_wall"),
        "material_type": "wood",
        "pbr_properties": {
            "color": [0.6, 0.5, 0.4],
            "texture_path": str(missing_albedo),
        },
    }
    mock_visualizer.mesh_entries = [entry]

    assert service.set_property("wood", "color", [0.2, 0.7, 0.3]) is True

    resolved = service.resolve_entry_material(entry)
    assert resolved.properties["color"] == (0.2, 0.7, 0.3)
    assert resolved.texture_policy.active_albedo_path is None
    assert resolved.texture_policy.color_editable is True
    binding = service.get_visual_binding(entry)
    assert binding.source is VisualMaterialSource.MANUAL
    assert binding.overrides["texture_path"] == str(missing_albedo)


def test_set_property_color_invalid_value(service: MaterialPBRService) -> None:
    """Verify set_property rejects invalid color values."""
    result = service.set_property("concrete", "color", "not a color")

    assert result is False
    assert "color" not in service.overrides.get("concrete", {})


def test_set_property_rejects_unknown_property(service: MaterialPBRService) -> None:
    """Unknown keys must not become silent renderer-specific material state."""
    result = service.set_property("concrete", "vendor_magic", 0.5)

    assert result is False
    assert "vendor_magic" not in service.overrides.get("concrete", {})


def test_follow_em_rebases_but_manual_and_profile_bindings_do_not(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": _scene_state("wall"),
        "material_type": "concrete",
        "pbr_properties": {"color": [0.4, 0.4, 0.4], "roughness": 0.7},
    }
    mock_visualizer.mesh_entries = [entry]

    assert service.set_property("concrete", "roughness", 0.91)
    manual = service.get_visual_binding(entry)
    assert manual.source is VisualMaterialSource.MANUAL
    assert service.get_visual_material_key(entry) == "concrete"

    entry["material_type"] = "glass"
    entry["pbr_properties"] = {"color": [0.2, 0.3, 0.8], "roughness": 0.05}
    assert service.get_effective_entry_properties(entry)["roughness"] == pytest.approx(0.91)
    assert service.get_visual_material_key(entry) == "concrete"

    service.clear_visual_binding(entry)
    assert service.get_visual_binding(entry).source is VisualMaterialSource.FOLLOW_EM
    assert service.get_effective_entry_properties(entry)["roughness"] == pytest.approx(0.05)
    assert service.get_visual_material_key(entry) == "glass"

    service.set_visual_binding(
        entry,
        VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="glass",
            preset="Gold",
        ),
    )
    entry["material_type"] = "brick"
    props = service.get_effective_entry_properties(entry)
    assert props["metallic"] == pytest.approx(1.0)
    assert props["color"] == MaterialPBRService.PRESETS["Gold"]["color"]
    assert service.get_visual_material_key(entry) == "glass"


def test_mixed_group_color_edit_changes_only_uniform_members(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    albedo = tmp_path / "brick.png"
    albedo.write_bytes(b"texture")
    uniform = {
        "name": "Uniform",
        "entry_type": "mesh",
        "mesh": _scene_state("uniform"),
        "material_type": "wood",
        "pbr_properties": {"color": [0.3, 0.2, 0.1], "roughness": 0.8},
    }
    textured = {
        "name": "Textured",
        "entry_type": "mesh",
        "mesh": _scene_state("textured"),
        "material_type": "wood",
        "pbr_properties": {
            "color": [0.9, 0.9, 0.9],
            "roughness": 0.7,
            "texture_path": str(albedo),
        },
    }
    mock_visualizer.mesh_entries = [uniform, textured]

    assert service.set_property("wood", "color", [0.1, 0.6, 0.2])

    assert service.get_effective_entry_properties(uniform)["color"] == [0.1, 0.6, 0.2]
    textured_props = service.get_effective_entry_properties(textured)
    assert textured_props["color"] == [0.9, 0.9, 0.9]
    assert textured_props["texture_path"] == str(albedo)
    assert service.get_visual_binding(uniform).source is VisualMaterialSource.MANUAL
    assert service.get_visual_binding(textured).source is VisualMaterialSource.FOLLOW_EM
    summary = service.summarize_material_group("wood")
    assert summary.uniform_color_members == 1
    assert summary.external_albedo_members == 1
    assert summary.color_editable is True


def test_vertex_color_entry_ignores_scalar_rgb_but_keeps_non_color_pbr(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    colors = np.asarray(
        [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]],
        dtype=float,
    )
    mesh = _scene_state("person")
    mesh.replace_payload(
        MeshPayload(
            vertices=mesh.payload.vertices,
            triangles=mesh.payload.triangles,
            vertex_colors=colors,
            color_source=SurfaceColorSource.VERTEX,
        )
    )
    entry = {
        "name": "Person",
        "entry_type": "mesh",
        "mesh": mesh,
        "material_type": "skin",
        "pbr_properties": {"color": [0.8, 0.6, 0.5], "roughness": 0.55},
    }
    mock_visualizer.mesh_entries = [entry]

    assert service.set_property("skin", "color", [0.0, 1.0, 0.0])
    assert service.set_property("skin", "roughness", 0.82)

    props = service.get_effective_entry_properties(entry)
    assert props["color"] == [1.0, 1.0, 1.0]
    assert props["roughness"] == pytest.approx(0.82)
    np.testing.assert_array_equal(mesh.payload.vertex_colors, colors)


def test_invalid_authored_and_saved_pbr_values_are_rejected_centrally(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    entry = {
        "name": "Bad",
        "entry_type": "mesh",
        "mesh": _scene_state("bad"),
        "material_type": "custom",
        "pbr_properties": {
            "color": [2.0, 0.0, 0.0],
            "roughness": float("nan"),
            "alpha": 4.0,
        },
    }
    mock_visualizer.mesh_entries = [entry]
    props = service.get_effective_entry_properties(entry)
    assert props["color"] == [0.7, 0.7, 0.7]
    assert props["roughness"] == pytest.approx(0.5)
    assert props["alpha"] == pytest.approx(1.0)

    service.overrides = {"concrete": {"roughness": 0.75}}
    (service.preset_dir / "invalid.json").write_text(
        '{"name":"invalid","overrides":{"concrete":{"roughness":9.0}}}'
    )
    assert service.load_preset("invalid") is False
    assert service.overrides == {"concrete": {"roughness": 0.75}}


def test_get_property_returns_override(service: MaterialPBRService) -> None:
    """Verify get_property returns override when set."""
    service.overrides["glass"] = {"roughness": 0.1}

    result = service.get_property("glass", "roughness", {"roughness": 0.5})

    assert result == 0.1


def test_get_property_returns_base_when_no_override(service: MaterialPBRService) -> None:
    """Verify get_property returns base value when no override."""
    result = service.get_property("glass", "roughness", {"roughness": 0.5})

    assert result == 0.5


def test_get_property_returns_zero_when_no_base(service: MaterialPBRService) -> None:
    """Verify get_property returns 0.0 when property not found anywhere."""
    result = service.get_property("glass", "roughness", {})

    assert result == 0.0


def test_get_effective_properties_merges_override(service: MaterialPBRService) -> None:
    """Verify get_effective_properties merges base and override."""
    service.overrides["concrete"] = {"roughness": 0.9}
    base_props = {"roughness": 0.5, "metallic": 0.0}

    result = service.get_effective_properties("concrete", base_props)

    assert result["roughness"] == 0.9  # Overridden
    assert result["metallic"] == 0.0  # From base


def test_get_entry_base_properties_preserves_resolved_material_id_texture(
    service: MaterialPBRService,
) -> None:
    """Material-panel rewrites should keep explicitly resolved texture IDs."""
    entry = {
        "material_type": "custom_wood",
        "pbr_properties": {
            "color": [0.3, 0.2, 0.1],
            "roughness": 0.7,
            "metallic": 0.0,
            "reflectance": 0.3,
            "alpha": 1.0,
        },
        "_resolved_texture_maps": {
            "texture_path": "/tmp/custom_wood.png",
        },
    }

    props = service.get_entry_base_properties(entry)

    assert props["texture_path"] == "/tmp/custom_wood.png"


def test_set_property_delegates_scene_material_refresh(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Material-type updates delegate to the scene's complete snapshot owner."""
    renderer = MockPBRRenderer()
    mock_visualizer.renderer = renderer
    mesh = _scene_state("wall_1")
    mock_visualizer.mesh_entries = [
        {
            "name": "Wall 1",
            "entry_type": "mesh",
            "mesh": mesh,
            "material_type": "default",
            "material_id": "mat-default",
            "color": [0.5, 0.5, 0.5],
            "pbr_properties": {
                "color": [0.5, 0.5, 0.5],
                "roughness": 0.6,
                "metallic": 0.0,
                "reflectance": 0.4,
            },
        }
    ]

    assert service.set_property("default", "color", [0.2, 0.3, 0.4])

    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        mock_visualizer.mesh_entries
    )
    assert mock_visualizer.mesh_entries[0]["color"] == [0.5, 0.5, 0.5]
    assert service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])["color"] == [
        0.2,
        0.3,
        0.4,
    ]
    assert mesh.id == "scene:wall_1::mesh"
    assert renderer.object_material_updates == []
    assert renderer.material_updates == []


def test_set_property_updates_wood_without_texture_maps(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Wood has no PBR albedo pack, so color edits stay flat in textured mode."""
    renderer = MockPBRRenderer()
    mock_visualizer.renderer = renderer
    mesh = _scene_state("wood_building")
    mock_visualizer.mesh_entries = [
        {
            "name": "Wood Building",
            "entry_type": "mesh",
            "mesh": mesh,
            "material_type": "wood",
            "material_id": "mat-itu_wood",
            "color": [0.27, 0.11, 0.06],
            "pbr_properties": {
                "color": [0.27, 0.11, 0.06],
                "roughness": 0.7,
                "metallic": 0.0,
                "reflectance": 0.3,
                "alpha": 1.0,
            },
        }
    ]

    assert service.set_property("wood", "color", [0.08, 0.47, 0.88])

    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        mock_visualizer.mesh_entries
    )
    props = service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])
    assert props["color"] == [0.08, 0.47, 0.88]
    assert props.get("texture_path") is None
    assert props.get("normal_map_path") is None
    assert props.get("roughness_map_path") is None
    assert props.get("ao_map_path") is None
    assert props.get("metallic_map_path") is None
    assert renderer.object_material_updates == []


def test_apply_presets_switch_texture_no_texture_and_back(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching presets must not leave stale texture maps behind."""
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    renderer = MockPBRRenderer()
    mock_visualizer.renderer = renderer
    mesh = _scene_state("switch_wall")
    mock_visualizer.mesh_entries = [
        {
            "name": "Switch Wall",
            "entry_type": "mesh",
            "mesh": mesh,
            "material_type": "concrete",
            "material_id": "mat-itu_concrete",
            "color": [0.48, 0.48, 0.45],
            "pbr_properties": {
                "color": [0.48, 0.48, 0.45],
                "roughness": 0.75,
                "metallic": 0.0,
                "reflectance": 0.35,
                "alpha": 1.0,
            },
        }
    ]

    assert service.apply_preset("Brick", material_type="concrete")
    brick_props = service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])
    assert brick_props["texture_path"] is not None

    assert service.apply_preset("Gold", material_type="concrete")
    gold_props = service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])
    assert gold_props["texture_path"] is None
    assert gold_props["normal_map_path"] is None
    assert gold_props["color"] == MaterialPBRService.PRESETS["Gold"]["color"]

    assert service.apply_preset("Concrete", material_type="concrete")
    concrete_props = service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])
    assert concrete_props["texture_path"] is not None

    assert service.apply_preset("Gold", material_type="concrete")
    back_to_no_texture = service.get_effective_entry_properties(mock_visualizer.mesh_entries[0])
    assert back_to_no_texture["texture_path"] is None
    assert back_to_no_texture["normal_map_path"] is None
    assert back_to_no_texture["roughness_map_path"] is None
    assert back_to_no_texture["ao_map_path"] is None
    assert back_to_no_texture["metallic_map_path"] is None
    assert renderer.object_material_updates == []
    assert (
        mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.call_args_list
        == [call(mock_visualizer.mesh_entries)] * 4
    )


def test_effective_entry_properties_apply_target_alpha_override(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Entry material resolution should preserve active global target alpha."""
    mock_visualizer.current_target_alpha = 0.35
    entry = {
        "name": "target_1",
        "entry_type": "target",
        "material_type": "glass",
        "color": [0.1, 0.2, 0.3],
        "pbr_alpha": 0.8,
    }

    props = service.get_effective_entry_properties(entry)

    assert props["color"] == [0.1, 0.2, 0.3]
    assert props["alpha"] == 0.35


def test_base_material_resolution_does_not_create_transient_highlight(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The material resolver leaves temporary highlight to the appearance resolver."""
    renderer = MockPBRRenderer()
    mock_visualizer.renderer = renderer
    mesh = _target_state("target_1")
    albedo = tmp_path / "albedo.png"
    albedo.write_bytes(b"texture")
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    entry = {
        "name": "target_1",
        "target_name": "target_1",
        "entry_type": "target",
        "mesh": mesh,
        "material_type": "concrete",
        "color": [0.2, 0.3, 0.4],
        "pbr_properties": {
            "color": [0.2, 0.3, 0.4],
            "roughness": 0.6,
            "metallic": 0.0,
            "reflectance": 0.4,
            "texture_path": str(albedo),
        },
    }
    mock_visualizer.target_entries = [entry]
    props = service.get_effective_entry_properties(entry)

    assert props["texture_path"] == str(albedo)
    assert "color_multiplier" not in props
    assert entry["color"] == [0.2, 0.3, 0.4]
    assert renderer.object_material_updates == []
    assert renderer.material_updates == []


def test_target_material_resolution_does_not_touch_renderer_or_target_owner(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Target material refresh never bypasses its semantic object owner."""
    renderer = MockPBRRenderer()
    renderer.set_material = Mock(return_value=False)
    renderer.ensure_object = Mock(return_value=True)
    renderer.set_named_material = Mock(return_value=True)
    renderer.ensure_named_geometry = Mock(return_value=True)
    mock_visualizer.renderer = renderer
    state = RenderObjectState(
        id="target:target_1::mesh",
        payload=MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    entry = {
        "name": "target_1",
        "target_name": "target_1",
        "entry_type": "target",
        "mesh": state,
        "visible": True,
        "_frame_visible": False,
        "material_type": "concrete",
        "color": [0.2, 0.3, 0.4],
    }
    mock_visualizer.target_entries = [entry]

    props = service.get_effective_entry_properties(entry)

    assert props["color"] == [0.2, 0.3, 0.4]
    mock_visualizer.target_service.refresh_target_entry_material.assert_not_called()
    assert state.visible is True
    renderer.set_material.assert_not_called()
    renderer.ensure_object.assert_not_called()
    renderer.set_named_material.assert_not_called()
    renderer.ensure_named_geometry.assert_not_called()


# =============================================================================
# Reset Tests
# =============================================================================


def test_reset_material_removes_override(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Verify reset_material removes the override."""
    # Add a mesh entry so _update_material_rendering returns True
    mock_mesh = _scene_state("mesh_0")
    mock_visualizer.mesh_entries = [
        {"material_type": "concrete", "mesh": mock_mesh, "pbr_properties": {}}
    ]
    service.overrides["concrete"] = {"roughness": 0.9}

    result = service.reset_material("concrete")

    assert result is True
    assert "concrete" not in service.overrides
    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        mock_visualizer.mesh_entries
    )


def test_reset_material_removes_override_no_meshes(service: MaterialPBRService) -> None:
    """Verify reset_material removes override even when no meshes exist."""
    service.overrides["concrete"] = {"roughness": 0.9}

    # Returns False because no meshes were updated, but override is still removed
    service.reset_material("concrete")

    # The override should be removed regardless of return value
    assert "concrete" not in service.overrides


def test_reset_material_nonexistent_returns_false(service: MaterialPBRService) -> None:
    """Verify reset_material returns False for non-existent material."""
    result = service.reset_material("nonexistent")

    assert result is False


def test_reset_material_restores_underlying_profile_instead_of_suppressing_it(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    entry = {
        "name": "Profiled wall",
        "material_type": "brick",
        "mesh": _scene_state("profiled-wall"),
    }
    service.set_visual_binding(
        entry,
        VisualMaterialBinding(
            source=VisualMaterialSource.MANUAL,
            material_type="concrete",
            overrides={"roughness": 0.9},
        ),
    )
    mock_visualizer.mesh_entries = [entry]
    service.overrides["concrete"] = {"roughness": 0.9}
    profile_binding = VisualMaterialBinding(
        source=VisualMaterialSource.PROFILE,
        material_type="concrete",
        preset="Gold",
    )
    mock_visualizer.visual_profile_service = SimpleNamespace(
        resolve_binding=Mock(return_value=profile_binding)
    )

    assert service.reset_material("concrete") is True

    assert service.get_visual_binding(entry) == profile_binding
    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        [entry]
    )


def test_clear_visual_assignment_clears_manual_and_profile_without_group_override(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    """Explicit assignments return to stored Follow EM bindings as one group."""
    manual_entry = {
        "name": "Manual wall",
        "material_type": "glass",
        "mesh": _scene_state("manual-wall"),
    }
    profile_entry = {
        "name": "Profile wall",
        "material_type": "brick",
        "mesh": _scene_state("profile-wall"),
    }
    service.set_visual_binding(
        manual_entry,
        VisualMaterialBinding(
            source=VisualMaterialSource.MANUAL,
            material_type="concrete",
            overrides={"roughness": 0.9},
        ),
    )
    service.set_visual_binding(
        profile_entry,
        VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            preset="Gold",
        ),
    )
    mock_visualizer.mesh_entries = [manual_entry, profile_entry]
    profile_resolver = Mock(
        return_value=VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            preset="Gold",
        )
    )
    mock_visualizer.visual_profile_service = SimpleNamespace(resolve_binding=profile_resolver)

    assert service.overrides == {}
    assert service.clear_visual_assignment("concrete") is True

    assert service.get_visual_binding(manual_entry).source is VisualMaterialSource.FOLLOW_EM
    assert service.get_visual_binding(profile_entry).source is VisualMaterialSource.FOLLOW_EM
    assert "_visual_material_binding" in manual_entry
    assert "_visual_material_binding" in profile_entry
    profile_resolver.assert_not_called()
    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        [manual_entry, profile_entry]
    )


def test_clear_visual_assignment_removes_group_override_without_entries(
    service: MaterialPBRService,
) -> None:
    """Detached group overrides remain clearable without renderer entries."""
    service.overrides["concrete"] = {"roughness": 0.9}

    assert service.clear_visual_assignment("concrete") is True
    assert "concrete" not in service.overrides


def test_reset_all_clears_overrides(service: MaterialPBRService) -> None:
    """Verify reset_all clears all overrides."""
    service.overrides["concrete"] = {"roughness": 0.9}
    service.overrides["glass"] = {"alpha": 0.2}

    result = service.reset_all()

    assert result is True
    assert len(service.overrides) == 0


def test_reset_all_refreshes_only_active_overrides_without_full_scene_render(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    """Reset republishes affected semantic owners instead of rebuilding the scene."""
    mock_visualizer.mesh_entries = [
        {
            "entry_type": "mesh",
            "material_type": "concrete",
            "mesh": _scene_state("wall"),
            "pbr_properties": {"color": [0.5, 0.5, 0.5]},
        }
    ]
    service.overrides = {
        "concrete": {"roughness": 0.9},
        "unused": {"roughness": 0.1},
    }

    assert service.reset_all()

    assert service.overrides == {}
    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        mock_visualizer.mesh_entries
    )
    mock_visualizer.scene_service.render_scene.assert_not_called()


def test_reset_all_returns_false_when_empty(service: MaterialPBRService) -> None:
    """Verify reset_all returns False when no overrides exist."""
    result = service.reset_all()

    assert result is False


# =============================================================================
# Apply Preset Tests
# =============================================================================


def test_apply_preset_to_material(service: MaterialPBRService) -> None:
    """Verify apply_preset applies preset to specific material."""
    result = service.apply_preset("Matte", material_type="concrete")

    assert result is True
    assert "concrete" in service.overrides
    assert service.overrides["concrete"]["roughness"] == 0.9


def test_apply_texture_backed_preset_carries_texture_pack(service: MaterialPBRService) -> None:
    """Building visual presets with PBR packs should load their texture maps."""
    from visualizer.src.materials.catalog import ITU_TO_PBR

    result = service.apply_preset("Brick", material_type="concrete")

    assert result is True
    overrides = service.overrides["concrete"]
    assert overrides["texture_path"] == ITU_TO_PBR["brick"]["texture_path"]
    assert overrides["normal_map_path"] == ITU_TO_PBR["brick"]["normal_map_path"]
    assert overrides["roughness_map_path"] == ITU_TO_PBR["brick"]["roughness_map_path"]
    assert overrides["ao_map_path"] == ITU_TO_PBR["brick"]["ao_map_path"]


def test_apply_nist_ctl_floor_preset_carries_texture_pack(
    service: MaterialPBRService,
) -> None:
    """The presentation floor preset should behave like other PBR packs."""
    from visualizer.src.materials.catalog import ITU_TO_PBR

    result = service.apply_preset("NIST CTL Floor", material_type="concrete")

    assert result is True
    overrides = service.overrides["concrete"]
    pack = ITU_TO_PBR["ground_nist_ctl_floor"]
    assert overrides["texture_path"] == pack["texture_path"]
    assert overrides["normal_map_path"] == pack["normal_map_path"]
    assert overrides["metallic_map_path"] == pack["metallic_map_path"]
    assert overrides["clearcoat"] == pytest.approx(0.12)


def test_apply_brick_after_nist_ctl_floor_resets_advanced_fields(
    service: MaterialPBRService,
) -> None:
    """Switching texture-backed presets should not inherit NIST floor styling."""
    from visualizer.src.materials.catalog import ITU_TO_PBR

    assert service.apply_preset("NIST CTL Floor", material_type="concrete")
    assert service.apply_preset("Brick", material_type="concrete")

    overrides = service.overrides["concrete"]
    assert overrides["texture_path"] == ITU_TO_PBR["brick"]["texture_path"]
    assert overrides["normal_map_path"] == ITU_TO_PBR["brick"]["normal_map_path"]
    assert overrides["metallic_map_path"] is None
    assert overrides["clearcoat"] == pytest.approx(0.0)
    assert overrides["clearcoat_roughness"] == pytest.approx(0.0)
    assert overrides["anisotropy"] == pytest.approx(0.0)
    assert overrides["color"] == MaterialPBRService.PRESETS["Brick"]["color"]


def test_apply_solid_preset_clears_existing_textures(service: MaterialPBRService) -> None:
    """Solid visual presets should not keep a previous texture pack."""
    service.overrides["concrete"] = {
        "texture_path": "/tmp/old_albedo.png",
        "normal_map_path": "/tmp/old_normal.png",
    }

    result = service.apply_preset("Gold", material_type="concrete")

    assert result is True
    overrides = service.overrides["concrete"]
    assert overrides["texture_path"] is None
    assert overrides["normal_map_path"] is None


def test_apply_preset_nonexistent_returns_false(service: MaterialPBRService) -> None:
    """Verify apply_preset returns False for non-existent preset."""
    result = service.apply_preset("NonExistent", material_type="concrete")

    assert result is False


def test_get_preset_properties(service: MaterialPBRService) -> None:
    """Verify get_preset_properties returns correct properties."""
    props = service.get_preset_properties("Matte")

    assert props is not None
    assert "roughness" in props
    assert "description" not in props  # Should exclude description


def test_get_preset_properties_nonexistent(service: MaterialPBRService) -> None:
    """Verify get_preset_properties returns None for non-existent preset."""
    props = service.get_preset_properties("NonExistent")

    assert props is None


# =============================================================================
# User Preset Save/Load Tests
# =============================================================================


def test_save_preset_creates_file(service: MaterialPBRService) -> None:
    """Verify save_preset creates a JSON file."""
    service.overrides["concrete"] = {"roughness": 0.8}

    result = service.save_preset("My Material Preset")

    assert result is True
    preset_file = service.preset_dir / "My_Material_Preset.json"
    assert preset_file.exists()


def test_save_preset_with_empty_overrides_fails(service: MaterialPBRService) -> None:
    """Verify save_preset fails when no overrides exist."""
    result = service.save_preset("Empty Preset")

    assert result is False


def test_save_preset_sanitizes_filename(service: MaterialPBRService) -> None:
    """Verify save_preset sanitizes special characters."""
    service.overrides["concrete"] = {"roughness": 0.8}

    result = service.save_preset("Test/Preset:Name")

    assert result is True
    # Should create file with sanitized name
    preset_file = service.preset_dir / "Test_Preset_Name.json"
    assert preset_file.exists()


def test_load_preset_restores_overrides(service: MaterialPBRService) -> None:
    """Verify load_preset restores saved overrides."""
    # Save a preset first
    service.overrides["glass"] = {"alpha": 0.3}
    service.save_preset("Glass Preset")

    # Clear and reload
    service.overrides.clear()
    result = service.load_preset("Glass Preset")

    assert result is True
    assert "glass" in service.overrides
    assert service.overrides["glass"]["alpha"] == 0.3


def test_load_preset_replaces_stale_manual_groups_and_properties(
    service: MaterialPBRService,
    mock_visualizer: MockVisualizer,
) -> None:
    concrete = {
        "name": "Concrete wall",
        "material_type": "concrete",
        "mesh": _scene_state("concrete-wall"),
        "pbr_properties": {"color": [0.1, 0.2, 0.3], "roughness": 0.4},
    }
    glass = {
        "name": "Glass wall",
        "material_type": "glass",
        "mesh": _scene_state("glass-wall"),
        "pbr_properties": {"alpha": 0.5},
    }
    service.set_visual_binding(
        concrete,
        VisualMaterialBinding(
            source=VisualMaterialSource.MANUAL,
            material_type="concrete",
            overrides={"color": [0.9, 0.8, 0.7], "clearcoat": 0.8},
        ),
    )
    service.set_visual_binding(
        glass,
        VisualMaterialBinding(
            source=VisualMaterialSource.MANUAL,
            material_type="glass",
            overrides={"alpha": 0.1},
        ),
    )
    mock_visualizer.mesh_entries = [concrete, glass]
    service.overrides = {
        "concrete": {"color": [0.9, 0.8, 0.7], "clearcoat": 0.8},
        "glass": {"alpha": 0.1},
    }
    preset_path = service.preset_dir / "Replacement.json"
    preset_path.write_text(
        json.dumps(
            {
                "name": "Replacement",
                "overrides": {"concrete": {"roughness": 0.2}},
            }
        )
    )

    assert service.load_preset("Replacement") is True

    concrete_binding = service.get_visual_binding(concrete)
    assert concrete_binding.source is VisualMaterialSource.MANUAL
    assert concrete_binding.overrides["roughness"] == pytest.approx(0.2)
    assert concrete_binding.overrides["color"] == [0.1, 0.2, 0.3]
    assert concrete_binding.overrides.get("clearcoat", 0.0) == pytest.approx(0.0)
    assert service.get_visual_binding(glass).source is VisualMaterialSource.FOLLOW_EM
    assert set(service.overrides) == {"concrete"}
    mock_visualizer.object_appearance_service.refresh_entry_appearance_batch.assert_called_once_with(
        [concrete, glass]
    )


def test_load_preset_nonexistent_returns_false(service: MaterialPBRService) -> None:
    """Verify load_preset returns False for non-existent preset."""
    result = service.load_preset("NonExistent Preset")

    assert result is False


def test_list_user_presets(service: MaterialPBRService) -> None:
    """Verify list_user_presets returns saved preset names."""
    # Save some presets
    service.overrides["concrete"] = {"roughness": 0.8}
    service.save_preset("Preset A")

    service.overrides["glass"] = {"alpha": 0.2}
    service.save_preset("Preset B")

    presets = service.list_user_presets()

    assert "Preset A" in presets
    assert "Preset B" in presets


def test_list_user_presets_empty_directory(service: MaterialPBRService) -> None:
    """Verify list_user_presets returns empty list when no presets."""
    presets = service.list_user_presets()

    assert presets == []


# =============================================================================
# Material Type Discovery Tests
# =============================================================================


def test_get_material_types_in_scene(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Verify get_material_types_in_scene returns unique material types."""
    mock_visualizer.mesh_entries = [
        {"material_type": "concrete"},
        {"material_type": "glass"},
        {"material_type": "concrete"},  # Duplicate
    ]

    result = service.get_material_types_in_scene()

    assert sorted(result) == ["concrete", "glass"]


def test_get_material_types_in_scene_empty(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Verify get_material_types_in_scene returns empty list when no meshes."""
    mock_visualizer.mesh_entries = []

    result = service.get_material_types_in_scene()

    assert result == []


def test_get_material_types_in_scene_skips_none(
    service: MaterialPBRService, mock_visualizer: MockVisualizer
) -> None:
    """Verify get_material_types_in_scene skips entries with None material."""
    mock_visualizer.mesh_entries = [
        {"material_type": "concrete"},
        {"material_type": None},
        {},
    ]

    result = service.get_material_types_in_scene()

    assert result == ["concrete"]


# =============================================================================
# Cleanup Tests
# =============================================================================


def test_cleanup_clears_overrides(service: MaterialPBRService) -> None:
    """Verify cleanup clears all overrides."""
    service.overrides["concrete"] = {"roughness": 0.8}

    service.cleanup()

    assert len(service.overrides) == 0


# =============================================================================
# Advanced PBR Tests (Tier 0 item 2 — Wave A + Wave B)
# =============================================================================


def test_set_clearcoat_creates_override(service: MaterialPBRService) -> None:
    """Wave A: clearcoat round-trips through set/get."""
    service.set_property("marble", "clearcoat", 0.6)
    assert service.overrides["marble"]["clearcoat"] == 0.6


def test_set_anisotropy_creates_override(service: MaterialPBRService) -> None:
    """Wave A: anisotropy round-trips through set/get."""
    service.set_property("metal", "anisotropy", 0.4)
    assert service.overrides["metal"]["anisotropy"] == 0.4


def test_set_emissive_color_creates_override(service: MaterialPBRService) -> None:
    """Wave A: emissive_color is stored as a list (tuple-like behavior)."""
    service.set_property("default", "emissive_color", [1.0, 0.5, 0.0])
    assert service.overrides["default"]["emissive_color"] == [1.0, 0.5, 0.0]


def test_set_emissive_color_invalid_value(service: MaterialPBRService) -> None:
    """emissive_color rejects non-tuple values via the same path as color."""
    result = service.set_property("default", "emissive_color", "red")
    assert result is False
    assert "emissive_color" not in service.overrides.get("default", {})


def test_set_transmission_creates_override(service: MaterialPBRService) -> None:
    """Wave B: transmission is accepted by the service even though pygfx ignores it."""
    service.set_property("glass", "transmission", 0.9)
    assert service.overrides["glass"]["transmission"] == 0.9


def test_set_glass_thickness_creates_override(service: MaterialPBRService) -> None:
    """Wave B: glass_thickness is accepted by the service."""
    service.set_property("glass", "glass_thickness", 0.5)
    assert service.overrides["glass"]["glass_thickness"] == 0.5
