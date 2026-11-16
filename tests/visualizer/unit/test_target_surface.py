"""Animated visual skins preserve geometry and obey the existing texture policy."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.scenarios.paths import PathPolicy
from visualizer.src.materials import texture_policy
from visualizer.src.scene.target_surface import load_target_surface, target_surface_for_profile
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.services.target_asset_cache import TargetAssetSource
from visualizer.src.services.target_service import TargetService
from visualizer.src.services.visual_profile_service import VisualProfileService
from visualizer.src.types.render_payloads import MeshPayload


@pytest.fixture
def surface_file(tmp_path):
    (tmp_path / "albedo.png").write_bytes(b"test texture")
    np.savez(
        tmp_path / "uv.npz",
        vertex_count=np.array(3),
        triangles=np.array([[0, 1, 2]]),
        triangle_uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
    )
    path = tmp_path / "surface.json"
    path.write_text(
        json.dumps(
            {
                "schema": "orchav.target_surface.v1",
                "texture": "albedo.png",
                "uv_map": "uv.npz",
            }
        )
    )
    return path


def test_uvs_remain_attached_across_poses_without_mutating_geometry(surface_file):
    surface = load_target_surface(surface_file)
    uv_storage = None
    for step in range(3):
        payload = MeshPayload(
            vertices=np.eye(3) * (step + 1),
            triangles=np.array([[0, 1, 2]]),
            vertex_colors=np.ones((3, 3)),
            cache_key=f"pose:{step}",
        )
        skinned = surface.apply(payload)
        np.testing.assert_array_equal(skinned.vertices, payload.vertices)
        np.testing.assert_array_equal(skinned.triangles, payload.triangles)
        assert skinned.vertex_colors is None
        assert payload.vertex_colors is not None
        assert payload.triangle_uvs is None
        assert skinned.cache_key != payload.cache_key
        assert not skinned.triangle_uvs.flags.writeable
        if uv_storage is not None:
            assert np.shares_memory(uv_storage, skinned.triangle_uvs)
        uv_storage = skinned.triangle_uvs


@pytest.mark.parametrize("vertices,faces", [(4, [[0, 1, 2]]), (3, [[2, 1, 0]])])
def test_rejects_incompatible_pose_topology(surface_file, vertices, faces):
    surface = load_target_surface(surface_file)
    with pytest.raises(ValueError, match="topology"):
        surface.apply(MeshPayload(vertices=np.zeros((vertices, 3)), triangles=np.array(faces)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("vertex_count", np.array(3.5)),
        ("triangles", np.array([[0, 1, 3]])),
        ("triangles", np.array([[0.0, 1.0, 2.0]])),
        ("triangle_uvs", np.zeros((2, 2))),
        ("triangle_uvs", np.full((3, 2), np.nan)),
    ],
)
def test_rejects_malformed_uv_maps(surface_file, field, value):
    uv_path = surface_file.parent / "uv.npz"
    with np.load(uv_path) as source:
        data = dict(source)
    data[field] = value
    np.savez(uv_path, **data)
    with pytest.raises(ValueError):
        load_target_surface(surface_file)


@pytest.mark.parametrize(
    "material",
    [
        {"roughness": True},
        {"metallic": 1.1},
        {"clearcoat": float("nan")},
        {"clearcoat_roughness_map_path": "/absolute.png"},
        {"roughness_map_path": "."},
        {"displacement": 0.2},
    ],
)
def test_rejects_invalid_authored_surface_material(surface_file, material):
    config = json.loads(surface_file.read_text())
    config["material"] = material
    surface_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_target_surface(surface_file)


def test_authored_finish_obeys_map_toggle_and_explicit_object_override(surface_file, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: surface_file.parent)
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", None)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    config = json.loads(surface_file.read_text())
    config["material"] = {
        "roughness": 1.0,
        "metallic": 1.0,
        "clearcoat": 0.7,
        "clearcoat_roughness": 1.0,
        "clearcoat_roughness_map_path": "albedo.png",
    }
    surface_file.write_text(json.dumps(config))
    surface = load_target_surface(surface_file)
    assert dict(surface.material)["clearcoat_roughness_map_path"] == str(
        surface_file.parent / "albedo.png"
    )
    viz = SimpleNamespace(mesh_entries=[], target_entries=[])
    viz.visual_profile_service = VisualProfileService(viz)
    viz.visual_profile_service.load_scenario_rules([{"match": {"name": "Car"}, "preset": "Satin"}])
    service = MaterialPBRService(viz)
    entry = {
        "name": "Car",
        "entry_type": "target",
        "material_type": "metal",
        "_target_surface": surface,
    }
    for enabled in [True, False, True]:
        texture_policy.set_texture_maps_override(enabled)
        payload = service.resolve_entry_material(entry).payload
        assert payload.roughness == (1 if enabled else service.PRESETS["Satin"]["roughness"])
        assert payload.metallic == (1 if enabled else service.PRESETS["Satin"]["metallic"])
        assert payload.clearcoat_roughness_map_path == (surface.texture_path if enabled else None)
    binding = service.get_visual_binding(entry)
    service.set_visual_binding(entry, replace(binding, overrides={"roughness": 0.41}))
    assert service.resolve_entry_material(entry).payload.roughness == 0.41
    assert entry["material_type"] == "metal"


def test_profile_resolves_sidecar_from_scenario_not_cwd(surface_file, monkeypatch):
    viz = SimpleNamespace(current_scenario_policy=PathPolicy(surface_file.parent / "scenario.yaml"))
    viz.visual_profile_service = VisualProfileService(viz)
    viz.visual_profile_service.load_scenario_rules(
        [
            {
                "match": {"name": "Pedestrian", "type": "target"},
                "preset": "Skin",
                "surface": "surface.json",
            }
        ]
    )
    monkeypatch.chdir(surface_file.parent.parent)
    surface = target_surface_for_profile(viz, "Pedestrian", "metal")
    assert surface.texture_path == str(surface_file.parent / "albedo.png")
    assert target_surface_for_profile(viz, "Drone", "metal") is None
    # A higher-priority profile without a skin must not inherit a lower rule's skin.
    viz.visual_profile_service.load_scenario_rules(
        [
            {"match": {"name": "Pedestrian"}, "preset": "Matte"},
            *viz.visual_profile_service.get_active_rules(),
        ]
    )
    assert target_surface_for_profile(viz, "Pedestrian", "metal") is None


def test_target_builder_applies_shared_surface_on_each_loaded_pose(surface_file):
    surface = load_target_surface(surface_file)
    service = TargetService(SimpleNamespace())
    spec = service._target_asset_build_spec(
        scale=1.0,
        orientation=(0, 0, 0),
        use_ply_position=False,
        pbr_props={"color": [1, 1, 1]},
        surface=surface,
    )
    for step in range(2):
        mesh = surface_file.parent / f"pose{step}.ply"
        mesh.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
            "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
            f"end_header\n{step} 0 0\n0 1 0\n0 0 1\n3 0 1 2\n"
        )
        source = TargetAssetSource.from_path("Pedestrian", mesh).resolve()
        asset = service._build_target_asset(source, spec)
        assert not asset.has_vertex_texture
        np.testing.assert_array_equal(asset.mesh.payload.triangle_uvs, surface.triangle_uvs)
        np.testing.assert_array_equal(asset.mesh.payload.vertices, asset.original_vertices)
        unskinned = service._build_target_asset(source, replace(spec, surface=None))
        assert unskinned.mesh.payload.triangle_uvs is None
        np.testing.assert_array_equal(asset.mesh.payload.vertices, unskinned.mesh.payload.vertices)


def test_skin_obeys_texture_toggle_and_retains_scientific_material(surface_file, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: surface_file.parent)
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", None)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    viz = SimpleNamespace(mesh_entries=[], target_entries=[])
    viz.visual_profile_service = VisualProfileService(viz)
    service = MaterialPBRService(viz)
    surface = load_target_surface(surface_file)
    entry = {
        "name": "Pedestrian",
        "target_name": "Pedestrian",
        "entry_type": "target",
        "material_type": "metal",
        "_target_surface": surface,
    }
    for enabled in [True, False, True]:
        texture_policy.set_texture_maps_override(enabled)
        resolved = service.resolve_entry_material(entry)
        assert resolved.payload.texture_path == (surface.texture_path if enabled else None)
        assert entry["material_type"] == "metal"
        assert entry["_target_surface"] is surface
        if enabled:
            assert resolved.payload.base_color[:3] == (1, 1, 1)
