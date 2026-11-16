"""PBR contracts shared by the live and static pygfx adapters."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.backends.pygfx_pbr import (
    apply_display_reflectance,
    needs_physical_material,
    pbr_material_classes,
)
from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material
from visualizer.src.types.render_payloads import MaterialPayload, SurfaceColorSource


def _gfx():
    return pytest.importorskip("pygfx")


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("shader", ["lit", "unlit"])
def test_static_payload_alpha_is_applied_once(alpha, vertex, shader):
    gfx = _gfx()
    native = make_mesh_material(
        gfx,
        MaterialPayload(base_color=(0.2, 0.4, 0.6, alpha), shader=shader),
        has_vertex_colors=vertex,
    )
    assert native.color.a == 1.0
    assert native.opacity == pytest.approx(alpha)
    if vertex:
        assert native.color_mode == "vertex"
    if shader == "unlit":
        assert type(native) is gfx.MeshBasicMaterial


@pytest.mark.parametrize("reflectance,ior", [(0.0, 1.0), (0.25, 11 / 9), (0.5, 1.5), (1.0, 7 / 3)])
def test_reflectance_has_bounded_filament_f0(reflectance, ior):
    gfx = _gfx()
    _, physical_cls, _ = pbr_material_classes(gfx)
    native = physical_cls()
    apply_display_reflectance(native, reflectance)
    assert native.ior == pytest.approx(ior)
    assert np.isfinite(native.ior)
    assert ((native.ior - 1) / (native.ior + 1)) ** 2 == pytest.approx(
        0.16 * reflectance**2, abs=1e-7
    )
    assert native.specular_intensity == 1.0


@pytest.mark.parametrize("reflectance", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("metalness", [0.0, 1.0])
def test_static_reflectance_survives_clearcoat_promotion(reflectance, metalness):
    gfx = _gfx()
    standard, physical, _ = pbr_material_classes(gfx)
    base = MaterialPayload(reflectance=reflectance, metallic=metalness)
    plain = make_mesh_material(gfx, base)
    coated = make_mesh_material(gfx, replace(base, clearcoat=0.3))
    assert isinstance(coated, physical)
    assert isinstance(plain, physical if reflectance != 0.5 else standard)
    plain_f0 = ((plain.ior - 1) / (plain.ior + 1)) ** 2 if hasattr(plain, "ior") else 0.04
    coated_f0 = ((coated.ior - 1) / (coated.ior + 1)) ** 2
    assert coated_f0 == pytest.approx(plain_f0)
    assert plain.metalness == coated.metalness == metalness
    assert needs_physical_material(base) is (reflectance != 0.5)


def _live_harness(vertex=False, ibl=False):
    from visualizer.src.renderers.pygfx.materials import PygfxMaterialMixin

    gfx = _gfx()

    class Harness(PygfxMaterialMixin):
        def _record_frame_update_metric(self, *_args, **_kwargs):
            pass

        def request_redraw(self):
            pass

    h = Harness()
    h._gfx = gfx
    h._unlit_mode_enabled = False
    h._ibl_loaded = ibl
    h._ibl_manager = SimpleNamespace(apply_to_material=lambda material: None)
    h._wireframe_enabled = False
    h._clipping_planes = ()
    h._materials = {}
    h._material_apply_signatures = {}
    h._texture_cache = {}
    h._texture_source_identities = {}
    geometry = gfx.plane_geometry()
    if vertex:
        geometry.colors = gfx.Buffer(np.tile(np.array([0.2, 0.4, 0.6, 0.75], np.float32), (4, 1)))
    h._objects = {"mesh": gfx.Mesh(geometry, h._build_mesh_material(vertex))}
    h._kinds = {"mesh": "mesh"}
    h._geometry_color_sources = {
        "mesh": SurfaceColorSource.VERTEX if vertex else SurfaceColorSource.MATERIAL
    }
    h._geometry_texcoords_available = {"mesh": True}
    return h


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("ibl", [False, True])
def test_live_alpha_unlit_and_promotion_roundtrip(vertex, ibl):
    gfx = _gfx()
    h = _live_harness(vertex, ibl)
    standard, physical, _ = pbr_material_classes(gfx)
    base = MaterialPayload(base_color=(0.2, 0.4, 0.6, 0.5), metallic=1.0)
    for requested in (
        base,
        replace(base, clearcoat=0.3),
        base,
        replace(base, reflectance=1.0),
        base,
    ):
        assert h.set_named_material("mesh", requested)
        native = h._objects["mesh"].material
        assert native.color.a == 1.0
        assert native.opacity == 0.5
        assert native.metalness == pytest.approx(1.0 if ibl else 0.9)
        assert isinstance(native, physical if needs_physical_material(requested) else standard)
        if hasattr(native, "ior"):
            assert ((native.ior - 1) / (native.ior + 1)) ** 2 == pytest.approx(
                0.16 * requested.reflectance**2
            )
            assert native.specular_intensity == 1.0
    assert h.set_named_material(
        "mesh", replace(base, shader="unlit", clearcoat=0.8, reflectance=1.0)
    )
    assert type(h._objects["mesh"].material) is gfx.MeshBasicMaterial
    if vertex:
        np.testing.assert_array_equal(
            h._objects["mesh"].geometry.colors.data[0], np.array([0.2, 0.4, 0.6, 0.75], np.float32)
        )


def test_hdr_cache_is_versioned_linear_and_float32(tmp_path):
    from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

    manager = PygfxIBLManager(_gfx(), tmp_path)
    manager._face_size = 4
    data = np.ones((6, 4, 4, 4), np.float32)
    data[..., :3] = [0.18, 4.0, 100_000.0]
    manager._write_cubemap_cache("neutral", data)
    cache_path = manager._cache_path("neutral")
    assert "linear_v2" in cache_path.name
    assert np.load(cache_path).dtype == np.float32
    texture = manager.load_ibl("neutral")
    assert texture.colorspace == "physical"
    assert texture.generate_mipmaps
    np.testing.assert_array_equal(texture.data, data)


def test_old_environment_cache_is_not_reused(tmp_path):
    from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

    manager = PygfxIBLManager(_gfx(), tmp_path)
    manager._face_size = 4
    old = tmp_path / ".pygfx_cache" / "neutral_face4_rgba.npy"
    old.parent.mkdir()
    np.save(old, np.ones((6, 4, 4, 4), np.float16))
    assert manager._load_cached_cubemap("neutral") is None


def test_visual_xml_clearcoat_fields_survive_catalog_defaults():
    from xml.etree import ElementTree as ET

    from visualizer.src.scene.io import MaterialHandler

    root = ET.fromstring(
        '<scene><bsdf type="diffuse" id="mat-concrete">'
        '<float name="clearcoat" value="0.4"/>'
        '<float name="clearcoat_roughness" value="0.15"/>'
        "</bsdf></scene>"
    )
    parsed = MaterialHandler.parse_materials(root)["mat-concrete"]["pbr_properties"]
    assert parsed["clearcoat"] == 0.4
    assert parsed["clearcoat_roughness"] == 0.15
    assert parsed["material_type"] == "concrete"


def test_clearcoat_roughness_map_survives_both_adapters_and_clears_on_toggle(tmp_path, monkeypatch):
    from PIL import Image

    from visualizer.src.materials import texture_policy
    from visualizer.src.materials.catalog import material_payload_to_pbr_kwargs
    from visualizer.src.services.material_pbr_service import MaterialPBRService
    from visualizer.src.types.render_payloads import material_payload_from_mapping

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", None)
    path = tmp_path / "coat.png"
    Image.fromarray(np.full((4, 4, 3), 85, np.uint8)).save(path)
    payload = material_payload_from_mapping(
        {
            "clearcoat": 0.7,
            "clearcoat_roughness": 1,
            "clearcoat_roughness_map_path": str(path),
        }
    )
    assert material_payload_to_pbr_kwargs(payload)["clearcoat_roughness_map_path"] == str(path)
    assert MaterialPBRService._validate_property("clearcoat_roughness_map_path", None) is None
    assert MaterialPBRService._validate_property("clearcoat_roughness_map_path", path) == str(path)
    static = make_mesh_material(_gfx(), payload)
    assert static.clearcoat_roughness_map is not None
    live = _live_harness(ibl=True)
    assert live.set_named_material("mesh", payload)
    assert live._objects["mesh"].material.clearcoat_roughness_map is not None
    texture_policy.set_texture_maps_override(False)
    assert live.set_named_material("mesh", payload)
    assert live._objects["mesh"].material.clearcoat_roughness_map is None
    texture_policy.set_texture_maps_override(True)
    assert live.set_named_material("mesh", payload)
    assert live._objects["mesh"].material.clearcoat_roughness_map is not None
    live._geometry_texcoords_available["mesh"] = False
    assert live.set_named_material("mesh", payload)
    assert live._objects["mesh"].material.clearcoat_roughness_map is None
