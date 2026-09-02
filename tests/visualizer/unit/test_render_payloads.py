"""Tests for the backend-neutral render payloads, focused on the
advanced PBR fields landed in Tier 0 item 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    SurfaceColorSource,
)

# =============================================================================
# MaterialPayload defaults
# =============================================================================


def test_default_payload_has_no_advanced_pbr() -> None:
    p = MaterialPayload()
    assert p.clearcoat == 0.0
    assert p.clearcoat_roughness == 0.0
    assert p.anisotropy == 0.0
    assert p.emissive_color == (0.0, 0.0, 0.0)
    assert p.emissive_intensity == 0.0
    assert p.transmission == 0.0
    assert p.glass_thickness == 0.0
    assert p.has_advanced_pbr is False


def test_clearcoat_triggers_advanced_pbr() -> None:
    p = MaterialPayload(clearcoat=0.6)
    assert p.has_advanced_pbr is True


def test_clearcoat_roughness_alone_triggers_advanced_pbr() -> None:
    p = MaterialPayload(clearcoat_roughness=0.2)
    assert p.has_advanced_pbr is True


def test_anisotropy_does_not_trigger_pygfx_advanced_pbr() -> None:
    """Anisotropy is Open3D-only on this release because pygfx 0.15.3
    has an upstream WGSL shader bug. ``has_advanced_pbr`` is the
    pygfx-side selector, so anisotropy alone must NOT switch pygfx to
    MeshPhysicalMaterial. The Open3D renderer applies it independently
    via ``base_anisotropy``.
    """
    p = MaterialPayload(anisotropy=0.4)
    assert p.has_advanced_pbr is False


def test_emissive_intensity_triggers_advanced_pbr() -> None:
    p = MaterialPayload(emissive_intensity=1.5)
    assert p.has_advanced_pbr is True


def test_emissive_color_alone_triggers_advanced_pbr() -> None:
    p = MaterialPayload(emissive_color=(1.0, 0.5, 0.0))
    assert p.has_advanced_pbr is True


def test_wave_b_only_does_not_trigger_advanced_pbr() -> None:
    """Transmission and glass_thickness are Open3D-only — pygfx must stay
    on the fast MeshStandardMaterial path even when these are non-default.
    """
    p = MaterialPayload(transmission=0.9, glass_thickness=0.5)
    assert p.has_advanced_pbr is False


def test_payload_is_frozen() -> None:
    p = MaterialPayload(clearcoat=0.5)
    with pytest.raises(Exception):
        p.clearcoat = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload,buffers",
    [
        (
            MeshPayload(
                vertices=np.zeros((3, 3), dtype=np.float32),
                triangles=np.array([[0, 1, 2]], dtype=np.int32),
                normals=np.zeros((3, 3), dtype=np.float32),
                vertex_colors=np.ones((3, 3), dtype=np.float32),
                triangle_uvs=np.zeros((3, 2), dtype=np.float32),
            ),
            ("vertices", "triangles", "normals", "vertex_colors", "triangle_uvs"),
        ),
        (
            LineSetPayload(
                points=np.zeros((2, 3), dtype=np.float32),
                lines=np.array([[0, 1]], dtype=np.int32),
                colors=np.ones((1, 3), dtype=np.float32),
            ),
            ("points", "lines", "colors"),
        ),
        (
            PointCloudPayload(
                points=np.zeros((2, 3), dtype=np.float32),
                colors=np.ones((2, 3), dtype=np.float32),
            ),
            ("points", "colors"),
        ),
    ],
)
def test_geometry_payload_buffers_are_read_only(payload, buffers) -> None:
    """Payload dataclass immutability includes its NumPy storage."""
    for name in buffers:
        buffer = getattr(payload, name)
        assert buffer.flags.writeable is False
        with pytest.raises(ValueError):
            buffer.flat[0] = 1


def test_geometry_payload_detaches_without_freezing_caller_owned_buffers() -> None:
    """Payload construction snapshots input without changing caller ownership."""
    vertices = np.zeros((3, 3), dtype=np.float32)

    payload = MeshPayload(
        vertices=vertices,
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )

    assert vertices.flags.writeable is True
    assert payload.vertices.flags.writeable is False
    assert not np.shares_memory(payload.vertices, vertices)
    vertices[0, 0] = 7.0
    assert payload.vertices[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        payload.vertices.setflags(write=True)


def test_geometry_payload_reuses_existing_immutable_snapshot() -> None:
    """Deriving payload metadata does not recopy already-immutable geometry."""
    source = MeshPayload(
        vertices=np.zeros((3, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )

    derived = MeshPayload(vertices=source.vertices, triangles=source.triangles)

    assert derived.vertices is source.vertices
    assert derived.triangles is source.triangles


def test_geometry_payload_detaches_from_readonly_view_of_mutable_storage() -> None:
    """A read-only view must not disguise storage its owner can still mutate."""
    backing = bytearray(np.zeros((2, 3), dtype=np.float32).tobytes())
    readonly_view = memoryview(backing).toreadonly()
    source = np.frombuffer(readonly_view, dtype=np.float32).reshape(2, 3)

    payload = PointCloudPayload(points=source)
    np.frombuffer(backing, dtype=np.float32)[0] = 9.0

    assert payload.points[0, 0] == pytest.approx(0.0)
    assert not np.shares_memory(payload.points, source)
    with pytest.raises(ValueError):
        payload.points.setflags(write=True)


def test_mesh_payload_for_pbr_material_applies_vertex_color_policy() -> None:
    from visualizer.src.types.render_payloads import mesh_payload_for_pbr_material

    source = MeshPayload(
        vertices=np.zeros((3, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        vertex_colors=np.ones((3, 3), dtype=np.float32),
        triangle_uvs=np.zeros((3, 2), dtype=np.float32),
        cache_key="old",
    )

    material_payload = mesh_payload_for_pbr_material(source, cache_key="new")
    preserved_payload = mesh_payload_for_pbr_material(
        source,
        color_source=SurfaceColorSource.VERTEX,
    )

    assert material_payload.vertex_colors is None
    assert material_payload.cache_key == "new"
    np.testing.assert_allclose(preserved_payload.vertex_colors, source.vertex_colors)
    assert preserved_payload.cache_key == "old"


# =============================================================================
# Pygfx material selection rule
# =============================================================================


def _make_renderer():
    """Construct a PygfxRenderer against a stub visualizer without booting Qt."""
    import pygfx as gfx

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _AppState:
        label_screen_space = True

    class _Viz:
        app_state = _AppState()

    r = PygfxRenderer(_Viz())
    r._scene = gfx.Scene()
    r._initialized = True
    return r


def test_make_mesh_material_plain_returns_standard_material() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    plain = MaterialPayload(roughness=0.6, metallic=0.0)
    mat = make_mesh_material(gfx, plain)
    assert isinstance(mat, gfx.MeshStandardMaterial)
    assert not isinstance(mat, gfx.MeshPhysicalMaterial)


def test_make_mesh_material_applies_reflectance() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    material = MaterialPayload(reflectance=0.22)
    mat = make_mesh_material(gfx, material)
    assert mat.reflectivity == pytest.approx(0.22, abs=1e-5)


def test_make_mesh_material_applies_normal_map_strength(monkeypatch, tmp_path) -> None:
    import pygfx as gfx
    from PIL import Image

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    normal_map = tmp_path / "normal.png"
    Image.new("RGBA", (1, 1), (128, 128, 255, 255)).save(normal_map)

    material = MaterialPayload(normal_map_path=str(normal_map), normal_map_strength=2.4)
    mat = make_mesh_material(gfx, material)
    assert mat.normal_map is not None
    assert mat.normal_scale == pytest.approx((2.4, 2.4), abs=1e-5)


def test_make_mesh_material_applies_transparent_alpha_state() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    material = MaterialPayload(base_color=(0.2, 0.7, 0.8, 0.35), shader="litTransparency")
    mat = make_mesh_material(gfx, material)
    assert mat.opacity == pytest.approx(0.35, abs=1e-5)
    assert mat.alpha_mode == "weighted_blend"
    assert mat.depth_write is False
    assert mat.depth_test is True


def test_make_mesh_material_advanced_returns_physical_material() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    fancy = MaterialPayload(clearcoat=0.6, clearcoat_roughness=0.1)
    mat = make_mesh_material(gfx, fancy)
    assert isinstance(mat, gfx.MeshPhysicalMaterial)
    # Pygfx's float32 storage rounds slightly, so use a tolerance.
    assert mat.clearcoat == pytest.approx(0.6, abs=1e-5)
    assert mat.clearcoat_roughness == pytest.approx(0.1, abs=1e-5)


def test_make_mesh_material_wave_b_only_stays_on_standard() -> None:
    """Transmission/glass_thickness alone must NOT trigger MeshPhysicalMaterial."""
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    wave_b = MaterialPayload(transmission=0.9, glass_thickness=0.5)
    mat = make_mesh_material(gfx, wave_b)
    assert isinstance(mat, gfx.MeshStandardMaterial)
    assert not isinstance(mat, gfx.MeshPhysicalMaterial)


def test_make_mesh_material_anisotropy_only_stays_on_standard() -> None:
    """Anisotropy alone must NOT trigger MeshPhysicalMaterial on pygfx —
    upstream shader bug in 0.15.3 makes anisotropy unrenderable. The
    Open3D renderer applies it via base_anisotropy independently.
    """
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    aniso = MaterialPayload(anisotropy=0.4)
    mat = make_mesh_material(gfx, aniso)
    assert isinstance(mat, gfx.MeshStandardMaterial)
    assert not isinstance(mat, gfx.MeshPhysicalMaterial)


def test_make_mesh_material_clearcoat_plus_anisotropy_uses_physical_without_aniso() -> None:
    """When clearcoat is set, pygfx swaps to MeshPhysicalMaterial. Anisotropy
    is silently dropped in that swap to avoid the upstream shader bug.
    """
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    combo = MaterialPayload(clearcoat=0.6, anisotropy=0.4)
    mat = make_mesh_material(gfx, combo)
    assert isinstance(mat, gfx.MeshPhysicalMaterial)
    assert mat.clearcoat == pytest.approx(0.6, abs=1e-5)
    # Anisotropy must be at the pygfx default, not 0.4.
    assert mat.anisotropy == 0.0


def test_make_mesh_material_emissive_returns_physical_material() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    glow = MaterialPayload(emissive_color=(1.0, 0.4, 0.0), emissive_intensity=2.0)
    mat = make_mesh_material(gfx, glow)
    assert isinstance(mat, gfx.MeshPhysicalMaterial)
    assert mat.emissive.r == pytest.approx(1.0, abs=1e-5)
    assert mat.emissive.g == pytest.approx(0.4, abs=1e-5)
    assert mat.emissive.b == pytest.approx(0.0, abs=1e-5)
    assert mat.emissive_intensity == pytest.approx(2.0, abs=1e-5)


def test_make_mesh_material_emissive_color_with_zero_intensity_is_off() -> None:
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material

    glow = MaterialPayload(emissive_color=(1.0, 0.4, 0.0), emissive_intensity=0.0)
    mat = make_mesh_material(gfx, glow)
    assert isinstance(mat, gfx.MeshPhysicalMaterial)
    assert mat.emissive_intensity == pytest.approx(0.0, abs=1e-5)


def test_pygfx_coerce_material_roundtrips_advanced_fields() -> None:
    r = _make_renderer()
    payload = r._coerce_material(
        {
            "color": [0.7, 0.6, 0.5],
            "roughness": 0.3,
            "clearcoat": 0.6,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.4,
            "emissive_color": (1.0, 0.5, 0.0),
            "emissive_intensity": 2.0,
            "transmission": 0.9,
            "glass_thickness": 0.5,
        }
    )
    assert payload.clearcoat == 0.6
    assert payload.clearcoat_roughness == 0.1
    assert payload.anisotropy == 0.4
    assert payload.emissive_color == (1.0, 0.5, 0.0)
    assert payload.emissive_intensity == 2.0
    assert payload.transmission == 0.9
    assert payload.glass_thickness == 0.5
    assert payload.has_advanced_pbr is True


def test_pygfx_coerce_material_wave_b_only_does_not_trigger_advanced() -> None:
    r = _make_renderer()
    payload = r._coerce_material({"transmission": 0.9, "glass_thickness": 0.5})
    assert payload.transmission == 0.9
    assert payload.glass_thickness == 0.5
    assert payload.has_advanced_pbr is False


def test_mesh_payload_to_pygfx_buffers_preserves_face_varying_uvs() -> None:
    from visualizer.src.backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers

    mesh = {
        "vertices": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "triangles": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        "normals": np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        "vertex_colors": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "triangle_uvs": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.3],
                [0.8, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    buffers = mesh_payload_to_pygfx_buffers(
        MeshPayload(
            vertices=mesh["vertices"],
            triangles=mesh["triangles"],
            normals=mesh["normals"],
            vertex_colors=mesh["vertex_colors"],
            triangle_uvs=mesh["triangle_uvs"],
        )
    )

    assert buffers["positions"].shape == (6, 3)
    assert buffers["indices"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert buffers["normals"].shape == (6, 3)
    assert buffers["colors"].shape == (6, 3)
    np.testing.assert_allclose(buffers["texcoords"], mesh["triangle_uvs"], atol=1e-6)
    np.testing.assert_allclose(buffers["positions"][0], mesh["vertices"][0], atol=1e-6)
    np.testing.assert_allclose(buffers["positions"][3], mesh["vertices"][0], atol=1e-6)


def test_mesh_payload_to_pygfx_buffers_keeps_indexed_mesh_when_uvs_share_corners() -> None:
    from visualizer.src.backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers

    mesh = {
        "vertices": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "triangles": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        "normals": np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        "vertex_colors": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "triangle_uvs": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    buffers = mesh_payload_to_pygfx_buffers(
        MeshPayload(
            vertices=mesh["vertices"],
            triangles=mesh["triangles"],
            normals=mesh["normals"],
            vertex_colors=mesh["vertex_colors"],
            triangle_uvs=mesh["triangle_uvs"],
        )
    )

    assert buffers["positions"].shape == (4, 3)
    assert buffers["indices"].tolist() == mesh["triangles"].tolist()
    assert buffers["normals"].shape == (4, 3)
    assert buffers["colors"].shape == (4, 3)
    np.testing.assert_allclose(
        buffers["texcoords"],
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_mesh_payload_to_pygfx_buffers_detaches_indexed_arrays_from_source_payload() -> None:
    from visualizer.src.backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    payload = MeshPayload(vertices=vertices, triangles=triangles)

    buffers = mesh_payload_to_pygfx_buffers(payload)
    buffers["positions"][0, 0] = 99.0
    buffers["indices"][0, 0] = 99

    assert payload.vertices[0, 0] == 0.0
    assert payload.triangles[0, 0] == 0


def test_mesh_payload_to_pygfx_buffers_reuses_persisted_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR", str(tmp_path / "mesh_cache"))
    monkeypatch.delenv("ORCHAV_DISABLE_PYGFX_MESH_BUFFER_CACHE", raising=False)
    pygfx_scene_helpers._PRUNED_PYGFX_MESH_CACHE_ROOTS.clear()

    payload = MeshPayload(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        normals=np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        vertex_colors=np.array([[1.0, 1.0, 1.0]] * 4, dtype=np.float32),
        triangle_uvs=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.3],
                [0.8, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        cache_key="city_scene.abc123/sample",
    )

    first = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(payload)
    cache_files = list((tmp_path / "mesh_cache").rglob("*.npz"))
    assert len(cache_files) == 1

    monkeypatch.setattr(
        pygfx_scene_helpers.np,
        "unique",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expected cache hit")),
    )

    second = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(payload)

    assert second["positions"].shape == first["positions"].shape
    np.testing.assert_allclose(second["positions"], first["positions"], atol=1e-6)
    np.testing.assert_allclose(second["indices"], first["indices"], atol=1e-6)
    np.testing.assert_allclose(second["texcoords"], first["texcoords"], atol=1e-6)


def test_mesh_payload_to_pygfx_buffers_cache_hit_returns_detached_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR", str(tmp_path / "mesh_cache"))
    monkeypatch.delenv("ORCHAV_DISABLE_PYGFX_MESH_BUFFER_CACHE", raising=False)
    pygfx_scene_helpers._PRUNED_PYGFX_MESH_CACHE_ROOTS.clear()
    pygfx_scene_helpers._PYGFX_MESH_BUFFER_MEMORY_CACHE.clear()

    payload = MeshPayload(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        triangle_uvs=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        cache_key="human_targets/walker/frame_00006",
    )

    first = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(payload)
    first["positions"][0, 0] = 42.0
    first["indices"][0, 0] = 42

    second = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(payload)

    assert second["positions"][0, 0] == 0.0
    assert second["indices"][0, 0] == 0


def test_mesh_payload_to_pygfx_buffers_memory_cache_separates_vertex_color_modes() -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    pygfx_scene_helpers._PYGFX_MESH_BUFFER_MEMORY_CACHE.clear()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        dtype=np.float32,
    )
    cache_key = "target_runtime/pedestrian/fitted_00001"

    flat_payload = MeshPayload(vertices=vertices, triangles=triangles, cache_key=cache_key)
    textured_payload = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        vertex_colors=colors,
        cache_key=cache_key,
    )

    flat_buffers = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(flat_payload)
    textured_buffers = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(textured_payload)
    flat_again = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(flat_payload)

    assert "colors" not in flat_buffers
    assert "colors" in textured_buffers
    np.testing.assert_allclose(textured_buffers["colors"], colors, atol=1e-6)
    assert "colors" not in flat_again


def test_mesh_payload_to_pygfx_buffers_memory_cache_separates_uv_modes() -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    pygfx_scene_helpers._PYGFX_MESH_BUFFER_MEMORY_CACHE.clear()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    triangle_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache_key = "target_runtime/car/car_painted"

    untextured_payload = MeshPayload(vertices=vertices, triangles=triangles, cache_key=cache_key)
    textured_payload = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        triangle_uvs=triangle_uvs,
        cache_key=cache_key,
    )

    untextured_buffers = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(untextured_payload)
    textured_buffers = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(textured_payload)
    untextured_again = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(untextured_payload)

    assert "texcoords" not in untextured_buffers
    assert "texcoords" in textured_buffers
    np.testing.assert_allclose(textured_buffers["texcoords"], triangle_uvs, atol=1e-6)
    assert "texcoords" not in untextured_again


def test_mesh_payload_to_pygfx_buffers_disable_env_skips_persisted_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR", str(tmp_path / "mesh_cache"))
    monkeypatch.setenv("ORCHAV_DISABLE_PYGFX_MESH_BUFFER_CACHE", "1")
    pygfx_scene_helpers._PRUNED_PYGFX_MESH_CACHE_ROOTS.clear()

    payload = MeshPayload(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        triangle_uvs=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        cache_key="city_scene.abc123/sample",
    )

    buffers = pygfx_scene_helpers.mesh_payload_to_pygfx_buffers(payload)

    assert "positions" in buffers
    assert not (tmp_path / "mesh_cache").exists()


# =============================================================================
# Module-level pbr_props_to_kwargs helper
# =============================================================================


def test_pbr_props_to_kwargs_carries_every_advanced_field() -> None:
    """The shared helper must forward every Wave A + Wave B field."""
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    props = {
        "color": [0.7, 0.6, 0.5],
        "roughness": 0.3,
        "metallic": 0.0,
        "reflectance": 0.6,
        "alpha": 1.0,
        "clearcoat": 0.6,
        "clearcoat_roughness": 0.1,
        "anisotropy": 0.4,
        "emissive_color": (1.0, 0.5, 0.0),
        "emissive_intensity": 2.0,
        "transmission": 0.9,
        "glass_thickness": 0.5,
    }
    kw = pbr_props_to_kwargs([0.7, 0.6, 0.5], props)
    for key in (
        "color",
        "roughness",
        "metallic",
        "reflectance",
        "alpha",
        "clearcoat",
        "clearcoat_roughness",
        "anisotropy",
        "emissive_color",
        "emissive_intensity",
        "transmission",
        "glass_thickness",
    ):
        assert key in kw, f"missing {key}"
    assert kw["clearcoat"] == 0.6
    assert kw["glass_thickness"] == 0.5
    assert kw["emissive_color"] == (1.0, 0.5, 0.0)


def test_pbr_props_to_kwargs_emissive_intensity_uses_base_color_by_default() -> None:
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    kw = pbr_props_to_kwargs(
        [0.2, 0.4, 0.9],
        {
            "emissive_intensity": 3.0,
        },
    )
    assert kw["emissive_color"] == (0.2, 0.4, 0.9)
    assert kw["emissive_intensity"] == 3.0


def test_pbr_props_to_kwargs_keeps_explicit_emissive_color() -> None:
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    kw = pbr_props_to_kwargs(
        [0.2, 0.4, 0.9],
        {
            "emissive_color": (1.0, 0.3, 0.1),
            "emissive_intensity": 3.0,
        },
    )
    assert kw["emissive_color"] == (1.0, 0.3, 0.1)


def test_pbr_props_to_kwargs_neutral_defaults() -> None:
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    kw = pbr_props_to_kwargs([0.5, 0.5, 0.5], {})
    assert kw["clearcoat"] == 0.0
    assert kw["anisotropy"] == 0.0
    assert kw["transmission"] == 0.0
    assert kw["glass_thickness"] == 0.0
    assert kw["emissive_color"] == (0.0, 0.0, 0.0)
    assert kw["emissive_intensity"] == 0.0


def test_pbr_props_to_kwargs_strips_texture_maps_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    kw = pbr_props_to_kwargs(
        [0.5, 0.5, 0.5],
        {
            "texture_path": "/tmp/albedo.png",
            "normal_map_path": "/tmp/normal.png",
            "roughness_map_path": "/tmp/roughness.png",
            "ao_map_path": "/tmp/ao.png",
            "metallic_map_path": "/tmp/metallic.png",
        },
    )

    assert kw["texture_path"] is None
    assert kw["normal_map_path"] is None
    assert kw["roughness_map_path"] is None
    assert kw["ao_map_path"] is None
    assert kw["metallic_map_path"] is None


def test_itu_marble_preset_produces_clearcoat_kwargs() -> None:
    """Real ITU preset → kwargs round-trip carries advanced fields."""
    from visualizer.src.materials.catalog import ITU_TO_PBR, pbr_props_to_kwargs

    marble = ITU_TO_PBR["marble"]
    kw = pbr_props_to_kwargs(marble["color"], marble)
    assert kw["clearcoat"] == 0.6
    assert kw["clearcoat_roughness"] == 0.1


def test_itu_metal_preset_produces_anisotropy_kwargs() -> None:
    from visualizer.src.materials.catalog import ITU_TO_PBR, pbr_props_to_kwargs

    metal = ITU_TO_PBR["metal"]
    kw = pbr_props_to_kwargs(metal["color"], metal)
    assert kw["anisotropy"] == 0.15


def test_itu_glass_preset_produces_transmission_kwargs() -> None:
    from visualizer.src.materials.catalog import ITU_TO_PBR, pbr_props_to_kwargs

    glass = ITU_TO_PBR["glass"]
    kw = pbr_props_to_kwargs(glass["color"], glass)
    assert kw["transmission"] == 0.9
    assert kw["glass_thickness"] == 0.5


def test_pygfx_set_named_material_uses_explicit_material_color_source() -> None:
    """A material-owned mesh without colors remains in uniform color mode."""
    r = _make_renderer()

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    payload = MeshPayload(vertices=verts, triangles=tris)
    name = "scene:merged_material_group::mesh"

    assert r.ensure_named_geometry(name, payload)
    assert r.set_named_material(
        name,
        MaterialPayload(base_color=(0.266, 0.109, 0.060, 1.0)),
    )

    obj = r._objects[name]
    assert getattr(obj.geometry, "colors", None) is None
    assert obj.material.color_mode == "uniform"
    assert obj.material.color.r == pytest.approx(0.266, abs=1e-5)
    assert obj.material.color.g == pytest.approx(0.109, abs=1e-5)
    assert obj.material.color.b == pytest.approx(0.060, abs=1e-5)
    assert r._geometry_color_sources[name] is SurfaceColorSource.MATERIAL


def test_pygfx_ensure_named_geometry_preserves_neutral_vertex_colors() -> None:
    """RenderObject/ensure_object paths must keep node and target vertex colors active."""
    r = _make_renderer()

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    payload = MeshPayload(
        vertices=verts,
        triangles=tris,
        vertex_colors=colors,
        color_source=SurfaceColorSource.VERTEX,
    )
    name = "node:tx_0::marker"

    ok = r.ensure_named_geometry(
        name,
        payload,
        material=MaterialPayload(base_color=(1.0, 0.0, 0.0, 1.0)),
        visible=True,
    )

    assert ok
    assert r._geometry_color_sources[name] is SurfaceColorSource.VERTEX
    obj = r._objects[name]
    assert getattr(obj.geometry, "colors", None) is not None
    assert obj.material.color_mode == "vertex"


def test_pygfx_clearcoat_roundtrip_returns_to_standard_material() -> None:
    """clearcoat 0 -> N -> 0 should not leave the mesh on MeshPhysicalMaterial."""
    import pygfx as gfx

    r = _make_renderer()

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    payload = MeshPayload(vertices=verts, triangles=tris)
    name = "wood_mesh"
    color = (0.266, 0.109, 0.060, 1.0)

    assert r.ensure_named_geometry(
        name,
        payload,
        material=MaterialPayload(
            base_color=color,
            roughness=0.7,
            metallic=0.0,
        ),
    )
    assert type(r._objects[name].material) is gfx.MeshStandardMaterial

    assert r.set_named_material(
        name,
        MaterialPayload(
            base_color=color,
            roughness=0.7,
            metallic=0.0,
            clearcoat=0.6,
            clearcoat_roughness=0.1,
        ),
    )
    assert type(r._objects[name].material) is gfx.MeshPhysicalMaterial

    assert r.set_named_material(
        name,
        MaterialPayload(
            base_color=color,
            roughness=0.7,
            metallic=0.0,
            clearcoat=0.0,
            clearcoat_roughness=0.0,
        ),
    )
    assert type(r._objects[name].material) is gfx.MeshStandardMaterial
    assert r._objects[name].material.color.r == pytest.approx(color[0], abs=1e-5)
    assert r._objects[name].material.color.g == pytest.approx(color[1], abs=1e-5)
    assert r._objects[name].material.color.b == pytest.approx(color[2], abs=1e-5)


def test_sync_pbr_properties_carries_advanced_fields() -> None:
    """Catalog PBR sync must propagate advanced PBR
    when the material type changes — the previous version stripped them.
    """
    from visualizer.src.services.material_properties import (
        sync_entry_pbr_properties_from_catalog,
    )

    entry: dict = {}
    sync_entry_pbr_properties_from_catalog(entry, "marble")
    props = entry["pbr_properties"]
    assert props["clearcoat"] == 0.6
    assert props["clearcoat_roughness"] == 0.1

    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "glass")
    props = entry["pbr_properties"]
    assert props["transmission"] == 0.9
    assert props["glass_thickness"] == 0.5

    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "metal")
    props = entry["pbr_properties"]
    assert props["anisotropy"] == 0.15

    # Default material must produce neutral advanced fields.
    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "concrete")
    props = entry["pbr_properties"]
    assert props["clearcoat"] == 0.0
    assert props["anisotropy"] == 0.0
    assert props["transmission"] == 0.0


def test_mesh_entry_to_payload_carries_advanced_fields() -> None:
    """scene.assembly.mesh_entry_to_payload must populate the advanced
    MaterialPayload fields for the notebook static path.
    """
    from visualizer.src.scene.assembly import mesh_entry_to_payload

    mesh = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_colors=np.ones((3, 3), dtype=np.float64),
    )
    entry = {
        "mesh": mesh,
        "name": "test_marble",
        "material_type": "marble",
        "material_id": None,
        "color": [0.7, 0.6, 0.5],
        "visible": True,
    }
    result = mesh_entry_to_payload(entry, texture_cache={}, index=0)
    assert result is not None
    _name, payload, material = result
    assert payload.vertex_colors is None
    assert material.clearcoat == 0.6
    assert material.clearcoat_roughness == 0.1
    assert material.has_advanced_pbr is True

    entry["material_type"] = "glass"
    result = mesh_entry_to_payload(entry, texture_cache={}, index=0)
    assert result is not None
    _name, _payload, material = result
    assert material.transmission == 0.9
    assert material.glass_thickness == 0.5
    # Wave B alone must NOT trigger MeshPhysicalMaterial selection.
    assert material.has_advanced_pbr is False
    # Absorption color flows through (slight cool tint for glass).
    assert material.absorption_color != (1.0, 1.0, 1.0)


# =============================================================================
# Helper coverage: every PBR call site uses pbr_props_to_kwargs
# =============================================================================


def test_pbr_props_to_kwargs_includes_absorption_color() -> None:
    """The shared helper must forward the new Wave B absorption_color field."""
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    kw = pbr_props_to_kwargs(
        [0.5, 0.5, 0.5],
        {"absorption_color": (0.9, 0.95, 1.0)},
    )
    assert kw["absorption_color"] == (0.9, 0.95, 1.0)


def test_pbr_props_to_kwargs_absorption_default_is_neutral() -> None:
    from visualizer.src.materials.catalog import pbr_props_to_kwargs

    kw = pbr_props_to_kwargs([0.5, 0.5, 0.5], {})
    assert kw["absorption_color"] == (1.0, 1.0, 1.0)


def test_itu_glass_preset_carries_absorption_and_higher_alpha() -> None:
    """Verify the glass preset redesign: alpha bumped, absorption_color set."""
    from visualizer.src.materials.catalog import ITU_TO_PBR

    glass = ITU_TO_PBR["glass"]
    assert glass["alpha"] == 0.7  # was 0.25 before the redesign
    assert glass["transmission"] == 0.9
    assert glass["glass_thickness"] == 0.5
    assert "absorption_color" in glass
    # Slight cool tint (green and blue near 1.0, red slightly less).
    assert glass["absorption_color"][0] < 1.0


def test_itu_water_preset_carries_absorption_and_higher_alpha() -> None:
    from visualizer.src.materials.catalog import ITU_TO_PBR

    water = ITU_TO_PBR["water"]
    assert water["alpha"] == 0.85  # was 0.7 before the redesign
    assert water["transmission"] == 0.6
    assert water["glass_thickness"] == 2.0
    assert "absorption_color" in water
