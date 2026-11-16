"""Real-GPU reference checks for linear HDR, alpha, and scientific colors.

Requires the pygfx_runtime profile and a functioning WGPU adapter. Runs on
hardware or a Vulkan software adapter; it never substitutes a mocked image.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.backends.pygfx_pbr import pbr_material_classes
from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material
from visualizer.src.types.render_payloads import MaterialPayload

pytestmark = pytest.mark.pygfx_runtime


def _render(material, *, background=None, colors=None, scene_environment=None):
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(64, 64), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")
    scene = gfx.Scene()
    if scene_environment is not None:
        scene.environment = gfx.TextureMap(scene_environment)
    geometry = gfx.plane_geometry(2, 2)
    if colors is not None:
        geometry.colors = gfx.Buffer(np.tile(np.asarray(colors, np.float32), (4, 1)))
    scene.add(gfx.Mesh(geometry, material))
    if background is not None:
        scene.add(background)
    camera = gfx.OrthographicCamera(2, 2)
    camera.local.position = (0, 0, 4)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    return np.asarray(canvas.draw()).copy()


def _environment(data, *, physical=True):
    import pygfx as gfx

    cube = np.ones((6, 16, 16, 4), np.float32)
    cube[..., :3] = data
    return gfx.Texture(
        cube,
        dim=2,
        size=(16, 16, 6),
        generate_mipmaps=True,
        colorspace="physical" if physical else "srgb",
    )


@pytest.mark.parametrize("physical_class", [False, True])
def test_hdr_radiance_scales_linearly_through_ibl(physical_class):
    import pygfx as gfx

    standard, physical, _ = pbr_material_classes(gfx)
    cls = physical if physical_class else standard
    bright = _render(
        cls(
            color="white",
            metalness=1,
            roughness=1,
            env_map=_environment(4.0),
            env_map_intensity=0.05,
        )
    )
    reference = _render(
        cls(
            color="white",
            metalness=1,
            roughness=1,
            env_map=_environment(0.2),
            env_map_intensity=1.0,
        )
    )
    np.testing.assert_allclose(bright[16:48, 16:48], reference[16:48, 16:48], atol=1)
    assert 90 < int(bright[32, 32, 0]) < 160


def test_hdr_mipmaps_average_linear_radiance():
    import pygfx as gfx

    standard, _, _ = pbr_material_classes(gfx)
    checker = ((np.indices((16, 16)).sum(axis=0) % 2)[..., None] * np.ones(3)).astype(np.float32)
    filtered = _render(
        standard(color="white", metalness=1, roughness=1, env_map=_environment(checker))
    )
    reference = _render(
        standard(color="white", metalness=1, roughness=1, env_map=_environment(0.5))
    )
    np.testing.assert_allclose(filtered[16:48, 16:48], reference[16:48, 16:48], atol=1)


def test_skybox_decodes_physical_and_srgb_sources_correctly():
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    _, _, skybox = pbr_material_classes(gfx)
    samples = []
    for physical, value in ((True, 0.18), (False, 0.46135613)):
        scene = gfx.Scene()
        scene.add(gfx.Background(None, skybox(map=_environment(value, physical=physical))))
        canvas = RenderCanvas(size=(32, 32), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")
        camera = gfx.PerspectiveCamera()
        canvas.request_draw(lambda: renderer.render(scene, camera))
        samples.append(np.asarray(canvas.draw())[16, 16].copy())
    np.testing.assert_allclose(samples[0], samples[1], atol=1)
    assert abs(int(samples[0][0]) - 118) <= 2


@pytest.mark.parametrize("vertex", [False, True])
def test_rendered_half_alpha_is_not_squared(vertex):
    import pygfx as gfx

    material = make_mesh_material(
        gfx, MaterialPayload(base_color=(1, 1, 1, 0.5), shader="unlit"), has_vertex_colors=vertex
    )
    image = _render(
        material,
        background=gfx.Background.from_color("black"),
        colors=(1, 1, 1, 1) if vertex else None,
    )
    # Weighted transparency over black: half linear white -> about 186 sRGB.
    assert 182 <= int(image[32, 32, 0]) <= 190


def test_lit_promotion_does_not_recolor_scientific_unlit_vertices():
    import pygfx as gfx

    color = np.array([33, 145, 140], np.float32) / 255
    unlit = make_mesh_material(
        gfx, MaterialPayload(shader="unlit", clearcoat=0.8, reflectance=1.0), has_vertex_colors=True
    )
    image = _render(unlit, colors=(*color, 1))
    reference = _render(gfx.MeshBasicMaterial(color_mode="vertex"), colors=(*color, 1))
    # pygfx vertex colors are physical-space inputs. Preserve its existing
    # interpretation exactly; this fix must not introduce a colormap gamma change.
    np.testing.assert_array_equal(image, reference)


def test_rendered_rgba_texture_alpha_multiplies_payload_once(tmp_path, monkeypatch):
    import pygfx as gfx
    from PIL import Image

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    pixels = np.full((4, 4, 4), 255, np.uint8)
    pixels[..., 3] = 128
    path = tmp_path / "rgba.png"
    Image.fromarray(pixels).save(path)
    native = make_mesh_material(
        gfx,
        MaterialPayload(
            base_color=(1, 1, 1, 0.5),
            shader="unlit",
            texture_path=str(path),
        ),
    )
    assert native.color.a == 1.0
    assert native.opacity == 0.5
    image = _render(native, background=gfx.Background.from_color("black"))
    # 128/255 texture alpha * 0.5 payload alpha = ~0.251 linear white.
    assert 134 <= int(image[32, 32, 0]) <= 140


def test_scene_level_linear_environment_matches_material_binding():
    import pygfx as gfx

    if not hasattr(gfx.Scene(), "environment"):
        pytest.skip("This pygfx version has no scene environment")
    standard, _, _ = pbr_material_classes(gfx)
    texture = _environment(0.18)
    per_material = _render(standard(color="white", roughness=1, metalness=1, env_map=texture))
    per_scene = _render(
        standard(color="white", roughness=1, metalness=1), scene_environment=texture
    )
    np.testing.assert_array_equal(per_scene, per_material)
