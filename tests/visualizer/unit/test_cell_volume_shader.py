"""Discrete cell faces retain exact geometry, depth, and source-cell picking."""

import numpy as np
import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.cell_volume import CellVolumeMaterial, CellVolumeShader


def test_cell_shader_replaces_surface_search_and_keeps_depth_separate_from_pick():
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(np.ones((3, 3, 3), np.float32), dim=3)),
        CellVolumeMaterial(lighting=True, surface_grid=True),
    )
    code = CellVolumeShader(volume).get_code()
    branch = code.split("$$ elif mode == 'iso'")[1].split("\n$$ else\n    fn raycast")[0]
    assert "substep_start_coord" not in branch
    assert "out.coord = center;" in branch
    assert "out.depth = ndc_pos.z / ndc_pos.w;" in branch
    assert "u_wobject.world_transform * vec4f(lattice - 0.5, 1.0)" in branch


@pytest.mark.pygfx_runtime
@pytest.mark.parametrize("lighting,grid", [(False, False), (True, False), (False, True)])
def test_discrete_cells_match_exact_boxes_and_pick_the_color_source(lighting, grid):
    from rendercanvas.offscreen import RenderCanvas
    from scipy.ndimage import binary_dilation, binary_erosion

    values = np.zeros((3, 3, 3), np.float32)
    values[0, 0, 0], values[1, 1, 1], values[2, 2, 2] = 2, 1, 3
    palette = np.array(
        [[0, 0, 0, 0], [0.1, 0.8, 0.95, 1], [0.73, 0.64, 1, 1], [0.97, 0.80, 0.33, 1]], np.float32
    )
    images, picks = [], []
    shift = np.array([10, -20, 30])
    scale = np.array([1.5, 1, 2])
    for exact in (False, True):
        canvas = RenderCanvas(size=(400, 400))
        renderer = gfx.WgpuRenderer(canvas)
        scene = gfx.Scene()
        scene.add(gfx.Background(None, gfx.BackgroundMaterial("#20252e")))
        scene.add(gfx.AmbientLight(intensity=1))
        cell_by_object = {}
        if exact:
            for z, y, x in np.argwhere(values > 0):
                box = gfx.Mesh(
                    gfx.box_geometry(*scale),
                    gfx.MeshBasicMaterial(color=palette[int(values[z, y, x])], pick_write=True),
                )
                box.local.position = shift + scale * [x, y, z]
                scene.add(box)
                cell_by_object[box] = (int(x), int(y), int(z))
        else:
            material = CellVolumeMaterial(
                lighting=lighting,
                surface_grid=grid,
                threshold=0.5,
                interpolation="nearest",
                pick_write=True,
            )
            material.map = gfx.TextureMap(
                gfx.Texture(palette, dim=1), filter="nearest", wrap="clamp"
            )
            material.maprange = (0, 3)
            material.alpha_mode = "dither"
            volume = gfx.Volume(gfx.Geometry(grid=gfx.Texture(values, dim=3)), material)
            volume.local.position = shift
            volume.local.scale = scale
            scene.add(volume)
        camera = gfx.OrthographicCamera(7, 7, depth_range=(0.1, 100))
        camera.local.position = shift + [10, -8, 12]
        camera.look_at(shift + scale)
        canvas.request_draw(lambda: renderer.render(scene, camera))
        im = np.asarray(canvas.draw()).copy()
        images.append(im)
        hits = {}
        for y in range(20, 380, 20):
            for x in range(20, 380, 20):
                pick = renderer.get_pick_info((x, y))
                obj = pick.get("world_object")
                if obj is not None:
                    hits[(x, y)] = cell_by_object.get(obj) if exact else pick.get("index")
        picks.append(hits)
        canvas.close()
    # Every interior sample and every exterior sample away from rasterized
    # edges agrees with geometric boxes, regardless of lighting or cell guides.
    masks = [np.any(image[:, :, :3] != image[0, 0, :3], axis=2) for image in images]
    if not grid:
        assert np.all(masks[0][binary_erosion(masks[1], iterations=2)])
    assert not masks[0][~binary_dilation(masks[1], iterations=2)].any()
    assert picks[0] == picks[1]
    assert set(picks[0].values()) == {(0, 0, 0), (1, 1, 1), (2, 2, 2)}
