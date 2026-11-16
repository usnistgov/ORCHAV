"""Real GPU volume-hit/depth regression, using only controlled scalar fixtures.

Run with a working WebGPU adapter; --output preserves native render evidence.
No ray tracer, detector or repository input is changed.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas

from visualizer.src.coverage.volume_inspection import scalar_template
from visualizer.src.renderers.pygfx.uniform_volume import (
    CellGridVolumeIsoMaterial,
    SceneVolumeIsoMaterial,
    UniformCellGridVolumeIsoMaterial,
    UniformVolumeIsoMaterial,
)


def render_iso(material_type, shift, opacity):
    z, y, x = np.mgrid[:16, :16, :16]
    values = np.asarray(8 - np.sqrt((x - 7.5) ** 2 + (y - 7.5) ** 2 + (z - 7.5) ** 2), np.float32)
    canvas = RenderCanvas(size=(480, 480))
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.Background(None, gfx.BackgroundMaterial((0.2, 0.2, 0.2, 1))))
    material = material_type(threshold=3, clim=(0, 8), pick_write=True, opacity=opacity)
    material.alpha_mode = "dither" if opacity == 1 else "weighted_blend"
    volume = gfx.Volume(gfx.Geometry(grid=gfx.Texture(values, dim=3)), material)
    volume.local.scale = (1.5, 1, 2)
    volume.local.position = shift
    # This plane is physically behind the entire visible sphere.
    plane = gfx.Mesh(gfx.plane_geometry(29, 19), gfx.MeshBasicMaterial(color="#df3333"))
    plane.local.position = (11.25 + shift[0], 7.5 + shift[1], shift[2])
    scene.add(volume, plane, gfx.AmbientLight(intensity=1))
    camera = gfx.OrthographicCamera(34, 34, depth_range=(0.1, 200))
    camera.local.position = (11.25 + shift[0], 7.5 + shift[1], 70 + shift[2])
    camera.look_at((11.25 + shift[0], 7.5 + shift[1], 15 + shift[2]))
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image = np.asarray(canvas.draw()).copy()
    pick = renderer.get_pick_info((240, 240))
    return image, pick.get("index"), pick.get("world_object") is volume


def render_mip():
    values = np.full((4, 4, 4), -90, np.float32)
    values[3] = -20  # nearer, above the display maximum
    values[0] = -10  # farther, still stronger: must win despite saturated colours
    payload = scalar_template(SimpleNamespace(values=values), -130, -50)
    canvas = RenderCanvas(size=(320, 320))
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    material = gfx.VolumeMipMaterial(interpolation="nearest", pick_write=True)
    material.map = gfx.TextureMap(
        gfx.Texture(payload.palette.copy(), dim=1), filter="nearest", wrap="clamp"
    )
    material.maprange = payload.limits
    volume = gfx.Volume(gfx.Geometry(grid=gfx.Texture(payload.values.copy(), dim=3)), material)
    volume.local.position = (10, -20, 30)
    scene.add(volume)
    camera = gfx.OrthographicCamera(5, 5, depth_range=(0.1, 100))
    camera.local.position = (11.5, -18.5, 45)
    camera.look_at((11.5, -18.5, 31.5))
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image = np.asarray(canvas.draw()).copy()
    pick = renderer.get_pick_info((160, 160))
    index = pick["index"]
    assert index[2] == 0 and values[tuple(reversed(index))] == -10
    return image, index


def render_slice_boundaries():
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    canvas = RenderCanvas(size=(400, 400))
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    material = gfx.VolumeSliceMaterial(
        interpolation="nearest", pick_write=True, plane=(0, 0, 1, -34), clim=(0, 23)
    )
    volume = gfx.Volume(gfx.Geometry(grid=gfx.Texture(values, dim=3)), material)
    volume.local.position = (10, 20, 30)
    volume.local.scale = (2, 3, 4)
    scene.add(volume)
    camera = gfx.OrthographicCamera(12, 12, depth_range=(0.1, 100))
    camera.local.position = (13, 23, 60)
    camera.look_at((13, 23, 34))
    canvas.request_draw(lambda: renderer.render(scene, camera))
    im = np.asarray(canvas.draw()).copy()
    picks = []
    for x, y in ((0, 0), (3, 0), (0, 2), (3, 2)):
        screen = (200 + (10 + 2 * x - 13) / 12 * 400, 200 - (20 + 3 * y - 23) / 12 * 400)
        pick = renderer.get_pick_info(screen)
        assert pick.get("index") == (x, y, 1), pick
        picks.append(pick["index"])
    assert renderer.get_pick_info((20, 200)).get("world_object") is None
    return im, picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    shift = (20, -10, 60)
    for opacity in (1.0, 0.55):
        for material in (
            gfx.VolumeIsoMaterial,
            SceneVolumeIsoMaterial,
            UniformVolumeIsoMaterial,
            CellGridVolumeIsoMaterial,
            UniformCellGridVolumeIsoMaterial,
        ):
            images = []
            picks = []
            for position in ((0, 0, 0), shift):
                im, index, hit = render_iso(material, position, opacity)
                name = f"{material.__name__}_{opacity}_{'translated' if any(position) else 'origin'}.png"
                iio.imwrite(args.output / name, im)
                images.append(im)
                picks.append((index, hit))
            equal = bool(np.array_equal(*images))
            results.append(
                dict(
                    material=material.__name__,
                    opacity=opacity,
                    translation_invariant=equal,
                    picks=picks,
                )
            )
            if material is not gfx.VolumeIsoMaterial:
                assert equal and picks[0] == picks[1] and picks[0][1], results[-1]
    im, index = render_mip()
    iio.imwrite(args.output / "mip_unsaturated_source_maximum.png", im)
    results.append(dict(mode="mip", index=index, raw_value=-10, units="dB", colour_maximum=-50))
    im, indices = render_slice_boundaries()
    iio.imwrite(args.output / "slice_translated_anisotropic_boundaries.png", im)
    results.append(dict(mode="slice", corner_indices=indices, outside_grid_rejected=True))
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
