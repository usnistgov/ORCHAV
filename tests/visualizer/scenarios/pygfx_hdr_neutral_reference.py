"""Render constant linear HDR references through the real ORCHAV IBL loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pygfx_pbr_reference import write_hdr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    import pygfx as gfx
    from PIL import Image
    from rendercanvas.offscreen import RenderCanvas

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material
    from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager
    from visualizer.src.types.render_payloads import MaterialPayload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = args.output.parent / (args.output.stem + "_inputs")
    asset_dir.mkdir(exist_ok=True)
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color((0.2, 0.2, 0.2, 1)))
    cases = ((0.04, 1), (0.18, 1), (0.5, 1), (1.0, 0.2), (4.0, 0.05), (0.2, 1.0))
    managers = []
    for index, (radiance, intensity) in enumerate(cases):
        name = f"constant_{index}"
        write_hdr(asset_dir / f"{name}.hdr", np.full((16, 32, 3), radiance, np.float32))
        manager = PygfxIBLManager(gfx, asset_dir)
        manager._face_size = 16
        manager.load_ibl(name)
        manager.set_intensity(intensity)
        managers.append(manager)
        mat = make_mesh_material(
            gfx,
            MaterialPayload(base_color=(1, 1, 1, 1), metallic=1, roughness=1),
            ibl_manager=manager,
        )
        patch = gfx.Mesh(gfx.plane_geometry(2.4, 1.3), mat)
        x, y = -3 + (index % 3) * 3, 1.3 - (index // 3) * 2.7
        patch.local.position = (x, y, 0)
        scene.add(patch)
        label = gfx.Text(
            text=f"linear HDR {radiance:g} x gain {intensity:g}",
            font_size=18,
            screen_space=True,
            material=gfx.TextMaterial(color="white"),
        )
        label.local.position = (x, y - 0.95, 0.1)
        scene.add(label)
    title = gfx.Text(
        text="Neutral HDR reference | lower row must match",
        font_size=24,
        screen_space=True,
        material=gfx.TextMaterial(color="white"),
    )
    title.local.position = (0, 2.9, 0.1)
    scene.add(title)
    camera = gfx.OrthographicCamera(10, 6.25)
    camera.local.position = (0, 0, 5)
    canvas = RenderCanvas(size=(1000, 625), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")
    canvas.request_draw(lambda: renderer.render(scene, camera))
    frame = np.asarray(canvas.draw()).copy()
    Image.fromarray(frame).save(args.output)
    samples = []
    for index, (radiance, intensity) in enumerate(cases):
        x, y = -3 + (index % 3) * 3, 1.3 - (index // 3) * 2.7
        col, row = round((x / 10 + 0.5) * 1000), round((0.5 - y / 6.25) * 625)
        samples.append(
            {
                "radiance": radiance,
                "intensity": intensity,
                "rgb8": frame[row, col, :3].tolist(),
                "sample_pixel": [col, row],
            }
        )
    manifest = {
        "repo_root": str(args.repo_root.resolve()),
        "cases": samples,
        "pygfx": gfx.__version__,
        "adapter": dict(renderer._device.adapter.info),
        "scope": "Real RGBE loading, linear cubemap filtering, rough metal IBL; no direct light or tone adjustment",
        "expected": "The three lower patches have equal radiance x gain = 0.2 and must match (within RGBE/8-bit quantization).",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
