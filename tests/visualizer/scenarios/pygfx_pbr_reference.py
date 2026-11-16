"""Deterministic, simulation-free PBR reference scene using ORCHAV builders.

Run this same script in separate processes with --repo-root pointing at the
baseline or candidate checkout. No render settings depend on before/after.
Generated HDR/texture inputs, camera, lights and palette are identical.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SIZE = (1280, 900)
CAMERA_POSITION = (0.0, -20.0, 22.0)
CAMERA_TARGET = (0.0, 0.0, 0.5)
PALETTE = np.array(
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], np.uint8
)


def write_hdr(path: Path, data: np.ndarray) -> None:
    """Write small deterministic, uncompressed RGBE reference input."""
    maximum = np.max(data, axis=-1)
    mantissa, exponent = np.frexp(maximum)
    scale = np.divide(mantissa * 256, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    rgbe = np.zeros((*maximum.shape, 4), np.uint8)
    rgbe[..., :3] = np.minimum(data * scale[..., None], 255).astype(np.uint8)
    rgbe[..., 3] = np.where(maximum > 0, exponent + 128, 0).astype(np.uint8)
    path.write_bytes(
        f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {data.shape[0]} +X {data.shape[1]}\n".encode()
        + rgbe.tobytes()
    )


def reference_inputs(directory: Path) -> Path:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    radiance = np.full((64, 128, 3), 0.18, np.float32)
    radiance[10:23, 42:65] = 4.0
    radiance[20:38, 96:108] = 1.0
    write_hdr(directory / "pbr_reference.hdr", radiance)
    checker = np.full((32, 32, 4), 255, np.uint8)
    grid = (np.indices((32, 32)).sum(axis=0) // 4) % 2 == 0
    checker[grid, :3] = (35, 90, 180)
    checker[..., 3] = 192
    Image.fromarray(checker).save(directory / "checker.png")
    return directory / "checker.png"


def build_scene(asset_dir: Path):
    """Return (scene, camera, manager), reusable by native export regression."""
    import pygfx as gfx

    from visualizer.src.backends.pygfx_scene_helpers import make_mesh_material
    from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager
    from visualizer.src.scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
    from visualizer.src.types.render_payloads import MaterialPayload

    texture_path = reference_inputs(asset_dir)
    manager = PygfxIBLManager(gfx, asset_dir)
    manager._face_size = 64
    manager.load_ibl("pbr_reference")
    manager.set_intensity(0.35)
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA))
    # Fixed modest direct rig isolates environment/material correctness without
    # clipping away the HDR comparison; this is not an app lighting preset.
    scene.add(gfx.AmbientLight("white", 0.12))
    key = gfx.DirectionalLight("white", 1.8)
    key.local.position = (-6, -7, 10)
    key.look_at((0, 0, 0))
    scene.add(key)

    def material(**kwargs):
        return make_mesh_material(gfx, MaterialPayload(**kwargs), ibl_manager=manager)

    floor = gfx.Mesh(
        gfx.box_geometry(15, 14, 0.12), material(base_color=(0.28, 0.29, 0.31, 1), roughness=0.95)
    )
    floor.local.position = (0, 0, -0.16)
    scene.add(floor)

    def label(text, position, size=18):
        obj = gfx.Text(
            text=text, font_size=size, screen_space=True, material=gfx.TextMaterial(color="white")
        )
        obj.local.position = position
        scene.add(obj)

    roughness = (0.08, 0.25, 0.55, 0.9)
    for row, (name, extra) in enumerate(
        (
            ("Dielectric", {}),
            ("Metal", {"metallic": 1.0}),
            ("Clearcoat 0.4", {"clearcoat": 0.4, "clearcoat_roughness": 0.15}),
        )
    ):
        y = 3.9 - row * 2.8
        label(name, (-6.2, y, 0.6), 16)
        for col, rough in enumerate(roughness):
            x = -3.8 + col * 2.6
            sphere = gfx.Mesh(
                gfx.sphere_geometry(0.86, 48, 32),
                material(base_color=(0.67, 0.7, 0.74, 1), roughness=rough, **extra),
            )
            sphere.local.position = (x, y, 0.78)
            scene.add(sphere)
            if row == 0:
                label(f"rough {rough:g}", (x, y + 1.2, 1.4), 16)

    for idx, reflectance in enumerate((0.0, 0.5, 1.0)):
        x = -4.8 + idx * 1.9
        sphere = gfx.Mesh(
            gfx.sphere_geometry(0.52, 40, 24),
            material(base_color=(0.3, 0.34, 0.4, 1), roughness=0.18, reflectance=reflectance),
        )
        sphere.local.position = (x, -4.5, 0.43)
        scene.add(sphere)
        label(f"reflectance {reflectance:g}", (x, -5.25, 0.1), 14)

    for idx, alpha in enumerate((0.25, 0.5, 0.75)):
        x = 1.0 + idx * 1.7
        panel = gfx.Mesh(
            gfx.plane_geometry(1.45, 1.25),
            material(base_color=(0.2, 0.7, 0.9, alpha), shader="unlit", clearcoat=0.4),
        )
        panel.local.position = (x, -4.5, 0.35)
        scene.add(panel)
        label(f"alpha {alpha:g}", (x, -5.3, 0.1), 14)

    textured = gfx.Mesh(
        gfx.plane_geometry(1.6, 1.6),
        material(base_color=(1, 1, 1, 0.5), shader="unlit", texture_path=str(texture_path)),
    )
    textured.local.position = (6.0, 0.0, 0.2)
    scene.add(textured)
    label("texture alpha x 0.5", (6.0, -1.1, 0.1), 14)

    # Immutable scientific colors bypass lit materials just as ORCHAV overlays
    # do. Both renders must retain this exact palette and per-vertex stream.
    for idx, rgb in enumerate(PALETTE):
        geo = gfx.plane_geometry(1.3, 0.55)
        geo.colors = gfx.Buffer(np.tile(np.r_[rgb / 255.0, 1].astype(np.float32), (4, 1)))
        patch = gfx.Mesh(geo, gfx.MeshBasicMaterial(color_mode="vertex", side="both"))
        patch.local.position = (-2.8 + idx * 1.4, -6.2, 0.1)
        scene.add(patch)
    label("Scientific color reference (unchanged)", (0, -7.1, 0.1), 16)

    # ORCHAV-style Z-up black Tx/Rx poles and familiar red/blue node markers.
    for x, name, color in ((-7, "Tx", "red"), (7, "Rx", "dodgerblue")):
        pole = gfx.Mesh(
            gfx.cylinder_geometry(0.065, 0.065, 2.6, 12), gfx.MeshBasicMaterial(color="black")
        )
        pole.local.position = (x, 4.2, 1.3)
        marker = gfx.Mesh(gfx.sphere_geometry(0.3, 24, 16), gfx.MeshBasicMaterial(color=color))
        marker.local.position = (x, 4.2, 2.6)
        scene.add(pole, marker)
        label(name, (x, 4.2, 3.2), 20)

    camera = gfx.OrthographicCamera(19, 19 * SIZE[1] / SIZE[0])
    camera.local.position = CAMERA_POSITION
    camera.world.reference_up = (0, 0, 1)
    camera.look_at(CAMERA_TARGET)
    return scene, camera, manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    os.environ["ORCHAV_ENABLE_TEXTURES"] = "1"
    os.environ.pop("ORCHAV_DISABLE_TEXTURES", None)
    import pygfx as gfx
    import wgpu
    from PIL import Image
    from rendercanvas.offscreen import RenderCanvas

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene, camera, manager = build_scene(args.output.parent / (args.output.stem + "_inputs"))
    canvas = RenderCanvas(size=SIZE, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="fxaa")
    canvas.request_draw(lambda: renderer.render(scene, camera))
    frame = np.asarray(canvas.draw()).copy()
    Image.fromarray(frame).save(args.output)
    manifest = {
        "repo_root": str(args.repo_root.resolve()),
        "output": str(args.output.resolve()),
        "size": list(SIZE),
        "camera_position": CAMERA_POSITION,
        "camera_target": CAMERA_TARGET,
        "camera_projection": "orthographic",
        "camera_width": 19,
        "background_srgb": [0.2, 0.2, 0.2],
        "ambient": 0.12,
        "directional": 1.8,
        "ibl_intensity": 0.35,
        "environment_linear_radiance": [0.18, 1.0, 4.0],
        "ibl_face_size": 64,
        "pygfx": gfx.__version__,
        "wgpu": wgpu.__version__,
        "adapter": dict(renderer._device.adapter.info),
        "ppaa": "fxaa",
        "scope": "Native ORCHAV material builders in a controlled backend scene; not a full application screenshot",
        "scientific_palette_rgb8": PALETTE.tolist(),
        "pixels_min": int(frame.min()),
        "pixels_max": int(frame.max()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
