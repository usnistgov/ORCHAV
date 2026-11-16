"""Prepare target UV skins and a portable look for the ordinary visualizer."""

import json
from pathlib import Path

import numpy as np
from demo_tour import paris_demo_tour
from dusk_look import paris_dusk
from PIL import Image
from vehicles import read_vehicle

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def scooter_surface(directory=HERE):
    """Pack repeating source albedos without moving or simplifying RF triangles."""
    folder = directory / "assets/vehicles/scooter"
    if (folder / "surface.json").exists():
        return
    parts, _, _ = read_vehicle("scooter", directory / "assets/vehicles")
    cell, padding, columns = 1024, 8, 4
    stride = cell + 2 * padding
    atlas = np.zeros((stride * 4, stride * columns, 3), np.uint8)
    faces, uvs, count = [], [], 0
    for i, (_, payload, material) in enumerate(parts):
        uv = payload.triangle_uvs.astype(np.float64)
        low, high = np.floor(uv.min(0)), np.ceil(uv.max(0))
        span = np.maximum(high - low, 1)
        texture = np.asarray(Image.open(material.texture_path).convert("RGB"))
        h, w = texture.shape[:2]
        xy = (np.arange(stride) - padding + 0.5) / cell
        xs = np.floor(((low[0] + xy * span[0]) % 1) * w).astype(int)
        ys = np.floor(((low[1] + xy * span[1]) % 1) * h).astype(int)
        x, y = (i % columns) * stride, (i // columns) * stride
        atlas[y : y + stride, x : x + stride] = texture[ys[:, None], xs[None, :]]
        packed = ((uv - low) / span * cell + [x + padding, y + padding]) / atlas.shape[1::-1]
        faces.append(payload.triangles + count)
        uvs.append(packed)
        count += len(payload.vertices)
    faces = np.concatenate(faces).astype(np.int32)
    # Match the existing sensing mesh exactly, including ordered topology.
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(folder / "target.ply"))
    assert len(mesh.vertices) == count
    assert np.array_equal(np.asarray(mesh.triangles), faces)
    Image.fromarray(atlas).save(folder / "albedo_atlas.png")
    np.savez_compressed(
        folder / "surface_uv.npz",
        triangles=faces,
        triangle_uvs=np.concatenate(uvs).astype(np.float32),
        vertex_count=np.int64(count),
    )
    (folder / "surface.json").write_text(
        json.dumps(
            dict(
                schema="orchav.target_surface.v1",
                texture="albedo_atlas.png",
                uv_map="surface_uv.npz",
                description="Original repeating CC0 albedos packed for native target rendering; RF topology unchanged.",
            ),
            indent=2,
        )
        + "\n"
    )


def view_defaults(directory=HERE):
    scooter_surface(directory)
    from surface_detail import prepare_surface_detail

    prepare_surface_detail(directory)
    recipe = dict(
        version=1,
        surface="original",
        unlit=False,
        textures=True,
        wireframe=False,
        background=[0.51, 0.71, 0.85],
        alpha=1.0,
        outlines=False,
        building_appearance="varied",
        facade_theme="haussmann",
        road_appearance="marked",
        authored_surface_textures=True,
        sky="default",
        skybox=False,
        ibl=5000.0,
        lights=dict(
            lighting_profile="custom",
            ambient_intensity=0.45,
            headlight_enabled=False,
            headlight_intensity=1.2,
            key_azimuth_deg=31.0,
            key_elevation_deg=-42.0,
            key_intensity=3.0,
            fill_azimuth_deg=-149.0,
            fill_elevation_deg=-60.0,
            fill_intensity=0.25,
            shadows_enabled=True,
        ),
        aa="fxaa",
        mipmaps=True,
        color_fidelity=False,
        shadow_quality="focus",
        shadow_radius=75.0,
        shadow_bias=2.0,
        depth_precision=True,
    )
    path = directory / "appearance.json"
    path.write_text(json.dumps(recipe, indent=2) + "\n")
    return dict(
        camera_pose=dict(eye=[-6, -129, 16], lookat=[-38, -55, 9], up=[0, 0, 1], fov=60),
        demo_tour=paris_demo_tour(),
        dusk=paris_dusk(),
        appearance_recipe="appearance.json",
        playback=dict(mode="fixed_fps", fps=25, stride=32),
        show_labels=False,
        show_target_labels=False,
        viewport_hud_enabled=False,
        label_callout=dict(
            mode="card",
            background="ink",
            opacity=0.92,
            offset_x=12,
            offset_y=24,
            marker_size=10,
            leader=True,
            on_top=True,
        ),
        sensing=dict(
            show_detections=True,
            scene_mode="tracks",
            show_track_labels=True,
            show_truth=False,
            full_history=False,
            history_frames=32,
            pin_freshest=True,
        ),
    )


if __name__ == "__main__":
    print(json.dumps(view_defaults(), indent=2))
