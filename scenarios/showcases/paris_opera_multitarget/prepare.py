"""Prepare a portable five-target Paris recording from explicit supplied walking input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "scripts"))
from walking import WALK_START, WALK_V, prepare_walking

RATE = 2000.0
CPI = 128
RADAR = np.array([-36.50568975108494, -44.60653584028523, 5.5])
ROUTES = {
    "Car_Outbound": ([-62.0, -65.0, 0.9090677518], [-2.28, 1.95, 0.0], "bmw"),
    "Car_Inbound": ([-4.4, -94.0, 0.9090677518], [-0.88, 3.05, 0.0], "bmw"),
    "Scooter": ([-9.2, -78.0, 0.7339999914], [-0.52, 1.8, 0.0], "scooter"),
    "Drone": ([-25.0, -82.0, 11.0], [-0.28, 1.65, 0.32], "drone"),
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_scene(directory):
    """Derive the surface scene from bundled original inputs, never a run cache."""
    source = HERE / "inputs/scene"
    destination = directory / "scene"
    shutil.copytree(source, destination)
    tree = ET.parse(destination / "scene.xml")
    root = tree.getroot()
    for node in root.findall(".//ref"):
        if node.get("id") in ("mat-itu_wet_ground", "mat-itu_medium_dry_ground"):
            node.set("id", "mat-itu_concrete")
    for node in list(root.findall("bsdf")):
        if node.get("id") in ("mat-itu_wet_ground", "mat-itu_medium_dry_ground"):
            root.remove(node)
    tree.write(destination / "scene.xml", encoding="utf-8", xml_declaration=True)
    sidecar = json.loads((source / "visual_surfaces.json").read_text())
    for mesh in sidecar["meshes"].values():
        for surface in mesh["surfaces"]:
            for key, value in surface["material"].items():
                if key.endswith("texture_path"):
                    local = Path("textures") / Path(value).name
                    if not (destination / local).is_file():
                        raise FileNotFoundError(local)
                    surface["material"][key] = local.as_posix()
    dump(destination / "visual_surfaces.json", sidecar)
    from clean_osm import main as clean_scene
    from paving import main as prepare_paving

    clean_scene(directory, HERE / "inputs")
    shutil.copy2(
        HERE / "inputs/paris_limestone_paving.png",
        directory / "scene_clean/textures/paris_limestone_paving.png",
    )
    prepare_paving(directory / "scene_clean")


def prepare_vehicles(directory):
    import open3d as o3d
    from PIL import Image
    from vehicles import read_vehicle

    destination = directory / "assets/vehicles"
    shutil.copytree(HERE / "inputs/vehicles", destination)
    shutil.copy2(destination / "bmw/car.ply", destination / "bmw/target.ply")
    parts, _, info = read_vehicle("scooter", destination)
    mesh = o3d.geometry.TriangleMesh()
    for _, payload, material in parts:
        vertices = payload.vertices.copy()
        uv = np.zeros((len(vertices), 2))
        uv[payload.triangles.reshape(-1)] = payload.triangle_uvs
        texture = np.asarray(Image.open(material.texture_path).convert("RGB"))
        h, w = texture.shape[:2]
        color = (
            texture[
                np.floor((uv[:, 1] % 1) * (h - 1)).astype(int),
                np.floor((uv[:, 0] % 1) * (w - 1)).astype(int),
            ]
            / 255.0
        )
        part = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(payload.triangles.copy()),
        )
        part.vertex_colors = o3d.utility.Vector3dVector(color)
        part.vertex_normals = o3d.utility.Vector3dVector(payload.normals.copy())
        mesh += part
    if not o3d.io.write_triangle_mesh(str(destination / "scooter/target.ply"), mesh):
        raise OSError("Could not write scooter mesh")
    info["rf_geometry"] = "All original triangles; canonical +X-front; metal RF surrogate"
    dump(destination / "scooter/conversion.json", info)
    shutil.copytree(ROOT / "libraries/targets/uav/large", directory / "assets/drone")
    shutil.copytree(HERE / "inputs/uav_surface", directory / "assets/drone_surface")


def legacy_walking(library, output):
    """Import the explicitly supplied historical 34-pose library without guessing a derivation."""
    import open3d as o3d

    from shared.animated_target_export import export_animated_target

    metadata_path = library / "mesh_library.json"
    metadata = json.loads(metadata_path.read_text())
    files = sorted(library.glob("smpl_walk_*.ply"))
    if len(files) != 34 or metadata.get("fps") != 30 or not metadata.get("cyclic"):
        raise ValueError(
            "historical alternative requires the declared 34-pose, 30 Hz cyclic library"
        )
    meshes = [o3d.io.read_triangle_mesh(str(path)) for path in files]
    faces = np.asarray(meshes[0].triangles)
    if not len(faces) or any(
        not np.array_equal(np.asarray(mesh.triangles), faces) for mesh in meshes
    ):
        raise ValueError("walking library must have nonempty identical ordered topology")
    vertices = np.array([np.asarray(mesh.vertices) for mesh in meshes])[..., [1, 0, 2]].copy()
    vertices[..., 0] *= -1
    return export_animated_target(
        output,
        asset_id="supplied_historical_walk",
        vertices=vertices,
        faces=faces,
        times_s=np.arange(34) / 30,
        root_translation=np.zeros((34, 3)),
        root_rotation_xyzw=np.tile([0, 0, 0, 1], (34, 1)),
        source_type="supplied_legacy_meshes",
        front_axis="+X",
        up_axis="+Z",
        source_records={
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in [metadata_path, *files]
        },
        adapter_metadata={
            "basis": "legacy (-y,x,z) rotation; no recentering",
            "cycle_policy": "34 sample slots / 30 Hz = 34/30 s",
            "source_metadata": metadata,
            "derivation_limit": "raw clip cycle selection/model normalization not established",
        },
    )


def copy_surface(source, destination):
    """Copy just the explicit external surface sidecar and its local resource closure."""
    config = json.loads(source.read_text())
    destination.mkdir(parents=True)
    for key in ("texture", "uv_map"):
        relative = Path(config[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("walking surface resources must be local relative paths")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.parent / relative, target)
    if config.get("material"):
        raise ValueError("walking surface must use a scalar preset and texture/UV resources only")
    dump(destination / "surface.json", config)


def prepare(
    output,
    epochs=80,
    rays=800000,
    start_s=0.0,
    *,
    walking_asset=None,
    walk_library=None,
    pose_period_s=None,
    walking_offset=(0.0, 0.0, 0.0),
    walking_surface=None,
):
    """Publish a new self-contained run; never mutate source inputs or existing output."""
    if epochs <= 0 or rays <= 0:
        raise ValueError("epochs and rays must be positive")
    if (walking_asset is None) == (walk_library is None):
        raise ValueError("supply exactly one walking asset or historical walk library")
    if walk_library is not None and pose_period_s is not None:
        raise ValueError("the historical library already declares its 34/30 s period")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        steps = epochs * CPI
        duration = (steps - 1) / RATE
        source = walking_asset
        if walk_library is not None:
            source = legacy_walking(walk_library, stage / "assets/source_walking")
            pose_period_s = 34 / 30
        prepare_walking(
            source,
            stage / "assets/walker",
            duration,
            start_s=start_s,
            pose_period_s=pose_period_s,
            offset_m=walking_offset,
        )
        prepare_scene(stage)
        prepare_vehicles(stage)
        config = yaml.safe_load((HERE / "scenario.yaml").read_text())
        config["timeline"] = {"steps": steps, "sample_rate_hz": RATE}
        config["raytracing"]["quality"]["custom"]["samples_per_src"] = rays
        for target in config["actors"]["targets"]:
            name = target["name"]
            if name not in ROUTES:
                continue
            start, velocity, _ = ROUTES[name]
            initial = np.array(start) + np.array(velocity) * start_s
            target["mobility"].update(
                start_m=initial.tolist(), end_m=(initial + np.array(velocity) * duration).tolist()
            )
        from native_appearance import view_defaults

        config["view_defaults"].update(view_defaults(stage))
        if walking_surface:
            copy_surface(walking_surface, stage / "assets/walking_surface")
            config["view_defaults"]["visual_profiles"].append(
                dict(
                    match=dict(name="Pedestrian", type="target"),
                    preset="Matte",
                    surface="assets/walking_surface/surface.json",
                )
            )
        shutil.copy2(HERE / "ATTRIBUTION.md", stage / "ATTRIBUTION.md")
        shutil.copy2(HERE / "inputs/asset_manifest.json", stage / "static_input_manifest.json")
        (stage / "scenario.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        dump(
            stage / "design.json",
            dict(
                epochs=epochs,
                duration_s=steps / RATE,
                last_sample_time_s=duration,
                route_offset_s=start_s,
                routes=ROUTES,
                radar=RADAR.tolist(),
                walker_start=WALK_START.tolist(),
                walker_velocity=WALK_V.tolist(),
                walking_offset_m=list(walking_offset),
                walking_pose_period_s=pose_period_s,
                preparation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        )
        output.mkdir()
        try:
            for item in stage.iterdir():
                item.replace(output / item.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        shutil.rmtree(stage)
    print(f"Prepared {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--walking-asset", type=Path, help="Existing user-supplied animated_target.json file"
    )
    inputs.add_argument(
        "--walk-library",
        type=Path,
        help="Existing 34-pose walking library directory containing mesh_library.json",
    )
    parser.add_argument(
        "--pose-period", type=float, help="Explicit source-pose repetition period in seconds"
    )
    parser.add_argument(
        "--walking-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z")
    )
    parser.add_argument("--walking-surface", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--rays", type=int, default=800000)
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Route/pose offset for a separately labeled short check",
    )
    args = parser.parse_args()
    if args.walking_asset is not None and not args.walking_asset.is_file():
        parser.error(
            f"--walking-asset must name an existing animated_target.json file: "
            f"{args.walking_asset}. Supply the actual path produced by your motion importer."
        )
    if args.walk_library is not None and (
        not args.walk_library.is_dir() or not (args.walk_library / "mesh_library.json").is_file()
    ):
        parser.error(
            f"--walk-library must name an existing walking library directory containing "
            f"mesh_library.json: {args.walk_library}. Supply the actual path to your local library."
        )
    prepare(
        args.output,
        args.epochs,
        args.rays,
        args.start,
        walking_asset=args.walking_asset,
        walk_library=args.walk_library,
        pose_period_s=args.pose_period,
        walking_offset=args.walking_offset,
        walking_surface=args.walking_surface,
    )
