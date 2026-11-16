"""Exclude cached underground OSM rooms incorrectly extruded above the plaza.

Only this showcase copy is changed. Every excluded building part must match a
source GeoJSON footprint and an explicit negative-level/underground tag. This
is a surface scene, so underground circulation is omitted rather than extruded.
Requires the optional OSM dependencies (pyproj, shapely) and Open3D.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def main(directory=HERE, inputs=HERE / "inputs"):
    source = directory / "scene"
    destination = directory / "scene_clean"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    metadata = json.loads((source / "osm_metadata.json").read_text())
    visuals = json.loads((source / "building_visuals.json").read_text())
    surfaces = json.loads((source / "visual_surfaces.json").read_text())
    origin = np.array(metadata["origin_utm"][:2])
    project = Transformer.from_crs(4326, 32631, always_xy=True).transform
    cache = inputs / "osm_building_parts.geojson"
    candidates = []
    for feature in json.loads(cache.read_text())["features"]:
        tags = {k: v for k, v in feature["properties"].items() if v is not None}
        levels = str(tags.get("level", "")).split(";")
        underground = tags.get("location") == "underground" or tags.get("tunnel") == "yes"
        try:
            underground |= (
                bool(levels) and max(map(float, levels)) <= 0 and min(map(float, levels)) < 0
            )
        except ValueError:
            pass
        if underground:
            polygon = transform(project, shape(feature["geometry"]))
            bounds = np.array(polygon.bounds).reshape(2, 2) - origin
            candidates.append((bounds, tags))
    report = []
    for filename, info in visuals["visual_building_faces"].items():
        mesh = o3d.io.read_triangle_mesh(str(source / filename))
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        keep = np.ones(len(triangles), dtype=bool)
        removed = []
        for first, count in info["parts"]:
            pts = vertices[triangles[first : first + count]].reshape(-1, 3)
            bounds = np.stack((pts[:, :2].min(0), pts[:, :2].max(0)))
            matches = [(b, tags) for b, tags in candidates if np.max(np.abs(bounds - b)) < 0.15]
            if matches:
                keep[first : first + count] = False
                removed.append((first, count))
                report.append(
                    dict(
                        mesh=filename,
                        first_triangle=first,
                        triangle_count=count,
                        tags=matches[0][1],
                        bounds_m=bounds.tolist(),
                        previous_maximum_height_m=float(pts[:, 2].max()),
                    )
                )
        if not removed:
            continue
        index = np.cumsum(keep) - 1
        mesh.triangles = o3d.utility.Vector3iVector(triangles[keep].copy())
        mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(str(destination / filename), mesh)
        info["parts"] = [
            [int(index[first]), count] for first, count in info["parts"] if keep[first]
        ]
        info["building_triangles"] -= sum(
            count for first, count in removed if first < info["building_triangles"]
        )
        info["total_triangles"] = int(keep.sum())
        for exclusion in info.get("facade_exclusions", []):
            exclusion["first_triangle"] = int(index[exclusion["first_triangle"]])
        metadata["visual_building_faces"][filename] = info
        if filename in surfaces["meshes"]:
            surface_mesh = surfaces["meshes"][filename]
            surface_mesh["source_sha256"] = hashlib.sha256(
                (destination / filename).read_bytes()
            ).hexdigest()
            for surface in surface_mesh["surfaces"]:
                old = np.array(surface["triangle_indices"])
                assert np.all(keep[old]), "Do not delete an authored landmark surface"
                surface["triangle_indices"] = index[old].tolist()
    visuals["provenance"] = dict(
        source="../scene/building_visuals.json",
        rf_mesh_or_material_changes=True,
        policy="Remove only matched OSM underground parts",
        audit="surface_scene_audit.json",
    )
    for name, data in (
        ("building_visuals.json", visuals),
        ("osm_metadata.json", metadata),
        ("visual_surfaces.json", surfaces),
        ("surface_scene_audit.json", report),
    ):
        (destination / name).write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Excluded {len(report)} explicitly underground parts ({sum(r['triangle_count'] for r in report)} triangles)"
    )


if __name__ == "__main__":
    main()
