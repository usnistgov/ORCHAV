"""Bind the authored limestone albedo to the existing plaza ground triangles.

UVs are measured in world metres. This only writes a visual sidecar; vertex
coordinates, triangle topology, motion and electromagnetic materials stay fixed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d

SCENE = Path(__file__).resolve().parents[1] / "scene_clean"


def main(scene=SCENE):
    path = scene / "visual_surfaces.json"
    surfaces = json.loads(path.read_text())
    tangent = np.array([0.965090900973, 0.261915163477])
    forward = np.array([0.261915163477, -0.965090900973])
    records = {}
    # Classified road segments retain their native paving/marking atlas.
    surfaces["meshes"].pop("mesh/streets.ply", None)
    for filename in ("mesh/ground.ply",):
        mesh = o3d.io.read_triangle_mesh(str(scene / filename))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        indices = np.arange(len(faces))
        points = vertices[faces[indices]].reshape(-1, 3)
        uv = np.column_stack((points[:, :2] @ tangent / 3.2, points[:, :2] @ forward / 2.0))
        surfaces["meshes"][filename] = dict(
            source_sha256=hashlib.sha256((scene / filename).read_bytes()).hexdigest(),
            surfaces=[
                dict(
                    id="showcase_limestone",
                    name="Authored limestone plaza paving",
                    triangle_indices=indices.tolist(),
                    triangle_uvs=uv.tolist(),
                    material=dict(
                        texture_path="textures/paris_limestone_paving.png",
                        color=[1, 1, 1],
                        metallic=0.0,
                        roughness=0.88,
                        reflectance=0.3,
                    ),
                )
            ],
        )
        records[filename] = len(indices)
    path.write_text(json.dumps(surfaces, separators=(",", ":")) + "\n")
    (scene / "paving_provenance.json").write_text(
        json.dumps(
            dict(
                origin=(
                    "Original limestone albedo generated with OpenAI image generation "
                    "for this showcase, 2026-10-23"
                ),
                file="textures/paris_limestone_paving.png",
                tile_extent_m=[3.2, 2.0],
                visual_only=True,
                rf_geometry_changed=False,
                bound_triangles=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
