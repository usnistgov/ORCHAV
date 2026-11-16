"""Bind the six-tile survey-drone atlas to the original ORCHAV large UAV.

Pass the matching mesh explicitly in an installed ORCHAV environment. This writes only
surface_uv.npz beside this script; geometry and recordings remain untouched.
The atlas is a 3-by-2 grid: shell/carbon/metal, then battery/lens/rubber.
"""

import argparse
from pathlib import Path

import numpy as np

from visualizer.src.scene.geometry_payload_factory import load_mesh_payload


def build(path: Path) -> None:
    mesh = load_mesh_payload(path)
    groups = []
    triangles = []
    group = ""
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "g":
            group = fields[1]
        elif fields and fields[0] == "f":
            triangles.append([int(value.split("/")[0]) - 1 for value in fields[1:]])
            groups.append(group)
    if not np.array_equal(triangles, mesh.triangles):
        raise ValueError("Expected the original ordered triangular UAV topology")
    groups = np.asarray(groups)
    uvs = np.empty((len(triangles), 3, 2), dtype=np.float32)
    for name in dict.fromkeys(groups):
        selected = np.flatnonzero(groups == name)
        points = mesh.vertices[mesh.triangles[selected]].astype(float)
        center = (points.min(axis=(0, 1)) + points.max(axis=(0, 1))) / 2
        local = points - center
        # Keep the weave aligned with each arm or blade, rather than world XY.
        yaw = 0.0
        if name.startswith("arm_"):
            yaw = np.arctan2(center[1], center[0])
        elif name.startswith("rotor_") and int(name.rsplit("_", 1)[1]) in (1, 2):
            yaw = np.pi / 2
        c, s = np.cos(yaw), np.sin(yaw)
        local = local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        span = np.maximum(np.ptp(local, axis=(0, 1)), 1e-9)
        normalized = local / span + 0.5
        normals = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
        for row, face in enumerate(selected):
            normal = normals[row]
            dominant = int(np.argmax(np.abs(normal)))
            axes = ((1, 2), (0, 2), (0, 1))[dominant]
            uv = normalized[row][:, axes].copy()
            uv[:, 1] = 1 - uv[:, 1]
            if name == "fuselage":
                tile = (0, 0)
            elif name == "battery":
                tile = (0, 1) if dominant == 2 else (2, 1)
            elif name == "camera" and normal[0] > 0:
                tile = (1, 1)
                uv = np.column_stack((1 - normalized[row, :, 1], 1 - normalized[row, :, 2]))
            elif name == "camera" or name.startswith("foot_"):
                tile = (2, 1)
            elif name.startswith(("motor_", "shaft_")):
                tile = (2, 0)
            else:
                tile = (1, 0)
            if tile not in {(0, 1), (1, 1)}:
                # Material grain should have the same scale along both axes,
                # including on long, narrow arms and landing struts.
                uv = local[row][:, axes] / max(span) + 0.5
                uv[:, 1] = 1 - uv[:, 1]
            # Inset the sampled rectangle to avoid filtering across tile boundaries.
            uvs[face] = (np.asarray(tile) + 0.02 + uv * 0.96) / [3, 2]
    destination = Path(__file__).with_name("surface_uv.npz")
    np.savez_compressed(
        destination,
        vertex_count=np.array(len(mesh.vertices), dtype=np.int64),
        triangles=mesh.triangles,
        triangle_uvs=uvs.reshape(-1, 2),
    )
    print(f"Wrote {destination}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    build(parser.parse_args().mesh.resolve())
