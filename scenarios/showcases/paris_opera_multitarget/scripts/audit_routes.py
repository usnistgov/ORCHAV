"""Check the authored routes against the static surface-city geometry."""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prepare import RADAR, ROUTES, WALK_START, WALK_V


def main(directory):
    scene = o3d.t.geometry.RaycastingScene()
    meshes = []
    for path in sorted((directory / "scene_clean/mesh").glob("buildings*.ply")):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            continue
        meshes.append(mesh)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    design = json.loads((directory / "design.json").read_text())
    times = np.linspace(0, design["last_sample_time_s"], 257) + design["route_offset_s"]
    routes = {
        name: np.asarray(start) + times[:, None] * velocity
        for name, (start, velocity, asset) in ROUTES.items()
    }
    routes["Pedestrian"] = (
        WALK_START + times[:, None] * WALK_V + design["walking_offset_m"] + [0, 0, 0.85]
    )
    result = dict(
        samples_per_route=len(times),
        duration_s=float(times[-1]),
        scope="Static building clearance and centre-ray visibility; pedestrian uses a root+0.85m proxy, excludes inter-target occlusion",
        targets={},
    )
    for name, points in routes.items():
        delta = points - RADAR
        ranges = np.linalg.norm(delta, axis=1)
        rays = np.column_stack(
            (np.broadcast_to(RADAR, points.shape), delta / ranges[:, None])
        ).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        distance = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
        result["targets"][name] = dict(
            static_los_blocked_samples=int(np.count_nonzero(hits < ranges - 0.02)),
            minimum_centre_to_building_surface_m=float(distance.min()),
            range_interval_m=[float(ranges.min()), float(ranges.max())],
            start_m=points[0].tolist(),
            end_m=points[-1].tolist(),
        )
    evidence = directory / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "route_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    import matplotlib.pyplot as plt
    from analyze import COLORS
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    fig.patch.set_facecolor("#0c1727")
    ax.set_facecolor("#111f32")
    for mesh in meshes:
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        ax.add_collection(
            PolyCollection(vertices[faces, :2], facecolors="#4c6075", edgecolors="none", alpha=0.75)
        )
    streets = o3d.io.read_triangle_mesh(str(directory / "scene_clean/mesh/streets.ply"))
    ax.add_collection(
        PolyCollection(
            np.asarray(streets.vertices)[np.asarray(streets.triangles), :2],
            facecolors="#28394e",
            edgecolors="none",
            zorder=0,
        )
    )
    for name, points in routes.items():
        color = COLORS[name]
        ax.plot(points[:, 0], points[:, 1], color=color, lw=3, label=name.replace("_", " "))
        ax.scatter(*points[0, :2], c=color, s=50, marker="o", zorder=5)
        ax.annotate(
            "",
            xy=points[-1, :2],
            xytext=points[-12, :2],
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
        )
    ax.scatter(*RADAR[:2], s=120, marker="^", c="#f0f5f9", label="Monostatic radar", zorder=6)
    ax.annotate("PALAIS GARNIER", (-37, -30), color="#eaf1f9", ha="center", weight="bold")
    ax.set(
        xlim=(-90, 10),
        ylim=(-108, -24),
        aspect="equal",
        xlabel="Local east (m)",
        ylabel="Local north (m)",
    )
    ax.tick_params(colors="#b7c9dd")
    ax.xaxis.label.set_color("#b7c9dd")
    ax.yaxis.label.set_color("#b7c9dd")
    for spine in ax.spines.values():
        spine.set_color("#435872")
    ax.legend(loc="lower left", facecolor="#111f32", edgecolor="#435872", labelcolor="#eaf1f9")
    ax.set_title("PARIS OPÉRA / 5.12 s MESH-SENSING SCENE", color="#eaf1f9", pad=16)
    fig.savefig(evidence / "routes.png", dpi=180)
    fig.savefig(evidence / "routes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    main(parser.parse_args().scenario.resolve())
