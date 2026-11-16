"""Make true-scale Paris target meshes visible within a linked street topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import ConnectionPatch, Patch
from mpl_toolkits.mplot3d import proj3d

from generator.figures.mesh_geometry import (
    INK,
    PALETTE,
    RADAR,
    _annotate,
    _background_order,
    _sha,
    _shaded,
    clip_triangles,
    crease_segments,
    tube_faces,
)
from generator.figures.temporal_mesh import (
    ACTOR_COLORS,
    ALPHAS,
    ROOT,
    _actor_layers,
    _axes_style,
    _collection,
    _ghost_overlay,
    load_temporal_inputs,
    save_flat_pdf,
)

SCENARIO = ROOT / "scenarios/sensing/applications/paris_opera_urban_tracking"
STREET_BOUNDS = (-57.0, -41.0, -82.0, -52.0, 0.0, 3.0)


def context_layers(scene_meshes, actors, bounds=None):
    """Return scene and true-scale endpoint faces in one depth-sorted layer.

    Only environmental faces are clipped for the explicitly labeled cutaway.
    Target mesh vertices are never translated, scaled, clipped or simplified here.
    """
    background, opaque, colors = [], [], []
    for name, vertices, faces in scene_meshes:
        triangles = vertices[faces]
        order = _background_order(name, vertices)
        if bounds is not None:
            triangles = clip_triangles(triangles, bounds)
        if not len(triangles):
            continue
        shades = _shaded(triangles, PALETTE.get(name, "#d9ccb4"))
        if order is not None:
            background.append((order, triangles, shades))
            continue
        opaque.append(triangles)
        colors.append(np.column_stack((shades, np.ones(len(triangles)))))
        if name.startswith("buildings"):
            edges = tube_faces(crease_segments(vertices, faces), 0.055 if bounds else 0.075)
            if bounds is not None:
                edges = clip_triangles(edges, bounds)
            if len(edges):
                opaque.append(edges)
                colors.append(np.tile([*to_rgb("#716e63"), 1], (len(edges), 1)))
    for actor in actors:
        faces, rgba = _actor_layers(actor)
        opaque.append(faces)
        colors.append(rgba)
    return background, np.concatenate(opaque), np.concatenate(colors)


def _draw_context(fig, ax, scene_meshes, actors, bounds=None):
    background, faces, colors = context_layers(scene_meshes, actors, bounds)
    for order, triangles, shades in background:
        _collection(ax, triangles, shades, order=order)
    _collection(ax, faces, colors)
    for actor in actors:
        ax.plot(
            *actor["positions"].T,
            color=ACTOR_COLORS.get(actor["name"], "#087b9d"),
            lw=1.3,
            zorder=3,
        )
        _ghost_overlay(fig, ax, actor)


def render_paris_context(output: Path, *, dpi=180):
    config, scene_meshes, actors, radar, indices, times, sources = load_temporal_inputs(SCENARIO)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(18, 10), dpi=dpi, facecolor="white")
    city = fig.add_axes((0.005, 0.30, 0.35, 0.51), projection="3d", computed_zorder=False)
    street = fig.add_axes((0.33, 0.14, 0.65, 0.70), projection="3d", computed_zorder=False)
    _axes_style(
        city, (np.array([-165, -190, 0]), np.array([115, 140, 62])), elev=56, azim=-65, zoom=1.12
    )
    limits = np.asarray(STREET_BOUNDS).reshape(3, 2)
    _axes_style(street, (limits[:, 0], limits[:, 1]), elev=48, azim=-135, zoom=1.08)
    _draw_context(fig, city, scene_meshes, actors)
    _draw_context(fig, street, scene_meshes, actors, STREET_BOUNDS)

    city.plot([radar[0]] * 2, [radar[1]] * 2, [0, radar[2]], color="#22272b", lw=2, zorder=4)
    city.scatter(
        *radar, marker="^", color=RADAR, s=45, edgecolors="white", linewidths=0.6, zorder=5
    )
    _annotate(city, radar, "Tx / Rx", (0.68, 0.29), RADAR, 9)
    xmin, xmax, ymin, ymax, _, _ = STREET_BOUNDS
    footprint = np.array(
        [
            [xmin, ymin, 0.3],
            [xmax, ymin, 0.3],
            [xmax, ymax, 0.3],
            [xmin, ymax, 0.3],
            [xmin, ymin, 0.3],
        ]
    )
    city.plot(*footprint.T, color="#3a7f87", lw=1.7, zorder=5)
    for actor in actors:
        color = ACTOR_COLORS.get(actor["name"], "#087b9d")
        where = (0.025, 0.22) if actor["name"] == "Car" else (0.025, 0.47)
        _annotate(city, actor["positions"][0], actor["name"], where, color, 9)
        where = (0.55, 0.13) if actor["name"] == "Car" else (0.10, 0.81)
        _annotate(
            street,
            actor["positions"][0],
            f"{actor['name']}\n{actor['distance_m']:.2f} m motion",
            where,
            color,
            12,
        )
        for sample, letter in ((0, "S"), (2, "E")):
            label_point = actor["positions"][indices[sample]].copy()
            label_point[2] = actor["meshes"][sample][:, :, 2].max() + 0.4
            street.text(
                *label_point,
                letter,
                color=color,
                fontsize=9,
                ha="center",
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
            )
    city.apply_aspect()
    street.apply_aspect()
    link_point = footprint[2]
    x, y, _ = proj3d.proj_transform(*link_point, city.get_proj())
    connector = ConnectionPatch(
        (x, y),
        (0.06, 0.58),
        "data",
        "axes fraction",
        axesA=city,
        axesB=street,
        color="#81a3a6",
        linewidth=1.1,
        linestyle=(0, (3, 4)),
        zorder=3,
    )
    fig.add_artist(connector)
    # This metric segment uses the same coordinates and projection as the street.
    bar = np.array([[xmin + 1, ymin + 2, 0.2], [xmin + 6, ymin + 2, 0.2]])
    street.plot(*bar.T, color="#53666c", lw=1.5, zorder=4)
    street.text(*(bar.mean(0) + [0, -0.6, 0]), "5 m", fontsize=9, color="#53666c", zorder=5)

    fig.text(0.035, 0.952, "SENSING TOPOLOGY", fontsize=11, fontweight="bold", color="#6a747c")
    fig.text(0.035, 0.908, "Paris Opéra · target meshes in their street context", fontsize=25)
    fig.text(
        0.035,
        0.869,
        "True-scale target meshes · sampled native poses",
        fontsize=12,
        color="#66727b",
    )
    fig.text(0.035, 0.800, "City locator", fontsize=14, fontweight="bold")
    fig.text(0.45, 0.823, "Street topology · linked zoom", fontsize=16, fontweight="bold")
    fig.text(
        0.45,
        0.797,
        "Same positions and scale · surrounding buildings cut away above 3 m",
        fontsize=10,
        color="#66727b",
    )
    fig.text(
        0.035, 0.247, "The outlined footprint is enlarged at right", fontsize=10, color="#66727b"
    )
    fig.text(0.035, 0.209, "Sampled instants", fontsize=12, fontweight="bold")
    time_labels = (
        f"S · Start  {times[0]:.4f} s\nMiddle  {times[1]:.4f} s\nE · End  {times[2]:.4f} s"
    )
    fig.text(0.035, 0.179, time_labels, fontsize=11, color="#66727b", linespacing=1.5, va="top")
    fig.legend(
        handles=[
            Patch(facecolor="#547c8a", label="S: solid start · E: solid end"),
            Patch(facecolor="#547c8a", alpha=ALPHAS[1], label="Middle sample: 24% opacity"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.64, 0.067),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.64,
        0.052,
        "Authored mesh dimensions and trajectory lengths · short motion naturally overlaps the poses",
        ha="center",
        fontsize=9.5,
        color="#66727b",
    )
    fig.text(
        0.5,
        0.025,
        "© OpenStreetMap contributors · ODbL · original scene scale; street view is an explicit cutaway",
        ha="center",
        fontsize=9,
        color="#66727b",
    )
    png, pdf = (
        output / "Paris_Street_Context_Meshes_3D.png",
        output / "Paris_Street_Context_Meshes_3D.pdf",
    )
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    save_flat_pdf(png, pdf, dpi)
    source_files = sources + [
        Path(__file__),
        ROOT / "generator/figures/temporal_mesh.py",
        ROOT / "generator/figures/mesh_geometry.py",
    ]
    receipt = {
        "renderer_sha256": _sha(Path(__file__)),
        "scenario": (
            str(SCENARIO.relative_to(ROOT)) if SCENARIO.is_relative_to(ROOT) else str(SCENARIO)
        ),
        "timeline": config["timeline"],
        "sample_indices": indices,
        "sample_times_s": times,
        "pose_opacity": ALPHAS,
        "street_cutaway_bounds_m": STREET_BOUNDS,
        "city_camera": {"elevation_deg": 56, "azimuth_deg": -65},
        "street_camera": {"elevation_deg": 48, "azimuth_deg": -135},
        "geometry": "Solid endpoint target faces share the depth-sorted building collection in BOTH city and street views. The midpoint is an object-level temporal overlay at the identical projection. Only environment faces are clipped for the labeled street cutaway; target geometry and scale are unchanged.",
        "source_files": {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): _sha(p)
            for p in sorted(set(source_files))
        },
        "actors": {
            a["name"]: {
                "positions_m": [a["positions"][i].tolist() for i in indices],
                "orientation_deg": a["orientation_deg"],
                "mesh_extent_m": a["extent_m"].tolist(),
                "extent_definition": a["extent_definition"],
                "animated": a["animated"],
                **(
                    {"root_affines": a["root_affines"], "animated_samples": a["animated_samples"]}
                    if a["animated"]
                    else {}
                ),
                "authored_asset_scale": a["scale"],
                "display_mesh_scale_multiplier": 1.0,
                "source_triangles": len(a["faces"]),
                "solid_endpoint_triangles_per_view": 2 * len(a["faces"]),
                "midpoint_triangles_per_view": len(a["faces"]),
                "actual_asset": (
                    str(a["actual_asset"].relative_to(ROOT))
                    if a["actual_asset"].is_relative_to(ROOT)
                    else str(a["actual_asset"])
                ),
                "requested_asset": (
                    str(a["requested_asset"].relative_to(ROOT))
                    if a["requested_asset"].is_relative_to(ROOT)
                    else str(a["requested_asset"])
                ),
                "asset_substitution": a["actual_asset"] != a["requested_asset"],
            }
            for a in actors
        },
        "qualification": "Rigid meshes use native world-AABB placement; animated meshes retain authenticated manifest-owned world geometry. Asset substitutions are recorded individually. Temporal overlays depict sampled poses. PDF embeds the PNG.",
        "outputs": {p.name: _sha(p) for p in (png, pdf)},
    }
    (output / "Paris_Street_Context_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    render_paris_context(args.output.resolve(), dpi=args.dpi)
    print(args.output / "Paris_Street_Context_Meshes_3D.png")  # noqa: T201 - CLI artifact path


if __name__ == "__main__":
    main()
