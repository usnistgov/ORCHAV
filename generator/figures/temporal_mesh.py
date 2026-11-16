"""Show authentic target meshes at sampled start, middle and end poses.

Example: python -m generator.figures.temporal_mesh --scenario
scenarios/sensing/applications/paris_opera_urban_tracking --output output/mesh-poses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generator.core.scenario_actors.preparation import prepare_scenario
from generator.core.scenario_actors.quaternion import Quaternion
from generator.figures.mesh_geometry import (
    INK,
    PALETTE,
    RADAR,
    _annotate,
    _background_order,
    _read_mesh,
    _sha,
    _shaded,
    crease_segments,
    tube_faces,
)
from generator.figures.topology_scene import scene_file
from shared.scenarios.parsers import parse_scene_config
from shared.scenarios.paths import PathPolicy, resolve_actor_resource

ROOT = Path(__file__).resolve().parents[2]
ACTOR_COLORS = {
    "Car": "#087b9d",
    "Pedestrian": "#d56335",
    "Human": "#d56335",
    "NearTarget": "#087b9d",
    "FarTarget": "#7852a1",
}
ALPHAS = (1.0, 0.24, 1.0)
STAGES = ("Start", "Middle sample", "End")
TITLES = {
    "paris_opera_urban_tracking": "Paris Opéra · car and pedestrian",
    "multi_target_tracking": "Two-target tracking · empty scene",
    "human_target_tracking": "Rigid human tracking · empty scene",
}


def quaternion_matrix(q):
    """Convert the native normalized w/x/y/z pose to a world rotation."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_temporal_inputs(scenario: Path):
    config_path = scenario / "scenario.yaml"
    config = yaml.safe_load(config_path.read_text())
    project_root = PathPolicy(config_path).project_root
    timeline = config["timeline"]
    if "sample_rate_hz" in timeline:
        duration = (timeline["steps"] - 1) / timeline["sample_rate_hz"]
        if "duration_s" in timeline and not np.isclose(duration, timeline["duration_s"]):
            raise ValueError("Conflicting scenario timeline")
        timeline["duration_s"] = duration
    indices = tuple(int(i) for i in (0, timeline["steps"] // 2, timeline["steps"] - 1))
    if len(set(indices)) != 3:
        raise ValueError("Three sampled poses require at least three timeline samples")
    sources = [
        config_path,
        Path(__file__).with_name("topology_scene.py"),
        ROOT / "shared/scenarios/parsers.py",
        ROOT / "shared/scenarios/paths.py",
    ] + [
        ROOT / "generator/core/scenario_actors" / f"{name}.py"
        for name in (
            "preparation",
            "mobility",
            "resources",
            "orientation",
            "quaternion",
            "types",
            "_adapters",
        )
    ]
    assets, alignments = {}, {}
    for spec in config["actors"]["targets"]:
        asset = spec["asset"]
        if asset["source"] == "animated_asset":
            continue
        if asset.get("switch_meshes", False) or asset["source"] != "directory":
            raise ValueError(
                "Temporal figures require a rigid directory asset or a native animated_asset"
            )
        directory = resolve_actor_resource(
            asset["path"], scenario_root=scenario, project_root=project_root
        )
        requested = directory / asset["pattern"]
        actual = requested
        if not actual.exists() and actual.with_suffix(".obj").exists():
            actual = actual.with_suffix(".obj")
        vertices, faces = _read_mesh(actual)
        scale = float(asset.get("scale", 1.0))
        centered = (vertices - (vertices.min(0) + vertices.max(0)) / 2) * scale
        metadata_path = directory / "target_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        offset = float(metadata.get("front", {}).get("yaw_offset_deg", 0))
        alignments[spec["name"]] = Quaternion.from_euler_deg(offset)
        assets[spec["name"]] = {
            "vertices": centered,
            "faces": faces,
            "scale": scale,
            "extent_m": np.ptp(centered, axis=0),
            "extent_definition": "rigid_asset_local",
            "animated": False,
            "requested_asset": requested,
            "actual_asset": actual,
        }
        sources.append(actual)
        if metadata_path.exists():
            sources.append(metadata_path)
        if spec["mobility"].get("graph_path"):
            path = Path(spec["mobility"]["graph_path"])
            sources.append(path if path.is_absolute() else scenario / path)
    animations = {}
    prepared = prepare_scenario(
        dict(config, project_root=project_root),
        base_dir=scenario,
        asset_alignments=alignments,
        _animated_assets=animations,
    )
    actors = []
    for pose in prepared.actors:
        if pose.role != "target":
            continue
        animation = animations.get(pose.name)
        if animation is not None:
            # Bound world vertices already contain articulation and the manifest
            # root transform. Applying the actor pose again would double motion.
            samples = [animation.binding.samples[i] for i in indices]
            meshes = [sample.vertices_m[animation.binding.faces] for sample in samples]
            positions = np.asarray(
                [sample.resolved_root_affine_4x4[:3, 3] for sample in animation.binding.samples]
            )
            manifest = animation.asset.manifest_path
            sources.append(manifest)
            sources.extend(
                manifest.parent / resource.path for resource in animation.asset.resources
            )
            sources.append(ROOT / "generator/core/scenario_actors/animated_asset.py")
            asset = {
                "scale": 1.0,
                "faces": animation.binding.faces,
                "extent_m": np.ptp(samples[0].vertices_m, axis=0),
                "extent_definition": "initial_world_aabb",
                "animated": True,
                "requested_asset": manifest,
                "actual_asset": manifest,
                "animated_samples": [sample.evidence_dict() for sample in samples],
                "root_affines": [sample.resolved_root_affine_4x4.tolist() for sample in samples],
            }
            euler = None  # The manifest's full affine is the placement authority.
        else:
            asset = assets[pose.name]
            positions = np.asarray(pose.positions_m)
            meshes = []
            for index in indices:
                rotation = quaternion_matrix(pose.orientation.quaternions[index])
                rotated = asset["vertices"] @ rotation.T
                world_vertices = rotated - (rotated.min(0) + rotated.max(0)) / 2 + positions[index]
                meshes.append(world_vertices[asset["faces"]])
            euler = [pose.orientation.euler_deg[i] for i in indices]
        delta = positions[-1] - positions[0]
        heading = float(np.degrees(np.arctan2(delta[1], delta[0])))
        actors.append(
            {
                **asset,
                "name": pose.name,
                "positions": positions,
                "meshes": meshes,
                "orientation_deg": euler,
                "heading_deg": heading,
                "distance_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            }
        )
    radar = np.asarray(next(p.positions_m[0] for p in prepared.actors if p.role == "tx"))
    _, _, scene_path = parse_scene_config(config, scenario, project_root)
    if scene_path is None:
        raise FileNotFoundError("Scene XML is unavailable; prepare the authored scene first")
    scene, scene_sources = scene_file(scene_path)
    sources.extend(scene_sources)
    meshes = [(entry["name"], entry["vertices"], entry["faces"]) for entry in scene]
    times = [prepared.timeline.timestamps_s[i] for i in indices]
    return config, meshes, actors, radar, indices, times, sorted(set(sources))


def _collection(ax, triangles, colors, *, order=2):
    return ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolors=colors,
            edgecolors="none",
            linewidths=0,
            antialiased=False,
            rasterized=True,
            zorder=order,
        )
    )


def _actor_layers(actor):
    color = ACTOR_COLORS.get(actor["name"], "#087b9d")
    faces, colors = [], []
    for index in (0, 2):
        mesh, alpha = actor["meshes"][index], 1.0
        faces.append(mesh)
        colors.append(
            np.column_stack((_shaded(mesh, color, strength=0.35), np.full(len(mesh), alpha)))
        )
    return np.concatenate(faces), np.concatenate(colors)


def _ghost_overlay(fig, ax, actor):
    """Apply transparency once to the complete midpoint, not to each triangle.

    An isolated opaque render uses the identical projection and pixel viewport.
    Its RGBA silhouette is then composited at uniform object-level opacity. This
    prevents dense reconstructed meshes from accumulating into an opaque ghost.
    These are temporal overlays, not a physical scene of concurrent actors.
    """
    ax.apply_aspect()
    ghost = plt.figure(figsize=fig.get_size_inches(), dpi=fig.dpi, facecolor=(0, 0, 0, 0))
    ghost_ax = ghost.add_axes(ax.get_position().bounds, projection="3d", computed_zorder=False)
    ghost_ax.set_axis_off()
    ghost_ax.patch.set_alpha(0)
    ghost_ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim(), zlim=ax.get_zlim())
    ghost_ax.get_proj = ax.get_proj
    mesh = actor["meshes"][1]
    color = ACTOR_COLORS.get(actor["name"], "#087b9d")
    _collection(ghost_ax, mesh, _shaded(mesh, color, strength=0.35))
    canvas = FigureCanvasAgg(ghost)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(ghost)
    fig.figimage(rgba, origin="upper", xo=0, yo=0, alpha=ALPHAS[1], zorder=1)


def _axes_style(ax, bounds, *, show_axes=False, elev=25, azim=-65, zoom=1):
    low, high = bounds
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low, zoom=zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    if show_axes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"].update({"color": (0.70, 0.74, 0.77, 0.22), "linewidth": 0.5})
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax.tick_params(labelsize=8, pad=0)
        if (high[2] - low[2]) / max(high[0] - low[0], high[1] - low[1]) < 0.12:
            ax.set_zticks([(low[2] + high[2]) / 2])
    else:
        ax.set_axis_off()


def save_flat_pdf(png: Path, pdf: Path, dpi: int):
    """Preserve uniform object opacity identically in raster and PDF backends."""
    pixels = plt.imread(png)[..., :3]
    page = plt.figure(figsize=(pixels.shape[1] / dpi, pixels.shape[0] / dpi), dpi=dpi)
    axes = page.add_axes((0, 0, 1, 1))
    axes.imshow(pixels, interpolation="none")
    axes.set_axis_off()
    page.savefig(pdf, dpi=dpi)
    plt.close(page)


def render_temporal(scenario: Path, output: Path, *, dpi=200):
    config, scene_meshes, actors, radar, indices, times, sources = load_temporal_inputs(scenario)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(16, 10), dpi=dpi, facecolor="white")
    main = fig.add_axes((0.015, 0.17, 0.60, 0.69), projection="3d", computed_zorder=False)
    layers = []
    colors = []
    for name, vertices, faces in scene_meshes:
        background = _background_order(name, vertices)
        triangles = vertices[faces]
        if background is not None:
            _collection(main, triangles, _shaded(triangles, PALETTE[name]), order=background)
            continue
        layers.append(triangles)
        colors.append(
            np.column_stack(
                (_shaded(triangles, PALETTE.get(name, "#d9ccb4")), np.ones(len(triangles)))
            )
        )
        if name.startswith("buildings"):
            edge_faces = tube_faces(crease_segments(vertices, faces), 0.075)
            layers.append(edge_faces)
            colors.append(np.tile([*to_rgb("#716e63"), 1], (len(edge_faces), 1)))
    for actor in actors:
        faces, rgba = _actor_layers(actor)
        layers.append(faces)
        colors.append(rgba)
    _collection(main, np.concatenate(layers), np.concatenate(colors))
    for actor in actors:
        p = actor["positions"]
        color = ACTOR_COLORS.get(actor["name"], "#087b9d")
        main.plot(*p.T, color=color, lw=1.7, zorder=3)
    main.plot([radar[0]] * 2, [radar[1]] * 2, [0, radar[2]], color="#22272b", lw=2.5, zorder=4)
    main.scatter(
        *radar, marker="^", color=RADAR, s=75, edgecolors="white", linewidths=0.8, zorder=5
    )
    if scene_meshes:
        _axes_style(
            main,
            (np.array([-165, -190, 0]), np.array([115, 140, 62])),
            elev=56,
            azim=-65,
            zoom=1.15,
        )
        _annotate(main, radar, "Tx / Rx", (0.54, 0.35), RADAR, 10)
    else:
        points = np.concatenate(
            [np.concatenate(a["meshes"]).reshape(-1, 3) for a in actors] + [radar[None]]
        )
        low, high = points.min(0), points.max(0)
        span = np.maximum(high - low, [1, 1, 1])
        low -= span * 0.10
        high += span * 0.10
        _axes_style(main, (low, high), show_axes=True, elev=28, azim=-60, zoom=0.95)
        _annotate(main, radar, "Tx / Rx", (0.05, 0.25), RADAR, 10)
        main.text2D(
            0.05,
            0.92,
            "Empty source scene\nAxes are a coordinate reference",
            transform=main.transAxes,
            fontsize=10,
            color="#66727b",
        )
    for actor in actors:
        _ghost_overlay(fig, main, actor)
    n = len(actors)
    for number, actor in enumerate(actors):
        height = 0.30 if n > 1 else 0.55
        bottom = 0.53 - number * 0.38 if n > 1 else 0.25
        ax = fig.add_axes((0.66, bottom, 0.32, height), projection="3d", computed_zorder=False)
        faces, rgba = _actor_layers(actor)
        _collection(ax, faces, rgba)
        points = np.concatenate(actor["meshes"]).reshape(-1, 3)
        low, high = points.min(0), points.max(0)
        span = np.maximum(high - low, [0.3, 0.3, 0.3])
        low -= span * 0.14
        high += span * 0.14
        _axes_style(ax, (low, high), elev=24, azim=actor["heading_deg"] - 65, zoom=1.0)
        _ghost_overlay(fig, ax, actor)
        for i, (mesh, t, label) in enumerate(zip(actor["meshes"], times, STAGES)):
            position = actor["positions"][indices[i]].copy()
            position[2] = mesh[:, :, 2].min()
            _annotate(ax, position, f"{label}\n{t:.4f} s", (0.06 + 0.35 * i, 0.025), INK, 8.5)
        extent = actor["extent_m"]
        fig.text(
            0.675,
            bottom + height + 0.018,
            f"{actor['name']} · {actor['distance_m']:.2f} m translation",
            fontsize=14,
            color=ACTOR_COLORS.get(actor["name"], INK),
        )
        fig.text(
            0.675,
            bottom + height - 0.010,
            f"Mesh {extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} m · "
            + ("initial world bounds" if actor["animated"] else "authored scale"),
            fontsize=9,
            color="#66727b",
        )
        # A metric bar in the same physical coordinates makes the independent
        # zoom explicit, without enlarging a mesh or stretching the trajectory.
        bar_length = 1.0 if span.max() > 2 else 0.25
        origin = low + span * np.array([0.1, 0.1, 0])
        if actor["name"] in ("NearTarget", "FarTarget"):
            origin[2] = high[2] + 0.10 * span.max()
        end = origin + np.array([bar_length, 0, 0])
        ax.plot(*np.stack((origin, end)).T, c="#606a70", lw=1.4, zorder=4)
        ax.text(*((origin + end) / 2), f" {bar_length:g} m", color="#606a70", fontsize=8, zorder=5)
    fig.text(0.045, 0.95, "SENSING SCENARIOS", fontsize=11, fontweight="bold", color="#6a747c")
    fig.text(0.045, 0.91, TITLES.get(scenario.name, scenario.name.replace("_", " ")), fontsize=25)
    fig.text(
        0.045,
        0.873,
        "Actual target meshes · solid start and end · translucent middle sample",
        fontsize=12,
        color="#66727b",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#547c8a", label="Start: solid"),
            Patch(facecolor="#547c8a", alpha=0.24, label="Middle sample: 24% opacity"),
            Patch(facecolor="#547c8a", label="End: solid"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.067),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=2.5,
    )
    fig.text(
        0.5,
        0.052,
        "Close-ups use independent camera zooms. Mesh dimensions, positions and sampled times are unchanged.",
        ha="center",
        fontsize=9.5,
        color="#66727b",
    )
    footer = (
        "© OpenStreetMap contributors · ODbL · scene heights without exaggeration"
        if scene_meshes
        else "Target motion · authored scenario geometry"
    )
    fig.text(0.5, 0.026, footer, ha="center", fontsize=9, color="#66727b")
    names = ("temporal_meshes_3d.png", "temporal_meshes_3d.pdf")
    fig.savefig(output / names[0], dpi=dpi)
    plt.close(fig)
    save_flat_pdf(output / names[0], output / names[1], dpi)
    receipt = {
        "scenario": str(scenario),
        "scenario_name": scenario.name,
        "timeline": config["timeline"],
        "sample_indices": indices,
        "sample_times_s": times,
        "pose_opacity": ALPHAS,
        "pdf_representation": "Flattened validated PNG, preserving the same effective object opacity as PNG.",
        "centering": "Rigid assets use the native world-AABB center; authenticated animated geometry keeps the manifest-owned world transform without recentering.",
        "opacity_compositing": "Uniform object-level RGBA midpoint overlay at identical projection, composited above opaque endpoint meshes; not per-triangle opacity.",
        "scope": "Native rigid poses or authenticated animated world geometry; no sensing recomputation; independent camera zoom only.",
        "source_files": {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): _sha(p) for p in sources
        },
        "renderer_sha256": _sha(Path(__file__)),
        "helper_sha256": _sha(ROOT / "generator/figures/mesh_geometry.py"),
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
                "requested_asset": str(a["requested_asset"]),
                "actual_asset": str(a["actual_asset"]),
                "asset_substitution": a["requested_asset"] != a["actual_asset"],
                "distance_m": a["distance_m"],
            }
            for a in actors
        },
        "outputs": {name: _sha(output / name) for name in names},
        "qualification": "When PLY is unavailable, the same-named upstream OBJ is explicitly recorded; geometry equivalence to PLY is not independently asserted. Transparent surfaces are illustrative temporal overlays, not concurrent physical targets.",
    }
    (output / "temporal_mesh_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    render_temporal(args.scenario.resolve(), args.output.resolve(), dpi=args.dpi)
    print(args.output / "temporal_meshes_3d.png")  # noqa: T201 - CLI artifact path


if __name__ == "__main__":
    main()
