"""Render an orthographic scenario overview without acquiring simulation frames.

Run ``python -m generator.figures.topology --help``. YAML-defined nodes and
directory and native animated targets use the generator's own pose preparation.
Unsupported or missing geometry raises an error instead of silently vanishing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generator.core.scenario_actors.preparation import prepare_scenario
from generator.core.scenario_actors.quaternion import Quaternion
from generator.core.target.mesh import (
    mesh_call_count_for_step,
    mesh_sequence_index,
    mesh_update_step_interval,
)
from shared.logging import get_logger
from shared.node_labels import monostatic_label_pairs, sensing_role_label
from shared.scenarios.loader import load_scenario
from shared.scenarios.parsers import parse_scene_config
from shared.scenarios.paths import PathPolicy, resolve_actor_resource

from .mesh_geometry import _read_mesh, _sha, _shaded, crease_segments, tube_faces
from .topology_scene import scene_file

COLORS = ("#007e9e", "#de7144", "#6b53a1", "#319177", "#b58331", "#bc467b")
INK, MUTED = "#263940", "#687b83"
VIEW_FRAME_STYLE = dict(facecolor="#f3f7fb", edgecolor="#b5c8d9", linewidth=1.8)
AXIS_COLORS = {"X": "#b85b61", "Y": "#398771", "Z": "#527cbb"}


def _panel_frame(fig, bounds):
    """Draw a softly rounded panel with corners sized in physical inches."""
    left, bottom, width, height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={0.10 / fig.get_figwidth()}",
            mutation_aspect=fig.get_figwidth() / fig.get_figheight(),
            transform=fig.transFigure,
            zorder=-2,
            **VIEW_FRAME_STYLE,
        )
    )


def rotation_matrix(q: Quaternion) -> np.ndarray:
    """World rotation for the generator's normalized w/x/y/z quaternion."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_indices(rotations: np.ndarray) -> list[int]:
    """Use actual start/middle/end samples only when any rotation changes.

    Compare the full series, not just its endpoints: a complete turn can end at
    the initial orientation. Rotation matrices also avoid quaternion sign flips.
    """
    if np.allclose(rotations, rotations[0], atol=1e-8, rtol=0):
        return [0]
    return list(dict.fromkeys((0, len(rotations) // 2, len(rotations) - 1)))


def is_stationary(positions: np.ndarray) -> bool:
    """Test the complete sampled path; a closed loop is still movement."""
    return bool(np.allclose(positions, positions[0], atol=1e-8, rtol=0))


def _node_marker(actor: dict) -> str:
    if actor["role"] == "radar" and actor["stationary"]:
        return "^"
    if actor["role"] != "target" and actor["stationary"]:
        return "D"
    return "o"


def _pose_sample_count(value: int) -> int:
    """Validate the display-only count for typed YAML and standalone callers."""
    if type(value) is not int or value not in (3, 5):
        raise ValueError("topology_pose_samples must be the integer 3 or 5")
    return value


def _pose_sample_indices(steps: int, count: int) -> list[int]:
    """Select authored sample indices, deduplicating short timelines."""
    count = _pose_sample_count(count)
    if count == 5:
        return sorted({0, steps // 4, steps // 2, 3 * steps // 4, steps - 1})
    return sorted({0, steps // 2, steps - 1})


def load_topology(scenario: Path, *, pose_samples: int | None = None) -> dict:
    """Load native geometry and sample start/end poses, with source hashes."""
    config_path = scenario / "scenario.yaml" if scenario.is_dir() else scenario
    config_path = config_path.resolve()
    root = config_path.parent
    cfg = load_scenario(config_path).model_dump(mode="json", exclude_none=True)
    if pose_samples is not None:
        cfg.setdefault("generator_summary", {}).setdefault("visualization", {})[
            "topology_pose_samples"
        ] = _pose_sample_count(pose_samples)
    policy = PathPolicy(config_path)
    scene_cfg = cfg
    if cfg["scene"].get("source") not in (None, "library", "local", "sionna"):
        cached = root / cfg["scene"].get("id", "scene.xml")
        if not cached.is_file():
            raise FileNotFoundError("Scene XML is unavailable; prepare the authored scene first")
        scene_cfg = dict(cfg, scene=dict(source="local", id=str(cached)))
    _, _, xml = parse_scene_config(scene_cfg, root, policy.project_root)
    if xml is None or not xml.is_file():
        raise FileNotFoundError("Scene XML is unavailable; prepare the authored scene first")
    scene, sources = scene_file(xml)
    sources.append(config_path)
    assets, alignments = {}, {}
    for spec in cfg["actors"].get("targets", []):
        if spec.get("representation") in {"constant_rcs", "tr38901"}:
            from generator.core.target.representation import NativeTargetConfig
            from shared.scenarios.actors import NativeMeshGeometrySpec, TargetActorSpec

            from .topology_targets import native_proxy_geometry

            native = TargetActorSpec.model_validate(spec)
            mesh_path = None
            if isinstance(native.geometry, NativeMeshGeometrySpec):
                mesh_path = resolve_actor_resource(
                    native.geometry.path, scenario_root=root, project_root=policy.project_root
                )
            config = NativeTargetConfig(
                name=native.name,
                representation=native.representation,
                geometry=native.geometry,
                constant_rcs=native.constant_rcs,
                tr38901=native.tr38901,
                resolved_mesh_path=mesh_path,
            )
            metadata = config.preview_metadata()
            vertices, faces, source = native_proxy_geometry(config, metadata, config_path)
            assets[native.name] = (
                dict(
                    scale=1.0,
                    switch_meshes=False,
                    start_index=0,
                    frame_stride=1,
                    mesh_end_behavior="hold_last",
                    geometry=(vertices, faces),
                    representation=config.representation,
                    proxy_geometry=metadata["geometry"],
                ),
                [source],
            )
            sources.extend(
                [
                    Path(__file__).with_name("topology_targets.py"),
                    Path(__file__).resolve().parents[1] / "core/target/representation.py",
                ]
            )
            continue
        asset = spec.get("asset", {})
        if asset.get("source") == "animated_asset":
            continue  # Authenticated binding is retained by prepare_scenario below.
        if asset.get("source") != "directory":
            raise ValueError("Topology requires a directory or native animated target")
        directory = resolve_actor_resource(
            asset["path"], scenario_root=root, project_root=policy.project_root
        )
        matches = sorted(directory.glob(asset["pattern"]))
        if not matches:
            raise FileNotFoundError(f"No target mesh matches {directory / asset['pattern']}")
        assets[spec["name"]] = (asset, matches)
        metadata = directory / "target_metadata.json"
        if metadata.is_file():
            offset = json.loads(metadata.read_text()).get("front", {}).get("yaw_offset_deg", 0)
            alignments[spec["name"]] = Quaternion.from_euler_deg(offset)
            sources.append(metadata)
    for specs in cfg["actors"].values():
        for spec in specs:
            for key in ("graph_path", "positions_path"):
                resource = spec.get("mobility", {}).get(key)
                if resource:
                    sources.append(
                        resolve_actor_resource(
                            resource, scenario_root=root, project_root=policy.project_root
                        )
                    )
    animations = {}
    prepared = prepare_scenario(
        cfg, base_dir=root, asset_alignments=alignments, _animated_assets=animations
    )
    for name, animation in animations.items():
        assets[name] = _animated_geometry(animation, sources)
    poses = [
        dict(
            name=p.name,
            role=p.role,
            positions=np.asarray(p.positions_m),
            rotations=np.array([rotation_matrix(q) for q in p.orientation.quaternions]),
            physical_path=p.mobility.has_physical_velocity,
        )
        for p in prepared.actors
    ]
    return _build_case(
        cfg,
        root,
        config_path,
        policy.project_root,
        xml,
        scene,
        sources,
        poses,
        prepared.timeline,
        assets,
        alignments,
    )


def _animated_geometry(animation, sources):
    """Use authenticated world vertices without centering or another transform."""
    asset = animation.asset
    resources = {r.resource_id: asset.manifest_path.parent / r.path for r in asset.resources}
    paths = [resources[s.resource_id] for s in asset.pose_samples]
    sources.append(asset.manifest_path)
    sources.extend(resources.values())
    return dict(animation=animation, switch_meshes=len(paths) > 1), paths


def load_runtime_topology(
    tx_configs,
    rx_configs,
    target_managers,
    simulation_config,
    scenario_context,
    actor_state_manager,
) -> dict:
    """Use the actual prepared runtime, including scripted actors and overrides."""
    from generator.core.scenario_actors.types import Timeline

    context = scenario_context
    xml = Path(context.scene_xml)
    scene, sources = scene_file(xml)
    root = Path(context.root)
    # The resolved YAML hash below identifies alternate YAMLs; runtime samples
    # identify Python overrides independently of the directory's default YAML.
    config_path = root / "scenario.yaml"
    if config_path.is_file():
        sources.append(config_path)
    driver = root / "generate.py"
    if driver.is_file():
        sources.append(driver)
    timeline = Timeline(simulation_config.num_steps, simulation_config.duration)
    cache = actor_state_manager.prepare_cached()
    poses, assets, alignments = [], {}, {}
    for role, configs in (
        ("tx", tx_configs),
        ("rx", rx_configs),
        ("target", [m.config for m in target_managers]),
    ):
        for index, config in enumerate(configs):
            positions = np.asarray(getattr(cache, role + "_positions")[index])
            angles = getattr(cache, role + "_orientations")[index]
            if not angles:
                angles = [(0, 0, 0)] * timeline.steps
            poses.append(
                dict(
                    name=config.name,
                    role=role,
                    positions=positions,
                    rotations=np.array(
                        [rotation_matrix(Quaternion.from_euler_deg(*a)) for a in angles]
                    ),
                    physical_path=bool(getattr(config.mobility, "has_physical_motion", True)),
                )
            )
            if role == "target":
                animation = getattr(config, "animated_asset", None)
                if animation is not None:
                    assets[config.name] = _animated_geometry(animation, sources)
                    continue
                if hasattr(config, "preview_metadata"):
                    from .topology_targets import native_proxy_geometry

                    metadata = config.preview_metadata()
                    vertices, faces, source = native_proxy_geometry(config, metadata, config_path)
                    assets[config.name] = (
                        dict(
                            scale=1.0,
                            switch_meshes=False,
                            start_index=0,
                            frame_stride=1,
                            mesh_end_behavior="hold_last",
                            geometry=(vertices, faces),
                            representation=config.representation,
                            proxy_geometry=metadata["geometry"],
                        ),
                        [source],
                    )
                    sources.append(Path(__file__).with_name("topology_targets.py"))
                    sources.append(
                        Path(__file__).resolve().parents[1] / "core/target/representation.py"
                    )
                    continue
                if not hasattr(config, "resolved_mesh_directory"):
                    raise ValueError("Topology requires a renderable target mesh")
                directory = Path(config.resolved_mesh_directory)
                paths = sorted(directory.glob(config.mesh_pattern))
                if not paths:
                    raise FileNotFoundError(f"No target meshes in {directory}")
                assets[config.name] = (
                    dict(
                        scale=config.scale,
                        switch_meshes=config.switch_meshes,
                        start_index=config.mesh_start_index,
                        frame_stride=config.mesh_frame_stride,
                        mesh_end_behavior=config.mesh_end_behavior,
                    ),
                    paths,
                )
                alignments[config.name] = Quaternion.from_euler_deg(
                    config.asset_front_yaw_offset_deg or 0
                )
                metadata = directory / "target_metadata.json"
                if metadata.is_file():
                    sources.append(metadata)
    sensing = getattr(context, "sensing", {})
    cfg = dict(
        scene=dict(source=context.scene_source, id=context.scene_id),
        sensing=dict(enabled=bool(getattr(sensing, "enabled", False))),
        timeline=dict(duration_s=timeline.duration_s),
        generator_summary=getattr(context, "generator_summary", None) or {},
        raytracing=dict(
            mesh_update_interval_s=getattr(simulation_config, "mesh_update_interval_s", None)
        ),
    )
    # Sensing configuration may be a mapping at this boundary.
    if isinstance(sensing, dict):
        cfg["sensing"]["enabled"] = bool(sensing.get("enabled", False))
    case = _build_case(
        cfg,
        root,
        config_path,
        context.project_root,
        xml,
        scene,
        sources,
        poses,
        timeline,
        assets,
        alignments,
    )
    case["receipt"]["resolved_yaml_sha256"] = context.summary_yaml_hash
    case["receipt"]["actor_source"] = "prepared_runtime"
    return case


def _build_case(
    cfg, root, config_path, project_root, xml, scene, sources, poses, timeline, assets, alignments
):
    indices = list(dict.fromkeys((0, timeline.steps - 1)))
    visualization = (cfg.get("generator_summary") or {}).get("visualization") or {}
    pose_samples = _pose_sample_count(visualization.get("topology_pose_samples", 3))
    mesh_interval = mesh_update_step_interval(
        duration=cfg["timeline"]["duration_s"],
        num_steps=timeline.steps,
        mesh_update_interval_s=cfg.get("raytracing", {}).get("mesh_update_interval_s"),
    )
    actors = []
    mesh_cache = {}
    for i, pose in enumerate(poses):
        positions = pose["positions"]
        rotations = pose["rotations"]
        animation = assets.get(pose["name"], ({}, []))[0].get("animation")
        if animation is not None:
            # The manifest owns root placement and orientation. Actor centers
            # used by RT may also shift with articulation; they are not root motion.
            affine = np.asarray([s.resolved_root_affine_4x4 for s in animation.binding.samples])
            positions = affine[:, :3, 3]
            linear = affine[:, :3, :3]
            rotations = (
                linear / np.linalg.norm(linear[:, :, 0], axis=1)[:, None, None]
            ) @ animation.asset.source_to_orchav_basis.T
        pointing_rotations = rotations
        if pose["name"] in alignments:
            # A mesh may face -X; its pointing arrow follows the declared front,
            # while its vertices retain the generator's asset-aligned rotation.
            pointing_rotations = rotations @ rotation_matrix(alignments[pose["name"]]).T
        meshes, mesh_samples, animated = [], [], False
        if pose["role"] == "target":
            asset, paths = assets[pose["name"]]
            animated = asset["switch_meshes"] and len(paths) > 1
            mesh_indices = (
                _pose_sample_indices(len(positions), pose_samples if animated else 3)
                if animated or not is_stationary(positions)
                else [0]
            )
            for idx in mesh_indices:
                if animation is not None:
                    sample = animation.binding.samples[idx]
                    mesh_index = next(
                        n
                        for n, p in enumerate(animation.asset.pose_samples)
                        if p.sample_id == sample.pose_sample_id
                    )
                    meshes.append(sample.vertices_m[animation.binding.faces])
                    mesh_samples.append(
                        dict(
                            index=idx,
                            mesh_index=mesh_index,
                            path=paths[mesh_index],
                            time_s=sample.scenario_time_s,
                            topology_visible=not is_stationary(positions) or idx == 0,
                            opacity=0.28 if idx not in indices else 1.0,
                            animated_sample=sample.evidence_dict(),
                        )
                    )
                    continue
                call = mesh_call_count_for_step(idx, mesh_interval) if asset["switch_meshes"] else 0
                mesh_index = mesh_sequence_index(
                    len(paths),
                    call,
                    mesh_start_index=asset["start_index"],
                    mesh_frame_stride=asset["frame_stride"],
                    mesh_end_behavior=asset["mesh_end_behavior"],
                )
                path = paths[mesh_index]
                if "geometry" in asset:
                    mesh_cache[path] = asset["geometry"]
                elif path not in mesh_cache:
                    mesh_cache[path] = _read_mesh(path)
                vertices, faces = mesh_cache[path]
                v = vertices * float(asset["scale"]) @ rotations[idx].T
                meshes.append((v - (v.min(0) + v.max(0)) / 2 + positions[idx])[faces])
                sources.append(path)
                mesh_samples.append(
                    dict(
                        index=idx,
                        mesh_index=mesh_index,
                        path=path,
                        time_s=timeline.timestamps_s[idx],
                        topology_visible=not is_stationary(positions) or idx == 0,
                        opacity=0.28 if idx not in indices else 1.0,
                    )
                )
        actors.append(
            dict(
                name=pose["name"],
                label=pose["name"],
                role=pose["role"],
                positions=positions,
                stationary=is_stationary(positions),
                meshes=meshes,
                mesh_samples=mesh_samples,
                animated=animated,
                color=COLORS[i % len(COLORS)],
                rotations=rotations,
                pointing_rotations=pointing_rotations,
                orientation_indices=orientation_indices(rotations),
                physical_path=pose["physical_path"],
                members=[pose["name"]],
                representation=assets.get(pose["name"], ({}, []))[0].get(
                    "representation", "mesh_material"
                ),
            )
        )
    # Presentation-only grouping requires colocation throughout the trajectory.
    if cfg.get("sensing", {}).get("enabled", False):
        tx, rx = ([a for a in actors if a["role"] == kind] for kind in ("tx", "rx"))
        pairs = monostatic_label_pairs([a["positions"] for a in tx], [a["positions"] for a in rx])
        named_labels = ((cfg.get("generator_summary") or {}).get("visualization") or {}).get(
            "actor_label_mode"
        ) == "name"
        for i, j in pairs.items():
            tx[i]["label"] = sensing_role_label(
                "tx",
                i,
                pairs,
                device_names=(tx[i]["name"], rx[j]["name"]) if named_labels else None,
            )
            if tx[i]["label"].startswith("Monostatic radar "):
                # The topology legend already supplies the actor's unique number.
                tx[i]["label"] = "Monostatic radar"
            tx[i]["role"] = "radar"
            tx[i]["members"].append(rx[j]["name"])
        actors = [a for a in actors if not any(a is rx[j] for j in pairs.values())]

    def source_name(path: Path) -> str:
        path = Path(path).resolve()
        if path.is_relative_to(project_root):
            return path.relative_to(project_root).as_posix()
        if cfg["scene"].get("source") == "sionna" and path.is_relative_to(xml.parent.parent):
            return "sionna_rt/scenes/" + path.relative_to(xml.parent.parent).as_posix()
        return str(path)

    receipt = dict(
        scenario=source_name(config_path),
        scene_source=cfg["scene"],
        source_files={source_name(p): _sha(p) for p in sorted(set(sources))},
        sample_indices=indices,
        sample_times_s=[timeline.timestamps_s[i] for i in indices],
        steps=timeline.steps,
        actors=[
            dict(
                name=a["label"],
                role=a["role"],
                members=a["members"],
                sample_positions_m=a["positions"][indices].tolist(),
                physical_path=a["physical_path"],
                stationary=a["stationary"],
                animated=a["animated"],
                mesh_samples=[
                    dict(
                        index=s["index"],
                        mesh_index=s["mesh_index"],
                        source=source_name(s["path"]),
                        time_s=s["time_s"],
                        topology_visible=s["topology_visible"],
                        opacity=s["opacity"],
                        **(
                            {"animated_sample": s["animated_sample"]}
                            if "animated_sample" in s
                            else {}
                        ),
                    )
                    for s in a["mesh_samples"]
                ],
                orientation_samples=[
                    dict(
                        index=i,
                        time_s=timeline.timestamps_s[i],
                        position_m=a["positions"][i].tolist(),
                        rotation_matrix=a["rotations"][i].tolist(),
                        forward_direction=a["pointing_rotations"][i, :, 0].tolist(),
                    )
                    for i in a["orientation_indices"]
                ],
            )
            for a in actors
        ],
    )
    for actor, recorded in zip(actors, receipt["actors"]):
        asset = assets.get(actor["name"], ({}, []))[0]
        if "animation" in asset:
            recorded["position_reference"] = "animated_asset_manifest_root"
            recorded["animated_asset"] = asset["animation"].catalog_metadata()
        if "representation" in asset:
            recorded["representation"] = asset["representation"]
            recorded["proxy_geometry"] = asset["proxy_geometry"]
    return dict(
        scene=scene,
        actors=actors,
        receipt=receipt,
        title=root.name.replace("_", " ").title(),
        sensing_enabled=bool(cfg.get("sensing", {}).get("enabled", False)),
        visualization=visualization,
    )


class _SolidArrowhead(path_effects.AbstractPathEffect):
    """Keep filled arrowhead outlines solid while leaving the shaft dashed."""

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        if rgbFace is None:
            renderer.draw_path(gc, tpath, affine, rgbFace)
            return
        head_gc = renderer.new_gc()
        head_gc.copy_properties(gc)
        head_gc.set_dashes(0, None)
        renderer.draw_path(head_gc, tpath, affine, rgbFace)
        head_gc.restore()


def _arrow_key(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    """Keep an actual arrow glyph, including opacity/dashing, in the legend."""
    return FancyArrowPatch(
        (0, height / 2),
        (width, height / 2),
        arrowstyle=orig_handle.get_arrowstyle(),
        mutation_scale=0.9 * fontsize * orig_handle.get_mutation_scale() / 11,
        facecolor=orig_handle.get_facecolor(),
        edgecolor=orig_handle.get_edgecolor(),
        alpha=orig_handle.get_alpha(),
        linestyle=orig_handle.get_linestyle(),
        lw=orig_handle.get_linewidth(),
        path_effects=orig_handle.get_path_effects(),
        shrinkA=0,
        shrinkB=0,
    )


def _orientation_style(color, *, middle=False, final=False):
    """Equal-sized arrows distinguish sampled poses through fill and opacity."""
    return dict(
        arrowstyle="-|>",
        mutation_scale=11,
        edgecolor=color,
        facecolor="white" if final else color,
        alpha=0.35 if middle else 1,
        linewidth=1.5,
        linestyle=(0, (2, 1.8)),
        path_effects=[_SolidArrowhead()],
    )


def _pointing_arrow(ax, position, direction, length, **style):
    """Project native directions, keeping arrowheads readable from above."""
    start = proj3d.proj_transform(*position, ax.get_proj())[:2]
    end = proj3d.proj_transform(*(position + length * direction), ax.get_proj())[:2]
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        zorder=5,
        arrowprops=dict(shrinkA=0, shrinkB=0, **style),
    )


def _target_camera(actor):
    forward = actor["pointing_rotations"][0, :, 0]
    return 18, np.rad2deg(np.arctan2(forward[1], forward[0])) + 40


@dataclass
class _TargetLegendKey:
    actor: dict
    label: str

    def get_label(self):
        return self.label


class _TargetLegendHandler(HandlerBase):
    """Orthographically project the first native mesh into its legend handle."""

    def create_artists(self, legend, original, xdescent, ydescent, width, height, fontsize, trans):
        actor = original.actor
        mesh = actor["meshes"][0]
        elevation, azimuth = np.deg2rad(_target_camera(actor))
        ce, se, ca, sa = np.cos(elevation), np.sin(elevation), np.cos(azimuth), np.sin(azimuth)
        basis = np.array([[-sa, ca, 0], [-se * ca, -se * sa, ce], [ce * ca, ce * sa, se]])
        projected = mesh @ basis.T
        lo, hi = projected.min((0, 1)), projected.max((0, 1))
        scale = min(width / max(hi[0] - lo[0], 1e-12), height / max(hi[1] - lo[1], 1e-12))
        xy = (projected[:, :, :2] - (lo[:2] + hi[:2]) / 2) * scale
        xy += [width / 2 - xdescent, height / 2 - ydescent]
        order = np.argsort(projected[:, :, 2].mean(1))
        return [
            PolyCollection(
                xy[order],
                facecolors=_shaded(mesh, actor["color"], strength=0.38)[order],
                edgecolors="none",
                linewidths=0,
                transform=trans,
            )
        ]


def _gallery_columns(actors, figure_width):
    """Keep five-pose galleries readable across single- and multiple-target views."""
    targets = [a for a in actors if a["role"] == "target"]
    if any(a["animated"] and len(a["meshes"]) > 3 for a in targets):
        return 1
    return min(len(targets), max(1, int(figure_width // 4)))


def _target_gallery(fig, actors, bottom, *, detail_actors=(), details=None):
    """One rigid-mesh close-up or the selected authored animation poses."""
    targets = [(i, a) for i, a in enumerate(actors, 1) if a["role"] == "target"]
    if not targets:
        return bottom
    from .topology_motion import overlapping_rigid_snapshots

    has_faded_detail = any(
        a["name"] in detail_actors and overlapping_rigid_snapshots(a) for _, a in targets
    )
    row_height = (2.1 if has_faded_detail else 1.8) / fig.get_figheight()
    # Keep close-ups readable on narrow single-view exports too.
    columns = _gallery_columns(actors, fig.get_figwidth())
    rows = (len(targets) + columns - 1) // columns
    padding = 0.10 / fig.get_figheight()
    header = 0.32 / fig.get_figheight()
    top = bottom + rows * row_height + header + 2 * padding
    _panel_frame(fig, (0.006, bottom, 0.988, top - bottom))
    fig.text(
        0.5,
        top - padding,
        (
            "Target motion"
            if detail_actors or any(a["animated"] for _, a in targets)
            else "Target detail"
        ),
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    bottom += padding
    for slot, (number, actor) in enumerate(targets):
        row, column = divmod(slot, columns)
        base = bottom + (rows - row - 1) * row_height
        left, group_width = column / columns, 1 / columns
        meshes = actor["meshes"] if actor["animated"] else actor["meshes"][:1]
        samples = actor["mesh_samples"] if actor["animated"] else actor["mesh_samples"][:1]
        centered = [m - actor["positions"][s["index"]] for m, s in zip(meshes, samples)]
        vertices = np.concatenate([m.reshape(-1, 3) for m in centered])
        lo, hi = vertices.min(0), vertices.max(0)
        extent = np.maximum(hi - lo, 0.05)
        lo, hi = lo - extent * 0.06, hi + extent * 0.06
        fig.text(
            left + group_width / 2,
            base + row_height * 0.94,
            f"{number}  {actor['label']}",
            ha="center",
            va="top",
            fontsize=10,
            color=actor["color"],
            fontweight="bold",
        )
        if actor["name"] in detail_actors:
            from .topology_motion import draw_motion_detail

            detail = draw_motion_detail(
                fig,
                actor,
                (
                    left + group_width * 0.08,
                    base + row_height * 0.20,
                    group_width * 0.84,
                    row_height * 0.62,
                ),
            )
            if details is not None:
                details.append(detail)
            plane = "".join(detail["projection_axes"])
            kind = (
                "Motion detail"
                if actor.get("representation", "mesh_material") == "mesh_material"
                else "RCS proxy motion"
            )
            fig.text(
                left + group_width / 2,
                base + row_height * 0.02,
                (
                    f"{kind} ({plane}) · "
                    + " / ".join(f"{time:g}" for time in detail["pose_times_s"].values())
                    + " s"
                    if "pose_times_s" in detail
                    else f"{kind} ({plane}) · dashed: {detail['start_time_s']:g} s · solid: {detail['end_time_s']:g} s"
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )
            continue
        for j, (mesh, sample) in enumerate(zip(centered, samples)):
            cell_width = group_width / len(meshes)
            ax = fig.add_axes(
                (left + j * cell_width, base + row_height * 0.13, cell_width, row_height * 0.73),
                projection="3d",
                facecolor="none",
            )
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
            ax.set_box_aspect(hi - lo, zoom=1.08)
            ax.set_proj_type("ortho")
            ax.view_init(*_target_camera(actor))
            ax.set_axis_off()
            ax.add_collection3d(
                Poly3DCollection(
                    mesh,
                    facecolors=_shaded(mesh, actor["color"], strength=0.38),
                    edgecolors="none",
                    linewidths=0,
                )
            )
            label = "First" if j == 0 else "Final" if j == len(meshes) - 1 else "Middle"
            if len(meshes) > 3 and label == "Middle":
                middle_index = len(actor["positions"]) // 2
                label = (
                    "Quarter"
                    if sample["index"] < middle_index
                    else "Three-quarter" if sample["index"] > middle_index else "Middle"
                )
            fig.text(
                left + (j + 0.5) * cell_width,
                base + row_height * 0.02,
                (
                    f"{label} · {sample['time_s']:g} s"
                    if actor["animated"]
                    else (
                        "RCS geometry proxy"
                        if actor.get("representation", "mesh_material") != "mesh_material"
                        else "Mesh detail"
                    )
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )
    return top


def _footer_legend(fig, handles, bottom, *, fontsize=9):
    """Fit a compact row to the canvas, wrapping only when labels require it."""
    if not handles:
        return None, bottom
    for columns in range(len(handles), 0, -1):
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, bottom),
            ncol=columns,
            frameon=False,
            fontsize=fontsize,
            borderaxespad=0,
            borderpad=0,
            columnspacing=1.1,
            handlelength=1.6,
            handleheight=2.6 if any(isinstance(h, _TargetLegendKey) for h in handles) else 0.7,
            handletextpad=0.45,
            labelspacing=0.4,
            handler_map={
                FancyArrowPatch: HandlerPatch(patch_func=_arrow_key),
                _TargetLegendKey: _TargetLegendHandler(),
            },
        )
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        if bounds.width <= fig.bbox.width * 0.97 or columns == 1:
            return legend, bottom + bounds.height / fig.bbox.height
        legend.remove()


def _footer_keys(fig, groups, bottom):
    """Wrap complete key groups, preserving left-to-right reading order.

    Orientation arrows have longer handles than position/material keys. Their
    group stays together even when a narrow figure needs another footer row.
    """
    rows, row, row_width = [], [], 0.0
    gap, row_gap = 0.012, 0.008
    for handles, handlelength in groups:
        if not handles:
            continue
        legend = fig.legend(
            handles=handles,
            loc="lower left",
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            borderaxespad=0,
            borderpad=0,
            columnspacing=1.0,
            handlelength=handlelength,
            handletextpad=0.45,
            handler_map={FancyArrowPatch: HandlerPatch(patch_func=_arrow_key)},
        )
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        width, height = bounds.width / fig.bbox.width, bounds.height / fig.bbox.height
        if row and row_width + gap + width > 0.97:
            rows.append((row, row_width))
            row, row_width = [], 0.0
        row_width += (gap if row else 0) + width
        row.append((legend, width, height))
    if row:
        rows.append((row, row_width))
    # Place from the bottom up so the authored groups read top-to-bottom.
    for row, width in reversed(rows):
        left = (1 - width) / 2
        for legend, width, height in row:
            legend.set_bbox_to_anchor((left, bottom))
            left += width + gap
        bottom += max(item[2] for item in row) + row_gap
    return bottom - row_gap


def _display_actors(actors):
    """Share one key for large stationary populations with identical orientation.

    Every node position remains drawn. A common orientation is represented once
    rather than overlaying hundreds of identical arrows and numbered labels.
    """
    display = [dict(actor) for actor in actors]
    grouped = {}
    for role in ("tx", "rx"):
        population = [i for i, a in enumerate(display) if a["role"] == role and a["stationary"]]
        if len(population) <= 24:
            continue
        first = display[population[0]]
        if not all(
            np.allclose(display[i]["rotations"], first["rotations"], atol=1e-8, rtol=0)
            for i in population
        ):
            continue
        for i in population:
            display[i]["color"] = first["color"]
            grouped[i] = population[0]
        first["label"] = (
            f"{len(population)} stationary {'receivers' if role == 'rx' else 'transmitters'}"
        )
    keys = [i for i in range(len(display)) if grouped.get(i, i) == i]
    numbers = {index: number for number, index in enumerate(keys, 1)}
    return display, keys, numbers, grouped


class _TopologyAxes(Axes3D):
    """Rectangular orthographic viewport; projected limits preserve equal scale."""

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, "active")


def _fit_topology_view(ax, points):
    """Fit projected geometry with uniform screen scale and a small label margin."""
    px, py, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    low = np.array([np.min(px), np.min(py)])
    high = np.array([np.max(px), np.max(py)])
    extent = np.maximum(high - low, 1e-8)
    box = ax.get_position(original=True)
    ratio = box.width * ax.figure.get_figwidth() / (box.height * ax.figure.get_figheight())
    height = max(extent[1], extent[0] / ratio) * 1.08
    width = height * ratio
    center = (low + high) / 2
    ax.viewLim.intervalx = (center[0] - width / 2, center[0] + width / 2)
    ax.viewLim.intervaly = (center[1] - height / 2, center[1] + height / 2)
    return box.y0 + box.height * (0.5 + extent[1] / (2 * height))


def _top_view_axes(ax, low, high):
    """Align light coordinate rulers with the true orthographic XY footprint."""
    bounds = np.array([low, high], dtype=float)
    # Keep a usable coordinate axis for a line or single-point layout too.
    for dimension in (0, 1):
        if bounds[1, dimension] - bounds[0, dimension] < 1e-8:
            bounds[:, dimension] += [-0.5, 0.5]
    px, py, _ = proj3d.proj_transform(*bounds.T, ax.get_proj())
    corners = ax.figure.transFigure.inverted().transform(
        ax.transData.transform(np.column_stack((px, py)))
    )
    left, bottom = corners.min(0)
    width, height = np.ptp(corners, axis=0)
    ruler = ax.figure.add_axes((left, bottom, width, height), facecolor="none")
    ruler.grid(False)
    ruler.set_xlim(bounds[:, 0])
    ruler.set_ylim(bounds[:, 1])
    for side in ("top", "right"):
        ruler.spines[side].set_visible(False)
    for side, label in (("bottom", "X"), ("left", "Y")):
        ruler.spines[side].set_position(("outward", 7))
        ruler.spines[side].set_color(AXIS_COLORS[label])
        ruler.spines[side].set_linewidth(0.9)
    ruler.tick_params(axis="both", labelsize=8, colors=MUTED, length=3, width=0.7, pad=4)
    # Long radial layouts can have a very narrow transverse ruler. Keep the
    # coordinates readable without stretching the scene to make labels fit.
    for axis, inches in (
        (ruler.xaxis, width * ax.figure.get_figwidth()),
        (ruler.yaxis, height * ax.figure.get_figheight()),
    ):
        axis.set_major_locator(MaxNLocator(nbins=max(1, min(4, int(inches / 0.65))), min_n_ticks=2))
    ruler.ticklabel_format(style="plain", useOffset=False)
    ruler.set_xlabel("X (m)", fontsize=9, color=AXIS_COLORS["X"], labelpad=6)
    ruler.set_ylabel("Y (m)", fontsize=9, color=AXIS_COLORS["Y"], labelpad=6)


def _axis_orientation_guide(ax, left, bottom):
    """Project a world-coordinate triad using the scene's actual camera."""
    fig = ax.figure
    # Use a local unit offset near the view center for stable projection of
    # scenes far from the world origin. Screen coordinates include panel aspect.
    origin = np.array([np.mean(limits) for limits in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
    points = origin + np.vstack((np.zeros(3), np.eye(3)))
    px, py, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    screen = ax.transData.transform(np.column_stack((px, py)))
    directions = screen[1:] - screen[0]
    directions /= np.linalg.norm(directions, axis=1).max()
    guide = fig.add_axes(
        (left + 0.025, bottom + 0.018, 1.05 / fig.get_figwidth(), 1.05 / fig.get_figheight()),
        facecolor="none",
    )
    guide.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal")
    guide.set_axis_off()
    for label, direction in zip("XYZ", directions):
        guide.annotate(
            "",
            xy=direction,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=AXIS_COLORS[label], lw=1.8, mutation_scale=9),
        )
        guide.text(
            *(direction * 1.18),
            label,
            color=AXIS_COLORS[label],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )
    guide.plot(0, 0, "o", color=MUTED, markersize=2.5)


def render_topology(case: dict, output: Path, *, view: str = "both", dpi: int = 170) -> None:
    """Draw orthographic views with one compact legend and native sampled poses."""
    if view not in {"3d", "top", "both"}:
        raise ValueError("view must be 3d, top or both")
    actors, key_indices, numbers, grouped = _display_actors(case["actors"])
    options = case.get("visualization", {})
    pose_samples = _pose_sample_count(options.get("topology_pose_samples", 3))
    motion_detail = options.get("topology_motion_detail", "auto")
    fixed_target_orientation = options.get("topology_fixed_target_orientation", "auto")

    motion_details = []
    scene = case["scene"]
    case["receipt"]["display_groups"] = [
        dict(
            label=actors[i]["label"],
            members=[actors[j]["name"] for j in grouped if grouped[j] == i],
            shared_orientation=True,
        )
        for i in key_indices
        if i in grouped
    ]
    points = [a["positions"] for a in actors] + [m["vertices"] for m in scene]
    points += [m.reshape(-1, 3) for a in actors for m in a["meshes"]]
    points = np.concatenate(points)
    lo, hi = points.min(0), points.max(0)
    extent = np.maximum(hi - lo, 1)
    lo, hi = lo - 0.04 * extent, hi + 0.04 * extent
    span = max(hi - lo)
    from .topology_motion import (
        POSE_LABELS,
        POSE_OPACITIES,
        annotate_pose_times,
        motion_detail_enabled,
        overlapping_rigid_snapshots,
        rigid_pose_style,
        transparent_mesh_faces,
    )

    detail_actors = {
        a["name"]
        for a in actors
        if motion_detail_enabled(
            a, motion_detail, sensing_enabled=case.get("sensing_enabled", False), scene_span=span
        )
    }
    faded_actors = {a["name"] for a in actors if overlapping_rigid_snapshots(a)}
    mesh_snapshot_styles = []

    def visible_orientations(actor):
        indices = actor["orientation_indices"]
        show_fixed = (
            actor["name"] not in detail_actors
            if fixed_target_orientation == "auto"
            else fixed_target_orientation
        )
        return [] if actor["role"] == "target" and len(indices) == 1 and not show_fixed else indices

    views = ("3d", "top") if view == "both" else (view,)
    panel_widths = (0.65, 0.35) if view == "both" else (1.0,)
    # Reuse the geometry and display colors between cameras.
    layers, opaque, colors = [], [], []
    materials = {}
    for mesh in scene:
        label = mesh["material_label"]
        key = (label, tuple(mesh["color"]))
        materials[key] = mesh["color"]
        tri = mesh["vertices"][mesh["faces"]]
        shade = _shaded(tri, mesh["color"])
        if mesh["background"]:
            layers.append((tri, shade, 0 if "street" not in mesh["name"] else 1))
        else:
            opaque.append(tri)
            colors.append(shade)
            edges = tube_faces(mesh["edges"], span * 0.00055, sides=4)
            if len(edges):
                opaque.append(edges)
                colors.append(np.tile(to_rgb("#41484c"), (len(edges), 1)))
    for actor in actors:
        seen = set()
        fade = actor["name"] in faded_actors
        snapshots = sorted(
            zip(actor["meshes"], actor["mesh_samples"]),
            # Prefer the latest mesh when a trajectory returns to an
            # identical sampled pose, before deduplicating native geometry.
            key=lambda item: -item[1]["index"] if fade else -item[1]["opacity"],
        )
        for mesh, sample in snapshots:
            if not sample["topology_visible"]:
                continue
            identity = mesh.tobytes()
            if identity not in seen:
                seen.add(identity)
                label, opacity = (
                    rigid_pose_style(actor, sample) if fade else (None, sample["opacity"])
                )
                polygons = transparent_mesh_faces(mesh) if fade else mesh
                opaque.append(polygons)
                shade = _shaded(
                    np.asarray([p[:3] for p in polygons]), actor["color"], strength=0.38
                )
                colors.append(np.column_stack((shade, np.full(len(shade), opacity))))
                if fade:
                    mesh_snapshot_styles.append(
                        dict(
                            name=actor["name"],
                            index=sample["index"],
                            time_s=sample["time_s"],
                            style="filled",
                            label=label,
                            opacity=opacity,
                        )
                    )
                edges = tube_faces(
                    crease_segments(mesh.reshape(-1, 3), np.arange(mesh.size // 3).reshape(-1, 3)),
                    span * 0.00055,
                    sides=4,
                )
                if len(edges):
                    opaque.append(edges)
                    colors.append(np.tile((*to_rgb("#41484c"), opacity), (len(edges), 1)))
    if opaque:
        rgba = [np.column_stack((c, np.ones(len(c)))) if c.shape[1] == 3 else c for c in colors]
        layers.append(([face for mesh in opaque for face in mesh], np.concatenate(rgba), 2))
    changing = any(len(a["orientation_indices"]) > 1 for a in actors)
    moving = any(not a["stationary"] for a in actors)
    scans = any(not a["physical_path"] and not a["stationary"] for a in actors)
    roll_only = any(
        len(a["orientation_indices"]) > 1
        and np.allclose(
            a["pointing_rotations"][:, :, 0], a["pointing_rotations"][0, :, 0], atol=1e-8, rtol=0
        )
        for a in actors
    )
    cameras = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK}):
        figure_width = 16 if len(views) == 2 else 9
        target_count = sum(a["role"] == "target" for a in actors)
        gallery_columns = max(1, _gallery_columns(actors, figure_width))
        gallery_rows = (target_count + gallery_columns - 1) // gallery_columns
        fig = plt.figure(
            figsize=(figure_width, 8 + 1.8 * gallery_rows + (0.52 if target_count else 0)),
            facecolor="white",
        )
        axes = []
        headings = []
        for column, current_view in enumerate(views):
            left, width = sum(panel_widths[:column]), panel_widths[column]
            ax = fig.add_axes(
                (left + 0.004, 0.09, width - 0.008, 0.87),
                axes_class=_TopologyAxes,
                computed_zorder=False,
            )
            axes.append(ax)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
            ax.set_box_aspect(hi - lo, zoom=1.28)
            ax.set_proj_type("ortho")
            elevation, azimuth = (90, -90) if current_view == "top" else (52 if scene else 30, -60)
            ax.view_init(elev=elevation, azim=azimuth)
            # Fit the complete 3D bounds, including tall room walls, inside the
            # panel. A fixed zoom can clip corners even with correct XYZ limits.
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            px, py, _ = proj3d.proj_transform(*corners.T, ax.get_proj())
            normalized = ax.transAxes.inverted().transform(
                ax.transData.transform(np.column_stack((px, py)))
            )
            radius = np.abs(normalized - 0.5).max()
            if radius > 0.47:
                ax.set_box_aspect(hi - lo, zoom=1.28 * 0.47 / radius)
            ax.set_axis_off()
            for tri, shade, order in layers:
                if len(tri):
                    ax.add_collection3d(
                        Poly3DCollection(
                            tri,
                            facecolors=shade,
                            edgecolors="none",
                            linewidths=0,
                            antialiased=False,
                            zorder=order,
                        )
                    )
            for actor_index, actor in enumerate(actors):
                p, color = actor["positions"], actor["color"]
                if not actor["stationary"]:
                    if actor["physical_path"]:
                        ax.plot(*p.T, color=color, lw=1.8, zorder=4)
                    else:
                        ax.scatter(*p.T, color=color, s=10, zorder=4)
                    # Position markers overlay meshes, orientation arrows and scene.
                    ax.scatter(
                        *p[-1],
                        marker="s",
                        color=color,
                        edgecolors="white",
                        linewidths=0.7,
                        s=23,
                        zorder=11,
                        depthshade=False,
                    )
                if actor["role"] != "target" or not actor["stationary"]:
                    ax.scatter(
                        *p[0],
                        marker=_node_marker(actor),
                        facecolors="white" if _node_marker(actor) == "D" else color,
                        edgecolors=color if actor["stationary"] else "white",
                        linewidths=1.5 if actor["stationary"] else 0.7,
                        s=50,
                        zorder=10,
                        depthshade=False,
                    )
                if actor_index not in numbers:
                    continue
                number = numbers[actor_index]
                samples = visible_orientations(actor)
                pointing = actor["pointing_rotations"]
                up_tick = len(samples) > 1 and np.allclose(
                    pointing[:, :, 0], pointing[0, :, 0], atol=1e-8, rtol=0
                )
                for index in samples:
                    middle = index not in (0, len(p) - 1)
                    final = index != 0 and index == len(p) - 1
                    kwargs = _orientation_style(color, middle=middle, final=final)
                    _pointing_arrow(
                        ax,
                        p[index],
                        pointing[index, :, 0],
                        span * 0.065,
                        **kwargs,
                    )
                    if up_tick:
                        _pointing_arrow(
                            ax,
                            p[index],
                            pointing[index, :, 2],
                            span * 0.032,
                            **kwargs,
                        )
                x, y, _ = proj3d.proj_transform(*p[0], ax.get_proj())
                ax.annotate(
                    str(number),
                    (x, y),
                    xytext=(10, 9) if actor["role"] == "target" else (5, 7),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
                    zorder=6,
                )
            headings.append(
                fig.text(
                    left + width / 2,
                    0.985,
                    "3D" if current_view == "3d" else "Top view",
                    ha="center",
                    va="top",
                    color=INK,
                    fontsize=11,
                    fontweight="bold",
                )
            )
            cameras.append(dict(view=current_view, elevation_deg=elevation, azimuth_deg=azimuth))
        handles = [
            (
                _TargetLegendKey(a, f"{i}  {a['label']}")
                if a["role"] == "target"
                else Line2D(
                    [],
                    [],
                    color=a["color"],
                    linestyle="none",
                    marker=_node_marker(a),
                    markerfacecolor="white" if _node_marker(a) == "D" else a["color"],
                    markeredgewidth=1.5,
                    markersize=6,
                    label=f"{i}  {a['label']}",
                )
            )
            for i, a in ((numbers[j], actors[j]) for j in key_indices)
        ]
        stationary_keys, position_keys = [], []
        if any(a["stationary"] and a["role"] not in {"target", "radar"} for a in actors):
            stationary_keys.append(
                Line2D(
                    [],
                    [],
                    color=INK,
                    marker="D",
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    linestyle="none",
                    label="Stationary node",
                )
            )
        if moving:
            position_keys.extend(
                [
                    Line2D(
                        [],
                        [],
                        color=MUTED,
                        marker="o",
                        linestyle="none",
                        markersize=6,
                        label="Start position",
                    ),
                    Line2D(
                        [],
                        [],
                        color=MUTED,
                        marker="s",
                        linestyle="none",
                        markersize=4,
                        label="End position",
                    ),
                ]
            )
        if moving:
            position_keys.append(Line2D([], [], color=MUTED, lw=1.8, label="Trajectory"))
        if scans:
            position_keys.append(
                Line2D([], [], color=MUTED, marker=".", linestyle="none", label="Grid samples")
            )
        orientation_keys = (
            [
                FancyArrowPatch(
                    (0, 0), (1, 0), **_orientation_style(INK), label="Initial orientation"
                )
            ]
            if any(visible_orientations(a) for a in actors)
            else []
        )
        if changing:
            orientation_keys.extend(
                [
                    FancyArrowPatch(
                        (0, 0),
                        (1, 0),
                        **_orientation_style(INK, middle=True),
                        label="Midpoint orientation",
                    ),
                    FancyArrowPatch(
                        (0, 0),
                        (1, 0),
                        **_orientation_style(INK, final=True),
                        label="Final orientation",
                    ),
                ]
            )
        groups = [(stationary_keys, 1.6), (position_keys, 1.6), (orientation_keys, 3.0)]
        if faded_actors:
            pose_color = (
                next(a["color"] for a in actors if a["name"] in faded_actors)
                if len(faded_actors) == 1
                else MUTED
            )
            groups.append(
                (
                    [
                        Patch(
                            facecolor=pose_color, edgecolor=pose_color, alpha=opacity, label=label
                        )
                        for label, opacity in zip(POSE_LABELS, POSE_OPACITIES)
                    ],
                    1.6,
                )
            )
        if roll_only:
            groups.append(
                ([Line2D([], [], color=MUTED, lw=1, label="Short arrow: local up (roll)")], 1.6)
            )
        material_groups = [
            ([Patch(facecolor=color, edgecolor="none", label=key[0])], 1.6)
            for key, color in sorted(materials.items())
        ]
        material_top = _footer_keys(fig, material_groups, 0.016)
        key_top = _footer_keys(fig, groups, material_top + (0.009 if material_groups else 0))
        actor_legend, footer_top = _footer_legend(fig, handles, key_top + 0.009, fontsize=10)
        for text, actor in zip(
            actor_legend.get_texts() if actor_legend else [], [actors[j] for j in key_indices]
        ):
            text.set_color(actor["color"])
        legend_bounds = Bbox.union(
            [legend.get_window_extent(fig.canvas.get_renderer()) for legend in fig.legends]
        ).transformed(fig.transFigure.inverted())
        # Pad the actual legend contents in points, independent of canvas size.
        pad_x, pad_y = np.array([8, 5]) / 72 / fig.get_size_inches()
        fig.add_artist(
            FancyBboxPatch(
                (legend_bounds.x0 - pad_x, legend_bounds.y0 - pad_y),
                legend_bounds.width + 2 * pad_x,
                legend_bounds.height + 2 * pad_y,
                transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.008",
                **VIEW_FRAME_STYLE,
                zorder=-1,
            )
        )
        footer_top = _target_gallery(
            fig, actors, footer_top + 0.012, detail_actors=detail_actors, details=motion_details
        )
        # Give the views all remaining height, including when a long key wraps.
        bottom = footer_top + 0.015
        # Include authored orientation tips in the fit, not just actor origins.
        fitting = [points]
        for actor in actors:
            indices = visible_orientations(actor)
            fitting.append(
                actor["positions"][indices]
                + span * 0.065 * actor["pointing_rotations"][indices, :, 0]
            )
        fit_points = np.concatenate(fitting)
        for column, ax in enumerate(axes):
            left, width = sum(panel_widths[:column]), panel_widths[column]
            # Reserve space inside the frame for the heading and coordinate rulers.
            _panel_frame(fig, (left + 0.006, bottom, width - 0.012, 0.995 - bottom))
            ax.set_facecolor("none")
            ax.set_position((left + 0.035, bottom + 0.05, width - 0.06, 0.88 - bottom))
            _fit_topology_view(ax, fit_points)
            # Project after the final fit so close native poses get readable labels.
            for actor in actors:
                if actor["name"] not in detail_actors | faded_actors:
                    continue
                indices = [sample["index"] for sample in actor["mesh_samples"]]
                if len(indices) != 3:
                    continue
                anchors = actor["positions"][indices]
                footprint = np.concatenate(actor["meshes"]).reshape(-1, 3)
                px, py, _ = proj3d.proj_transform(*anchors.T, ax.get_proj())
                fx, fy, _ = proj3d.proj_transform(*footprint.T, ax.get_proj())
                annotate_pose_times(
                    ax,
                    np.column_stack((px, py)),
                    np.column_stack((fx, fy)),
                    actor["color"],
                    side="right" if views[column] == "top" else "left",
                )
            headings[column].set_verticalalignment("top")
            headings[column].set_y(0.965)
            if views[column] == "3d":
                _axis_orientation_guide(ax, left, bottom)
            if views[column] == "top":
                _top_view_axes(ax, points.min(0), points.max(0))
                size_x, size_y = np.ptp(points[:, :2], axis=0)
                fig.text(
                    left + width / 2,
                    0.942,
                    f"{size_x:,.3g} × {size_y:,.3g} m",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=MUTED,
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        # One opaque image remains readable on both light and dark documentation pages.
        fig.savefig(
            output,
            dpi=dpi,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
    camera = dict(projection="orthographic", view=view, bounds_m=[lo.tolist(), hi.tolist()])
    if len(cameras) == 1:
        camera.update(cameras[0])
    else:
        camera["panels"] = cameras
    case["receipt"].update(
        material_legend=[
            dict(label=key[0], display_rgb=list(color)) for key, color in sorted(materials.items())
        ],
        output_sha256=_sha(output),
        camera=camera,
        layout=dict(
            panel_widths=list(panel_widths),
            transparent=False,
            projected_fit=True,
            view_frames=dict(VIEW_FRAME_STYLE),
            panel_corner_radius_in=0.10,
            axis_colors=dict(AXIS_COLORS),
        ),
        motion_details=motion_details,
        mesh_snapshot_styles=mesh_snapshot_styles,
        topology_options=dict(
            pose_samples=pose_samples,
            motion_detail=motion_detail,
            fixed_target_orientation=fixed_target_orientation,
            actor_label_mode=options.get("actor_label_mode", "role"),
        ),
        renderer_sha256=_sha(Path(__file__)),
        geometry_loader_sha256=_sha(Path(__file__).with_name("topology_scene.py")),
        node_label_helper_sha256=_sha(
            Path(__file__).resolve().parents[2] / "shared/node_labels.py"
        ),
    )
    if motion_details or faded_actors:
        case["receipt"]["motion_renderer_sha256"] = _sha(
            Path(__file__).with_name("topology_motion.py")
        )


def write_receipt(case: dict, output: Path, command: list[str]) -> None:
    """Record the inputs, prepared poses and image identity of one rendering."""
    for package in ("numpy", "matplotlib", "sionna-rt", "open3d"):
        try:
            case["receipt"].setdefault("versions", {})[package] = importlib.metadata.version(
                package
            )
        except importlib.metadata.PackageNotFoundError:
            pass
    case["receipt"]["command"] = command
    output.with_suffix(".json").write_text(json.dumps(case["receipt"], indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", type=Path, required=True, help="Scenario directory or YAML path"
    )
    parser.add_argument(
        "--output", type=Path, help="PNG/PDF path; default: scenario/summary/topology.png"
    )
    parser.add_argument("--view", choices=("3d", "top", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--pose-samples",
        type=int,
        choices=(3, 5),
        help="Override the number of authored animated-target poses in the topology figure",
    )
    parser.add_argument(
        "--motion-detail",
        choices=("auto", "true", "false"),
        help="Override the configured target motion detail without changing the scenario YAML",
    )
    args = parser.parse_args(argv)
    root = args.scenario if args.scenario.is_dir() else args.scenario.parent
    suffix = {"3d": "3d", "top": "2d", "both": ""}[args.view]
    output = args.output or root / "summary" / f"topology{suffix}.png"
    case = load_topology(args.scenario, pose_samples=args.pose_samples)
    if args.motion_detail is not None:
        case.setdefault("visualization", {})["topology_motion_detail"] = {
            "auto": "auto",
            "true": True,
            "false": False,
        }[args.motion_detail]
    render_topology(case, output, view=args.view, dpi=args.dpi)
    command = [
        "python",
        "-m",
        "generator.figures.topology",
        "--scenario",
        str(args.scenario),
        "--output",
        str(output),
        "--view",
        args.view,
        "--dpi",
        str(args.dpi),
    ]
    if args.motion_detail is not None:
        command.extend(["--motion-detail", args.motion_detail])
    if args.pose_samples is not None:
        command.extend(["--pose-samples", str(args.pose_samples)])
    write_receipt(case, output, command)
    get_logger(__name__).info("Topology figure: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
