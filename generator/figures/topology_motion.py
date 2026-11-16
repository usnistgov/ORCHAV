"""Optional true-scale motion details for native rigid target poses."""

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb

from .mesh_geometry import crease_segments

POSE_OPACITIES = (0.16, 0.42, 0.88)
POSE_LABELS = ("Start", "Middle", "End")


def annotate_pose_times(ax, anchors, footprint, color, *, side="right"):
    """Label three native poses outside their projected mesh, with separated leaders."""
    anchors = np.asarray(anchors)
    screen = ax.transData.transform(anchors)
    outline = ax.transData.transform(footprint)
    pixels_per_point = ax.figure.dpi / 72
    right = side == "right"
    x = outline[:, 0].max() if right else outline[:, 0].min()
    x += (8 if right else -8) * pixels_per_point
    order = np.argsort(screen[:, 1], kind="stable")
    y = screen[order, 1].copy()
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + 14 * pixels_per_point)
    y += screen[:, 1].mean() - y.mean()
    for i, height in zip(order, y):
        ax.annotate(
            rf"$t_{{\mathrm{{{POSE_LABELS[i].lower()}}}}}$",
            xy=anchors[i],
            xytext=(np.array([x, height]) - screen[i]) / pixels_per_point,
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7),
            annotation_clip=False,
            zorder=30,
        )


def rigid_pose_style(actor: dict, sample: dict) -> tuple[str, float]:
    """Identify the three native rigid poses without changing their geometry."""
    index = next(
        i for i, item in enumerate(actor["mesh_samples"]) if item["index"] == sample["index"]
    )
    slot = 0 if index == 0 else 2 if index == len(actor["mesh_samples"]) - 1 else 1
    return POSE_LABELS[slot], POSE_OPACITIES[slot]


def transparent_mesh_faces(mesh: np.ndarray) -> list[np.ndarray]:
    """Join convex coplanar triangle pairs to avoid transparent diagonal seams.

    Only triangles sharing an edge can merge. The area check rejects concave
    unions; curved, disconnected and degenerate faces retain their triangles.
    Original vertices and the represented surface are preserved.
    """
    vertices, inverse = np.unique(mesh.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(areas[:, None], 1e-12)
    edges = {}
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((a, b))), []).append(i)
    merged, polygons = set(), []
    for adjacent in edges.values():
        if len(adjacent) != 2:
            continue
        a, b = adjacent
        if a in merged or b in merged or min(areas[a], areas[b]) < 1e-12:
            continue
        if abs(normals[a] @ normals[b]) < 1 - 1e-10:
            continue
        if not np.allclose((mesh[b] - mesh[a, 0]) @ normals[a], 0, atol=1e-10, rtol=0):
            continue
        ids = np.unique(np.concatenate((faces[a], faces[b])))
        if len(ids) != 4:
            continue
        polygon = vertices[ids]
        centered = polygon - polygon.mean(0)
        u = centered[0] / np.linalg.norm(centered[0])
        v = np.cross(normals[a], u)
        polygon = polygon[np.argsort(np.arctan2(centered @ v, centered @ u))]
        sides = np.roll(polygon, -1, axis=0) - polygon
        turns = np.cross(sides, np.roll(sides, -1, axis=0)) @ normals[a]
        if not (np.all(turns > 0) or np.all(turns < 0)):
            continue
        centered = polygon - polygon.mean(0)
        area = abs(np.cross(centered, np.roll(centered, -1, axis=0)).sum(0) @ normals[a])
        if not np.isclose(area, areas[a] + areas[b], rtol=1e-10, atol=1e-12):
            continue
        polygons.append(polygon)
        merged.update((a, b))
    polygons.extend(triangle for i, triangle in enumerate(mesh) if i not in merged)
    return polygons


def overlapping_rigid_snapshots(actor: dict) -> bool:
    """Find rigid sampled poses whose bounding boxes overlap or nearly touch.

    A five-percent mesh-size gap also needs pose shading to remain distinguishable
    at overview scale. Articulated meshes keep their separate pose rendering.
    """
    if (
        actor["role"] != "target"
        or actor["stationary"]
        or actor["animated"]
        or not actor["physical_path"]
    ):
        return False
    meshes = [
        mesh.reshape(-1, 3)
        for mesh, sample in zip(actor["meshes"], actor["mesh_samples"])
        if sample["topology_visible"]
    ]
    bounds = [(mesh.min(0), mesh.max(0)) for mesh in meshes]
    for i, (low, high) in enumerate(bounds):
        for other_low, other_high in bounds[i + 1 :]:
            gap = np.maximum(np.maximum(low - other_high, other_low - high), 0)
            tolerance = 0.05 * min(max(high - low), max(other_high - other_low))
            if np.linalg.norm(gap) <= tolerance:
                return True
    return False


def motion_detail_enabled(actor, setting, *, sensing_enabled, scene_span):
    """Select small physical translations, with explicit boolean overrides."""
    if (
        actor["role"] != "target"
        or actor["stationary"]
        or actor["animated"]
        or not actor["physical_path"]
        or np.linalg.norm(actor["positions"][-1] - actor["positions"][0]) <= 1e-8
    ):
        return False
    if setting != "auto":
        return bool(setting)
    if not sensing_enabled:
        return False
    # Overlapping mesh snapshots or travel occupying <=5% of the overview are
    # hard to read. Cap the latter at three target lengths so the close-up can
    # still show recognizable geometry. Use the whole trajectory, so a long
    # loop is not called small merely because it ends near its initial position.
    target_span = float(np.ptp(actor["meshes"][0].reshape(-1, 3), axis=0).max())
    excursion = float(np.linalg.norm(actor["positions"] - actor["positions"][0], axis=1).max())
    return excursion <= 3 * target_span and (
        overlapping_rigid_snapshots(actor) or excursion <= max(target_span, 0.05 * scene_span)
    )


def draw_motion_detail(fig, actor, bounds):
    """Draw projected poses without changing their relative placement."""
    positions = actor["positions"]
    delta = positions[-1] - positions[0]
    distance = float(np.linalg.norm(delta))
    size = np.ptp(actor["meshes"][0].reshape(-1, 3), axis=0)
    # A side view keeps tall meshes (such as people) recognizable. Otherwise
    # prefer the top view unless the motion is predominantly vertical.
    side_view = size[2] > 1.5 * max(size[:2]) or abs(delta[2]) > np.linalg.norm(delta[:2])
    dims = [int(np.argmax(abs(delta[:2]))), 2] if side_view else [0, 1]
    direction = delta[dims] / np.linalg.norm(delta[dims])
    normal = np.array([-direction[1], direction[0]])
    fade = overlapping_rigid_snapshots(actor)
    meshes = actor["meshes"] if fade else [actor["meshes"][0], actor["meshes"][-1]]
    origin = positions[0, dims]
    projected = [mesh[:, :, dims] - origin for mesh in meshes]
    vertices = np.concatenate(projected).reshape(-1, 2)
    lo, hi = vertices.min(0), vertices.max(0)
    span = max(hi - lo)
    ax = fig.add_axes(bounds, facecolor="none")
    color = actor["color"]
    for i, (mesh, xy) in enumerate(zip(meshes, projected)):
        opacity = rigid_pose_style(actor, actor["mesh_samples"][i])[1] if fade else 1
        if fade:
            xy = [polygon[:, dims] - origin for polygon in transparent_mesh_faces(mesh)]
            fill = (*to_rgb(color), opacity)
        else:
            fill = np.array(to_rgb(color)) * (0.08 if i == 0 else 0.18)
            fill += 1 - (0.08 if i == 0 else 0.18)
        ax.add_collection(PolyCollection(xy, facecolors=[fill], edgecolors="none", zorder=i))
        edges = crease_segments(mesh.reshape(-1, 3), np.arange(mesh.size // 3).reshape(-1, 3))
        ax.add_collection(
            LineCollection(
                edges[:, :, dims] - origin,
                colors=(*to_rgb(color), opacity),
                linewidths=1.1,
                linestyles="-" if fade else "--" if i == 0 else "-",
                zorder=3 + i,
            )
        )
    path = positions[:, dims] - origin
    ax.plot(*path.T, color=color, lw=1.2, zorder=5)
    for position, marker, marker_size, zorder in ((path[0], "o", 50, 10), (path[-1], "s", 23, 11)):
        ax.scatter(
            *position,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            s=marker_size,
            zorder=zorder,
        )
    offset = vertices @ normal
    dimension_offset = float(offset.max()) + span * 0.20
    start = normal * dimension_offset
    end = delta[dims] + start
    for center, tip in zip((np.zeros(2), delta[dims]), (start, end)):
        ax.plot(
            *np.array([center + normal * (offset.max() + span * 0.05), tip]).T,
            color="#75858d",
            lw=0.7,
        )
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.3, shrinkA=0, shrinkB=0),
    )
    middle = (start + end) / 2 + normal * span * 0.12
    distance_label = f"{distance:.3g} m"
    if not np.isclose(distance, np.linalg.norm(delta[dims]), rtol=1e-4):
        distance_label += " (3D)"
    ax.text(
        *middle,
        distance_label,
        ha="left" if normal[0] > 0.5 else "right" if normal[0] < -0.5 else "center",
        va="bottom" if normal[1] > 0.5 else "top" if normal[1] < -0.5 else "center",
        fontsize=9,
        color=color,
    )
    limits = np.vstack([vertices, path, start, end, middle])
    ax.set(
        xlim=(limits[:, 0].min() - span * 0.22, limits[:, 0].max() + span * 0.22),
        ylim=(limits[:, 1].min() - span * 0.22, limits[:, 1].max() + span * 0.22),
        aspect="equal",
    )
    ax.set_axis_off()
    ax.apply_aspect()
    indices = [sample["index"] for sample in actor["mesh_samples"]]
    if fade and len(indices) == 3:
        annotate_pose_times(ax, positions[indices][:, dims] - origin, vertices, color, side="left")
    result = {
        "name": actor["name"],
        "projection_axes": ["XYZ"[i] for i in dims],
        "displacement_m": distance,
        "start_position_m": positions[0].tolist(),
        "end_position_m": positions[-1].tolist(),
        "start_time_s": actor["mesh_samples"][0]["time_s"],
        "end_time_s": actor["mesh_samples"][-1]["time_s"],
    }
    if fade:
        result["pose_times_s"] = {
            rigid_pose_style(actor, sample)[0]: sample["time_s"] for sample in actor["mesh_samples"]
        }
    return result
