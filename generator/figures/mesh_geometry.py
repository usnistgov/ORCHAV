"""Mesh-only drawing helpers shared by installed figures and scenario scripts.

These operations preserve source geometry; they neither acquire nor process
sensing measurements. Scenario assets remain explicit source-tree inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d

COLORS = {"Car": "#087b9d", "Pedestrian": "#d56335"}
RADAR = "#783b81"
INK = "#27333b"
PALETTE = {
    "buildings_concrete": "#d9ccb4",
    "buildings_glass": "#b4cbd1",
    "buildings_metal": "#d9ccb4",
    "ground": "#eeece5",
    "streets": "#929ca3",
}


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Use the normal ORCHAV mesh dependency, or its existing trimesh extra."""
    try:
        import open3d as o3d
    except ImportError:
        import trimesh

        mesh = trimesh.load(path, process=False, force="mesh")
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(faces):
        raise ValueError(f"Mesh contains no triangles: {path}")
    return vertices, faces


def crease_segments(vertices: np.ndarray, faces: np.ndarray, angle_deg: float = 28.0):
    """Return boundaries and true creases, excluding coplanar triangle diagonals.

    Weld coincident coordinates only for adjacency, without changing the mesh.
    Opposite triangle winding is ignored when comparing adjacent face normals.
    """
    if not 0.0 < angle_deg < 90.0:
        raise ValueError("Crease angle must be between 0 and 90 degrees")
    _, representative, inverse = np.unique(
        np.round(vertices, 6), axis=0, return_index=True, return_inverse=True
    )
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    adjacency = {}
    for index, triangle in enumerate(inverse[faces]):
        if lengths[index] < 1e-10:
            continue
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            if a != b:
                adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    threshold = np.cos(np.deg2rad(angle_deg))
    retained = []
    for edge, adjacent in adjacency.items():
        crease = len(adjacent) == 1 or any(
            abs(float(normals[a] @ normals[b])) < threshold
            for a in adjacent
            for b in adjacent
            if a < b
        )
        if crease:
            retained.append(vertices[representative[list(edge)]])
    return np.asarray(retained, dtype=float).reshape(-1, 2, 3)


def tube_faces(segments: np.ndarray, radius: float, sides: int = 5) -> np.ndarray:
    """Small polygonal cylinders share scene depth ordering with mesh faces."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if not len(segments):
        return np.empty((0, 3, 3))
    direction = segments[:, 1] - segments[:, 0]
    length = np.linalg.norm(direction, axis=1)
    segments, direction, length = (
        segments[length > 1e-9],
        direction[length > 1e-9],
        length[length > 1e-9],
    )
    direction /= length[:, None]
    reference = np.tile([0.0, 0.0, 1.0], (len(direction), 1))
    reference[np.abs(direction[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
    u = np.cross(direction, reference)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(direction, u)
    angles = np.arange(sides) * 2 * np.pi / sides
    ring = radius * (
        np.cos(angles)[None, :, None] * u[:, None] + np.sin(angles)[None, :, None] * v[:, None]
    )
    start, end = segments[:, 0, None] + ring, segments[:, 1, None] + ring
    faces = []
    for j in range(sides):
        k = (j + 1) % sides
        faces.extend(
            (
                np.stack((start[:, j], end[:, j], end[:, k]), axis=1),
                np.stack((start[:, j], end[:, k], start[:, k]), axis=1),
            )
        )
    return np.concatenate(faces)


def clip_triangles(triangles: np.ndarray, bounds: tuple[float, ...]):
    """Clip to a local XY window, retaining actual sloping roof/wall geometry."""
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    if not len(triangles):
        return triangles
    limits = np.asarray(bounds).reshape(-1, 2)
    coordinates = triangles[:, :, : len(limits)]
    inside = np.all((coordinates >= limits[:, 0]) & (coordinates <= limits[:, 1]), axis=(1, 2))
    outside = np.any(
        (coordinates.max(1) < limits[:, 0]) | (coordinates.min(1) > limits[:, 1]), axis=1
    )
    result = list(triangles[inside])
    for triangle in triangles[~inside & ~outside]:
        polygon = list(triangle)
        for axis, limit, sign in tuple(
            (i // 2, value, 1 if i % 2 == 0 else -1) for i, value in enumerate(bounds)
        ):
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                inside_a = sign * (a[axis] - limit) >= -1e-10
                inside_b = sign * (b[axis] - limit) >= -1e-10
                if inside_a:
                    clipped.append(a)
                if inside_a != inside_b:
                    fraction = (limit - a[axis]) / (b[axis] - a[axis])
                    clipped.append(a + fraction * (b - a))
            polygon = clipped
            if not polygon:
                break
        for index in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[index], polygon[index + 1]])
    return np.asarray(result, dtype=float).reshape(-1, 3, 3)


def _background_order(name: str, vertices: np.ndarray):
    """Flat ground and lifted roads are ordered behind all above-ground meshes."""
    expected = {"ground": (0, 0.0), "streets": (1, 0.15)}
    if name not in expected:
        return None
    order, height = expected[name]
    if not np.allclose(vertices[:, 2], height, atol=1e-6):
        raise ValueError("This overview requires the authored flat ground and road layers")
    return order


def _shaded(triangles: np.ndarray, color: str, strength: float = 0.24):
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    light = np.array([-0.45, -0.5, 0.74])
    # Cached facades can have inconsistent winding. Two-sided diffuse shading
    # gives coplanar triangles identical tones instead of diagonal light patches.
    shade = (1 - strength) + strength * np.abs(normals @ light)
    return np.asarray(to_rgb(color))[None] * shade[:, None]


def _plot_route(ax, route, bounds, color):
    """Clip mapped context to the displayed window without extending motion."""
    for a, b in zip(route[:-1], route[1:]):
        low, high = 0.0, 1.0
        if bounds is not None:
            delta = b - a
            for axis in range(3):
                if abs(delta[axis]) < 1e-12:
                    if not bounds[2 * axis] <= a[axis] <= bounds[2 * axis + 1]:
                        high = -1.0
                        break
                    continue
                near = (bounds[2 * axis] - a[axis]) / delta[axis]
                far = (bounds[2 * axis + 1] - a[axis]) / delta[axis]
                low, high = max(low, min(near, far)), min(high, max(near, far))
        if low <= high:
            points = np.stack((a + low * (b - a), a + high * (b - a)))
            ax.plot(*points.T, color=color, lw=1.8, linestyle=(0, (3, 3)), alpha=0.7, zorder=4)


def _add_mesh_layer(layers, triangles, color, bounds=None, shade=True):
    if bounds is not None:
        triangles = clip_triangles(triangles, bounds)
    if len(triangles):
        colors = _shaded(triangles, color) if shade else np.tile(to_rgb(color), (len(triangles), 1))
        layers.append((triangles, colors))


def _annotate(ax, xyz, text, xytext, color=INK, fontsize=11):
    x, y, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    ax.annotate(
        text,
        (x, y),
        xytext=xytext,
        textcoords="axes fraction",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 3.5},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9, "shrinkA": 4, "shrinkB": 4},
        annotation_clip=False,
    )
