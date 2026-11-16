"""Draw native scene faces and creases without a Python pixel rasterizer.

Like the topology figures, these views submit mesh collections to Matplotlib.
The XY view retains the original triangles and sorts them by mean height;
open spaces and disconnected buildings are never filled by a convex hull.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..mesh_geometry import _shaded, crease_segments


@dataclass
class MeshScene:
    faces: np.ndarray
    colors: np.ndarray
    edges: np.ndarray
    points: np.ndarray
    bounds: np.ndarray


def prepare_mesh_scene(scene_geometry, color_for_mesh) -> MeshScene:
    """Prepare exact world-space geometry once for reuse across figure panels."""
    faces, colors, edges, points, bounds = [], [], [], [], []
    for entry in scene_geometry:
        mesh = entry["mesh"]
        vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
        if not len(vertices):
            continue
        bounds.append(vertices[np.isfinite(vertices).all(axis=1)])
        indices = np.asarray(getattr(mesh, "triangles", getattr(mesh, "faces", [])), dtype=int)
        if not indices.size:
            points.append(vertices[np.isfinite(vertices).all(axis=1)])
            continue
        indices = indices.reshape(-1, 3)
        triangles = vertices[indices]
        valid = np.isfinite(triangles).all(axis=(1, 2))
        triangles, indices = triangles[valid], indices[valid]
        faces.append(triangles)
        colors.append(_shaded(triangles, color_for_mesh(entry)))
        edges.append(crease_segments(vertices, indices))
    triangles = np.concatenate(faces) if faces else np.empty((0, 3, 3))
    shades = np.concatenate(colors) if colors else np.empty((0, 3))
    order = np.argsort(triangles[:, :, 2].mean(axis=1), kind="stable")
    vertices = np.concatenate(bounds) if bounds else np.empty((0, 3))
    return MeshScene(
        faces=triangles[order],
        colors=shades[order],
        edges=np.concatenate(edges) if edges else np.empty((0, 2, 3)),
        points=np.concatenate(points) if points else np.empty((0, 3)),
        bounds=(
            np.array([vertices.min(axis=0), vertices.max(axis=0)])
            if len(vertices)
            else np.empty((0, 3))
        ),
    )


def draw_mesh_scene(ax, scene: MeshScene, *, style="fill", alpha=1.0, zorder=0, floor=False):
    """Draw filled faces or true creases on a 2D/3D axis in world coordinates."""
    is_3d = getattr(ax, "name", "") == "3d"
    faces, edges = scene.faces, scene.edges
    if floor and len(scene.bounds):
        faces, edges = faces.copy(), edges.copy()
        faces[:, :, 2] = edges[:, :, 2] = scene.bounds[0, 2]
    if style == "fill" and len(faces):
        collection_type = Poly3DCollection if is_3d else PolyCollection
        collection = collection_type(
            faces if is_3d else faces[:, :, :2],
            facecolors=scene.colors,
            edgecolors="none",
            linewidths=0,
            antialiased=False,
            alpha=alpha,
            zorder=zorder,
        )
        if is_3d:
            ax.add_collection3d(collection)
        else:
            ax.add_collection(collection)
    if len(edges):
        collection_type = Line3DCollection if is_3d else LineCollection
        collection = collection_type(
            edges if is_3d else edges[:, :, :2],
            colors="#41484c",
            linewidths=0.5,
            alpha=alpha,
            zorder=zorder + 0.1,
        )
        if is_3d:
            ax.add_collection3d(collection)
        else:
            ax.add_collection(collection)
    if len(scene.points):
        coordinates = scene.points.T if is_3d else scene.points[:, :2].T
        ax.scatter(*coordinates, s=1, color="#41484c", alpha=alpha, zorder=zorder)
    if len(scene.bounds):
        if is_3d:
            ax.auto_scale_xyz(*scene.bounds.T)
        else:
            ax.update_datalim(scene.bounds[:, :2])
            ax.autoscale_view()
