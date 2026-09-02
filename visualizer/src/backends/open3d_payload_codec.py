"""Open3D <-> payload conversion utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d

from ..types.render_payloads import (
    GeometryPayload,
    LineSetPayload,
    MeshPayload,
    OrientationFramePayload,
    PointCloudPayload,
)


def _writable_native_array(values: Any, dtype: np.dtype | type) -> np.ndarray:
    """Return a detached, C-contiguous buffer accepted by Open3D.

    Renderer-neutral payload arrays are deliberately read-only. Open3D's
    ``Vector*Vector`` constructors require writable NumPy storage, so the
    backend boundary owns the unavoidable copy instead of weakening payload
    immutability.
    """
    return np.array(values, dtype=dtype, copy=True, order="C")


def mesh_payload_to_o3d(payload: MeshPayload) -> o3d.geometry.TriangleMesh:
    """Convert MeshPayload to Open3D TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(_writable_native_array(payload.vertices, float))
    mesh.triangles = o3d.utility.Vector3iVector(_writable_native_array(payload.triangles, np.int32))

    if payload.normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            _writable_native_array(payload.normals, float)
        )
    if payload.vertex_colors is not None:
        colors = payload.vertex_colors
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(_writable_native_array(colors, float))
    if payload.triangle_uvs is not None:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            _writable_native_array(payload.triangle_uvs, float)
        )

    return mesh


def lines_payload_to_o3d(payload: LineSetPayload) -> o3d.geometry.LineSet:
    """Convert LineSetPayload to Open3D LineSet."""
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(_writable_native_array(payload.points, float))
    lineset.lines = o3d.utility.Vector2iVector(_writable_native_array(payload.lines, np.int32))
    if payload.colors is not None:
        colors = payload.colors
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]
        lineset.colors = o3d.utility.Vector3dVector(_writable_native_array(colors, float))
    return lineset


def points_payload_to_o3d(payload: PointCloudPayload) -> o3d.geometry.PointCloud:
    """Convert PointCloudPayload to Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_writable_native_array(payload.points, float))
    if payload.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(_writable_native_array(payload.colors, float))
    return pcd


def orientation_frame_payload_to_o3d(payload: OrientationFramePayload) -> o3d.geometry.TriangleMesh:
    """Convert an orientation-frame payload to Open3D's native coordinate frame."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(float(payload.size), 0.0))
    frame.compute_vertex_normals()
    return frame


def geometry_payload_to_o3d(payload: GeometryPayload | Any) -> Any:
    """Convert payload to Open3D geometry; passthrough if already native."""
    if isinstance(payload, MeshPayload):
        return mesh_payload_to_o3d(payload)
    if isinstance(payload, LineSetPayload):
        return lines_payload_to_o3d(payload)
    if isinstance(payload, PointCloudPayload):
        return points_payload_to_o3d(payload)
    if isinstance(payload, OrientationFramePayload):
        return orientation_frame_payload_to_o3d(payload)
    return payload


def o3d_mesh_to_payload(mesh: o3d.geometry.TriangleMesh) -> MeshPayload:
    """Convert Open3D TriangleMesh to MeshPayload."""
    normals = np.array(mesh.vertex_normals, copy=True) if mesh.has_vertex_normals() else None
    colors = np.array(mesh.vertex_colors, copy=True) if mesh.has_vertex_colors() else None
    uvs = np.array(mesh.triangle_uvs, copy=True) if mesh.has_triangle_uvs() else None
    return MeshPayload(
        vertices=np.array(mesh.vertices, copy=True),
        triangles=np.array(mesh.triangles, copy=True),
        normals=normals,
        vertex_colors=colors,
        triangle_uvs=uvs,
    )


def o3d_lines_to_payload(lineset: o3d.geometry.LineSet) -> LineSetPayload:
    """Convert Open3D LineSet to LineSetPayload."""
    colors = np.array(lineset.colors, copy=True) if len(lineset.colors) else None
    return LineSetPayload(
        points=np.array(lineset.points, copy=True),
        lines=np.array(lineset.lines, copy=True),
        colors=colors,
    )


def o3d_points_to_payload(pcd: o3d.geometry.PointCloud) -> PointCloudPayload:
    """Convert Open3D PointCloud to PointCloudPayload."""
    colors = np.array(pcd.colors, copy=True) if len(pcd.colors) else None
    return PointCloudPayload(points=np.array(pcd.points, copy=True), colors=colors)
