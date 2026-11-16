"""Renderer-neutral geometry from retained native sensing-target metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..types.render_payloads import MeshPayload
from .geometry_payload_factory import load_mesh_payload


def native_proxy_payload(metadata: Mapping[str, Any], mesh_path: Path | None = None) -> MeshPayload:
    """Build the recorded absorbing proxy, never a second scattering surface."""
    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Native sensing target has no retained geometry")
    if geometry.get("type") == "mesh":
        if mesh_path is None:
            raise ValueError("Native mesh proxy requires a resolved path")
        mesh = load_mesh_payload(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Native mesh proxy is empty")
        scaling = np.asarray(geometry.get("scaling", (1.0, 1.0, 1.0)), dtype=float)
        if scaling.shape != (3,) or not np.all(np.isfinite(scaling)) or np.any(scaling <= 0):
            raise ValueError("Native proxy scaling must be three positive finite values")
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        return MeshPayload(
            vertices=(vertices - center) * scaling + center,
            triangles=mesh.triangles,
            cache_key=f"native-proxy:{mesh_path}:{tuple(scaling)}",
        )
    if geometry.get("type") != "box":
        raise ValueError("Unknown native proxy geometry")
    dimensions = np.asarray(
        [geometry.get(key) for key in ("length_m", "width_m", "height_m")], dtype=float
    )
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("Native box requires retained positive effective dimensions")
    vertices = (
        np.array(
            [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=float,
        )
        * dimensions
        / 2
    )
    triangles = np.array(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    return MeshPayload(
        vertices=vertices, triangles=triangles, cache_key=f"native-box:{tuple(dimensions)}"
    )


def native_scattering_points(
    metadata: Mapping[str, Any], *, world: bool = True
) -> tuple[tuple[int, ...], np.ndarray]:
    """Read exact point IDs and positions without inferring visibility or RCS."""
    key = "scattering_points_world_m" if world else "scattering_points_local_m"
    raw = metadata.get(key, ())
    points = np.asarray(raw, dtype=np.float64)
    ids = tuple(metadata.get("scattering_point_ids", ()))
    if points.size == 0 and not ids:
        return (), np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Native scattering points must be finite three-vectors")
    if len(ids) != len(points) or len(set(ids)) != len(ids):
        raise ValueError("Native scattering point IDs must be unique and match positions")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
        for value in ids
    ):
        raise ValueError("Native scattering point IDs must be nonnegative integers")
    return tuple(int(value) for value in ids), points
