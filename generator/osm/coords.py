# Portions of this file are adapted from NVlabs/sionna-large-radio-maps
# (sionna_lrm/scene/utils.py), which carries the following notice:
#
#   Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#
# The NVIDIA code is itself adapted from geo2sigmap
# (https://github.com/functions-lab/geo2sigmap).
#
# See generator/osm/NOTICE.md for the full list of changes.

"""Coordinate utilities for GPS / UTM conversion and geometry helpers."""

from __future__ import annotations

import math

import numpy as np
import pyproj
from pyproj import CRS, Transformer
from shapely.geometry import LinearRing


def get_utm_crs(lon: float, lat: float) -> CRS:
    """Return the UTM CRS best matching a given lon/lat point."""
    utm_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    return CRS.from_epsg(utm_list[0].code)


def bbox_from_center(
    lat: float,
    lon: float,
    size_m: float,
    width_m: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute a GPS bounding box from a center point and size in meters.

    Parameters
    ----------
    lat, lon : float
        Center point in WGS84 degrees.
    size_m : float
        Side length in meters (used for both axes unless *width_m* is given).
    width_m : float, optional
        Width in meters (east-west). If ``None``, *size_m* is used.

    Returns
    -------
    tuple
        ``(min_lat, min_lon, max_lat, max_lon)`` in WGS84 degrees.
    """
    if width_m is None:
        width_m = size_m

    utm_crs = get_utm_crs(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(lon, lat)
    half_w, half_h = width_m / 2, size_m / 2

    min_lon, min_lat = to_wgs.transform(cx - half_w, cy - half_h)
    max_lon, max_lat = to_wgs.transform(cx + half_w, cy + half_h)
    return (min_lat, min_lon, max_lat, max_lon)


def unique_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove duplicate 2-D coordinates while preserving order."""
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for c in coords:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def reorder_localize(
    ring: LinearRing,
    cx: float,
    cy: float,
) -> list[tuple[float, float]]:
    """Ensure clockwise winding and translate to a local origin."""
    if hasattr(ring, "is_ccw") and ring.is_ccw:
        ring = LinearRing(list(ring.coords)[::-1])
    return [(x - cx, y - cy) for x, y in ring.coords]


def _ground_grid(
    bbox: tuple[float, float, float, float],
    to_utm: Transformer,
    origin: np.ndarray,
    pad_m: float,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build the XY grid and face indices shared by flat and terrain meshes.

    Returns (gx_local, gy_local, faces, nx, ny) where gx/gy are origin-
    subtracted UTM coordinate grids (ravelled).
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    corners = np.array(
        [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
        ]
    )
    x_utm, y_utm = to_utm.transform(corners[:, 0], corners[:, 1])
    x_min, x_max = x_utm.min() - pad_m, x_utm.max() + pad_m
    y_min, y_max = y_utm.min() - pad_m, y_utm.max() + pad_m

    nx = max(2, int(math.ceil((x_max - x_min) / resolution_m)) + 1)
    ny = max(2, int(math.ceil((y_max - y_min) / resolution_m)) + 1)

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")

    gx_local = gx.ravel() - origin[0]
    gy_local = gy.ravel() - origin[1]

    idx = np.arange(nx * ny).reshape(ny, nx)
    i0 = idx[:-1, :-1].ravel()
    i1 = idx[:-1, 1:].ravel()
    i2 = idx[1:, :-1].ravel()
    i3 = idx[1:, 1:].ravel()
    faces = np.concatenate(
        [
            np.stack([i0, i1, i2], axis=-1),
            np.stack([i1, i3, i2], axis=-1),
        ],
        axis=0,
    ).astype(np.int32)

    return gx_local, gy_local, faces, nx, ny


def flat_ground_mesh(
    bbox: tuple[float, float, float, float],
    to_utm: Transformer,
    origin: np.ndarray,
    pad_m: float = 200.0,
    resolution_m: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat triangulated ground plane in UTM coordinates.

    Parameters
    ----------
    bbox : tuple
        ``(min_lat, min_lon, max_lat, max_lon)`` in WGS84 degrees.
    to_utm : Transformer
        WGS84→UTM transformer.
    origin : ndarray
        3-element UTM origin ``[x, y, 0]`` subtracted from all vertices.
    pad_m : float
        Extra padding around the bounding box in meters.
    resolution_m : float
        Grid cell size in meters.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (M, 3)
    """
    gx_local, gy_local, faces, nx, ny = _ground_grid(
        bbox,
        to_utm,
        origin,
        pad_m,
        resolution_m,
    )
    vertices = np.stack(
        [gx_local, gy_local, np.zeros(nx * ny)],
        axis=-1,
    ).astype(np.float32)
    return vertices, faces


def terrain_ground_mesh(
    bbox: tuple[float, float, float, float],
    to_utm: Transformer,
    origin: np.ndarray,
    heightmap,
    origin_z: float,
    pad_m: float = 200.0,
    resolution_m: float = 50.0,
    return_grid: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, float | int | list[float]]]
):
    """Create a triangulated ground mesh with real terrain elevation.

    Parameters
    ----------
    bbox : tuple
        ``(min_lat, min_lon, max_lat, max_lon)`` in WGS84 degrees.
    to_utm : Transformer
        WGS84→UTM transformer.
    origin : ndarray
        3-element UTM origin ``[x, y, 0]`` subtracted from all vertices.
    heightmap : HeightMap
        Heightmap instance for elevation queries.
    origin_z : float
        Elevation at scene center (subtracted from all z-values).
    pad_m : float
        Extra padding around the bounding box in meters.
    resolution_m : float
        Grid cell size in meters.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (M, 3)
    """
    gx_local, gy_local, faces, nx, ny = _ground_grid(
        bbox,
        to_utm,
        origin,
        pad_m,
        resolution_m,
    )
    # Query real elevations at each grid vertex (UTM absolute coords).
    x_abs = gx_local + origin[0]
    y_abs = gy_local + origin[1]
    heights = heightmap.heights_from_utm_batch(x_abs, y_abs) - origin_z

    # Clamp invalid heights.
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    vertices = np.stack(
        [gx_local, gy_local, heights],
        axis=-1,
    ).astype(np.float32)
    if not return_grid:
        return vertices, faces

    grid_meta = {
        "x_min_local": float(gx_local.min()),
        "y_min_local": float(gy_local.min()),
        "x_step_m": float((gx_local.max() - gx_local.min()) / max(nx - 1, 1)),
        "y_step_m": float((gy_local.max() - gy_local.min()) / max(ny - 1, 1)),
        "nx": int(nx),
        "ny": int(ny),
        "heights_m": heights.astype(np.float32).tolist(),
    }
    return vertices, faces, grid_meta
