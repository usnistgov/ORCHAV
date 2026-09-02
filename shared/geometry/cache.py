#!/usr/bin/env python3
"""Shared scene-geometry cache and bounds helpers.

Provides one neutral place to load XML/Open3D scene geometry once and compute
XY/XYZ bounds for generator core, visualization summaries, and tools. Bounds are
padded so downstream grids and cameras have a stable margin around the physical
mesh extents.
"""

from pathlib import Path
from typing import Any

import numpy as np

from shared.cache_sizing import estimate_retained_bytes
from shared.logging import get_logger

logger = get_logger(__name__)

_GEOM_CACHE: dict[str, list[Any]] = {}
_XY_BOUNDS_CACHE: dict[int, Any] = {}
_XYZ_BOUNDS_CACHE: dict[int, Any] = {}
_GEOM_CACHE_REVISIONS: dict[
    str,
    tuple[tuple[str, int | None, int | None], ...],
] = {}
_GEOM_CACHE_HITS = 0
_GEOM_CACHE_MISSES = 0
_GEOM_CACHE_CLEARS = 0
_GEOM_CACHE_PEAK_BYTES = 0
_GEOMETRY_BUFFER_ATTRIBUTES = (
    "vertices",
    "triangles",
    "vertex_normals",
    "vertex_colors",
    "triangle_normals",
    "triangle_uvs",
    "triangle_material_ids",
)
# Sentinel value for "bounds computation was attempted and no usable vertices
# were found"; this lets repeated callers avoid re-parsing unusable geometry.
_CACHE_NONE = object()
DEFAULT_SCENE_HEIGHT_BOUNDS_M = (0.0, 3.0)
SCENE_BOUNDS_PADDING_FRACTION = 0.1
MIN_HORIZONTAL_BOUNDS_PADDING_M = 5.0
MIN_VERTICAL_BOUNDS_PADDING_M = 1.0
# XY-only coverage defaults use a tighter pad than full 3D scene framing.
XY_BOUNDS_PADDING_FRACTION = 0.05
MIN_XY_BOUNDS_PADDING_M = 2.0
MIN_BOUNDS_SPAN_M = 1e-3

XYBounds = tuple[float, float, float, float]
XYZBounds = tuple[float, float, float, float, float, float]


def _normalized_source_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve())
    except OSError:
        return str(candidate.absolute())


def _file_revision(path: str | Path) -> tuple[str, int | None, int | None]:
    """Return a cheap source-file identity for on-demand drift telemetry."""
    normalized = _normalized_source_path(path)
    try:
        stat = Path(normalized).stat()
    except OSError:
        return normalized, None, None
    return normalized, int(stat.st_size), int(stat.st_mtime_ns)


def _scene_source_revision(
    scene_xml: str | Path,
    scene_geometry: list[Any],
) -> tuple[tuple[str, int | None, int | None], ...]:
    """Return XML/mesh revisions already described by cached geometry entries."""
    paths = {_normalized_source_path(scene_xml)}
    for mesh_info in scene_geometry:
        if not isinstance(mesh_info, dict):
            continue
        for field in ("source_xml", "full_path"):
            value = mesh_info.get(field)
            if value:
                paths.add(_normalized_source_path(str(value)))
    return tuple(_file_revision(path) for path in sorted(paths))


def _known_geometry_buffers() -> list[np.ndarray]:
    """Return explicit CPU mesh buffers without traversing opaque objects."""
    buffers: list[np.ndarray] = []
    for scene_geometry in _GEOM_CACHE.values():
        for mesh_info in scene_geometry:
            mesh = mesh_info.get("mesh") if isinstance(mesh_info, dict) else None
            if mesh is None:
                continue
            for attribute in _GEOMETRY_BUFFER_ATTRIBUTES:
                try:
                    array = np.asarray(getattr(mesh, attribute))
                except (AttributeError, TypeError, ValueError):
                    continue
                if array.size and array.dtype != object:
                    buffers.append(array)
    return buffers


def _known_geometry_bytes() -> int:
    return estimate_retained_bytes(*_known_geometry_buffers())


def _update_geometry_peak() -> int:
    global _GEOM_CACHE_PEAK_BYTES

    current_bytes = _known_geometry_bytes()
    _GEOM_CACHE_PEAK_BYTES = max(_GEOM_CACHE_PEAK_BYTES, current_bytes)
    return current_bytes


def _geometry_cache_key(scene_geometry: list[Any]) -> int | None:
    try:
        return id(scene_geometry)
    except TypeError:
        return None


def clear_scene_geometry_cache() -> None:
    """Clear in-memory geometry and bounds caches.

    Primarily useful for tests and long-lived interactive sessions that change
    scene files in place.
    """
    global _GEOM_CACHE_CLEARS

    _GEOM_CACHE.clear()
    _XY_BOUNDS_CACHE.clear()
    _XYZ_BOUNDS_CACHE.clear()
    _GEOM_CACHE_REVISIONS.clear()
    _GEOM_CACHE_CLEARS += 1


def get_scene_geometry_cache_stats() -> dict[str, int]:
    """Return process-local geometry cache telemetry without invalidating data.

    Byte totals include known Open3D-style CPU mesh buffers only. File revision
    mismatches compare the cached XML/mesh stat signatures on demand and are a
    diagnostic signal; cache hits do not stat files or evict stale entries.
    """
    current_bytes = _update_geometry_peak()
    revision_mismatches = sum(
        _scene_source_revision(key, geometry) != _GEOM_CACHE_REVISIONS[key]
        for key, geometry in _GEOM_CACHE.items()
        if key in _GEOM_CACHE_REVISIONS
    )
    return {
        "entries": len(_GEOM_CACHE),
        "mesh_entries": sum(len(geometry) for geometry in _GEOM_CACHE.values()),
        "xy_bounds_entries": len(_XY_BOUNDS_CACHE),
        "xyz_bounds_entries": len(_XYZ_BOUNDS_CACHE),
        "current_bytes": current_bytes,
        "peak_bytes": _GEOM_CACHE_PEAK_BYTES,
        "hits": _GEOM_CACHE_HITS,
        "misses": _GEOM_CACHE_MISSES,
        "clears": _GEOM_CACHE_CLEARS,
        "evictions": 0,
        "revision_tracked_entries": len(_GEOM_CACHE_REVISIONS),
        "source_revision_mismatches": revision_mismatches,
    }


def _load_geometry(scene_xml: Path) -> list[Any] | None:
    """Load scene geometry through the optional Open3D-backed XML reader."""
    try:
        from shared.geometry.scene import load_scene_geometry
    except ImportError as e:
        logger.debug("Scene geometry loader unavailable: %s", e)
        return None
    try:
        return load_scene_geometry(str(scene_xml))
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not load geometry for %s: %s", scene_xml, e)
        return None


def get_scene_geometry(
    scenario_context: Any | None = None,
    scene_xml: str | Path | None = None,
) -> list[Any] | None:
    """Return cached scene geometry for the scenario if available.

    ``scenario_context.scene_xml`` is the primary source when available; direct
    ``scene_xml`` is useful for tests and tools that operate outside the full
    scenario loader.
    """
    global _GEOM_CACHE_HITS, _GEOM_CACHE_MISSES

    key = None
    if scenario_context is not None and getattr(scenario_context, "scene_xml", None):
        scene_xml = scenario_context.scene_xml
    if scene_xml is None:
        return None
    scene_xml = Path(scene_xml)
    key = str(scene_xml.resolve())
    if key in _GEOM_CACHE:
        _GEOM_CACHE_HITS += 1
        return _GEOM_CACHE[key]
    _GEOM_CACHE_MISSES += 1
    geom = _load_geometry(scene_xml)
    if geom is not None:
        _GEOM_CACHE[key] = geom
        _GEOM_CACHE_REVISIONS[key] = _scene_source_revision(scene_xml, geom)
        _update_geometry_peak()
    return geom


def _collect_valid_vertices(
    scene_geometry: list[Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Collect finite XY and Z vertex arrays from mesh metadata entries."""
    xy_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for mesh_info in scene_geometry:
        mesh = mesh_info.get("mesh") if isinstance(mesh_info, dict) else None
        if mesh is None:
            continue
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
        if vertices.size == 0 or vertices.ndim != 2 or vertices.shape[1] < 2:
            continue
        xy_values = vertices[:, :2]
        xy_valid = np.isfinite(xy_values).all(axis=1)
        if np.any(xy_valid):
            xy_chunks.append(xy_values[xy_valid])
        if vertices.shape[1] >= 3:
            z_values = vertices[:, 2]
            z_valid = xy_valid & np.isfinite(z_values)
            if np.any(z_valid):
                z_chunks.append(z_values[z_valid])

    xy = np.concatenate(xy_chunks, axis=0) if xy_chunks else None
    z = np.concatenate(z_chunks, axis=0) if z_chunks else None
    return xy, z


def _pad_interval(
    min_v: float,
    max_v: float,
    *,
    frac: float = SCENE_BOUNDS_PADDING_FRACTION,
    min_pad: float = MIN_HORIZONTAL_BOUNDS_PADDING_M,
) -> tuple[float, float]:
    span = max(max_v - min_v, MIN_BOUNDS_SPAN_M)
    pad = max(frac * span, min_pad)
    return min_v - pad, max_v + pad


def compute_xy_bounds_from_geometry(
    scene_geometry: list[Any],
) -> XYBounds | None:
    """Compute padded XY bounds from shared scene-geometry mesh metadata."""
    if not scene_geometry:
        return None

    key = _geometry_cache_key(scene_geometry)
    if key is not None:
        cached = _XY_BOUNDS_CACHE.get(key, None)
        if cached is _CACHE_NONE:
            return None
        if cached is not None:
            return cached
    try:
        xy, _ = _collect_valid_vertices(scene_geometry)
        if xy is None:
            result = None
        else:
            x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
            y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
            x_min, x_max = _pad_interval(
                x_min,
                x_max,
                frac=XY_BOUNDS_PADDING_FRACTION,
                min_pad=MIN_XY_BOUNDS_PADDING_M,
            )
            y_min, y_max = _pad_interval(
                y_min,
                y_max,
                frac=XY_BOUNDS_PADDING_FRACTION,
                min_pad=MIN_XY_BOUNDS_PADDING_M,
            )
            result = (x_min, x_max, y_min, y_max)
        if key is not None:
            _XY_BOUNDS_CACHE[key] = result if result is not None else _CACHE_NONE
        return result
    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
        logger.debug("Failed to compute XY bounds from geometry: %s", e)
        if key is not None:
            _XY_BOUNDS_CACHE[key] = _CACHE_NONE
        return None


def compute_xyz_bounds_from_geometry(
    scene_geometry: list[Any],
) -> XYZBounds | None:
    """Compute padded XYZ bounds from geometry meshes."""
    if not scene_geometry:
        return None

    key = _geometry_cache_key(scene_geometry)
    if key is not None:
        cached = _XYZ_BOUNDS_CACHE.get(key, None)
        if cached is _CACHE_NONE:
            return None
        if cached is not None:
            return cached
    try:
        xy, z = _collect_valid_vertices(scene_geometry)
        if xy is None:
            result = None
        else:
            x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
            y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
            if z is not None:
                z_min, z_max = float(np.min(z)), float(np.max(z))
            else:
                z_min, z_max = DEFAULT_SCENE_HEIGHT_BOUNDS_M
            x_min, x_max = _pad_interval(x_min, x_max)
            y_min, y_max = _pad_interval(y_min, y_max)
            z_min, z_max = _pad_interval(z_min, z_max, min_pad=MIN_VERTICAL_BOUNDS_PADDING_M)
            result = (x_min, x_max, y_min, y_max, z_min, z_max)
        if key is not None:
            _XYZ_BOUNDS_CACHE[key] = result if result is not None else _CACHE_NONE
        return result
    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
        logger.debug("Failed to compute XYZ bounds from geometry: %s", e)
        if key is not None:
            _XYZ_BOUNDS_CACHE[key] = _CACHE_NONE
        return None
