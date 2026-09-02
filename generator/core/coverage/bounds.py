"""Coverage grid bounds resolution.

Coverage maps are sampled on a regular XY grid at one or more configured
heights. Callers may provide explicit bounds through ``coverage.grid.bbox_xy``;
otherwise this module derives bounds from the already-loaded Sionna/Mitsuba
scene and then from cached XML/Open3D geometry. The returned bbox includes X, Y,
and Z intervals because ``solver.py`` uses XY for the radio-map plane and Z to
choose coverage heights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shared.geometry.cache import compute_xyz_bounds_from_geometry, get_scene_geometry
from shared.logging import get_logger

from ..configuration import CoverageConfig

logger = get_logger(__name__)

__all__ = [
    "CoverageBBox",
    "CoverageBoundsError",
    "resolve_coverage_bbox",
]

# Auto-discovered bounds get padding so grid edges are not tight to walls/meshes.
SCENE_BOUNDS_PADDING_FRACTION = 0.1
MIN_HORIZONTAL_BOUNDS_PADDING_M = 5.0
MIN_VERTICAL_BOUNDS_PADDING_M = 1.0
MIN_BOUNDS_SPAN_M = 1e-3

CoverageBBox = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class CoverageBoundsError(ValueError):
    """Raised when coverage grid bounds cannot be resolved from config or scene."""


def resolve_coverage_bbox(
    coverage_config: CoverageConfig,
    *,
    scene: Any | None = None,
    scenario_context: Any | None = None,
) -> CoverageBBox:
    """Return explicit coverage bounds or derive them from available scene geometry.

    Explicit config bounds are authoritative. Auto mode prefers runtime mesh
    bounds from the loaded scene to avoid extra file parsing, then falls back to
    the shared XML geometry cache when runtime objects do not expose mesh bounds.
    """
    # Non-``None`` means the user provided numeric YAML bounds such as
    # ``coverage.grid.bbox_xy: [[x_min, x_max], [y_min, y_max]]``. Treat those
    # as authoritative and skip all scene-bound discovery/cache work.
    if coverage_config.bbox is not None:
        bbox = coverage_config.bbox
        logger.info("Using provided bbox: %s", bbox)
        return bbox

    # Prefer runtime mesh bounds when Sionna/Mitsuba exposes them. Some scene
    # objects or test paths do not expose usable mesh bboxes, so XML geometry
    # loading remains a fallback rather than a separate "custom scene" path.
    bbox = bbox_from_loaded_scene_meshes(scene)
    if bbox is None:
        # Reuse ORCHAV's XML geometry loader when a scenario XML exists but the
        # already-loaded runtime scene cannot provide bounds.
        bbox = bbox_from_scenario_xml_geometry(scenario_context)
    if bbox is None:
        raise CoverageBoundsError(
            "coverage.grid.bbox_xy: auto requires scene geometry bounds. "
            "Set coverage.grid.bbox_xy explicitly for scenes without geometry bounds."
        )

    bbox = snap_bbox_to_resolution(bbox, coverage_config.resolution)
    logger.info("Auto-computed bbox from scene geometry: %s", bbox)
    return bbox


def bbox_from_loaded_scene_meshes(scene: Any | None) -> CoverageBBox | None:
    """Derive padded bounds from mesh bboxes exposed by a loaded Sionna scene."""
    objects = getattr(scene, "objects", None)
    if not objects:
        return None

    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    for obj in _iter_objects(objects):
        mesh = getattr(obj, "mi_mesh", None) or getattr(obj, "_mi_mesh", None)
        bbox_fn = getattr(mesh, "bbox", None)
        if bbox_fn is None:
            continue
        try:
            bbox = bbox_fn() if callable(bbox_fn) else bbox_fn
            min_point = _get_bbox_endpoint(bbox, "min")
            max_point = _get_bbox_endpoint(bbox, "max")
            if min_point is None or max_point is None:
                continue
            mins.append(min_point)
            maxs.append(max_point)
        except (TypeError, ValueError, AttributeError, IndexError):
            continue

    if not mins or not maxs:
        return None
    min_arr = np.asarray(mins, dtype=float)
    max_arr = np.asarray(maxs, dtype=float)
    valid = np.isfinite(min_arr).all(axis=1) & np.isfinite(max_arr).all(axis=1)
    if not np.any(valid):
        return None
    min_arr = min_arr[valid]
    max_arr = max_arr[valid]
    return _pad_bounds(
        float(np.nanmin(min_arr[:, 0])),
        float(np.nanmax(max_arr[:, 0])),
        float(np.nanmin(min_arr[:, 1])),
        float(np.nanmax(max_arr[:, 1])),
        float(np.nanmin(min_arr[:, 2])),
        float(np.nanmax(max_arr[:, 2])),
    )


def bbox_from_scenario_xml_geometry(scenario_context: Any | None) -> CoverageBBox | None:
    """Derive unpadded bounds from cached XML/Open3D geometry for a scenario."""
    scene_xml = getattr(scenario_context, "scene_xml", None)
    if scene_xml is None:
        return None

    try:
        geometry = get_scene_geometry(scenario_context=scenario_context)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("Could not load scene geometry for coverage bounds: %s", exc)
        return None

    bounds = compute_xyz_bounds_from_geometry(geometry or [])
    if bounds is None:
        return None
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def snap_bbox_to_resolution(
    bbox: CoverageBBox,
    resolution: tuple[float, ...],
) -> CoverageBBox:
    """Expand bounds outward so grid edges align with the configured resolution.

    Snapping outward preserves full scene coverage while making the grid origin
    and cell spacing deterministic for HDF5 metadata and visual overlays.
    """
    try:
        dx, dy = resolution if len(resolution) == 2 else (resolution[0], resolution[1])
        dz = resolution[2] if len(resolution) >= 3 else 1.0
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
        if dx > 0:
            x_min = math.floor(x_min / dx) * dx
            x_max = math.ceil(x_max / dx) * dx
        if dy > 0:
            y_min = math.floor(y_min / dy) * dy
            y_max = math.ceil(y_max / dy) * dy
        if dz > 0:
            z_min = math.floor(z_min / dz) * dz
            z_max = math.ceil(z_max / dz) * dz
        return (
            (float(x_min), float(x_max)),
            (float(y_min), float(y_max)),
            (float(z_min), float(z_max)),
        )
    except (TypeError, ValueError, IndexError):
        return bbox


def _iter_objects(objects: Any) -> list[Any]:
    """Normalize Sionna scene object containers across dict/list-like versions."""
    if isinstance(objects, dict):
        return list(objects.values())
    return list(objects)


def _get_bbox_endpoint(bbox: Any, attr: str) -> tuple[float, float, float] | None:
    """Return one bbox endpoint from Mitsuba-style method or attribute objects."""
    point = getattr(bbox, attr, None)
    if callable(point):
        point = point()
    if point is None:
        return None
    return (_coord(point, 0, "x"), _coord(point, 1, "y"), _coord(point, 2, "z"))


def _coord(point: Any, index: int, attr: str) -> float:
    """Read one coordinate from either named ``x/y/z`` fields or sequence access."""
    value = getattr(point, attr, None)
    if value is None:
        value = point[index]
    return float(value)


def _pad_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> CoverageBBox:
    """Apply horizontal and vertical padding policies to raw scene extents."""
    x_min, x_max = _pad_interval(x_min, x_max, min_pad=MIN_HORIZONTAL_BOUNDS_PADDING_M)
    y_min, y_max = _pad_interval(y_min, y_max, min_pad=MIN_HORIZONTAL_BOUNDS_PADDING_M)
    z_min, z_max = _pad_interval(z_min, z_max, min_pad=MIN_VERTICAL_BOUNDS_PADDING_M)
    return (
        (float(x_min), float(x_max)),
        (float(y_min), float(y_max)),
        (float(z_min), float(z_max)),
    )


def _pad_interval(
    min_value: float,
    max_value: float,
    *,
    min_pad: float,
) -> tuple[float, float]:
    """Pad an interval by 10 percent of span with a minimum absolute margin."""
    span = max(float(max_value) - float(min_value), MIN_BOUNDS_SPAN_M)
    pad = max(SCENE_BOUNDS_PADDING_FRACTION * span, min_pad)
    return float(min_value) - pad, float(max_value) + pad
