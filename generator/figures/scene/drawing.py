#!/usr/bin/env python3
"""Scene geometry drawing utilities for generator-side static figures.

The functions here draw ORCHAV scene geometry on Matplotlib axes for summary
figures, coverage overlays, and diagnostic comparisons. They are deliberately
best-effort: missing optional geometry libraries, very large meshes, or invalid
mesh data fall back to simpler drawing modes instead of failing the generator
run.

Two cache layers are used for expensive floor plans. In-memory caches reuse
geometry objects within one process, while persistent NPZ caches under
``ORCHAV_SUMMARY_CACHE_DIR`` or ``.cache/orchav/summary_geometry`` are keyed by
scene XML/mesh file signatures and render parameters.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from generator.core.pipeline.progress import StderrProgress
from shared.cache_sizing import estimate_retained_bytes
from shared.logging import get_logger

logger = get_logger(__name__)

_SCENE_2D_COMMAND_CACHE: Dict[int, Tuple[Tuple[str, Tuple[Any, ...], Dict[str, Any]], ...]] = {}
_RASTERIZED_SCENE_CACHE: Dict[
    Tuple[int, float, float, bool, Optional[Tuple[float, float]]],
    Tuple[np.ndarray, float, float, float, float],
] = {}
_VECTOR_CACHE_HITS = 0
_VECTOR_CACHE_MISSES = 0
_VECTOR_CACHE_WRITES = 0
_VECTOR_CACHE_PEAK_BYTES = 0
_RASTER_CACHE_HITS = 0
_RASTER_CACHE_MISSES = 0
_RASTER_CACHE_WRITES = 0
_RASTER_CACHE_PEAK_BYTES = 0
_FIGURE_CACHE_CLEARS = 0
_FIGURE_CACHE_PEAK_BYTES = 0
# Guardrails for scene summaries. Large map-derived scenes can contain enough
# vertices to make exact hulls or full-resolution rasters impractical.
MAX_RASTER_GRID_DIMENSION = 4000
DEFAULT_SCENE_SPAN_M = 100.0
CITY_BLOCK_CELL_DIVISOR = 90.0
MIN_CITY_BLOCK_CELL_SIZE_M = 8.0
MAX_3D_CITY_BLOCKS = 900
MAX_WIREFRAME_EDGES_TOTAL = 36_000
MAX_WIREFRAME_EDGES_PER_MESH = 5_000
MAX_MESH_POINTS_PER_MESH = 5_000
MAX_MESH_TRIANGLES = 30_000
MAX_HULL_INPUT_POINTS = 20_000
MAX_HULL_SIMPLICES = 20_000
MIN_RASTER_PROGRESS_PRIMITIVES = 200
_MATERIAL_COLOR_RULES: Tuple[Tuple[str, str, Tuple[float, float, float]], ...] = (
    ("glass", "Glass", (0.38, 0.62, 0.85)),
    ("brick", "Brick", (0.62, 0.28, 0.18)),
    ("metal", "Metal", (0.56, 0.59, 0.63)),
    ("marble", "Marble", (0.78, 0.74, 0.65)),
    ("wood", "Wood", (0.55, 0.36, 0.22)),
    ("concrete", "Concrete", (0.50, 0.50, 0.48)),
    ("asphalt", "Asphalt", (0.20, 0.20, 0.20)),
    ("vegetation", "Vegetation", (0.25, 0.48, 0.24)),
    ("water", "Water", (0.28, 0.52, 0.75)),
    ("ground", "Ground", (0.46, 0.42, 0.34)),
    ("plane", "Ground", (0.46, 0.42, 0.34)),
)
_SUMMARY_ENV_CACHE_VERSION = "summary-env-v1"
_AUTO_Z_RANGE_CACHE_POLICY = "scaled-floor-v1"


def _update_figure_cache_peaks() -> tuple[int, int, int]:
    global _FIGURE_CACHE_PEAK_BYTES
    global _RASTER_CACHE_PEAK_BYTES
    global _VECTOR_CACHE_PEAK_BYTES

    vector_bytes = (
        estimate_retained_bytes(_SCENE_2D_COMMAND_CACHE) if _SCENE_2D_COMMAND_CACHE else 0
    )
    raster_bytes = (
        estimate_retained_bytes(_RASTERIZED_SCENE_CACHE) if _RASTERIZED_SCENE_CACHE else 0
    )
    total_bytes = (
        estimate_retained_bytes(
            _SCENE_2D_COMMAND_CACHE,
            _RASTERIZED_SCENE_CACHE,
        )
        if _SCENE_2D_COMMAND_CACHE or _RASTERIZED_SCENE_CACHE
        else 0
    )
    _VECTOR_CACHE_PEAK_BYTES = max(_VECTOR_CACHE_PEAK_BYTES, vector_bytes)
    _RASTER_CACHE_PEAK_BYTES = max(_RASTER_CACHE_PEAK_BYTES, raster_bytes)
    _FIGURE_CACHE_PEAK_BYTES = max(_FIGURE_CACHE_PEAK_BYTES, total_bytes)
    return vector_bytes, raster_bytes, total_bytes


def clear_scene_figure_caches() -> None:
    """Clear process-local vector/raster caches while retaining telemetry peaks."""
    global _FIGURE_CACHE_CLEARS

    _SCENE_2D_COMMAND_CACHE.clear()
    _RASTERIZED_SCENE_CACHE.clear()
    _FIGURE_CACHE_CLEARS += 1


def get_scene_figure_cache_stats() -> dict[str, int]:
    """Return local in-memory figure cache telemetry.

    Persistent NPZ artifacts are intentionally excluded: that cache has no
    bounded inventory, and recursively scanning its root would make a stats
    query perform unrelated filesystem work.
    """
    vector_bytes, raster_bytes, total_bytes = _update_figure_cache_peaks()
    return {
        "vector_entries": len(_SCENE_2D_COMMAND_CACHE),
        "vector_current_bytes": vector_bytes,
        "vector_peak_bytes": _VECTOR_CACHE_PEAK_BYTES,
        "vector_hits": _VECTOR_CACHE_HITS,
        "vector_misses": _VECTOR_CACHE_MISSES,
        "vector_writes": _VECTOR_CACHE_WRITES,
        "raster_entries": len(_RASTERIZED_SCENE_CACHE),
        "raster_current_bytes": raster_bytes,
        "raster_peak_bytes": _RASTER_CACHE_PEAK_BYTES,
        "raster_hits": _RASTER_CACHE_HITS,
        "raster_misses": _RASTER_CACHE_MISSES,
        "raster_writes": _RASTER_CACHE_WRITES,
        "current_bytes": total_bytes,
        "peak_bytes": _FIGURE_CACHE_PEAK_BYTES,
        "clears": _FIGURE_CACHE_CLEARS,
        "evictions": 0,
    }


def _store_rasterized_scene(
    key: Tuple[int, float, float, bool, Optional[Tuple[float, float]]],
    value: Tuple[np.ndarray, float, float, float, float],
) -> None:
    global _RASTER_CACHE_WRITES

    _RASTERIZED_SCENE_CACHE[key] = value
    _RASTER_CACHE_WRITES += 1
    _update_figure_cache_peaks()


def _as_float3(values: Sequence[Any]) -> Tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _as_float6(values: Sequence[Any]) -> Tuple[float, float, float, float, float, float]:
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


class _RecordingAxis:
    """Minimal axis shim that records drawing commands for replay."""

    def __init__(self) -> None:
        self.commands: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def _record(self, method: str, args: Sequence[Any], kwargs: Dict[str, Any]) -> None:
        self.commands.append((method, tuple(args), dict(kwargs)))

    def scatter(self, *args, **kwargs):
        self._record("scatter", args, kwargs)

    def plot(self, *args, **kwargs):
        self._record("plot", args, kwargs)

    def fill(self, *args, **kwargs):
        self._record("fill", args, kwargs)


def _scene_cache_key(scene_geometry: Optional[List[Any]]) -> Optional[int]:
    if scene_geometry is None:
        return None
    try:
        return id(scene_geometry)
    except TypeError:
        return None


def _raster_cache_key(
    scene_geometry: Optional[List[Any]],
    *,
    resolution: float,
    blur_sigma: float,
    edge_enhancement: bool,
    z_range: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[int, float, float, bool, Optional[Tuple[float, float]]]]:
    scene_key = _scene_cache_key(scene_geometry)
    if scene_key is None:
        return None
    normalized_z_range = None if z_range is None else (float(z_range[0]), float(z_range[1]))
    return (
        scene_key,
        round(float(resolution), 6),
        round(float(blur_sigma), 6),
        bool(edge_enhancement),
        normalized_z_range,
    )


def _find_project_root() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


def _summary_cache_root() -> Optional[Path]:
    configured = os.environ.get("ORCHAV_SUMMARY_CACHE_DIR")
    root = Path(configured) if configured else _find_project_root() / ".cache" / "orchav"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Summary environment cache disabled: %s", exc)
        return None
    return root / "summary_geometry"


def _file_signature(path: Path, *, hash_contents: bool = False) -> Optional[Dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    signature: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        try:
            signature["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            pass
    return signature


def _scene_environment_signature(scene_geometry: Optional[List[Any]]) -> Optional[str]:
    """Return a stable cache signature for geometry source files and materials.

    The cache key is based on file identity/content, not Python object identity,
    so a rerun can reuse rasterized summaries when the scene XML and mesh files
    did not change.
    """
    if not scene_geometry:
        return None

    source_xml = None
    mesh_parts: List[Dict[str, Any]] = []
    for mesh_info in scene_geometry:
        if not isinstance(mesh_info, dict):
            continue
        if source_xml is None and mesh_info.get("source_xml"):
            source_xml = Path(str(mesh_info["source_xml"]))
        mesh_path = mesh_info.get("full_path")
        mesh_stat = _file_signature(Path(mesh_path)) if mesh_path else None
        mesh_parts.append(
            {
                "name": str(mesh_info.get("name", "")),
                "rel_path": str(mesh_info.get("rel_path", "")),
                "shape_id": str(mesh_info.get("shape_id", "")),
                "material_id": str(mesh_info.get("material_id", "")),
                "mesh": mesh_stat,
            }
        )

    xml_signature = _file_signature(source_xml, hash_contents=True) if source_xml else None
    if xml_signature is None and not mesh_parts:
        return None

    payload = {
        "version": _SUMMARY_ENV_CACHE_VERSION,
        "source_xml": xml_signature,
        "meshes": sorted(mesh_parts, key=lambda item: (item["rel_path"], item["name"])),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary_cache_path(
    scene_geometry: Optional[List[Any]], artifact: str, params: Dict[str, Any]
) -> Optional[Path]:
    """Return the persistent cache path for a summary-rendering artifact."""
    root = _summary_cache_root()
    signature = _scene_environment_signature(scene_geometry)
    if root is None or signature is None:
        return None
    params_payload = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    params_hash = hashlib.sha256(params_payload).hexdigest()[:16]
    return root / signature[:2] / signature / f"{artifact}_{params_hash}.npz"


def _load_npz_cache(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("Ignoring invalid summary environment cache %s: %s", path, exc)
        return None


def _save_npz_cache(path: Optional[Path], **arrays: Any) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Failed to write summary environment cache %s: %s", path, exc)


def _cached_rasterized_floor_plan(
    scene_geometry: List[Any],
    *,
    resolution: float,
    blur_sigma: float,
    edge_enhancement: bool,
    z_range: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[np.ndarray, float, float, float, float]]:
    """Return a cached rasterized floor plan or generate and persist one.

    The return tuple is ``(rgba_image, x_min, x_max, y_min, y_max)``. ``None``
    signals that callers should fall back to vector or mesh drawing.
    """
    global _RASTER_CACHE_HITS, _RASTER_CACHE_MISSES

    memory_key = _raster_cache_key(
        scene_geometry,
        resolution=resolution,
        blur_sigma=blur_sigma,
        edge_enhancement=edge_enhancement,
        z_range=z_range,
    )
    if memory_key is not None:
        if memory_key in _RASTERIZED_SCENE_CACHE:
            _RASTER_CACHE_HITS += 1
            logger.debug("[RASTERIZE] Using in-memory floor plan cache")
            return _RASTERIZED_SCENE_CACHE[memory_key]
        _RASTER_CACHE_MISSES += 1

    raster_cache_params = {
        "resolution": round(float(resolution), 6),
        "blur_sigma": round(float(blur_sigma), 6),
        "edge_enhancement": bool(edge_enhancement),
        "z_range": None if z_range is None else [float(z_range[0]), float(z_range[1])],
    }
    if z_range is None:
        raster_cache_params["auto_z_range_policy"] = _AUTO_Z_RANGE_CACHE_POLICY

    disk_path = _summary_cache_path(scene_geometry, "raster", raster_cache_params)
    cached = _load_npz_cache(disk_path)
    if cached is not None:
        try:
            result = (
                cached["image"],
                float(cached["extent"][0]),
                float(cached["extent"][1]),
                float(cached["extent"][2]),
                float(cached["extent"][3]),
            )
            if memory_key is not None:
                _store_rasterized_scene(memory_key, result)
            logger.info("[RASTERIZE] Loaded floor plan from persistent summary cache")
            return result
        except (KeyError, TypeError, ValueError, IndexError):
            pass

    logger.info("[RASTERIZE] Generating floor plan (this may take a few seconds)...")
    result = create_rasterized_floor_plan(
        scene_geometry,
        resolution=resolution,
        blur_sigma=blur_sigma,
        edge_enhancement=edge_enhancement,
        z_range=z_range,
    )
    if result is not None:
        if memory_key is not None:
            _store_rasterized_scene(memory_key, result)
        image, x_min, x_max, y_min, y_max = result
        _save_npz_cache(
            disk_path,
            image=image,
            extent=np.asarray([x_min, x_max, y_min, y_max], dtype=np.float64),
        )
    return result


def create_rasterized_floor_plan(
    scene_geometry: List[Any],
    resolution: float = 0.01,
    blur_sigma: float = 0.5,
    edge_enhancement: bool = True,
    z_range: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[np.ndarray, float, float, float, float]]:
    """Create a rasterized 2D floor plan from 3D scene geometry.

    Mesh triangles are projected onto an RGBA grid with a Z-buffer; meshes
    without triangles fall back to vertex rasterization. Grid dimensions are
    capped, so a large scene may use a coarser effective resolution than the
    requested value.

    Args:
        scene_geometry: List of mesh dictionaries with 'mesh', 'name', 'color'
        resolution: Requested grid resolution in meters per pixel (default: 0.01 m)
        blur_sigma: Gaussian blur sigma for anti-aliasing (0=off, 1=strong)
        edge_enhancement: If True, enhance edges for crisper boundaries
        z_range: Optional mesh-overlap interval ``(z_min, z_max)``. When omitted,
            the interval begins up to 0.1 m above the scene minimum (5% of
            scene height) and ends 2.5 m above that minimum for scenes shorter
            than 10 m, or at the scene maximum for taller scenes.

    Returns:
        ``(rgba_image, x_min, x_max, y_min, y_max)``, or ``None`` when no
        usable raster can be produced.
    """
    try:
        from scipy.ndimage import gaussian_filter, sobel

        # Step 1: Compute bounding box across all meshes
        all_vertices = []
        for mesh_info in scene_geometry:
            mesh = mesh_info["mesh"]
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                all_vertices.append(vertices)

        if not all_vertices:
            logger.warning("[RASTERIZE] No vertices found in scene geometry")
            return None

        all_vertices = np.vstack(all_vertices)
        x_coords = all_vertices[:, 0]
        y_coords = all_vertices[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Add padding (5% of range)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = max(x_range, y_range) * 0.05
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Auto-detect a useful vertical slice. Indoor scenes usually need
        # furniture/wall height, while outdoor scenes need full building height.
        if z_range is None:
            z_min_global = all_vertices[:, 2].min()
            z_max_global = all_vertices[:, 2].max()
            scene_height = z_max_global - z_min_global
            floor_offset = _scaled_floor_offset(scene_height)

            # Smart detection: indoor vs outdoor scenes
            # Indoor scenes: low ceilings (< 10m), filter to furniture height (0.1m to 2.5m)
            # Outdoor scenes: tall structures (>= 10m), include everything above ground
            if scene_height < 10.0:
                # Indoor scene - filter to furniture range
                z_range = (
                    z_min_global + floor_offset,
                    min(z_min_global + 2.5, z_max_global),
                )
                logger.info(
                    f"[RASTERIZE] Indoor scene detected (height={scene_height:.1f}m). "
                    f"Z range: {z_range[0]:.2f}m to {z_range[1]:.2f}m"
                )
            else:
                # Outdoor scene - include all structures above ground level
                z_range = (z_min_global + floor_offset, z_max_global)
                logger.info(
                    f"[RASTERIZE] Outdoor scene detected (height={scene_height:.1f}m). "
                    f"Z range: {z_range[0]:.2f}m to {z_range[1]:.2f}m (full height)"
                )
            logger.info(f"[RASTERIZE] Global Z range: {z_min_global:.2f}m to {z_max_global:.2f}m")

        # Filter meshes by Z range and compute their min Z for sorting
        filtered_meshes = []
        for mesh_info in scene_geometry:
            mesh = mesh_info["mesh"]
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                continue

            z_coords = vertices[:, 2]
            z_min_mesh = z_coords.min()
            z_max_mesh = z_coords.max()

            # Check if mesh overlaps with desired Z range
            if z_max_mesh >= z_range[0] and z_min_mesh <= z_range[1]:
                filtered_meshes.append((z_min_mesh, mesh_info))
                logger.debug(
                    f"[RASTERIZE] Including '{mesh_info.get('name', 'unknown')}' "
                    f"(Z: {z_min_mesh:.2f}m to {z_max_mesh:.2f}m)"
                )
            else:
                logger.debug(
                    f"[RASTERIZE] Filtering out '{mesh_info.get('name', 'unknown')}' "
                    f"(Z: {z_min_mesh:.2f}m to {z_max_mesh:.2f}m, outside range)"
                )

        # Sort by Z min (render from bottom to top)
        filtered_meshes.sort(key=lambda x: x[0])
        logger.info(
            f"[RASTERIZE] Rendering {len(filtered_meshes)}/{len(scene_geometry)} meshes "
            f"(filtered by Z range)"
        )

        grid_width = int(np.ceil((x_max - x_min) / resolution))
        grid_height = int(np.ceil((y_max - y_min) / resolution))

        max_dimension = MAX_RASTER_GRID_DIMENSION
        if grid_width > max_dimension or grid_height > max_dimension:
            scale_factor = max_dimension / max(grid_width, grid_height)
            grid_width = int(grid_width * scale_factor)
            grid_height = int(grid_height * scale_factor)
            resolution = (x_max - x_min) / grid_width
            logger.info(
                f"[RASTERIZE] Limiting grid size: {grid_width}x{grid_height}, resolution={resolution:.3f}m"
            )

        logger.info(
            f"[RASTERIZE] Creating {grid_width}x{grid_height} grid (resolution={resolution:.3f}m)"
        )

        raster_jobs = []
        total_primitives = 0
        for _, mesh_info in filtered_meshes:
            mesh = mesh_info["mesh"]
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                continue
            triangles = _mesh_triangles(mesh)
            primitive_count = len(triangles) if triangles is not None else len(vertices)
            total_primitives += primitive_count
            raster_jobs.append((mesh_info, vertices, triangles))

        progress = None
        completed_primitives = 0
        progress_interval = max(1, total_primitives // 100)
        if total_primitives >= MIN_RASTER_PROGRESS_PRIMITIVES:
            progress = StderrProgress(
                first_step=0,
                total_steps=total_primitives,
                label="Scene raster",
            )

        def note_raster_progress() -> None:
            nonlocal completed_primitives
            completed_primitives += 1
            if progress is None:
                return
            if (
                completed_primitives == total_primitives
                or completed_primitives % progress_interval == 0
            ):
                progress.update(completed_primitives - 1)

        # Create RGB grid for colored buildings + Z-buffer for occlusion
        occupancy_r = np.zeros((grid_height, grid_width), dtype=np.float32)
        occupancy_g = np.zeros((grid_height, grid_width), dtype=np.float32)
        occupancy_b = np.zeros((grid_height, grid_width), dtype=np.float32)
        occupancy_alpha = np.zeros((grid_height, grid_width), dtype=np.float32)
        z_buffer = np.full((grid_height, grid_width), -np.inf, dtype=np.float32)

        # Rasterize bottom-to-top with a z-buffer so higher geometry can occlude
        # lower geometry in the top-down texture.
        try:
            for mesh_info, vertices, triangles in raster_jobs:
                material_key, _, material_color = _scene_material_descriptor(
                    str(mesh_info.get("name", "")), mesh_info.get("material_id")
                )
                color = (
                    material_color
                    if material_key != "other"
                    else mesh_info.get("color", material_color)
                )
                if isinstance(color, str):
                    # Convert color name to RGB if needed
                    import matplotlib.colors as mcolors

                    try:
                        color = mcolors.to_rgb(color)
                    except (ValueError, TypeError):
                        color = [0.7, 0.7, 0.7]

                if triangles is not None and len(triangles) > 0:
                    # Rasterize triangles with Z-buffer
                    _rasterize_triangles(
                        vertices,
                        triangles,
                        occupancy_r,
                        occupancy_g,
                        occupancy_b,
                        occupancy_alpha,
                        z_buffer,
                        color,
                        x_min,
                        y_min,
                        resolution,
                        progress_callback=note_raster_progress,
                    )
                else:
                    # Fallback: rasterize vertices as points with Z-buffer
                    _rasterize_points(
                        vertices,
                        occupancy_r,
                        occupancy_g,
                        occupancy_b,
                        occupancy_alpha,
                        z_buffer,
                        color,
                        x_min,
                        y_min,
                        resolution,
                        progress_callback=note_raster_progress,
                    )
        finally:
            if progress is not None:
                progress.newline()

        if blur_sigma > 0:
            logger.debug(f"[RASTERIZE] Applying Gaussian blur (sigma={blur_sigma})")
            occupancy_r = gaussian_filter(occupancy_r, sigma=blur_sigma)
            occupancy_g = gaussian_filter(occupancy_g, sigma=blur_sigma)
            occupancy_b = gaussian_filter(occupancy_b, sigma=blur_sigma)
            occupancy_alpha = gaussian_filter(occupancy_alpha, sigma=blur_sigma)

        if edge_enhancement:
            logger.debug("[RASTERIZE] Applying edge enhancement")
            # Compute gradient magnitude
            alpha_grad_x = sobel(occupancy_alpha, axis=1, mode="constant")
            alpha_grad_y = sobel(occupancy_alpha, axis=0, mode="constant")
            edge_magnitude = np.sqrt(alpha_grad_x**2 + alpha_grad_y**2)

            # Normalize and enhance edges
            if edge_magnitude.max() > 0:
                edge_magnitude = edge_magnitude / edge_magnitude.max()
                # Darken edges slightly for definition
                edge_factor = 1.0 - (edge_magnitude * 0.3)
                occupancy_r *= edge_factor
                occupancy_g *= edge_factor
                occupancy_b *= edge_factor

        image = np.zeros((grid_height, grid_width, 4), dtype=np.float32)
        image[:, :, 0] = occupancy_r
        image[:, :, 1] = occupancy_g
        image[:, :, 2] = occupancy_b
        image[:, :, 3] = occupancy_alpha

        # Clip values to [0, 1]
        image = np.clip(image, 0, 1)

        occupancy = (occupancy_alpha > 0).sum() / occupancy_alpha.size
        if occupancy == 0:
            logger.info(
                "[RASTERIZE] Scene geometry projects to zero filled area; using vector fallback"
            )
            return None

        logger.info(
            f"[RASTERIZE] Successfully created floor plan: {grid_width}x{grid_height}, "
            f"occupancy: {occupancy * 100:.1f}%"
        )

        return (image, x_min, x_max, y_min, y_max)

    except ImportError as e:
        logger.warning(f"[RASTERIZE] Missing dependencies: {e}")
        return None
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ):  # defensive catch: plot resilience
        logger.warning("[RASTERIZE] Failed to create rasterized floor plan", exc_info=True)
        return None


def _scaled_floor_offset(scene_height: float) -> float:
    """Return a non-negative floor offset capped at the nominal 0.1 m."""
    return min(0.1, max(0.0, float(scene_height)) * 0.05)


def _mesh_triangles(mesh: Any) -> Optional[np.ndarray]:
    """Return mesh triangle indices when available."""
    try:
        if hasattr(mesh, "triangles"):
            triangles = np.asarray(mesh.triangles)
        elif hasattr(mesh, "faces"):
            triangles = np.asarray(mesh.faces)
        else:
            return None
    except (AttributeError, ValueError, TypeError):
        return None
    return triangles if len(triangles) > 0 else None


def _rasterize_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    occupancy_r: np.ndarray,
    occupancy_g: np.ndarray,
    occupancy_b: np.ndarray,
    occupancy_alpha: np.ndarray,
    z_buffer: np.ndarray,
    color: Sequence[float],
    x_min: float,
    y_min: float,
    resolution: float,
    progress_callback=None,
) -> None:
    """Rasterize mesh triangles onto occupancy grids using scan-line algorithm with Z-buffering."""
    grid_height, grid_width = occupancy_alpha.shape

    for tri_idx in triangles:
        # Get triangle vertices (X, Y, Z coordinates)
        if len(tri_idx) < 3:
            continue

        v0_3d = vertices[tri_idx[0]]
        v1_3d = vertices[tri_idx[1]]
        v2_3d = vertices[tri_idx[2]]

        v0 = v0_3d[:2]
        v1 = v1_3d[:2]
        v2 = v2_3d[:2]

        # Average Z height of triangle (for Z-buffering)
        z_avg = (v0_3d[2] + v1_3d[2] + v2_3d[2]) / 3.0

        # Convert to grid coordinates
        p0 = _world_to_grid(v0, x_min, y_min, resolution, grid_width, grid_height)
        p1 = _world_to_grid(v1, x_min, y_min, resolution, grid_width, grid_height)
        p2 = _world_to_grid(v2, x_min, y_min, resolution, grid_width, grid_height)

        # Compute bounding box
        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(grid_width - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(grid_height - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        # Fill triangle using barycentric coordinates with Z-test
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if _point_in_triangle((x, y), p0, p1, p2):
                    # Z-test: only render if this triangle is above previous content
                    if z_avg > z_buffer[y, x]:
                        occupancy_r[y, x] = color[0]
                        occupancy_g[y, x] = color[1]
                        occupancy_b[y, x] = color[2]
                        occupancy_alpha[y, x] = 1.0
                        z_buffer[y, x] = z_avg
        if progress_callback is not None:
            progress_callback()


def _rasterize_points(
    vertices: np.ndarray,
    occupancy_r: np.ndarray,
    occupancy_g: np.ndarray,
    occupancy_b: np.ndarray,
    occupancy_alpha: np.ndarray,
    z_buffer: np.ndarray,
    color: Sequence[float],
    x_min: float,
    y_min: float,
    resolution: float,
    progress_callback=None,
) -> None:
    """Rasterize mesh vertices as points (fallback when triangles unavailable) with Z-buffering."""
    grid_height, grid_width = occupancy_alpha.shape

    for vertex in vertices:
        px, py = _world_to_grid(vertex[:2], x_min, y_min, resolution, grid_width, grid_height)
        x, y = int(px), int(py)
        if 0 <= x < grid_width and 0 <= y < grid_height:
            z = vertex[2]
            # Z-test: only render if this point is above previous content
            if z > z_buffer[y, x]:
                occupancy_r[y, x] = color[0]
                occupancy_g[y, x] = color[1]
                occupancy_b[y, x] = color[2]
                occupancy_alpha[y, x] = 1.0
                z_buffer[y, x] = z
        if progress_callback is not None:
            progress_callback()


def _world_to_grid(
    point: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    width: int,
    height: int,
    y_max: float | None = None,
) -> Tuple[float, float]:
    """Convert world coordinates to raster grid coordinates.

    The resulting grid is later displayed with ``origin='lower'``, so the y-axis
    remains in scene coordinates rather than image row coordinates.
    """
    px = (point[0] - x_min) / resolution
    py = (point[1] - y_min) / resolution
    return (px, py)


def _point_in_triangle(
    p: Tuple[float, float],
    v0: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
) -> bool:
    """Check if point p is inside triangle (v0, v1, v2) using barycentric coordinates."""
    # Compute barycentric coordinates
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return False

    a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
    b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
    c = 1.0 - a - b

    return a >= 0 and b >= 0 and c >= 0


def _build_scene_geometry_2d_commands(scene_geometry: Optional[List[Any]]):
    """Record vector drawing commands once for replay on later Matplotlib axes."""
    global _VECTOR_CACHE_HITS, _VECTOR_CACHE_MISSES, _VECTOR_CACHE_WRITES

    if not scene_geometry:
        return tuple()

    key = _scene_cache_key(scene_geometry)
    if key is not None and key in _SCENE_2D_COMMAND_CACHE:
        _VECTOR_CACHE_HITS += 1
        return _SCENE_2D_COMMAND_CACHE[key]
    if key is not None:
        _VECTOR_CACHE_MISSES += 1

    recorder = _RecordingAxis()
    _plot_scene_geometry_2d_impl(recorder, scene_geometry)
    commands = tuple(recorder.commands)
    if key is not None:
        _SCENE_2D_COMMAND_CACHE[key] = commands
        _VECTOR_CACHE_WRITES += 1
        _update_figure_cache_peaks()
    return commands


def plot_scene_geometry_2d(
    ax,
    scene_geometry,
    rendering_mode: str = "rasterized",
    resolution: float = 0.05,
    show_material_legend: bool = False,
):
    """Plot 3D scene geometry as a 2D top-view projection.

    Args:
        ax: Matplotlib axis to plot on
        scene_geometry: List of mesh dictionaries
        rendering_mode: Rendering method
            - "rasterized" (default): Clean floor plan with anti-aliasing
            - "vector": Alpha-shape, convex-hull, or point projections
            - "auto": Try rasterized, fallback to vector
        resolution: Grid resolution for rasterized mode (meters per pixel)
        show_material_legend: If True, add material-family entries to the legend.
    """
    try:
        # Rasterized and auto modes try the floor-plan raster first.
        if rendering_mode in ("rasterized", "auto"):
            rasterized_data = _cached_rasterized_floor_plan(
                scene_geometry,
                resolution=resolution,
                blur_sigma=0.8,
                edge_enhancement=True,
            )

            # Plot rasterized image
            if rasterized_data is not None:
                image, x_min, x_max, y_min, y_max = rasterized_data
                ax.imshow(
                    image,
                    extent=[x_min, x_max, y_min, y_max],
                    origin="lower",
                    aspect="auto",
                    interpolation="bilinear",
                    zorder=0,
                )
                if show_material_legend:
                    _add_material_legend_entries(ax, _scene_material_scores(scene_geometry))
                logger.info("[RASTERIZE] Successfully plotted floor plan")
                return  # Success!
            elif rendering_mode == "rasterized":
                logger.info("[RASTERIZE] Rasterized floor plan unavailable; using vector mode")

        # Fallback to vector mode (alpha shapes/scatter)
        if rendering_mode in ("vector", "auto", "rasterized"):
            logger.info("[VECTOR] Using vector rendering (alpha shapes/scatter)")
            commands = _build_scene_geometry_2d_commands(scene_geometry)
            for method, args, kwargs in commands:
                try:
                    getattr(ax, method)(*args, **kwargs)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    RuntimeError,
                    OSError,
                ) as draw_err:  # defensive catch: plot resilience
                    logger.debug(f"Replay failed for {method}: {draw_err}")
            if show_material_legend:
                _add_material_legend_entries(ax, _scene_material_scores(scene_geometry))

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ) as e:  # defensive catch: plot resilience
        logger.warning(f"Could not plot scene geometry: {e}")


def _plot_scene_geometry_2d_impl(ax, scene_geometry):
    """Build vector 2D drawing commands from mesh vertices and hull fallbacks."""
    try:
        # First pass: check if any mesh is too complex for consistent visualization
        has_complex_mesh = False
        for mesh_info in scene_geometry:
            mesh = mesh_info["mesh"]
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                x_range = x_coords.max() - x_coords.min()
                y_range = y_coords.max() - y_coords.min()
                point_density = (
                    len(vertices) / (x_range * y_range) if x_range > 0 and y_range > 0 else 1.0
                )

                # Check if this mesh is too complex for consistent visualization
                # Use thresholds that allow alpha shapes for moderate complexity (like etoile)
                is_too_complex = (
                    len(vertices) > 5000
                    or point_density > 200  # Very large meshes only
                    or (  # Very dense meshes only
                        len(vertices) > 2000 and point_density > 100
                    )  # Large and dense
                )

                if is_too_complex:
                    has_complex_mesh = True
                    break

        if has_complex_mesh:
            logger.info(
                "[SCENE GEOMETRY] Detected complex meshes - using scatter plot for all meshes for consistency"
            )

        # Second pass: render all meshes
        for mesh_info in scene_geometry:
            mesh = mesh_info["mesh"]
            name = mesh_info.get("name", "mesh")
            color = mesh_info.get("color", [0.7, 0.7, 0.7])

            # Get vertices from the mesh
            vertices = np.asarray(mesh.vertices)

            if len(vertices) == 0:
                continue

            # Project to 2D (top view: keep X and Y, ignore Z)
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]

            # Create a 2D projection using multiple approaches for better shape representation
            # Use scatter plot for all meshes if any mesh is too complex (for visual consistency)
            if has_complex_mesh:
                logger.debug(
                    f"   [SCATTER] Using scatter for {len(vertices)} vertices (complex scene)"
                )
                ax.scatter(x_coords, y_coords, c=color, s=1, alpha=0.6)
                logger.debug("   [SCATTER] Plotted all vertices")
            # Use complex reconstruction for meshes with more than 10 vertices (when no complex meshes)
            elif len(vertices) > 10:
                # For complex meshes, use multiple techniques
                try:
                    from scipy.spatial import ConvexHull

                    # Get 2D points
                    points_2d = np.column_stack([x_coords, y_coords])

                    # Debug: Basic mesh info
                    logger.debug(f"[SCENE GEOMETRY] Mesh '{name}': {len(vertices)} vertices")
                    logger.debug(f"   X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
                    logger.debug(f"   Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
                    logger.debug(
                        f"   Z range: {vertices[:, 2].min():.2f} to {vertices[:, 2].max():.2f}"
                    )

                    # Method 1: Alpha shape (concave hull) for better boundary detection
                    try:
                        from scipy.spatial import Delaunay
                        from shapely.geometry import Polygon

                        # Create alpha shape
                        def alpha_shape(points, alpha):
                            coords = np.array(points)
                            if len(coords) < 3:
                                return np.array([])

                            try:
                                tri = Delaunay(coords)
                                edges = set()

                                for ia, ib, ic in tri.simplices:
                                    pa = coords[ia]
                                    pb = coords[ib]
                                    pc = coords[ic]

                                    # Calculate triangle sides
                                    a = np.linalg.norm(pb - pc)
                                    b = np.linalg.norm(pa - pc)
                                    c = np.linalg.norm(pa - pb)

                                    # Skip degenerate triangles
                                    if a < 1e-10 or b < 1e-10 or c < 1e-10:
                                        continue

                                    # Calculate circumradius
                                    s = (a + b + c) / 2.0
                                    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

                                    if area > 1e-10:
                                        circum_r = a * b * c / (4.0 * area)

                                        # Add edges if circumradius is small enough
                                        if circum_r < 1.0 / alpha:
                                            edges.add((ia, ib))
                                            edges.add((ib, ic))
                                            edges.add((ic, ia))

                                # Convert edges to boundary points
                                if len(edges) > 0:
                                    edge_points = []
                                    for i, j in edges:
                                        edge_points.append(coords[i])
                                        edge_points.append(coords[j])

                                    if len(edge_points) > 0:
                                        return np.array(edge_points)

                                return np.array([])

                            except (
                                ValueError,
                                TypeError,
                                KeyError,
                                IndexError,
                                RuntimeError,
                                OSError,
                            ) as e:  # defensive catch: plot resilience
                                logger.warning(f"   [ALPHA SHAPE] Error in alpha_shape: {e}")
                                return np.array([])

                        # Try alpha shape with adaptive alpha based on point density
                        x_range = x_coords.max() - x_coords.min()
                        y_range = y_coords.max() - y_coords.min()
                        point_density = (
                            len(points_2d) / (x_range * y_range)
                            if x_range > 0 and y_range > 0
                            else 1.0
                        )

                        # Smart complexity detection: skip alpha shape for very complex geometries
                        vertex_count = len(vertices)
                        is_too_complex = (
                            vertex_count > 5000
                            or point_density > 200  # Very large meshes only
                            or (  # Very dense meshes only
                                vertex_count > 2000 and point_density > 100
                            )  # Large and dense
                        )

                        if is_too_complex:
                            logger.debug(
                                f"   [COMPLEXITY] Mesh too complex for alpha shape (vertices: {vertex_count}, density: {point_density:.1f})"
                            )
                            logger.debug("   [FALLBACK] Using scatter plot for complex geometry")
                            # Skip alpha shape and go directly to scatter
                            ax.scatter(x_coords, y_coords, c=color, s=1, alpha=0.6)
                            logger.debug(
                                f"   [SCATTER] Plotted {len(x_coords)} vertices as scatter"
                            )
                            continue

                        # Adaptive alpha: higher density = lower alpha (more detail)
                        if point_density > 100:  # Very high density (indoor complex geometry)
                            alpha = 0.2  # Less restrictive for complex indoor shapes
                            density_category = "VERY_HIGH"
                        elif point_density > 10:  # High density
                            alpha = 0.1  # Moderate detail
                            density_category = "HIGH"
                        elif point_density > 0.1:  # Medium density
                            alpha = 0.05  # More detail
                            density_category = "MEDIUM"
                        else:  # Low density
                            alpha = 0.001  # Less detail, more robust
                            density_category = "LOW"

                        logger.debug(
                            f"   [ALPHA SHAPE] Point density: {point_density:.6f} ({density_category})"
                        )
                        logger.debug(f"   [ALPHA SHAPE] Using alpha: {alpha}")
                        logger.debug(
                            f"   [ALPHA SHAPE] X range: {x_range:.2f}, Y range: {y_range:.2f}"
                        )

                        alpha_points = alpha_shape(points_2d, alpha)

                        if len(alpha_points) > 0:
                            logger.debug(
                                f"   [ALPHA SHAPE] SUCCESS: Generated {len(alpha_points)} boundary points"
                            )
                            # Plot alpha shape
                            ax.plot(
                                alpha_points[:, 0],
                                alpha_points[:, 1],
                                color=color,
                                linewidth=1.5,
                                alpha=0.8,
                            )

                            # Fill alpha shape
                            if len(alpha_points) >= 3:
                                try:
                                    from shapely.geometry import Polygon

                                    poly = Polygon(alpha_points)
                                    if poly.is_valid:
                                        x, y = poly.exterior.xy
                                        ax.fill(x, y, color=color, alpha=0.15)
                                        logger.debug("   [ALPHA SHAPE] Filled polygon successfully")
                                    else:
                                        logger.debug(
                                            "   [ALPHA SHAPE] Invalid polygon, skipping fill"
                                        )
                                except (
                                    ValueError,
                                    TypeError,
                                    KeyError,
                                    IndexError,
                                    RuntimeError,
                                    OSError,
                                ) as e:  # defensive catch: plot resilience
                                    logger.warning(f"   [ALPHA SHAPE] Fill error: {e}")
                        else:
                            logger.debug("   [ALPHA SHAPE] FAILED: No boundary points generated")
                            # Fallback to convex hull
                            try:
                                logger.debug("   [FALLBACK] Trying Convex Hull...")
                                hull = ConvexHull(points_2d)
                                hull_points = points_2d[hull.vertices]
                                hull_points = np.vstack([hull_points, hull_points[0]])
                                ax.plot(
                                    hull_points[:, 0],
                                    hull_points[:, 1],
                                    color=color,
                                    linewidth=1.5,
                                    alpha=0.8,
                                )
                                ax.fill(
                                    hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.15
                                )
                                logger.debug(
                                    f"   [CONVEX HULL] SUCCESS: Generated {len(hull_points)} hull points"
                                )
                            except (
                                ValueError,
                                TypeError,
                                KeyError,
                                IndexError,
                                RuntimeError,
                                OSError,
                            ) as e:  # defensive catch: plot resilience
                                logger.warning(f"   [CONVEX HULL] FAILED: {e}")
                                # Ultimate fallback: use bounding box for very sparse data
                                logger.debug("   [FALLBACK] Using Bounding Box...")
                                x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
                                y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

                                # Create bounding box
                                bbox_points = np.array(
                                    [
                                        [x_min, y_min],
                                        [x_max, y_min],
                                        [x_max, y_max],
                                        [x_min, y_max],
                                        [x_min, y_min],
                                    ]
                                )

                                ax.plot(
                                    bbox_points[:, 0],
                                    bbox_points[:, 1],
                                    color=color,
                                    linewidth=1.5,
                                    alpha=0.8,
                                )
                                ax.fill(
                                    bbox_points[:, 0], bbox_points[:, 1], color=color, alpha=0.15
                                )
                                logger.debug(
                                    f"   [BOUNDING BOX] SUCCESS: Bbox ({x_min:.2f},{y_min:.2f}) to ({x_max:.2f},{y_max:.2f})"
                                )

                    except ImportError:
                        logger.warning(
                            "   [IMPORT ERROR] Shapely not available, using Convex Hull fallback"
                        )
                        # Fallback to convex hull if shapely not available
                        hull = ConvexHull(points_2d)
                        hull_points = points_2d[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.plot(
                            hull_points[:, 0],
                            hull_points[:, 1],
                            color=color,
                            linewidth=1.5,
                            alpha=0.8,
                        )
                        ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.15)
                        logger.debug(
                            f"   [CONVEX HULL] SUCCESS: Generated {len(hull_points)} hull points"
                        )

                    # Method 2: Add density-based scatter for internal details
                    # Sample points for density visualization
                    if len(points_2d) > 1000:
                        # Downsample for performance
                        indices = np.random.choice(len(points_2d), 500, replace=False)
                        sample_points = points_2d[indices]
                        logger.debug(
                            f"   [SCATTER] Downsampled from {len(points_2d)} to {len(sample_points)} points"
                        )
                    else:
                        sample_points = points_2d
                        logger.debug(f"   [SCATTER] Using all {len(sample_points)} points")

                    # Plot internal structure as semi-transparent scatter
                    ax.scatter(
                        sample_points[:, 0], sample_points[:, 1], color=color, s=0.5, alpha=0.3
                    )
                    logger.debug("   [SCATTER] Plotted internal structure")

                except ImportError:
                    logger.warning(
                        "   [IMPORT ERROR] SciPy not available, using basic scatter fallback"
                    )
                    # Ultimate fallback: plot vertices as scatter
                    ax.scatter(x_coords, y_coords, color=color, s=1, alpha=0.6)
                    logger.debug(f"   [BASIC SCATTER] Plotted {len(x_coords)} vertices")
            else:
                logger.debug(f"   [SIMPLE MESH] Drawing outline for {len(vertices)} vertices")
                points_2d = np.column_stack([x_coords, y_coords])
                unique_points = np.unique(np.round(points_2d, decimals=9), axis=0)
                if len(unique_points) >= 3:
                    center = unique_points.mean(axis=0)
                    angles = np.arctan2(
                        unique_points[:, 1] - center[1],
                        unique_points[:, 0] - center[0],
                    )
                    ordered = unique_points[np.argsort(angles)]
                    closed = np.vstack([ordered, ordered[0]])
                    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5, alpha=0.85)
                    ax.fill(ordered[:, 0], ordered[:, 1], color=color, alpha=0.12)
                elif len(unique_points) == 2:
                    ax.plot(
                        unique_points[:, 0],
                        unique_points[:, 1],
                        color=color,
                        linewidth=2.0,
                        alpha=0.85,
                    )
                else:
                    ax.scatter(x_coords, y_coords, color=color, s=6, alpha=0.8)
                logger.debug("   [SIMPLE MESH] Drew simple mesh outline")

        logger.info(f"Plotted {len(scene_geometry)} scene objects")

    except ImportError as e:
        logger.warning(f"Could not plot scene geometry: {e}")
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ) as e:  # defensive catch: plot resilience
        logger.warning(f"Error plotting scene geometry: {e}")


def plot_scene_geometry_3d(
    ax, scene_geometry, rendering_mode: str = "floor_plan", alpha: float = 0.3
):
    """Plot 3D scene geometry as a backdrop on a 3D axis.

    Args:
        ax: Matplotlib 3D axis
        scene_geometry: List of mesh dictionaries with 'mesh', 'name', 'color'
        rendering_mode: Rendering method
            - "floor_plan" (default): Cached raster floor texture, with mesh fallback
            - "mesh": Bounded triangles, convex hulls, or sampled points
            - "wireframe": Bounded mesh-edge collection
            - "hybrid": Floor texture with semi-transparent 3D meshes
            - "city": Floor plan + simplified building blocks for city-scale scenes
        alpha: Transparency for geometry (0=invisible, 1=opaque)
    """
    logger.info(f"[3D RENDER] Rendering with mode={rendering_mode!r}, alpha={alpha}")
    if rendering_mode == "floor_plan":
        _plot_scene_geometry_3d_floor_plan(ax, scene_geometry, alpha)
    elif rendering_mode == "wireframe":
        _plot_scene_geometry_3d_wireframe(ax, scene_geometry, alpha)
    elif rendering_mode == "hybrid":
        _plot_scene_geometry_3d_hybrid(ax, scene_geometry, alpha)
    elif rendering_mode == "city":
        _plot_scene_geometry_3d_city(ax, scene_geometry, alpha)
    else:
        _plot_scene_geometry_3d_mesh(ax, scene_geometry, alpha)


def _plot_scene_geometry_3d_floor_plan(ax, scene_geometry, alpha: float = 0.3):
    """Render a cached floor texture at ground level, falling back to mesh drawing."""
    try:
        resolution = 0.1
        blur_sigma = 0.5
        edge_enhancement = True
        raster_result = _cached_rasterized_floor_plan(
            scene_geometry,
            resolution=resolution,
            blur_sigma=blur_sigma,
            edge_enhancement=edge_enhancement,
        )

        if raster_result is None:
            logger.warning(
                "[3D FLOOR PLAN] Could not create rasterized floor plan, falling back to mesh mode"
            )
            _plot_scene_geometry_3d_mesh(ax, scene_geometry, alpha)
            return

        image, x_min, x_max, y_min, y_max = raster_result

        # Get image dimensions
        height, width = image.shape[:2]

        # Create a dense meshgrid matching the image resolution
        # Use fewer points for performance (subsample the image)
        subsample = max(1, min(width, height) // 200)  # Aim for ~200x200 grid
        x_steps = width // subsample
        y_steps = height // subsample

        x = np.linspace(x_min, x_max, x_steps)
        y = np.linspace(y_min, y_max, y_steps)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # Ground plane at Z=0

        # Subsample the image to match the meshgrid
        if subsample > 1:
            image_subsampled = image[::subsample, ::subsample]
        else:
            image_subsampled = image

        # Ensure image matches grid dimensions
        # We need (y_steps, x_steps) image to create (y_steps-1, x_steps-1) quads
        if image_subsampled.shape[0] != y_steps or image_subsampled.shape[1] != x_steps:
            # Resize image to match grid
            from scipy.ndimage import zoom

            y_factor = y_steps / image_subsampled.shape[0]
            x_factor = x_steps / image_subsampled.shape[1]
            image_subsampled = zoom(image_subsampled, (y_factor, x_factor, 1), order=1)

        # Extract colors for each quad (face)
        # For plot_surface, we need one color per face (quad)
        # A face is defined by 4 vertices, so we take the average or just use top-left
        colors = np.asarray(image_subsampled[:-1, :-1]).copy()  # Shape: (y_steps-1, x_steps-1, 4)

        # Apply alpha to the entire image
        colors[:, :, 3] = colors[:, :, 3] * alpha

        # Plot the floor plan as a textured surface
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=colors,
            shade=False,
            zorder=0,
            antialiased=False,
        )

        logger.debug(f"[3D FLOOR PLAN] Rendered {x_steps}x{y_steps} textured floor plan at Z=0")

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ):  # defensive catch: plot resilience
        logger.warning(
            "[3D FLOOR PLAN] Failed, falling back to mesh mode",
            exc_info=True,
        )
        _plot_scene_geometry_3d_mesh(ax, scene_geometry, alpha)


def _plot_scene_geometry_3d_hybrid(ax, scene_geometry, alpha: float = 0.3):
    """Combine a ground-level floor texture with translucent 3D meshes."""
    # Keep the floor visible without letting it compete with vertical geometry.
    floor_alpha = min(alpha * 1.0, 0.4)
    _plot_scene_geometry_3d_floor_plan(ax, scene_geometry, floor_alpha)

    # Mesh drawing applies its own opacity caps, so its input needs a stronger alpha.
    mesh_alpha = min(alpha * 1.8, 0.5)
    _plot_scene_geometry_3d_mesh(ax, scene_geometry, mesh_alpha)

    logger.info(
        f"[3D HYBRID] Rendered floor plan (α={floor_alpha:.2f}) + meshes (α={mesh_alpha:.2f})"
    )


def _scene_material_descriptor(
    name: str, material_id: Optional[str] = None
) -> Tuple[str, str, Tuple[float, float, float]]:
    """Return a stable material key, label, and color for scene summary figures."""
    for text in (name, material_id or ""):
        lowered = text.lower()
        for key, label, color in _MATERIAL_COLOR_RULES:
            if key in lowered:
                return key, label, color
    return "other", "Other material", (0.48, 0.50, 0.52)


def _add_material_legend_entries(
    ax,
    material_scores: Dict[str, Tuple[str, Tuple[float, float, float], float]],
    *,
    max_entries: int = 6,
) -> None:
    """Add invisible proxy markers so material colors appear in the figure legend."""
    if not material_scores:
        return
    seen = getattr(ax, "_orchav_scene_material_legend_keys", set())
    ranked = sorted(material_scores.items(), key=lambda item: item[1][2], reverse=True)
    for key, (label, color, _) in ranked[:max_entries]:
        if key in seen:
            continue
        if getattr(ax, "name", "") == "3d":
            ax.scatter(
                [],
                [],
                [],
                marker="s",
                s=36,
                c=[color],
                alpha=0.82,
                label=f"Material: {label}",
            )
        else:
            ax.scatter(
                [],
                [],
                marker="s",
                s=36,
                c=[color],
                alpha=0.82,
                label=f"Material: {label}",
            )
        seen.add(key)
    setattr(ax, "_orchav_scene_material_legend_keys", seen)


def _scene_material_scores(
    scene_geometry: Optional[List[Any]],
) -> Dict[str, Tuple[str, Tuple[float, float, float], float]]:
    material_scores: Dict[str, Tuple[str, Tuple[float, float, float], float]] = {}
    for mesh_info in scene_geometry or []:
        if not isinstance(mesh_info, dict):
            continue
        name = str(mesh_info.get("name", ""))
        material_key, material_label, color = _scene_material_descriptor(
            name, mesh_info.get("material_id")
        )
        mesh = mesh_info.get("mesh")
        score = 1.0
        if mesh is not None:
            vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
            if vertices.size:
                x_span = float(vertices[:, 0].max() - vertices[:, 0].min())
                y_span = float(vertices[:, 1].max() - vertices[:, 1].min())
                score = max(x_span * y_span, float(len(vertices)), 1.0)
        label, existing_color, existing_score = material_scores.get(
            material_key, (material_label, color, 0.0)
        )
        material_scores[material_key] = (label, existing_color, existing_score + score)
    return material_scores


def _is_city_building_mesh(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("building") or lowered.startswith("buildings")


def _grid_aabbs(
    mesh, *, cell_size: float, min_height: float = 1.0
) -> List[Tuple[float, float, float, float, float, float]]:
    """Return fast XY-grid AABBs for an aggregate city mesh."""
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
    if vertices.size == 0:
        return []

    try:
        triangles = np.asarray(getattr(mesh, "triangles", []), dtype=int)
    except (TypeError, ValueError):
        triangles = np.empty((0, 3), dtype=int)

    if triangles.size == 0:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        height = float(maxs[2] - mins[2])
        if height < min_height:
            return []
        return [
            (
                float(mins[0]),
                float(maxs[0]),
                float(mins[1]),
                float(maxs[1]),
                float(mins[2]),
                float(maxs[2]),
            )
        ]

    tri_vertices = vertices[triangles]
    centroids = tri_vertices.mean(axis=1)
    z_min = tri_vertices[:, :, 2].min(axis=1)
    z_max = tri_vertices[:, :, 2].max(axis=1)

    x0 = float(vertices[:, 0].min())
    y0 = float(vertices[:, 1].min())
    ix = np.floor((centroids[:, 0] - x0) / cell_size).astype(np.int64)
    iy = np.floor((centroids[:, 1] - y0) / cell_size).astype(np.int64)
    cell_ids = np.stack([ix, iy], axis=1)
    unique_cells, inverse = np.unique(cell_ids, axis=0, return_inverse=True)
    cell_z_min = np.full(len(unique_cells), np.inf, dtype=float)
    cell_z_max = np.full(len(unique_cells), -np.inf, dtype=float)
    np.minimum.at(cell_z_min, inverse, z_min)
    np.maximum.at(cell_z_max, inverse, z_max)

    boxes = []
    for cell_index, (cell_x, cell_y) in enumerate(unique_cells):
        z_low = float(cell_z_min[cell_index])
        z_high = float(cell_z_max[cell_index])
        if z_high - z_low < min_height:
            continue
        x_min = x0 + float(cell_x) * cell_size
        y_min = y0 + float(cell_y) * cell_size
        boxes.append(
            (
                x_min,
                x_min + cell_size,
                y_min,
                y_min + cell_size,
                z_low,
                z_high,
            )
        )

    return boxes


def _box_faces(box: Tuple[float, float, float, float, float, float]):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0 or (z_max - z_min) <= 0:
        return []
    p000 = (x_min, y_min, z_min)
    p100 = (x_max, y_min, z_min)
    p110 = (x_max, y_max, z_min)
    p010 = (x_min, y_max, z_min)
    p001 = (x_min, y_min, z_max)
    p101 = (x_max, y_min, z_max)
    p111 = (x_max, y_max, z_max)
    p011 = (x_min, y_max, z_max)
    return [
        [p001, p101, p111, p011],  # roof
        [p000, p100, p101, p001],
        [p100, p110, p111, p101],
        [p110, p010, p011, p111],
        [p010, p000, p001, p011],
    ]


def _city_building_entries(
    scene_geometry: Optional[List[Any]], *, cell_size: float
) -> List[
    Tuple[
        float, Tuple[float, float, float, float, float, float], Tuple[float, float, float], str, str
    ]
]:
    params = {"cell_size": round(float(cell_size), 6), "min_height": 1.0}
    cache_path = _summary_cache_path(scene_geometry, "city_blocks", params)
    cached = _load_npz_cache(cache_path)
    if cached is not None:
        try:
            boxes = np.asarray(cached["boxes"], dtype=float)
            scores = np.asarray(cached["scores"], dtype=float)
            colors = np.asarray(cached["colors"], dtype=float)
            keys = [str(value) for value in cached["keys"]]
            labels = [str(value) for value in cached["labels"]]
            logger.info("[3D CITY] Loaded simplified building blocks from persistent cache")
            return [
                (
                    float(scores[i]),
                    _as_float6(boxes[i]),
                    _as_float3(colors[i]),
                    keys[i],
                    labels[i],
                )
                for i in range(len(scores))
            ]
        except (KeyError, ValueError, TypeError, IndexError):
            pass

    entries = []
    for mesh_info in scene_geometry or []:
        name = str(mesh_info.get("name", ""))
        if not _is_city_building_mesh(name):
            continue
        mesh = mesh_info.get("mesh")
        if mesh is None:
            continue
        material_key, material_label, color = _scene_material_descriptor(
            name, mesh_info.get("material_id")
        )
        for box in _grid_aabbs(mesh, cell_size=cell_size, min_height=1.0):
            x_min, x_max, y_min, y_max, z_min, z_max = box
            footprint_area = max((x_max - x_min) * (y_max - y_min), 0.0)
            height = z_max - z_min
            if footprint_area < 4.0 or height < 1.0:
                continue
            score = footprint_area * max(height, 1.0)
            entries.append((score, box, color, material_key, material_label))

    if entries:
        _save_npz_cache(
            cache_path,
            scores=np.asarray([entry[0] for entry in entries], dtype=np.float64),
            boxes=np.asarray([entry[1] for entry in entries], dtype=np.float64),
            colors=np.asarray([entry[2] for entry in entries], dtype=np.float64),
            keys=np.asarray([entry[3] for entry in entries], dtype=np.str_),
            labels=np.asarray([entry[4] for entry in entries], dtype=np.str_),
        )
    return entries


def _plot_scene_geometry_3d_city(ax, scene_geometry, alpha: float = 0.55):
    """Render city-scale scenes as floor context plus simplified 3D buildings."""
    floor_alpha = min(alpha * 0.45, 0.25)
    _plot_scene_geometry_3d_floor_plan(ax, scene_geometry, floor_alpha)

    all_vertices = []
    for mesh_info in scene_geometry or []:
        mesh = mesh_info.get("mesh")
        if mesh is None:
            continue
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
        if vertices.size:
            all_vertices.append(vertices)
    if all_vertices:
        vertices = np.vstack(all_vertices)
        scene_span = max(
            float(vertices[:, 0].max() - vertices[:, 0].min()),
            float(vertices[:, 1].max() - vertices[:, 1].min()),
        )
    else:
        scene_span = DEFAULT_SCENE_SPAN_M
    cell_size = max(scene_span / CITY_BLOCK_CELL_DIVISOR, MIN_CITY_BLOCK_CELL_SIZE_M)

    building_entries = _city_building_entries(scene_geometry, cell_size=cell_size)

    if not building_entries:
        logger.info("[3D CITY] No building blocks found; falling back to wireframe context")
        _plot_scene_geometry_3d_wireframe(ax, scene_geometry, min(alpha, 0.25))
        return

    max_blocks = MAX_3D_CITY_BLOCKS
    if len(building_entries) > max_blocks:
        building_entries = sorted(building_entries, key=lambda item: item[0], reverse=True)[
            :max_blocks
        ]

    faces = []
    facecolors = []
    edgecolors = []
    block_alpha = min(max(alpha, 0.35), 0.72)
    material_scores: Dict[str, Tuple[str, Tuple[float, float, float], float]] = {}
    for score, box, color, material_key, material_label in building_entries:
        box_faces = _box_faces(box)
        faces.extend(box_faces)
        facecolors.extend([(*color, block_alpha)] * len(box_faces))
        edgecolors.extend([(*color, min(block_alpha + 0.12, 0.85))] * len(box_faces))
        label, existing_color, existing_score = material_scores.get(
            material_key, (material_label, color, 0.0)
        )
        material_scores[material_key] = (label, existing_color, existing_score + score)

    if faces:
        poly = Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=0.15,
            zsort="average",
        )
        ax.add_collection3d(poly)

    legend_scores = _scene_material_scores(scene_geometry)
    for material_key, (material_label, color, score) in material_scores.items():
        existing_label, existing_color, existing_score = legend_scores.get(
            material_key, (material_label, color, 0.0)
        )
        legend_scores[material_key] = (
            existing_label,
            existing_color,
            existing_score + score,
        )
    _add_material_legend_entries(ax, legend_scores)
    logger.info("[3D CITY] Rendered %d simplified building block(s)", len(building_entries))


def _wireframe_segments(
    scene_geometry: Optional[List[Any]], *, max_edges_total: int, max_edges_per_mesh: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = {
        "max_edges_total": int(max_edges_total),
        "max_edges_per_mesh": int(max_edges_per_mesh),
    }
    cache_path = _summary_cache_path(scene_geometry, "wireframe", params)
    cached = _load_npz_cache(cache_path)
    if cached is not None:
        try:
            logger.info("[3D WIREFRAME] Loaded sampled edge set from persistent cache")
            return (
                np.asarray(cached["segments"], dtype=float),
                np.asarray(cached["colors"], dtype=float),
                np.asarray(cached["keys"], dtype=np.str_),
                np.asarray(cached["labels"], dtype=np.str_),
            )
        except (KeyError, ValueError, TypeError):
            pass

    all_segments = []
    all_colors = []
    all_keys = []
    all_labels = []
    remaining_edges = max_edges_total

    for mesh_info in scene_geometry or []:
        if remaining_edges <= 0:
            break
        try:
            mesh = mesh_info["mesh"]
            name = str(mesh_info.get("name", ""))
            material_key, material_label, base_color = _scene_material_descriptor(
                name, mesh_info.get("material_id")
            )
            vertices = np.asarray(mesh.vertices, dtype=float)
            if vertices.size == 0:
                continue

            faces = None
            try:
                if hasattr(mesh, "triangles"):
                    faces = np.asarray(mesh.triangles)
                elif hasattr(mesh, "faces"):
                    faces = np.asarray(mesh.faces)
            except (AttributeError, ValueError, TypeError):
                faces = None

            if faces is None or faces.size == 0 or faces.shape[1] < 3:
                continue

            faces = np.asarray(faces[:, :3], dtype=np.int64)
            face_budget = max(max_edges_per_mesh, min(len(faces), remaining_edges * 2))
            if len(faces) > face_budget:
                sample_idx = np.linspace(0, len(faces) - 1, face_budget, dtype=np.int64)
                faces = faces[sample_idx]

            edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
            edges.sort(axis=1)
            edges = np.unique(edges, axis=0)
            valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
            valid &= (edges[:, 0] < len(vertices)) & (edges[:, 1] < len(vertices))
            edges = edges[valid]
            if len(edges) == 0:
                continue

            mesh_budget = min(max_edges_per_mesh, remaining_edges)
            if len(edges) > mesh_budget:
                edge_idx = np.linspace(0, len(edges) - 1, mesh_budget, dtype=np.int64)
                edges = edges[edge_idx]

            segments = vertices[edges]
            all_segments.append(segments)
            all_colors.append(np.tile(np.asarray(base_color, dtype=float), (len(segments), 1)))
            all_keys.extend([material_key] * len(segments))
            all_labels.extend([material_label] * len(segments))
            remaining_edges -= len(segments)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ):
            continue

    if all_segments:
        segments_arr = np.vstack(all_segments).astype(np.float64)
        colors_arr = np.vstack(all_colors).astype(np.float64)
        keys_arr = np.asarray(all_keys, dtype=np.str_)
        labels_arr = np.asarray(all_labels, dtype=np.str_)
    else:
        segments_arr = np.empty((0, 2, 3), dtype=np.float64)
        colors_arr = np.empty((0, 3), dtype=np.float64)
        keys_arr = np.empty((0,), dtype=np.str_)
        labels_arr = np.empty((0,), dtype=np.str_)

    _save_npz_cache(
        cache_path,
        segments=segments_arr,
        colors=colors_arr,
        keys=keys_arr,
        labels=labels_arr,
    )
    return segments_arr, colors_arr, keys_arr, labels_arr


def _plot_scene_geometry_3d_wireframe(ax, scene_geometry, alpha: float = 0.3):
    """
    Render scene as wireframe (edges only) for a lighter visualization.
    """
    try:
        max_edges_total = MAX_WIREFRAME_EDGES_TOTAL
        max_edges_per_mesh = MAX_WIREFRAME_EDGES_PER_MESH
        segments, colors, keys, labels = _wireframe_segments(
            scene_geometry,
            max_edges_total=max_edges_total,
            max_edges_per_mesh=max_edges_per_mesh,
        )
        if len(segments):
            line_alpha = min(max(alpha, 0.05), 0.95)
            rgba = np.column_stack([colors, np.full(len(colors), line_alpha)])
            collection = Line3DCollection(
                segments,
                colors=rgba,
                linewidths=0.28,
                linestyles="solid",
            )
            ax.add_collection3d(collection)

        material_scores: Dict[str, Tuple[str, Tuple[float, float, float], float]] = {}
        for key, label, color in zip(keys, labels, colors):
            existing_label, existing_color, existing_score = material_scores.get(
                key, (label, _as_float3(color), 0.0)
            )
            material_scores[key] = (existing_label, existing_color, existing_score + 1.0)

        _add_material_legend_entries(ax, material_scores)

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ) as e:  # defensive catch: plot resilience
        logger.debug(f"[3D WIREFRAME] Failed: {e}")


def _plot_scene_geometry_3d_mesh(ax, scene_geometry, alpha: float = 0.3):
    """Render bounded triangles, then convex hulls or sampled points as fallbacks."""
    try:
        max_points_per_mesh = MAX_MESH_POINTS_PER_MESH
        # Dark neutral geometry remains legible when layered over a floor texture.
        base_color = (0.4, 0.4, 0.4)
        face_alpha = min(alpha * 0.7, 0.35)
        edge_alpha = min(alpha * 1.0, 0.5)
        scatter_alpha = min(alpha * 0.9, 0.45)

        # Optional convex hull (requires SciPy)
        try:
            from scipy.spatial import ConvexHull

            have_scipy = True
        except ImportError:
            have_scipy = False

        for mesh_info in scene_geometry:
            try:
                mesh = mesh_info["mesh"]
                vertices = np.asarray(mesh.vertices)
                if vertices.size == 0:
                    continue

                # Basic density/size check
                n = len(vertices)

                # Preferred: render actual mesh triangles if available
                faces = None
                try:
                    if hasattr(mesh, "triangles"):
                        faces = np.asarray(mesh.triangles)
                    elif hasattr(mesh, "faces"):
                        faces = np.asarray(mesh.faces)
                except (AttributeError, ValueError, TypeError):
                    faces = None
                if faces is not None and faces.size > 0 and faces.shape[1] >= 3:
                    # Limit number of triangles for performance
                    max_tris = MAX_MESH_TRIANGLES
                    m = faces.shape[0]
                    if m > max_tris:
                        step = int(np.ceil(m / max_tris))
                        faces = faces[::step]
                    # Build triangle vertex arrays
                    try:
                        tris = [vertices[idx[:3]] for idx in faces]
                        poly = Poly3DCollection(
                            tris, facecolor=base_color, edgecolor=base_color, linewidths=0.2
                        )
                        poly.set_alpha(face_alpha)
                        poly.set_edgecolor((*base_color, edge_alpha))
                        ax.add_collection3d(poly)
                        continue
                    except (RuntimeError, ValueError, IndexError):
                        # Fall back to hull/scatter
                        pass

                # Decide whether to attempt hull (works well for coarse meshes)
                use_hull = have_scipy and n >= 8
                if use_hull:
                    # For very dense meshes, subsample before hull for performance
                    if n > MAX_HULL_INPUT_POINTS:
                        idx = np.random.choice(n, size=MAX_HULL_INPUT_POINTS, replace=False)
                        pts = vertices[idx]
                    else:
                        pts = vertices
                    try:
                        hull = ConvexHull(pts)
                        # If hull is too complex, fallback to scatter to avoid heavy draw
                        if len(hull.simplices) > MAX_HULL_SIMPLICES:
                            raise RuntimeError("Hull too complex for plotting")

                        # Build triangle list for Poly3DCollection
                        triangles = [pts[simplex] for simplex in hull.simplices]
                        poly = Poly3DCollection(
                            triangles, facecolor=base_color, edgecolor=base_color, linewidths=0.3
                        )
                        poly.set_alpha(face_alpha)
                        poly.set_edgecolor((*base_color, edge_alpha))
                        ax.add_collection3d(poly)
                        continue
                    except (RuntimeError, ValueError, IndexError):
                        # Fall back to scatter if hull fails (coplanar or too few points)
                        pass

                # Scatter fallback (downsample if needed)
                if n > max_points_per_mesh:
                    idx = np.linspace(0, n - 1, max_points_per_mesh, dtype=int)
                    pts = vertices[idx]
                else:
                    pts = vertices
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=0.5,
                    c=[base_color],
                    alpha=scatter_alpha,
                    depthshade=False,
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                RuntimeError,
                OSError,
            ):  # defensive catch: plot resilience
                # Keep plotting other meshes even if one fails
                continue
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ) as e:  # defensive catch: plot resilience
        logger.debug(f"Could not plot 3D scene geometry: {e}")
