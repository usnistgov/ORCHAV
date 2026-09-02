"""Internal mesh helpers for target assets.

This module keeps low-level mesh compatibility policy out of ``TargetManager``:
it maps mesh playback cadence onto simulation steps, normalizes PLY vertex color
encoding for Mitsuba, and derives target positions from mesh geometry when a
scenario asks to use positions embedded in a PLY sequence.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Literal

from shared.logging import get_logger

logger = get_logger(__name__)

MeshEndBehavior = Literal["loop", "hold_last"]


def mesh_sequence_index(
    mesh_count: int,
    mesh_call_count: int,
    *,
    mesh_start_index: int = 0,
    mesh_frame_stride: int = 1,
    mesh_end_behavior: MeshEndBehavior = "loop",
) -> int:
    """Select one mesh index for a playback update.

    Looping preserves the historical modulo behavior. ``hold_last`` advances
    from the configured start using the configured stride, then clamps to the
    final source mesh instead of wrapping to the beginning.
    """
    if mesh_count <= 0:
        return 0
    requested_index = mesh_start_index + int(mesh_call_count) * mesh_frame_stride
    if mesh_end_behavior == "hold_last":
        return min(max(0, requested_index), mesh_count - 1)
    if mesh_end_behavior != "loop":
        raise ValueError(
            f"mesh_end_behavior must be 'loop' or 'hold_last', got {mesh_end_behavior!r}"
        )
    return requested_index % mesh_count


_PLY_INTEGER_COLOR_TYPES = {
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
}
_NORMALIZED_PLY_CACHE: dict[tuple[str, int, int], str] = {}
_NORMALIZED_PLY_CACHE_DIR = Path(tempfile.gettempdir()) / "orchav_ply_float_colors"


def mesh_update_step_interval(
    *,
    duration: float | None,
    num_steps: int | None,
    mesh_update_interval_s: float | None,
) -> int | None:
    """Return the target mesh update cadence in simulation steps.

    ``mesh_update_interval_s`` is a wall-clock interval for animated target
    meshes. ORCHAV maps it onto the simulation step grid using the same
    convention as ray tracing: ``duration / num_steps``.

    The interval exists so mesh playback does not have to advance on every ray
    tracing step. Some frame products need geometry to remain fixed over a
    multi-step acquisition window; callers can use this conversion, then align
    updates to those boundaries before replacing the scene object.
    """
    if mesh_update_interval_s is None:
        return None
    if not num_steps or duration is None or duration <= 0:
        return None

    dt = float(duration) / max(int(num_steps), 1)
    return max(1, math.ceil(float(mesh_update_interval_s) / dt))


def mesh_call_count_for_step(frame_idx: int, mesh_step_interval: int | None) -> int:
    """Return the mesh playback call count corresponding to a frame index.

    This is distinct from ``frame_idx`` when mesh updates are intentionally
    slower than the ray-tracing step rate.
    """
    return max(0, int(frame_idx)) // max(1, int(mesh_step_interval or 1))


def _read_ply_header_lines(mesh_path: Path) -> list[str]:
    """Read the ASCII PLY header without loading the full mesh payload."""
    lines: list[str] = []
    with open(mesh_path, "rb") as handle:
        for _ in range(256):
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="replace").strip()
            lines.append(line)
            if line == "end_header":
                break
    return lines


def _ply_header_has_faces(mesh_path: str | Path) -> bool | None:
    """Return whether a PLY header declares faces, or ``None`` if unreadable."""
    path = Path(mesh_path)
    if path.suffix.lower() != ".ply":
        return None

    try:
        header_lines = _read_ply_header_lines(path)
    except OSError as exc:
        logger.debug("Could not inspect PLY header for %s: %s", path, exc)
        return None

    for line in header_lines:
        lowered = line.lower()
        if lowered.startswith("element face") or lowered.startswith("element polygon"):
            return True
    return False


def _has_integer_vertex_colors(mesh_path: Path) -> bool:
    """Return whether a PLY uses integer RGB vertex color properties."""
    if mesh_path.suffix.lower() != ".ply":
        return False

    try:
        header_lines = _read_ply_header_lines(mesh_path)
    except OSError as exc:
        logger.debug("Could not inspect PLY header for %s: %s", mesh_path, exc)
        return False

    in_vertex_section = False
    color_types: dict[str, str] = {}
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex_section = parts[1] == "vertex"
            continue
        if not in_vertex_section:
            continue
        if len(parts) >= 3 and parts[0] == "property":
            prop_type = parts[1].lower()
            prop_name = parts[2].lower()
            if prop_name in {"red", "green", "blue"}:
                color_types[prop_name] = prop_type

    return len(color_types) == 3 and all(
        prop_type in _PLY_INTEGER_COLOR_TYPES for prop_type in color_types.values()
    )


def _write_float_vertex_color_ply(source_path: Path, output_path: Path) -> None:
    """Rewrite a PLY with normalized float RGB vertex colors for Mitsuba.

    Some asset tools write PLY colors as integer ``red/green/blue`` properties.
    Mitsuba's mesh ingestion path expects color attributes to be float values in
    ``[0, 1]``. This helper preserves geometry, faces, and normals while writing
    a temporary compatibility copy with float color properties.
    """
    import numpy as np
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(source_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {source_path}")
    if not mesh.has_vertex_colors():
        raise ValueError(f"Mesh has no vertex colors: {source_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)
    if triangles.size == 0:
        raise ValueError(f"Mesh has no faces: {source_path}")

    if not mesh.has_vertex_normals() or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    colors = np.clip(np.asarray(mesh.vertex_colors, dtype=np.float32), 0.0, 1.0)

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Converted by ORCHAV to float vertex colors for Mitsuba",
            f"element vertex {len(vertices)}",
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            "property float red",
            "property float green",
            "property float blue",
            f"element face {len(triangles)}",
            "property list uchar uint vertex_indices",
            "end_header\n",
        ]
    ).encode("ascii")

    with open(output_path, "wb") as handle:
        handle.write(header)
        for vertex, normal, color in zip(vertices, normals, colors):
            handle.write(
                struct.pack(
                    "<6d3f",
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                )
            )
        for triangle in triangles:
            handle.write(
                struct.pack(
                    "<B3I",
                    3,
                    int(triangle[0]),
                    int(triangle[1]),
                    int(triangle[2]),
                )
            )


def _prepare_mesh_path_for_mitsuba(mesh_path: str) -> str:
    """Return a Mitsuba-compatible mesh path, creating a cached copy if needed."""
    source_path = Path(mesh_path)
    if not _has_integer_vertex_colors(source_path):
        return mesh_path

    try:
        stat = source_path.stat()
    except OSError as exc:
        logger.debug("Could not stat mesh %s for float-color cache: %s", source_path, exc)
        return mesh_path

    cache_key = (str(source_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached_path = _NORMALIZED_PLY_CACHE.get(cache_key)
    if cached_path and Path(cached_path).exists():
        return cached_path

    digest = hashlib.sha256(
        f"{cache_key[0]}:{cache_key[1]}:{cache_key[2]}".encode("utf-8")
    ).hexdigest()[:16]
    output_path = _NORMALIZED_PLY_CACHE_DIR / f"{source_path.stem}.{digest}.ply"
    if output_path.exists():
        _NORMALIZED_PLY_CACHE[cache_key] = str(output_path)
        return str(output_path)

    try:
        _NORMALIZED_PLY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")
        _write_float_vertex_color_ply(source_path, temp_path)
        os.replace(temp_path, output_path)
        _NORMALIZED_PLY_CACHE[cache_key] = str(output_path)
        logger.debug(
            "Normalized integer vertex colors for Mitsuba: %s -> %s",
            source_path,
            output_path,
        )
        return str(output_path)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        logger.warning(
            "Could not normalize vertex colors for %s; using original PLY: %s",
            source_path,
            exc,
        )
        return mesh_path


def positions_from_ply_aabb(
    mesh_files: list[str],
    steps: int,
    duration: float,
    mesh_start_index: int = 0,
    mesh_frame_stride: int = 1,
    mesh_update_interval_s: float | None = None,
    mesh_end_behavior: MeshEndBehavior = "loop",
) -> list[tuple[float, float, float]]:
    """Derive per-step target positions from PLY mesh AABB centers.

    When ``use_ply_position`` is enabled, the mesh sequence carries its own
    world-space motion. We approximate each mesh frame's target position by the
    center of its axis-aligned bounding box, then interpolate across simulation
    steps using the same mesh update cadence that controls live scene-object
    replacement.
    """
    import numpy as np
    import open3d as o3d

    n_meshes = len(mesh_files)
    if n_meshes == 0:
        return [(0.0, 0.0, 0.0)] * steps

    centers: list[Any] = []
    for call_idx in range(n_meshes):
        # Apply the same start/stride policy used by TargetManager so cached
        # positions and live mesh playback stay aligned.
        mesh_idx = mesh_sequence_index(
            n_meshes,
            call_idx,
            mesh_start_index=mesh_start_index,
            mesh_frame_stride=mesh_frame_stride,
            mesh_end_behavior=mesh_end_behavior,
        )
        mesh_path = mesh_files[mesh_idx]
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                centers.append(np.zeros(3))
            else:
                centers.append((vertices.max(axis=0) + vertices.min(axis=0)) / 2.0)
        except (OSError, RuntimeError):
            logger.warning("Could not read mesh %s for AABB center", mesh_path)
            centers.append(np.zeros(3))

    switch_every = (
        mesh_update_step_interval(
            duration=duration,
            num_steps=steps,
            mesh_update_interval_s=mesh_update_interval_s,
        )
        or 1
    )

    positions: list[tuple[float, float, float]] = []
    mesh_call = 0
    for step_idx in range(steps):
        # Interpolate between mesh AABB centers while the live mesh remains on a
        # lower-rate cadence. This keeps exported positions smooth even when
        # scene-object replacement is intentionally less frequent.
        mesh_call = step_idx // switch_every
        frac = (step_idx % switch_every) / max(switch_every, 1)
        idx_a = min(mesh_call, len(centers) - 1)
        idx_b = min(mesh_call + 1, len(centers) - 1)
        center = centers[idx_a] + frac * (centers[idx_b] - centers[idx_a])
        positions.append((float(center[0]), float(center[1]), float(center[2])))

    logger.info(
        "PLY AABB trajectory: %d meshes -> %d steps (switch every %d), "
        "start=(%.2f,%.2f,%.2f) end=(%.2f,%.2f,%.2f)",
        len(centers),
        steps,
        switch_every,
        centers[0][0],
        centers[0][1],
        centers[0][2],
        centers[min(mesh_call, len(centers) - 1)][0],
        centers[min(mesh_call, len(centers) - 1)][1],
        centers[min(mesh_call, len(centers) - 1)][2],
    )
    return positions
