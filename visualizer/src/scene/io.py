"""
Scene I/O module for loading and managing 3D environments.

Handles Mitsuba XML scene parsing, mesh loading (PLY/OBJ),
material extraction, and scene assembly.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np

from shared.geometry.transforms import parse_lightweight_shape_transform
from shared.logging import get_logger

from ..diagnostics.cache_telemetry import record_cache_event, set_cache_inventory
from ..io.config_handlers import TextFileHandler
from ..materials.catalog import (
    infer_material_type_from_id,
    is_known_material_type,
    material_id_stem,
    material_preset,
    normalize_material_type_name,
)
from ..types.render_payloads import MeshPayload
from ..utils import geometry
from .geometry_payload_factory import load_mesh_payload
from .uv_cache_store import (
    clear_uv_cache,
    close_uv_cache_stores,
    get_uv_cache_info,
    get_uv_cache_store,
    prune_uv_cache,
)

logger = get_logger("orchav")

_PATCH_PROJECTION_TRIANGLE_LIMIT = 20_000
_ORIENTATION_BIN_DEGREES = 7.5
_BOX_PROJECTION_UV_CACHE_VERSION = "20260414_uv_v1"
_SCENE_PAYLOAD_CACHE_VERSION = "20260702_scene_payload_v1"
_DEFAULT_ORCHAV_CACHE_ROOT = Path("~/.orchav/cache").expanduser()
_UV_CACHE_SUBDIR = "uvcache"
_SCENE_PAYLOAD_CACHE_SUBDIR = "scene_payloads"
_DEFAULT_UV_CACHE_MAX_AGE_DAYS = 14
_DEFAULT_UV_CACHE_MAX_BYTES = 256 * 1024 * 1024
_PRUNED_UV_CACHE_ROOTS: set[str] = set()
_DEFAULT_SCENE_PAYLOAD_CACHE_MAX_AGE_DAYS = 30
_DEFAULT_SCENE_PAYLOAD_CACHE_MAX_BYTES = 512 * 1024 * 1024
_PRUNED_SCENE_PAYLOAD_CACHE_ROOTS: set[str] = set()


class MaterialInfo(TypedDict):
    """Information about a material from the scene."""

    id: str
    color: list[float]
    xml_element: Any  # XML element reference


def _uv_cache_root() -> Path:
    """Return the directory used for persisted box-projection UV caches."""
    configured = os.environ.get("ORCHAV_UV_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        base_root = os.environ.get("ORCHAV_CACHE_DIR")
        if base_root:
            root = Path(base_root).expanduser() / _UV_CACHE_SUBDIR
        else:
            root = _DEFAULT_ORCHAV_CACHE_ROOT / _UV_CACHE_SUBDIR

    root.mkdir(parents=True, exist_ok=True)
    _prune_uv_cache_root(root)
    return root


def get_uv_cache_root() -> Path:
    """Return the persisted UV cache root used by the visualizer."""
    return _uv_cache_root()


def _sanitize_cache_segment(value: str) -> str:
    """Return a filesystem-friendly cache directory segment."""
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    sanitized = sanitized.strip("._")
    return sanitized or "cache"


def _scene_payload_cache_root() -> Path | None:
    """Return the neutral scene-payload cache root, or None when disabled/unavailable."""
    if os.environ.get("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE") == "1":
        return None
    configured = os.environ.get("ORCHAV_SCENE_PAYLOAD_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        base_root = os.environ.get("ORCHAV_CACHE_DIR")
        root = (
            Path(base_root).expanduser() / _SCENE_PAYLOAD_CACHE_SUBDIR
            if base_root
            else _DEFAULT_ORCHAV_CACHE_ROOT / _SCENE_PAYLOAD_CACHE_SUBDIR
        )
    try:
        root = root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Neutral scene payload cache root is not writable: %s", root)
        return None
    _prune_scene_payload_cache_root(root)
    return root


def _scene_payload_cache_limits() -> tuple[int, int]:
    """Return age and byte limits for neutral scene payload caches."""
    try:
        max_age = max(
            0,
            int(
                float(
                    os.environ.get(
                        "ORCHAV_SCENE_PAYLOAD_CACHE_MAX_AGE_DAYS",
                        _DEFAULT_SCENE_PAYLOAD_CACHE_MAX_AGE_DAYS,
                    )
                )
            ),
        )
    except (TypeError, ValueError, OverflowError):
        max_age = _DEFAULT_SCENE_PAYLOAD_CACHE_MAX_AGE_DAYS
    try:
        max_bytes = max(
            0,
            int(
                float(
                    os.environ.get(
                        "ORCHAV_SCENE_PAYLOAD_CACHE_MAX_BYTES",
                        _DEFAULT_SCENE_PAYLOAD_CACHE_MAX_BYTES,
                    )
                )
            ),
        )
    except (TypeError, ValueError, OverflowError):
        max_bytes = _DEFAULT_SCENE_PAYLOAD_CACHE_MAX_BYTES
    return max_age, max_bytes


def _prune_scene_payload_cache_root(root: Path) -> None:
    """Prune scene payloads by last use and a root byte budget once per process."""
    root_key = str(root.resolve(strict=False))
    if root_key in _PRUNED_SCENE_PAYLOAD_CACHE_ROOTS:
        return
    _PRUNED_SCENE_PAYLOAD_CACHE_ROOTS.add(root_key)
    _enforce_scene_payload_cache_limits(root)


def _enforce_scene_payload_cache_limits(root: Path) -> None:
    """Enforce scene-payload age and byte limits after startup or a write."""
    max_age_days, max_bytes = _scene_payload_cache_limits()
    cutoff = time.time() - max_age_days * 24 * 60 * 60
    files: list[tuple[float, int, Path]] = []
    try:
        for path in root.glob("*.npz"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((float(stat.st_mtime), int(stat.st_size), path))
    except OSError:
        record_cache_event("scene_payload", "inventory_failure")
        return

    total_bytes = sum(size for _mtime, size, _path in files)
    removed = 0
    for mtime, size, path in sorted(files):
        expired = max_age_days > 0 and mtime < cutoff
        over_budget = max_bytes > 0 and total_bytes > max_bytes
        if not expired and not over_budget:
            continue
        try:
            path.unlink()
            total_bytes -= size
            removed += 1
        except OSError:
            continue
    if removed:
        record_cache_event("scene_payload", "pruned", count=removed)
    set_cache_inventory(
        "scene_payload",
        entries=max(0, len(files) - removed),
        byte_count=max(0, total_bytes),
        root=str(root),
    )


def get_scene_payload_cache_info() -> dict[str, Any]:
    """Return current neutral scene-payload disk inventory and limits."""
    root = _scene_payload_cache_root()
    if root is None:
        return {"enabled": False, "entries": 0, "bytes": 0, "root": None}
    entries = 0
    byte_count = 0
    for path in root.glob("*.npz"):
        try:
            byte_count += int(path.stat().st_size)
            entries += 1
        except OSError:
            continue
    max_age_days, max_bytes = _scene_payload_cache_limits()
    set_cache_inventory(
        "scene_payload",
        entries=entries,
        byte_count=byte_count,
        root=str(root),
    )
    return {
        "enabled": True,
        "entries": entries,
        "bytes": byte_count,
        "root": str(root),
        "max_age_days": max_age_days,
        "max_bytes": max_bytes,
    }


def clear_scene_payload_cache() -> dict[str, int]:
    """Remove neutral scene payloads under the active cache root."""
    root = _scene_payload_cache_root()
    if root is None:
        return {"files": 0, "bytes": 0}
    removed_files = 0
    removed_bytes = 0
    for path in root.glob("*.npz"):
        try:
            size = int(path.stat().st_size)
            path.unlink()
            removed_files += 1
            removed_bytes += size
        except OSError:
            continue
    record_cache_event(
        "scene_payload",
        "clear",
        count=removed_files,
        byte_count=removed_bytes,
    )
    set_cache_inventory("scene_payload", entries=0, byte_count=0, root=str(root))
    return {"files": removed_files, "bytes": removed_bytes}


def _path_signature(path: Path) -> dict[str, Any] | None:
    """Return a cheap invalidation signature for a scene or mesh file."""
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return {
        "path": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "size": int(stat.st_size),
    }


def _mesh_shape_records(xml_root: Any, xml_path: str) -> list[dict[str, Any]]:
    """Return cache/build records for mesh-backed XML shapes."""
    base_dir = os.path.dirname(xml_path)
    records: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(xml_root.findall("shape")):
        shape_type = shape.get("type")
        if shape_type not in ["ply", "obj"]:
            continue

        fn_elem = shape.find("string[@name='filename']")
        ref_elem = shape.find("ref[@name='bsdf']")
        if ref_elem is None:
            ref_elem = shape.find("ref[@id]")

        if fn_elem is None or ref_elem is None:
            continue

        rel_path = fn_elem.get("value")
        if not rel_path:
            continue
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
        transform_state = parse_lightweight_shape_transform(
            shape,
            source_xml=xml_path,
            shape_index=shape_index,
        )
        records.append(
            {
                "shape_index": shape_index,
                "shape": shape,
                "rel_path": rel_path,
                "full_path": full_path,
                "material_id": ref_elem.get("id"),
                "transform_state": transform_state,
            }
        )
    return records


def _scene_payload_cache_fingerprint(
    xml_root: Any, xml_path: str, records: list[dict[str, Any]]
) -> str | None:
    """Return the cache fingerprint for the XML plus referenced mesh files."""
    xml_sig = _path_signature(Path(xml_path))
    if xml_sig is None:
        return None
    xml_sig["content_sha256"] = hashlib.sha256(ET.tostring(xml_root, encoding="utf-8")).hexdigest()

    mesh_sigs: list[dict[str, Any]] = []
    for record in records:
        mesh_sig = _path_signature(Path(str(record["full_path"])))
        if mesh_sig is None:
            return None
        # Reuse this authoritative revision throughout material/UV/buffer
        # preparation. Re-statting every mesh at each cache layer turns one
        # safe fingerprint pass into several thousand duplicate syscalls.
        record["source_signature"] = dict(mesh_sig)
        mesh_sigs.append(
            {
                "shape_index": int(record["shape_index"]),
                "rel_path": str(record["rel_path"]),
                **mesh_sig,
            }
        )

    payload = {
        "schema_version": _SCENE_PAYLOAD_CACHE_VERSION,
        "xml": xml_sig,
        "meshes": mesh_sigs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_payload_cache_path(xml_path: str, fingerprint: str) -> Path | None:
    """Return the neutral scene payload cache path for one fingerprint."""
    root = _scene_payload_cache_root()
    if root is None:
        return None
    label = _sanitize_cache_segment(Path(xml_path).stem)
    return root / f"{label}.{fingerprint[:16]}.npz"


def _pack_scene_payload_arrays(
    arrays: list[Any | None],
    *,
    width: int,
    dtype: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length 2D arrays into one array plus entry offsets."""
    parts: list[np.ndarray] = []
    offsets = [0]
    present: list[bool] = []
    for value in arrays:
        if value is None:
            present.append(False)
            offsets.append(offsets[-1])
            continue
        array = np.asarray(value, dtype=dtype).reshape((-1, width))
        present.append(True)
        parts.append(array)
        offsets.append(offsets[-1] + len(array))

    packed = (
        np.vstack(parts).astype(dtype, copy=False) if parts else np.empty((0, width), dtype=dtype)
    )
    return (
        packed,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(present, dtype=bool),
    )


def _scene_payload_slice(data: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    """Return one immutable entry slice with a single ownership copy."""
    start = int(offsets[index])
    end = int(offsets[index + 1])
    array = np.ascontiguousarray(data[start:end])
    storage = array.tobytes(order="C")
    return np.frombuffer(storage, dtype=array.dtype).reshape(array.shape)


def _scene_payload_optional_slice(
    data: np.ndarray,
    offsets: np.ndarray,
    present: np.ndarray,
    index: int,
) -> np.ndarray | None:
    """Return an optional copied entry slice from a packed scene-payload array."""
    if not bool(present[index]):
        return None
    return _scene_payload_slice(data, offsets, index)


def _scene_payload_vector(value: Any) -> np.ndarray:
    """Return a 3-vector for persisted scene-entry center fields."""
    return np.asarray(value, dtype=np.float64).reshape((3,))


def _store_scene_payload_cache(
    cache_path: Path,
    fingerprint: str,
    entries: list[dict[str, Any]],
) -> None:
    """Persist neutral transformed scene payloads for repeat scene opens."""
    if not entries:
        return

    start = time.perf_counter()
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npz")
    try:
        metadata: list[dict[str, Any]] = []
        meshes: list[MeshPayload] = []
        for entry in entries:
            mesh = entry.get("mesh")
            if not isinstance(mesh, MeshPayload):
                return
            meshes.append(mesh)
            metadata.append(
                {
                    "name": entry.get("name"),
                    "material_id": entry.get("material_id"),
                    "material_type": entry.get("material_type", "default"),
                    "pbr_properties": entry.get("pbr_properties") or {},
                    "transform_state": entry.get("transform_state") or {},
                    "color": entry.get("color", [0.7, 0.7, 0.7]),
                    "rel_path": entry.get("rel_path"),
                    "shape_index": int(entry.get("shape_index", -1)),
                    "mesh_cache_key": mesh.cache_key,
                    "source_signature": entry.get("_source_signature"),
                }
            )

        vertices, vertices_offsets, _ = _pack_scene_payload_arrays(
            [mesh.vertices for mesh in meshes],
            width=3,
            dtype=np.float64,
        )
        triangles, triangles_offsets, _ = _pack_scene_payload_arrays(
            [mesh.triangles for mesh in meshes],
            width=3,
            dtype=np.int32,
        )
        original_vertices, original_vertices_offsets, _ = _pack_scene_payload_arrays(
            [entry.get("original_vertices") for entry in entries],
            width=3,
            dtype=np.float64,
        )
        normals, normals_offsets, normals_present = _pack_scene_payload_arrays(
            [mesh.normals for mesh in meshes],
            width=3,
            dtype=np.float64,
        )
        vertex_colors, vertex_colors_offsets, vertex_colors_present = _pack_scene_payload_arrays(
            [mesh.vertex_colors for mesh in meshes],
            width=3,
            dtype=np.float64,
        )
        triangle_uvs, triangle_uvs_offsets, triangle_uvs_present = _pack_scene_payload_arrays(
            [mesh.triangle_uvs for mesh in meshes],
            width=2,
            dtype=np.float64,
        )

        np.savez(
            tmp_path,
            schema_version=np.asarray(_SCENE_PAYLOAD_CACHE_VERSION),
            fingerprint=np.asarray(fingerprint),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            original_centers=np.vstack(
                [_scene_payload_vector(entry["original_center"]) for entry in entries]
            ),
            position_after_scale_rotation=np.vstack(
                [_scene_payload_vector(entry["position_after_scale_rotation"]) for entry in entries]
            ),
            current_centers=np.vstack(
                [_scene_payload_vector(entry["current_center"]) for entry in entries]
            ),
            vertices=vertices,
            vertices_offsets=vertices_offsets,
            triangles=triangles,
            triangles_offsets=triangles_offsets,
            original_vertices=original_vertices,
            original_vertices_offsets=original_vertices_offsets,
            normals=normals,
            normals_offsets=normals_offsets,
            normals_present=normals_present,
            vertex_colors=vertex_colors,
            vertex_colors_offsets=vertex_colors_offsets,
            vertex_colors_present=vertex_colors_present,
            triangle_uvs=triangle_uvs,
            triangle_uvs_offsets=triangle_uvs_offsets,
            triangle_uvs_present=triangle_uvs_present,
        )
        os.replace(tmp_path, cache_path)
        _enforce_scene_payload_cache_limits(cache_path.parent)
        try:
            byte_count = int(cache_path.stat().st_size)
        except OSError:
            byte_count = 0
        record_cache_event(
            "scene_payload",
            "write",
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            byte_count=byte_count,
        )
        logger.debug("Stored neutral scene payload cache: %s", cache_path)
    except (OSError, TypeError, ValueError) as exc:
        record_cache_event(
            "scene_payload",
            "write_failure",
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
        logger.debug("Failed to store neutral scene payload cache %s: %s", cache_path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_scene_payload_cache(
    cache_path: Path,
    fingerprint: str,
    xml_root: Any,
) -> list[dict[str, Any]] | None:
    """Load neutral transformed scene payloads and reattach current XML references."""
    start = time.perf_counter()
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["schema_version"].item()) != _SCENE_PAYLOAD_CACHE_VERSION:
                record_cache_event("scene_payload", "invalid")
                return None
            if str(data["fingerprint"].item()) != fingerprint:
                record_cache_event("scene_payload", "invalid")
                return None

            metadata = json.loads(str(data["metadata_json"].item()))
            count = len(metadata)
            vertices = data["vertices"]
            vertices_offsets = data["vertices_offsets"]
            triangles = data["triangles"]
            triangles_offsets = data["triangles_offsets"]
            original_vertices = data["original_vertices"]
            original_vertices_offsets = data["original_vertices_offsets"]
            normals = data["normals"]
            normals_offsets = data["normals_offsets"]
            normals_present = data["normals_present"]
            vertex_colors = data["vertex_colors"]
            vertex_colors_offsets = data["vertex_colors_offsets"]
            vertex_colors_present = data["vertex_colors_present"]
            triangle_uvs = data["triangle_uvs"]
            triangle_uvs_offsets = data["triangle_uvs_offsets"]
            triangle_uvs_present = data["triangle_uvs_present"]
            original_centers = data["original_centers"]
            position_after_scale_rotation = data["position_after_scale_rotation"]
            current_centers = data["current_centers"]

            required_offsets = (
                vertices_offsets,
                triangles_offsets,
                original_vertices_offsets,
            )
            if any(len(offsets) != count + 1 for offsets in required_offsets):
                return None
            for present in (
                normals_present,
                vertex_colors_present,
                triangle_uvs_present,
            ):
                if len(present) != count:
                    return None

            materials = MaterialHandler.parse_materials(xml_root)
            shapes = xml_root.findall("shape")
            entries: list[dict[str, Any]] = []
            for index, item in enumerate(metadata):
                shape_index = int(item.get("shape_index", -1))
                if shape_index < 0 or shape_index >= len(shapes):
                    return None
                material_id = item.get("material_id")
                material_info = materials.get(material_id, {})

                mesh = MeshPayload(
                    vertices=_scene_payload_slice(vertices, vertices_offsets, index),
                    triangles=_scene_payload_slice(triangles, triangles_offsets, index),
                    normals=_scene_payload_optional_slice(
                        normals,
                        normals_offsets,
                        normals_present,
                        index,
                    ),
                    vertex_colors=_scene_payload_optional_slice(
                        vertex_colors,
                        vertex_colors_offsets,
                        vertex_colors_present,
                        index,
                    ),
                    triangle_uvs=_scene_payload_optional_slice(
                        triangle_uvs,
                        triangle_uvs_offsets,
                        triangle_uvs_present,
                        index,
                    ),
                    cache_key=item.get("mesh_cache_key"),
                )

                entries.append(
                    {
                        "name": item.get("name"),
                        "mesh": mesh,
                        "material_id": material_id,
                        "material_type": item.get("material_type", "default"),
                        "pbr_properties": dict(item.get("pbr_properties") or {}),
                        "original_center": np.array(original_centers[index], copy=True),
                        "position_after_scale_rotation": np.array(
                            position_after_scale_rotation[index],
                            copy=True,
                        ),
                        "current_center": np.array(current_centers[index], copy=True),
                        "original_vertices": _scene_payload_slice(
                            original_vertices,
                            original_vertices_offsets,
                            index,
                        ),
                        "transform_state": dict(item.get("transform_state") or {}),
                        "color": list(item.get("color") or [0.7, 0.7, 0.7]),
                        "visible": True,
                        "show_label": False,
                        "highlighted": False,
                        "id_edit": None,
                        "entry_type": "mesh",
                        "xml_bsdf": material_info.get("xml_element"),
                        "xml_shape": shapes[shape_index],
                        "rel_path": item.get("rel_path"),
                        "shape_index": shape_index,
                        "_source_signature": item.get("source_signature"),
                    }
                )
            logger.debug("Loaded neutral scene payload cache: %s", cache_path)
            try:
                byte_count = int(cache_path.stat().st_size)
                os.utime(cache_path, None)
            except OSError:
                byte_count = 0
            record_cache_event(
                "scene_payload",
                "hit",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                byte_count=byte_count,
            )
            return entries
    except (OSError, KeyError, TypeError, ValueError) as exc:
        record_cache_event(
            "scene_payload",
            "read_failure",
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
        logger.debug("Failed to load neutral scene payload cache %s: %s", cache_path, exc)
        return None


def _find_scenario_root_for_source(source: Path) -> Path | None:
    """Return the nearest ancestor directory containing ``scenario.yaml``."""
    current = source.parent if source.suffix else source
    for parent in (current, *current.parents):
        if (parent / "scenario.yaml").is_file():
            return parent
    return None


def _uv_cache_namespace(source: Path) -> str:
    """Return a readable per-scenario namespace for cached UV files."""
    scenario_root = _find_scenario_root_for_source(source)
    if scenario_root is not None:
        label = _sanitize_cache_segment(scenario_root.name)
        digest_source = str(scenario_root)
    else:
        label = _sanitize_cache_segment(source.parent.name or source.stem)
        digest_source = str(source.parent)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{label}.{digest}"


def _uv_cache_max_age_days() -> int:
    """Return the stale-entry pruning threshold in days."""
    raw_value = os.environ.get("ORCHAV_UV_CACHE_MAX_AGE_DAYS")
    if raw_value is None:
        return _DEFAULT_UV_CACHE_MAX_AGE_DAYS
    try:
        return max(0, int(float(raw_value)))
    except (TypeError, ValueError):
        return _DEFAULT_UV_CACHE_MAX_AGE_DAYS


def _uv_cache_max_bytes() -> int:
    """Return the aggregate generated-UV disk budget."""
    raw_value = os.environ.get("ORCHAV_UV_CACHE_MAX_BYTES")
    if raw_value is None:
        return _DEFAULT_UV_CACHE_MAX_BYTES
    try:
        return max(0, int(float(raw_value)))
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_UV_CACHE_MAX_BYTES


def _prune_uv_cache_root(root: Path) -> None:
    """Prune inactive scene-level UV stores once per process."""
    root_key = str(root.resolve(strict=False))
    if root_key in _PRUNED_UV_CACHE_ROOTS:
        return
    _PRUNED_UV_CACHE_ROOTS.add(root_key)

    removed_files = prune_uv_cache(
        root,
        max_age_days=_uv_cache_max_age_days(),
        max_bytes=_uv_cache_max_bytes(),
    )
    if removed_files:
        logger.info("Pruned %d stale UV cache stores from %s", removed_files, root)


def finalize_uv_cache_stores() -> int:
    """Commit and close scene UV stores, then enforce the aggregate disk policy."""
    root = _uv_cache_root()
    close_uv_cache_stores(root=root, flush=True)
    removed_files = prune_uv_cache(
        root,
        max_age_days=_uv_cache_max_age_days(),
        max_bytes=_uv_cache_max_bytes(),
    )
    if removed_files:
        logger.info("Pruned %d generated UV cache stores from %s", removed_files, root)
    return removed_files


def _box_projection_uv_cache_identity(
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_signature: dict[str, Any] | None = None,
) -> tuple[Any, str, str]:
    """Return the namespace store, entry key, and source signature."""
    source = Path(source_path).expanduser().resolve(strict=False)
    supplied_path = str((source_signature or {}).get("path") or "")
    signature_matches = bool(
        source_signature and supplied_path and Path(supplied_path).resolve(strict=False) == source
    )
    if signature_matches:
        source_size = source_signature.get("size")
        source_mtime_ns = source_signature.get("mtime_ns")
        source_ctime_ns = source_signature.get("ctime_ns")
    else:
        try:
            stat = source.stat()
            source_size = int(stat.st_size)
            source_mtime_ns = int(stat.st_mtime_ns)
            source_ctime_ns = int(stat.st_ctime_ns)
        except OSError:
            source_size = None
            source_mtime_ns = None
            source_ctime_ns = None

    payload = {
        "version": _BOX_PROJECTION_UV_CACHE_VERSION,
        "source_path": str(source),
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "source_ctime_ns": source_ctime_ns,
        "transform_state": transform_state or {},
        "scale": round(float(scale), 9),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    store = get_uv_cache_store(_uv_cache_root(), _uv_cache_namespace(source))
    return store, digest, encoded


def _box_projection_uv_cache_path(
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_signature: dict[str, Any] | None = None,
) -> Path:
    """Return the scene-level UV store path for one mesh identity."""
    store, _cache_key, _signature = _box_projection_uv_cache_identity(
        source_path=source_path,
        transform_state=transform_state,
        scale=scale,
        source_signature=source_signature,
    )
    return store.path


def resolve_box_projection_uv_cache_path(
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_signature: dict[str, Any] | None = None,
) -> Path:
    """Return the persisted UV cache-file path for one mesh/transform/scale tuple."""
    return _box_projection_uv_cache_path(
        source_path=source_path,
        transform_state=transform_state,
        scale=scale,
        source_signature=source_signature,
    )


def box_projection_uv_cache_contains(
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_signature: dict[str, Any] | None = None,
) -> bool:
    """Return whether the scene-level store contains one generated UV entry."""
    store, cache_key, _signature = _box_projection_uv_cache_identity(
        source_path=source_path,
        transform_state=transform_state,
        scale=scale,
        source_signature=source_signature,
    )
    return store.contains(cache_key)


def _load_cached_box_projection_uvs(
    mesh: Any,
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_revision: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Load cached UVs for a transformed architectural mesh if present."""
    store, cache_key, source_signature = _box_projection_uv_cache_identity(
        source_path=source_path,
        transform_state=transform_state,
        scale=scale,
        source_signature=source_revision,
    )
    expected_shape = (len(np.asarray(mesh.triangles)) * 3, 2)
    return store.get(
        cache_key,
        expected_shape=expected_shape,
        source_signature=source_signature,
    )


def _store_cached_box_projection_uvs(
    uvs: np.ndarray,
    *,
    source_path: str,
    transform_state: dict[str, Any] | None,
    scale: float,
    source_revision: dict[str, Any] | None = None,
) -> None:
    """Persist box-projected UVs for reuse on later launches."""
    store, cache_key, source_signature = _box_projection_uv_cache_identity(
        source_path=source_path,
        transform_state=transform_state,
        scale=scale,
        source_signature=source_revision,
    )
    store.put(cache_key, uvs, source_signature=source_signature)


def clear_persistent_uv_cache() -> dict[str, int]:
    """Clear every persisted generated-UV store under the active cache root."""
    return clear_uv_cache(_uv_cache_root())


def get_persistent_uv_cache_info() -> dict[str, int | str]:
    """Return generated-UV disk inventory and configured lifecycle limits."""
    info = get_uv_cache_info(_uv_cache_root())
    info["max_age_days"] = _uv_cache_max_age_days()
    info["max_bytes"] = _uv_cache_max_bytes()
    return info


def load_or_generate_box_projection_uvs(
    mesh: Any,
    *,
    scale: float,
    cache_source_path: str | None = None,
    transform_state: dict[str, Any] | None = None,
    source_signature: dict[str, Any] | None = None,
    cache_miss_callback: Callable[[], None] | None = None,
) -> np.ndarray | None:
    """Load cached architectural UVs when possible, otherwise generate them.

    The cache is intentionally modest: it stores only the generated UV
    array, keyed by the source mesh file, its file timestamp/size, the
    applied transform, the UV scale, and an algorithm version string.
    This speeds up repeat launches of the same textured city-scale scenes
    without trying to persist renderer-specific GPU objects.
    """
    if cache_source_path:
        cached = _load_cached_box_projection_uvs(
            mesh,
            source_path=cache_source_path,
            transform_state=transform_state,
            scale=scale,
            source_revision=source_signature,
        )
        if cached is not None:
            return cached
        if cache_miss_callback is not None:
            cache_miss_callback()

    generated = _generate_box_projection_uvs(mesh, scale=scale)
    if generated is not None:
        generated = np.ascontiguousarray(generated, dtype=np.float32)
    if generated is not None and cache_source_path:
        _store_cached_box_projection_uvs(
            generated,
            source_path=cache_source_path,
            transform_state=transform_state,
            scale=scale,
            source_revision=source_signature,
        )
    return generated


class MaterialHandler:
    """Handles material (BSDF) operations from Mitsuba XML scenes."""

    @staticmethod
    def parse_pbr_properties_from_bsdf(bsdf_element) -> dict[str, Any]:  # pragma: no cover
        """
        Extract PBR properties from a BSDF XML element.

        Reads material type and PBR properties (roughness, metallic, reflectance, alpha)
        from the XML, with fallback to catalog defaults.

        Args:
            bsdf_element: BSDF XML element

        Returns:
            Dict with PBR properties: material_type, color, roughness, metallic, reflectance, alpha, thickness
        """
        material_type = "default"
        explicit_props: set[str] = set()
        type_elem = bsdf_element.find("./string[@name='type']")
        if type_elem is not None:
            material_type = type_elem.get("value", "default")
        material_type = normalize_material_type_name(material_type)

        defaults = material_preset(material_type)

        # Override with XML-specified values if present
        props = defaults.copy()

        rgb_el = bsdf_element.find(".//rgb[@name='reflectance']")
        if rgb_el is None:
            rgb_el = bsdf_element.find(".//rgb[@name='color']")
        if rgb_el is not None:
            vals = [float(v.strip(",")) for v in rgb_el.get("value").split()]
            props["color"] = vals
            explicit_props.add("color")

        for prop_name in [
            "roughness",
            "metallic",
            "reflectance",
            "alpha",
            "thickness",
            "normal_map_strength",
            "uv_scale_meters",
        ]:
            prop_elem = bsdf_element.find(f"./float[@name='{prop_name}']")
            if prop_elem is not None:
                try:
                    props[prop_name] = float(prop_elem.get("value"))
                    explicit_props.add(prop_name)
                except (ValueError, TypeError):
                    pass  # Keep default value

        for prop_name in [
            "texture_path",
            "normal_map_path",
            "roughness_map_path",
            "ao_map_path",
            "metallic_map_path",
            "shader_variant",
        ]:
            prop_elem = bsdf_element.find(f"./string[@name='{prop_name}']")
            if prop_elem is not None:
                value = prop_elem.get("value")
                props[prop_name] = value if value else None
                explicit_props.add(prop_name)

        props["material_type"] = material_type
        props["_explicit_props"] = sorted(explicit_props)
        return props

    @staticmethod
    def _normalize_material_type_name(material_type: str) -> str:
        """Normalize XML material type strings through the shared catalog."""
        return normalize_material_type_name(material_type)

    @staticmethod
    def _infer_material_type_from_id(material_id: str | None) -> str:
        """
        Infer material type from material ID name.

        Handles patterns like:
        - mat-itu_wood -> wood
        - mat-itu_ceiling_board -> ceiling_board
        - itu_concrete -> concrete
        - concrete_mat -> concrete

        Args:
            material_id: Material ID string from XML

        Returns:
            Inferred material type or "default" if unknown
        """
        return infer_material_type_from_id(material_id)

    @staticmethod
    def _material_id_stem(material_id: str | None) -> str:
        """Return the normalized material ID stem used for visual inference."""
        return material_id_stem(material_id)

    @staticmethod
    def parse_materials(xml_root) -> dict[str, MaterialInfo]:  # pragma: no cover
        """
        Parse materials from Mitsuba XML scene.

        Args:
            xml_root: Root element of the XML scene

        Returns:
            Dictionary mapping material ID to material info
        """
        mat_map = {}

        for bsdf in xml_root.findall("bsdf"):
            mid = bsdf.get("id")

            pbr_props = MaterialHandler.parse_pbr_properties_from_bsdf(bsdf)

            # If material_type is still "default", try to infer from material ID.
            # Ground helper scenes intentionally decouple propagation and visuals,
            # e.g. mat-ground_asphalt with type=concrete for Sionna RT.
            explicit_props = set(pbr_props.pop("_explicit_props", []))
            inferred_type = MaterialHandler._infer_material_type_from_id(mid)
            id_stem = MaterialHandler._material_id_stem(mid)
            if inferred_type != "default" and (
                id_stem.startswith("ground_")
                or pbr_props["material_type"] == "default"
                or not is_known_material_type(pbr_props["material_type"])
            ):
                pbr_props["material_type"] = inferred_type
                # Also update PBR defaults based on inferred type.
                type_defaults = material_preset(inferred_type)
                for key, value in type_defaults.items():
                    if key not in explicit_props:
                        pbr_props[key] = value

            # Maintain backward compatibility - color is at top level
            mat_map[mid] = {
                "id": mid,
                "color": pbr_props["color"],
                "xml_element": bsdf,
                "material_type": pbr_props["material_type"],
                "pbr_properties": pbr_props,  # Full PBR property dict
            }

        logger.debug(f"Parsed {len(mat_map)} materials from XML scene")
        return mat_map

    @staticmethod
    def update_material_color(xml_bsdf_element, new_color: list[float]) -> None:  # pragma: no cover
        """
        Update the color of a material in the XML.

        Args:
            xml_bsdf_element: The BSDF XML element to update
            new_color: New RGB color values [r, g, b]
        """
        try:
            logger.debug(f"Updating material color for BSDF element: {xml_bsdf_element.tag}")
            logger.debug(f"BSDF element attributes: {xml_bsdf_element.attrib}")

            # Work directly with the XML tree structure; find() results can be detached.

            # Find the rgb element by iterating through the tree structure
            rgb_element_to_update = None
            parent_of_rgb = None

            # Look for rgb element directly under the BSDF
            # Try both 'reflectance' (standard) and 'color' (itu-radio-material)
            for child in xml_bsdf_element:
                if child.tag == "rgb" and child.get("name") in ("reflectance", "color"):
                    rgb_element_to_update = child
                    parent_of_rgb = xml_bsdf_element
                    logger.debug(f"Found rgb element directly in BSDF: {child.attrib}")
                    break

            # If not found, look for it in a diffuse BSDF sub-element
            if rgb_element_to_update is None:
                for child in xml_bsdf_element:
                    if child.tag == "bsdf" and child.get("type") == "diffuse":
                        for subchild in child:
                            if subchild.tag == "rgb" and subchild.get("name") == "reflectance":
                                rgb_element_to_update = subchild
                                parent_of_rgb = child
                                logger.debug(
                                    f"Found rgb element in diffuse BSDF: {subchild.attrib}"
                                )
                                break
                        if rgb_element_to_update is not None:
                            break

            # If still not found, create the proper structure
            if rgb_element_to_update is None:
                logger.debug("No existing rgb element found, creating new structure")

                bsdf_type = xml_bsdf_element.get("type", "")
                # For itu-radio-material, use 'color' instead of 'reflectance'
                rgb_name = "color" if bsdf_type == "itu-radio-material" else "reflectance"

                if bsdf_type == "diffuse":
                    rgb_element_to_update = ET.SubElement(
                        xml_bsdf_element,
                        "rgb",
                        {"name": rgb_name, "value": " ".join(f"{v:.6f}" for v in new_color)},
                    )
                    parent_of_rgb = xml_bsdf_element
                    logger.debug(
                        f"Created rgb element directly in diffuse BSDF: {rgb_element_to_update.attrib}"
                    )
                elif bsdf_type == "itu-radio-material":
                    # For itu-radio-material, add rgb element directly with name='color'
                    rgb_element_to_update = ET.SubElement(
                        xml_bsdf_element,
                        "rgb",
                        {"name": "color", "value": " ".join(f"{v:.6f}" for v in new_color)},
                    )
                    parent_of_rgb = xml_bsdf_element
                    logger.debug(
                        f"Created rgb element directly in itu-radio-material BSDF: {rgb_element_to_update.attrib}"
                    )
                else:
                    diff = ET.SubElement(
                        xml_bsdf_element, "bsdf", {"type": "diffuse", "name": "bsdf"}
                    )
                    rgb_element_to_update = ET.SubElement(
                        diff,
                        "rgb",
                        {"name": "reflectance", "value": " ".join(f"{v:.6f}" for v in new_color)},
                    )
                    parent_of_rgb = diff
                    logger.debug(
                        f"Created diffuse BSDF with rgb element: {rgb_element_to_update.attrib}"
                    )
            else:
                old_value = rgb_element_to_update.get("value")
                logger.debug(f"Updating existing rgb element from {old_value} to new color")

                rgb_element_to_update.set("value", " ".join(f"{v:.6f}" for v in new_color))
                logger.debug(f"Updated rgb element: {rgb_element_to_update.attrib}")

                # Verify that the tree was updated.
                if parent_of_rgb is not None:
                    # Find the same element again in the tree to verify
                    for child in parent_of_rgb:
                        if child.tag == "rgb" and child.get("name") == "reflectance":
                            if child is rgb_element_to_update:
                                logger.debug("Element identity verified - same object")
                            else:
                                logger.warning("Element identity mismatch - different objects")

                            tree_value = child.get("value")
                            expected_value = " ".join(f"{v:.6f}" for v in new_color)

                            if tree_value == expected_value:
                                logger.debug(f"Tree update verified: {tree_value}")
                            else:
                                logger.error(
                                    f"Tree update failed! Expected: {expected_value}, Got: {tree_value}"
                                )
                                # Force update the tree element
                                child.set("value", expected_value)
                                logger.debug(f"Force-updated tree element: {child.attrib}")
                            break
                else:
                    logger.warning("Could not verify tree update - no parent element")

            # FINAL VERIFICATION: Ensure the XML tree contains the correct value
            if parent_of_rgb is not None:
                for child in parent_of_rgb:
                    if child.tag == "rgb" and child.get("name") == "reflectance":
                        final_value = child.get("value")
                        expected_value = " ".join(f"{v:.6f}" for v in new_color)

                        if final_value == expected_value:
                            logger.debug(
                                f"Final verification: XML tree correctly updated to {final_value}"
                            )
                        else:
                            logger.error(
                                f"Final verification failed: Expected {expected_value}, got {final_value}"
                            )
                            # Last attempt: force update
                            child.set("value", expected_value)
                            logger.debug(f"Last-resort update applied: {child.attrib}")
                        break
            else:
                logger.warning("Could not perform final verification - no parent element")

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to update material color: {e}")
            logger.error(f"BSDF element structure: {[child.tag for child in xml_bsdf_element]}")
            raise

    @staticmethod
    def test_xml_modification():  # pragma: no cover
        """Run the standalone XML material-color update diagnostic."""
        try:
            root = ET.Element("scene")

            bsdf = ET.SubElement(root, "bsdf", {"id": "test_material", "type": "diffuse"})

            logger.info("Testing XML modification...")
            logger.info(f"Before modification: {[child.tag for child in bsdf]}")

            MaterialHandler.update_material_color(bsdf, [0.8, 0.2, 0.3])

            logger.info(f"After modification: {[child.tag for child in bsdf]}")

            rgb_el = bsdf.find('rgb[@name="reflectance"]')
            if rgb_el is not None:
                logger.info(f"Color update successful: {rgb_el.attrib}")
            else:
                logger.error("Color update failed - no rgb element found")

            test_file = "/tmp/test_xml_modification.xml"
            XMLSceneHandler.save_xml_scene(root, test_file)

            with open(test_file) as f:
                content = f.read()
                if "reflectance" in content and "0.800000 0.200000 0.300000" in content:
                    logger.info("XML saving successful - color preserved")
                else:
                    logger.error("XML saving failed - color not preserved")
                    logger.error(f"File content: {content}")

            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)

        except (OSError, ValueError, RuntimeError):
            logger.exception("XML modification test failed")


def _canonicalize_horizontal_axes(u_axes: np.ndarray) -> np.ndarray:
    """Flip horizontal axes so equivalent directions choose one stable sign."""
    canonical = np.asarray(u_axes, dtype=np.float64).copy()
    x_dominant = np.abs(canonical[:, 0]) >= np.abs(canonical[:, 1])
    flip_mask = np.where(x_dominant, canonical[:, 0] < 0.0, canonical[:, 1] < 0.0)
    canonical[flip_mask] *= -1.0
    return canonical


def _quantize_horizontal_axes(u_axes: np.ndarray, *, bin_degrees: float) -> np.ndarray:
    """Snap horizontal directions to coarse angle bins for stable large-mesh UVs."""
    angles = np.arctan2(u_axes[:, 1], u_axes[:, 0])
    step = np.deg2rad(bin_degrees)
    snapped = np.round(angles / step) * step
    quantized = np.stack(
        [np.cos(snapped), np.sin(snapped), np.zeros_like(snapped)],
        axis=1,
    )
    return _canonicalize_horizontal_axes(quantized)


def _generate_orientation_binned_uvs(
    *,
    local_vertices: np.ndarray,
    triangles: np.ndarray,
    face_normals: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Fast UV projection for large meshes using vectorized orientation bins."""
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tri_count = len(triangles)
    u_axis_unit = np.zeros((tri_count, 3), dtype=np.float64)
    v_axis_unit = np.zeros((tri_count, 3), dtype=np.float64)

    abs_nz = np.abs(face_normals[:, 2])
    flat_mask = abs_nz >= 0.95
    vertical_mask = abs_nz <= 0.15
    sloped_mask = ~(flat_mask | vertical_mask)

    if np.any(flat_mask):
        u_axis_unit[flat_mask] = world_x
        v_axis_unit[flat_mask] = world_y

    if np.any(vertical_mask):
        raw_u = np.cross(
            np.broadcast_to(world_up, (int(np.sum(vertical_mask)), 3)), face_normals[vertical_mask]
        )
        raw_u_norm = np.linalg.norm(raw_u, axis=1)
        safe_mask = raw_u_norm >= 1e-8
        raw_u[safe_mask] /= raw_u_norm[safe_mask, None]
        raw_u[~safe_mask] = world_x
        u_axis_unit[vertical_mask] = _quantize_horizontal_axes(
            raw_u,
            bin_degrees=_ORIENTATION_BIN_DEGREES,
        )
        v_axis_unit[vertical_mask] = world_up

    if np.any(sloped_mask):
        raw_u = np.cross(face_normals[sloped_mask], world_up)
        raw_u_norm = np.linalg.norm(raw_u, axis=1)
        safe_mask = raw_u_norm >= 1e-8
        raw_u[safe_mask] /= raw_u_norm[safe_mask, None]
        raw_u[~safe_mask] = world_x
        quantized_u = _quantize_horizontal_axes(
            raw_u,
            bin_degrees=_ORIENTATION_BIN_DEGREES,
        )
        raw_v = np.cross(face_normals[sloped_mask], quantized_u)
        raw_v_norm = np.linalg.norm(raw_v, axis=1)
        safe_v_mask = raw_v_norm >= 1e-8
        raw_v[safe_v_mask] /= raw_v_norm[safe_v_mask, None]
        raw_v[~safe_v_mask] = world_y
        flip_mask = raw_v[:, 2] < 0.0
        quantized_u[flip_mask] *= -1.0
        raw_v[flip_mask] *= -1.0
        u_axis_unit[sloped_mask] = quantized_u
        v_axis_unit[sloped_mask] = raw_v

    tri_verts = local_vertices[triangles]
    u_coords = np.sum(tri_verts * u_axis_unit[:, None, :], axis=-1)
    v_coords = np.sum(tri_verts * v_axis_unit[:, None, :], axis=-1)
    triangle_uvs = np.stack([u_coords * scale, v_coords * scale], axis=-1).reshape(-1, 2)
    logger.debug(
        "Generated %d orientation-binned UVs (%d triangles, %.1f%% flat, %.1f%% vertical)",
        len(triangle_uvs),
        tri_count,
        100.0 * float(np.mean(flat_mask)) if tri_count else 0.0,
        100.0 * float(np.mean(vertical_mask)) if tri_count else 0.0,
    )
    return triangle_uvs


def _generate_box_projection_uvs(mesh: Any, scale: float = 1.0) -> "np.ndarray | None":
    """Generate UV coordinates for architectural meshes without artist-authored UVs.

    **Hybrid projection.** Smaller meshes use connected planar patches:
    triangles are grouped into connected regions with near-matching
    normals, then each patch gets a shared basis. Larger meshes use a
    vectorized orientation-binned fallback that snaps wall/roof
    directions to stable angle bins. The fast path avoids expensive
    Python adjacency construction on giant pre-merged city meshes while
    keeping texture direction coherent.

    The patch basis is chosen from world-aligned rules:

    * near-vertical patch: ``v = world_up`` and ``u`` runs along the wall
    * near-flat patch: project in world ``(x, y)``
    * sloped patch: ``u`` follows the horizontal ridge direction and
      ``v`` climbs the slope

    Coordinates are still anchored to the mesh bbox min after projection,
    so the mapping stays numerically stable on far-from-origin scenes
    while preserving world-coordinate directions.

    Vertices are origin-shifted to the mesh bbox min before projection so
    UVs live in ``[0, mesh_extent]`` regardless of where the mesh sits in
    world space. Large generated city meshes are routinely hundreds of
    metres away from the world origin; without the shift,
    UVs at ``scale=0.5`` ran past 500 tile repeats, aliased through
    mipmap LOD, and distant walls collapsed to a near-solid average
    colour of the texture instead of showing the tile pattern.

    History: replaced a dominant-axis world box projection 2026-04-14
    (stretched walls on rotated buildings), then a per-triangle
    face-local basis (fixed some roofs but let marble/brick direction
    drift between adjacent faces). The hybrid below keeps the
    world-coordinate feel while staying fast enough for giant
    building batches.

    Args:
        mesh: Mesh-like object with vertices and triangles.
        scale: UV scale factor. The final UV extent is
            ``local_coord * scale``, so ``scale=0.5`` means 1 texture tile
            per 2 m of surface. Callers typically pass
            ``1.0 / uv_scale_meters`` from the ITU preset.

    Returns:
        ``(N_tri*3, 2)`` float64 array of UV coordinates, or ``None`` if
        the mesh has no triangles or UV generation failed.
    """
    try:
        import numpy as np

        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        triangles = np.asarray(getattr(mesh, "triangles", []), dtype=np.int64)
        if vertices.size == 0 or triangles.size == 0:
            return None
        vertices = vertices.reshape((-1, 3))
        triangles = triangles.reshape((-1, 3))

        if len(triangles) == 0:
            return None

        face_normals: np.ndarray | None = None
        has_triangle_normals = getattr(mesh, "has_triangle_normals", None)
        if callable(has_triangle_normals):
            try:
                if bool(has_triangle_normals()):
                    face_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
            except (RuntimeError, ValueError, AttributeError):
                face_normals = None
        if face_normals is None:
            compute_triangle_normals = getattr(mesh, "compute_triangle_normals", None)
            if callable(compute_triangle_normals):
                try:
                    compute_triangle_normals()
                    face_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
                except (RuntimeError, ValueError, AttributeError):
                    face_normals = None
        if face_normals is None or len(face_normals) != len(triangles):
            tri_vertices = vertices[triangles]
            face_normals = np.cross(
                tri_vertices[:, 1] - tri_vertices[:, 0],
                tri_vertices[:, 2] - tri_vertices[:, 0],
            )
            lengths = np.linalg.norm(face_normals, axis=1)
            valid = lengths > 1e-12
            face_normals[valid] = face_normals[valid] / lengths[valid, None]
            face_normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Origin-shift so UVs land in [0, mesh_extent] regardless of world
        # placement. Fixes mipmap LOD collapse on meshes far from origin.
        bbox_min = vertices.min(axis=0)
        local_vertices = vertices - bbox_min

        if len(triangles) > _PATCH_PROJECTION_TRIANGLE_LIMIT:
            return _generate_orientation_binned_uvs(
                local_vertices=local_vertices,
                triangles=triangles,
                face_normals=face_normals,
                scale=scale,
            )

        # Connected near-coplanar patches
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for tri_idx, tri in enumerate(triangles):
            i0, i1, i2 = (int(tri[0]), int(tri[1]), int(tri[2]))
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                key = (a, b) if a < b else (b, a)
                edge_to_triangles.setdefault(key, []).append(tri_idx)

        adjacency: list[list[int]] = [[] for _ in range(len(triangles))]
        coplanar_cos = float(np.cos(np.deg2rad(12.0)))
        for tri_list in edge_to_triangles.values():
            if len(tri_list) < 2:
                continue
            for idx_a in range(len(tri_list) - 1):
                tri_a = tri_list[idx_a]
                normal_a = face_normals[tri_a]
                for idx_b in range(idx_a + 1, len(tri_list)):
                    tri_b = tri_list[idx_b]
                    normal_b = face_normals[tri_b]
                    if float(np.dot(normal_a, normal_b)) >= coplanar_cos:
                        adjacency[tri_a].append(tri_b)
                        adjacency[tri_b].append(tri_a)

        patch_ids = np.full(len(triangles), -1, dtype=np.int32)
        patch_count = 0
        stack: list[int] = []
        for tri_idx in range(len(triangles)):
            if patch_ids[tri_idx] >= 0:
                continue
            patch_ids[tri_idx] = patch_count
            stack.append(tri_idx)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if patch_ids[neighbor] >= 0:
                        continue
                    patch_ids[neighbor] = patch_count
                    stack.append(neighbor)
            patch_count += 1

        u_axis_unit = np.zeros((len(triangles), 3), dtype=np.float64)
        v_axis_unit = np.zeros((len(triangles), 3), dtype=np.float64)
        flat_patch_count = 0
        vertical_patch_count = 0

        for patch_id in range(patch_count):
            tri_mask = patch_ids == patch_id
            patch_normals = face_normals[tri_mask]
            avg_normal = patch_normals.mean(axis=0)
            avg_norm = float(np.linalg.norm(avg_normal))
            if avg_norm < 1e-8:
                avg_normal = world_up.copy()
            else:
                avg_normal /= avg_norm

            abs_nz = abs(float(avg_normal[2]))
            if abs_nz >= 0.95:
                flat_patch_count += 1
                u_axis = world_x.copy()
                v_axis = world_y.copy()
            elif abs_nz <= 0.15:
                vertical_patch_count += 1
                u_axis = np.cross(world_up, avg_normal)
                u_norm = float(np.linalg.norm(u_axis))
                if u_norm < 1e-8:
                    u_axis = world_x.copy()
                else:
                    u_axis /= u_norm
                if abs(float(u_axis[0])) >= abs(float(u_axis[1])):
                    if u_axis[0] < 0.0:
                        u_axis *= -1.0
                elif u_axis[1] < 0.0:
                    u_axis *= -1.0
                v_axis = world_up.copy()
            else:
                u_axis = np.cross(avg_normal, world_up)
                u_norm = float(np.linalg.norm(u_axis))
                if u_norm < 1e-8:
                    u_axis = world_x.copy()
                else:
                    u_axis /= u_norm
                if abs(float(u_axis[0])) >= abs(float(u_axis[1])):
                    if u_axis[0] < 0.0:
                        u_axis *= -1.0
                elif u_axis[1] < 0.0:
                    u_axis *= -1.0
                v_axis = np.cross(avg_normal, u_axis)
                v_norm = float(np.linalg.norm(v_axis))
                if v_norm < 1e-8:
                    v_axis = world_y.copy()
                else:
                    v_axis /= v_norm
                if v_axis[2] < 0.0:
                    u_axis *= -1.0
                    v_axis *= -1.0

            u_axis_unit[tri_mask] = u_axis
            v_axis_unit[tri_mask] = v_axis

        # Triangle vertex positions: (N_tri, 3, 3) — 3 corners × xyz.
        tri_verts = local_vertices[triangles]

        # Project each corner onto (u_axis, v_axis). Broadcast axis over
        # the 3 corners via [:, None, :] → (N_tri, 1, 3) × (N_tri, 3, 3).
        u_coords = np.sum(tri_verts * u_axis_unit[:, None, :], axis=-1)  # (N_tri, 3)
        v_coords = np.sum(tri_verts * v_axis_unit[:, None, :], axis=-1)  # (N_tri, 3)

        u_coords *= scale
        v_coords *= scale

        # Interleave into (N_tri, 3, 2) then reshape to (N_tri*3, 2)
        triangle_uvs = np.stack([u_coords, v_coords], axis=-1).reshape(-1, 2)

        logger.debug(
            "Generated %d patch-projected UVs (%d triangles, %d patches, "
            "%d flat patches, %d vertical patches)",
            len(triangle_uvs),
            len(triangles),
            patch_count,
            flat_patch_count,
            vertical_patch_count,
        )
        return triangle_uvs

    except (RuntimeError, ValueError, IndexError) as e:
        logger.debug(f"Failed to generate patch-projected UVs: {e}")
        return None


def _parse_ply_uvs(file_path: str) -> list[tuple[float, float]] | None:
    """Parse UV coordinates from a PLY file.

    Open3D does not load PLY vertex-property UVs, so this parser reads them
    directly.

    Args:
        file_path: Path to the PLY file

    Returns:
        List of (u, v) tuples per vertex, or None if no UVs found
    """
    try:
        # Check if the PLY file uses ASCII format; binary PLY files
        # cannot be parsed as text and don't support this UV extraction.
        with open(file_path, "rb") as fb:
            for raw_line in fb:
                header_line = raw_line.decode("ascii", errors="replace").strip()
                if header_line.startswith("format"):
                    if "ascii" not in header_line:
                        return None
                    break
                if header_line == "end_header":
                    break

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Parse header
        vertex_count = 0
        face_count = 0
        u_index = -1
        v_index = -1
        prop_index = 0

        header_end = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "end_header":
                header_end = i + 1
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line.startswith("property") and "vertex" not in line:
                parts = line.split()
                if len(parts) >= 3:
                    prop_name = parts[-1]
                    if prop_name in ("u", "s", "texture_u"):
                        u_index = prop_index
                    elif prop_name in ("v", "t", "texture_v"):
                        v_index = prop_index
                    prop_index += 1

        if u_index < 0 or v_index < 0:
            return None

        # Parse vertex data
        vertex_uvs = []
        for i in range(vertex_count):
            line = lines[header_end + i].strip()
            values = line.split()
            u = float(values[u_index])
            v = float(values[v_index])
            vertex_uvs.append((u, v))

        # Parse faces and expand vertex UVs to triangle UVs
        triangle_uvs = []
        for i in range(face_count):
            line = lines[header_end + vertex_count + i].strip()
            values = line.split()
            num_verts = int(values[0])
            if num_verts == 3:
                idx0, idx1, idx2 = int(values[1]), int(values[2]), int(values[3])
                triangle_uvs.append(vertex_uvs[idx0])
                triangle_uvs.append(vertex_uvs[idx1])
                triangle_uvs.append(vertex_uvs[idx2])

        return triangle_uvs if triangle_uvs else None

    except (OSError, ValueError, IndexError) as e:
        logger.debug(f"Failed to parse UVs from {file_path}: {e}")
        return None


class MeshLoader:
    """Renderer-neutral mesh loader for scene XML imports."""

    @staticmethod
    def load_mesh(
        file_path: str, material_color: list[float], auto_generate_uvs: bool = True
    ) -> MeshPayload:  # pragma: no cover
        """
        Load a mesh file and apply material color when no vertex colors exist.

        Args:
            file_path: Path to the mesh file
            material_color: RGB color values [r, g, b]
            auto_generate_uvs: If True, generate UV coordinates using box projection
                              for meshes that don't have them. Useful for texture mapping.

        Returns:
            Renderer-neutral mesh payload.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Mesh file not found: {file_path}")

            mesh = load_mesh_payload(file_path)
            if file_path.lower().endswith(".ply") and mesh.triangle_uvs is None:
                uvs = _parse_ply_uvs(file_path)
                if uvs:
                    mesh = replace(mesh, triangle_uvs=np.asarray(uvs, dtype=np.float64))
                    logger.debug(f"Loaded {len(uvs)} UV coordinates from {file_path}")
                elif auto_generate_uvs:
                    auto_uvs = _generate_box_projection_uvs(mesh, scale=0.5)
                    if auto_uvs is not None:
                        mesh = replace(mesh, triangle_uvs=np.asarray(auto_uvs, dtype=np.float64))
                        logger.debug(
                            f"Auto-generated {len(auto_uvs)} UV coordinates for {file_path}"
                        )
            elif auto_generate_uvs and mesh.triangle_uvs is None:
                auto_uvs = _generate_box_projection_uvs(mesh, scale=0.5)
                if auto_uvs is not None:
                    mesh = replace(mesh, triangle_uvs=np.asarray(auto_uvs, dtype=np.float64))

            # Preserve vertex colors from the PLY if present, otherwise apply
            # the material color. This allows textured ground planes (grass,
            # cobblestone, asphalt) to show their baked vertex colors.
            if mesh.vertex_colors is not None and len(mesh.vertex_colors) > 0:
                logger.debug(f"Preserving {len(mesh.vertex_colors)} vertex colors from {file_path}")
            else:
                mesh = geometry.paint_mesh_payload(mesh, material_color)

            logger.debug(f"Loaded mesh: {file_path}")
            return mesh

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to load mesh {file_path}: {e}")
            raise


class XMLSceneHandler:
    """Handles Mitsuba XML scene file operations."""

    @staticmethod
    def load_xml_scene(file_path: str) -> Any:  # pragma: no cover
        """
        Load and parse a Mitsuba XML scene file.

        Args:
            file_path: Path to the XML scene file

        Returns:
            Parsed XML root element

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be parsed
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"XML scene file not found: {file_path}")

            # Strip malformed comments before ElementTree parses the document.
            content = TextFileHandler.read_text_file(file_path, encoding="utf-8")

            lines = content.split("\n")
            fixed_lines = []
            skip_comment_block = False

            for line in lines:
                stripped = line.strip()

                if stripped.startswith("<!--") and stripped.endswith("-->"):
                    continue
                elif stripped.startswith("<!--"):
                    skip_comment_block = True
                    continue
                elif skip_comment_block and "-->" in stripped:
                    skip_comment_block = False
                    continue
                # Skip lines within comment block
                elif skip_comment_block:
                    continue
                else:
                    # Keep valid XML content
                    fixed_lines.append(line)

            fixed_content = "\n".join(fixed_lines)

            if not fixed_content.strip():
                raise ValueError("No valid XML content found after removing comments")

            # Parse the fixed XML content
            import io

            tree = ET.parse(io.StringIO(fixed_content))
            xml_root = tree.getroot()

            logger.debug(f"Successfully parsed XML scene: {file_path}")
            return xml_root

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to parse XML scene {file_path}: {e}")
            raise

    @staticmethod
    def save_xml_scene(xml_root, file_path: str) -> None:  # pragma: no cover
        """
        Save a Mitsuba XML scene to file.

        Args:
            xml_root: Root element of the XML scene
            file_path: Path to save the XML file
        """
        try:
            logger.debug(f"Saving XML scene to: {file_path}")
            logger.debug(f"XML root has {len(xml_root)} direct children")

            for child in xml_root:
                logger.debug(f"Root element: {child.tag} (id: {child.get('id', 'N/A')})")

                if child.tag == "bsdf":
                    logger.debug(f"  BSDF {child.get('id', 'N/A')}:")
                    for subchild in child:
                        logger.debug(
                            f"    Sub: {subchild.tag} (name: {subchild.get('name', 'N/A')})"
                        )
                        if subchild.tag == "rgb":
                            logger.debug(f"      RGB value: {subchild.get('value', 'N/A')}")
                        elif subchild.tag == "bsdf":
                            logger.debug(f"      Sub-BSDF type: {subchild.get('type', 'N/A')}")
                            for subsubchild in subchild:
                                if subsubchild.tag == "rgb":
                                    logger.debug(
                                        f"        RGB value: {subsubchild.get('value', 'N/A')}"
                                    )

            # Verify that the XML tree contains the expected modifications.
            logger.debug("Verifying XML tree modifications before saving...")

            for child in xml_root:
                if child.tag == "bsdf":
                    bsdf_id = child.get("id", "N/A")
                    logger.debug(f"  Checking BSDF {bsdf_id}...")

                    # Look for rgb elements in this BSDF
                    for subchild in child:
                        if subchild.tag == "rgb" and subchild.get("name") == "reflectance":
                            color_value = subchild.get("value")
                            logger.debug(f"    RGB value: {color_value}")
                        elif subchild.tag == "bsdf" and subchild.get("type") == "diffuse":
                            for subsubchild in subchild:
                                if (
                                    subsubchild.tag == "rgb"
                                    and subsubchild.get("name") == "reflectance"
                                ):
                                    color_value = subsubchild.get("value")
                                    logger.debug(f"    RGB value (in diffuse): {color_value}")

            lines = XMLSceneHandler._assemble_scene_lines(xml_root)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            logger.debug(f"Successfully saved XML scene to: {file_path}")

            # Verify the saved file contains the expected content
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    if "reflectance" in content:
                        logger.debug("Saved XML contains 'reflectance' keyword")
                    else:
                        logger.warning("Saved XML does not contain 'reflectance' keyword")

                    # Count rgb elements
                    rgb_count = content.count("<rgb")
                    logger.debug(f"Saved XML contains {rgb_count} rgb elements")

            except OSError as verify_e:
                logger.warning(f"Could not verify saved XML content: {verify_e}")

        except (OSError, ValueError) as e:
            logger.error(f"Failed to save XML scene to {file_path}: {e}")
            raise

    @staticmethod
    def _assemble_scene_lines(xml_root) -> list[str]:  # pragma: no cover
        """Assemble XML lines in the stable manual serialization order."""
        logger.debug("Using manual XML assembly (old working method)")
        lines: list[str] = []
        lines.append('<?xml version="1.0"?>')
        lines.append('<scene version="2.1.0">')
        lines.append("")
        lines.append("<!-- Defaults, these can be set via the command line: -Darg=value -->")
        lines.append("")
        lines.append("<!-- Camera and Rendering Parameters -->")
        integ = xml_root.find("integrator")
        if integ is not None:
            lines.extend(XMLSceneHandler._elem_to_lines(integ))
        lines.append("")
        lines.append("<!-- Materials -->")
        for bsdf in xml_root.findall("bsdf"):
            lines.extend(XMLSceneHandler._elem_to_lines(bsdf))
        lines.append("")
        lines.append("<!-- Emitters -->")
        emi = xml_root.find("emitter")
        if emi is not None:
            lines.extend(XMLSceneHandler._elem_to_lines(emi))
        lines.append("")
        lines.append("<!-- Shapes -->")
        for shape in xml_root.findall("shape"):
            if shape.get("type") == "ply":
                # DEBUG: Verify shape has transform before serializing
                transform = shape.find("transform[@name='to_world']")
                if transform is not None:
                    translate = transform.find("translate")
                    if translate is not None:
                        logger.debug(
                            f"Shape '{shape.get('id', 'unknown')}' has translate: x={translate.get('x', 'N/A')}, y={translate.get('y', 'N/A')}, z={translate.get('z', 'N/A')}"
                        )
                    else:
                        logger.warning(
                            f"Shape '{shape.get('id', 'unknown')}' has transform but NO translate element!"
                        )
                else:
                    logger.debug(f"Shape '{shape.get('id', 'unknown')}' has no transform element")
                lines.extend(XMLSceneHandler._elem_to_lines(shape))
        lines.append("")
        lines.append("<!-- Volumes -->")
        lines.append("")
        lines.append("</scene>")
        return lines

    @staticmethod
    def serialize_xml_scene(xml_root) -> str:  # pragma: no cover
        """Serialize the current XML tree to a string using the manual assembler."""
        lines = XMLSceneHandler._assemble_scene_lines(xml_root)
        return "\n".join(lines)

    @staticmethod
    def _elem_to_lines(elem, indent="\t"):  # pragma: no cover
        """Convert one XML element to the stable tab-indented line format."""
        if elem is None:
            return []
        try:
            text = ET.tostring(elem, encoding="unicode")
            # simple prettify
            lines = text.strip().split(">")
            out = []
            for line in lines:
                if not line.strip():
                    continue
                out.append(indent + line.strip() + ">")
            return out
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing XML element: {e}")
            return []

    @staticmethod
    def debug_xml_structure(xml_root, label: str = "XML Structure") -> None:  # pragma: no cover
        """
        Debug method to print the XML structure for troubleshooting.

        Args:
            xml_root: Root element of the XML scene
            label: Label for the debug output
        """
        try:
            logger.debug(f"=== {label} ===")
            logger.debug(f"Root tag: {xml_root.tag}")
            logger.debug(f"Root attributes: {xml_root.attrib}")
            logger.debug(f"Number of children: {len(xml_root)}")

            for i, child in enumerate(xml_root):
                logger.debug(f"Child {i}: {child.tag} (id: {child.get('id', 'N/A')})")

                if child.tag == "bsdf":
                    logger.debug(f"  BSDF {child.get('id', 'N/A')}:")
                    for subchild in child:
                        logger.debug(
                            f"    Sub: {subchild.tag} (name: {subchild.get('name', 'N/A')})"
                        )
                        if subchild.tag == "rgb":
                            logger.debug(f"      RGB value: {subchild.get('value', 'N/A')}")

                elif child.tag == "shape":
                    logger.debug(f"  Shape type: {child.get('type', 'N/A')}")
                    for subchild in child:
                        logger.debug(
                            f"    Sub: {subchild.tag} = {subchild.text or subchild.get('value', 'N/A')}"
                        )

            logger.debug(f"=== End {label} ===")

        except (ValueError, AttributeError) as e:
            logger.error(f"Error debugging XML structure: {e}")


def build_scene_from_root(xml_root, xml_path):  # pragma: no cover
    """
    Build mesh_entries from an existing XML root (without re-parsing).
    This preserves one parsed tree for both editing and serialization.

    Args:
        xml_root: Existing XML root element
        xml_path: Path to the XML file (for resolving relative paths)

    Returns:
        List of mesh entries with references to the same XML tree
    """
    try:
        fingerprint_start = time.perf_counter()
        records = _mesh_shape_records(xml_root, xml_path)
        fingerprint = _scene_payload_cache_fingerprint(xml_root, xml_path, records)
        record_cache_event(
            "scene_payload",
            "fingerprint",
            elapsed_ms=(time.perf_counter() - fingerprint_start) * 1000.0,
            count=len(records),
        )
        cache_path = (
            _scene_payload_cache_path(xml_path, fingerprint)
            if fingerprint is not None and records
            else None
        )
        if cache_path is not None and cache_path.is_file():
            cached_entries = _load_scene_payload_cache(cache_path, fingerprint, xml_root)
            if cached_entries is not None:
                return cached_entries
            record_cache_event("scene_payload", "miss")
        elif cache_path is not None:
            record_cache_event("scene_payload", "miss")
            logger.warning(
                "Neutral scene payload cache miss for %s; first load may take longer "
                "while the cache is created. Future loads should be faster.",
                xml_path,
            )

        # Parse materials from the existing root
        materials = MaterialHandler.parse_materials(xml_root)
        mesh_entries = []
        base_dir = os.path.dirname(xml_path)

        for record in records:
            shape = record["shape"]
            rel_path = record["rel_path"]
            full_path = record["full_path"]
            material_id = record["material_id"]

            material_info = materials.get(material_id, {"color": [0.7, 0.7, 0.7]})
            color = material_info["color"]
            material_type = material_info.get("material_type", "default")
            pbr_properties = dict(material_info.get("pbr_properties", {}) or {})
            from ..materials.texture_policy import TEXTURE_MAP_KEYS

            for key in TEXTURE_MAP_KEYS:
                texture_value = pbr_properties.get(key)
                if not texture_value:
                    continue
                texture_path = Path(str(texture_value)).expanduser()
                if not texture_path.is_absolute():
                    texture_path = Path(base_dir) / texture_path
                pbr_properties[key] = str(texture_path)

            # Vegetation material override: shapes whose PLY is ``vegetation.ply``
            # are tree trunks + canopies.  Sionna RT keeps them bound to
            # ``mat-itu_wood`` so the ITU-R P.833 wood attenuation model still
            # drives propagation, but the visualizer renders them with the
            # green ``vegetation`` PBR entry so the scene shows green trees
            # instead of brown ones.
            if rel_path and os.path.basename(rel_path) == "vegetation.ply":
                veg_pbr = material_preset("vegetation")
                material_type = "vegetation"
                color = list(veg_pbr["color"])
                pbr_properties = dict(veg_pbr)

            # UV generation is deferred until material policy selects a texture.
            mesh = MeshLoader.load_mesh(full_path, color, auto_generate_uvs=False)
            name = os.path.splitext(os.path.basename(rel_path))[0]

            vertices = geometry.mesh_vertices(mesh)
            if vertices is None:
                continue
            original_mesh_center = geometry.mesh_center(mesh).copy()
            original_vertices = np.asarray(vertices, dtype=float).copy()

            transform_state = record["transform_state"]

            mesh, position_after_scale_rotation, final_center = geometry.apply_transform_to_payload(
                mesh, original_vertices, original_mesh_center, transform_state
            )

            mesh_entries.append(
                {
                    "name": name,
                    "mesh": mesh,
                    "material_id": material_id,
                    "material_type": material_type,
                    "pbr_properties": pbr_properties,
                    "original_center": original_mesh_center,
                    "position_after_scale_rotation": position_after_scale_rotation,
                    "current_center": final_center,
                    "original_vertices": original_vertices,
                    "transform_state": transform_state,
                    "color": color,
                    "visible": True,
                    "show_label": False,
                    "highlighted": False,
                    "id_edit": None,
                    "entry_type": "mesh",
                    "xml_bsdf": material_info.get("xml_element"),
                    "xml_shape": shape,
                    "rel_path": rel_path,
                    "shape_index": record["shape_index"],
                    "_source_signature": record.get("source_signature"),
                }
            )

        if cache_path is not None:
            _store_scene_payload_cache(cache_path, fingerprint, mesh_entries)
        logger.debug(f"Built {len(mesh_entries)} mesh entries from existing XML root")
        return mesh_entries

    except (OSError, ValueError, RuntimeError):
        logger.exception("Failed to build scene from XML root")
        raise
