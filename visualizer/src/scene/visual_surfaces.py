"""Load authored visual surfaces without changing a scene's radio geometry.

``visual_surfaces.json`` sits beside the scene XML. Its version-one records bind
explicit source triangle selections and per-corner UVs to local texture assets.
The source-file fingerprint makes selections fail closed after mesh changes.
Apply this after restoring neutral scene payloads and geographic classifications:
the disk cache and editable XML must continue to describe the original scene.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from shared.logging import get_logger

from ..materials.scene_variation import valid_parts
from ..materials.texture_policy import TEXTURE_MAP_KEYS
from ..services.object_identity import ensure_scene_entry_identity
from ..types.render_payloads import MeshPayload, SurfaceColorSource

logger = get_logger("orchav.visual_surfaces")
_SCALARS = frozenset({"roughness", "metallic", "reflectance", "alpha", "clearcoat", "transmission"})


def _material(record: Any, directory: Path) -> dict[str, Any]:
    """Validate the bounded optical-material contract and resolve local maps."""
    if not isinstance(record, dict) or not record.get("texture_path"):
        raise ValueError("each visual surface requires a material texture_path")
    unknown = set(record) - set(TEXTURE_MAP_KEYS) - _SCALARS - {"color"}
    if unknown:
        raise ValueError(f"unknown visual material fields: {sorted(unknown)}")
    result: dict[str, Any] = {key: None for key in TEXTURE_MAP_KEYS}
    for key in TEXTURE_MAP_KEYS:
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            raise ValueError(f"{key} must be a path relative to the visual sidecar")
        path = (directory / value).resolve()
        if not path.is_file():
            raise ValueError(f"visual texture is unavailable: {path}")
        result[key] = str(path)
    for key in _SCALARS:
        if key in record:
            value = record[key]
            if type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and between zero and one")
            result[key] = float(value)
    color = record.get("color", [1.0, 1.0, 1.0])
    if (
        not isinstance(color, list)
        or len(color) != 3
        or any(type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1 for v in color)
    ):
        raise ValueError("visual material color must contain three finite values in [0, 1]")
    result["color"] = list(color)
    result["_explicit_props"] = sorted(record)
    return result


def _compact_mesh(
    source: MeshPayload, faces: np.ndarray, uvs: np.ndarray | None, key: str, *, authored: bool
) -> tuple[MeshPayload, np.ndarray]:
    """Preserve triangle corners and winding while giving each piece local bounds."""
    triangles = source.triangles[faces]
    vertex_ids, inverse = np.unique(triangles, return_inverse=True)
    normals = source.normals
    colors = source.vertex_colors
    if normals is not None and len(normals) != len(source.vertices):
        raise ValueError("visual surfaces require per-vertex normals")
    if colors is not None and len(colors) != len(source.vertices):
        raise ValueError("visual surfaces require per-vertex colors")
    return (
        MeshPayload(
            vertices=source.vertices[vertex_ids],
            triangles=inverse.reshape(-1, 3).astype(source.triangles.dtype),
            normals=normals[vertex_ids] if normals is not None else None,
            vertex_colors=colors[vertex_ids] if colors is not None and not authored else None,
            triangle_uvs=uvs,
            color_source=SurfaceColorSource.MATERIAL if authored else source.color_source,
            cache_key=key,
        ),
        vertex_ids,
    )


def _payload_fingerprint(mesh: MeshPayload) -> str:
    """Identify transformed source buffers, including repeated XML instances.

    A sidecar selects source-file faces, but the loader can instantiate that
    file under different transforms. Neutral/notebook renderers also consume
    the payload cache key, before SceneService assigns its desktop cache key.
    """
    digest = hashlib.sha256()
    for name in ("vertices", "triangles", "normals", "vertex_colors", "triangle_uvs"):
        value = getattr(mesh, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    digest.update(str(mesh.color_source).encode())
    return digest.hexdigest()


def _remaining_classification(
    entry: dict[str, Any], source: dict[str, Any], keep: np.ndarray
) -> None:
    """Reindex geographic facade parts after removing authored surface faces."""
    count = source.get("_osm_building_faces")
    if type(count) is not int or not 0 < count <= len(keep):
        return
    new_count = int(keep[:count].sum())
    if not new_count:
        return
    entry["_osm_building_faces"] = new_count
    parts = source.get("_osm_building_parts")
    if not valid_parts(parts, count):
        return
    excluded = {tuple(item) for item in source.get("_osm_facade_exclusions", ())}
    offset = 0
    new_parts, new_exclusions = [], []
    for first, size in parts:
        retained = int(keep[first : first + size].sum())
        if retained:
            new_parts.append([offset, retained])
            if (first, size) in excluded:
                new_exclusions.append((offset, retained))
            offset += retained
    entry["_osm_building_parts"] = new_parts
    entry["_osm_facade_exclusions"] = tuple(new_exclusions)


def _partition_entry(entry: dict[str, Any], record: Any, directory: Path) -> list[dict[str, Any]]:
    """Validate all selections before returning any replacement for one source."""
    mesh = entry.get("mesh")
    if not isinstance(mesh, MeshPayload) or not isinstance(record, dict):
        raise ValueError("visual surfaces require a neutral source mesh and mapping record")
    source_path = directory / str(entry["rel_path"])
    expected = record.get("source_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("visual surfaces require a source_sha256 fingerprint")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
        raise ValueError("visual surface source mesh fingerprint changed")
    surfaces = record.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("surfaces must be a nonempty list")
    if entry.get("_osm_road_segments"):
        raise ValueError("visual surface partitioning of classified road segments is unsupported")

    # A mapping edit must invalidate GPU buffers even when the source PLY has
    # not changed. Texture pixel changes follow the existing texture cache.
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, allow_nan=False).encode()
        + _payload_fingerprint(mesh).encode()
    ).hexdigest()
    keep = np.ones(len(mesh.triangles), dtype=bool)
    ids = {"remainder"}
    selected = []
    for surface in surfaces:
        if not isinstance(surface, dict):
            raise ValueError("surface records must be objects")
        identity = surface.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", identity):
            raise ValueError("surface id must contain lowercase letters, digits, '_' or '-'")
        if identity in ids:
            raise ValueError("surface ids must be unique and cannot be 'remainder'")
        ids.add(identity)
        name = surface.get("name", identity)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("surface name must be nonempty text")
        indices = surface.get("triangle_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(v) is not int or not 0 <= v < len(keep) for v in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("triangle_indices must be distinct source triangle indices")
        faces = np.asarray(indices, dtype=np.int64)
        if not np.all(keep[faces]):
            raise ValueError("visual surface triangle selections overlap")
        keep[faces] = False
        uvs = np.asarray(surface.get("triangle_uvs"), dtype=np.float32)
        if uvs.shape != (3 * len(faces), 2) or not np.isfinite(uvs).all():
            raise ValueError("triangle_uvs must provide three finite UV pairs per selected face")
        selected.append((identity, name, faces, uvs, _material(surface.get("material"), directory)))

    source = dict(entry)
    ensure_scene_entry_identity(source)
    base_id = source["stable_mesh_id"]
    retained_faces = np.flatnonzero(keep)
    source_uvs = mesh.triangle_uvs
    if source_uvs is not None and source_uvs.shape != (len(mesh.triangles) * 3, 2):
        raise ValueError("source mesh has invalid per-corner UVs")
    if len(retained_faces):
        remaining_uvs = (
            source_uvs.reshape(-1, 3, 2)[retained_faces].reshape(-1, 2)
            if source_uvs is not None
            else None
        )
        selected.insert(
            0, ("remainder", entry.get("name", "mesh"), retained_faces, remaining_uvs, None)
        )

    result = []
    for identity, name, faces, uvs, material in selected:
        key = f"visual-surfaces-v1/{digest}/{identity}"
        payload, vertex_ids = _compact_mesh(mesh, faces, uvs, key, authored=material is not None)
        child = {k: v for k, v in entry.items() if not k.startswith("_osm_")}
        child.update(
            name=name,
            display_name=name,
            mesh=payload,
            stable_mesh_id=f"{base_id}_visual_{identity}",
            current_center=np.mean(payload.vertices, axis=0),
            _visual_surface_partition=True,
            _visual_surface_source_name=entry.get("name", "mesh"),
            _visual_surface_id=identity,
            _visual_surface_key=key,
        )
        if "_geographic_face_layers" in entry:
            child["_geographic_face_layers"] = entry["_geographic_face_layers"][faces]
        original_vertices = entry.get("original_vertices")
        if original_vertices is not None:
            child["original_vertices"] = np.asarray(original_vertices)[vertex_ids].copy()
        ensure_scene_entry_identity(child)
        if material is not None:
            # A display toggle can hide this authored map while retaining
            # the original material assignment and unchanged radio geometry.
            child["_visual_surface_base_properties"] = dict(entry.get("pbr_properties") or {})
            child["_visual_surface_base_color"] = entry.get("color")
            child["pbr_properties"] = {**dict(entry.get("pbr_properties") or {}), **material}
            child["color"] = material["color"]
            # XML still refers to the unchanged radio material; mark authored
            # maps explicitly so procedural styling cannot replace them.
            child["texture_path"] = material["texture_path"]
        else:
            _remaining_classification(child, source, keep)
        result.append(child)
    return result


def apply_visual_surfaces(entries: list[dict[str, Any]], xml_path: str) -> list[dict[str, Any]]:
    """Apply optional sidecar mappings to fresh entries, atomically per source mesh.

    Invalid metadata retains the original entry and emits a diagnostic. The
    function never writes source files, mutates the input entries or changes XML
    references. Derived entries are explicitly marked for source-edit guards.
    """
    path = Path(xml_path).with_name("visual_surfaces.json")
    if not path.is_file():
        return entries
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError("visual surface schema_version must be 1")
        records = document.get("meshes")
        if not isinstance(records, dict):
            raise ValueError("visual surface meshes must be an object")
    except (OSError, ValueError) as exc:
        logger.warning("Keeping original scene appearance: %s: %s", path, exc)
        return entries
    result = []
    for entry in entries:
        rel_path = entry.get("rel_path")
        if rel_path not in records:
            result.append(entry)
            continue
        try:
            result.extend(_partition_entry(entry, records[rel_path], path.parent))
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Keeping original appearance for %s: %s", rel_path, exc)
            result.append(entry)
    return result
