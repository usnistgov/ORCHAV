"""Classify and partition geographic meshes for display, leaving RF sources intact.

New OSM exports bind triangle ranges to source-file fingerprints. Older exports
can identify building prefixes and a few dedicated meshes; ambiguous geometry
stays unclassified and visible. Classification precedes authored facade mapping,
then partitions follow it so texture UVs and facade exclusions survive both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from shared.logging import get_logger

from ..services.object_identity import ensure_scene_entry_identity
from ..types.render_payloads import MeshPayload
from .visual_surfaces import _compact_mesh, _payload_fingerprint, _remaining_classification

logger = get_logger("orchav.geographic_layers")
LAYER_LABELS = {
    "buildings": "Buildings",
    "roads": "Roads",
    "ground": "Ground",
    "vegetation": "Vegetation",
    "water": "Water",
    "barriers": "Barriers / walls / fences",
}
_CODES = {name: i + 1 for i, name in enumerate(LAYER_LABELS)}
_NAMES = {value: key for key, value in _CODES.items()}


def _explicit_layers(record: dict, mesh: MeshPayload, path: Path) -> np.ndarray:
    if record.get("total_triangles") != len(mesh.triangles):
        raise ValueError("geographic triangle count changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("source_sha256"):
        raise ValueError("geographic source fingerprint changed")
    codes = np.zeros(len(mesh.triangles), dtype=np.uint8)
    for layer, ranges in record["layers"].items():
        if layer not in _CODES or not isinstance(ranges, list):
            raise ValueError("unknown geographic layer or invalid ranges")
        for first, size in ranges:
            if (
                type(first) is not int
                or type(size) is not int
                or first < 0
                or size <= 0
                or first + size > len(codes)
                or codes[first : first + size].any()
            ):
                raise ValueError("invalid or overlapping geographic triangle ranges")
            codes[first : first + size] = _CODES[layer]
    return codes


def _legacy_layers(entry: dict, metadata: dict) -> np.ndarray:
    """Recover only categories supported by the old OSM export contract."""
    codes = np.zeros(len(entry["mesh"].triangles), dtype=np.uint8)
    rel = entry.get("rel_path")
    if rel not in metadata.get("artifacts", {}).get("mesh_files", []):
        return codes
    name = Path(rel).name
    dedicated = {"ground.ply": "ground", "water.ply": "water", "vegetation.ply": "vegetation"}
    if name in dedicated:
        codes[:] = _CODES[dedicated[name]]
    elif name == "streets.ply" or name.startswith("streets_"):
        codes[:] = _CODES["roads"]
    else:
        record = metadata.get("visual_building_faces", {}).get(rel, {})
        count, total = record.get("building_triangles"), record.get("total_triangles")
        if type(count) is not int or total != len(codes) or not 0 <= count <= total:
            return codes
        codes[:count] = _CODES["buildings"]
        config = metadata.get("osm_config", {})
        # Only these RF groups received barriers. Wood can also contain trees
        # in old exports, so never assume its tail is all hedge geometry.
        barrier_group = name in {"buildings_concrete.ply", "buildings_metal.ply"} or (
            name == "buildings_wood.ply"
            and (config.get("green_vegetation") is True or config.get("vegetation") is False)
        )
        if config.get("barriers") is True and barrier_group:
            codes[count:] = _CODES["barriers"]
    return codes


def annotate_geographic_layers(entries: list[dict], xml_path: str) -> None:
    """Attach transient per-face classifications to freshly loaded neutral entries."""
    path = Path(xml_path).with_name("osm_metadata.json")
    if not path.is_file():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        records = metadata.get("visual_geographic_layers")
        if records is not None and not isinstance(records, dict):
            raise ValueError("geographic layer metadata must be a mapping")
        for entry in entries:
            mesh = entry.get("mesh")
            if not isinstance(mesh, MeshPayload):
                continue
            try:
                if records is None:
                    codes = _legacy_layers(entry, metadata)
                elif entry.get("rel_path") in records:
                    codes = _explicit_layers(
                        records[entry["rel_path"]], mesh, path.parent / entry["rel_path"]
                    )
                else:
                    continue
                if codes.any():
                    entry["_geographic_face_layers"] = codes
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                logger.warning(
                    "Keeping unclassified geometry for %s: %s", entry.get("rel_path"), exc
                )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Geographic layers unavailable: %s", exc)


def partition_geographic_layers(entries: list[dict]) -> list[dict]:
    """Split mixed display meshes by category without modifying their XML or files."""
    result = []
    for entry in entries:
        codes = entry.pop("_geographic_face_layers", None)
        if codes is None:
            result.append(entry)
            continue
        categories = np.unique(codes)
        if len(categories) == 1:
            if categories[0]:
                entry["geographic_layer"] = _NAMES[int(categories[0])]
            result.append(entry)
            continue
        ensure_scene_entry_identity(entry)
        mesh = entry["mesh"]
        fingerprint = _payload_fingerprint(mesh)
        for code in categories:
            layer = _NAMES.get(int(code), "unclassified")
            keep = codes == code
            faces = np.flatnonzero(keep)
            uvs = mesh.triangle_uvs
            if uvs is not None:
                uvs = uvs.reshape(-1, 3, 2)[faces].reshape(-1, 2)
            selection = hashlib.sha256(faces.tobytes()).hexdigest()
            key = f"geographic-v1/{fingerprint}/{layer}/{selection}"
            payload, vertex_ids = _compact_mesh(mesh, faces, uvs, key, authored=False)
            child = {k: v for k, v in entry.items() if not k.startswith("_osm_")}
            child.update(
                name=f"{entry.get('name', 'mesh')} ({layer})",
                mesh=payload,
                stable_mesh_id=f"{entry['stable_mesh_id']}_geo_{layer}",
                current_center=np.mean(payload.vertices, axis=0),
                _geographic_layer_partition=True,
                _geographic_source_name=entry.get("name", "mesh"),
                _geographic_source_mesh_id=entry["stable_mesh_id"],
                _geographic_layer_key=key,
                display_name=f"{entry.get('display_name', entry.get('name', 'mesh'))} ({layer})",
            )
            if code:
                child["geographic_layer"] = layer
            if entry.get("original_vertices") is not None:
                child["original_vertices"] = np.asarray(entry["original_vertices"])[
                    vertex_ids
                ].copy()
            _remaining_classification(child, entry, keep)
            ensure_scene_entry_identity(child)
            result.append(child)
    return result
