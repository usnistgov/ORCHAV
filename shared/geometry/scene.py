"""Shared scene-geometry loader for Mitsuba XML files.

This module provides a lightweight, dependency-tolerant loader that extracts
mesh geometry for computing bounds and simple visual summaries. It reads shape
references, world transforms, and basic material color hints from XML and
returns neutral dictionaries so callers can use geometry without importing
generator or visualizer services.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.logging import get_logger

from .transforms import parse_lightweight_shape_transform

logger = get_logger("shared.geometry.scene")


def _parse_rgb(value: Optional[str]) -> Optional[list[float]]:
    if not value:
        return None
    try:
        rgb = [float(v) for v in value.strip().split()]
    except (TypeError, ValueError):
        return None
    if len(rgb) != 3:
        return None
    return [max(0.0, min(1.0, value)) for value in rgb]


def _parse_bsdf_colors(xml_root: ET.Element) -> dict[str, list[float]]:
    """Extract simple reflectance colors keyed by Mitsuba BSDF id."""
    colors: dict[str, list[float]] = {}
    for bsdf in xml_root.findall("bsdf"):
        bsdf_id = bsdf.get("id")
        if not bsdf_id:
            continue
        rgb_elem = bsdf.find("rgb[@name='reflectance']")
        if rgb_elem is None:
            rgb_elem = bsdf.find("rgb")
        color = _parse_rgb(rgb_elem.get("value") if rgb_elem is not None else None)
        if color is not None:
            colors[bsdf_id] = color
    return colors


def _build_rotation_matrix_from_angles(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    """Build a 3x3 rotation matrix using Mitsuba/Sionna order (Z -> Y -> X)."""
    yaw, pitch, roll = map(np.radians, rotation_deg)
    rotation_matrix = np.eye(3)

    if abs(roll) > 1e-6:
        rot_x = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )
        rotation_matrix = rot_x @ rotation_matrix

    if abs(pitch) > 1e-6:
        rot_y = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        rotation_matrix = rot_y @ rotation_matrix

    if abs(yaw) > 1e-6:
        rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = rot_z @ rotation_matrix

    return rotation_matrix


def _build_transform_matrix(transform_state: dict[str, Any]) -> np.ndarray:
    """Construct a 4x4 transform matrix (scale -> rotate -> translate)."""
    scale = float(transform_state.get("scale", 1.0))
    rotation = tuple(transform_state.get("rotation", (0.0, 0.0, 0.0)))
    translate_vec = np.asarray(transform_state.get("translate", (0.0, 0.0, 0.0)), dtype=float)

    matrix = np.eye(4)
    scale_matrix = np.diag([scale, scale, scale, 1.0])
    matrix = scale_matrix @ matrix

    rotation_matrix = _build_rotation_matrix_from_angles(rotation)
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix
    matrix = rotation_matrix_4x4 @ matrix

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translate_vec
    matrix = translation_matrix @ matrix
    return matrix


def load_scene_geometry(xml_path: str) -> list[dict[str, Any]]:
    """Load mesh geometry from a Mitsuba XML file.

    Returns a list of mesh entries with at least:
    - name: mesh basename
    - mesh: Open3D mesh instance
    - rel_path: path as specified in the XML

    Optional fields such as ``material_id`` and ``color`` preserve enough scene
    context for bounds, summaries, and lightweight previews.

    Missing Open3D or an unreadable XML document returns an empty list;
    unreadable or empty individual meshes are skipped. Mesh transforms must fit
    the exact lightweight subset documented by ``shared.geometry.transforms``.
    Unsupported forms raise instead of producing approximate placement.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not available; cannot load scene geometry for %s", xml_path)
        return []

    xml_path = str(xml_path)
    base_dir = str(Path(xml_path).parent)

    try:
        xml_root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        logger.error("Failed to parse scene XML %s: %s", xml_path, exc)
        return []
    bsdf_colors = _parse_bsdf_colors(xml_root)

    mesh_entries: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(xml_root.findall("shape")):
        shape_type = shape.get("type")
        if shape_type not in ("ply", "obj"):
            continue

        fn_elem = shape.find("string[@name='filename']")
        if fn_elem is None:
            continue

        rel_path = fn_elem.get("value", "")
        if not rel_path:
            continue

        transform_state = parse_lightweight_shape_transform(
            shape,
            source_xml=xml_path,
            shape_index=shape_index,
        )
        full_path = Path(rel_path) if Path(rel_path).is_absolute() else Path(base_dir) / rel_path
        try:
            mesh = o3d.io.read_triangle_mesh(str(full_path))
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to load mesh %s: %s", full_path, exc)
            continue

        if mesh is None or mesh.is_empty():
            logger.warning("Loaded empty mesh from %s", full_path)
            continue

        try:
            mesh.compute_vertex_normals()
        except (RuntimeError, ValueError):
            pass

        try:
            mesh.transform(_build_transform_matrix(transform_state))
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Failed to apply transform to mesh %s: %s", full_path, exc)

        name = Path(rel_path).stem
        bsdf_ref = shape.find("ref[@name='bsdf']")
        material_id = bsdf_ref.get("id") if bsdf_ref is not None else None
        entry: dict[str, Any] = {
            "name": name,
            "mesh": mesh,
            "rel_path": rel_path,
            "full_path": str(full_path),
            "source_xml": str(Path(xml_path)),
            "shape_id": shape.get("id"),
            "material_id": material_id,
        }
        if material_id in bsdf_colors:
            entry["color"] = bsdf_colors[material_id]
        mesh_entries.append(entry)

    logger.debug("Loaded %d mesh entries from %s", len(mesh_entries), xml_path)
    return mesh_entries
