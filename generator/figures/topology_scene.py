"""Exact XML meshes and display colors for geometry-only topology figures.

Adapted from the reviewed October 2026 topology gallery. Material fallback
matches Sionna RT's ITU display palette; it does not change radio materials.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

from .mesh_geometry import _read_mesh, crease_segments

# Sionna RT 2.2.0 ITURadioMaterial.ITU_MATERIAL_COLORS; display only.
# Native defaults, not invented per-building colors or RF parameters.
ITU_MATERIAL_COLORS = {
    "marble": (0.701, 0.644, 0.485),
    "concrete": (0.539, 0.539, 0.539),
    "wood": (0.266, 0.109, 0.06),
    "metal": (0.22, 0.22, 0.254),
    "brick": (0.402, 0.112, 0.087),
    "glass": (0.168, 0.139, 0.509),
    "floorboard": (0.539, 0.386, 0.025),
    "ceiling_board": (0.376, 0.539, 0.117),
    "chipboard": (0.509, 0.159, 0.323),
    "plasterboard": (0.051, 0.539, 0.133),
    "plywood": (0.136, 0.076, 0.539),
    "clear_acrylic": (0.8, 0.9, 0.95),
    "vinyl_tile": (0.75, 0.75, 0.72),
    "carpet_tile": (0.32, 0.35, 0.42),
    "asphalt_concrete": (0.18, 0.18, 0.18),
    "very_dry_ground": (0.539, 0.319, 0.223),
    "medium_dry_ground": (0.539, 0.181, 0.076),
    "wet_ground": (0.539, 0.027, 0.147),
}


def material_rgb(bsdf):
    """Resolve native explicit-color precedence, then native ITU defaults."""
    for property_name in ("color", "reflectance", "base_color"):
        element = bsdf.find(f"rgb[@name='{property_name}']")
        if element is not None:
            values = [float(x) for x in element.get("value").replace(",", " ").split()]
            if len(values) == 1:
                values *= 3
            if len(values) != 3:
                raise ValueError("Material RGB requires one or three components")
            return np.asarray(values)
    if bsdf.get("type") == "itu-radio-material":
        kind = bsdf.find("string[@name='type']")
        if kind is not None and kind.get("value") in ITU_MATERIAL_COLORS:
            return np.asarray(ITU_MATERIAL_COLORS[kind.get("value")])
    return None


def is_etoile_ground(path, shape, asset, vertices):
    """Only the verified native Étoile Plane is a ground-name exception."""
    return (
        Path(path).name == "etoile.xml"
        and shape.get("id") == "mesh-Plane"
        and asset.name == "Plane.ply"
        and np.ptp(vertices[:, 2]) < 1e-7
    )


def transform_matrix(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    for child in element:
        op = np.eye(4)
        if child.tag in ("translate", "scale"):
            default = 0 if child.tag == "translate" else 1
            values = [float(x) for x in child.get("value", "").replace(",", " ").split()]
            if len(values) == 1:
                values *= 3
            if not values:
                values = [float(child.get(axis, default)) for axis in "xyz"]
            if len(values) != 3:
                raise ValueError("Unsupported transform vector")
            if child.tag == "translate":
                op[:3, 3] = values
            else:
                op[:3, :3] = np.diag(values)
        elif child.tag == "matrix":
            op = np.asarray([float(x) for x in child.get("value").split()]).reshape(4, 4)
        elif child.tag == "rotate":
            axis = np.array([float(child.get(a, 0)) for a in "xyz"])
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = np.deg2rad(float(child.get("angle")))
            op[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
        else:
            raise ValueError(f"Unsupported authored shape transform: {child.tag}")
        matrix = op @ matrix
    return matrix


def scene_file(path):
    path = Path(path)
    tree = ET.parse(path).getroot()
    materials, material_names = {}, {}
    for bsdf in tree.findall("bsdf"):
        kind = bsdf.find("string[@name='type']")
        native_name = kind.get("value") if kind is not None else bsdf.get("id", "Unassigned")
        material_names[bsdf.get("id")] = native_name.replace("_", " ").title()
        rgb = material_rgb(bsdf)
        if rgb is not None:
            materials[bsdf.get("id")] = np.clip(0.32 + 0.68 * rgb, 0, 1)
    result, inputs = [], [path]
    for shape in tree.findall("shape"):
        kind = shape.get("type")
        filename = shape.find("string[@name='filename']")
        if kind not in ("ply", "obj") or filename is None:
            raise ValueError(f"Unadapted authored scene shape {kind}; geometry was not omitted")
        asset = (path.parent / filename.get("value")).resolve()
        vertices, faces = _read_mesh(asset)
        matrix = transform_matrix(shape.find("transform[@name='to_world']"))
        vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        name = shape.get("id", asset.stem)
        ref = shape.find("ref[@name='bsdf']")
        if ref is None:
            ref = shape.find("ref")
        material = ref.get("id", "") if ref is not None else ""
        lower = (name + " " + material).lower()
        etoile_ground = is_etoile_ground(path, shape, asset, vertices)
        color = materials.get(material, to_rgb("#d7cfbc"))
        material_label = material_names.get(material, material or "Unassigned")
        if etoile_ground or any(s in lower for s in ("ground", "floor", "bottom", "terrain")):
            color = to_rgb("#e9e7df")
            material_label = f"Ground ({material_label.lower()})"
        elif any(s in lower for s in ("street", "asphalt", "road")):
            color = to_rgb("#9fa9ad")
            material_label = f"Road ({material_label.lower()})"
        elif "glass" in lower:
            color = to_rgb("#adc6cd")
        flat = np.ptp(vertices[:, 2]) < 1e-7
        background = flat and (
            etoile_ground
            or any(s in lower for s in ("ground", "floor", "bottom", "street", "terrain"))
        )
        edges = crease_segments(vertices, faces) if not background else np.empty((0, 2, 3))
        result.append(
            {
                "name": name,
                "vertices": vertices,
                "faces": faces,
                "color": color,
                "material_id": material,
                "material_label": material_label,
                "background": background,
                "edges": edges,
            }
        )
        inputs.append(asset)
    return result, inputs
