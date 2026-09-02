"""Shared immutable identity and geometry naming helpers.

This module keeps renderer geometry names stable across UI display-name changes.
"""

from __future__ import annotations

import re
from typing import Any

from .object_key_service import make_node_key, make_scene_key, make_target_key

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def normalize_token(value: object) -> str:
    """Normalize a display token into stable key/name text."""
    text = str(value).strip().lower().replace(" ", "_")
    text = _SANITIZE_RE.sub("_", text)
    return text or "unknown"


def make_entity_id(domain: str, entity_id: object) -> str:
    """Return immutable entity id like ``scene:wall_01`` or ``node:tx_0``."""
    return f"{normalize_token(domain)}:{normalize_token(entity_id)}"


def make_component_geometry_name(entity_id: str, component: str) -> str:
    """Return stable geometry name like ``scene:wall_01::mesh``."""
    return f"{entity_id}::{normalize_token(component)}"


def make_node_entity_id(kind: str, index: int) -> str:
    """Return the immutable entity id for one TX/RX node index."""
    return make_entity_id("node", f"{normalize_token(kind)}_{int(index)}")


def make_node_geometry_name(kind: str, index: int, component: str) -> str:
    """Return the stable renderer geometry name for one node component."""
    return make_component_geometry_name(make_node_entity_id(kind, index), component)


def make_scene_entry_geometry_name(entry: dict[str, Any], component: str = "mesh") -> str:
    """Return the stable renderer geometry name for a scene entry component."""
    object_id = entry.get("object_id")
    if not object_id:
        object_id = make_entity_id(
            "scene", entry.get("stable_mesh_id") or entry.get("name") or "mesh"
        )
    return make_component_geometry_name(str(object_id), component)


def make_target_entry_geometry_name(entry: dict[str, Any], component: str = "mesh") -> str:
    """Return the stable renderer geometry name for a target entry component."""
    object_id = entry.get("object_id")
    if not object_id:
        target_id = (
            entry.get("stable_target_id")
            or entry.get("target_name")
            or entry.get("name")
            or "target"
        )
        object_id = make_entity_id("target", target_id)
    return make_component_geometry_name(str(object_id), component)


def ensure_scene_entry_identity(entry: dict[str, Any], index: int | None = None) -> str:
    """Populate immutable identity fields for a scene mesh entry."""
    stable_mesh_id = entry.get("stable_mesh_id")
    if not stable_mesh_id:
        xml_shape = entry.get("xml_shape")
        shape_id = xml_shape.get("id") if hasattr(xml_shape, "get") else None
        if shape_id:
            base = shape_id
        elif entry.get("rel_path"):
            base = entry.get("rel_path")
        elif entry.get("name"):
            base = entry.get("name")
        elif index is not None:
            base = f"mesh_{index}"
        else:
            base = f"mesh_{id(entry.get('mesh'))}"
        stable_mesh_id = normalize_token(base)
        entry["stable_mesh_id"] = stable_mesh_id

    object_id = make_entity_id("scene", stable_mesh_id)
    entry["object_id"] = object_id
    entry["object_key"] = make_scene_key(stable_mesh_id, "mesh")
    entry["geometry_name"] = make_component_geometry_name(object_id, "mesh")
    if "display_name" not in entry:
        entry["display_name"] = entry.get("name", str(stable_mesh_id))
    return object_id


def ensure_target_entry_identity(entry: dict[str, Any], index: int | None = None) -> str:
    """Populate immutable identity fields for a target entry."""
    target_name = entry.get("target_name") or entry.get("node_name") or entry.get("name")
    if not target_name:
        target_name = f"target_{index if index is not None else 0}"
    if not entry.get("target_name"):
        entry["target_name"] = str(target_name)

    stable_target_id = entry.get("stable_target_id") or normalize_token(target_name)
    entry["stable_target_id"] = stable_target_id

    object_id = make_entity_id("target", stable_target_id)
    entry["object_id"] = object_id
    entry["object_key"] = make_target_key(stable_target_id, "mesh")
    entry["geometry_name"] = make_component_geometry_name(object_id, "mesh")
    if "display_name" not in entry:
        entry["display_name"] = entry.get("name", str(target_name))
    return object_id


def ensure_node_entry_identity(entry: dict[str, Any], kind: str, index: int) -> str:
    """Populate immutable identity fields for a TX/RX entry."""
    object_id = make_node_entity_id(kind, index)
    entry["object_id"] = object_id
    entry["object_key"] = make_node_key(kind, index, "marker")
    entry["marker_geometry_name"] = make_component_geometry_name(object_id, "marker")
    entry["label_geometry_name"] = make_component_geometry_name(object_id, "label")
    if "display_name" not in entry:
        entry["display_name"] = entry.get("name", f"{kind.upper()}{index + 1}")
    return object_id
