"""Stable object-key helpers for visual state/session restore."""

from __future__ import annotations

import re
from typing import Optional, TypeAlias

ObjectKey: TypeAlias = str

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _norm(value: object) -> str:
    """Normalize one object-key part for session-safe keys."""
    text = str(value).strip().lower().replace(" ", "_")
    text = _SANITIZE_RE.sub("_", text)
    return text or "unknown"


def make_object_key(
    domain: str,
    entity_id: object,
    component: str = "mesh",
    sub_component: Optional[str] = None,
) -> ObjectKey:
    """Build a deterministic object key."""
    parts = [_norm(domain), _norm(entity_id), _norm(component)]
    if sub_component:
        parts.append(_norm(sub_component))
    return ":".join(parts)


def make_node_key(kind: str, index: int, component: str) -> ObjectKey:
    """Node key helper (e.g. ``node:tx_0:marker``)."""
    return make_object_key("node", f"{kind}_{index}", component)


def make_target_key(target_name: str, component: str) -> ObjectKey:
    """Target key helper."""
    return make_object_key("target", target_name, component)


def make_scene_key(mesh_name: str, component: str = "mesh") -> ObjectKey:
    """Scene mesh key helper."""
    return make_object_key("scene", mesh_name, component)


def make_beamforming_key(mesh_id: object, component: str = "mesh") -> ObjectKey:
    """Beamforming key helper."""
    return make_object_key("beamforming", mesh_id, component)


def make_ground_key(name: str = "custom", component: str = "grid") -> ObjectKey:
    """Ground grid key helper."""
    return make_object_key("ground", name, component)
