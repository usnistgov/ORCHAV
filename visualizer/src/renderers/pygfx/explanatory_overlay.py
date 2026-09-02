"""Shared policy for pygfx path-explanation overlays.

RF X-Ray and selected-MPC inspection both draw transient geometry over the
canonical bulk MPC layer.  This helper keeps those overlays non-pickable so
the bulk line remains the single source of viewport path identity.
"""

from __future__ import annotations

from typing import Any


def configure_explanatory_overlay(
    renderer: Any,
    name: str,
    *,
    native_object: Any = None,
) -> bool:
    """Mark one registered/native overlay as explanatory and non-pickable."""
    obj = native_object
    if obj is None:
        obj = getattr(renderer, "_objects", {}).get(str(name))
    if obj is None:
        return False

    material = getattr(obj, "material", None)
    if material is not None and hasattr(material, "pick_write"):
        try:
            material.pick_write = False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    metadata_by_name = getattr(renderer, "_pick_metadata", None)
    if isinstance(metadata_by_name, dict):
        metadata = dict(metadata_by_name.get(str(name), {}))
        metadata["pickable"] = False
        metadata_by_name[str(name)] = metadata
    return True


__all__ = ["configure_explanatory_overlay"]
