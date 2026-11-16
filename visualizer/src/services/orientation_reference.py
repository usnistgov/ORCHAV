"""Read the displayed actor inventory and pose for orientation inspection.

Use frame mirrors for TX/RX and stable target identity for targets. Inspection
never changes Context selections, camera tracking, recorded poses or filters.
"""

import numpy as np

from ..scene.geometry_helpers import resolve_node_label
from ..scene.target_transforms import target_entry_anchor_position
from .object_identity import normalize_token


def target_reference_key(entry: dict) -> str:
    """Use the same stable identity as target renderer objects."""
    identity = entry.get("stable_target_id") or normalize_token(
        entry.get("target_name") or entry.get("name") or ""
    )
    return f"target:{identity}"


def orientation_actor_choices(viz) -> list[tuple[str, str]]:
    """List individual actors without adding an ambiguous 'all' selection."""
    state = getattr(viz, "app_state", None)
    choices = []
    for kind in ("tx", "rx"):
        positions = getattr(viz, f"current_{kind}_positions", ())
        for index in range(len(positions) if positions is not None else 0):
            label = resolve_node_label(
                kind.upper(),
                index,
                getattr(state, f"{kind}_labels", ()),
                label_mode=getattr(state, "node_label_mode", "role"),
                device_names=getattr(state, f"{kind}_device_names", ()),
            )
            role = f"{kind.upper()}{index + 1}"
            choices.append((role if label == role else f"{role} · {label}", f"{kind}:{index}"))
    custom = getattr(state, "target_labels", ())
    for index, entry in enumerate(getattr(viz, "target_entries", ())):
        name = (
            custom[index]
            if index < len(custom) and custom[index]
            else entry.get("name", f"Target {index + 1}")
        )
        choices.append((f"Target {index + 1} · {name}", target_reference_key(entry)))
    return choices


def _vector(value) -> np.ndarray | None:
    """Accept only finite three-component positions or Euler angles."""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return vector if vector.shape == (3,) and np.isfinite(vector).all() else None


def orientation_actor_pose(viz, actor: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve the current world anchor and Sionna yaw/pitch/roll in radians."""
    kind, _, key = actor.partition(":")
    if kind in ("tx", "rx") and key.isdecimal():
        index = int(key)
        positions = getattr(viz, f"current_{kind}_positions", ())
        orientations = getattr(viz, f"current_{kind}_orientations", ())
        position = positions[index] if positions is not None and index < len(positions) else None
        orientation = (
            orientations[index] if orientations is not None and index < len(orientations) else None
        )
        return _vector(position), _vector(orientation)
    if kind == "target":
        for entry in getattr(viz, "target_entries", ()):
            if target_reference_key(entry) != actor:
                continue
            if entry.get("_frame_visible", True) is False:
                return None, None
            position = target_entry_anchor_position(entry)
            orientation = entry.get("orientation_radians", entry.get("orientation"))
            return _vector(position), _vector(orientation)
    return None, None
