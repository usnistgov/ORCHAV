"""Scene-level label helpers above renderer-specific text implementations.

These helpers resolve user/session/scenario label policy into text intent and
delegate actual label object creation to the active renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from shared.logging import get_logger

from ..model import RenderObjectState, make_text_label_state
from ..services.object_identity import (
    ensure_scene_entry_identity,
    ensure_target_entry_identity,
    make_scene_entry_geometry_name,
    make_target_entry_geometry_name,
)
from ..types.render_payloads import MeshPayload
from ..utils.geometry import mesh_center
from .target_transforms import target_entry_anchor_position

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.geometry_helpers")

NodeLabelMode = Literal["role", "name"]


def ensure_scene_mesh_render_state(
    entry: dict[str, Any],
    index: int | None = None,
) -> RenderObjectState:
    """Return the canonical persistent render state for one scene mesh entry.

    Scene I/O deliberately caches neutral :class:`MeshPayload` objects.  The
    desktop visualizer adds application ownership here, after stable scene
    identity is known.  Wrapping reuses the exact cached payload object; it
    does not duplicate vertex, triangle, UV, or color buffers.

    Once wrapped, a persistent scene entry must keep the same stable renderer
    ID.  Treating an ID or payload-type mismatch as an invariant violation is
    preferable to creating a second renderer object through a named fallback.
    """
    ensure_scene_entry_identity(entry, index)
    expected_id = make_scene_entry_geometry_name(entry, "mesh")
    mesh = entry.get("mesh")

    if isinstance(mesh, RenderObjectState):
        if mesh.id != expected_id:
            raise ValueError(
                f"Scene mesh render ID mismatch: expected {expected_id!r}, got {mesh.id!r}"
            )
        if not isinstance(mesh.payload, MeshPayload):
            raise TypeError(
                "Persistent scene mesh state must contain a MeshPayload, "
                f"got {type(mesh.payload).__name__}"
            )
        mesh.visible = bool(entry.get("visible", mesh.visible))
        mesh.metadata.setdefault("type", "scene_mesh")
        entry["entry_type"] = "mesh"
        entry["geometry_name"] = expected_id
        return mesh

    if not isinstance(mesh, MeshPayload):
        raise TypeError(
            "Persistent scene entry mesh must be a MeshPayload or RenderObjectState, "
            f"got {type(mesh).__name__}"
        )

    state = RenderObjectState(
        id=expected_id,
        payload=mesh,
        visible=bool(entry.get("visible", True)),
        metadata={"type": "scene_mesh"},
    )
    entry["entry_type"] = "mesh"
    entry["mesh"] = state
    entry["geometry_name"] = expected_id
    return state


def require_target_mesh_render_state(
    entry: dict[str, Any],
    index: int | None = None,
) -> RenderObjectState:
    """Return a target's canonical persistent state or raise on corruption."""
    ensure_target_entry_identity(entry, index)
    expected_id = make_target_entry_geometry_name(entry, "mesh")
    mesh = entry.get("mesh")
    if not isinstance(mesh, RenderObjectState) or not isinstance(mesh.payload, MeshPayload):
        raise TypeError("Persistent target entry mesh must be a mesh RenderObjectState")
    if mesh.id != expected_id:
        raise ValueError(
            f"Target mesh render ID mismatch: expected {expected_id!r}, got {mesh.id!r}"
        )
    entry["entry_type"] = "target"
    entry["geometry_name"] = expected_id
    return mesh


def normalize_node_label_mode(mode: Any) -> NodeLabelMode:
    """Normalize node label mode values used by UI and sessions."""
    normalized = str(mode or "role").strip().lower()
    return "name" if normalized == "name" else "role"


def resolve_node_label(
    node_type: str,
    index: int,
    custom_labels: Sequence[str] = (),
    *,
    label_mode: Any = "name",
    device_names: Sequence[str] = (),
) -> str:
    """Return the display label for a TX or RX node.

    Runtime custom labels always take precedence because they represent an
    explicit Rename action. ``role`` mode otherwise returns compact role/index
    labels. ``name`` mode otherwise uses scenario/frame device names before the
    compact role/index fallback.

    Args:
        node_type: ``"TX"`` or ``"RX"``.
        index: 0-based node index.
        custom_labels: Tuple of custom labels from ``AppState.tx_labels``
            or ``AppState.rx_labels``.
        label_mode: ``"role"`` for ``TX1``/``RX1``, or ``"name"`` for
            custom/scenario names.
        device_names: Tuple of source device names from scenario/frame data.

    Returns:
        Human-readable node label string.
    """
    role_label = f"{node_type}{index + 1}"
    if custom_labels and index < len(custom_labels) and custom_labels[index]:
        return custom_labels[index]
    if normalize_node_label_mode(label_mode) == "role":
        return role_label
    if device_names and index < len(device_names) and device_names[index]:
        return device_names[index]
    return role_label


def create_text_label(
    label_id: str,
    text: str,
    color: Any,
    font_size: float = 0.3,
    *,
    position: Any = (0.0, 0.0, 0.0),
    visible: bool = True,
    screen_space: bool = True,
) -> RenderObjectState:
    """Create a persistent label state for the common renderer contract.

    Args:
        label_id: Stable renderer object ID.
        text: Label text
        color: RGB color [r, g, b]
        font_size: Scale factor for the text (default 0.3)
        position: Absolute world anchor position.
        visible: Effective initial visibility.
        screen_space: Whether supported backends keep a constant pixel size.
    """
    return make_text_label_state(
        label_id,
        text,
        color,
        font_size=font_size,
        position=position,
        visible=visible,
        screen_space=screen_space,
    )


def create_building_labels(visualizer: OrchavVisualizer) -> None:
    """Generate labels for building meshes."""
    viz = visualizer
    viz.building_labels.clear()
    screen_space = bool(getattr(getattr(viz, "app_state", None), "label_screen_space", True))

    for index, mesh_entry in enumerate(viz.mesh_entries):
        if "name" in mesh_entry and "mesh" in mesh_entry and mesh_entry["mesh"] is not None:
            mesh = ensure_scene_mesh_render_state(mesh_entry, index)
            building_name = mesh_entry["name"]
            label_color = mesh_entry.get("color", [0.8, 0.8, 0.8])
            center = mesh_center(mesh)
            label = create_text_label(
                f"bldg_label_{index}",
                building_name,
                label_color,
                position=np.asarray(center, dtype=float) + np.array([0.0, 0.0, 2.0]),
                screen_space=screen_space,
            )
            viz.building_labels.append(label)

    logger.info("Created %s building labels", len(viz.building_labels))


def create_target_labels(visualizer: OrchavVisualizer) -> None:
    """Generate labels for target meshes.

    Uses custom labels from ``AppState.target_labels`` when available,
    falling back to the target entry's scene name.  Positions labels
    using the shared label offsets (``label_offset_x/y/z``).
    """
    viz = visualizer
    viz.target_labels.clear()

    custom_labels = getattr(getattr(viz, "app_state", None), "target_labels", ())
    offset_x = getattr(viz, "label_offset_x", 1.5)
    offset_y = getattr(viz, "label_offset_y", 0.0)
    offset_z = getattr(viz, "label_offset_z", 1.0)
    screen_space = bool(getattr(getattr(viz, "app_state", None), "label_screen_space", True))

    for i, target_entry in enumerate(viz.target_entries):
        if "mesh" not in target_entry or target_entry["mesh"] is None:
            continue

        default_name = target_entry.get("name", f"Target{i + 1}")
        if custom_labels and i < len(custom_labels) and custom_labels[i]:
            text = custom_labels[i]
        else:
            text = default_name

        anchor = target_entry_anchor_position(target_entry)
        if anchor is None:
            continue
        ensure_target_entry_identity(target_entry, i)
        label_color = target_entry.get("color", [0.8, 0.8, 0.8])
        label = create_text_label(
            make_target_entry_geometry_name(target_entry, "label"),
            text,
            label_color,
            position=[anchor[0] + offset_x, anchor[1] + offset_y, anchor[2] + offset_z],
            screen_space=screen_space,
        )

        viz.target_labels.append(label)

    logger.info("Created %s target labels", len(viz.target_labels))
