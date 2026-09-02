"""Renderer-neutral visualizer model objects.

The classes here describe what the visualizer wants rendered. Backend-native
objects are created later, inside renderer packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    OrientationFramePayload,
    RenderPayload,
    TextLabelPayload,
)


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only shallow copy for stable render-object metadata."""
    if metadata is None:
        return MappingProxyType({})
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class Transform:
    """World transform stored as a 4x4 homogeneous matrix."""

    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))

    def __post_init__(self) -> None:
        """Normalize and freeze the homogeneous transform matrix."""
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {matrix.shape}")
        matrix = np.array(matrix, dtype=float, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> "Transform":
        """Return an identity transform."""
        return cls()

    @classmethod
    def from_translation(cls, translation: Any) -> "Transform":
        """Create a transform from an XYZ translation."""
        vec = np.asarray(translation, dtype=float).reshape(-1)
        if vec.size != 3:
            raise ValueError(f"Translation must contain 3 values, got {vec.size}")
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = vec[:3]
        return cls(matrix)

    @property
    def translation(self) -> np.ndarray:
        """Return a copy of the XYZ translation."""
        return self.matrix[:3, 3].copy()


@dataclass(frozen=True, slots=True)
class Visibility:
    """Visibility state for a render object."""

    visible: bool = True


@dataclass(frozen=True, slots=True)
class MaterialState:
    """Material state attached to a render object."""

    payload: MaterialPayload = field(default_factory=MaterialPayload)


@dataclass(frozen=True, slots=True)
class RenderObject:
    """Declarative renderer input keyed by a stable object ID."""

    id: str
    payload: RenderPayload
    material: Optional[MaterialState | MaterialPayload] = None
    transform: Transform | np.ndarray = field(default_factory=Transform.identity)
    visibility: Visibility | bool = field(default_factory=Visibility)
    is_edge: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize convenient constructor forms into immutable render state."""
        object_id = str(self.id).strip()
        if not object_id:
            raise ValueError("RenderObject id must be non-empty")
        object.__setattr__(self, "id", object_id)
        if isinstance(self.material, MaterialPayload):
            object.__setattr__(self, "material", MaterialState(self.material))
        if not isinstance(self.transform, Transform):
            object.__setattr__(self, "transform", Transform(np.asarray(self.transform)))
        if isinstance(self.visibility, bool):
            object.__setattr__(self, "visibility", Visibility(self.visibility))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def material_payload(self) -> Optional[MaterialPayload]:
        """Return the underlying material payload, if one is attached."""
        if self.material is None:
            return None
        return self.material.payload

    @property
    def visible(self) -> bool:
        """Return the boolean visibility state."""
        return bool(self.visibility.visible)

    @property
    def transform_matrix(self) -> np.ndarray:
        """Return the 4x4 transform matrix."""
        return self.transform.matrix


@dataclass(slots=True)
class RenderObjectState:
    """Mutable app-level render state keyed by one stable renderer object ID.

    It keeps neutral payloads, renderer IDs, material, transform, and
    visibility in app-owned state. Geometry edits stay in explicit helper
    functions so this type does not mimic a backend-native mesh API.
    """

    id: str
    payload: RenderPayload
    material: MaterialPayload = field(default_factory=MaterialPayload)
    world_transform: Transform = field(default_factory=Transform.identity)
    visible: bool = True
    is_edge: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize stable object id and transform inputs for app-owned state."""
        object_id = str(self.id).strip()
        if not object_id:
            raise ValueError("RenderObjectState id must be non-empty")
        self.id = object_id
        if not isinstance(self.world_transform, Transform):
            self.world_transform = Transform(np.asarray(self.world_transform, dtype=float))

    def __hash__(self) -> int:
        """Hash handles by stable renderer object id."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Compare handles by stable renderer object id."""
        if isinstance(other, RenderObjectState):
            return self.id == other.id
        return False

    def replace_payload(self, payload: RenderPayload) -> None:
        """Replace immutable renderer content with a new payload object."""
        self.payload = payload

    def to_render_object(self, *, effective_visible: Optional[bool] = None) -> RenderObject:
        """Return an immutable renderer input snapshot.

        ``visible`` remains application-owned intent. Callers that resolve
        parent or global visibility policy can pass the final renderer value
        without overwriting that intent.
        """
        return RenderObject(
            id=self.id,
            payload=self.payload,
            material=self.material,
            transform=self.world_transform,
            visibility=self.visible if effective_visible is None else bool(effective_visible),
            is_edge=self.is_edge,
            metadata=self.metadata,
        )


def make_text_label_state(
    label_id: str,
    text: str,
    color: Any,
    *,
    font_size: float = 0.3,
    position: Any = (0.0, 0.0, 0.0),
    visible: bool = True,
    screen_space: bool = True,
    outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    outline_thickness: float = 0.15,
) -> RenderObjectState:
    """Build one persistent renderer-neutral text label."""
    rgb = np.asarray(color, dtype=float).reshape(-1)
    if rgb.size < 3 or not np.all(np.isfinite(rgb[:3])):
        raise ValueError("Text label color must contain three finite values")
    return RenderObjectState(
        id=label_id,
        payload=TextLabelPayload(
            text=str(text),
            font_size=float(font_size),
            screen_space=bool(screen_space),
            outline_color=outline_color,
            outline_thickness=outline_thickness,
        ),
        material=MaterialPayload(
            base_color=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
            shader="unlit",
        ),
        world_transform=Transform.from_translation(position),
        visible=bool(visible),
        metadata={"type": "text_label"},
    )


def replace_render_state_payload(state: RenderObjectState, payload: RenderPayload) -> None:
    """Replace a state payload through its explicit mutation point."""
    state.replace_payload(payload)


def _readonly_view(values: Any, *, dtype: Any = None) -> np.ndarray:
    """Return a non-writable NumPy view without copying payload storage."""
    array = np.asarray(values, dtype=dtype)
    view = array.view()
    view.setflags(write=False)
    return view


def render_state_points(state: RenderObjectState) -> np.ndarray:
    """Return payload coordinates from a renderer-neutral state object."""
    payload = state.payload
    if isinstance(payload, MeshPayload):
        return _readonly_view(payload.vertices, dtype=float)
    if isinstance(payload, LineSetPayload):
        return _readonly_view(payload.points, dtype=float)
    if isinstance(payload, OrientationFramePayload):
        size = max(float(payload.size), 0.0)
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [size, 0.0, 0.0],
                [0.0, size, 0.0],
                [0.0, 0.0, size],
            ],
            dtype=float,
        )
    return _readonly_view(
        getattr(payload, "points", np.empty((0, 3))),
        dtype=float,
    )


def set_render_state_points(state: RenderObjectState, values: Any) -> None:
    """Replace payload coordinates on a renderer-neutral state object."""
    points = np.asarray(values, dtype=float)
    if isinstance(state.payload, MeshPayload):
        state.replace_payload(replace(state.payload, vertices=points, cache_key=None))
    elif isinstance(state.payload, LineSetPayload):
        state.replace_payload(replace(state.payload, points=points))
    elif hasattr(state.payload, "points"):
        state.replace_payload(replace(state.payload, points=points))
    else:
        raise AttributeError("Payload does not expose editable coordinates")


def render_state_triangles(state: RenderObjectState) -> np.ndarray:
    """Return mesh triangle indices when the state stores a mesh payload."""
    if isinstance(state.payload, MeshPayload):
        return _readonly_view(state.payload.triangles, dtype=np.int32)
    return _readonly_view(np.empty((0, 3), dtype=np.int32))


def set_render_state_triangles(state: RenderObjectState, values: Any) -> None:
    """Replace mesh triangle indices on a mesh-backed state object."""
    if not isinstance(state.payload, MeshPayload):
        raise AttributeError("Payload does not expose triangles")
    state.replace_payload(
        replace(
            state.payload,
            triangles=np.asarray(values, dtype=np.int32),
            cache_key=None,
        )
    )


def render_state_colors(state: RenderObjectState) -> Optional[np.ndarray]:
    """Return payload color buffers from a renderer-neutral state object."""
    if isinstance(state.payload, MeshPayload):
        colors = state.payload.vertex_colors
    else:
        colors = getattr(state.payload, "colors", None)
    return None if colors is None else _readonly_view(colors, dtype=float)


def set_render_state_colors(state: RenderObjectState, values: Any) -> None:
    """Replace payload color buffers on a renderer-neutral state object."""
    colors = np.asarray(values, dtype=float)
    if isinstance(state.payload, MeshPayload):
        state.replace_payload(replace(state.payload, vertex_colors=colors, cache_key=None))
    elif isinstance(state.payload, LineSetPayload):
        state.replace_payload(replace(state.payload, colors=colors))
    elif hasattr(state.payload, "colors"):
        state.replace_payload(replace(state.payload, colors=colors))
    else:
        raise AttributeError("Payload does not expose editable colors")


def render_state_bounds(state: RenderObjectState) -> tuple[np.ndarray, np.ndarray]:
    """Return local axis-aligned bounds for payload coordinates."""
    points = render_state_points(state)
    if points.size == 0:
        empty = np.zeros(3, dtype=float)
        return empty, empty
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    return points.min(axis=0), points.max(axis=0)


def render_state_local_center(state: RenderObjectState) -> np.ndarray:
    """Return the local mean coordinate of a renderer-neutral state object."""
    points = render_state_points(state)
    if points.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(points, dtype=float).reshape((-1, 3)).mean(axis=0)


def render_state_center(state: RenderObjectState) -> np.ndarray:
    """Return payload center after the state object's world transform."""
    center = np.append(render_state_local_center(state), 1.0)
    transformed = state.world_transform.matrix @ center
    return np.asarray(transformed[:3], dtype=float)


def render_state_aabb_center(state: RenderObjectState) -> np.ndarray:
    """Return the local AABB center for a renderer-neutral state object."""
    min_bound, max_bound = render_state_bounds(state)
    return (min_bound + max_bound) / 2.0


def offset_render_state_transform(
    state: RenderObjectState,
    delta: Any,
    *,
    relative: bool = True,
) -> None:
    """Move state transform by a relative offset or to an absolute center."""
    vec = np.asarray(delta, dtype=float).reshape(-1)
    if vec.size < 3:
        return
    vec = vec[:3]
    offset = vec if relative else vec - render_state_center(state)
    matrix = np.array(state.world_transform.matrix, dtype=float, copy=True)
    matrix[:3, 3] += offset
    state.world_transform = Transform(matrix)


def compose_render_state_transform(state: RenderObjectState, matrix: Any) -> None:
    """Compose a homogeneous transform onto the state world transform."""
    update = np.asarray(matrix, dtype=float)
    if update.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {update.shape}")
    state.world_transform = Transform(update @ state.world_transform.matrix)


def transform_render_state_payload(state: RenderObjectState, matrix: Any) -> None:
    """Apply a homogeneous transform directly to stored payload coordinates."""
    update = np.asarray(matrix, dtype=float)
    if update.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {update.shape}")
    points = render_state_points(state)
    if points.size == 0:
        return
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    hom = np.column_stack([points, np.ones(len(points), dtype=float)])
    transformed = (update @ hom.T).T[:, :3]
    set_render_state_points(state, transformed)


def tint_render_state_payload(state: RenderObjectState, color: Any) -> None:
    """Apply a uniform color to payload buffers and the material base color."""
    rgb = np.asarray(color, dtype=float).reshape(-1)
    if rgb.size < 3:
        return
    rgb = np.clip(rgb[:3], 0.0, 1.0)
    if isinstance(state.payload, MeshPayload):
        state.replace_payload(
            replace(
                state.payload,
                vertex_colors=np.tile(rgb, (len(state.payload.vertices), 1)),
                cache_key=None,
            )
        )
    elif isinstance(state.payload, LineSetPayload):
        state.replace_payload(
            replace(
                state.payload,
                colors=np.tile(rgb, (len(state.payload.lines), 1)),
            )
        )
    state.material = replace(
        state.material,
        base_color=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
    )


@dataclass(frozen=True, slots=True)
class SceneObject:
    """Domain object for a static scene mesh."""

    object_id: str
    render_id: str
    display_name: Optional[str] = None
    mesh_source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata for stable scene-object identity."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TargetInstance:
    """Domain object for a target mesh instance at a frame."""

    object_id: str
    render_id: str
    target_name: str
    frame_index: Optional[int] = None
    mesh_source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata for stable target-instance identity."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class NodeMarker:
    """Domain identity for a TX/RX visual marker and its optional label."""

    object_id: str
    render_id: str
    kind: str
    index: int
    label_render_id: Optional[str] = None
    display_name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize node kind/index and freeze marker metadata."""
        object.__setattr__(self, "kind", str(self.kind).lower())
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class VisualEntity:
    """Runtime scene entity synced through renderer-neutral object APIs.

    Targets, TX markers, and RX markers all use this contract: services own a
    stable entity id plus neutral payload/material/transform state, while
    renderer packages translate that payload into Open3D or pygfx objects.
    Labels are optional components keyed beside the primary render object.
    """

    entity_id: str
    category: str
    render_object: RenderObject
    display_name: Optional[str] = None
    label_render_object: Optional[RenderObject] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize entity identity and freeze semantic metadata."""
        entity_id = str(self.entity_id).strip()
        if not entity_id:
            raise ValueError("VisualEntity entity_id must be non-empty")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "category", str(self.category).lower())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def render_id(self) -> str:
        """Stable renderer object id for the entity's primary geometry."""
        return self.render_object.id

    @property
    def label_id(self) -> Optional[str]:
        """Stable renderer object id for the optional text label."""
        if self.label_render_object is None:
            return None
        return self.label_render_object.id

    @property
    def visible(self) -> bool:
        """Return the primary object visibility."""
        return self.render_object.visible


@dataclass(frozen=True, slots=True)
class TrajectoryPreview:
    """Domain object for a preview trajectory path."""

    object_id: str
    render_id: str
    kind: str
    points: np.ndarray
    marker_render_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize trajectory kind and freeze XYZ preview points."""
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"TrajectoryPreview points must have shape (N, 3), got {points.shape}")
        points = np.array(points, dtype=float, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "kind", str(self.kind).lower())
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
