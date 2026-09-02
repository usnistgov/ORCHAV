"""Typed boundary between Scenario Builder UI and its pygfx viewport.

Nothing in this module imports Qt or pygfx. The authoring workspace and tests
exchange immutable values here; backend-native event and object details remain
inside :mod:`visualizer.src.authoring.viewport` and ``interaction``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, TypeAlias
from uuid import UUID

import numpy as np

from ..types.render_payloads import MaterialPayload, MeshPayload
from .domain import MobilityControl
from .mobility_control_rig import MobilityControlRig

Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]


class AuthoringTool(str, Enum):
    """Mutually exclusive tools available in the authoring viewport."""

    SELECT = "select"
    PLACE = "place"
    MOVE = "move"
    WAYPOINT = "waypoint"


class PointerPhase(str, Enum):
    """Normalized pointer lifecycle phases."""

    DOWN = "down"
    UP = "up"
    MOVE = "move"
    DOUBLE_CLICK = "double_click"
    LEAVE = "leave"
    WHEEL = "wheel"


class TransformPhase(str, Enum):
    """One transient transform edit lifecycle."""

    BEGIN = "begin"
    UPDATE = "update"
    COMMIT = "commit"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class PointerInput:
    """Backend-neutral pointer input in logical viewport pixels."""

    phase: PointerPhase
    position: Vec2
    button: int = 0
    buttons: tuple[int, ...] = ()
    modifiers: frozenset[str] = field(default_factory=frozenset)
    wheel_delta: Vec2 = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class KeyboardInput:
    """Backend-neutral keyboard input routed by the active authoring tool."""

    key: str
    pressed: bool
    modifiers: frozenset[str] = field(default_factory=frozenset)
    repeat: bool = False


@dataclass(frozen=True, slots=True)
class HitResult:
    """Resolved authoring hit in world space."""

    world_position: Vec3
    renderer_object_id: str | None = None
    actor_id: UUID | None = None
    component: str | None = None
    vertex_index: int | None = None
    surface: bool = False


@dataclass(frozen=True, slots=True)
class TransformInput:
    """World-space transform emitted by the persistent authoring gizmo."""

    actor_id: UUID
    phase: TransformPhase
    matrix: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        matrix: np.ndarray = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("authoring transform must be a finite 4x4 matrix")


@dataclass(frozen=True, slots=True)
class RenderTurnInput:
    """A renderer ``before_render`` boundary used to coalesce overlay work."""

    sequence: int


class ActorVisualState(str, Enum):
    """Shape-and-color status presented for an authored actor."""

    COMPLETE = "complete"
    PENDING = "pending"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class OverlayVisibility(str, Enum):
    """Actor scope for optional authoring viewport overlays."""

    OFF = "off"
    SELECTED = "selected"
    ALL = "all"


class PreviewProvenance(str, Enum):
    """Evidence behind the trajectory presented by the viewport."""

    GENERATOR_PREPARED = "generator_prepared"
    AUTHORED_DRAFT = "authored_draft"


class TrajectoryDisplayMode(str, Enum):
    """Visual meaning of a prepared position sequence."""

    PATH = "path"
    OBSERVATIONS = "observations"


@dataclass(frozen=True, slots=True)
class TargetOverlayAsset:
    """Cached renderer-neutral target mesh state supplied by the compiler."""

    cache_key: str
    payload: MeshPayload
    material: MaterialPayload
    local_to_actor: tuple[tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        if not str(self.cache_key).strip():
            raise ValueError("target overlay cache key must be non-empty")
        if not isinstance(self.payload, MeshPayload):
            raise TypeError("target overlay payload must be a MeshPayload")
        if not isinstance(self.material, MaterialPayload):
            raise TypeError("target overlay material must be a MaterialPayload")
        transform: np.ndarray = np.asarray(self.local_to_actor, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("target local-to-actor transform must be a finite 4x4 matrix")


@dataclass(frozen=True, slots=True)
class SceneOverlayAsset:
    """One immutable, renderer-neutral mesh from the selected base scene."""

    cache_key: str
    name: str
    payload: MeshPayload
    material: MaterialPayload

    def __post_init__(self) -> None:
        if not str(self.cache_key).strip():
            raise ValueError("scene overlay cache key must be non-empty")
        if not str(self.name).strip():
            raise ValueError("scene overlay name must be non-empty")
        if not isinstance(self.payload, MeshPayload):
            raise TypeError("scene overlay payload must be a MeshPayload")
        if not isinstance(self.material, MaterialPayload):
            raise TypeError("scene overlay material must be a MaterialPayload")


@dataclass(frozen=True, slots=True)
class ActorOverlaySnapshot:
    """Complete renderer-neutral overlay state for one authored actor."""

    actor_id: UUID
    role: str
    name: str
    positions: tuple[Vec3, ...]
    frame_samples: tuple[Vec3, ...] = ()
    current_position: Vec3 | None = None
    look_at_position: Vec3 | None = None
    mobility_controls: tuple[MobilityControl, ...] = ()
    mobility_control_rig: MobilityControlRig | None = None
    authored_path: tuple[Vec3, ...] = ()
    selected: bool = False
    visible: bool = True
    locked: bool = False
    trajectory_visible: bool = True
    frame_samples_visible: bool = False
    closed_trajectory: bool = False
    trajectory_hovered: bool = False
    hovered_control_key: str | None = None
    preview_provenance: PreviewProvenance = PreviewProvenance.GENERATOR_PREPARED
    trajectory_display: TrajectoryDisplayMode = TrajectoryDisplayMode.PATH
    trajectory_geometry_key: str = ""
    frame_samples_geometry_key: str = ""
    status: ActorVisualState = ActorVisualState.COMPLETE
    orientation_matrix: tuple[tuple[float, float, float, float], ...] | None = None
    pending_positions: tuple[Vec3, ...] = ()
    pending_trajectory_display: TrajectoryDisplayMode = TrajectoryDisplayMode.PATH
    pending_current_position: Vec3 | None = None
    pending_look_at_position: Vec3 | None = None
    pending_mobility_control_rig: MobilityControlRig | None = None
    pending_orientation_matrix: tuple[tuple[float, float, float, float], ...] | None = None
    pending_target_asset: TargetOverlayAsset | None = None
    pending_geometry_key: str = ""
    mobility_draft_pending: bool = False
    orientation_draft_pending: bool = False
    group_origin_position: Vec3 | None = None
    group_frame_matrix: tuple[tuple[float, float, float, float], ...] | None = None
    transform_rotation_enabled: bool = True
    target_asset: TargetOverlayAsset | None = None

    def __post_init__(self) -> None:
        points: np.ndarray = np.asarray(self.positions, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("authoring overlay positions must be finite XYZ points")
        frame_samples: np.ndarray = np.asarray(self.frame_samples, dtype=float)
        if frame_samples.size and (
            frame_samples.ndim != 2
            or frame_samples.shape[1:] != (3,)
            or not np.all(np.isfinite(frame_samples))
        ):
            raise ValueError("generator frame samples must contain finite XYZ points")
        object.__setattr__(self, "preview_provenance", PreviewProvenance(self.preview_provenance))
        object.__setattr__(
            self,
            "trajectory_display",
            TrajectoryDisplayMode(self.trajectory_display),
        )
        object.__setattr__(
            self,
            "pending_trajectory_display",
            TrajectoryDisplayMode(self.pending_trajectory_display),
        )
        for control in self.mobility_controls:
            if not isinstance(control, MobilityControl):
                raise TypeError("authoring mobility controls must be MobilityControl values")
            position: np.ndarray = np.asarray(control.position, dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("authoring mobility controls must have finite XYZ positions")
        if self.mobility_control_rig is not None and not isinstance(
            self.mobility_control_rig, MobilityControlRig
        ):
            raise TypeError("authoring mobility control rig must be a MobilityControlRig")
        if self.pending_mobility_control_rig is not None and not isinstance(
            self.pending_mobility_control_rig,
            MobilityControlRig,
        ):
            raise TypeError("pending mobility control rig must be a MobilityControlRig")
        if self.hovered_control_key is not None:
            if self.mobility_control_rig is None:
                raise ValueError("a hovered mobility control requires a control rig")
            self.mobility_control_rig.control(self.hovered_control_key)
        authored_path: np.ndarray = np.asarray(self.authored_path, dtype=float)
        if authored_path.size and (
            authored_path.ndim != 2
            or authored_path.shape[1:] != (3,)
            or not np.all(np.isfinite(authored_path))
        ):
            raise ValueError("authored control path must contain finite XYZ points")
        for field_name, vector_value in (
            ("current_position", self.current_position),
            ("look_at_position", self.look_at_position),
            ("pending_current_position", self.pending_current_position),
            ("pending_look_at_position", self.pending_look_at_position),
            ("group_origin_position", self.group_origin_position),
        ):
            if vector_value is None:
                continue
            vector: np.ndarray = np.asarray(vector_value, dtype=float)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{field_name} must be a finite XYZ point")
        pending_positions: np.ndarray = np.asarray(self.pending_positions, dtype=float)
        if pending_positions.size and (
            pending_positions.ndim != 2
            or pending_positions.shape[1:] != (3,)
            or not np.all(np.isfinite(pending_positions))
        ):
            raise ValueError("pending positions must contain finite XYZ points")
        for field_name, matrix_value in (
            ("orientation_matrix", self.orientation_matrix),
            ("pending_orientation_matrix", self.pending_orientation_matrix),
            ("group_frame_matrix", self.group_frame_matrix),
        ):
            if matrix_value is None:
                continue
            matrix: np.ndarray = np.asarray(matrix_value, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{field_name} must be a finite 4x4 matrix")


@dataclass(frozen=True, slots=True)
class OverlaySnapshot:
    """Atomic authoring overlay revision supplied to the viewport."""

    document_id: UUID
    revision: int
    scene_assets: tuple[SceneOverlayAsset, ...] = ()
    actors: tuple[ActorOverlaySnapshot, ...] = ()
    work_plane_z: float = 0.0
    work_plane_visible: bool = True
    grid_snap_m: float | None = None
    placement_ghost: Vec3 | None = None
    placement_guide_start: Vec3 | None = None
    orientation_axes_visibility: OverlayVisibility = OverlayVisibility.SELECTED
    look_at_visibility: OverlayVisibility = OverlayVisibility.SELECTED

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("overlay revision must be non-negative")
        if not np.isfinite(float(self.work_plane_z)):
            raise ValueError("work plane height must be finite")
        if self.grid_snap_m is not None:
            snap = float(self.grid_snap_m)
            if not np.isfinite(snap) or snap <= 0.0:
                raise ValueError("grid snap must be positive and finite")
        if self.placement_ghost is not None:
            ghost: np.ndarray = np.asarray(self.placement_ghost, dtype=float)
            if ghost.shape != (3,) or not np.all(np.isfinite(ghost)):
                raise ValueError("placement ghost must be a finite XYZ point")
        if self.placement_guide_start is not None:
            guide_start: np.ndarray = np.asarray(self.placement_guide_start, dtype=float)
            if guide_start.shape != (3,) or not np.all(np.isfinite(guide_start)):
                raise ValueError("placement guide start must be a finite XYZ point")
        scene_keys = tuple(asset.cache_key for asset in self.scene_assets)
        if len(scene_keys) != len(set(scene_keys)):
            raise ValueError("overlay snapshot scene cache keys must be unique")
        actor_ids = tuple(actor.actor_id for actor in self.actors)
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("overlay snapshot actor IDs must be unique")


ViewportInput: TypeAlias = (
    PointerInput | KeyboardInput | HitResult | TransformInput | RenderTurnInput
)
ViewportEventSink: TypeAlias = Callable[[ViewportInput], None]


def stable_renderer_id(
    document_id: UUID | str,
    actor_id: UUID | str,
    component: str,
) -> str:
    """Return the stable renderer ID for one document/actor component."""

    component_token = str(component).strip().lower().replace(" ", "_")
    if not component_token or ":" in component_token:
        raise ValueError("renderer component must be a non-empty colon-free token")
    return f"authoring:{UUID(str(document_id))}:{UUID(str(actor_id))}:{component_token}"


def parse_renderer_id(value: str) -> tuple[UUID, UUID, str] | None:
    """Parse an authoring renderer ID, returning ``None`` for foreign IDs."""

    parts = str(value).split(":", 3)
    if len(parts) != 4 or parts[0] != "authoring" or not parts[3]:
        return None
    try:
        return UUID(parts[1]), UUID(parts[2]), parts[3]
    except (ValueError, AttributeError):
        return None


def stable_scene_renderer_id(document_id: UUID | str, cache_key: str) -> str:
    """Return a stable, non-actor renderer ID for one base-scene mesh."""

    normalized_key = str(cache_key).strip()
    if not normalized_key:
        raise ValueError("scene cache key must be non-empty")
    digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()[:24]
    return f"authoring:{UUID(str(document_id))}:scene_{digest}"


class ScenarioAuthoringViewportPort(Protocol):
    """Feature-specific port implemented only by the embedded pygfx host."""

    @property
    def widget(self) -> Any:
        """Return the final-parent Qt viewport widget."""
        ...

    @property
    def active(self) -> bool:
        """Return whether the authoring interaction session is registered."""
        ...

    def activate(self, sink: ViewportEventSink) -> None:
        """Activate one renderer-lifetime authoring event route."""
        ...

    def deactivate(self) -> None:
        """Remove every authoring event route and transient tool state."""
        ...

    def set_tool(self, tool: AuthoringTool) -> None:
        """Select the active authoring interaction tool."""
        ...

    def set_camera_mode(self, mode: str) -> bool:
        """Switch between the Orbit and Fly camera controllers."""
        ...

    def begin_drag_plane(self, z: float, source: HitResult) -> None:
        """Continue one picked handle or actor-body drag on a horizontal plane."""
        ...

    def begin_control_drag(self, constraint: str, source: HitResult) -> None:
        """Continue one semantic handle drag using its declared constraint."""
        ...

    def end_drag(self) -> None:
        """End the active fixed-plane handle drag, if any."""
        ...

    def clear_transform_gizmo(self) -> None:
        """Hide and detach the persistent semantic gizmo without destroying it."""
        ...

    def show_transform_gizmo(self, actor_id: UUID) -> bool:
        """Attach the persistent gizmo to the selected actor's semantic pose."""
        ...

    def reconcile(self, snapshot: OverlaySnapshot) -> None:
        """Converge renderer objects to one complete overlay snapshot."""
        ...

    def current_snapshot(self) -> OverlaySnapshot | None:
        """Return the most recently reconciled complete snapshot, if any."""
        ...

    def focus_actor(self, actor_id: UUID) -> bool:
        """Focus the camera on one actor, returning whether it was found."""
        ...

    def fit_all(self) -> bool:
        """Fit the camera to all authored geometry."""
        ...

    def renderer_objects(self) -> Mapping[str, Any]:
        """Return a read-only diagnostic view of authoring-owned objects."""
        ...
