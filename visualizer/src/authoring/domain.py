"""Immutable, renderer-neutral values owned by the authoring document.

Canonical mobility and orientation fields come directly from
``shared.scenarios.actors``. The authoring layer adds only session identity,
selection state, presentation flags, and UUID-backed relationship handling.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeAlias, cast
from uuid import UUID, uuid4

from shared.scenarios.actors import (
    ActorRole,
    AlignMotionOrientationSpec,
    CatalogAssetSpec,
    CircularMobilitySpec,
    DirectoryAssetSpec,
    FileAssetSpec,
    FixedOrientationSpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    MobilitySpec,
    OrientationSpec,
    SpinOrientationSpec,
    StandaloneMobilitySpec,
    StationaryMobilitySpec,
    TargetAssetSpec,
    WaypointMobilitySpec,
)

Vector3: TypeAlias = tuple[float, float, float]
Mobility: TypeAlias = MobilitySpec
Orientation: TypeAlias = OrientationSpec
_YamlScalar: TypeAlias = str | int | float | bool | None

# Short aliases keep the authoring API readable while the shared schema
# classes remain the sole owners of serialized fields and validation.
Stationary = StationaryMobilitySpec
Linear = LinearMobilitySpec
Waypoint = WaypointMobilitySpec
Circular = CircularMobilitySpec
FixedOrientation = FixedOrientationSpec
KeyframesOrientation = KeyframesOrientationSpec
SpinOrientation = SpinOrientationSpec
AlignMotionOrientation = AlignMotionOrientationSpec
LookAtOrientation = LookAtOrientationSpec


def vector3(value: object) -> Vector3:
    """Return *value* as a finite three-component builtin-float tuple."""

    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("a three-component numeric coordinate is required")
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("a three-component numeric coordinate is required") from exc
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise ValueError("a coordinate must contain exactly three finite values")
    return values


def mobility_kind(mobility: Mobility) -> str:
    """Return the canonical mobility discriminator."""

    return str(mobility.type)


def orientation_kind(orientation: Orientation) -> str:
    """Return the canonical orientation discriminator."""

    return str(orientation.type)


class QualityPreset(str, Enum):
    """Ray-tracing presets available in the authoring workspace."""

    ULTRA_LOW = "ultra-low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class SceneReference:
    """Exact scene source/id pair selected by the author."""

    source: str
    id: str


@dataclass(frozen=True, slots=True)
class TimelineSettings:
    """Timeline and fixed generator-quality settings."""

    steps: int = 30
    duration_s: float = 3.0
    quality: QualityPreset = QualityPreset.ULTRA_LOW
    export_path_metrics: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality", QualityPreset(self.quality))


@dataclass(frozen=True, slots=True)
class _FrozenYamlSequence:
    values: tuple["_FrozenYamlValue", ...] = ()


@dataclass(frozen=True, slots=True)
class _FrozenYamlMapping:
    items: tuple[tuple[str, "_FrozenYamlValue"], ...] = ()


_FrozenYamlValue: TypeAlias = _YamlScalar | _FrozenYamlSequence | _FrozenYamlMapping


def _freeze_yaml_value(value: object) -> _FrozenYamlValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, _FrozenYamlValue]] = []
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("scenario source snapshot keys must be strings")
            items.append((key, _freeze_yaml_value(nested)))
        return _FrozenYamlMapping(tuple(items))
    if isinstance(value, (list, tuple)):
        return _FrozenYamlSequence(tuple(_freeze_yaml_value(item) for item in value))
    raise TypeError("scenario source snapshots support only YAML mappings, sequences, and scalars")


def _thaw_yaml_value(value: _FrozenYamlValue) -> Any:
    if isinstance(value, _FrozenYamlMapping):
        return {key: _thaw_yaml_value(nested) for key, nested in value.items}
    if isinstance(value, _FrozenYamlSequence):
        return [_thaw_yaml_value(nested) for nested in value.values]
    return value


@dataclass(frozen=True, slots=True)
class ScenarioSourceBinding:
    """Stable document UUID bound to one record in the imported source mapping."""

    id: UUID
    path: tuple[str | int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", UUID(str(self.id)))
        object.__setattr__(self, "path", tuple(self.path))


@dataclass(frozen=True, slots=True)
class ScenarioSourceSnapshot:
    """Complete immutable validated YAML used as the merge base.

    The Builder projects actors, groups, scene, and timeline controls out of
    this mapping. Serialization always merges those projections back into a
    detached copy, so schema-known configuration outside the editable surface
    is retained without maintaining a field allowlist.
    """

    _root: _FrozenYamlMapping = field(default_factory=_FrozenYamlMapping, repr=False)
    actor_bindings: tuple[ScenarioSourceBinding, ...] = ()
    group_bindings: tuple[ScenarioSourceBinding, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        *,
        actor_bindings: Iterable[ScenarioSourceBinding] = (),
        group_bindings: Iterable[ScenarioSourceBinding] = (),
    ) -> "ScenarioSourceSnapshot":
        """Freeze a validated YAML mapping and its stable record bindings."""

        frozen = _freeze_yaml_value(mapping)
        if not isinstance(frozen, _FrozenYamlMapping):
            raise TypeError("scenario source snapshot must be a mapping")
        return cls(frozen, tuple(actor_bindings), tuple(group_bindings))

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached mutable copy suitable for merged serialization."""

        thawed = _thaw_yaml_value(self._root)
        if not isinstance(thawed, dict):
            raise TypeError("scenario source snapshot must be a mapping")
        return thawed

    @property
    def is_empty(self) -> bool:
        return not self._root.items

    @property
    def top_level_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _value in self._root.items)

    def has_path(self, path: str) -> bool:
        """Return whether a dotted mapping path is present."""

        current: _FrozenYamlValue = self._root
        missing = object()
        for part in path.split("."):
            if not isinstance(current, _FrozenYamlMapping):
                return False
            match = next(
                (value for key, value in current.items if key == part),
                missing,
            )
            if match is missing:
                return False
            current = cast(_FrozenYamlValue, match)
        return True

    def without_path(self, path: str) -> "ScenarioSourceSnapshot":
        """Return a copy without one mapping path and newly empty parent mappings."""

        parts = tuple(part for part in path.split(".") if part)
        if not parts:
            return self
        root = self.to_mapping()
        current: dict[str, Any] = root
        parents: list[tuple[dict[str, Any], str]] = []
        for part in parts[:-1]:
            nested = current.get(part)
            if not isinstance(nested, dict):
                return self
            parents.append((current, part))
            current = nested
        if parts[-1] not in current:
            return self
        current.pop(parts[-1])
        for parent, key in reversed(parents):
            if parent.get(key) == {}:
                parent.pop(key)
            else:
                break
        return ScenarioSourceSnapshot.from_mapping(
            root,
            actor_bindings=self.actor_bindings,
            group_bindings=self.group_bindings,
        )

    @staticmethod
    def _record(
        root: Mapping[str, Any],
        binding: ScenarioSourceBinding,
    ) -> dict[str, Any] | None:
        current: Any = root
        for part in binding.path:
            if isinstance(part, int):
                if not isinstance(current, list) or part >= len(current):
                    return None
                current = current[part]
            else:
                if not isinstance(current, Mapping) or part not in current:
                    return None
                current = current[part]
        return dict(current) if isinstance(current, Mapping) else None

    def actor_record(self, actor_id: UUID | str) -> dict[str, Any] | None:
        """Return a detached imported record for one actor UUID."""

        wanted = UUID(str(actor_id))
        binding = next(
            (candidate for candidate in self.actor_bindings if candidate.id == wanted),
            None,
        )
        return None if binding is None else self._record(self.to_mapping(), binding)

    def group_record(self, group_id: UUID | str) -> dict[str, Any] | None:
        """Return a detached imported record for one group UUID."""

        wanted = UUID(str(group_id))
        binding = next(
            (candidate for candidate in self.group_bindings if candidate.id == wanted),
            None,
        )
        return None if binding is None else self._record(self.to_mapping(), binding)


@dataclass(frozen=True, slots=True)
class FacetCapability:
    """Editable/locked state for one Builder-owned projection facet."""

    facet: str
    path: str
    editable: bool = True
    reason: str = ""
    subject_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.subject_id is not None:
            object.__setattr__(self, "subject_id", UUID(str(self.subject_id)))
        if not self.editable and not self.reason.strip():
            raise ValueError("locked facet capabilities require an exact reason")


@dataclass(frozen=True, slots=True)
class ScenarioDependency:
    """One semantic input whose relative reference changes with scenario root."""

    source_path: Path
    relative_path: str
    kind: str
    origin_path: str
    external: bool = False
    problem: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        normalized = str(PurePosixPath(str(self.relative_path).replace("\\", "/")))
        if normalized in {"", "."}:
            raise ValueError("scenario dependency destinations must name a path")
        object.__setattr__(self, "relative_path", normalized)

    @property
    def available(self) -> bool:
        return self.source_path.exists()


class MobilityControlKind(str, Enum):
    """Semantic control-point roles exposed by mobility editors."""

    POSITION = "position"
    START = "start"
    END = "end"
    WAYPOINT = "waypoint"
    CENTER = "center"
    RIM = "rim"


@dataclass(frozen=True, slots=True)
class MobilityControl:
    """One immutable, renderer-neutral handle for authored mobility geometry."""

    kind: MobilityControlKind
    position: Vector3
    ordinal: int | None = None

    def __post_init__(self) -> None:
        kind = MobilityControlKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "position", vector3(self.position))
        if kind is MobilityControlKind.WAYPOINT:
            if (
                self.ordinal is None
                or isinstance(self.ordinal, bool)
                or not isinstance(self.ordinal, int)
                or self.ordinal < 0
            ):
                raise ValueError("waypoint controls require a non-negative ordinal")
        elif self.ordinal is not None:
            raise ValueError("only waypoint controls may have an ordinal")


def mobility_controls(mobility: Mobility) -> tuple[MobilityControl, ...]:
    """Return semantic handles for mobility models with direct point controls."""

    if isinstance(mobility, StationaryMobilitySpec):
        return (MobilityControl(MobilityControlKind.POSITION, mobility.position_m),)
    if isinstance(mobility, LinearMobilitySpec):
        return (
            MobilityControl(MobilityControlKind.START, mobility.start_m),
            MobilityControl(MobilityControlKind.END, mobility.end_m),
        )
    if isinstance(mobility, WaypointMobilitySpec):
        return tuple(
            MobilityControl(MobilityControlKind.WAYPOINT, point, ordinal=index)
            for index, point in enumerate(mobility.points_m)
        )
    if isinstance(mobility, CircularMobilitySpec):
        start_angle = math.radians(mobility.start_angle_deg)
        rim = (
            mobility.center_m[0] + mobility.radius_m * math.cos(start_angle),
            mobility.center_m[1] + mobility.radius_m * math.sin(start_angle),
            mobility.center_m[2],
        )
        return (
            MobilityControl(MobilityControlKind.CENTER, mobility.center_m),
            MobilityControl(MobilityControlKind.RIM, rim),
        )
    return ()


def update_mobility_control(
    mobility: Mobility,
    control: MobilityControl,
    world_position: Vector3,
) -> Mobility:
    """Return a simple-family mobility spec with one semantic control moved."""

    position = vector3(world_position)
    kind = control.kind

    def updated(**changes: object) -> Mobility:
        values = mobility.model_dump(mode="python")
        values.update(changes)
        return cast(Mobility, type(mobility).model_validate(values))

    if isinstance(mobility, StationaryMobilitySpec):
        if kind is not MobilityControlKind.POSITION:
            raise ValueError(f"{kind.value} is not a stationary mobility control")
        return updated(position_m=position)
    if isinstance(mobility, LinearMobilitySpec):
        if kind is MobilityControlKind.START:
            return updated(start_m=position)
        if kind is MobilityControlKind.END:
            return updated(end_m=position)
        raise ValueError(f"{kind.value} is not a linear mobility control")
    if isinstance(mobility, WaypointMobilitySpec):
        if kind is not MobilityControlKind.WAYPOINT or control.ordinal is None:
            raise ValueError(f"{kind.value} is not a waypoint mobility control")
        if control.ordinal >= len(mobility.points_m):
            raise ValueError(f"unknown waypoint control ordinal: {control.ordinal}")
        points = list(mobility.points_m)
        points[control.ordinal] = position
        return updated(points_m=tuple(points))
    if isinstance(mobility, CircularMobilitySpec):
        if kind is MobilityControlKind.CENTER:
            return updated(center_m=position)
        if kind is MobilityControlKind.RIM:
            dx = position[0] - mobility.center_m[0]
            dy = position[1] - mobility.center_m[1]
            return updated(
                radius_m=math.hypot(dx, dy),
                start_angle_deg=math.degrees(math.atan2(dy, dx)),
            )
        raise ValueError(f"{kind.value} is not a circular mobility control")
    raise ValueError(f"{mobility_kind(mobility)!r} has no simple mobility control")


def convert_mobility(
    mobility: Mobility,
    target_kind: str,
    anchor: Vector3,
) -> Mobility:
    """Convert through the complete mobility adapter registry."""

    from .mobility_models import convert_mobility as convert

    return convert(mobility, target_kind, vector3(anchor), duration_s=3.0, seed=0)


def translate_mobility(mobility: Mobility, offset: Vector3) -> Mobility:
    """Translate a mobility through its registered spatial adapter."""

    from .mobility_models import translate_mobility as translate

    return translate(mobility, vector3(offset))


@dataclass(frozen=True, slots=True)
class TargetAsset:
    """Shared catalog/file/directory target asset projected for editing."""

    asset_id: str
    mesh_directory: str
    mesh_pattern: str = "*.ply"
    material: str = "glass"
    scale: float = 1.0
    mesh_animation: bool = True
    source: str = "catalog"
    path: str | None = None
    start_index: int = 0
    frame_stride: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", float(self.scale))
        if self.source not in {"catalog", "file", "directory"}:
            raise ValueError(f"unsupported target asset source: {self.source!r}")
        if self.source != "catalog" and not self.path:
            object.__setattr__(self, "path", self.asset_id)

    @classmethod
    def from_catalog_id(
        cls,
        asset_id: str,
        *,
        mesh_pattern: str = "*.ply",
        material: str = "glass",
        scale: float = 1.0,
        mesh_animation: bool = True,
    ) -> "TargetAsset":
        """Create a target asset rooted in the repository target catalog."""

        normalized_id = str(PurePosixPath(str(asset_id).replace("\\", "/")))
        # Constructing the shared value here applies the canonical catalog ID
        # and scale validation before the authoring-only preview fields exist.
        CatalogAssetSpec(
            id=normalized_id,
            material_type=material,
            scale=scale,
            switch_meshes=mesh_animation,
        )
        return cls(
            asset_id=normalized_id,
            mesh_directory=f"libraries/targets/{normalized_id}",
            mesh_pattern=mesh_pattern,
            material=material,
            scale=scale,
            mesh_animation=mesh_animation,
        )

    @classmethod
    def from_spec(cls, spec: TargetAssetSpec) -> "TargetAsset":
        """Create an editable projection without changing its locator semantics."""

        if isinstance(spec, CatalogAssetSpec):
            return cls.from_catalog_id(
                spec.id,
                material=spec.material_type,
                scale=spec.scale,
                mesh_animation=spec.switch_meshes,
            )
        if isinstance(spec, FileAssetSpec):
            path = str(spec.path)
            return cls(
                asset_id=path,
                mesh_directory=str(PurePosixPath(path).parent),
                mesh_pattern=PurePosixPath(path).name,
                material=spec.material_type,
                scale=spec.scale,
                mesh_animation=False,
                source="file",
                path=path,
            )
        if isinstance(spec, DirectoryAssetSpec):
            path = str(spec.path)
            return cls(
                asset_id=path,
                mesh_directory=path,
                mesh_pattern=spec.pattern,
                material=spec.material_type,
                scale=spec.scale,
                mesh_animation=spec.switch_meshes,
                source="directory",
                path=path,
                start_index=spec.start_index,
                frame_stride=spec.frame_stride,
            )
        raise TypeError(f"unsupported target asset spec: {type(spec).__name__}")

    def to_mapping(self) -> dict[str, Any]:
        """Return an unchecked schema-shaped mapping.

        The compiler owns validation so it can report exact YAML paths instead
        of letting an invalid in-progress editor value raise during projection.
        """

        if self.source == "catalog":
            return {
                "source": "catalog",
                "id": self.asset_id,
                "material_type": self.material,
                "scale": self.scale,
                "switch_meshes": self.mesh_animation,
            }
        if self.source == "file":
            return {
                "source": "file",
                "path": str(self.path or self.asset_id),
                "material_type": self.material,
                "scale": self.scale,
            }
        return {
            "source": "directory",
            "path": str(self.path or self.asset_id),
            "pattern": self.mesh_pattern,
            "material_type": self.material,
            "scale": self.scale,
            "switch_meshes": self.mesh_animation,
            "start_index": self.start_index,
            "frame_stride": self.frame_stride,
        }


def _stationary_at_origin() -> StationaryMobilitySpec:
    return StationaryMobilitySpec(position_m=(0.0, 0.0, 0.0))


def _fixed_orientation() -> FixedOrientationSpec:
    return FixedOrientationSpec()


@dataclass(frozen=True, slots=True)
class AuthoringActor:
    """One TX, RX, or target in an authoring document."""

    id: UUID
    name: str
    role: ActorRole
    mobility: Mobility = field(default_factory=_stationary_at_origin)
    orientation: Orientation = field(default_factory=_fixed_orientation)
    target: TargetAsset | None = None
    visible: bool = True
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", UUID(str(self.id)))
        object.__setattr__(self, "role", ActorRole(self.role))

    @classmethod
    def create(
        cls,
        role: ActorRole | str,
        name: str,
        *,
        position: Vector3 = (0.0, 0.0, 0.0),
        target: TargetAsset | None = None,
        actor_id: UUID | str | None = None,
    ) -> "AuthoringActor":
        """Create a stationary, statically oriented actor."""

        return cls(
            id=UUID(str(actor_id)) if actor_id is not None else uuid4(),
            name=name,
            role=ActorRole(role),
            mobility=StationaryMobilitySpec(position_m=vector3(position)),
            orientation=FixedOrientationSpec(),
            target=target,
        )

    def with_changes(self, **changes: object) -> "AuthoringActor":
        """Return a changed copy while preserving immutable identity and role."""

        if "id" in changes and UUID(str(changes["id"])) != self.id:
            raise ValueError("actor id is immutable")
        if "role" in changes and ActorRole(changes["role"]) != self.role:
            raise ValueError("actor role is immutable")
        return cast(AuthoringActor, cast(Any, replace)(self, **changes))


@dataclass(frozen=True, slots=True)
class AuthoringGroup:
    """One optional shared mobility group with immutable session identity."""

    id: UUID
    name: str
    mobility: StandaloneMobilitySpec = field(default_factory=_stationary_at_origin)
    deviation: GroupDeviationSpec | None = None
    visible: bool = True
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", UUID(str(self.id)))
        if isinstance(self.mobility, GroupMemberMobilitySpec):
            raise ValueError("groups cannot reference another group")

    @classmethod
    def create(
        cls,
        name: str,
        *,
        position: Vector3 = (0.0, 0.0, 0.0),
        group_id: UUID | str | None = None,
    ) -> "AuthoringGroup":
        """Create a stationary group with a stable or generated identity."""

        return cls(
            id=UUID(str(group_id)) if group_id is not None else uuid4(),
            name=name,
            mobility=StationaryMobilitySpec(position_m=vector3(position)),
        )

    def with_changes(self, **changes: object) -> "AuthoringGroup":
        """Return an updated group while preserving its immutable identity."""

        if "id" in changes and UUID(str(changes["id"])) != self.id:
            raise ValueError("group id is immutable")
        return cast(AuthoringGroup, cast(Any, replace)(self, **changes))


class ResourceKind(str, Enum):
    """Kinds of external mobility data copied into a scenario directory."""

    NETWORK_GRAPH = "network_graph"
    POSITION_SEQUENCE = "position_sequence"


def canonical_authoring_resource_path(raw_path: Path | str) -> str | None:
    """Normalize a scenario-owned resource path and reject traversal.

    Absolute paths and ordinary relative paths are external resource sources,
    so they return ``None``. A value in the ``resources`` namespace must name
    a file below that directory and may not contain parent traversal.
    """

    raw = str(raw_path)
    normalized = raw.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise ValueError("mobility resource paths must not contain parent traversal")
    path = Path(raw)
    posix_path = PurePosixPath(normalized)
    if path.is_absolute() or posix_path.is_absolute() or PureWindowsPath(raw).is_absolute():
        return None
    if not posix_path.parts or posix_path.parts[0] != "resources":
        return None
    if len(posix_path.parts) < 2:
        raise ValueError("mobility resources must name a file below resources/")
    return str(posix_path)


def resolve_owned_authoring_resource(
    scenario_directory: Path | str,
    relative_path: Path | str,
) -> Path:
    """Resolve one canonical resource without permitting link-based escape."""

    canonical = canonical_authoring_resource_path(relative_path)
    if canonical is None:
        raise ValueError("scenario-owned mobility resources must be below resources/")
    scenario_root = Path(scenario_directory).resolve()
    destination = (scenario_root / Path(canonical)).resolve()
    if not destination.is_relative_to(scenario_root):
        raise ValueError(f"mobility resource resolves outside the scenario directory: {canonical}")
    return destination


@dataclass(frozen=True, slots=True)
class AuthoringResource:
    """Source file and its stable scenario-relative destination."""

    kind: ResourceKind
    source_path: Path
    relative_path: str

    def __post_init__(self) -> None:
        source = Path(self.source_path).resolve()
        canonical = canonical_authoring_resource_path(self.relative_path)
        if canonical is None:
            raise ValueError("authoring resources must be stored below resources/")
        object.__setattr__(self, "kind", ResourceKind(self.kind))
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "relative_path", canonical)


class SubjectKind(str, Enum):
    """Kinds that may be selected and addressed by validation issues."""

    ACTOR = "actor"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class AuthoringSubject:
    """Stable selection/reference to an actor or group."""

    kind: SubjectKind
    id: UUID

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SubjectKind(self.kind))
        object.__setattr__(self, "id", UUID(str(self.id)))


@dataclass(frozen=True, slots=True)
class AuthoringScenario:
    """Complete immutable content owned by :class:`ScenarioDocument`."""

    document_id: UUID = field(default_factory=uuid4)
    scene: SceneReference | None = None
    timeline: TimelineSettings = field(default_factory=TimelineSettings)
    actors: tuple[AuthoringActor, ...] = ()
    groups: tuple[AuthoringGroup, ...] = ()
    resources: tuple[AuthoringResource, ...] = ()
    source_snapshot: ScenarioSourceSnapshot = field(default_factory=ScenarioSourceSnapshot)
    facet_capabilities: tuple[FacetCapability, ...] = ()
    dependencies: tuple[ScenarioDependency, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", UUID(str(self.document_id)))
        object.__setattr__(self, "actors", tuple(self.actors))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "facet_capabilities", tuple(self.facet_capabilities))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not isinstance(self.source_snapshot, ScenarioSourceSnapshot):
            raise TypeError("source snapshot must be a ScenarioSourceSnapshot")

    def capability(
        self,
        facet: str,
        subject_id: UUID | str | None = None,
    ) -> FacetCapability:
        """Return an explicit facet capability, defaulting to editable."""

        wanted = UUID(str(subject_id)) if subject_id is not None else None
        return next(
            (
                capability
                for capability in self.facet_capabilities
                if capability.facet == facet and capability.subject_id == wanted
            ),
            FacetCapability(facet=facet, path=facet, subject_id=wanted),
        )

    def resource(self, relative_path: str) -> AuthoringResource | None:
        """Look up a resource by normalized scenario-relative POSIX path."""

        normalized = str(PurePosixPath(str(relative_path).replace("\\", "/")))
        return next(
            (resource for resource in self.resources if resource.relative_path == normalized),
            None,
        )

    def actor(self, actor_id: UUID | str) -> AuthoringActor | None:
        wanted = UUID(str(actor_id))
        return next((actor for actor in self.actors if actor.id == wanted), None)

    def actor_by_name(self, name: str) -> AuthoringActor | None:
        return next((actor for actor in self.actors if actor.name == name), None)

    def group(self, group_id: UUID | str) -> AuthoringGroup | None:
        wanted = UUID(str(group_id))
        return next((group for group in self.groups if group.id == wanted), None)

    def group_by_name(self, name: str) -> AuthoringGroup | None:
        return next((group for group in self.groups if group.name == name), None)

    def subject(self, reference: AuthoringSubject | None) -> AuthoringActor | AuthoringGroup | None:
        """Resolve a typed actor/group reference without changing selection."""

        if reference is None:
            return None
        if reference.kind is SubjectKind.ACTOR:
            return self.actor(reference.id)
        return self.group(reference.id)

    def replace_actor(self, replacement: AuthoringActor) -> "AuthoringScenario":
        """Replace an actor while preserving its identity and radio role."""

        existing = self.actor(replacement.id)
        if existing is None:
            raise KeyError(f"unknown actor id: {replacement.id}")
        if existing.role != replacement.role:
            raise ValueError("actor role is immutable")
        return replace(
            self,
            actors=tuple(
                replacement if actor.id == replacement.id else actor for actor in self.actors
            ),
        )

    def replace_group(self, replacement: AuthoringGroup) -> "AuthoringScenario":
        """Replace an existing group while preserving its identity."""

        if self.group(replacement.id) is None:
            raise KeyError(f"unknown group id: {replacement.id}")
        return replace(
            self,
            groups=tuple(
                replacement if group.id == replacement.id else group for group in self.groups
            ),
        )


def renderer_actor_id(document_id: UUID | str, actor_id: UUID | str) -> str:
    """Return a stable renderer namespace derived only from immutable IDs."""

    return f"authoring:{UUID(str(document_id))}:actor:{UUID(str(actor_id))}"


def renderer_group_id(document_id: UUID | str, group_id: UUID | str) -> str:
    """Return a stable renderer namespace for an authored group."""

    return f"authoring:{UUID(str(document_id))}:group:{UUID(str(group_id))}"
