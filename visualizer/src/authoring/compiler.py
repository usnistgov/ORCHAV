"""Canonical YAML compilation and generator-backed authoring validation."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import UUID

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from generator.core.configuration import load_simulation_config
from generator.core.materials.target_materials import validate_target_material_type
from generator.core.scenario_actors import (
    derive_group_member_mobility,
    group_offset_from_world_position,
    prepare_mobility_with_resources,
    prepare_scenario,
)
from generator.core.scenario_actors.runtime import (
    prepare_actor_runtime,
    target_asset_alignments,
)
from generator.core.scenario_actors.types import PreparedMobility, Timeline
from shared.scenarios.actors import (
    SCENARIO_SCHEMA_VERSION,
    AlignMotionOrientationSpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    MeshSequenceMobilitySpec,
    MobilitySpec,
    NetworkRouteMobilitySpec,
    OrientationSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
    StandaloneMobilitySpec,
    StationaryMobilitySpec,
    TargetAssetSpec,
)
from shared.scenarios.loader import ScenarioConfiguration, load_scenario_configuration
from shared.scenarios.parsers import resolve_sionna_scene_xml
from shared.scenarios.paths import find_project_root
from shared.scenarios.yaml import validate_scenario_data

from .assets import PreviewAssetCompiler, ScenePreviewAsset, TargetPreviewAsset
from .domain import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    Mobility,
    ResourceKind,
    ScenarioDependency,
    SceneReference,
    Vector3,
    canonical_authoring_resource_path,
    resolve_owned_authoring_resource,
    vector3,
)

DOCUMENT_VERSION = 2
FIXED_FRAME_FORMAT = "h5"
FIXED_FRAME_DIRECTORY = "frames"
FIXED_FRAME_PATTERN = "mpc_frames_*.h5"
FIXED_CHUNK_SIZE = 100
FIXED_COMPRESSION = "lzf"
PRESERVED_RAYTRACING_FIELDS = frozenset(
    {
        "view",
        "carrier_frequency_hz",
        "bandwidth_hz",
        "temperature_k",
        "antenna",
        "materials",
        "scene_materials",
    }
)


class IssueSeverity(str, Enum):
    """Severity used by the Problems drawer and save gate."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One machine-addressable authoring or generator validation problem."""

    severity: IssueSeverity
    code: str
    path: str
    message: str
    actor_id: UUID | None = None
    group_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ActorSamples:
    """Generator-prepared positions and degree-valued orientations."""

    positions: tuple[Vector3, ...]
    orientations: tuple[Vector3, ...]
    velocities_mps: tuple[Vector3, ...] = ()
    forward_vectors: tuple[Vector3, ...] = ()
    has_physical_velocity: bool = True


@dataclass(frozen=True, slots=True)
class GroupSamples:
    """Generator-prepared positions for one authored group path."""

    positions: tuple[Vector3, ...]
    prepared_mobility: PreparedMobility = field(repr=False, compare=False)
    timeline: Timeline = field(repr=False, compare=False)
    frame_transforms: tuple[
        tuple[tuple[float, float, float, float], ...],
        ...,
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        origins = self.positions
        axis_positions = tuple(
            derive_group_member_mobility(
                self.prepared_mobility,
                offset,
                self.timeline,
            ).positions_m
            for offset in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        transforms = []
        for index, origin in enumerate(origins):
            right, forward, up = tuple(
                tuple(
                    float(value - base)
                    for value, base in zip(
                        positions[index],
                        origin,
                        strict=True,
                    )
                )
                for positions in axis_positions
            )
            transforms.append(
                (
                    (right[0], forward[0], up[0], origin[0]),
                    (right[1], forward[1], up[1], origin[1]),
                    (right[2], forward[2], up[2], origin[2]),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        object.__setattr__(self, "frame_transforms", tuple(transforms))

    @property
    def has_physical_velocity(self) -> bool:
        """Return whether samples represent continuous physical motion."""

        return bool(self.prepared_mobility.has_physical_velocity)

    def frame_transform(
        self,
        *,
        step: int,
    ) -> tuple[tuple[float, float, float, float], ...]:
        """Return the canonical right/forward/up group frame at one sample."""

        index = int(step)
        if index < 0 or index >= len(self.positions):
            raise IndexError(f"group sample index is out of range: {index}")
        return self.frame_transforms[index]

    def offsets_for_world_positions(
        self,
        world_positions: tuple[Vector3, ...],
        *,
        step: int,
    ) -> tuple[Vector3, ...]:
        """Project world positions through the prepared group's local frame."""

        return tuple(
            vector3(
                group_offset_from_world_position(
                    self.prepared_mobility,
                    position,
                    step=step,
                )
            )
            for position in world_positions
        )


@dataclass(frozen=True, slots=True)
class CompiledRuntime:
    """Generator objects constructed from the canonical YAML mapping."""

    scenario_configuration: ScenarioConfiguration
    simulation_config: Any
    transmitters: tuple[Any, ...]
    receivers: tuple[Any, ...]
    targets: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Canonical output, structured problems, runtime objects, and samples."""

    mapping: dict[str, Any]
    yaml_text: str
    issues: tuple[ValidationIssue, ...]
    samples: Mapping[UUID, ActorSamples]
    group_samples: Mapping[UUID, GroupSamples] = field(default_factory=dict)
    runtime: CompiledRuntime | None = None
    resolved_scene_path: str | None = None
    scene_assets: tuple[ScenePreviewAsset, ...] = ()
    target_assets: Mapping[UUID, tuple[TargetPreviewAsset, ...]] = field(default_factory=dict)
    generation_issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether no error-severity issue blocks save/generation."""
        return not any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    def issues_for_actor(self, actor_id: UUID | str) -> tuple[ValidationIssue, ...]:
        """Return issues attached to one immutable actor identity."""
        wanted = UUID(str(actor_id))
        return tuple(issue for issue in self.issues if issue.actor_id == wanted)

    def issues_for_group(self, group_id: UUID | str) -> tuple[ValidationIssue, ...]:
        """Return issues attached to one immutable group identity."""

        wanted = UUID(str(group_id))
        return tuple(issue for issue in self.issues if issue.group_id == wanted)

    def group_offsets(
        self,
        group_id: UUID | str,
        world_positions: tuple[Vector3, ...],
        *,
        step: int,
    ) -> tuple[Vector3, ...]:
        """Project positions through the exact prepared group used by preview."""

        wanted = UUID(str(group_id))
        samples = self.group_samples.get(wanted)
        if samples is None:
            raise KeyError(f"prepared samples are unavailable for group {wanted}")
        return samples.offsets_for_world_positions(world_positions, step=step)


def _restore_nested_discriminators(value: Any, dumped: Any) -> Any:
    """Restore model discriminators removed by recursive default exclusion."""

    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        restored = {
            key: _restore_nested_discriminators(getattr(value, key), item)
            for key, item in dumped.items()
        }
        discriminator = getattr(value, "type", None)
        if discriminator is None:
            return restored
        return {
            "type": str(discriminator),
            **{key: item for key, item in restored.items() if key != "type"},
        }
    if isinstance(value, (list, tuple)) and isinstance(dumped, list):
        return [_restore_nested_discriminators(source, item) for source, item in zip(value, dumped)]
    if isinstance(value, Mapping) and isinstance(dumped, dict):
        return {
            key: _restore_nested_discriminators(value[key], item) for key, item in dumped.items()
        }
    return dumped


def _spec_mapping(spec: BaseModel) -> dict[str, Any]:
    """Dump a shared immutable specification with concise schema defaults."""

    dumped = spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    return cast(dict[str, Any], _restore_nested_discriminators(spec, dumped))


def _mobility_mapping(
    mobility: Mobility,
    scenario: AuthoringScenario,
) -> dict[str, Any]:
    mapping = _spec_mapping(mobility)
    if isinstance(mobility, GroupMemberMobilitySpec):
        group = _group_reference(scenario, mobility.group)
        mapping["group"] = group.name if group is not None else ""
    return mapping


def _orientation_mapping(
    actor: AuthoringActor,
    scenario: AuthoringScenario,
) -> dict[str, Any]:
    orientation = actor.orientation
    mapping = _spec_mapping(orientation)
    if isinstance(orientation, LookAtOrientationSpec) and orientation.actor is not None:
        target = _actor_reference(scenario, orientation.actor)
        mapping["actor"] = target.name if target is not None else ""
    return mapping


def _actor_reference(
    scenario: AuthoringScenario,
    reference: str,
) -> AuthoringActor | None:
    try:
        actor_id = UUID(str(reference))
    except ValueError:
        return scenario.actor_by_name(str(reference))
    return scenario.actor(actor_id)


def _actor_reference_id(
    scenario: AuthoringScenario,
    reference: str,
) -> UUID | None:
    actor = _actor_reference(scenario, reference)
    return actor.id if actor is not None else None


def _group_reference(
    scenario: AuthoringScenario,
    reference: str,
) -> AuthoringGroup | None:
    try:
        group_id = UUID(str(reference))
    except ValueError:
        return scenario.group_by_name(str(reference))
    return scenario.group(group_id)


class _MobilityResourceProblem(ValueError):
    """One resource failure that maps directly to a validation issue code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _mobility_resource_field(
    mobility: Mobility,
) -> tuple[str, str, ResourceKind] | None:
    if isinstance(mobility, MeshSequenceMobilitySpec):
        return (
            "positions_path",
            mobility.positions_path,
            ResourceKind.POSITION_SEQUENCE,
        )
    if isinstance(mobility, NetworkRouteMobilitySpec):
        return (
            "graph_path",
            mobility.graph_path or "street_network.graphml",
            ResourceKind.NETWORK_GRAPH,
        )
    return None


def _resolve_authoring_mobility_resource(
    mobility: Mobility,
    *,
    scenario_root: Path,
    resources: tuple[AuthoringResource, ...],
    dependencies: tuple[ScenarioDependency, ...] = (),
) -> tuple[str, str, Path] | None:
    """Resolve a resource only after its path, containment, and kind are safe."""

    details = _mobility_resource_field(mobility)
    if details is None:
        return None
    field_name, raw_path, expected_kind = details
    dependency = next(
        (
            item
            for item in dependencies
            if item.relative_path == str(PurePosixPath(raw_path.replace("\\", "/")))
            and item.kind in {expected_kind.value, "osm_artifact"}
        ),
        None,
    )
    if dependency is not None:
        if dependency.problem:
            raise _MobilityResourceProblem(
                "mobility.resource.outside_scenario",
                dependency.problem,
            )
        if dependency.source_path.is_file():
            return field_name, raw_path, dependency.source_path
        raise _MobilityResourceProblem(
            "mobility.resource.unresolved",
            f"Mobility resource was not found: {dependency.source_path}",
        )
    try:
        canonical = canonical_authoring_resource_path(raw_path)
    except ValueError as exc:
        raise _MobilityResourceProblem(
            "mobility.resource.invalid_path",
            str(exc),
        ) from exc

    if canonical is not None:
        try:
            owned_source = resolve_owned_authoring_resource(
                scenario_root,
                canonical,
            )
        except ValueError as exc:
            raise _MobilityResourceProblem(
                "mobility.resource.outside_scenario",
                str(exc),
            ) from exc
        registrations = tuple(
            resource for resource in resources if resource.relative_path == canonical
        )
        if any(resource.kind != expected_kind for resource in registrations):
            registered_kinds = ", ".join(
                sorted({resource.kind.value for resource in registrations})
            )
            raise _MobilityResourceProblem(
                "mobility.resource.kind_collision",
                f"Mobility resource {canonical!r} is registered as "
                f"{registered_kinds}, not {expected_kind.value}.",
            )
        registered_sources = {resource.source_path.resolve() for resource in registrations}
        if len(registered_sources) > 1:
            raise _MobilityResourceProblem(
                "mobility.resource.registration_collision",
                f"Mobility resource {canonical!r} has multiple registered sources.",
            )
        if registrations:
            registered_source = registrations[0].source_path.resolve()
            if registered_source.is_file():
                return field_name, raw_path, registered_source
            missing = registered_source
        elif owned_source.is_file():
            return field_name, raw_path, owned_source
        else:
            missing = owned_source
        raise _MobilityResourceProblem(
            "mobility.resource.unresolved",
            f"Mobility resource was not found: {missing}",
        )

    candidate = Path(raw_path)
    source = (
        candidate.resolve() if candidate.is_absolute() else (scenario_root / candidate).resolve()
    )
    if not candidate.is_absolute() and not source.is_relative_to(scenario_root):
        raise _MobilityResourceProblem(
            "mobility.resource.outside_scenario",
            f"Mobility resource resolves outside the scenario directory: {raw_path}",
        )
    if not source.is_file():
        raise _MobilityResourceProblem(
            "mobility.resource.unresolved",
            f"Mobility resource was not found: {source}",
        )
    return field_name, raw_path, source


def _prepare_authoring_mobility(
    mobility: StandaloneMobilitySpec,
    timeline: Timeline,
    *,
    scenario_root: Path,
    resources: tuple[AuthoringResource, ...],
    dependencies: tuple[ScenarioDependency, ...] = (),
    path: str,
) -> PreparedMobility:
    """Prepare one mobility using its owned or pending authoring resource."""

    prepared_spec = mobility
    resolved = _resolve_authoring_mobility_resource(
        mobility,
        scenario_root=scenario_root,
        resources=resources,
        dependencies=dependencies,
    )
    if resolved is not None:
        field_name, _raw_path, source = resolved
        prepared_spec = cast(
            StandaloneMobilitySpec,
            mobility.model_copy(
                update={field_name: str(source)},
            ),
        )

    return prepare_mobility_with_resources(
        prepared_spec,
        timeline,
        base_dir=scenario_root,
        path=path,
    )


def _actor_mapping(actor: AuthoringActor, scenario: AuthoringScenario) -> dict[str, Any]:
    result = scenario.source_snapshot.actor_record(actor.id) or {}
    result["name"] = actor.name
    mobility = _mobility_mapping(actor.mobility, scenario)
    if not _same_spec(result.get("mobility"), mobility, _MOBILITY_ADAPTER):
        result["mobility"] = mobility
    orientation = _orientation_mapping(actor, scenario)
    if not ("orientation" not in result and orientation == {"type": "fixed"}) and not _same_spec(
        result.get("orientation"), orientation, _ORIENTATION_ADAPTER
    ):
        result["orientation"] = orientation
    if actor.role is ActorRole.TARGET and actor.target is not None:
        asset = actor.target.to_mapping()
        if not _same_spec(result.get("asset"), asset, _TARGET_ASSET_ADAPTER):
            result["asset"] = asset
    return result


_MOBILITY_ADAPTER = TypeAdapter(MobilitySpec)
_ORIENTATION_ADAPTER = TypeAdapter(OrientationSpec)
_TARGET_ASSET_ADAPTER = TypeAdapter(TargetAssetSpec)
_DEVIATION_ADAPTER = TypeAdapter(GroupDeviationSpec)


def _same_spec(
    original: Any,
    projected: Mapping[str, Any],
    adapter: TypeAdapter[Any],
) -> bool:
    """Compare schema meaning so unchanged explicit defaults remain byte-semantic."""

    if not isinstance(original, Mapping):
        return False
    try:
        return bool(
            adapter.validate_python(dict(original)) == adapter.validate_python(dict(projected))
        )
    except ValidationError:
        return False


def _group_mapping(group: AuthoringGroup, scenario: AuthoringScenario) -> dict[str, Any]:
    result = scenario.source_snapshot.group_record(group.id) or {}
    result["name"] = group.name
    mobility = _spec_mapping(group.mobility)
    if not _same_spec(result.get("mobility"), mobility, _MOBILITY_ADAPTER):
        result["mobility"] = mobility
    if group.deviation is not None:
        deviation = _spec_mapping(group.deviation)
        if not _same_spec(result.get("deviation"), deviation, _DEVIATION_ADAPTER):
            result["deviation"] = deviation
    else:
        result.pop("deviation", None)
    return result


def merged_scenario_mapping(scenario: AuthoringScenario) -> dict[str, Any]:
    """Merge Builder-owned projections into a copy of the validated source."""

    mapping = scenario.source_snapshot.to_mapping()
    mapping["schema_version"] = SCENARIO_SCHEMA_VERSION
    if scenario.scene is not None:
        source_scene = mapping.get("scene")
        same_scene = (
            isinstance(source_scene, Mapping)
            and source_scene.get("source") == scenario.scene.source
            and source_scene.get("id") == scenario.scene.id
        )
        if not same_scene:
            # Replacing an OSM scene is explicit: its generation block no
            # longer describes the newly selected source.
            mapping["scene"] = {
                "source": scenario.scene.source,
                "id": scenario.scene.id,
            }
    else:
        mapping.pop("scene", None)

    timeline = scenario.timeline
    timeline_mapping = mapping.get("timeline")
    if not isinstance(timeline_mapping, dict):
        timeline_mapping = {}
        mapping["timeline"] = timeline_mapping
    timeline_mapping["steps"] = timeline.steps
    timeline_mapping["duration_s"] = timeline.duration_s

    raytracing = mapping.get("raytracing")
    if not isinstance(raytracing, dict):
        raytracing = {"enabled": True}
        mapping["raytracing"] = raytracing
    raytracing["export_path_metrics"] = timeline.export_path_metrics
    quality = raytracing.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        raytracing["quality"] = quality
    quality["preset"] = timeline.quality.value

    if scenario.groups:
        mapping["groups"] = [_group_mapping(group, scenario) for group in scenario.groups]
    else:
        mapping.pop("groups", None)

    actor_mapping = mapping.get("actors")
    if not isinstance(actor_mapping, dict):
        actor_mapping = {}
        mapping["actors"] = actor_mapping
    tx = [
        _actor_mapping(actor, scenario) for actor in scenario.actors if actor.role is ActorRole.TX
    ]
    rx = [
        _actor_mapping(actor, scenario) for actor in scenario.actors if actor.role is ActorRole.RX
    ]
    targets = [
        _actor_mapping(actor, scenario)
        for actor in scenario.actors
        if actor.role is ActorRole.TARGET
    ]
    actor_mapping["tx"] = tx
    actor_mapping["rx"] = rx
    if targets:
        actor_mapping["targets"] = targets
    else:
        actor_mapping.pop("targets", None)

    visualizer = mapping.get("visualizer")
    if not isinstance(visualizer, dict):
        visualizer = {}
        mapping["visualizer"] = visualizer
    marker = visualizer.get("scenario_builder")
    if not isinstance(marker, dict):
        marker = {}
        visualizer["scenario_builder"] = marker
    marker["document_version"] = DOCUMENT_VERSION
    return mapping


def canonical_scenario_mapping(scenario: AuthoringScenario) -> dict[str, Any]:
    """Compatibility name for the schema-first merged mapping."""

    return merged_scenario_mapping(scenario)


def canonical_yaml(mapping: Mapping[str, Any]) -> str:
    """Return deterministic, safe YAML for a canonical scenario mapping."""
    return str(
        yaml.safe_dump(
            dict(mapping),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    )


class ScenarioCompiler:
    """Compile, validate, construct, and prepare one authoring scenario."""

    def __init__(self, project_root: Path | None = None) -> None:
        start = Path(__file__).resolve().parent
        self.project_root = (
            Path(project_root).resolve() if project_root is not None else find_project_root(start)
        )
        self._preview_assets = PreviewAssetCompiler(self.project_root)

    def compile(
        self,
        scenario: AuthoringScenario | Any,
        *,
        scenario_directory: Path | None = None,
    ) -> CompilationResult:
        """Return canonical YAML and all available generator-backed evidence."""
        if not isinstance(scenario, AuthoringScenario) and hasattr(scenario, "scenario"):
            scenario = scenario.scenario
        if not isinstance(scenario, AuthoringScenario):
            raise TypeError("compile() requires an AuthoringScenario or ScenarioDocument")

        root = (
            Path(scenario_directory).resolve()
            if scenario_directory is not None
            else self.project_root
        )
        mapping = canonical_scenario_mapping(scenario)
        yaml_text = canonical_yaml(mapping)
        issues = self._domain_issues(scenario)
        issues.extend(self._dependency_issues(scenario))
        issues.extend(self._asset_issues(scenario, root))
        issues.extend(self._prepared_mobility_orientation_issues(scenario, root))

        resolved_scene_path: str | None = None
        scene_assets: tuple[ScenePreviewAsset, ...] = ()
        if scenario.scene is not None and not any(
            issue.severity is IssueSeverity.ERROR and issue.path.startswith("scene")
            for issue in issues
        ):
            (
                resolved_scene_path,
                scene_assets,
                scene_preview_issues,
            ) = self._compile_scene_preview_assets(scenario, root)
            issues.extend(scene_preview_issues)

        if not self._has_errors(issues):
            issues.extend(self._schema_issues(mapping))

        runtime: CompiledRuntime | None = None
        samples: dict[UUID, ActorSamples] = {}
        group_samples: dict[UUID, GroupSamples] = {}
        if not self._has_errors(issues):
            try:
                runtime, samples, group_samples = self._construct_and_prepare(
                    scenario,
                    yaml_text,
                    root,
                )
            # Runtime construction is a validation boundary whose adapters may
            # raise unrelated exception types. Convert ordinary failures into a
            # structured problem so the Problems drawer remains authoritative.
            except Exception as exc:
                issues.append(self._runtime_issue(scenario, exc))
        else:
            samples = self._prepare_partial_samples(scenario, issues, root)
            group_samples = self._prepare_partial_group_samples(
                scenario,
                issues,
                root,
            )

        if runtime is not None and samples:
            issues.extend(self._prepared_orientation_issues(scenario, samples))

        target_assets: dict[UUID, tuple[TargetPreviewAsset, ...]] = {}
        if samples:
            # Partial samples keep valid targets visible while unrelated
            # document errors continue to block save and generation.
            target_assets, target_preview_issues = self._compile_target_preview_assets(
                scenario,
                root,
                samples,
                report_missing_samples=runtime is not None,
            )
            issues.extend(target_preview_issues)

        return CompilationResult(
            mapping=mapping,
            yaml_text=yaml_text,
            issues=tuple(issues),
            samples=samples,
            group_samples=group_samples,
            runtime=runtime,
            resolved_scene_path=resolved_scene_path,
            scene_assets=scene_assets,
            target_assets=target_assets,
            generation_issues=tuple(self._generation_issues(mapping)),
        )

    def validate(
        self,
        scenario: AuthoringScenario | Any,
        *,
        scenario_directory: Path | None = None,
    ) -> tuple[ValidationIssue, ...]:
        """Validate through the same complete compile path used for saving."""
        return self.compile(scenario, scenario_directory=scenario_directory).issues

    @staticmethod
    def group_offsets(
        mobility: StandaloneMobilitySpec,
        timeline_steps: int,
        duration_s: float,
        world_positions: tuple[Vector3, ...],
        *,
        step: int = 0,
        scenario_directory: Path | None = None,
        resources: tuple[AuthoringResource, ...] = (),
    ) -> tuple[Vector3, ...]:
        """Project world positions into one canonically prepared group frame."""

        prepared = _prepare_authoring_mobility(
            mobility,
            Timeline(int(timeline_steps), float(duration_s)),
            scenario_root=Path(scenario_directory or Path.cwd()).resolve(),
            resources=resources,
            path="group.mobility",
        )
        return tuple(
            vector3(
                group_offset_from_world_position(
                    prepared,
                    position,
                    step=step,
                )
            )
            for position in world_positions
        )

    @staticmethod
    def _has_errors(issues: list[ValidationIssue]) -> bool:
        return any(issue.severity is IssueSeverity.ERROR for issue in issues)

    @staticmethod
    def _issue(
        code: str,
        path: str,
        message: str,
        actor: AuthoringActor | None = None,
        group: AuthoringGroup | None = None,
        severity: IssueSeverity = IssueSeverity.ERROR,
    ) -> ValidationIssue:
        return ValidationIssue(
            severity,
            code,
            path,
            message,
            actor.id if actor is not None else None,
            group.id if group is not None else None,
        )

    @staticmethod
    def _actor_paths(scenario: AuthoringScenario) -> dict[UUID, str]:
        counters = {role: 0 for role in ActorRole}
        paths: dict[UUID, str] = {}
        for actor in scenario.actors:
            index = counters[actor.role]
            counters[actor.role] += 1
            section = "targets" if actor.role is ActorRole.TARGET else actor.role.value
            prefix = f"actors.{section}"
            paths[actor.id] = f"{prefix}.{index}"
        return paths

    @staticmethod
    def _group_paths(scenario: AuthoringScenario) -> dict[UUID, str]:
        return {group.id: f"groups.{index}" for index, group in enumerate(scenario.groups)}

    def _domain_issues(self, scenario: AuthoringScenario) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        paths = self._actor_paths(scenario)
        scene = scenario.scene
        if scene is None:
            issues.append(self._issue("scene.required", "scene", "Select a scene before saving."))
        else:
            if not scene.id.strip():
                issues.append(self._issue("scene.id.required", "scene.id", "Scene id is required."))

        timeline = scenario.timeline
        if isinstance(timeline.steps, bool) or not isinstance(timeline.steps, int):
            issues.append(
                self._issue("timeline.steps.type", "timeline.steps", "Steps must be an integer.")
            )
        elif timeline.steps < 1:
            issues.append(
                self._issue("timeline.steps.minimum", "timeline.steps", "Steps must be at least 1.")
            )
        if not math.isfinite(timeline.duration_s) or timeline.duration_s < 0.0:
            issues.append(
                self._issue(
                    "timeline.duration.invalid",
                    "timeline.duration_s",
                    "Duration must be finite and nonnegative.",
                )
            )

        groups_by_id = {str(group.id): group for group in scenario.groups}

        def effective_mobility(actor: AuthoringActor) -> Mobility:
            mobility = actor.mobility
            if not isinstance(mobility, GroupMemberMobilitySpec):
                return mobility
            group = groups_by_id.get(mobility.group)
            return group.mobility if group is not None else mobility

        moving = any(
            not isinstance(
                effective_mobility(actor),
                (StationaryMobilitySpec, GroupMemberMobilitySpec),
            )
            for actor in scenario.actors
        ) or any(
            not isinstance(group.mobility, StationaryMobilitySpec) for group in scenario.groups
        )
        moving = moving or any(
            isinstance(
                actor.orientation,
                (
                    KeyframesOrientationSpec,
                    SpinOrientationSpec,
                    RandomOrientationSpec,
                ),
            )
            for actor in scenario.actors
        )
        if moving and isinstance(timeline.steps, int) and timeline.steps < 2:
            issues.append(
                self._issue(
                    "timeline.moving_steps",
                    "timeline.steps",
                    "Moving documents require at least two steps.",
                )
            )
        if moving and (not math.isfinite(timeline.duration_s) or timeline.duration_s <= 0.0):
            issues.append(
                self._issue(
                    "timeline.moving_duration",
                    "timeline.duration_s",
                    "Moving documents require a positive duration.",
                )
            )

        role_counts = {
            role: sum(actor.role is role for actor in scenario.actors) for role in ActorRole
        }
        if role_counts[ActorRole.TX] == 0:
            issues.append(
                self._issue("actors.tx.required", "actors.tx", "At least one TX is required.")
            )
        if role_counts[ActorRole.RX] == 0:
            issues.append(
                self._issue("actors.rx.required", "actors.rx", "At least one RX is required.")
            )

        names: dict[str, AuthoringActor] = {}
        ids: set[UUID] = set()
        for actor in scenario.actors:
            base = paths[actor.id]
            if actor.id in ids:
                issues.append(
                    self._issue(
                        "actor.id.duplicate",
                        f"{base}.id",
                        f"Actor UUID {actor.id} is duplicated.",
                        actor,
                    )
                )
            ids.add(actor.id)
            if not actor.name.strip():
                issues.append(
                    self._issue(
                        "actor.name.required",
                        f"{base}.name",
                        "Actor name is required.",
                        actor,
                    )
                )
            elif actor.name in names:
                issues.append(
                    self._issue(
                        "actor.name.duplicate",
                        f"{base}.name",
                        f"Actor name {actor.name!r} is already in use.",
                        actor,
                    )
                )
            else:
                names[actor.name] = actor

            issues.extend(self._mobility_issues(actor, scenario, base))
            issues.extend(self._orientation_issues(actor, scenario, base))
            if actor.role is ActorRole.TARGET:
                issues.extend(self._target_issues(actor, base))
            elif actor.target is not None:
                issues.append(
                    self._issue(
                        "actor.target.unexpected",
                        f"{base}.target",
                        "Only target-role actors may have a target asset.",
                        actor,
                    )
                )
        group_paths = self._group_paths(scenario)
        group_names: dict[str, AuthoringGroup] = {}
        group_ids: set[UUID] = set()
        member_counts = {group.id: 0 for group in scenario.groups}
        member_roles: dict[UUID, set[ActorRole]] = {group.id: set() for group in scenario.groups}
        for actor in scenario.actors:
            mobility = actor.mobility
            if not isinstance(mobility, GroupMemberMobilitySpec):
                continue
            group = _group_reference(scenario, mobility.group)
            if group is not None:
                member_counts[group.id] += 1
                member_roles[group.id].add(actor.role)

        for group in scenario.groups:
            base = group_paths[group.id]
            if group.id in group_ids:
                issues.append(
                    self._issue(
                        "group.id.duplicate",
                        f"{base}.id",
                        f"Group UUID {group.id} is duplicated.",
                        group=group,
                    )
                )
            group_ids.add(group.id)
            if not group.name.strip():
                issues.append(
                    self._issue(
                        "group.name.required",
                        f"{base}.name",
                        "Group name is required.",
                        group=group,
                    )
                )
            elif group.name in group_names:
                issues.append(
                    self._issue(
                        "group.name.duplicate",
                        f"{base}.name",
                        f"Group name {group.name!r} is already in use.",
                        group=group,
                    )
                )
            else:
                group_names[group.name] = group
            if member_counts[group.id] < 2:
                issues.append(
                    self._issue(
                        "group.members.minimum",
                        base,
                        "A group requires at least two actor members.",
                        group=group,
                    )
                )
            if (
                isinstance(group.mobility, MeshSequenceMobilitySpec)
                and member_roles[group.id]
                and member_roles[group.id] != {ActorRole.TARGET}
            ):
                issues.append(
                    self._issue(
                        "group.mobility.target_only",
                        f"{base}.mobility.type",
                        "Mesh-sequence group mobility requires only target members.",
                        group=group,
                    )
                )
        return issues

    def _mobility_issues(
        self,
        actor: AuthoringActor,
        scenario: AuthoringScenario,
        base: str,
    ) -> list[ValidationIssue]:
        mobility = actor.mobility
        if isinstance(mobility, MeshSequenceMobilitySpec) and actor.role is not ActorRole.TARGET:
            return [
                self._issue(
                    "mobility.mesh_sequence.target_only",
                    f"{base}.mobility.type",
                    "Mesh-sequence mobility is valid only for target actors.",
                    actor,
                )
            ]
        if isinstance(mobility, GroupMemberMobilitySpec):
            group = _group_reference(scenario, mobility.group)
            if group is None:
                return [
                    self._issue(
                        "mobility.group_member.missing_group",
                        f"{base}.mobility.group",
                        f"Group UUID {mobility.group} does not exist.",
                        actor,
                    )
                ]
        return []

    def _orientation_issues(
        self,
        actor: AuthoringActor,
        scenario: AuthoringScenario,
        base: str,
    ) -> list[ValidationIssue]:
        orientation = actor.orientation
        if (
            isinstance(orientation, KeyframesOrientationSpec)
            and orientation.keyframes
            and orientation.keyframes[-1].time_s > scenario.timeline.duration_s
        ):
            return [
                self._issue(
                    "orientation.keyframes.after_timeline",
                    (f"{base}.orientation.keyframes." f"{len(orientation.keyframes) - 1}.time_s"),
                    "Orientation keyframes must not exceed the timeline duration.",
                    actor,
                )
            ]
        if not isinstance(orientation, LookAtOrientationSpec) or orientation.actor is None:
            return []
        target = _actor_reference(scenario, orientation.actor)
        if target is None:
            return [
                self._issue(
                    "orientation.look_at.missing_target",
                    f"{base}.orientation.actor",
                    f"Look-at target UUID {orientation.actor} does not exist.",
                    actor,
                )
            ]
        if target.id == actor.id:
            return [
                self._issue(
                    "orientation.look_at.self",
                    f"{base}.orientation.actor",
                    "An actor cannot look at itself.",
                    actor,
                )
            ]
        return []

    def _prepared_mobility_orientation_issues(
        self,
        scenario: AuthoringScenario,
        scenario_root: Path,
    ) -> list[ValidationIssue]:
        """Validate derived orientation requirements against sampled motion."""

        if not any(
            isinstance(actor.orientation, AlignMotionOrientationSpec) for actor in scenario.actors
        ):
            return []
        try:
            timeline = Timeline(
                int(scenario.timeline.steps),
                float(scenario.timeline.duration_s),
            )
        except ValueError:
            return []

        paths = self._actor_paths(scenario)
        group_paths = self._group_paths(scenario)
        prepared_groups: dict[UUID, PreparedMobility] = {}
        issues: list[ValidationIssue] = []
        for actor in scenario.actors:
            if not isinstance(actor.orientation, AlignMotionOrientationSpec):
                continue
            try:
                if isinstance(actor.mobility, GroupMemberMobilitySpec):
                    group = _group_reference(scenario, actor.mobility.group)
                    if group is None:
                        continue
                    prepared = prepared_groups.get(group.id)
                    if prepared is None:
                        prepared = _prepare_authoring_mobility(
                            group.mobility,
                            timeline,
                            scenario_root=scenario_root,
                            resources=scenario.resources,
                            dependencies=scenario.dependencies,
                            path=f"{group_paths[group.id]}.mobility",
                        )
                        prepared_groups[group.id] = prepared
                else:
                    prepared = _prepare_authoring_mobility(
                        actor.mobility,
                        timeline,
                        scenario_root=scenario_root,
                        resources=scenario.resources,
                        dependencies=scenario.dependencies,
                        path=f"{paths[actor.id]}.mobility",
                    )
            except Exception:
                # Mobility/schema diagnostics own preparation failures. This
                # pass only adds evidence when valid mobility has no tangent.
                continue
            if prepared.has_physical_velocity:
                continue
            issues.append(
                self._issue(
                    "orientation.align_motion.no_physical_motion",
                    f"{paths[actor.id]}.orientation",
                    "Align-with-motion orientation requires a prepared trajectory "
                    "with physical movement.",
                    actor,
                )
            )
        return issues

    def _target_issues(self, actor: AuthoringActor, base: str) -> list[ValidationIssue]:
        target = actor.target
        if target is None:
            return [
                self._issue(
                    "target.asset.required",
                    f"{base}.asset",
                    "Choose a catalog target asset.",
                    actor,
                )
            ]
        issues: list[ValidationIssue] = []
        if target.source == "catalog":
            normalized = target.mesh_directory.replace("\\", "/")
            path = PurePosixPath(normalized)
            normalized_asset_id = str(PurePosixPath(target.asset_id.replace("\\", "/")))
            expected_directory = f"libraries/targets/{normalized_asset_id}"
            if (
                not normalized.startswith("libraries/targets/")
                or ".." in path.parts
                or path.is_absolute()
            ):
                issues.append(
                    self._issue(
                        "target.asset.not_catalog",
                        f"{base}.asset.id",
                        "Catalog target assets must stay inside libraries/targets.",
                        actor,
                    )
                )
            elif normalized != expected_directory:
                issues.append(
                    self._issue(
                        "target.asset.catalog_mismatch",
                        f"{base}.asset.id",
                        "Target mesh directory must exactly match its catalog asset id "
                        f"({expected_directory!r}).",
                        actor,
                    )
                )
        if not target.asset_id.strip():
            issues.append(
                self._issue(
                    "target.asset.id_required",
                    f"{base}.asset.id",
                    "Target catalog asset id is required.",
                    actor,
                )
            )
        if not target.mesh_pattern.strip():
            issues.append(
                self._issue(
                    "target.asset.pattern_required",
                    f"{base}.asset.id",
                    "Target mesh pattern is required.",
                    actor,
                )
            )
        if not target.material.strip():
            issues.append(
                self._issue(
                    "target.material.required",
                    f"{base}.asset.material_type",
                    "Target material is required.",
                    actor,
                )
            )
        else:
            try:
                validate_target_material_type(target.material)
            except ValueError as exc:
                issues.append(
                    self._issue(
                        "target.material.unsupported",
                        f"{base}.asset.material_type",
                        str(exc),
                        actor,
                    )
                )
        if not math.isfinite(target.scale) or target.scale <= 0.0:
            issues.append(
                self._issue(
                    "target.scale.positive",
                    f"{base}.asset.scale",
                    "Target scale must be finite and positive.",
                    actor,
                )
            )
        return issues

    def _asset_issues(
        self,
        scenario: AuthoringScenario,
        scenario_root: Path,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        scene = scenario.scene
        if scene is not None and scene.id.strip():
            try:
                scene_path = self._resolve_scene_path(
                    scene,
                    scenario_root,
                    scenario.dependencies,
                )
            except Exception as exc:
                issues.append(
                    self._issue(
                        "scene.asset.resolve_failed",
                        "scene.id",
                        f"Scene asset could not be resolved: {exc}",
                    )
                )
                scene_path = None
            if scene.source == "local" and scene_path is not None:
                if scene_path.suffix.lower() != ".xml":
                    issues.append(
                        self._issue(
                            "scene.asset.not_xml",
                            "scene.id",
                            "Local scenes must select an XML file.",
                        )
                    )
            elif scene.source == "library" and scene_path is not None:
                library_root = (self.project_root / "libraries" / "scenes").resolve()
                if not scene_path.is_relative_to(library_root):
                    issues.append(
                        self._issue(
                            "scene.asset.not_library",
                            "scene.id",
                            "Library scenes must stay inside the ORCHAV scene catalog.",
                        )
                    )
                    scene_path = None
            if scene_path is None or not scene_path.is_file():
                issues.append(
                    self._issue(
                        "scene.asset.unresolved",
                        "scene.id",
                        f"Scene asset could not be resolved: {scene.id}",
                    )
                )

        paths = self._actor_paths(scenario)
        group_paths = self._group_paths(scenario)

        def check_mobility_resource(
            mobility: Mobility,
            base: str,
            *,
            actor: AuthoringActor | None = None,
            group: AuthoringGroup | None = None,
        ) -> None:
            details = _mobility_resource_field(mobility)
            if details is None:
                return
            field, _raw_path, _kind = details
            try:
                _resolve_authoring_mobility_resource(
                    mobility,
                    scenario_root=scenario_root,
                    resources=scenario.resources,
                    dependencies=scenario.dependencies,
                )
            except _MobilityResourceProblem as exc:
                issues.append(
                    self._issue(
                        exc.code,
                        f"{base}.mobility.{field}",
                        str(exc),
                        actor,
                        group,
                    )
                )

        for actor in scenario.actors:
            check_mobility_resource(
                actor.mobility,
                paths[actor.id],
                actor=actor,
            )
        for group in scenario.groups:
            check_mobility_resource(
                group.mobility,
                group_paths[group.id],
                group=group,
            )

        for actor in scenario.actors:
            target = actor.target
            if actor.role is not ActorRole.TARGET or target is None:
                continue
            base = paths[actor.id]
            pattern_path = PurePosixPath(target.mesh_pattern.replace("\\", "/"))
            if pattern_path.is_absolute() or ".." in pattern_path.parts:
                issues.append(
                    self._issue(
                        "target.asset.pattern_not_catalog",
                        f"{base}.asset.id",
                        "Target mesh patterns must stay inside the selected catalog asset.",
                        actor,
                    )
                )
                continue
            try:
                matches = self._target_mesh_paths(
                    actor,
                    scenario_root,
                    scenario.dependencies,
                )
            except (OSError, ValueError) as exc:
                issues.append(
                    self._issue(
                        "target.asset.pattern_invalid",
                        f"{base}.asset.id",
                        f"Target mesh pattern is invalid: {exc}",
                        actor,
                    )
                )
                continue
            if not matches:
                issues.append(
                    self._issue(
                        "target.asset.empty",
                        f"{base}.asset.id",
                        f"No target meshes match {target.mesh_pattern!r}.",
                        actor,
                    )
                )
        return issues

    def _dependency_issues(
        self,
        scenario: AuthoringScenario,
    ) -> list[ValidationIssue]:
        """Report portability warnings and exact copy/generation blockers."""

        issues: list[ValidationIssue] = []
        destinations: dict[str, Path] = {}
        for dependency in scenario.dependencies:
            if dependency.problem:
                issues.append(
                    self._issue(
                        "dependency.unsafe",
                        dependency.origin_path,
                        dependency.problem,
                    )
                )
                continue
            if not dependency.source_path.exists():
                issues.append(
                    self._issue(
                        "dependency.missing",
                        dependency.origin_path,
                        f"Scenario dependency was not found: {dependency.source_path}",
                    )
                )
                continue
            if dependency.external:
                issues.append(
                    self._issue(
                        "dependency.absolute_external",
                        dependency.origin_path,
                        "Absolute external dependency is preserved and was not copied: "
                        f"{dependency.source_path}",
                        severity=IssueSeverity.WARNING,
                    )
                )
                continue
            previous = destinations.get(dependency.relative_path)
            if previous is not None and previous != dependency.source_path:
                issues.append(
                    self._issue(
                        "dependency.collision",
                        dependency.origin_path,
                        f"Dependency destination {dependency.relative_path!r} is also "
                        f"claimed by {previous}.",
                    )
                )
            destinations[dependency.relative_path] = dependency.source_path
        return issues

    def _generation_issues(
        self,
        mapping: Mapping[str, Any],
    ) -> list[ValidationIssue]:
        """Return Builder limits that must fail before subprocess launch."""

        issues: list[ValidationIssue] = []
        data = mapping.get("data")
        data_mapping = data if isinstance(data, Mapping) else {}
        if data_mapping.get("mode", "files") != "files":
            issues.append(
                self._issue(
                    "generation.data.mode",
                    "data.mode",
                    "Builder Generate supports only file-mode HDF5 output; "
                    "use orchav-generator from the CLI.",
                )
            )
        files = data_mapping.get("files")
        files_mapping = files if isinstance(files, Mapping) else {}
        if str(files_mapping.get("format") or "h5").lower() not in {"h5", "hdf5"}:
            issues.append(
                self._issue(
                    "generation.data.format",
                    "data.files.format",
                    "Builder Generate supports only HDF5 frames; use "
                    "orchav-generator from the CLI.",
                )
            )
        frame_directory = str(files_mapping.get("directory") or FIXED_FRAME_DIRECTORY).strip()
        if frame_directory != FIXED_FRAME_DIRECTORY:
            issues.append(
                self._issue(
                    "generation.frames.unsafe",
                    "data.files.directory",
                    "Builder Generate publishes only to <scenario>/frames; "
                    "data.files.directory is a read-only frame selection.",
                )
            )

        raytracing = mapping.get("raytracing")
        raytracing_mapping = raytracing if isinstance(raytracing, Mapping) else {}
        coverage = mapping.get("coverage")
        coverage_mapping = coverage if isinstance(coverage, Mapping) else {}
        coverage_save = coverage_mapping.get("save")
        coverage_save_mapping = coverage_save if isinstance(coverage_save, Mapping) else {}
        coverage_data = coverage_save_mapping.get("data")
        coverage_data_mapping = coverage_data if isinstance(coverage_data, Mapping) else {}
        raytracing_writes_frames = bool(raytracing_mapping.get("enabled", False))
        coverage_writes_frames = bool(coverage_mapping.get("enabled", False)) and bool(
            coverage_data_mapping.get("enabled", True)
        )
        if not raytracing_writes_frames and not coverage_writes_frames:
            issues.append(
                self._issue(
                    "generation.frames.not_requested",
                    "raytracing.enabled",
                    "Builder Generate requires a frame-producing ray-tracing or "
                    "coverage run. Use orchav-generator for summary-only or other "
                    "derived-only workflows.",
                )
            )
        return issues

    @staticmethod
    def _mobility_resource_paths(
        scenario: AuthoringScenario,
    ) -> tuple[str, ...]:
        paths: list[str] = []
        mobilities = [
            *(actor.mobility for actor in scenario.actors),
            *(group.mobility for group in scenario.groups),
        ]
        for mobility in mobilities:
            if isinstance(mobility, MeshSequenceMobilitySpec):
                paths.append(mobility.positions_path)
            elif isinstance(mobility, NetworkRouteMobilitySpec):
                paths.append(mobility.graph_path or "street_network.graphml")
        return tuple(dict.fromkeys(paths))

    @staticmethod
    def _stage_validation_assets(
        scenario: AuthoringScenario,
        scenario_root: Path,
        validation_root: Path,
    ) -> None:
        """Copy semantic dependencies needed by the isolated validation YAML."""

        copied: set[Path] = set()
        for dependency in scenario.dependencies:
            if dependency.external or dependency.problem or not dependency.source_path.is_file():
                continue
            destination = (validation_root / Path(dependency.relative_path)).resolve()
            if not destination.is_relative_to(validation_root):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependency.source_path, destination)
            copied.add(destination)

        mobilities = (
            *(actor.mobility for actor in scenario.actors),
            *(group.mobility for group in scenario.groups),
        )
        for mobility in mobilities:
            resolved = _resolve_authoring_mobility_resource(
                mobility,
                scenario_root=scenario_root,
                resources=scenario.resources,
                dependencies=scenario.dependencies,
            )
            if resolved is None:
                continue
            _field_name, relative, source = resolved
            destination = (
                validation_root / Path(relative)
                if not Path(relative).is_absolute()
                else validation_root / "resources" / source.name
            )
            if destination.resolve() in copied:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _validation_yaml_text(
        scenario: AuthoringScenario,
        yaml_text: str,
        scenario_root: Path,
    ) -> str:
        """Return merged YAML; relative inputs are copied into validation root."""

        return yaml_text

    def _resolve_scene_path(
        self,
        scene: SceneReference,
        scenario_root: Path,
        dependencies: tuple[ScenarioDependency, ...] = (),
    ) -> Path | None:
        """Resolve the exact selected scene source/id pair without loading it."""
        if scene.source == "local":
            if not Path(scene.id).is_absolute():
                normalized = str(PurePosixPath(scene.id.replace("\\", "/")))
                dependency = next(
                    (
                        item
                        for item in dependencies
                        if item.kind == "scene_xml" and item.relative_path == normalized
                    ),
                    None,
                )
                if dependency is not None:
                    return dependency.source_path
            return (scenario_root / scene.id).resolve()
        if scene.source == "library":
            return (self.project_root / "libraries" / "scenes" / scene.id).resolve()
        if scene.source == "sionna":
            resolved = resolve_sionna_scene_xml(scene.id)
            return Path(resolved).resolve() if resolved is not None else None
        if scene.source == "osm":
            dependency = next(
                (
                    item
                    for item in dependencies
                    if item.relative_path == "scene.xml" and item.kind == "osm_artifact"
                ),
                None,
            )
            if dependency is not None:
                return dependency.source_path
            return (scenario_root / "scene.xml").resolve()
        return None

    def _target_mesh_paths(
        self,
        actor: AuthoringActor,
        scenario_root: Path,
        dependencies: tuple[ScenarioDependency, ...] = (),
    ) -> tuple[Path, ...]:
        """Return generator-equivalent deterministic mesh playback order."""
        target = actor.target
        if target is None:
            return ()
        if target.source == "file":
            candidate = Path(str(target.path or target.asset_id))
            if not candidate.is_absolute():
                normalized = str(PurePosixPath(str(candidate).replace("\\", "/")))
                dependency = next(
                    (
                        item
                        for item in dependencies
                        if item.relative_path == normalized
                        and item.kind in {"target_file", "target_directory_file"}
                    ),
                    None,
                )
                if dependency is not None:
                    return (dependency.source_path,) if dependency.source_path.is_file() else ()
            path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (scenario_root / candidate).resolve()
            )
            return (path,) if path.is_file() else ()
        normalized = target.mesh_directory.replace("\\", "/")
        dependency_matches = tuple(
            item.source_path
            for item in dependencies
            if item.kind == "target_directory_file"
            and PurePosixPath(item.relative_path).match(
                f"{normalized.rstrip('/')}/{target.mesh_pattern}"
            )
            and item.source_path.is_file()
        )
        if dependency_matches:
            matches = tuple(sorted(dependency_matches, key=str))
            return matches[target.start_index :: target.frame_stride]
        if normalized.startswith("libraries/"):
            directory = (self.project_root / PurePosixPath(normalized)).resolve()
        else:
            directory = (scenario_root / target.mesh_directory).resolve()
        matches = tuple(
            sorted(
                (path.resolve() for path in directory.glob(target.mesh_pattern) if path.is_file()),
                key=str,
            )
        )
        if target.source == "directory":
            return matches[target.start_index :: target.frame_stride]
        return matches

    def _compile_scene_preview_assets(
        self,
        scenario: AuthoringScenario,
        scenario_root: Path,
    ) -> tuple[
        str | None,
        tuple[ScenePreviewAsset, ...],
        list[ValidationIssue],
    ]:
        """Compile a selected valid scene even while the document is incomplete."""
        issues: list[ValidationIssue] = []
        scene_assets: tuple[ScenePreviewAsset, ...] = ()
        resolved_scene_path: str | None = None

        scene = scenario.scene
        if scene is not None:
            try:
                scene_path = self._resolve_scene_path(
                    scene,
                    scenario_root,
                    scenario.dependencies,
                )
                if scene_path is None:
                    raise FileNotFoundError(scene.id)
                resolved_scene_path = str(scene_path)
                scene_assets = self._preview_assets.build_scene_assets(scene, scene_path)
            except Exception as exc:
                resolved_scene_path = None
                issues.append(
                    self._issue(
                        "scene.asset.load_failed",
                        "scene.id",
                        f"Scene preview asset loading failed: {exc}",
                    )
                )
        return resolved_scene_path, scene_assets, issues

    def _compile_target_preview_assets(
        self,
        scenario: AuthoringScenario,
        scenario_root: Path,
        samples: Mapping[UUID, ActorSamples],
        *,
        report_missing_samples: bool,
    ) -> tuple[
        dict[UUID, tuple[TargetPreviewAsset, ...]],
        list[ValidationIssue],
    ]:
        """Compile target frames after generator samples are authoritative."""
        issues: list[ValidationIssue] = []
        target_assets: dict[UUID, tuple[TargetPreviewAsset, ...]] = {}

        paths = self._actor_paths(scenario)
        for actor in scenario.actors:
            if actor.role is not ActorRole.TARGET:
                continue
            base = paths[actor.id]
            prepared = samples.get(actor.id)
            if prepared is None:
                if not report_missing_samples:
                    continue
                issues.append(
                    self._issue(
                        "target.preview.samples_missing",
                        base,
                        "Generator-prepared target samples are unavailable.",
                        actor,
                    )
                )
                continue
            try:
                mesh_paths = self._target_mesh_paths(
                    actor,
                    scenario_root,
                    scenario.dependencies,
                )
                target_assets[actor.id] = self._preview_assets.build_target_assets(
                    actor,
                    mesh_paths,
                    prepared,
                )
            except Exception as exc:
                issues.append(
                    self._issue(
                        "target.asset.load_failed",
                        f"{base}.asset.id",
                        f"Target preview asset loading failed: {exc}",
                        actor,
                    )
                )
        return target_assets, issues

    def _schema_issues(self, mapping: dict[str, Any]) -> list[ValidationIssue]:
        try:
            validate_scenario_data(mapping)
        except ValueError as exc:
            message = str(exc)
            path = "scenario"
            for line in message.splitlines():
                marker = "- '"
                if marker in line:
                    path = line.split(marker, 1)[1].split("'", 1)[0]
                    break
                marker = "Unknown key '"
                if marker in line:
                    path = line.split(marker, 1)[1].split("'", 1)[0]
                    break
            return [self._issue("schema.invalid", path, message)]
        return []

    def _prepare_partial_samples(
        self,
        scenario: AuthoringScenario,
        issues: list[ValidationIssue],
        scenario_root: Path,
    ) -> dict[UUID, ActorSamples]:
        """Prepare valid dependency components while the document cannot save.

        Missing scenes or radio roles are save blockers, not trajectory inputs.
        Actor-local errors remove that actor, and look-at owners are removed
        whenever their referenced actor is unavailable. Each remaining
        reference component is evaluated by the canonical pose kernel.
        """

        if any(
            issue.severity is IssueSeverity.ERROR and issue.code.startswith("timeline.")
            for issue in issues
        ):
            return {}

        blocked_ids = {
            issue.actor_id
            for issue in issues
            if issue.severity is IssueSeverity.ERROR and issue.actor_id is not None
        }
        blocked_group_ids = {
            issue.group_id
            for issue in issues
            if issue.severity is IssueSeverity.ERROR
            and issue.group_id is not None
            and issue.code != "group.members.minimum"
        }
        blocked_ids.update(
            actor.id
            for actor in scenario.actors
            if isinstance(actor.mobility, GroupMemberMobilitySpec)
            and (
                (group := _group_reference(scenario, actor.mobility.group)) is not None
                and group.id in blocked_group_ids
            )
        )
        eligible_ids = {actor.id for actor in scenario.actors} - blocked_ids

        # Reference owners are not sampled with invented stand-ins. Propagate
        # dependency removal so a chain of look-at references stays atomic.
        while True:
            dependent_ids = {
                actor.id
                for actor in scenario.actors
                if actor.id in eligible_ids
                and isinstance(actor.orientation, LookAtOrientationSpec)
                and actor.orientation.actor is not None
                and _actor_reference_id(scenario, actor.orientation.actor) not in eligible_ids
            }
            if not dependent_ids:
                break
            eligible_ids.difference_update(dependent_ids)

        if not eligible_ids:
            return {}

        neighbors: dict[UUID, set[UUID]] = {actor_id: set() for actor_id in eligible_ids}
        for actor in scenario.actors:
            if actor.id not in eligible_ids:
                continue
            orientation = actor.orientation
            if isinstance(orientation, LookAtOrientationSpec) and orientation.actor is not None:
                target_id = _actor_reference_id(scenario, orientation.actor)
                if target_id in eligible_ids:
                    neighbors[actor.id].add(target_id)
                    neighbors[target_id].add(actor.id)

        for group in scenario.groups:
            members = [
                actor.id
                for actor in scenario.actors
                if actor.id in eligible_ids
                and isinstance(actor.mobility, GroupMemberMobilitySpec)
                and _group_reference(scenario, actor.mobility.group) == group
            ]
            for member in members:
                neighbors[member].update(other for other in members if other != member)

        components: list[set[UUID]] = []
        remaining = set(eligible_ids)
        for actor in scenario.actors:
            if actor.id not in remaining:
                continue
            component: set[UUID] = set()
            pending = [actor.id]
            while pending:
                actor_id = pending.pop()
                if actor_id not in remaining:
                    continue
                remaining.remove(actor_id)
                component.add(actor_id)
                pending.extend(neighbors[actor_id] - component)
            components.append(component)

        samples: dict[UUID, ActorSamples] = {}
        for component in components:
            component_scenario = replace(
                scenario,
                actors=tuple(actor for actor in scenario.actors if actor.id in component),
                groups=tuple(
                    group
                    for group in scenario.groups
                    if any(
                        actor.id in component
                        and isinstance(actor.mobility, GroupMemberMobilitySpec)
                        and _group_reference(scenario, actor.mobility.group) == group
                        for actor in scenario.actors
                    )
                ),
            )
            mapping = canonical_scenario_mapping(component_scenario)
            try:
                _actor_runtime, component_samples = self._construct_actors_and_prepare(
                    component_scenario,
                    mapping,
                    scenario_root,
                )
            # Partial preview is best-effort evidence. Full validation issues
            # remain the authoritative save/generation diagnostics, and one
            # generator-level component failure must not suppress its peers.
            except Exception:
                continue
            samples.update(component_samples)

        # Geometry-dependent warnings use the same prepared evidence in partial
        # and complete documents. Only error-severity issues suppress samples.
        prepared_issues = self._prepared_orientation_issues(scenario, samples)
        issues.extend(prepared_issues)
        for issue in prepared_issues:
            if issue.severity is IssueSeverity.ERROR and issue.actor_id is not None:
                samples.pop(issue.actor_id, None)
        return samples

    def _prepare_partial_group_samples(
        self,
        scenario: AuthoringScenario,
        issues: list[ValidationIssue],
        scenario_root: Path,
    ) -> dict[UUID, GroupSamples]:
        """Prepare self-contained group paths despite unrelated save blockers."""

        if any(
            issue.severity is IssueSeverity.ERROR and issue.code.startswith("timeline.")
            for issue in issues
        ):
            return {}
        blocked_group_ids = {
            issue.group_id
            for issue in issues
            if issue.severity is IssueSeverity.ERROR
            and issue.group_id is not None
            and issue.code != "group.members.minimum"
        }
        timeline = Timeline(
            int(scenario.timeline.steps),
            float(scenario.timeline.duration_s),
        )
        samples: dict[UUID, GroupSamples] = {}
        for group in scenario.groups:
            if group.id in blocked_group_ids:
                continue
            try:
                prepared = _prepare_authoring_mobility(
                    group.mobility,
                    timeline,
                    scenario_root=scenario_root,
                    resources=scenario.resources,
                    dependencies=scenario.dependencies,
                    path=f"{self._group_paths(scenario)[group.id]}.mobility",
                )
            except Exception:
                continue
            samples[group.id] = GroupSamples(
                tuple(vector3(point) for point in prepared.positions_m),
                prepared,
                timeline,
            )
        return samples

    def _construct_and_prepare(
        self,
        scenario: AuthoringScenario,
        yaml_text: str,
        scenario_root: Path,
    ) -> tuple[
        CompiledRuntime,
        dict[UUID, ActorSamples],
        dict[UUID, GroupSamples],
    ]:
        def construct(
            validation_yaml: Path,
            loader_scenario_path: Path,
        ) -> tuple[
            CompiledRuntime,
            dict[UUID, ActorSamples],
            dict[UUID, GroupSamples],
        ]:
            scenario_config = load_scenario_configuration(
                loader_scenario_path,
                project_root=self.project_root,
                yaml_path=validation_yaml,
            )
            simulation_config = load_simulation_config(scenario_config)
            simulation_config.validate()
            actor_runtime = prepare_actor_runtime(scenario_config)
            samples = self._samples_from_prepared(scenario, actor_runtime.prepared)
            group_samples = self._group_samples_from_prepared(
                scenario,
                actor_runtime.prepared,
            )

            runtime = CompiledRuntime(
                scenario_configuration=scenario_config,
                simulation_config=simulation_config,
                transmitters=tuple(actor_runtime.transmitters),
                receivers=tuple(actor_runtime.receivers),
                targets=tuple(actor_runtime.targets),
            )
            return runtime, samples, group_samples

        scene = scenario.scene
        needs_staging = (
            bool(any(not dependency.external for dependency in scenario.dependencies))
            or bool(self._mobility_resource_paths(scenario))
            or bool(
                scene is not None and scene.source == "local" and not Path(scene.id).is_absolute()
            )
        )
        validation_parent = scenario_root if scenario_root.is_dir() else self.project_root
        if needs_staging:
            with tempfile.TemporaryDirectory(
                prefix=".orchav-authoring-validation-",
                dir=validation_parent,
            ) as temporary_name:
                validation_root = Path(temporary_name)
                self._stage_validation_assets(
                    scenario,
                    scenario_root,
                    validation_root,
                )
                validation_yaml = validation_root / "scenario.yaml"
                validation_yaml.write_text(
                    self._validation_yaml_text(
                        scenario,
                        yaml_text,
                        scenario_root,
                    ),
                    encoding="utf-8",
                )
                return construct(validation_yaml, validation_yaml)

        descriptor, validation_name = tempfile.mkstemp(
            prefix=".orchav-authoring-validation-",
            suffix=".yaml",
            dir=validation_parent,
            text=True,
        )
        validation_yaml = Path(validation_name)
        os.close(descriptor)
        try:
            validation_yaml.write_text(yaml_text, encoding="utf-8")
            loader_scenario_path = (
                scenario_root if scenario_root.is_dir() else scenario_root / "scenario.yaml"
            )
            return construct(validation_yaml, loader_scenario_path)
        finally:
            validation_yaml.unlink(missing_ok=True)

    def _construct_actors_and_prepare(
        self,
        scenario: AuthoringScenario,
        mapping: dict[str, Any],
        scenario_root: Path,
    ) -> tuple[Any, dict[UUID, ActorSamples]]:
        """Prepare actor poses and associate them with document-local UUIDs."""

        scenario_model = validate_scenario_data(mapping)
        preview_configuration = SimpleNamespace(
            actors=scenario_model.actors,
            root=scenario_root,
            targets_dir=self.project_root / "libraries" / "targets",
        )
        prepared = prepare_scenario(
            scenario_model,
            base_dir=scenario_root,
            asset_alignments=target_asset_alignments(preview_configuration),
        )
        return prepared, self._samples_from_prepared(scenario, prepared)

    @staticmethod
    def _samples_from_prepared(
        scenario: AuthoringScenario,
        prepared: Any,
    ) -> dict[UUID, ActorSamples]:
        """Associate serialized actor names with document-local UUIDs."""

        samples: dict[UUID, ActorSamples] = {}
        for authored_actor in scenario.actors:
            prepared_actor = prepared.actor(authored_actor.name)
            samples[authored_actor.id] = ActorSamples(
                tuple(vector3(point) for point in prepared_actor.positions_m),
                tuple(vector3(angles) for angles in prepared_actor.orientation.euler_deg),
                tuple(vector3(velocity) for velocity in prepared_actor.mobility.velocities_mps),
                tuple(vector3(forward) for forward in prepared_actor.mobility.forward_vectors),
                bool(prepared_actor.mobility.has_physical_velocity),
            )
        return samples

    @staticmethod
    def _group_samples_from_prepared(
        scenario: AuthoringScenario,
        prepared: Any,
    ) -> dict[UUID, GroupSamples]:
        by_name = {group.name: group for group in prepared.groups}
        timeline = Timeline(
            int(scenario.timeline.steps),
            float(scenario.timeline.duration_s),
        )
        return {
            authored_group.id: GroupSamples(
                tuple(
                    vector3(point) for point in by_name[authored_group.name].mobility.positions_m
                ),
                by_name[authored_group.name].mobility,
                timeline,
            )
            for authored_group in scenario.groups
            if authored_group.name in by_name
        }

    def _runtime_issue(self, scenario: AuthoringScenario, exc: Exception) -> ValidationIssue:
        message = str(exc) or exc.__class__.__name__
        for actor in scenario.actors:
            if actor.name and actor.name in message:
                base = self._actor_paths(scenario)[actor.id]
                return self._issue(
                    "generator.actor_invalid",
                    base,
                    f"Generator construction failed: {message}",
                    actor,
                )
        return self._issue(
            "generator.invalid",
            "scenario",
            f"Generator construction failed: {message}",
        )

    def _prepared_orientation_issues(
        self,
        scenario: AuthoringScenario,
        samples: Mapping[UUID, ActorSamples],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        paths = self._actor_paths(scenario)
        for actor in scenario.actors:
            orientation = actor.orientation
            if not isinstance(orientation, LookAtOrientationSpec):
                continue
            own = samples.get(actor.id)
            if own is None:
                continue
            if orientation.actor is not None:
                target_id = _actor_reference_id(scenario, orientation.actor)
                target = samples.get(target_id) if target_id is not None else None
                if target is None:
                    continue
                target_positions = target.positions
                target_path = "actor"
            else:
                assert orientation.point_m is not None
                target_positions = (vector3(orientation.point_m),) * len(own.positions)
                target_path = "point_m"
            if any(
                math.dist(own_position, target_position) <= 1e-9
                for own_position, target_position in zip(
                    own.positions,
                    target_positions,
                    strict=True,
                )
            ):
                issues.append(
                    self._issue(
                        "orientation.look_at.coincident",
                        f"{paths[actor.id]}.orientation.{target_path}",
                        "Look-at owner and target coincide at one or more timeline "
                        "steps; canonical hold semantics keep the previous "
                        "orientation (or the authored offsets at the first step).",
                        actor,
                        severity=IssueSeverity.WARNING,
                    )
                )
        return issues
