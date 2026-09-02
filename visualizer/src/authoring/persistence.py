"""Schema-aware loading, scenario copying, and atomic scenario persistence."""

from __future__ import annotations

import filecmp
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID, uuid4

import yaml
from pydantic import TypeAdapter, ValidationError

from generator.core.configuration.defaults import (
    DEFAULT_EXPORT_PATH_METRICS,
    DEFAULT_QUALITY_PRESET,
)
from shared.scenarios.actors import (
    SCENARIO_SCHEMA_VERSION,
    GroupSpec,
    MobilitySpec,
    OrientationSpec,
    StandaloneMobilitySpec,
    TargetAssetSpec,
)
from shared.scenarios.yaml import validate_scenario_data

from .compiler import (
    DOCUMENT_VERSION,
    IssueSeverity,
    ScenarioCompiler,
    ValidationIssue,
)
from .dependencies import ScenarioDependencyCollector
from .document import DocumentOwnership, ScenarioDocument
from .domain import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    FacetCapability,
    QualityPreset,
    ResourceKind,
    ScenarioSourceBinding,
    ScenarioSourceSnapshot,
    SceneReference,
    TargetAsset,
    TimelineSettings,
    canonical_authoring_resource_path,
    resolve_owned_authoring_resource,
)
from .model_capabilities import (
    MOBILITY_CAPABILITIES,
    ORIENTATION_CAPABILITIES,
    mobility_capability,
    orientation_capability,
)
from .undo import UndoStack

_SCHEMA_UNION_TAGS = (
    frozenset(MOBILITY_CAPABILITIES)
    | frozenset(ORIENTATION_CAPABILITIES)
    | frozenset(
        {
            "fit_duration",
            "constant_speed",
            "catalog",
            "file",
            "directory",
        }
    )
)


def _schema_error_path(location: object) -> str:
    """Return a user field path without Pydantic union-branch labels."""

    if not isinstance(location, tuple):
        return str(location)
    return ".".join(str(part) for part in location if str(part) not in _SCHEMA_UNION_TAGS)


class LoadDisposition(str, Enum):
    """Authoring policy decision for a loaded YAML document."""

    OWNED_EDITABLE = "owned_editable"
    READ_ONLY = "read_only"


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Result of checking the YAML surface the builder can round-trip."""

    disposition: LoadDisposition
    issues: tuple[ValidationIssue, ...]

    @property
    def compatible(self) -> bool:
        return self.disposition is not LoadDisposition.READ_ONLY


@dataclass(frozen=True, slots=True)
class AuthoringLoadResult:
    """Loaded source plus an editable document or read-only domain snapshot."""

    disposition: LoadDisposition
    issues: tuple[ValidationIssue, ...]
    document: ScenarioDocument | None
    scenario: AuthoringScenario | None
    raw_mapping: dict[str, Any]
    source_path: Path

    @property
    def editable(self) -> bool:
        return self.disposition is LoadDisposition.OWNED_EDITABLE

    @property
    def read_only(self) -> bool:
        return self.disposition is LoadDisposition.READ_ONLY


class ScenarioSaveError(ValueError):
    """Raised when validation prevents an authoring document save."""

    def __init__(self, issues: tuple[ValidationIssue, ...]):
        self.issues = issues
        summary = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(f"Scenario cannot be saved: {summary}")


class _ScenarioParseError(ValueError):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(message)


def _as_bool(value: Any, path: str, *, default: bool | None = None) -> bool:
    """Parse schema-compatible booleans without Python truthiness surprises."""
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "on", "t", "true", "y", "yes"}:
            return True
        if normalized in {"0", "off", "f", "false", "n", "no"}:
            return False
    raise _ScenarioParseError(path, "A boolean value is required.")


class CompatibilityAnalyzer:
    """Classify complete documents through the shared schema and extensions."""

    def analyze(self, mapping: Mapping[str, Any]) -> CompatibilityReport:
        """Classify editability independently of optional Builder metadata."""
        issues: list[ValidationIssue] = []
        if not isinstance(mapping, Mapping):
            issues.append(self._issue("scenario.mapping", "Scenario YAML must be a mapping."))
            return CompatibilityReport(LoadDisposition.READ_ONLY, tuple(issues))

        schema_version = mapping.get("schema_version")
        if type(schema_version) is not int or schema_version != SCENARIO_SCHEMA_VERSION:
            issues.append(
                self._issue(
                    "schema_version",
                    f"Unsupported schema version: {schema_version!r}.",
                    "compatibility.schema_version",
                )
            )
        else:
            issues.extend(self._shared_schema_issues(mapping))

        visualizer = mapping.get("visualizer")
        if isinstance(visualizer, Mapping):
            marker = visualizer.get("scenario_builder")
            if "scenario_builder" in visualizer and not isinstance(marker, Mapping):
                issues.append(
                    self._issue(
                        "visualizer.scenario_builder",
                        "Scenario Builder metadata must be a mapping.",
                    )
                )
            elif isinstance(marker, Mapping):
                version = marker.get("document_version")
                if type(version) is not int or version != DOCUMENT_VERSION:
                    issues.append(
                        self._issue(
                            "visualizer.scenario_builder.document_version",
                            f"Unsupported Scenario Builder document version: {version!r}.",
                            "compatibility.document_version",
                        )
                    )

        if issues:
            return CompatibilityReport(LoadDisposition.READ_ONLY, tuple(issues))
        return CompatibilityReport(LoadDisposition.OWNED_EDITABLE, ())

    @staticmethod
    def _issue(path: str, message: str, code: str = "compatibility.unsupported") -> ValidationIssue:
        return ValidationIssue(IssueSeverity.ERROR, code, path, message)

    @classmethod
    def _shared_schema_issues(cls, mapping: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        """Return exact-path failures from the shared strict scenario schema."""

        try:
            validate_scenario_data(dict(mapping))
        except ValueError as exc:
            cause = exc.__cause__
            if isinstance(cause, ValidationError):
                return tuple(
                    cls._issue(
                        _schema_error_path(error["loc"]),
                        str(error["msg"]),
                        "compatibility.schema_invalid",
                    )
                    for error in cause.errors()
                )
            return (
                cls._issue(
                    "schema",
                    str(exc),
                    "compatibility.schema_invalid",
                ),
            )
        return ()


def _resolve_yaml_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "scenario.yaml"
    return candidate.resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a YAML mapping: {path}")
    return data


def _as_point(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise _ScenarioParseError(path, "A three-component coordinate is required.")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise _ScenarioParseError(path, "Coordinate values must be numeric.") from exc


def _as_float(value: Any, path: str) -> float:
    """Parse one numeric scalar while retaining an exact compatibility path."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise _ScenarioParseError(path, "A numeric value is required.") from exc


_MOBILITY_ADAPTER = TypeAdapter(MobilitySpec)
_STANDALONE_MOBILITY_ADAPTER = TypeAdapter(StandaloneMobilitySpec)
_ORIENTATION_ADAPTER = TypeAdapter(OrientationSpec)
_TARGET_ASSET_ADAPTER = TypeAdapter(TargetAssetSpec)


def resource_relative_path(source_path: Path | str) -> str:
    """Return a deterministic destination below the scenario resources folder."""

    source = Path(source_path).resolve()
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
    stem = source.stem or "resource"
    return f"resources/{stem}-{digest}{source.suffix}"


def resolve_authoring_resource(
    raw_path: Path | str,
    kind: ResourceKind | str,
    scenario_directory: Path | str,
    registered_resources: Iterable[AuthoringResource] = (),
) -> AuthoringResource:
    """Resolve an external or canonical resource path without losing its source.

    Before the first save, an editor already contains the canonical
    ``resources/...`` value while the file still lives at its registered
    external source. After save or reopen, the same value resolves to the
    scenario-owned file. Supporting both states makes repeated Apply operations
    idempotent.
    """

    raw = str(raw_path)
    path = Path(raw)
    normalized_kind = ResourceKind(kind)
    scenario_root = Path(scenario_directory).resolve()
    canonical_relative = canonical_authoring_resource_path(raw)

    resources_by_path = {resource.relative_path: resource for resource in registered_resources}
    if canonical_relative is not None:
        registered = resources_by_path.get(canonical_relative)
        if registered is not None and registered.kind != normalized_kind:
            raise ValueError(
                f"Resource {canonical_relative!r} is registered as "
                f"{registered.kind.value}, not {normalized_kind.value}."
            )
        owned_source = resolve_owned_authoring_resource(
            scenario_root,
            canonical_relative,
        )
        if registered is not None and registered.source_path.is_file():
            source = registered.source_path
        elif owned_source.is_file():
            source = owned_source
        else:
            missing = registered.source_path if registered is not None else owned_source
            raise ValueError(f"Authoring resource was not found: {missing}")
        return AuthoringResource(normalized_kind, source, canonical_relative)

    source = path.resolve() if path.is_absolute() else (scenario_root / path).resolve()
    if not path.is_absolute() and not source.is_relative_to(scenario_root):
        raise ValueError(f"Mobility resource resolves outside the scenario directory: {raw}")
    if not source.is_file():
        raise ValueError(f"Authoring resource was not found: {source}")
    relative = resource_relative_path(source)
    registered = resources_by_path.get(relative)
    if registered is not None and registered.kind != normalized_kind:
        raise ValueError(
            f"Resource {relative!r} is registered as "
            f"{registered.kind.value}, not {normalized_kind.value}."
        )
    return AuthoringResource(normalized_kind, source, relative)


def _internalize_resource(
    data: dict[str, Any],
    field_name: str,
    kind: ResourceKind,
    field_path: str,
    source_directory: Path | None,
    resources: list[AuthoringResource] | None,
) -> None:
    raw = data.get(field_name)
    if raw is None or resources is None:
        return
    raw_path = Path(str(raw))
    try:
        canonical = canonical_authoring_resource_path(raw_path)
    except ValueError as exc:
        raise _ScenarioParseError(field_path, str(exc)) from exc
    if source_directory is None:
        return
    if canonical is not None:
        try:
            source = resolve_owned_authoring_resource(
                source_directory,
                canonical,
            )
        except ValueError as exc:
            raise _ScenarioParseError(field_path, str(exc)) from exc
        relative = canonical
    else:
        source = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (source_directory / raw_path).resolve()
        )
        if not raw_path.is_absolute() and not source.is_relative_to(source_directory):
            raise _ScenarioParseError(
                field_path,
                f"Mobility resource resolves outside the scenario directory: {raw}",
            )
        relative = resource_relative_path(source)
    data[field_name] = relative
    resource = AuthoringResource(kind, source, relative)
    existing = next(
        (item for item in resources if item.relative_path == relative),
        None,
    )
    if existing is not None and existing.kind != kind:
        raise _ScenarioParseError(
            field_path,
            f"Mobility resource {relative!r} is already used as "
            f"{existing.kind.value}, not {kind.value}.",
        )
    if existing is None:
        resources.append(resource)


def _validation_parse_error(path: str, exc: ValidationError) -> _ScenarioParseError:
    first = exc.errors()[0] if exc.errors() else None
    suffix = _schema_error_path(first["loc"]) if first is not None else ""
    issue_path = f"{path}.{suffix}" if suffix else path
    message = str(first["msg"]) if first is not None else str(exc)
    return _ScenarioParseError(issue_path, message)


def _parse_mobility(
    value: Any,
    path: str,
    group_name_to_id: Mapping[str, UUID],
    source_directory: Path | None = None,
    resources: list[AuthoringResource] | None = None,
):
    if not isinstance(value, Mapping):
        raise _ScenarioParseError(path, "Mobility must be a mapping.")
    data = dict(value)
    if data.get("type") == "network_route":
        _internalize_resource(
            data,
            "graph_path",
            ResourceKind.NETWORK_GRAPH,
            f"{path}.graph_path",
            source_directory,
            resources,
        )
    elif data.get("type") == "mesh_sequence":
        _internalize_resource(
            data,
            "positions_path",
            ResourceKind.POSITION_SEQUENCE,
            f"{path}.positions_path",
            source_directory,
            resources,
        )
    if data.get("type") == "group_member":
        group_name = str(data.get("group", ""))
        group_id = group_name_to_id.get(group_name)
        if group_id is None:
            raise _ScenarioParseError(
                f"{path}.group",
                f"Group was not found: {group_name!r}.",
            )
        data["group"] = str(group_id)
    try:
        return _MOBILITY_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise _validation_parse_error(path, exc) from exc


def _parse_group_mobility(
    value: Any,
    path: str,
    source_directory: Path | None = None,
    resources: list[AuthoringResource] | None = None,
):
    if not isinstance(value, Mapping):
        raise _ScenarioParseError(path, "Group mobility must be a mapping.")
    data = dict(value)
    if data.get("type") == "network_route":
        _internalize_resource(
            data,
            "graph_path",
            ResourceKind.NETWORK_GRAPH,
            f"{path}.graph_path",
            source_directory,
            resources,
        )
    elif data.get("type") == "mesh_sequence":
        _internalize_resource(
            data,
            "positions_path",
            ResourceKind.POSITION_SEQUENCE,
            f"{path}.positions_path",
            source_directory,
            resources,
        )
    try:
        return _STANDALONE_MOBILITY_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise _validation_parse_error(path, exc) from exc


def _parse_orientation(
    value: Any,
    path: str,
    name_to_id: Mapping[str, UUID],
):
    if value is None:
        value = {"type": "fixed"}
    if not isinstance(value, Mapping):
        raise _ScenarioParseError(path, "Orientation must be a mapping.")
    data = dict(value)
    if data.get("type") == "look_at" and data.get("actor") is not None:
        target_name = str(data["actor"])
        target_id = name_to_id.get(target_name)
        if target_id is None:
            raise _ScenarioParseError(
                f"{path}.actor",
                f"Look-at actor was not found: {target_name!r}.",
            )
        data["actor"] = str(target_id)
    try:
        return _ORIENTATION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise _validation_parse_error(path, exc) from exc


def _opaque_name_reference_paths(
    mapping: Mapping[str, Any],
    actor_names: set[str],
) -> dict[str, tuple[str, ...]]:
    """Find exact actor-name values outside Builder-owned actor/group records."""

    found: dict[str, list[str]] = {name: [] for name in actor_names}

    def visit(value: Any, path: tuple[str | int, ...]) -> None:
        if path and path[0] in {"actors", "groups"}:
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, (*path, str(key)))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*path, index))
        elif isinstance(value, str) and value in found:
            found[value].append(".".join(str(part) for part in path))

    visit(mapping, ())
    return {name: tuple(paths) for name, paths in found.items() if paths}


def _facet_capabilities(
    mapping: Mapping[str, Any],
    actors: Iterable[AuthoringActor],
    groups: Iterable[AuthoringGroup],
    specs: Iterable[tuple[ActorRole, Mapping[str, Any], str]],
) -> tuple[FacetCapability, ...]:
    """Project per-facet editability without changing document disposition."""

    actor_values = tuple(actors)
    group_values = tuple(groups)
    source_paths = {str(entry.get("name")): path for _role, entry, path in specs}
    opaque_references = _opaque_name_reference_paths(
        mapping,
        {actor.name for actor in actor_values},
    )
    capabilities: list[FacetCapability] = [
        FacetCapability("scene", "scene"),
    ]
    for actor in actor_values:
        base = source_paths[actor.name]
        opaque_paths = opaque_references.get(actor.name, ())
        capabilities.append(
            FacetCapability(
                "identity",
                f"{base}.name",
                editable=not opaque_paths,
                reason=(
                    "Rename/delete is locked because an unhandled actor reference "
                    f"exists at {', '.join(opaque_paths)}."
                    if opaque_paths
                    else ""
                ),
                subject_id=actor.id,
            )
        )
        mobility = mobility_capability(str(actor.mobility.type))
        capabilities.append(
            FacetCapability(
                "mobility",
                f"{base}.mobility",
                editable=mobility.editable,
                reason=(
                    ""
                    if mobility.editable
                    else f"Mobility model {actor.mobility.type!r} is preserved read-only."
                ),
                subject_id=actor.id,
            )
        )
        orientation = orientation_capability(str(actor.orientation.type))
        capabilities.append(
            FacetCapability(
                "orientation",
                f"{base}.orientation",
                editable=orientation.editable,
                reason=(
                    ""
                    if orientation.editable
                    else f"Orientation model {actor.orientation.type!r} is preserved read-only."
                ),
                subject_id=actor.id,
            )
        )
        if actor.target is not None:
            locator_editable = actor.target.source == "catalog"
            capabilities.append(
                FacetCapability(
                    "target_asset",
                    f"{base}.asset",
                    editable=locator_editable,
                    reason=(
                        ""
                        if locator_editable
                        else (
                            f"Imported {actor.target.source} locator, pattern, indices, "
                            "and stride are preserved read-only; material, scale, and "
                            "applicable mesh animation remain editable."
                        )
                    ),
                    subject_id=actor.id,
                )
            )
    for index, group in enumerate(group_values):
        capability = mobility_capability(str(group.mobility.type))
        capabilities.append(
            FacetCapability(
                "group",
                f"groups.{index}",
                editable=capability.editable,
                reason=(
                    ""
                    if capability.editable
                    else f"Group mobility model {group.mobility.type!r} is preserved read-only."
                ),
                subject_id=group.id,
            )
        )
    return tuple(capabilities)


def scenario_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
) -> AuthoringScenario:
    """Convert a structurally compatible mapping into immutable domain values."""
    document_id = uuid4()
    resource_root = Path(source_directory).resolve() if source_directory is not None else None
    resources: list[AuthoringResource] = []
    group_entries = mapping.get("groups") or []
    if not isinstance(group_entries, list):
        raise _ScenarioParseError("groups", "Groups must be a list.")
    group_name_to_id: dict[str, UUID] = {}
    for index, entry in enumerate(group_entries):
        if not isinstance(entry, Mapping):
            raise _ScenarioParseError(f"groups.{index}", "Group entry must be a mapping.")
        name = entry.get("name")
        if not isinstance(name, str):
            raise _ScenarioParseError(f"groups.{index}.name", "Group name must be a string.")
        group_name_to_id[name] = uuid4()

    authoring_groups: list[AuthoringGroup] = []
    for index, entry in enumerate(group_entries):
        path = f"groups.{index}"
        name = str(entry["name"])
        try:
            parsed_group = GroupSpec.model_validate(dict(entry))
        except ValidationError as exc:
            raise _validation_parse_error(path, exc) from exc
        authoring_groups.append(
            AuthoringGroup(
                id=group_name_to_id[name],
                name=name,
                mobility=_parse_group_mobility(
                    entry.get("mobility"),
                    f"{path}.mobility",
                    resource_root,
                    None,
                ),
                deviation=parsed_group.deviation,
            )
        )

    specs: list[tuple[ActorRole, Mapping[str, Any], str]] = []
    actor_sections = mapping.get("actors") or {}
    for role, section in (
        (ActorRole.TX, "tx"),
        (ActorRole.RX, "rx"),
        (ActorRole.TARGET, "targets"),
    ):
        for index, entry in enumerate(actor_sections.get(section, []) or []):
            if not isinstance(entry, Mapping):
                raise _ScenarioParseError(
                    f"actors.{section}.{index}", "Actor entry must be a mapping."
                )
            specs.append((role, entry, f"actors.{section}.{index}"))

    name_to_id: dict[str, UUID] = {}
    for _, entry, path in specs:
        name = entry.get("name")
        if not isinstance(name, str):
            raise _ScenarioParseError(f"{path}.name", "Actor name must be a string.")
        name_to_id[name] = uuid4()

    timeline_mapping = mapping.get("timeline")
    if not isinstance(timeline_mapping, Mapping):
        raise _ScenarioParseError("timeline", "Timeline settings are required.")
    try:
        steps = int(timeline_mapping["steps"])
        duration_s = float(timeline_mapping["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _ScenarioParseError("timeline", f"Invalid timeline settings: {exc}") from exc

    authoring_actors: list[AuthoringActor] = []
    for role, entry, path in specs:
        name = str(entry["name"])
        target: TargetAsset | None = None
        if role is ActorRole.TARGET:
            asset = entry.get("asset")
            if not isinstance(asset, Mapping):
                raise _ScenarioParseError(f"{path}.asset", "Target asset must be a mapping.")
            try:
                target = TargetAsset.from_spec(_TARGET_ASSET_ADAPTER.validate_python(dict(asset)))
            except (TypeError, ValueError, ValidationError) as exc:
                raise _ScenarioParseError(
                    f"{path}.asset",
                    f"Invalid target asset: {exc}",
                ) from exc
        authoring_actors.append(
            AuthoringActor(
                id=name_to_id[name],
                name=name,
                role=role,
                mobility=_parse_mobility(
                    entry.get("mobility"),
                    f"{path}.mobility",
                    group_name_to_id,
                    resource_root,
                    None,
                ),
                orientation=_parse_orientation(
                    entry.get("orientation"),
                    f"{path}.orientation",
                    name_to_id,
                ),
                target=target,
            )
        )

    scene_mapping = mapping.get("scene")
    if not isinstance(scene_mapping, Mapping):
        raise _ScenarioParseError("scene", "Scene selection is required.")
    scene = SceneReference(str(scene_mapping.get("source", "")), str(scene_mapping.get("id", "")))
    raytracing = mapping.get("raytracing") or {}
    quality = raytracing.get("quality") or {}
    try:
        timeline = TimelineSettings(
            steps=steps,
            duration_s=duration_s,
            quality=QualityPreset(quality.get("preset") or DEFAULT_QUALITY_PRESET),
            export_path_metrics=_as_bool(
                raytracing.get("export_path_metrics"),
                "raytracing.export_path_metrics",
                default=DEFAULT_EXPORT_PATH_METRICS,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise _ScenarioParseError("timeline", f"Invalid timeline settings: {exc}") from exc
    dependencies = (
        ScenarioDependencyCollector().collect(mapping, resource_root)
        if resource_root is not None
        else ()
    )
    return AuthoringScenario(
        document_id=document_id,
        scene=scene,
        timeline=timeline,
        actors=tuple(authoring_actors),
        groups=tuple(authoring_groups),
        resources=tuple(resources),
        source_snapshot=ScenarioSourceSnapshot.from_mapping(
            mapping,
            actor_bindings=(
                ScenarioSourceBinding(
                    name_to_id[str(entry["name"])],
                    tuple(int(part) if part.isdigit() else part for part in path.split(".")),
                )
                for _role, entry, path in specs
            ),
            group_bindings=(
                ScenarioSourceBinding(
                    group_name_to_id[str(entry["name"])],
                    ("groups", index),
                )
                for index, entry in enumerate(group_entries)
            ),
        ),
        facet_capabilities=_facet_capabilities(
            mapping,
            authoring_actors,
            authoring_groups,
            specs,
        ),
        dependencies=dependencies,
    )


def load_for_authoring(
    path: Path | str,
    *,
    undo_stack: UndoStack | None = None,
    analyzer: CompatibilityAnalyzer | None = None,
) -> AuthoringLoadResult:
    """Open canonical schema-v2 YAML in place without writing it.

    Compatible alternate YAML filenames remain explicit copy sources because
    Scenario Builder saves and generates only ``<scenario>/scenario.yaml``.
    """
    source_path = _resolve_yaml_path(path)
    try:
        mapping = _load_mapping(source_path)
    except (ValueError, yaml.YAMLError) as exc:
        issue = ValidationIssue(
            IssueSeverity.ERROR,
            "compatibility.yaml_invalid",
            "scenario",
            str(exc),
        )
        return AuthoringLoadResult(
            LoadDisposition.READ_ONLY,
            (issue,),
            None,
            None,
            {},
            source_path,
        )
    report = (analyzer or CompatibilityAnalyzer()).analyze(mapping)
    if not report.compatible:
        return AuthoringLoadResult(
            report.disposition,
            report.issues,
            None,
            None,
            mapping,
            source_path,
        )
    try:
        scenario = scenario_from_mapping(
            mapping,
            source_directory=source_path.parent,
        )
    except _ScenarioParseError as exc:
        issue = ValidationIssue(
            IssueSeverity.ERROR,
            "compatibility.parse",
            exc.path,
            str(exc),
        )
        return AuthoringLoadResult(
            LoadDisposition.READ_ONLY,
            (*report.issues, issue),
            None,
            None,
            mapping,
            source_path,
        )
    disposition = report.disposition
    issues = report.issues
    if disposition is LoadDisposition.OWNED_EDITABLE and source_path.name != "scenario.yaml":
        disposition = LoadDisposition.READ_ONLY
        issues = (
            *issues,
            ValidationIssue(
                IssueSeverity.WARNING,
                "compatibility.canonical_filename_required",
                "scenario",
                "Writable Scenario Builder documents must use the canonical "
                "filename scenario.yaml in their own scenario directory.",
            ),
        )
    document = (
        ScenarioDocument.loaded(scenario, source_path, undo_stack=undo_stack)
        if disposition is LoadDisposition.OWNED_EDITABLE
        else None
    )
    return AuthoringLoadResult(
        disposition,
        issues,
        document,
        scenario,
        mapping,
        source_path,
    )


def create_scenario_copy(
    source_path: Path | str,
    destination_directory: Path | str,
    *,
    undo_stack: UndoStack | None = None,
    analyzer: CompatibilityAnalyzer | None = None,
) -> ScenarioDocument:
    """Create an unsaved canonical copy document targeting a new directory."""
    result = load_for_authoring(source_path, analyzer=analyzer)
    if result.disposition is LoadDisposition.READ_ONLY or result.scenario is None:
        raise ValueError("Scenario is not compatible with Scenario Builder copying.")
    if result.source_path.name != "scenario.yaml":
        raise ValueError("Scenario copies require a canonical source named scenario.yaml.")
    destination = Path(destination_directory).resolve()
    destination_path = (destination / "scenario.yaml").resolve()
    if destination_path == result.source_path:
        raise ValueError("Scenario copy requires a destination different from the source.")
    if destination_path.exists():
        raise FileExistsError(
            f"Scenario copy destination already contains scenario.yaml: {destination_path}"
        )
    return ScenarioDocument(
        result.scenario,
        path=destination_path,
        ownership=DocumentOwnership.COPIED,
        undo_stack=undo_stack,
        saved=False,
    )


def atomic_write_scenario_yaml(path: Path | str, yaml_text: str) -> Path:
    """Atomically replace ``scenario.yaml`` via a same-directory temporary file."""
    destination = Path(path).resolve()
    if destination.name != "scenario.yaml":
        raise ValueError("authoring documents must be saved as scenario.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".scenario.yaml.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(yaml_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


@dataclass(slots=True)
class _ResourcePromotion:
    destination: Path
    staged: Path
    backup: Path
    backup_created: bool = False
    replacement_installed: bool = False


@dataclass(slots=True)
class _ResourceSaveTransaction:
    staging_directory: Path | None
    promotions: tuple[_ResourcePromotion, ...]

    def promote(self) -> None:
        """Install staged resources while retaining rollback copies."""

        for promotion in self.promotions:
            promotion.destination.parent.mkdir(parents=True, exist_ok=True)
            if promotion.destination.exists():
                os.replace(promotion.destination, promotion.backup)
                promotion.backup_created = True
            os.replace(promotion.staged, promotion.destination)
            promotion.replacement_installed = True

    def rollback(self) -> None:
        """Restore every resource destination to its pre-save state."""

        for promotion in reversed(self.promotions):
            if promotion.replacement_installed:
                promotion.destination.unlink(missing_ok=True)
                promotion.replacement_installed = False
            if promotion.backup_created and promotion.backup.exists():
                os.replace(promotion.backup, promotion.destination)
                promotion.backup_created = False

    def cleanup(self) -> None:
        """Remove transaction-only files after commit or rollback."""

        if self.staging_directory is not None:
            shutil.rmtree(self.staging_directory, ignore_errors=True)


def _stage_document_resources(
    scenario: AuthoringScenario,
    scenario_directory: Path,
    *,
    allow_owned_destination_update: bool,
) -> _ResourceSaveTransaction:
    candidates: list[tuple[Path, Path]] = []
    for resource in scenario.resources:
        destination = (scenario_directory / resource.relative_path).resolve()
        if not destination.is_relative_to(scenario_directory):
            raise ValueError(
                f"Resource destination escapes the scenario directory: " f"{resource.relative_path}"
            )
        source = resource.source_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Mobility resource not found: {source}")
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if filecmp.cmp(source, destination, shallow=False):
                continue
            if not allow_owned_destination_update:
                raise FileExistsError(
                    f"Refusing to overwrite a different scenario resource: {destination}"
                )
        candidates.append((source, destination))

    dependency_destinations: dict[Path, Path] = {}
    for dependency in scenario.dependencies:
        source = dependency.source_path.resolve()
        if dependency.problem:
            raise ValueError(dependency.problem)
        if not source.exists():
            raise FileNotFoundError(
                f"Scenario dependency not found for {dependency.origin_path}: {source}"
            )
        if dependency.external:
            continue
        if not source.is_file():
            raise FileNotFoundError(
                f"Scenario dependency is not a readable file for "
                f"{dependency.origin_path}: {source}"
            )
        destination = (scenario_directory / Path(dependency.relative_path)).resolve()
        if not destination.is_relative_to(scenario_directory):
            raise ValueError(
                f"Scenario dependency destination escapes the scenario directory "
                f"for {dependency.origin_path}: {dependency.relative_path}"
            )
        previous = dependency_destinations.get(destination)
        if previous is not None and previous != source:
            raise ValueError(
                f"Scenario dependencies collide at {destination}: {previous} and {source}"
            )
        if previous == source:
            continue
        dependency_destinations[destination] = source
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if filecmp.cmp(source, destination, shallow=False):
                continue
            if not allow_owned_destination_update:
                raise FileExistsError(
                    f"Refusing to overwrite a different scenario dependency: {destination}"
                )
        candidates.append((source, destination))

    if not candidates:
        return _ResourceSaveTransaction(None, ())

    scenario_directory.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=".scenario-resources.",
            dir=scenario_directory,
        )
    )
    promotions: list[_ResourcePromotion] = []
    try:
        for index, (source, destination) in enumerate(candidates):
            staged = staging_directory / f"resource-{index}.staged"
            backup = staging_directory / f"resource-{index}.backup"
            shutil.copy2(source, staged)
            promotions.append(
                _ResourcePromotion(
                    destination=destination,
                    staged=staged,
                    backup=backup,
                )
            )
    except BaseException:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return _ResourceSaveTransaction(staging_directory, tuple(promotions))


def _project_library_save_issues(
    scenario: AuthoringScenario,
    scenario_directory: Path,
    project_root: Path,
) -> tuple[ValidationIssue, ...]:
    """Reject destinations that would lose project-library resolution.

    The Builder compiler can preview library assets from its own checkout even
    when a scenario is being saved elsewhere. The Generator resolves those
    same references from the saved scenario's ORCHAV project root. Refuse the
    save before writing a YAML file that the Generator could not run.
    """

    resolved_root = Path(project_root).resolve()
    if scenario_directory.resolve().is_relative_to(resolved_root):
        return ()

    issues: list[ValidationIssue] = []
    scene = scenario.scene
    if scene is not None and scene.source == "library":
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                "save.library_scene.outside_project",
                "scene.id",
                "ORCHAV library scenes require a scenario directory inside the "
                f"active ORCHAV project root ({resolved_root}). Choose a directory "
                "inside that root, or select a local or Sionna scene before saving.",
            )
        )

    target_index = 0
    for actor in scenario.actors:
        if actor.role is not ActorRole.TARGET:
            continue
        target = actor.target
        if target is not None and target.source == "catalog":
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    "save.catalog_target.outside_project",
                    f"actors.targets.{target_index}.asset.id",
                    "ORCHAV catalog targets require a scenario directory inside the "
                    f"active ORCHAV project root ({resolved_root}). Choose a directory "
                    "inside that root, or use a scenario-local file or directory target "
                    "before saving.",
                    actor.id,
                )
            )
        target_index += 1
    return tuple(issues)


def save_document(
    document: ScenarioDocument,
    destination: Path | str | None = None,
    *,
    compiler: ScenarioCompiler | None = None,
    compile_lock: AbstractContextManager[object] | None = None,
) -> Path:
    """Validate completely, atomically save, then mark the document clean.

    ``compile_lock`` lets an interactive owner serialize this synchronous
    validation with its background compiler.  Standalone persistence callers
    do not need to provide one.
    """
    if document.read_only:
        raise PermissionError("the authoring document is read-only")
    target: Path
    if destination is None:
        if document.path is None:
            raise ValueError("a scenario directory is required for the first save")
        target = document.path
    else:
        selected = Path(destination)
        if selected.name == "scenario.yaml":
            target = selected
        elif selected.suffix.lower() in {".yaml", ".yml"}:
            raise ValueError("select a scenario directory, not an arbitrary YAML filename")
        else:
            target = selected / "scenario.yaml"
    target = target.resolve()
    current_path = document.path.resolve() if document.path is not None else None
    saving_owned_in_place = document.ownership is DocumentOwnership.OWNED and current_path == target
    if target.exists() and not saving_owned_in_place:
        raise FileExistsError(
            f"Refusing to overwrite an existing scenario outside this owned document: {target}"
        )
    active_compiler = compiler or ScenarioCompiler()
    project_root = getattr(active_compiler, "project_root", None)
    if project_root is not None:
        project_library_issues = _project_library_save_issues(
            document.scenario,
            target.parent,
            Path(project_root),
        )
        if project_library_issues:
            raise ScenarioSaveError(project_library_issues)
    guard = compile_lock if compile_lock is not None else nullcontext()
    with guard:
        result = active_compiler.compile(document.scenario, scenario_directory=target.parent)
    blocking = tuple(issue for issue in result.issues if issue.severity is IssueSeverity.ERROR)
    if blocking:
        raise ScenarioSaveError(blocking)
    resource_transaction = _stage_document_resources(
        document.scenario,
        target.parent,
        allow_owned_destination_update=saving_owned_in_place,
    )
    try:
        resource_transaction.promote()
        atomic_write_scenario_yaml(target, result.yaml_text)
    except BaseException:
        try:
            resource_transaction.rollback()
        except BaseException as rollback_error:
            recovery_path = resource_transaction.staging_directory
            raise RuntimeError(
                "Scenario save failed and resource rollback was incomplete; "
                f"recovery files remain at {recovery_path}."
            ) from rollback_error
        resource_transaction.cleanup()
        raise
    resource_transaction.cleanup()
    document.mark_saved(target)
    return target
