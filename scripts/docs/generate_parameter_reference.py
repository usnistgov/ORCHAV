#!/usr/bin/env python3
"""Generate schema-derived Markdown blocks for ORCHAV parameter references."""

from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.scenarios.actors import (
    ActorsSpec,
    AlignMotionOrientationSpec,
    CatalogAssetSpec,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    DirectoryAssetSpec,
    Figure8MobilitySpec,
    FileAssetSpec,
    FitDurationTraversalSpec,
    FixedOrientationSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    GroupSpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    OrientationKeyframeSpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    RxActorSpec,
    SampledMobilitySpec,
    SpinOrientationSpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    TargetActorSpec,
    TimelineSpec,
    TxActorSpec,
    WaypointMobilitySpec,
)
from shared.scenarios.model import (
    AntennaArrayModel,
    AntennaModel,
    CoverageDistributionFigureModel,
    CoverageGridModel,
    CoverageMetricsModel,
    CoverageModel,
    CoverageSaveDataModel,
    CoverageSaveFigureModel,
    CoverageSaveModel,
    CoverageSolverModel,
    CoverageTxModel,
    DataConfigModel,
    FilesConfigModel,
    GeneratorSummaryModel,
    GeneratorSummaryOutputModel,
    LiveGrpcConfigModel,
    LiveGrpcEndpointsModel,
    MaterialOverrideModel,
    MpcVisibilityDefaultsModel,
    PathFilterModel,
    RayTracingModel,
    RayTracingQualityModel,
    RemoteHdf5ConfigModel,
    ScenarioModel,
    SceneMaterialsModel,
    SceneModel,
    ViewDefaultsModel,
    VisualizationConfigModel,
)

SCENARIO_BEGIN_MARKER = "<!-- BEGIN GENERATED: scenario-parameters -->"
SCENARIO_END_MARKER = "<!-- END GENERATED: scenario-parameters -->"


@dataclass(frozen=True)
class ModelSection:
    title: str
    yaml_prefix: str
    model_class: type[BaseModel]
    exclude_fields: frozenset[str] = frozenset()
    values_overrides: dict[str, str] | None = None
    default_overrides: dict[str, str] | None = None
    summary: str | None = None
    anchor: str | None = None


# Actor specifications predate the schema-derived reference and intentionally
# keep their runtime models compact. These user-facing descriptions fill the
# documentation metadata gap without duplicating Pydantic defaults or
# constraints, which continue to come from ``model_json_schema()``.
FIELD_DESCRIPTION_FALLBACKS = {
    "actor": "Name of the actor to face. Use exactly one of `actor` or `point_m`.",
    "after_end": "Behavior after reaching the path end: hold, loop, or reverse direction.",
    "allow_pitch": (
        "Whether computed facing may include pitch. False keeps yaw-only tracking before "
        "offsets."
    ),
    "alpha": (
        "Gauss-Markov correlation factor: 0 gives independent updates and 1 preserves "
        "constant velocity."
    ),
    "amplitude_m": "Maximum displacement from `center_m` along the configured axis, in meters.",
    "asset": "Target mesh asset mapping used for rendering and propagation.",
    "block_size_m": "Distance between adjacent Manhattan-grid nodes, in meters.",
    "center_m": "World-space path center as `[x, y, z]` in meters.",
    "clockwise": "Whether the horizontal path runs clockwise when viewed from world +Z.",
    "deviation": "Optional seeded per-sample displacement limits for group members.",
    "direction_std_deg": "Standard deviation of Gauss-Markov heading noise, in degrees.",
    "duration_s": "Total endpoint-inclusive scenario duration in seconds.",
    "end_altitude_m": "World-space Z coordinate at the end of the spiral, in meters.",
    "end_m": "World-space path endpoint as `[x, y, z]` in meters.",
    "end_node": (
        "Cached-graph node where a shortest-path route ends. Provide it with `start_node`."
    ),
    "forward": "Offset along the moving group's local forward axis, in meters.",
    "frame_stride": "Use every Nth mesh after `start_index` in the matched directory sequence.",
    "frequency_hz": "Number of complete motion cycles per second.",
    "grid_height": "Number of Manhattan-grid nodes along world Y.",
    "grid_width": "Number of Manhattan-grid nodes along world X.",
    "group": "Name of the shared mobility group followed by this actor.",
    "heading_deg": (
        "Counterclockwise rotation of the survey's local +X axis from world +X, in degrees."
    ),
    "height_m": "Survey rectangle extent along its local +Y axis, in meters.",
    "id": "Catalog-relative target asset identifier using forward-slash path segments.",
    "initial_position_m": "World-space starting position as `[x, y, z]` in meters.",
    "interpolation": "Interpolation used between consecutive authored or loaded path points.",
    "keyframes": "Time-ordered orientation samples to interpolate.",
    "length_m": "Distance from the pendulum pivot to the actor, in meters.",
    "material_type": "Electromagnetic material family assigned to the target mesh.",
    "mesh_end_behavior": (
        "Behavior when mesh playback reaches or crosses the sequence end: wrap by "
        "modulo or hold the final source mesh."
    ),
    "max_angle_deg": "Maximum pendulum displacement from its zero-angle position, in degrees.",
    "max_forward_m": "Maximum absolute seeded displacement along local forward, in meters.",
    "max_pitch_rate_deg_s": "Optional maximum pitch change rate, in degrees per second.",
    "max_right_m": "Maximum absolute seeded displacement along local right, in meters.",
    "max_up_m": "Maximum absolute seeded displacement along local up, in meters.",
    "max_yaw_rate_deg_s": "Optional maximum yaw change rate, in degrees per second.",
    "mean_direction_deg": (
        "Mean Gauss-Markov heading in world XY, counterclockwise from +X, in degrees."
    ),
    "mean_speed_mps": "Mean Gauss-Markov speed, in meters per second.",
    "min_distance_m": (
        "Minimum Euclidean distance between accepted Poisson-disk samples, in meters."
    ),
    "mobility": "Mobility mapping that supplies positions across the shared timeline.",
    "name": "Scenario-level name used by actor and group references.",
    "offset_m": "Actor offset in the moving group's local right, forward, and up frame.",
    "orientation": ("Optional orientation mapping. Omission uses fixed zero yaw, pitch, and roll."),
    "origin_m": "World-space origin of the survey rectangle as `[x, y, z]` in meters.",
    "origin_xy_m": (
        "World-space `[x, y]` coordinate of the initial Manhattan-grid node, in meters."
    ),
    "pattern": "Glob pattern selecting mesh files within the target asset directory.",
    "pause_range_s": "`[minimum, maximum]` pause duration sampled at each stop, in seconds.",
    "phase_deg": "Motion phase at timeline time zero, in degrees.",
    "pitch_deg": "Pitch component of the right-handed Z/Y/X orientation, in degrees.",
    "pitch_limits_deg": "Optional `[minimum, maximum]` clamp for computed pitch, in degrees.",
    "pitch_offset_deg": "Pitch offset added to the computed orientation, in degrees.",
    "pitch_range_deg": "`[minimum, maximum]` range for seeded pitch samples, in degrees.",
    "pivot_m": "World-space pendulum pivot as `[x, y, z]` in meters.",
    "plane": "World coordinate plane containing the motion.",
    "point_m": (
        "Fixed world-space point to face as `[x, y, z]` in meters. This is mutually "
        "exclusive with `actor`."
    ),
    "points_m": "Ordered world-space path points, each `[x, y, z]` in meters.",
    "position_key": "Array or dataset key used by NPZ, YAML, JSON, and HDF5 position resources.",
    "position_m": "Fixed world-space position as `[x, y, z]` in meters.",
    "positions_m": "World-space positions, one `[x, y, z]` value for each timeline sample.",
    "positions_path": ("Path to a local position-data file containing an N-by-3 array in meters."),
    "power_dbm": "Optional transmitter power in dBm.",
    "radius_m": "Horizontal path radius in meters.",
    "rate_deg_s": "Signed angular rotation rate, in degrees per second.",
    "right": "Offset along the moving group's local right axis, in meters.",
    "roll_deg": "Roll component of the right-handed Z/Y/X orientation, in degrees.",
    "roll_offset_deg": "Roll offset added to the computed orientation, in degrees.",
    "roll_range_deg": "`[minimum, maximum]` range for seeded roll samples, in degrees.",
    "route": "Network route strategy: a weighted shortest path or a seeded random walk.",
    "row_spacing_m": "Requested maximum spacing between adjacent survey passes, in meters.",
    "rx": "Receiver actor mappings.",
    "sampling": "Independent position sampling strategy: uniform or Poisson disk.",
    "scale": "Positive uniform scale applied to target mesh geometry.",
    "seed": "Portable deterministic non-negative signed 32-bit random seed.",
    "size_m": "Scale of the figure-eight path from its center, in meters.",
    "smoothing_time_s": (
        "Exponential orientation-smoothing time constant in seconds. Zero disables smoothing."
    ),
    "speed_mps": "Constant physical path speed in meters per second.",
    "speed_range_mps": "`[minimum, maximum]` speed range sampled in meters per second.",
    "speed_std_mps": "Standard deviation of Gauss-Markov speed noise, in meters per second.",
    "start_altitude_m": "World-space Z coordinate at the start of the spiral, in meters.",
    "start_angle_deg": "Initial horizontal azimuth from world +X, in degrees.",
    "start_corner": (
        "XY corner used for the first grid point. Left/right selects X, and bottom/top "
        "selects Y."
    ),
    "start_index": "Zero-based index of the first matched mesh used from the directory sequence.",
    "start_m": "World-space path start as `[x, y, z]` in meters.",
    "start_node": (
        "Cached-graph node where a shortest-path route begins. Provide it with `end_node`."
    ),
    "steps": "Number of endpoint-inclusive samples in the shared scenario timeline.",
    "switch_meshes": "Whether target animation advances through the asset's mesh sequence.",
    "targets": "Target actor mappings.",
    "time_s": "Elapsed scenario time of this orientation keyframe, in seconds.",
    "travel_mode": "Travel mode used to filter compatible edges in the cached network graph.",
    "traversal": "Path timing policy: fit the path to the timeline or use constant speed.",
    "traversal_pattern": (
        "Grid visitation order: alternating snake rows or same-direction raster rows."
    ),
    "turn_probability": (
        "Probability of choosing a new direction at each Manhattan-grid intersection."
    ),
    "turns": "Number of complete path cycles.",
    "tx": "Transmitter actor mappings.",
    "up": "Offset along the moving group's local up axis, in meters.",
    "update_interval_s": (
        "Time each seeded random orientation is held. Omission resamples every timeline step."
    ),
    "width_m": "Survey rectangle extent along its local +X axis, in meters.",
    "x_bounds_m": "Inclusive world-X range `[minimum, maximum]` in meters.",
    "x_steps": "Number of endpoint-inclusive grid coordinates along world X.",
    "y_bounds_m": "Inclusive world-Y range `[minimum, maximum]` in meters.",
    "y_steps": "Number of endpoint-inclusive grid coordinates along world Y.",
    "yaw_deg": "Yaw component of the right-handed Z/Y/X orientation, in degrees.",
    "yaw_limits_deg": "Optional `[minimum, maximum]` clamp for computed yaw, in degrees.",
    "yaw_offset_deg": "Yaw offset added to the computed orientation, in degrees.",
    "yaw_range_deg": "`[minimum, maximum]` range for seeded yaw samples, in degrees.",
    "z_bounds_m": "Inclusive world-Z range `[minimum, maximum]` in meters.",
    "z_steps": "Number of endpoint-inclusive grid coordinates along world Z.",
}

FIELD_DESCRIPTION_OVERRIDES = {
    ("ScenarioModel", "timeline"): "Shared endpoint-inclusive scenario timeline.",
    ("ScenarioModel", "actors"): "Transmitter, receiver, and target actor collections.",
    ("ScenarioModel", "groups"): "Optional shared mobility groups referenced by actors.",
    ("ScenarioModel", "scene"): "Scene selection mapping.",
    ("ScenarioModel", "data"): "Frame generation or playback data-mode mapping.",
    ("ScenarioModel", "raytracing"): "Optional generator ray-tracing settings.",
    ("ScenarioModel", "coverage"): "Optional coverage-map computation settings.",
    ("ScenarioModel", "view_defaults"): "Optional portable visualizer view defaults.",
    ("ScenarioModel", "generator_summary"): "Optional post-run summary-figure settings.",
    ("ScenarioModel", "live_grpc_endpoints"): "Optional Live Generator endpoint aliases.",
    ("TimelineSpec", "steps"): (
        "Number of endpoint-inclusive samples prepared for every actor and group."
    ),
    ("TimelineSpec", "duration_s"): (
        "Elapsed scenario time in seconds from the first sample at 0 s to the final sample."
    ),
    ("ActorSpec", "name"): "Globally unique actor name used by references and output metadata.",
    ("ActorSpec", "mobility"): (
        "Mobility model that supplies this actor's positions on the shared timeline."
    ),
    ("ActorSpec", "orientation"): (
        "Orientation model that supplies this actor's facing direction on the shared timeline."
    ),
    ("TxActorSpec", "power_dbm"): (
        "Optional transmitter power in dBm. Omission applies no actor-specific power override."
    ),
    ("TargetActorSpec", "asset"): "Renderable mesh asset definition used for this target.",
    ("ActorsSpec", "tx"): "Transmitters, in stable scenario order.",
    ("ActorsSpec", "rx"): "Receivers, in stable scenario order.",
    ("ActorsSpec", "targets"): "Mesh-backed target actors, in stable scenario order.",
    ("GroupSpec", "name"): "Unique group name referenced by group-member mobility.",
    ("GroupSpec", "mobility"): (
        "Standalone mobility model prepared once as the shared group path."
    ),
    ("GroupSpec", "deviation"): (
        "Optional seeded per-sample displacement bounds applied independently to each group "
        "member."
    ),
    ("GroupDeviationSpec", "max_right_m"): (
        "Maximum absolute seeded displacement in meters along the group's local right axis."
    ),
    ("GroupDeviationSpec", "max_forward_m"): (
        "Maximum absolute seeded displacement in meters along the group's local forward axis."
    ),
    ("GroupDeviationSpec", "max_up_m"): (
        "Maximum absolute seeded displacement in meters along the group's local up axis."
    ),
    ("GroupOffsetSpec", "right"): ("Signed offset in meters along the group's local right axis."),
    ("GroupOffsetSpec", "forward"): (
        "Signed offset in meters along the group's local forward axis."
    ),
    ("GroupOffsetSpec", "up"): "Signed offset in meters along the group's local up axis.",
    ("FitDurationTraversalSpec", "type"): (
        "Select complete-path traversal over the scenario duration."
    ),
    ("ConstantSpeedTraversalSpec", "type"): "Select traversal at a physical speed.",
    ("ConstantSpeedTraversalSpec", "speed_mps"): (
        "Travel speed along the authored path in meters per second."
    ),
    ("ConstantSpeedTraversalSpec", "after_end"): (
        "Behavior after reaching the path end: stay there, wrap to the start, or reverse "
        "direction."
    ),
    ("OscillatingMobilitySpec", "axis"): (
        "Nonzero world-space direction vector of the oscillation. Normalization is automatic."
    ),
    ("SpinOrientationSpec", "axis"): (
        "Orientation component to rotate continuously: yaw, pitch, or roll."
    ),
    ("SpiralMobilitySpec", "center_m"): (
        "World-space spiral-axis position. X and Y are used, while the altitude fields " "define Z."
    ),
    ("FileAssetSpec", "path"): (
        "Path to one target mesh file, relative to the scenario unless otherwise resolved."
    ),
    ("DirectoryAssetSpec", "path"): (
        "Path to a target mesh directory, relative to the scenario unless otherwise resolved."
    ),
    ("CatalogAssetSpec", "source"): "Selects an asset from the ORCHAV target catalog.",
    ("FileAssetSpec", "source"): "Selects one target mesh file.",
    ("DirectoryAssetSpec", "source"): "Selects a directory-backed target mesh sequence.",
    ("FixedOrientationSpec", "yaw_deg"): (
        "Constant actor yaw in degrees using the right-handed Z/Y/X convention."
    ),
    ("FixedOrientationSpec", "pitch_deg"): (
        "Constant actor pitch in degrees using the right-handed Z/Y/X convention."
    ),
    ("FixedOrientationSpec", "roll_deg"): (
        "Constant actor roll in degrees using the right-handed Z/Y/X convention."
    ),
    ("OrientationKeyframeSpec", "yaw_deg"): "Yaw at this keyframe, in degrees.",
    ("OrientationKeyframeSpec", "pitch_deg"): "Pitch at this keyframe, in degrees.",
    ("OrientationKeyframeSpec", "roll_deg"): "Roll at this keyframe, in degrees.",
    ("SpinOrientationSpec", "yaw_deg"): (
        "Initial yaw before the configured spin is applied, in degrees."
    ),
    ("SpinOrientationSpec", "pitch_deg"): (
        "Initial pitch before the configured spin is applied, in degrees."
    ),
    ("SpinOrientationSpec", "roll_deg"): (
        "Initial roll before the configured spin is applied, in degrees."
    ),
}


def _markdown_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _code(value: str) -> str:
    return f"`{_markdown_escape(value)}`"


def _schema_type(schema: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in schema:
        # Pydantic class names are an implementation detail. Nested scenario
        # models are represented in YAML as mappings and documented below by
        # their public YAML path and selector.
        return "mapping"

    variants = schema.get("oneOf") or schema.get("anyOf")
    if variants:
        parts = [_schema_type(part, defs) for part in variants]
        deduped = list(dict.fromkeys(parts))
        return " | ".join(deduped)

    if "enum" in schema or "const" in schema:
        declared_type = schema.get("type")
        return declared_type if isinstance(declared_type, str) else "string"

    schema_type = schema.get("type")
    if schema_type == "array":
        prefix_items = schema.get("prefixItems")
        if prefix_items:
            return "[" + ", ".join(_schema_type(item, defs) for item in prefix_items) + "]"
        item_schema = schema.get("items")
        if item_schema:
            return f"sequence of {_schema_type(item_schema, defs)}"
        return "sequence"
    if schema_type == "object":
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            value_type = _schema_type(additional, defs)
            return f"mapping of string keys to {value_type} values"
        return "mapping"
    if isinstance(schema_type, list):
        return " | ".join(str(item) for item in schema_type)
    if schema_type:
        return str(schema_type)
    return "object"


def _schema_values_and_constraints(schema: dict[str, Any]) -> str:
    values: list[str] = []
    constraints: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        if "enum" in node:
            values.extend(_code(str(value)) for value in node["enum"])
        if "const" in node:
            values.append(_code(str(node["const"])))
        if "minimum" in node:
            constraints.append(f">= {node['minimum']}")
        if "ge" in node:
            constraints.append(f">= {node['ge']}")
        if "exclusiveMinimum" in node:
            constraints.append(f"> {node['exclusiveMinimum']}")
        if "gt" in node:
            constraints.append(f"> {node['gt']}")
        if "maximum" in node:
            constraints.append(f"<= {node['maximum']}")
        if "le" in node:
            constraints.append(f"<= {node['le']}")
        if "exclusiveMaximum" in node:
            constraints.append(f"< {node['exclusiveMaximum']}")
        if "lt" in node:
            constraints.append(f"< {node['lt']}")
        if "minItems" in node:
            constraints.append(f"min items {node['minItems']}")
        if "maxItems" in node:
            constraints.append(f"max items {node['maxItems']}")
        for part in node.get("oneOf", []) or node.get("anyOf", []):
            if isinstance(part, dict) and part.get("type") != "null":
                visit(part)

    visit(schema)
    entries = list(dict.fromkeys(values + constraints))
    return ", ".join(entries) if entries else "-"


def _format_when_omitted(model_class: type[BaseModel], field_name: str) -> str:
    field = model_class.model_fields[field_name]
    if field.default is not PydanticUndefined:
        default = field.default
    elif field.default_factory is not None:
        try:
            default = field.default_factory()
        except TypeError:
            return "dynamic"
    else:
        return "required"

    if default is None:
        return "not set"
    if default is True:
        return "`true`"
    if default is False:
        return "`false`"
    if isinstance(default, str):
        return _code(default)
    if isinstance(default, (int, float)):
        return _code(str(default))
    if isinstance(default, list):
        return _code(_compact_repr(default))
    return _code(_compact_repr(default))


def _compact_repr(value: Any) -> str:
    if isinstance(value, BaseModel):
        return _compact_repr(value.model_dump(exclude_none=True))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_compact_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_compact_repr(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Enum):
        return _compact_repr(value.value)
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _model_summary(model_class: type[BaseModel], override: str | None) -> str:
    if override is not None:
        return override
    if not model_class.__doc__:
        return ""
    doc = inspect.cleandoc(model_class.__doc__)
    return doc.split("\n\n", maxsplit=1)[0]


def _field_description(
    *,
    title: str,
    model_class: type[BaseModel],
    field_name: str,
    schema_property: dict[str, Any],
) -> str:
    description = schema_property.get("description")
    if description:
        return str(description)

    for candidate_class in model_class.__mro__:
        override = FIELD_DESCRIPTION_OVERRIDES.get((candidate_class.__name__, field_name))
        if override:
            return override

    if field_name == "type":
        return f"Selects the {title.lower()} model."

    fallback = FIELD_DESCRIPTION_FALLBACKS.get(field_name)
    if fallback:
        return fallback

    raise RuntimeError(f"{model_class.__name__}.{field_name} needs a public description")


def _model_table(
    *,
    title: str,
    yaml_prefix: str,
    model_class: type[BaseModel],
    exclude_fields: frozenset[str] = frozenset(),
    values_overrides: dict[str, str] | None = None,
    default_overrides: dict[str, str] | None = None,
    summary: str | None = None,
    anchor: str | None = None,
) -> str:
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})
    public_anchor = anchor or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    lines = [f'<a id="{public_anchor}"></a>']
    lines.extend(["", f"### {title}", "", _model_summary(model_class, summary), ""])
    if yaml_prefix:
        lines.extend([f"YAML path: `{yaml_prefix}`.", ""])

    discriminator = None
    for field_name in ("type", "source"):
        prop = properties.get(field_name, {})
        if "const" in prop:
            discriminator = f"{field_name}: {prop['const']}"
            break
    if discriminator is not None:
        lines.extend([f"Selector: `{discriminator}`.", ""])

    lines.extend(
        [
            "| YAML key | Type / values | Default / requirement | Meaning |",
            "|---|---|---:|---|",
        ]
    )
    for field_name in model_class.model_fields:
        if field_name in exclude_fields:
            continue
        prop = copy.deepcopy(properties.get(field_name, {}))
        key = f"{yaml_prefix}.{field_name}" if yaml_prefix else field_name
        field_type = _schema_type(prop, defs)
        default = (default_overrides or {}).get(field_name) or _format_when_omitted(
            model_class, field_name
        )
        values = (
            (values_overrides or {}).get(field_name)
            or prop.get("values")
            or _schema_values_and_constraints(prop)
        )
        accepted = field_type if values == "-" else f"{field_type}; {values}"
        description = _field_description(
            title=title,
            model_class=model_class,
            field_name=field_name,
            schema_property=prop,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _code(key),
                    _markdown_escape(accepted),
                    default,
                    _markdown_escape(description),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _append_sections(
    blocks: list[str],
    heading: str,
    sections: list[ModelSection],
    *,
    introduction: str | None = None,
    example_name: str | None = None,
    example_yaml: str | None = None,
) -> None:
    blocks.extend([heading, ""])
    if introduction:
        blocks.extend([introduction, ""])
    if example_name and example_yaml:
        blocks.extend(
            [
                f"<!-- BEGIN EXAMPLE: {example_name} -->",
                "```yaml",
                example_yaml.strip(),
                "```",
                f"<!-- END EXAMPLE: {example_name} -->",
                "",
            ]
        )
    for index, section in enumerate(sections):
        if index:
            blocks.append("")
        blocks.append(
            _model_table(
                title=section.title,
                yaml_prefix=section.yaml_prefix,
                model_class=section.model_class,
                exclude_fields=section.exclude_fields,
                values_overrides=section.values_overrides,
                default_overrides=section.default_overrides,
                summary=section.summary,
                anchor=section.anchor,
            )
        )
    blocks.append("")


def _generate_scenario_parameters() -> str:
    document_sections = [
        ModelSection(
            "Scenario document",
            "",
            ScenarioModel,
            exclude_fields=frozenset({"sensing"}),
            values_overrides={
                "debug_level": "`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`",
            },
            default_overrides={
                "actors": "empty role lists",
                "data": "`files` mode with standard `frames/` settings",
                "debug_level": "`WARNING`",
            },
            summary="Root-level format, logging, and component-extension fields.",
            anchor="scenario-document",
        ),
        ModelSection(
            "Scene",
            "scene",
            SceneModel,
            exclude_fields=frozenset({"osm"}),
            values_overrides={"source": "`library`, `local`, `sionna`"},
            summary="Scene selection: library, local XML, or Sionna built-in.",
            anchor="scene",
        ),
        ModelSection(
            "Timeline",
            "timeline",
            TimelineSpec,
            summary="Endpoint-inclusive sampling timeline shared by every actor and group.",
            anchor="timeline",
        ),
    ]
    actor_sections = [
        ModelSection(
            "Actor collections",
            "actors",
            ActorsSpec,
            summary="Actors grouped by transmitter, receiver, and target role.",
            anchor="actors",
        ),
        ModelSection(
            "Transmitter actor",
            "actors.tx[]",
            TxActorSpec,
            summary="A transmitter with mobility, orientation, and optional transmit power.",
            anchor="actor-transmitter",
        ),
        ModelSection(
            "Receiver actor",
            "actors.rx[]",
            RxActorSpec,
            summary="A receiver with mobility and orientation.",
            anchor="actor-receiver",
        ),
        ModelSection(
            "Target actor",
            "actors.targets[]",
            TargetActorSpec,
            summary="A target with mobility, orientation, and a renderable asset.",
            anchor="actor-target",
        ),
        ModelSection(
            "Group",
            "groups[]",
            GroupSpec,
            summary="Shared standalone mobility referenced by two or more actors.",
            anchor="group",
        ),
        ModelSection(
            "Group deviation",
            "groups[].deviation",
            GroupDeviationSpec,
            summary=(
                "Seeded per-sample displacement bounds for members of a group. At least "
                "one of the three maximum bounds must be greater than zero."
            ),
            anchor="group-deviation",
        ),
        ModelSection(
            "Group member offset",
            "actors.<role>[].mobility.offset_m",
            GroupOffsetSpec,
            summary="Actor offset in the moving group's right, forward, and up frame.",
            anchor="group-member-offset",
        ),
    ]
    mobility_sections = [
        ModelSection(
            "Fit-duration traversal",
            "actors.<role>[].mobility.traversal",
            FitDurationTraversalSpec,
            summary="Traverse the complete path once over the scenario duration.",
            anchor="traversal-fit-duration",
        ),
        ModelSection(
            "Constant-speed traversal",
            "actors.<role>[].mobility.traversal",
            ConstantSpeedTraversalSpec,
            summary="Traverse at a physical speed with explicit end-of-path behavior.",
            anchor="traversal-constant-speed",
        ),
        ModelSection(
            "Stationary mobility",
            "actors.<role>[].mobility",
            StationaryMobilitySpec,
            summary="Keep the actor at one world-space position.",
            anchor="mobility-stationary",
        ),
        ModelSection(
            "Linear mobility",
            "actors.<role>[].mobility",
            LinearMobilitySpec,
            summary="Follow a straight path between two world-space positions.",
            anchor="mobility-linear",
        ),
        ModelSection(
            "Waypoint mobility",
            "actors.<role>[].mobility",
            WaypointMobilitySpec,
            summary="Follow an ordered path containing at least two waypoints.",
            anchor="mobility-waypoint",
        ),
        ModelSection(
            "Sampled mobility",
            "actors.<role>[].mobility",
            SampledMobilitySpec,
            summary="Use exactly one authoritative physical position per timeline step.",
            anchor="mobility-sampled",
        ),
        ModelSection(
            "Circular mobility",
            "actors.<role>[].mobility",
            CircularMobilitySpec,
            summary="Follow one or more turns around a horizontal circle.",
            anchor="mobility-circular",
        ),
        ModelSection(
            "Survey mobility",
            "actors.<role>[].mobility",
            SurveyMobilitySpec,
            summary="Follow a lawnmower survey path across a rotated horizontal rectangle.",
            anchor="mobility-survey",
        ),
        ModelSection(
            "Grid-scan mobility",
            "actors.<role>[].mobility",
            GridScanMobilitySpec,
            summary="Visit a regular three-dimensional grid in raster or snake order.",
            anchor="mobility-grid-scan",
        ),
        ModelSection(
            "Oscillating mobility",
            "actors.<role>[].mobility",
            OscillatingMobilitySpec,
            summary="Oscillate around a center along a nonzero world-space axis.",
            anchor="mobility-oscillating",
        ),
        ModelSection(
            "Pendulum mobility",
            "actors.<role>[].mobility",
            PendulumMobilitySpec,
            summary="Swing around a pivot in one coordinate plane.",
            anchor="mobility-pendulum",
        ),
        ModelSection(
            "Figure-eight mobility",
            "actors.<role>[].mobility",
            Figure8MobilitySpec,
            summary="Follow a figure-eight path in one coordinate plane.",
            anchor="mobility-figure-eight",
        ),
        ModelSection(
            "Spiral mobility",
            "actors.<role>[].mobility",
            SpiralMobilitySpec,
            summary="Follow a circular path while changing altitude.",
            anchor="mobility-spiral",
        ),
        ModelSection(
            "Random-sampling mobility",
            "actors.<role>[].mobility",
            RandomSamplingMobilitySpec,
            summary=(
                "Sample independent positions within a bounded volume. `poisson_disk` "
                "requires `min_distance_m`. With `uniform`, omit that field."
            ),
            anchor="mobility-random-sampling",
        ),
        ModelSection(
            "Gauss-Markov mobility",
            "actors.<role>[].mobility",
            GaussMarkovMobilitySpec,
            summary="Generate correlated motion within a bounded volume.",
            anchor="mobility-gauss-markov",
        ),
        ModelSection(
            "Random-waypoint mobility",
            "actors.<role>[].mobility",
            RandomWaypointMobilitySpec,
            summary="Choose seeded destinations, speeds, and pauses within a bounded volume.",
            anchor="mobility-random-waypoint",
        ),
        ModelSection(
            "Manhattan-grid mobility",
            "actors.<role>[].mobility",
            ManhattanGridMobilitySpec,
            summary="Move along a seeded orthogonal street grid.",
            anchor="mobility-manhattan-grid",
        ),
        ModelSection(
            "Network-route mobility",
            "actors.<role>[].mobility",
            NetworkRouteMobilitySpec,
            summary=(
                "Follow a route on a referenced spatial network. `random_walk` requires "
                "`seed` and does not accept `start_node` or `end_node`. `shortest_path` "
                "accepts those two node fields only as a pair."
            ),
            anchor="mobility-network-route",
        ),
        ModelSection(
            "Mesh-sequence mobility",
            "actors.targets[].mobility",
            MeshSequenceMobilitySpec,
            summary="Load target position samples from an external resource.",
            anchor="mobility-mesh-sequence",
        ),
        ModelSection(
            "Group-member mobility",
            "actors.<role>[].mobility",
            GroupMemberMobilitySpec,
            summary="Follow a named group's mobility with an optional local offset.",
            anchor="mobility-group-member",
        ),
    ]
    orientation_sections = [
        ModelSection(
            "Fixed orientation",
            "actors.<role>[].orientation",
            FixedOrientationSpec,
            summary="Hold fixed yaw, pitch, and roll angles in degrees.",
            anchor="orientation-fixed",
        ),
        ModelSection(
            "Orientation keyframe",
            "actors.<role>[].orientation.keyframes[]",
            OrientationKeyframeSpec,
            summary="One time-stamped yaw, pitch, and roll sample.",
            anchor="orientation-keyframe",
        ),
        ModelSection(
            "Keyframed orientation",
            "actors.<role>[].orientation",
            KeyframesOrientationSpec,
            summary="Interpolate through at least two time-ordered orientation samples.",
            anchor="orientation-keyframes",
        ),
        ModelSection(
            "Motion-aligned orientation",
            "actors.<role>[].orientation",
            AlignMotionOrientationSpec,
            summary="Point along the actor's motion and apply optional angular constraints.",
            anchor="orientation-align-motion",
        ),
        ModelSection(
            "Look-at orientation",
            "actors.<role>[].orientation",
            LookAtOrientationSpec,
            summary="Point toward exactly one named actor or world-space point.",
            anchor="orientation-look-at",
        ),
        ModelSection(
            "Spin orientation",
            "actors.<role>[].orientation",
            SpinOrientationSpec,
            summary="Rotate continuously around one local orientation axis.",
            anchor="orientation-spin",
        ),
        ModelSection(
            "Random orientation",
            "actors.<role>[].orientation",
            RandomOrientationSpec,
            summary="Generate seeded angles within configured ranges.",
            anchor="orientation-random",
        ),
    ]
    target_asset_sections = [
        ModelSection(
            "Catalog target asset",
            "actors.targets[].asset",
            CatalogAssetSpec,
            summary="Resolve a target mesh from the ORCHAV asset catalog.",
            anchor="target-asset-catalog",
        ),
        ModelSection(
            "File target asset",
            "actors.targets[].asset",
            FileAssetSpec,
            summary="Load one target mesh from a local file.",
            anchor="target-asset-file",
        ),
        ModelSection(
            "Directory target asset",
            "actors.targets[].asset",
            DirectoryAssetSpec,
            summary="Load an ordered target mesh sequence from a local directory.",
            anchor="target-asset-directory",
        ),
    ]
    raytracing_sections = [
        ModelSection(
            "Ray tracing",
            "raytracing",
            RayTracingModel,
            default_overrides={
                "view": "`top`",
                "quality": "`low` preset",
                "carrier_frequency_hz": "`28e9`",
                "bandwidth_hz": "`2e9`",
                "cir_time_steps": "`1`",
            },
            anchor="raytracing",
        ),
        ModelSection(
            "Ray-tracing quality",
            "raytracing.quality",
            RayTracingQualityModel,
            default_overrides={"preset": "`low`"},
            anchor="raytracing-quality",
        ),
        ModelSection(
            "Scene-material defaults",
            "raytracing.scene_materials",
            SceneMaterialsModel,
            anchor="raytracing-scene-materials",
        ),
        ModelSection(
            "Antenna arrays",
            "raytracing.antenna",
            AntennaModel,
            anchor="raytracing-antenna",
        ),
        ModelSection(
            "TX and RX antenna-array fields",
            "raytracing.antenna.<role>",
            AntennaArrayModel,
            summary=(
                "The same fields apply below `raytracing.antenna.tx` and "
                "`raytracing.antenna.rx`."
            ),
            anchor="raytracing-antenna-array",
        ),
        ModelSection(
            "Path filtering",
            "raytracing.path_filter",
            PathFilterModel,
            anchor="raytracing-path-filter",
        ),
        ModelSection(
            "Material override",
            "raytracing.materials.<name>",
            MaterialOverrideModel,
            summary=(
                "A coefficient alone uses the Lambertian pattern. An explicit pattern "
                "also requires `scattering_coefficient`. `directive` and "
                "`backscattering` require `alpha_r`, while `g-rer` requires `alpha_g`. "
                "Other patterns reject alpha parameters, and `alpha_r` and `alpha_g` "
                "cannot be combined."
            ),
            anchor="raytracing-material-override",
        ),
    ]
    coverage_sections = [
        ModelSection(
            "Coverage settings",
            "coverage",
            CoverageModel,
            anchor="coverage-settings",
        ),
        ModelSection(
            "Coverage grid",
            "coverage.grid",
            CoverageGridModel,
            summary=(
                "Regular planar coverage grid. `resolution_m` must contain exactly two "
                "positive values. An explicit `bbox_xy` must contain two two-value ranges."
            ),
            anchor="coverage-grid",
        ),
        ModelSection(
            "Coverage solver",
            "coverage.solver",
            CoverageSolverModel,
            anchor="coverage-solver",
        ),
        ModelSection(
            "Coverage metrics",
            "coverage.metrics",
            CoverageMetricsModel,
            values_overrides={
                "store": "`path_gain_linear`, `rss_w`, `sinr_linear`",
                "derived": (
                    "`path_gain_db`, `path_loss_db`, `best_path_loss_db`, `rss_dbm`, "
                    "`best_rss_dbm`, `sum_rss_dbm`, `sinr_db`, `serving_tx`, "
                    "`tx_margin_db`"
                ),
            },
            anchor="coverage-metrics",
        ),
        ModelSection(
            "Coverage transmitter mode",
            "coverage.tx",
            CoverageTxModel,
            anchor="coverage-transmitter-mode",
        ),
        ModelSection(
            "Coverage output",
            "coverage.save",
            CoverageSaveModel,
            summary=(
                "Coverage output settings for data files and figures. Enabling figures "
                "requires persisted coverage data, so `save.data.enabled` cannot be false."
            ),
            anchor="coverage-output",
        ),
        ModelSection(
            "Coverage data output",
            "coverage.save.data",
            CoverageSaveDataModel,
            anchor="coverage-data-output",
        ),
        ModelSection(
            "Coverage figure output",
            "coverage.save.figure",
            CoverageSaveFigureModel,
            anchor="coverage-figure-output",
        ),
        ModelSection(
            "Coverage distribution figure",
            "coverage.save.figure.distribution",
            CoverageDistributionFigureModel,
            anchor="coverage-distribution-figure",
        ),
    ]
    data_sections = [
        ModelSection("Data mode", "data", DataConfigModel, anchor="data"),
        ModelSection("Local HDF5 files", "data.files", FilesConfigModel, anchor="data-files"),
        ModelSection(
            "Live Generator connection",
            "data.live_grpc",
            LiveGrpcConfigModel,
            values_overrides={"endpoint": "`host:port` or `grpc://host:port`; port 1-65535"},
            anchor="data-live-grpc",
        ),
        ModelSection(
            "Remote HDF5 connection",
            "data.remote_hdf5",
            RemoteHdf5ConfigModel,
            anchor="data-remote-hdf5",
        ),
        ModelSection(
            "Live Generator endpoint aliases",
            "live_grpc_endpoints",
            LiveGrpcEndpointsModel,
            anchor="live-grpc-endpoints",
        ),
    ]
    summary_sections = [
        ModelSection(
            "Generator-summary settings",
            "generator_summary",
            GeneratorSummaryModel,
            values_overrides={
                "create": ("`scene2d`, `scene3d`, `speed`, `orientation`, `angular_velocity`")
            },
            anchor="generator-summary-settings",
        ),
        ModelSection(
            "Generator-summary image format",
            "generator_summary.output",
            GeneratorSummaryOutputModel,
            anchor="generator-summary-output",
        ),
        ModelSection(
            "Generator-summary presentation",
            "generator_summary.visualization",
            VisualizationConfigModel,
            values_overrides={
                "scene2d_mode": "`rasterized`, `vector`, `auto`",
                "scene3d_mode": "`floor_plan`, `mesh`, `wireframe`, `hybrid`, `city`",
            },
            anchor="generator-summary-presentation",
        ),
    ]
    visualizer_sections = [
        ModelSection(
            "Visualizer view defaults",
            "view_defaults",
            ViewDefaultsModel,
            anchor="view-defaults",
        ),
        ModelSection(
            "MPC visibility defaults",
            "view_defaults.mpc_visibility",
            MpcVisibilityDefaultsModel,
            anchor="view-defaults-mpc-visibility",
        ),
    ]
    blocks = [
        SCENARIO_BEGIN_MARKER,
        "",
        "<!-- Generated by scripts/docs/generate_parameter_reference.py; do not edit this block by hand. -->",
        "",
    ]
    _append_sections(
        blocks,
        "## Common Scenario Fields",
        document_sections,
        introduction=(
            "These fields identify the schema, select the scene, define the shared timeline, "
            "and control root-level logging or component settings. The root `visualizer` "
            "mapping is component-owned. See [Visualizer Scenario Defaults]"
            "(../visualizer/scenario_defaults.md) for supported subkeys."
        ),
    )
    _append_sections(
        blocks,
        "## Actors And Groups",
        actor_sections,
        introduction=(
            "Actors are grouped by role. Every actor has one mobility mapping. Orientation "
            "defaults to fixed zero yaw, pitch, and roll."
        ),
    )
    _append_sections(
        blocks,
        "## Mobility",
        mobility_sections,
        introduction="Select exactly one mobility model with its `type` field.",
        example_name="mobility",
        example_yaml="""
mobility:
  type: linear
  start_m: [0.0, 0.0, 1.5]
  end_m: [10.0, 0.0, 1.5]
""",
    )
    _append_sections(
        blocks,
        "## Orientation",
        orientation_sections,
        introduction=(
            "Orientation is optional. Add one mapping when the actor should not use the "
            "fixed zero-angle default."
        ),
        example_name="orientation",
        example_yaml="""
orientation:
  type: look_at
  actor: RX1
""",
    )
    _append_sections(
        blocks,
        "## Target Assets",
        target_asset_sections,
        introduction="Every target actor selects one catalog, file, or directory asset.",
        example_name="target-asset",
        example_yaml="""
asset:
  source: catalog
  id: car
  material_type: metal
""",
    )
    _append_sections(
        blocks,
        "## Ray Tracing",
        raytracing_sections,
        introduction=(
            "Enable this block to generate MPC paths. Omitted quality and RF values use "
            "the effective defaults shown below."
        ),
        example_name="raytracing",
        example_yaml="""
raytracing:
  enabled: true
  quality:
    preset: low
""",
    )
    _append_sections(
        blocks,
        "## Coverage",
        coverage_sections,
        introduction="Coverage maps use their own grid, solver, metric, and output settings.",
        example_name="coverage",
        example_yaml="""
coverage:
  enabled: true
  grid:
    resolution_m: [5.0, 5.0]
    heights_m: [1.5]
""",
    )
    _append_sections(
        blocks,
        "## Data Modes",
        data_sections,
        introduction=(
            "Use local HDF5 files by default, Live Generator for requested computation, or "
            "Remote HDF5 Playback for read-only access to existing frames."
        ),
        example_name="data-mode",
        example_yaml="""
data:
  mode: files
  files:
    directory: frames
""",
    )
    _append_sections(
        blocks,
        "## Generator Summary",
        summary_sections,
        introduction="Generate standalone topology and motion figures after a run.",
        example_name="generator-summary",
        example_yaml="""
generator_summary:
  enabled: true
  create: [scene2d, speed]
""",
    )
    _append_sections(
        blocks,
        "## Visualizer Defaults",
        visualizer_sections,
        introduction=(
            "Use `view_defaults` for portable initial camera, filtering, and visibility "
            "preferences. The root `visualizer` mapping is component-owned and is listed "
            "under Common Scenario Fields."
        ),
        example_name="view-defaults",
        example_yaml="""
view_defaults:
  camera_view: isometric
  color_mode: mpc_type
  mpc_visibility:
    enabled: true
    paths: true
    bounce_points: false
""",
    )
    blocks.append(SCENARIO_END_MARKER)
    return "\n".join(blocks)


@dataclass(frozen=True)
class Target:
    path: Path
    generator: Callable[[], str]
    begin_marker: str
    end_marker: str


TARGETS: dict[str, Target] = {
    "scenario": Target(
        PROJECT_ROOT / "docs" / "reference" / "scenario_yaml.md",
        _generate_scenario_parameters,
        SCENARIO_BEGIN_MARKER,
        SCENARIO_END_MARKER,
    ),
}


def _replace_block(text: str, generated: str, begin_marker: str, end_marker: str) -> str:
    try:
        start = text.index(begin_marker)
        end = text.index(end_marker, start) + len(end_marker)
    except ValueError as exc:
        raise SystemExit(
            f"Could not find generated block markers {begin_marker!r} / {end_marker!r}"
        ) from exc
    return text[:start] + generated + text[end:]


def _check(target: Target, expected: str) -> int:
    path = target.path
    current = path.read_text(encoding="utf-8")
    updated = _replace_block(current, expected, target.begin_marker, target.end_marker)
    if current == updated:
        return 0
    diff = difflib.unified_diff(
        current.splitlines(),
        updated.splitlines(),
        fromfile=str(path),
        tofile=f"{path} (generated)",
        lineterm="",
    )
    sys.stderr.write("\n".join(diff) + "\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Update the generated block")
    mode.add_argument(
        "--check", action="store_true", help="Verify that the generated block is current"
    )
    args = parser.parse_args(argv)

    target = TARGETS[args.target]
    generated = target.generator()

    if args.write:
        current = target.path.read_text(encoding="utf-8")
        target.path.write_text(
            _replace_block(current, generated, target.begin_marker, target.end_marker),
            encoding="utf-8",
        )
        return 0
    return _check(target, generated)


if __name__ == "__main__":
    raise SystemExit(main())
