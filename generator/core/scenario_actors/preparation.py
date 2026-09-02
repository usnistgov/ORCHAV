"""Whole-scenario actor and optional-group reference resolution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from ._adapters import discriminator, value
from .errors import PosePreparationError
from .mobility import derive_group_member_mobility, prepare_mobility
from .orientation import apply_asset_alignment, prepare_orientation
from .quaternion import Quaternion
from .types import (
    ActorRole,
    PreparedActorPose,
    PreparedGroupPose,
    PreparedMobility,
    PreparedScenario,
    Timeline,
)


def prepare_scenario(
    scenario_model: object,
    *,
    base_dir: str | Path | None = None,
    asset_alignments: Mapping[str, Quaternion | object] | None = None,
) -> PreparedScenario:
    """Prepare all actors into one immutable renderer-neutral index.

    ``scenario_model`` is normally ``shared.scenarios.model.ScenarioModel``.
    Structural access also permits a ScenarioConfiguration exposing the same
    ``timeline``, ``actors``, and ``groups`` attributes and small test doubles.
    ``base_dir`` is reserved for resource-backed mobility resolution; ordinary
    actor/group preparation performs no filesystem or renderer work.
    """

    scenario = _unwrap_scenario(scenario_model)
    timeline_spec = value(scenario, "timeline")
    timeline = Timeline(
        steps=int(value(timeline_spec, "steps")),
        duration_s=float(value(timeline_spec, "duration_s")),
    )
    actor_records = _collect_actor_records(value(scenario, "actors", default={}))
    _validate_actor_names(actor_records)

    group_specs = tuple(value(scenario, "groups", default=()) or ())
    groups_by_name: dict[str, object] = {}
    for index, group_spec in enumerate(group_specs):
        name = str(value(group_spec, "name"))
        if not name:
            raise PosePreparationError(
                "empty_group_name", f"groups[{index}].name", "group name must not be empty"
            )
        if name in groups_by_name:
            raise PosePreparationError(
                "duplicate_group_name", f"groups[{index}].name", f"duplicate group name {name!r}"
            )
        groups_by_name[name] = group_spec

    member_counts = {name: 0 for name in groups_by_name}
    for _, actor, path in actor_records:
        mobility_spec = value(actor, "mobility")
        if discriminator(mobility_spec) != "group_member":
            continue
        group_name = str(value(mobility_spec, "group"))
        if group_name not in groups_by_name:
            raise PosePreparationError(
                "missing_group_reference",
                f"{path}.mobility.group",
                f"group {group_name!r} does not exist",
                actor_name=str(value(actor, "name")),
            )
        member_counts[group_name] += 1
    invalid_groups = sorted(name for name, count in member_counts.items() if count < 2)
    if invalid_groups:
        raise PosePreparationError(
            "invalid_group_member_count",
            "groups",
            "groups require at least two members: " + ", ".join(invalid_groups),
        )

    # Every member samples the same prepared group path.
    prepared_groups: dict[str, PreparedMobility] = {}
    prepared_group_poses: list[PreparedGroupPose] = []
    for index, group_spec in enumerate(group_specs):
        name = str(value(group_spec, "name"))
        mobility_spec = value(group_spec, "mobility")
        if discriminator(mobility_spec) == "group_member":
            raise PosePreparationError(
                "nested_group",
                f"groups[{index}].mobility",
                "groups cannot reference other groups",
            )
        mobility = prepare_mobility_with_resources(
            mobility_spec,
            timeline,
            base_dir=base_dir,
            path=f"groups[{index}].mobility",
        )
        prepared_groups[name] = mobility
        prepared_group_poses.append(PreparedGroupPose(name, mobility))

    prepared_mobility: dict[str, PreparedMobility] = {}
    for _, actor, path in actor_records:
        name = str(value(actor, "name"))
        if _has_actor_level_position(actor):
            raise PosePreparationError(
                "ambiguous_position_owner",
                path,
                "position belongs inside mobility, not on the actor",
                actor_name=name,
            )
        mobility_spec = value(actor, "mobility")
        if discriminator(mobility_spec) == "group_member":
            group_name = str(value(mobility_spec, "group"))
            group_spec = groups_by_name[group_name]
            offset = value(mobility_spec, "offset_m", default=(0.0, 0.0, 0.0))
            deviation_spec = value(group_spec, "deviation", default=None)
            deviation_bounds = None
            deviation_seed = None
            if deviation_spec is not None:
                deviation_bounds = (
                    float(value(deviation_spec, "max_right_m", default=0.0)),
                    float(value(deviation_spec, "max_forward_m", default=0.0)),
                    float(value(deviation_spec, "max_up_m", default=0.0)),
                )
                deviation_seed = _member_seed(int(value(deviation_spec, "seed")), name)
            prepared_mobility[name] = derive_group_member_mobility(
                prepared_groups[group_name],
                offset,
                timeline,
                deviation=deviation_bounds,
                seed=deviation_seed,
                path=f"{path}.mobility",
            )
        else:
            prepared_mobility[name] = prepare_mobility_with_resources(
                mobility_spec,
                timeline,
                base_dir=base_dir,
                path=f"{path}.mobility",
            )

    alignments = asset_alignments or {}
    prepared_actors: list[PreparedActorPose] = []
    for role, actor, path in actor_records:
        name = str(value(actor, "name"))
        orientation_spec = value(actor, "orientation", default=None)
        orientation = prepare_orientation(
            orientation_spec,
            timeline,
            prepared_mobility[name],
            references=prepared_mobility,
            path=f"{path}.orientation",
        )
        if name in alignments:
            if role != "target":
                raise PosePreparationError(
                    "asset_alignment_non_target",
                    f"{path}.orientation",
                    "asset alignment can only be applied to target actors",
                    actor_name=name,
                )
            orientation = apply_asset_alignment(
                orientation,
                alignments[name],
                path=f"{path}.asset.alignment",
            )
        prepared_actors.append(
            PreparedActorPose(
                name=name,
                role=role,
                mobility=prepared_mobility[name],
                orientation=orientation,
            )
        )

    unknown_alignments = sorted(set(alignments) - set(prepared_mobility))
    if unknown_alignments:
        raise PosePreparationError(
            "missing_actor_reference",
            "asset_alignments",
            "alignment references unknown actors: " + ", ".join(unknown_alignments),
        )
    return PreparedScenario(timeline, tuple(prepared_actors), tuple(prepared_group_poses))


def _collect_actor_records(actors: object) -> tuple[tuple[ActorRole, object, str], ...]:
    records: list[tuple[ActorRole, object, str]] = []
    actor_sections: tuple[tuple[str, ActorRole], ...] = (
        ("tx", "tx"),
        ("rx", "rx"),
        ("targets", "target"),
    )
    for section, role in actor_sections:
        entries = value(actors, section, default=()) or ()
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise PosePreparationError(
                "invalid_actor_section",
                f"actors.{section}",
                "actor section must be a sequence",
            )
        for index, actor in enumerate(entries):
            records.append((role, actor, f"actors.{section}[{index}]"))
    return tuple(records)


def _validate_actor_names(records: tuple[tuple[ActorRole, object, str], ...]) -> None:
    seen: set[str] = set()
    for _, actor, path in records:
        name = str(value(actor, "name"))
        if not name:
            raise PosePreparationError(
                "empty_actor_name", f"{path}.name", "actor name must not be empty"
            )
        if name in seen:
            raise PosePreparationError(
                "duplicate_actor_name",
                f"{path}.name",
                f"duplicate actor name {name!r}",
                actor_name=name,
            )
        seen.add(name)


def _unwrap_scenario(candidate: object) -> object:
    if _has_field(candidate, "timeline") and _has_field(candidate, "actors"):
        return candidate
    for field_name in ("scenario_model", "model", "scenario"):
        nested = value(candidate, field_name, default=None)
        if nested is not None and _has_field(nested, "timeline") and _has_field(nested, "actors"):
            return nested
    raise PosePreparationError(
        "invalid_scenario_input",
        "$",
        "prepare_scenario requires a scenario object with timeline and actors",
    )


def prepare_mobility_with_resources(
    mobility_spec: object,
    timeline: Timeline,
    *,
    base_dir: str | Path | None,
    path: str,
) -> PreparedMobility:
    """Prepare one mobility specification with local resource resolution."""

    mobility_type = discriminator(mobility_spec)
    if mobility_type in ("mesh_sequence", "network_route"):
        # Resource-backed evaluation is isolated to keep the ordinary kernel
        # deterministic and free of renderer/network dependencies.
        from .resources import prepare_resource_mobility

        return prepare_resource_mobility(
            mobility_spec,
            timeline,
            base_dir=base_dir,
            path=path,
        )
    return prepare_mobility(mobility_spec, timeline, path=path)


def _has_actor_level_position(actor: object) -> bool:
    return _has_field(actor, "position") or _has_field(actor, "position_m")


def _has_field(obj: object, name: str) -> bool:
    if isinstance(obj, Mapping):
        return name in obj
    return hasattr(obj, name)


def _member_seed(group_seed: int, actor_name: str) -> int:
    digest = hashlib.sha256(f"{group_seed}:{actor_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
