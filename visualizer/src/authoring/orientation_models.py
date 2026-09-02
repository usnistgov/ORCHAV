"""Schema-direct orientation values and authoring reference adapters.

The shared scenario models own every persisted orientation field.  This module
adds only authoring metadata, conversion defaults, and the boundary translation
between immutable UUID actor references in a live document and actor names in
scenario YAML.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, TypeVar, cast
from uuid import UUID

from pydantic import TypeAdapter

from shared.scenarios.actors import (
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    OrientationKeyframeSpec,
    OrientationSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
)

Vector3: TypeAlias = tuple[float, float, float]
Range2: TypeAlias = tuple[float, float]
OrientationModel: TypeAlias = OrientationSpec
OrientationModelType: TypeAlias = (
    type[FixedOrientationSpec]
    | type[KeyframesOrientationSpec]
    | type[AlignMotionOrientationSpec]
    | type[LookAtOrientationSpec]
    | type[SpinOrientationSpec]
    | type[RandomOrientationSpec]
)


class OrientationKind(str, Enum):
    """Canonical orientation discriminators."""

    FIXED = "fixed"
    KEYFRAMES = "keyframes"
    ALIGN_MOTION = "align_motion"
    LOOK_AT = "look_at"
    SPIN = "spin"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class OrientationModelDescriptor:
    """Immutable presentation and construction metadata for one model."""

    kind: OrientationKind
    label: str
    model_type: OrientationModelType


ORIENTATION_MODELS: Mapping[OrientationKind, OrientationModelDescriptor] = MappingProxyType(
    {
        OrientationKind.FIXED: OrientationModelDescriptor(
            OrientationKind.FIXED,
            "Fixed",
            FixedOrientationSpec,
        ),
        OrientationKind.KEYFRAMES: OrientationModelDescriptor(
            OrientationKind.KEYFRAMES,
            "Keyframes",
            KeyframesOrientationSpec,
        ),
        OrientationKind.ALIGN_MOTION: OrientationModelDescriptor(
            OrientationKind.ALIGN_MOTION,
            "Align with motion",
            AlignMotionOrientationSpec,
        ),
        OrientationKind.LOOK_AT: OrientationModelDescriptor(
            OrientationKind.LOOK_AT,
            "Look at",
            LookAtOrientationSpec,
        ),
        OrientationKind.SPIN: OrientationModelDescriptor(
            OrientationKind.SPIN,
            "Spin",
            SpinOrientationSpec,
        ),
        OrientationKind.RANDOM: OrientationModelDescriptor(
            OrientationKind.RANDOM,
            "Random",
            RandomOrientationSpec,
        ),
    }
)

ORIENTATION_MODEL_TYPES: Mapping[OrientationKind, OrientationModelType] = MappingProxyType(
    {kind: descriptor.model_type for kind, descriptor in ORIENTATION_MODELS.items()}
)
ORIENTATION_MODEL_LABELS: Mapping[OrientationKind, str] = MappingProxyType(
    {kind: descriptor.label for kind, descriptor in ORIENTATION_MODELS.items()}
)

_ORIENTATION_TYPES = tuple(ORIENTATION_MODEL_TYPES.values())
_ORIENTATION_ADAPTER: TypeAdapter[OrientationModel] = TypeAdapter(OrientationSpec)


class ActorIdentity(Protocol):
    """Minimal session actor surface needed to resolve a look-at reference."""

    @property
    def id(self) -> UUID | str: ...

    @property
    def name(self) -> str: ...


ActorT = TypeVar("ActorT", bound=ActorIdentity)
ActorNameResolver: TypeAlias = (
    Mapping[UUID | str, str] | Iterable[ActorIdentity] | Callable[[UUID], str | None]
)


def _kind(value: OrientationKind | str) -> OrientationKind:
    if isinstance(value, OrientationKind):
        return value
    normalized = str(value).strip().lower()
    try:
        return OrientationKind(normalized)
    except ValueError as exc:
        raise ValueError(f"unsupported orientation kind: {value!r}") from exc


def _vector3(value: object, *, label: str) -> Vector3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{label} must contain exactly three finite values")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three finite values") from exc
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain exactly three finite values")
    return result


def _uuid(value: UUID | str, *, label: str = "actor id") -> UUID:
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid UUID") from exc


def orientation_kind(orientation: OrientationModel) -> OrientationKind:
    """Return the typed discriminator for a shared orientation value."""

    if not isinstance(orientation, _ORIENTATION_TYPES):
        raise TypeError(f"unsupported orientation value: {type(orientation).__name__}")
    return OrientationKind(orientation.type)


def actor_look_at_orientation(
    target_actor_id: UUID | str,
    *,
    allow_pitch: bool = True,
    smoothing_time_s: float = 0.0,
    max_yaw_rate_deg_s: float | None = None,
    max_pitch_rate_deg_s: float | None = None,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
    roll_offset_deg: float = 0.0,
    yaw_limits_deg: Range2 | None = None,
    pitch_limits_deg: Range2 | None = None,
) -> LookAtOrientationSpec:
    """Create an actor-target look-at value with a canonical UUID token.

    Persisted YAML uses an actor name. The authoring document keeps the same
    relationship as a UUID so a rename cannot retarget it.
    """

    return LookAtOrientationSpec(
        actor=str(_uuid(target_actor_id, label="target actor id")),
        allow_pitch=allow_pitch,
        smoothing_time_s=smoothing_time_s,
        max_yaw_rate_deg_s=max_yaw_rate_deg_s,
        max_pitch_rate_deg_s=max_pitch_rate_deg_s,
        yaw_offset_deg=yaw_offset_deg,
        pitch_offset_deg=pitch_offset_deg,
        roll_offset_deg=roll_offset_deg,
        yaw_limits_deg=yaw_limits_deg,
        pitch_limits_deg=pitch_limits_deg,
    )


def point_look_at_orientation(
    point_m: Vector3,
    *,
    allow_pitch: bool = True,
    smoothing_time_s: float = 0.0,
    max_yaw_rate_deg_s: float | None = None,
    max_pitch_rate_deg_s: float | None = None,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
    roll_offset_deg: float = 0.0,
    yaw_limits_deg: Range2 | None = None,
    pitch_limits_deg: Range2 | None = None,
) -> LookAtOrientationSpec:
    """Create a point-target look-at value."""

    return LookAtOrientationSpec(
        point_m=_vector3(point_m, label="point_m"),
        allow_pitch=allow_pitch,
        smoothing_time_s=smoothing_time_s,
        max_yaw_rate_deg_s=max_yaw_rate_deg_s,
        max_pitch_rate_deg_s=max_pitch_rate_deg_s,
        yaw_offset_deg=yaw_offset_deg,
        pitch_offset_deg=pitch_offset_deg,
        roll_offset_deg=roll_offset_deg,
        yaw_limits_deg=yaw_limits_deg,
        pitch_limits_deg=pitch_limits_deg,
    )


def look_at_actor_id(orientation: LookAtOrientationSpec) -> UUID | None:
    """Return an internal actor target UUID, or ``None`` for a point target."""

    if orientation.actor is None:
        return None
    return _uuid(orientation.actor, label="look-at actor reference")


def resolve_actor_look_at(
    orientation: LookAtOrientationSpec,
    actors: Iterable[ActorT],
) -> ActorT | None:
    """Resolve an internal UUID reference to its current actor object."""

    target_id = look_at_actor_id(orientation)
    if target_id is None:
        return None
    for actor in actors:
        if _uuid(actor.id) == target_id:
            return actor
    return None


def resolve_actor_look_at_reference(
    orientation: LookAtOrientationSpec,
    actor_id_by_name: Mapping[str, UUID | str],
) -> LookAtOrientationSpec:
    """Translate a parsed YAML actor name to the authoring UUID convention."""

    actor_name = orientation.actor
    if actor_name is None:
        return orientation
    if actor_name in actor_id_by_name:
        target_id = _uuid(
            actor_id_by_name[actor_name],
            label=f"actor id for {actor_name!r}",
        )
    else:
        # This also makes the helper idempotent for an already-internal value.
        try:
            target_id = _uuid(actor_name, label="look-at actor reference")
        except ValueError as exc:
            raise KeyError(f"unknown look-at actor name: {actor_name!r}") from exc
    values = orientation.model_dump(mode="python")
    values["actor"] = str(target_id)
    return cast(LookAtOrientationSpec, LookAtOrientationSpec.model_validate(values))


def _forward_target_point(position_m: Vector3, euler_deg: Vector3) -> Vector3:
    yaw_rad = math.radians(euler_deg[0])
    pitch_rad = math.radians(euler_deg[1])
    horizontal = math.cos(pitch_rad)
    return (
        position_m[0] + horizontal * math.cos(yaw_rad),
        position_m[1] + horizontal * math.sin(yaw_rad),
        position_m[2] - math.sin(pitch_rad),
    )


def default_orientation(
    target_kind: OrientationKind | str,
    current_euler_deg: Vector3 = (0.0, 0.0, 0.0),
    current_position_m: Vector3 = (0.0, 0.0, 0.0),
    *,
    duration_s: float = 1.0,
    target_actor_id: UUID | str | None = None,
    target_point_m: Vector3 | None = None,
    spin_rate_deg_s: float = 30.0,
    random_seed: int = 0,
) -> OrientationModel:
    """Build a valid model, seeding explicit angles from the visible pose.

    A look-at conversion uses the supplied actor or point.  Without either, it
    creates a point one metre along the current forward axis, which preserves
    the visible yaw, pitch, and roll without inventing a document relationship.
    """

    kind = _kind(target_kind)
    yaw, pitch, roll = _vector3(current_euler_deg, label="current_euler_deg")
    position = _vector3(current_position_m, label="current_position_m")

    if kind is OrientationKind.FIXED:
        return FixedOrientationSpec(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
    if kind is OrientationKind.KEYFRAMES:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("keyframes orientation requires a positive finite duration_s")
        angles = {
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
        }
        return KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0, **angles),
                OrientationKeyframeSpec(time_s=duration, **angles),
            )
        )
    if kind is OrientationKind.ALIGN_MOTION:
        return AlignMotionOrientationSpec()
    if kind is OrientationKind.LOOK_AT:
        if target_actor_id is not None and target_point_m is not None:
            raise ValueError("look-at conversion accepts an actor or a point, not both")
        if target_actor_id is not None:
            return actor_look_at_orientation(
                target_actor_id,
                roll_offset_deg=roll,
            )
        point = (
            _vector3(target_point_m, label="target_point_m")
            if target_point_m is not None
            else _forward_target_point(position, (yaw, pitch, roll))
        )
        return point_look_at_orientation(point, roll_offset_deg=roll)
    if kind is OrientationKind.SPIN:
        return SpinOrientationSpec(
            axis="yaw",
            rate_deg_s=spin_rate_deg_s,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )
    return RandomOrientationSpec(seed=random_seed)


def convert_orientation(
    orientation: OrientationModel,
    target_kind: OrientationKind | str,
    current_euler_deg: Vector3,
    current_position_m: Vector3 = (0.0, 0.0, 0.0),
    **defaults: Any,
) -> OrientationModel:
    """Convert to another schema model while preserving the visible pose seed.

    Converting to the current kind returns the original immutable value, which
    preserves every canonical field rather than rebuilding only editor-visible
    fields.  Keyword arguments are forwarded to :func:`default_orientation`.
    """

    kind = _kind(target_kind)
    if orientation_kind(orientation) is kind:
        return orientation
    return default_orientation(
        kind,
        current_euler_deg,
        current_position_m,
        **defaults,
    )


def orientation_from_mapping(
    mapping: Mapping[str, object],
    *,
    actor_id_by_name: Mapping[str, UUID | str] | None = None,
) -> OrientationModel:
    """Validate a schema mapping and internalize any actor-name reference."""

    orientation = cast(
        OrientationModel,
        _ORIENTATION_ADAPTER.validate_python(dict(mapping)),
    )
    if not isinstance(orientation, LookAtOrientationSpec) or orientation.actor is None:
        return orientation
    if actor_id_by_name is not None:
        return resolve_actor_look_at_reference(orientation, actor_id_by_name)
    # Enforce the document invariant even when the caller already supplied an
    # internal mapping.
    target_id = look_at_actor_id(orientation)
    if orientation.actor == str(target_id):
        return orientation
    values = orientation.model_dump(mode="python")
    values["actor"] = str(target_id)
    return cast(LookAtOrientationSpec, LookAtOrientationSpec.model_validate(values))


def _actor_name(
    target_id: UUID,
    resolver: ActorNameResolver | None,
) -> str:
    if resolver is None:
        raise ValueError("serializing an actor look-at requires current actor names")
    if callable(resolver):
        name = resolver(target_id)
        if name is not None:
            return str(name)
    elif isinstance(resolver, Mapping):
        name = resolver.get(target_id)
        if name is None:
            name = resolver.get(str(target_id))
        if name is not None:
            return str(name)
    else:
        actor = next(
            (candidate for candidate in resolver if _uuid(candidate.id) == target_id),
            None,
        )
        if actor is not None:
            return str(actor.name)
    raise KeyError(f"unknown look-at actor id: {target_id}")


def orientation_to_mapping(
    orientation: OrientationModel,
    actors: ActorNameResolver | None = None,
    *,
    exclude_none: bool = False,
) -> dict[str, object]:
    """Return the complete schema mapping, resolving UUIDs to current names.

    The shared model performs the recursive dump so every canonical field is
    retained.  Actor-target look-at is the sole authoring-only transformation.
    """

    orientation_kind(orientation)
    result = cast(
        dict[str, object],
        orientation.model_dump(mode="python", exclude_none=exclude_none),
    )
    if isinstance(orientation, LookAtOrientationSpec) and orientation.actor is not None:
        target_id = look_at_actor_id(orientation)
        assert target_id is not None
        result["actor"] = _actor_name(target_id, actors)
    return result


serialize_orientation = orientation_to_mapping


__all__ = [
    "ActorIdentity",
    "ActorNameResolver",
    "ORIENTATION_MODELS",
    "ORIENTATION_MODEL_LABELS",
    "ORIENTATION_MODEL_TYPES",
    "OrientationKind",
    "OrientationModel",
    "OrientationModelDescriptor",
    "OrientationModelType",
    "actor_look_at_orientation",
    "convert_orientation",
    "default_orientation",
    "look_at_actor_id",
    "orientation_from_mapping",
    "orientation_kind",
    "orientation_to_mapping",
    "point_look_at_orientation",
    "resolve_actor_look_at",
    "resolve_actor_look_at_reference",
    "serialize_orientation",
]
