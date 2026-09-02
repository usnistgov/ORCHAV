"""Schema-direct orientation authoring and UUID reference boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from uuid import UUID, uuid4

import pytest

from shared.scenarios.actors import (
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    OrientationKeyframeSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
)
from visualizer.src.authoring.orientation_models import (
    ORIENTATION_MODEL_LABELS,
    ORIENTATION_MODEL_TYPES,
    ORIENTATION_MODELS,
    OrientationKind,
    actor_look_at_orientation,
    convert_orientation,
    default_orientation,
    look_at_actor_id,
    orientation_from_mapping,
    orientation_kind,
    orientation_to_mapping,
    point_look_at_orientation,
    resolve_actor_look_at,
)


@dataclass(frozen=True)
class _Actor:
    id: UUID
    name: str


def test_registry_is_typed_complete_and_immutable() -> None:
    expected = {
        OrientationKind.FIXED: FixedOrientationSpec,
        OrientationKind.KEYFRAMES: KeyframesOrientationSpec,
        OrientationKind.ALIGN_MOTION: AlignMotionOrientationSpec,
        OrientationKind.LOOK_AT: LookAtOrientationSpec,
        OrientationKind.SPIN: SpinOrientationSpec,
        OrientationKind.RANDOM: RandomOrientationSpec,
    }

    assert dict(ORIENTATION_MODEL_TYPES) == expected
    assert tuple(ORIENTATION_MODELS) == tuple(OrientationKind)
    assert all(ORIENTATION_MODEL_LABELS[kind] for kind in OrientationKind)
    assert all(
        descriptor.kind is kind
        and descriptor.model_type is expected[kind]
        and descriptor.label == ORIENTATION_MODEL_LABELS[kind]
        for kind, descriptor in ORIENTATION_MODELS.items()
    )
    with pytest.raises(TypeError):
        ORIENTATION_MODELS[OrientationKind.FIXED] = ORIENTATION_MODELS[  # type: ignore[index]
            OrientationKind.SPIN
        ]
    with pytest.raises(FrozenInstanceError):
        ORIENTATION_MODELS[OrientationKind.FIXED].label = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("orientation", "kind"),
    (
        (FixedOrientationSpec(), OrientationKind.FIXED),
        (
            KeyframesOrientationSpec(
                keyframes=(
                    OrientationKeyframeSpec(time_s=0.0),
                    OrientationKeyframeSpec(time_s=1.0),
                )
            ),
            OrientationKind.KEYFRAMES,
        ),
        (AlignMotionOrientationSpec(), OrientationKind.ALIGN_MOTION),
        (LookAtOrientationSpec(point_m=(1.0, 0.0, 0.0)), OrientationKind.LOOK_AT),
        (SpinOrientationSpec(rate_deg_s=30.0), OrientationKind.SPIN),
        (RandomOrientationSpec(seed=7), OrientationKind.RANDOM),
    ),
)
def test_orientation_kind_covers_every_shared_schema_model(orientation, kind) -> None:
    assert orientation_kind(orientation) is kind


def test_default_and_conversion_seed_explicit_angles_from_current_euler() -> None:
    euler = (45.0, -20.0, 7.0)

    fixed = default_orientation("fixed", euler)
    keyframes = default_orientation("keyframes", euler, duration_s=4.0)
    spin = default_orientation("spin", euler, spin_rate_deg_s=-15.0)

    assert fixed == FixedOrientationSpec(yaw_deg=45.0, pitch_deg=-20.0, roll_deg=7.0)
    assert isinstance(keyframes, KeyframesOrientationSpec)
    assert keyframes.keyframes == (
        OrientationKeyframeSpec(
            time_s=0.0,
            yaw_deg=45.0,
            pitch_deg=-20.0,
            roll_deg=7.0,
        ),
        OrientationKeyframeSpec(
            time_s=4.0,
            yaw_deg=45.0,
            pitch_deg=-20.0,
            roll_deg=7.0,
        ),
    )
    assert spin == SpinOrientationSpec(
        axis="yaw",
        rate_deg_s=-15.0,
        yaw_deg=45.0,
        pitch_deg=-20.0,
        roll_deg=7.0,
    )

    source = RandomOrientationSpec(
        seed=11,
        yaw_range_deg=(-5.0, 6.0),
        pitch_range_deg=(-7.0, 8.0),
        roll_range_deg=(-9.0, 10.0),
        update_interval_s=0.25,
    )
    assert convert_orientation(source, "random", euler) is source
    assert convert_orientation(source, "fixed", euler) == fixed


def test_default_orientation_constructs_all_six_schema_variants() -> None:
    models = {
        kind: default_orientation(kind, (10.0, -5.0, 2.0), duration_s=2.0)
        for kind in OrientationKind
    }

    assert {orientation_kind(model) for model in models.values()} == set(OrientationKind)
    assert isinstance(models[OrientationKind.RANDOM], RandomOrientationSpec)
    assert models[OrientationKind.RANDOM].seed == 0
    assert isinstance(models[OrientationKind.ALIGN_MOTION], AlignMotionOrientationSpec)


def test_actor_look_at_uses_uuid_internally_and_current_name_when_serialized() -> None:
    target_id = uuid4()
    target = _Actor(target_id, "Receiver before rename")
    orientation = actor_look_at_orientation(
        target_id,
        allow_pitch=False,
        smoothing_time_s=0.25,
        max_yaw_rate_deg_s=40.0,
        max_pitch_rate_deg_s=20.0,
        yaw_offset_deg=1.0,
        pitch_offset_deg=2.0,
        roll_offset_deg=3.0,
        yaw_limits_deg=(-60.0, 70.0),
        pitch_limits_deg=(-20.0, 30.0),
    )

    assert orientation.actor == str(target_id)
    assert look_at_actor_id(orientation) == target_id
    assert resolve_actor_look_at(orientation, (target,)) is target

    renamed = _Actor(target_id, "Receiver after rename")
    mapping = orientation_to_mapping(orientation, (renamed,))

    assert mapping == {
        **orientation.model_dump(mode="python"),
        "actor": "Receiver after rename",
    }


def test_point_look_at_remains_a_normal_schema_value() -> None:
    orientation = point_look_at_orientation(
        (1.0, 2.0, 3.0),
        allow_pitch=False,
        smoothing_time_s=0.5,
        yaw_offset_deg=4.0,
        pitch_offset_deg=5.0,
        roll_offset_deg=6.0,
        yaw_limits_deg=(-90.0, 90.0),
        pitch_limits_deg=(-30.0, 30.0),
    )

    assert look_at_actor_id(orientation) is None
    assert resolve_actor_look_at(orientation, ()) is None
    assert orientation_to_mapping(orientation) == orientation.model_dump(mode="python")


def test_look_at_default_uses_current_position_and_forward_euler() -> None:
    orientation = default_orientation(
        "look_at",
        (90.0, -30.0, 8.0),
        (10.0, 20.0, 30.0),
    )

    assert isinstance(orientation, LookAtOrientationSpec)
    assert orientation.actor is None
    assert orientation.point_m == pytest.approx(
        (
            10.0,
            20.0 + 3.0**0.5 / 2.0,
            30.5,
        )
    )
    assert orientation.roll_offset_deg == 8.0


def test_parse_and_serialize_preserve_every_canonical_field() -> None:
    target_id = uuid4()
    mappings = (
        FixedOrientationSpec(
            yaw_deg=1.0,
            pitch_deg=2.0,
            roll_deg=3.0,
        ).model_dump(mode="python"),
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(
                    time_s=0.25,
                    yaw_deg=1.0,
                    pitch_deg=2.0,
                    roll_deg=3.0,
                ),
                OrientationKeyframeSpec(
                    time_s=1.75,
                    yaw_deg=4.0,
                    pitch_deg=5.0,
                    roll_deg=6.0,
                ),
            )
        ).model_dump(mode="python"),
        AlignMotionOrientationSpec(
            allow_pitch=False,
            smoothing_time_s=0.25,
            yaw_offset_deg=1.0,
            pitch_offset_deg=2.0,
            roll_offset_deg=3.0,
            max_yaw_rate_deg_s=90.0,
            max_pitch_rate_deg_s=45.0,
        ).model_dump(mode="python"),
        LookAtOrientationSpec(
            actor="Target",
            allow_pitch=False,
            smoothing_time_s=0.5,
            max_yaw_rate_deg_s=60.0,
            max_pitch_rate_deg_s=30.0,
            yaw_offset_deg=1.0,
            pitch_offset_deg=2.0,
            roll_offset_deg=3.0,
            yaw_limits_deg=(-80.0, 85.0),
            pitch_limits_deg=(-40.0, 45.0),
        ).model_dump(mode="python"),
        SpinOrientationSpec(
            axis="roll",
            rate_deg_s=-25.0,
            yaw_deg=1.0,
            pitch_deg=2.0,
            roll_deg=3.0,
        ).model_dump(mode="python"),
        RandomOrientationSpec(
            seed=123,
            yaw_range_deg=(-10.0, 20.0),
            pitch_range_deg=(-30.0, 40.0),
            roll_range_deg=(-50.0, 60.0),
            update_interval_s=0.125,
        ).model_dump(mode="python"),
    )

    for source in mappings:
        orientation = orientation_from_mapping(
            source,
            actor_id_by_name={"Target": target_id},
        )
        serialized = orientation_to_mapping(
            orientation,
            {target_id: "Target"},
        )
        assert serialized == source


def test_reference_and_conversion_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="valid UUID"):
        actor_look_at_orientation("not-a-uuid")
    with pytest.raises(KeyError, match="unknown look-at actor id"):
        orientation_to_mapping(actor_look_at_orientation(uuid4()), ())
    with pytest.raises(ValueError, match="positive finite duration"):
        default_orientation("keyframes", duration_s=0.0)
    with pytest.raises(ValueError, match="actor or a point"):
        default_orientation(
            "look_at",
            target_actor_id=uuid4(),
            target_point_m=(0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="unsupported orientation kind"):
        default_orientation("teleport")
