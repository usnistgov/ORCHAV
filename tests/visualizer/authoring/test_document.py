"""Scenario Builder domain and document command tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from shared.scenarios.actors import (
    CircularMobilitySpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    LinearMobilitySpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringScenario,
    CommandStack,
    ScenarioCompiler,
    ScenarioDocument,
    SceneReference,
    TimelineSettings,
)
from visualizer.src.authoring.mobility_models import translate_mobility
from visualizer.src.authoring.orientation_models import (
    actor_look_at_orientation,
    look_at_actor_id,
)


def test_new_document_uses_contract_defaults_with_empty_scene() -> None:
    document = ScenarioDocument.new()

    assert document.scenario.scene == SceneReference("library", "empty/empty.xml")
    assert document.scenario.timeline == TimelineSettings()
    assert document.scenario.timeline.steps == 30
    assert document.scenario.timeline.duration_s == 3.0
    assert document.scenario.timeline.export_path_metrics is True
    assert document.revision == 0
    assert document.can_undo is False
    assert document.dirty is True


@pytest.mark.parametrize(
    ("mobility", "anchors"),
    (
        (
            StationaryMobilitySpec(position_m=(1, 2, 3)),
            ((3, 1, 3.5),),
        ),
        (
            LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6)),
            ((3, 1, 3.5), (6, 4, 6.5)),
        ),
        (
            WaypointMobilitySpec(points_m=((1, 2, 3), (4, 5, 6))),
            ((3, 1, 3.5), (6, 4, 6.5)),
        ),
        (
            CircularMobilitySpec(
                center_m=(1, 2, 3),
                radius_m=4.0,
                start_angle_deg=15.0,
                clockwise=False,
            ),
            ((3, 1, 3.5),),
        ),
    ),
)
def test_translate_mobility_moves_every_anchor_without_changing_shape(mobility, anchors) -> None:
    translated = translate_mobility(mobility, (2.0, -1.0, 0.5))

    if isinstance(translated, StationaryMobilitySpec):
        actual = (translated.position_m,)
    elif isinstance(translated, LinearMobilitySpec):
        actual = (translated.start_m, translated.end_m)
    elif isinstance(translated, WaypointMobilitySpec):
        actual = translated.points_m
    else:
        actual = (translated.center_m,)
        assert translated.radius_m == mobility.radius_m
        assert translated.start_angle_deg == mobility.start_angle_deg
        assert translated.clockwise == mobility.clockwise
    assert actual == anchors


def test_rename_preserves_uuid_lookat_reference_and_changes_serialized_name() -> None:
    target = AuthoringActor.create(ActorRole.RX, "RX1", position=(5, 0, 1))
    observer = replace(
        AuthoringActor.create(ActorRole.TX, "TX1", position=(0, 0, 1)),
        orientation=actor_look_at_orientation(target.id),
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(observer, target),
        )
    )

    document.rename_actor(target.id, "Receiver")

    persisted_observer = document.scenario.actor(observer.id)
    assert persisted_observer is not None
    assert look_at_actor_id(persisted_observer.orientation) == target.id
    mapping = ScenarioCompiler().compile(document).mapping
    assert mapping["actors"]["tx"][0]["orientation"] == {
        "type": "look_at",
        "actor": "Receiver",
    }


def test_delete_leaves_uuid_reference_explicitly_invalid_and_undo_restores() -> None:
    target = AuthoringActor.create(ActorRole.RX, "RX1", position=(5, 0, 1))
    observer = replace(
        AuthoringActor.create(ActorRole.TX, "TX1", position=(0, 0, 1)),
        orientation=actor_look_at_orientation(target.id),
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(observer, target),
        )
    )

    document.remove_actor(target.id)
    result = ScenarioCompiler().compile(document)
    assert any(issue.code == "orientation.look_at.missing_target" for issue in result.issues)

    document.undo()
    assert document.scenario.actor(target.id) == target
    restored_observer = document.scenario.actor(observer.id)
    assert restored_observer is not None
    assert look_at_actor_id(restored_observer.orientation) == target.id


def test_role_and_id_are_immutable_during_actor_replacement() -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)

    with pytest.raises(ValueError, match="role is immutable"):
        document.update_actor(actor.id, role=ActorRole.RX)
    with pytest.raises(ValueError, match="id is immutable"):
        document.update_actor(
            actor.id,
            id=AuthoringActor.create(ActorRole.TX, "Other").id,
        )


def test_transient_drag_commits_as_one_undo_entry_and_cancel_is_lossless() -> None:
    stack = CommandStack()
    document = ScenarioDocument.new(undo_stack=stack)
    actor = document.add_default_actor(ActorRole.TX)
    stack.clear()
    original = document.scenario.actor(actor.id)
    assert original is not None

    document.begin_transient_edit("Move TX1")
    document.update_transient_actor(
        original.with_changes(mobility=StationaryMobilitySpec(position_m=(1, 0, 0)))
    )
    document.update_transient_actor(
        original.with_changes(mobility=StationaryMobilitySpec(position_m=(2, 0, 0)))
    )
    assert document.revision == 1
    document.commit_transient_edit()

    assert stack.count == 1
    moved_actor = document.scenario.actor(actor.id)
    assert moved_actor is not None
    assert moved_actor.mobility == StationaryMobilitySpec(position_m=(2, 0, 0))
    document.undo()
    assert document.scenario.actor(actor.id) == original

    document.begin_transient_edit("Move TX1")
    document.update_transient_actor(
        original.with_changes(mobility=StationaryMobilitySpec(position_m=(9, 0, 0)))
    )
    document.cancel_transient_edit()
    assert document.scenario.actor(actor.id) == original
    assert stack.count == 1


def test_transient_state_cannot_be_marked_saved(tmp_path: Path) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.begin_transient_edit("Move TX1")
    document.update_transient_actor(
        actor.with_changes(mobility=StationaryMobilitySpec(position_m=(4.0, 0.0, 0.0)))
    )

    with pytest.raises(RuntimeError, match="transient edit"):
        document.mark_saved(tmp_path / "scenario.yaml")

    assert document.dirty is True
    document.cancel_transient_edit()
    assert document.scenario.actor(actor.id) == actor


def test_saved_scenario_becomes_dirty_and_undo_returns_to_clean(tmp_path: Path) -> None:
    scenario = AuthoringScenario(scene=SceneReference("library", "empty/empty.xml"))
    document = ScenarioDocument.loaded(scenario, tmp_path / "scenario.yaml")
    assert document.dirty is False

    document.set_timeline(replace(document.scenario.timeline, duration_s=4.0))
    assert document.dirty is True
    document.undo()
    assert document.dirty is False


def test_group_rename_preserves_uuid_member_references_and_serializes_current_name() -> None:
    group = AuthoringGroup.create("Formation")
    tx = AuthoringActor.create(ActorRole.TX, "TX1").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-1.0),
        )
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX1").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=1.0),
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(tx, rx),
            groups=(group,),
        )
    )

    document.rename_group(group.id, "Renamed Formation")

    current_tx = document.scenario.actor(tx.id)
    assert current_tx is not None
    assert current_tx.mobility.group == str(group.id)
    mapping = ScenarioCompiler().compile(document).mapping
    assert mapping["groups"][0]["name"] == "Renamed Formation"
    assert mapping["actors"]["tx"][0]["mobility"]["group"] == "Renamed Formation"
