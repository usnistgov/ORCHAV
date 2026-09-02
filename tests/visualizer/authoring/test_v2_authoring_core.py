"""End-to-end authoring coverage for shared actor models and groups."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import yaml

from shared.scenarios.actors import (
    ActorRole,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    MeshSequenceMobilitySpec,
    RandomOrientationSpec,
    StationaryMobilitySpec,
)
from visualizer.src.authoring.compiler import ScenarioCompiler, canonical_scenario_mapping
from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.domain import (
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    ResourceKind,
    SceneReference,
    TargetAsset,
    TimelineSettings,
)
from visualizer.src.authoring.persistence import (
    load_for_authoring,
    resource_relative_path,
    save_document,
    scenario_from_mapping,
)


def _base_actors() -> tuple[AuthoringActor, AuthoringActor]:
    return (
        AuthoringActor.create(ActorRole.TX, "TX", position=(0.0, 0.0, 1.0)),
        AuthoringActor.create(ActorRole.RX, "RX", position=(1.0, 0.0, 1.0)),
    )


def test_new_document_omits_redundant_file_data_defaults() -> None:
    mapping = canonical_scenario_mapping(ScenarioDocument.new().scenario)

    assert "data" not in mapping


def test_group_and_actor_references_serialize_current_names_and_reopen_as_uuids() -> None:
    group = AuthoringGroup.create("Formation").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        )
    )
    tx, rx = _base_actors()
    tx = tx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-1.0),
        ),
        orientation=LookAtOrientationSpec(actor=str(rx.id)),
    )
    rx = rx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=1.0),
        ),
        orientation=RandomOrientationSpec(seed=42),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx, rx),
        groups=(group,),
    )

    mapping = canonical_scenario_mapping(scenario)

    assert mapping["groups"][0]["name"] == "Formation"
    assert mapping["actors"]["tx"][0]["mobility"]["group"] == "Formation"
    assert mapping["actors"]["tx"][0]["orientation"]["actor"] == "RX"

    reopened = scenario_from_mapping(mapping)
    reopened_tx = reopened.actor_by_name("TX")
    reopened_rx = reopened.actor_by_name("RX")
    reopened_group = reopened.group_by_name("Formation")
    assert reopened_tx is not None and reopened_rx is not None
    assert reopened_group is not None
    assert UUID(reopened_tx.mobility.group) == reopened_group.id
    assert UUID(reopened_tx.orientation.actor or "") == reopened_rx.id


def test_dangling_uuid_references_do_not_fall_back_to_uuid_shaped_names() -> None:
    missing_actor_id = uuid4()
    missing_group_id = uuid4()
    tx = AuthoringActor.create(ActorRole.TX, "TX").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(missing_group_id),
            offset_m=GroupOffsetSpec(),
        ),
        orientation=LookAtOrientationSpec(actor=str(missing_actor_id)),
    )
    rx = AuthoringActor.create(ActorRole.RX, str(missing_actor_id))
    misleading_group = AuthoringGroup.create(str(missing_group_id))
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx, rx),
        groups=(misleading_group,),
    )

    mapping = canonical_scenario_mapping(scenario)
    result = ScenarioCompiler().compile(scenario)

    assert mapping["actors"]["tx"][0]["mobility"]["group"] == ""
    assert mapping["actors"]["tx"][0]["orientation"]["actor"] == ""
    assert any(issue.code == "mobility.group_member.missing_group" for issue in result.issues)
    assert any(issue.code == "orientation.look_at.missing_target" for issue in result.issues)


def test_group_compile_uses_generator_prepared_group_and_member_samples() -> None:
    group = AuthoringGroup.create("Pair").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(5.0, 0.0, 1.0),
        )
    )
    tx, rx = _base_actors()
    tx = tx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-2.0),
        )
    )
    rx = rx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=2.0),
        )
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=5, duration_s=2.0),
        actors=(tx, rx),
        groups=(group,),
    )

    result = ScenarioCompiler().compile(scenario)

    assert result.valid
    assert result.group_samples[group.id].positions[0] == (0.0, 0.0, 1.0)
    assert result.samples[tx.id].positions[0] == (0.0, 2.0, 1.0)
    assert result.samples[rx.id].positions[0] == (0.0, -2.0, 1.0)
    assert result.group_offsets(
        group.id,
        (
            result.samples[tx.id].positions[0],
            result.samples[rx.id].positions[0],
        ),
        step=0,
    ) == ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_incomplete_group_still_has_generator_prepared_preview_samples() -> None:
    group = AuthoringGroup.create("DraftGroup").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(1.0, 2.0, 3.0),
            end_m=(4.0, 5.0, 6.0),
        )
    )
    result = ScenarioCompiler().compile(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            groups=(group,),
        )
    )

    assert not result.valid
    assert result.group_samples[group.id].positions[0] == (1.0, 2.0, 3.0)
    assert result.group_samples[group.id].positions[-1] == (4.0, 5.0, 6.0)


def test_incomplete_resource_group_uses_registered_source_for_preview(
    tmp_path: Path,
) -> None:
    source = tmp_path / "group_positions.npy"
    np.save(
        source,
        np.asarray(((1.0, 0.0, 2.0), (5.0, 0.0, 2.0)), dtype=np.float64),
    )
    relative = resource_relative_path(source)
    group = AuthoringGroup.create("DraftResourceGroup").with_changes(
        mobility=MeshSequenceMobilitySpec(positions_path=relative),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=4, duration_s=2.0),
        groups=(group,),
        resources=(
            AuthoringResource(
                ResourceKind.POSITION_SEQUENCE,
                source,
                relative,
            ),
        ),
    )

    result = ScenarioCompiler().compile(
        scenario,
        scenario_directory=tmp_path / "draft",
    )
    offsets = ScenarioCompiler.group_offsets(
        group.mobility,
        scenario.timeline.steps,
        scenario.timeline.duration_s,
        ((1.0, 2.0, 2.0),),
        scenario_directory=tmp_path / "draft",
        resources=scenario.resources,
    )

    assert not result.valid
    assert result.group_samples[group.id].positions[0] == (1.0, 0.0, 2.0)
    assert result.group_samples[group.id].positions[-1] == (5.0, 0.0, 2.0)
    assert len(offsets) == 1


def test_resource_backed_target_is_copied_saved_reopened_and_prepared(
    tmp_path: Path,
    authoring_project_root: Path,
) -> None:
    source = tmp_path / "external_positions.npy"
    positions = np.asarray(
        ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (4.0, 1.0, 1.0)),
        dtype=np.float64,
    )
    np.save(source, positions)
    relative = resource_relative_path(source)
    tx, rx = _base_actors()
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "MovingTarget",
        target=TargetAsset.from_catalog_id("cube"),
    ).with_changes(
        mobility=MeshSequenceMobilitySpec(positions_path=relative),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=5, duration_s=2.0),
        actors=(tx, rx, target),
        resources=(
            AuthoringResource(
                ResourceKind.POSITION_SEQUENCE,
                source,
                relative,
            ),
        ),
    )
    document = ScenarioDocument(scenario)
    destination = authoring_project_root / "saved"
    compiler = ScenarioCompiler(authoring_project_root)

    saved = save_document(document, destination, compiler=compiler)

    mapping = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert mapping["actors"]["targets"][0]["mobility"]["positions_path"] == relative
    assert (destination / relative).read_bytes() == source.read_bytes()
    assert document.scenario.resources[0].source_path == (destination / relative).resolve()

    source.unlink()
    document.set_timeline(TimelineSettings(steps=6, duration_s=3.0))
    save_document(document, compiler=compiler)
    assert (destination / relative).is_file()

    reopened = load_for_authoring(destination)
    assert reopened.document is not None
    result = compiler.compile(
        reopened.document.scenario,
        scenario_directory=destination,
    )
    reopened_target = reopened.document.scenario.actor_by_name("MovingTarget")
    assert reopened_target is not None
    assert result.valid
    assert result.samples[reopened_target.id].positions[0] == (0.0, 0.0, 1.0)


def test_document_prunes_resources_when_their_mobility_is_replaced() -> None:
    tx, rx = _base_actors()
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Target",
        target=TargetAsset.from_catalog_id("cube"),
    ).with_changes(mobility=MeshSequenceMobilitySpec(positions_path="resources/positions.npy"))
    resource = AuthoringResource(
        ResourceKind.POSITION_SEQUENCE,
        Path(__file__),
        "resources/positions.npy",
    )
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(tx, rx, target),
            resources=(resource,),
        )
    )

    document.replace_actor(
        target.with_changes(mobility=StationaryMobilitySpec(position_m=(0.0, 0.0, 0.0)))
    )

    assert document.scenario.resources == ()
    document.undo()
    assert document.scenario.resources == (resource,)
