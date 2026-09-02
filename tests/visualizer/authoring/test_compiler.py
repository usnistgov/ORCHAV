"""Canonical compiler and generator-prepared preview parity tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shared.scenarios.actors import (
    AlignMotionOrientationSpec,
    CircularMobilitySpec,
    FixedOrientationSpec,
    GroupMemberMobilitySpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    OrientationKeyframeSpec,
    OscillatingMobilitySpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringScenario,
    IssueSeverity,
    ScenarioCompiler,
    SceneReference,
    TargetAsset,
    TimelineSettings,
)
from visualizer.src.authoring.mobility_models import (
    MOBILITY_MODELS,
    MobilityKind,
    default_mobility,
)
from visualizer.src.authoring.orientation_models import (
    OrientationKind,
    actor_look_at_orientation,
    point_look_at_orientation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


CONTEXT_FREE_MOBILITY_CASES = tuple(
    pytest.param(kind, role, id=f"{kind.value}-{role.value}")
    for kind, descriptor in MOBILITY_MODELS.items()
    if not descriptor.requires_context
    for role in sorted(descriptor.allowed_roles, key=lambda actor_role: actor_role.value)
)


def _orientation_for_compile(
    kind: OrientationKind,
    look_at_target: AuthoringActor,
):
    if kind is OrientationKind.FIXED:
        return FixedOrientationSpec(
            yaw_deg=12.0,
            pitch_deg=3.0,
            roll_deg=-2.0,
        )
    if kind is OrientationKind.KEYFRAMES:
        return KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(
                    time_s=4.0,
                    yaw_deg=90.0,
                    pitch_deg=-10.0,
                    roll_deg=5.0,
                ),
            )
        )
    if kind is OrientationKind.ALIGN_MOTION:
        return AlignMotionOrientationSpec(
            allow_pitch=False,
            smoothing_time_s=0.25,
            yaw_offset_deg=10.0,
            max_yaw_rate_deg_s=180.0,
        )
    if kind is OrientationKind.LOOK_AT:
        return actor_look_at_orientation(
            look_at_target.id,
            allow_pitch=False,
            smoothing_time_s=0.25,
            yaw_offset_deg=5.0,
        )
    if kind is OrientationKind.SPIN:
        return SpinOrientationSpec(
            axis="yaw",
            rate_deg_s=135.0,
            yaw_deg=-45.0,
            pitch_deg=10.0,
            roll_deg=-5.0,
        )
    return RandomOrientationSpec(
        seed=17,
        yaw_range_deg=(-90.0, 90.0),
        pitch_range_deg=(-30.0, 30.0),
        roll_range_deg=(-10.0, 10.0),
        update_interval_s=0.5,
    )


def _scenario_with_subject(role: ActorRole, mobility) -> tuple[AuthoringScenario, AuthoringActor]:
    subject = replace(
        AuthoringActor.create(
            role,
            {
                ActorRole.TX: "TXSubject",
                ActorRole.RX: "RXSubject",
                ActorRole.TARGET: "TargetSubject",
            }[role],
            target=(
                TargetAsset.from_catalog_id("cube", mesh_pattern="cube.ply")
                if role is ActorRole.TARGET
                else None
            ),
        ),
        mobility=mobility,
        orientation=FixedOrientationSpec(
            yaw_deg=12.0,
            pitch_deg=3.0,
            roll_deg=-2.0,
        ),
    )
    actors: list[AuthoringActor] = []
    if role is not ActorRole.TX:
        actors.append(AuthoringActor.create(ActorRole.TX, "TXBase", position=(-2, 0, 1)))
    actors.append(subject)
    if role is not ActorRole.RX:
        actors.append(AuthoringActor.create(ActorRole.RX, "RXBase", position=(8, 0, 1)))
    return (
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            timeline=TimelineSettings(steps=7, duration_s=4.0),
            actors=tuple(actors),
        ),
        subject,
    )


def test_canonical_mapping_uses_default_file_output_settings() -> None:
    scenario, _subject = _scenario_with_subject(
        ActorRole.RX,
        StationaryMobilitySpec(position_m=(0.0, 0.0, 1.0)),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert result.valid, result.issues
    assert "data" not in result.mapping


def test_generation_issues_reject_summary_only_workflows() -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)

    issues = compiler._generation_issues(
        {
            "raytracing": {"enabled": False},
            "generator_summary": {"enabled": True, "create": ["scene2d"]},
        }
    )

    assert {issue.code for issue in issues} == {"generation.frames.not_requested"}


@pytest.mark.parametrize(("mobility_kind", "role"), CONTEXT_FREE_MOBILITY_CASES)
def test_every_context_free_mobility_compiles_for_each_allowed_role(
    mobility_kind: MobilityKind,
    role: ActorRole,
) -> None:
    mobility = default_mobility(
        mobility_kind,
        (2.0, 3.0, 1.0),
        duration_s=4.0,
        seed=23,
    )
    scenario, subject = _scenario_with_subject(role, mobility)

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert result.valid, result.issues
    assert result.runtime is not None
    configs = {
        ActorRole.TX: result.runtime.transmitters,
        ActorRole.RX: result.runtime.receivers,
        ActorRole.TARGET: result.runtime.targets,
    }[role]
    config = next(config for config in configs if config.name == subject.name)
    section = "targets" if role is ActorRole.TARGET else role.value
    mapped = next(
        entry for entry in result.mapping["actors"][section] if entry["name"] == subject.name
    )
    assert mapped["mobility"]["type"] == mobility_kind.value
    expected_positions = tuple(tuple(point) for point in config.mobility.prepared_positions())
    expected_orientations = tuple(tuple(angles) for angles in config.orientation.orientations())
    np.testing.assert_allclose(result.samples[subject.id].positions, expected_positions)
    np.testing.assert_allclose(result.samples[subject.id].orientations, expected_orientations)


@pytest.mark.parametrize("role", list(ActorRole), ids=lambda role: role.value)
@pytest.mark.parametrize(
    "orientation_kind",
    list(OrientationKind),
    ids=lambda kind: kind.value,
)
def test_every_orientation_compiles_for_each_actor_role(
    role: ActorRole,
    orientation_kind: OrientationKind,
) -> None:
    scenario, subject = _scenario_with_subject(
        role,
        LinearMobilitySpec(
            start_m=(2.0, 3.0, 1.0),
            end_m=(6.0, 3.0, 1.0),
        ),
    )
    look_at_target = next(actor for actor in scenario.actors if actor.id != subject.id)
    orientation = _orientation_for_compile(orientation_kind, look_at_target)
    subject = replace(subject, orientation=orientation)
    scenario = replace(
        scenario,
        actors=tuple(subject if actor.id == subject.id else actor for actor in scenario.actors),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert result.valid, result.issues
    assert result.runtime is not None
    section = "targets" if role is ActorRole.TARGET else role.value
    mapped_actors = result.mapping["actors"][section]
    mapped = next(entry for entry in mapped_actors if entry["name"] == subject.name)
    expected_mapping = orientation.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )
    expected_mapping = {
        "type": orientation_kind.value,
        **{key: value for key, value in expected_mapping.items() if key != "type"},
    }
    if orientation_kind is OrientationKind.LOOK_AT:
        expected_mapping["actor"] = look_at_target.name
    assert mapped["orientation"] == expected_mapping

    configs = {
        ActorRole.TX: result.runtime.transmitters,
        ActorRole.RX: result.runtime.receivers,
        ActorRole.TARGET: result.runtime.targets,
    }[role]
    config = next(config for config in configs if config.name == subject.name)
    expected = tuple(tuple(angles) for angles in config.orientation.orientations())
    np.testing.assert_allclose(result.samples[subject.id].orientations, expected)


@pytest.mark.parametrize(
    "mobility",
    (
        LinearMobilitySpec(
            start_m=(2.0, 3.0, 1.0),
            end_m=(2.0, 3.0, 1.0),
        ),
        OscillatingMobilitySpec(
            center_m=(2.0, 3.0, 1.0),
            axis=(1.0, 0.0, 0.0),
            amplitude_m=0.0,
            frequency_hz=1.0,
        ),
    ),
    ids=("zero-length-linear", "zero-amplitude-oscillating"),
)
def test_align_motion_rejects_prepared_trajectories_without_physical_motion(
    mobility,
) -> None:
    scenario, subject = _scenario_with_subject(ActorRole.RX, mobility)
    subject = replace(subject, orientation=AlignMotionOrientationSpec())
    scenario = replace(
        scenario,
        actors=tuple(subject if actor.id == subject.id else actor for actor in scenario.actors),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    issue = next(
        issue
        for issue in result.issues
        if issue.code == "orientation.align_motion.no_physical_motion"
    )
    assert not result.valid
    assert issue.path == "actors.rx.0.orientation"
    assert issue.actor_id == subject.id


def test_align_motion_rejects_a_constant_prepared_group_path() -> None:
    group = AuthoringGroup.create("Formation").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(1.0, 2.0, 3.0),
            end_m=(1.0, 2.0, 3.0),
        )
    )
    tx = replace(
        AuthoringActor.create(ActorRole.TX, "TX1"),
        mobility=GroupMemberMobilitySpec(group=str(group.id)),
    )
    rx = replace(
        AuthoringActor.create(ActorRole.RX, "RX1"),
        mobility=GroupMemberMobilitySpec(group=str(group.id)),
        orientation=AlignMotionOrientationSpec(),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=7, duration_s=4.0),
        actors=(tx, rx),
        groups=(group,),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    issue = next(
        issue
        for issue in result.issues
        if issue.code == "orientation.align_motion.no_physical_motion"
    )
    assert not result.valid
    assert issue.path == "actors.rx.0.orientation"
    assert issue.actor_id == rx.id


@pytest.mark.parametrize("target_kind", ("actor", "point"))
@pytest.mark.parametrize("scene_selected", (True, False), ids=("complete", "partial-preview"))
def test_coincident_look_at_forms_share_hold_warning_semantics(
    target_kind: str,
    scene_selected: bool,
) -> None:
    position = (2.0, 3.0, 1.0)
    tx = AuthoringActor.create(ActorRole.TX, "TX1", position=position)
    orientation = (
        actor_look_at_orientation(tx.id, yaw_offset_deg=17.0)
        if target_kind == "actor"
        else point_look_at_orientation(position, yaw_offset_deg=17.0)
    )
    rx = replace(
        AuthoringActor.create(ActorRole.RX, "RX1", position=position),
        orientation=orientation,
    )
    scenario = AuthoringScenario(
        scene=(SceneReference("library", "empty/empty.xml") if scene_selected else None),
        timeline=TimelineSettings(steps=3, duration_s=2.0),
        actors=(tx, rx),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    warning = next(
        issue for issue in result.issues if issue.code == "orientation.look_at.coincident"
    )
    target_field = "actor" if target_kind == "actor" else "point_m"
    assert warning.severity is IssueSeverity.WARNING
    assert warning.path == f"actors.rx.0.orientation.{target_field}"
    assert warning.actor_id == rx.id
    assert "hold semantics" in warning.message
    assert rx.id in result.samples
    np.testing.assert_allclose(
        result.samples[rx.id].orientations,
        ((17.0, 0.0, 0.0),) * 3,
    )
    assert result.valid is scene_selected


def test_circular_inspector_degrees_are_serialized_as_degrees() -> None:
    circular = CircularMobilitySpec(
        center_m=(1, 2, 3),
        radius_m=4.0,
        start_angle_deg=90.0,
        clockwise=True,
    )
    scenario, subject = _scenario_with_subject(ActorRole.TX, circular)

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)

    assert result.mapping["actors"]["tx"][0]["mobility"]["start_angle_deg"] == pytest.approx(90.0)
    assert circular.start_angle_deg == pytest.approx(90.0)
    assert result.samples[subject.id].positions[0] == pytest.approx((1.0, 6.0, 3.0))


def test_validation_returns_exact_paths_for_core_contract_errors() -> None:
    tx = replace(
        AuthoringActor.create(ActorRole.TX, "same"),
        mobility=WaypointMobilitySpec(points_m=((0, 0, 0), (1, 0, 0))),
        orientation=AlignMotionOrientationSpec(),
    )
    rx = AuthoringActor.create(ActorRole.RX, "same", position=(2, 0, 0))
    target = replace(
        AuthoringActor.create(
            ActorRole.TARGET,
            "Target1",
            target=TargetAsset(
                asset_id="cube",
                mesh_directory="libraries/targets/cube",
                mesh_pattern="cube.ply",
                scale=0.0,
            ),
        ),
        mobility=CircularMobilitySpec(center_m=(0, 0, 0), radius_m=1.0),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=1, duration_s=0.0),
        actors=(tx, rx, target),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)
    by_code = {issue.code: issue.path for issue in result.issues}

    assert by_code["actor.name.duplicate"] == "actors.rx.0.name"
    assert by_code["target.scale.positive"] == "actors.targets.0.asset.scale"
    assert by_code["timeline.moving_steps"] == "timeline.steps"
    assert by_code["timeline.moving_duration"] == "timeline.duration_s"


def test_catalog_asset_id_and_directory_must_match_exactly() -> None:
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Target1",
        target=TargetAsset(
            asset_id="cube",
            mesh_directory="libraries/targets/car",
            mesh_pattern="car.ply",
        ),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
            target,
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(issue for issue in result.issues if issue.code == "target.asset.catalog_mismatch")
    assert issue.path == "actors.targets.0.asset.id"
    assert issue.actor_id == target.id


def test_target_material_must_exist_in_generator_sionna_registry() -> None:
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Target1",
        target=TargetAsset.from_catalog_id(
            "cube",
            mesh_pattern="cube.ply",
            material="not_a_real_itu_material",
        ),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
            target,
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(issue for issue in result.issues if issue.code == "target.material.unsupported")
    assert issue.path == "actors.targets.0.asset.material_type"
    assert issue.actor_id == target.id
    assert "not_a_real_itu_material" in issue.message


def test_catalog_mesh_pattern_cannot_escape_selected_asset() -> None:
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Target1",
        target=TargetAsset.from_catalog_id("cube", mesh_pattern="../car/car.ply"),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
            target,
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(
        issue for issue in result.issues if issue.code == "target.asset.pattern_not_catalog"
    )
    assert issue.path == "actors.targets.0.asset.id"
    assert issue.actor_id == target.id


def test_local_scene_requires_an_xml_asset() -> None:
    scenario = AuthoringScenario(
        scene=SceneReference("local", "README.md"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=PROJECT_ROOT)

    assert any(issue.code == "scene.asset.not_xml" for issue in result.issues)


def test_relative_local_scene_validation_never_writes_through_its_id(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    source = tmp_path / "sibling.xml"
    source.write_bytes((PROJECT_ROOT / "libraries/scenes/empty/empty.xml").read_bytes())
    old_escape_destination = scenario_root / "sibling.xml"
    sentinel = b"scenario-owned sentinel"
    old_escape_destination.write_bytes(sentinel)
    scenario = AuthoringScenario(
        scene=SceneReference("local", "../sibling.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=scenario_root,
    )

    assert result.valid, result.issues
    assert result.mapping["scene"] == {
        "source": "local",
        "id": "../sibling.xml",
    }
    assert result.resolved_scene_path == str(source.resolve())
    assert old_escape_destination.read_bytes() == sentinel


def test_generator_owned_exceptions_become_structured_validation_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, _subject = _scenario_with_subject(
        ActorRole.TX,
        StationaryMobilitySpec(position_m=(0, 0, 1)),
    )
    compiler = ScenarioCompiler(PROJECT_ROOT)

    class GeneratorBoundaryError(Exception):
        pass

    def fail_construction(*_args, **_kwargs):
        raise GeneratorBoundaryError("construction failed")

    monkeypatch.setattr(compiler, "_construct_and_prepare", fail_construction)

    result = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    assert result.runtime is None
    assert result.samples == {}
    assert any(
        issue.code == "generator.invalid" and "construction failed" in issue.message
        for issue in result.issues
    )


def test_resolved_target_metadata_offset_matches_generator_when_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = replace(
        AuthoringActor.create(
            ActorRole.TARGET,
            "Car",
            target=TargetAsset.from_catalog_id("car", mesh_pattern="car.ply"),
        ),
        mobility=StationaryMobilitySpec(position_m=(2, 0, 0)),
        orientation=FixedOrientationSpec(
            yaw_deg=5.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
            target,
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(scenario, scenario_directory=tmp_path)

    assert result.valid, result.issues
    assert result.runtime is not None
    runtime_target = result.runtime.targets[0]
    expected = tuple(tuple(value) for value in runtime_target.orientation.orientations())
    assert result.samples[target.id].orientations == expected
    assert result.samples[target.id].orientations[0] == pytest.approx((-175.0, 0.0, 0.0))
