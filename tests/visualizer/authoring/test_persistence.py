"""Ownership, compatibility, semantic reopen, and atomic-write tests."""

import errno
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import networkx as nx
import pytest
import yaml

from generator.core.configuration.defaults import (
    DEFAULT_EXPORT_PATH_METRICS,
    DEFAULT_QUALITY_PRESET,
)
from shared.scenarios.actors import (
    AlignMotionOrientationSpec,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    FixedOrientationSpec,
    GroupMemberMobilitySpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    NetworkRouteMobilitySpec,
    OrientationKeyframeSpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    SampledMobilitySpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    CompatibilityAnalyzer,
    LoadDisposition,
    QualityPreset,
    ScenarioCompiler,
    ScenarioDocument,
    ScenarioSaveError,
    ScenarioSourceSnapshot,
    SceneReference,
    TargetAsset,
    TimelineSettings,
    atomic_write_scenario_yaml,
    canonical_scenario_mapping,
    create_scenario_copy,
    load_for_authoring,
    save_document,
    scenario_from_mapping,
)
from visualizer.src.authoring import persistence as persistence_module
from visualizer.src.authoring.domain import ResourceKind
from visualizer.src.authoring.orientation_models import look_at_actor_id
from visualizer.src.authoring.persistence import resource_relative_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).with_name("fixtures")


def _external_mapping() -> dict:
    return {
        "schema_version": 2,
        "scene": {"source": "library", "id": "empty/empty.xml"},
        "timeline": {"steps": 30, "duration_s": 3.0},
        "raytracing": {"enabled": True},
        "actors": {
            "tx": [
                {
                    "name": "TX1",
                    "mobility": {"type": "stationary", "position_m": [0, 0, 1]},
                }
            ],
            "rx": [
                {
                    "name": "RX1",
                    "mobility": {"type": "stationary", "position_m": [2, 0, 1]},
                }
            ],
        },
    }


def _known_passthrough_mapping() -> dict:
    mapping = _external_mapping()
    mapping["debug_level"] = "WARNING"
    mapping["view_defaults"] = {
        "camera_view": "isometric",
        "merge_scene_meshes": True,
    }
    mapping["raytracing"].update(
        {
            "export_path_metrics": True,
            "quality": {
                "preset": "custom",
                "custom": {
                    "max_depth": 4,
                    "samples_per_src": 75_000,
                    "diffuse_reflection": True,
                    "seed": 42,
                },
            },
            "materials": {
                "concrete": {"scattering_coefficient": 0.3},
                "metal": {"scattering_coefficient": 0.2},
            },
        }
    )
    return mapping


def _write_yaml(path: Path, mapping: dict) -> None:
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")


def _create_directory_link(link: Path, target: Path) -> None:
    """Create a directory symlink, using a Windows junction when necessary."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return
        message = f"{completed.stdout}\n{completed.stderr}".strip()
        if "access is denied" in message.lower() or "privilege" in message.lower():
            pytest.skip(f"directory link creation is not permitted: {message}")
        pytest.fail(f"could not create a Windows directory junction: {message}")
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM}:
            pytest.skip(f"directory symlink creation is not permitted: {exc}")
        raise


def _with_resource_mobility(
    *,
    actor_path: bool,
    resource_path: str,
) -> dict:
    mapping = _external_mapping()
    mapping["visualizer"] = {"scenario_builder": {"document_version": 2}}
    mobility = {
        "type": "network_route",
        "graph_path": resource_path,
        "start_node": "start",
        "end_node": "end",
    }
    if actor_path:
        mapping["actors"]["tx"][0]["mobility"] = mobility
    else:
        mapping["groups"] = [{"name": "RouteGroup", "mobility": mobility}]
        for section in ("tx", "rx"):
            mapping["actors"][section][0]["mobility"] = {
                "type": "group_member",
                "group": "RouteGroup",
            }
    return mapping


@pytest.mark.parametrize(
    ("fixture_name", "disposition", "problem_paths"),
    [
        ("supported_owned.yaml", LoadDisposition.OWNED_EDITABLE, set()),
        ("compatible_external.yaml", LoadDisposition.OWNED_EDITABLE, set()),
        (
            "unsupported_external.yaml",
            LoadDisposition.OWNED_EDITABLE,
            set(),
        ),
        (
            "future_owned.yaml",
            LoadDisposition.READ_ONLY,
            {"visualizer.scenario_builder.document_version"},
        ),
    ],
)
def test_authoring_compatibility_fixtures(
    fixture_name: str,
    disposition: LoadDisposition,
    problem_paths: set[str],
) -> None:
    mapping = yaml.safe_load((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"))

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is disposition
    assert {issue.path for issue in report.issues} == problem_paths


def test_supported_fixture_parses_actor_models_and_catalog_target() -> None:
    mapping = yaml.safe_load((FIXTURE_ROOT / "supported_owned.yaml").read_text(encoding="utf-8"))

    scenario = scenario_from_mapping(mapping)

    assert isinstance(scenario.actors[0].mobility, StationaryMobilitySpec)
    assert isinstance(scenario.actors[0].orientation, FixedOrientationSpec)
    assert isinstance(scenario.actors[1].mobility, LinearMobilitySpec)
    assert isinstance(scenario.actors[1].orientation, AlignMotionOrientationSpec)
    assert isinstance(scenario.actors[2].mobility, WaypointMobilitySpec)
    assert isinstance(scenario.actors[2].orientation, LookAtOrientationSpec)
    assert isinstance(scenario.actors[3].mobility, CircularMobilitySpec)
    assert scenario.actors[3].target == TargetAsset.from_catalog_id("cube")
    assert look_at_actor_id(scenario.actors[2].orientation) == scenario.actors[3].id


def test_each_reopen_rebuilds_look_at_references_with_fresh_actor_ids() -> None:
    mapping = yaml.safe_load((FIXTURE_ROOT / "supported_owned.yaml").read_text(encoding="utf-8"))

    first = scenario_from_mapping(mapping)
    second = scenario_from_mapping(mapping)

    first_rx = first.actor_by_name("RX Waypoint")
    first_target = first.actor_by_name("Target Circular")
    second_rx = second.actor_by_name("RX Waypoint")
    second_target = second.actor_by_name("Target Circular")
    assert first_rx is not None and isinstance(first_rx.orientation, LookAtOrientationSpec)
    assert first_target is not None
    assert second_rx is not None and isinstance(second_rx.orientation, LookAtOrientationSpec)
    assert second_target is not None
    assert look_at_actor_id(first_rx.orientation) == first_target.id
    assert look_at_actor_id(second_rx.orientation) == second_target.id
    assert first_target.id != second_target.id


def test_noncanonical_yaml_is_read_only_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "external.yaml"
    original_mapping = _external_mapping()
    _write_yaml(source, original_mapping)
    original_bytes = source.read_bytes()

    loaded = load_for_authoring(source)

    assert loaded.disposition is LoadDisposition.READ_ONLY
    assert loaded.document is None
    assert loaded.scenario is not None
    assert {issue.code for issue in loaded.issues} == {"compatibility.canonical_filename_required"}
    assert source.read_bytes() == original_bytes


def test_unmarked_canonical_yaml_opens_in_place_without_writing_and_save_adds_marker(
    authoring_project_root: Path,
) -> None:
    scenario_root = authoring_project_root / "scenario"
    scenario_root.mkdir()
    source = scenario_root / "scenario.yaml"
    mapping = _known_passthrough_mapping()
    _write_yaml(source, mapping)
    original_bytes = source.read_bytes()

    loaded = load_for_authoring(source)

    assert loaded.disposition is LoadDisposition.OWNED_EDITABLE
    assert loaded.document is not None
    assert loaded.document.path == source.resolve()
    assert not loaded.document.dirty
    assert source.read_bytes() == original_bytes

    saved = save_document(
        loaded.document,
        compiler=ScenarioCompiler(authoring_project_root),
    )
    saved_mapping = yaml.safe_load(saved.read_text(encoding="utf-8"))

    assert saved == source.resolve()
    assert saved_mapping["visualizer"]["scenario_builder"] == {"document_version": 2}
    assert saved_mapping["debug_level"] == mapping["debug_level"]
    assert saved_mapping["view_defaults"] == mapping["view_defaults"]
    assert saved_mapping["raytracing"]["materials"] == mapping["raytracing"]["materials"]


@pytest.mark.parametrize(
    ("actor_path", "expected_path"),
    (
        (True, "actors.tx.0.mobility.graph_path"),
        (False, "groups.0.mobility.graph_path"),
    ),
)
def test_load_preserves_canonical_resource_link_escape_for_save_diagnostics(
    tmp_path: Path,
    actor_path: bool,
    expected_path: str,
) -> None:
    scenario_root = tmp_path / "scenario"
    outside_root = tmp_path / "outside"
    scenario_root.mkdir()
    outside_root.mkdir()
    (outside_root / "network.graphml").write_text("outside", encoding="utf-8")
    _create_directory_link(scenario_root / "resources", outside_root)
    scenario_yaml = scenario_root / "scenario.yaml"
    _write_yaml(
        scenario_yaml,
        _with_resource_mobility(
            actor_path=actor_path,
            resource_path="resources/network.graphml",
        ),
    )

    loaded = load_for_authoring(scenario_yaml)

    assert loaded.disposition is LoadDisposition.OWNED_EDITABLE
    assert loaded.scenario is not None
    dependency = next(
        dependency
        for dependency in loaded.scenario.dependencies
        if dependency.origin_path == expected_path
    )
    assert "outside the source scenario" in dependency.problem


@pytest.mark.parametrize(
    ("actor_path", "expected_path"),
    (
        (True, "actors.tx.0.mobility.graph_path"),
        (False, "groups.0.mobility.graph_path"),
    ),
)
def test_load_preserves_resource_parent_traversal_for_save_diagnostics(
    tmp_path: Path,
    actor_path: bool,
    expected_path: str,
) -> None:
    scenario_yaml = tmp_path / "scenario.yaml"
    _write_yaml(
        scenario_yaml,
        _with_resource_mobility(
            actor_path=actor_path,
            resource_path="resources/../network.graphml",
        ),
    )

    loaded = load_for_authoring(scenario_yaml)

    assert loaded.disposition is LoadDisposition.OWNED_EDITABLE
    assert loaded.scenario is not None
    dependency = next(
        dependency
        for dependency in loaded.scenario.dependencies
        if dependency.origin_path == expected_path
    )
    assert "parent traversal" in dependency.problem


def test_load_preserves_shared_resource_kind_uses_for_validation(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    resources = scenario_root / "resources"
    resources.mkdir(parents=True)
    (resources / "shared.data").write_bytes(b"shared resource")
    mapping = _with_resource_mobility(
        actor_path=True,
        resource_path="resources/shared.data",
    )
    mapping["actors"]["targets"] = [
        {
            "name": "Target",
            "mobility": {
                "type": "mesh_sequence",
                "positions_path": "resources/shared.data",
            },
            "asset": {"source": "catalog", "id": "cube"},
        }
    ]
    scenario_yaml = scenario_root / "scenario.yaml"
    _write_yaml(scenario_yaml, mapping)

    loaded = load_for_authoring(scenario_yaml)

    assert loaded.disposition is LoadDisposition.OWNED_EDITABLE
    assert loaded.scenario is not None
    assert {
        dependency.kind
        for dependency in loaded.scenario.dependencies
        if dependency.relative_path == "resources/shared.data"
    } == {"network_graph", "position_sequence"}


def test_compiler_blocks_actor_preview_through_canonical_resource_link_escape(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    outside_root = tmp_path / "outside"
    scenario_root.mkdir()
    outside_root.mkdir()
    external = outside_root / "network.graphml"
    external.write_text("must not be read", encoding="utf-8")
    _create_directory_link(scenario_root / "resources", outside_root)
    tx = AuthoringActor.create(ActorRole.TX, "TX1").with_changes(
        mobility=NetworkRouteMobilitySpec(
            graph_path="resources/network.graphml",
            start_node="start",
            end_node="end",
        )
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX1", position=(2.0, 0.0, 1.0))
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx, rx),
        resources=(
            AuthoringResource(
                ResourceKind.NETWORK_GRAPH,
                external,
                "resources/network.graphml",
            ),
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=scenario_root,
    )

    issue = next(
        issue for issue in result.issues if issue.code == "mobility.resource.outside_scenario"
    )
    assert issue.path == "actors.tx.0.mobility.graph_path"
    assert issue.actor_id == tx.id
    assert tx.id not in result.samples


def test_relative_local_scene_validation_does_not_write_through_directory_links(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    source_root = tmp_path / "scene-source"
    source_root.mkdir()
    source = source_root / "scene.xml"
    source.write_bytes((PROJECT_ROOT / "libraries/scenes/empty/empty.xml").read_bytes())
    destination_root = tmp_path / "scene-destination"
    destination_root.mkdir()
    escaped_destination = destination_root / "scene.xml"
    sentinel = b"outside sentinel"
    escaped_destination.write_bytes(sentinel)
    _create_directory_link(tmp_path / "linked-scene", source_root)
    _create_directory_link(scenario_root / "linked-scene", destination_root)
    scenario = AuthoringScenario(
        scene=SceneReference("local", "../linked-scene/scene.xml"),
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
    assert result.mapping["scene"]["id"] == "../linked-scene/scene.xml"
    assert result.resolved_scene_path == str(source.resolve())
    assert escaped_destination.read_bytes() == sentinel


@pytest.mark.parametrize(
    ("graph_path", "resource", "expected_code"),
    (
        (
            "resources/../network.graphml",
            None,
            "mobility.resource.invalid_path",
        ),
        (
            "resources/network.graphml",
            ResourceKind.POSITION_SEQUENCE,
            "mobility.resource.kind_collision",
        ),
    ),
)
def test_compiler_rejects_resource_traversal_and_kind_collision_at_actor_field(
    tmp_path: Path,
    graph_path: str,
    resource: ResourceKind | None,
    expected_code: str,
) -> None:
    source = tmp_path / "network.graphml"
    source.write_text("not read", encoding="utf-8")
    tx = AuthoringActor.create(ActorRole.TX, "TX1").with_changes(
        mobility=NetworkRouteMobilitySpec(
            graph_path=graph_path,
            start_node="start",
            end_node="end",
        )
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            tx,
            AuthoringActor.create(ActorRole.RX, "RX1", position=(2.0, 0.0, 1.0)),
        ),
        resources=(
            (
                AuthoringResource(
                    resource,
                    source,
                    "resources/network.graphml",
                ),
            )
            if resource is not None
            else ()
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=tmp_path / "scenario",
    )

    issue = next(issue for issue in result.issues if issue.code == expected_code)
    assert issue.path == "actors.tx.0.mobility.graph_path"
    assert issue.actor_id == tx.id
    assert tx.id not in result.samples


def test_compiler_blocks_group_members_and_group_preview_for_resource_link_escape(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    outside_root = tmp_path / "outside"
    scenario_root.mkdir()
    outside_root.mkdir()
    external = outside_root / "network.graphml"
    external.write_text("must not be read", encoding="utf-8")
    _create_directory_link(scenario_root / "resources", outside_root)
    group = AuthoringGroup.create("RouteGroup").with_changes(
        mobility=NetworkRouteMobilitySpec(
            graph_path="resources/network.graphml",
            start_node="start",
            end_node="end",
        )
    )
    tx = AuthoringActor.create(ActorRole.TX, "TX1").with_changes(
        mobility=GroupMemberMobilitySpec(group=str(group.id))
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX1").with_changes(
        mobility=GroupMemberMobilitySpec(group=str(group.id))
    )
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx, rx),
        groups=(group,),
        resources=(
            AuthoringResource(
                ResourceKind.NETWORK_GRAPH,
                external,
                "resources/network.graphml",
            ),
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=scenario_root,
    )

    issue = next(
        issue for issue in result.issues if issue.code == "mobility.resource.outside_scenario"
    )
    assert issue.path == "groups.0.mobility.graph_path"
    assert issue.group_id == group.id
    assert group.id not in result.group_samples
    assert tx.id not in result.samples
    assert rx.id not in result.samples


@pytest.mark.parametrize("fixture_name", ["compatible_external.yaml", "supported_owned.yaml"])
def test_schema_invalid_mapping_is_read_only_and_never_parsed_lossily(
    fixture_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = yaml.safe_load((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"))
    mapping["timeline"]["steps"] = 1.9
    source = tmp_path / fixture_name
    _write_yaml(source, mapping)
    original_bytes = source.read_bytes()

    def reject_lossy_parse(_mapping):
        pytest.fail("schema-invalid mappings must not reach scenario_from_mapping")

    monkeypatch.setattr(persistence_module, "scenario_from_mapping", reject_lossy_parse)

    loaded = load_for_authoring(source)

    assert loaded.disposition is LoadDisposition.READ_ONLY
    assert loaded.document is None
    assert loaded.scenario is None
    assert loaded.raw_mapping["timeline"]["steps"] == 1.9
    issue = next(issue for issue in loaded.issues if issue.path == "timeline.steps")
    assert issue.code == "compatibility.schema_invalid"
    assert source.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="not compatible"):
        create_scenario_copy(source, tmp_path / "copy")
    assert source.read_bytes() == original_bytes


def test_copy_applies_builder_defaults_and_round_trips_semantically(
    authoring_project_root: Path,
) -> None:
    source_root = authoring_project_root / "source"
    source_root.mkdir()
    source = source_root / "scenario.yaml"
    _write_yaml(source, _external_mapping())
    destination = authoring_project_root / "copy"

    document = create_scenario_copy(source, destination)

    assert document.scenario.timeline.steps == 30
    assert document.scenario.timeline.duration_s == 3.0
    assert document.scenario.timeline.quality.value == DEFAULT_QUALITY_PRESET
    assert document.scenario.timeline.export_path_metrics is DEFAULT_EXPORT_PATH_METRICS
    assert document.dirty is True
    assert not (destination / "scenario.yaml").exists()

    saved_path = save_document(
        document,
        compiler=ScenarioCompiler(authoring_project_root),
    )
    reopened = load_for_authoring(saved_path)

    assert reopened.disposition is LoadDisposition.OWNED_EDITABLE
    assert reopened.document is not None
    assert canonical_scenario_mapping(reopened.document.scenario) == canonical_scenario_mapping(
        document.scenario
    )
    assert reopened.document.dirty is False
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == _external_mapping()


def test_schema_known_noneditable_configuration_is_preserved_canonically() -> None:
    mapping = _known_passthrough_mapping()

    report = CompatibilityAnalyzer().analyze(mapping)
    scenario = scenario_from_mapping(mapping)
    canonical = canonical_scenario_mapping(scenario)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()
    assert scenario.timeline.quality is QualityPreset.CUSTOM
    assert canonical["debug_level"] == mapping["debug_level"]
    assert canonical["view_defaults"] == mapping["view_defaults"]
    assert canonical["raytracing"]["materials"] == mapping["raytracing"]["materials"]
    assert canonical["raytracing"]["quality"] == mapping["raytracing"]["quality"]
    assert canonical["visualizer"] == {"scenario_builder": {"document_version": 2}}


def test_sampled_mobility_is_preserved_and_locked_as_one_authoring_facet() -> None:
    mapping = _external_mapping()
    mapping["timeline"] = {"steps": 3, "duration_s": 2.0}
    positions = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    mapping["actors"]["tx"][0]["mobility"] = {
        "type": "sampled",
        "positions_m": positions,
    }

    scenario = scenario_from_mapping(mapping)
    tx = scenario.actor_by_name("TX1")
    assert tx is not None
    assert isinstance(tx.mobility, SampledMobilitySpec)
    assert canonical_scenario_mapping(scenario)["actors"]["tx"][0]["mobility"] == {
        "type": "sampled",
        "positions_m": positions,
    }

    capability = scenario.capability("mobility", tx.id)
    assert not capability.editable
    assert capability.reason == "Mobility model 'sampled' is preserved read-only."

    document = ScenarioDocument(scenario)
    with pytest.raises(PermissionError, match="preserved read-only"):
        document.replace_actor(
            tx.with_changes(mobility=StationaryMobilitySpec(position_m=(0.0, 0.0, 0.0)))
        )


def test_source_snapshot_copy_save_and_owned_reopen_are_stable(
    authoring_project_root: Path,
) -> None:
    source_root = authoring_project_root / "source"
    source_root.mkdir()
    source = source_root / "scenario.yaml"
    mapping = _known_passthrough_mapping()
    _write_yaml(source, mapping)
    source_bytes = source.read_bytes()
    document = create_scenario_copy(source, authoring_project_root / "copy")
    expected = canonical_scenario_mapping(document.scenario)

    saved_path = save_document(
        document,
        compiler=ScenarioCompiler(authoring_project_root),
    )
    reopened = load_for_authoring(saved_path)

    assert source.read_bytes() == source_bytes
    assert reopened.disposition is LoadDisposition.OWNED_EDITABLE
    assert reopened.document is not None
    assert canonical_scenario_mapping(reopened.document.scenario) == expected


def test_canonical_builder_fields_override_imported_values_without_losing_siblings() -> None:
    mapping = _known_passthrough_mapping()
    document = ScenarioDocument(scenario_from_mapping(mapping))
    tx = document.scenario.actor_by_name("TX1")
    assert tx is not None

    document.set_scene(SceneReference("sionna", "box"))
    document.rename_actor(tx.id, "TX Edited")
    document.set_timeline(
        TimelineSettings(
            steps=12,
            duration_s=6.0,
            quality=QualityPreset.CUSTOM,
            export_path_metrics=False,
        )
    )

    canonical = canonical_scenario_mapping(document.scenario)
    assert canonical["scene"] == {"source": "sionna", "id": "box"}
    assert canonical["timeline"] == {"steps": 12, "duration_s": 6.0}
    assert canonical["actors"]["tx"][0]["name"] == "TX Edited"
    assert canonical["raytracing"]["enabled"] is True
    assert canonical["raytracing"]["export_path_metrics"] is False
    assert canonical["raytracing"]["quality"] == mapping["raytracing"]["quality"]
    assert canonical["debug_level"] == mapping["debug_level"]
    assert canonical["view_defaults"] == mapping["view_defaults"]
    assert canonical["raytracing"]["materials"] == mapping["raytracing"]["materials"]
    assert canonical["visualizer"]["scenario_builder"] == {"document_version": 2}


def test_explicit_quality_replacement_is_undoable_and_removes_hidden_custom_values() -> None:
    mapping = _known_passthrough_mapping()
    document = ScenarioDocument(scenario_from_mapping(mapping))
    original_quality = mapping["raytracing"]["quality"]

    document.set_timeline(
        TimelineSettings(
            steps=document.scenario.timeline.steps,
            duration_s=document.scenario.timeline.duration_s,
            quality=QualityPreset.LOW,
            export_path_metrics=document.scenario.timeline.export_path_metrics,
        ),
        replace_source_quality=True,
    )

    assert canonical_scenario_mapping(document.scenario)["raytracing"]["quality"] == {
        "preset": "low"
    }
    assert not document.scenario.source_snapshot.has_path("raytracing.quality.custom")
    assert (
        canonical_scenario_mapping(document.scenario)["raytracing"]["materials"]
        == mapping["raytracing"]["materials"]
    )

    document.undo()

    assert (
        canonical_scenario_mapping(document.scenario)["raytracing"]["quality"] == original_quality
    )
    assert document.scenario.source_snapshot.has_path("raytracing.quality.custom")

    document.redo()

    assert canonical_scenario_mapping(document.scenario)["raytracing"]["quality"] == {
        "preset": "low"
    }
    assert not document.scenario.source_snapshot.has_path("raytracing.quality.custom")


def test_source_snapshot_has_no_mutable_aliases() -> None:
    mapping = _known_passthrough_mapping()
    scenario = scenario_from_mapping(mapping)
    expected = canonical_scenario_mapping(scenario)

    mapping["debug_level"] = "DEBUG"
    mapping["view_defaults"]["merge_scene_meshes"] = False
    mapping["raytracing"]["materials"]["concrete"]["scattering_coefficient"] = 0.9
    mapping["raytracing"]["quality"]["custom"]["max_depth"] = 99

    detached = scenario.source_snapshot.to_mapping()
    detached["debug_level"] = "INFO"
    detached["view_defaults"]["camera_view"] = "top"
    detached["raytracing"]["quality"]["custom"]["seed"] = 999

    assert canonical_scenario_mapping(scenario) == expected


def test_merged_mapping_preserves_schema_known_source_paths() -> None:
    scenario = scenario_from_mapping(_external_mapping())
    scenario = replace(
        scenario,
        timeline=TimelineSettings(
            steps=30,
            duration_s=3.0,
            quality=QualityPreset.LOW,
            export_path_metrics=False,
        ),
        source_snapshot=ScenarioSourceSnapshot.from_mapping(
            {
                "raytracing": {
                    "enabled": False,
                    "export_path_metrics": True,
                    "quality": {
                        "preset": "ultra",
                        "custom": {"max_depth": 3},
                    },
                    "path_filter": {"max_paths_per_pair": 5},
                    "materials": {
                        "concrete": {"scattering_coefficient": 0.3},
                    },
                },
                "visualizer": {"renderer": "unreviewed"},
                "coverage": {"enabled": True},
            }
        ),
    )

    canonical = canonical_scenario_mapping(scenario)

    assert canonical["raytracing"]["enabled"] is False
    assert canonical["raytracing"]["export_path_metrics"] is False
    assert canonical["raytracing"]["quality"] == {
        "preset": "low",
        "custom": {"max_depth": 3},
    }
    assert canonical["raytracing"]["materials"] == {"concrete": {"scattering_coefficient": 0.3}}
    assert canonical["raytracing"]["path_filter"] == {"max_paths_per_pair": 5}
    assert canonical["visualizer"] == {
        "renderer": "unreviewed",
        "scenario_builder": {"document_version": 2},
    }
    assert canonical["coverage"] == {"enabled": True}


def test_unknown_root_field_remains_read_only_without_lossy_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _known_passthrough_mapping()
    mapping["experimental_section"] = {"enabled": True}
    source = tmp_path / "external.yaml"
    _write_yaml(source, mapping)
    original_bytes = source.read_bytes()

    def reject_lossy_parse(_mapping, **_kwargs):
        pytest.fail("unknown fields must not reach scenario_from_mapping")

    monkeypatch.setattr(persistence_module, "scenario_from_mapping", reject_lossy_parse)

    loaded = load_for_authoring(source)

    assert loaded.disposition is LoadDisposition.READ_ONLY
    assert loaded.document is None
    assert loaded.scenario is None
    assert {issue.path for issue in loaded.issues} == {"experimental_section"}
    assert loaded.raw_mapping["experimental_section"] == {"enabled": True}
    assert source.read_bytes() == original_bytes


def test_schema_known_raytracing_field_is_importable_and_preserved() -> None:
    mapping = _known_passthrough_mapping()
    mapping["raytracing"]["path_filter"] = {"max_paths_per_pair": 5}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()


def test_owned_save_omits_session_ids_and_reopen_allocates_fresh_ids(
    authoring_project_root: Path,
) -> None:
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1", position=(0, 0, 1)),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(2, 0, 1)),
        ),
    )
    document = ScenarioDocument(scenario)

    path = save_document(
        document,
        authoring_project_root / "owned",
        compiler=ScenarioCompiler(authoring_project_root),
    )
    first = load_for_authoring(path)
    second = load_for_authoring(path)

    assert first.document is not None
    assert second.document is not None
    assert first.raw_mapping["visualizer"]["scenario_builder"] == {"document_version": 2}
    first_ids = {actor.id for actor in first.document.scenario.actors}
    second_ids = {actor.id for actor in second.document.scenario.actors}
    assert first.document.scenario.document_id != scenario.document_id
    assert second.document.scenario.document_id != first.document.scenario.document_id
    assert first_ids.isdisjoint({actor.id for actor in scenario.actors})
    assert second_ids.isdisjoint(first_ids)


@pytest.mark.parametrize(
    ("scene", "target", "expected_code", "expected_path"),
    (
        (
            SceneReference("library", "empty/empty.xml"),
            None,
            "save.library_scene.outside_project",
            "scene.id",
        ),
        (
            SceneReference("sionna", "empty"),
            TargetAsset.from_catalog_id("cube", mesh_pattern="cube.ply"),
            "save.catalog_target.outside_project",
            "actors.targets.0.asset.id",
        ),
    ),
)
def test_save_blocks_project_library_references_outside_project_root(
    authoring_project_root: Path,
    scene: SceneReference,
    target: TargetAsset | None,
    expected_code: str,
    expected_path: str,
) -> None:
    actors = [
        AuthoringActor.create(ActorRole.TX, "TX1", position=(0, 0, 1)),
        AuthoringActor.create(ActorRole.RX, "RX1", position=(2, 0, 1)),
    ]
    if target is not None:
        actors.append(
            AuthoringActor.create(
                ActorRole.TARGET,
                "Target1",
                position=(1, 0, 1),
                target=target,
            )
        )
    document = ScenarioDocument(AuthoringScenario(scene=scene, actors=tuple(actors)))
    destination = authoring_project_root.parent / "outside-project"

    with pytest.raises(ScenarioSaveError) as caught:
        save_document(
            document,
            destination,
            compiler=ScenarioCompiler(authoring_project_root),
        )

    assert [(issue.code, issue.path) for issue in caught.value.issues] == [
        (expected_code, expected_path)
    ]
    assert "inside the active ORCHAV project root" in str(caught.value)
    assert not destination.exists()


def test_owned_save_refreshes_a_changed_registered_resource_transactionally(
    tmp_path: Path,
    authoring_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "external" / "network.graphml"
    external.parent.mkdir()

    def write_graph(midpoint_y: float) -> None:
        graph = nx.Graph()
        graph.add_node("start", x_local=0.0, y_local=0.0)
        graph.add_node("middle", x_local=1.0, y_local=midpoint_y)
        graph.add_node("end", x_local=2.0, y_local=0.0)
        graph.add_edge("start", "middle", length=1.0)
        graph.add_edge("middle", "end", length=1.0)
        nx.write_graphml(graph, external)

    write_graph(0.0)
    relative = resource_relative_path(external)
    tx = AuthoringActor.create(ActorRole.TX, "TX1").with_changes(
        mobility=NetworkRouteMobilitySpec(
            graph_path=relative,
            start_node="start",
            end_node="end",
            seed=3,
        )
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX1", position=(3.0, 0.0, 1.0))
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            timeline=TimelineSettings(steps=3, duration_s=2.0),
            actors=(tx, rx),
            resources=(
                AuthoringResource(
                    ResourceKind.NETWORK_GRAPH,
                    external,
                    relative,
                ),
            ),
        )
    )
    saved_path = save_document(
        document,
        authoring_project_root / "owned",
        compiler=ScenarioCompiler(authoring_project_root),
    )
    owned_resource = saved_path.parent / relative
    first_bytes = owned_resource.read_bytes()
    first_yaml = saved_path.read_bytes()

    write_graph(1.0)
    document.replace_actor_with_resources(
        tx.with_changes(
            mobility=tx.mobility.model_copy(update={"seed": 7}),
        ),
        (
            AuthoringResource(
                ResourceKind.NETWORK_GRAPH,
                external,
                relative,
            ),
        ),
    )
    revision_before_failed_save = document.revision
    path_before_failed_save = document.path
    assert document.dirty
    changed_result = ScenarioCompiler(authoring_project_root).compile(
        document.scenario,
        scenario_directory=saved_path.parent,
    )
    assert changed_result.valid, changed_result.issues
    assert changed_result.samples[tx.id].positions[1][1] == pytest.approx(1.0)

    def fail_yaml_promotion(*_args, **_kwargs) -> None:
        raise OSError("injected YAML promotion failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            persistence_module,
            "atomic_write_scenario_yaml",
            fail_yaml_promotion,
        )
        with pytest.raises(OSError, match="injected YAML promotion failure"):
            save_document(document, compiler=ScenarioCompiler(authoring_project_root))

    assert saved_path.read_bytes() == first_yaml
    assert owned_resource.read_bytes() == first_bytes
    assert document.path == path_before_failed_save
    assert document.revision == revision_before_failed_save
    assert document.dirty

    save_document(document, compiler=ScenarioCompiler(authoring_project_root))

    assert owned_resource.read_bytes() == external.read_bytes()
    assert owned_resource.read_bytes() != first_bytes


def test_save_compilation_uses_the_callers_serialization_guard(tmp_path: Path) -> None:
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1", position=(0, 0, 1)),
            AuthoringActor.create(ActorRole.RX, "RX1", position=(2, 0, 1)),
        ),
    )
    document = ScenarioDocument(scenario)

    class _Guard:
        entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, _exception_type, _exception, _traceback):
            self.entered = False
            return False

    guard = _Guard()

    class _GuardCheckingCompiler:
        def compile(self, current, *, scenario_directory=None):
            assert guard.entered is True
            return ScenarioCompiler(PROJECT_ROOT).compile(
                current,
                scenario_directory=scenario_directory,
            )

    saved = save_document(
        document,
        tmp_path / "serialized-save",
        compiler=_GuardCheckingCompiler(),
        compile_lock=guard,
    )

    assert saved.exists()
    assert guard.entered is False


def test_keyframe_and_spin_orientations_reopen_semantically(
    authoring_project_root: Path,
) -> None:
    external = _external_mapping()
    external["actors"]["tx"][0]["orientation"] = {
        "type": "keyframes",
        "keyframes": [
            {"time_s": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"time_s": 3.0, "yaw_deg": 90.0, "pitch_deg": -10.0, "roll_deg": 5.0},
        ],
    }
    external["actors"]["rx"][0]["orientation"] = {
        "type": "spin",
        "axis": "yaw",
        "rate_deg_s": 180.0,
        "yaw_deg": -45.0,
        "pitch_deg": 10.0,
        "roll_deg": -5.0,
    }

    report = CompatibilityAnalyzer().analyze(external)
    parsed = scenario_from_mapping(external)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()
    assert parsed.actors[0].orientation == KeyframesOrientationSpec(
        keyframes=(
            OrientationKeyframeSpec(
                time_s=0.0,
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=0.0,
            ),
            OrientationKeyframeSpec(
                time_s=3.0,
                yaw_deg=90.0,
                pitch_deg=-10.0,
                roll_deg=5.0,
            ),
        )
    )
    assert parsed.actors[1].orientation == SpinOrientationSpec(
        axis="yaw",
        rate_deg_s=180.0,
        yaw_deg=-45.0,
        pitch_deg=10.0,
        roll_deg=-5.0,
    )

    document = ScenarioDocument(parsed)
    saved_path = save_document(
        document,
        authoring_project_root / "owned-orientations",
        compiler=ScenarioCompiler(authoring_project_root),
    )
    reopened = load_for_authoring(saved_path)

    assert reopened.disposition is LoadDisposition.OWNED_EDITABLE
    assert reopened.document is not None
    assert canonical_scenario_mapping(reopened.document.scenario) == canonical_scenario_mapping(
        parsed
    )


def test_coverage_root_section_is_importable() -> None:
    mapping = _external_mapping()
    mapping["coverage"] = {"enabled": True}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["actors"]["rx"][0].update(
            {
                "mobility": {
                    "type": "random_sampling",
                    "x_bounds_m": [0, 1],
                    "y_bounds_m": [2, 4],
                    "z_bounds_m": [1, 3],
                    "seed": 7,
                }
            }
        ),
        lambda data: data["actors"]["tx"][0].update(
            {
                "orientation": {
                    "type": "random",
                    "seed": 2,
                    "yaw_range_deg": [-90, 90],
                    "pitch_range_deg": [-30, 30],
                    "roll_range_deg": [-10, 10],
                    "update_interval_s": 0.5,
                }
            }
        ),
        lambda data: data["actors"]["rx"][0].update(
            {
                "mobility": {
                    "type": "linear",
                    "start_m": [2, 0, 1],
                    "end_m": [8, 0, 1],
                    "traversal": {
                        "type": "constant_speed",
                        "speed_mps": 2.0,
                        "after_end": "ping_pong",
                    },
                }
            }
        ),
        lambda data: data["actors"]["tx"][0].update(
            {
                "mobility": {
                    "type": "linear",
                    "start_m": [0, 0, 1],
                    "end_m": [3, 0, 1],
                },
                "orientation": {
                    "type": "align_motion",
                    "allow_pitch": False,
                    "smoothing_time_s": 0.25,
                    "yaw_offset_deg": 10.0,
                    "max_yaw_rate_deg_s": 90.0,
                },
            }
        ),
    ],
    ids=("random-sampling", "random-orientation", "constant-speed", "align-motion"),
)
def test_complete_shared_models_are_importable_and_parse_semantically(mutation) -> None:
    mapping = _external_mapping()
    mutation(mapping)

    report = CompatibilityAnalyzer().analyze(mapping)
    parsed = scenario_from_mapping(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()
    assert isinstance(
        parsed.actors[1].mobility,
        (
            StationaryMobilitySpec,
            LinearMobilitySpec,
            RandomSamplingMobilitySpec,
        ),
    )
    assert isinstance(
        parsed.actors[0].orientation,
        (FixedOrientationSpec, RandomOrientationSpec, AlignMotionOrientationSpec),
    )
    if isinstance(parsed.actors[1].mobility, LinearMobilitySpec):
        assert isinstance(
            parsed.actors[1].mobility.traversal,
            ConstantSpeedTraversalSpec,
        )


def test_groups_reopen_with_uuid_backed_member_references() -> None:
    mapping = _external_mapping()
    mapping["groups"] = [
        {
            "name": "Convoy",
            "mobility": {
                "type": "linear",
                "start_m": [0, 0, 1],
                "end_m": [6, 0, 1],
            },
            "deviation": {
                "max_right_m": 0.5,
                "max_forward_m": 1.0,
                "max_up_m": 0.25,
                "seed": 13,
            },
        }
    ]
    mapping["actors"]["tx"][0]["mobility"] = {
        "type": "group_member",
        "group": "Convoy",
        "offset_m": {"right": -1.0, "forward": 0.0, "up": 0.5},
    }
    mapping["actors"]["rx"][0]["mobility"] = {
        "type": "group_member",
        "group": "Convoy",
        "offset_m": {"right": 1.0, "forward": 0.0, "up": 0.5},
    }

    report = CompatibilityAnalyzer().analyze(mapping)
    parsed = scenario_from_mapping(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()
    assert len(parsed.groups) == 1
    group = parsed.groups[0]
    assert all(
        isinstance(actor.mobility, GroupMemberMobilitySpec)
        and UUID(actor.mobility.group) == group.id
        for actor in parsed.actors
    )
    canonical = canonical_scenario_mapping(parsed)
    assert canonical["groups"][0]["name"] == "Convoy"
    assert {
        canonical["actors"]["tx"][0]["mobility"]["group"],
        canonical["actors"]["rx"][0]["mobility"]["group"],
    } == {"Convoy"}


def test_nonuniform_keyframes_are_importable_and_preserve_times() -> None:
    mapping = _external_mapping()
    mapping["actors"]["tx"][0]["orientation"] = {
        "type": "keyframes",
        "keyframes": [
            {"time_s": 0.0},
            {"time_s": 1.0},
            {"time_s": 3.0},
        ],
    }

    report = CompatibilityAnalyzer().analyze(mapping)
    parsed = scenario_from_mapping(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()
    orientation = parsed.actors[0].orientation
    assert isinstance(orientation, KeyframesOrientationSpec)
    assert tuple(keyframe.time_s for keyframe in orientation.keyframes) == (0.0, 1.0, 3.0)


def test_file_target_asset_is_importable_with_locked_locator() -> None:
    mapping = _external_mapping()
    mapping["actors"]["targets"] = [
        {
            "name": "Target1",
            "mobility": {"type": "stationary", "position_m": [1, 1, 0]},
            "asset": {"source": "file", "path": "car.ply"},
        }
    ]

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    parsed = scenario_from_mapping(mapping)
    target = parsed.actors[-1]
    assert target.target is not None
    assert target.target.source == "file"
    capability = parsed.capability("target_asset", target.id)
    assert not capability.editable
    assert capability.path == "actors.targets.0.asset"


def test_future_builder_document_version_is_read_only_at_marker_path() -> None:
    mapping = _external_mapping()
    mapping["visualizer"] = {"scenario_builder": {"document_version": 3}}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.READ_ONLY
    assert "visualizer.scenario_builder.document_version" in {issue.path for issue in report.issues}


def test_malformed_builder_marker_is_read_only_at_marker_path() -> None:
    mapping = _external_mapping()
    mapping["visualizer"] = {"scenario_builder": "unsupported"}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.READ_ONLY
    assert {issue.path for issue in report.issues} == {"visualizer.scenario_builder"}


def test_unsupported_scenario_version_reports_only_the_version_path() -> None:
    mapping = _external_mapping()
    mapping["schema_version"] = 3

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.READ_ONLY
    assert {issue.path for issue in report.issues} == {"schema_version"}


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_builder_document_version_requires_an_exact_integer(version) -> None:
    mapping = _external_mapping()
    mapping["visualizer"] = {"scenario_builder": {"document_version": version}}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.READ_ONLY
    assert "visualizer.scenario_builder.document_version" in {issue.path for issue in report.issues}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", "11111111-1111-4111-8111-111111111111"),
        ("actors", []),
    ],
)
def test_builder_marker_preserves_nonownership_metadata(field: str, value) -> None:
    mapping = _external_mapping()
    mapping["visualizer"] = {"scenario_builder": {"document_version": 2, field: value}}

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.OWNED_EDITABLE
    assert report.issues == ()


def test_actor_entries_reject_serialized_session_identity() -> None:
    mapping = _external_mapping()
    mapping["actors"]["tx"][0]["id"] = "22222222-2222-4222-8222-222222222222"

    report = CompatibilityAnalyzer().analyze(mapping)

    assert report.disposition is LoadDisposition.READ_ONLY
    assert "actors.tx.0.id" in {issue.path for issue in report.issues}


def test_boolean_strings_are_parsed_without_python_truthiness() -> None:
    mapping = _external_mapping()
    mapping["raytracing"]["export_path_metrics"] = "false"
    mapping["actors"]["tx"][0] = {
        "name": "TX1",
        "mobility": {
            "type": "circular",
            "center_m": [0, 0, 1],
            "radius_m": 2,
            "clockwise": "false",
        },
    }

    parsed = scenario_from_mapping(mapping)

    assert parsed.timeline.export_path_metrics is False
    assert parsed.actors[0].mobility.clockwise is False


def test_scenario_copy_rejects_the_source_scenario_path(tmp_path: Path) -> None:
    source = tmp_path / "scenario.yaml"
    _write_yaml(source, _external_mapping())

    with pytest.raises(ValueError, match="different from the source"):
        create_scenario_copy(source, tmp_path)

    assert yaml.safe_load(source.read_text(encoding="utf-8")) == _external_mapping()


def test_copy_and_save_refuse_existing_arbitrary_scenario(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "scenario.yaml"
    _write_yaml(source, _external_mapping())
    destination = tmp_path / "destination"
    destination.mkdir()
    existing = destination / "scenario.yaml"
    existing.write_text("arbitrary: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains"):
        create_scenario_copy(source, destination)

    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(
                AuthoringActor.create(ActorRole.TX, "TX1"),
                AuthoringActor.create(ActorRole.RX, "RX1", position=(1, 0, 0)),
            ),
        )
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_document(document, destination, compiler=ScenarioCompiler(PROJECT_ROOT))

    assert existing.read_text(encoding="utf-8") == "arbitrary: true\n"


def test_atomic_replace_failure_preserves_scenario_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "scenario.yaml"
    destination.write_text("current\n", encoding="utf-8")

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("visualizer.src.authoring.persistence.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_scenario_yaml(destination, "replacement\n")

    assert destination.read_text(encoding="utf-8") == "current\n"
    assert list(tmp_path.glob(".scenario.yaml.*.tmp")) == []


def test_atomic_writer_rejects_arbitrary_yaml_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario.yaml"):
        atomic_write_scenario_yaml(tmp_path / "draft.yaml", "schema_version: 2\n")


def test_session_identifiers_are_valid_uuid_values_after_reopen(tmp_path: Path) -> None:
    source = tmp_path / "external.yaml"
    _write_yaml(source, _external_mapping())

    loaded = load_for_authoring(source)

    assert loaded.scenario is not None
    assert isinstance(loaded.scenario.document_id, UUID)
    assert all(isinstance(actor.id, UUID) for actor in loaded.scenario.actors)
