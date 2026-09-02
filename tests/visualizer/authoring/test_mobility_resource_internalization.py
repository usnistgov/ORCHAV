"""Repeated-Apply coverage for resource-backed mobility models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from shared.scenarios.actors import (
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
)
from visualizer.src.authoring.compiler import ScenarioCompiler
from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.domain import (
    ActorRole,
    AuthoringActor,
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
)
from visualizer.src.authoring.viewport_port import AuthoringTool
from visualizer.src.authoring.workspace import ScenarioAuthoringWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _MobilityEditorStub:
    def __init__(self, mobility) -> None:
        self.value = mobility

    def mobility(self):
        return self.value


class _WorkspaceHarness:
    _apply_mobility = ScenarioAuthoringWorkspace._apply_mobility
    _internalize_mobility_resource = ScenarioAuthoringWorkspace._internalize_mobility_resource

    def __init__(self, document: ScenarioDocument, project_root: Path, mobility) -> None:
        self.document = document
        self.mobility_editor = _MobilityEditorStub(mobility)
        self._project_directory = project_root
        self._waypoint_session_actor_id = None
        self._mobility_draft_pending = False
        self._tool = AuthoringTool.SELECT

    def _project_root(self) -> Path:
        return self._project_directory


def _external_mobility(model: str, source: Path):
    if model == "network_route":
        return NetworkRouteMobilitySpec(
            route="shortest_path",
            seed=3,
            graph_path=str(source),
            start_node="A",
            end_node="B",
        )
    return MeshSequenceMobilitySpec(
        positions_path=str(source),
        position_key="positions",
        interpolation="linear",
    )


def _changed_mobility(mobility):
    if isinstance(mobility, NetworkRouteMobilitySpec):
        return mobility.model_copy(update={"seed": 19})
    return mobility.model_copy(update={"interpolation": "step"})


def _resource_details(mobility) -> tuple[str, ResourceKind]:
    if isinstance(mobility, NetworkRouteMobilitySpec):
        assert mobility.graph_path is not None
        return mobility.graph_path, ResourceKind.NETWORK_GRAPH
    return mobility.positions_path, ResourceKind.POSITION_SEQUENCE


def _document_with_selected_actor() -> ScenarioDocument:
    actor = AuthoringActor.create(ActorRole.TX, "Actor")
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    return document


@pytest.mark.parametrize(
    ("model", "filename"),
    (("network_route", "network.graphml"), ("mesh_sequence", "positions.npy")),
)
def test_unsaved_second_apply_reuses_the_pending_registered_resource(
    tmp_path: Path,
    model: str,
    filename: str,
) -> None:
    source = tmp_path / "external" / filename
    source.parent.mkdir()
    source.write_bytes(b"external resource")
    document = _document_with_selected_actor()
    harness = _WorkspaceHarness(
        document,
        tmp_path / "draft",
        _external_mobility(model, source),
    )

    harness._apply_mobility()
    first_actor = document.selected_actor
    assert first_actor is not None
    first_path, expected_kind = _resource_details(first_actor.mobility)
    expected_relative = resource_relative_path(source)
    assert first_path == expected_relative
    assert document.scenario.resources == (
        AuthoringResource(expected_kind, source, expected_relative),
    )

    harness.mobility_editor.value = _changed_mobility(first_actor.mobility)
    harness._apply_mobility()

    second_actor = document.selected_actor
    assert second_actor is not None
    second_path, _kind = _resource_details(second_actor.mobility)
    assert second_path == expected_relative
    assert document.scenario.resources == (
        AuthoringResource(expected_kind, source, expected_relative),
    )


@pytest.mark.parametrize(
    ("model", "filename"),
    (("network_route", "network.graphml"), ("mesh_sequence", "positions.npy")),
)
def test_saved_reopened_second_apply_reuses_the_owned_resource(
    tmp_path: Path,
    model: str,
    filename: str,
) -> None:
    source = tmp_path / "external" / filename
    source.parent.mkdir()
    source.write_bytes(b"external resource")
    document = _document_with_selected_actor()
    first_harness = _WorkspaceHarness(
        document,
        tmp_path / "draft",
        _external_mobility(model, source),
    )
    first_harness._apply_mobility()
    first_actor = document.selected_actor
    assert first_actor is not None
    relative, expected_kind = _resource_details(first_actor.mobility)

    saved_root = tmp_path / "saved"
    owned_source = saved_root / relative
    owned_source.parent.mkdir(parents=True)
    owned_source.write_bytes(source.read_bytes())
    reopened_scenario = replace(
        document.scenario,
        resources=(AuthoringResource(expected_kind, owned_source, relative),),
    )
    reopened = ScenarioDocument.loaded(
        reopened_scenario,
        saved_root / "scenario.yaml",
    )
    reopened.select(first_actor.id)
    reopened_harness = _WorkspaceHarness(
        reopened,
        tmp_path / "unused-project-root",
        _changed_mobility(first_actor.mobility),
    )

    reopened_harness._apply_mobility()

    second_actor = reopened.selected_actor
    assert second_actor is not None
    second_path, _kind = _resource_details(second_actor.mobility)
    assert second_path == relative
    assert reopened.scenario.resources == (
        AuthoringResource(expected_kind, owned_source, relative),
    )


def test_real_position_sequence_saves_reopens_and_reapplies(
    tmp_path: Path,
    authoring_project_root: Path,
) -> None:
    source = tmp_path / "external" / "positions.npy"
    source.parent.mkdir()
    positions = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (2.0, 1.0, 1.5),
            (4.0, 0.0, 2.0),
        ),
        dtype=np.float64,
    )
    np.save(source, positions)
    tx = AuthoringActor.create(ActorRole.TX, "TX1", position=(-2.0, 0.0, 1.0))
    rx = AuthoringActor.create(ActorRole.RX, "RX1", position=(6.0, 0.0, 1.0))
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Target",
        target=TargetAsset.from_catalog_id("cube"),
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            timeline=TimelineSettings(steps=5, duration_s=2.0),
            actors=(tx, rx, target),
        )
    )
    document.select(target.id)
    first_harness = _WorkspaceHarness(
        document,
        tmp_path / "draft",
        MeshSequenceMobilitySpec(positions_path=str(source)),
    )

    assert first_harness._apply_mobility()
    saved = save_document(
        document,
        authoring_project_root / "saved",
        compiler=ScenarioCompiler(authoring_project_root),
    )
    reopened_result = load_for_authoring(saved)
    assert reopened_result.document is not None
    reopened = reopened_result.document
    reopened_target = reopened.scenario.actor_by_name("Target")
    assert reopened_target is not None
    assert isinstance(reopened_target.mobility, MeshSequenceMobilitySpec)
    relative = reopened_target.mobility.positions_path
    reopened.select(reopened_target.id)
    second_harness = _WorkspaceHarness(
        reopened,
        saved.parent,
        reopened_target.mobility.model_copy(update={"interpolation": "step"}),
    )

    assert second_harness._apply_mobility()

    reapplied_target = reopened.selected_actor
    assert reapplied_target is not None
    assert isinstance(reapplied_target.mobility, MeshSequenceMobilitySpec)
    assert reapplied_target.mobility.positions_path == relative
    assert reapplied_target.mobility.interpolation == "step"
    assert reopened.scenario.resources == (
        AuthoringResource(
            ResourceKind.POSITION_SEQUENCE,
            saved.parent / relative,
            relative,
        ),
    )
    compiled = ScenarioCompiler(authoring_project_root).compile(
        reopened.scenario,
        scenario_directory=saved.parent,
    )
    assert compiled.valid, compiled.issues
    assert compiled.samples[reapplied_target.id].positions[0] == tuple(positions[0])
    assert compiled.samples[reapplied_target.id].positions[-1] == tuple(positions[-1])


def test_real_network_graph_saves_reopens_and_reapplies_seed_only(
    tmp_path: Path,
    authoring_project_root: Path,
) -> None:
    source = tmp_path / "external" / "network.graphml"
    source.parent.mkdir()
    graph = nx.Graph()
    graph.add_node("A", x_local=0.0, y_local=0.0)
    graph.add_node("B", x_local=2.0, y_local=1.0)
    graph.add_node("C", x_local=4.0, y_local=0.0)
    graph.add_edge("A", "B", length=2.25)
    graph.add_edge("B", "C", length=2.25)
    nx.write_graphml(graph, source)
    tx = AuthoringActor.create(ActorRole.TX, "TX1")
    rx = AuthoringActor.create(ActorRole.RX, "RX1", position=(6.0, 0.0, 1.0))
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            timeline=TimelineSettings(steps=5, duration_s=2.0),
            actors=(tx, rx),
        )
    )
    document.select(tx.id)
    first_harness = _WorkspaceHarness(
        document,
        tmp_path / "draft",
        NetworkRouteMobilitySpec(
            graph_path=str(source),
            start_node="A",
            end_node="C",
            seed=3,
        ),
    )

    assert first_harness._apply_mobility()
    first_tx = document.selected_actor
    assert first_tx is not None
    assert isinstance(first_tx.mobility, NetworkRouteMobilitySpec)
    expected_relative = resource_relative_path(source)
    assert first_tx.mobility.graph_path == expected_relative
    saved = save_document(
        document,
        authoring_project_root / "saved-route",
        compiler=ScenarioCompiler(authoring_project_root),
    )
    reopened_result = load_for_authoring(saved)
    assert reopened_result.document is not None
    reopened = reopened_result.document
    reopened_tx = reopened.scenario.actor_by_name("TX1")
    assert reopened_tx is not None
    assert isinstance(reopened_tx.mobility, NetworkRouteMobilitySpec)
    assert reopened_tx.mobility.graph_path == expected_relative
    reopened.select(reopened_tx.id)
    second_harness = _WorkspaceHarness(
        reopened,
        saved.parent,
        reopened_tx.mobility.model_copy(update={"seed": 19}),
    )

    assert second_harness._apply_mobility()

    reapplied_tx = reopened.selected_actor
    assert reapplied_tx is not None
    assert isinstance(reapplied_tx.mobility, NetworkRouteMobilitySpec)
    assert reapplied_tx.mobility.graph_path == expected_relative
    assert reapplied_tx.mobility.seed == 19
    assert reopened.scenario.resources == (
        AuthoringResource(
            ResourceKind.NETWORK_GRAPH,
            saved.parent / expected_relative,
            expected_relative,
        ),
    )
    compiled = ScenarioCompiler(authoring_project_root).compile(
        reopened.scenario,
        scenario_directory=saved.parent,
    )
    assert compiled.valid, compiled.issues
    assert compiled.samples[reapplied_tx.id].positions[0] == (0.0, 0.0, 0.0)
    assert compiled.samples[reapplied_tx.id].positions[-1] == (4.0, 0.0, 0.0)
