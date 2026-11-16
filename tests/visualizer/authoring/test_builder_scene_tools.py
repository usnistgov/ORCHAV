"""Street routes, independent environments, spatial checks and sensing guides."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import networkx as nx
import numpy as np
import pytest
import yaml

from generator.core.scenario_actors.resources import prepare_resource_mobility
from generator.core.scenario_actors.types import Timeline
from shared.scenarios.actors import NetworkRouteMobilitySpec
from visualizer.src.authoring import (
    ScenarioCompiler,
    ScenarioDocument,
    canonical_scenario_mapping,
    load_for_authoring,
    save_document,
)
from visualizer.src.authoring.domain import (
    ActorRole,
    AuthoringActor,
    AuthoringResource,
    ResourceKind,
)
from visualizer.src.authoring.environment_library import EnvironmentLibrary
from visualizer.src.authoring.sensing_guides import range_gate_lines, sensing_guides
from visualizer.src.authoring.spatial_checks import SpatialScene
from visualizer.src.authoring.street_dialog import StreetPlacementDialog
from visualizer.src.authoring.street_routes import StreetNetwork, street_names
from visualizer.src.authoring.viewport_port import ActorOverlaySnapshot
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def graph_path(tmp_path):
    graph = nx.MultiDiGraph()
    for node, xy in {
        "a": (0, 0),
        "b": (10, 0),
        "c": (20, 0),
        "d": (30, 0),
        "e": (100, 0),
        "f": (110, 0),
        "g": (10, 10),
    }.items():
        graph.add_node(node, x_local=xy[0], y_local=xy[1])
    for a, b, name, highway in (
        ("a", "b", "Start Street", "residential"),
        ("b", "c", "Middle Street", "residential"),
        ("c", "d", "End Street", "residential"),
        ("e", "f", "Island Street", "residential"),
        ("b", "g", "Footpath", "footway"),
    ):
        graph.add_edge(a, b, name=name, highway=highway, length=10.0)
    graph["a"]["b"][0]["geometry_local"] = json.dumps([[0, 0], [5, 2], [10, 0]])
    path = tmp_path / "street_network.graphml"
    nx.write_graphml(graph, path)
    return path


def test_named_streets_use_canonical_connected_routes(graph_path):
    network = StreetNetwork(graph_path, "car")
    assert "Footpath" not in network.streets
    assert "Footpath" in StreetNetwork(graph_path, "pedestrian").streets
    options = dict(
        count=2, seed=4, start_street="Start Street", end_street="End Street", altitude_m=0.9
    )
    routes = network.routes(**options)
    assert routes == network.routes(**options)
    assert len({r.points[0] for r in routes}) == 2
    for route in routes:
        spec = NetworkRouteMobilitySpec(
            graph_path=str(graph_path),
            travel_mode="car",
            start_node=route.start,
            end_node=route.end,
            altitude_m=0.9,
        )
        prepared = prepare_resource_mobility(
            spec, Timeline(2, 1.0), base_dir=graph_path.parent, path="test"
        )
        np.testing.assert_allclose(prepared.positions_m[0], route.points[0])
        np.testing.assert_allclose(prepared.positions_m[-1], route.points[-1])
    assert (5.0, 2.0, 0.9) in next(r for r in routes if r.start == "a").points


def test_unconnected_one_way_and_oversubscribed_streets_are_clear_errors(graph_path):
    network = StreetNetwork(graph_path, "car")
    for start, end in (("Start Street", "Island Street"), ("End Street", "Start Street")):
        with pytest.raises(ValueError, match="No connected route"):
            network.routes(count=1, seed=1, start_street=start, end_street=end)
    with pytest.raises(ValueError, match="Only 2"):
        network.routes(count=3, seed=1, start_street="Start Street", end_street="End Street")
    assert street_names("['Rue A', 'Rue B']") == ("Rue A", "Rue B")


def test_route_dialog_requires_fresh_preview(qtbot, graph_path):
    dialog = StreetPlacementDialog(graph_path, ("cube",))
    qtbot.addWidget(dialog)
    dialog.count.setValue(1)
    dialog.start_street.setCurrentText("Start Street")
    dialog.end_street.setCurrentText("End Street")
    dialog.preview_routes()
    assert dialog.apply.isEnabled() and dialog.routes
    dialog.seed.setValue(2)
    assert not dialog.apply.isEnabled() and not dialog.routes
    dialog.end_street.setCurrentText("Island Street")
    dialog.preview_routes()
    assert "No connected route" in dialog.status.text()
    assert not dialog.apply.isEnabled()


def test_malformed_graph_is_an_actionable_dialog_error(qtbot, tmp_path):
    path = tmp_path / "bad.graphml"
    path.write_text("<not-valid-xml")
    dialog = StreetPlacementDialog(path, ("cube",))
    qtbot.addWidget(dialog)
    assert "Cannot load this graph" in dialog.status.text()
    assert not dialog.apply.isEnabled()


def base_document(root):
    root.mkdir(exist_ok=True)
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    mapping = canonical_scenario_mapping(document.scenario)
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    mapping["sensing"] = {"enabled": False, "acquisition": {"bandwidth_hz": 5e8}}
    (root / "scene.xml").write_text('<scene version="3.0.0"><shape type="cube"/></scene>')
    (root / "scenario.yaml").write_text(yaml.safe_dump(mapping))
    return load_for_authoring(root).document


def test_street_batch_is_one_undo_and_resources_save_portably(tmp_path, graph_path):
    root = tmp_path / "source"
    doc = base_document(root)
    resource = AuthoringResource(
        ResourceKind.NETWORK_GRAPH, graph_path, "resources/streets.graphml"
    )
    actors = tuple(
        AuthoringActor.create(ActorRole.RX, f"Walker{i}").with_changes(
            mobility=NetworkRouteMobilitySpec(
                graph_path=resource.relative_path, start_node="a", end_node="d"
            )
        )
        for i in range(2)
    )
    before = doc.scenario
    doc.add_actors_with_resources(actors, (resource,))
    assert len(doc.actors) == 4
    doc.undo()
    assert doc.scenario == before
    doc.redo()
    destination = save_document(doc, tmp_path / "copied", compiler=ScenarioCompiler(PROJECT))
    assert (destination.parent / resource.relative_path).is_file()
    reopened = load_for_authoring(destination).document
    assert ScenarioCompiler(PROJECT).compile(reopened, scenario_directory=destination.parent).valid


def test_environment_snapshot_preserves_current_actors_settings_and_undo(tmp_path, graph_path):
    import shutil

    source = tmp_path / "source"
    doc = base_document(source)
    shutil.copy2(graph_path, source / graph_path.name)
    library = EnvironmentLibrary(tmp_path / "library", project_root=PROJECT)
    entry = library.save(
        doc.scenario, source, "Test square", resolved_scene=str(source / "scene.xml")
    )
    assert (entry.root / "street_network.graphml").is_file()
    assert not (entry.root / "scenario.yaml").exists()
    assert "actors" not in entry.metadata and "sensing" not in entry.metadata
    (source / "scene.xml").unlink()
    dependencies = library.dependencies(entry)
    other = base_document(tmp_path / "other")
    before = other.scenario
    other.use_environment(entry.metadata["scene"], dependencies)
    assert other.actors == before.actors
    assert (
        canonical_scenario_mapping(other.scenario)["sensing"]
        == canonical_scenario_mapping(before)["sensing"]
    )
    other.undo()
    assert other.scenario == before
    other.redo()
    saved = save_document(other, tmp_path / "reused", compiler=ScenarioCompiler(PROJECT))
    assert saved.is_file() and (saved.parent / "street_network.graphml").is_file()
    reopened = load_for_authoring(saved).document
    assert any(d.relative_path == "street_network.graphml" for d in reopened.scenario.dependencies)
    assert len(library.entries()) == 1


def test_environment_missing_input_cannot_publish(tmp_path):
    doc = base_document(tmp_path / "source")
    (tmp_path / "source/scene.xml").unlink()
    library = EnvironmentLibrary(tmp_path / "library", project_root=PROJECT)
    with pytest.raises(ValueError, match="available"):
        library.save(doc.scenario, tmp_path / "source", "Missing")
    assert not library.entries()


def test_library_can_snapshot_a_browsed_absolute_scene(tmp_path):
    from visualizer.src.authoring.domain import SceneReference

    scene = tmp_path / "custom.xml"
    scene.write_text('<scene version="3.0.0"><shape type="cube"/></scene>')
    doc = ScenarioDocument.new()
    doc.set_scene(SceneReference("local", str(scene)))
    library = EnvironmentLibrary(tmp_path / "library", project_root=PROJECT)
    entry = library.save(doc.scenario, PROJECT, "Custom", resolved_scene=scene)
    scene.unlink()
    assert entry.metadata["scene"]["id"] == "custom.xml"
    assert all(d.available and not d.external for d in library.dependencies(entry))


def asset(name, vertices, triangles):
    return SimpleNamespace(
        name=name,
        mesh=MeshPayload(np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32)),
        material=MaterialPayload(),
    )


def spatial_scene():
    ground = asset(
        "ground", [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]], [[0, 1, 2], [0, 2, 3]]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    cube = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 4],
            [1, -1, 4],
            [1, 1, 4],
            [-1, 1, 4],
        ]
    )
    buildings = asset(
        "buildings",
        np.concatenate([cube + [-4, 0, 0], cube + [4, 0, 0]]),
        np.concatenate([faces, faces + 8]),
    )
    return SpatialScene((ground, buildings))


def test_placement_uses_building_interiors_not_aggregate_bounds():
    scene = spatial_scene()
    assert not scene.check(None, "Between buildings", (0, 0, 1))
    assert scene.check(None, "Inside", (-4, 0, 1))[0].code == "inside_building"
    assert scene.check(None, "Outside", (15, 0, 1))[0].code == "outside"
    assert scene.check(None, "Buried mesh", (0, 0, 1), bottom_z=-0.5)[0].code == "below_ground"
    fixed = scene.corrected_position((0, 0, 1), bottom_z=-0.5)
    assert fixed[2] == pytest.approx(1.52)
    clear = scene.corrected_position((-4, 0, 1), clear=True)
    assert not scene.check(None, "Fixed", clear)


def radio(role, position, rotation=None):
    return ActorOverlaySnapshot(
        uuid4(),
        role,
        role,
        (position,),
        current_position=position,
        orientation_matrix=tuple(map(tuple, np.eye(4) if rotation is None else rotation)),
    )


def test_sensing_range_is_correct_bistatic_geometry_and_configuration_unchanged():
    for points in range_gate_lines((0, 0, 0), (6, 0, 0), 10):
        np.testing.assert_allclose(
            np.linalg.norm(points, axis=1) + np.linalg.norm(points - [6, 0, 0], axis=1),
            10,
            atol=1e-10,
        )
    assert not range_gate_lines((0, 0, 0), (6, 0, 0), 5)
    mapping = {
        "sensing": {
            "enabled": True,
            "processing": {"range": {"interpretation": "bistatic"}},
            "detection": {"filters": {"maximum_range_m": 10}},
        }
    }
    before = copy.deepcopy(mapping)
    guides, note = sensing_guides(mapping, [radio("tx", (0, 0, 0)), radio("rx", (6, 0, 0))])
    assert len(guides) == 3 and "range gates" in note
    assert mapping == before


def test_sensor_boresight_rotates_but_path_filter_is_world_referenced():
    yaw90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mapping = {
        "sensing": {
            "enabled": True,
            "processing": {
                "path_selection": {
                    "enabled": True,
                    "azimuth_min_deg": -30,
                    "azimuth_max_deg": 30,
                    "elevation_min_deg": -10,
                    "elevation_max_deg": 10,
                }
            },
        }
    }
    first, _ = sensing_guides(mapping, [radio("rx", (0, 0, 0))])
    second, _ = sensing_guides(mapping, [radio("rx", (0, 0, 0), yaw90)])
    np.testing.assert_allclose(first[0].payload.points[1], [10, 0, 0])
    np.testing.assert_allclose(second[0].payload.points[1], [0, 10, 0])
    np.testing.assert_allclose(first[1].payload.points, second[1].payload.points)
    mapping["sensing"]["detection"] = {"filters": {"maximum_range_m": 50}}
    without_tx, _ = sensing_guides(mapping, [radio("rx", (0, 0, 0))])
    assert len(without_tx) == 2
