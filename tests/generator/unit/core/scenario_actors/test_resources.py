"""Focused resource-boundary tests for actor mobility preparation."""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from generator.core.scenario_actors import PosePreparationError, prepare_scenario
from shared.scenarios.actors import (
    ActorsSpec,
    CatalogAssetSpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    TargetActorSpec,
    TimelineSpec,
)
from shared.scenarios.model import ScenarioModel

pytestmark = pytest.mark.unit

NETWORK_ROUTE_FIXTURE = (
    Path(__file__).resolve().parents[3] / "fixtures" / "network_route" / "street_network.graphml"
)
OPTIONAL_MAP_MODULES = ("boto3", "pyproj", "shapely", "triangle")


def _network_route_scenario(
    *,
    graph_path: str | None = None,
    start_node: str = "start",
    end_node: str = "end",
) -> ScenarioModel:
    return ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="vehicle",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=NetworkRouteMobilitySpec(
                        graph_path=graph_path,
                        start_node=start_node,
                        end_node=end_node,
                    ),
                ),
            )
        ),
    )


def test_mesh_sequence_loads_relative_to_explicit_base_dir(tmp_path) -> None:
    np.save(
        tmp_path / "positions.npy",
        np.asarray(((0, 0, 0), (2, 0, 0), (2, 2, 0)), dtype=np.float64),
    )
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="mesh",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=MeshSequenceMobilitySpec(positions_path="positions.npy"),
                ),
            )
        ),
    )

    prepared = prepare_scenario(scenario, base_dir=tmp_path)

    assert prepared.actor("mesh").positions_m == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0),
    )


def test_mesh_sequence_libraries_path_uses_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scenario_root = project_root / "scenarios" / "example"
    resource_path = project_root / "libraries" / "motion" / "positions.npy"
    scenario_root.mkdir(parents=True)
    resource_path.parent.mkdir(parents=True)
    (tmp_path / "README.md").write_text("workspace", encoding="utf-8")
    for marker in ("config", "visualizer", "shared"):
        (tmp_path / marker).mkdir()
    np.save(
        resource_path,
        np.asarray(((0, 0, 0), (2, 0, 0), (2, 2, 0)), dtype=np.float64),
    )
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="mesh",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=MeshSequenceMobilitySpec(
                        positions_path="libraries/motion/positions.npy"
                    ),
                ),
            )
        ),
    )

    prepared = prepare_scenario(scenario, base_dir=scenario_root)

    assert prepared.actor("mesh").positions_m[-1] == (2.0, 2.0, 0.0)


def test_network_route_uses_default_cached_graph_without_osm_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in OPTIONAL_MAP_MODULES:
        monkeypatch.setitem(sys.modules, module_name, None)

    prepared = prepare_scenario(
        _network_route_scenario(),
        base_dir=NETWORK_ROUTE_FIXTURE.parent,
    )

    assert prepared.actor("vehicle").positions_m == (
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (4.0, 0.0, 0.0),
    )


def test_network_route_reads_node_link_json_with_links_key(tmp_path: Path) -> None:
    graph_path = tmp_path / "route.json"
    graph_path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {"id": "start", "x_local": 0.0, "y_local": 0.0},
                    {"id": "end", "x_local": 4.0, "y_local": 0.0},
                ],
                "links": [
                    {
                        "source": "start",
                        "target": "end",
                        "length": 4.0,
                        "highway": "residential",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_scenario(
        _network_route_scenario(graph_path=graph_path.name),
        base_dir=tmp_path,
    )

    assert prepared.actor("vehicle").positions_m == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    )


def test_network_route_reports_missing_networkx_as_a_structured_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "networkx", None)

    with pytest.raises(PosePreparationError) as raised:
        prepare_scenario(
            _network_route_scenario(),
            base_dir=NETWORK_ROUTE_FIXTURE.parent,
        )

    assert raised.value.issue.code == "missing_network_route_dependency"
    assert raised.value.issue.path == "actors.targets[0].mobility"
    assert raised.value.issue.message == ("networkx is required to prepare a cached network_route")


@pytest.mark.parametrize(
    ("travel_mode", "expected_midpoint"),
    [
        ("pedestrian", (1.0, 0.0, 1.5)),
        ("car", (1.0, 2.0, 1.5)),
    ],
)
def test_network_route_filters_edges_by_actor_travel_mode(
    tmp_path,
    travel_mode: str,
    expected_midpoint: tuple[float, float, float],
) -> None:
    graph = nx.Graph()
    graph.add_node("start", x_local=0.0, y_local=0.0)
    graph.add_node("walk", x_local=1.0, y_local=0.0)
    graph.add_node("drive", x_local=1.0, y_local=2.0)
    graph.add_node("end", x_local=2.0, y_local=0.0)
    graph.add_edge("start", "walk", length=1.0, highway="footway")
    graph.add_edge("walk", "end", length=1.0, highway="footway")
    graph.add_edge("start", "drive", length=2.0, highway="motorway")
    graph.add_edge("drive", "end", length=2.0, highway="motorway")
    nx.write_graphml(graph, tmp_path / "routes.graphml")
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="traveler",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=NetworkRouteMobilitySpec(
                        travel_mode=travel_mode,
                        altitude_m=1.5,
                        graph_path="routes.graphml",
                        start_node="start",
                        end_node="end",
                    ),
                ),
            )
        ),
    )

    prepared = prepare_scenario(scenario, base_dir=tmp_path)

    assert prepared.actor("traveler").positions_m[1] == expected_midpoint


def test_network_random_walk_starts_from_a_node_with_an_outgoing_edge(tmp_path) -> None:
    graph = nx.DiGraph()
    graph.add_node("sink", x_local=2.0, y_local=0.0)
    graph.add_node("source", x_local=0.0, y_local=0.0)
    graph.add_edge("source", "sink", length=2.0)
    nx.write_graphml(graph, tmp_path / "directed.graphml")
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="walker",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=NetworkRouteMobilitySpec(
                        route="random_walk",
                        seed=1,
                        graph_path="directed.graphml",
                    ),
                ),
            )
        ),
    )

    prepared = prepare_scenario(scenario, base_dir=tmp_path)

    assert prepared.actor("walker").positions_m == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )


def test_network_random_walk_rejects_graphs_without_a_moving_edge(tmp_path) -> None:
    graph = nx.DiGraph()
    graph.add_node("one", x_local=0.0, y_local=0.0)
    graph.add_node("two", x_local=1.0, y_local=0.0)
    graph.add_edge("one", "one", length=0.0)
    graph.add_edge("two", "two", length=0.0)
    nx.write_graphml(graph, tmp_path / "self-loops.graphml")
    scenario = ScenarioModel(
        schema_version=2,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec(
            targets=(
                TargetActorSpec(
                    name="walker",
                    asset=CatalogAssetSpec(id="car"),
                    mobility=NetworkRouteMobilitySpec(
                        route="random_walk",
                        seed=2,
                        graph_path="self-loops.graphml",
                    ),
                ),
            )
        ),
    )

    with pytest.raises(PosePreparationError, match="outgoing edge"):
        prepare_scenario(scenario, base_dir=tmp_path)
