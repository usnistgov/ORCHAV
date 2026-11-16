"""Named-street selection using the Generator's cached routing graph."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from generator.core.scenario_actors.resources import load_network_graph, network_route_points
from shared.scenarios.paths import resolve_actor_resource


@dataclass(frozen=True)
class StreetRoute:
    start: str
    end: str
    points: tuple
    length_m: float


def street_names(value: object) -> tuple[str, ...]:
    """Handle both scalar OSM names and GraphML's serialized name lists."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(sorted({str(v).strip() for v in values if v and str(v).strip()}))


class StreetNetwork:
    """Offline road preview and deterministic, reachable route selection."""

    def __init__(self, path: Path, travel_mode: str):
        self.path = Path(path)
        self.graph = load_network_graph(self.path, travel_mode=travel_mode)
        self.streets: dict[str, set] = {}
        for start, end, data in self.graph.edges(data=True):
            for name in street_names(data.get("name")):
                self.streets.setdefault(name, set()).update((start, end))

    def routes(
        self,
        *,
        count: int,
        seed: int,
        start_street: str | None = None,
        end_street: str | None = None,
        altitude_m: float = 0.0,
    ) -> tuple[StreetRoute, ...]:
        """Choose distinct starts; fail atomically if the requested batch cannot fit.

        Endpoints lie on the selected streets. A junction can belong to several
        streets; zero-length routes are excluded. Directed reachability and
        weighted paths have exactly the Generator's meaning.
        """
        if not 1 <= count <= 100:
            raise ValueError("Choose between 1 and 100 actors.")
        for name in (start_street, end_street):
            if name is not None and name not in self.streets:
                raise ValueError(f"Street is not available for this travel mode: {name}")
        starts = sorted(self.streets[start_street] if start_street else self.graph, key=str)
        ends = self.streets[end_street] if end_street else set(self.graph)
        rng = np.random.default_rng(seed)
        result = []
        positions = set()
        for index in rng.permutation(len(starts)):
            start = starts[int(index)]
            lengths, paths = nx.single_source_dijkstra(self.graph, start, weight="length")
            candidates = sorted((n for n in ends if n in lengths and lengths[n] > 1e-6), key=str)
            if not candidates:
                continue
            end = candidates[int(rng.integers(len(candidates)))]
            points = network_route_points(self.graph, paths[end], altitude_m=altitude_m)
            origin = tuple(round(v, 4) for v in points[0])
            if origin in positions:
                continue
            positions.add(origin)
            length = float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())
            result.append(StreetRoute(str(start), str(end), points, length))
            if len(result) == count:
                return tuple(result)
        if not result:
            raise ValueError(
                "No connected route between these streets for this travel mode. "
                "Choose another street or travel mode; one-way restrictions are respected."
            )
        raise ValueError(
            f"Only {len(result)} distinct connected starting positions are available. "
            "Reduce the count or broaden the street selection. No actors were added."
        )

    def edges(self):
        """Yield canonical local polylines, including curved streets."""
        from ..scene.street_plan import graph_polylines

        yield from graph_polylines(self.graph)


def scenario_graph(scenario, root: Path) -> Path | None:
    """Prefer collected scene graphs, including unsaved OSM build artifacts."""
    for item in (*scenario.dependencies, *scenario.resources):
        if (
            "graph" in str(item.kind) or item.source_path.name == "street_network.graphml"
        ) and item.source_path.is_file():
            return item.source_path
    for actor in scenario.actors:
        raw = getattr(actor.mobility, "graph_path", None)
        if raw:
            path = resolve_actor_resource(raw, scenario_root=root)
            if path.is_file():
                return path
    if scenario.scene and scenario.scene.source == "local":
        path = resolve_actor_resource(scenario.scene.id, scenario_root=root)
        candidate = path.parent / "street_network.graphml"
        if candidate.is_file():
            return candidate
    candidate = root / "street_network.graphml"
    return candidate if candidate.is_file() else None
