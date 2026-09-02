"""Renderer-neutral loaders for resource-backed mobility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from shared.scenarios.paths import resolve_actor_resource

from ._adapters import discriminator, value
from .errors import PosePreparationError
from .mobility import (
    _as_position,
    _finalize,
    _polyline_length,
    _sample_polyline,
    _traversal_distances,
)
from .types import Position3, PreparedMobility, Timeline

_NETWORK_HIGHWAYS: dict[str, frozenset[str]] = {
    "pedestrian": frozenset(
        {
            "footway",
            "path",
            "pedestrian",
            "steps",
            "living_street",
            "residential",
            "service",
            "track",
            "unclassified",
            "tertiary",
            "secondary",
            "primary",
        }
    ),
    "bike": frozenset(
        {
            "cycleway",
            "path",
            "living_street",
            "residential",
            "service",
            "unclassified",
            "tertiary",
            "secondary",
            "primary",
            "trunk",
        }
    ),
    "car": frozenset(
        {
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
            "living_street",
            "unclassified",
            "service",
        }
    ),
}


def prepare_resource_mobility(
    spec: object,
    timeline: Timeline,
    *,
    base_dir: str | Path | None,
    path: str,
) -> PreparedMobility:
    """Resolve a mesh-position sequence or cached network graph locally."""

    mobility_type = discriminator(spec)
    if mobility_type == "mesh_sequence":
        points = _load_mesh_positions(spec, base_dir=base_dir, path=path)
        return _prepare_mesh_points(points, spec, timeline, path=path)
    if mobility_type == "network_route":
        points = _load_network_route(spec, base_dir=base_dir, path=path)
        distances = _traversal_distances(_polyline_length(points), spec, timeline, path=path)
        return _finalize(
            _sample_polyline(points, distances),
            timeline,
            physical_velocity=True,
            path=path,
        )
    raise PosePreparationError(
        "unsupported_resource_mobility",
        path,
        f"unsupported resource mobility type {mobility_type!r}",
    )


def _load_mesh_positions(
    spec: object,
    *,
    base_dir: str | Path | None,
    path: str,
) -> tuple[Position3, ...]:
    resource_path = _resolve_path(
        str(value(spec, "positions_path")),
        base_dir=base_dir,
        path=f"{path}.positions_path",
    )
    key = str(value(spec, "position_key", default="positions"))
    suffix = resource_path.suffix.lower()
    try:
        if suffix == ".npy":
            raw = np.load(resource_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(resource_path, allow_pickle=False) as archive:
                if key not in archive:
                    raise KeyError(key)
                raw = archive[key]
        elif suffix in (".yaml", ".yml"):
            payload = yaml.safe_load(resource_path.read_text(encoding="utf-8"))
            raw = _mapping_value(payload, key)
        elif suffix == ".json":
            payload = json.loads(resource_path.read_text(encoding="utf-8"))
            raw = _mapping_value(payload, key)
        elif suffix in (".csv", ".txt"):
            raw = np.loadtxt(resource_path, delimiter="," if suffix == ".csv" else None)
        elif suffix in (".h5", ".hdf5"):
            import h5py

            with h5py.File(resource_path, "r") as handle:
                raw = np.asarray(handle[key])
        else:
            raise ValueError(f"unsupported position resource suffix {suffix!r}")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise PosePreparationError(
            "invalid_position_resource",
            f"{path}.positions_path",
            str(exc),
        ) from exc
    return _normalize_points(raw, path=f"{path}.positions_path")


def _prepare_mesh_points(
    points: tuple[Position3, ...],
    spec: object,
    timeline: Timeline,
    *,
    path: str,
) -> PreparedMobility:
    if len(points) == 1:
        positions = [points[0]] * timeline.steps
    else:
        distances = _traversal_distances(_polyline_length(points), spec, timeline, path=path)
        interpolation = str(value(spec, "interpolation", default="linear"))
        if interpolation == "linear":
            positions = _sample_polyline(points, distances)
        elif interpolation == "step":
            positions = _sample_step(points, distances)
        else:
            raise PosePreparationError(
                "invalid_mesh_interpolation",
                f"{path}.interpolation",
                "mesh sequence interpolation must be linear or step",
            )
    return _finalize(positions, timeline, physical_velocity=True, path=path)


def _load_network_route(
    spec: object,
    *,
    base_dir: str | Path | None,
    path: str,
) -> tuple[Position3, ...]:
    graph_path_raw = value(spec, "graph_path", default=None)
    if graph_path_raw is None:
        if base_dir is None:
            raise PosePreparationError(
                "network_route_unresolved",
                f"{path}.graph_path",
                "network_route requires graph_path or a scenario directory containing "
                "street_network.graphml",
            )
        graph_path = _resolve_path(
            "street_network.graphml",
            base_dir=base_dir,
            path=f"{path}.graph_path",
        )
    else:
        graph_path = _resolve_path(
            str(graph_path_raw),
            base_dir=base_dir,
            path=f"{path}.graph_path",
        )
    try:
        import networkx as nx
    except ImportError as exc:
        raise PosePreparationError(
            "missing_network_route_dependency",
            path,
            "networkx is required to prepare a cached network_route",
        ) from exc
    try:
        if graph_path.suffix.lower() in (".graphml", ".xml"):
            graph = nx.read_graphml(graph_path)
        elif graph_path.suffix.lower() == ".json":
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            graph = nx.node_link_graph(payload, edges="links")
        else:
            raise ValueError("cached network route must be GraphML or node-link JSON")
    except (OSError, ValueError, nx.NetworkXException) as exc:
        raise PosePreparationError("invalid_network_graph", f"{path}.graph_path", str(exc)) from exc

    _normalize_network_edge_lengths(graph, path=f"{path}.graph_path")
    travel_mode = str(value(spec, "travel_mode", default="pedestrian"))
    graph = _filter_network_graph(graph, travel_mode)
    nodes = sorted(graph.nodes, key=str)
    if len(nodes) < 2:
        raise PosePreparationError(
            "invalid_network_graph",
            f"{path}.graph_path",
            f"network graph needs at least two nodes usable by {travel_mode!r}",
        )

    route_type = str(value(spec, "route", default="shortest_path"))
    seed_raw = value(spec, "seed", default=None)
    seed = 0 if seed_raw is None else int(seed_raw)
    start_raw = value(spec, "start_node", default=None)
    end_raw = value(spec, "end_node", default=None)
    try:
        if route_type == "shortest_path":
            if end_raw is not None:
                start = _resolve_node(graph, start_raw, path=f"{path}.start_node")
                end = _resolve_node(graph, end_raw, path=f"{path}.end_node")
            else:
                start, end = _seeded_route_endpoints(graph, seed)
            route_nodes = nx.shortest_path(graph, start, end, weight="length")
        elif route_type == "random_walk":
            start = _seeded_random_walk_start(graph, seed, path=path)
            route_nodes = _random_walk_nodes(graph, start, seed, max(2, len(nodes)))
        else:
            raise ValueError(f"unsupported network route strategy {route_type!r}")
    except nx.NetworkXException as exc:
        raise PosePreparationError("network_route_not_found", path, str(exc)) from exc
    altitude = float(value(spec, "altitude_m", default=0.0))
    return _expand_network_route(
        graph,
        route_nodes,
        altitude_m=altitude,
        path=f"{path}.graph_path",
    )


def _resolve_path(raw: str, *, base_dir: str | Path | None, path: str) -> Path:
    if base_dir is None and not Path(raw).is_absolute():
        raise PosePreparationError(
            "missing_resource_base_dir",
            path,
            "relative resource path requires prepare_scenario(base_dir=...)",
        )
    scenario_root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    candidate = resolve_actor_resource(raw, scenario_root=scenario_root)
    if not candidate.is_file():
        raise PosePreparationError(
            "missing_resource", path, f"resource does not exist: {candidate}"
        )
    return candidate


def _mapping_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key not in payload:
            raise KeyError(key)
        return payload[key]
    if key == "positions":
        return payload
    raise KeyError(key)


def _normalize_points(raw: object, *, path: str) -> tuple[Position3, ...]:
    array: npt.NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 1 or not np.isfinite(array).all():
        raise PosePreparationError(
            "invalid_position_resource",
            path,
            "position resource must be a finite N x 3 numeric array",
        )
    return tuple(_as_position(row) for row in array)


def _sample_step(
    points: tuple[Position3, ...],
    distances: tuple[float, ...],
) -> list[Position3]:
    array: npt.NDArray[np.float64] = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return [
        points[min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 1)]
        for distance in distances
    ]


def _filter_network_graph(graph: Any, travel_mode: str) -> Any:
    """Return a graph containing edges usable by the requested actor mode."""

    if travel_mode == "drone":
        return graph
    allowed = _NETWORK_HIGHWAYS.get(travel_mode)
    if allowed is None:
        raise PosePreparationError(
            "invalid_network_travel_mode",
            "network_route.travel_mode",
            f"unsupported network travel mode {travel_mode!r}",
        )

    filtered = graph.copy()
    multigraph = bool(filtered.is_multigraph())
    edge_records = (
        filtered.edges(keys=True, data=True)
        if multigraph
        else ((start, end, None, data) for start, end, data in filtered.edges(data=True))
    )
    removed: list[tuple[object, object, object | None]] = []
    for start, end, key, attributes in edge_records:
        highway_types = _highway_types(attributes.get("highway"))
        if highway_types and highway_types.isdisjoint(allowed):
            removed.append((start, end, key))
    if multigraph:
        filtered.remove_edges_from(removed)
    else:
        filtered.remove_edges_from((start, end) for start, end, _ in removed)
    filtered.remove_nodes_from(node for node in tuple(filtered.nodes) if filtered.degree(node) == 0)
    return filtered


def _normalize_network_edge_lengths(graph: Any, *, path: str) -> None:
    """Make cached GraphML edge weights numeric before route selection."""

    edge_records = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else ((start, end, None, data) for start, end, data in graph.edges(data=True))
    )
    for start, end, _key, attributes in edge_records:
        try:
            length = float(attributes["length"])
        except (KeyError, TypeError, ValueError):
            points = _network_edge_geometry(graph, start, end, attributes, path=path)
            length = float(
                np.linalg.norm(
                    np.diff(np.asarray(points, dtype=np.float64), axis=0),
                    axis=1,
                ).sum()
            )
        if not np.isfinite(length) or length < 0.0:
            raise PosePreparationError(
                "invalid_network_graph",
                path,
                f"edge {start!r} -> {end!r} has invalid length {length!r}",
            )
        attributes["length"] = length


def _highway_types(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip().lower() for item in raw if str(item).strip())
    text = str(raw).strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return frozenset(str(item).strip().lower() for item in decoded if str(item).strip())
    return frozenset({text.lower()}) if text else frozenset()


def _seeded_route_endpoints(graph: Any, seed: int) -> tuple[object, object]:
    """Choose a deterministic reachable start/end pair from a cached graph."""

    import networkx as nx

    nodes = np.asarray(sorted(graph.nodes, key=str), dtype=object)
    rng = np.random.default_rng(seed)
    for start_index in rng.permutation(len(nodes)):
        start = nodes[int(start_index)]
        if graph.is_directed():
            reachable = sorted(nx.descendants(graph, start), key=str)
        else:
            reachable = sorted(
                set(nx.node_connected_component(graph, start)) - {start},
                key=str,
            )
        if reachable:
            end = reachable[int(rng.integers(0, len(reachable)))]
            return start, end
    raise PosePreparationError(
        "network_route_not_found",
        "network_route",
        "network graph has no route connecting two distinct nodes",
    )


def _seeded_random_walk_start(graph: Any, seed: int, *, path: str) -> object:
    """Choose a deterministic node with at least one outgoing walk edge."""

    candidates = sorted(
        (
            node
            for node in graph.nodes
            if any(neighbor != node for neighbor in graph.neighbors(node))
        ),
        key=str,
    )
    if not candidates:
        raise PosePreparationError(
            "network_route_not_found",
            path,
            "network graph has no node with an outgoing edge",
        )
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def _expand_network_route(
    graph: Any,
    route_nodes: list[object],
    *,
    altitude_m: float,
    path: str,
) -> tuple[Position3, ...]:
    """Expand graph edges into their cached local-coordinate polylines."""

    if not route_nodes:
        raise PosePreparationError(
            "network_route_not_found",
            path,
            "network route contains no nodes",
        )
    if len(route_nodes) == 1:
        first_xy = _network_node_xy(graph, route_nodes[0], path=path)
        return ((first_xy[0], first_xy[1], altitude_m),)

    points: list[Position3] = []
    for start, end in zip(route_nodes, route_nodes[1:]):
        edge_points, bridge_offset = _network_edge_points(graph, start, end, path=path)
        if not points:
            first_x, first_y = edge_points[0]
            points.append((first_x, first_y, altitude_m + bridge_offset))
        points.extend(
            (float(x_coord), float(y_coord), altitude_m + bridge_offset)
            for x_coord, y_coord in edge_points[1:]
        )
    return tuple(points)


def _network_edge_points(
    graph: Any,
    start: object,
    end: object,
    *,
    path: str,
) -> tuple[tuple[tuple[float, float], ...], float]:
    edge_data = graph.get_edge_data(start, end)
    if edge_data is None:
        return (
            _network_node_xy(graph, start, path=path),
            _network_node_xy(graph, end, path=path),
        ), 0.0
    candidates = tuple(edge_data.values()) if graph.is_multigraph() else (edge_data,)
    best: tuple[tuple[tuple[float, float], ...], float] | None = None
    best_length = float("inf")
    for attributes in candidates:
        points = _network_edge_geometry(graph, start, end, attributes, path=path)
        length_raw = attributes.get("length")
        try:
            length = float(length_raw)
        except (TypeError, ValueError):
            length = float(
                np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1).sum()
            )
        if length < best_length:
            try:
                bridge_offset = float(attributes.get("bridge_z_offset", 0.0))
            except (TypeError, ValueError):
                bridge_offset = 0.0
            best = points, bridge_offset
            best_length = length
    if best is None:
        raise PosePreparationError(
            "invalid_network_graph",
            path,
            f"edge {start!r} -> {end!r} has no usable attributes",
        )
    return best


def _network_edge_geometry(
    graph: Any,
    start: object,
    end: object,
    attributes: dict[str, Any],
    *,
    path: str,
) -> tuple[tuple[float, float], ...]:
    raw = attributes.get("geometry_local")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            points = tuple((float(point[0]), float(point[1])) for point in raw)
        except (IndexError, TypeError, ValueError):
            points = ()
        if points:
            start_xy = _network_node_xy(graph, start, path=path)
            direct = float(np.linalg.norm(np.asarray(points[0]) - np.asarray(start_xy)))
            reverse = float(np.linalg.norm(np.asarray(points[-1]) - np.asarray(start_xy)))
            return tuple(reversed(points)) if reverse < direct else points
    return (
        _network_node_xy(graph, start, path=path),
        _network_node_xy(graph, end, path=path),
    )


def _network_node_xy(graph: Any, node: object, *, path: str) -> tuple[float, float]:
    attributes = graph.nodes[node]
    if "x_local" in attributes and "y_local" in attributes:
        return float(attributes["x_local"]), float(attributes["y_local"])
    if "x" in attributes and "y" in attributes:
        return float(attributes["x"]), float(attributes["y"])
    raise PosePreparationError(
        "network_node_missing_position",
        path,
        f"node {node!r} lacks local or x/y coordinates",
    )


def _resolve_node(graph: Any, raw: object, *, path: str) -> object:
    if raw in graph:
        return raw
    normalized = str(raw)
    if normalized in graph:
        return normalized
    raise PosePreparationError("missing_network_node", path, f"node {raw!r} does not exist")


def _random_walk_nodes(graph: Any, start: object, seed: int, count: int) -> list[object]:
    rng = np.random.default_rng(seed)
    route = [start]
    for _ in range(count - 1):
        neighbors = [neighbor for neighbor in graph.neighbors(route[-1]) if neighbor != route[-1]]
        if not neighbors:
            break
        route.append(neighbors[int(rng.integers(0, len(neighbors)))])
    return route
