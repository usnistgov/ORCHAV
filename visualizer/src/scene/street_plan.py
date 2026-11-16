"""Schematic street geometry and metric-grid style shared by both workspaces."""

from __future__ import annotations

import math

import numpy as np

from ..types.render_payloads import LineSetPayload

BACKGROUND = "#17232d"
GRID_COLOR = "#30414e"
STREET_COLOR = "#78828d"


def grid_spacing(width, height):
    return 10 ** math.floor(math.log10(max(width, height, 1) / 8))


def graph_polylines(graph):
    """Use canonical curved local edges, including the Builder's travel-mode filter."""
    from generator.core.scenario_actors.resources import network_route_points

    for start, end in graph.edges():
        points = network_route_points(graph, [start, end])
        if not np.isfinite(points).all():
            raise ValueError("Street graph positions must be finite local coordinates in metres.")
        yield points


def load_street_polylines(path):
    from generator.core.scenario_actors.resources import load_network_graph

    return tuple(graph_polylines(load_network_graph(path, travel_mode="pedestrian")))


def schematic_payloads(bounds, streets):
    """Return a bounded grid and flattened streets, without changing source positions."""
    low, high = np.asarray(bounds, dtype=float)
    center = (low + high) / 2
    extent = np.maximum(high - low, 20)
    low, high = center - extent * 0.55, center + extent * 0.55
    spacing = grid_spacing(*extent)
    segments = []
    for x in np.arange(math.floor(low[0] / spacing) * spacing, high[0] + spacing, spacing):
        segments.append(((x, low[1], 0), (x, high[1], 0)))
    for y in np.arange(math.floor(low[1] / spacing) * spacing, high[1] + spacing, spacing):
        segments.append(((low[0], y, 0), (high[0], y, 0)))
    grid = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
    points, lines = [], []
    for street in streets:
        start = len(points)
        points.extend((p[0], p[1], 0.02) for p in street)
        lines.extend((i, i + 1) for i in range(start, len(points) - 1))
    return (
        LineSetPayload(points=grid, lines=np.arange(len(grid), dtype=np.int32).reshape(-1, 2)),
        LineSetPayload(
            points=np.asarray(points, dtype=np.float32).reshape(-1, 3),
            lines=np.asarray(lines, dtype=np.int32).reshape(-1, 2),
        ),
    )
