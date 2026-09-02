"""Renderer-neutral payload helpers for trajectory-builder previews."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..model import RenderObject, Transform
from ..types.render_payloads import LineSetPayload, MaterialPayload
from .geometry_payload_factory import make_sphere_payload


def _rgba(color: Iterable[float]) -> tuple[float, float, float, float]:
    """Coerce user preview colors into clamped RGBA material values."""
    values = np.asarray(list(color), dtype=float).reshape(-1)
    if values.size < 3:
        values = np.asarray([1.0, 1.0, 1.0], dtype=float)
    rgb = np.clip(values[:3], 0.0, 1.0)
    alpha = float(values[3]) if values.size >= 4 else 1.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(np.clip(alpha, 0.0, 1.0)))


def make_trajectory_marker_object(
    render_id: str,
    point: Iterable[float],
    *,
    radius: float,
    color: Iterable[float],
) -> RenderObject:
    """Return a render object for one trajectory preview marker."""
    return RenderObject(
        id=render_id,
        payload=make_sphere_payload(radius=radius, color=color),
        material=MaterialPayload(base_color=_rgba(color), roughness=0.5),
        transform=Transform.from_translation(point),
        metadata={"type": "trajectory_preview_marker"},
    )


def make_trajectory_line_object(
    render_id: str,
    points: Iterable[Iterable[float]],
    *,
    color: Iterable[float],
) -> RenderObject | None:
    """Return a render object for a polyline trajectory preview."""
    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        return None
    pts = pts[:, :3]
    line_count = len(pts) - 1
    indices = np.column_stack([np.arange(line_count), np.arange(1, len(pts))]).astype(np.int32)
    colors = np.tile(np.asarray(_rgba(color)[:3], dtype=float), (line_count, 1))
    return RenderObject(
        id=render_id,
        payload=LineSetPayload(points=pts, lines=indices, colors=colors),
        material=MaterialPayload(base_color=_rgba(color), shader="unlit"),
        is_edge=True,
        metadata={"type": "trajectory_preview_line"},
    )
