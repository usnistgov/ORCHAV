"""Renderer-neutral orientation-frame payload helpers."""

from __future__ import annotations

import numpy as np

from ..model import RenderObjectState, Transform
from ..types.render_payloads import LineSetPayload, MaterialPayload, OrientationFramePayload

DEFAULT_ORIENTATION_FRAME_THICKNESS = 4.0


def orientation_frame_size(size: float) -> float:
    """Normalize an orientation-frame axis length."""
    return max(float(size), 0.0)


def make_orientation_frame_payload(
    size: float,
    *,
    thickness: float = DEFAULT_ORIENTATION_FRAME_THICKNESS,
) -> OrientationFramePayload:
    """Return a semantic renderer-neutral RGB coordinate-frame payload."""
    return OrientationFramePayload(
        size=orientation_frame_size(size),
        thickness=max(float(thickness), 0.0),
    )


def make_orientation_frame_lines_payload(size: float) -> LineSetPayload:
    """Return a line-based orientation-frame fallback payload."""
    axis_size = orientation_frame_size(size)
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_size, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, axis_size, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, axis_size],
        ],
        dtype=float,
    )
    lines = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0],
            [0.0, 0.25, 1.0, 1.0],
        ],
        dtype=float,
    )
    return LineSetPayload(points=points, lines=lines, colors=colors)


def make_orientation_frame_handle(
    render_id: str,
    *,
    size: float,
    thickness: float = DEFAULT_ORIENTATION_FRAME_THICKNESS,
    transform: Transform | np.ndarray | None = None,
    visible: bool = False,
) -> RenderObjectState:
    """Return a renderer-neutral RGB axis frame handle."""
    axis_size = orientation_frame_size(size)
    axis_thickness = max(float(thickness), 0.0)
    world_transform = transform if isinstance(transform, Transform) else Transform.identity()
    if transform is not None and not isinstance(transform, Transform):
        world_transform = Transform(np.asarray(transform, dtype=float))
    return RenderObjectState(
        id=render_id,
        payload=make_orientation_frame_payload(axis_size, thickness=axis_thickness),
        material=MaterialPayload(
            base_color=(1.0, 1.0, 1.0, 1.0),
            shader="unlit",
            line_width=axis_thickness,
        ),
        world_transform=world_transform,
        visible=bool(visible),
        is_edge=False,
        metadata={
            "type": "orientation_frame",
            "size": axis_size,
            "thickness": axis_thickness,
            "pickable": False,
        },
    )
