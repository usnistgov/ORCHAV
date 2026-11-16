"""Recorded radio-position annotations shared by coverage and sensing views."""

from __future__ import annotations

import numpy as np

from ..model import RenderObject, Transform
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    PointCloudPayload,
    TextLabelPayload,
)


def site_locator_objects(position, name, axes, object_ids, *, readable=False, color=None):
    """Anchor a dot at the recorded position and a display-only leader above it.

    The stem is a locator, not a physical antenna mast. Readable labels are
    screen-sized annotations with a translucent backing on supporting renderers.
    Neither their height nor their opacity changes the recorded radio position.
    """
    if color is None:
        color = (0.0, 0.0, 0.0, 1.0) if readable else (1.0, 0.3, 0.85, 1.0)
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        return []
    span = max(float(np.ptp(axis)) for axis in axes)
    label_position = position.copy()
    label_position[2] = max(position[2], float(axes[2][-1])) + max(2.0, span * 0.08) * (
        2 if readable else 1
    )
    coordinates = ", ".join(f"{value:g}" for value in position)
    metadata = (
        dict(
            depth_test=False,
            depth_write=False,
            render_order=21000,
            alpha_mode="blend",
            pickable=False,
        )
        if readable
        else {}
    )
    label_metadata = dict(metadata)
    label_metadata["locator_role"] = "radio"
    label_metadata["locator_companions"] = list(object_ids[:2])
    label_metadata["label_connector_origin"] = position.tolist()
    if readable:
        label_metadata["label_background_rgba"] = (1, 1, 1, 0.8)
        # Keep the complete card above the stem, with a screen-sized gap.
        label_metadata["label_anchor"] = "bottom-center"
        label_metadata["label_anchor_offset"] = 12.0
        label_metadata["label_connector_origin"] = position.tolist()
        label_metadata["label_bold"] = True
    return [
        RenderObject(
            id=object_ids[0],
            payload=PointCloudPayload(points=position[None, :]),
            material=MaterialPayload(base_color=color, point_size=20 if readable else 14),
            metadata=metadata,
        ),
        RenderObject(
            id=object_ids[1],
            payload=LineSetPayload(
                points=np.stack((position, label_position)),
                lines=np.array([[0, 1]], dtype=np.int32),
            ),
            material=MaterialPayload(base_color=color, line_width=4 if readable else 2.5),
            metadata=metadata,
        ),
        RenderObject(
            id=object_ids[2],
            payload=TextLabelPayload(
                text=name if readable else f"{name}\n({coordinates}) m",
                font_size=0.6 if readable else 0.3,
                screen_space=True,
                outline_thickness=0.03 if readable else 0.15,
            ),
            material=MaterialPayload(base_color=(0, 0, 0, 1) if readable else color),
            transform=Transform.from_translation(label_position),
            metadata=label_metadata,
        ),
    ]
