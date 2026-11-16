"""Renderer-neutral selected-trial and geometric reference overlays."""

from __future__ import annotations

import numpy as np

from shared.sensing.volume.inspection import zero_radial_polygon

from ..model import RenderObject, Transform
from ..types.label_style import LabelCalloutStyle
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    TextLabelPayload,
)


def _line(name, points, color, *, through=False, strip=False):
    points = np.asarray(points, dtype=np.float32)
    pairs = (
        np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
        if strip
        else np.arange(len(points), dtype=np.int32).reshape(-1, 2)
    )
    return RenderObject(
        id="sensing:volume:" + name,
        payload=LineSetPayload(points=points, lines=pairs),
        material=MaterialPayload(base_color=color, line_width=2),
        metadata={
            "depth_compare": "<=",
            "depth_test": not through,
            "depth_write": not through,
            "render_order": 100 if through else 0,
        },
    )


def _label(name, position, text, color, through=False, *, callout_offset=None):
    metadata = {
        "depth_compare": "<=",
        "depth_test": not through,
        "depth_write": False,
        "render_order": 101 if through else 0,
    }
    if callout_offset is not None:
        style = LabelCalloutStyle(
            mode="callout",
            offset_x=callout_offset[0],
            offset_y=callout_offset[1],
            marker_size=10,
            on_top=through,
        )
        metadata.update(
            alpha_mode="blend",
            render_order=31000 if through else 0,
            label_callout=dict(style.to_dict(), base_position=np.asarray(position).tolist()),
            label_connector_origin=np.asarray(position).tolist(),
            label_background_rgba=(*(0.75 + 0.25 * np.asarray(color[:3])), 0.92),
            label_anchor_offset=8,
        )
    return RenderObject(
        id="sensing:volume:" + name,
        payload=TextLabelPayload(text=text, font_size=0.22),
        transform=Transform.from_translation(position),
        material=MaterialPayload(base_color=(0, 0, 0, 1) if callout_offset else color),
        metadata=metadata,
    )


def trial_objects(reader, key, rows, selected=None, *, vectors=False, through=False):
    """Never place missing report coordinates or mutate below-ground estimates."""
    trial = reader.cells.get(key)
    if not trial:
        return []
    ref = np.asarray(trial["position_m"], float)
    white = (1.0, 1.0, 1.0, 1.0)
    cyan = (0.1, 0.8, 0.95, 1.0)
    orange = (1.0, 0.65, 0.2, 1.0)
    objects = [
        _label(
            "tested_label",
            ref,
            (
                "Tested target"
                if trial.get("status") == "measured"
                else "Authored position · not measured"
            ),
            white,
            through,
            callout_offset=(-90, -50),
        ),
    ]
    for index, row in enumerate(rows):
        if row["position_m"] is None:
            continue
        p = np.asarray(row["position_m"], float)
        chosen = row["detection_id"] == selected
        color = cyan if row["matched"] else orange
        objects.append(
            RenderObject(
                id=f"sensing:volume:report:{index}",
                payload=PointCloudPayload(points=p[None, :].astype(np.float32)),
                material=MaterialPayload(base_color=color, point_size=15 if chosen else 10),
                metadata={
                    "depth_compare": "<=",
                    "depth_test": not through,
                    "depth_write": not through,
                    "render_order": 100 if through else 0,
                },
            )
        )
        if row["matched"] or chosen:
            objects.append(
                _label(
                    f"report_label:{index}",
                    p,
                    f"{'match' if row['matched'] else 'extra'} · {row['error_m']:.3f} m",
                    color,
                    through,
                    callout_offset=(95, -55 if row["matched"] else 55),
                )
            )
            if vectors:
                objects.append(_line(f"error_vector:{index}", [ref, p], color, through=through))
    return objects


def geometry_objects(
    reader, heading, forward, *, show_forward=False, show_motion=False, show_zero=False
):
    sensor = np.asarray(reader.metadata["sensor_position_m"], float)
    velocity = np.asarray(reader.headings[heading]["velocity_m_s"], float)
    span = np.linalg.norm([np.ptp(reader.axes[n]) for n in ("x", "y", "z")]) * 0.13
    objects = []
    for name, vector, enabled, color, label in (
        (
            "forward",
            forward,
            show_forward,
            (1.0, 0.85, 0.2, 1.0),
            "Array forward — orientation only",
        ),
        (
            "motion",
            velocity,
            show_motion,
            (0.7, 1.0, 0.5, 1.0),
            f"Translation {np.linalg.norm(velocity):g} m/s — not attitude",
        ),
    ):
        if not enabled or vector is None or np.linalg.norm(vector) == 0:
            continue
        direction = np.asarray(vector) / np.linalg.norm(vector)
        tip = sensor + span * direction
        axis = np.eye(3)[np.argmin(abs(direction))]
        side = np.cross(direction, axis) * span * 0.12
        objects.append(
            _line(
                name,
                [
                    sensor,
                    tip,
                    tip,
                    tip - span * 0.2 * direction + side,
                    tip,
                    tip - span * 0.2 * direction - side,
                ],
                color,
            )
        )
        objects.append(_label(name + "_label", tip, label, color))
    if show_zero:
        polygon = zero_radial_polygon(sensor, velocity, reader.axes)
        if len(polygon) >= 3:
            faces = np.array([[0, i, i + 1] for i in range(1, len(polygon) - 1)], np.int32)
            objects.append(
                RenderObject(
                    id="sensing:volume:zero_radial",
                    payload=MeshPayload(vertices=polygon.astype(np.float32), triangles=faces),
                    material=MaterialPayload(base_color=(0.2, 1.0, 0.8, 0.18), shader="unlit"),
                )
            )
            objects.append(
                _line(
                    "zero_radial_border",
                    np.vstack((polygon, polygon[0])),
                    (0.2, 1.0, 0.8, 1.0),
                    strip=True,
                )
            )
            objects.append(
                _label(
                    "zero_radial_label",
                    polygon.mean(axis=0),
                    "Zero direct radial velocity — geometry",
                    (0.2, 1.0, 0.8, 1.0),
                )
            )
    return objects
