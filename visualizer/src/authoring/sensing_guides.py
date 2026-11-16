"""Visualize configured sensing geometry without claiming detection coverage."""

from __future__ import annotations

import numpy as np

from shared.scenarios.sensing_groups import normalize_sensing_groups

from ..types.render_payloads import LineSetPayload
from .viewport_port import GuideOverlay


def line_guide(name, polylines, color=(0.2, 0.8, 1.0, 1.0)) -> GuideOverlay:
    points, lines = [], []
    for polyline in polylines:
        start = len(points)
        points.extend(polyline)
        lines.extend((i, i + 1) for i in range(start, len(points) - 1))
    return GuideOverlay(
        name,
        LineSetPayload(
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            np.asarray(lines, dtype=np.int32).reshape(-1, 2),
        ),
        color,
    )


def range_gate_lines(tx, rx, path_length):
    """Ellipsoid whose two focus distances sum to the configured path length."""
    tx, rx = np.asarray(tx, float), np.asarray(rx, float)
    delta = rx - tx
    separation = np.linalg.norm(delta)
    a, c = float(path_length) / 2, separation / 2
    if a <= c or a <= 0:
        return []
    b = np.sqrt(a * a - c * c)
    x = delta / separation if separation > 1e-9 else np.array((1.0, 0.0, 0.0))
    helper = np.array((0.0, 0.0, 1.0)) if abs(x[2]) < 0.9 else np.array((0.0, 1.0, 0.0))
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    centre = (tx + rx) / 2
    t = np.linspace(0, 2 * np.pi, 97)
    return [
        centre + np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
        for u, v in ((a * x, b * y), (a * x, b * z), (b * y, b * z))
    ]


def sensing_guides(mapping: dict, actors, *, display_length: float = 10.0):
    """Use prepared radio poses, imported range gates and world-angle path filters.

    Sionna path phi_r/theta_r filters are world angles. They are deliberately
    separate from the radio-local +X boresight and do not rotate with the array.
    """
    raw = mapping.get("sensing")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return (), "No enabled sensing configuration."
    config = normalize_sensing_groups(raw)
    radios = [
        a for a in actors if a.role in {"tx", "rx"} and a.visible and a.current_position is not None
    ]
    guides, notes = [], ["Arrows: sensor-local +X boresight. Guides are not predicted coverage."]
    maximum = config.get("max_detection_range_m")
    length = min(float(maximum), display_length) if maximum and maximum > 0 else display_length
    length = max(length, 0.1)
    for actor in radios:
        if actor.orientation_matrix is None:
            continue
        origin = np.asarray(actor.current_position)
        rotation = np.asarray(actor.orientation_matrix)[:3, :3]
        forward, side = rotation[:, 0], rotation[:, 1]
        tip = origin + forward * length
        guides.append(
            line_guide(
                f"boresight:{actor.actor_id}",
                [
                    (origin, tip),
                    (
                        tip - forward * length * 0.2 + side * length * 0.08,
                        tip,
                        tip - forward * length * 0.2 - side * length * 0.08,
                    ),
                ],
            )
        )
    mode = config.get("range_mode")
    transmitters, receivers = [a for a in radios if a.role == "tx"], [
        a for a in radios if a.role == "rx"
    ]
    if mode in {"monostatic", "bistatic"}:
        for tx in transmitters:
            for rx in receivers:
                for key in ("min_detection_range_m", "max_detection_range_m"):
                    distance = config.get(key)
                    if distance is not None and distance > 0:
                        lines = range_gate_lines(
                            tx.current_position,
                            rx.current_position,
                            distance * (2 if mode == "monostatic" else 1),
                        )
                        if lines:
                            guides.append(
                                line_guide(
                                    f"range:{tx.actor_id}:{rx.actor_id}:{key}",
                                    lines,
                                    (0.95, 0.65, 0.15, 0.8),
                                )
                            )
        if any(k in config for k in ("min_detection_range_m", "max_detection_range_m")):
            notes.append("Gold: configured range gates (TX–RX path-length surfaces).")
    elif maximum is not None:
        notes.append("Range guides need an explicit monostatic/bistatic interpretation.")
    if config.get("aoa_filter_enabled"):
        az0, az1 = config.get("aoa_az_min_deg"), config.get("aoa_az_max_deg")
        el0, el1 = config.get("aoa_el_min_deg"), config.get("aoa_el_max_deg")
        if any(v is not None for v in (az0, az1, el0, el1)):
            az0, az1 = -180 if az0 is None else az0, 180 if az1 is None else az1
            el0, el1 = -90 if el0 is None else el0, 90 if el1 is None else el1
            if az1 < az0:
                az1 += 360
            azimuth = np.deg2rad(np.linspace(az0, az1, 65))
            elevation = np.deg2rad(np.linspace(el0, el1, 33))

            def directions(az, el):
                return np.stack(
                    np.broadcast_arrays(
                        np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
                    ),
                    axis=-1,
                )

            for rx in receivers:
                origin = np.asarray(rx.current_position)
                radius = (
                    float(maximum)
                    if maximum
                    and maximum > 0
                    and transmitters
                    and np.allclose([t.current_position for t in transmitters], origin)
                    else display_length
                )
                curves = [origin + radius * directions(azimuth, np.deg2rad(e)) for e in (el0, el1)]
                curves += [
                    origin + radius * directions(np.deg2rad(a), elevation) for a in (az0, az1)
                ]
                curves += [(origin, curve[0]) for curve in curves]
                guides.append(line_guide(f"angles:{rx.actor_id}", curves, (0.75, 0.4, 1.0, 0.9)))
            notes.append(
                "Purple: configured world-angle path selection; arc radius is a display aid."
            )
    return tuple(guides), " ".join(notes)
