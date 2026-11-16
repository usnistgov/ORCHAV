"""Renderer-neutral sample textures and semantic viewport legends."""

from dataclasses import replace

import numpy as np

from ..model import Transform
from ..services.viewport_hud_service import AnalysisLegend
from ..types.render_payloads import VolumePayload
from ..utils.colors import ensure_viridis_lut


def volume_template(reader, heading, region="outcomes", *, outcomes=None):
    """Encode categorical cells or occupied hit/miss cells for GPU ray marching."""
    if outcomes is None:
        outcomes = reader.outcome_grid(heading)
    miss_color, hit_color = [0.95, 0.45, 0.1, 1], [0.1, 0.8, 0.95, 1]
    if region == "outcomes":
        values = np.where(np.isfinite(outcomes), outcomes + 1, 0).astype(np.float32)
        palette = np.array([[0, 0, 0, 0], miss_color, hit_color], dtype=np.float32)
        payload = VolumePayload(values=values, palette=palette, limits=(0, 2))
    elif region in {"hit", "miss", "blocked"}:
        # NaN (unmeasured/excluded) must not become a missed cell. Occupied
        # samples retain their category color; zero stays transparent in slices.
        if region == "blocked":
            values = (reader.geometry_grid(heading) == 1).astype(np.float32)
            color = [0.65, 0.4, 0.95, 1]
        else:
            values = (outcomes == (1 if region == "hit" else 0)).astype(np.float32)
            color = hit_color if region == "hit" else miss_color
        palette = np.array([[0, 0, 0, 0], color], dtype=np.float32)
        payload = VolumePayload(values=values, palette=palette, limits=(0, 1), mode="iso")
    else:
        raise ValueError(f"Unknown outcome region: {region}")
    matrix = np.eye(4)
    for i, name in enumerate(("x", "y", "z")):
        matrix[i, i] = reader.axes[name][1] - reader.axes[name][0]
        matrix[i, 3] = reader.axes[name][0]
    return payload, Transform(matrix)


def error_volume_template(reader, heading, snapshot, maximum):
    """Color reports by nearest-world-position error; keep no reports separate."""
    payload, transform = volume_template(reader, heading)
    measured = np.isfinite(snapshot["outcomes"])
    errors = snapshot["nearest_error_m"]
    counts = snapshot["report_counts"]
    values = np.zeros(reader.shape, dtype=np.float32)
    values[measured & (counts == 0)] = 1
    values[measured & (counts > 0) & ~np.isfinite(errors)] = 2
    valid = measured & np.isfinite(errors)
    values[valid] = 3 + np.minimum(errors[valid] / maximum, 1) * 255
    palette = np.vstack(
        (
            [0, 0, 0, 0],
            [0.95, 0.45, 0.1, 1],
            [0.6, 0.6, 0.6, 1],
            np.column_stack((ensure_viridis_lut(), np.ones(256))),
        )
    ).astype(np.float32)
    return replace(payload, values=values, palette=palette, limits=(0, 258), mode="iso"), transform


def volume_legend(reader, heading, basis, display, settings):
    """Describe the visible quantity without conflating presence and matching."""
    lines = []
    if len(reader.headings) > 1:
        lines.append(reader.headings[heading]["name"])
    colorbar_label, colorbar_range = "", None
    if basis == "reports":
        title = (
            "Nearest estimate position error"
            if reader.is_fusion
            else "Nearest report position error"
        )
        swatches = (("No reports", "#f2731a"), ("Position unavailable", "#999999"))
        colorbar_label = "Error (m, capped)"
        colorbar_range = (0, settings["error_max"])
    elif basis == "geometry":
        title = "Direct path blockage"
        swatches = (("Blocked", "#a666f2"),)
    else:
        title = "Matched / missed"
        if basis == "range_velocity":
            lines.append(
                f"Match: range ≤ {settings['range_gate']:g} m · velocity ≤ {settings['velocity_gate']:g} m/s"
            )
        else:
            gate = (
                reader.metadata["position_gate_m"]
                if basis == "saved"
                else settings["position_gate"]
            )
            lines.append(f"Match: position error ≤ {gate:g} m")
        swatches = (("Matched", "#1accf2"), ("Missed", "#f2731a"))
        if display == "hit":
            swatches = swatches[:1]
        elif display == "miss":
            swatches = swatches[1:]
    return AnalysisLegend(
        title=title,
        lines=tuple(lines),
        swatches=swatches,
        colorbar_label=colorbar_label,
        colorbar_range=colorbar_range,
    )


def inspection_template(reader, heading, values, *, colors=None, maximum=None):
    """Encode categories or nonnegative counts with a distinct unknown code."""
    payload, transform = volume_template(reader, heading)
    valid = np.isfinite(values)
    encoded = np.zeros(values.shape, np.float32)
    if colors is not None:
        palette = [[0, 0, 0, 0]] + [
            [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1] for color in colors
        ]
        encoded[valid] = values[valid] + 1
    else:
        palette = np.vstack(([0, 0, 0, 0], np.column_stack((ensure_viridis_lut(), np.ones(256)))))
        encoded[valid] = 1 + np.minimum(values[valid] / maximum, 1) * 255
    return (
        replace(
            payload,
            values=encoded,
            palette=np.asarray(palette, np.float32),
            limits=(0, len(palette) - 1),
            mode="iso",
        ),
        transform,
    )
