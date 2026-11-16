"""Plane display and region summaries over immutable recorded coverage cells."""

from dataclasses import replace

import numpy as np

from shared.coverage.availability import CoverageAvailability

from ..model import RenderObject
from ..types.render_payloads import LineSetPayload, MaterialPayload
from .analysis import build_coverage_isoline, coverage_isoline_levels, coverage_metric_valid_mask
from .smoothing import smooth_coverage_values


def plane_values(model, axis, index, smoothing="none", region=None):
    """Extract and optionally smooth one plane, preserving its missing mask."""
    values = np.take(model.values, index, axis=2 - axis)
    valid = coverage_metric_valid_mask(values, model.quantity)
    valid &= ~np.take(model.zero_contributions, index, axis=2 - axis)
    if region is not None:
        valid &= np.take(region, index, axis=2 - axis)
    return smooth_coverage_values(np.where(valid, values, np.nan), smoothing)


def plane_model(model, values, axis, index):
    """A one-plane display model; original model remains authoritative for picks."""
    axes = list(model.axes)
    axes[axis] = axes[axis][index : index + 1]
    zeros = np.expand_dims(np.take(model.zero_contributions, index, axis=2 - axis), 2 - axis)
    return replace(
        model, values=np.expand_dims(values, 2 - axis), axes=tuple(axes), zero_contributions=zeros
    )


def contour_object(model, values, axis, index, count, object_id):
    """Reuse the common isoline algorithm and place it in XY, XZ or YZ."""
    horizontal, vertical = [dimension for dimension in range(3) if dimension != axis]
    step = model.spacing
    levels = coverage_isoline_levels(values, count, model.quantity)
    points, lines = [], []
    offset = 0
    for level in levels:
        xy, segments = build_coverage_isoline(
            values,
            grid_origin=np.array([model.axes[i][0] - step[i] / 2 for i in (horizontal, vertical)]),
            grid_spacing=np.array([step[i] for i in (horizontal, vertical)]),
            z_level=model.axes[axis][index],
            level=level,
            z_offset=0.04,
        )
        if not len(segments):
            continue
        xyz = np.empty_like(xy)
        xyz[:, horizontal], xyz[:, vertical], xyz[:, axis] = xy[:, 0], xy[:, 1], xy[:, 2]
        points.append(xyz)
        lines.append(segments + offset)
        offset += len(xyz)
    if not points:
        return None, levels
    return (
        RenderObject(
            id=object_id,
            payload=LineSetPayload(np.concatenate(points), np.concatenate(lines)),
            material=MaterialPayload(
                base_color=(0.04, 0.04, 0.04, 1), line_width=2.0, shader="unlit"
            ),
        ),
        levels,
    )


def region_summary(model, region, passed=None, *, axis=None, index=0):
    """Count original cells; the denominator never depends on the filter result."""
    select = (lambda a: np.take(a, index, axis=2 - axis)) if axis is not None else (lambda a: a)
    eligible = select(region)
    zeros = select(model.zero_contributions) & eligible
    finite = coverage_metric_valid_mask(select(model.values), model.quantity) & eligible & ~zeros
    invalid = (
        (select(model.availability) == CoverageAvailability.INVALID) & eligible
        if model.availability is not None
        else np.zeros_like(eligible)
    )
    total = int(eligible.sum())
    result = dict(
        total=total, finite=int(finite.sum()), zero=int(zeros.sum()), invalid=int(invalid.sum())
    )
    result["unknown"] = total - result["finite"] - result["zero"] - result["invalid"]
    if passed is not None:
        count = int((select(passed) & eligible).sum())
        result.update(passing=count, percent=100 * count / total if total else 0.0)
    return result
