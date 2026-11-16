"""Independent slice/filter/region contracts, including source immutability."""

from dataclasses import replace

import numpy as np
import pytest

from shared.coverage.spatial import coverage_roi_mask
from shared.coverage.volume import CoverageVolume, coverage_threshold_mask
from visualizer.src.coverage.volume_analysis import (
    contour_object,
    plane_model,
    plane_values,
    region_summary,
)
from visualizer.src.coverage.volume_inspection import scalar_template


@pytest.fixture
def model():
    z, y, x = np.mgrid[:4, :5, :6]
    values = (-90 + 2 * x + 3 * y + 4 * z).astype(np.float32)
    values[1, 1, 1] = np.nan
    zero = np.zeros(values.shape, bool)
    zero[0, 0, 0] = True
    values[0, 0, 0] = -270
    reasons = np.where(np.isnan(values), 0, np.where(zero, 2, 1)).astype(np.uint8)
    return CoverageVolume(
        values,
        (np.arange(6) * 2 + 10, np.arange(5) * 3 - 20, np.arange(4) * 4 + 5),
        "A",
        "rss_dbm/A",
        zero,
        ("fixture", 0, 0),
        reasons,
        ("A",),
    )


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_smoothing_preserves_holes_and_stays_within_plane(model, axis):
    original = model.values.copy()
    raw = plane_values(model, axis, 1)
    smooth = plane_values(model, axis, 1, "linear")
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(smooth))
    modified = model.values.copy()
    selector = [slice(None)] * 3
    selector[2 - axis] = 0
    modified[tuple(selector)] += 1000
    np.testing.assert_allclose(
        smooth, plane_values(replace(model, values=modified), axis, 1, "linear")
    )
    np.testing.assert_array_equal(model.values, original)
    one = plane_model(model, smooth, axis, 1)
    assert one.values.shape[2 - axis] == 1
    assert one.axes[axis][0] == model.axes[axis][1]


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_contour_segments_use_correct_world_axes(model, axis):
    values = plane_values(model, axis, 2)
    obj, levels = contour_object(model, values, axis, 2, 4, "test")
    assert obj is not None and len(levels) == 4
    xyz = obj.payload.points
    np.testing.assert_allclose(xyz[:, axis], model.axes[axis][2] + 0.04)
    for dimension in set(range(3)) - {axis}:
        assert xyz[:, dimension].min() >= model.axes[dimension][0]
        assert xyz[:, dimension].max() <= model.axes[dimension][-1]


def test_threshold_palette_keeps_failed_cells_transparent_with_grey_missing(model):
    passed = coverage_threshold_mask(model, -75)
    visible = passed | np.isnan(model.values) | model.zero_contributions
    template = scalar_template(model, -90, -55, visible_mask=visible, show_nonfinite=True)
    assert template.palette[0, 3] == 0
    assert np.all(template.values[~visible] == template.limits[0])
    assert template.values[1, 1, 1] > template.limits[0]
    assert np.all(template.palette[1:3, 3] == 1)
    assert np.all(template.values[passed] > template.threshold)


def test_loss_filter_and_projection_order(model):
    loss = replace(model, values=-model.values, quantity="path_loss_db/A")
    passed = coverage_threshold_mask(loss, 75)
    np.testing.assert_array_equal(passed, coverage_threshold_mask(model, -75))
    template = scalar_template(loss, 55, 90, visible_mask=passed)
    assert template.values[3, 4, 5] > template.values[3, 2, 5]
    assert template.values[0, 0, 0] == template.limits[0]


def test_region_counts_have_fixed_denominator_and_do_not_fill_missing(model):
    bounds = [[10, 14], [-20, -14], [5, 9]]
    region = coverage_roi_mask(model.axes, bounds)
    assert region.sum() == 18
    summary = region_summary(model, region, coverage_threshold_mask(model, -85))
    assert summary["total"] == 18
    assert summary["zero"] == 1 and summary["unknown"] == 1 and summary["finite"] == 16
    strict = region_summary(model, region, coverage_threshold_mask(model, -60))
    assert strict["total"] == 18 and strict["passing"] == 0
    assert region_summary(model, region, axis=2, index=3)["total"] == 0
    with pytest.raises(ValueError, match="no recorded"):
        coverage_roi_mask(model.axes, [[10.1, 10.2], [-20, -14], [5, 9]])
