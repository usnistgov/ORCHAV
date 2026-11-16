"""Recorded interaction identity and deterministic finite-persistence semantics."""

import numpy as np
import pytest

from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry


def fixture():
    paths = [[(float(j), float(i), 0) for j in range(6)] for i in range(3)]
    catalog = _catalog([10, 20, 30], points=paths)
    types = np.array([0, 1, 99, 37, 0, 0] * 3, dtype=np.uint8)
    catalog.canonical_data.itype[:] = types
    catalog.canonical_data.order[:] = np.repeat([4, 5, 6], 6)
    return catalog


def test_global_metadata_follows_reordered_subset_and_keeps_unknown_virtual():
    catalog = fixture()
    original = catalog.canonical_data.itype.copy()
    g = CinematicPulseGeometry(catalog, [2, 0])
    np.testing.assert_array_equal(g.interaction_path_ids, [2, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(g.interaction_point_indices, [13, 14, 15, 1, 2, 3])
    np.testing.assert_array_equal(g.interaction_types, [1, 99, 37, 1, 99, 37])
    np.testing.assert_array_equal(g.interaction_orders, [6, 6, 6, 4, 4, 4])
    np.testing.assert_array_equal(g.interaction_ordinals, [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(
        g.vertices, catalog.canonical_data.points[g.interaction_point_indices]
    )
    np.testing.assert_array_equal(catalog.canonical_data.itype, original)
    assert not g.interaction_point_indices.flags.writeable
    assert not g.interaction_orders.flags.writeable


def test_metadata_stays_aligned_beyond_former_geometry_budget():
    paths = [[(float(j), float(i), 0) for j in range(12000)] for i in range(2)]
    catalog = _catalog([1, 2], points=paths)
    catalog.canonical_data.itype[1:11999] = 1
    g = CinematicPulseGeometry(catalog, [0, 1])
    assert g.omitted_count == 0
    assert g.ids.tolist() == [0, 1]
    assert len(g.interaction_path_ids) == len(g.vertices) == 11998
    assert np.all(g.interaction_path_ids == 0)
    assert g.interaction_point_indices.max() == 11998


def test_dwell_holds_then_fades_and_scrubs_without_history():
    g = CinematicPulseGeometry(fixture(), [0])
    t = g.interaction_ns[0]
    hold = g.extent * 2 / 5
    fade = g.extent * 0.035
    kwargs = dict(interaction_mode="dwell", dwell_seconds=2)
    assert g.sample(t - 0.001, **kwargs).interaction_colors[0, 3] == 0
    assert g.sample(t, **kwargs).interaction_colors[0, 3] == pytest.approx(0.85)
    assert g.sample(t + hold, **kwargs).interaction_colors[0, 3] == pytest.approx(0.85)
    assert g.sample(t + hold + fade / 2, **kwargs).interaction_colors[0, 3] == pytest.approx(0.425)
    assert g.sample(t + hold + fade + 0.001, **kwargs).interaction_colors[0, 3] == 0
    assert g.sample(t, **kwargs).interaction_colors[0, 3] == pytest.approx(0.85)
    assert g.sample(t - 0.001, **kwargs).interaction_colors[0, 3] == 0


def test_zero_dwell_equals_flash_and_retains_path_motion():
    g = CinematicPulseGeometry(fixture(), [0])
    for time in (0, g.interaction_ns[0], g.extent):
        flash = g.sample(time)
        zero = g.sample(time, interaction_mode="dwell", dwell_seconds=0)
        np.testing.assert_allclose(flash.interaction_colors, zero.interaction_colors)
        np.testing.assert_array_equal(flash.leading, zero.leading)
        np.testing.assert_array_equal(flash.trail, zero.trail)


@pytest.mark.parametrize("duration", [-1, np.nan, np.inf, -np.inf, True])
def test_invalid_dwell_rejected(duration):
    with pytest.raises(ValueError):
        CinematicPulseGeometry(fixture(), [0]).sample(
            0, interaction_mode="dwell", dwell_seconds=duration
        )


def test_empty_metadata_retains_consistent_dtypes_and_shapes():
    g = CinematicPulseGeometry(_catalog([1]), [0])
    assert g.vertices.shape == (0, 3)
    for item in (
        g.interaction_path_ids,
        g.interaction_point_indices,
        g.interaction_types,
        g.interaction_orders,
        g.interaction_ordinals,
    ):
        assert item.shape == (0,)
        assert np.issubdtype(item.dtype, np.integer)
    assert g.sample(0, interaction_mode="dwell").interaction_colors.shape == (0, 4)
