"""Pure geometry controls preserve source paths and geometric arrival ordering."""

import numpy as np
import pytest

from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry
from visualizer.src.experiments.path_analysis import SPEED_OF_LIGHT_M_PER_NS, geometric_progress


def geometry():
    catalog = _catalog(
        [100, 200],
        points=[[(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]] * 2,
    )
    catalog._canonical_data.itype[[1, 2, 5, 6]] = [1, 2, 1, 2]
    return CinematicPulseGeometry(catalog, [0, 1])


@pytest.mark.parametrize("mode,factor", [("short", 0.025), ("medium", 0.055), ("long", 0.1)])
def test_trail_modes_change_only_trailing_geometry(mode, factor):
    g = geometry()
    time = g.extent * 0.8
    sample = g.sample(time, trail_mode=mode)
    default = g.sample(time)
    np.testing.assert_array_equal(sample.leading, default.leading)
    np.testing.assert_array_equal(sample.interactions, default.interactions)
    np.testing.assert_array_equal(sample.interaction_colors, default.interaction_colors)
    for index, fraction in enumerate(np.linspace(0.1, 1, g.TRAIL_SAMPLES)):
        expected = geometric_progress(g.catalog, g.ids, time - g.extent * factor * fraction)
        np.testing.assert_allclose(
            sample.trail[index * 2 : index * 2 + 2], expected.positions, atol=1e-6
        )
    assert sample.trail.shape == (20, 3)


def test_default_trail_is_existing_medium_and_invalid_modes_fail():
    g = geometry()
    assert g.trail_ns == g.extent * 0.055
    np.testing.assert_array_equal(g.sample(1).trail, g.sample(1, trail_mode="medium").trail)
    with pytest.raises(ValueError):
        g.sample(1, trail_mode="unbounded")


def test_interaction_step_deduplicates_and_never_fabricates_boundaries():
    g = geometry()
    a, b = 3 / SPEED_OF_LIGHT_M_PER_NS, 6 / SPEED_OF_LIGHT_M_PER_NS
    assert g.interaction_time(0, 1) == a
    assert g.interaction_time(a, 1) == b
    assert g.interaction_time(b, 1) is None
    assert g.interaction_time(g.extent, -1) == b
    assert g.interaction_time(b, -1) == a
    assert g.interaction_time(a, -1) is None
    assert g.interaction_time(a + 0.001, -1) == a
    assert g.interaction_time(b - 0.001, 1) == b


def test_empty_unknown_interactions_have_no_steps():
    g = CinematicPulseGeometry(_catalog([1]), [0])
    assert g.interaction_time(0, 1) is None
    assert g.interaction_time(100, -1) is None


@pytest.mark.parametrize("time", [np.nan, np.inf, -np.inf])
def test_invalid_step_time(time):
    with pytest.raises(ValueError):
        geometry().interaction_time(time, 1)


@pytest.mark.parametrize("direction", [True, 0, 2, -2])
def test_invalid_step_direction(direction):
    with pytest.raises(ValueError):
        geometry().interaction_time(0, direction)
