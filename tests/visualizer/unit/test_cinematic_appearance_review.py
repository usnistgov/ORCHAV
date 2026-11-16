"""Independent parity checks against existing full-frame MPC appearance."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_interaction_metadata import fixture
from visualizer.src.experiments.cinematic_appearance import (
    interaction_colors,
    interaction_label_specs,
)
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry
from visualizer.src.metrics.mpc_canon import colorize


def core():
    rng = np.random.default_rng(42)
    return SimpleNamespace(
        _order_palette=rng.random((10, 3)).astype(np.float32),
        _type_palette=rng.random((100, 3)).astype(np.float32),
        _viridis256=rng.random((256, 3)).astype(np.float32),
        _resolve_material_colors=lambda distinct: {"stone": np.array([0.8, 0.4, 0.2])},
    )


@pytest.mark.parametrize(
    "mode",
    ["delay", "path_loss", "reflection_order", "mpc_type", "reconstruction_type", "material"],
)
def test_subset_colors_match_full_canonical_colors_with_ranked_repeated_indices(mode):
    catalog = fixture()
    canon = catalog.canonical_data
    canon.delay[:] = np.repeat([1, 50, 100], 6)
    canon.loss[:] = np.repeat([30, 70, 120], 6)
    canon.delay_min, canon.delay_max = 1, 100
    canon.loss_min, canon.loss_max = 30, 120
    canon.material_names = np.array(["stone", "stone", "absent", "stone", "", ""] * 3)
    p = core()
    # Both nonmonotonic strongest-path order and duplicate vertices are valid mappings.
    indices = np.array([14, 13, 1, 15, 14])
    before = canon.points.copy()
    expected = colorize(
        canon,
        np.ones(18, bool),
        mode,
        p._order_palette,
        p._type_palette,
        p._viridis256,
        material_colors=p._resolve_material_colors(False),
    )[indices]
    actual, note = interaction_colors(catalog, indices, p, mode)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(canon.points, before)
    assert "existing MPC" in note


def test_fallback_and_empty_colors_are_explicit():
    catalog = fixture()
    actual, note = interaction_colors(catalog, [1], None, "delay")
    np.testing.assert_allclose(actual, [[0.7, 0.7, 0.7]])
    assert "unavailable" in note
    assert interaction_colors(catalog, [], None, "delay")[0].shape == (0, 3)


def test_labels_disambiguate_path_vertex_and_unknown_virtual_types():
    geometry = CinematicPulseGeometry(fixture(), [2, 0])
    assert interaction_label_specs(geometry, enabled=False, show_types=True) == ()
    labels = interaction_label_specs(geometry, enabled=True, show_types=True)
    assert labels[:3] == (
        (3, "P0 · I1 · Specular"),
        (4, "P0 · I2 · Virtual"),
        (5, "P0 · I3 · Unknown"),
    )
    assert all("reflection" not in label.lower() for _, label in labels)
    plain = interaction_label_specs(geometry, enabled=True, show_types=False)
    assert plain[0] == (3, "P0 · I1")


def test_label_budget_is_deterministic():
    g = SimpleNamespace(
        vertices=np.zeros((100, 3)),
        interaction_path_ids=np.repeat([9, 2], 50),
        interaction_ordinals=np.tile(np.arange(50), 2),
        interaction_types=np.ones(100, int),
    )
    labels = interaction_label_specs(g, enabled=True, show_types=False)
    assert len(labels) == 32
    assert labels[0] == (50, "P2 · I0")
    assert labels[-1] == (81, "P2 · I31")
