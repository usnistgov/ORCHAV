"""Tests for the distinct material color palette generator."""

import numpy as np
import pytest

from visualizer.src.utils.colors import (
    clear_distinct_material_palette_cache,
    get_distinct_material_palette,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear palette cache before each test."""
    clear_distinct_material_palette_cache()
    yield
    clear_distinct_material_palette_cache()


def test_palette_returns_correct_count():
    for n in (1, 5, 10, 20):
        palette = get_distinct_material_palette(n)
        assert palette.shape == (n, 3), f"Expected ({n}, 3), got {palette.shape}"


def test_palette_colors_are_distinct():
    palette = get_distinct_material_palette(10)
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            assert not np.array_equal(palette[i], palette[j]), f"Colors {i} and {j} are identical"


def test_palette_values_in_range():
    palette = get_distinct_material_palette(20)
    assert palette.min() >= 0.0
    assert palette.max() <= 1.0


def test_palette_cached():
    p1 = get_distinct_material_palette(8)
    p2 = get_distinct_material_palette(8)
    assert p1 is p2


def test_palette_different_count_regenerates():
    p1 = get_distinct_material_palette(5)
    p2 = get_distinct_material_palette(10)
    assert p1 is not p2
    assert p1.shape == (5, 3)
    assert p2.shape == (10, 3)


def test_palette_large_count():
    palette = get_distinct_material_palette(50)
    assert palette.shape == (50, 3)
    assert palette.dtype == np.float32
    assert palette.min() >= 0.0
    assert palette.max() <= 1.0


def test_palette_medium_count():
    """Test the tab20+tab20b path (21-40 materials)."""
    palette = get_distinct_material_palette(30)
    assert palette.shape == (30, 3)
    assert palette.dtype == np.float32


def test_clear_cache():
    p1 = get_distinct_material_palette(5)
    clear_distinct_material_palette_cache()
    p2 = get_distinct_material_palette(5)
    assert p1 is not p2
    np.testing.assert_array_equal(p1, p2)
