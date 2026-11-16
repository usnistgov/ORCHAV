"""Revision-aware bounds reuse must preserve malformed/mutable input handling."""

import gc
import weakref

import numpy as np

from visualizer.src.renderers.pygfx.mpc import PygfxMpcMixin


def test_invalid_indices_preserve_matching_colors_and_point_count_invalidation():
    r = PygfxMpcMixin()
    points = np.zeros((4, 3), np.float32)
    lines = np.array([[0, 3], [1, 2]], np.int32)
    lines.setflags(write=False)
    colors = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    normalize = r._normalize_mpc_lines_input
    assert normalize(points, lines, colors, revision=(1,))[1] is lines
    assert normalize(points, lines, colors, revision=(1,))[1] is lines
    _, filtered, shades = normalize(points[:3], lines, colors, revision=(1,))
    np.testing.assert_array_equal(filtered, [[1, 2]])
    np.testing.assert_array_equal(shades, [[0, 1, 0]])
    # A mutable caller is not allowed to inherit the earlier immutable success.
    lines.setflags(write=True)
    lines[0] = (-1, 3)
    _, filtered, shades = normalize(points, lines, colors, revision=(1,))
    np.testing.assert_array_equal(filtered, [[1, 2]])
    np.testing.assert_array_equal(shades, [[0, 1, 0]])


def test_unversioned_readonly_views_are_revalidated():
    r = PygfxMpcMixin()
    owner = np.array([[0, 1]], np.int32)
    view = owner.view()
    view.setflags(write=False)
    points = np.zeros((2, 3), np.float32)
    assert len(r._normalize_mpc_lines_input(points, view, None)[1]) == 1
    owner[0, 1] = 9
    assert len(r._normalize_mpc_lines_input(points, view, None)[1]) == 0
    assert not getattr(r, "_mpc_valid_index_cache", {})


def test_validation_cache_is_bounded_weak_and_checks_array_identity(monkeypatch):
    import visualizer.src.renderers.pygfx.mpc as mpc

    monkeypatch.setattr(mpc, "_MPC_VALID_INDEX_CACHE_LIMIT", 2)
    r = PygfxMpcMixin()
    refs = []
    for i in range(4):
        lines = np.array([[0, 1]], np.int32)
        lines.setflags(write=False)
        refs.append(weakref.ref(lines))
        assert r._valid_mpc_line_indices(lines, 2, revision=(i,))
    assert len(r._mpc_valid_index_cache) == 2
    del lines
    gc.collect()
    assert all(ref() is None for ref in refs)
    changed = np.array([[0, 7]], np.int32)
    changed.setflags(write=False)
    assert not r._valid_mpc_line_indices(changed, 2, revision=(3,))


def test_input_shapes_and_empty_sets_still_follow_normalization_contract():
    r = PygfxMpcMixin()
    points = np.zeros((3, 3), np.float32)
    assert r._normalize_mpc_lines_input(points, np.zeros((2, 3)), None) is None
    assert r._normalize_mpc_lines_input(np.zeros(3), [[0, 1]], None) is None
    assert r._normalize_mpc_lines_input(points, [], None)[1].shape == (0, 2)
    flat = r._normalize_mpc_lines_input(points, [0, 1, 1, 2], None)[1]
    np.testing.assert_array_equal(flat, [[0, 1], [1, 2]])
