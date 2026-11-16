"""Exact grid boundaries distinguish acquired outcomes from unavailable cells."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.sensing.volume_cell_edges import cell_edge_objects
from visualizer.src.utils import cell_edges


def axes():
    return {"x": np.array([10.0, 12, 14]), "y": np.array([20.0, 23]), "z": np.array([30.0, 34])}


def grid(available, **kwargs):
    return cell_edges.grid_edge_objects(axes(), available, enabled=True, **kwargs)


def test_default_off_and_hidden_do_not_inspect_or_allocate_source():
    assert cell_edge_objects(None, 0, enabled=False) == ([], ())
    assert cell_edge_objects(None, 0, enabled=True, visible=False) == ([], ())
    assert cell_edges.grid_edge_objects(None, None, enabled=False) == ([], ())


def test_sensing_status_distinguishes_no_reports_native_omission_and_no_acquisition():
    records = {
        (0, 0, 0, 0): dict(status="measured", tp=1, fp=0),
        (0, 0, 0, 1): dict(status="measured", tp=0, fp=0, native_evidence="not_included"),
        (0, 0, 0, 2): dict(status="pending"),
        (0, 0, 1, 0): dict(status="excluded_geometry"),
        (0, 0, 1, 1): dict(status="excluded_building"),
        (1, 0, 1, 2): dict(status="measured", tp=1, fp=0),
    }
    before = deepcopy(records)
    reader = SimpleNamespace(axes=axes(), cells=records)
    original_axes = {name: value.copy() for name, value in reader.axes.items()}
    objects, legend = cell_edge_objects(reader, 0, enabled=True, include_unavailable=True)
    by_name = {obj.id.rsplit(":", 1)[-1]: obj for obj in objects}
    assert set(by_name) == {"available", "unavailable"}
    # Adjacent cubes share four edges: 12 + 12 - 4, not 24 overlapping edges.
    assert len(by_name["available"].payload.lines) == 20
    np.testing.assert_allclose(by_name["available"].payload.points.min(axis=0), [9, 18.5, 28])
    np.testing.assert_allclose(by_name["available"].payload.points.max(axis=0), [13, 21.5, 32])
    assert "2 measured cells" in legend[0]
    assert any("10 excluded / unmeasured" in line for line in legend)
    assert any("no-report trials" in line and "no acquired outcome" in line for line in legend)
    assert records == before
    for name, values in reader.axes.items():
        np.testing.assert_array_equal(values, original_axes[name])
        assert values.flags.writeable
    for obj in objects:
        assert obj.metadata["depth_test"] and not obj.metadata["depth_write"]
        assert not obj.payload.points.flags.writeable


@pytest.mark.parametrize("normal,index", [("x", 1), ("y", 0), ("z", 1)])
def test_slice_rectangles_lie_on_exact_sample_centre_plane(normal, index):
    source = np.ones((2, 2, 3), bool)
    objects, _ = grid(source, display="slice", normal=normal, index=index)
    points = objects[0].payload.points
    n = "xyz".index(normal)
    np.testing.assert_array_equal(points[:, n], axes()[normal][index])
    dimensions = [axis for axis in range(3) if axis != n]
    lengths = [len(axes()["xyz"[axis]]) for axis in dimensions]
    expected_lines = lengths[0] * (lengths[1] + 1) + (lengths[0] + 1) * lengths[1]
    assert len(objects[0].payload.lines) == expected_lines
    for axis in dimensions:
        values = axes()["xyz"[axis]]
        step = values[1] - values[0]
        assert points[:, axis].min() == values[0] - step / 2
        assert points[:, axis].max() == values[-1] + step / 2
    assert source.all()


@pytest.mark.parametrize("axis,index,lower", [("x", 1, True), ("y", 1, False), ("z", 0, True)])
def test_cutaway_keeps_whole_cells_and_exact_half_cell_boundary(axis, index, lower):
    source = np.ones((2, 2, 3), bool)
    objects, _ = grid(source, cut=(axis, index, lower))
    points = objects[0].payload.points
    dimension = "xyz".index(axis)
    values = axes()[axis]
    step = values[1] - values[0]
    expected = values[index] + (step / 2 if lower else -step / 2)
    assert (points[:, dimension].max() if lower else points[:, dimension].min()) == expected
    assert source.all()
    # Switching to a slice ignores the previously selected 3D cutaway.
    sliced, _ = grid(source, display="slice", normal="z", index=1, cut=(axis, index, lower))
    reference, _ = grid(source, display="slice", normal="z", index=1)
    np.testing.assert_equal(sliced[0].payload.points, reference[0].payload.points)


def test_dashed_unknown_edges_preserve_bounds_and_do_not_invent_measured_support():
    source = np.zeros((2, 2, 3), bool)
    empty, legend = grid(source)
    assert empty == [] and "0 computed" in legend[0]
    objects, legend = grid(source, include_unavailable=True)
    assert len(objects) == 1 and objects[0].id.endswith(":unavailable")
    # Full regular-grid edge count: edges along each axis, deduplicated.
    edges = 3 * 3 * 3 + 4 * 2 * 3 + 4 * 3 * 2
    assert len(objects[0].payload.lines) == 4 * edges
    np.testing.assert_allclose(objects[0].payload.points.min(axis=0), [9, 18.5, 28])
    np.testing.assert_allclose(objects[0].payload.points.max(axis=0), [15, 24.5, 36])
    pairs = objects[0].payload.points[objects[0].payload.lines]
    # Every dash follows one physical coordinate direction; no diagonals.
    assert np.all(np.count_nonzero(pairs[:, 1] != pairs[:, 0], axis=1) == 1)
    assert "12 unavailable" in legend[2]


def test_full_coverage_grid_deduplicates_before_budget():
    shape = (17, 28, 32)
    grid_axes = {
        name: np.arange(n, dtype=float) for name, n in zip("xyz", shape[::-1], strict=True)
    }
    objects, _ = cell_edges.grid_edge_objects(grid_axes, np.ones(shape, bool), enabled=True)
    assert len(objects) == 1
    assert len(objects[0].payload.lines) == 32 * 29 * 18 + 33 * 28 * 18 + 33 * 29 * 17
    assert len(objects[0].payload.lines) < cell_edges.MAX_CELL_EDGE_SEGMENTS
    assert np.prod(shape) * 12 > cell_edges.MAX_CELL_EDGE_SEGMENTS


def test_budget_counts_dashes_and_refuses_instead_of_subsampling(monkeypatch):
    source = np.zeros((2, 2, 3), bool)
    objects, _ = grid(source, include_unavailable=True)
    exact_count = len(objects[0].payload.lines)
    monkeypatch.setattr(cell_edges, "MAX_CELL_EDGE_SEGMENTS", exact_count - 1)
    objects, legend = grid(source, include_unavailable=True)
    assert objects == []
    assert (
        "disabled" in legend[0] and "slice" in legend[1] and "no cells were subsampled" in legend[1]
    )
    monkeypatch.setattr(cell_edges, "MAX_CELL_EDGE_SEGMENTS", exact_count)
    objects, _ = grid(source, include_unavailable=True)
    assert len(objects[0].payload.lines) == exact_count
    assert not source.any()


def test_threshold_filters_and_render_modes_do_not_relabel_sampling_support():
    source = np.ones((2, 2, 3), bool)
    reference, _ = grid(source)
    for mode in ("hit", "miss", "mip", "iso"):
        objects, legend = grid(source, display=mode, prefix="coverage")
        assert objects[0].id == "coverage:available"
        np.testing.assert_array_equal(objects[0].payload.points, reference[0].payload.points)
        assert "independent" in legend[1]


def test_invalid_support_and_outside_slices_fail_clearly():
    with pytest.raises(ValueError, match="boolean Z/Y/X"):
        grid(np.ones((2, 2, 3), float))
    with pytest.raises(ValueError, match="boolean Z/Y/X"):
        grid(np.ones((2, 3, 2), bool))
    with pytest.raises(ValueError, match="outside"):
        grid(np.ones((2, 2, 3), bool), display="slice", normal="x", index=3)


def test_slice_edges_follow_volume_pass_without_bypassing_native_scene_depth():
    import pygfx

    from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin

    source = np.ones((2, 2, 3), bool)
    native = pygfx.Line(
        pygfx.Geometry(positions=np.zeros((2, 3), np.float32)), pygfx.LineSegmentMaterial()
    )
    renderer = SimpleNamespace(_objects={"analysis:cell_edges:available": native})
    volume = pygfx.VolumeSliceMaterial(alpha_mode="weighted_blend")
    # Reuse the same native object when returning from a slice to 3D: the
    # original queue/order must be restored, not retained from the slice.
    for display in ("both", "slice", "both", "slice"):
        objects, _ = grid(source, display=display)
        PygfxGeometryMixin._sync_render_object_metadata(renderer, objects[0])
        if display == "slice":
            assert native.material.alpha_mode == "blend"
            assert native.material.render_queue == volume.render_queue
            assert native.render_order == 10
        else:
            assert native.material.alpha_mode == "auto"
            assert native.material.render_queue < volume.render_queue
            assert native.render_order == 0
        assert native.material.depth_test
        assert not native.material.depth_write
        assert native.material.depth_compare == "<="


def test_absent_alpha_metadata_keeps_native_defaults_and_invalid_mode_is_ignored():
    from dataclasses import replace

    import pygfx

    from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin

    objects, _ = grid(np.ones((2, 2, 3), bool))
    native = pygfx.Line(
        pygfx.Geometry(positions=np.zeros((2, 3), np.float32)), pygfx.LineSegmentMaterial()
    )
    renderer = SimpleNamespace(_objects={objects[0].id: native})
    default = (native.material.alpha_mode, native.material.depth_write, native.render_order)
    PygfxGeometryMixin._sync_render_object_metadata(renderer, replace(objects[0], metadata={}))
    assert (native.material.alpha_mode, native.material.depth_write, native.render_order) == default
    PygfxGeometryMixin._sync_render_object_metadata(
        renderer, replace(objects[0], metadata={"alpha_mode": "unknown"})
    )
    assert (native.material.alpha_mode, native.material.depth_write, native.render_order) == default


def test_selected_cell_edges_preserve_availability_and_exact_boundaries():
    available = np.ones((2, 2, 3), bool)
    selected = np.zeros_like(available)
    selected[0, 0, 0] = True
    before = available.copy()
    objects, legend = grid(available, selected=selected)
    assert len(objects) == 1 and len(objects[0].payload.lines) == 12
    np.testing.assert_allclose(objects[0].payload.points.min(axis=0), [9, 18.5, 28])
    np.testing.assert_allclose(objects[0].payload.points.max(axis=0), [11, 21.5, 32])
    np.testing.assert_array_equal(available, before)
    assert any("follow the selected display region" in line for line in legend)


@pytest.mark.parametrize("mask", [np.ones((2, 2, 3)), np.ones((2, 3, 2), bool)])
def test_invalid_cell_selection_rejected(mask):
    with pytest.raises(ValueError, match="boolean mask matching"):
        grid(np.ones((2, 2, 3), bool), selected=mask)
