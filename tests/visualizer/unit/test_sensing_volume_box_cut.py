"""Corner and interior cuts preserve source values, grid support and view state."""

import json

import numpy as np
import pytest

from visualizer.src.sensing.volume_cutaway import VolumeCutawayControls, cut_volume
from visualizer.src.types.render_payloads import VolumePayload
from visualizer.src.utils.cell_edges import grid_edge_objects
from visualizer.src.utils.volume_cut import BoxCut


def axes():
    return {axis: origin + 3 + 6 * np.arange(16) for axis, origin in zip("xyz", (-63, -58, 27))}


def test_poster_corner_exact_dimensions_source_immutable_and_state_roundtrip(qtbot):
    controls = VolumeCutawayControls()
    qtbot.addWidget(controls)
    controls.configure(axes())
    controls.mode.setCurrentIndex(controls.mode.findData("box"))
    controls.active.setChecked(True)
    cut = controls.cut("both")
    assert cut == BoxCut(((0, 8), (0, 8), (8, 16)))
    state = json.loads(json.dumps(controls.state()))
    assert state["box_bounds"] == {"x": [-63, -15], "y": [-58, -10], "z": [75, 123]}
    assert "cell boundaries" in controls.description("both")
    original = VolumePayload(
        values=np.ones((16, 16, 16)),
        palette=np.array([[0, 0, 0, 0], [1, 1, 1, 1]]),
        limits=(0, 1),
    )
    cropped = cut_volume(original, cut)
    assert np.count_nonzero(cropped.values) == 4096 - 512
    assert np.all(cropped.values[8:, :8, :8] == 0)
    assert original.values.all() and cropped.palette is original.palette
    assert controls.cut("slice") is None
    assert cut_volume(original, controls.cut("slice")) is original
    restored = VolumeCutawayControls()
    qtbot.addWidget(restored)
    restored.configure(axes(), state)
    assert restored.cut("mip") == cut
    assert restored.state() == state
    controls.active.setChecked(False)
    assert controls.cut("both") is None


def test_legacy_plane_state_and_crossed_box_bounds(qtbot):
    controls = VolumeCutawayControls()
    qtbot.addWidget(controls)
    controls.configure(axes(), dict(active=True, axis="x", coordinate=-60, keep_lower=True))
    assert controls.cut("both") == ("x", 0, True)
    controls.mode.setCurrentIndex(controls.mode.findData("box"))
    for lower, upper in controls.box_bounds.values():
        lower.setCurrentIndex(lower.count() - 1)
        assert lower.currentIndex() < upper.currentIndex()
        upper.setCurrentIndex(0)
        assert lower.currentIndex() < upper.currentIndex()
    assert controls.cut("both") == BoxCut(((0, 1), (0, 1), (0, 1)))


@pytest.mark.parametrize("unavailable", [False, True])
def test_box_removes_edge_support_including_unknown_cells_and_keeps_cut_faces(unavailable):
    grid_axes = {axis: np.arange(3, dtype=float) * 6 + 3 for axis in "xyz"}
    available = np.ones((3, 3, 3), dtype=bool)
    available[2, 0, 0] = False
    selected = np.ones_like(available)
    cut = BoxCut(((0, 1), (0, 1), (2, 3)))
    objects, legend = grid_edge_objects(
        grid_axes,
        available,
        enabled=True,
        include_unavailable=unavailable,
        cut=cut,
        selected=selected,
    )
    assert "26 computed" in legend[0]
    assert len(objects) == 1  # The only unknown cell was removed, not relabelled.
    points = objects[0].payload.points
    assert not np.any(np.all(points == [0, 0, 18], axis=1))
    assert np.any(np.all(points == [0, 0, 12], axis=1))  # exposed horizontal face
    assert selected.all() and np.count_nonzero(available) == 26
    sliced, _ = grid_edge_objects(grid_axes, available, enabled=True, display="slice", cut=cut)
    reference, _ = grid_edge_objects(grid_axes, available, enabled=True, display="slice")
    np.testing.assert_array_equal(sliced[0].payload.points, reference[0].payload.points)


def test_interior_box_is_intersection_of_bounds_and_validated():
    data = np.ones((5, 4, 3))
    payload = VolumePayload(values=data, palette=np.ones((2, 4)), limits=(0, 1))
    result = cut_volume(payload, BoxCut(((1, 2), (1, 3), (1, 4))))
    assert np.count_nonzero(result.values) == 60 - 6
    assert data.all()
    for bounds in (((0, 4), (0, 1), (0, 1)), ((1, 1), (0, 1), (0, 1))):
        with pytest.raises(ValueError, match="outside"):
            cut_volume(payload, BoxCut(bounds))
