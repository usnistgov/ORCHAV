"""3D sweeps retain recorded cells, source identities and bounded display caches."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from shared.coverage.hdf5 import CoverageHDF5Reader
from tests.visualizer.unit.test_unified_coverage_panel import loaded
from tests.visualizer.unit.test_unified_coverage_panel import panel_view as panel_view


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize(
    "metric", ["rss_dbm/A", "path_gain_linear/A", "path_loss_db/A", "serving_tx"]
)
def test_3d_sweep_reveals_exact_layers_and_edges(panel_view, qtbot, monkeypatch, axis, metric):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    controller.handle_coverage_metric_changed(metric)
    w = loaded(qtbot, panel)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.cut_axis.setCurrentIndex(w.cut_axis.findData(axis))
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    raw, template = w.model.values, w.template.values
    selected, height = w.selected_indices, viz.coverage_height_index
    monkeypatch.setattr(CoverageHDF5Reader, "read_metric", Mock(side_effect=AssertionError("read")))
    w._play()
    assert w.sweep.isActive() and w.cut_enabled.isChecked()
    assert w.cut_position.currentIndex() == 0
    for index in range(len(w.model.axes[axis])):
        payload = viz.renderer.objects[w.OBJECT_ID].payload
        expected = template.copy()
        hidden = [slice(None)] * 3
        hidden[2 - axis] = slice(index + 1, None)
        expected[tuple(hidden)] = payload.limits[0]
        np.testing.assert_array_equal(payload.values, expected)
        assert payload.palette[0, 3] == 0
        assert payload.limits[0] < payload.threshold
        assert not np.any(payload.values[tuple(hidden)] > payload.threshold)
        for obj in w.edge_objects:
            assert obj.payload.points[:, axis].max() <= (
                w.model.axes[axis][index] + w.model.spacing[axis] / 2 + 1e-6
            )
        w.opacity.setValue(0.6 if index % 2 else 0.7)
        assert viz.renderer.objects[w.OBJECT_ID].payload.values is payload.values
        assert w.template.values is template and w.model.values is raw
        w._step()
    assert not w.sweep.isActive()
    assert w.selected_indices == selected and viz.coverage_height_index == height
    w.cut_enabled.setChecked(False)
    assert viz.renderer.objects[w.OBJECT_ID].payload.values is template


def test_3d_sweep_pause_loop_and_bookmark(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.cut_axis.setCurrentIndex(0)
    w.loop.setChecked(True)
    w.rate.setValue(4)
    w._play()
    assert w.sweep.interval() == 250
    for _ in range(w.cut_position.count()):
        w._step()
    assert w.cut_position.currentIndex() == 0 and w.sweep.isActive()
    w._step()
    w._play()
    assert not w.sweep.isActive() and w.cut_position.currentIndex() == 1
    path = tmp_path / "cut.json"
    w._save_view(path)
    values = viz.renderer.objects[w.OBJECT_ID].payload.values.copy()
    w.cut_axis.setCurrentIndex(2)
    w.cut_enabled.setChecked(False)
    w._load_view(path)
    assert w.cut_enabled.isChecked() and w.cut_axis.currentData() == 0
    assert w.cut_position.currentIndex() == 1 and not w.sweep.isActive()
    np.testing.assert_array_equal(viz.renderer.objects[w.OBJECT_ID].payload.values, values)
    old = json.loads(path.read_text())
    for name in ("cut_enabled", "cut_axis", "cut_coordinate"):
        old.pop(name)
    path.write_text(json.dumps(old))
    w._load_view(path)
    assert not w.cut_enabled.isChecked()
    w._play()
    w.mode.setCurrentIndex(w.mode.findData("mip"))
    assert not w.sweep.isActive() and w.sweep_controls.isHidden()


def test_region_controls_clamp_to_saved_grid(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    for axis, pair in zip(w.model.axes, w.region_bounds, strict=True):
        for spin in pair:
            assert spin.minimum() == pytest.approx(axis[0])
            assert spin.maximum() == pytest.approx(axis[-1])
            spin.setValue(axis[0] - 100)
            spin.stepDown()
            assert spin.value() == pytest.approx(axis[0])
            spin.setValue(axis[-1] + 100)
            spin.stepUp()
            assert spin.value() == pytest.approx(axis[-1])
