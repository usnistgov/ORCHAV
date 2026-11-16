"""Cluster colors represent saved categories, not a continuous strength scale."""

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QRectF

from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit.test_sensing_stage_dashboard import _replay_event_for, _request
from visualizer.src.sensing.detector_map import DetectorMap
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle


def test_sparse_cluster_ids_have_compact_discrete_color_bands(qtbot):
    values = np.array([[np.nan, 71, 76], [104, 71, np.nan]])
    saved = values.copy()
    plot = DetectorMap("Cluster labels", pg.colormap.get("viridis"), categorical=True)
    qtbot.addWidget(plot)
    assert plot.set_map(values, "Cluster ID", (0, 104), QRectF(0, 0, 3, 2), ("first",))
    assert plot.cluster_ids == (71, 76, 104)
    np.testing.assert_allclose(plot.image.image, [[np.nan, 2], [0, 0], [1, np.nan]], equal_nan=True)
    np.testing.assert_allclose(values, saved, equal_nan=True)
    assert plot.image.getLevels() == pytest.approx((-0.5, 2.5))
    assert plot.colorbar.axis._tickLevels == [[(0, "71"), (1, "76"), (2, "104")]]
    lut = plot.colorbar.colorMap().getLookupTable(nPts=256, alpha=True)
    assert len(np.unique(lut, axis=0)) == 3
    assert plot.colorbar.colorMapMenu is False

    image = plot.image.image
    assert not plot.set_map(values, "Cluster ID", (0, 104), QRectF(0, 0, 3, 2), ("first",))
    plot.select_cell(1, 1)
    assert plot.image.image is image
    plot.select_cell()
    assert plot.image.image is image


def test_ten_visible_clusters_have_ten_distinct_colors(qtbot):
    plot = DetectorMap("Cluster labels", pg.colormap.get("viridis"), categorical=True)
    qtbot.addWidget(plot)
    values = np.array([[71, 76, 78, 83, 86, 90, 94, 96, 101, 104]])
    plot.set_map(values, "Cluster ID", (0, 104), QRectF(0, 0, 10, 1), (1,))
    lut = plot.colorbar.colorMap().getLookupTable(nPts=256, alpha=True)
    assert len(np.unique(lut, axis=0)) == 10
    assert len(plot.colorbar.axis._tickLevels[0]) == 10
    plot.set_map(np.array([[76]]), "Cluster ID", (0, 76), QRectF(0, 0, 1, 1), (2,))
    assert plot.cluster_ids == (76,)
    assert plot.image.getLevels() == pytest.approx((-0.5, 0.5))
    assert len(np.unique(plot.colorbar.colorMap().getLookupTable(), axis=0)) == 1


@pytest.mark.parametrize("background", [np.nan, -1])
def test_empty_and_unrecorded_clusters_do_not_show_a_fake_scale(qtbot, background):
    plot = DetectorMap("Cluster labels", pg.colormap.get("viridis"), categorical=True)
    qtbot.addWidget(plot)
    plot.set_map(np.full((2, 3), background), "Cluster ID", (0, 1), QRectF(0, 0, 3, 2), (1,))
    assert plot.cluster_ids == ()
    assert not plot.colorbar.isVisible()
    assert np.isnan(plot.image.image).all()
    plot.set_map(None, "Not recorded", (0, 1), None, (2,))
    assert plot.cluster_ids == ()
    assert not plot.colorbar.isVisible()
    assert plot.image.image is None


def test_dashboard_keeps_saved_ids_while_displaying_category_indices(qtbot):
    sensing = _typed_sensing()
    bundle = build_detection_stage_bundle(sensing)
    product = bundle.product("range_doppler_cluster_labels")
    labels = np.full_like(product.values, -1)
    labels[0, 1, 1:3] = (71, 104)
    bundle = replace(
        bundle,
        products=tuple(
            replace(p, values=labels) if p.kind == product.kind else p for p in bundle.products
        ),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._open_result_detail("diagnostics", "cfar")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    plot = window.stage_cluster_plot
    assert plot.cluster_ids == (71, 104)
    assert "2 clusters" in plot.getPlotItem().titleLabel.text
    assert plot.image.image[1, 1] == 0
    assert plot.image.image[2, 1] == 1
    window._detection_stage_kind = product.kind
    window._select_inspection_coordinate(2, 1)
    assert "value=104 1" in window.detection_stage_text.toPlainText()
    assert bundle.product(product.kind).values[0, 1, 2] == 104


def test_power_map_keeps_its_continuous_scale(qtbot):
    plot = DetectorMap("Power", pg.colormap.get("viridis"))
    qtbot.addWidget(plot)
    values = np.array([[0.0, 10.0, 100.0]])
    plot.set_map(values, "Power (dB)", (0, 100), QRectF(0, 0, 3, 1), (1,))
    np.testing.assert_array_equal(plot.image.image, values.T)
    assert plot.image.getLevels() == pytest.approx((0, 100))
    assert len(np.unique(plot.colorbar.colorMap().getLookupTable(), axis=0)) > 100
