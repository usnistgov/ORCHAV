"""Plot retirement clears only owned pyqtgraph view registrations."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from shiboken6 import isValid

from visualizer.src.widgets.dashboard_plot import DashboardPlot
from visualizer.src.widgets.inline_plot import InlinePlot
from visualizer.src.widgets.plot_lifecycle import ManagedPlotWidget, unregister_plot_views


@pytest.mark.parametrize("kind", ["sensing", "fusion", "metrics"])
def test_dashboard_close_unregisters_nested_views_without_touching_other_plots(
    qapp, monkeypatch, tmp_path, kind
):
    from visualizer.src.metrics.viz_metrics import MetricsDashboard
    from visualizer.src.sensing import fusion_window
    from visualizer.src.sensing.sensing_window import SensingDashboard

    monkeypatch.setattr(
        fusion_window,
        "FusionInspectionSession",
        lambda: SimpleNamespace(open=lambda _path: Future(), close=lambda: None),
    )
    other = pg.PlotWidget()
    other_view = other.getViewBox()
    baseline = set(pg.ViewBox.AllViews)
    try:
        for _ in range(2):
            if kind == "fusion":
                dashboard = fusion_window.FusionWindow(SimpleNamespace(renderer=None), tmp_path)
            else:
                dashboard = SensingDashboard() if kind == "sensing" else MetricsDashboard()
            # Retain Python wrappers deliberately: registry cleanup must not
            # depend on cyclic collection after their Qt owners are destroyed.
            owned = set(pg.ViewBox.AllViews) - baseline
            assert owned
            assert any(isinstance(view.parentItem(), pg.ColorBarItem) for view in owned)
            dashboard.close()
            dashboard.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
            assert not owned.intersection(pg.ViewBox.AllViews)
            assert other_view in pg.ViewBox.AllViews
    finally:
        unregister_plot_views(other)
        other.close()


def test_clearing_inline_heatmaps_retires_only_the_removed_colorbar(qapp):
    plot = InlinePlot()
    view = plot.getViewBox()
    try:
        for _ in range(3):
            plot.heatmap([0, 1], [0, 1], np.ones((2, 2)), limits=(0, 1))
            colorbar_view = plot._colorbars[0].getViewBox()
            assert colorbar_view in pg.ViewBox.AllViews
            plot.clear()
            assert colorbar_view not in pg.ViewBox.AllViews
            assert view in pg.ViewBox.AllViews
        unregister_plot_views(plot)
        unregister_plot_views(plot)  # Retirement is safe before repeated closes.
        assert view not in pg.ViewBox.AllViews
    finally:
        unregister_plot_views(plot)
        plot.close()


@pytest.mark.parametrize("plot_type", [ManagedPlotWidget, DashboardPlot, InlinePlot])
@pytest.mark.parametrize("retirement", ["close", "delete", "parent_delete", "close_parent_delete"])
def test_plot_retires_nested_views_when_closed_or_deleted_by_qt(qapp, plot_type, retirement):
    other = InlinePlot()
    other_view = other.getViewBox()
    parent = QWidget()
    plot = plot_type()
    QVBoxLayout(parent).addWidget(plot)
    image = pg.ImageItem(np.ones((2, 2)))
    plot.addItem(image)
    colorbar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"))
    colorbar.setImageItem(image, insert_in=plot.getPlotItem())
    owned = {plot.getViewBox(), colorbar.getViewBox()}
    try:
        if retirement in {"close", "close_parent_delete"}:
            plot.close()
            plot.close()
            assert not owned.intersection(pg.ViewBox.AllViews)
            if retirement == "close_parent_delete":
                # PlotWidget.close detaches; controllers restore Qt ownership
                # before their containing dashboard is deleted.
                plot.setParent(parent)
                parent.deleteLater()
            else:
                plot.deleteLater()
        elif retirement == "delete":
            plot.deleteLater()
        else:
            parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert not isValid(plot)
        assert not owned.intersection(pg.ViewBox.AllViews)
        assert other_view in pg.ViewBox.AllViews
    finally:
        other.close()
        if isValid(parent):
            parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("plot_type", [ManagedPlotWidget, DashboardPlot, InlinePlot])
def test_reparenting_a_live_plot_keeps_its_views_registered(qapp, plot_type):
    parent = QWidget()
    destination = QWidget()
    plot = plot_type()
    QVBoxLayout(parent).addWidget(plot)
    view = plot.getViewBox()
    plot.hide()
    QVBoxLayout(destination).addWidget(plot)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert isValid(plot) and view in pg.ViewBox.AllViews
    plot.plot([0, 1], [1, 2])
    destination.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert view not in pg.ViewBox.AllViews


@pytest.mark.parametrize("kind", ["statistics", "sensing", "processing", "experiments", "slices"])
@pytest.mark.parametrize("retirement", ["parent_delete", "close_parent_delete"])
def test_plot_owners_retire_all_views_without_relying_on_close_event(qapp, kind, retirement):
    """Qt parent deletion and explicit release cover the actual plot owners."""
    from visualizer.src.experiments.window import ExperimentWindow
    from visualizer.src.panels.statistics_plot import StatisticsPlotCard
    from visualizer.src.sensing.processing_panel import ProcessingPanel
    from visualizer.src.sensing.sensing_window import SensingDashboard
    from visualizer.src.sensing.volume_inspection_controls import LinkedSlice

    parent = QMainWindow() if kind == "experiments" else QWidget()
    other = ManagedPlotWidget()
    other_view = other.getViewBox()
    baseline = set(pg.ViewBox.AllViews)
    if kind == "statistics":
        owner = StatisticsPlotCard("delay_time_chart", "Saved delays", str, lambda *_: 0)
        owner.set_chart(
            "delay_time",
            {
                "power_db": np.array([[-10.0, -20.0], [-5.0, -15.0]]),
                "x_edges": [0.0, 1.0, 2.0],
                "delay_edges": [0.0, 1.0, 2.0],
                "xlabel": "Time (s)",
                "coordinates": [0.5, 1.5],
                "frame_indices": [0, 1],
            },
            "Saved delay/time graph",
        )
    elif kind == "sensing":
        owner = SensingDashboard()
    elif kind == "processing":
        owner = ProcessingPanel()
    elif kind == "experiments":
        owner = ExperimentWindow(parent)
    else:
        owner = LinkedSlice()
        owner._ensure_plot()
    if kind != "experiments":
        QVBoxLayout(parent).addWidget(owner)
    if kind == "statistics":
        parent.show()
        qapp.processEvents()
        assert owner._colorbar is not None
    owned = set(pg.ViewBox.AllViews) - baseline
    assert owned
    try:
        if retirement == "close_parent_delete":
            if kind == "processing":
                owner.release()
                owner.release()
                assert not owned.intersection(pg.ViewBox.AllViews)
            else:
                owner.close()
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert not isValid(owner)
        assert not owned.intersection(pg.ViewBox.AllViews)
        assert other_view in pg.ViewBox.AllViews
    finally:
        other.close()
        other.deleteLater()
        if isValid(parent):
            parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_dashboard_parent_deletion_preserves_a_reparented_live_figure(qapp):
    from visualizer.src.sensing.sensing_window import SensingDashboard

    parent, destination = QWidget(), QWidget()
    baseline = set(pg.ViewBox.AllViews)
    dashboard = SensingDashboard()
    QVBoxLayout(parent).addWidget(dashboard)
    parent.show()
    qapp.processEvents()
    assert dashboard.angular_spectrum_plot.isHidden()
    plot = dashboard.rd_plot
    borrowed = {plot.getViewBox(), dashboard.rd_colorbar.getViewBox()}
    owned = set(pg.ViewBox.AllViews) - baseline
    # A live-figure dock borrows the existing widget by reparenting it.
    QVBoxLayout(destination).addWidget(plot)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    try:
        assert not isValid(dashboard)
        assert isValid(plot)
        assert owned.intersection(pg.ViewBox.AllViews) == borrowed
        curve = plot.plot([0, 1], [1, 2])
        np.testing.assert_array_equal(curve.yData, [1, 2])
    finally:
        destination.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not owned.intersection(pg.ViewBox.AllViews)


@pytest.mark.parametrize("kind", ["volume", "volume_with_slices", "volume_comparison"])
def test_volume_close_retires_embedded_plots(qapp, tmp_path, kind):
    from tests.shared.test_sensing_volume_inspection import write_fixture
    from visualizer.src.sensing.volume_comparison import VolumeComparisonDialog
    from visualizer.src.sensing.volume_window import SensingVolumeWindow

    reader = write_fixture(tmp_path / "volume")
    parent = QWidget()
    parent.renderer = SimpleNamespace(
        remove_object=lambda _id: None, set_analysis_legend=lambda _legend: None
    )
    other = InlinePlot()
    baseline = set(pg.ViewBox.AllViews)
    try:
        for _ in range(2):
            window = (
                VolumeComparisonDialog(parent, initial_volume=reader.path)
                if kind == "volume_comparison"
                else SensingVolumeWindow(parent, reader.path)
            )
            if kind == "volume_with_slices":
                for view in window.inspection_controls.slice_view.views.values():
                    view._ensure_plot()
            owned = set(pg.ViewBox.AllViews) - baseline
            assert owned
            window.close()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
            assert not isValid(window)
            assert not owned.intersection(pg.ViewBox.AllViews)
            assert other.getViewBox() in pg.ViewBox.AllViews
    finally:
        other.close()
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
