"""Nested plot scenes are destroyed with the dashboard, not orphaned by close()."""

import gc
import weakref

import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, Qt

from visualizer.src.sensing.sensing_window import SensingWindow


def test_all_nested_plots_follow_actual_dashboard_deletion(qtbot):
    references = []
    for _ in range(4):
        window = SensingWindow()
        window.setAttribute(Qt.WA_DeleteOnClose)
        plots = window.findChildren(pg.PlotWidget)
        assert window.processing_panel.before_plot in plots
        assert window.evaluation_panel.plots.matches in plots
        references.extend(weakref.ref(plot) for plot in plots)
        window.close()
        assert all(plot.plotItem is None for plot in plots)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not shiboken6.isValid(window)
        assert all(not shiboken6.isValid(plot) for plot in plots)
        del plots, window
        gc.collect()
    assert all(ref() is None or not shiboken6.isValid(ref()) for ref in references)
