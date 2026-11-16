"""Retire pyqtgraph view registrations before their Qt owners are deleted."""

import pyqtgraph as pg
from shiboken6 import isValid


class ManagedPlotWidget(pg.PlotWidget):
    """Preserve PlotWidget styling while retiring views on either Qt close path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        connect_plot_cleanup(self)

    def close(self):
        if self.plotItem is None:
            return
        # PlotWidget.close clears the scene before QWidget destruction. Retire
        # nested color-bar views while their graphics ownership is still valid.
        unregister_plot_views(self.plotItem)
        super().close()


def connect_plot_cleanup(plot: pg.PlotWidget) -> None:
    """Retire views when Qt deletes a plot without calling PlotWidget.close()."""
    # The owner's destroyed signal arrives before its graphics children die.
    # Capture the item rather than the partly destroyed QWidget wrapper.
    plot_item = plot.getPlotItem()
    plot.destroyed.connect(lambda: unregister_plot_views(plot_item))


def unregister_plot_views(plot: pg.PlotWidget | pg.PlotItem) -> None:
    """Unregister a plot's own views, including nested color-bar views.

    PlotItem.close() removes its main ViewBox but does not unregister nested
    ColorBarItem views. Python references can outlive their deleted Qt objects,
    leaving pyqtgraph's application-exit handler with stale signal wrappers.
    Follow graphics ownership so closing one plot leaves other plots alone.
    """
    # Explicit closure can delete the graphics before their QWidget owner.
    if not isValid(plot):
        return
    root = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
    pending = [root] if root is not None else []
    while pending:
        item = pending.pop()
        pending.extend(item.childItems())
        if isinstance(item, pg.ViewBox) and item in pg.ViewBox.AllViews:
            item.unregister()
