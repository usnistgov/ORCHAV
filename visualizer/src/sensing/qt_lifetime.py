"""Keep cyclic Qt object cleanup on the application thread during background I/O."""

from __future__ import annotations

import gc

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication


def ensure_gui_garbage_collection() -> None:
    """Install pyqtgraph's GUI-thread collector once for the application lifetime.

    Plot menus and graphics scenes contain Python reference cycles. A storage
    worker's allocation can otherwise trigger collection of a closed dashboard
    and destroy Qt objects from that worker. Explicit plot closure still frees
    resources promptly; the timer handles remaining cycles at Python's normal
    generation thresholds. Respect an embedding application's disabled GC.
    """
    app = QApplication.instance()
    if app is None or hasattr(app, "_orchav_plot_collector") or not gc.isenabled():
        return
    if QThread.currentThread() != app.thread():
        raise RuntimeError("The plot garbage collector must be installed on Qt's thread")

    from pyqtgraph.util.garbage_collector import GarbageCollector

    collector = GarbageCollector(interval=1.0)
    collector.timer.setParent(app)
    # Keep this shared owner beyond an individual dashboard/session. Readers
    # can finish after a window closes, and other application plots have cycles.
    app._orchav_plot_collector = collector
