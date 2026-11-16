"""Hidden object-tree layout must not reset cells from accessibility reads."""

from types import SimpleNamespace

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QAccessible
from shiboken6 import isValid

from visualizer.src.panels.object_panel import ObjectManagementPanel


def test_hidden_scene_tree_settles_before_accessibility_cell_state(qapp, monkeypatch):
    class RecordingTree(ObjectManagementPanel._ObjectTreeView):
        def __init__(self, *args, **kwargs):
            self.layout_calls = 0
            super().__init__(*args, **kwargs)

        def doItemsLayout(self):
            self.layout_calls += 1
            super().doItemsLayout()

    monkeypatch.setattr(ObjectManagementPanel, "_ObjectTreeView", RecordingTree)
    panel = ObjectManagementPanel(SimpleNamespace(xml_root=object()))
    widget = panel.create_panel()
    tree = panel.widgets["object_tree"]
    assert not tree.isVisible()  # The Objects tab need not be open during reload.
    try:
        for count in (1, 2, 1, 3):
            panel.populate_object_list(
                [{"name": f"wall{i}", "material_type": f"metal{i}"} for i in range(count)],
                [],
            )
            # Model/header work can queue another hidden-view layout after the
            # synchronous population pass, as in the native scene transition.
            tree.scheduleDelayedItemsLayout()
            panel._apply_object_column_widths()
            for _ in range(3):
                qapp.processEvents()
            accessible = QAccessible.queryAccessibleInterface(tree)
            before = tree.layout_calls
            for i in range(accessible.childCount()):
                cell = accessible.child(i)
                if cell is not None and cell.isValid():
                    cell.state()
            # A layout from state()->visualRect() deletes the very interface
            # AT-SPI is reading on Qt6.8. Observe that cause without crashing
            # the test process or requiring a system accessibility service.
            assert tree.layout_calls == before
    finally:
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_layout_settler_is_coalesced_and_owned_by_tree(qapp):
    tree = ObjectManagementPanel._ObjectTreeView(lambda: None)
    timer = tree._layout_settler
    tree.settle_layout()
    timer_id = timer.timerId()
    tree.settle_layout()
    assert timer.isActive()
    assert timer.parent() is tree
    assert timer.isSingleShot()
    assert timer_id >= 0
    tree.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(timer)
    qapp.processEvents()  # No callback against a deleted view.
