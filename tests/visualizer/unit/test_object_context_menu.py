"""Context cancellation must not dispatch a nonexistent menu action."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QPoint

from visualizer.src.panels.object_panel import ObjectManagementPanel


@pytest.mark.parametrize("kind", ["mesh", "group", "tx", "rx", "target"])
@pytest.mark.parametrize("selection", [None, "Focus Camera", "Node Properties…"])
def test_context_menu_dispatch_requires_an_existing_action(qapp, monkeypatch, kind, selection):
    panel = ObjectManagementPanel(SimpleNamespace())
    entry = {"entry_type": kind}
    panel.widgets["object_tree"] = SimpleNamespace(
        indexAt=lambda _pos: None,
        viewport=lambda: SimpleNamespace(mapToGlobal=lambda pos: pos),
    )
    panel._entry_from_index = lambda _index: entry
    panel._is_online_mode = lambda: False
    panel._show_node_properties_dialog = Mock()
    panel._focus_camera_on_entry = Mock()
    actions = {}

    def add_action(name):
        action = Mock()
        actions[name] = action
        return action

    menu = SimpleNamespace(
        addAction=add_action,
        addSeparator=lambda: None,
        exec=lambda _pos: actions.get(selection),
    )
    monkeypatch.setattr("visualizer.src.panels.object_panel.QMenu", lambda _tree: menu)
    panel._on_tree_context_menu(QPoint())
    assert panel._show_node_properties_dialog.call_count == int(
        selection == "Node Properties…" and kind in {"tx", "rx", "target"}
    )
    assert panel._focus_camera_on_entry.call_count == int(selection == "Focus Camera")
