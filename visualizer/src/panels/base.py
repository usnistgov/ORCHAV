"""Shared base class for visualizer panels.

Panels are lightweight widget factories: they keep a reference to the parent
visualizer/panel manager and register child widgets in ``self.widgets`` so
``UIPanelManager`` can wire signals and mirror compatibility attributes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtWidgets import QGroupBox


class BasePanel:
    """Provide common parent storage, widget registry, and group-box helpers."""

    def __init__(self, parent_widget: Any) -> None:
        """Store the parent owner and initialize the panel widget registry."""
        self.parent = parent_widget
        self.widgets: Dict[str, Any] = {}

    def create_group_box(self, title: str, *, style: Optional[str] = None) -> QGroupBox:
        """Create a group box using the application stylesheet by default.

        A ``style`` override is available for the few panels that need local Qt
        styling, but most panel appearance should remain centralized in the app
        theme.
        """
        group = QGroupBox(title)
        if style is not None:
            group.setStyleSheet(style)
        return group

    def create_subgroup_box(self, title: str) -> QGroupBox:
        """Create a nested group box with the same styling contract."""
        return self.create_group_box(title)
