"""Inspector scrolling with explicit protection against accidental field edits."""

from PySide6.QtWidgets import QScrollArea, QWidget

from ..widgets.input_guard import protect_controls


class InspectorScrollArea(QScrollArea):
    """Resizable section container with focus-aware wheel editing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        protect_controls(self, require_wheel_focus=True)
