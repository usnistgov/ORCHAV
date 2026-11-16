"""Collapsible section widget used by tabbed visualizer panels.

``UIPanelManager`` wraps most panel groups in these sections so tab pages can
present dense controls without overwhelming the user. The widget emits a simple
boolean signal and leaves persistence/layout policy to its caller.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class _SectionBody(QWidget):
    """Reserve wrapped content height even with a fixed vertical size policy."""

    def sizeHint(self):
        hint = super().sizeHint()
        layout = self.layout()
        if layout is not None and layout.hasHeightForWidth():
            hint.setHeight(max(hint.height(), layout.totalHeightForWidth(self.width())))
        return hint

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()


class CollapsibleSection(QWidget):
    """Section container with a text-only toggle header and body layout."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        parent=None,
        start_open: bool = True,
        *,
        expand_content: bool = False,
        flat_header: bool = False,
    ):
        """Create the toggle header and body container for one panel section."""
        super().__init__(parent)
        chevron = "\u25be" if start_open else "\u25b8"
        self._title = title
        self._expand_content = expand_content
        if expand_content:
            self.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding if start_open else QSizePolicy.Fixed
            )
        self.setProperty("defaultExpanded", bool(start_open))
        button_type = QPushButton if flat_header else QToolButton
        self._toggle = button_type(text=f" {chevron}  {title}", checkable=True, checked=start_open)
        self._toggle.setProperty("sectionHeader", True)
        if not flat_header:
            self._toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.clicked.connect(self._on_toggled)

        self._body = _SectionBody()
        self._body.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding if expand_content else QSizePolicy.Fixed
        )
        self._body.setLayout(QVBoxLayout())
        self._body.layout().setContentsMargins(2, 2, 2, 2)
        self._body.setVisible(start_open)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self._toggle)
        root.addWidget(self._body, int(expand_content))

    def _on_toggled(self, checked: bool):
        """Apply header/body state and emit the public toggle signal."""
        chevron = "\u25be" if checked else "\u25b8"
        self._toggle.setText(f" {chevron}  {self._title}")
        self._body.setVisible(checked)
        if self._expand_content:
            self.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding if checked else QSizePolicy.Fixed
            )
        self.toggled.emit(bool(checked))

    def collapse(self) -> None:
        """Collapse this section."""
        self._toggle.setChecked(False)
        self._on_toggled(False)

    def expand(self) -> None:
        """Expand this section."""
        self._toggle.setChecked(True)
        self._on_toggled(True)

    def is_expanded(self) -> bool:
        """Return whether this section body is currently visible."""
        return bool(self._toggle.isChecked())

    def content_layout(self):
        """Return the body layout where callers should add section widgets."""
        return self._body.layout()
