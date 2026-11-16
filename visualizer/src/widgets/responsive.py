"""Small width-aware layouts for the resizable visualizer sidebar."""

from PySide6.QtCore import QEvent, QSize
from PySide6.QtWidgets import QBoxLayout, QHBoxLayout, QStackedWidget, QWidget

from .wrapping_layout import WrappingLayout


class CurrentPageStack(QStackedWidget):
    """Let the visible form determine size, rather than the largest hidden form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else QSize(0, 0)

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else QSize(0, 0)

    def hasHeightForWidth(self):
        page = self.currentWidget()
        return page is not None and page.hasHeightForWidth()

    def heightForWidth(self, width):
        page = self.currentWidget()
        return page.heightForWidth(width) if page is not None else -1


class ResponsiveRow(QWidget):
    """Keep related groups adjacent when they fit, otherwise stack them."""

    def __init__(self, parent=None, *, spacing=6):
        super().__init__(parent)
        self.row = QBoxLayout(QBoxLayout.TopToBottom, self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(spacing)

    def minimumSizeHint(self):
        width = max(
            (self.row.itemAt(i).minimumSize().width() for i in range(self.row.count())), default=0
        )
        return QSize(width, 0)

    def _reflow(self):
        required = sum(self.row.itemAt(i).minimumSize().width() for i in range(self.row.count()))
        required += self.row.spacing() * max(0, self.row.count() - 1)
        direction = QBoxLayout.LeftToRight if self.width() >= required else QBoxLayout.TopToBottom
        if self.row.direction() != direction:
            self.row.setDirection(direction)

    def resizeEvent(self, event):
        self._reflow()
        super().resizeEvent(event)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            self._reflow()
        return super().event(event)


def wrap_control_row(row, groups):
    """Move an unmounted row into wrapping groups, retaining label/control pairs."""
    flow = WrappingLayout()
    flow.setSpacing(row.spacing())
    for count in groups:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(row.spacing())
        for _ in range(count):
            item = row.takeAt(0)
            if item is not None:
                if item.widget() is not None:
                    layout.addWidget(item.widget())
                else:
                    layout.addItem(item)
        flow.addWidget(container)
    while row.count():
        item = row.takeAt(0)
        if item.widget() is not None:
            flow.addWidget(item.widget())
    return flow
