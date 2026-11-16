"""Wrap widget toolbars without imposing a desktop-sized minimum width."""

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class WrappingLayout(QLayout):
    """Lay out visible controls in rows, centered within each row's height."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect, *, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        line = []

        def place_line():
            if apply:
                for item, item_x, width, height in line:
                    item.setGeometry(QRect(item_x, y + (line_height - height) // 2, width, height))

        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            width = min(size.width(), max(item.minimumSize().width(), area.width()))
            if x > area.x() and x + width > area.right() + 1:
                place_line()
                line.clear()
                x, y, line_height = area.x(), y + line_height + self.spacing(), 0
            height = (
                max(size.height(), item.heightForWidth(width))
                if item.hasHeightForWidth()
                else size.height()
            )
            line.append((item, x, width, height))
            x += width + self.spacing()
            line_height = max(line_height, height)
        place_line()
        return y + line_height - rect.y() + margins.bottom()
