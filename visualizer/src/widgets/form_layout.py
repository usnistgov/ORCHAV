"""Align input-form controls while retaining Qt's sizing and row wrapping."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QBoxLayout,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSlider,
)

from .wrapping_layout import WrappingLayout


def _single_line(item):
    """Identify input rows, excluding multiline text and tall editor containers."""
    widget = item.widget()
    if isinstance(widget, QLabel):
        return not widget.wordWrap() and "\n" not in widget.text()
    if isinstance(widget, (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit)):
        return True
    if isinstance(widget, QSlider):
        return widget.orientation() == Qt.Orientation.Horizontal
    layout = widget.layout() if widget is not None else item.layout()
    if isinstance(layout, QBoxLayout):
        if layout.direction() not in (QBoxLayout.LeftToRight, QBoxLayout.RightToLeft):
            return False
    elif not isinstance(layout, WrappingLayout):
        return False
    controls = [layout.itemAt(index) for index in range(layout.count())]
    controls = [control for control in controls if not control.isEmpty()]
    if not controls or not all(_single_line(control) for control in controls):
        return False
    # A wrapping selector can occupy multiple rows in a narrow inspector.
    centers = [control.geometry().center().y() for control in controls]
    return max(centers) - min(centers) <= 1


class AlignedFormLayout(QFormLayout):
    """Center ordinary input labels and fields without changing their size hints.

    Qt can cap form labels below the styled field height even when labelAlignment
    requests vertical centering. Adjust only the completed single-line row;
    stacked rows, multiline fields and explicit top alignment remain native.
    """

    def setGeometry(self, rect):
        super().setGeometry(rect)
        for row in range(self.rowCount()):
            label = self.itemAt(row, QFormLayout.LabelRole)
            field = self.itemAt(row, QFormLayout.FieldRole)
            if label is None or field is None or label.isEmpty() or field.isEmpty():
                continue
            if label.alignment() & Qt.AlignmentFlag.AlignTop:
                continue
            if not _single_line(label) or not _single_line(field):
                continue
            label_rect, field_rect = label.geometry(), field.geometry()
            if label_rect.bottom() < field_rect.top() or field_rect.bottom() < label_rect.top():
                continue
            top = min(label_rect.top(), field_rect.top())
            bottom = max(label_rect.bottom(), field_rect.bottom())
            height = bottom - top + 1
            for item, geometry in ((label, label_rect), (field, field_rect)):
                geometry.moveTop(top + (height - geometry.height()) // 2)
                item.setGeometry(geometry)
