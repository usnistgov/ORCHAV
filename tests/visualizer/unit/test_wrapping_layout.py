"""Wrapping toolbar rows align mixed controls without changing their sizing contract."""

from PySide6.QtCore import QRect, QSize
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel, QPushButton, QWidget

from visualizer.src.widgets.responsive import wrap_control_row
from visualizer.src.widgets.wrapping_layout import WrappingLayout


def _fixed_controls(qtbot, sizes):
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = WrappingLayout(parent)
    controls = []
    for width, height in sizes:
        control = QWidget()
        control.setFixedSize(width, height)
        layout.addWidget(control)
        controls.append(control)
    parent.show()
    return parent, layout, controls


def test_mixed_controls_share_row_center_and_keep_natural_sizes(qtbot):
    parent, layout, controls = _fixed_controls(qtbot, [(40, 20), (50, 40), (30, 25)])
    layout.setGeometry(QRect(0, 0, 150, 80))

    assert [control.width() for control in controls] == [40, 50, 30]
    assert [control.height() for control in controls] == [20, 40, 25]
    centers = [control.geometry().center().y() for control in controls]
    assert max(centers) - min(centers) <= 1
    assert [control.x() for control in controls] == [0, 46, 102]
    assert layout.heightForWidth(150) == 40


def test_wrapped_rows_center_independently_and_preserve_boundary(qtbot):
    parent, layout, controls = _fixed_controls(qtbot, [(40, 20), (50, 40), (60, 14), (30, 30)])
    layout.setGeometry(QRect(0, 0, 96, 100))

    assert [control.geometry() for control in controls] == [
        QRect(0, 10, 40, 20),
        QRect(46, 0, 50, 40),
        QRect(0, 54, 60, 14),
        QRect(66, 46, 30, 30),
    ]
    assert layout.heightForWidth(96) == 76
    assert layout.heightForWidth(95) > 76


def test_margins_and_nonzero_origin_apply_to_all_controls(qtbot):
    parent, layout, controls = _fixed_controls(qtbot, [(40, 20), (50, 40)])
    layout.setContentsMargins(3, 5, 7, 9)
    layout.setGeometry(QRect(10, 20, 110, 100))

    assert [control.geometry() for control in controls] == [
        QRect(13, 35, 40, 20),
        QRect(59, 25, 50, 40),
    ]
    assert layout.heightForWidth(110) == 54
    assert layout.minimumSize() == QSize(60, 54)


def test_hidden_controls_do_not_set_row_height_or_consume_width(qtbot):
    parent, layout, controls = _fixed_controls(qtbot, [(40, 20), (80, 70), (40, 30)])
    controls[1].hide()
    layout.setGeometry(QRect(0, 0, 86, 100))

    assert controls[0].geometry() == QRect(0, 5, 40, 20)
    assert controls[2].geometry() == QRect(46, 0, 40, 30)
    assert layout.heightForWidth(86) == 30
    controls[1].show()
    layout.setGeometry(QRect(0, 0, 86, 150))
    assert layout.heightForWidth(86) == 132


def test_height_for_width_measures_without_moving_widgets(qtbot):
    parent, layout, controls = _fixed_controls(qtbot, [(40, 20), (50, 40)])
    layout.setGeometry(QRect(0, 0, 100, 100))
    before = [control.geometry() for control in controls]

    assert layout.heightForWidth(60) == 66
    assert [control.geometry() for control in controls] == before


def test_width_dependent_control_keeps_its_required_height(qtbot):
    class WrappedControl(QWidget):
        def sizeHint(self):
            return QSize(80, 20)

        def minimumSizeHint(self):
            return QSize(10, 20)

        def heightForWidth(self, width):
            return 40 if width < 80 else 20

    parent = QWidget()
    qtbot.addWidget(parent)
    layout = WrappingLayout(parent)
    wrapped = WrappedControl()
    policy = wrapped.sizePolicy()
    policy.setHeightForWidth(True)
    wrapped.setSizePolicy(policy)
    layout.addWidget(wrapped)
    parent.show()
    layout.setGeometry(QRect(0, 0, 50, 100))

    assert wrapped.geometry() == QRect(0, 0, 50, 40)
    assert layout.heightForWidth(50) == 40
    layout.setGeometry(QRect(0, 0, 100, 100))
    assert wrapped.geometry() == QRect(0, 0, 80, 20)
    assert layout.heightForWidth(100) == 20


def test_grouped_toolbar_centers_checkboxes_label_spinbox_and_button(qtbot):
    from PySide6.QtWidgets import QHBoxLayout

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet("QDoubleSpinBox { min-height: 30px; padding: 3px; }")
    row = QHBoxLayout()
    controls = [QCheckBox("Axes"), QLabel("Marker:"), QDoubleSpinBox(), QPushButton("Rename")]
    for control in controls:
        row.addWidget(control)
    layout = wrap_control_row(row, [1, 2, 1])
    parent.setLayout(layout)
    parent.resize(600, 100)
    parent.show()
    layout.activate()

    centers = [control.mapTo(parent, control.rect().center()).y() for control in controls]
    assert max(centers) - min(centers) <= 1
    assert controls[0].height() < controls[2].height()
