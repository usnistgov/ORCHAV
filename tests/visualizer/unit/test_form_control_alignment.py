"""Input labels follow styled controls without moving tall or stacked form rows."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QStyleFactory,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from visualizer.src.app.theme import ThemeManager, generate_application_stylesheet
from visualizer.src.authoring.mobility_editor import MobilityEditor
from visualizer.src.authoring.orientation_editor import OrientationEditor
from visualizer.src.authoring.target_representation_editor import TargetRepresentationEditor
from visualizer.src.panels.advanced_camera_panel import AdvancedCameraPanel
from visualizer.src.panels.label_style_controls import LabelStyleControls
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.sensing.scene_explanation import SceneExplanationControls
from visualizer.src.sensing.volume_evaluation import VolumeEvaluationControls
from visualizer.src.widgets.form_layout import AlignedFormLayout


@pytest.fixture(params=["Fusion", "Windows"])
def qt_style(qapp, request):
    if request.param not in QStyleFactory.keys():
        pytest.skip(f"Qt style {request.param} is unavailable")
    previous = qapp.style().objectName()
    qapp.setStyle(request.param)
    yield
    qapp.setStyle(previous)


def _workspace(qtbot, scale):
    workspace = QWidget()
    workspace.setProperty("workflowWorkspace", True)
    manager = ThemeManager()
    manager._scale = scale
    workspace.setStyleSheet(manager._scaled_stylesheet(generate_application_stylesheet()))
    qtbot.addWidget(workspace)
    return workspace


def _editor(kind):
    if kind == "labels":
        visualizer = SimpleNamespace(
            app_state=SimpleNamespace(label_callout={"mode": "card"}),
            renderer=SimpleNamespace(capabilities=RendererCapabilities(label_callouts=True)),
            ui_controller=SimpleNamespace(handle_label_callout_changed=lambda _: None),
        )
        return LabelStyleControls(visualizer)
    if kind == "mobility":
        editor = MobilityEditor()
        editor.type_combo.setCurrentIndex(editor.type_combo.findData("circular"))
        editor.circular_advanced_section.set_expanded(True)
        return editor
    if kind == "orientation":
        editor = OrientationEditor()
        editor.type_combo.setCurrentIndex(editor.type_combo.findData("spin"))
        return editor
    if kind == "target":
        editor = TargetRepresentationEditor()
        editor.representation.setCurrentIndex(editor.representation.findData("constant_rcs"))
        return editor
    return VolumeEvaluationControls(collapsible=False)


@pytest.mark.parametrize("scale", [1.0, 1.3])
@pytest.mark.parametrize("kind", ["labels", "mobility", "orientation", "target", "evaluation"])
def test_real_editor_labels_center_on_themed_spinboxes_and_combos(qtbot, qt_style, scale, kind):
    workspace = _workspace(qtbot, scale)
    editor = _editor(kind)
    layout = QVBoxLayout(workspace)
    layout.addWidget(editor)
    workspace.resize(950, 1000)
    workspace.show()
    qtbot.waitUntil(editor.isVisible)
    checked = []
    for form in editor.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            field = form.itemAt(row, QFormLayout.FieldRole)
            if label is None or field is None or field.widget() is None:
                continue
            control = field.widget()
            if not isinstance(control, (QDoubleSpinBox, QComboBox)) or not control.isVisible():
                continue
            assert abs(label.geometry().center().y() - control.geometry().center().y()) <= 1
            checked.append(control)
    assert any(isinstance(control, QDoubleSpinBox) for control in checked)
    assert any(isinstance(control, QComboBox) for control in checked)


@pytest.mark.parametrize("scale", [1.0, 1.3])
def test_form_preserves_tall_fields_stacked_rows_and_explicit_top_alignment(qtbot, qt_style, scale):
    workspace = _workspace(qtbot, scale)
    form = AlignedFormLayout(workspace)
    tall = QPlainTextEdit()
    tall.setFixedHeight(150)
    form.addRow("Notes", tall)
    summary = QLabel("A saved result with several lines of explanation. " * 8)
    summary.setWordWrap(True)
    summary.setFixedWidth(200)
    form.addRow("Summary", summary)
    field = QDoubleSpinBox()
    form.addRow("Explicit top", field)
    top_label = form.labelForField(field)
    form.setAlignment(top_label, Qt.AlignmentFlag.AlignTop)
    workspace.resize(500, 500)
    workspace.show()
    qtbot.waitUntil(workspace.isVisible)

    for control in (tall, summary, field):
        label = form.labelForField(control)
        assert label.geometry().top() == control.geometry().top()
        assert label.geometry().center().y() < control.geometry().center().y()

    form.setRowWrapPolicy(QFormLayout.WrapAllRows)
    form.activate()
    for control in (tall, summary, field):
        assert form.labelForField(control).geometry().bottom() < control.geometry().top()


@pytest.mark.parametrize("scale", [1.0, 1.3])
def test_wrapped_report_label_follows_first_control_line(qtbot, qt_style, scale):
    workspace = _workspace(qtbot, scale)
    layout = QVBoxLayout(workspace)
    controls = SceneExplanationControls(QCheckBox())
    layout.addWidget(controls)
    controls.phase.setFixedWidth(150)
    controls.play.setFixedWidth(220)
    workspace.resize(250, 450)
    workspace.show()
    qtbot.waitUntil(lambda: controls.play.y() > controls.phase.geometry().bottom())

    label = next(label for label in controls.findChildren(QLabel) if label.text() == "Report")
    assert abs(label.geometry().center().y() - controls.phase.geometry().center().y()) <= 1
    assert label.geometry().bottom() < controls.play.geometry().top()


def test_input_widgets_in_label_column_and_horizontal_groups_stay_centered(qtbot, qt_style):
    workspace = _workspace(qtbot, 1.3)
    form = AlignedFormLayout(workspace)
    checkbox, spin = QCheckBox("Smooth"), QDoubleSpinBox()
    combo, button = QComboBox(), QPushButton("Focus")
    form.addRow(checkbox, spin)
    form.addRow(combo, button)
    vector = QHBoxLayout()
    vector.addWidget(QLabel("X"))
    coordinate = QDoubleSpinBox()
    vector.addWidget(coordinate)
    form.addRow("Position", vector)
    workspace.resize(500, 250)
    workspace.show()
    qtbot.waitUntil(workspace.isVisible)

    for label, field in ((checkbox, spin), (combo, button)):
        assert abs(label.geometry().center().y() - field.geometry().center().y()) <= 1
    vector_label = form.labelForField(vector)
    assert abs(vector_label.geometry().center().y() - coordinate.geometry().center().y()) <= 1
    assert checkbox.height() < spin.height()


def test_advanced_camera_page_resizes_after_theme_scaling_and_tab_changes(qtbot, qt_style):
    workspace = _workspace(qtbot, 1.0)
    owner = SimpleNamespace(app_state=None)
    panel = AdvancedCameraPanel(owner)
    group = panel.create_panel()
    layout = QVBoxLayout(workspace)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    layout.addWidget(group)
    workspace.resize(1000, 1400)
    workspace.show()
    tabs = group.findChild(QTabWidget)
    manager = ThemeManager()
    heights = []
    for scale in (1.0, 1.3, 1.0):
        manager._scale = scale
        workspace.setStyleSheet(manager._scaled_stylesheet(generate_application_stylesheet()))
        for index in (0, 2, 0):
            tabs.setCurrentIndex(index)
            page = tabs.currentWidget()
            qtbot.waitUntil(lambda: page.height() >= page.minimumSizeHint().height())
            form = page.layout()
            previous_bottom = -1
            for row in range(form.rowCount()):
                items = [
                    form.itemAt(row, role)
                    for role in (QFormLayout.LabelRole, QFormLayout.FieldRole)
                ]
                rectangles = [item.geometry() for item in items if item and not item.isEmpty()]
                if rectangles:
                    assert min(rect.top() for rect in rectangles) > previous_bottom
                    previous_bottom = max(rect.bottom() for rect in rectangles)
            assert previous_bottom < page.height()
        heights.append(tabs.height())
    assert heights[1] > heights[0]
    assert heights[2] == heights[0]
