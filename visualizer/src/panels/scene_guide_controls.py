"""Compact controls for native environment measurements and orientation guides."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..renderers.protocol import renderer_capabilities


def create_dimensions_control(panel):
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    checkbox = QCheckBox("Environment dimensions")
    checkbox.setToolTip(
        "Show X, Y and Z spans of the complete static environment in metres. "
        "Moving targets, paths and guide graphics do not change these measurements."
    )
    panel.widgets["environment_dimensions_cb"] = checkbox
    panel.widgets["environment_dimensions_row"] = container
    row.addWidget(checkbox)
    style = QComboBox()
    style.addItem("Size labels", "size_labels")
    style.addItem("Paper axes", "paper_axes")
    style.setToolTip(
        "Size labels show the environment spans. Paper axes show world coordinates "
        "with X, Y and Z labels and numeric ticks in metres."
    )
    panel.widgets["environment_dimensions_style"] = style
    row.addWidget(style)
    row.addStretch()
    style.hide()

    def apply(enabled):
        renderer = getattr(panel.parent, "renderer", None)
        if renderer_capabilities(renderer).scene_guides:
            renderer.set_environment_dimensions_visible(enabled)
        sync_guide_controls(panel)

    def apply_style(_index):
        renderer = getattr(panel.parent, "renderer", None)
        if renderer_capabilities(renderer).scene_guides:
            renderer.set_environment_dimensions_style(style.currentData())

    checkbox.toggled.connect(apply)
    style.currentIndexChanged.connect(apply_style)
    return container


def create_grid_compass_controls(panel):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    row = QHBoxLayout()
    grid = QCheckBox("Ground grid")
    grid.setToolTip("A ground-plane grid centered on the complete static environment.")
    panel.widgets["ground_grid_cb"] = grid
    row.addWidget(grid)
    density = QWidget()
    density_row = QHBoxLayout(density)
    density_row.setContentsMargins(0, 0, 0, 0)
    density_row.addWidget(QLabel("Density:"))
    divisions = QSpinBox()
    divisions.setRange(2, 500)
    divisions.setValue(20)
    divisions.setSuffix(" cells")
    divisions.setKeyboardTracking(False)
    divisions.setToolTip("Cells along each side of the grid; higher values give closer lines.")
    density_row.addWidget(divisions)
    panel.widgets["ground_grid_density"] = density
    panel.widgets["ground_grid_divisions"] = divisions
    row.addWidget(density)
    row.addStretch()
    layout.addLayout(row)

    def toggle_grid(enabled):
        panel._on_ground_grid_toggled(enabled)
        sync_guide_controls(panel)

    def change_density(value):
        renderer = getattr(panel.parent, "renderer", None)
        if renderer_capabilities(renderer).scene_guides:
            renderer.set_ground_grid_divisions(value)

    grid.toggled.connect(toggle_grid)
    divisions.valueChanged.connect(change_density)
    density.hide()
    compass = QCheckBox("North compass")
    panel.widgets["compass_cb"] = compass

    def toggle_compass(enabled):
        renderer = getattr(panel.parent, "renderer", None)
        if renderer_capabilities(renderer).compass:
            renderer.set_compass_visible(enabled)
        sync_guide_controls(panel)

    compass.toggled.connect(toggle_compass)
    layout.addWidget(compass)
    return container


def sync_guide_controls(panel):
    renderer = getattr(panel.parent, "renderer", None)
    capabilities = renderer_capabilities(renderer)
    widgets = panel.widgets
    for key, visible in (
        ("environment_dimensions_cb", capabilities.scene_guides),
        ("environment_dimensions_row", capabilities.scene_guides),
        ("compass_cb", capabilities.compass),
    ):
        if key in widgets:
            widgets[key].setVisible(visible)
    if capabilities.scene_guides:
        dimension_state = renderer.get_environment_dimensions_state()
        for key, checked in (
            ("show_axes_cb", renderer.get_axes_visible()),
            ("environment_dimensions_cb", dimension_state["enabled"]),
        ):
            if key in widgets:
                with QSignalBlocker(widgets[key]):
                    widgets[key].setChecked(checked)
        if "ground_grid_divisions" in widgets:
            with QSignalBlocker(widgets["ground_grid_divisions"]):
                widgets["ground_grid_divisions"].setValue(renderer.get_ground_grid_divisions())
        if "environment_dimensions_style" in widgets:
            style = widgets["environment_dimensions_style"]
            with QSignalBlocker(style):
                style.setCurrentIndex(style.findData(dimension_state.get("style", "size_labels")))
            style.setVisible(dimension_state["enabled"])
    if "ground_grid_density" in widgets:
        widgets["ground_grid_density"].setVisible(
            capabilities.scene_guides and renderer.get_ground_grid_visible()
        )
    if capabilities.compass and "compass_cb" in widgets:
        compass = widgets["compass_cb"]
        state = renderer.get_compass_state()
        compass.setEnabled(state["available"])
        compass.setToolTip(
            f"{state['label']}. Rotates with the camera; requires the master HUD switch."
            if state["available"]
            else state["reason"]
        )
        with QSignalBlocker(compass):
            compass.setChecked(state["enabled"])


def restore_guide_controls(panel, state):
    """Restore optional workspace preferences through their renderer owners."""
    renderer = getattr(panel.parent, "renderer", None)
    capabilities = renderer_capabilities(renderer)
    if capabilities.scene_guides:
        style = state.get("environment_dimensions_style")
        if style in ("size_labels", "paper_axes"):
            renderer.set_environment_dimensions_style(style)
        divisions = state.get("ground_grid_divisions")
        if isinstance(divisions, int) and not isinstance(divisions, bool) and 2 <= divisions <= 500:
            renderer.set_ground_grid_divisions(divisions)
        if "show_environment_dimensions" in state:
            renderer.set_environment_dimensions_visible(bool(state["show_environment_dimensions"]))
    if capabilities.compass and "show_compass" in state:
        renderer.set_compass_visible(bool(state["show_compass"]))
    sync_guide_controls(panel)
