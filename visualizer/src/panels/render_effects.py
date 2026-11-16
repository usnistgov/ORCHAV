"""Optional presentation controls using the ordinary renderer capability contract."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QWidget,
)

from ..renderers.protocol import renderer_capabilities


def _renderer(panel):
    return getattr(panel.parent, "renderer", None)


def create_dusk_controls(panel):
    checkbox = QCheckBox("Scenario blue hour")
    checkbox.setToolTip(
        "Authored dusk lighting, facade accents and vehicle lamps. "
        "Available when the scenario includes a lighting recipe; off by default."
    )
    panel.widgets["dusk_cb"] = checkbox

    def apply(enabled):
        renderer = _renderer(panel)
        if renderer_capabilities(renderer).dusk_lighting:
            renderer.set_dusk_enabled(enabled)
            panel._sync_from_visualizer()

    checkbox.toggled.connect(apply)
    return checkbox


def create_contour_controls(panel):
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    checkbox = QCheckBox("Section contours")
    checkbox.setToolTip("Outline scene surfaces where enabled cutaway planes intersect them.")
    width = QDoubleSpinBox()
    width.setRange(0.5, 6)
    width.setValue(2)
    width.setSingleStep(0.5)
    width.setSuffix(" px")
    width.setToolTip("Section line width")
    row.addWidget(checkbox)
    row.addWidget(width)
    row.addStretch()
    panel.widgets.update(
        section_contours_row=container, section_contours_cb=checkbox, section_contours_width=width
    )

    def apply(*_):
        width.setVisible(checkbox.isChecked())
        renderer = _renderer(panel)
        if renderer_capabilities(renderer).section_contours:
            renderer.set_section_contours(checkbox.isChecked(), width=width.value())

    checkbox.toggled.connect(apply)
    width.valueChanged.connect(apply)
    width.hide()
    return container


def sync_effect_controls(panel):
    """Mirror renderer state after source changes, scripts and Demo takeover."""
    renderer = _renderer(panel)
    capabilities = renderer_capabilities(renderer)
    widgets = panel.widgets
    for key, capability in (
        ("dusk_cb", "dusk_lighting"),
        ("section_contours_row", "section_contours"),
    ):
        if key in widgets:
            widgets[key].setVisible(getattr(capabilities, capability))
    if capabilities.dusk_lighting and "dusk_cb" in widgets:
        state = renderer.get_dusk_state()
        checkbox = widgets["dusk_cb"]
        with QSignalBlocker(checkbox):
            checkbox.setChecked(state["enabled"])
        checkbox.setEnabled(state["available"])
    if capabilities.section_contours and "section_contours_cb" in widgets:
        state = renderer.get_section_contour_state()
        with QSignalBlocker(widgets["section_contours_cb"]):
            widgets["section_contours_cb"].setChecked(state["enabled"])
        with QSignalBlocker(widgets["section_contours_width"]):
            widgets["section_contours_width"].setValue(state["width"])
        widgets["section_contours_width"].setVisible(state["enabled"])
