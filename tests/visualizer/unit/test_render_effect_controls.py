"""Optional effects disclose settings and reflect renderer state without redraw loops."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QGroupBox

from visualizer.src.panels.render_effects import (
    create_contour_controls,
    create_dusk_controls,
    sync_effect_controls,
)
from visualizer.src.renderers.protocol import RendererCapabilities


@pytest.fixture
def panel(qtbot):
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(dusk_lighting=True, section_contours=True),
        dusk={"available": True, "enabled": False},
        contour={"enabled": False, "width": 2},
    )
    renderer.get_dusk_state = lambda: renderer.dusk
    renderer.get_section_contour_state = lambda: renderer.contour
    renderer.set_dusk_enabled = Mock(
        side_effect=lambda enabled: renderer.dusk.update(enabled=enabled)
    )
    renderer.set_section_contours = Mock(
        side_effect=lambda enabled, **kw: renderer.contour.update(enabled=enabled, **kw)
    )
    panel = SimpleNamespace(
        parent=SimpleNamespace(renderer=renderer), widgets={}, create_subgroup_box=QGroupBox
    )
    panel._sync_from_visualizer = lambda: sync_effect_controls(panel)
    for create in (create_dusk_controls, create_contour_controls):
        widget = create(panel)
        qtbot.addWidget(widget)
        widget.show()
    sync_effect_controls(panel)
    return panel


def test_effects_are_off_and_settings_expand_only_when_enabled(panel):
    widgets = panel.widgets
    assert widgets["section_contours_width"].isHidden()
    widgets["section_contours_cb"].setChecked(True)
    widgets["dusk_cb"].setChecked(True)
    assert not widgets["section_contours_width"].isHidden()
    assert panel.parent.renderer.dusk["enabled"]
    widgets["section_contours_cb"].setChecked(False)
    assert widgets["section_contours_width"].isHidden()


def test_source_and_script_sync_does_not_echo_renderer_changes(panel):
    renderer = panel.parent.renderer
    renderer.dusk.update(available=False, enabled=False)
    renderer.contour.update(enabled=True, width=3)
    sync_effect_controls(panel)
    assert not panel.widgets["dusk_cb"].isEnabled()
    assert panel.widgets["section_contours_width"].value() == 3
    renderer.set_dusk_enabled.assert_not_called()
    renderer.set_section_contours.assert_not_called()
    renderer.capabilities = RendererCapabilities()
    sync_effect_controls(panel)
    assert all(panel.widgets[key].isHidden() for key in ("dusk_cb", "section_contours_row"))
