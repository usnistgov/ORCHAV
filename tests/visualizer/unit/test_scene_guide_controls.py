"""Guide controls keep density conditional and restore actual renderer state."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.panels.render_panel import RenderPanel
from visualizer.src.panels.scene_guide_controls import restore_guide_controls
from visualizer.src.renderers.protocol import RendererCapabilities


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(RenderPanel, "_load_ibl_library_from_disk", lambda _self: None)
    state = dict(
        axes=False,
        grid=False,
        divisions=20,
        dimensions=False,
        dimension_style="size_labels",
        compass=False,
    )
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(
            axes=True, ground_grid=True, scene_guides=True, compass=True
        ),
        get_axes_visible=lambda: state["axes"],
        get_ground_grid_visible=lambda: state["grid"],
        get_ground_grid_divisions=lambda: state["divisions"],
        get_environment_dimensions_state=lambda: {
            "enabled": state["dimensions"],
            "style": state["dimension_style"],
        },
        get_compass_visible=lambda: state["compass"],
        get_compass_state=lambda: dict(
            available=True, enabled=state["compass"], label="Geographic north", reason=""
        ),
    )
    for method, key in (
        ("show_axes", "axes"),
        ("set_ground_grid_visible", "grid"),
        ("set_ground_grid_divisions", "divisions"),
        ("set_environment_dimensions_visible", "dimensions"),
        ("set_environment_dimensions_style", "dimension_style"),
        ("set_compass_visible", "compass"),
    ):
        setattr(
            renderer, method, Mock(side_effect=lambda value, key=key: state.update({key: value}))
        )
    panel = RenderPanel(SimpleNamespace(renderer=renderer))
    group = panel.create_scene_view_panel()
    qtbot.addWidget(group)
    group.setFixedWidth(440)
    group.show()
    panel._sync_from_visualizer()
    panel.test_state = state
    panel.test_group = group
    yield panel
    panel.cleanup()


def test_density_disclosure_and_native_settings(panel, qtbot):
    widgets = panel.widgets
    assert widgets["ground_grid_density"].isHidden()
    widgets["ground_grid_cb"].setChecked(True)
    widgets["ground_grid_divisions"].setValue(80)
    widgets["environment_dimensions_cb"].setChecked(True)
    assert not widgets["environment_dimensions_style"].isHidden()
    widgets["environment_dimensions_style"].setCurrentIndex(1)
    widgets["compass_cb"].setChecked(True)
    widgets["show_axes_cb"].setChecked(True)
    qtbot.wait(20)
    assert panel.test_state == dict(
        axes=True,
        grid=True,
        divisions=80,
        dimensions=True,
        dimension_style="paper_axes",
        compass=True,
    )
    assert not widgets["ground_grid_density"].isHidden()
    assert widgets["ground_grid_divisions"].height() >= 20
    assert panel.test_group.minimumSizeHint().width() <= 440
    widgets["ground_grid_cb"].setChecked(False)
    assert widgets["ground_grid_density"].isHidden()
    widgets["environment_dimensions_cb"].setChecked(False)
    assert widgets["environment_dimensions_style"].isHidden()


def test_source_sync_and_workspace_restore_are_not_echoed(panel):
    panel.test_state.update(axes=True, grid=True, divisions=40, dimensions=True, compass=True)
    panel._sync_from_visualizer()
    assert panel.widgets["show_axes_cb"].isChecked()
    assert panel.widgets["ground_grid_divisions"].value() == 40
    assert panel.widgets["environment_dimensions_cb"].isChecked()
    assert panel.widgets["compass_cb"].isChecked()
    renderer = panel.parent.renderer
    renderer.set_ground_grid_divisions.assert_not_called()
    renderer.set_environment_dimensions_visible.assert_not_called()
    renderer.set_compass_visible.assert_not_called()
    restore_guide_controls(
        panel,
        dict(
            ground_grid_divisions=60,
            show_environment_dimensions=False,
            show_compass=False,
            environment_dimensions_style="paper_axes",
        ),
    )
    assert panel.test_state["divisions"] == 60
    assert not panel.widgets["environment_dimensions_cb"].isChecked()
    assert not panel.widgets["compass_cb"].isChecked()
    assert panel.test_state["dimension_style"] == "paper_axes"
    restore_guide_controls(
        panel, dict(ground_grid_divisions=100000, environment_dimensions_style="invalid")
    )
    assert panel.test_state["divisions"] == 60
    assert panel.test_state["dimension_style"] == "paper_axes"


def test_cartesian_scene_disables_compass_with_reason(panel):
    panel.parent.renderer.get_compass_state = lambda: dict(
        available=False, enabled=False, label="", reason="No geographic coordinates in this scene."
    )
    panel._sync_from_visualizer()
    checkbox = panel.widgets["compass_cb"]
    assert not checkbox.isEnabled()
    assert not checkbox.isChecked()
    assert "No geographic" in checkbox.toolTip()
    assert not checkbox.isHidden()


def test_backend_without_guides_hides_optional_controls(panel):
    panel.parent.renderer.capabilities = RendererCapabilities()
    panel._sync_from_visualizer()
    for key in ("ground_grid_cb", "ground_grid_density", "environment_dimensions_cb", "compass_cb"):
        assert panel.widgets[key].isHidden()
