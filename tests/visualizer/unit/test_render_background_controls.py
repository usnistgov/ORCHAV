"""Backdrop disclosure must preserve independent IBL and restored appearances."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QRadioButton

from visualizer.src.panels.render_panel import RenderPanel
from visualizer.src.renderers.protocol import RendererCapabilities


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(RenderPanel, "_load_ibl_library_from_disk", lambda _self: None)
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(ibl=True, skybox=True),
        skybox=False,
        intensity=2000,
        get_ibl_name=lambda: "default",
        set_ibl=Mock(return_value=True),
    )
    renderer.get_skybox_visible = lambda: renderer.skybox
    renderer.get_ibl_intensity = lambda: renderer.intensity
    renderer.show_skybox = Mock(side_effect=lambda value: setattr(renderer, "skybox", value))
    renderer.set_ibl_intensity = Mock(
        side_effect=lambda value: setattr(renderer, "intensity", value)
    )
    parent = SimpleNamespace(renderer=renderer, scene_appearance_service=Mock())
    panel = RenderPanel(parent)
    group = panel.create_lighting_panel()
    qtbot.addWidget(group)
    group.show()
    panel._sync_from_visualizer()
    yield panel
    panel.cleanup()


def test_backdrop_radios_are_exclusive_and_expand_only_the_relevant_controls(panel, qtbot):
    sky = panel.widgets["skybox_cb"]
    solid = panel.widgets["background_solid_radio"]
    color = panel.widgets["background_color_row"]
    environment = panel.widgets["environment_group"]
    assert isinstance(sky, QRadioButton) and isinstance(solid, QRadioButton)
    assert solid.isChecked() and not sky.isChecked()
    assert not color.isHidden() and not environment.is_expanded()

    qtbot.mouseClick(sky, Qt.LeftButton)
    assert sky.isChecked() and not solid.isChecked()
    assert color.isHidden() and environment.is_expanded()
    qtbot.mouseClick(solid, Qt.LeftButton)
    assert solid.isChecked() and not sky.isChecked()
    assert not color.isHidden() and not environment.is_expanded()
    assert [call.args[0] for call in panel.parent.renderer.show_skybox.call_args_list] == [
        True,
        False,
    ]
    panel.parent.renderer.set_ibl.assert_not_called()
    panel.parent.renderer.set_ibl_intensity.assert_not_called()


def test_renderer_sync_restores_both_radios_without_echoing_or_reclosing_manual_ibl(panel):
    renderer = panel.parent.renderer
    environment = panel.widgets["environment_group"]
    renderer.skybox = True
    panel._sync_from_visualizer()
    assert panel.widgets["skybox_cb"].isChecked() and environment.is_expanded()
    renderer.skybox = False
    panel._sync_from_visualizer()
    assert panel.widgets["background_solid_radio"].isChecked()
    assert not panel.widgets["skybox_cb"].isChecked()
    assert not environment.is_expanded()
    environment.expand()
    panel._sync_from_visualizer()
    assert environment.is_expanded()
    renderer.show_skybox.assert_not_called()
    renderer.set_ibl_intensity.assert_not_called()


def test_blocked_settings_restore_can_leave_an_exclusive_skybox_radio(panel):
    panel._apply_settings_to_widgets({"skybox_cb": True})
    snapshot = panel._snapshot_settings({"skybox_cb": None})
    panel._apply_settings_to_widgets({"skybox_cb": False, "bg_combo": "White"})
    assert panel.widgets["background_solid_radio"].isChecked()
    assert not panel.widgets["skybox_cb"].isChecked()
    assert panel.parent.renderer.skybox is False
    assert not panel.widgets["background_color_row"].isHidden()
    panel._apply_settings_to_widgets(snapshot)
    assert panel.widgets["skybox_cb"].isChecked()
    assert panel.parent.renderer.skybox is True
    assert panel.widgets["environment_group"].is_expanded()


def test_solid_background_can_still_change_environment_map_and_intensity(panel):
    panel.widgets["environment_group"].expand()
    panel._on_ibl_changed("Default")
    panel.widgets["ibl_intensity_spin"].setValue(3500)
    renderer = panel.parent.renderer
    renderer.set_ibl.assert_called_once_with("default")
    renderer.set_ibl_intensity.assert_called_once_with(3500.0)
    assert not renderer.skybox
    assert panel.widgets["background_solid_radio"].isChecked()
    renderer.show_skybox.assert_not_called()


def test_renderer_without_skybox_retains_color_and_environment_lighting(panel):
    panel.parent.renderer.capabilities = RendererCapabilities(ibl=True)
    panel._sync_from_visualizer()
    assert panel.widgets["skybox_cb"].isHidden()
    assert panel.widgets["background_solid_radio"].isHidden()
    assert not panel.widgets["background_color_row"].isHidden()
    assert not panel.widgets["environment_group"].isHidden()


@pytest.mark.parametrize("action", ["solid", "sky", "ibl"])
def test_manual_backdrop_action_updates_blue_hour_checkbox(panel, action):
    renderer = panel.parent.renderer
    renderer.capabilities = RendererCapabilities(ibl=True, skybox=True, dusk_lighting=True)
    state = {"enabled": True, "available": True}
    renderer.get_dusk_state = lambda: dict(state)
    panel._sync_from_visualizer()
    assert panel.widgets["dusk_cb"].isChecked()
    if action == "solid":
        panel.parent.scene_appearance_service.set_background_preset.side_effect = (
            lambda *_: state.update(enabled=False)
        )
        panel._on_background_changed("White")
    elif action == "sky":
        renderer.show_skybox.side_effect = lambda *_: state.update(enabled=False)
        panel._on_skybox_toggled(True)
    else:
        renderer.set_ibl.side_effect = lambda *_: state.update(enabled=False) or True
        panel._on_ibl_changed("Default")
    assert not panel.widgets["dusk_cb"].isChecked()
