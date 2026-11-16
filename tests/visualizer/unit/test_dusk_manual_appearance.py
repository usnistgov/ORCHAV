"""Ordinary backdrop controls take over cleanly from temporary blue-hour state."""

from unittest.mock import Mock

import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.renderer import PygfxRenderer


@pytest.fixture
def renderer():
    owner = PygfxRenderer(None)
    owner._gfx = gfx
    owner._scene = gfx.Scene()
    owner._install_default_lights()
    owner._ibl_manager._background = gfx.Background.from_color("#406080")
    owner._ibl_manager._scene = owner._scene
    owner.set_background_color([0.3, 0.4, 0.5, 1])
    owner.configure_dusk({"label": "Test blue hour"})
    yield owner
    owner.close()


def _backgrounds(renderer):
    # Sunset keeps the original backdrop resident for an exact daylight return.
    return [
        child
        for child in renderer._scene.children
        if isinstance(child, gfx.Background) and child.visible
    ]


@pytest.mark.parametrize("sky", [False, True])
def test_backdrop_choice_retires_gradient_and_restores_normal_light_colors(renderer, sky):
    original = tuple(renderer._ambient_light.color)
    renderer.set_dusk_enabled(True)
    gradient = renderer._dusk_background
    assert _backgrounds(renderer) == [gradient]
    renderer.show_skybox(sky)
    assert not renderer.get_dusk_state()["enabled"]
    assert gradient.parent is None
    assert len(_backgrounds(renderer)) == 1
    assert renderer.get_skybox_visible() is sky
    assert tuple(renderer._ambient_light.color) == original


def test_solid_color_choice_survives_later_idempotent_dusk_disable(renderer):
    renderer.set_dusk_enabled(True)
    gradient = renderer._dusk_background
    renderer.set_background_color([0.7, 0.6, 0.5, 1])
    assert not renderer.get_dusk_state()["enabled"]
    assert gradient.parent is None and len(_backgrounds(renderer)) == 1
    renderer.set_dusk_enabled(False)
    assert renderer._clear_color == (0.7, 0.6, 0.5, 1)


@pytest.mark.parametrize("success", [False, True])
def test_environment_choice_hands_off_but_failed_load_preserves_dusk(
    renderer, monkeypatch, success
):
    original = renderer.get_light_rig_state()
    renderer.set_dusk_enabled(True)

    def load(_name):
        assert not renderer.get_dusk_state()["enabled"]
        assert renderer.get_light_rig_state() == original
        return success

    monkeypatch.setattr(renderer, "_load_ibl_into_scene", load)
    assert renderer.set_ibl("test.hdr") is success
    assert renderer.get_dusk_state()["enabled"] is not success
    assert len(_backgrounds(renderer)) == 1


def test_internal_environment_binding_does_not_replace_active_gradient(renderer, monkeypatch):
    renderer.set_dusk_enabled(True)
    gradient = renderer._dusk_background
    monkeypatch.setattr(renderer._ibl_manager, "apply_to_scene", Mock(return_value=True))
    renderer._apply_active_ibl_to_scene_materials()
    assert renderer.get_dusk_state()["enabled"]
    assert _backgrounds(renderer) == [gradient]


def test_close_restores_persistent_light_and_background_fields_for_reuse(renderer):
    original = (renderer.get_light_rig_state(), renderer.get_ibl_intensity(), renderer._clear_color)
    renderer.set_dusk_enabled(True)
    renderer.close()
    assert (
        renderer.get_light_rig_state(),
        renderer.get_ibl_intensity(),
        renderer._clear_color,
    ) == original
    assert not renderer.get_dusk_state()["enabled"]


def test_scenario_recipe_syncs_existing_direct_checkbox(qapp, renderer):
    from types import SimpleNamespace

    from visualizer.src.app.scenario_presentation import apply_presentation_defaults
    from visualizer.src.panels.render_effects import create_dusk_controls
    from visualizer.src.panels.render_panel import RenderPanel

    renderer.configure_dusk(None)
    viz = SimpleNamespace(renderer=renderer)
    panel = RenderPanel(viz)
    checkbox = create_dusk_controls(panel)
    panel._sync_from_visualizer = panel.sync_presentation_effects
    viz.ui_manager = SimpleNamespace(panels={"render": panel})
    panel.sync_presentation_effects()
    assert not checkbox.isEnabled() and not checkbox.isChecked()
    toggles = Mock()
    checkbox.toggled.connect(toggles)
    for _replacement in range(2):
        apply_presentation_defaults(viz, {"dusk": {"label": "Immediate direct control"}})
        assert checkbox.isEnabled() and not checkbox.isChecked()
        toggles.assert_not_called()
        checkbox.click()
        assert renderer.get_dusk_state()["enabled"] and checkbox.isChecked()
        checkbox.click()
        assert not renderer.get_dusk_state()["enabled"]
        checkbox.click()
        toggles.reset_mock()
        apply_presentation_defaults(viz, {})
        assert not checkbox.isEnabled() and not checkbox.isChecked()
        assert not renderer.get_dusk_state()["available"]
        assert not renderer.get_dusk_state()["enabled"]
        toggles.assert_not_called()
