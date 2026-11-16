"""Presentation recipes share real palette, appearance and Demo state owners."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shared.statistics.themes import Theme, theme_manager
from shared.statistics.themes.palettes import DARK_PRESENTATION_COLORMAP
from tests.visualizer.unit import test_appearance_presets, test_dusk_manual_appearance
from tests.visualizer.unit.test_demo_service import Appearance
from tests.visualizer.unit.test_demo_service import rig as rig
from visualizer.src.materials.view_styles import dark_presentation_ground, styled_surface
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services.demo_service import DemoService
from visualizer.src.services.mpc_appearance_service import MpcAppearanceService
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload, SurfaceColorSource

appearance = test_appearance_presets.service
dusk_renderer = test_dusk_manual_appearance.renderer


@pytest.fixture
def demo(appearance, qapp):
    viz = appearance.viz
    viz.ready = True
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.current_scenario_path = "test-scene"
    viz.camera_controller = SimpleNamespace(presentation=lambda: None)
    viz.appearance_preset_service = appearance
    viz.mpc_appearance_service = MpcAppearanceService(viz)
    service = DemoService(viz)
    yield service
    service.close()


def test_palette_is_available_to_figures_and_live_path_lut(appearance):
    from visualizer.src.utils.colors import ensure_continuous_lut

    cmap = Theme(
        name="presentation", continuous_colormap=DARK_PRESENTATION_COLORMAP
    ).get_continuous_cmap()
    assert cmap(0.0)[:3] == pytest.approx((0.12, 0.94, 0.77))
    assert cmap(1.0)[:3] == pytest.approx((1.0, 0.72, 0.27))
    theme_manager.set_colormap("continuous", DARK_PRESENTATION_COLORMAP)
    lut = ensure_continuous_lut()
    np.testing.assert_allclose(lut[[0, -1]], [cmap(0.0)[:3], cmap(1.0)[:3]], rtol=1e-6)


@pytest.mark.parametrize(
    ("material_type", "ground", "color"),
    [
        ("concrete", True, (0.12, 0.17, 0.23)),
        ("concrete", False, (0.39, 0.46, 0.54)),
        ("itu_metal", False, (0.23, 0.29, 0.37)),
    ],
)
def test_matte_context_preserves_authored_material_and_actor(material_type, ground, color):
    material = MaterialPayload(base_color=(0.8, 0.2, 0.1, 0.6), texture_path="authored.png")
    styled, source = styled_surface(
        material, SurfaceColorSource.VERTEX, "dark", False, material_type, False, ground
    )
    assert styled.base_color == (*color, 0.6)
    assert styled.roughness == 0.95 and styled.reflectance == 0.15
    assert styled.texture_path is None and source is SurfaceColorSource.MATERIAL
    target, target_source = styled_surface(
        material, SurfaceColorSource.VERTEX, "dark", False, material_type, True, ground
    )
    assert target is material and target.texture_path == "authored.png"
    assert target_source is SurfaceColorSource.VERTEX


def test_ground_classification_refreshes_after_geometry_replacement():
    payload = MeshPayload(
        vertices=np.array([[0.0, 0, 0], [5, 0, 0], [0, 5, 0]]),
        triangles=np.array([[0, 1, 2]]),
    )
    entry = {"material_type": "concrete"}
    assert dark_presentation_ground(entry, payload)
    cached = entry["_presentation_height"]
    assert dark_presentation_ground(entry, payload)
    assert entry["_presentation_height"] is cached
    wall = replace(payload, vertices=np.array([[0.0, 0, 0], [5, 0, 0], [0, 0, 12]]))
    assert not dark_presentation_ground(entry, wall)
    assert np.all(payload.vertices[:, 2] == 0)


def test_saved_scene_look_preserves_palette_and_accepts_legacy_recipe(appearance):
    theme_manager.set_colormap("continuous", "plasma")
    before = appearance.capture()
    appearance.apply("Dark")
    dark = appearance.capture()
    assert dark["background"] == [0.035, 0.055, 0.085]
    assert dark["lights"]["shadows_enabled"] and dark["ibl"] == 0
    assert not dark["textures"] and not dark["skybox"]
    appearance.save("My dark look")
    assert theme_manager.current.continuous_colormap == "plasma"
    appearance.restore(before)
    assert theme_manager.current.continuous_colormap == "plasma"
    appearance.apply("My dark look")
    assert appearance.capture() == dark
    legacy = {**before, "continuous_colormap": DARK_PRESENTATION_COLORMAP}
    appearance.restore(legacy)
    assert theme_manager.current.continuous_colormap == "plasma"


def test_demo_holds_dark_look_and_restores_custom_appearance_and_path_choices(demo):
    viz = demo.viz
    viz.set_state(
        color_mode="path_loss",
        topk_render_enabled=True,
        topk_render_max_paths=7,
        mpc_visibility=MpcVisibility(enabled=False),
    )
    theme_manager.set_colormap("continuous", "plasma")
    state = viz.app_state
    appearance = viz.appearance_preset_service
    before = appearance.capture()
    metadata = deepcopy((appearance.active_name, appearance.previous, appearance.last_applied))
    # Default lighting cycles and explicit sky cycles cannot replace this look.
    assert demo.start(orbit=False, playback=False, presentation="Dark", cycle_ibl=True)
    assert viz.app_state.color_mode == "delay"
    assert viz.app_state.topk_render_max_paths == 20 and viz.app_state.topk_render_enabled
    assert viz.app_state.mpc_visibility == MpcVisibility(bounce_points=False)
    assert viz.app_state.step == state.step
    assert viz.app_state.mpc_allowed_orders == state.mpc_allowed_orders
    assert theme_manager.current.continuous_colormap == DARK_PRESENTATION_COLORMAP
    dark = appearance.capture()
    demo.tick(demo._next_look + 100)
    assert appearance.capture() == dark
    demo.stop()
    assert viz.app_state == state
    assert appearance.capture() == before
    assert (appearance.active_name, appearance.previous, appearance.last_applied) == metadata


def test_partial_start_failure_restores_palette_and_state(demo, monkeypatch):
    viz = demo.viz
    before = viz.appearance_preset_service.capture()
    state = viz.app_state
    setter = viz.set_state
    failed = False

    def fail_once(**changes):
        nonlocal failed
        if "color_mode" in changes and not failed:
            failed = True
            raise RuntimeError("path update failed")
        setter(**changes)

    monkeypatch.setattr(viz, "set_state", fail_once)
    assert not demo.start(orbit=False, playback=False, presentation="Dark")
    assert not demo.active
    assert viz.app_state == state
    assert viz.appearance_preset_service.capture() == before


@pytest.mark.parametrize("progress", [0.35, 1.0])
def test_dark_demo_replaces_dusk_temporarily_and_restores_its_progress(
    demo, dusk_renderer, progress
):
    demo.viz.renderer = dusk_renderer
    assert dusk_renderer.set_dusk_progress(progress)
    before = dusk_renderer.get_dusk_state()
    ambient = tuple(dusk_renderer._ambient_light.color)
    assert demo.start(orbit=False, playback=False, presentation="Dark")
    assert not dusk_renderer.get_dusk_state()["enabled"]
    demo.stop()
    assert dusk_renderer.get_dusk_state() == before
    assert tuple(dusk_renderer._ambient_light.color) == ambient


def test_path_and_palette_restore_even_if_appearance_restore_fails(demo, monkeypatch):
    viz = demo.viz
    theme_manager.set_colormap("continuous", "plasma")
    state = viz.app_state
    assert demo.start(orbit=False, playback=False, presentation="Dark")
    monkeypatch.setattr(
        viz.appearance_preset_service, "restore", Mock(side_effect=RuntimeError("lost renderer"))
    )
    demo.stop()
    for key in ("color_mode", "topk_render_enabled", "topk_render_max_paths", "mpc_visibility"):
        assert getattr(viz.app_state, key) == getattr(state, key)
    assert theme_manager.current.continuous_colormap == "plasma"
    assert "lost renderer" in demo.status


@pytest.mark.parametrize("replacement_palette", [None, "inferno"])
def test_scene_change_releases_owned_palette_without_overwriting_new_scene(
    demo, replacement_palette
):
    viz = demo.viz
    theme_manager.set_colormap("continuous", "plasma")
    assert demo.start(orbit=False, playback=False, presentation="Dark")
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.set_state(color_mode="path_loss", topk_render_max_paths=81)
    if replacement_palette:
        theme_manager.set_colormap("continuous", replacement_palette)
    demo.tick()
    assert not demo.active
    assert theme_manager.current.continuous_colormap == (replacement_palette or "plasma")
    assert viz.app_state.color_mode == "path_loss" and viz.app_state.topk_render_max_paths == 81


def test_demo_panel_temporarily_disables_conflicting_look_controls(rig, qtbot):
    viz, _service, _tools = rig
    viz.appearance_preset_service = Appearance(viz.renderer)
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        panel.widgets["demo_look"].setCurrentText("Dark")
        for key in ("demo_lighting", "demo_ibl", "demo_appearance", "demo_interval"):
            assert not panel.widgets[key].isEnabled()
        panel.service.start = Mock()
        panel.start()
        assert panel.service.start.call_args.kwargs["presentation"] == "Dark"
        panel.widgets["demo_look"].setCurrentText("Current")
        assert panel.widgets["demo_appearance"].isEnabled()
        assert panel.widgets["demo_appearance"].isChecked()
    finally:
        panel.cleanup()
