"""Independent scene/path controls, persistence and temporary Demo takeover."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pygfx as gfx
import pytest

from shared.statistics.themes import theme_manager
from shared.statistics.themes.palettes import GLOW_COLORMAP
from tests.visualizer.unit import (
    test_appearance_presets,
    test_building_style,
    test_dark_presentation,
)
from visualizer.src.materials.building_style import apply_building_style
from visualizer.src.materials.catalog import material_preset
from visualizer.src.materials.view_styles import PRESETS, styled_surface
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel
from visualizer.src.renderers.effect_settings import DEFAULT_BLOOM
from visualizer.src.renderers.pygfx.canvas import build_pygfx_effect_passes
from visualizer.src.services.mpc_appearance_service import MpcAppearanceService
from visualizer.src.services.session_service import SessionService
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import MaterialPayload, SurfaceColorSource

appearance = test_appearance_presets.service
entry = test_building_style.entry
demo = test_dark_presentation.demo


@pytest.mark.parametrize("scene", list(PRESETS))
def test_scene_presets_preserve_path_appearance_and_filters(appearance, scene):
    assert list(PRESETS) == ["Detailed", "Paper", "Plan", "Dark", "Glow"]
    viz = appearance.viz
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    viz.set_state(
        color_mode="path_loss",
        topk_render_enabled=True,
        topk_render_max_paths=91,
        mpc_allowed_orders=frozenset({0, 2}),
        mpc_visibility=MpcVisibility(enabled=False, paths=False),
    )
    theme_manager.set_colormap("continuous", "plasma")
    theme_manager.set_colormap("categorical", "tab20")
    viz.renderer.set_line_width(3.5)
    viz.renderer.set_point_size(6.0)
    paths = mpc.capture()
    state = viz.app_state
    appearance.apply(scene)
    assert mpc.capture() == paths
    assert viz.app_state == replace(
        state,
        antialiasing_mode="fxaa",
        texture_mipmaps=True,
    )
    assert appearance.capture()["bloom"]["enabled"] is (scene == "Glow")


def test_previous_and_saved_scene_looks_preserve_later_path_edits(appearance):
    viz = appearance.viz
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    appearance.apply("Dark")
    mpc.apply("Paper")
    viz.renderer.set_line_width(3.5)
    theme_manager.set_colormap("continuous", "plasma")
    appearance.mark_if_custom()
    assert appearance.active_name == "Dark"
    before = appearance.capture()
    appearance.save("My look")
    assert "mpc_appearance" not in appearance.load("My look")
    appearance.mark_if_custom()
    assert appearance.active_name == "My look"
    appearance.apply("Glow")
    viz.set_state(color_mode="delay")
    viz.renderer.set_point_size(7.0)
    paths = mpc.capture()
    appearance.restore(appearance.previous)
    assert appearance.capture() == before
    assert mpc.capture() == paths
    appearance.apply("Paper")
    appearance.apply("My look")
    assert appearance.capture() == before
    assert mpc.capture() == paths


def test_legacy_saved_look_keeps_paths_but_snapshot_restore_replays_them(appearance):
    viz = appearance.viz
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    appearance.apply("Dark")
    mpc.apply("Paper")
    snapshot = appearance.capture(include_mpc=True)
    (appearance.directory / "Legacy.json").write_text(json.dumps(snapshot))
    appearance.apply("Detailed")
    mpc.apply("Standard")
    paths = mpc.capture()
    appearance.apply("Legacy")
    assert appearance.surface == "dark"
    assert mpc.capture() == paths
    appearance.restore(snapshot)
    assert mpc.capture() == snapshot["mpc_appearance"]


def test_invalid_path_settings_in_saved_look_leave_scene_and_paths_unchanged(appearance):
    viz = appearance.viz
    viz.mpc_appearance_service = MpcAppearanceService(viz)
    before = appearance.capture(include_mpc=True)
    bad = deepcopy(before)
    bad.update(surface="glow", background=[0.1, 0.2, 0.3])
    bad["mpc_appearance"]["continuous_colormap"] = "missing-palette"
    with pytest.raises(ValueError, match="Unknown MPC palette"):
        appearance.restore(bad)
    assert appearance.capture(include_mpc=True) == before


def test_paper_shows_material_identity_and_restores_fine_edges(appearance):
    appearance.viz.renderer.set_edge_line_width(8)
    appearance.apply("Glow")
    appearance.apply("Paper")
    recipe = appearance.capture()
    assert recipe["surface"] == "radio" and recipe["unlit"]
    assert recipe["background"] == [1, 1, 1] and recipe["outlines"]
    assert recipe["edge_width"] == 1 and not recipe["bloom"]["enabled"]
    assert not recipe["lights"]["shadows_enabled"] and not recipe["skybox"]
    styled, _ = styled_surface(
        MaterialPayload(),
        SurfaceColorSource.MATERIAL,
        recipe["surface"],
        recipe["unlit"],
        "brick",
        False,
    )
    assert styled.base_color[:3] == tuple(material_preset("brick")["color"])


@pytest.mark.parametrize("parts", [False, True])
def test_detailed_explicitly_selects_sunny_and_best_supported_facades(
    appearance, monkeypatch, parts
):
    viz = appearance.viz
    viz.mesh_entries = [{"_osm_building_parts": [[0, 2]]}] if parts else [{}]
    monkeypatch.setattr(
        "visualizer.src.materials.appearance_recipes.cached_environment",
        lambda _: "cached.hdr",
    )
    viz.renderer.get_ibl_name = Mock(return_value="sky_sunset")
    viz.renderer.set_ibl = Mock(return_value=True)
    viz.renderer.apply_environment_lighting = Mock()
    viz.authored_surface_textures = False
    appearance.apply("Detailed")
    viz.renderer.set_ibl.assert_called_once_with("sky_sunny")
    viz.renderer.apply_environment_lighting.assert_called_once_with("sky_sunny")
    assert viz.renderer.get_skybox_visible()
    assert viz.building_appearance == ("varied" if parts else "procedural")
    assert viz.road_appearance == "marked" and viz.authored_surface_textures


def test_mixed_scene_uses_repeated_facades_for_buildings_without_parts(entry):
    original = entry["mesh"].payload
    styled = apply_building_style(entry, {}, enabled=True, eligible=True, mode="varied")
    assert styled["texture_path"]
    assert entry["_building_style_mode"][0] == "procedural"
    assert entry["mesh"].payload.triangle_uvs is not None
    assert (entry["mesh"].payload.vertices == original.vertices).all()


def test_mpc_presets_leave_hidden_layers_scene_and_filters_alone(appearance):
    viz = appearance.viz
    viz.set_state(
        topk_render_enabled=True,
        topk_render_max_paths=91,
        mpc_visibility=MpcVisibility(enabled=False, paths=False),
    )
    mpc = MpcAppearanceService(viz)
    scene = appearance.capture()
    for name in ("Standard", "Paper", "Dark", "Glow"):
        before = viz.app_state
        mpc.apply(name)
        assert mpc.current_name() == name
        assert appearance.capture() == scene
        assert viz.app_state == replace(
            before, color_mode=viz.app_state.color_mode, mpc_visibility=viz.app_state.mpc_visibility
        )
        assert not viz.app_state.mpc_visibility.enabled and not viz.app_state.mpc_visibility.paths
    mpc.apply("Paper")
    assert viz.app_state.color_mode == "delay"


def test_paths_controls_follow_internal_demo_recipes_and_manual_edits(appearance, qtbot):
    viz = appearance.viz
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    panel = MPCVisualizationPanel(viz)
    group = panel.create_panel()
    qtbot.addWidget(group)
    try:
        assert "appearance_preset_combo" not in panel.widgets
        mpc.apply("Glow")
        assert panel.widgets["continuous_cmap_combo"].currentText() == GLOW_COLORMAP
        assert panel.widgets["line_width_spin"].value() == 2.2
        panel.widgets["line_width_spin"].setValue(3.5)
        assert viz.renderer.get_line_width() == 3.5
        scene = appearance.capture()
        mpc.apply("Dark")
        assert mpc.current_name() == "Dark"
        assert appearance.capture() == scene
        mpc.apply("Glow")
        assert panel.widgets["line_width_spin"].value() == 2.2
        theme_manager.set_colormap("categorical", "tab20")
        assert panel.widgets["categorical_cmap_combo"].currentData() == "tab20"
    finally:
        panel.cleanup()


@pytest.mark.parametrize("color_mode", ["delay", "path_loss"])
def test_scalar_modes_preserve_semantic_palette_and_manual_palette_choices(
    appearance, qtbot, color_mode
):
    viz = appearance.viz
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    mpc.apply("Standard")
    panel = MPCVisualizationPanel(viz)
    group = panel.create_panel()
    qtbot.addWidget(group)
    try:
        assert "appearance_preset_combo" not in panel.widgets
        before = mpc.capture()
        viz.set_state(color_mode=color_mode)
        panel.sync_coloring_from_state(viz.app_state)
        assert panel.widgets[f"{color_mode}_rb"].isChecked()
        assert mpc.capture() == {**before, "color_mode": color_mode}
        assert panel.widgets["continuous_cmap_combo"].currentText() == "RdYlGn_r"
        from visualizer.src.utils.colors import ensure_continuous_lut

        colors = ensure_continuous_lut()
        assert colors[0, 1] > colors[0, 0]  # Small delay/loss: green.
        assert colors[-1, 0] > colors[-1, 1]  # Large delay/loss: red.
        panel.widgets["continuous_cmap_combo"].setCurrentText("plasma")
        assert mpc.capture() == {
            **before,
            "color_mode": color_mode,
            "continuous_colormap": "plasma",
        }
    finally:
        panel.cleanup()
        group.deleteLater()


@pytest.mark.parametrize(
    "invalid",
    [
        dict(line_width=float("nan")),
        dict(point_size=-2),
        dict(continuous_colormap="unknown"),
        dict(bounce_points=1),
    ],
)
def test_bad_mpc_recipe_is_rejected_before_mutation(appearance, invalid):
    mpc = MpcAppearanceService(appearance.viz)
    before = mpc.capture()
    with pytest.raises(ValueError):
        mpc.restore({**before, **invalid})
    assert mpc.capture() == before


def test_live_bloom_survives_aa_background_and_export_and_overrides_launch_flags(
    appearance, monkeypatch
):
    from visualizer.src.renderers.pygfx.high_resolution import _copy_effects

    r = appearance.viz.renderer
    r._gfx = gfx
    r._renderer = SimpleNamespace(effect_passes=())
    settings = {**DEFAULT_BLOOM, "enabled": True, "bloom_strength": 0.18}
    r.set_bloom_settings(settings)
    for update in (
        lambda: r.set_antialiasing_mode("fxaa"),
        lambda: r.set_background_color([0.1, 0.2, 0.3, 1]),
    ):
        update()
        bloom = next(
            p for p in r._renderer.effect_passes if type(p).__name__ == "PhysicalBasedBloomPass"
        )
        assert bloom.bloom_strength == 0.18
    effects = _copy_effects(r._renderer)
    copied = next(p for p in effects if type(p).__name__ == "PhysicalBasedBloomPass")
    assert copied is not bloom and copied.bloom_strength == bloom.bloom_strength
    monkeypatch.setenv("ORCHAV_PYGFX_ENABLE_BLOOM", "1")
    assert not any(
        type(p).__name__ == "PhysicalBasedBloomPass"
        for p in build_pygfx_effect_passes(gfx, clear_color=(0, 0, 0, 1), bloom=DEFAULT_BLOOM)
    )
    r.set_bloom_settings(DEFAULT_BLOOM)
    assert all(type(p).__name__ != "PhysicalBasedBloomPass" for p in r._renderer.effect_passes)


def test_bloom_custom_recipe_and_old_recipe_roundtrip(appearance):
    appearance.apply("Glow")
    recipe = appearance.capture()
    appearance.save("Saved glow")
    appearance.apply("Paper")
    appearance.apply("Saved glow")
    assert appearance.capture() == recipe
    old = {key: value for key, value in recipe.items() if key not in {"bloom", "edge_width"}}
    appearance.restore(old)
    assert not appearance.capture()["bloom"]["enabled"]
    before = appearance.capture()
    with pytest.raises(ValueError):
        appearance.restore({**before, "bloom": {**DEFAULT_BLOOM, "bloom_strength": float("nan")}})
    assert appearance.capture() == before


def test_workspace_saves_separate_scene_and_mpc_looks_and_reads_old_palette(appearance):
    viz = appearance.viz
    viz.appearance_preset_service = appearance
    mpc = viz.mpc_appearance_service = MpcAppearanceService(viz)
    viz.scene_appearance_service.set_building_transparency = lambda value: setattr(
        viz, "current_building_alpha", value
    )
    viz.scene_appearance_service.set_target_transparency = lambda value: setattr(
        viz, "current_target_alpha", value
    )
    session = SessionService(viz)
    appearance.apply("Glow")
    mpc.apply("Dark")
    saved = session._get_rendering_state()
    assert saved["appearance"]["bloom"]["enabled"]
    assert saved["mpc_appearance"] == mpc.capture()
    original = deepcopy(saved)
    appearance.apply("Paper")
    mpc.apply("Standard")
    session._restore_rendering_state(saved)
    assert appearance.capture() == saved["appearance"]
    assert mpc.capture() == saved["mpc_appearance"]
    assert saved == original
    session._restore_rendering_state(
        {"appearance": {**saved["appearance"], "continuous_colormap": "plasma"}}
    )
    assert theme_manager.current.continuous_colormap == "plasma"


@pytest.mark.parametrize("fixed", [True, False])
def test_demo_restores_bloom_mpc_sizes_palettes_and_filters_after_glow(demo, fixed):
    viz = demo.viz
    scene = viz.appearance_preset_service.capture()
    paths = viz.mpc_appearance_service.capture()
    state = viz.app_state
    assert demo.start(
        orbit=False,
        playback=False,
        presentation="Glow" if fixed else None,
        cycle_appearance=not fixed,
    )
    if not fixed:
        for _ in range(4):
            demo.tick(demo._next_look)
    assert viz.renderer.get_bloom_settings()["enabled"]
    assert viz.mpc_appearance_service.current_name() == "Glow"
    assert viz.app_state.topk_render_max_paths == 20
    demo.stop()
    assert viz.appearance_preset_service.capture() == scene
    assert viz.mpc_appearance_service.capture() == paths
    assert viz.app_state == state
