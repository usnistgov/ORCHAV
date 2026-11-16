"""Display recipes preserve scientific state and survive material/JSON round trips."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shared.statistics.themes import theme_manager
from visualizer.src.materials import texture_policy
from visualizer.src.materials.view_styles import PRESETS, styled_surface
from visualizer.src.services.appearance_preset_service import AppearancePresetService
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload, SurfaceColorSource


@pytest.fixture
def service(tmp_path, monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(theme_manager, "_current_theme", theme_manager.current)
    # Startup code writes os.environ directly. Record even originally absent
    # keys so monkeypatch restores them after each launch-policy case.
    for key in ("ORCHAV_DISABLE_TEXTURES", "ORCHAV_ENABLE_TEXTURES"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", None)
    monkeypatch.setattr(
        "visualizer.src.materials.appearance_recipes.cached_environment", lambda _: None
    )
    from visualizer.src.state import create_initial_state, update_state

    viz = SimpleNamespace(
        mesh_entries=[],
        target_entries=[],
        schedule_update=Mock(),
        facade_theme="paris",
        building_appearance="varied",
        current_target_alpha=0.7,
        app_state=replace(create_initial_state(), step=12),
    )
    viz.set_state = lambda **changes: setattr(
        viz, "app_state", update_state(viz.app_state, **changes)
    )
    viz.renderer = PygfxRenderer(viz)
    viz.scene_appearance_service = SimpleNamespace(
        set_edge_visibility=lambda value: setattr(viz, "outlines_enabled", value),
        set_background_color=lambda value: setattr(viz, "current_background_color", list(value)),
    )
    viz.object_appearance_service = SimpleNamespace(refresh_entry_appearance_batch=lambda _: True)
    viz.material_pbr_service = MaterialPBRService(viz)
    return AppearancePresetService(viz, tmp_path)


@pytest.mark.parametrize("explicit_scenario", [True, False])
@pytest.mark.parametrize(
    ("flag", "saved_textures", "no_resume", "env_enabled", "expected"),
    [
        ("--enable-textures", False, False, False, True),
        ("--disable-textures", True, False, False, False),
        (None, False, False, True, False),
        (None, True, False, False, True),
        (None, False, False, False, False),
        ("--enable-textures", False, True, False, True),
        (None, True, True, False, False),
        ("--enable-textures", None, False, False, True),
        ("--disable-textures", None, False, False, False),
    ],
)
def test_startup_texture_precedence_preserves_workspace_and_later_choices(
    service, monkeypatch, explicit_scenario, flag, saved_textures, no_resume, env_enabled, expected
):
    from visualizer.src.app import startup_workflow
    from visualizer.src.services.session_service import (
        SessionService,
        WorkspaceSnapshot,
        WorkspaceSnapshotSummary,
    )

    root = service.directory / "scenario"
    appearance = {**service.capture(), "textures": saved_textures, "wireframe": True}
    payload = {
        "camera": {"format": "orbit-v1", "eye": [10.0, 20.0, 30.0]},
        "rendering": {"appearance": appearance} if saved_textures is not None else {},
    }
    original = deepcopy(payload)
    selection = WorkspaceSnapshot(
        summary=WorkspaceSnapshotSummary(
            path=service.directory / "autosave.json",
            scenario_root=root,
            scenario_name="scenario",
            created_at=datetime(2026, 10, 13),
            frame=0,
            is_autosave=True,
        ),
        payload=payload,
    )
    select_workspace = Mock(return_value=selection)
    monkeypatch.setattr(startup_workflow, "select_startup_workspace", select_workspace)
    if env_enabled:
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    argv = ["--scenario", str(root)] if explicit_scenario else []
    if flag:
        argv.append(flag)
    if no_resume:
        argv.append("--no-resume")
    args = startup_workflow.parse_cli_args(argv)
    startup_workflow.apply_launch_environment(args)

    viz = service.viz
    viz.appearance_preset_service = service
    viz._scene_only_mode = True
    viz._deferred_init = Mock()
    viz.open_scenario = Mock(return_value=SimpleNamespace(succeeded=True, scenario_root=root))
    session = SimpleNamespace(
        viz=viz, _restore_mpc_appearance_state=Mock(), _get_panel=lambda _: None
    )

    def load_session(snapshot, **kwargs):
        # Exercise real session appearance restoration and material publication,
        # without loading scene assets, frame data, or the user's autosaves.
        SessionService._restore_rendering_state(session, snapshot.payload["rendering"])
        return True

    session.load_session = Mock(side_effect=load_session)
    viz.session_service = session
    startup_workflow._complete_cli_startup(
        args=args,
        app=SimpleNamespace(processEvents=lambda: None),
        reporter=SimpleNamespace(note=Mock(), task=lambda _: nullcontext()),
        viz=viz,
    )

    assert texture_policy.textures_globally_enabled() is expected
    assert selection.payload == original
    if no_resume:
        select_workspace.assert_not_called()
        session.load_session.assert_not_called()
    else:
        session.load_session.assert_called_once()
        restored = session.load_session.call_args.args[0]
        assert restored.summary == selection.summary
        assert restored.camera == selection.camera
        viz._deferred_init.assert_called_once_with(pending_camera=selection.camera)
        if saved_textures is not None:
            # The disable flag must not reject the rest of the appearance preset.
            assert service.capture() == {**appearance, "textures": expected}
        else:
            assert "appearance" not in restored.payload["rendering"]

    if flag == "--enable-textures" and saved_textures is False and not no_resume:
        service.restore(appearance)  # A later manual workspace choice still wins.
        assert texture_policy.textures_globally_enabled() is False
        assert viz.material_pbr_service.set_texture_maps_enabled(True)
        assert texture_policy.textures_globally_enabled() is True


def test_builtin_progression_preserves_experiment_and_actor_state(service):
    original = vars(service.viz.app_state).copy()
    for name in PRESETS:
        service.apply(name)
        assert vars(service.viz.app_state) == {
            **original,
            "antialiasing_mode": "fxaa",
            "texture_mipmaps": True,
        }
        assert service.viz.current_target_alpha == 0.7
        assert service.viz.facade_theme == "paris"
        assert service.viz.building_appearance == "procedural"
        assert not service.capture()["wireframe"]
        assert service.active_name == ("Custom" if name == "Plan" else name)
    service.apply("Detailed inspection")
    assert texture_policy.textures_globally_enabled()
    assert service.surface == "original" and not service.unlit


@pytest.mark.parametrize("restore_saved", [False, True])
def test_manual_appearance_takes_over_from_dusk_before_saving_previous(service, restore_saved):
    import pygfx as gfx

    renderer = service.viz.renderer
    renderer._gfx = gfx
    renderer._scene = gfx.Scene()
    renderer._install_default_lights()
    renderer.configure_dusk({"label": "Test blue hour"})
    original = service.capture()
    original_color = tuple(renderer._ambient_light.color)
    renderer.set_dusk_enabled(True)
    if restore_saved:
        service.restore(original)
    else:
        service.apply("Paper")
    assert not renderer.get_dusk_state()["enabled"]
    assert renderer._dusk_background is None
    assert tuple(renderer._ambient_light.color) == original_color
    assert service.previous == original
    renderer.close()


def test_custom_json_and_workspace_snapshot_roundtrip(service):
    service.apply("Detailed inspection")
    service.viz.renderer.set_key_light_intensity(1.7)
    service.viz.renderer.show_skybox(False)
    service.mark_if_custom()
    assert service.active_name == "Custom"
    captured = service.capture()
    service.save("My look")
    assert service.names() == ["My look"]
    assert (service.directory / "My look.json").stat().st_size < 4096
    service.apply("Glow")
    service.apply("My look")
    assert service.capture() == captured
    service.apply("Paper")
    service.restore(captured)
    assert service.capture() == captured


def test_display_state_remains_consistent_after_preset_and_restore(service):
    from visualizer.src.panels.render_panel import RenderPanel

    original = service.capture()
    service.apply("Paper")
    viz = service.viz
    assert viz.app_state.antialiasing_mode == viz.renderer.get_antialiasing_mode() == "fxaa"
    assert viz.app_state.texture_mipmaps == viz.renderer.get_texture_mipmaps_enabled()
    # Exercise the workspace synchronization path which used to revert the
    # preset's renderer settings from stale AppState values.
    panel = RenderPanel(viz)
    panel._push_clipping_planes_to_renderer = lambda: None
    panel.apply_runtime_controls_from_state()
    assert service.capture()["aa"] == "fxaa"
    service.restore(original)
    panel.apply_runtime_controls_from_state()
    assert service.capture()["aa"] == original["aa"]
    assert service.capture()["mipmaps"] == original["mipmaps"]


@pytest.mark.parametrize("old_mode", ["ddaa", "ppaa"])
def test_legacy_appearance_aa_restores_as_fxaa_without_mutating_saved_recipe(service, old_mode):
    recipe = {**service.capture(), "aa": old_mode}
    original = deepcopy(recipe)

    service.restore(recipe)

    assert recipe == original
    assert service.capture()["aa"] == "fxaa"
    assert service.viz.app_state.antialiasing_mode == "fxaa"


@pytest.mark.parametrize(
    "update",
    [
        dict(version=2),
        dict(alpha=float("nan")),
        dict(alpha=2),
        dict(shadow_bias=15),
        dict(textures="yes"),
        dict(surface="bogus"),
        dict(surface=[]),
        dict(bloom={}),
        dict(bloom=[]),
    ],
)
def test_invalid_recipe_rejected_before_mutation(service, update):
    before = service.capture()
    with pytest.raises(ValueError):
        service.restore({**before, **update})
    assert service.capture() == before


@pytest.mark.parametrize("name", ["../escape", "bad/name", "", " ", " padded ", "Paper", "Custom"])
def test_preset_paths_and_builtin_names_are_protected(service, name):
    with pytest.raises(ValueError):
        service.save(name)
    assert not list(service.directory.glob("*.json"))


def test_explicit_texture_lock_is_respected_before_any_change(service, monkeypatch):
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    before = service.capture()
    with pytest.raises(ValueError, match="locked off"):
        service.apply("Detailed inspection")
    assert service.capture() == before


def test_missing_sky_has_no_download_or_hidden_success(service):
    service.apply("Outdoor presentation")
    assert "not installed" in service.notice
    assert not service.capture()["skybox"]
    assert service.capture()["textures"]


def test_style_preserves_originals_and_target_colors():
    base = MaterialPayload(base_color=(0.1, 0.5, 0.8, 0.6), metallic=0.7, texture_path="map.png")
    neutral, source = styled_surface(
        base, SurfaceColorSource.VERTEX, "geometry", True, "concrete", False
    )
    assert neutral.shader == "unlit" and neutral.texture_path is None
    assert neutral.base_color[3] == 0.6 and source == SurfaceColorSource.MATERIAL
    target, source = styled_surface(
        base, SurfaceColorSource.VERTEX, "geometry", True, "concrete", True
    )
    assert target.base_color == base.base_color and source == SurfaceColorSource.VERTEX
    original, _ = styled_surface(
        base, SurfaceColorSource.VERTEX, "original", False, "concrete", False
    )
    assert original is base and base.texture_path == "map.png"


def test_snapshot_color_choice_never_mutates_vertex_data():
    from visualizer.src.model import RenderObjectState
    from visualizer.src.services.scene_render_sync import SceneRenderSync

    payload = MeshPayload(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        triangles=np.array([[0, 1, 2]]),
        vertex_colors=np.ones((3, 3)),
        color_source=SurfaceColorSource.VERTEX,
    )
    state = RenderObjectState(id="scene:mesh", payload=payload)
    received = []
    sync = SceneRenderSync(
        SimpleNamespace(
            renderer=SimpleNamespace(ensure_object=lambda value: received.append(value) or True)
        )
    )
    for _ in range(2):
        assert sync.ensure_object(state, snapshot_color_source=SurfaceColorSource.MATERIAL)
    assert state.payload is payload and payload.color_source == SurfaceColorSource.VERTEX
    assert received[0].payload is received[1].payload
    assert received[0].payload.vertices is payload.vertices
    assert received[0].payload.color_source == SurfaceColorSource.MATERIAL


def test_wireframe_survives_native_material_swap_and_excludes_overlay():
    from tests.visualizer.unit.test_pygfx_pbr_correctness import _live_harness

    h = _live_harness()
    h._wireframe_enabled = True
    h._is_wireframe_candidate = lambda name: name == "mesh"
    assert h.set_named_material("mesh", MaterialPayload(shader="unlit"))
    assert h._objects["mesh"].material.wireframe
    h._is_wireframe_candidate = lambda name: False
    assert h.set_named_material("mesh", MaterialPayload(shader="lit"))
    assert not h._objects["mesh"].material.wireframe


def test_depth_precision_recipe_roundtrip_and_older_recipe_compatibility(service):
    recipe = service.capture()
    recipe["depth_precision"] = True
    service.restore(recipe)
    assert service.viz.renderer.get_depth_precision_enabled()
    assert service.capture()["depth_precision"] is True
    old_recipe = {k: v for k, v in recipe.items() if k != "depth_precision"}
    service.restore(old_recipe)
    assert service.viz.renderer.get_depth_precision_enabled()
    with pytest.raises(ValueError, match="depth_precision"):
        service.restore({**recipe, "depth_precision": "yes"})
