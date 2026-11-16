"""Independent notebook looks and shared TX/RX annotation behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.statistics.themes import theme_manager
from visualizer.src.materials.texture_policy import (
    texture_maps_override,
    textures_globally_enabled,
    view_texture_maps,
)
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.notebook.presentation import APPEARANCES, NotebookPresentation
from visualizer.src.pipeline.core import MPCCore
from visualizer.src.scene.assembly import mesh_entry_to_payload
from visualizer.src.types.render_payloads import MeshPayload

gfx = pytest.importorskip("pygfx")
pytestmark = pytest.mark.headless


@pytest.fixture(autouse=True)
def no_sky_download(monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.materials.appearance_recipes.cached_environment", lambda _: None
    )


@pytest.mark.parametrize("appearance", APPEARANCES)
def test_looks_do_not_change_process_settings(appearance, monkeypatch):
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    theme, textures = theme_manager.current, texture_maps_override()
    view = NotebookPresentation(gfx, appearance, "card")
    assert view.settings["bloom"]["enabled"] == (appearance == "Glow")
    assert view.mpc["color_mode"] == (
        "delay" if appearance in ("Dark", "Glow") else "reflection_order"
    )
    assert theme_manager.current is theme
    assert texture_maps_override() == textures


@pytest.mark.parametrize("appearance", ["Plan", "missing", "paper"])
def test_unsupported_appearance_has_an_actionable_error(appearance):
    with pytest.raises(ValueError, match="Notebook appearance must be one of"):
        NotebookPresentation(gfx, appearance, "plain")


def test_texture_scope_restores_policy_after_nested_failure(monkeypatch):
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    before = textures_globally_enabled()
    with view_texture_maps(True):
        assert textures_globally_enabled()
        with pytest.raises(RuntimeError), view_texture_maps(False):
            assert not textures_globally_enabled()
            raise RuntimeError("assembly failed")
        assert textures_globally_enabled()
    assert textures_globally_enabled() == before
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    with view_texture_maps(True):
        assert not textures_globally_enabled()
    with pytest.raises(ValueError, match="Detailed requires textures"):
        NotebookPresentation(gfx, "Detailed", "plain")


@pytest.mark.parametrize("mode", ["callout", "card"])
def test_cards_follow_anchors_and_keep_pixel_sizes_across_camera_changes(mode):
    view = NotebookPresentation(
        gfx,
        "Paper",
        {
            "mode": mode,
            "tx_marker_size": 30,
            "rx_marker_size": 12,
        },
        tx_color=(0.1, 0.8, 0.2),
        rx_color=(0.8, 0.1, 0.4),
    )
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 30)
    canvas = SimpleNamespace(get_logical_size=lambda: (800, 600))
    for x in (0.0, 3.0):
        scene.clear()
        view.clear_dynamic()
        tx = np.array([[x, 0, 0]])
        rx = np.array([[x + 0.05, 0, 0]])
        view.add_node_labels(scene, tx, rx)
        for distance in (30, 60):
            camera.local.position = (0, 0, distance)
            view.prepare_draw(canvas, camera)
            for kind, position, size in (("tx", tx[0], 30), ("rx", rx[0], 12)):
                native = view._objects[f"node:{kind}_0"]
                marker = native._orchav_callout_group.children[2]
                np.testing.assert_allclose(marker.geometry.positions.data[0], position)
                assert marker.material.size == size
        assert len(view._label_backgrounds) == 2
        assert len(view._objects) == 2


def test_palette_changes_colors_without_changing_geometry_or_other_views():
    raw = {
        "canonical_data": CanonicalStepData(
            points=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
            order=np.array([0, 1, 1], dtype=np.uint8),
            itype=np.array([0, 1, 0], dtype=np.uint8),
            delay=np.array([1, 2, 3], dtype=np.float32),
            loss=np.array([10, 20, 30], dtype=np.float32),
            delay_min=1,
            delay_max=3,
        ),
        "tx_positions": np.array([[0, 0, 0]]),
        "rx_positions": np.array([[2, 0, 0]]),
    }
    dark = NotebookPresentation(gfx, "Dark", "plain")
    core = MPCCore(subscribe_to_theme=False)
    first = core.create_view_model(0, raw, "delay", color_palette=dark.palette)
    paper = NotebookPresentation(gfx, "Paper", "plain")
    second = core.create_view_model(0, raw, "delay", color_palette=paper.palette)
    again = core.create_view_model(0, raw, "delay", color_palette=dark.palette)
    assert first is not None and second is not None and again is not None
    np.testing.assert_array_equal(first.mpc_points, second.mpc_points)
    np.testing.assert_array_equal(first.mpc_lines, second.mpc_lines)
    np.testing.assert_array_equal(first.mpc_colors, again.mpc_colors)
    assert not np.array_equal(first.mpc_colors, second.mpc_colors)
    np.testing.assert_array_equal(first.mpc_bounce_colors, again.mpc_bounce_colors)
    assert core._on_theme_change not in theme_manager._listeners


def test_surface_style_changes_only_display_material():
    payload = MeshPayload(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    entry = {"name": "wall", "mesh": payload, "material_type": "concrete"}
    name, mesh, material = mesh_entry_to_payload(dict(entry), {})
    first = NotebookPresentation(gfx, "Paper", "plain")
    second = NotebookPresentation(gfx, "Dark", "plain")
    for view in (first, second):
        view.add_mesh(gfx.Scene(), name, mesh, material, dict(entry))
    assert type(first._objects[name].material) is gfx.MeshBasicMaterial
    assert type(second._objects[name].material) is not gfx.MeshBasicMaterial
    assert entry["mesh"] is payload
    np.testing.assert_array_equal(mesh.vertices, payload.vertices)


def test_explicit_path_options_override_the_preset(tmp_path):
    from visualizer.src.notebook.pygfx import PygfxNotebookViz

    viz = object.__new__(PygfxNotebookViz)
    viz._mesh_entries, viz._texture_cache = [], {}
    viz._project_root = tmp_path
    viz._scenario = SimpleNamespace(root=tmp_path)
    viz._build_frame_data = MagicMock(return_value=(None, None))
    viz._add_dynamic = MagicMock()
    scene, _ = viz._build_pygfx_scene(
        0,
        "reflection_order",
        "all",
        "all",
        True,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        4.0,
        9.0,
        1.0,
        "neutral_outdoor",
        1.0,
        appearance="Glow",
    )
    assert viz._build_frame_data.call_args.args[1] == "reflection_order"
    assert viz._build_frame_data.call_args.args[6] is True
    assert viz._add_dynamic.call_args.args[-3:] == (4.0, 9.0, 1.0)
    assert scene._orchav_presentation.settings["bloom"]["enabled"]
