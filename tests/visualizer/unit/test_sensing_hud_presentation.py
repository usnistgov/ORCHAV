"""Sensing HUD content remains concise and responds while playback is paused."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from visualizer.src import extensions
from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter
from visualizer.src.sensing import runtime
from visualizer.src.sensing.path_context import AnalyticalPathContext
from visualizer.src.state import create_initial_state


class _Renderer(PygfxOverlayMixin):
    renderer_id = "pygfx"

    def __init__(self, **sensing):
        self.visualizer = SimpleNamespace(
            renderer=self,
            app_state=replace(
                create_initial_state(),
                extension_state={"sensing": {"show_detections": True, **sensing}},
            ),
            extension_services={"sensing": object()},
        )
        self._hud_overlay_specs = {}
        self._hud_overlay_labels = {}
        self._hud_suppressed = False
        self.last_frame_packet = None

    def _set_hud_overlay(self, overlay_id, **spec):
        spec["visible"] = spec["visible"] and self._hud_role_enabled(spec["role"])
        self._hud_overlay_specs[overlay_id] = spec

    def _clear_hud_overlay(self, overlay_id):
        if overlay_id in self._hud_overlay_specs:
            self._hud_overlay_specs[overlay_id]["visible"] = False

    def _reposition_hud_overlays(self):
        pass

    def _update_mpc_hud_overlays(self, _packet):
        pass

    def _update_trajectory_hud_overlay(self):
        pass

    def _sync_marker_legend_from_state(self, _packet):
        pass

    def _hide_tooltip(self):
        pass

    def request_redraw(self):
        pass


@pytest.mark.parametrize("hud_mode", ["compact", "detailed"])
@pytest.mark.parametrize("scene_mode", ["detections", "tracks", "evidence", "full"])
def test_sensing_legend_explains_symbols_without_repeating_controls(hud_mode, scene_mode):
    renderer = _Renderer(scene_mode=scene_mode, show_velocity_arrows=True)
    renderer.visualizer.app_state = replace(
        renderer.visualizer.app_state, viewport_hud_mode=hud_mode
    )
    adapter = PygfxSensingAdapter(renderer)
    html = adapter._scene_legend_html()
    assert "× Detections" in html
    assert "rings: range only" in html
    assert "+ Ground truth" in html
    assert ("◇ Tracks" in html) == (scene_mode in {"tracks", "full"})
    if scene_mode in {"tracks", "full"}:
        assert "Muted: tentative · faded: historical" in html
        assert "arrows: velocity" in html
    assert ("History cloud" in html) == (scene_mode in {"evidence", "full"})
    for unwanted in ("Markers overlay", "all prior results", "History: all", "Sensing:"):
        assert unwanted not in html


@pytest.mark.parametrize(
    "timestamp, expected", [(0.0, " at 0.000 s"), (None, ""), (float("nan"), "")]
)
def test_held_result_keeps_qualifier_without_inventing_a_time(timestamp, expected):
    renderer = _Renderer(scene_mode="tracks")
    adapter = PygfxSensingAdapter(renderer)
    adapter._selection_frame = {"stale": True, "result_time_s": timestamp}
    html = adapter._scene_legend_html()
    assert f"Held result{expected} (faded)" in html
    assert "None" not in html and "nan" not in html


def test_legend_respects_disabled_3d_sensing_truth_and_marker_glyphs():
    renderer = _Renderer(show_truth=False, use_marker_glyphs=False)
    adapter = PygfxSensingAdapter(renderer)
    html = adapter._scene_legend_html()
    assert "Detections" in html
    assert "×" not in html and "Ground truth" not in html
    renderer.visualizer.app_state = create_initial_state()
    assert adapter._scene_legend_html() == ""


@pytest.mark.parametrize("source_available", [False, True])
def test_status_only_identifies_visible_historical_geometry(source_available):
    from tests.visualizer.unit.test_analytical_path_context import _runtime

    viz, service, packet, _renderer, _objects, _calls, _generation = _runtime(
        source_available=source_available
    )
    context = service.path_context = AnalyticalPathContext()
    raw = {"_source": {"source_rt_frame_idx": 0, "timestamp": 0.0}}
    context.prepare(viz, service, raw, 0, packet)
    assert runtime._status_chips(viz, service, packet) == ()
    assert not context.historical_paths_visible
    raw["_source"]["timestamp"] = 0.01
    context.prepare(viz, service, raw, 1, packet)
    assert runtime._status_chips(viz, service, packet) == (
        ("Historical paths",) if source_available else ()
    )
    assert context.historical_paths_visible is source_available
    # Changing controls while paused immediately removes an irrelevant chip.
    viz.app_state = replace(
        viz.app_state, mpc_visibility=replace(viz.app_state.mpc_visibility, enabled=False)
    )
    assert runtime._status_chips(viz, service, packet) == ()


@pytest.mark.parametrize("category", ["viewport_hud_enabled", "viewport_hud_show_legends"])
def test_paused_hud_toggle_restores_sensing_legend_without_a_frame_read(monkeypatch, category):
    renderer = _Renderer(scene_mode="tracks")
    adapter = PygfxSensingAdapter(renderer)
    adapter._selection_frame = {"stale": True, "result_time_s": 0.5}
    monkeypatch.setitem(runtime._ANNOTATION_ADAPTERS, renderer, adapter)
    monkeypatch.setattr(extensions, "_RUNTIME_EXTENSIONS", {})
    monkeypatch.setattr(extensions, "ensure_external_extensions_loaded", lambda: None)
    runtime.register()
    adapter.update_scene_legend()
    original = renderer._hud_overlay_specs["sensing_scene_legend"]["html"]
    for enabled in (False, True):
        renderer.visualizer.app_state = replace(
            renderer.visualizer.app_state, **{category: enabled}
        )
        renderer.refresh_viewport_hud()
        spec = renderer._hud_overlay_specs["sensing_scene_legend"]
        assert spec["visible"] is enabled
        assert spec["html"] == original
    # This fixture has no provider, pipeline, cache, frame-sync or geometry API.


def test_runtime_hud_refresh_ignores_uninstantiated_features(monkeypatch):
    renderer = _Renderer()
    monkeypatch.setattr(extensions, "_RUNTIME_EXTENSIONS", {})
    monkeypatch.setattr(extensions, "ensure_external_extensions_loaded", lambda: None)
    runtime.register()
    extensions.refresh_runtime_hud(renderer)
    assert renderer not in runtime._ANNOTATION_ADAPTERS
    assert not renderer._hud_overlay_specs
