"""Cinematic style shares the recorded geometric clock and bounded lifecycle."""

from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pytest

from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry, cinematic_subset
from visualizer.src.experiments.path_analysis import SPEED_OF_LIGHT_M_PER_NS, geometric_progress
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay


def test_subset_strongest_and_deterministic_fallback():
    catalog = _catalog(list(range(120)), losses=list(range(120))[::-1])
    ids, note = cinematic_subset(catalog, np.arange(120), 10)
    np.testing.assert_array_equal(ids, np.arange(119, 109, -1))
    assert "10/120" in note
    for limit in (10, 50, 100):
        unknown = _catalog(list(range(120)), known_loss=False)
        ids, note = cinematic_subset(unknown, np.arange(120)[::-1], limit)
        np.testing.assert_array_equal(ids, np.arange(limit))
        assert "canonical-ID" in note
    with pytest.raises(ValueError):
        cinematic_subset(catalog, [0], 12)


def test_clock_cache_parity_trails_and_recorded_interaction():
    catalog = _catalog([99], points=[[(0, 0, 0), (3, 0, 0), (3, 4, 0)]])
    catalog._canonical_data.itype[1] = 1
    geometry = CinematicPulseGeometry(catalog, [0])
    bounce = 3 / SPEED_OF_LIGHT_M_PER_NS
    assert geometry.sample(bounce - 0.01).interaction_colors[0, 3] == 0
    assert geometry.sample(bounce).interaction_colors[0, 3] == pytest.approx(0.85)
    assert geometry.sample(bounce + geometry.extent * 0.04).interaction_colors[0, 3] == 0
    for time in (-1, 0, 5, bounce, 15, 30):
        sample = geometry.sample(time)
        expected = geometric_progress(catalog, [0], time)
        np.testing.assert_allclose(sample.leading, expected.positions)
        assert sample.trail.shape == (10, 3)
        for j, fraction in enumerate(np.linspace(0.1, 1, 10)):
            expected = geometric_progress(catalog, [0], time - geometry.trail_ns * fraction)
            np.testing.assert_allclose(sample.trail[j], expected.positions[0], atol=1e-6)


def test_unknown_interaction_has_no_flash_and_empty_geometry():
    catalog = _catalog([1], points=[[(0, 0, 0), (1, 0, 0), (1, 1, 0)]])
    assert len(CinematicPulseGeometry(catalog, [0]).sample(1).interactions) == 0
    assert len(CinematicPulseGeometry(catalog, []).sample(1).leading) == 0


def renderer():
    return SimpleNamespace(
        renderer_id="pygfx",
        _gfx=gfx,
        _scene=gfx.Scene(),
        request_redraw=lambda: None,
        _apply_clipping_to_material=lambda m: None,
    )


def test_buffers_reused_and_toggle_scene_playback_cleanup(qtbot):
    viz = SimpleNamespace(renderer=renderer(), animation_running=False)
    service = ExperimentService(viz)
    service.window = ExperimentWindow()
    qtbot.addWidget(service.window)
    service.window.show()
    service.overlay = ExperimentOverlay(viz.renderer)
    service.catalog = _catalog([1, 2])
    service.window.cinematic_changed.connect(service._cinematic_changed)
    service.window.play_changed.connect(service.play)
    service._emphasize(np.array([0, 1]), "test")
    service.window.play.setChecked(True)
    assert service._clock.isActive()
    head = service.overlay.layers["cinematic_head"]
    position_buffer = head.geometry.positions
    service.clock(1)
    assert service.overlay.layers["cinematic_head"] is head
    assert head.geometry.positions is position_buffer
    service.set_presentation_playing(True)
    assert not service._clock.isActive()
    assert not service.window.play.isChecked()
    assert not service.window.play.isEnabled()
    assert not any(k.startswith("cinematic") for k in service.overlay.layers)
    service.set_presentation_playing(False)
    assert not service.window.play.isChecked()
    for _ in range(3):
        service._emphasize(np.array([0, 1]), "new frozen frame")
        service.window.play.setChecked(True)
        service.clear_overlays()
        assert not service._clock.isActive()
        assert not service.overlay.layers
        assert service._cinematic_geometry is None
    service.close()


def test_native_pygfx_cinematic_buffers_render(qapp):
    import os

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Set ORCHAV_RUN_PYGFX_TESTS=1 for native renderer smoke")
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    native = PygfxRenderer(SimpleNamespace())
    try:
        native.initialize_visualizer(width=128, height=128)
        catalog = _catalog([1], points=[[(0, 0, 0), (1, 0, 0), (1, 1, 0)]])
        catalog._canonical_data.itype[1] = 1
        geometry = CinematicPulseGeometry(catalog, [0])
        overlay = ExperimentOverlay(native)
        overlay.cinematic_routes(catalog, [0])
        for moment in (0, geometry.interaction_ns[0], geometry.extent):
            overlay.cinematic_pulses(geometry.sample(moment))
            native._renderer.render(native._scene, native._camera)
            assert native._renderer.snapshot().shape[-1] == 4
        overlay.clear()
        assert not overlay.layers
    finally:
        native.close()
