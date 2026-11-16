"""Presentation-only marker visibility and elapsed cinematic pacing."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_pulses import renderer
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay


def catalog():
    result = _catalog(
        [10, 20], points=[[(0, 0, 0), (3, 0, 0), (3, 4, 0)], [(0, 0, 0), (0, 8, 0), (3, 8, 0)]]
    )
    result._canonical_data.itype[1] = 1
    result._canonical_data.itype[4] = 2
    return result


@pytest.fixture
def service(qtbot, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("visualizer.src.experiments.service.time.monotonic", lambda: now[0])
    obj = ExperimentService(SimpleNamespace(renderer=renderer(), animation_running=False))
    obj.window = ExperimentWindow()
    qtbot.addWidget(obj.window)
    obj.window.show()
    obj.window.tabs.setTabEnabled(0, True)
    obj.overlay = ExperimentOverlay(obj.viz.renderer)
    obj.catalog = catalog()
    obj.window.cinematic_changed.connect(obj._cinematic_changed)
    obj.window.interaction_mode_changed.connect(obj._interaction_mode_changed)
    obj.window.presentation_speed_changed.connect(obj._presentation_speed_changed)
    obj.window.presentation_style_changed.connect(obj._interaction_mode_changed)
    obj.window.restart_requested.connect(obj._restart_presentation)
    obj.window.interaction_step_requested.connect(obj._step_interaction)
    obj.window.play_changed.connect(obj.play)
    obj.window.clock_changed.connect(obj.clock)
    obj._emphasize(np.array([0, 1]), "selected paths")
    yield obj, now
    obj.close()


def test_marker_modes_are_subset_only_and_reversible():
    geometry = CinematicPulseGeometry(catalog(), [0])
    assert len(geometry.vertices) == 1
    bounce = geometry.interaction_ns[0]
    late = geometry.extent + 100
    assert geometry.sample(late).interaction_colors[0, 3] == 0
    assert geometry.sample(late, interaction_mode="reached").interaction_colors[
        0, 3
    ] == pytest.approx(0.48)
    assert geometry.sample(bounce - 0.01, interaction_mode="reached").interaction_colors[0, 3] == 0
    assert geometry.sample(0, interaction_mode="all").interaction_colors[0, 3] == pytest.approx(
        0.48
    )
    for mode in ("flash", "reached", "all"):
        np.testing.assert_array_equal(
            geometry.sample(10, interaction_mode=mode).leading, geometry.sample(10).leading
        )
    with pytest.raises(ValueError):
        geometry.sample(0, interaction_mode="made up")


@pytest.mark.parametrize("speed", [0.25, 0.5, 1.0, 2.0])
def test_cinematic_speed_uses_elapsed_time_not_tick_count(service, speed):
    obj, now = service
    obj.window.presentation_speed.setCurrentIndex(obj.window.presentation_speed.findData(speed))
    obj.window.play.setChecked(True)
    end = obj._clock_end
    now[0] += 0.7
    obj._tick()
    assert obj._clock_time == pytest.approx(end * 0.7 * speed / 5)
    now[0] += 0.3
    obj._tick()
    assert obj._clock_time == pytest.approx(end * speed / 5)
    assert obj._clock_end == end


def test_speed_changes_use_old_rate_until_change_and_pause_skips_elapsed(service):
    obj, now = service
    obj.window.play.setChecked(True)
    end = obj._clock_end
    now[0] += 1
    obj.window.presentation_speed.setCurrentIndex(3)
    assert obj._clock_time == pytest.approx(end / 5)
    now[0] += 0.5
    obj._tick()
    assert obj._clock_time == pytest.approx(end * 0.4)
    obj.window.play.setChecked(False)
    now[0] += 100
    obj.window.play.setChecked(True)
    now[0] += 0.25
    obj._tick()
    assert obj._clock_time == pytest.approx(end * 0.5)


def test_mode_change_reuses_buffers_and_preserves_clock(service):
    obj, now = service
    obj.window.play.setChecked(True)
    obj.window.play.setChecked(False)
    obj.clock(obj._clock_end)
    layer = obj.overlay.layers["cinematic_flash"]
    buffer = layer.geometry.positions
    geometry = obj._cinematic_geometry
    obj.window.interaction_mode.setCurrentIndex(1)
    assert obj.overlay.layers["cinematic_flash"] is layer
    assert layer.geometry.positions is buffer
    assert obj._cinematic_geometry is geometry
    assert obj._clock_time == obj._clock_end
    assert not obj._clock.isActive()
    assert np.all(layer.geometry.colors.data[:, 3] > 0)
    obj.clock(0)
    assert not np.any(layer.geometry.colors.data[:, 3])
    obj.window.interaction_mode.setCurrentIndex(2)
    assert np.all(layer.geometry.colors.data[:, 3] > 0)
    obj.clear_overlays()
    assert not obj.overlay.layers
    assert not obj._clock.isActive()


def test_cinematic_options_are_available_without_a_checkbox(service):
    obj, now = service
    assert not hasattr(obj.window, "cinematic")
    assert obj.window.interaction_mode.currentData() == "flash"
    assert obj.window.presentation_speed.currentData() == 1
    assert obj.window.interaction_mode.isEnabled()
    assert obj.window.presentation_speed.isEnabled()
    assert obj._cinematic_geometry is not None
    assert not obj._clock.isActive()
    obj.window.play.setChecked(True)
    now[0] += 1
    obj._tick()
    assert obj._clock_time == pytest.approx(obj._clock_end / 5)


def test_seek_restart_and_playback_cancellation(service):
    obj, now = service
    obj.window.play.setChecked(True)
    end = obj._clock_end
    now[0] += 1
    obj.clock(end * 0.1)
    now[0] += 0.5
    obj._tick()
    assert obj._clock_time == pytest.approx(end * 0.2)
    now[0] += 50
    obj._tick()
    assert obj._clock_time == end
    assert not obj.window.play.isChecked()
    assert not obj._clock.isActive()
    obj.window.play.setChecked(True)
    now[0] += 0.5
    obj._tick()
    assert obj._clock_time == pytest.approx(end * 0.1)
    obj.set_presentation_playing(True)
    assert not obj._clock.isActive()
    assert not obj.window.play.isChecked()
    assert not obj.window.interaction_mode.isEnabled()
    assert not obj.window.presentation_speed.isEnabled()


@pytest.mark.parametrize("mode", ["hold", "once", "loop"])
def test_end_modes_and_restart(service, mode):
    obj, now = service
    obj.window.presentation_end.setCurrentIndex(obj.window.presentation_end.findData(mode))
    obj.window.play.setChecked(True)
    end = obj._clock_end
    now[0] += 5.5
    obj._tick()
    if mode == "loop":
        assert obj._clock_time == pytest.approx(end * 0.1)
        assert obj._clock.isActive()
    else:
        assert obj._clock_time == end
        assert not obj._clock.isActive()
        assert ("cinematic_head" in obj.overlay.layers) == (mode == "hold")
        assert "cinematic_halo" in obj.overlay.layers
    obj.window.restart.click()
    assert obj._clock_time == 0
    assert obj._clock.isActive()
    assert "cinematic_head" in obj.overlay.layers


def test_steps_pause_at_known_vertices_and_rewind_reached(service):
    obj, now = service
    obj.window.play.setChecked(True)
    obj.window.interaction_mode.setCurrentIndex(1)
    times = sorted(obj._cinematic_geometry.interaction_ns)
    obj.window.next_interaction.click()
    assert not obj._clock.isActive()
    assert obj._clock_time == pytest.approx(times[0])
    obj.window.next_interaction.click()
    assert obj._clock_time == pytest.approx(times[1])
    obj.window.previous_interaction.click()
    assert obj._clock_time == pytest.approx(times[0])
    assert np.count_nonzero(obj.overlay.layers["cinematic_flash"].geometry.colors.data[:, 3]) == 1
    obj.window.previous_interaction.click()
    assert obj._clock_time == pytest.approx(times[0])


def test_appearance_options_reuse_routes_and_buffers(service):
    obj, now = service
    obj.window.play.setChecked(True)
    obj.window.play.setChecked(False)
    obj.clock(obj._clock_end * 0.7)
    halo = obj.overlay.layers["cinematic_halo"]
    trail = obj.overlay.layers["cinematic_trail"]
    buffer = trail.geometry.positions
    positions = buffer.data.copy()
    obj.window.trail_mode.setCurrentIndex(2)
    assert trail is obj.overlay.layers["cinematic_trail"]
    assert buffer is trail.geometry.positions
    assert not np.array_equal(positions, buffer.data)
    obj.window.halo_opacity.setValue(0.02)
    assert halo is obj.overlay.layers["cinematic_halo"]
    assert halo.material.color.a == pytest.approx(0.02)
    assert obj._clock_time == pytest.approx(obj._clock_end * 0.7)
    assert not obj._clock.isActive()
    assert obj.window.presentation_options.is_expanded() is False
