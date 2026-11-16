"""Independent scientific and lifecycle boundaries for propagation options."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_pulses import renderer
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay


@pytest.fixture
def service(qtbot, monkeypatch):
    wall = [100.0]
    monkeypatch.setattr("visualizer.src.experiments.service.time.monotonic", lambda: wall[0])
    obj = ExperimentService(SimpleNamespace(renderer=renderer(), animation_running=False))
    obj.window = ExperimentWindow()
    qtbot.addWidget(obj.window)
    obj.window.show()
    obj.window.tabs.setTabEnabled(0, True)
    obj.window.tabs.setCurrentIndex(0)
    obj.window.presentation_options.expand()
    obj.overlay = ExperimentOverlay(obj.viz.renderer)
    obj.window.cinematic_changed.connect(obj._cinematic_changed)
    obj.window.play_changed.connect(obj.play)
    obj.window.interaction_mode_changed.connect(obj._interaction_mode_changed)
    obj.window.presentation_speed_changed.connect(obj._presentation_speed_changed)
    obj.window.presentation_style_changed.connect(obj._interaction_mode_changed)
    obj.window.restart_requested.connect(obj._restart_presentation)
    obj.window.interaction_step_requested.connect(obj._step_interaction)
    obj.catalog = _catalog([99], points=[[(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]])
    obj.catalog._canonical_data.itype[1:3] = [1, 2]
    obj._emphasize(np.array([0]), "review")
    obj.window.play.setChecked(True)
    yield obj, wall
    obj.close()


def test_reached_scrub_is_pure_and_preserves_recorded_data(service):
    obj, _ = service
    geometry = obj._cinematic_geometry
    before = obj.catalog.canonical_data.points.copy()
    first, second = geometry.interaction_ns
    samples = [
        geometry.sample(t, interaction_mode="reached") for t in (second + 1, first - 1, first)
    ]
    assert np.all(samples[0].interaction_colors[:, 3] > 0)
    assert np.all(samples[1].interaction_colors[:, 3] == 0)
    assert samples[2].interaction_colors[0, 3] > 0
    assert samples[2].interaction_colors[1, 3] == 0
    np.testing.assert_array_equal(before, obj.catalog.canonical_data.points)
    np.testing.assert_allclose(
        geometry.sample(-1, interaction_mode="all").interaction_colors[:, 3], [0.48, 0.48]
    )


def test_paused_mode_change_updates_same_native_buffers(service):
    obj, _ = service
    obj.play(False)
    obj.clock(obj._clock_end)
    layer = obj.overlay.layers["cinematic_flash"]
    positions = layer.geometry.positions
    colors = layer.geometry.colors
    obj.window.interaction_mode.setCurrentIndex(1)
    assert obj.overlay.layers["cinematic_flash"] is layer
    assert layer.geometry.positions is positions
    assert layer.geometry.colors is colors
    assert np.all(colors.data[:, 3] > 0)
    obj.clock(0)
    assert np.all(colors.data[:, 3] == 0)
    assert not obj._clock.isActive()


def test_changing_marker_style_does_not_discard_elapsed_time(service):
    obj, wall = service
    wall[0] += 1.0
    obj.window.interaction_mode.setCurrentIndex(1)
    assert obj._clock_time == pytest.approx(obj._clock_end / 5)
    wall[0] += 1.0
    obj._tick()
    assert obj._clock_time == pytest.approx(obj._clock_end * 2 / 5)


def test_pause_resume_does_not_count_paused_wall_time(service):
    obj, wall = service
    wall[0] += 0.5
    obj._tick()
    position = obj._clock_time
    obj.play(False)
    wall[0] += 100
    obj.play(True)
    wall[0] += 0.5
    obj._tick()
    assert obj._clock_time == pytest.approx(position + obj._clock_end / 10)
    obj.set_presentation_playing(True)
    assert not obj._clock.isActive()
    assert obj._clock_last_wall is None
    assert not obj.window.interaction_mode.isEnabled()
    assert not obj.window.presentation_speed.isEnabled()
    assert obj._cinematic_geometry is None


def test_scene_playback_disables_clock_until_paused_and_played(service):
    obj, wall = service
    obj.set_presentation_playing(True)
    assert not obj._clock.isActive()
    assert not obj.window.play.isEnabled()
    obj.set_presentation_playing(False)
    assert obj.window.play.isEnabled()
    assert not obj._clock.isActive()
    obj.window.play.setChecked(True)
    wall[0] += 1
    obj._tick()
    assert obj._clock_time == pytest.approx(obj._clock_end / 5)
    np.testing.assert_array_equal(obj._selected_population, [0])


def test_loop_wraps_elapsed_and_resets_reached_markers(service):
    obj, wall = service
    obj.window.interaction_mode.setCurrentIndex(1)
    obj.window.presentation_end.setCurrentIndex(obj.window.presentation_end.findData("loop"))
    wall[0] += 11.0  # Two cycles plus one second, even if a timer tick was delayed.
    obj._tick()
    assert obj._clock_time == pytest.approx(obj._clock_end / 5)
    assert obj._clock.isActive()
    assert np.all(obj.overlay.layers["cinematic_flash"].geometry.colors.data[:, 3] == 0)
    obj.clear_overlays()
    assert not obj._clock.isActive()
    assert not obj.overlay.layers
    assert obj._clock_last_wall is None


@pytest.mark.parametrize("end_mode", ["hold", "once"])
def test_end_modes_stop_timer_and_restart(service, end_mode):
    obj, wall = service
    obj.window.presentation_end.setCurrentIndex(obj.window.presentation_end.findData(end_mode))
    obj.window.interaction_mode.setCurrentIndex(1)
    halo = obj.overlay.layers["cinematic_halo"]
    wall[0] += 6
    obj._tick()
    assert obj._clock_time == obj._clock_end
    assert not obj._clock.isActive()
    assert not obj.window.play.isChecked()
    assert obj.overlay.layers["cinematic_halo"] is halo
    if end_mode == "once":
        assert "cinematic_flash" not in obj.overlay.layers
    else:
        assert np.all(obj.overlay.layers["cinematic_flash"].geometry.colors.data[:, 3] > 0)
    obj.window.restart.click()
    assert obj._clock_time == 0
    assert obj._clock.isActive()
    assert obj.window.play.isChecked()
    assert np.all(obj.overlay.layers["cinematic_flash"].geometry.colors.data[:, 3] == 0)
    assert obj.overlay.layers["cinematic_halo"] is halo


def test_interaction_buttons_pause_exact_time_and_handle_bounds(service):
    obj, _ = service
    first, second = obj._cinematic_geometry.interaction_ns
    obj.window.next_interaction.click()
    assert obj._clock_time == first
    assert not obj._clock.isActive()
    obj.window.next_interaction.click()
    assert obj._clock_time == second
    obj.window.next_interaction.click()
    assert obj._clock_time == second
    obj.window.previous_interaction.click()
    assert obj._clock_time == first
    obj.window.previous_interaction.click()
    assert obj._clock_time == first
    assert not obj._clock.isActive()


def test_appearance_updates_reuse_bounded_buffers_and_no_timer(service):
    obj, _ = service
    obj.play(False)
    obj.clock(obj._clock_end / 2)
    halo = obj.overlay.layers["cinematic_halo"]
    halo_buffer = halo.geometry.positions
    trails = obj.overlay.layers["cinematic_trail"]
    trail_buffer = trails.geometry.positions
    obj.window.trail_mode.setCurrentIndex(2)
    obj.window.halo_opacity.setValue(0)
    assert obj.overlay.layers["cinematic_trail"] is trails
    assert trails.geometry.positions is trail_buffer
    assert obj.overlay.layers["cinematic_halo"] is halo
    assert halo.geometry.positions is halo_buffer
    assert halo.material.color.a == 0
    obj.window.halo_opacity.setValue(0.30)
    assert halo.material.color.a == pytest.approx(0.30)
    assert not obj._clock.isActive()


def test_native_renderer_all_options_draw_and_release(qapp):
    import os

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Native renderer opt-in")
    from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    native = PygfxRenderer(SimpleNamespace())
    try:
        native.initialize_visualizer(width=128, height=128)
        catalog = _catalog([1], points=[[(0, 0, 0), (1, 0, 0), (1, 1, 0)]])
        catalog._canonical_data.itype[1] = 1
        geometry = CinematicPulseGeometry(catalog, [0])
        overlay = ExperimentOverlay(native)
        overlay.cinematic_routes(catalog, [0])
        for mode in ("flash", "reached", "all", "dwell"):
            for trail in ("short", "medium", "long"):
                for time in (0, geometry.interaction_ns[0], geometry.extent, 0):
                    overlay.cinematic_pulses(
                        geometry.sample(time, interaction_mode=mode, trail_mode=trail),
                        interaction_types=geometry.interaction_types,
                        type_markers=True,
                        label_specs=((0, "P0 · I1 · Specular"),),
                    )
                    overlay.set_cinematic_halo_opacity(0.30)
                    native._renderer.render(native._scene, native._camera)
                    assert native._renderer.snapshot().shape[-1] == 4
        overlay.clear_cinematic_pulses()
        assert set(overlay.layers) == {"cinematic_halo"}
        overlay.clear()
        assert not native._presentation_materials
    finally:
        native.close()


def test_paused_glyph_labels_dwell_toggle_preserves_buffers_and_cleans(service):
    import pygfx as gfx

    obj, _ = service
    obj.play(False)
    obj.viz.app_state = SimpleNamespace(show_mpc_type_markers=True)
    obj.window.interaction_labels.setChecked(True)
    obj.window.interaction_mode.setCurrentIndex(obj.window.interaction_mode.findData("dwell"))
    obj.window.interaction_dwell.setValue(0.1)
    g = obj._cinematic_geometry
    t = g.interaction_ns[0]
    obj.clock(t)
    flash = obj.overlay.layers["cinematic_flash"]
    buffer = flash.geometry.positions
    assert isinstance(flash.material, gfx.PointsMarkerMaterial)
    first = obj.overlay.layers["cinematic_label_0"]
    assert first.visible
    assert not obj.overlay.layers["cinematic_label_1"].visible
    obj.clock(t + g.extent * (0.1 / 5 + 0.036))
    assert not first.visible
    obj.clock(t)
    assert obj.overlay.layers["cinematic_label_0"] is first
    assert first.visible
    obj.viz.app_state.show_mpc_type_markers = False
    obj._interaction_mode_changed()
    assert not isinstance(flash.material, gfx.PointsMarkerMaterial)
    assert flash.geometry.positions is buffer
    obj.window.interaction_labels.setChecked(False)
    assert not any(k.startswith("cinematic_label_") for k in obj.overlay.layers)
    assert not obj._clock.isActive()
    obj.clear_overlays()
    assert not any(k.startswith("cinematic_") for k in obj.overlay.layers)


def test_existing_marker_refresh_hook_without_bulk_packet_updates_same_frame(service):
    import pygfx as gfx

    from visualizer.src.renderers.pygfx.mpc import PygfxMpcMixin

    obj, _ = service
    obj.play(False)
    obj.viz.app_state = SimpleNamespace(show_mpc_type_markers=False)
    obj.viz.extension_services = {"experiments": obj}
    renderer = obj.viz.renderer
    renderer.visualizer = obj.viz
    renderer.last_frame_packet = None
    renderer._set_marker_legend_visible = lambda visible: None
    before = obj.overlay.layers["cinematic_flash"].geometry.positions
    obj.viz.app_state.show_mpc_type_markers = True
    PygfxMpcMixin.refresh_mpc_point_markers(renderer)
    flash = obj.overlay.layers["cinematic_flash"]
    assert isinstance(flash.material, gfx.PointsMarkerMaterial)
    assert flash.geometry.positions is before
    assert not obj._clock.isActive()
    obj.clear_overlays()
    PygfxMpcMixin.refresh_mpc_point_markers(renderer)
    assert not obj.overlay.layers


def test_same_frame_color_mutation_preserves_buffers_and_uses_canonical_rgb(service):
    from tests.visualizer.unit.test_cinematic_appearance_review import core
    from visualizer.src.experiments.cinematic_appearance import interaction_colors

    obj, _ = service
    obj.play(False)
    obj.viz.app_state = SimpleNamespace(color_mode="reflection_order")
    obj.viz.mpc_core = core()
    obj.window.match_mpc_colors.setChecked(True)
    obj.clock(obj._clock_end / 2)
    flash = obj.overlay.layers["cinematic_flash"]
    positions, colors = flash.geometry.positions, flash.geometry.colors
    for mode in ("delay", "mpc_type", "path_loss", "reflection_order", "material"):
        obj.viz.app_state.color_mode = mode
        obj.refresh_cinematic_style()
        expected, _ = interaction_colors(
            obj.catalog, obj._cinematic_geometry.interaction_point_indices, obj.viz.mpc_core, mode
        )
        np.testing.assert_allclose(colors.data[:, :3], expected)
        assert flash.geometry.positions is positions
        assert flash.geometry.colors is colors
        assert not obj._clock.isActive()


def test_labels_on_top_but_markers_depth_tested_and_both_clipped(service):
    obj, _ = service
    obj.play(False)
    plane = [(0.0, 0.0, 1.0, -0.5)]
    obj.viz.renderer._apply_clipping_to_material = lambda material: setattr(
        material, "clipping_planes", plane
    )
    obj.viz.app_state = SimpleNamespace(show_mpc_type_markers=True)
    obj.window.interaction_labels.setChecked(True)
    obj.clock(obj._cinematic_geometry.interaction_ns[0])
    label = obj.overlay.layers["cinematic_label_0"]
    marker = obj.overlay.layers["cinematic_flash"]
    assert not label.material.depth_test
    assert label.material.render_queue == 4100
    assert marker.material.depth_test
    np.testing.assert_allclose(label.material.clipping_planes, plane)
    np.testing.assert_allclose(marker.material.clipping_planes, plane)


def test_typed_edge_and_label_alpha_follow_dwell_and_reuse_buffers(service):
    obj, _ = service
    obj.play(False)
    obj.viz.app_state = SimpleNamespace(show_mpc_type_markers=True)
    obj.window.interaction_labels.setChecked(True)
    obj.window.interaction_mode.setCurrentIndex(obj.window.interaction_mode.findData("dwell"))
    obj.window.interaction_dwell.setValue(0.1)
    g = obj._cinematic_geometry
    arrival = g.interaction_ns[0]
    hold = g.extent * 0.1 / 5
    fade = g.extent * 0.035
    obj.clock(arrival - 0.001)
    edge_buffer = obj.overlay.layers["cinematic_flash"].geometry.edge_colors
    for moment, expected in (
        (arrival - 0.001, 0),
        (arrival, 0.85),
        (arrival + hold + fade / 2, 0.425),
        (arrival + hold + fade + 0.001, 0),
    ):
        obj.clock(moment)
        flash = obj.overlay.layers["cinematic_flash"]
        assert flash.geometry.edge_colors is edge_buffer
        assert flash.material.edge_color_mode == "vertex"
        np.testing.assert_allclose(edge_buffer.data[:, 3], flash.geometry.colors.data[:, 3])
        assert edge_buffer.data[0, 3] == pytest.approx(expected)
        label = obj.overlay.layers["cinematic_label_0"]
        assert label.material.opacity == pytest.approx(expected)
        assert label.visible == (expected > 0)


def test_native_zero_alpha_typed_glyphs_leave_no_ghost_pixels(qapp):
    import os

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Native renderer opt-in")
    import pygfx as gfx

    from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    native = PygfxRenderer(SimpleNamespace())
    try:
        native.initialize_visualizer(width=128, height=128)
        native._scene.clear()
        native._scene.add(gfx.Background(None, gfx.BackgroundMaterial("#101820")))
        camera = gfx.OrthographicCamera(4, 4)
        camera.local.z = 5

        def snapshot():
            native._renderer.render(native._scene, camera)
            return native._renderer.snapshot().copy()

        empty = snapshot()
        catalog = _catalog([1], points=[[(-1, 0, 0), (0, 0, 0), (1, 0, 0)]])
        catalog.canonical_data.itype[1] = 1
        geometry = CinematicPulseGeometry(catalog, [0])
        overlay = ExperimentOverlay(native)
        arrival = geometry.interaction_ns[0]
        for moment, visible in ((0, False), (arrival, True), (geometry.extent * 2, False)):
            sample = geometry.sample(moment, interaction_mode="flash")
            sample.leading_colors[:, 3] = 0
            sample.trail_colors[:, 3] = 0
            overlay.cinematic_pulses(
                sample,
                interaction_types=geometry.interaction_types,
                type_markers=True,
                label_specs=((0, "P0 · I1"),),
            )
            pixels = snapshot()
            if visible:
                assert np.any(pixels != empty)
            else:
                np.testing.assert_array_equal(pixels, empty)
        overlay.clear()
    finally:
        native.close()
