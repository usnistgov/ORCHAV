"""Visibility and content policy for the actual Qt HUD labels."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QWidget

from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
from visualizer.src.services.viewport_hud_service import AnalysisLegend
from visualizer.src.state import create_initial_state, update_state


@pytest.fixture
def renderer(qtbot):
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(app_state=create_initial_state())
    renderer._container = QWidget()
    renderer._container.resize(900, 600)
    qtbot.addWidget(renderer._container)
    renderer._hud_overlay_labels = {}
    renderer._hud_overlay_specs = {}
    renderer._tooltip_label = None
    renderer._clipping_planes = []
    renderer.last_frame_packet = None
    renderer.request_redraw = lambda: None
    return renderer


@pytest.mark.parametrize(
    "bounds,labels",
    [
        ((1e-12, 1e-6), ("1e-12", "1e-06")),
        ((5.25e-9, 1.25e-4), ("5.25e-09", "0.000125")),
        ((-120.25, -60.5), ("-120.25", "-60.50")),
        ((0, 1), ("0.00", "1.00")),
    ],
)
def test_analysis_legend_keeps_small_linear_endpoints_readable(renderer, bounds, labels):
    renderer.set_analysis_legend(
        AnalysisLegend(title="Coverage", colorbar_label="linear", colorbar_range=bounds)
    )
    html = renderer._hud_overlay_labels["analysis_legend"].text()
    assert all(label in html for label in labels)


@pytest.mark.parametrize("mode", ["compact", "detailed"])
def test_routine_settings_and_empty_paths_do_not_create_status(renderer, monkeypatch, mode):
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.overlays.runtime_status_chips", lambda *_: ()
    )
    renderer.visualizer.app_state = create_initial_state(
        viewport_hud_mode=mode, show_mpc_type_markers=True
    )
    renderer._aa_mode = "fxaa"
    renderer._ground_grid_visible = True
    renderer._depth_pass_enabled = True
    monkeypatch.setenv("ORCHAV_PYGFX_ENABLE_EDL", "1")
    renderer._update_status_chip_overlay(SimpleNamespace(mpc_lines=[], mpc_points=[]))
    assert "render_status" not in renderer._hud_overlay_labels

    renderer._clipping_planes = [object()]
    renderer._update_status_chip_overlay()
    assert "Cutaway" in renderer._hud_overlay_labels["render_status"].text()


def test_missing_time_never_creates_a_label_and_clears_previous_time(renderer):
    renderer.set_simulation_time_label("")
    assert "simulation_time" not in renderer._hud_overlay_labels

    renderer.set_simulation_time_label("Simulation time: 0.000000 s")
    label = renderer._hud_overlay_labels["simulation_time"]
    assert not label.isHidden()
    renderer.set_simulation_time_label("")
    assert label.isHidden()
    assert not renderer._hud_overlay_specs["simulation_time"]["visible"]


def test_map_attribution_survives_hidden_hud_until_map_is_removed(renderer):
    renderer.set_map_attribution("© OpenStreetMap contributors")
    label = renderer._hud_overlay_labels["map_attribution"]
    assert not label.isHidden()
    renderer.visualizer.app_state = update_state(
        renderer.visualizer.app_state, viewport_hud_enabled=False
    )
    renderer.refresh_viewport_hud()
    assert not label.isHidden()
    renderer.set_map_attribution("© OpenStreetMap contributors · provider")
    assert "provider" in label.text() and not label.isHidden()
    renderer.set_map_attribution("")
    assert label.isHidden()


def test_clock_updates_keep_the_existing_layout(renderer, monkeypatch):
    renderer.set_simulation_time_label("Simulation time: 0.000000 s")
    label = renderer._hud_overlay_labels["simulation_time"]

    def unexpected_layout():
        pytest.fail("A running clock must not reflow every HUD widget")

    monkeypatch.setattr(renderer, "_reposition_hud_overlays", unexpected_layout)
    renderer.set_simulation_time_label("Simulation time: 1.000000 s")
    assert label.text() == "Simulation time: 1.000000 s"


@pytest.mark.parametrize("field", ["viewport_hud_enabled", "viewport_hud_show_status"])
def test_paused_clock_restores_after_visibility_toggle(renderer, field):
    renderer._update_mpc_hud_overlays = lambda _packet: None
    renderer._update_trajectory_hud_overlay = lambda: None
    renderer._refresh_feature_hud_overlays = lambda: None
    renderer.set_simulation_time_label("Simulation time: 2.500000 s")
    label = renderer._hud_overlay_labels["simulation_time"]

    renderer.visualizer.app_state = update_state(renderer.visualizer.app_state, **{field: False})
    renderer.refresh_viewport_hud()
    assert label.isHidden()
    renderer.visualizer.app_state = update_state(renderer.visualizer.app_state, **{field: True})
    renderer.refresh_viewport_hud()
    assert not label.isHidden()
    assert label.text() == "Simulation time: 2.500000 s"

    renderer.visualizer.app_state = update_state(renderer.visualizer.app_state, **{field: False})
    renderer.set_simulation_time_label("")
    renderer.visualizer.app_state = update_state(renderer.visualizer.app_state, **{field: True})
    renderer.refresh_viewport_hud()
    assert label.isHidden()


def test_style_switch_restyles_existing_paused_labels_and_hover(renderer):
    def show_legend(_packet):
        renderer._set_hud_overlay(
            "example",
            html="<b>Path legend</b>",
            visible=True,
            role="legend",
            corner="top_right",
            priority=10,
        )

    renderer._update_mpc_hud_overlays = show_legend
    renderer._update_trajectory_hud_overlay = lambda: None
    renderer._refresh_feature_hud_overlays = lambda: None
    renderer._setup_tooltip()
    renderer.set_simulation_time_label("Simulation time: 0.000000 s")
    show_legend(None)
    clock = renderer._hud_overlay_labels["simulation_time"]
    legend = renderer._hud_overlay_labels["example"]
    appearances = set()
    for style in ("classic", "card", "minimal"):
        renderer.visualizer.app_state = update_state(
            renderer.visualizer.app_state, viewport_hud_style=style
        )
        renderer.refresh_viewport_hud()
        assert renderer._hud_overlay_labels["simulation_time"] is clock
        assert renderer._hud_overlay_labels["example"] is legend
        assert clock.text() == "Simulation time: 0.000000 s"
        assert legend.text() == "<b>Path legend</b>"
        assert renderer._hud_overlay_specs["example"]["style"] == style
        assert renderer._tooltip_label.styleSheet() == renderer._hud_overlay_stylesheet(
            "annotation"
        )
        appearances.add(legend.styleSheet())
    assert len(appearances) == 3
