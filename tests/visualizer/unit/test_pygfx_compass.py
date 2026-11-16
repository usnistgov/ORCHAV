"""Compass heading stability, HUD ownership, caching and export composition."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from visualizer.src.renderers.pygfx import compass
from visualizer.src.renderers.pygfx.capture import PygfxCaptureMixin
from visualizer.src.renderers.pygfx.compass import PygfxCompassMixin, north_heading
from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin


class CompassRenderer(PygfxCompassMixin, PygfxOverlayMixin, PygfxCaptureMixin):
    def __init__(self, container):
        self._container = container
        self._hud_overlay_specs = {}
        self._hud_overlay_labels = {}
        self._camera = SimpleNamespace(world=SimpleNamespace(matrix=np.eye(4)))
        self.visualizer = SimpleNamespace(app_state=SimpleNamespace(viewport_hud_enabled=True))
        self.request_redraw = Mock()
        self.configure_compass((0, 1))


@pytest.fixture
def renderer(qtbot):
    container = QWidget()
    container.resize(500, 400)
    qtbot.addWidget(container)
    container.show()
    yield CompassRenderer(container)


@pytest.mark.parametrize("camera_class", ["PerspectiveCamera", "OrthographicCamera"])
@pytest.mark.parametrize("height", [0, 0.00001, 20, 1000000])
def test_north_heading_stays_stable_at_horizon_and_overhead(camera_class, height):
    gfx = pytest.importorskip("pygfx")
    camera = getattr(gfx, camera_class)()
    camera.world.reference_up = (0, 0, 1)
    camera.local.position = (0, -10, height)
    camera.look_at((0, 0, 0))
    assert north_heading(camera.world.matrix) == pytest.approx(0, abs=1e-6)
    camera.local.position = (-10, 0, height)
    camera.look_at((0, 0, 0))
    assert north_heading(camera.world.matrix) == pytest.approx(-np.pi / 2, abs=1e-6)


def test_degenerate_and_rolled_camera_inputs_are_bounded():
    assert north_heading(np.zeros((4, 4))) is None
    assert north_heading(np.full((4, 4), np.nan)) is None
    assert north_heading(np.eye(3)) is None
    rolled = np.eye(4)
    rolled[:3, :3] = [[0, -1, 0], [0, 0, 1], [1, 0, 0]]
    assert north_heading(rolled) == pytest.approx(np.pi / 2)
    assert north_heading(np.eye(4), (1, 1)) == pytest.approx(np.pi / 4)


def test_only_geographic_sources_enable_compass_and_source_clear_retires_it(renderer):
    renderer.configure_compass(None, reason="This Cartesian scene has no geographic mapping.")
    assert renderer.get_compass_state() == {
        "available": False,
        "enabled": False,
        "label": "Geographic north at scene origin",
        "reason": "This Cartesian scene has no geographic mapping.",
    }
    assert not renderer.set_compass_visible(True)
    assert not renderer._hud_overlay_labels
    renderer.configure_compass((0.02, 1), label="Geographic north at scene origin")
    assert renderer.get_compass_state()["available"]
    assert renderer.set_compass_visible(True)
    assert renderer.get_compass_state()["enabled"]
    assert renderer.set_compass_visible(False)
    assert renderer.get_compass_state()["available"]
    assert renderer.set_compass_visible(True)
    renderer.configure_compass((1, 0))  # Another geographic scene still starts off.
    assert not renderer.get_compass_visible() and not renderer._hud_overlay_labels
    assert renderer.set_compass_visible(True)
    renderer._clear_compass()
    assert not renderer.get_compass_state()["available"]
    assert not renderer.get_compass_state()["enabled"]
    assert not renderer._hud_overlay_labels
    assert not renderer.set_compass_visible(True)


@pytest.mark.parametrize("direction", [(0, 0), (float("nan"), 1), (1, 2, 3)])
def test_invalid_compass_configuration_does_not_destroy_current_source(renderer, direction):
    renderer.set_compass_visible(True)
    with pytest.raises(ValueError, match="Compass north"):
        renderer.configure_compass(direction)
    assert renderer.get_compass_state()["available"] and renderer.get_compass_visible()


def test_off_has_no_widget_work_and_unchanged_view_reuses_pixmap(renderer, monkeypatch):
    paint = Mock(wraps=compass._compass_pixmap)
    monkeypatch.setattr(compass, "_compass_pixmap", paint)
    renderer._sync_compass()
    assert not renderer._hud_overlay_labels
    paint.assert_not_called()
    assert renderer.set_compass_visible(True)
    label = renderer._hud_overlay_labels["compass"]
    assert label.isVisible() and label.testAttribute(Qt.WA_TransparentForMouseEvents)
    first_key = label.pixmap().cacheKey()
    redraws = renderer.request_redraw.call_count
    for _ in range(5):
        renderer._camera.world.matrix[:3, 3] += 1  # Translation is not a heading change.
        renderer._sync_compass()
    paint.assert_called_once()
    assert label.pixmap().cacheKey() == first_key
    assert renderer.request_redraw.call_count == redraws
    renderer._camera.world.matrix[:2, :2] = [[0, -1], [1, 0]]
    renderer._sync_compass()
    assert paint.call_count == 2 and label.pixmap().cacheKey() != first_key
    assert renderer.request_redraw.call_count == redraws  # QWidget update, no extra GPU draw.
    assert renderer.set_compass_visible(False)
    assert not renderer.get_compass_visible()
    assert "compass" not in renderer._hud_overlay_labels
    assert "compass" not in renderer._hud_overlay_specs
    assert renderer._compass_signature is None
    renderer._sync_compass()
    assert paint.call_count == 2


def test_master_hud_roundtrip_keeps_compass_intent_and_restores_widget(renderer):
    renderer.visualizer.app_state.viewport_hud_show_legends = False
    renderer.visualizer.app_state.viewport_hud_show_annotations = False
    renderer.set_compass_visible(True)
    label = renderer._hud_overlay_labels["compass"]
    renderer.visualizer.app_state.viewport_hud_enabled = False
    renderer._sync_compass()
    assert renderer.get_compass_visible() and not label.isVisible()
    renderer.visualizer.app_state.viewport_hud_enabled = True
    renderer._sync_compass()
    assert renderer._hud_overlay_labels["compass"] is label
    assert label.isVisible() and renderer._hud_overlay_specs["compass"]["visible"]


def test_compass_uses_hud_style_layout_and_capture(renderer, monkeypatch):
    renderer._set_hud_overlay(
        "legend",
        html="Range / Doppler",
        visible=True,
        role="legend",
        corner="top_right",
        priority=10,
    )
    renderer.set_compass_visible(True)
    label = renderer._hud_overlay_labels["compass"]
    legend = renderer._hud_overlay_labels["legend"]
    assert label.geometry().bottom() < legend.geometry().top()
    assert label.width() <= 100 and label.height() <= 100
    first = label.pixmap().cacheKey()
    renderer.visualizer.app_state.viewport_hud_style = "card"
    renderer._sync_compass()
    assert label.pixmap().cacheKey() != first
    assert renderer._hud_overlay_specs["compass"]["style"] == "card"
    assert label.geometry().bottom() < legend.geometry().top()
    # The ordinary capture compositor consumes the same registered QLabel.
    base = np.zeros((400, 500, 3), dtype=np.uint8)
    captured = renderer._composite_visible_qt_overlays(base)
    area = captured[label.y() : label.y() + label.height(), label.x() : label.x() + label.width()]
    assert area.max() > 0
    renderer._clear_compass()
    without = renderer._composite_visible_qt_overlays(base)
    assert not np.array_equal(captured, without)
