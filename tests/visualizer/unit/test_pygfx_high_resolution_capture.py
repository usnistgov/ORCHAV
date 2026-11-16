"""Native GPU checks for resolution, frozen callouts, tiling and restoration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

gfx = pytest.importorskip("pygfx")
from rendercanvas.offscreen import RenderCanvas

from visualizer.src.renderers.pygfx.callouts import sync_callout
from visualizer.src.renderers.pygfx.capture import PygfxCaptureMixin
from visualizer.src.renderers.pygfx.high_resolution import ScreenshotCancelled, _render_view
from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin

pytestmark = pytest.mark.pygfx_runtime


class CaptureOwner(PygfxCaptureMixin, PygfxLabelMixin):
    def __init__(self):
        self._gfx = gfx
        self._initialized = True
        self._canvas = RenderCanvas(size=(320, 240), pixel_ratio=1)
        self._renderer = gfx.WgpuRenderer(self._canvas, pixel_scale=1)
        self._renderer.effect_passes = ()
        self._scene = gfx.Scene()
        self._scene.add(gfx.Background.from_color("#ddd"))
        self._camera = gfx.OrthographicCamera(8, 6)
        self._camera.local.z = 10
        self._objects = {}
        self._minimap_enabled = False
        self.layout_calls = 0
        for index in range(25):
            mesh = gfx.Mesh(gfx.box_geometry(0.012, 3, 0.1), gfx.MeshBasicMaterial(color="#222"))
            mesh.local.x = -3 + index * 0.25
            self._scene.add(mesh)

    def _update_label_backgrounds(self):
        self.layout_calls += 1
        super()._update_label_backgrounds()

    def add_callout(self, name, position):
        native = gfx.Text(
            text=name,
            font_size=16,
            screen_space=True,
            material=gfx.TextMaterial(
                color="#111", aa=True, alpha_mode="blend", depth_test=False, depth_write=False
            ),
            render_order=31000,
        )
        native.local.position = position
        self._objects[name] = native
        self._scene.add(native)
        self._sync_label_background(name, (1, 1, 1, 0.95), connector_origin=position)
        sync_callout(
            self,
            name,
            dict(
                base_position=position,
                offset_x=24,
                offset_y=20,
                marker_size=12,
                on_top=True,
                leader=True,
            ),
            {"label_background_rgba": (1, 1, 1, 0.95)},
        )
        return native


@pytest.fixture
def owner():
    result = CaptureOwner()
    yield result
    result._canvas.close()


def camera_snapshot(camera):
    return {
        "state": camera.get_state(),
        "matrix": camera.projection_matrix.copy(),
        "size": camera._view_size,
        "offset": camera._view_offset,
    }


def assert_camera_unchanged(camera, snapshot):
    for key, value in snapshot["state"].items():
        np.testing.assert_equal(camera.get_state()[key], value)
    np.testing.assert_array_equal(camera.projection_matrix, snapshot["matrix"])
    assert camera._view_size == snapshot["size"]
    assert camera._view_offset == snapshot["offset"]


def test_dimensions_native_detail_and_default_compatibility(owner, monkeypatch):
    plain = owner.export_screenshot_to_array()
    high = owner.export_screenshot_to_array(4)
    assert plain.shape == (240, 320, 3)
    assert high.shape == (960, 1280, 3)
    enlarged = np.asarray(Image.fromarray(plain).resize((1280, 960), Image.Resampling.BILINEAR))
    assert np.mean(np.abs(high.astype(float) - enlarged.astype(float))) > 0.2
    monkeypatch.setattr(owner, "_capture_canvas_frame", lambda **kw: plain)
    np.testing.assert_array_equal(owner.export_screenshot_to_array(), plain)
    np.testing.assert_array_equal(owner.export_screenshot_to_array(0), plain)
    assert owner.export_screenshot_to_array(0.5).shape == (120, 160, 3)
    with pytest.raises(ValueError, match="finite"):
        owner.export_screenshot_to_array(float("inf"))


def test_tiles_freeze_full_image_callouts_and_restore_state(owner):
    labels = [owner.add_callout(f"Rx {index}", (index * 0.15 - 0.2, 0, 0)) for index in range(3)]
    owner.export_screenshot_to_array()
    original = camera_snapshot(owner._camera)
    canvas_size = owner._canvas.get_physical_size()
    renderer_id = id(owner._renderer)
    full = owner.export_screenshot_to_array(4)
    positions = [label.world.position.copy() for label in labels]
    calls = owner.layout_calls
    tiled = owner.export_screenshot_to_array(4, max_tile_size=256)
    assert owner.layout_calls == calls + 1
    np.testing.assert_array_equal(tiled, full)
    for label, position in zip(labels, positions):
        np.testing.assert_array_equal(label.world.position, position)
    assert_camera_unchanged(owner._camera, original)
    assert owner._canvas.get_physical_size() == canvas_size
    assert id(owner._renderer) == renderer_id


def test_restores_camera_on_mid_export_cancel_and_render_failure(owner, monkeypatch):
    owner.export_screenshot_to_array()
    owner._camera.set_view_offset(400, 300, 40, 30, 320, 240)
    original = camera_snapshot(owner._camera)
    count = 0

    def cancel():
        nonlocal count
        count += 1
        return count == 3

    with pytest.raises(ScreenshotCancelled):
        owner.export_screenshot_to_array(4, max_tile_size=256, cancel_check=cancel)
    assert_camera_unchanged(owner._camera, original)

    def broken(*args, **kwargs):
        raise RuntimeError("injected renderer failure")

    monkeypatch.setattr(gfx, "WgpuRenderer", broken)
    with pytest.raises(RuntimeError, match="injected"):
        owner.export_screenshot_to_array(4)
    assert_camera_unchanged(owner._camera, original)


def test_new_scene_content_is_not_stale_after_repeated_export(owner):
    first = owner.export_screenshot_to_array(2)
    owner._scene.add(gfx.Mesh(gfx.box_geometry(2, 2, 0.2), gfx.MeshBasicMaterial(color="red")))
    second = owner.export_screenshot_to_array(2)
    assert np.count_nonzero(first != second) > 1000


def test_different_camera_and_existing_subview_are_preserved(owner):
    owner._camera = gfx.PerspectiveCamera(40)
    owner._camera.local.z = 10
    owner._camera.set_view_offset(400, 300, 40, 30, 320, 240)
    owner.export_screenshot_to_array()
    original = camera_snapshot(owner._camera)
    full = owner.export_screenshot_to_array(2)
    tiled = owner.export_screenshot_to_array(2, max_tile_size=256)
    np.testing.assert_array_equal(full, tiled)
    assert_camera_unchanged(owner._camera, original)


@pytest.mark.parametrize("effect_class", ["FXAAPass", "DDAAPass"])
def test_aa_overlap_and_nonlocal_effect_limit(owner, effect_class):
    owner.add_callout("Rx scientific label", (0, 0, 0))
    owner._renderer.effect_passes = (getattr(gfx.renderers.wgpu, effect_class)(),)
    effect = owner._renderer.effect_passes[0]
    full = owner.export_screenshot_to_array(2)
    tiled = owner.export_screenshot_to_array(2, max_tile_size=256)
    assert np.mean(np.abs(full.astype(float) - tiled.astype(float))) < 0.02
    assert owner._renderer.effect_passes == (effect,)
    owner._renderer.effect_passes = (gfx.renderers.wgpu.PhysicalBasedBloomPass(),)
    with pytest.raises(ValueError, match="Tiled export cannot preserve"):
        owner.export_screenshot_to_array(2, max_tile_size=256)


def test_hud_composited_once_at_final_resolution(owner, monkeypatch):
    calls = []

    def composite(frame):
        calls.append(frame.shape)
        return frame

    monkeypatch.setattr(owner, "_composite_visible_qt_overlays", composite)
    owner.export_screenshot_to_array(2, include_hud=True, max_tile_size=256)
    assert calls == [(480, 640, 3)]


@pytest.mark.parametrize("scale", [1, 2])
def test_map_attribution_is_exported_without_optional_hud(owner, monkeypatch, scale):
    calls = []
    owner._hud_overlay_specs = {"map_attribution": {"visible": True}}

    def composite(frame, *, attribution_only=False):
        calls.append((frame.shape, attribution_only))
        return frame

    monkeypatch.setattr(owner, "_composite_visible_qt_overlays", composite)
    owner.export_screenshot_to_array(scale, include_hud=False, max_tile_size=256)
    assert calls == [((240 * scale, 320 * scale, 3), True)]


def test_screen_coordinate_overlay_keeps_full_logical_coordinates(owner):
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#ddd"))
    text = gfx.Text(
        text="Minimap",
        font_size=16,
        screen_space=True,
        material=gfx.TextMaterial(color="black"),
        anchor="top-left",
    )
    text.local.position = (20, 20, 0)
    scene.add(text)
    camera = gfx.ScreenCoordsCamera(invert_y=True)
    full = _render_view(
        owner,
        scene,
        camera,
        output_size=(640, 480),
        logical_size=(320, 240),
        max_tile_size=None,
        cancel_check=None,
    )
    tiled = _render_view(
        owner,
        scene,
        camera,
        output_size=(640, 480),
        logical_size=(320, 240),
        max_tile_size=256,
        cancel_check=None,
    )
    np.testing.assert_array_equal(full, tiled)
    assert np.any(full[30:100, 30:200] < 100)


def test_qt_legend_is_rerendered_at_output_resolution(owner, qtbot):
    from PySide6.QtWidgets import QLabel, QWidget

    container = QWidget()
    container.resize(320, 240)
    legend = QLabel("Power (dBm)\n−100      −60", container)
    legend.setStyleSheet("background: white; color: black; font-size: 14px;")
    legend.setGeometry(12, 12, 130, 50)
    qtbot.addWidget(container)
    container.show()
    owner._container = container
    owner._hud_overlay_labels = {"power": legend}
    clean = owner.export_screenshot_to_array(2)
    with_hud = owner.export_screenshot_to_array(2, include_hud=True, max_tile_size=256)
    assert np.any(clean[24:124, 24:284] != with_hud[24:124, 24:284])
    np.testing.assert_array_equal(clean[130:], with_hud[130:])
    small = owner._render_qt_widget_rgba(legend, width=130, height=50)
    large = owner._render_qt_widget_rgba(legend, width=260, height=100)
    enlarged = np.asarray(Image.fromarray(small).resize((260, 100), Image.Resampling.BILINEAR))
    assert np.mean(np.abs(large.astype(float) - enlarged.astype(float))) > 0.2


def test_minimap_and_overlay_survive_tiled_export(owner):
    owner._minimap_enabled = True
    owner._minimap_camera = gfx.OrthographicCamera(8, 6)
    owner._minimap_camera.local.z = 10
    owner._minimap_overlay_camera = gfx.ScreenCoordsCamera(invert_y=True)
    owner._minimap_overlay_scene = gfx.Scene()
    title = gfx.Text(
        text="Minimap",
        font_size=12,
        screen_space=True,
        anchor="top-left",
        material=gfx.TextMaterial(color="blue"),
    )
    title.local.position = (5, 5, 0)
    owner._minimap_overlay_scene.add(title)
    owner._viewport_hud_policy = lambda: SimpleNamespace(enabled=True)
    owner.compute_scene_bounds = lambda **kw: (np.zeros(3), np.ones(3))
    owner._compute_minimap_rect = lambda: (220, 10, 90, 70)
    initial = camera_snapshot(owner._minimap_camera)
    full = owner.export_screenshot_to_array(2, include_hud=True)
    tiled = owner.export_screenshot_to_array(2, include_hud=True, max_tile_size=128)
    np.testing.assert_array_equal(full, tiled)
    assert np.any(full[20:160, 440:620, 2] > full[20:160, 440:620, 0])
    assert_camera_unchanged(owner._minimap_camera, initial)


def test_hidpi_and_fractional_scale_use_current_physical_pixels(owner):
    owner._canvas.set_pixel_ratio(2)
    owner._canvas.set_logical_size(320, 240)
    base = owner.export_screenshot_to_array()
    assert base.shape == (480, 640, 3)
    high = owner.export_screenshot_to_array(1.5, max_tile_size=256)
    assert high.shape == (720, 960, 3)
    assert owner._canvas.get_logical_size() == (320, 240)
    assert owner._canvas.get_pixel_ratio() == 2


def test_file_export_reports_render_error_without_saving(owner, tmp_path, monkeypatch):
    destination = tmp_path / "figure.png"

    def fail(**kwargs):
        raise ValueError("Reduce resolution for this effect")

    monkeypatch.setattr(owner, "export_screenshot_to_array", fail)
    assert not owner.export_screenshot(str(destination), resolution_scale=4)
    assert owner.last_screenshot_error == "Reduce resolution for this effect"
    assert not destination.exists()
    destination.write_bytes(b"previous successful figure")
    assert not owner.export_screenshot(str(destination), resolution_scale=4)
    assert destination.read_bytes() == b"previous successful figure"


def test_progress_is_monotonic_and_finishes_after_hud(owner, monkeypatch):
    progress = []
    monkeypatch.setattr(
        owner, "_composite_visible_qt_overlays", lambda image: (progress.append("hud"), image)[1]
    )
    owner.export_screenshot_to_array(
        2, include_hud=True, max_tile_size=256, progress_callback=progress.append
    )
    assert progress[0] == 0
    assert progress[-2:] == ["hud", 1.0]
    numeric = [value for value in progress if isinstance(value, float)]
    assert numeric == sorted(numeric)
    assert len(numeric) > 3
