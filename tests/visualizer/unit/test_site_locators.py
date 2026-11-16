"""Readable locator annotations must preserve positions and screen dimensions."""

from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.model import RenderObject
from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin
from visualizer.src.services.site_locator_service import site_locator_objects
from visualizer.src.types.render_payloads import TextLabelPayload


def test_readable_locator_changes_annotation_without_moving_radio():
    axes = tuple(np.array([0, 100]) for _ in range(3))
    ordinary = site_locator_objects([10, 20, 8], "Tx1", axes, ("point", "stem", "label"))
    readable = site_locator_objects(
        [10, 20, 8], "Tx1", axes, ("point", "stem", "label"), readable=True
    )
    for objects in (ordinary, readable):
        np.testing.assert_array_equal(objects[0].payload.points, [[10, 20, 8]])
        np.testing.assert_array_equal(objects[1].payload.points[0], [10, 20, 8])
    assert ordinary[1].payload.points[1, 2] == 108
    assert readable[1].payload.points[1, 2] == 116
    assert readable[2].payload.text == "Tx1"
    assert readable[2].metadata["label_background_rgba"] == (1, 1, 1, 0.8)
    assert readable[2].metadata["label_anchor"] == "bottom-center"
    assert readable[2].metadata["label_anchor_offset"] == 12.0
    assert readable[2].metadata["label_connector_origin"] == [10, 20, 8]
    assert readable[2].metadata["label_bold"]
    assert readable[0].material_payload.base_color == (0, 0, 0, 1)
    assert readable[1].material_payload.base_color == (0, 0, 0, 1)
    assert ordinary[0].material_payload.base_color == (1, 0.3, 0.85, 1)
    assert "label_background_rgba" not in ordinary[2].metadata


def test_native_bold_can_be_disabled_without_changing_text():
    gfx = pytest.importorskip("pygfx")
    native = gfx.Text(text="Tx1", screen_space=True, material=gfx.TextMaterial())
    renderer = SimpleNamespace(_objects={"label": native})
    payload = TextLabelPayload(text="Tx1", screen_space=True)

    def glyph_formats():
        native._update_object()
        data = native.geometry.glyph_data.data
        return data["format"][data["size"] > 0].copy()

    plain = glyph_formats()
    assert plain.size == 3
    PygfxGeometryMixin._sync_render_object_metadata(
        renderer, RenderObject(id="label", payload=payload, metadata={"label_bold": True})
    )
    assert not np.array_equal(glyph_formats(), plain)
    PygfxGeometryMixin._sync_render_object_metadata(
        renderer, RenderObject(id="label", payload=payload, metadata={"label_bold": False})
    )
    np.testing.assert_array_equal(glyph_formats(), plain)


def test_native_label_card_follows_zoom_resize_text_and_visibility():
    gfx = pytest.importorskip("pygfx")
    native = gfx.Text(
        text="Tx1",
        font_size=36,
        screen_space=True,
        anchor="bottom-center",
        anchor_offset=12,
        material=gfx.TextMaterial(),
    )
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 20)
    extent = [600, 400]
    renderer = SimpleNamespace(
        _gfx=gfx,
        _objects={"label": native},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: tuple(extent)),
    )
    PygfxLabelMixin._sync_label_background(renderer, "label", (1, 1, 1, 0.8))

    def verify_pixels():
        PygfxLabelMixin._update_label_backgrounds(renderer)
        plate = renderer._label_backgrounds["label"][1]
        corners = np.array([[-0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1]])
        clip = (
            camera.projection_matrix @ camera.world.inverse_matrix @ plate.world.matrix @ corners.T
        ).T
        pixels = clip[:, :2] / clip[:, 3:] * np.array(extent) / 2
        np.testing.assert_allclose(
            pixels[1] - pixels[0], native._aabb[1, :2] - native._aabb[0, :2] + 10, atol=1e-4
        )
        assert pixels[0, 1] > 0  # The card clears the stem endpoint.
        border = plate.children[0]
        assert border.material.thickness == 1.5
        assert border.render_order == native.render_order - 1
        return plate

    plate = verify_pixels()
    original_width = native._aabb[1, 0] - native._aabb[0, 0]
    native.set_text("A longer direct native annotation")
    assert verify_pixels() is plate
    assert native._aabb[1, 0] - native._aabb[0, 0] > original_width
    camera.local.position = (0, 0, 50)
    extent[:] = [1200, 800]
    native.set_text("Elevated sensor")
    native.font_size = 42
    assert verify_pixels() is plate
    PygfxLabelMixin._sync_label_background(renderer, "label", (1, 1, 1, 0.8), [-2, 0, 0])
    PygfxLabelMixin._update_label_backgrounds(renderer)
    assert native.anchor == "middle-left"
    assert native._aabb[0, 0] == 12
    native.visible = False
    PygfxLabelMixin._update_label_backgrounds(renderer)
    assert not plate.visible
    PygfxLabelMixin._sync_label_background(renderer, "label", None)
    assert not native.children
    assert "label" not in renderer._label_backgrounds
