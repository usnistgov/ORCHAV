"""Real ISO renders must leave unrevealed coverage cells transparent."""

import os
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1",
        reason="set ORCHAV_RUN_PYGFX_TESTS=1 to run GPU-backed visual checks",
    ),
]


@pytest.mark.parametrize("axis", "xyz")
def test_negative_db_sweep_does_not_paint_unrevealed_cells(axis):
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    from visualizer.src.coverage.volume_inspection import scalar_template
    from visualizer.src.renderers.pygfx.volume import ensure_volume
    from visualizer.src.utils.volume_cut import cut_volume

    model = SimpleNamespace(
        values=np.linspace(-100, -50, 8**3).reshape(8, 8, 8), quantity="rss_dbm"
    )
    payload = replace(scalar_template(model, -100, -50), mode="iso", opacity=1)
    canvas = RenderCanvas(size=(192, 192), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#ffffff"))
    handles = count()
    owner = SimpleNamespace(
        _initialized=True,
        _scene=scene,
        _gfx=gfx,
        _objects={},
        _kinds={},
        _name_to_handle={},
        _handle_to_name={},
        _reverse_objects={},
        _geometry_upload_center={},
        _allocate_handle=lambda: next(handles),
        request_redraw=lambda: None,
    )
    camera = gfx.OrthographicCamera(12, 12)
    camera.local.position = (3.5, -50, 3.5) if axis == "z" else (3.5, 3.5, 50)
    camera.look_at((3.5, 3.5, 3.5))
    canvas.request_draw(lambda: renderer.render(scene, camera))

    def visible_pixels(current):
        assert ensure_volume(owner, "coverage", current)
        return np.asarray(canvas.draw())[:, :, :3].min(axis=2) < 240

    try:
        full = visible_pixels(payload)
        cut = visible_pixels(cut_volume(payload, (axis, 2, True)))
        assert full.sum() > 1000
        # Exactly three of eight complete cell layers are revealed. Filling the
        # rest with zero dB instead would paint the entire volume's footprint.
        assert cut.sum() / full.sum() == pytest.approx(3 / 8, abs=0.01)
        assert not (cut & ~full).any()
        np.testing.assert_array_equal(visible_pixels(payload), full)
    finally:
        canvas.close()
