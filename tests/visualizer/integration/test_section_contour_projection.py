"""GPU regression for native slice expansion across the camera near plane."""

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1",
        reason="set ORCHAV_RUN_PYGFX_TESTS=1 to run GPU-backed visual checks",
    ),
]


@pytest.fixture
def render_slice():
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    from visualizer.src.renderers.pygfx.section_contours import _section_material_class

    canvas = RenderCanvas(size=(256, 256), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")
    camera = gfx.PerspectiveCamera(60, 1, depth_range=(0.1, 100))
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000000"))

    def render(a, b):
        # A triangle whose intersection with y=0 is exactly segment A→B.
        positions = np.asarray([np.add(a, [0, 1, 0]), np.add(a, [0, -1, 0]), b], np.float32)
        mesh = gfx.Mesh(
            gfx.Geometry(positions=positions, indices=np.asarray([[0, 1, 2]], np.int32)),
            _section_material_class()(color="#ffffff", thickness=3, plane=(0, 1, 0, 0)),
        )
        scene.add(mesh)
        canvas.request_draw(lambda: renderer.render(scene, camera))
        pixels = np.asarray(canvas.draw()).copy()
        scene.remove(mesh)
        return pixels[:, :, :3].max(axis=2) > 50

    yield render
    canvas.close()


def test_near_crossing_keeps_visible_half_without_mirrored_streak(render_slice):
    mask = render_slice((-0.5, 0, -3), (1, 0, 1))
    y, x = np.nonzero(mask)
    # A projects near x91. The part before the near plane extends rightward
    # out of the frustum. Dividing the behind-eye endpoint by w first instead
    # draws a false segment to A's left (native 0.17: x46..91 on this fixture).
    assert 85 <= x.min() <= 95
    assert x.max() == 255
    assert y.min() >= 125 and y.max() <= 130
    assert 350 <= len(x) <= 700


@pytest.mark.parametrize(
    "a,b",
    [
        ((-0.5, 0, 3), (1, 0, 1)),  # Entire intersection is behind the eye.
        ((-0.5, 0, -3), (-1, 0, -6)),  # Zero screen-space segment length.
    ],
)
def test_behind_and_degenerate_segments_do_not_draw(render_slice, a, b):
    assert not render_slice(a, b).any()


def test_visible_front_section_is_retained(render_slice):
    y, x = np.nonzero(render_slice((-0.5, 0, -3), (0.5, 0, -3)))
    assert 85 <= x.min() <= 95 and 160 <= x.max() <= 170
    assert y.min() >= 125 and y.max() <= 130
    assert 150 <= len(x) <= 350
