"""Integration smoke test for pygfx renderer backend.

Skipped unless the pygfx runtime profile is explicitly enabled.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
)


def _pygfx_available() -> bool:
    try:
        import pygfx  # noqa: F401
        from PySide6 import QtWidgets as _QtWidgets  # noqa: F401
        from rendercanvas.qt import RenderCanvas as _RenderCanvas  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        (not _pygfx_available()) or (os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1"),
        reason="pygfx runtime tests disabled (set ORCHAV_RUN_PYGFX_TESTS=1) or runtime unavailable",
    ),
]


def _make_frame_geometry(step: int):
    n_points = 200
    rng = np.random.RandomState(seed=step)

    pts = rng.rand(n_points, 3).astype(np.float32) * 100.0
    lines = np.column_stack([np.arange(0, n_points - 1), np.arange(1, n_points)]).astype(np.int32)
    line_colors = rng.rand(n_points - 1, 3).astype(np.float32)
    mpc_lines = LineSetPayload(points=pts, lines=lines, colors=line_colors)

    bounce_pts = pts[::2]
    bounce_colors = rng.rand(len(bounce_pts), 3).astype(np.float32)
    mpc_points = PointCloudPayload(points=bounce_pts, colors=bounce_colors)

    ground_verts = np.array(
        [[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], dtype=np.float32
    )
    ground_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ground_mesh = MeshPayload(vertices=ground_verts, triangles=ground_tris)

    return {
        "mpc_lines": mpc_lines,
        "mpc_points": mpc_points,
        "ground": ground_mesh,
    }


@pytest.fixture
def renderer(qapp):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    r = PygfxRenderer(SimpleNamespace())
    try:
        r.initialize_visualizer(width=128, height=128)
    except Exception as exc:
        pytest.skip(f"pygfx unavailable in this environment: {exc}")
    yield r
    r.close()


class TestPygfxPlayback:
    def test_playback_10_frames(self, renderer):
        for step in range(10):
            geom = _make_frame_geometry(step)
            with renderer.batch_updates():
                renderer.ensure_named_geometry("mpc_lines", geom["mpc_lines"])
                renderer.ensure_named_geometry("mpc_points", geom["mpc_points"])
                renderer.ensure_named_geometry("ground", geom["ground"])

            assert renderer.has_named_geometry("mpc_lines")
            assert renderer.has_named_geometry("mpc_points")
            assert renderer.has_named_geometry("ground")

    def test_visibility_and_material(self, renderer):
        geom = _make_frame_geometry(0)
        renderer.ensure_named_geometry("ground", geom["ground"])
        assert renderer.set_named_visibility("ground", False) is True
        assert renderer.is_named_visible("ground") is False

        assert renderer.set_named_visibility("ground", True) is True
        mat = MaterialPayload(base_color=(0.3, 0.3, 0.3, 1.0), roughness=0.8, metallic=0.0)
        assert renderer.set_named_material("ground", mat) is True

    def test_camera_set_get(self, renderer):
        from visualizer.src.types.camera_state import CameraState

        cam = CameraState(
            eye=(50.0, 50.0, 50.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=45.0,
        )
        assert renderer.set_camera_state(cam) is True
        got = renderer.get_camera_state()
        assert got is not None
        assert abs(got.fov_deg - 45.0) < 5.0

    def test_cleanup(self, renderer):
        geom = _make_frame_geometry(0)
        renderer.ensure_named_geometry("mpc_lines", geom["mpc_lines"])
        renderer.ensure_named_geometry("mpc_points", geom["mpc_points"])
        renderer.ensure_named_geometry("ground", geom["ground"])

        renderer.remove_named_geometry("mpc_lines")
        renderer.remove_named_geometry("mpc_points")
        renderer.remove_named_geometry("ground")

        assert not renderer.has_named_geometry("mpc_lines")
        assert not renderer.has_named_geometry("mpc_points")
        assert not renderer.has_named_geometry("ground")

    def test_camera_restore_loop(self, renderer):
        from visualizer.src.types.camera_state import CameraState

        for idx in range(20):
            cam = CameraState(
                eye=(50.0 + idx, 35.0 + idx * 0.5, 20.0),
                lookat=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov_deg=45.0,
            )
            assert renderer.set_camera_state(cam) is True
            got = renderer.get_camera_state()
            assert got is not None
            assert abs(got.eye[0] - cam.eye[0]) < 1e-2

    @pytest.mark.skipif(
        os.getenv("ORCHAV_RUN_PYGFX_SOAK_TESTS", "0") != "1",
        reason="set ORCHAV_RUN_PYGFX_SOAK_TESTS=1 for extended soak playback",
    )
    @pytest.mark.soak
    def test_playback_300_frames_soak(self, renderer):
        for step in range(320):
            geom = _make_frame_geometry(step)
            with renderer.batch_updates():
                renderer.ensure_named_geometry("mpc_lines", geom["mpc_lines"])
                renderer.ensure_named_geometry("mpc_points", geom["mpc_points"])
                renderer.ensure_named_geometry("ground", geom["ground"])
            if step % 8 == 0:
                renderer.set_named_visibility("mpc_points", (step // 8) % 2 == 0)

        stats = renderer.get_runtime_stats() if hasattr(renderer, "get_runtime_stats") else {}
        failures = int(stats.get("present_failures", 0))
        assert failures == 0
