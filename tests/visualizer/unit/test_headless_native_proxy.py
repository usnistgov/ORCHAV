"""Headless native boxes use retained geometry and preserve their absorbing style."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from weakref import WeakSet

import numpy as np
import pytest

from visualizer.src.scene.assembly import target_metadata_to_payload


def metadata():
    return {
        "name": "proxy",
        "representation": "tr38901",
        "geometry": {"type": "box", "length_m": 4.5, "width_m": 1.8, "height_m": 1.6},
        "current_position": [37.312, 0, 1],
        "orientation": [0, 0, 0],
    }


@pytest.mark.parametrize("representation", ["constant_rcs", "tr38901"])
@pytest.mark.parametrize("yaw,expected", [(0, [4.5, 1.8, 1.6]), (np.pi / 2, [1.8, 4.5, 1.6])])
def test_retained_native_box_pose_and_material(representation, yaw, expected):
    data = metadata()
    data["representation"] = representation
    data["orientation"][0] = yaw
    original = deepcopy(data)
    name, mesh, material = target_metadata_to_payload(data, Path("."), Path("."))
    assert name == "proxy"
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), expected, atol=1e-7)
    np.testing.assert_allclose(np.mean(mesh.vertices, axis=0), data["current_position"])
    assert material.base_color == (0.2, 0.4, 1.0, 0.25)
    assert data == original


def test_ordinary_missing_mesh_behavior_unchanged():
    assert target_metadata_to_payload({"name": "mesh"}, Path("."), Path(".")) is None


def test_retained_box_dimensions_are_not_scaled_twice():
    data = metadata()
    data["scale"] = 3.0
    data["geometry"]["scaling"] = [2.0, 3.0, 4.0]
    _, mesh, _ = target_metadata_to_payload(data, Path("."), Path("."))
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), [4.5, 1.8, 1.6])


def test_native_box_uses_full_zyx_pose():
    from scipy.spatial.transform import Rotation

    from visualizer.src.scene.native_target_geometry import native_proxy_payload

    data = metadata()
    data["orientation"] = [0.7, -0.4, 0.2]
    data["current_position"] = [7.0, -11.0, 3.0]
    local = native_proxy_payload(data)
    rotation = Rotation.from_euler("ZYX", data["orientation"])
    expected = rotation.apply(local.vertices.copy()) + data["current_position"]
    _, mesh, _ = target_metadata_to_payload(data, Path("."), Path("."))
    np.testing.assert_allclose(mesh.vertices, expected)


def test_equal_size_native_boxes_keep_independent_cached_poses():
    from visualizer.src.backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers

    first = metadata()
    second = metadata()
    second["current_position"] = [-10.0, 8.0, 4.0]
    second["orientation"] = [np.pi / 2, 0.0, 0.0]
    for data in (first, second, first):
        _, mesh, _ = target_metadata_to_payload(data, Path("."), Path("."))
        buffers = mesh_payload_to_pygfx_buffers(mesh)
        np.testing.assert_allclose(buffers["positions"], mesh.vertices, rtol=1e-6)


def test_native_box_material_supports_transparent_pygfx_rendering():
    gfx = pytest.importorskip("pygfx")
    from visualizer.src.backends.pygfx_scene_helpers import payload_to_pygfx_mesh

    _, mesh, material = target_metadata_to_payload(metadata(), Path("."), Path("."))
    native = payload_to_pygfx_mesh(gfx, mesh, material)
    assert isinstance(native.material, gfx.MeshStandardMaterial)
    assert native.material.alpha_mode == "weighted_blend"
    assert native.material.opacity == pytest.approx(0.25)
    assert not native.material.depth_write
    assert native.material.depth_test


@pytest.mark.parametrize("animated", [False, True])
def test_notebook_routes_preserve_native_proxy_material(monkeypatch, animated):
    gfx = pytest.importorskip("pygfx")
    from visualizer.src.backends import pygfx_scene_helpers as helpers
    from visualizer.src.notebook import presentation
    from visualizer.src.notebook import pygfx as notebook

    viz = object.__new__(notebook.PygfxNotebookViz)
    viz._project_root = Path(".")
    viz._scenario = SimpleNamespace(root=Path("."))
    viz._mesh_entries = []
    viz._texture_cache = {}
    viz._frame_source = SimpleNamespace(list_frames=lambda: [0, 1])
    viz._canvases = WeakSet()
    first, second = metadata(), metadata()
    second["current_position"] = [-10.0, 8.0, 4.0]
    second["representation"] = "constant_rcs"
    frames = [first, second]
    monkeypatch.setattr(viz, "_setup_ibl", lambda *_: None)
    monkeypatch.setattr(
        viz,
        "_build_frame_data",
        lambda frame, *_, **_kw: ({"targets_metadata": [frames[frame]]}, None),
    )
    monkeypatch.setattr(
        viz, "_extract_node_positions", lambda _: (np.empty((0, 3)), np.empty((0, 3)))
    )
    monkeypatch.setattr(viz, "_extract_mesh_bboxes", lambda: [])
    monkeypatch.setattr(viz, "_add_node_labels", lambda *_: None)
    meshes = []
    original = helpers.payload_to_pygfx_mesh

    def capture(*args, **kwargs):
        mesh = original(*args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(helpers, "payload_to_pygfx_mesh", capture)
    monkeypatch.setattr(presentation, "payload_to_pygfx_mesh", capture)
    if animated:
        import rendercanvas.auto

        canvas = MagicMock()
        canvas.get_closed.return_value = False
        renderer = MagicMock()
        monkeypatch.setattr(notebook, "_running_in_jupyter", lambda: False)
        monkeypatch.setattr(rendercanvas.auto, "RenderCanvas", lambda **_: canvas)
        monkeypatch.setattr(rendercanvas.auto.loop, "run", lambda: None)
        monkeypatch.setattr(gfx, "OrbitController", lambda *_a, **_kw: MagicMock())
        monkeypatch.setattr(
            "visualizer.src.renderers.pygfx.canvas.create_wgpu_renderer",
            lambda *_a, **_kw: renderer,
        )
        clock = [0.0]
        monkeypatch.setattr("time.perf_counter", lambda: clock[0])
        viz.animate(frames=[0, 1], fps=1.0, center=[0, 0, 0], distance=60)
        callback = canvas.request_draw.call_args.args[0]
        callback()
        clock[0] = 1.0
        callback()
        clock[0] = 2.0
        callback()
        expected = [first, second, first]
    else:
        for frame in (0, 1, 0):
            viz._build_pygfx_scene(
                frame,
                "reflection_order",
                "all",
                "all",
                False,
                False,
                False,
                None,
                None,
                45,
                30,
                60,
                [0, 0, 0],
                2,
                5,
                1,
                "neutral_outdoor",
                0,
            )
        expected = [first, second, first]
    assert len(meshes) == len(expected)
    for mesh, data in zip(meshes, expected, strict=True):
        assert mesh.material.alpha_mode == "weighted_blend"
        assert mesh.material.opacity == pytest.approx(0.25)
        np.testing.assert_allclose(
            np.mean(mesh.geometry.positions.data, axis=0), data["current_position"], rtol=1e-6
        )


@pytest.mark.parametrize(
    "field,value", [("current_position", [0, np.nan, 0]), ("orientation", [0, 0])]
)
def test_malformed_native_pose_rejected(field, value):
    data = metadata()
    data[field] = value
    with pytest.raises(ValueError, match="finite position and orientation"):
        target_metadata_to_payload(data, Path("."), Path("."))
