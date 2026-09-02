from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if "visualizer" in sys.modules:
    pkg = sys.modules["visualizer"]
    pkg_path = str(REPO_ROOT / "visualizer")
    if hasattr(pkg, "__path__") and pkg_path not in list(pkg.__path__):
        pkg.__path__.append(pkg_path)

from visualizer.src.diagnostics.debug_capture import (
    collect_debug_inventory,
    collect_focus_points,
    deterministic_camera_from_bounds,
    deterministic_camera_from_points,
)
from visualizer.src.services.object_identity import (
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from visualizer.src.types.camera_state import CameraState


class _Bounds:
    def __init__(self, min_bound, max_bound):
        self.min_bound = np.asarray(min_bound, dtype=float)
        self.max_bound = np.asarray(max_bound, dtype=float)


class _Renderer:
    renderer_type = "pygfx"

    def __init__(self):
        target_name = "target:walker::mesh"
        node_name = "node:tx_0::marker"
        self._name_to_handle = {target_name: object(), node_name: object()}
        self._hidden = set()
        self._external_geometry_names = {111: "external_geom_111"}
        self._positions = {
            target_name: np.array([0.0, 0.0, 0.0]),
            node_name: np.array([1.0, 2.0, 3.0]),
        }

    def compute_scene_bounds(self):
        return _Bounds([-2.0, -1.0, 0.0], [4.0, 5.0, 6.0])

    def get_camera_state(self):
        return CameraState(
            eye=(10.0, 11.0, 12.0),
            lookat=(1.0, 2.0, 3.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=60.0,
        )

    def get_runtime_stats(self):
        return {"present_attempts": 3}

    def has_named_geometry(self, name: str) -> bool:
        return name in self._name_to_handle

    def get_named_geometry_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._name_to_handle))

    def is_named_visible(self, name: str):
        return name not in self._hidden if name in self._name_to_handle else None

    def get_named_position(self, name: str):
        return self._positions.get(name)

    def _external_name_for_geometry(self, geometry):
        return getattr(geometry, "external_name", None)


class _Geometry:
    def __init__(self, center, external_name=None):
        self._center = np.asarray(center, dtype=float)
        self.external_name = external_name

    def get_center(self):
        return self._center


class _Viz:
    def __init__(self):
        self.renderer = _Renderer()
        target_entry = {
            "target_name": "Walker",
            "stable_target_id": "walker",
            "object_id": "target:walker",
            "mesh": _Geometry([0.0, 0.0, 0.0], external_name="external_geom_target"),
            "_frame_visible": True,
            "position": [0.0, 0.0, 0.0],
        }
        self.target_entries = [target_entry]
        self.tx_markers = [_Geometry([1.0, 2.0, 3.0], external_name="external_geom_tx")]
        self.rx_markers = []


def test_deterministic_camera_from_bounds_auto_iso():
    camera = deterministic_camera_from_bounds(_Bounds([0.0, 0.0, 0.0], [10.0, 20.0, 5.0]))
    assert camera is not None
    assert camera.lookat == (5.0, 10.0, 2.5)
    assert camera.eye[0] < camera.lookat[0]
    assert camera.eye[1] < camera.lookat[1]
    assert camera.eye[2] > camera.lookat[2]


def test_deterministic_camera_from_points_target_focus_is_tight():
    camera = deterministic_camera_from_points(
        [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
        preset="target_focus",
    )
    assert camera is not None
    assert camera.lookat == (1.0, 1.0, 0.5)
    assert camera.eye[2] > camera.lookat[2]


def test_collect_debug_inventory_reports_origin_and_external_duplicates():
    viz = _Viz()
    target_geometry = make_target_entry_geometry_name(viz.target_entries[0], "mesh")
    target_alias = f"geometry_{id(viz.target_entries[0]['mesh'])}"
    viz.renderer._name_to_handle[target_alias] = object()
    payload = collect_debug_inventory(viz, label="frame_0000", step=0)

    tx_geometry = make_node_geometry_name("tx", 0, "marker")

    assert payload["renderer"] == "pygfx"
    assert payload["ghost_checks"]["origin_target_meshes"] == [target_geometry]
    assert payload["ghost_checks"]["target_external_duplicates"] == ["external_geom_target"]
    assert payload["ghost_checks"]["target_named_alias_duplicates"] == [target_alias]
    assert payload["ghost_checks"]["node_external_duplicates"] == ["external_geom_tx"]
    assert target_geometry in payload["named_geometry"]["target_named_names"]
    assert tx_geometry in payload["named_geometry"]["node_named_names"]
    assert payload["renderer_state"]["object_count"] == 3


def test_collect_focus_points_uses_targets_and_nodes():
    viz = _Viz()
    points = collect_focus_points(viz)
    assert [round(float(v), 3) for v in points[0]] == [0.0, 0.0, 0.0]
    assert [round(float(v), 3) for v in points[1]] == [1.0, 2.0, 3.0]


def test_collect_focus_points_targets_scope_excludes_nodes():
    viz = _Viz()
    points = collect_focus_points(viz, scope="targets")
    assert len(points) == 1
    assert [round(float(v), 3) for v in points[0]] == [0.0, 0.0, 0.0]
