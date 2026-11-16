"""Exact saved-state covariance presentation across frame and playback changes."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from visualizer.src.sensing.selection import SensingSelection
from visualizer.src.sensing.uncertainty_scene import TrackingUncertaintyScene


class Renderer:
    renderer_id = "pygfx"

    def __init__(self):
        self.objects = {}

    def ensure_object(self, obj):
        self.objects[obj.id] = obj
        return True

    def remove_object(self, name):
        self.objects.pop(name, None)
        return True


def frame():
    return {
        "rx_positions": np.array([[10.0, 20.0, 30.0]]),
        "rx_orientations": np.zeros((1, 3)),
        "sensing": {
            "pair_axis": [{"frame_pair_index": 0, "pair_index": 0, "pair_id": "pair"}],
            "temporal": {
                "clock": {"clock_id": "clock"},
                "acquisition": {"pair_id": "pair", "acquisition_id": "a"},
                "result": {"processing_run_id": "run", "reference_time_s": 1.0},
            },
            "detections": [],
            "detection_filter_metadata": {
                "tracker": {
                    "enabled": True,
                    "decisions": {
                        "schema_version": 1,
                        "update_time_s": 1.0,
                        "state_labels": ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"],
                        "tracks": [
                            {
                                "track_id": 7,
                                "mean": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                                "confirmed": False,
                                "misses": 2,
                                "hits": 3,
                                "covariance": np.diag([1.0, 4.0, 9.0, 1.0, 1.0, 1.0]).tolist(),
                            }
                        ],
                        "assignments": [],
                        "candidates": [],
                    },
                }
            },
        },
    }


def test_native_pose_epoch_and_cached_geometry():
    scene, renderer, data = TrackingUncertaintyScene(), Renderer(), frame()
    assert scene.sync(renderer, data, time_s=1.0, source=("source",), selected_only=False)
    assert len(renderer.objects) == 3
    mesh = renderer.objects["sensing_uncertainty_7"]
    np.testing.assert_allclose(mesh.transform.translation, [11, 22, 33])
    assert mesh.metadata["status"] == "coasting"
    assert mesh.material_payload.base_color[3] == 0.10  # tentative retains identity
    scene.refresh(renderer, selected_only=False, sigma=2.0)
    updated = renderer.objects[mesh.id]
    assert updated.payload is mesh.payload
    np.testing.assert_allclose(np.linalg.norm(updated.transform_matrix[:3, :3], axis=0), [2, 4, 6])
    bad = deepcopy(data)
    bad["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["update_time_s"] = 2.0
    scene.sync(renderer, bad, time_s=1.0, selected_only=False)
    assert not renderer.objects
    assert "exact result epoch" in scene.status


def test_selection_source_missing_covariance_and_replacement():
    scene, renderer, data = TrackingUncertaintyScene(), Renderer(), frame()
    selection = SensingSelection(("clock", "run", "pair"), "a", 7, source=("source",))
    scene.sync(renderer, data, time_s=1.0, source=("source",), selection=selection)
    assert len(renderer.objects) == 3
    scene.sync(renderer, data, time_s=1.0, source=("other",), selection=selection)
    assert not renderer.objects
    data = deepcopy(data)
    data["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["tracks"][0][
        "covariance"
    ] = None
    scene.sync(renderer, data, time_s=1.0, selected_only=False)
    assert not renderer.objects
    assert "not recorded" in scene.status
    for _ in range(3):
        scene.sync(renderer, frame(), time_s=1.0, selected_only=False)
        assert len(renderer.objects) == 3
        scene.clear(renderer)
        assert not renderer.objects


def test_controls_preserve_uncertainty_across_playback(qtbot):
    from visualizer.src.sensing.uncertainty_controls import TrackingUncertaintyControls

    control = TrackingUncertaintyControls()
    qtbot.addWidget(control)
    assert not control.enabled.isChecked()
    control.enabled.setChecked(True)
    control.set_playing(True)
    assert control.enabled.isChecked() and control.enabled.isEnabled()
    control.set_playing(False)
    assert control.enabled.isEnabled() and control.enabled.isChecked()


def test_disabled_runtime_never_builds_or_reads_uncertainty(monkeypatch):
    from visualizer.src.sensing import runtime

    viz = SimpleNamespace(app_state=SimpleNamespace(extension_state={}), renderer=Renderer())
    runtime._sync_tracking_uncertainty(viz, object(), 0)
    assert not hasattr(viz, "_tracking_uncertainty_scene")


def test_no_selection_shows_all_tracks_then_selection_narrows_them():
    data = frame()
    tracks = data["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["tracks"]
    tracks.append(dict(deepcopy(tracks[0]), track_id=8))
    renderer, scene = Renderer(), TrackingUncertaintyScene()
    scene.sync(renderer, data, time_s=1.0, source=("source",))
    assert len(renderer.objects) == 6
    selection = SensingSelection(("clock", "run", "pair"), "a", track_id=7, source=("source",))
    scene.refresh(renderer, selection=selection)
    assert len(renderer.objects) == 3 and "sensing_uncertainty_7" in renderer.objects
    scene.refresh(renderer, selection=None)
    assert len(renderer.objects) == 6
