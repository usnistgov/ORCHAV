"""Independent boundary regressions for opt-in saved covariance rendering."""

from dataclasses import replace
from types import SimpleNamespace

from tests.visualizer.unit.test_tracking_uncertainty_scene import Renderer, frame
from visualizer.src.sensing.selection import SensingSelection
from visualizer.src.sensing.uncertainty_scene import TrackingUncertaintyScene


def test_detection_selection_cannot_cross_acquisition_or_collection():
    data, renderer, scene = frame(), Renderer(), TrackingUncertaintyScene()
    data["sensing"]["detections"] = [{"track_id": 7}]
    selection = SensingSelection(
        ("clock", "run", "pair"), "a", track_id=7, detection_index=0, source=("source",)
    )
    scene.sync(renderer, data, time_s=1.0, source=("source",), selection=selection)
    assert renderer.objects
    scene.refresh(renderer, selection=replace(selection, acquisition_id="other"))
    assert not renderer.objects
    scene.refresh(
        renderer, selection=replace(selection, detection_collection="detections_before_tracking")
    )
    assert not renderer.objects


def test_disabled_after_use_has_no_additional_redraws(monkeypatch):
    from visualizer.src.sensing import runtime

    renderer, scene = Renderer(), TrackingUncertaintyScene()
    draws = []
    renderer._canvas = SimpleNamespace(request_draw=lambda: draws.append(True))
    scene.sync(renderer, frame(), time_s=1.0, selected_only=False)
    viz = SimpleNamespace(app_state=object(), renderer=renderer, _tracking_uncertainty_scene=scene)
    monkeypatch.setattr(runtime, "state_value", lambda *_: False)
    runtime.refresh_tracking_uncertainty(viz, clear=True)
    assert not renderer.objects
    draws.clear()
    for _ in range(5):
        runtime._sync_tracking_uncertainty(viz, object(), 0)
    assert draws == []


def test_runtime_accepts_only_exact_dashboard_source_and_step(monkeypatch):
    from visualizer.src.io import packed_frame_payload
    from visualizer.src.sensing import runtime
    from visualizer.src.sensing.dashboard_data import DashboardSession

    data, renderer = frame(), Renderer()
    request = SimpleNamespace(
        source_identity=("provider",), effective_display_step=4, display_frame=data
    )
    window = SimpleNamespace(_last_request=request)
    panel = SimpleNamespace(_sensing_window=window)
    viz = SimpleNamespace(
        app_state=object(),
        renderer=renderer,
        ui_manager=SimpleNamespace(panels={"sensing": panel}),
    )
    values = {
        "show_tracking_uncertainty": True,
        "show_detections": True,
        "uncertainty_sigma": 1.0,
        "uncertainty_selected_only": False,
    }
    monkeypatch.setattr(runtime, "state_value", lambda _, key: values[key])
    monkeypatch.setattr(packed_frame_payload, "visualizer_frame_provider", lambda _: object())
    monkeypatch.setattr(DashboardSession, "provider_identity", lambda _: ("provider",))
    # The source identity does not depend on a dashboard session being present.
    runtime._sync_tracking_uncertainty(viz, object(), 4)
    assert renderer.objects
    assert viz._tracking_uncertainty_scene.source == ("provider",)
    request.source_identity = ("other_provider",)
    runtime._sync_tracking_uncertainty(viz, object(), 4)
    assert not renderer.objects
    request.source_identity = ("provider",)
    request.effective_display_step = 3
    runtime._sync_tracking_uncertainty(viz, object(), 4)
    assert not renderer.objects
    request.effective_display_step = 4
    runtime._sync_tracking_uncertainty(viz, object(), 4)
    assert renderer.objects
    viz.animation_running = True
    request.effective_display_step = 5
    request.display_frame = frame()
    request.display_frame["rx_positions"][:] = 50
    runtime._sync_tracking_uncertainty(viz, object(), 5)
    assert renderer.objects
    assert viz._tracking_uncertainty_step == 5
    assert renderer.objects["sensing_uncertainty_7"].transform.translation[0] == 51


def test_verified_world_bundle_requires_exact_epoch_and_keeps_tentative_coast():
    import numpy as np

    from tests.visualizer.unit.test_sensing_tracking_presentation import SavedReader
    from visualizer.src.sensing.fusion_inspection import FusionSelection

    class Reader(SavedReader):
        def epoch(self, index):
            result = super().epoch(index)
            result["tracks"][0]["covariance"] = np.diag([1, 4, 9, 1, 1, 1])
            return result

    scene, renderer = TrackingUncertaintyScene(), Renderer()
    binding = SimpleNamespace(reader=Reader(), sensor_id="sensor")
    selection = FusionSelection("saved-run", 1, 0)
    scene.sync(renderer, {}, binding=binding, time_s=1.0, selection=selection)
    assert len(renderer.objects) == 3
    mesh = renderer.objects["sensing_uncertainty_1"]
    np.testing.assert_allclose(mesh.transform.translation, [10, 0, 0])
    assert mesh.metadata["status"] == "coasting"
    assert mesh.material_payload.base_color[3] == 0.10
    scene.sync(renderer, {}, binding=binding, time_s=1.01, selection=selection)
    assert not renderer.objects
    scene.sync(
        renderer, {}, binding=binding, time_s=1.0, selection=FusionSelection("other-run", 1, 0)
    )
    assert not renderer.objects


def test_runtime_rotates_with_accepted_result_pose_not_current_scene_pose(monkeypatch):
    import numpy as np

    from visualizer.src.io import packed_frame_payload
    from visualizer.src.sensing import runtime
    from visualizer.src.sensing.dashboard_data import DashboardSession

    current, accepted, renderer = frame(), frame(), Renderer()
    current["rx_positions"][:] = 999
    accepted["rx_orientations"][0, 0] = np.pi / 2
    calls = []

    def resolve(raw, **kwargs):
        assert raw is current
        calls.append(kwargs)
        return SimpleNamespace(display_frame=accepted)

    service = SimpleNamespace(resolve_display_sensing_frame=resolve)
    viz = SimpleNamespace(app_state=object(), renderer=renderer)
    values = {
        "show_tracking_uncertainty": True,
        "show_detections": True,
        "uncertainty_sigma": 1.0,
        "uncertainty_selected_only": False,
    }
    monkeypatch.setattr(runtime, "state_value", lambda _, key: values[key])
    monkeypatch.setattr(packed_frame_payload, "visualizer_frame_provider", lambda _: object())
    monkeypatch.setattr(DashboardSession, "provider_identity", lambda _: ("provider",))
    runtime._sync_tracking_uncertainty(viz, service, 4, current)
    mesh = renderer.objects["sensing_uncertainty_7"]
    np.testing.assert_allclose(mesh.transform.translation, [8, 21, 33], atol=1e-6)
    transform = mesh.transform_matrix[:3, :3]
    np.testing.assert_allclose(transform @ transform.T, np.diag([4, 1, 9]), atol=1e-6)
    assert calls == [
        {
            "scene_frame_id": 4,
            "storage_frame_id": 4,
            "product_names": ("detections",),
            "pin_freshest": False,
        }
    ]
    values["show_tracking_uncertainty"] = False
    runtime._sync_tracking_uncertainty(viz, service, 4, current)
    assert len(calls) == 1 and not renderer.objects


def test_singular_covariance_keeps_exact_zero_axes_without_singular_mesh():
    import numpy as np

    data, renderer, scene = frame(), Renderer(), TrackingUncertaintyScene()
    covariance = data["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["tracks"][0][
        "covariance"
    ]
    covariance[0][0] = 0
    scene.sync(renderer, data, time_s=1.0, selected_only=False)
    assert set(renderer.objects) == {
        "sensing_uncertainty_7_outline",
        "sensing_uncertainty_7_center",
    }
    outline = renderer.objects["sensing_uncertainty_7_outline"]
    np.testing.assert_array_equal(outline.payload.points[:, 0], 11)
    np.testing.assert_allclose(outline.transform_matrix, np.eye(4))
    assert "zero-volume" in scene.status
    # A new saved snapshot with exactly zero positional covariance is a point.
    data = frame()
    covariance = data["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["tracks"][0][
        "covariance"
    ]
    for index in range(3):
        covariance[index][index] = 0
    scene.sync(renderer, data, time_s=1.0, selected_only=False)
    assert set(renderer.objects) == {"sensing_uncertainty_7_center"}


def test_panel_cleanup_clears_uncertainty_before_closing_children(monkeypatch):
    from visualizer.src.panels.sensing_panel import SensingPanel
    from visualizer.src.sensing import runtime

    calls = []
    parent = SimpleNamespace(renderer=object())
    panel = SimpleNamespace(
        _closed=False,
        parent=parent,
        saved_track_scene=SimpleNamespace(
            close=lambda: None, overlay=SimpleNamespace(clear=lambda _: None)
        ),
        _retire_dashboard_session=lambda: None,
        _refresh_timer=None,
        _dashboard_poll_timer=None,
        dashboard_host=None,
        _sensing_window=None,
        _dashboard_generation=0,
        widgets={},
    )
    monkeypatch.setattr(
        runtime, "refresh_tracking_uncertainty", lambda viz, **kwargs: calls.append((viz, kwargs))
    )
    SensingPanel.cleanup(panel)
    assert calls == [(parent, {"clear": True})]


def test_native_pygfx_singular_covariance_renders_and_cleans_up(qapp):
    import os

    import pytest

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Native pygfx smoke requires ORCHAV_RUN_PYGFX_TESTS=1")
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace())
    try:
        renderer.initialize_visualizer(width=128, height=128)
        scene = TrackingUncertaintyScene()
        data = frame()
        data["sensing"]["detection_filter_metadata"]["tracker"]["decisions"]["tracks"][0][
            "covariance"
        ][0][0] = 0
        assert scene.sync(renderer, data, time_s=1.0, selected_only=False)
        renderer._renderer.render(renderer._scene, renderer._camera)
        image = renderer._renderer.snapshot()
        assert image.ndim == 3 and image.shape[-1] == 4 and image.size > 0
        assert len(scene.overlay.owned_ids) == 2
        assert scene.clear(renderer)
        renderer._renderer.render(renderer._scene, renderer._camera)
        assert not scene.overlay.owned_ids
    finally:
        renderer.close()
