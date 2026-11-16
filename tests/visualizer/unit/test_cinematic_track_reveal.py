"""Finish the existing reveal with separately recorded detection and track anchors."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from tests.visualizer.unit.test_cinematic_reveal import cinematic  # noqa: F401
from tests.visualizer.unit.test_report_geometry import report_request
from visualizer.src.experiments.cinematic_reveal import CinematicReveal
from visualizer.src.services.sensing_service import SensingService


def test_final_stage_frames_displaced_associated_saved_track(cinematic):  # noqa: F811
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    renderer._apply_clipping_to_material = lambda material: None
    sensing = request.display_frame["sensing"]
    sensing["detections"][0].update(track_x_m=40.0, track_y_m=5.0, track_z_m=3.0)
    expected = SensingService.dashboard_world_detections(request.display_frame)[0][
        "world_track_position"
    ]
    controls._play()
    session.sync(request, selection, dashboard)
    assert not session.overlay.layers["track"].visible  # no early track reveal
    controls.phase.setCurrentIndex(3)
    session.sync(request, selection, dashboard)
    session._fade_tick(session.fade_start + 1)
    detection = dashboard.selected_world_position()
    center, radius, note = session.stages["position"]
    np.testing.assert_allclose(center, (detection + expected) / 2)
    assert radius > np.linalg.norm(expected - detection) / 2
    np.testing.assert_allclose(session.overlay.layers["track"].geometry.positions.data[0], expected)
    assert session.overlay.layers["track"].visible
    assert "TRACK — associated saved track 7" in note and "unavailable" not in note
    assert camera.transition is not None
    session.clear()
    assert not session.fade.isActive()


def test_track_anchor_rejects_inexact_source_acquisition_and_incomplete_state():
    request, selection = report_request()
    row = request.display_frame["sensing"]["detections"][0]
    row.update(track_x_m=2.0, track_y_m=3.0)
    assert CinematicReveal._associated_track_position(request, selection) is None
    row["track_z_m"] = 4.0
    assert CinematicReveal._associated_track_position(request, selection) is not None
    for changed in (
        replace(selection, source=("another",)),
        replace(selection, acquisition_id="another"),
        replace(selection, detection_collection="raw_detections"),
    ):
        assert CinematicReveal._associated_track_position(request, changed) is None
    row["track_z_m"] = np.nan
    assert CinematicReveal._associated_track_position(request, selection) is None


def test_track_anchor_rejects_wrong_diagnostic_epoch():
    from copy import deepcopy

    from tests.visualizer.unit.test_tracking_uncertainty_scene import frame

    request, selection = report_request()
    sensing = request.display_frame["sensing"]
    sensing["detection_filter_metadata"] = deepcopy(frame()["sensing"]["detection_filter_metadata"])
    decisions = sensing["detection_filter_metadata"]["tracker"]["decisions"]
    timestamp = sensing["temporal"]["result"]["reference_time_s"]
    decisions["update_time_s"] = timestamp
    assert CinematicReveal._associated_track_position(request, selection) is not None
    decisions["update_time_s"] = timestamp + 1
    assert CinematicReveal._associated_track_position(request, selection) is None


def test_native_saved_track_anchor_renders(qapp):
    import os

    if os.environ.get("ORCHAV_RUN_PYGFX_TESTS") != "1":
        pytest.skip("Native pygfx smoke requires explicit opt-in")
    from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace())
    try:
        renderer.initialize_visualizer(width=128, height=128)
        overlay = ExperimentOverlay(renderer)
        overlay.track_position([1.0, 2.0, 3.0], "#aabbcc")
        renderer._renderer.render(renderer._scene, renderer._camera)
        assert renderer._renderer.snapshot().shape[-1] == 4
        overlay.clear()
    finally:
        renderer.close()
