"""Camera anchors come from saved data, independently of overlay visibility."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from visualizer.src.panels.sensing_panel import SensingPanel
from visualizer.src.sensing.selection import select_estimate
from visualizer.src.sensing.selection_geometry import selected_world_position
from visualizer.src.sensing.sensing_window import SensingWindow


def spatial_request():
    request = replace(_request(), source_identity=("saved recording",))
    request.display_frame.update(
        rx_positions=np.array([[10.0, 20.0, 30.0]]),
        rx_orientations=np.array([[np.pi / 2, 0.0, 0.0]]),
    )
    sensing = request.display_frame["sensing"]
    sensing["detections"] = [{"range_m": 5.0, "aoa_az_deg": 0.0, "track_id": 8}]
    selection = replace(select_estimate(sensing, detection_index=0), source=request.source_identity)
    return request, selection


def test_saved_detection_uses_bound_pose_without_rendered_overlays_or_truth():
    request, selection = spatial_request()
    request = replace(request, state_frame={"rx_positions": [[-500, -500, -500]]})
    np.testing.assert_allclose(selected_world_position(request, selection), [10, 25, 30])
    assert "x_m" not in request.display_frame["sensing"]["detections"][0]


@pytest.mark.parametrize(
    "change", ["source", "scope", "acquisition", "collection", "pose", "range-only"]
)
def test_stale_or_nonspatial_selection_has_no_camera_anchor(change):
    request, selection = spatial_request()
    if change == "source":
        selection = replace(selection, source=("other",))
    elif change == "scope":
        selection = replace(selection, scope=("other", "run", "pair"))
    elif change == "acquisition":
        selection = replace(selection, acquisition_id="other")
    elif change == "collection":
        selection = replace(selection, detection_collection="other")
    elif change == "pose":
        request.display_frame.pop("rx_orientations")
    else:
        request.display_frame["sensing"]["detections"][0].pop("aoa_az_deg")
    assert selected_world_position(request, selection) is None


def test_native_coasting_track_position_is_projected_from_receiver_local_coordinates():
    request, selection = spatial_request()
    request.display_frame["sensing"]["detections"] = []
    selection = replace(selection, detection_index=None)
    track = {"track_id": 8, "position_frame": "receiver local", "position_m": [4, 2, 1]}
    np.testing.assert_allclose(
        selected_world_position(request, selection, track=track), [8, 24, 31]
    )
    track["position_m"] = None  # A radial tracker has no unique Cartesian anchor.
    assert selected_world_position(request, selection, track=track) is None


def test_saved_world_track_uses_world_position_without_projecting_again():
    request, _ = spatial_request()
    track = {"track_id": 2, "position_frame": "world", "position_m": [4, 2, 1]}
    np.testing.assert_allclose(selected_world_position(request, None, track=track), [4, 2, 1])
    track["position_m"] = [np.nan, 2, 1]
    assert selected_world_position(request, None, track=track) is None


def test_dashboard_center_button_requires_a_saved_spatial_anchor(qtbot):
    request, _ = spatial_request()
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    window.update_from_request(request)
    window.select_detection_index(0)
    assert window.locate_selection_button.isEnabled()
    np.testing.assert_allclose(window.selected_world_position(), [10, 25, 30])
    window.set_linked_selection(None)
    assert not window.locate_selection_button.isEnabled()
    request.display_frame["sensing"]["detections"][0].pop("aoa_az_deg")
    window.select_detection_index(0)
    assert not window.locate_selection_button.isEnabled()
    assert "range shell" in window.locate_selection_button.toolTip()


@pytest.mark.parametrize("presentation", [True, False])
def test_center_action_moves_camera_without_enabling_detection_overlays(presentation):
    request, selection = spatial_request()
    positions = []
    renderer = SimpleNamespace(focus_camera=positions.append)
    parent = SimpleNamespace(
        renderer=renderer,
        app_state=SimpleNamespace(extension_state={"sensing": {"show_detections": False}}),
        camera_controller=(
            SimpleNamespace(presentation=lambda: SimpleNamespace(focus=positions.append))
            if presentation
            else None
        ),
    )
    panel = SimpleNamespace(
        parent=parent,
        _sensing_window=SimpleNamespace(
            selected_world_position=lambda: selected_world_position(request, selection)
        ),
        _set_status=lambda _: pytest.fail("Available saved position was rejected"),
    )
    panel.selected_camera_position = lambda: SensingPanel.selected_camera_position(panel)
    SensingPanel._locate_selection(panel)
    np.testing.assert_allclose(positions, [[10, 25, 30]])
    assert not parent.app_state.extension_state["sensing"]["show_detections"]
