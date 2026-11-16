"""Detection rows and persistent tracks have distinct selection lifetimes."""

from copy import deepcopy

from tests.visualizer.unit import test_sensing_optional_selection as selection_fixtures
from tests.visualizer.unit.test_sensing_stage_dashboard import _replay_event_for, _request
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products

detector = selection_fixtures.detector
from visualizer.src.sensing.selection import matches_detection, select_estimate
from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle


def test_report_selection_matches_one_row_even_when_track_identity_is_shared(detector):
    window, request = detector
    sensing = request.display_frame["sensing"]
    sensing["detections"][0]["track_id"] = 7
    sensing["detections"].append(dict(sensing["detections"][0]))
    report = select_estimate(sensing, detection_index=0)
    assert matches_detection(report, sensing, 0, sensing["detections"][0])
    assert not matches_detection(report, sensing, 1, sensing["detections"][1])
    track = select_estimate(sensing, track_id=7)
    assert all(
        matches_detection(track, sensing, i, row) for i, row in enumerate(sensing["detections"])
    )


def test_report_expires_on_next_acquisition_but_explicit_track_persists(detector):
    window, request = detector
    sensing = request.display_frame["sensing"]
    sensing["detections"][0]["track_id"] = 7
    following = deepcopy(sensing)
    following["temporal"]["acquisition"]["acquisition_id"] = "6411913b-405d-4e2c-80aa-c3cdc6af5acd"
    _rematerialize_products(following)
    bundle = build_detection_stage_bundle(following)
    next_request = _request(following, bundle, _replay_event_for(bundle))

    window.select_detection_index(0)
    assert window._linked_selection.detection_index == 0
    window.update_from_request(next_request)
    assert window._linked_selection is None
    assert not window.result_browser.table.selectionModel().hasSelection()

    window.set_linked_selection(select_estimate(following, track_id=7))
    window.update_from_request(request)
    assert window._linked_selection.track_id == 7
    assert window._linked_selection.detection_index is None
