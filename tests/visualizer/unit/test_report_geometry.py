"""Saved-bin geometry retains exact source, pose and beam semantics."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tests.visualizer.unit.test_experimental_range_geometry import _request
from tests.visualizer.unit.test_volume_antenna import recorded_array
from visualizer.src.experiments.report_geometry import (
    build_angular_cell,
    receive_pattern_for_request,
    selected_beam_index,
)
from visualizer.src.sensing.selection import select_estimate
from visualizer.src.sensing.volume_antenna import receive_metadata_signature, receive_power
from visualizer.src.utils.aperture_geometry import yaw_pitch_roll_to_rotation_matrix


def report_request():
    request, old = _request(mode="monostatic", tx=(10, 20, 3), rx=(10, 20, 3))
    request.display_frame["rx_orientations"] = np.array([[0.4, -0.2, 0.1]])
    sensing = request.display_frame["sensing"]
    sensing["rx_array"] = recorded_array()
    sensing["detection_filter_metadata"] = {
        "array_detection_mode": "beam_scan",
        "beam_count": 45,
        "angular_hemisphere": "rx_local_positive_x",
    }
    sensing["detections"] = [{"range_bin": 1, "doppler_bin": 1, "beam_index": 20, "track_id": 7}]
    selection = replace(
        select_estimate(sensing, detection_index=0), source=request.source_identity, cell=old.cell
    )
    return request, selection


def test_coarse_angular_cell_keeps_saved_bin_boundaries_and_rotated_world_pose():
    request, selection = report_request()
    original = deepcopy(request.display_frame)
    pattern = receive_pattern_for_request(request)
    cell = build_angular_cell(request, selection)
    rotation = yaw_pitch_roll_to_rotation_matrix(*pattern.orientation_rad)
    local = (cell.vertices - pattern.position_m) @ rotation
    radius = np.linalg.norm(local, axis=1)
    np.testing.assert_allclose(radius[:81], 16, atol=3e-6)
    np.testing.assert_allclose(radius[81:], 27, atol=3e-6)
    directions = local / radius[:, None]
    center = pattern.beam_directions[20]
    assert np.max(np.abs(directions[:, 1] - center[1])) == pytest.approx(0.125, abs=2e-7)
    assert np.max(np.abs(directions[:, 2] - center[2])) == pytest.approx(0.125, abs=2e-7)
    assert np.min(directions[:, 0]) >= 0
    assert cell.beam_index == 20
    assert "not uncertainty" in cell.label
    assert "range_doppler_angle_by_pair" not in request.display_frame["sensing"]
    np.testing.assert_array_equal(request.display_frame["rx_positions"], original["rx_positions"])
    assert request.display_frame["sensing"]["detections"] == original["sensing"]["detections"]


@pytest.mark.parametrize(
    "mutation", ["source", "acquisition", "pose", "beam", "bin", "array", "pair"]
)
def test_unavailable_or_mismatched_saved_geometry_is_rejected(mutation):
    request, selection = report_request()
    sensing = request.display_frame["sensing"]
    if mutation == "source":
        selection = replace(selection, source=("another",))
    elif mutation == "acquisition":
        selection = replace(selection, acquisition_id="another")
    elif mutation == "pose":
        request.display_frame["rx_orientations"][:] = np.nan
    elif mutation == "beam":
        sensing["detections"][0]["beam_index"] = 45
    elif mutation == "bin":
        sensing["detections"][0]["range_bin"] = 2
    elif mutation == "array":
        sensing["rx_array"]["element_pattern"] = "unknown"
    else:
        sensing["pair_axis"][0]["pair_id"] = "another"
    with pytest.raises(ValueError):
        build_angular_cell(request, selection)


def test_array_pose_is_rejected_when_display_epoch_is_stale():
    request, _ = report_request()
    request.display_frame["timestamp_s"] += 1
    with pytest.raises(ValueError, match="epoch"):
        receive_pattern_for_request(request)


@pytest.mark.parametrize("field", ["array", "carrier", "processing", "pair", "sensor"])
def test_array_fingerprint_proves_current_configuration_and_pair_without_heavy_products(field):
    request, _selection = report_request()
    sensing = request.display_frame["sensing"]
    signature = receive_metadata_signature(sensing)
    assert isinstance(signature, str) and len(signature) == 64
    sensing["unused_large_product"] = np.ones(1_000_000)
    sensing["detections"] = []
    assert receive_metadata_signature(sensing) == signature
    if field == "array":
        sensing["rx_array"]["horizontal_spacing_wavelengths"] *= 2
    elif field == "carrier":
        sensing["config_resolved"]["carrier_freq_hz"] *= 2
    elif field == "processing":
        sensing["detection_filter_metadata"]["angular_hemisphere"] = "other"
    elif field == "pair":
        sensing["pair_axis"][0]["frame_pair_index"] += 1
    else:
        sensing["temporal"]["acquisition"]["rx_sensor_id"] = "another sensor"
    assert receive_metadata_signature(sensing) != signature
    sensing.pop("rx_array")
    assert receive_metadata_signature(sensing) is None


def test_selected_beam_matches_independent_complex_sum_and_front_back_symmetry():
    request, selection = report_request()
    pattern = receive_pattern_for_request(request)
    beam = selected_beam_index(request, selection, pattern)
    samples = np.array([[1, 0, 0], [0.8, 0.6, 0], [0.6, 0, 0.8]])
    positions = np.array(
        [[0, (col - 3.5) * 0.5, (3.5 - row) * 0.5] for col in range(8) for row in range(8)]
    )
    direction = pattern.beam_directions[beam]
    weights = np.exp(2j * np.pi * positions @ direction) / 8
    expected = np.abs(weights.conj() @ np.exp(2j * np.pi * positions @ samples.T)) ** 2
    measured = receive_power(pattern, samples, "beam", beam_index=beam)
    np.testing.assert_allclose(measured, expected, atol=3e-14)
    np.testing.assert_allclose(
        measured, receive_power(pattern, samples * [-1, 1, 1], "beam", beam_index=beam)
    )
    assert receive_power(pattern, [direction], "beam", beam_index=beam)[0] == pytest.approx(64)
    with pytest.raises(ValueError):
        receive_power(pattern, samples, "beam", beam_index=999)
