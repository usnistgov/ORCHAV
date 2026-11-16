"""Independent array-factor references and actual volume-control lifetime tests."""

from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from shared.sensing.arrays import planar_spatial_fft
from tests.visualizer.unit.test_volume_evidence_workflow import opened as _opened
from visualizer.src.sensing.volume_antenna import (
    ReceivePattern,
    pattern_from_metadata,
    receive_power,
    response_object,
)

opened = _opened


@pytest.fixture
def pattern():
    return ReceivePattern(8, 8, 0.5, 0.5, (12, 23, 34), (0, 0, 0), 3.5e9, True, "test")


def test_complex_direct_sum_matches_native_fft_and_full_sphere(pattern):
    """Independent physical coordinates reproduce every native coarse power."""
    positions = np.array(
        [[0, (col - 3.5) * 0.5, (3.5 - row) * 0.5] for col in range(8) for row in range(8)]
    )
    directions = np.array([[1, 0, 0], [0.8, 0.6, 0], [0.6, 0, 0.8], [0.5, -0.5, 2**-0.5]])
    samples = np.exp(2j * np.pi * positions @ directions.T)
    native = planar_spatial_fft(
        samples,
        num_rows=8,
        num_cols=8,
        vertical_spacing_wavelengths=0.5,
        horizontal_spacing_wavelengths=0.5,
    )
    z, y = np.meshgrid(
        native.elevation_direction_cosine, native.azimuth_direction_cosine, indexing="ij"
    )
    visible = y * y + z * z < 1
    expected = np.abs(native.spectrum[visible]) ** 2
    # Direct complex weights are reconstructed separately from response code.
    beams = np.column_stack(
        (np.sqrt(1 - y[visible] ** 2 - z[visible] ** 2), y[visible], z[visible])
    )
    weights = np.exp(2j * np.pi * positions @ beams.T) / 8
    np.testing.assert_allclose(np.abs(weights.conj().T @ samples) ** 2, expected, atol=3e-14)
    assert len(pattern.beam_directions) == 45
    np.testing.assert_allclose(
        receive_power(pattern, directions, "scan"), expected.max(axis=0), atol=3e-14
    )
    broadside = np.abs(samples.sum(axis=0) / 8) ** 2
    np.testing.assert_allclose(
        receive_power(pattern, directions, "broadside"), broadside, atol=3e-14
    )
    for mode in ("broadside", "scan"):
        np.testing.assert_array_equal(
            receive_power(pattern, directions, mode),
            receive_power(pattern, directions * [-1, 1, 1], mode),
        )
    np.testing.assert_allclose(
        receive_power(pattern, pattern.beam_directions, "scan"), 64, atol=3e-14
    )


def test_pattern_world_pose_radius_and_immutable_payload(pattern):
    baseline = response_object(pattern, "broadside", 20, 0.4)
    moved = replace(pattern, position_m=(-3, 5, 8), orientation_rad=(np.pi / 2, 0, 0))
    rotated = response_object(moved, "broadside", 20, 0.4)
    local = baseline.payload.vertices - pattern.position_m
    expected = local[:, [1, 0, 2]] * [-1, 1, 1] + moved.position_m
    np.testing.assert_allclose(rotated.payload.vertices, expected, atol=4e-6)
    assert np.max(np.linalg.norm(local, axis=1)) == pytest.approx(20)
    assert np.min(local[:, 0]) == pytest.approx(-20)
    assert np.max(local[:, 0]) == pytest.approx(20)
    assert not baseline.payload.vertices.flags.writeable
    assert baseline.metadata["type"] == "rx_array_response"
    assert len(baseline.payload.vertices) == 7381
    assert len(baseline.payload.triangles) == 14400


def recorded_array():
    return dict(
        rows=8,
        columns=8,
        element_count=64,
        element_order="sionna_column_major_top_to_bottom",
        element_pattern="iso",
        polarization="V",
        local_horizontal_direction="+y",
        local_vertical_direction="+z",
        spatial_phase_convention="positive_phase_toward_positive_y_and_z",
        spatial_fft_normalization="unitary_ortho",
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        dict(rows=True),
        dict(element_count=128),
        dict(element_pattern="tr38901"),
        dict(polarization="VH"),
        dict(element_order="row_major"),
        dict(horizontal_spacing_wavelengths=0),
        dict(rows=17, columns=17, element_count=289),
    ],
)
def test_unsupported_metadata_is_explicitly_unavailable(mutation):
    array = recorded_array() | mutation
    with pytest.raises(ValueError):
        pattern_from_metadata(
            array, [0, 0, 0], [0, 0, 0], dict(carrier_freq_hz=3.5e9), {}, source="test"
        )


def test_coarse_beam_count_is_validated_and_non_scan_is_not_invented():
    args = recorded_array(), [0, 0, 0], [0, 0, 0], dict(carrier_freq_hz=3.5e9)
    with pytest.raises(ValueError, match="coarse beam count"):
        pattern_from_metadata(
            *args,
            dict(
                array_detection_mode="beam_scan",
                beam_count=1024,
                angular_hemisphere="rx_local_positive_x",
            ),
            source="test",
        )
    p = pattern_from_metadata(*args, {}, source="test")
    assert not p.scan_enabled
    with pytest.raises(ValueError, match="not supported"):
        receive_power(p, [[1, 0, 0]], "scan")


def test_real_controls_enable_off_restore_and_close_clean(opened, pattern, monkeypatch, qtbot):
    window, objects, legends = opened
    monkeypatch.setattr(
        "visualizer.src.sensing.volume_antenna_controls.load_receive_pattern", lambda _: pattern
    )
    before = window.evaluation_snapshot
    a = window.antenna
    a.mode.setCurrentIndex(a.mode.findData("broadside"))
    qtbot.waitUntil(lambda: "sensing:volume:rx_response" in objects)
    assert window.evaluation_snapshot is before
    assert any("not detection coverage" in line for line in legends[-1].lines)
    assert "front/back" in a.note.text()
    a.mode.setCurrentIndex(a.mode.findData("scan"))
    a.radius.setValue(24)
    state = window.workflow.context()["view"]["antenna"]
    assert state["mode"] == "scan" and state["radius"] == 24
    a.mode.setCurrentIndex(a.mode.findData("off"))
    assert "sensing:volume:rx_response" not in objects
    assert not any("not detection coverage" in line for line in legends[-1].lines)
    a.restore(state)
    window._draw()
    assert "sensing:volume:rx_response" in objects
    assert any("Arbitrary radius 24 m" in line for line in legends[-1].lines)
    assert window.evaluation_snapshot is before
    window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(window) and not objects


def test_unavailable_pattern_keeps_outcomes_and_close_safe(opened, qtbot):
    window, objects, _ = opened
    before = window.evaluation_snapshot
    window.antenna.mode.setCurrentIndex(1)
    qtbot.waitUntil(lambda: window.antenna.failed)
    assert "unavailable" in window.antenna.note.text()
    assert "sensing:outcome_volume" in objects
    assert "sensing:volume:rx_response" not in objects
    assert window.evaluation_snapshot is before


def test_close_while_pattern_worker_pending_does_not_publish(opened, monkeypatch, pattern):
    window, objects, _ = opened
    pending = Future()
    monkeypatch.setattr(window.executor, "submit", lambda *args, **kwargs: pending)
    window.antenna.mode.setCurrentIndex(1)
    assert window.antenna.pending is pending
    window.close()
    pending.set_result(pattern)
    window._poll()
    assert not objects
