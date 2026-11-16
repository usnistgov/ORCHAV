"""Independent edge-case regressions for array observability and fusion isolation."""

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.detection.beam_scan import refine_array_directions
from shared.sensing.fusion.core import (
    COMPONENTS,
    FusionConfig,
    MonostaticFusion,
    Observation,
    Sensor,
    gated_assignment,
    measurement_model,
)
from shared.sensing.fusion.quality import native_observation


@pytest.mark.parametrize("rows,cols", [(8, 1), (1, 1), (1, 8), (8, 8)])
@pytest.mark.parametrize("factor", [1, 2, 8, 16])
def test_singleton_axis_is_assumed_zero_and_not_a_fusion_measurement(rows, cols, factor):
    detection = dict(doppler_bin=0, range_bin=0, range_m=10.0, aoa_az_deg=0.0, aoa_el_deg=0.0)
    original = deepcopy(detection)
    result = refine_array_directions(
        [detection],
        np.ones((rows * cols, 1, 1), complex),
        num_rows=rows,
        num_cols=cols,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
        oversampling=factor,
    )[0]
    assert result["aoa_az_deg"] == 0
    assert result["aoa_el_deg"] == 0
    assert result["aoa_az_observable"] == (cols > 1 and rows > 1)
    assert result["azimuth_direction_cosine_observable"] == (cols > 1)
    assert result["aoa_el_observable"] == (rows > 1)
    assert detection == original
    observation = native_observation("s", 1, 0, 0.0, result, {}, {})
    assert ("azimuth_rad" in observation.components) == (cols > 1 and rows > 1)
    assert ("elevation_rad" in observation.components) == (rows > 1)
    assert observation.has_position == (rows > 1 and cols > 1)


@pytest.mark.parametrize("bad_id", ["a_bad", "z_bad"])
@pytest.mark.parametrize("time", [0.0, 1.0])
def test_geometry_rejection_does_not_discard_valid_epoch_observation(bad_id, time):
    sensor = Sensor("s", (0, 0, 0), (0, 0, 0))
    processor = MonostaticFusion([sensor], FusionConfig(confirmation_epochs=1))
    if time:
        processor.step(0.0, [Observation("s", "initial", 0.0, 10.0, 0.0, 0.0, 0.0)])
    output = processor.step(
        time,
        [
            Observation("s", bad_id, time, 10.0, 0.0, np.pi / 2 - 1e-8, 0.0),
            Observation("s", "good", time, 10.0, 0.0, 0.0, 0.0),
        ],
    )
    assert processor.time_s == time
    assert processor.next_id == 2
    assert len(output["tracks"]) == 1
    assert output["tracks"][0]["track_id"] == 1
    assert output["tracks"][0]["sources"][0]["detection_id"] == "good"
    rejection = [r for r in output["geometry_rejections"] if r.get("stage") == "birth"]
    assert len(rejection) == 1 and rejection[0]["detection_id"] == bad_id
    assert rejection[0]["track_id"] is None


@pytest.mark.parametrize("gate", [1, 1.0, np.int64(1)])
def test_integer_gate_preserves_fractional_assignment_cost(gate):
    costs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert gated_assignment(costs, gate) == [(0, 1), (1, 0)]


def test_unexpected_birth_failure_still_rolls_back_valid_earlier_birth(monkeypatch):
    sensor = Sensor("s", (0, 0, 0), (0, 0, 0))
    processor = MonostaticFusion([sensor])
    update = MonostaticFusion._update

    def fail(self, track, observation, sensor, **kwargs):
        update(self, track, observation, sensor, **kwargs)
        if observation.detection_id == "z_bad":
            raise np.linalg.LinAlgError("unexpected numerical failure")

    with monkeypatch.context() as context:
        context.setattr(MonostaticFusion, "_update", fail)
        with pytest.raises(np.linalg.LinAlgError):
            processor.step(
                0.0,
                [
                    Observation("s", "a_good", 0, 10, 0, 0, 0),
                    Observation("s", "z_bad", 0, 20, 0, 0, 0),
                ],
            )
    assert processor.time_s is None and processor.next_id == 1 and not processor.tracks
    assert len(processor.step(0.0, [Observation("s", "retry", 0, 10, 0, 0, 0)])["tracks"]) == 1


def test_reordered_components_and_correlated_covariance_match_jacobian_submatrix():
    sensor = Sensor("s", (1, 2, 3), (0.3, -0.2, 0.1))
    state = np.array([12.0, 5.0, 4.0, 0.1, 0.2, 0.3])
    full, jac = measurement_model(state, sensor)
    components = (COMPONENTS[3], COMPONENTS[1], COMPONENTS[0])
    partial, partial_jac = measurement_model(state, sensor, components)
    np.testing.assert_array_equal(partial, full[[3, 1, 0]])
    np.testing.assert_array_equal(partial_jac, jac[[3, 1, 0]])
    covariance = np.array(
        [
            [1, 0.02, 0.03, 0.04],
            [0.02, 0.2, 0.01, 0.03],
            [0.03, 0.01, 0.3, 0.02],
            [0.04, 0.03, 0.02, 0.4],
        ]
    )
    sensor = Sensor("s", (1, 2, 3), (0.3, -0.2, 0.1), tuple(map(tuple, covariance)))
    observation = Observation("s", "d", 0, full[0], full[1], None, full[3], components)
    np.testing.assert_array_equal(
        MonostaticFusion([sensor])._measurement_covariance(observation, sensor),
        covariance[np.ix_([3, 1, 0], [3, 1, 0])],
    )


@pytest.mark.parametrize("elevation", [-35.0, -10.0, 17.0, 40.0])
def test_vertical_ula_recovers_elevation_without_inventing_azimuth(elevation):
    rows = 16
    uz = np.sin(np.radians(elevation))
    data = np.exp(-2j * np.pi * 0.5 * np.arange(rows) * uz)[:, None, None]
    # A nearby coarse elevation suffices for the local interpolation search.
    result = refine_array_directions(
        [dict(doppler_bin=0, range_bin=0, range_m=10, aoa_az_deg=0, aoa_el_deg=elevation)],
        data,
        num_rows=rows,
        num_cols=1,
        horizontal_spacing_wavelengths=0.5,
        vertical_spacing_wavelengths=0.5,
        oversampling=16,
    )[0]
    assert result["aoa_az_deg"] == 0 and not result["aoa_az_observable"]
    assert abs(result["aoa_el_deg"] - elevation) < 0.3


def test_polar_threshold_applies_only_when_angles_are_used():
    from shared.sensing.fusion.core import GeometryRejection

    sensor = Sensor("s", (0, 0, 0), (0, 0, 0))
    state = np.array([0.5e-6, 0.0, 10.0, 1.0, 2.0, 3.0])
    with pytest.raises(GeometryRejection):
        measurement_model(state, sensor)
    prediction, jac = measurement_model(state, sensor, (COMPONENTS[0], COMPONENTS[3]))
    assert np.all(np.isfinite(prediction)) and np.all(np.isfinite(jac))
    state[0] = 2e-6
    assert np.all(np.isfinite(measurement_model(state, sensor)[1]))


@pytest.mark.parametrize("rows,cols", [(8, 1), (1, 1), (1, 8), (8, 8)])
def test_coarse_beam_scan_marks_unobservable_angles_before_tracking(rows, cols):
    from shared.sensing.detection.beam_scan import run_beam_scan_detection_pipeline
    from shared.sensing.detection.pipeline import DetectionPipelineConfig
    from shared.sensing.signal_processing import RangeDopplerResult

    power = np.ones((1, 24, 24))
    power[0, 12, 12] = 1000.0
    rd = RangeDopplerResult(power[0], np.arange(24.0), np.arange(24.0) - 12, None)
    config = DetectionPipelineConfig(
        cfar_guard_cells=(1, 1),
        cfar_reference_cells=(2, 2),
        cfar_threshold_scale=3,
        min_snr_db=0,
    )
    seen = []

    def tracker(detections):
        seen.extend(deepcopy(detections))
        return detections, {}

    result = run_beam_scan_detection_pipeline(
        rd,
        beam_power=power,
        beam_azimuth_deg=np.array([0.0]),
        beam_elevation_deg=np.array([0.0]),
        config=config,
        wavelength_m=1.0,
        num_rows=rows,
        num_cols=cols,
        post_detection_filter=tracker,
    )
    assert len(result.detections) == len(seen) == 1
    detection = result.detections[0]
    assert seen[0]["aoa_az_observable"] == (rows > 1 and cols > 1)
    assert seen[0]["aoa_el_observable"] == (rows > 1)
    observation = native_observation("s", 0, 0, 0.0, detection, {}, {})
    assert observation.has_position == (rows > 1 and cols > 1)
    assert ("azimuth_rad" in observation.components) == (rows > 1 and cols > 1)
    assert ("elevation_rad" in observation.components) == (rows > 1)
