"""Exact shared-kernel parity against the preserved high-precision implementation."""

import numpy as np
import pytest

from shared.sensing.range_response import (
    C0,
    FrequencyNoise,
    RangeResponseConfig,
    frequency_to_delay,
    paths_to_frequency,
)
from tests.shared.sensing_cfr_reference import make_case, reference_paths_to_frequency


@pytest.mark.parametrize("elements", [1, 16, 128])
@pytest.mark.parametrize("n", [320, 384, 512])
@pytest.mark.parametrize("kind", ["shared", "element_specific"])
def test_noise_metadata_and_downstream_bitwise_parity(elements, n, kind):
    args, kwargs = make_case(elements, 9, 8, n, kind)
    expected, expected_meta = reference_paths_to_frequency(*args, **kwargs)
    actual, actual_meta = paths_to_frequency(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_meta == expected_meta
    for window in ("boxcar", "hann"):
        config = RangeResponseConfig("cfr_ifft", n, window)
        opts = dict(
            bandwidth=kwargs["bandwidth"], config=config, fft_size=512, range_mode="monostatic"
        )
        a, axes_a, meta_a = frequency_to_delay(actual, **opts)
        b, axes_b, meta_b = frequency_to_delay(expected, **opts)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            np.abs(np.fft.fft(a, axis=1)) ** 2, np.abs(np.fft.fft(b, axis=1)) ** 2
        )
        assert meta_a == meta_b
        for key in axes_a:
            np.testing.assert_array_equal(axes_a[key], axes_b[key])


@pytest.mark.parametrize("shape", ["shared", "element_specific", "different_masks"])
def test_cross_path_time_frequency_block_boundaries(shape):
    args, kwargs = make_case(3, 259, 133, 513)
    a, delays, valid, times = args
    delays = np.broadcast_to(delays, (3, 259)).copy()
    valid = np.broadcast_to(valid, (3, 259)).copy()
    if shape == "element_specific":
        delays[1, 3] = np.nextafter(delays[0, 3], np.inf)
    if shape == "different_masks":
        valid[1, 128] = False
    a[0, 25, 2] = np.nan
    a[1, 180, 4] = np.inf
    delays[:, 2] = np.nan  # invalid padding must not prevent safe reuse
    args = (a, delays, valid, times)
    expected, expected_meta = reference_paths_to_frequency(*args, **kwargs)
    actual, actual_meta = paths_to_frequency(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_meta == expected_meta
    assert actual_meta["nonfinite_path_count"] == 2


@pytest.mark.parametrize("residual", [0.0, 1e-13])
@pytest.mark.parametrize("n", [320, 384, 512])
def test_destructive_interference_and_boundary_delay(residual, n):
    times = 3 + np.arange(8) / 1000
    delays = np.array([3.25 / 500e6, 3.25 / 500e6, np.nextafter(n / 500e6, 0)])
    coeff = np.broadcast_to(
        np.array([1 + 2j, -(1 + 2j) * (1 - residual), 1e-8j])[None, :, None], (16, 3, 8)
    ).copy()
    kwargs = dict(bandwidth=500e6, config=RangeResponseConfig("cfr_ifft", n))
    args = (coeff, delays, np.ones(3, bool), times)
    expected, expected_meta = reference_paths_to_frequency(*args, **kwargs)
    actual, actual_meta = paths_to_frequency(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_meta == expected_meta
    delays[-1] = n / 500e6
    with pytest.raises(ValueError, match="ambiguity period"):
        paths_to_frequency(*args, **kwargs)


@pytest.mark.parametrize("elements", [1, 16, 128])
def test_empty_path_channels_and_seeded_noise(elements):
    kwargs = dict(
        bandwidth=500e6,
        config=RangeResponseConfig("cfr_ifft", 320, frequency_noise=FrequencyNoise(1e-8, 42)),
        realization_index=8,
    )
    args = (
        np.empty((elements, 0, 4), complex),
        np.empty(0),
        np.empty(0, bool),
        np.arange(4) / 1000,
    )
    expected, expected_meta = reference_paths_to_frequency(*args, **kwargs)
    actual, actual_meta = paths_to_frequency(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_meta == expected_meta


def test_smaller_grids_keep_paths_and_reject_aliasing():
    times = np.arange(4) / 1000
    a = np.ones((1, 1, 4), complex)
    for n in (320, 384, 512):
        kwargs = dict(bandwidth=500e6, config=RangeResponseConfig("cfr_ifft", n))
        paths_to_frequency(a, np.array([81.45 * 2 / C0]), np.ones(1, bool), times, **kwargs)
        with pytest.raises(ValueError, match="ambiguity period"):
            paths_to_frequency(a, np.array([n / 500e6]), np.ones(1, bool), times, **kwargs)


def test_default_legacy_model_unchanged():
    assert RangeResponseConfig().model == "point_bin_floor"


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_noncontiguous_coefficients_and_explicit_equal_delays(dtype):
    args, kwargs = make_case(16, 17, 16, 384)
    a, delays, valid, times = args
    a = a.astype(dtype)[..., ::2]
    delays = np.broadcast_to(delays, (16, 17)).copy()
    # Invalid padding does not contribute and need not share a delay.
    delays[:, 2] = np.arange(16) + np.nan
    args = (a, delays, valid, times[::2])
    expected, expected_meta = reference_paths_to_frequency(*args, **kwargs)
    actual, actual_meta = paths_to_frequency(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_meta == expected_meta


def test_reuse_reduces_only_kernel_evaluations(monkeypatch):
    args, kwargs = make_case(16, 259, 4, 512)
    original = np.exp
    calls = []

    def counted(value, *a, **kw):
        calls.append(value.shape)
        return original(value, *a, **kw)

    monkeypatch.setattr(np, "exp", counted)
    expected, _ = reference_paths_to_frequency(*args, **kwargs)
    before = list(calls)
    calls.clear()
    actual, _ = paths_to_frequency(*args, **kwargs)
    assert len(before) == 16 * len(calls)
    assert max(np.prod(shape) for shape in calls) <= 128 * 256
    np.testing.assert_array_equal(actual, expected)


def test_actual_detector_results_and_stage_artifacts_are_identical():
    from dataclasses import fields

    from shared.sensing.detection.pipeline import DetectionPipelineConfig, run_detection_pipeline
    from shared.sensing.signal_processing import RangeDopplerResult

    times = np.arange(32) / 1000
    delays = np.array([40.3, 100.4, 200.1]) / 500e6
    tones = np.array([125, -187.5, 62.5])
    a = np.broadcast_to(
        np.array([1, 0.7j, 0.2])[:, None] * np.exp(2j * np.pi * tones[:, None] * times), (16, 3, 32)
    ).copy()
    config = RangeResponseConfig("cfr_ifft", 320, "hann", FrequencyNoise(1e-8, 21))
    kwargs = dict(bandwidth=500e6, config=config)
    args = (a, delays, np.ones(3, bool), times)
    outputs = [
        function(*args, **kwargs)[0]
        for function in (reference_paths_to_frequency, paths_to_frequency)
    ]
    detections = []
    for output in outputs:
        delay, axes, _ = frequency_to_delay(
            output, bandwidth=500e6, config=config, fft_size=320, range_mode="monostatic"
        )
        combined = delay.sum(axis=0) / np.sqrt(16)
        rd = np.fft.fftshift(np.fft.fft(combined, axis=0), axes=0) / np.sqrt(32)
        result = RangeDopplerResult(
            np.abs(rd) ** 2, axes["range_axis_m"], np.fft.fftshift(np.fft.fftfreq(32, 0.001)), rd
        )
        detections.append(
            run_detection_pipeline(
                result,
                config=DetectionPipelineConfig((1, 2), (3, 5), 8, cfar_max_dynamic_range_db=40),
                wavelength_m=C0 / 28e9,
                return_stage_artifacts=True,
            )
        )
    assert detections[0].detections  # Ensure the control actually exercises detections.
    import json

    # Missing angles deliberately publish NaN Cartesian coordinates. Canonical
    # serialization compares their exact values without NaN != NaN semantics.
    def canonical(value):
        return json.dumps(value, sort_keys=True, default=lambda item: np.asarray(item).tolist())

    assert canonical(detections[0].detections) == canonical(detections[1].detections)
    assert canonical(detections[0].metadata) == canonical(detections[1].metadata)
    for field in fields(detections[0].stages):
        left = getattr(detections[0].stages, field.name)
        right = getattr(detections[1].stages, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    from copy import deepcopy

    from shared.sensing.tracker import KalmanTracker, MofNTracker, TrackerConfig

    for tracker_type in (MofNTracker, KalmanTracker):
        trackers = [tracker_type(TrackerConfig(record_diagnostics=True)) for _ in range(2)]
        for index in range(3):
            results = [
                tracker.filter_detections(
                    deepcopy(detection.detections), update_time_s=index * 0.064
                )
                for tracker, detection in zip(trackers, detections)
            ]
            normalized = [
                json.dumps(result, sort_keys=True, default=lambda value: np.asarray(value).tolist())
                for result in results
            ]
            assert normalized[0] == normalized[1]
        assert results[0][0]  # Confirmed tracks, including association/state updates.


def test_sharing_predicate_never_gathers_all_element_delay_rows():
    from shared.sensing.range_response import _project_valid_paths, frequency_grid

    class NoElementPathGather(np.ndarray):
        def __getitem__(self, key):
            if (
                isinstance(key, tuple)
                and isinstance(key[0], slice)
                and len(key) > 1
                and isinstance(key[1], np.ndarray)
            ):
                raise AssertionError("sharing eligibility allocated an element-by-path gather")
            return super().__getitem__(key)

    args, _ = make_case(16, 259, 4, 512)
    a, delays, valid, _ = args
    delays = np.broadcast_to(delays, (16, 259)).copy().view(NoElementPathGather)
    valid = np.broadcast_to(valid, (16, 259))
    frequencies = frequency_grid(500e6, 512)
    output = _project_valid_paths(a, delays, valid, frequencies)
    assert output.shape == (16, 4, 512)
