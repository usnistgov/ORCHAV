"""Tests for renderer-independent playback cadence calculations."""

import pytest

from visualizer.src.playback import (
    DEFAULT_PLAYBACK_INTERVAL_MS,
    PlaybackCadenceTracker,
    PlaybackMode,
    fixed_fps_interval_ms,
    format_playback_fps,
    normalize_playback_mode,
    real_time_interval_ms,
    timestamp_interval_ms,
)


@pytest.mark.parametrize(
    ("fps", "expected_ms"),
    [(30, 33), (60, 17), (120, 8)],
)
def test_fixed_fps_interval_is_expressed_as_frame_start_cadence(fps, expected_ms):
    assert fixed_fps_interval_ms(fps) == expected_ms


def test_real_time_interval_preserves_total_scenario_pace_and_respects_stride():
    assert real_time_interval_ms(12.0, 12) == 1000
    assert real_time_interval_ms(12.0, 12, frame_stride=2) == 2000


@pytest.mark.parametrize("duration, count", [(None, 10), (1.0, 1), (-1.0, 10)])
def test_real_time_interval_falls_back_without_valid_scenario_timing(duration, count):
    assert real_time_interval_ms(duration, count) == DEFAULT_PLAYBACK_INTERVAL_MS


def test_timestamp_interval_accepts_seconds_and_rejects_zero_delta():
    assert timestamp_interval_ms(1.25, 1.5) == 250
    assert timestamp_interval_ms(1.25, 1.25) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (25.2, "25"),
        (2.5, "2.5"),
        (0.4, "0.4"),
        (0.04, "0.04"),
        (0.001, "<0.01"),
        (0.0, None),
    ],
)
def test_format_playback_fps_uses_adaptive_precision(value, expected):
    assert format_playback_fps(value) == expected


def test_invalid_persisted_playback_mode_defaults_to_maximum():
    assert normalize_playback_mode("removed-mode") is PlaybackMode.MAXIMUM


def test_playback_cadence_measures_completed_frame_intervals():
    tracker = PlaybackCadenceTracker()

    tracker.record_completion(10.0)
    tracker.record_completion(10.04)
    tracker.record_completion(10.08)

    assert tracker.frames_per_second() == pytest.approx(25.0)
    assert tracker.mean_interval_ms() == pytest.approx(40.0)


def test_playback_cadence_reports_recent_interval_percentiles():
    tracker = PlaybackCadenceTracker()
    for timestamp in (0.0, 0.01, 0.03, 0.07, 0.15):
        tracker.record_completion(timestamp)

    assert tracker.percentile_interval_ms(50.0) == pytest.approx(30.0)
    assert tracker.percentile_interval_ms(95.0) == pytest.approx(74.0)


def test_playback_cadence_reset_and_invalid_timestamps_do_not_create_samples():
    tracker = PlaybackCadenceTracker()
    tracker.record_completion(2.0)
    tracker.record_completion(1.0)
    tracker.record_completion(float("nan"))
    assert tracker.frames_per_second() is None

    tracker.record_completion(2.05)
    assert tracker.frames_per_second() == pytest.approx(20.0)

    tracker.reset()
    assert tracker.frames_per_second() is None
    assert tracker.mean_interval_ms() is None
    assert tracker.percentile_interval_ms(95.0) is None


def test_playback_cadence_uses_a_bounded_recent_window():
    tracker = PlaybackCadenceTracker(max_samples=2)
    for timestamp in (0.0, 1.0, 1.5, 1.75):
        tracker.record_completion(timestamp)

    assert tracker.frames_per_second() == pytest.approx(2.0 / 0.75)
