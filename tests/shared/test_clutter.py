from __future__ import annotations

import numpy as np

from shared.sensing.detection.clutter import SlidingBaselineRemover


def test_sliding_baseline_remover_tracks_previous_frames():
    remover = SlidingBaselineRemover(window=2, clip_zero=True)

    sample1 = np.array([1.0, 2.0], dtype=np.float32)
    result1 = remover.process(sample1)
    assert result1.frames_used == 0
    np.testing.assert_allclose(result1.baseline, np.zeros_like(sample1))
    np.testing.assert_allclose(result1.cleaned, sample1)

    sample2 = np.array([2.0, 1.0], dtype=np.float32)
    result2 = remover.process(sample2)
    assert result2.frames_used == 1
    np.testing.assert_allclose(result2.baseline, sample1)
    np.testing.assert_allclose(result2.cleaned, np.clip(sample2 - sample1, 0.0, None))


def test_sliding_baseline_contract_exposes_warmup_delay_and_nonlinearity():
    remover = SlidingBaselineRemover(window=3, clip_zero=True)

    first = remover.process(np.array([2.0]))
    assert first.processing["history_samples_used"] == 0
    assert first.processing["transient_length_updates"] == 3
    assert first.processing["output_latency_samples"] == 0
    assert first.processing["steady_state"] is False
    assert first.processing["complete_operator_lti"] is False

    remover.process(np.array([2.0]))
    remover.process(np.array([2.0]))
    steady = remover.process(np.array([2.0]))
    np.testing.assert_allclose(steady.cleaned, [0.0])
    assert steady.processing["history_delay_samples"] == [1, 3]
    assert steady.processing["steady_state"] is True


def test_unclipped_sliding_baseline_matches_independent_tone_response():
    window = 4
    frequency = 0.125
    remover = SlidingBaselineRemover(window=window, clip_zero=False)
    samples = np.exp(2j * np.pi * frequency * np.arange(64))
    outputs = np.asarray([remover.process(np.asarray(value)).cleaned for value in samples])

    delays = np.arange(1, window + 1, dtype=np.float64)
    expected_response = 1.0 - np.mean(np.exp(-2j * np.pi * frequency * delays))
    np.testing.assert_allclose(
        outputs[-16:] / samples[-16:],
        expected_response,
        atol=1e-13,
    )


def test_sliding_baseline_accelerating_sequence_uses_only_prior_samples():
    remover = SlidingBaselineRemover(window=2, clip_zero=False)
    sequence = np.arange(6, dtype=np.float64) ** 2
    outputs = [float(remover.process(np.asarray(value)).cleaned) for value in sequence]

    assert outputs[0] == 0.0  # first-sample pass-through for x[0] == 0
    assert outputs[1] == 1.0  # x[1] - x[0]
    assert outputs[4] == 16.0 - (4.0 + 9.0) / 2.0
