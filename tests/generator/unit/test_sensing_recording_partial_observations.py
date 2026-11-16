"""Canonical warm-up observations remain real samples, not completed results."""

from copy import deepcopy

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from shared.extensions.sensing import finalize_sensing_v4_publication
from shared.frames.frame_set_writer import FrameSetWriter
from shared.sensing.delay import extract_delay_domain_samples
from shared.sensing.motion_analysis import (
    MotionRecipe,
    _partial_canonical_delay_observation,
    compute_motion_spectra,
    load_motion_recording,
)
from tests.generator.unit.test_sensing_processor import _create_processor, _FakePaths
from tests.shared.test_sensing_postprocessor import _make_standard_frame
from tests.shared.test_sensing_sequential_cfr import (
    bind_results,
    processor_for,
    pulses_for,
    run_pulses,
)


def delay_run(profile, count, *, noise):
    cfg = SensingConfig(
        enabled=True,
        recording_profile=profile,
        processing_mode="sequential_stft",
        output_detections=False,
        bandwidth=100e6,
        prf_hz=1000,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="always",
        noise_enabled=noise,
        noise_snr_db=10 if noise else None,
        rng_seed=42,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)
    paths = _FakePaths([np.exp(0.4j * i) for i in range(count)], delay_s=20e-9)
    rows = [processor.process(None, i, paths, source_time_s=2 + i / 1000) for i in range(count)]
    bind_results(processor, rows)
    for row in rows:
        finalize_sensing_v4_publication(row)
    return rows


@pytest.mark.parametrize("profile", ["reprocessing", "full"])
@pytest.mark.parametrize("count", [3, 6])
@pytest.mark.parametrize("noise", [False, True])
def test_partial_delay_observations_roundtrip_without_duplicate_samples(
    tmp_path, profile, count, noise
):
    reference = delay_run(None, count, noise=noise)
    rows = delay_run(profile, count, noise=noise)
    canonical = "slow_time_by_pair_rx_element"
    for i, row in enumerate(rows):
        assert row["recording"]["canonical_observation"] == canonical
        assert "cir_by_pair_rx_element" not in row
        assert sum(d["representation"] == "complex_voltage" for d in row["products"].values()) == 1
        if i < 3:
            assert "result" not in row["temporal"]
            assert extract_delay_domain_samples(row) is None
            assert row[canonical].shape == (1, 1, 1, 16)
            np.testing.assert_array_equal(
                row[canonical][0, 0, 0], reference[i]["cir_by_pair_rx_element"][0, 0]
            )
            assert row["products"][canonical]["temporal_support"] == {
                "type": "acquisition_sample",
                "selection_policy": "last_valid_acquisition_sample",
                "sample_index": i,
            }
        else:
            assert row[canonical].shape == (1, 1, 4, 16)
            np.testing.assert_array_equal(row[canonical], reference[i][canonical])
    directory = tmp_path / "recording"
    with FrameSetWriter.create_new(directory, compression="lzf") as writer:
        for i, row in enumerate(rows):
            writer.append(_make_standard_frame(frame_idx=i, sensing=row))
        writer.finalize()
    samples, times, ranges, _ = load_motion_recording(directory)
    np.testing.assert_array_equal(
        samples, np.stack([r["cir_by_pair_rx_element"][0, 0] for r in reference])
    )
    np.testing.assert_array_equal(times, 2 + np.arange(count) / 1000)
    result = compute_motion_spectra(
        samples, times, ranges, MotionRecipe(window_samples=3, hop_samples=1)
    )
    assert len(result["time_s"]) == count - 2
    corrupted = deepcopy(rows[0])
    corrupted["temporal"]["acquisition"]["samples"][0]["sample_time_s"] += 0.5
    with pytest.raises(ValueError, match="exact last acquired"):
        _partial_canonical_delay_observation(corrupted)


def test_short_canonical_cfr_uses_the_same_explicit_motion_path(tmp_path):
    processor = processor_for(samples=4, recording_profile="reprocessing")
    rows = bind_results(processor, run_pulses(processor, pulses_for("approaching", 3)))
    directory = tmp_path / "frequency"
    with FrameSetWriter.create_new(directory, compression="lzf") as writer:
        for i, row in enumerate(rows):
            writer.append(_make_standard_frame(frame_idx=i, sensing=row))
        writer.finalize()
    samples, times, _, provenance = load_motion_recording(directory)
    assert samples.shape == (3, 16)
    np.testing.assert_array_equal(times, 2 + np.arange(3) / 1000)
    assert provenance["source_kind"] == "baseband_frequency_observation"
    assert all("result" not in r["temporal"] for r in rows)
