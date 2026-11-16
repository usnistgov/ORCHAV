from __future__ import annotations

from generator.core.sensing.timing import merge_sensing_timing_metadata


def test_merge_sensing_timing_metadata_preserves_stage_profile() -> None:
    base_timing = {
        "frame_dt_s": 0.1,
        "pri_s": 0.01,
        "coherent_samples_per_cpi": 4,
        "cpi_duration_s": 0.04,
        "revisit_interval_s": 0.1,
        "cpi_index": 2,
        "pri_index_in_cpi": 3,
        "timing_role": "cpi_end",
    }
    extra_timing = {
        "stage_profile_ms": {
            "cir_projection": 1.25,
            "rd_fft": 2.5,
        }
    }

    merged = merge_sensing_timing_metadata(base_timing, extra_timing)

    assert merged["frame_dt_s"] == 0.1
    assert merged["timing_role"] == "cpi_end"
    assert merged["stage_profile_ms"]["cir_projection"] == 1.25
    assert merged["stage_profile_ms"]["rd_fft"] == 2.5
