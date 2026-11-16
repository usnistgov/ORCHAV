"""Load retained sequential CFR into the actual Qt inspection/history views."""

import numpy as np

from shared.sensing.loader import load_frames_sensing
from tests.shared.test_sensing_postprocessor import _make_standard_frame, _write_packed_frame_set
from tests.shared.test_sensing_sequential_cfr import (
    bind_results,
    processor_for,
    pulses_for,
    run_pulses,
)
from visualizer.src.sensing.cir_provider import extract_timestamped_fast_time_samples
from visualizer.src.sensing.processing_inspection import inspect_processing
from visualizer.src.sensing.processing_panel import ProcessingPanel


def test_sequential_recording_loads_frequency_and_advances_history_once(qtbot, tmp_path):
    processor = processor_for()
    results = bind_results(processor, run_pulses(processor, pulses_for("oscillating", 17)))
    _write_packed_frame_set(
        tmp_path,
        *[_make_standard_frame(frame_idx=i, sensing=results[i]) for i in (15, 16)],
    )
    loaded = [frame["_typed_sensing"] for frame in load_frames_sensing(tmp_path)]
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.show()
    for i, sensing in enumerate(loaded):
        model = inspect_processing(sensing, "frequency")
        np.testing.assert_array_equal(model.samples, sensing["cfr_by_pair_rx_element"][0])
        panel.set_model(model, "frequency")
        panel.range_bin.setValue(3)
        qtbot.wait(10)
        np.testing.assert_allclose(panel.before_curve.yData, abs(model.samples[0, 3]))
        batch = extract_timestamped_fast_time_samples(sensing)
        assert batch.alignment_error is None
        # The service advances by one acquired pulse, even though the saved
        # frequency and delay products each contain the full overlapping window.
        assert len(batch.samples) == 1
        np.testing.assert_allclose(batch.sample_times_s, [2 + (15 + i) / 1000])
        np.testing.assert_array_equal(
            batch.samples[0], sensing["slow_time_by_pair_rx_element"][0, 0, -1]
        )
    panel.release()
