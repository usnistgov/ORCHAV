"""Retained synthetic evidence must use normal storage and fail hash tampering."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from shared.frames.providers import Hdf5Provider
from shared.scenarios import load_scenario_configuration
from shared.sensing.loader import load_frames_sensing
from shared.sensing.volume.reader import SensingVolume


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_trial_uses_canonical_provider_truth_and_verified_hashes(tmp_path):
    root = Path(__file__).resolve().parents[3]
    scripts = root / "scenarios/visualizer/volume_cell_inspection"
    demo = _load(scripts / "demo.py", "volume_qa_demo")
    retained = _load(scripts / "retained_trial.py", "volume_qa_retained")
    demo.prepare(tmp_path)
    result = retained.prepare_retained_trial(tmp_path)
    reader = SensingVolume(tmp_path)
    native = reader.verify_native((0, 0, 0, 2))
    assert native == Path(result["native"])
    assert load_scenario_configuration(native, project_root=root).sensing["enabled"] is True
    provider = Hdf5Provider(native)
    try:
        frame = provider.load_frame(0)
        assert frame.sensing["detections"][0]["detection_id"] == "controlled-report"
        assert frame.sensing["range_doppler_power_by_pair"].shape == (1, 9, 17)
        assert frame.sensing["range_doppler_power_by_pair"][0, 4, 8] == 1.0
        truth = frame.sensing["truth"]["targets"][0]
        assert truth["target_id"] == "controlled-volume-target"
        np.testing.assert_array_equal(truth["position_m"], [14, -18, 7])
        np.testing.assert_array_equal(frame.rx_positions, [[7, -12, 4]])
    finally:
        provider.close()
    assert len(load_frames_sensing(native)) == 1
    chunk = next((native / "frames").glob("*.h5"))
    with chunk.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        reader.verify_native((0, 0, 0, 2))
