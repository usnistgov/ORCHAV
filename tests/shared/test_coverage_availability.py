"""Zero solver contribution, invalid estimates and historical unknowns stay distinct."""

import h5py
import numpy as np
import pytest

from shared.coverage.availability import classify_solver_gain, validate_availability
from shared.coverage.volume import load_coverage_volume
from tests.shared.test_coverage_volume import write_volume


def test_raw_estimates_keep_zero_evidence_before_masking():
    raw = np.array([1e-5, 0, -1, np.nan, np.inf])
    codes = classify_solver_gain(raw)
    np.testing.assert_equal(codes, [1, 2, 3, 3, 3])
    old_gain = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    np.testing.assert_equal(validate_availability(codes, old_gain), codes)
    for bad in ([1, 1, 3, 3, 3], [0, 2, 3, 3, 3], [1, 255, 3, 3, 3]):
        with pytest.raises(ValueError):
            validate_availability(bad, old_gain)


def test_reader_preserves_old_unknown_and_reads_new_evidence(tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    old = load_coverage_volume(path, "A", "path_gain_db")
    assert old.availability[0, 0, 1] == 0
    with h5py.File(path, "r+") as f:
        gain = f["values/path_gain_linear"][()]
        codes = classify_solver_gain(gain)
        gain[gain == 0] = np.nan
        f["values/path_gain_linear"][...] = gain
        ds = f.create_dataset("availability/path_gain", data=codes)
        ds.attrs["schema_version"] = 1
    new = load_coverage_volume(path, "A", "path_gain_db")
    assert new.availability[0, 0, 0] == 2
    assert new.availability[0, 0, 1] == 3
    assert np.isneginf(new.values[0, 0, 0])
    assert np.isnan(new.values[0, 0, 1])
    with h5py.File(path, "r+") as f:
        f["availability/path_gain"][0, 0, 0, 0, 2] = 2
    with pytest.raises(ValueError, match="disagrees"):
        load_coverage_volume(path, "A", "path_gain_db")
