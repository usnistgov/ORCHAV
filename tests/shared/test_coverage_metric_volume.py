"""Volume metrics use the canonical coverage formulas without reinterpretation."""

import h5py
import numpy as np
import pytest

from shared.coverage.hdf5 import CoverageHDF5Reader
from shared.coverage.volume import load_coverage_metric_volume
from tests.shared.test_coverage_volume import write_volume


@pytest.mark.parametrize(
    "metric",
    [
        "path_gain_linear/A",
        "path_gain_db/B",
        "path_loss_db/A",
        "rss_w/B",
        "rss_dbm/A",
        "best_path_loss_db",
        "best_rss_dbm",
        "serving_path_gain_linear",
        "sum_rss_dbm",
        "sinr_db",
        "sinr_db/A",
        "sinr_linear/B",
        "serving_tx",
        "tx_margin_db",
    ],
)
def test_same_values_as_ordinary_reader_for_all_metric_families(tmp_path, metric):
    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h5:
        h5.require_group("metadata").attrs["noise_power_w"] = 1e-12
    before = path.read_bytes()
    expected = CoverageHDF5Reader(path).read_metric(metric)
    volume = load_coverage_metric_volume(path, metric)
    np.testing.assert_array_equal(volume.values, expected)
    np.testing.assert_array_equal(volume.axes[0], [11, 13, 15, 17, 19])
    np.testing.assert_array_equal(volume.axes[2], [7, 11, 15])
    assert not volume.values.flags.writeable
    assert volume.transmitter_names == ("A", "B")
    assert path.read_bytes() == before


def test_volume_metric_reads_one_height_at_a_time(tmp_path, monkeypatch):
    path = write_volume(tmp_path / "coverage.h5")
    original = CoverageHDF5Reader.read_metric
    heights = []

    def observe(self, metric, **kwargs):
        heights.append(kwargs["height_indices"])
        return original(self, metric, **kwargs)

    monkeypatch.setattr(CoverageHDF5Reader, "read_metric", observe)
    load_coverage_metric_volume(path, "serving_tx")
    assert heights == [0, 1, 2]


def test_old_zero_contribution_is_not_plotted_as_clamped_minus270_dbm(tmp_path):
    from shared.coverage.volume import coverage_threshold_mask

    path = write_volume(tmp_path / "coverage.h5")
    model = load_coverage_metric_volume(path, "rss_dbm/A")
    # Keep the canonical reader's raw value, but its physical zero is separate
    # from a very weak finite observation and never passes a finite threshold.
    assert model.values[0, 0, 0] < -260
    assert model.zero_contributions[0, 0, 0]
    assert not coverage_threshold_mask(model, -300)[0, 0, 0]
    assert not model.zero_contributions[0, 0, 1]  # historical unknown


def test_generic_volume_rejects_contradictory_availability(tmp_path):
    from shared.coverage.availability import classify_solver_gain

    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h5:
        codes = classify_solver_gain(h5["values/path_gain_linear"][()])
        codes[0, 0, 0, 0, 2] = 2  # claim zero for a positive gain
        ds = h5.create_dataset("availability/path_gain", data=codes)
        ds.attrs["schema_version"] = 1
    with pytest.raises(ValueError):
        load_coverage_metric_volume(path, "rss_dbm/A")
