"""Independent coordinate, signal and allocation contracts for coverage volumes."""

import h5py
import numpy as np
import pytest

from shared.coverage.volume import ALL_TRANSMITTERS, display_samples, load_coverage_volume


def write_volume(path):
    """Write asymmetric translated planes with exact known gains for two TXs."""
    gains = np.full((1, 3, 2, 4, 5), 1e-9, dtype=np.float32)
    gains[0, 0, 0, 0, :3] = [0, np.nan, 1e-6]
    with h5py.File(path, "w") as h:
        h.attrs["coverage_schema_version"] = 2
        h.attrs["coverage_storage_layout"] = "canonical_derived_v2"
        h.create_dataset("values/path_gain_linear", data=gains)
        h.create_dataset("grid/shape_xyz", data=[5, 4, 3])
        h.create_dataset("grid/origin", data=[10, -20, 7])
        h.create_dataset("grid/spacing_xy", data=[2, 3])
        h.create_dataset("grid/heights_m", data=[7, 11, 15])
        h.create_dataset("tx/names", data=[b"A", b"B"])
        h.create_dataset("tx/powers_dbm", data=[0, 23])
    return path


def test_exact_centres_signal_units_and_zero_unknown_distinction(tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    volume = load_coverage_volume(path, "A", "path_gain_db")
    np.testing.assert_equal(volume.axes[0], [11, 13, 15, 17, 19])
    np.testing.assert_equal(volume.axes[1], [-18.5, -15.5, -12.5, -9.5])
    np.testing.assert_equal(volume.axes[2], [7, 11, 15])
    assert volume.spacing == (2, 3, 4)
    assert np.isneginf(volume.values[0, 0, 0])
    assert np.isnan(volume.values[0, 0, 1])
    assert volume.values[0, 0, 2] == pytest.approx(-60)
    assert not volume.values.flags.writeable
    rss = load_coverage_volume(path, "B", "rss_dbm")
    np.testing.assert_allclose(rss.values, -67, atol=1e-5)
    samples = display_samples(volume, -100, -70)
    assert samples[0, 0, 0] == 0  # Known zero stays blank, not a weak finite power.
    assert samples[0, 0, 1] == 0
    assert samples[0, 0, 2] == 256
    region = display_samples(volume, -100, -70, threshold=-80)
    assert np.count_nonzero(region) == 1
    assert region[0, 0, 2] == 256
    with pytest.raises(ValueError, match="minimum"):
        display_samples(volume, -70, -100)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("irregular", "uniformly"),
        ("oversize", "2,000,000"),
        ("wrong_shape", "dimensions"),
    ],
)
def test_reject_before_dense_reads(tmp_path, monkeypatch, mutation, reason):
    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h:
        if mutation == "irregular":
            h["grid/heights_m"][...] = [7, 11, 16]
        elif mutation == "oversize":
            h["grid/shape_xyz"][...] = [10000, 10000, 3]
        else:
            h["grid/shape_xyz"][...] = [6, 4, 3]

    def fail_read(*args, **kwargs):
        raise AssertionError("Dense read before validating shape")

    monkeypatch.setattr("shared.coverage.volume.CoverageHDF5Reader.read_metric", fail_read)
    with pytest.raises(ValueError, match=reason):
        load_coverage_volume(path, "A", "path_gain_db")


def test_combined_volume_selects_power_not_gain_and_preserves_unknown(tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h:
        h["values/path_gain_linear"][0, 0, :, 1, 0] = [0, 0]
        h["tx/powers_dbm"][1] = 42
    original = path.read_bytes()
    power = load_coverage_volume(path, ALL_TRANSMITTERS, "rss_dbm")
    gain = load_coverage_volume(path, ALL_TRANSMITTERS, "path_gain_db")
    # A has stronger gain here, but B's 42 dBm power makes its link stronger.
    assert gain.values[0, 0, 2] == pytest.approx(-60)
    assert power.values[0, 0, 2] == pytest.approx(-48)
    assert np.isnan(power.values[0, 0, 1])  # A's estimate is unknown.
    assert np.isneginf(power.values[0, 1, 0])  # Both recorded contributions zero.
    assert power.zero_contributions[0, 1, 0]
    assert not power.values.flags.writeable
    assert path.read_bytes() == original


def test_path_loss_uses_power_winner_and_reversed_threshold(tmp_path):
    from shared.coverage.volume import coverage_threshold_mask

    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h:
        h["tx/powers_dbm"][1] = 42
        h["values/path_gain_linear"][0, 0, :, 1, 0] = [0, 0]
    original = path.read_bytes()
    single = load_coverage_volume(path, "A", "path_loss_db")
    combined = load_coverage_volume(path, ALL_TRANSMITTERS, "path_loss_db")
    assert single.values[0, 0, 2] == pytest.approx(60)
    assert np.isposinf(single.values[0, 0, 0])
    assert np.isnan(single.values[0, 0, 1])
    assert combined.values[0, 0, 2] == pytest.approx(90)  # B wins by power, not gain.
    assert np.isnan(combined.values[0, 0, 1])
    assert combined.zero_contributions[0, 1, 0]
    assert coverage_threshold_mask(single, 80).sum() == 1
    assert np.count_nonzero(display_samples(single, 50, 100, threshold=80)) == 1
    assert path.read_bytes() == original
