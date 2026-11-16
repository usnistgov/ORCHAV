"""Retained point reads and reproducible region-limited power comparisons."""

import numpy as np
import pytest

from shared.coverage.comparison import analyze_definition, file_identity, native_coverage_bundle
from shared.coverage.hdf5 import CoverageHDF5Reader
from tests.shared.test_coverage_volume import write_volume


def test_cell_read_uses_only_transmitter_vector_and_canonical_metrics(tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    seen = []
    reader = CoverageHDF5Reader(
        path, read_observer=lambda name, selection, shape: seen.append((name, selection, shape))
    )
    original = path.read_bytes()
    result = reader.read_cell((2, 1, 2))
    gains = [r for r in seen if r[0] == "/values/path_gain_linear"]
    assert len(gains) == 1 and gains[0][2] == (2,)
    assert gains[0][1] == (0, 2, slice(None), 1, 2)
    for index, name in enumerate(result["names"]):
        for metric in ("rss_dbm", "sinr_db"):
            expected = CoverageHDF5Reader(path).read_metric(f"{metric}/{name}", height_indices=2)[
                0, 1, 2
            ]
            assert result[metric][index] == pytest.approx(expected)
    assert path.read_bytes() == original
    zero = reader.read_cell((0, 0, 0))
    assert zero["availability"][0] == 2
    with pytest.raises(ValueError):
        reader.read_cell((-1, 0, 0))


def test_roi_comparison_keeps_same_samples_values_and_denominator(tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    full = native_coverage_bundle(path)
    bounds = [[11, 15], [-18.5, -15.5], [7, 11]]
    eligible = np.ones((3, 20), dtype=bool)
    selected = native_coverage_bundle(path, roi_bounds=bounds, eligible=eligible)
    assert eligible.all()  # Region analysis must not mutate caller-supplied support.
    assert selected["group_names"].tolist() == ["7 m", "11 m"]
    np.testing.assert_array_equal(selected["rss_dbm"], full["rss_dbm"][:2])
    assert selected["eligible"].sum() == 12
    definition = dict(
        schema_version=1,
        kind="coverage_comparison",
        thresholds_dbm=[-80],
        sources=[
            dict(
                path=str(path),
                sha256=file_identity(path)["sha256"],
                kind="native_coverage",
                label="ROI",
                roi_bounds_m=bounds,
            )
        ],
    )
    result = analyze_definition(definition)
    assert len(result["rows"]) == 2
    assert all(row["eligible_samples"] == 6 for row in result["rows"])


def test_cell_read_rejects_unknown_availability_schema(tmp_path):
    import h5py

    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h5:
        dataset = h5.create_dataset(
            "availability/path_gain", data=np.ones((1, 3, 2, 4, 5), dtype=np.uint8)
        )
        dataset.attrs["schema_version"] = 999
    with pytest.raises(ValueError, match="availability schema"):
        CoverageHDF5Reader(path).read_cell((2, 1, 2))
