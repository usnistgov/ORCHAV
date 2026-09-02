"""Selective-read tests for compact coverage HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from shared.coverage.hdf5 import CoverageHDF5Reader
from shared.coverage.schema import (
    COVERAGE_HDF5_SCHEMA_VERSION,
    COVERAGE_HDF5_STORAGE_LAYOUT,
    derive_coverage_metric_layer,
)


def _write_coverage_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path_gain = (np.arange(1 * 3 * 2 * 4 * 5, dtype=np.float32).reshape(1, 3, 2, 4, 5) + 1.0) * 1e-9
    serving_tx = np.asarray(
        [
            [
                np.zeros((4, 5), dtype=np.int8),
                np.ones((4, 5), dtype=np.int8),
                np.zeros((4, 5), dtype=np.int8),
            ]
        ]
    )
    with h5py.File(path, "w") as h5:
        h5.attrs["coverage_schema_version"] = COVERAGE_HDF5_SCHEMA_VERSION
        h5.attrs["coverage_storage_layout"] = COVERAGE_HDF5_STORAGE_LAYOUT
        values = h5.create_group("values")
        values.create_dataset(
            "path_gain_linear",
            data=path_gain,
            chunks=(1, 1, 1, 4, 5),
        )
        derived = h5.create_group("derived")
        derived.create_dataset("serving_tx", data=serving_tx, chunks=(1, 1, 4, 5))
        tx = h5.create_group("tx")
        tx.create_dataset("names", data=np.asarray([b"TX1", b"TX2"]))
        tx.create_dataset("powers_dbm", data=np.asarray([0.0, 3.0], dtype=np.float32))
        metadata = h5.create_group("metadata")
        metadata.attrs["noise_power_w"] = 1e-12
    return path_gain, serving_tx


def _canonical_reads(
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]],
) -> list[tuple[str, tuple[Any, ...], tuple[int, ...]]]:
    return [read for read in reads if read[0] == "/values/path_gain_linear"]


def test_path_gain_single_height_and_tx_uses_one_hyperslab(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []

    selected = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    ).read_path_gain(height_indices=2, transmitter_indices=1)

    np.testing.assert_array_equal(selected, path_gain[:, 2:3, 1:2, :, :])
    assert _canonical_reads(reads) == [
        (
            "/values/path_gain_linear",
            (
                slice(0, 1),
                slice(2, 3),
                slice(1, 2),
                slice(None),
                slice(None),
            ),
            (1, 1, 1, 4, 5),
        )
    ]
    assert selected.nbytes == path_gain[:, 2:3, 1:2, :, :].nbytes
    assert selected.nbytes < path_gain.nbytes


def test_per_tx_recipe_reads_only_requested_height_and_tx(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []
    reader = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    )

    selected = reader.read_metric("rss_w/TX2", height_indices=[1])
    expected = derive_coverage_metric_layer(
        path_gain,
        "rss_w/TX2",
        tx_power_dbm=np.asarray([0.0, 3.0], dtype=np.float32),
        noise_power_w=1e-12,
        tx_names=["TX1", "TX2"],
    )[1:2]

    np.testing.assert_allclose(selected, expected)
    assert _canonical_reads(reads) == [
        (
            "/values/path_gain_linear",
            (
                slice(0, 1),
                slice(1, 2),
                slice(1, 2),
                slice(None),
                slice(None),
            ),
            (1, 1, 1, 4, 5),
        )
    ]


def test_scalar_recipe_reads_requested_height_but_all_required_txs(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []
    reader = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    )

    selected = reader.read_metric("sinr_db", height_indices=0)
    expected = derive_coverage_metric_layer(
        path_gain,
        "sinr_db",
        tx_power_dbm=np.asarray([0.0, 3.0], dtype=np.float32),
        noise_power_w=1e-12,
        tx_names=["TX1", "TX2"],
    )[0:1]

    np.testing.assert_allclose(selected, expected)
    assert _canonical_reads(reads) == [
        (
            "/values/path_gain_linear",
            (
                slice(0, 1),
                slice(0, 1),
                slice(None),
                slice(None),
                slice(None),
            ),
            (1, 1, 2, 4, 5),
        )
    ]


@pytest.mark.parametrize("metric_name", ["sinr_linear/TX2", "sinr_db/TX2"])
def test_selected_tx_sinr_reads_all_txs_and_uses_others_as_interference(
    tmp_path: Path,
    metric_name: str,
) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []
    reader = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    )

    selected = reader.read_metric(metric_name, height_indices=1)
    tx1_power_w = np.float32(1e-3)
    tx2_power_w = np.float32(10.0 ** ((3.0 - 30.0) / 10.0))
    signal = path_gain[0, 1, 1] * tx2_power_w
    interference = path_gain[0, 1, 0] * tx1_power_w
    expected = signal / (interference + np.float32(1e-12))
    if metric_name.startswith("sinr_db"):
        expected = 10.0 * np.log10(np.maximum(expected, 1e-30))

    np.testing.assert_allclose(selected[0], expected, rtol=1e-6, atol=0.0)
    assert _canonical_reads(reads) == [
        (
            "/values/path_gain_linear",
            (
                slice(0, 1),
                slice(1, 2),
                slice(None),
                slice(None),
                slice(None),
            ),
            (1, 1, 2, 4, 5),
        )
    ]


def test_best_server_sinr_matches_selected_serving_tx_at_high_dynamic_range() -> None:
    path_gain = np.asarray(
        [[[[[1.0]], [[1.0e-20]], [[2.0e-20]]]]],
        dtype=np.float32,
    )
    kwargs = {
        "tx_power_dbm": np.zeros(3, dtype=np.float32),
        "noise_power_w": 1.0e-30,
        "tx_names": ["Serving", "Weak1", "Weak2"],
    }

    best_server = derive_coverage_metric_layer(path_gain, "sinr_linear", **kwargs)
    selected_server = derive_coverage_metric_layer(
        path_gain,
        "sinr_linear/Serving",
        **kwargs,
    )

    np.testing.assert_allclose(best_server, selected_server, rtol=1.0e-6, atol=0.0)
    assert best_server.item() == pytest.approx(1.0 / 3.0e-20, rel=1.0e-6)


def test_serving_path_gain_reads_all_txs_and_follows_strongest_rss(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    _write_coverage_file(coverage_file)
    with h5py.File(coverage_file, "r+") as h5:
        path_gain = h5["values/path_gain_linear"]
        path_gain[:, 0, 0, :, :] = 2e-9
        path_gain[:, 0, 1, :, :] = 1.5e-9

    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []
    selected = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    ).read_metric("serving_path_gain_linear", height_indices=0)

    # TX2 has lower raw gain, but its 3 dBm power makes its received power
    # stronger than TX1 at 0 dBm.
    np.testing.assert_allclose(selected, np.full((1, 4, 5), 1.5e-9, dtype=np.float32))
    assert _canonical_reads(reads) == [
        (
            "/values/path_gain_linear",
            (
                slice(0, 1),
                slice(0, 1),
                slice(None),
                slice(None),
                slice(None),
            ),
            (1, 1, 2, 4, 5),
        )
    ]


def test_materialized_metric_does_not_read_canonical_tensor(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    _, serving_tx = _write_coverage_file(coverage_file)
    reads: list[tuple[str, tuple[Any, ...], tuple[int, ...]]] = []
    reader = CoverageHDF5Reader(
        coverage_file,
        read_observer=lambda name, selection, shape: reads.append((name, selection, shape)),
    )

    selected = reader.read_metric("serving_tx", height_indices=2)

    np.testing.assert_array_equal(selected, serving_tx[0, 2:3])
    assert _canonical_reads(reads) == []
    assert reads == [
        (
            "/derived/serving_tx",
            (
                slice(0, 1),
                slice(2, 3),
                slice(None),
                slice(None),
            ),
            (1, 1, 4, 5),
        )
    ]


def test_unbounded_metric_read_preserves_full_height_result(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)

    selected = CoverageHDF5Reader(coverage_file).read_metric("best_path_loss_db")
    expected = derive_coverage_metric_layer(
        path_gain,
        "best_path_loss_db",
        tx_power_dbm=np.asarray([0.0, 3.0], dtype=np.float32),
        noise_power_w=1e-12,
        tx_names=["TX1", "TX2"],
    )

    np.testing.assert_allclose(selected, expected)
    assert selected.shape == (3, 4, 5)


def test_sum_power_recipe_preserves_cells_without_finite_coverage(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    _write_coverage_file(coverage_file)
    with h5py.File(coverage_file, "r+") as h5:
        h5["values/path_gain_linear"][:, :, :, 0, 0] = np.nan

    selected = CoverageHDF5Reader(coverage_file).read_metric("sum_rss_dbm")

    assert selected.shape == (3, 4, 5)
    assert np.isnan(selected[:, 0, 0]).all()
    assert np.isfinite(selected[:, 0, 1:]).all()


@pytest.mark.parametrize(
    "metric_name",
    [
        "path_gain_linear/TX2",
        "path_gain_db/TX1",
        "path_loss_db/TX2",
        "rss_w/TX1",
        "rss_dbm/TX2",
        "best_path_loss_db",
        "best_rss_dbm",
        "sum_rss_dbm",
        "serving_path_gain_linear",
        "sinr_linear",
        "sinr_db",
        "sinr_linear/TX1",
        "sinr_db/TX2",
        "tx_margin_db",
    ],
)
def test_selective_metric_formulas_match_existing_numeric_semantics(
    tmp_path: Path,
    metric_name: str,
) -> None:
    coverage_file = tmp_path / "coverage.h5"
    path_gain, _ = _write_coverage_file(coverage_file)

    selected = CoverageHDF5Reader(coverage_file).read_metric(
        metric_name,
        height_indices=[2, 0],
    )
    expected = derive_coverage_metric_layer(
        path_gain,
        metric_name,
        tx_power_dbm=np.asarray([0.0, 3.0], dtype=np.float32),
        noise_power_w=1e-12,
        tx_names=["TX1", "TX2"],
    )[[2, 0]]

    np.testing.assert_allclose(selected, expected, rtol=1e-6, atol=0.0, equal_nan=True)


def test_selective_reader_rejects_out_of_range_axes(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.h5"
    _write_coverage_file(coverage_file)
    reader = CoverageHDF5Reader(coverage_file)

    with np.testing.assert_raises_regex(IndexError, "coverage height index 3"):
        reader.read_path_gain(height_indices=3)
    with np.testing.assert_raises_regex(IndexError, "coverage transmitter index 2"):
        reader.read_path_gain(transmitter_indices=2)
