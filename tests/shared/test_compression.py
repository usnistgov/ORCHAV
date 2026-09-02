"""Tests for HDF5 compression profiles."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from shared.frames.contracts import PathMetric
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_data(
    *,
    frame_index: int = 0,
    tx_x: float = 0.0,
) -> StandardMPCFrame:
    """Create a complete compact frame for compression tests."""

    metrics = {
        PathMetric.DELAY_NS: [np.asarray([12.5], dtype=np.float32)],
        PathMetric.PATH_LOSS_DB: [np.asarray([85.3], dtype=np.float32)],
        PathMetric.AOA_AZ_DEG: [np.asarray([45.0], dtype=np.float32)],
        PathMetric.AOA_EL_DEG: [np.asarray([10.0], dtype=np.float32)],
        PathMetric.AOD_AZ_DEG: [np.asarray([225.0], dtype=np.float32)],
        PathMetric.AOD_EL_DEG: [np.asarray([-5.0], dtype=np.float32)],
    }
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[tx_x, 0.0, 5.0]], dtype=np.float64),
        rx_positions=np.asarray([[10.0, 0.0, 1.5]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        vertices_by_pair=[
            np.asarray(
                [[[5.0, 0.0, 3.0], [7.0, 1.0, 2.0]]],
                dtype=np.float32,
            )
        ],
        interactions_by_pair=[np.asarray([[1, 1]], dtype=np.uint8)],
        path_lengths_by_pair=[np.asarray([2], dtype=np.int64)],
        material_names_by_pair=[np.asarray([["mat-itu_concrete", "mat-itu_glass"]])],
        material_itu_types_by_pair=[np.asarray([["concrete", "glass"]])],
        metrics_by_pair=metrics,
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        provenance={"provider": "test", "frame_idx": frame_index},
    )


def _write_packed_frame_set(
    scenario_root: Path,
    frames: list[StandardMPCFrame],
    *,
    compression: str,
) -> Path:
    """Write one provider-readable packed-v2 chunk plus its manifest."""
    frames_dir = scenario_root / "frames"
    chunk = write_packed_mpc_frame_chunk(
        frames_dir,
        frames,
        generation_id="compression-test-generation",
        compression=compression,
    )
    manifest = manifest_from_chunks(
        generation_id="compression-test-generation",
        frame_set_id="compression-test-frame-set",
        chunks=[chunk],
        compression={"codec": compression},
        segmentation={"max_frames": len(frames)},
        provenance={"fixture": "compression-tests"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir / chunk.file


# ---------------------------------------------------------------------------
# Task 1: Compression profile resolution
# ---------------------------------------------------------------------------


class TestHDF5CompressionRoundtrip:
    """Verify supported v2 profiles produce provider-readable frame sets."""

    @pytest.mark.parametrize("profile", ["fast", "balanced", "compact"])
    def test_roundtrip_per_profile(self, tmp_path: Path, profile: str):
        frame_data = _make_frame_data()
        scenario_root = tmp_path / "scenario"

        fpath = _write_packed_frame_set(
            scenario_root,
            [frame_data],
            compression=profile,
        )
        assert fpath.exists()

        # Verify it can be read back via the provider
        provider = Hdf5Provider(scenario_root)
        try:
            frame = provider.load_frame(0)

            np.testing.assert_allclose(frame.tx_positions, np.array([[0.0, 0.0, 5.0]]))
            np.testing.assert_allclose(frame.rx_positions, np.array([[10.0, 0.0, 1.5]]))
            assert frame.num_tx == 1
            assert frame.num_rx == 1
            np.testing.assert_allclose(
                frame.bounce_xyz_m,
                np.array([[5.0, 0.0, 3.0], [7.0, 1.0, 2.0]]),
            )
        finally:
            provider.close()

    @pytest.mark.parametrize(
        ("profile", "expected_filter", "expected_level"),
        [
            ("fast", None, None),
            ("none", None, None),
            ("balanced", "lzf", None),
            ("lzf", "lzf", None),
            ("compact", "gzip", 4),
            ("gzip", "gzip", 4),
            ("gzip-4", "gzip", 4),
            ("Balanced", "lzf", None),
        ],
    )
    def test_packed_write_per_profile(
        self,
        tmp_path: Path,
        profile: str,
        expected_filter: str | None,
        expected_level: int | None,
    ):
        """Verify every supported spelling configures packed numeric datasets."""
        frame_data = _make_frame_data()
        scenario_root = tmp_path / "scenario"
        fpath = _write_packed_frame_set(
            scenario_root,
            [frame_data],
            compression=profile,
        )

        with h5py.File(fpath, "r") as f:
            assert f.attrs["storage_layout"] == "packed_ragged_v2"
            dataset = f["bounces/xyz_m"]
            assert dataset.compression == expected_filter
            assert dataset.compression_opts == expected_level

    @pytest.mark.parametrize("profile", ["maximum", "gzip-9", "szip", "nonexistent"])
    def test_unsupported_profile_is_rejected(self, tmp_path: Path, profile: str):
        with pytest.raises(ValueError, match="compression must be one of"):
            _write_packed_frame_set(
                tmp_path / "scenario",
                [_make_frame_data()],
                compression=profile,
            )

    def test_fast_produces_no_compression(self, tmp_path: Path):
        """Confirm 'fast' profile writes without any filter."""
        fpath = _write_packed_frame_set(
            tmp_path / "scenario",
            [_make_frame_data()],
            compression="fast",
        )

        with h5py.File(fpath, "r") as f:
            ds = f["frames/tx_position_m"]
            assert ds.compression is None

    def test_compact_produces_gzip(self, tmp_path: Path):
        """Confirm 'compact' profile writes gzip-4."""
        fpath = _write_packed_frame_set(
            tmp_path / "scenario",
            [_make_frame_data()],
            compression="compact",
        )

        with h5py.File(fpath, "r") as f:
            ds = f["bounces/xyz_m"]
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4

    def test_multi_frame_packed_chunk_is_provider_readable(self, tmp_path: Path):
        scenario_root = tmp_path / "scenario"

        first = _make_frame_data(frame_index=0)
        second = _make_frame_data(frame_index=1, tx_x=1.0)

        _write_packed_frame_set(
            scenario_root,
            [first, second],
            compression="fast",
        )

        provider = Hdf5Provider(scenario_root)
        try:
            assert provider.list_frames() == [0, 1]
            frame = provider.load_frame(1)
            np.testing.assert_allclose(frame.tx_positions, np.array([[1.0, 0.0, 5.0]]))
        finally:
            provider.close()
