"""Tests for the FileProvider base class architecture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.frames.base import FormatHandler
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.providers import FileProvider, Hdf5Provider
from shared.frames.types import StandardMPCFrame


def _make_minimal_frame() -> StandardMPCFrame:
    """Create a minimal valid StandardMPCFrame for testing."""
    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        tx_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.array([[10.0, 0.0, 0.0]], dtype=np.float64),
        tx_names=("tx-0",),
        rx_names=("rx-0",),
        vertices_by_pair=[np.empty((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.array([0], dtype=np.int64)],
    )


class StubHandler(FormatHandler):
    """Minimal FormatHandler for testing FileProvider logic."""

    def __init__(self, source: Path, frames: dict[int, StandardMPCFrame]):
        super().__init__(source)
        self._frames = frames

    def can_handle(self) -> bool:
        return True

    def list_frames(self) -> list[int]:
        return sorted(self._frames)

    def has_frame(self, step: int) -> bool:
        return step in self._frames

    def load_frame(self, step: int) -> StandardMPCFrame:
        return self._frames[step]


class TestFileProviderBase:
    """Tests for the FileProvider base class using a stub handler."""

    def test_list_frames(self, tmp_path: Path):
        frame = _make_minimal_frame()
        handler = StubHandler(tmp_path, {0: frame, 3: frame})
        provider = FileProvider(tmp_path, handler)

        assert provider.list_frames() == [0, 3]

    def test_has_frame(self, tmp_path: Path):
        frame = _make_minimal_frame()
        handler = StubHandler(tmp_path, {0: frame})
        provider = FileProvider(tmp_path, handler)

        assert provider.has_frame(0) is True
        assert provider.has_frame(1) is False

    def test_load_frame_returns_complete_frame(self, tmp_path: Path):
        frame = _make_minimal_frame()
        handler = StubHandler(tmp_path, {0: frame})
        provider = FileProvider(tmp_path, handler)

        loaded = provider.load_frame(0)
        assert loaded.num_tx == 1

    def test_load_frame_rejects_noncanonical_handler_result(self, tmp_path: Path):
        handler = StubHandler(tmp_path, {0: {}})  # type: ignore[dict-item]
        provider = FileProvider(tmp_path, handler)

        with pytest.raises(TypeError, match="must return StandardMPCFrame"):
            provider.load_frame(0)

    def test_info_property(self, tmp_path: Path):
        frame = _make_minimal_frame()
        handler = StubHandler(tmp_path, {0: frame, 1: frame, 2: frame})
        provider = FileProvider(tmp_path, handler)

        info = provider.info
        assert info.name == "FileProvider"
        assert info.total_frames == 3
        assert str(tmp_path) in info.source

    def test_is_bulk_default(self, tmp_path: Path):
        handler = StubHandler(tmp_path, {})
        provider = FileProvider(tmp_path, handler)
        assert provider.is_bulk is False

    def test_bulk_files_default(self, tmp_path: Path):
        handler = StubHandler(tmp_path, {})
        provider = FileProvider(tmp_path, handler)
        assert provider.bulk_files == []


class TestHdf5ProviderDiscovery:
    """Tests for Hdf5Provider initialization and error handling."""

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Hdf5Provider(tmp_path / "nonexistent")

    def test_empty_frames_dir_raises(self, tmp_path: Path):
        """A current HDF5 frame set requires ``frames_manifest.json``."""
        (tmp_path / "frames").mkdir()
        with pytest.raises(FileNotFoundError):
            Hdf5Provider(tmp_path)
