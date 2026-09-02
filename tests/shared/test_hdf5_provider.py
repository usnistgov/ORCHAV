"""Provider-level tests for manifest-driven packed HDF5 v2 frame sets."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.frames.hdf5 import (
    HDF5FormatHandler,
    ScenarioFrameSetIntegrityError,
)
from shared.frames.providers import Hdf5Provider
from tests.shared.test_hdf5_format_handler_v2 import _write_minimal_packed_frame_set


class TestHdf5Provider:
    """Exercise provider behavior at the packed-v2 format boundary."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Hdf5Provider(tmp_path / "missing")

    def test_manifest_drives_listing_loading_and_metadata(self, tmp_path: Path) -> None:
        chunk_path = _write_minimal_packed_frame_set(tmp_path, frame_id=7)

        provider = Hdf5Provider(tmp_path)
        try:
            assert provider.list_frames() == [7]
            assert provider.has_frame(7)
            assert not provider.has_frame(8)
            assert provider.is_bulk
            assert provider.bulk_files == [str(chunk_path)]

            frame = provider.load_frame(7)
            assert frame.num_tx == 1
            assert frame.num_rx == 1

            info = provider.info
            assert info.total_frames == 1
            assert info.generation_id == "test-generation"
            assert info.frame_set_id == "test-frame-set"
        finally:
            provider.close()

    def test_missing_manifest_frame_raises_file_not_found(self, tmp_path: Path) -> None:
        _write_minimal_packed_frame_set(tmp_path)
        provider = Hdf5Provider(tmp_path)
        try:
            with pytest.raises(FileNotFoundError, match="not listed"):
                provider.load_frame(99)
        finally:
            provider.close()

    def test_non_manifest_index_and_chunk_are_not_a_runtime_frame_set(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frames_index.json").write_text("{}", encoding="utf-8")
        (frames_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        (frames_dir / "mpc_frames_00000-00000.h5").touch()

        handler = HDF5FormatHandler(tmp_path)
        assert not handler.can_handle()
        assert handler.list_frames() == []
        assert handler.bulk_files == []
        assert handler.generation_id is None
        assert handler.frame_set_id is None

        with pytest.raises(FileNotFoundError):
            Hdf5Provider(tmp_path)

    def test_invalid_v2_manifest_uses_public_integrity_exception(
        self,
        tmp_path: Path,
    ) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frames_manifest.json").write_text("{invalid", encoding="utf-8")

        with pytest.raises(ScenarioFrameSetIntegrityError, match="Could not read"):
            HDF5FormatHandler(tmp_path)

    def test_refresh_detects_new_manifest_and_reopens_after_close(
        self,
        tmp_path: Path,
    ) -> None:
        handler = HDF5FormatHandler(tmp_path)
        assert not handler.can_handle()

        _write_minimal_packed_frame_set(tmp_path)
        handler.refresh()
        assert handler.can_handle()
        assert handler.load_frame(4).num_tx == 1
        assert handler._packed_reader is not None
        assert handler._packed_reader.open_handle_count == 1

        handler.close()
        assert handler._packed_reader.open_handle_count == 0

        handler.refresh()
        try:
            assert handler.list_frames() == [4]
            assert handler.load_frame(4).num_rx == 1
        finally:
            handler.close()
