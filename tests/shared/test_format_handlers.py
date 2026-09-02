"""Tests for the FormatHandler ABC and packed-v2 HDF5 handler."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.frames.base import FormatHandler
from shared.frames.hdf5 import HDF5FormatHandler
from tests.shared.test_hdf5_format_handler_v2 import _write_minimal_packed_frame_set


class TestFormatHandlerABC:
    """Test that FormatHandler enforces its abstract interface."""

    def test_cannot_instantiate_abc(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            FormatHandler(tmp_path)

    def test_subclass_must_implement_all_methods(self) -> None:
        class IncompleteHandler(FormatHandler):
            def can_handle(self) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteHandler(Path("."))


class TestHDF5FormatHandler:
    """Test manifest-only discovery and listing."""

    def test_can_handle_false_for_missing_dir(self, tmp_path: Path) -> None:
        handler = HDF5FormatHandler(tmp_path)
        assert not handler.can_handle()

    def test_can_handle_false_for_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "frames").mkdir()
        handler = HDF5FormatHandler(tmp_path)
        assert not handler.can_handle()

    def test_list_frames_empty_without_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "frames").mkdir()
        handler = HDF5FormatHandler(tmp_path)
        assert handler.list_frames() == []

    def test_can_handle_manifest_driven_packed_chunks(self, tmp_path: Path) -> None:
        _write_minimal_packed_frame_set(tmp_path, frame_id=5)
        handler = HDF5FormatHandler(tmp_path)
        try:
            assert handler.can_handle()
            assert handler.list_frames() == [5]
        finally:
            handler.close()

    def test_has_frame_uses_manifest_inventory(self, tmp_path: Path) -> None:
        _write_minimal_packed_frame_set(tmp_path, frame_id=5)
        handler = HDF5FormatHandler(tmp_path)
        try:
            assert handler.has_frame(5)
            assert not handler.has_frame(6)
        finally:
            handler.close()

    def test_custom_frames_subdir(self, tmp_path: Path) -> None:
        _write_minimal_packed_frame_set(
            tmp_path,
            frame_id=9,
            frames_subdir="custom_frames",
        )
        handler = HDF5FormatHandler(tmp_path, frames_subdir="custom_frames")
        try:
            assert handler.can_handle()
            assert handler.list_frames() == [9]
        finally:
            handler.close()

    def test_legacy_metadata_does_not_enable_handler(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frames_index.json").write_text("{}", encoding="utf-8")
        (frames_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        (frames_dir / "mpc_frames_00000-00000.h5").touch()

        handler = HDF5FormatHandler(tmp_path)
        assert not handler.can_handle()
