"""Manifest-driven HDF5 v2 format handler.

``frames_manifest.json`` is the sole runtime entry point for an MPC frame set.
The manifest identifies the packed HDF5 chunks, their ordered frame IDs, and
the stable generation/frame-set identities exposed by providers.  Physical
chunk validation and projection reads are delegated to
:class:`~shared.frames.packed_hdf5.PackedHDF5Reader`.

Other index files and HDF5 layouts are not runtime inputs for this handler.
Format-specific benchmark decoders remain outside the provider path.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from shared.logging import get_logger

from .base import FormatHandler
from .contracts import FrameReadRequest
from .manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameManifestError,
)
from .packed import FrameProjection
from .packed_hdf5 import PackedHDF5Reader
from .types import StandardMPCFrame

logger = get_logger(__name__)


# Frame-set integrity failures use the manifest validator's exception type.
ScenarioFrameSetIntegrityError = FrameManifestError


class HDF5FormatHandler(FormatHandler):
    """Adapt one manifest-driven packed HDF5 v2 frame set to ``FormatHandler``."""

    def __init__(self, source: str | Path, frames_subdir: str = "frames") -> None:
        super().__init__(source)
        self.frames_dir = self.source / frames_subdir
        self._packed_reader: PackedHDF5Reader | None = None
        self._detect_layout()

    def _detect_layout(self) -> None:
        """Install a reader only when the authoritative v2 manifest exists."""
        manifest_path = self.frames_dir / FRAMES_MANIFEST_FILENAME
        if not manifest_path.is_file():
            return

        # PackedHDF5Reader validates the manifest and advertised file metadata
        # without opening any HDF5 chunk. HDF5 handles stay lazy until a frame
        # or projection is requested.
        self._packed_reader = PackedHDF5Reader(self.frames_dir)
        logger.info(
            "Detected manifest-driven packed HDF5 v2 format: %d files",
            len(self._packed_reader.bulk_files),
        )

    def refresh(self) -> None:
        """Close current HDF5 resources and reload the manifest from disk."""
        self.close()
        self._packed_reader = None
        self._detect_layout()

    @property
    def is_bulk(self) -> bool:
        """Return whether a manifest-driven multi-frame set is installed."""
        return self._packed_reader is not None

    @property
    def bulk_files(self) -> list[Path]:
        """Return manifest-advertised HDF5 chunks in manifest order."""
        reader = self._packed_reader
        return [] if reader is None else reader.bulk_files

    @property
    def generation_id(self) -> str | None:
        """Return the authoritative frame-set generation identifier."""
        reader = self._packed_reader
        return None if reader is None else reader.manifest.generation_id

    @property
    def frame_set_id(self) -> str | None:
        """Return the authoritative stable frame-set identity."""
        reader = self._packed_reader
        return None if reader is None else reader.manifest.frame_set_id

    def close(self) -> None:
        """Close every lazily opened HDF5 chunk handle."""
        if self._packed_reader is not None:
            self._packed_reader.close()

    def can_handle(self) -> bool:
        """Return whether a valid packed-v2 manifest was installed."""
        return self._packed_reader is not None

    def list_frames(self) -> list[int]:
        """Return manifest-advertised frame IDs without opening HDF5."""
        reader = self._packed_reader
        return [] if reader is None else reader.frame_ids

    def has_frame(self, step: int) -> bool:
        """Answer frame membership from manifest metadata only."""
        reader = self._packed_reader
        return reader is not None and reader.has_frame(step)

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load one complete canonical frame directly from HDF5."""
        reader = self._packed_reader
        if reader is None:
            raise FileNotFoundError(
                f"No {FRAMES_MANIFEST_FILENAME} frame set found under {self.frames_dir}"
            )
        if not reader.has_frame(step):
            raise FileNotFoundError(
                f"Frame {step} is not listed in " f"{self.frames_dir / FRAMES_MANIFEST_FILENAME}"
            )
        return reader.load_standard_frame(step)

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        """Load one true selective projection from packed HDF5."""
        reader = self._packed_reader
        if reader is None:
            raise NotImplementedError(
                "Selective frame projections require the packed HDF5 v2 layout"
            )
        return reader.load_projection(step, request)

    def iter_frame_projections(
        self,
        steps: Sequence[int],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Yield packed-v2 projections in caller order."""
        reader = self._packed_reader
        if reader is None:
            raise NotImplementedError(
                "Selective frame projections require the packed HDF5 v2 layout"
            )
        return reader.iter_projections(steps, request)


__all__ = [
    "HDF5FormatHandler",
    "ScenarioFrameSetIntegrityError",
]
