"""Shared file-based frame providers.

``FileProvider`` is the format-agnostic base class for any file-backed
provider. ``Hdf5Provider`` is the concrete reader for generated HDF5 frame
sets. Both return validated ``StandardMPCFrame`` objects and hide the physical
file layout from visualizer and analysis code.

Adding a new native read format requires:
1. Subclass :class:`FormatHandler` (see ``shared.frames.base``).
2. Subclass :class:`FileProvider` with a ``__init__`` that constructs the
   handler and passes it to ``super().__init__()``.

External producers that publish ORCHAV frame sets do not need a provider; they
adapt their records to ``StandardMPCFrame`` and use ``FrameSetWriter``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from shared.logging import get_logger

from .base import FormatHandler
from .contracts import FrameReadRequest
from .hdf5 import HDF5FormatHandler
from .packed import FrameProjection
from .provider_base import DataProvider, ProviderCapability, ProviderInfo
from .types import StandardMPCFrame

logger = get_logger(__name__)


class FileProvider(DataProvider):
    """Format-agnostic base class for file-backed data providers.

    Delegates storage I/O to a :class:`FormatHandler` and enforces that its
    complete-frame methods return ``StandardMPCFrame`` objects. Construction
    is the validation boundary for those objects.

    Args:
        source: Path to the dataset root directory.
        handler: A :class:`FormatHandler` instance that knows how to read
            the specific file format.
    """

    def __init__(self, source: str | Path, handler: FormatHandler) -> None:
        self.source = Path(source)
        self._handler = handler

    # ---------------------------------------------------------------- provider
    def list_frames(self) -> list[int]:
        return self._handler.list_frames()

    def has_frame(self, step: int) -> bool:
        return self._handler.has_frame(step)

    def load_frame(self, step: int) -> StandardMPCFrame:
        logger.debug("Loading frame %s via %s from %s", step, type(self).__name__, self.source)
        frame = self._handler.load_frame(step)
        if not isinstance(frame, StandardMPCFrame):
            raise TypeError(
                f"{type(self._handler).__name__}.load_frame() must return StandardMPCFrame"
            )
        return frame

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        """Delegate a selective read without constructing a complete frame."""
        logger.debug(
            "Loading frame %s projection=%s via %s from %s",
            step,
            request,
            type(self).__name__,
            self.source,
        )
        return self._handler.load_frame_projection(step, request)

    def iter_frame_projections(
        self,
        steps: Iterable[int],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Delegate ordered batch projections to the format handler."""
        return self._handler.iter_frame_projections(steps, request)

    def close(self) -> None:
        """Release resources held by the underlying format handler."""
        self._handler.close()

    # ---------------------------------------------------------------- metadata
    @property
    def info(self) -> ProviderInfo:
        """Expose provider metadata for UI/diagnostics."""
        frames = self.list_frames()
        return ProviderInfo(
            name=type(self).__name__,
            source=str(self.source),
            total_frames=len(frames),
            capabilities=ProviderCapability.RANDOM_ACCESS,
            generation_id=self._handler.generation_id,
            frame_set_id=self._handler.frame_set_id,
        )

    @property
    def is_bulk(self) -> bool:
        """Whether the dataset uses bulk files."""
        return self._handler.is_bulk

    @property
    def bulk_files(self) -> list[str]:
        """Return the list of bulk file paths (if any)."""
        return [str(path) for path in self._handler.bulk_files]


class Hdf5Provider(FileProvider):
    """HDF5 file-based provider.

    Reads frames from an on-disk HDF5 dataset using
    :class:`HDF5FormatHandler`.

    Args:
        source: Path to the scenario root directory.
        frames_subdir: Subdirectory under *source* containing HDF5 frame
            files (default ``"frames"``).
    """

    def __init__(self, source: str | Path, frames_subdir: str = "frames") -> None:
        handler = HDF5FormatHandler(Path(source), frames_subdir=frames_subdir)
        if not handler.can_handle():
            raise FileNotFoundError(
                f"No HDF5 frames directory found under {Path(source).resolve()}/{frames_subdir}"
            )
        super().__init__(source, handler)


__all__ = ["FileProvider", "Hdf5Provider"]
