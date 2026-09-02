"""Storage-reader boundary beneath shared frame providers.

``FormatHandler`` owns format-specific discovery and random access while
``FileProvider`` owns the common provider API. A handler returns complete
``StandardMPCFrame`` values; formats with selective I/O may override the
projection methods, while simpler formats use the complete-frame fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from .adapters import project_standard_mpc_frame
from .contracts import FrameReadRequest
from .packed import FrameProjection
from .types import StandardMPCFrame


class FormatHandler(ABC):
    """
    Base class for file-based frame format handlers.

    A handler encapsulates all file format specific logic (discovery, listing,
    loading) so that higher-level providers can stay agnostic of the underlying
    storage layout.
    """

    def __init__(self, source: str | Path):
        self.source = Path(source)

    @abstractmethod
    def can_handle(self) -> bool:
        """Return True if this handler can operate on ``source``."""

    @abstractmethod
    def list_frames(self) -> list[int]:
        """Return the sorted list of available frame indices."""

    @abstractmethod
    def has_frame(self, step: int) -> bool:
        """Return True if ``step`` can be loaded."""

    @abstractmethod
    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load ``step`` and return a StandardMPCFrame."""

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        """Load a logical projection, falling back to a complete frame read."""
        return project_standard_mpc_frame(
            self.load_frame(step),
            request,
        )

    def iter_frame_projections(
        self,
        steps: Iterable[int],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Yield logical projections in caller order."""
        for step in steps:
            yield self.load_frame_projection(int(step), request)

    @property
    def generation_id(self) -> str | None:
        """Stable generation identifier when the physical format supplies one."""
        return None

    @property
    def frame_set_id(self) -> str | None:
        """Stable frame-set identifier when the physical format supplies one."""
        return None

    @property
    def is_bulk(self) -> bool:
        """Whether the handler uses bulk (multi-frame) files."""
        return False

    @property
    def bulk_files(self) -> list[Path]:
        """Return bulk file paths (empty for non-bulk handlers)."""
        return []

    def close(self) -> None:
        """Release any open resources held by the handler."""
        return
