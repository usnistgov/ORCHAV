"""Common frame-provider interface for ``StandardMPCFrame`` sources.

Providers hide where frames come from: local HDF5 files, remote HDF5 gRPC, live
streaming, or reconstructed measurement caches. Their common promise is that
``load_frame()`` returns a validated ``StandardMPCFrame``. Generator-specific
writers and raw Sionna RT objects stay outside this shared provider contract.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Flag, auto
from typing import Any, Callable, Optional

from .adapters import project_standard_mpc_frame
from .contracts import FrameReadRequest
from .packed import FrameProjection
from .types import StandardMPCFrame


class ProviderCapability(Flag):
    """
    Capabilities that a data provider may support.

    Use these flags to query provider features at runtime:
        if provider.capabilities & ProviderCapability.STREAMING:
            provider.subscribe(callback)
    """

    NONE = 0
    STREAMING = auto()  # Supports real-time frame streaming
    CACHING = auto()  # Has internal frame cache
    RANDOM_ACCESS = auto()  # Can load any frame directly (not just sequential)
    WRITE = auto()  # Can write/save frames
    OVERRIDES = auto()  # Supports position overrides for on-demand computation


@dataclass
class ProviderInfo:
    """
    Metadata about a data provider instance.

    Attributes:
        name: Provider type name (e.g., "Hdf5Provider", "GrpcProvider")
        source: Data source identifier (file path, gRPC endpoint, etc.)
        total_frames: Total number of frames available (-1 if unknown/streaming)
        frame_rate: Frames per second (0 if unknown)
        capabilities: Bitmask of supported capabilities
        generation_id: Stable generation identifier when supplied by the source
        frame_set_id: Stable frame-set identifier when supplied by the source
    """

    name: str
    source: str
    total_frames: int = -1
    frame_rate: float = 0.0
    capabilities: ProviderCapability = ProviderCapability.RANDOM_ACCESS
    generation_id: str | None = None
    frame_set_id: str | None = None


class DataProvider(ABC):
    """Abstract base class for canonical MPC frame providers.

    File, remote, streaming, and measurement-backed providers all expose the
    same access pattern once their source has been normalized to
    ``StandardMPCFrame``. Code that consumes frames should depend on this class
    rather than on concrete HDF5, protobuf, or generator output details.

    Lifecycle:
        1. Create provider instance
        2. Call open() to initialize
        3. Use list_frames(), has_frame(), load_frame() as needed
        4. Call close() when done

    Example:
        provider = Hdf5Provider("/path/to/scenario")
        provider.open()
        try:
            frames = provider.list_frames()
            for idx in frames:
                frame = provider.load_frame(idx)
                process(frame)
        finally:
            provider.close()
    """

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def open(self) -> None:
        """
        Initialize the data provider.

        This method should be called before any data access. For file-based
        providers, this might validate the file path. For streaming providers,
        this establishes the connection.

        Raises:
            ConnectionError: If unable to connect (streaming)
            FileNotFoundError: If data files not found (file-based)
            ValueError: If data format is invalid
        """
        return None

    def close(self) -> None:
        """
        Release resources and close connections.

        This method should be called when done with the provider.
        After close(), the provider should not be used.
        """
        return None

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_frames(self) -> list[int]:
        """
        List all available frame indices.

        Returns:
            Sorted list of available frame indices (0-based)

        Note:
            For streaming providers, this returns frames currently available
            in the buffer, which may change over time.
        """
        ...

    @abstractmethod
    def has_frame(self, step: int) -> bool:
        """
        Check if a specific frame is available.

        Args:
            step: Frame index (0-based)

        Returns:
            True if the frame can be loaded, False otherwise
        """
        ...

    @abstractmethod
    def load_frame(self, step: int) -> StandardMPCFrame:
        """
        Load a specific frame and return it in standard format.

        Args:
            step: Frame index (0-based)

        Returns:
            StandardMPCFrame with ray tracing data

        Raises:
            KeyError: If frame index is not available
            ValueError: If frame data is invalid
        """
        ...

    # -------------------------------------------------------------------------
    # Optional Methods - Override for specific capabilities
    # -------------------------------------------------------------------------

    @property
    def info(self) -> ProviderInfo:
        """
        Get provider metadata.

        Returns:
            ProviderInfo with name, source, and capabilities
        """
        try:
            total_frames = len(self.list_frames())
        except (ConnectionError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError):
            total_frames = 0
        return ProviderInfo(
            name=self.__class__.__name__,
            source="unknown",
            total_frames=total_frames,
            capabilities=ProviderCapability.RANDOM_ACCESS,
        )

    @property
    def capabilities(self) -> ProviderCapability:
        """Shorthand for info.capabilities."""
        return self.info.capabilities

    def subscribe(self, on_frame: Callable[[int, StandardMPCFrame], None]) -> None:
        """
        Subscribe to streaming frame updates.

        Only available if STREAMING capability is supported.

        Args:
            on_frame: Callback function called with (frame_idx, frame_data)
                      when a new frame arrives
        """
        return None

    def unsubscribe(self, on_frame: Callable[[int, StandardMPCFrame], None]) -> None:
        """
        Unsubscribe from streaming frame updates.

        Args:
            on_frame: Previously registered callback to remove
        """
        return None

    def load_frame_with_overrides(
        self, step: int, overrides: list[dict[str, Any]]
    ) -> Optional[StandardMPCFrame]:
        """
        Load a frame with position overrides applied.

        Only available if OVERRIDES capability is supported.
        This triggers frame recomputation with the specified positions.

        Args:
            step: Frame index (0-based)
            overrides: List of override dicts with keys:
                - name: Node name (e.g., "TX1", "RX2")
                - type: Node type ("tx", "rx", "target")
                - position: [x, y, z] coordinates
                - orientation: Optional [yaw, pitch, roll]

        Returns:
            Recomputed StandardMPCFrame, or None if not supported
        """
        # Default: not supported
        return None

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        """Load a logical frame projection.

        Providers with selective storage should override this method. The
        default loads one complete canonical frame and retains references only
        to the requested components.
        """
        frame = self.load_frame(step)
        if frame is None:
            raise KeyError(f"Frame {step} is not available from {type(self).__name__}")
        return project_standard_mpc_frame(
            frame,
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

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "DataProvider":
        """Context manager entry - opens the provider."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the provider."""
        self.close()
