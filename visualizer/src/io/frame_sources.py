"""Frame-source implementations and factory helpers for the visualizer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Optional, Sequence

from shared.frames.provider_base import DataProvider
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame

from .scenario_config import DEFAULT_GRPC_SIONNA, DEFAULT_REMOTE_HDF5_SERVER

if TYPE_CHECKING:
    from shared.frames.remote_hdf5 import RemoteHdf5Provider

try:
    from shared.logging import get_logger
except ImportError:
    import logging

    def get_logger(name):
        """Return a basic logger when shared logging is unavailable."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_logger("orchav")

_OPTIONAL_GRPC_MODULES = ("grpc", "google.protobuf")


def _is_optional_grpc_import_error(exc: ModuleNotFoundError) -> bool:
    """Return whether *exc* identifies the optional gRPC runtime."""
    missing = exc.name or ""
    return missing == "google" or any(
        missing == module or missing.startswith(f"{module}.") for module in _OPTIONAL_GRPC_MODULES
    )


def _raise_missing_grpc_transport(
    feature: str,
    exc: ModuleNotFoundError,
) -> NoReturn:
    """Translate only missing optional transport imports into setup guidance."""
    if not _is_optional_grpc_import_error(exc):
        raise exc
    raise RuntimeError(
        f"{feature} requires the optional gRPC transport; " 'run python -m pip install -e ".[grpc]"'
    ) from exc


class FrameSource:
    """Abstract interface for frame data sources."""

    def open(self) -> None:
        """Open or initialize the frame source."""
        raise NotImplementedError

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load a specific frame and return the standard MPC frame format."""
        raise NotImplementedError

    def has_frame(self, step: int) -> bool:
        """Return whether a specific frame exists."""
        raise NotImplementedError

    def list_frames(self) -> list[int]:
        """List all available frame numbers."""
        raise NotImplementedError

    def close(self) -> None:
        """Release source resources; stateless implementations may do nothing."""
        return None


class FileSource(FrameSource):
    """File-based frame source that routes to the appropriate data provider."""

    def __init__(self, root: Path, directory: str | Path, fmt: str) -> None:
        """Store file-mode scenario settings without opening frame data yet."""
        self.root = root
        self.directory = str(directory)
        self.fmt = fmt
        self.provider: Optional[DataProvider] = None
        self._opened = False

    def open(self) -> None:
        """Initialize the appropriate provider based on format."""
        if self._opened:
            return
        self._opened = True

        if self.fmt in ["hdf5", "h5"]:
            frames_subdir = self.directory or "frames"
            try:
                self.provider = Hdf5Provider(str(self.root), frames_subdir=frames_subdir)
            except FileNotFoundError:
                logger.warning(
                    "No HDF5 frames found under %s/%s - scene-only mode",
                    self.root,
                    frames_subdir,
                )
                self.provider = None
                return
        else:
            raise ValueError(f"Unsupported format: {self.fmt}. Only HDF5 (hdf5/h5) is supported.")

        logger.debug("FileSource opened with %s provider for root: %s", self.fmt, self.root)

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load a frame by delegating to the provider."""
        if not self._opened:
            self.open()

        if self.provider is None:
            raise FileNotFoundError("No frames available (scene-only mode)")

        return self.provider.load_frame(step)

    def has_frame(self, step: int) -> bool:
        """Return frame availability through the provider."""
        if not self._opened:
            self.open()

        if self.provider is None:
            return False

        return self.provider.has_frame(step)

    def list_frames(self) -> list[int]:
        """List available frames through the provider."""
        if not self._opened:
            self.open()

        if self.provider is None:
            return []

        return self.provider.list_frames()

    def close(self) -> None:
        """Close the active file provider and permit a clean later reopen."""
        provider = self.provider
        self.provider = None
        self._opened = False
        if provider is not None:
            provider.close()


class RemoteHdf5Source(FrameSource):
    """Frame source for remote pre-generated HDF5 frames."""

    def __init__(
        self,
        server_address: str,
        cache_size: int = 50,
        connect_timeout: float = 10.0,
        frame_index_ttl_s: float = 0.0,
    ) -> None:
        """Validate remote-HDF5 connection and cache settings."""
        if int(cache_size) < 1:
            raise ValueError("remote_hdf5.cache_size must be >= 1")
        if float(connect_timeout) <= 0.0:
            raise ValueError("remote_hdf5.connect_timeout must be > 0")
        if float(frame_index_ttl_s) < 0.0:
            raise ValueError("remote_hdf5.frame_index_ttl_s must be >= 0")
        self.server_address = server_address
        self.cache_size = int(cache_size)
        self.connect_timeout = float(connect_timeout)
        self.frame_index_ttl_s = float(frame_index_ttl_s)
        self.provider: Optional["RemoteHdf5Provider"] = None
        self.is_initialized = False

    def open(self) -> None:
        """Connect to the remote frame file server."""
        if self.is_initialized and self.provider is not None:
            logger.debug("RemoteHdf5Source already initialized")
            return

        try:
            from shared.frames.remote_hdf5 import RemoteHdf5Provider
        except ModuleNotFoundError as exc:
            _raise_missing_grpc_transport("Remote HDF5 playback", exc)

        logger.info("Connecting to remote frame server at %s", self.server_address)
        self.provider = RemoteHdf5Provider(
            self.server_address,
            cache_size=self.cache_size,
            connect_timeout=self.connect_timeout,
            frame_index_ttl_s=self.frame_index_ttl_s,
        )
        try:
            self.provider.open()
        except (OSError, RuntimeError):
            self.provider = None
            self.is_initialized = False
            raise
        else:
            self.is_initialized = True
            logger.info(
                "RemoteHdf5Source connected: %d frames available",
                len(self.provider.list_frames()),
            )

    def close(self) -> None:
        """Disconnect from the remote server."""
        if self.provider:
            self.provider.close()
            self.provider = None
        self.is_initialized = False

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load a frame from the remote server."""
        if not self.provider or not self.is_initialized:
            self.open()
        return self.provider.load_frame(step)

    def has_frame(self, step: int) -> bool:
        """Check if a frame is available on the remote server."""
        if not self.provider or not self.is_initialized:
            self.open()
        return self.provider.has_frame(step)

    def list_frames(self) -> list[int]:
        """List available frames from the remote server."""
        if not self.provider or not self.is_initialized:
            self.open()
        return self.provider.list_frames()

    def prefetch_frames(self, start: int, count: int = 10) -> None:
        """Start background prefetching of frames for smooth animation."""
        if self.provider:
            self.provider.prefetch_frames(start, count)

    @property
    def metadata(self) -> dict:
        """Return server metadata."""
        if self.provider:
            return self.provider.metadata
        return {}

    @property
    def frame_set_id(self) -> str:
        """Return the connected remote HDF5 frame-set id, if known."""
        if self.provider:
            return self.provider.frame_set_id
        return ""


class LiveGrpcSource(FrameSource):
    """Frame source for live generator streams and edit/request commands."""

    def __init__(self, endpoint: str, *, buffer_size: int = 50) -> None:
        """Store live gRPC connection settings without connecting eagerly."""
        self.endpoint = endpoint
        self.buffer_size = buffer_size
        self.provider: Optional[Any] = None
        self.is_initialized = False

    def open(self) -> None:
        """Initialize gRPC connection and start streaming."""
        if self.is_initialized and self.provider is not None:
            logger.debug("LiveGrpcSource already initialized")
            return

        try:
            from .grpc_provider import GrpcProvider
        except ModuleNotFoundError as exc:
            _raise_missing_grpc_transport("Live visualizer playback", exc)

        try:
            logger.info("LiveGrpcSource opening connection to: %s", self.endpoint)

            self.provider = GrpcProvider(self.endpoint, buffer_size=self.buffer_size)

            self.provider.open()

            self.is_initialized = True
            logger.info("LiveGrpcSource connected to: %s", self.endpoint)

        except (OSError, RuntimeError) as e:
            logger.error("Failed to open LiveGrpcSource: %s", e)
            logger.error("Endpoint: %s", self.endpoint)
            self.is_initialized = False
            self.provider = None
            raise

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load a frame from the gRPC provider."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            return self.provider.load_frame(step)
        except (OSError, RuntimeError) as e:
            logger.error("Error loading frame %d from gRPC: %s", step, e)
            raise

    def has_frame(self, step: int) -> bool:
        """Check if a frame is available through the gRPC provider."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            return self.provider.has_frame(step)
        except (OSError, RuntimeError) as e:
            logger.error("Error checking frame %d availability: %s", step, e)
            return False

    def list_frames(self) -> list[int]:
        """List available frames through the gRPC provider."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            return self.provider.list_frames()
        except (OSError, RuntimeError) as e:
            logger.error("Error listing frames: %s", e)
            return []

    def subscribe(self, on_frame: Any) -> None:
        """Subscribe to streaming updates."""
        if not self.is_initialized or self.provider is None:
            logger.warning("Cannot subscribe: LiveGrpcSource not initialized")
            return

        try:
            self.provider.subscribe(on_frame)
        except (OSError, RuntimeError) as e:
            logger.error("Error subscribing to frame updates: %s", e)

    def request_frame(self, frame_idx: int) -> bool:
        """Request a specific frame from the gRPC server."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            return self.provider.request_frame(frame_idx)
        except (OSError, RuntimeError) as e:
            logger.error("Error requesting frame %d from gRPC: %s", frame_idx, e)
            return False

    def get_connection_status(self) -> dict:
        """Get connection status from the gRPC provider."""
        if not self.is_initialized or self.provider is None:
            return {"connected": False, "error": "Not initialized"}

        try:
            return self.provider.get_connection_status()
        except (OSError, RuntimeError) as e:
            logger.error("Error getting connection status: %s", e)
            return {"connected": False, "error": str(e)}

    def get_buffer_status(self) -> dict:
        """Get buffer status from the gRPC provider."""
        if not self.is_initialized or self.provider is None:
            return {"buffer_size": 0, "error": "Not initialized"}

        try:
            return self.provider.get_buffer_status()
        except (OSError, RuntimeError) as e:
            logger.error("Error getting buffer status: %s", e)
            return {"buffer_size": 0, "error": str(e)}

    def clear_buffer(self, *, pause_stream: bool = False, preserve_position: bool = True) -> None:
        """Clear the local frame buffer."""
        if not self.is_initialized or self.provider is None:
            return

        try:
            self.provider.clear_buffer(
                pause_stream=pause_stream,
                preserve_position=preserve_position,
            )
        except (OSError, RuntimeError) as e:
            logger.error("Error clearing buffer: %s", e)

    def request_cache_flush(self, reason: str = "Manual cache flush") -> None:
        """Flush both client and server caches."""
        if not self.is_initialized or self.provider is None:
            return
        try:
            self.provider.request_cache_flush(flush_client=True, flush_server=True, reason=reason)
        except (OSError, RuntimeError) as e:
            logger.error("Error requesting cache flush: %s", e)

    def update_object_via_xml(
        self, xml_root, *, scene_name: str = "scene_modified.xml", reason: str = "Edit Properties"
    ) -> bool:
        """Send updated XML scene to the generator."""
        if not self.is_initialized or self.provider is None:
            self.open()
        if self.provider is None:
            return False
        try:
            return self.provider.update_object_via_xml(
                xml_root,
                scene_name=scene_name,
                reason=reason,
            )
        except (OSError, RuntimeError) as e:
            logger.error("Error sending object update: %s", e)
            return False

    def update_node_properties(
        self,
        *,
        node_type: str,
        node_name: str,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[float] = None,
        flush_cache: bool = True,
        origin: str = "node_edit",
        details: Optional[str] = None,
    ) -> bool:
        """Send TX/RX/target property updates through the provider."""
        if not self.is_initialized or self.provider is None:
            self.open()
        if self.provider is None:
            return False
        try:
            return self.provider.update_node_properties(
                node_type=node_type,
                node_name=node_name,
                position=position,
                orientation=orientation,
                scale=scale,
                flush_cache=flush_cache,
                origin=origin,
                details=details,
            )
        except (OSError, RuntimeError) as e:
            logger.error("Error sending node update: %s", e)
            return False

    def subscribe_to_frames(self, callback) -> None:
        """Subscribe to frame updates."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            self.provider.subscribe_to_frames(callback)
        except (OSError, RuntimeError) as e:
            logger.error("Error subscribing to frames: %s", e)

    def unsubscribe_from_frames(self, callback) -> None:
        """Unsubscribe from frame updates."""
        if not self.is_initialized or self.provider is None:
            return

        try:
            self.provider.unsubscribe_from_frames(callback)
        except (OSError, RuntimeError) as e:
            logger.error("Error unsubscribing from frames: %s", e)

    def request_frame_with_parameters(
        self, frame_idx: int, tx_overrides: dict = None, rx_overrides: dict = None
    ) -> Optional[StandardMPCFrame]:
        """Request a frame with custom TX/RX position parameters."""
        if not self.is_initialized or self.provider is None:
            self.open()

        try:
            return self.provider.request_frame_with_parameters(
                frame_idx, tx_overrides, rx_overrides
            )
        except (OSError, RuntimeError) as e:
            logger.error("Error requesting frame %d with parameters: %s", frame_idx, e)
            return None

    def load_frame_with_overrides(
        self, step: int, overrides: list[dict[str, Any]]
    ) -> Optional[Any]:
        """Load a live frame with TX/RX/target overrides applied by the provider."""
        if not self.is_initialized or self.provider is None:
            logger.warning("Cannot load frame with overrides: LiveGrpcSource not initialized")
            return None

        try:
            return self.provider.load_frame_with_overrides(step, overrides)
        except (OSError, RuntimeError) as e:
            logger.error("Error loading frame %d with overrides: %s", step, e)
            return None

    def close(self) -> None:
        """Close the gRPC connection and clean up."""
        if self.provider is not None:
            try:
                self.provider.close()
                logger.info("LiveGrpcSource closed")
            except (OSError, RuntimeError) as e:
                logger.error("Error closing LiveGrpcSource: %s", e)
            finally:
                self.provider = None
                self.is_initialized = False


def make_frame_source(scn: Any) -> FrameSource:
    """Create the appropriate frame source based on scenario configuration."""
    if scn.data_mode == "files":
        files_spec = scn.data_spec.get("files", {})
        directory = files_spec.get("directory", "frames")
        fmt = files_spec.get("format", "h5")

        return FileSource(scn.root, directory, fmt)

    if scn.data_mode == "live_grpc":
        live_grpc_spec = scn.data_spec.get("live_grpc", {})
        endpoint = (
            live_grpc_spec.get("endpoint")
            or scn.live_grpc_endpoints.get("sionna")
            or DEFAULT_GRPC_SIONNA
        )
        buffer_size = int(live_grpc_spec.get("buffer_size", 50))
        logger.info(
            "Creating LiveGrpcSource for endpoint: %s (buffer_size=%d)", endpoint, buffer_size
        )
        return LiveGrpcSource(endpoint, buffer_size=buffer_size)

    if scn.data_mode == "remote_hdf5":
        remote_spec = scn.data_spec.get("remote_hdf5", {})
        server_address = remote_spec.get("server", DEFAULT_REMOTE_HDF5_SERVER)
        cache_size = int(remote_spec.get("cache_size", 50))
        connect_timeout = float(remote_spec.get("connect_timeout", 10.0))
        frame_index_ttl_s = float(remote_spec.get("frame_index_ttl_s", 0.0))

        logger.info(
            "Creating RemoteHdf5Source for server: %s (cache_size=%d)",
            server_address,
            cache_size,
        )
        return RemoteHdf5Source(server_address, cache_size, connect_timeout, frame_index_ttl_s)

    from .frame_source_extensions import (
        create_registered_frame_source,
        registered_frame_source_modes,
    )

    source = create_registered_frame_source(scn)
    if source is not None:
        return source

    known_modes = ["files", "live_grpc", "remote_hdf5"] + registered_frame_source_modes()
    logger.debug("Known visualizer data modes: %s", ", ".join(sorted(set(known_modes))))
    raise ValueError(f"Unknown data mode: {scn.data_mode}")
