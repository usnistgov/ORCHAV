"""Remote provider for reading a generated HDF5 frame set via gRPC.

This provider is the shared client side of the remote-HDF5 workflow. It connects
to a ``FrameFileServer`` pinned to one manifest identity, fetches complete
pre-generated frames as protobuf messages, and returns ``StandardMPCFrame``
objects through the normal provider interface. It does not transfer HDF5 files
or expose their chunk layout. The server implementation lives in
``generator.io.grpc.file_server`` because it owns the generator output
directory and its immutable startup snapshot.

Usage:
    from shared.frames.remote_hdf5 import RemoteHdf5Provider

    provider = RemoteHdf5Provider("192.168.1.100:50052")
    provider.open()
    frame = provider.load_frame(0)
    provider.close()
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import grpc

from shared.cache_sizing import estimate_retained_bytes
from shared.frames.protobuf import standard_mpc_frame_from_proto
from shared.grpc_transport import (
    DEFAULT_GRPC_CONNECT_TIMEOUT_S,
    DEFAULT_GRPC_UNARY_TIMEOUT_S,
    GRPC_MESSAGE_OPTIONS,
    format_grpc_endpoint,
    parse_grpc_endpoint,
)
from shared.logging import get_logger

from .provider_base import DataProvider, ProviderCapability, ProviderInfo
from .types import StandardMPCFrame

logger = get_logger(__name__)

# gRPC channel liveness configuration. Request completion is bounded separately
# by the unary RPC timeout supplied to every finite remote operation.
GRPC_KEEPALIVE_TIME_MS = 30_000  # Send keepalive ping every 30 s
GRPC_KEEPALIVE_TIMEOUT_MS = 10_000  # Wait 10 s for keepalive response
DEFAULT_REMOTE_FRAME_CACHE_SIZE = 50
PREFETCH_CLOSE_JOIN_TIMEOUT_S = 2.0
PREFETCH_RESTART_JOIN_TIMEOUT_S = 1.0

# Import the gRPC generated code
try:
    from shared.protos import visualizer_pb2 as _visualizer_pb2
    from shared.protos import visualizer_pb2_grpc as _visualizer_pb2_grpc

    visualizer_pb2: Any = _visualizer_pb2
    visualizer_pb2_grpc: Any = _visualizer_pb2_grpc
except ImportError as e:
    logger.error("Could not import gRPC generated code: %s", e)
    logger.error("Make sure the protobuf files are generated.")
    raise


class RemoteHdf5Provider(DataProvider):
    """Provider that fetches pre-generated frames from a remote FrameFileServer.

    This provider connects to a gRPC server that serves pre-generated HDF5 frames,
    allowing remote visualization without local file access.

    Features:
    - LRU caching of recently fetched frames
    - Background prefetching for animation

    Connection failures are reported to the caller. Reconnecting is an explicit
    ``close()`` followed by ``open()`` so failed requests are never replayed.
    """

    def __init__(
        self,
        server_address: str,
        cache_size: int = DEFAULT_REMOTE_FRAME_CACHE_SIZE,
        connect_timeout: float = DEFAULT_GRPC_CONNECT_TIMEOUT_S,
        rpc_timeout: float = DEFAULT_GRPC_UNARY_TIMEOUT_S,
        frame_index_ttl_s: float = 0.0,
    ):
        """Initialize the remote HDF5 provider.

        Args:
            server_address: gRPC server address (e.g., "192.168.1.100:50052")
            cache_size: Maximum number of frames to cache locally
            connect_timeout: Connection timeout in seconds
            rpc_timeout: Deadline in seconds for each finite server request
            frame_index_ttl_s: Optional TTL for refreshing available frame list
        """
        if int(cache_size) < 1:
            raise ValueError("cache_size must be >= 1")
        if float(connect_timeout) <= 0.0:
            raise ValueError("connect_timeout must be > 0")
        if float(rpc_timeout) <= 0.0:
            raise ValueError("rpc_timeout must be > 0")
        if float(frame_index_ttl_s) < 0.0:
            raise ValueError("frame_index_ttl_s must be >= 0")

        server_host, server_port = parse_grpc_endpoint(server_address)
        self.server_address = format_grpc_endpoint(server_host, server_port)
        self.cache_size = int(cache_size)
        self.connect_timeout = float(connect_timeout)
        self.rpc_timeout = float(rpc_timeout)
        self.frame_index_ttl_s = float(frame_index_ttl_s)

        self._channel: Optional[grpc.Channel] = None
        self._stub: Any = None
        self._connected = False

        # Frame cache (LRU)
        self._frame_cache: OrderedDict[int, StandardMPCFrame] = OrderedDict()
        self._frame_cache_sizes: dict[int, int] = {}
        self._cache_lock = threading.Lock()

        # Metadata cache
        self._metadata: Optional[dict] = None
        self._frame_indices: Optional[list[int]] = None
        self._frame_index_set: set[int] = set()
        self._frame_index_last_refresh_s: float = 0.0

        # Cache statistics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_current_bytes: int = 0
        self._cache_peak_bytes: int = 0
        self._cache_evictions: int = 0
        self._last_fetch_latency_ms: float = 0.0

        # Prefetch thread
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop = threading.Event()

    def _clear_frame_cache_locked(self) -> None:
        """Clear cache entries and current-byte accounting while lock-held."""
        self._frame_cache.clear()
        self._frame_cache_sizes.clear()
        self._cache_current_bytes = 0

    def _store_frame_locked(
        self,
        step: int,
        frame: StandardMPCFrame,
        retained_bytes: int,
    ) -> None:
        """Store one frame and update exact per-entry accounting while lock-held."""
        previous_bytes = self._frame_cache_sizes.pop(step, 0)
        self._cache_current_bytes -= previous_bytes

        self._frame_cache[step] = frame
        self._frame_cache_sizes[step] = retained_bytes
        self._frame_cache.move_to_end(step)
        self._cache_current_bytes += retained_bytes
        self._cache_peak_bytes = max(
            self._cache_peak_bytes,
            self._cache_current_bytes,
        )

        while len(self._frame_cache) > self.cache_size:
            evicted_step, _evicted_frame = self._frame_cache.popitem(last=False)
            self._cache_current_bytes -= self._frame_cache_sizes.pop(evicted_step, 0)
            self._cache_evictions += 1

    def open(self) -> None:
        """Connect to the remote frame server."""
        if self._connected:
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            raise RuntimeError(
                "Cannot reopen RemoteHdf5Provider while its previous prefetch worker is stopping"
            )
        self._prefetch_thread = None
        with self._cache_lock:
            self._clear_frame_cache_locked()
            self._metadata = None
            self._frame_indices = None
            self._frame_index_set.clear()
            self._frame_index_last_refresh_s = 0.0

        try:
            # Create gRPC channel with options for better performance
            options = [
                *GRPC_MESSAGE_OPTIONS,
                ("grpc.keepalive_time_ms", GRPC_KEEPALIVE_TIME_MS),
                ("grpc.keepalive_timeout_ms", GRPC_KEEPALIVE_TIMEOUT_MS),
            ]
            self._channel = grpc.insecure_channel(self.server_address, options=options)

            # Wait for connection
            try:
                grpc.channel_ready_future(self._channel).result(timeout=self.connect_timeout)
            except grpc.FutureTimeoutError as exc:
                raise ConnectionError(
                    f"Timeout connecting to {self.server_address} " f"after {self.connect_timeout}s"
                ) from exc

            self._stub = visualizer_pb2_grpc.FrameFileServiceStub(self._channel)

            # Fetch initial metadata
            self._fetch_metadata()
            self._connected = True

            logger.info(
                "Connected to FrameFileServer at %s (%d frames available)",
                self.server_address,
                len(self._frame_indices or []),
            )

        except (OSError, RuntimeError) as e:
            self.close()
            raise ConnectionError(f"Failed to connect to {self.server_address}: {e}") from e

    def close(self) -> None:
        """Disconnect from the remote frame server."""
        self._prefetch_stop.set()
        channel = self._channel
        self._channel = None
        self._stub = None
        self._connected = False
        if channel is not None:
            channel.close()

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=PREFETCH_CLOSE_JOIN_TIMEOUT_S)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.warning(
                "Remote HDF5 prefetch worker is still stopping; reopen is temporarily unavailable"
            )
        else:
            self._prefetch_thread = None

        with self._cache_lock:
            self._clear_frame_cache_locked()
            self._metadata = None
            self._frame_indices = None
            self._frame_index_set.clear()
            self._frame_index_last_refresh_s = 0.0

        logger.info("Disconnected from FrameFileServer")

    def _mark_transport_failed(self, reason: str) -> None:
        """Make a failed channel unusable until an explicit close/open cycle."""
        self._prefetch_stop.set()
        channel = self._channel
        self._channel = None
        self._stub = None
        self._connected = False
        if channel is not None:
            channel.close()

        with self._cache_lock:
            self._clear_frame_cache_locked()
            self._metadata = None
            self._frame_indices = None
            self._frame_index_set.clear()
            self._frame_index_last_refresh_s = 0.0
        logger.error("Remote HDF5 transport ended: %s", reason)

    def _fetch_metadata(self) -> None:
        """Fetch and commit one coherent metadata and frame-index snapshot."""
        if not self._stub:
            raise ConnectionError("Not connected to server")

        try:
            response = self._stub.GetFileServerMetadata(
                visualizer_pb2.FileServerMetadataRequest(),
                timeout=self.rpc_timeout,
            )
            if not response.success:
                raise RuntimeError(f"Failed to get metadata: {response.message}")
            if not response.snapshot_valid:
                raise RuntimeError(
                    "Remote frame snapshot is invalid: "
                    f"{response.snapshot_error or response.message}"
                )

            new_frame_set_id = str(response.frame_set_id or "")
            if not new_frame_set_id:
                raise RuntimeError("Remote metadata did not identify its frame set")
            material_properties_json = getattr(response, "material_properties_json", "")
            new_metadata = {
                "total_frames": response.total_frames,
                "num_tx": response.num_tx,
                "num_rx": response.num_rx,
                "num_targets": response.num_targets,
                "scene_name": response.scene_name,
                "source_directory": response.source_directory,
                "is_bulk_format": response.is_bulk_format,
                "frame_set_id": new_frame_set_id,
                "manifest_schema_version": response.manifest_schema_version,
                "first_frame_idx": response.first_frame_idx,
                "last_frame_idx": response.last_frame_idx,
                "chunk_size": response.chunk_size,
                "total_files": response.total_files,
                "snapshot_valid": response.snapshot_valid,
                "snapshot_error": response.snapshot_error,
                "git_sha": response.git_sha,
                "quality_profile_json": response.quality_profile_json,
                "material_properties_json": material_properties_json,
                "material_properties": self._decode_json_dict(material_properties_json),
            }

            list_response = self._stub.ListAvailableFrames(
                visualizer_pb2.ListFramesRequest(),
                timeout=self.rpc_timeout,
            )
            if not list_response.success:
                raise RuntimeError(f"Failed to list frames: {list_response.message}")
            list_frame_set_id = str(list_response.frame_set_id or "")
            if list_frame_set_id != new_frame_set_id:
                raise RuntimeError(
                    "Remote metadata and frame list describe different frame sets "
                    f"({new_frame_set_id!r} != {list_frame_set_id!r})"
                )

            new_frame_indices = list(list_response.frame_indices)
            previous_frame_set_id = (self._metadata or {}).get("frame_set_id")
            with self._cache_lock:
                if previous_frame_set_id != new_frame_set_id:
                    self._clear_frame_cache_locked()
                self._metadata = new_metadata
                self._frame_indices = new_frame_indices
                self._frame_index_set = set(new_frame_indices)
                self._frame_index_last_refresh_s = time.monotonic()
            if previous_frame_set_id and previous_frame_set_id != new_frame_set_id:
                logger.info(
                    "Remote HDF5 frame set changed (%s -> %s); committed a new snapshot",
                    previous_frame_set_id,
                    new_frame_set_id,
                )

        except grpc.RpcError as e:
            message = f"gRPC error fetching metadata: {e}"
            self._mark_transport_failed(message)
            raise ConnectionError(message) from e
        except RuntimeError as e:
            message = f"Remote HDF5 snapshot refresh failed: {e}"
            self._mark_transport_failed(message)
            raise ConnectionError(message) from e

    @staticmethod
    def _decode_json_dict(payload: object) -> dict[str, Any]:
        """Decode optional JSON metadata fields into dictionaries."""
        if not isinstance(payload, str) or not payload.strip():
            return {}
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def refresh(self) -> None:
        """Refresh remote metadata and available frame list."""
        if not self._connected:
            return
        self._fetch_metadata()

    def _maybe_refresh_frame_index(self) -> None:
        if not self._connected:
            return
        if self._frame_indices is None:
            self._fetch_metadata()
            return
        if self.frame_index_ttl_s <= 0.0:
            return
        age = time.monotonic() - self._frame_index_last_refresh_s
        if age >= self.frame_index_ttl_s:
            self._fetch_metadata()

    def _convert_protobuf_to_frame(
        self,
        frame_pb: Any,
        *,
        expected_frame_index: int | None = None,
    ) -> StandardMPCFrame:
        """Decode one validated compact frame from the remote service."""
        frame = standard_mpc_frame_from_proto(frame_pb)
        if expected_frame_index is not None and frame.frame_index != expected_frame_index:
            raise ValueError(
                f"Encoded frame index {frame.frame_index} does not match "
                f"requested frame {expected_frame_index}"
            )
        return frame

    def list_frames(self) -> list[int]:
        """Return list of available frame indices."""
        if not self._connected and self._frame_indices is None:
            return []
        self._maybe_refresh_frame_index()
        return list(self._frame_indices or [])

    def has_frame(self, step: int) -> bool:
        """Check if a frame is available."""
        self._maybe_refresh_frame_index()
        return step in self._frame_index_set

    def load_frame(self, step: int) -> StandardMPCFrame:
        """Load a frame from the remote server (with local caching)."""
        cached_frame: Optional[StandardMPCFrame] = None
        with self._cache_lock:
            expected_frame_set_id = str((self._metadata or {}).get("frame_set_id") or "")
            if step in self._frame_cache:
                self._frame_cache.move_to_end(step)
                logger.debug("Frame %d loaded from cache", step)
                cached_frame = self._frame_cache[step]
                self._cache_hits += 1
            else:
                self._cache_misses += 1
        if cached_frame is not None:
            return cached_frame

        # Fetch from server
        if not self._stub:
            raise ConnectionError("Not connected to server")
        if not expected_frame_set_id:
            message = "Remote HDF5 provider has no active frame-set identity"
            self._mark_transport_failed(message)
            raise ConnectionError(message)

        try:
            fetch_start = time.monotonic()
            response = self._stub.GetPreGeneratedFrame(
                visualizer_pb2.PreGeneratedFrameRequest(frame_idx=step),
                timeout=self.rpc_timeout,
            )

            response_frame_set_id = str(response.frame_set_id or "")
            if response_frame_set_id != expected_frame_set_id:
                message = (
                    "Remote HDF5 frame identity changed while loading "
                    f"frame {step}: {expected_frame_set_id!r} -> {response_frame_set_id!r}"
                )
                self._mark_transport_failed(message)
                raise ConnectionError(message)
            if not response.success:
                raise KeyError(f"Frame {step} not available: {response.message}")
            if int(response.frame_idx) != step:
                raise ValueError(
                    f"Remote response frame {response.frame_idx} does not match "
                    f"requested frame {step}"
                )

            frame = self._convert_protobuf_to_frame(
                response.frame_data,
                expected_frame_index=step,
            )
            retained_bytes = estimate_retained_bytes(frame)

            snapshot_changed = False
            with self._cache_lock:
                current_frame_set_id = str((self._metadata or {}).get("frame_set_id") or "")
                if current_frame_set_id != expected_frame_set_id:
                    snapshot_changed = True
                else:
                    self._store_frame_locked(step, frame, retained_bytes)
            if snapshot_changed:
                message = (
                    "Remote HDF5 snapshot changed before frame admission "
                    f"({expected_frame_set_id!r} -> {current_frame_set_id!r})"
                )
                self._mark_transport_failed(message)
                raise ConnectionError(message)

            fetch_latency_ms = (time.monotonic() - fetch_start) * 1000
            with self._cache_lock:
                self._last_fetch_latency_ms = fetch_latency_ms
            logger.debug(
                "Frame %d loaded from server (%.1f ms round-trip, %.1f ms server)",
                step,
                fetch_latency_ms,
                response.load_time_ms,
            )
            return frame

        except grpc.RpcError as e:
            message = f"gRPC error loading frame {step}: {e}"
            self._mark_transport_failed(message)
            raise ConnectionError(message) from e

    def prefetch_frames(self, start: int, count: int = 10) -> None:
        """Start background prefetching of frames.

        Args:
            start: Starting frame index
            count: Number of frames to prefetch
        """
        if not self._connected:
            return

        # Stop any existing prefetch
        self._prefetch_stop.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=PREFETCH_RESTART_JOIN_TIMEOUT_S)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.warning("Previous remote HDF5 prefetch is still stopping")
            return

        self._prefetch_thread = None
        self._prefetch_stop.clear()

        def _prefetch_worker():
            indices = self.list_frames()
            if not indices:
                return

            # Find frames to prefetch (starting from 'start')
            try:
                start_pos = indices.index(start)
            except ValueError:
                start_pos = 0

            frames_to_fetch = indices[start_pos : start_pos + count]

            for frame_idx in frames_to_fetch:
                if self._prefetch_stop.is_set():
                    break

                # Skip if already cached
                with self._cache_lock:
                    if frame_idx in self._frame_cache:
                        continue

                try:
                    self.load_frame(frame_idx)
                except (ConnectionError, OSError, RuntimeError, ValueError) as e:
                    logger.warning("Prefetch failed for frame %d: %s", frame_idx, e)

        self._prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    @property
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        frames = self.list_frames()
        return ProviderInfo(
            name="RemoteHdf5Provider",
            source=self.server_address,
            total_frames=len(frames),
            capabilities=ProviderCapability.RANDOM_ACCESS | ProviderCapability.CACHING,
            frame_set_id=self.frame_set_id or None,
        )

    @property
    def metadata(self) -> dict:
        """Return cached server metadata."""
        return self._metadata or {}

    @property
    def frame_set_id(self) -> str:
        """Return the connected server snapshot id, if known."""
        return str((self._metadata or {}).get("frame_set_id") or "")

    @property
    def is_connected(self) -> bool:
        """Return whether connected to server."""
        return self._connected

    @property
    def cached_frame_count(self) -> int:
        """Return the number of frames currently in the local cache."""
        with self._cache_lock:
            return len(self._frame_cache)

    @property
    def cached_frame_indices(self) -> list[int]:
        """Return list of frame indices currently in the local cache."""
        with self._cache_lock:
            return list(self._frame_cache.keys())

    @property
    def cache_hit_ratio(self) -> float:
        """Return cache hit ratio (0.0–1.0). Returns 0.0 if no requests yet."""
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            return self._cache_hits / total if total > 0 else 0.0

    @property
    def cache_stats(self) -> dict:
        """Return cache statistics dict."""
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            return {
                "frame_set_id": (self._metadata or {}).get("frame_set_id"),
                "cached_entries": len(self._frame_cache),
                "max_entries": self.cache_size,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "total": total,
                "hit_ratio": self._cache_hits / total if total > 0 else 0.0,
                "last_fetch_latency_ms": self._last_fetch_latency_ms,
                "current_bytes": self._cache_current_bytes,
                "peak_bytes": self._cache_peak_bytes,
                "evictions": self._cache_evictions,
            }


__all__ = ["RemoteHdf5Provider"]
