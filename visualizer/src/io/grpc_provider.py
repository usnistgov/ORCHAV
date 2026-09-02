#!/usr/bin/env python3
"""Live gRPC frame provider for the ORCHAV visualizer.

``GrpcProvider`` adapts the generator ``StreamFrames`` bidirectional RPC to the
visualizer ``DataProvider`` surface. It owns the client-side streaming thread,
request/waiter queues, local frame buffer, flow-control acknowledgements,
protobuf-to-frame conversion, and live edit/update side channels.
"""

import math
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import grpc

from shared.grpc_transport import (
    DEFAULT_GRPC_CONNECT_TIMEOUT_S,
    GRPC_MESSAGE_OPTIONS,
    format_grpc_endpoint,
    parse_grpc_endpoint,
)
from shared.logging import get_logger

logger = get_logger(__name__)

_FRAME_REQUEST_ERROR_CODES = frozenset(
    {
        "INVALID_FRAME",
        "COMPUTATION_FAILED",
        "ENCODING_FAILED",
        "FRAME_TOO_LARGE",
    }
)

try:
    from shared.protos import visualizer_pb2, visualizer_pb2_grpc
except ImportError as e:
    logger.error("Could not import gRPC generated code: %s", e)
    logger.error("Make sure the protobuf files are generated.")
    raise
logger.info(
    "Using visualizer protobuf modules: %s, %s",
    getattr(visualizer_pb2, "__file__", "unknown"),
    getattr(visualizer_pb2_grpc, "__file__", "unknown"),
)

from shared.frames.adapters import project_standard_mpc_frame
from shared.frames.contracts import FrameReadRequest
from shared.frames.packed import FrameProjection
from shared.frames.provider_base import DataProvider, ProviderCapability, ProviderInfo
from shared.frames.types import StandardMPCFrame

from ..scene.io import XMLSceneHandler
from .protobuf_deserializer import frame_from_proto

# Default playback jump detection thresholds (in frames)
DEFAULT_BACKWARD_JUMP_THRESHOLD = 5  # Backward jumps >= this trigger buffer clear
DEFAULT_FORWARD_JUMP_THRESHOLD = 10  # Forward jumps >= this trigger seeking state
_NON_PLAYBACK_FRAME_ORIGINS = frozenset({"projection"})


class LiveGrpcControllerBusyError(ConnectionError):
    """Raised when another visualizer owns the generator's live controller slot."""


def _validated_vector3(
    values: Sequence[float],
    *,
    label: str,
) -> Optional[tuple[float, float, float]]:
    """Return three finite floats or report a rejected live node update."""
    try:
        if len(values) != 3:
            logger.error("%s must contain exactly 3 values, got %s", label, len(values))
            return None
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.error("%s must contain exactly 3 numeric values: %s", label, exc)
        return None
    if not all(math.isfinite(value) for value in normalized):
        logger.error("%s values must all be finite", label)
        return None
    return normalized[0], normalized[1], normalized[2]


class PlaybackState(Enum):
    """Playback state machine states for live frame navigation."""

    IDLE = "idle"
    PLAYING_FORWARD = "playing_forward"
    PLAYING_BACKWARD = "playing_backward"
    SEEKING = "seeking"
    PAUSED = "paused"
    END_OF_TRACE = "end_of_trace"


@dataclass
class FrameTelemetry:
    """Request/response telemetry stored for live gRPC diagnostics."""

    frame_idx: int
    source: str  # "buffer" | "server" | "cache" | "timeout" | "error"
    requested_at: float
    received_at: Optional[float] = None
    latency_ms: Optional[float] = None
    buffer_size: int = 0
    buffer_utilization: float = 0.0
    origin: str = "user"
    error_message: Optional[str] = None


@dataclass(frozen=True)
class PendingFrameRequest:
    """Panel-facing snapshot of one pending live frame request."""

    frame_idx: int
    request_timestamp: Optional[float] = None
    origin: str = "server"
    has_waiter: bool = False
    age_seconds: Optional[float] = None


class PlaybackStateManager:
    """Manage playback state transitions and seek-buffer policy."""

    def __init__(
        self,
        backward_jump_threshold: int = DEFAULT_BACKWARD_JUMP_THRESHOLD,
        forward_jump_threshold: int = DEFAULT_FORWARD_JUMP_THRESHOLD,
        on_state_change: Optional[Callable[[PlaybackState, PlaybackState], None]] = None,
    ):
        """Configure jump thresholds and optional state-change callback."""
        self.state = PlaybackState.IDLE
        self.current_frame = -1
        self.backward_jump_threshold = backward_jump_threshold
        self.forward_jump_threshold = forward_jump_threshold
        self.on_state_change = on_state_change
        self._displayed_frames: Set[int] = set()  # Track which frames were actually displayed

    def begin_seek(
        self, target_idx: int, clear_buffer_callback: Optional[Callable[[], None]] = None
    ) -> tuple[bool, bool]:
        """
        Begin seeking to target frame.

        Returns:
            (should_clear_buffer, is_significant_jump)
        """
        old_state = self.state
        old_frame = self.current_frame

        # Determine if this is a significant jump
        is_backward = old_frame >= 0 and target_idx < old_frame
        is_forward = old_frame >= 0 and target_idx > old_frame
        jump_distance = abs(target_idx - old_frame) if old_frame >= 0 else 0

        should_clear = False
        is_significant = False

        # Special case: Detect looping (jumping from high frame to 0)
        # Don't clear buffer on loop - preserve frames for smooth looping
        is_loop = target_idx == 0 and old_frame > 10 and is_backward

        if is_backward and jump_distance >= self.backward_jump_threshold:
            # Don't clear buffer if this is a loop - preserve frames for reuse
            if not is_loop:
                should_clear = True
            is_significant = True
            self.state = PlaybackState.SEEKING
        elif is_forward and jump_distance >= self.forward_jump_threshold:
            is_significant = True
            self.state = PlaybackState.SEEKING
        elif old_state == PlaybackState.END_OF_TRACE:
            # Coming back from end of trace
            self.state = PlaybackState.SEEKING
            should_clear = True

        self.current_frame = target_idx

        if self.state != old_state and self.on_state_change:
            self.on_state_change(old_state, self.state)

        if should_clear and clear_buffer_callback:
            clear_buffer_callback()

        return should_clear, is_significant

    def ack_displayed(self, frame_idx: int) -> None:
        """Record that a frame was displayed to the user"""
        self.current_frame = frame_idx
        self._displayed_frames.add(frame_idx)

        # Transition out of seeking state
        if self.state == PlaybackState.SEEKING:
            self.state = PlaybackState.IDLE

    def begin_playback(self, direction: str = "forward") -> None:
        """Begin playback in specified direction"""
        old_state = self.state
        if direction == "forward":
            self.state = PlaybackState.PLAYING_FORWARD
        elif direction == "backward":
            self.state = PlaybackState.PLAYING_BACKWARD
        else:
            self.state = PlaybackState.IDLE

        if self.state != old_state and self.on_state_change:
            self.on_state_change(old_state, self.state)

    def handle_eof(self) -> None:
        """Handle end of trace"""
        old_state = self.state
        self.state = PlaybackState.END_OF_TRACE
        if self.on_state_change:
            self.on_state_change(old_state, self.state)

    def was_displayed(self, frame_idx: int) -> bool:
        """Check if frame was actually displayed"""
        return frame_idx in self._displayed_frames

    def pause(self) -> None:
        """Pause playback"""
        old_state = self.state
        self.state = PlaybackState.PAUSED
        if self.on_state_change:
            self.on_state_change(old_state, self.state)

    def reset(self) -> None:
        """Reset to idle state"""
        old_state = self.state
        self.state = PlaybackState.IDLE
        self.current_frame = -1
        if self.on_state_change:
            self.on_state_change(old_state, self.state)


class GrpcConnectionManager:
    """Own one gRPC channel and establish it only when explicitly requested."""

    def __init__(self, endpoint: str):
        """Store endpoint identity without connecting."""
        self.endpoint = endpoint
        self.channel = None
        self.stub = None
        self.connection_lock = threading.Lock()
        self.is_connected = False
        self.last_error = None

    def ensure_connection(self) -> bool:
        """Return the current connection or make one bounded connection attempt."""
        with self.connection_lock:
            if self.is_connected and self.channel is not None and self.stub is not None:
                return True

            return self._establish_connection()

    def _establish_connection(self) -> bool:
        """Establish new gRPC connection"""
        try:
            host, port = parse_grpc_endpoint(self.endpoint)
            address = format_grpc_endpoint(host, port)

            logger.info(f"Connecting to gRPC server at: {address}")
            logger.debug(f"Full endpoint: {self.endpoint}")

            options = [
                *GRPC_MESSAGE_OPTIONS,
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ]

            self.channel = grpc.insecure_channel(address, options=options)
            self.stub = visualizer_pb2_grpc.GeneratorServiceStub(self.channel)

            # Test connection
            request = visualizer_pb2.GetGeneratorStatusRequest()
            response = self.stub.GetGeneratorStatus(
                request,
                timeout=DEFAULT_GRPC_CONNECT_TIMEOUT_S,
            )

            if response.success:
                if bool(getattr(response, "is_streaming", False)):
                    raise LiveGrpcControllerBusyError(
                        "Live gRPC generator at "
                        f"{self.endpoint} already has an active controlling visualizer. "
                        "Close that visualizer before starting another live session."
                    )
                self.is_connected = True
                self.last_error = None
                logger.info(f"Successfully connected to gRPC server: {response.message}")
                return True
            else:
                raise grpc.RpcError(f"Server returned error: {response.message}")

        except LiveGrpcControllerBusyError as exc:
            self.is_connected = False
            self.last_error = exc
            logger.error("%s", exc)
            self._close_unlocked()
            return False
        except (grpc.RpcError, OSError, RuntimeError, ValueError) as e:
            self.is_connected = False
            self.last_error = e
            logger.error(
                "Failed to connect to gRPC server at %s: %s", self.endpoint, self._error_summary(e)
            )
            self._close_unlocked()
            return False

    def _error_summary(self, error: Exception) -> str:
        """Return a concise user-facing connection error summary."""
        if isinstance(error, grpc.RpcError):
            details_fn = getattr(error, "details", None)
            code_fn = getattr(error, "code", None)
            details = details_fn() if callable(details_fn) else str(error)
            code = code_fn() if callable(code_fn) else None
            code_name = getattr(code, "name", None)
            if code_name and details:
                return f"{code_name}: {details}"
            if details:
                return str(details)
        return str(error)

    def close(self) -> None:
        """Close the current channel without initiating another connection."""
        with self.connection_lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        """Close channel state while ``connection_lock`` is held by the caller."""
        if self.channel is not None:
            try:
                self.channel.close()
            except (OSError, RuntimeError) as e:
                logger.warning(f"Error closing gRPC channel: {e}")
        self.channel = None
        self.stub = None
        self.is_connected = False


class StreamingFrameBuffer:
    """Thread-safe frame buffer with display-anchored prefetch eviction."""

    def __init__(
        self,
        max_buffer_size: int = 10,
        on_evict: Optional[Callable[[List[int]], None]] = None,
        lookback: int = 0,
        lookahead: Optional[int] = None,
    ):
        """Create a bounded frame buffer and optional eviction callback."""
        self.buffer: OrderedDict[int, StandardMPCFrame] = OrderedDict()
        self.max_size = max_buffer_size
        self.lock = threading.Lock()
        self.latest_frame = -1
        self.oldest_frame = -1
        self.frame_timestamps: Dict[int, float] = {}
        self._on_evict = on_evict

        # Sliding window for prefetch rejection
        self.lookback = lookback
        # Default lookahead: use provided value, or default to buffer size
        # UI can override this via prefetch_lookahead control
        self.lookahead = lookahead if lookahead is not None else max_buffer_size
        self.current_display = -1  # Anchor for sliding window

    def set_display_position(self, frame_idx: int) -> None:
        """Update the current display position for sliding window calculations"""
        with self.lock:
            self.current_display = frame_idx

    def is_in_prefetch_window(self, frame_idx: int) -> bool:
        """Check if frame is within the prefetch window"""
        if self.current_display < 0:
            return True  # No position set, accept all

        window_min = self.current_display - self.lookback
        window_max = self.current_display + self.lookahead
        return window_min <= frame_idx <= window_max

    def add_frame(self, frame_idx: int, frame_data: StandardMPCFrame) -> None:
        """Add frame to buffer with size management"""
        evicted: List[int] = []
        with self.lock:
            if frame_idx in self.buffer:
                # Move to end (most recently used)
                self.buffer.move_to_end(frame_idx)
            else:
                self.buffer[frame_idx] = frame_data

            self.frame_timestamps[frame_idx] = time.time()
            self.latest_frame = max(self.latest_frame, frame_idx)
            if self.oldest_frame < 0 or frame_idx < self.oldest_frame:
                self.oldest_frame = frame_idx

            # Manage buffer size - evict frames intelligently based on proximity to current display
            if self.max_size > 0 and len(self.buffer) > self.max_size:
                frames_to_remove = []
                # Distance eviction preserves the display frame and its nearest neighbors.
                candidate_frames = []
                for key in self.buffer.keys():
                    if self.current_display >= 0 and key == self.current_display:
                        # Never evict the current display frame
                        continue
                    distance = (
                        abs(key - self.current_display)
                        if self.current_display >= 0
                        else float("inf")
                    )
                    candidate_frames.append((distance, key))

                candidate_frames.sort(key=lambda x: (-x[0], x[1]))

                # Evict frames starting with those farthest from current_display
                while len(self.buffer) > self.max_size and candidate_frames:
                    _, frame_to_evict = candidate_frames.pop(0)
                    if frame_to_evict in self.buffer:
                        frames_to_remove.append(frame_to_evict)
                        del self.buffer[frame_to_evict]
                        if frame_to_evict in self.frame_timestamps:
                            del self.frame_timestamps[frame_to_evict]
                        evicted.append(frame_to_evict)

                # FIFO is the final overflow policy when distance candidates
                # cannot satisfy the limit; current_display remains protected.
                while len(self.buffer) > self.max_size:
                    oldest_key, _ = next(iter(self.buffer.items()))
                    # Still protect current_display even in fallback
                    if self.current_display >= 0 and oldest_key == self.current_display:
                        # Skip current_display, try next oldest
                        if len(self.buffer) > 1:
                            keys = list(self.buffer.keys())
                            keys.sort()
                            for key in keys:
                                if key != self.current_display:
                                    oldest_key = key
                                    break
                        else:
                            # Only current_display left, can't evict
                            break
                    frames_to_remove.append(oldest_key)
                    del self.buffer[oldest_key]
                    if oldest_key in self.frame_timestamps:
                        del self.frame_timestamps[oldest_key]
                    evicted.append(oldest_key)

                # Recompute the lower bound only when its owner was evicted.
                if evicted and min(evicted) == self.oldest_frame:
                    if self.buffer:
                        self.oldest_frame = min(self.buffer.keys())
                    else:
                        self.oldest_frame = -1

                logger.debug(
                    f"[BUFFER] Overflow! Removed {len(evicted)} frames: {evicted[:5]}{'...' if len(evicted) > 5 else ''} (current_display={self.current_display})"
                )

            logger.debug(
                f"[BUFFER] Added frame {frame_idx} (size: {len(self.buffer)}/{self.max_size}, range: [{self.oldest_frame}, {self.latest_frame}])"
            )

        if evicted and self._on_evict:
            self._on_evict(evicted)

    def get_frame(self, frame_idx: int) -> Optional[StandardMPCFrame]:
        """Get frame from buffer (moves to end for LRU behavior)"""
        with self.lock:
            frame = self.buffer.get(frame_idx)
            if frame is not None:
                # Move to end (most recently used)
                self.buffer.move_to_end(frame_idx)
            available_frames = sorted(self.buffer.keys())

        # Log outside the lock to avoid holding it too long
        if frame is not None:
            logger.debug(
                f"[BUFFER] Retrieved frame {frame_idx} (available: {len(available_frames)} frames)"
            )
        else:
            logger.debug(
                f"[BUFFER] Frame {frame_idx} NOT in buffer (available: {len(available_frames)} frames)"
            )
        return frame

    def get_latest_frame(self) -> Optional[StandardMPCFrame]:
        """Get most recent frame"""
        with self.lock:
            if self.latest_frame >= 0 and self.latest_frame in self.buffer:
                return self.buffer[self.latest_frame]
            return None

    def get_available_frames(self) -> List[int]:
        """Get list of available frame indices"""
        with self.lock:
            return sorted(self.buffer.keys())

    def get_window_range(self) -> tuple[int, int]:
        """Get the contiguous window range [oldest, newest]"""
        with self.lock:
            if not self.buffer:
                return (-1, -1)
            return (self.oldest_frame, self.latest_frame)

    def clear(self, *, notify_eviction: bool = True) -> None:
        """Clear the buffer and optionally report dropped frames."""
        evicted: List[int] = []
        with self.lock:
            if self.buffer:
                evicted = sorted(list(self.buffer.keys()))
            logger.debug(f"[BUFFER] Clearing buffer (was holding {len(evicted)} frames)")
            self.buffer.clear()
            self.frame_timestamps.clear()
            self.latest_frame = -1
            self.oldest_frame = -1
        if notify_eviction and evicted and self._on_evict:
            self._on_evict(evicted)

    def set_max_size(self, new_size: int):
        """Update the max buffer size and evict frames if necessary."""
        evicted: List[int] = []
        with self.lock:
            self.max_size = new_size
            if self.max_size >= 0 and len(self.buffer) > self.max_size:
                candidate_frames = []
                for key in self.buffer.keys():
                    if self.current_display >= 0 and key == self.current_display:
                        # Never evict the current display frame
                        continue
                    distance = (
                        abs(key - self.current_display)
                        if self.current_display >= 0
                        else float("inf")
                    )
                    candidate_frames.append((distance, key))

                candidate_frames.sort(key=lambda x: (-x[0], x[1]))

                # Evict frames starting with those farthest from current_display
                while len(self.buffer) > self.max_size and candidate_frames:
                    _, frame_to_evict = candidate_frames.pop(0)
                    if frame_to_evict in self.buffer:
                        del self.buffer[frame_to_evict]
                        if frame_to_evict in self.frame_timestamps:
                            del self.frame_timestamps[frame_to_evict]
                        evicted.append(frame_to_evict)

                # FIFO resolves any remaining overflow while protecting current_display.
                while len(self.buffer) > self.max_size:
                    oldest_key, _ = next(iter(self.buffer.items()))
                    if self.current_display >= 0 and oldest_key == self.current_display:
                        # Skip current_display, try next oldest
                        if len(self.buffer) > 1:
                            keys = list(self.buffer.keys())
                            keys.sort()
                            for key in keys:
                                if key != self.current_display:
                                    oldest_key = key
                                    break
                        else:
                            # Only current_display left, can't evict
                            break
                    del self.buffer[oldest_key]
                    if oldest_key in self.frame_timestamps:
                        del self.frame_timestamps[oldest_key]
                    evicted.append(oldest_key)

                if evicted and min(evicted) == self.oldest_frame:
                    if self.buffer:
                        self.oldest_frame = min(self.buffer.keys())
                    else:
                        self.oldest_frame = -1

                logger.debug(
                    f"[BUFFER] Resized to {new_size}, removed {len(evicted)} frames (current_display={self.current_display})"
                )
        if evicted and self._on_evict:
            self._on_evict(evicted)


class GrpcProvider(DataProvider):
    """Live generator data provider backed by the ``StreamFrames`` RPC."""

    REQUEST_TIMEOUT = 30.0
    OBJECT_UPDATE_TIMEOUT = 120.0
    ACK_BATCH_INTERVAL_SECONDS = 1.0

    def __init__(self, endpoint: str, *, buffer_size: int = 50):
        """Initialize live transport state without opening the network connection."""
        self.endpoint = endpoint
        self.connection_manager = GrpcConnectionManager(endpoint)
        self.streaming_buffer = StreamingFrameBuffer(
            max_buffer_size=buffer_size, on_evict=self._handle_evicted_frames
        )
        self.subscribers: List[Callable[[int, StandardMPCFrame], None]] = []
        self.frame_info_cache: Optional[Dict[str, Any]] = None
        self.cache_timestamp = 0.0
        self.cache_ttl = 5.0

        self._request_queue: "queue.Queue[Optional[visualizer_pb2.FrameRequest]]" = queue.Queue()
        self._pending_lock = threading.Lock()
        self._pending_frames: Dict[
            int, List[Optional["queue.Queue[Optional[StandardMPCFrame]]"]]
        ] = {}
        self._non_playback_pending_frames: Set[int] = set()
        self._initial_prefetch_sent = False  # Track if initial prefetch flow control was sent
        self._initial_prefetch_time = 0.0  # When initial prefetch was sent
        self._shutdown = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None

        self._last_error: Optional[str] = None
        self._disconnect_reason: Optional[str] = "Provider is not open"
        self._last_displayed_frame = -1
        self._generation_epoch = 0
        self._stale_frames_dropped = 0

        # Playback state management
        self.playback_state = PlaybackStateManager(
            backward_jump_threshold=DEFAULT_BACKWARD_JUMP_THRESHOLD,
            forward_jump_threshold=DEFAULT_FORWARD_JUMP_THRESHOLD,
            on_state_change=self._on_playback_state_change,
        )

        # Track displayed frames separately from acknowledged frames
        # (for proper eviction handling - only acknowledge displayed frames)
        self._displayed_frames: Set[int] = set()
        self._discarded_frames: Set[int] = set()  # Frames evicted before display

        # For batched acknowledgements
        self._acknowledged_frames: Set[int] = set()
        self._ack_lock = threading.Lock()
        self._unacknowledged_frames_count = 0
        self._ack_timer: Optional[threading.Timer] = None

        # Parameter update coordination
        self._param_update_waiter: Optional[
            "queue.Queue[visualizer_pb2.ParameterUpdateResponse]"
        ] = None
        self._param_update_waiter_lock = threading.Lock()

        # Object update coordination
        self._object_update_waiter: Optional["queue.Queue[visualizer_pb2.ObjectUpdateResponse]"] = (
            None
        )
        self._object_update_waiter_lock = threading.Lock()
        self._object_update_listeners: List[
            Callable[[visualizer_pb2.ObjectUpdateResponse], None]
        ] = []

        # Request history tracking (max 100 entries)
        self._request_history: List[Dict[str, Any]] = []
        self._request_history_max = 100
        self._request_history_lock = threading.Lock()
        self._prefetch_enabled = True
        self._play_direction = 1

        # Latency tracking
        self._pending_requests: Dict[int, float] = {}  # frame_idx -> request_timestamp
        self._pending_request_origins: Dict[int, str] = {}
        self._generation_times: List[float] = []  # Track last 100 generation times
        self._generation_times_max = 100
        self._last_flow_control_time = 0.0

        # Connection health tracking
        self._connection_start_time: Optional[float] = None
        self._last_error_timestamp: Optional[float] = None

    # Provider Info
    @property
    def info(self) -> ProviderInfo:
        """Return provider metadata with streaming capabilities."""
        frame_info = self._get_frame_info()
        total = frame_info.get("total_frames", -1) if frame_info else -1
        return ProviderInfo(
            name="GrpcProvider",
            source=self.endpoint,
            total_frames=total,
            capabilities=(
                ProviderCapability.STREAMING
                | ProviderCapability.CACHING
                | ProviderCapability.RANDOM_ACCESS
                | ProviderCapability.OVERRIDES
            ),
        )

    # Lifecycle
    def open(self) -> None:
        """Establish the gRPC connection and start the streaming loop."""
        if self._stream_thread is not None and self._stream_thread.is_alive():
            if self._shutdown.is_set():
                raise RuntimeError(
                    "Cannot reopen live gRPC provider while its previous stream is stopping"
                )
            return
        self._stream_thread = None
        logger.info("GrpcProvider connecting to %s", self.endpoint)
        self._shutdown.clear()
        self._disconnect_reason = None
        self._last_error = None
        self._last_error_timestamp = None
        self._request_queue = queue.Queue()
        if not self.connection_manager.ensure_connection():
            connection_error = self.connection_manager.last_error
            if isinstance(connection_error, LiveGrpcControllerBusyError):
                self._disconnect_reason = str(connection_error)
                raise connection_error
            detail = self.connection_manager._error_summary(self.connection_manager.last_error)
            self._disconnect_reason = detail
            raise ConnectionError(
                "Live gRPC server is unavailable at "
                f"{self.endpoint}: {detail}. Start the generator for this scenario "
                "or update live_grpc_endpoints.sionna to the running server address."
            )
        self._reset_session_state()
        self._connection_start_time = time.time()
        self._start_stream_loop()

    def _reset_session_state(self) -> None:
        """Reset values whose meaning is scoped to one server connection."""
        self.frame_info_cache = None
        self.cache_timestamp = 0.0
        self._generation_epoch = 0
        self._stale_frames_dropped = 0
        self._initial_prefetch_sent = False
        self._initial_prefetch_time = 0.0
        self._last_displayed_frame = -1
        self._displayed_frames.clear()
        self._discarded_frames.clear()
        self.streaming_buffer.clear(notify_eviction=False)
        self.streaming_buffer.set_display_position(-1)
        self.playback_state.reset()
        with self._ack_lock:
            self._acknowledged_frames.clear()
            self._unacknowledged_frames_count = 0

    def close(self) -> None:
        """Stop streaming, close the connection, and release resources."""
        self._shutdown.set()
        self._disconnect_reason = "Provider closed"
        self._connection_start_time = None
        self._stop_ack_timer()
        self._request_queue.put(None)
        self.connection_manager.close()

        stream_thread = self._stream_thread
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)
        stream_still_alive = stream_thread is not None and stream_thread.is_alive()
        if stream_still_alive:
            logger.warning(
                "Live gRPC stream thread did not stop within the close timeout; "
                "reopen remains disabled until it exits"
            )
        else:
            self._stream_thread = None

        self._release_pending_waiters("Provider closed")

        self.clear_buffer()
        if not stream_still_alive:
            self._request_queue = queue.Queue()
        self.subscribers.clear()
        logger.info("gRPC provider closed")

    # Pending request snapshots/cancellation
    def get_pending_frame_requests(self) -> List[PendingFrameRequest]:
        """Return a thread-safe snapshot of pending live frame requests."""
        now = time.time()
        with self._pending_lock:
            pending = [
                (frame_idx, any(waiter is not None for waiter in waiters))
                for frame_idx, waiters in self._pending_frames.items()
            ]

        with self._request_history_lock:
            snapshots = [
                PendingFrameRequest(
                    frame_idx=frame_idx,
                    request_timestamp=self._pending_requests.get(frame_idx),
                    origin=self._pending_request_origins.get(frame_idx, "server"),
                    has_waiter=has_waiter,
                    age_seconds=(
                        max(0.0, now - self._pending_requests[frame_idx])
                        if frame_idx in self._pending_requests
                        else None
                    ),
                )
                for frame_idx, has_waiter in pending
            ]
        return sorted(snapshots, key=lambda item: item.frame_idx)

    def _cancel_pending_frame_locked(self, frame_idx: int) -> bool:
        """Cancel one pending request while ``_pending_lock`` is already held."""
        if frame_idx not in self._pending_frames:
            return False
        waiters = self._pending_frames.pop(frame_idx)
        self._non_playback_pending_frames.discard(frame_idx)
        for waiter in waiters:
            if waiter is not None:
                try:
                    waiter.put_nowait(None)
                except queue.Full:
                    pass
        with self._request_history_lock:
            self._pending_requests.pop(frame_idx, None)
            self._pending_request_origins.pop(frame_idx, None)
        return True

    def cancel_pending_frame_request(self, frame_idx: int) -> bool:
        """Cancel a pending live frame request and notify its waiter if present."""
        with self._pending_lock:
            cancelled = self._cancel_pending_frame_locked(int(frame_idx))
        if cancelled:
            logger.info("Cancelled pending gRPC frame request %s", frame_idx)
        return cancelled

    def cancel_pending_frame_requests(
        self, frame_indices: Optional[Sequence[int]] = None
    ) -> List[int]:
        """Cancel selected pending live frame requests, or all pending requests."""
        with self._pending_lock:
            if frame_indices is None:
                targets = list(self._pending_frames.keys())
            else:
                targets = [int(frame_idx) for frame_idx in frame_indices]
            cancelled = [
                frame_idx for frame_idx in targets if self._cancel_pending_frame_locked(frame_idx)
            ]
        if cancelled:
            logger.info("Cancelled %d pending gRPC frame request(s)", len(cancelled))
        return cancelled

    def _transport_available(self) -> bool:
        """Return whether the explicitly opened stream can accept requests."""
        return bool(
            self._disconnect_reason is None
            and not self._shutdown.is_set()
            and self.connection_manager.is_connected
            and self.connection_manager.stub is not None
        )

    def _release_pending_waiters(self, reason: str) -> None:
        """Fail all in-flight operations immediately after transport termination."""
        with self._pending_lock:
            pending_frames = list(self._pending_frames.items())
            self._pending_frames.clear()
            self._non_playback_pending_frames.clear()

        with self._request_history_lock:
            request_context = {
                frame_idx: (
                    self._pending_requests.pop(frame_idx, time.time()),
                    self._pending_request_origins.pop(frame_idx, "server"),
                )
                for frame_idx, _waiters in pending_frames
            }

        for frame_idx, waiters in pending_frames:
            request_timestamp, origin = request_context[frame_idx]
            self._record_request(
                frame_idx,
                "GET_FRAME",
                request_timestamp,
                False,
                error_message=reason,
                source="server",
                origin=origin,
            )
            for waiter in waiters:
                if waiter is not None:
                    try:
                        waiter.put_nowait(None)
                    except queue.Full:
                        pass

        with self._param_update_waiter_lock:
            param_waiter = self._param_update_waiter
            self._param_update_waiter = None
        if param_waiter is not None:
            try:
                param_waiter.put_nowait(
                    visualizer_pb2.ParameterUpdateResponse(success=False, message=reason)
                )
            except queue.Full:
                pass

        with self._object_update_waiter_lock:
            object_waiter = self._object_update_waiter
            self._object_update_waiter = None
        if object_waiter is not None:
            try:
                object_waiter.put_nowait(
                    visualizer_pb2.ObjectUpdateResponse(success=False, message=reason)
                )
            except queue.Full:
                pass

    def _handle_stream_disconnect(self, reason: str) -> None:
        """Make a failed stream terminal until the provider is explicitly reopened."""
        if self._shutdown.is_set():
            return
        self._disconnect_reason = reason
        self._last_error = reason
        self._last_error_timestamp = time.time()
        self._connection_start_time = None
        self._shutdown.set()
        self._stop_ack_timer()
        self.connection_manager.close()
        self._release_pending_waiters(reason)
        self._request_queue = queue.Queue()

    # Streaming loop
    def _start_stream_loop(self) -> None:
        """Start the background StreamFrames loop and advertise initial capacity."""
        if self._stream_thread and self._stream_thread.is_alive():
            return
        self._shutdown.clear()
        self._stream_thread = threading.Thread(target=self._run_stream_loop, daemon=True)
        self._stream_thread.start()
        self._acknowledged_frames.clear()
        # EOF keeps the stream reusable for navigation, so the ack timer restarts here.
        self._start_ack_timer()
        lookahead = (
            getattr(self.streaming_buffer, "lookahead", None) or self.streaming_buffer.max_size
        )
        logger.info(
            f"[INIT] Sending initial flow control: buffer_capacity={self.streaming_buffer.max_size} "
            f"(transmission limit), server_prefetch_limit={lookahead} (computation limit)"
        )
        self._enqueue_flow_control(0, -1)
        self._initial_prefetch_sent = True
        self._initial_prefetch_time = time.time()

    def _on_playback_state_change(self, old_state: PlaybackState, new_state: PlaybackState) -> None:
        """Callback when playback state changes"""
        logger.debug(f"[PLAYBACK] State change: {old_state.value} -> {new_state.value}")

    def _run_stream_loop(self) -> None:
        """Run one bidirectional StreamFrames RPC until shutdown or disconnect."""
        logger.debug("Streaming loop started")
        stub = self.connection_manager.stub
        disconnect_reason: Optional[str] = None

        def request_iterator():
            """Yield queued outbound stream requests until shutdown."""
            while not self._shutdown.is_set():
                try:
                    request = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if request is None:
                    logger.debug("Request iterator received shutdown signal")
                    return
                yield request

        if stub is None:
            disconnect_reason = "Live gRPC stream could not start: connection is unavailable"
        else:
            try:
                # This production stream intentionally has no overall deadline.
                for response in stub.StreamFrames(request_iterator()):
                    if self._shutdown.is_set():
                        break
                    self._handle_stream_response(response)
                if not self._shutdown.is_set():
                    disconnect_reason = "Live gRPC stream ended unexpectedly"
            except grpc.RpcError as exc:
                if not self._shutdown.is_set():
                    disconnect_reason = (
                        "Live gRPC stream disconnected: "
                        f"{self.connection_manager._error_summary(exc)}"
                    )
                    logger.error("StreamFrames RPC terminated: %s", disconnect_reason)

        if disconnect_reason is not None:
            self._handle_stream_disconnect(disconnect_reason)
        logger.debug("Streaming loop terminated")

    # Request helpers
    def _enqueue_get_frame(self, frame_idx: int) -> None:
        """Queue a frame request and track it for latency/status reporting."""
        request = visualizer_pb2.FrameRequest(
            get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=frame_idx)
        )
        # Track request for latency measurement
        request_timestamp = time.time()
        with self._request_history_lock:
            self._pending_requests[frame_idx] = request_timestamp
        self._request_queue.put(request)

    def _enqueue_overrides(self, overrides: List[Dict[str, Any]]) -> None:
        """Queue TX/RX/target overrides for the next live frame request."""
        if not overrides:
            return

        type_mapping = {
            "tx": visualizer_pb2.NODE_TYPE_TX,
            "rx": visualizer_pb2.NODE_TYPE_RX,
            "target": visualizer_pb2.NODE_TYPE_TARGET,
        }

        override_messages = []
        for override in overrides:
            category = override.get("type", "").lower()
            node_type = type_mapping.get(category, visualizer_pb2.NODE_TYPE_UNSPECIFIED)
            position = override.get("position", [0.0, 0.0, 0.0])
            orientation = override.get("orientation", [])
            override_messages.append(
                visualizer_pb2.NodeOverride(
                    name=override.get("name", ""),
                    type=node_type,
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    orientation=[float(o) for o in orientation],
                )
            )

        request = visualizer_pb2.FrameRequest(
            override_cmd=visualizer_pb2.NodeOverrideList(overrides=override_messages)
        )
        # Track override request
        self._record_request(-1, "OVERRIDE", time.time(), True)
        self._request_queue.put(request)

    def _enqueue_flow_control(
        self,
        frames_consumed: int,
        current_display: int,
        *,
        buffer_capacity: Optional[int] = None,
        server_prefetch_limit: Optional[int] = None,
    ) -> None:
        """
        Send flow control signal to server.

        buffer_capacity: Maximum frames to send to client (transmission limit).
                         Should be set to max_buffer_size to avoid wasted bandwidth.

        server_prefetch_limit: Maximum frames to compute ahead on server (computation limit).
                               Frames are computed and cached but NOT sent automatically.
                               Should be set to lookahead for maximum server-side prefetching.
        """
        if buffer_capacity is None:
            # Transmission limit: only send what client can hold
            buf_cap = self.streaming_buffer.max_size
        else:
            buf_cap = buffer_capacity

        if server_prefetch_limit is None:
            # Computation limit: compute ahead based on UI lookahead
            lookahead = (
                getattr(self.streaming_buffer, "lookahead", None) or self.streaming_buffer.max_size
            )
            prefetch_limit = lookahead
        else:
            prefetch_limit = server_prefetch_limit

        logger.debug(
            f"[FLOW_CONTROL] Enqueuing: frames_consumed={frames_consumed}, current_display={current_display}, "
            f"buffer_capacity={buf_cap} (transmission limit), server_prefetch_limit={prefetch_limit} (computation limit)"
        )
        request = visualizer_pb2.FrameRequest(
            flow_control=visualizer_pb2.FlowControlSignal(
                frames_consumed=frames_consumed,
                buffer_capacity=buf_cap,
                current_display_frame_idx=current_display,
                server_prefetch_limit=prefetch_limit,
            )
        )
        self._request_queue.put(request)
        self._last_flow_control_time = time.time()
        if frames_consumed > 0 or buffer_capacity is not None:
            details = f"frames_consumed={frames_consumed}, capacity={buf_cap}"
            self._record_request(
                frame_idx=current_display,
                request_type="FLOW_CONTROL",
                request_timestamp=self._last_flow_control_time,
                success=True,
                source="client",
                origin="flow_control",
                details=details,
            )

    # Response handling
    def _handle_stream_response(self, response: visualizer_pb2.FrameResponse) -> None:
        """Dispatch one server stream response into buffers, waiters, or listeners."""
        resp_type = response.WhichOneof("response_type")
        frame_idx = response.frame_idx
        response_epoch = int(getattr(response, "generation_epoch", 0) or 0)

        if resp_type == "param_update_response":
            param_resp = response.param_update_response
            self._generation_epoch = max(
                self._generation_epoch,
                int(getattr(param_resp, "generation_epoch", response_epoch) or response_epoch),
            )
            now = time.time()
            self._record_request(
                -1,
                "PARAM_UPDATE_ACK",
                now,
                param_resp.success,
                error_message=None if param_resp.success else param_resp.message,
            )

            if param_resp.success:
                logger.info(
                    "Parameter update acknowledged (cache_flushed=%s)",
                    param_resp.cache_flushed,
                )
                if param_resp.cache_flushed:
                    self._invalidate_local_frame_state()
                # Resume with a small window; invalidated state restarts at frame 0.
                current_frame = (
                    self._last_displayed_frame if self._last_displayed_frame >= 0 else -1
                )
                target_capacity = (
                    min(5, self.streaming_buffer.max_size) if self.streaming_buffer.max_size else 0
                )
                self._enqueue_flow_control(
                    0,
                    current_frame,
                    buffer_capacity=target_capacity,
                    server_prefetch_limit=target_capacity,
                )
            else:
                logger.error("Parameter update failed: %s", param_resp.message)
                # Ensure stream resumes even if update failed
                self._enqueue_flow_control(0, self._last_displayed_frame)

            waiter = None
            with self._param_update_waiter_lock:
                waiter = self._param_update_waiter
                self._param_update_waiter = None
            if waiter is not None:
                try:
                    waiter.put_nowait(param_resp)
                except queue.Full:
                    pass

            return

        if resp_type == "object_update_response":
            obj_resp = response.object_update_response
            self._generation_epoch = max(
                self._generation_epoch,
                int(getattr(obj_resp, "generation_epoch", response_epoch) or response_epoch),
            )
            timestamp = time.time()
            details = (
                f"name={obj_resp.object_name or 'unknown'}, "
                f"cache_flushed={obj_resp.cache_flushed}, "
                f"recompute={obj_resp.recomputation_triggered}"
            )
            if obj_resp.success:
                logger.info(
                    "[OBJECT_UPDATE] Ack success (cache_flushed=%s, recomputation=%s)",
                    obj_resp.cache_flushed,
                    obj_resp.recomputation_triggered,
                )
                self._record_request(
                    -1,
                    "OBJECT_UPDATE_ACK",
                    timestamp,
                    True,
                    details=details,
                )
                if obj_resp.cache_flushed:
                    self._invalidate_local_frame_state()
                    target_capacity = (
                        min(5, self.streaming_buffer.max_size)
                        if self.streaming_buffer.max_size
                        else 0
                    )
                    self._enqueue_flow_control(
                        0,
                        -1,
                        buffer_capacity=target_capacity,
                        server_prefetch_limit=target_capacity,
                    )
                else:
                    self._enqueue_get_frame(0)
            else:
                logger.error("[OBJECT_UPDATE] Ack failed: %s", obj_resp.message)
                self._record_request(
                    -1,
                    "OBJECT_UPDATE_ACK",
                    timestamp,
                    False,
                    error_message=obj_resp.message,
                    details=details,
                )

            waiter = None
            with self._object_update_waiter_lock:
                waiter = self._object_update_waiter
                self._object_update_waiter = None
            if waiter is not None:
                try:
                    waiter.put_nowait(obj_resp)
                except queue.Full:
                    pass

            for callback in list(self._object_update_listeners):
                try:
                    callback(obj_resp)
                except Exception as exc:  # broad catch: callback dispatcher
                    logger.error("Object update listener raised: %s", exc)
            return

        if resp_type == "cache_flush_response":
            cache_resp = response.cache_flush_response
            self._generation_epoch = max(
                self._generation_epoch,
                int(getattr(cache_resp, "generation_epoch", response_epoch) or response_epoch),
            )
            timestamp = time.time()
            if cache_resp.success:
                logger.info(
                    "[CACHE_FLUSH] Ack received (client=%s, server=%s, frames_cleared=%s)",
                    cache_resp.client_cache_flushed,
                    cache_resp.server_cache_flushed,
                    cache_resp.frames_cleared,
                )
                self._record_request(-1, "CACHE_FLUSH", timestamp, True)
                if cache_resp.client_cache_flushed or cache_resp.server_cache_flushed:
                    self._invalidate_local_frame_state()
                    target_capacity = (
                        min(5, self.streaming_buffer.max_size)
                        if self.streaming_buffer.max_size
                        else 0
                    )
                    self._enqueue_flow_control(
                        0,
                        -1,
                        buffer_capacity=target_capacity,
                        server_prefetch_limit=target_capacity,
                    )
            else:
                logger.error("[CACHE_FLUSH] Failed: %s", cache_resp.message)
                self._record_request(
                    -1, "CACHE_FLUSH", timestamp, False, error_message=cache_resp.message
                )
            return

        if resp_type == "frame_data":
            if response_epoch < self._generation_epoch:
                self._stale_frames_dropped += 1
                logger.debug(
                    "[STREAM] Dropping stale frame %s from epoch %s (current epoch=%s)",
                    frame_idx,
                    response_epoch,
                    self._generation_epoch,
                )
                with self._pending_lock:
                    self._cancel_pending_frame_locked(frame_idx)
                return
            if response_epoch > self._generation_epoch:
                self._generation_epoch = response_epoch
            logger.debug(f"[STREAM] Received frame_data response for frame {frame_idx}")
            with self._pending_lock:
                is_non_playback_request = frame_idx in self._non_playback_pending_frames
                waiters = self._pending_frames.pop(frame_idx, [])
                self._non_playback_pending_frames.discard(frame_idx)
            request_timestamp = None
            origin = "server"
            with self._request_history_lock:
                request_timestamp = self._pending_requests.pop(frame_idx, None)
                origin = self._pending_request_origins.pop(frame_idx, "server")
            buffer_size_snapshot, buffer_util_snapshot = self._current_buffer_metrics()

            # The sliding window rejects playback/prefetch responses made stale by
            # a later seek. Sequential projection scans are non-playback consumers
            # and must not be discarded merely because playback moved meanwhile.
            if not is_non_playback_request and not self.streaming_buffer.is_in_prefetch_window(
                frame_idx
            ):
                logger.debug(
                    f"[STREAM] Discarding frame {frame_idx} (outside prefetch window, current_display={self.streaming_buffer.current_display})"
                )
                # Still record the request as received (but discarded)
                if request_timestamp is not None:
                    generation_time = (time.time() - request_timestamp) * 1000.0
                    self._record_request(
                        frame_idx,
                        "GET_FRAME",
                        request_timestamp,
                        True,
                        generation_time=generation_time,
                        error_message="Frame discarded - outside prefetch window",
                        source="server",
                        origin=origin,
                        buffer_size=buffer_size_snapshot,
                        buffer_utilization=buffer_util_snapshot,
                    )
                for waiter in waiters:
                    if waiter is not None:
                        try:
                            waiter.put_nowait(None)
                        except queue.Full:
                            pass
                return

            frame = self._frame_from_proto(response.frame_data, frame_idx)
            if frame is None:
                logger.error("Stream frame decode failed for frame %s", frame_idx)
                timestamp = request_timestamp or time.time()
                generation_time = (
                    (time.time() - request_timestamp) * 1000.0 if request_timestamp else None
                )
                self._record_request(
                    frame_idx,
                    "GET_FRAME",
                    timestamp,
                    False,
                    generation_time=generation_time,
                    error_message="Failed to decode frame",
                    source="server",
                    origin=origin,
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )
                for waiter in waiters:
                    if waiter is not None:
                        try:
                            waiter.put_nowait(None)
                        except queue.Full:
                            pass
                return

            logger.debug(
                f"📥 [STREAM] Decoded frame {frame_idx} successfully (request_timestamp={request_timestamp})"
            )

            if request_timestamp is not None:
                generation_time = (time.time() - request_timestamp) * 1000.0
                self._record_request(
                    frame_idx,
                    "GET_FRAME",
                    request_timestamp,
                    True,
                    generation_time=generation_time,
                    source="server",
                    origin=origin,
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )
            else:
                now = time.time()
                self._record_request(
                    frame_idx,
                    "GET_FRAME",
                    now,
                    True,
                    source="server",
                    origin=origin,
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )

            # Synchronous callers share one transport request for this index;
            # unsolicited and fire-and-forget responses populate the buffer.
            synchronous_waiters = [waiter for waiter in waiters if waiter is not None]
            if synchronous_waiters:
                logger.debug(
                    "[STREAM] Delivering frame %s to %s waiting caller(s)",
                    frame_idx,
                    len(synchronous_waiters),
                )
                for waiter in synchronous_waiters:
                    try:
                        waiter.put_nowait(frame)
                    except queue.Full:
                        logger.warning("[STREAM] Waiter queue full for frame %s", frame_idx)
            else:
                # No active waiter - add to buffer for future requests
                logger.debug(
                    f"[SAVE] [STREAM] Frame {frame_idx} added to buffer (no active waiter, will be available for future requests)"
                )
                self.streaming_buffer.add_frame(frame_idx, frame)

            for callback in list(self.subscribers):
                try:
                    callback(frame_idx, frame)
                except Exception as exc:  # broad catch: callback dispatcher
                    logger.error("Frame subscriber raised: %s", exc)

        elif resp_type == "error":
            error_code = response.error.code
            message = response.error.message
            if error_code not in _FRAME_REQUEST_ERROR_CODES:
                logger.error("Live stream command error %s: %s", error_code or "UNKNOWN", message)
                buffer_size_snapshot, buffer_util_snapshot = self._current_buffer_metrics()
                self._record_request(
                    -1,
                    "STREAM_COMMAND",
                    time.time(),
                    False,
                    error_message=f"{error_code or 'UNKNOWN'}: {message}",
                    origin="server",
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )
                return

            logger.error(
                "Streaming frame error %s for frame %s: %s",
                error_code,
                frame_idx,
                message,
            )

            with self._request_history_lock:
                request_timestamp = self._pending_requests.pop(frame_idx, None)
                origin = self._pending_request_origins.pop(frame_idx, "server")
            buffer_size_snapshot, buffer_util_snapshot = self._current_buffer_metrics()

            if request_timestamp is not None:
                generation_time = (time.time() - request_timestamp) * 1000.0
                self._record_request(
                    frame_idx,
                    "GET_FRAME",
                    request_timestamp,
                    False,
                    generation_time=generation_time,
                    error_message=message,
                    origin=origin,
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )
            else:
                self._record_request(
                    frame_idx,
                    "GET_FRAME",
                    time.time(),
                    False,
                    error_message=message,
                    origin=origin,
                    buffer_size=buffer_size_snapshot,
                    buffer_utilization=buffer_util_snapshot,
                )

            with self._pending_lock:
                waiters = self._pending_frames.pop(frame_idx, [])
                self._non_playback_pending_frames.discard(frame_idx)
            for waiter in waiters:
                if waiter is not None:
                    try:
                        waiter.put_nowait(None)
                    except queue.Full:
                        pass

        elif resp_type == "eof":
            logger.info("Received end-of-stream at frame %s", response.eof.last_frame_idx)
            self.playback_state.handle_eof()
            # Don't set _shutdown on EOF - allow user to navigate back
            # The stream should continue to accept requests

        elif resp_type == "status_update":
            logger.debug(
                "Status update received: ready=%s streaming=%s",
                response.status_update.is_ready,
                response.status_update.is_streaming,
            )

    # Frame conversion utilities (delegated to protobuf_deserializer)
    def _frame_from_proto(
        self, frame_pb: visualizer_pb2.FrameData, frame_idx: int
    ) -> Optional[StandardMPCFrame]:
        """Decode one protobuf frame through the shared deserializer."""
        return frame_from_proto(frame_pb, frame_idx)

    # Public API used by the visualizer
    def load_frame(self, step: int, *, origin: str = "user") -> Optional[StandardMPCFrame]:
        """Load one playback frame while applying seek, buffer, and prefetch policy."""
        return self._load_frame(step, origin=origin, affects_playback=True)

    def iter_frame_projections(
        self,
        steps: Iterable[int],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Yield sequential projections without moving the playback window."""
        for raw_step in steps:
            step = int(raw_step)
            frame = self._load_frame(step, origin="projection", affects_playback=False)
            if frame is None:
                raise KeyError(f"Frame {step} is not available from {type(self).__name__}")
            yield project_standard_mpc_frame(frame, request)

    def _load_frame(
        self,
        step: int,
        *,
        origin: str,
        affects_playback: bool,
    ) -> Optional[StandardMPCFrame]:
        """Request one frame, optionally applying playback-state side effects."""
        request_start = time.time()
        buffer_size_snapshot = len(self.streaming_buffer.get_available_frames())
        buffer_utilization_snapshot = (
            buffer_size_snapshot / self.streaming_buffer.max_size
            if self.streaming_buffer.max_size
            else 0.0
        )
        telemetry = FrameTelemetry(
            frame_idx=step,
            source="unknown",
            requested_at=request_start,
            buffer_size=buffer_size_snapshot,
            buffer_utilization=buffer_utilization_snapshot,
            origin=origin,
        )
        previous_displayed = self._last_displayed_frame
        direction_hint = 1
        if previous_displayed >= 0 and step < previous_displayed:
            direction_hint = -1

        is_significant_jump = False
        if affects_playback:
            _should_clear, is_significant_jump = self.playback_state.begin_seek(
                step, clear_buffer_callback=self.clear_buffer
            )
            self.streaming_buffer.set_display_position(step)

            # Cancel playback requests made stale by the new display position.
            self._cancel_distant_pending_requests(step)

        cached = self.streaming_buffer.get_frame(step)
        if cached is not None:
            if not affects_playback:
                return cached
            result = self._serve_cached_frame(step, cached, telemetry, is_significant_jump)
            if result is not None:
                self._maybe_prefetch(step, direction_hint)
            return result

        # Frame not in buffer - request from server
        telemetry.source = "server"

        # Note: With incremental prefetch, frames arrive quickly as they're computed
        # The periodic buffer check in the wait loop below will catch prefetched frames immediately

        logger.debug(f"[LOAD_FRAME] Frame {step} not in buffer, requesting from server")

        if not self._transport_available():
            logger.warning("Cannot load frame %s: not connected", step)
            telemetry.source = "error"  # Mark as error instead of leaving as "unknown"
            telemetry.error_message = self._disconnect_reason or "Not connected"
            telemetry.latency_ms = (time.time() - telemetry.requested_at) * 1000.0
            self._record_telemetry(telemetry)
            return None

        self._start_stream_loop()

        waiter: "queue.Queue[Optional[StandardMPCFrame]]" = queue.Queue(maxsize=1)
        with self._pending_lock:
            request_already_pending = step in self._pending_frames
            if request_already_pending:
                self._pending_frames[step].append(waiter)
            else:
                self._pending_frames[step] = [waiter]
            if origin in _NON_PLAYBACK_FRAME_ORIGINS:
                self._non_playback_pending_frames.add(step)
            elif not request_already_pending:
                self._non_playback_pending_frames.discard(step)

        if not request_already_pending:
            with self._request_history_lock:
                self._pending_requests[step] = request_start
                self._pending_request_origins[step] = origin
            with self._ack_lock:
                self._acknowledged_frames.discard(step)
            self._enqueue_get_frame(step)

        frame = None
        waiter_released = False
        start_wait = time.time()
        check_interval = 0.05
        max_wait_time = self.REQUEST_TIMEOUT

        while (time.time() - start_wait) < max_wait_time:
            # First check if frame arrived in buffer (from prefetch or previous request)
            cached = self.streaming_buffer.get_frame(step)
            if cached is not None:
                logger.debug(
                    f"[LOAD_FRAME] Frame {step} found in buffer while waiting, using cached version"
                )
                frame = cached
                # Deliver to waiter if still waiting
                try:
                    waiter.put_nowait(frame)
                except queue.Full:
                    pass
                break

            # Try to get frame from waiter (non-blocking check)
            try:
                frame = waiter.get_nowait()
                if frame is None:
                    waiter_released = True
                break
            except queue.Empty:
                # Frame not ready yet - wait a bit and check again
                time.sleep(check_interval)

        # If still no frame, do final blocking wait with remaining time
        if frame is None and not waiter_released:
            remaining_time = max_wait_time - (time.time() - start_wait)
            if remaining_time > 0:
                try:
                    frame = waiter.get(timeout=remaining_time)
                except queue.Empty:
                    logger.error("[LOAD_FRAME] Timed out waiting for frame %s", step)
                    frame = None
                    telemetry.source = "timeout"
                    telemetry.error_message = "Timed out waiting for server"

        if waiter_released:
            telemetry.source = "error"
            telemetry.error_message = self._disconnect_reason or "Live gRPC stream disconnected"

        telemetry.received_at = time.time()
        if telemetry.received_at and telemetry.requested_at:
            telemetry.latency_ms = (telemetry.received_at - telemetry.requested_at) * 1000.0

        # Clean up pending requests
        with self._pending_lock:
            waiters = self._pending_frames.get(step)
            if waiters is not None and waiter in waiters:
                waiters.remove(waiter)
                if not waiters:
                    self._pending_frames.pop(step, None)
                    self._non_playback_pending_frames.discard(step)
            request_still_pending = step in self._pending_frames
        with self._request_history_lock:
            if not request_still_pending:
                self._pending_requests.pop(step, None)
                self._pending_request_origins.pop(step, None)

        orderly_close = bool(
            waiter_released
            and self._shutdown.is_set()
            and self._disconnect_reason == "Provider closed"
        )

        if frame is not None:
            logger.debug(f"[LOAD_FRAME] Received frame {step} from server")
            if affects_playback:
                if self.streaming_buffer.get_frame(step) is None:
                    self.streaming_buffer.add_frame(step, frame)

                self._last_displayed_frame = step
                self._displayed_frames.add(step)
                self.playback_state.ack_displayed(step)

                # Send flow control for significant jumps
                if is_significant_jump:
                    logger.debug("[LOAD_FRAME] Significant jump, sending flow control")
                    self._enqueue_flow_control(0, step)
                else:
                    self._acknowledge_frame(step)
                # The stream handler records network requests; this path records
                # separate telemetry only for buffer hits.
                self._maybe_prefetch(step, direction_hint)
        elif orderly_close:
            logger.debug("[LOAD_FRAME] Cancelled frame %s because the provider closed", step)
        else:
            logger.error(f"[LOAD_FRAME] Failed to load frame {step}")
            # Ensure source is set even on failure (should already be set, but be explicit)
            if telemetry.source == "unknown":
                telemetry.source = "error"
            # Record failed requests from load_frame (stream handler won't record failures the same way)
            self._record_telemetry(telemetry)

        return frame

    def _serve_cached_frame(
        self,
        step: int,
        frame: StandardMPCFrame,
        telemetry: FrameTelemetry,
        is_significant_jump: bool,
    ) -> Optional[StandardMPCFrame]:
        """Serve frame from local buffer while keeping request history accurate."""
        with self._pending_lock:
            is_pending = step in self._pending_frames

        logger.debug(
            "[BUFFER] Serving frame %s from cache (pending=%s, buffer_size=%s)",
            step,
            is_pending,
            len(self.streaming_buffer.get_available_frames()),
        )

        telemetry.source = "buffer"
        telemetry.received_at = time.time()
        telemetry.latency_ms = (telemetry.received_at - telemetry.requested_at) * 1000.0

        self._last_displayed_frame = step
        self._displayed_frames.add(step)
        self.playback_state.ack_displayed(step)

        if is_significant_jump:
            self._enqueue_flow_control(0, step)
        else:
            self._acknowledge_frame(step)

        self._record_telemetry(telemetry)

        return frame

    def _maybe_prefetch(self, center_step: int, direction: int) -> None:
        """Attempt to keep the buffer filled ahead of the current playback direction.

        This is CLIENT-SIDE prefetch: actively requests specific frames ahead for smooth playback.
        The lookahead value comes from the UI prefetch_lookahead control.
        """
        if not self._prefetch_enabled or self.streaming_buffer.max_size <= 0:
            return
        direction = -1 if direction < 0 else 1
        lookahead = getattr(self.streaming_buffer, "lookahead", None)
        if lookahead is None or lookahead <= 0:
            lookahead = self.streaming_buffer.max_size  # Default to buffer size

        total_frames = None
        frame_info = self._get_frame_info()
        if frame_info:
            total_frames = frame_info.get("total_frames")
            if total_frames and total_frames > 0:
                logger.debug(f"[PREFETCH] Total frames: {total_frames}, center_step: {center_step}")

        max_requests = min(lookahead, self.streaming_buffer.max_size)

        available_frames = set(self.streaming_buffer.get_available_frames())
        with self._pending_lock:
            pending_frames = set(self._pending_frames.keys())

        offsets = range(1, lookahead + 1)
        scheduled = 0
        for offset in offsets:
            if scheduled >= max_requests:
                break
            if direction < 0:
                target = center_step - offset
            else:
                target = center_step + offset
            if target < 0:
                continue
            if total_frames is not None and target >= total_frames:
                logger.debug(
                    f"[PREFETCH] Skipping frame {target} (beyond total_frames={total_frames})"
                )
                continue
            if target in available_frames or target in pending_frames:
                continue
            if not self.streaming_buffer.is_in_prefetch_window(target):
                continue

            # Skip prefetches that the distance-based policy would immediately
            # evict after the buffered and pending frames are considered.
            current_display = self.streaming_buffer.current_display
            if current_display >= 0:
                target_distance = abs(target - current_display)
                buffer_size = len(available_frames)

                # If buffer is full or nearly full, check if target would be evicted
                if buffer_size >= self.streaming_buffer.max_size - 1:
                    # Calculate distances of all frames currently in buffer (excluding current_display which is protected)
                    buffer_distances = []
                    for f in available_frames:
                        if f != current_display:  # current_display is protected, don't count it
                            buffer_distances.append(abs(f - current_display))
                    max_buffer_distance = max(buffer_distances) if buffer_distances else 0

                    # Also consider pending frames that will arrive before target
                    # (they'll be in buffer when target arrives)
                    pending_distances = []
                    for f in pending_frames:
                        if f != target and f != current_display:
                            pending_distances.append(abs(f - current_display))
                    max_pending_distance = max(pending_distances) if pending_distances else 0

                    # The target frame would be evicted immediately if:
                    # 1. Buffer is completely full (max_size), AND
                    # 2. Target is farther from current_display than ALL frames in buffer AND pending
                    #
                    # If target is closer than some frames, it will replace the farthest ones.
                    # If target is farther than everything, it will be evicted immediately.
                    if buffer_size >= self.streaming_buffer.max_size:
                        # Buffer is completely full
                        if (
                            target_distance > max_buffer_distance
                            and target_distance > max_pending_distance
                        ):
                            # Target is farther than everything in buffer and pending
                            # It will be evicted immediately when it arrives
                            logger.debug(
                                "[PREFETCH] Skipping frame %s (distance=%d > max_buffer_distance=%d, would be evicted at current_display=%s)",
                                target,
                                target_distance,
                                max(max_buffer_distance, max_pending_distance),
                                current_display,
                            )
                            continue
                        elif target_distance == max_buffer_distance and max_buffer_distance > 0:
                            # Target is at the same distance as the farthest frame
                            # With FIFO eviction for ties, the oldest frame gets evicted
                            # Since target is new, it might stay, but it's risky - skip to be safe
                            logger.debug(
                                "[PREFETCH] Skipping frame %s (distance=%d == max_buffer_distance=%d, buffer full, risky at current_display=%s)",
                                target,
                                target_distance,
                                max_buffer_distance,
                                current_display,
                            )
                            continue

            origin = "prefetch_backward" if direction < 0 else "prefetch_forward"
            if self.prefetch_frame(target, origin=origin):
                scheduled += 1

        if scheduled:
            logger.debug(
                "[PREFETCH] Requested %s frame(s) around %s (direction=%s, lookahead=%s, max_requests=%s)",
                scheduled,
                center_step,
                "backward" if direction < 0 else "forward",
                lookahead,
                max_requests,
            )

    def _cancel_distant_pending_requests(self, current_display: int) -> None:
        """Cancel pending frame requests that are too far from the current display position."""
        if current_display < 0:
            return

        # Requests farther than one buffer span cannot survive distance eviction.
        max_safe_distance = self.streaming_buffer.max_size

        cancelled = []
        with self._pending_lock:
            # Cancellation mutates the pending map, so iterate over a snapshot.
            pending_keys = list(self._pending_frames.keys())
            for frame_idx in pending_keys:
                if frame_idx in self._non_playback_pending_frames:
                    continue
                distance = abs(frame_idx - current_display)
                # Cancel if frame is too far and outside prefetch window
                if distance > max_safe_distance or not self.streaming_buffer.is_in_prefetch_window(
                    frame_idx
                ):
                    if self._cancel_pending_frame_locked(frame_idx):
                        cancelled.append(frame_idx)

        if cancelled:
            logger.debug(
                "[PREFETCH] Cancelled %d distant pending request(s): %s (current_display=%s)",
                len(cancelled),
                cancelled[:5],
                current_display,
            )

    def set_prefetch_enabled(self, enabled: bool) -> None:
        """Enable or disable asynchronous prefetch requests."""
        self._prefetch_enabled = bool(enabled)
        logger.info("[PREFETCH] Enabled=%s", self._prefetch_enabled)

    def set_play_direction(self, direction: int) -> None:
        """Record forward/backward playback direction for prefetch policy."""
        self._play_direction = -1 if direction < 0 else 1
        logger.debug("[PREFETCH] Play direction set to %s", self._play_direction)

    def _current_buffer_metrics(self) -> tuple[int, float]:
        """Return current buffer size and utilization snapshot."""
        available = len(self.streaming_buffer.get_available_frames())
        utilization = (
            available / self.streaming_buffer.max_size if self.streaming_buffer.max_size else 0.0
        )
        return available, utilization

    def _record_telemetry(self, telemetry: FrameTelemetry) -> None:
        """Record frame telemetry for debugging"""
        if telemetry.source == "buffer":
            return
        is_success = telemetry.received_at is not None and telemetry.source not in (
            "error",
            "timeout",
        )
        entry = {
            "frame_idx": telemetry.frame_idx,
            "request_type": "GET_FRAME",
            "request_timestamp": telemetry.requested_at,
            "success": is_success,
            "generation_time_ms": telemetry.latency_ms,
            "source": telemetry.source,
            "buffer_size": telemetry.buffer_size,
            "buffer_utilization": telemetry.buffer_utilization,
            "origin": telemetry.origin,
            "error_message": telemetry.error_message,
        }
        with self._request_history_lock:
            self._request_history.insert(0, entry)
            if len(self._request_history) > self._request_history_max:
                self._request_history = self._request_history[: self._request_history_max]

    def has_frame(self, step: int) -> bool:
        """Return whether a frame is buffered or advertised by server metadata."""
        if self.streaming_buffer.get_frame(step) is not None:
            return True
        info = self._get_frame_info()
        if not info:
            return False
        total_frames = int(info.get("total_frames") or 0)
        if total_frames > 0:
            return 0 <= step < total_frames
        return step in info.get("available_frames", [])

    def list_frames(self) -> List[int]:
        """Return sorted frame indices from server metadata and local knowledge."""
        info = self._get_frame_info()
        if not info:
            return []
        frames = list(info.get("available_frames", []))
        total_frames = int(info.get("total_frames") or 0)
        if total_frames > 0:
            known = set(frames)
            known.update(range(total_frames))
            frames = list(known)
        frames.sort()
        return frames

    def load_frame_with_overrides(
        self, step: int, overrides: List[Dict[str, Any]]
    ) -> Optional[StandardMPCFrame]:
        """Apply live node overrides, clear stale frames, and load the target frame."""
        self._enqueue_overrides(overrides)
        self.clear_buffer()
        return self.load_frame(step)

    def request_frame(self, frame_idx: int, *, origin: str = "manual") -> bool:
        """Synchronously request one frame and return whether it was received."""
        frame = self.load_frame(frame_idx, origin=origin)
        return frame is not None

    def prefetch_frame(self, frame_idx: int, *, origin: str = "prefetch") -> bool:
        """Request a frame asynchronously without blocking the UI thread."""
        if not self._prefetch_enabled:
            logger.debug("[PREFETCH] Prefetch disabled; skipping frame %s", frame_idx)
            return False
        if frame_idx < 0:
            logger.debug("[PREFETCH] Ignoring negative frame index %s", frame_idx)
            return False
        if self.streaming_buffer.get_frame(frame_idx) is not None:
            logger.debug("[PREFETCH] Frame %s already in buffer, skipping", frame_idx)
            return False
        if not self._transport_available():
            logger.debug("[PREFETCH] Cannot prefetch frame %s: not connected", frame_idx)
            return False

        self._start_stream_loop()

        with self._pending_lock:
            if frame_idx in self._pending_frames:
                logger.debug(
                    "[PREFETCH] Frame %s already pending; skipping duplicate prefetch", frame_idx
                )
                return False
            # None distinguishes fire-and-forget prefetch from a blocking waiter.
            self._pending_frames[frame_idx] = [None]
            self._non_playback_pending_frames.discard(frame_idx)

        with self._ack_lock:
            self._acknowledged_frames.discard(frame_idx)

        request_timestamp = time.time()
        with self._request_history_lock:
            self._pending_requests[frame_idx] = request_timestamp
            self._pending_request_origins[frame_idx] = origin

        self._enqueue_get_frame(frame_idx)
        logger.debug("[PREFETCH] Enqueued frame %s (origin=%s)", frame_idx, origin)
        return True

    def preload_frames(self, frame_indices: List[int]) -> int:
        """Synchronously request a list of frames and count successful loads."""
        loaded = 0
        for frame_idx in frame_indices:
            if self.request_frame(frame_idx, origin="preload"):
                loaded += 1
        return loaded

    def subscribe(self, on_frame: Callable[[int, StandardMPCFrame], None]) -> None:
        """Compatibility alias for ``subscribe_to_frames``."""
        self.subscribe_to_frames(on_frame)

    def subscribe_to_frames(self, callback: Callable[[int, StandardMPCFrame], None]) -> None:
        """Register a callback invoked when a stream frame is decoded."""
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def unsubscribe_from_frames(self, callback: Callable[[int, StandardMPCFrame], None]) -> None:
        """Remove a frame callback if it is currently registered."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def get_connection_status(self) -> dict:
        """Return a panel-facing snapshot of live connection and request state."""
        with self._request_history_lock:
            connection_uptime = (
                (time.time() - self._connection_start_time) if self._connection_start_time else 0.0
            )
            avg_latency = self._calculate_average_latency()
            pending_request_count = len(self._pending_requests)
        with self._ack_lock:
            pending_ack_count = self._unacknowledged_frames_count

        available_frames = self.streaming_buffer.get_available_frames()
        buffer_size = len(available_frames)
        free_capacity = (
            max(0, self.streaming_buffer.max_size - buffer_size)
            if self.streaming_buffer.max_size
            else 0
        )

        status = {
            "connected": self.connection_manager.is_connected,
            "endpoint": self.endpoint,
            "streaming_active": self._stream_thread is not None and self._stream_thread.is_alive(),
            "buffer_size": buffer_size,
            "last_error": self._last_error,
            "connection_uptime": connection_uptime,
            "last_error_timestamp": self._last_error_timestamp,
            "average_latency_ms": avg_latency,
            "request_queue_depth": self._request_queue.qsize(),
            "pending_ack_count": pending_ack_count,
            "pending_request_count": pending_request_count,
            "prefetch_window": self.streaming_buffer.max_size,
            "buffer_free_capacity": free_capacity,
            "discarded_frames": len(self._discarded_frames),
            "generation_epoch": self._generation_epoch,
            "stale_frames_dropped": self._stale_frames_dropped,
            "last_flow_control_at": self._last_flow_control_time,
        }
        frame_info = self._get_frame_info()
        if frame_info:
            status.update(
                {
                    "total_frames": frame_info.get("total_frames", 0),
                    "available_frames": len(frame_info.get("available_frames", [])),
                    "duration": frame_info.get("duration", 0.0),
                    "frame_rate": frame_info.get("frame_rate", 0.0),
                }
            )
        return status

    def get_buffer_status(self) -> dict:
        """Return a panel-facing snapshot of buffered live frames."""
        available_frames = self.streaming_buffer.get_available_frames()
        return {
            "buffer_size": len(available_frames),
            "max_buffer_size": self.streaming_buffer.max_size,
            "available_frames": available_frames,
            "latest_frame": self.streaming_buffer.latest_frame if available_frames else None,
            "buffer_utilization": (
                len(available_frames) / self.streaming_buffer.max_size
                if self.streaming_buffer.max_size
                else 0.0
            ),
        }

    def set_buffer_size(self, size: int) -> None:
        """Dynamically update the streaming buffer size."""
        logger.info(f"Setting streaming buffer size to {size}")
        self.streaming_buffer.set_max_size(size)
        # Advertise new capacity immediately
        self._enqueue_flow_control(0, self._last_displayed_frame)

    def set_prefetch_lookahead(self, value: int) -> None:
        """Set client/server compute-ahead distance and advertise it immediately."""
        lookahead = max(1, int(value))
        self.streaming_buffer.lookahead = lookahead
        self._enqueue_flow_control(
            0,
            self._last_displayed_frame,
            server_prefetch_limit=lookahead,
        )

    def pause_streaming(self) -> None:
        """Signal the server to stop sending frames by setting buffer capacity to 0."""
        logger.info("Pausing stream by setting buffer capacity to 0.")
        self._enqueue_flow_control(0, self._last_displayed_frame, buffer_capacity=0)

    def resume_streaming(self) -> None:
        """Signal the server to resume sending frames by advertising buffer capacity."""
        logger.info("Resuming stream by advertising buffer capacity.")
        self._enqueue_flow_control(0, self._last_displayed_frame)

    def add_object_update_listener(
        self, callback: Callable[[visualizer_pb2.ObjectUpdateResponse], None]
    ) -> None:
        """Register a callback for object/node update acknowledgements."""
        if callback not in self._object_update_listeners:
            self._object_update_listeners.append(callback)

    def remove_object_update_listener(
        self, callback: Callable[[visualizer_pb2.ObjectUpdateResponse], None]
    ) -> None:
        """Remove an object update callback if registered."""
        if callback in self._object_update_listeners:
            self._object_update_listeners.remove(callback)

    def _clear_local_buffer_state(self, *, preserve_position: bool) -> int:
        """Discard buffered frames and requests without notifying the server."""
        current_frame = self._last_displayed_frame  # Preserve current position
        buffer_before_clear = sorted(self.streaming_buffer.get_available_frames())
        frame_preview = buffer_before_clear[:10]
        logger.debug(
            "[CLEAR_BUFFER] Clearing buffer at frame %s (holding %d frames, sample=%s%s)",
            current_frame,
            len(buffer_before_clear),
            frame_preview,
            "..." if len(buffer_before_clear) > len(frame_preview) else "",
        )

        self.streaming_buffer.clear(notify_eviction=False)

        # Reset acknowledgment state without consuming the cleared frames.
        with self._ack_lock:
            pending_acks = self._unacknowledged_frames_count
            self._acknowledged_frames.clear()
            self._unacknowledged_frames_count = 0
            logger.debug("[CLEAR_BUFFER] Cleared %d pending acknowledgments", pending_acks)

        # Preserve a valid display anchor only when the caller requests it.
        if preserve_position and current_frame >= 0:
            current_display = current_frame
            # Cancel pending requests that are too far from the preserved position
            self._cancel_distant_pending_requests(current_display)
        else:
            current_display = -1
            self._last_displayed_frame = -1
            self.streaming_buffer.set_display_position(-1)
            cancelled = self.cancel_pending_frame_requests()
            if cancelled:
                logger.debug(
                    "[CLEAR_BUFFER] Cancelled %d pending request(s) due to buffer clear",
                    len(cancelled),
                )
        if not preserve_position or current_display < 0:
            self._last_displayed_frame = -1
        return current_display

    def clear_buffer(self, *, pause_stream: bool = False, preserve_position: bool = True) -> None:
        """Clear buffered frames and advertise the resulting capacity.

        Flow control advertises the retained display position with
        ``frames_consumed=0``. Pending acknowledgments are discarded because
        clearing buffered frames does not consume them. ``pause_stream`` also
        advertises zero capacity; ``preserve_position=False`` resets the display
        position and cancels all pending frame requests.
        """
        current_display = self._clear_local_buffer_state(preserve_position=preserve_position)
        if pause_stream:
            logger.debug(
                "[CLEAR_BUFFER] Pausing stream: frames_consumed=0, current_display=%s, buffer_capacity=0",
                current_display,
            )
            self._enqueue_flow_control(0, current_display, buffer_capacity=0)
        else:
            logger.debug(
                "[CLEAR_BUFFER] Sending flow control: frames_consumed=0, current_display=%s (free capacity will be calculated)",
                current_display,
            )
            # Don't pass buffer_capacity - let _enqueue_flow_control calculate free capacity
            # After clear, buffer is empty so free_capacity = max_size
            self._enqueue_flow_control(0, current_display)

    def _invalidate_local_frame_state(self) -> None:
        """Discard frames and request state invalidated by a server cache epoch change."""
        self._clear_local_buffer_state(preserve_position=False)
        self._displayed_frames.clear()
        self._discarded_frames.clear()

    def request_cache_flush(
        self,
        *,
        flush_client: bool = True,
        flush_server: bool = True,
        reason: str = "Manual cache flush",
    ) -> None:
        """
        Ask the server to flush its frame cache and optionally clear the local buffer.
        """
        logger.info(
            "[CACHE_FLUSH] Requesting cache flush (client=%s, server=%s, reason=%s)",
            flush_client,
            flush_server,
            reason,
        )
        if flush_client:
            self._invalidate_local_frame_state()
        if flush_client or flush_server:
            self._enqueue_flow_control(0, -1, buffer_capacity=0, server_prefetch_limit=0)

        request = visualizer_pb2.FrameRequest(
            cache_flush=visualizer_pb2.CacheFlushRequest(
                flush_client=flush_client,
                flush_server=flush_server,
                reason=reason,
            )
        )
        self._request_queue.put(request)

    def update_object_via_xml(
        self,
        xml_root,
        *,
        scene_name: str = "scene_modified.xml",
        reason: str = "Edit Properties",
        flush_cache: bool = True,
    ) -> bool:
        """
        Serialize the current XML scene and send it to the generator via ObjectUpdate.
        """
        if xml_root is None:
            logger.error("Cannot send object update: xml_root is None")
            return False

        try:
            xml_payload = XMLSceneHandler.serialize_xml_scene(xml_root)
        except (ValueError, RuntimeError) as exc:
            logger.error(f"Failed to serialize XML scene for update: {exc}")
            return False

        if not xml_payload.strip():
            logger.error("Serialized XML scene is empty; aborting object update")
            return False

        if not self._transport_available():
            logger.error("Cannot send object update: not connected to generator")
            return False

        self._start_stream_loop()

        request = visualizer_pb2.FrameRequest(
            object_update=visualizer_pb2.ObjectUpdate(
                object_name=scene_name or "scene_modified",
                object_type=visualizer_pb2.NODE_TYPE_UNSPECIFIED,
                flush_cache=flush_cache,
                xml_payload=xml_payload,
            )
        )

        waiter: "queue.Queue[visualizer_pb2.ObjectUpdateResponse]" = queue.Queue(maxsize=1)
        with self._object_update_waiter_lock:
            if self._object_update_waiter is not None:
                logger.warning(
                    "Another object update is still in flight; rejecting the XML scene update"
                )
                return False
            self._object_update_waiter = waiter

        logger.info(
            "[OBJECT_UPDATE] Sending XML payload (%d bytes) reason='%s'", len(xml_payload), reason
        )
        self._request_queue.put(request)

        try:
            response = waiter.get(timeout=self.OBJECT_UPDATE_TIMEOUT)
        except queue.Empty:
            logger.error(
                "[OBJECT_UPDATE] No acknowledgment after %.0f seconds; the scene rebuild "
                "may still be running and its result is unknown. Further object updates "
                "remain blocked until the server responds or disconnects.",
                self.OBJECT_UPDATE_TIMEOUT,
            )
            return False

        if response.success:
            logger.info(
                "[OBJECT_UPDATE] Server applied update (cache_flushed=%s, recomputation=%s)",
                response.cache_flushed,
                response.recomputation_triggered,
            )
            return True

        logger.error("[OBJECT_UPDATE] Server reported failure: %s", response.message)
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
        """
        Send a targeted node property update (TX/RX/Target) via ObjectUpdate.
        """
        if not node_name:
            logger.error("Cannot send node update: node_name is empty")
            return False

        if isinstance(node_type, str):
            node_type_key = node_type.lower()
        else:
            node_type_key = str(node_type).lower()

        type_mapping = {
            "tx": visualizer_pb2.NODE_TYPE_TX,
            "rx": visualizer_pb2.NODE_TYPE_RX,
            "target": visualizer_pb2.NODE_TYPE_TARGET,
        }
        node_enum = type_mapping.get(node_type_key)
        if node_enum is None:
            logger.error("Unsupported node type for update: %s", node_type)
            return False

        normalized_position = None
        if position is not None:
            normalized_position = _validated_vector3(position, label="Position")
            if normalized_position is None:
                return False

        normalized_orientation = None
        if orientation is not None:
            normalized_orientation = _validated_vector3(orientation, label="Orientation")
            if normalized_orientation is None:
                return False

        normalized_scale = None
        if scale is not None:
            if node_type_key != "target":
                logger.error("Scale updates are supported only for targets")
                return False
            try:
                normalized_scale = float(scale)
            except (TypeError, ValueError, OverflowError) as exc:
                logger.error("Scale must be numeric: %s", exc)
                return False
            if not math.isfinite(normalized_scale) or normalized_scale <= 0.0:
                logger.error("Scale must be finite and positive")
                return False

        if not self._transport_available():
            logger.error("Cannot send node update: not connected to generator")
            return False

        self._start_stream_loop()

        update_msg = visualizer_pb2.ObjectUpdate(
            object_name=node_name,
            object_type=node_enum,
            flush_cache=flush_cache,
        )

        if normalized_position is not None:
            update_msg.has_position = True
            update_msg.x, update_msg.y, update_msg.z = normalized_position

        if normalized_orientation is not None:
            update_msg.has_orientation = True
            update_msg.orientation.extend(normalized_orientation)

        if normalized_scale is not None:
            update_msg.has_scale = True
            update_msg.scale = normalized_scale

        request = visualizer_pb2.FrameRequest(object_update=update_msg)
        waiter: "queue.Queue[visualizer_pb2.ObjectUpdateResponse]" = queue.Queue(maxsize=1)
        with self._object_update_waiter_lock:
            if self._object_update_waiter is not None:
                logger.warning(
                    "Another object update is already in-flight; rejecting new node update"
                )
                return False
            self._object_update_waiter = waiter

        request_details = details
        if not request_details:
            fields = []
            if normalized_position is not None:
                fields.append(f"pos={tuple(round(value, 3) for value in normalized_position)}")
            if normalized_orientation is not None:
                fields.append(
                    f"orient={tuple(round(value, 2) for value in normalized_orientation)}"
                )
            if normalized_scale is not None:
                fields.append(f"scale={normalized_scale:.3f}")
            field_str = ", ".join(fields) if fields else "no fields"
            request_details = f"{node_type_key}:{node_name} ({field_str})"

        self._record_request(
            -1,
            "NODE_UPDATE",
            time.time(),
            True,
            origin=origin,
            source="client",
            details=request_details,
        )

        self._request_queue.put(request)

        try:
            response = waiter.get(timeout=self.REQUEST_TIMEOUT)
        except queue.Empty:
            logger.error(
                "[NODE_UPDATE] Timed out waiting for server acknowledgment; "
                "further object updates remain blocked until the late response "
                "arrives or the transport disconnects"
            )
            self._record_request(
                -1,
                "NODE_UPDATE",
                time.time(),
                False,
                origin=origin,
                source="client",
                error_message="Timeout waiting for node update response",
                details=request_details,
            )
            return False

        with self._object_update_waiter_lock:
            if self._object_update_waiter is waiter:
                self._object_update_waiter = None

        if response.success:
            logger.info(
                "[NODE_UPDATE] Server applied update for %s (cache_flushed=%s, recomputation=%s)",
                node_name,
                response.cache_flushed,
                response.recomputation_triggered,
            )
            return True

        logger.error("[NODE_UPDATE] Server reported failure: %s", response.message)
        self._record_request(
            -1,
            "NODE_UPDATE",
            time.time(),
            False,
            origin=origin,
            source="client",
            error_message=response.message or "Node update rejected",
            details=request_details,
        )
        return False

    def get_frame_timestamps(self) -> Dict[int, float]:
        """Return a copy of frame arrival timestamps keyed by frame index."""
        return self.streaming_buffer.frame_timestamps.copy()

    def _start_ack_timer(self) -> None:
        """Start the acknowledgment timer, which remains active after EOF."""

        if self._ack_timer is not None:
            return
        self._ack_timer = threading.Timer(self.ACK_BATCH_INTERVAL_SECONDS, self._handle_ack_timer)
        self._ack_timer.daemon = True
        self._ack_timer.start()

    def _stop_ack_timer(self) -> None:
        """Stops the periodic timer and sends any final acknowledgements."""
        if self._ack_timer:
            self._ack_timer.cancel()
            self._ack_timer = None
        # Send any remaining acks on close
        self._send_batched_acknowledgements()

    def _handle_ack_timer(self) -> None:
        """
        Called by the timer to send batched acks and reschedule itself.
        Timer continues even after EOF (only stops on explicit close).
        """
        self._ack_timer = None

        # Only send acks if stream is still active (not closed)
        if not self._shutdown.is_set():
            self._send_batched_acknowledgements()
            # Reschedule the timer
            self._start_ack_timer()

    def _send_batched_acknowledgements(self) -> None:
        """Sends a flow control signal for all unacknowledged frames."""
        count_to_ack = 0
        with self._ack_lock:
            if self._unacknowledged_frames_count > 0:
                count_to_ack = self._unacknowledged_frames_count
                self._unacknowledged_frames_count = 0

        if count_to_ack > 0:
            logger.debug(f"Acknowledging {count_to_ack} batched frames.")
            self._enqueue_flow_control(count_to_ack, self._last_displayed_frame)

    def _acknowledge_frame(self, frame_idx: int) -> None:
        """
        Marks a frame as acknowledged and increments the batch counter.
        Only acknowledges frames that were actually displayed.
        """
        # Only acknowledge if frame was displayed
        if frame_idx not in self._displayed_frames:
            logger.debug(f"[ACK] Skipping acknowledgment for frame {frame_idx} (not displayed)")
            return

        with self._ack_lock:
            if frame_idx in self._acknowledged_frames:
                return
            self._acknowledged_frames.add(frame_idx)
            self._unacknowledged_frames_count += 1

    def _handle_evicted_frames(self, frame_indices: List[int]) -> None:
        """
        Handle evicted frames. Only acknowledge frames that were displayed.
        Frames evicted before display are marked as discarded and not acknowledged.
        """
        if not frame_indices:
            return

        to_credit: List[int] = []
        to_discard: List[int] = []

        for idx in frame_indices:
            if idx in self._displayed_frames:
                # Frame was displayed - acknowledge it
                with self._ack_lock:
                    if idx not in self._acknowledged_frames:
                        self._acknowledged_frames.add(idx)
                        to_credit.append(idx)
            else:
                # Frame was evicted before display - mark as discarded
                self._discarded_frames.add(idx)
                to_discard.append(idx)
                logger.debug(
                    f"[EVICT] Frame {idx} evicted before display (discarded, not acknowledged)"
                )

        # Only send flow control for frames that were displayed
        if to_credit:
            logger.debug(
                f"[EVICT] Acknowledging {len(to_credit)} displayed frames that were evicted"
            )
            self._enqueue_flow_control(len(to_credit), self._last_displayed_frame)

        if to_discard:
            logger.debug(f"[EVICT] Discarded {len(to_discard)} frames that were never displayed")

    # Request history and telemetry
    def _record_request(
        self,
        frame_idx: int,
        request_type: str,
        request_timestamp: float,
        success: bool,
        generation_time: Optional[float] = None,
        error_message: Optional[str] = None,
        source: str = "server",
        origin: str = "user",
        buffer_size: Optional[int] = None,
        buffer_utilization: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Record a request in the history with timing information."""
        entry = {
            "frame_idx": frame_idx,
            "request_type": request_type,
            "request_timestamp": request_timestamp,
            "success": success,
            "generation_time_ms": generation_time,
            "error_message": error_message,
            "source": source,
            "origin": origin,
            "buffer_size": buffer_size,
            "buffer_utilization": buffer_utilization,
            "details": details,
        }

        with self._request_history_lock:
            self._request_history.insert(0, entry)  # Insert at beginning (most recent first)

            if len(self._request_history) > self._request_history_max:
                self._request_history = self._request_history[: self._request_history_max]

            if generation_time is not None and success:
                self._generation_times.append(generation_time)
                if len(self._generation_times) > self._generation_times_max:
                    self._generation_times = self._generation_times[-self._generation_times_max :]

    def get_request_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent request history."""
        with self._request_history_lock:
            return self._request_history[:limit] if limit > 0 else self._request_history.copy()

    def _calculate_average_latency(self) -> float:
        """Calculate average latency from generation times."""
        if not self._generation_times:
            return 0.0
        return sum(self._generation_times) / len(self._generation_times)

    def get_latency_statistics(self) -> Dict[str, float]:
        """Get latency statistics (average, min, max)."""
        with self._request_history_lock:
            if not self._generation_times:
                return {
                    "average_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "count": 0,
                }

            return {
                "average_ms": sum(self._generation_times) / len(self._generation_times),
                "min_ms": min(self._generation_times),
                "max_ms": max(self._generation_times),
                "count": len(self._generation_times),
            }

    def update_raytracing_params(
        self,
        preset: str,
        config: Dict[str, Any],
        flush_cache: bool = True,
    ) -> Optional[visualizer_pb2.ParameterUpdateResponse]:
        """Update raytracing parameters via gRPC ParameterUpdate request."""
        try:
            if not self._transport_available():
                logger.error("Cannot update parameters: not connected")
                return None

            # Dynamic protobuf modules may expose descriptors before generated classes.
            descriptor = visualizer_pb2.DESCRIPTOR
            has_rt_config = "RaytracingConfig" in descriptor.message_types_by_name
            has_param_update = "ParameterUpdate" in descriptor.message_types_by_name

            if not has_rt_config or not has_param_update:
                missing = []
                if not has_rt_config:
                    missing.append("RaytracingConfig")
                if not has_param_update:
                    missing.append("ParameterUpdate")
                raise AttributeError(
                    f"Protobuf messages {', '.join(missing)} not found in descriptor. "
                    "Please regenerate protobuf files with: python scripts/protobuf/generate_protobuf.py"
                )

            try:
                RaytracingConfig = visualizer_pb2.RaytracingConfig
                ParameterUpdate = visualizer_pb2.ParameterUpdate
            except AttributeError:
                # Reload once when descriptors exist but generated classes are absent.
                logger.warning("Protobuf classes not found in module, attempting reload...")
                import importlib

                importlib.reload(visualizer_pb2)

                try:
                    RaytracingConfig = visualizer_pb2.RaytracingConfig
                    ParameterUpdate = visualizer_pb2.ParameterUpdate
                except AttributeError:
                    raise AttributeError(
                        "RaytracingConfig and ParameterUpdate classes not available. "
                        "The protobuf file was regenerated but Python hasn't picked up the changes. "
                        "Please RESTART the visualizer completely (close and reopen) to reload the protobuf module."
                    )

            rt_config = RaytracingConfig(
                max_depth=int(config.get("max_depth", 4)),
                samples_per_src=int(config.get("samples_per_src", 100000)),
                max_num_paths_per_src=int(config.get("max_num_paths_per_src", 100000)),
                los=bool(config.get("los", True)),
                specular_reflection=bool(config.get("specular_reflection", True)),
                diffuse_reflection=bool(config.get("diffuse_reflection", True)),
                refraction=bool(config.get("refraction", False)),
                diffraction=bool(config.get("diffraction", False)),
                synthetic_array=bool(config.get("synthetic_array", True)),
                seed=int(config.get("seed", 42)),
            )

            param_update = ParameterUpdate(
                preset=preset,
                raytracing_config=rt_config,
                flush_cache=flush_cache,
            )

            # Prepare waiter for acknowledgement before enqueuing request
            wait_queue: "queue.Queue[visualizer_pb2.ParameterUpdateResponse]" = queue.Queue(
                maxsize=1
            )
            with self._param_update_waiter_lock:
                if self._param_update_waiter is not None:
                    logger.warning("Parameter update already in progress; rejecting new request")
                    return None
                self._param_update_waiter = wait_queue

            request = visualizer_pb2.FrameRequest(parameter_update=param_update)
            self._request_queue.put(request)

            # Optionally clear local buffer immediately so UI reflects reset
            if flush_cache:
                self.clear_buffer(pause_stream=True)
                logger.info("Cleared local buffer after parameter update")

            logger.info(
                "Parameter update request sent: preset=%s, flush_cache=%s (awaiting acknowledgement)",
                preset,
                flush_cache,
            )

            try:
                response_msg = wait_queue.get(timeout=self.REQUEST_TIMEOUT)
            except queue.Empty:
                logger.error(
                    "Timed out waiting for parameter update response; further parameter "
                    "updates remain blocked until the late response arrives or the "
                    "transport disconnects"
                )
                self._record_request(
                    -1,
                    "PARAM_UPDATE_ACK",
                    time.time(),
                    False,
                    error_message="Timeout waiting for parameter update response",
                )
                return None

            with self._param_update_waiter_lock:
                if self._param_update_waiter is wait_queue:
                    self._param_update_waiter = None

            if response_msg.success:
                logger.info(
                    "Parameter update succeeded: %s",
                    response_msg.message or "applied",
                )
            else:
                logger.error(
                    "Parameter update rejected: %s",
                    response_msg.message or "no message",
                )

            return response_msg

        except AttributeError as e:
            # Protobuf messages not available - provide helpful error message
            error_msg = (
                f"Protobuf messages not available: {str(e)}. "
                "Please run: python scripts/protobuf/generate_protobuf.py"
            )
            logger.error(error_msg)
            self._record_request(-1, "PARAM_UPDATE", time.time(), False, error_message=error_msg)
            # Re-raise so UI can show the error
            raise
        except (OSError, RuntimeError) as e:
            logger.error("Error updating raytracing parameters: %s", e, exc_info=True)
            self._record_request(-1, "PARAM_UPDATE", time.time(), False, error_message=str(e))
            return None

    # Frame info helpers
    def _get_frame_info(self) -> Optional[Dict[str, Any]]:
        """Fetch and cache server frame-range metadata."""
        now = time.time()
        if hasattr(self, "frame_info_cache"):
            if self.frame_info_cache is not None and now - self.cache_timestamp < getattr(
                self, "cache_ttl", 5.0
            ):
                return self.frame_info_cache
        else:
            self.frame_info_cache = None
            self.cache_timestamp = 0.0
            self.cache_ttl = 5.0

        try:
            if not self._transport_available():
                return None

            request = visualizer_pb2.GetFrameInfoRequest()
            response = self.connection_manager.stub.GetFrameInfo(request, timeout=5.0)
            if response.success:
                self.frame_info_cache = {
                    "total_frames": response.total_frames,
                    "duration": response.duration,
                    "frame_rate": response.frame_rate,
                    "available_frames": list(response.available_frames),
                    "generation_epoch": int(getattr(response, "generation_epoch", 0) or 0),
                }
                self._generation_epoch = max(
                    self._generation_epoch,
                    self.frame_info_cache["generation_epoch"],
                )
                self.cache_timestamp = now
                return self.frame_info_cache
        except grpc.RpcError as exc:
            summary = self.connection_manager._error_summary(exc)
            message = f"GetFrameInfo failed: {summary}"
            logger.error(message)
            code_fn = getattr(exc, "code", None)
            code = code_fn() if callable(code_fn) else None
            if code == grpc.StatusCode.UNAVAILABLE:
                self._handle_stream_disconnect(message)
            else:
                self._last_error = message
                self._last_error_timestamp = time.time()
        except (OSError, RuntimeError) as exc:
            logger.error("Error getting frame info: %s", exc)
        return None
