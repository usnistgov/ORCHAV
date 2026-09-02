"""Lifecycle handles returned by non-blocking pipeline backends."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.grpc_transport import DEFAULT_GRPC_SHUTDOWN_GRACE_S

if TYPE_CHECKING:
    from ...io.grpc.live_server import GeneratorFrameCache, GeneratorService
    from .context import PipelineContext


@dataclass
class StreamingHandle:
    """Lifecycle handle for a running streaming generator server.

    The handle owns both the server and the ``PipelineContext`` that created the
    services handed to it. Closing or shutting down the handle releases those
    services; callers that keep the server alive must keep this handle alive too.
    """

    server: Any
    generator_service: GeneratorService | None
    frame_cache: GeneratorFrameCache
    services: dict[str, Any]
    server_thread: threading.Thread | None = None
    pipeline_context: PipelineContext | None = None
    _closed: bool = field(default=False, init=False)
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def is_alive(self) -> bool:
        """Return whether the owned server appears to still be running."""
        if self._closed:
            return False
        if self.server is not None and hasattr(self.server, "wait_for_termination"):
            try:
                # grpc.Server returns True when wait_for_termination times out.
                # With a zero timeout, that means the server is still running.
                return bool(self.server.wait_for_termination(timeout=0))
            except (TypeError, ValueError):
                return True
        if self.server_thread is not None:
            return self.server_thread.is_alive()
        return self.server is not None

    def wait_for_termination(self, timeout: float | None = None) -> None:
        """Wait for termination or timeout without releasing owned resources."""
        if self.server is not None and hasattr(self.server, "wait_for_termination"):
            self.server.wait_for_termination(timeout=timeout)
            return
        if self.server_thread is not None:
            self.server_thread.join(timeout=timeout)

    def close(self, timeout_s: float = DEFAULT_GRPC_SHUTDOWN_GRACE_S) -> None:
        """Stop the owned server and release its pipeline context exactly once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        try:
            if self.server is not None and hasattr(self.server, "stop"):
                stop_event = self.server.stop(timeout_s)
                if stop_event is not None and hasattr(stop_event, "wait"):
                    stop_event.wait(timeout=timeout_s)
        finally:
            try:
                if self.server_thread is not None and self.server_thread.is_alive():
                    self.server_thread.join(timeout=timeout_s)
            finally:
                try:
                    close_service = getattr(self.generator_service, "close", None)
                    if callable(close_service):
                        close_service()
                finally:
                    if self.pipeline_context is not None:
                        self.pipeline_context.__exit__(None, None, None)
                        self.pipeline_context = None

    def shutdown(self, timeout_s: float = DEFAULT_GRPC_SHUTDOWN_GRACE_S) -> None:
        """Delegate to the complete ``close`` lifecycle."""
        self.close(timeout_s=timeout_s)
