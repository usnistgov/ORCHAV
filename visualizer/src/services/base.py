"""Minimal service lifecycle helpers shared by visualizer services."""

from __future__ import annotations

from typing import Protocol


class ServiceBase(Protocol):
    """Protocol describing a lightweight service lifecycle."""

    _running: bool

    def start(self) -> None:
        """Start the service."""

    def stop(self) -> None:
        """Stop the service."""

    def is_running(self) -> bool:
        """Return whether the service is running."""


class BaseService:
    """Minimal base class implementing the service lifecycle."""

    def __init__(self):
        """Initialize a stopped lifecycle flag."""
        self._running = False

    def start(self) -> None:
        """Mark the service as running."""
        self._running = True

    def stop(self) -> None:
        """Mark the service as stopped."""
        self._running = False

    def is_running(self) -> bool:
        """Return the current lifecycle flag."""
        return self._running
