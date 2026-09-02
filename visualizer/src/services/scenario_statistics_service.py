"""Provider-driven scenario statistics scanning.

This service keeps statistics I/O separate from playback preloading. It asks a
provider for the exact topology, interaction, and metric projection required by
the statistics accumulator, streams one projection at a time, and never retains
the raw frames. A generation token and cancellation event prevent a scan from a
previous scenario from publishing progress, results, errors, or cache entries.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from shared.frames.provider_base import DataProvider
from shared.logging import get_logger

from ..metrics.scenario_statistics import (
    SCENARIO_STATISTICS_REQUEST,
    ScenarioStatisticsAccumulator,
)
from .base import BaseService
from .statistics_cache_service import StatisticsCacheService

logger = get_logger("orchav.scenario_statistics")

StatisticsProgressCallback = Callable[[int, int], None]


class StatisticsCollectionCancelled(RuntimeError):
    """Raised internally when a caller retires an in-flight statistics scan."""


@dataclass(frozen=True, slots=True)
class ScenarioStatisticsResult:
    """One completed cache lookup or provider scan."""

    stats: dict[str, Any]
    from_cache: bool
    generation: int


class ScenarioStatisticsService(BaseService):
    """Stream scenario statistics from a selective frame provider.

    ``collect`` is the deterministic synchronous core and is useful for tests
    and batch callers. ``start_collection`` runs the same path on a daemon
    thread for the Qt application. Starting another collection atomically
    cancels the previous generation.
    """

    def __init__(self, cache_service: StatisticsCacheService) -> None:
        """Bind the persisted cache and initialize generation state."""

        super().__init__()
        self.cache_service = cache_service
        self._lock = threading.RLock()
        self._generation = 0
        self._cancel_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def current_generation(self) -> int:
        """Return the generation allowed to publish callbacks."""

        with self._lock:
            return self._generation

    def collect(
        self,
        provider: DataProvider,
        *,
        scenario: Any | None = None,
        steps: Iterable[int] | None = None,
        force: bool = False,
        on_progress: StatisticsProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        generation: int = 0,
    ) -> ScenarioStatisticsResult:
        """Return cached or freshly streamed statistics for ``provider``.

        Cache lookup deliberately precedes ``list_frames`` and
        ``iter_frame_projections``. For packed HDF5 v2 this lookup needs only
        the already-parsed manifest identity, so a hit performs no HDF5 reads.
        """

        cancellation = cancel_event or threading.Event()
        self._raise_if_cancelled(cancellation)

        if not force:
            cached = self.cache_service.load_cached_stats(
                scenario,
                provider=provider,
            )
            self._raise_if_cancelled(cancellation)
            if cached is not None:
                return ScenarioStatisticsResult(
                    stats=cached,
                    from_cache=True,
                    generation=generation,
                )

        frame_steps = (
            tuple(int(step) for step in provider.list_frames())
            if steps is None
            else tuple(int(step) for step in steps)
        )
        total_frames = len(frame_steps)
        accumulator = ScenarioStatisticsAccumulator()
        projections = iter(
            provider.iter_frame_projections(
                frame_steps,
                SCENARIO_STATISTICS_REQUEST,
            )
        )

        consumed = 0
        for current in range(1, total_frames + 1):
            self._raise_if_cancelled(cancellation)
            try:
                projection = next(projections)
            except StopIteration as exc:
                raise ValueError(
                    f"Expected {total_frames} frame projections, received {consumed}"
                ) from exc
            self._raise_if_cancelled(cancellation)
            accumulator.add_projection(projection)
            consumed = current
            if on_progress is not None:
                on_progress(current, total_frames)

        self._raise_if_cancelled(cancellation)
        try:
            next(projections)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"Expected {total_frames} frame projections, received more than {total_frames}"
            )

        stats = accumulator.finalize()
        self._raise_if_cancelled(cancellation)
        self.cache_service.save_cached_stats(
            stats,
            scenario,
            provider=provider,
        )
        self._raise_if_cancelled(cancellation)
        return ScenarioStatisticsResult(
            stats=stats,
            from_cache=False,
            generation=generation,
        )

    def start_collection(
        self,
        provider: DataProvider,
        *,
        scenario: Any | None = None,
        steps: Iterable[int] | None = None,
        force: bool = False,
        on_progress: StatisticsProgressCallback | None = None,
        on_complete: Callable[[ScenarioStatisticsResult], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> int:
        """Start one background cache lookup/scan and return its generation."""

        frame_steps = None if steps is None else tuple(int(step) for step in steps)
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._generation += 1
            generation = self._generation
            cancellation = threading.Event()
            self._cancel_event = cancellation

            def _publish_progress(current: int, total: int) -> None:
                if self._is_current(generation, cancellation) and on_progress is not None:
                    on_progress(current, total)

            def _worker() -> None:
                try:
                    result = self.collect(
                        provider,
                        scenario=scenario,
                        steps=frame_steps,
                        force=force,
                        on_progress=_publish_progress,
                        cancel_event=cancellation,
                        generation=generation,
                    )
                except StatisticsCollectionCancelled:
                    logger.debug("Statistics generation %d was cancelled", generation)
                    return
                except Exception as exc:  # provider boundary: report without killing Qt
                    if self._is_current(generation, cancellation) and on_error is not None:
                        on_error(exc)
                    return

                if self._is_current(generation, cancellation) and on_complete is not None:
                    on_complete(result)

            thread = threading.Thread(
                target=_worker,
                name=f"scenario-statistics-{generation}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return generation

    def cancel_collection(self) -> None:
        """Retire the current generation without blocking the caller."""

        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._generation += 1
            self._cancel_event = None
            self._thread = None

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait for the current worker in tests and non-Qt batch callers."""

        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def stop(self) -> None:
        """Cancel active work and mark the service stopped."""

        self.cancel_collection()
        super().stop()

    def _is_current(
        self,
        generation: int,
        cancellation: threading.Event,
    ) -> bool:
        with self._lock:
            return (
                generation == self._generation
                and cancellation is self._cancel_event
                and not cancellation.is_set()
            )

    @staticmethod
    def _raise_if_cancelled(cancellation: threading.Event) -> None:
        if cancellation.is_set():
            raise StatisticsCollectionCancelled


def resolve_statistics_provider(source: Any) -> DataProvider | None:
    """Return the shared provider behind a visualizer frame source."""

    if isinstance(source, DataProvider):
        return source
    provider = getattr(source, "provider", None)
    return provider if isinstance(provider, DataProvider) else None


__all__ = [
    "ScenarioStatisticsResult",
    "ScenarioStatisticsService",
    "StatisticsCollectionCancelled",
    "resolve_statistics_provider",
]
