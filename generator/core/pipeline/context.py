"""Service lifetime management for one pipeline run.

Pipeline modules request services from this context instead of constructing them
directly. The context validates the shared simulation config on entry, lazily
creates each service once, and guarantees cleanup at the end of the run.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import TypeVar, cast

from shared.logging import get_logger

from ..configuration import SimulationConfig
from ..exceptions import ServiceError
from ..services.base import BaseService

logger = get_logger(__name__)

_ServiceT = TypeVar("_ServiceT", bound=BaseService)


class PipelineContext:
    """Own service instances and cleanup for one file or streaming pipeline run."""

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config
        self.services: dict[str, BaseService] = {}
        self.start_time = 0.0

    def __enter__(self) -> PipelineContext:
        self.start_time = time.time()
        logger.info("Pipeline execution started")
        if hasattr(self.simulation_config, "validate"):
            self.simulation_config.validate()
            logger.info("Configuration validated successfully")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        duration = time.time() - self.start_time
        # Services hold scene graphs, prepared actor-state caches, and solver
        # state. Cleanup is centralized here so file and streaming backends do
        # not each need to know every service's resource details.
        for service_name, service in self.services.items():
            cleanup_fn = getattr(service, "cleanup", None)
            if not callable(cleanup_fn):
                continue
            try:
                cleanup_fn()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("Cleanup failed for %s: %s", service_name, cleanup_exc)
        if exc_type:
            logger.error("Pipeline failed after %.2fs: %s", duration, exc_val)
            return False

        logger.info("Pipeline completed successfully in %.2fs", duration)
        return False

    def get_service(self, service_cls: type[_ServiceT]) -> _ServiceT:
        """Return the run-scoped service instance for ``service_cls``."""
        service_name = service_cls.__name__
        if service_name not in self.services:
            try:
                logger.info("Creating %s...", service_name)
                service = service_cls(self.simulation_config)
                self.services[service_name] = service
            except (ValueError, TypeError, RuntimeError, OSError) as e:
                raise ServiceError(f"Failed to create {service_name}: {e}") from e

        return cast(_ServiceT, self.services[service_name])
