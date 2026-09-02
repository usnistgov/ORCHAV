"""Shared foundation for run-scoped generator services.

Services keep the same ``SimulationConfig`` object used by the pipeline.  That
is intentional: setup services may normalize values such as coverage options or
the effective number of steps, and later services in the same run must observe
those updates.
"""

import logging
from typing import Any


class BaseService:
    """Common configuration and logger setup for concrete services."""

    def __init__(self, simulation_config: Any):
        """Store the global simulation config and create a service logger."""
        self.simulation_config = simulation_config
        self.logger = logging.getLogger(f"generator.services.{self.__class__.__name__}")
