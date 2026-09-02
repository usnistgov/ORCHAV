"""Exception hierarchy for ORCHAV generator failures.

The generator raises these exceptions at subsystem boundaries so callers can
distinguish invalid inputs, simulation failures, I/O failures, and service
startup problems while still catching :class:`GeneratorError` for any
generator-owned failure.
"""

__all__ = [
    "GeneratorError",
    "ConfigurationError",
    "ComputationError",
    "GeneratorIOError",
    "ServiceError",
]


class GeneratorError(Exception):
    """Base class for failures owned by the generator package."""


class ConfigurationError(GeneratorError):
    """Raised when scenario YAML or derived runtime configuration is invalid."""


class ComputationError(GeneratorError):
    """Raised when simulation, propagation, coverage, or postprocessing fails."""


class GeneratorIOError(GeneratorError):
    """Raised when generator input or output resources cannot be accessed."""


class ServiceError(GeneratorError):
    """Raised when a generator service cannot initialize or complete work."""
