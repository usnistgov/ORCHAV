"""Dispatchers for generator and sensing summary figure generation.

The generator dispatcher is cheap to import. Sensing summary generation pulls in
heavier optional sensing helpers, so it is exposed through ``__getattr__`` and
loaded only when a sensing-enabled pipeline asks for it.
"""

from typing import TYPE_CHECKING

from .generator import maybe_generate_generator_summary

if TYPE_CHECKING:
    from .sensing import maybe_generate_sensing_summary

__all__ = ["maybe_generate_generator_summary", "maybe_generate_sensing_summary"]


def __getattr__(name: str):
    """Lazy-load sensing-only figure dispatch to keep normal generator imports light."""
    if name == "maybe_generate_sensing_summary":
        from .sensing import maybe_generate_sensing_summary

        return maybe_generate_sensing_summary
    raise AttributeError(name)
