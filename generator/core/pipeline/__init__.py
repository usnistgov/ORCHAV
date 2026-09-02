"""Public entry points for generator pipeline orchestration.

The pipeline package is the top-level conductor: it resolves scenario-facing
configs, chooses file or streaming execution, and wires services together. The
services own the heavy work: scene construction, actor-state preparation,
coverage computation, and ray tracing. This module keeps imports lazy so callers
can import lightweight handles without initializing Sionna/Mitsuba machinery.
"""

from typing import TYPE_CHECKING, Any

from .context import PipelineContext
from .handles import StreamingHandle
from .progress import ProgressInfo

if TYPE_CHECKING:
    from .dispatch import perform_pipeline as perform_pipeline
    from .offline_pipeline import perform_offline_pipeline as perform_offline_pipeline
    from .streaming import perform_pipeline_streaming as perform_pipeline_streaming

__all__ = [
    "PipelineContext",
    "ProgressInfo",
    "StreamingHandle",
    "perform_pipeline",
    "perform_offline_pipeline",
    "perform_pipeline_streaming",
]


def __getattr__(name: str) -> Any:
    if name == "perform_pipeline":
        from .dispatch import perform_pipeline

        return perform_pipeline
    if name == "perform_offline_pipeline":
        from .offline_pipeline import perform_offline_pipeline

        return perform_offline_pipeline
    if name == "perform_pipeline_streaming":
        from .streaming import perform_pipeline_streaming

        return perform_pipeline_streaming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
