"""Subsections and helper widgets for data-source panels.

`DataSourcePanel` owns mode selection and file/remote-HDF5 summaries, while
`LiveGrpcModePanel` composes these focused live-gRPC sections into the shared
panel widget registry.
"""

from .connection_section import ConnectionStatusSection
from .performance_section import PerformanceSection
from .raytracing_section import RaytracingControlSection
from .streaming_section import StreamingControlSection
from .widgets import (
    FrameComparisonDialog,
    FrameTimelineWidget,
)

__all__ = [
    "ConnectionStatusSection",
    "FrameComparisonDialog",
    "FrameTimelineWidget",
    "PerformanceSection",
    "RaytracingControlSection",
    "StreamingControlSection",
]
