"""Visualizer-facing sensing helpers.

The package exposes CIR providers for frame-derived sensing products and the
typed update payload consumed by the optional pop-out sensing dashboard. Core
DSP result types live in :mod:`shared.sensing.signal_processing`; the visualizer layer adapts
frame dictionaries, cached results, and UI state into those shared contracts.

The panel owns a ``dashboard_data.DashboardSession`` for background selective
reads and a bounded recording overview. ``sensing_window`` consumes its typed
requests on Qt's thread; saved RD power, not playback visits, defines the
recording's motion spectrum. ``qt_lifetime`` keeps cyclic plot cleanup on that
same thread while readers run.

``recording_cache`` stores disposable, source-fingerprinted indices and overview
data across restarts. ``result_navigation`` resolves physical times to authored
scene bindings. ``selection`` defines scoped saved identities; ``linked_selection``
and the renderer adapter present them using the shared ``display_style`` palette.
"""

from shared.sensing.signal_processing import RangeDopplerResult, RangeProfileResult

from .cir_provider import (
    CIRComputationResult,
    CIRProvider,
    StoredCIRProvider,
)
from .update_request import SensingUpdateRequest

__all__ = [
    "CIRComputationResult",
    "CIRProvider",
    "StoredCIRProvider",
    "SensingUpdateRequest",
    "RangeProfileResult",
    "RangeDopplerResult",
]
