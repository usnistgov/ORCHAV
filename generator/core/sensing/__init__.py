"""Resolve sensing acquisition, processing, and frame timing contracts.

Start with ``resolve_sensing_execution_config`` in ``acquisition``. The
``processor.create_sensing_processor`` factory consumes that resolved contract
and delegates numerical work to ``strategies``; ``timing`` describes when each
emitted product was acquired.

Exports are loaded lazily so generic generator orchestration does not import
sensing processors or timing helpers while the extension is disabled.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ACQUISITION_EXPORTS = frozenset(
    {
        "ANALYTIC_ORACLE_MODE",
        "COHERENT_MODE",
        "ResolvedSensingExecutionConfig",
        "SEQUENTIAL_MODE",
        "SensingAcquisitionPlan",
        "SensingProcessingPlan",
        "SensingTopologyRequirement",
        "resolve_sensing_execution_config",
        "resolve_sensing_execution_values",
    }
)
_TIMING_EXPORTS = frozenset({"build_sensing_timing_metadata", "merge_sensing_timing_metadata"})

__all__ = sorted(_ACQUISITION_EXPORTS | _TIMING_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _ACQUISITION_EXPORTS:
        return getattr(import_module(".acquisition", __name__), name)
    if name in _TIMING_EXPORTS:
        return getattr(import_module(".timing", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
