"""Public Python API for ORCHAV scenario generation.

``perform_pipeline`` runs a validated scenario through the file or streaming
backend.  The configuration classes describe normalized generator settings and
explicit runtime actors.  Immutable scenario and actor specifications live in
``shared.scenarios``, which is the single authoring and validation authority.

Facade attributes load on first access so importing :mod:`generator` does not
initialize Sionna, Mitsuba, or target-mesh runtimes.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "ORCHAV Team"

_LazyImport = tuple[str, str]

_PUBLIC_LAZY_IMPORTS: dict[str, _LazyImport] = {
    "AntennaConfig": (".core.configuration", "AntennaConfig"),
    "CoverageConfig": (".core.configuration", "CoverageConfig"),
    "ReceiverConfig": (".core.configuration", "ReceiverConfig"),
    "SimulationConfig": (".core.configuration", "SimulationConfig"),
    "TransmitterConfig": (".core.configuration", "TransmitterConfig"),
    "build_simulation_config": (".core.configuration", "build_simulation_config"),
    "TargetConfig": (".core.target", "TargetConfig"),
    "AlignMotionOrientationSpec": (".core.orientation", "AlignMotionOrientationSpec"),
    "FixedOrientationSpec": (".core.orientation", "FixedOrientationSpec"),
    "KeyframesOrientationSpec": (".core.orientation", "KeyframesOrientationSpec"),
    "LookAtOrientationSpec": (".core.orientation", "LookAtOrientationSpec"),
    "OrientationKeyframeSpec": (".core.orientation", "OrientationKeyframeSpec"),
    "RandomOrientationSpec": (".core.orientation", "RandomOrientationSpec"),
    "SpinOrientationSpec": (".core.orientation", "SpinOrientationSpec"),
    "ProgressInfo": (".core.pipeline", "ProgressInfo"),
    "perform_pipeline": (".core.pipeline", "perform_pipeline"),
    "perform_offline_pipeline": (".core.pipeline", "perform_offline_pipeline"),
    "perform_pipeline_streaming": (".core.pipeline", "perform_pipeline_streaming"),
}

_LAZY_IMPORTS: dict[str, _LazyImport] = dict(_PUBLIC_LAZY_IMPORTS)
_LOADED: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Resolve one public facade attribute on first access."""
    if name in _LOADED:
        return _LOADED[name]

    import_target = _LAZY_IMPORTS.get(name)
    if import_target is None:
        raise AttributeError(f"module 'generator' has no attribute {name!r}")

    module_path, attribute_name = import_target
    module = importlib.import_module(module_path, package="generator")
    attribute = getattr(module, attribute_name)
    _LOADED[name] = attribute
    return attribute


def __dir__() -> list[str]:
    """Return the supported facade names for interactive discovery."""
    return sorted([*__all__, "__author__", "__version__"])


__all__ = [
    "AntennaConfig",
    "AlignMotionOrientationSpec",
    "CoverageConfig",
    "FixedOrientationSpec",
    "KeyframesOrientationSpec",
    "LookAtOrientationSpec",
    "OrientationKeyframeSpec",
    "ProgressInfo",
    "RandomOrientationSpec",
    "ReceiverConfig",
    "SimulationConfig",
    "TargetConfig",
    "TransmitterConfig",
    "SpinOrientationSpec",
    "build_simulation_config",
    "perform_offline_pipeline",
    "perform_pipeline",
    "perform_pipeline_streaming",
]
