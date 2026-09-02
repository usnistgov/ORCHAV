"""Canonical orientation models and quaternion preparation facade.

The immutable models come from :mod:`shared.scenarios.actors`; evaluation is
owned by :mod:`generator.core.scenario_actors`. Scripted generator code may
import the models here for convenience. Every exposed model and evaluator
resolves to those shared specifications and the canonical quaternion kernel.
"""

from __future__ import annotations

from shared.scenarios.actors import (
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    OrientationKeyframeSpec,
    OrientationSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
)

from ..scenario_actors import (
    PreparedOrientation,
    Quaternion,
    Timeline,
    apply_asset_alignment,
    prepare_orientation,
)
from .adapters import (
    smoothing_time_from_step_fraction,
)
from .base import Orientation3, PreparedOrientationSource, orientation_to_tuple

__all__ = [
    "AlignMotionOrientationSpec",
    "FixedOrientationSpec",
    "KeyframesOrientationSpec",
    "LookAtOrientationSpec",
    "Orientation3",
    "OrientationKeyframeSpec",
    "OrientationSpec",
    "PreparedOrientation",
    "PreparedOrientationSource",
    "Quaternion",
    "RandomOrientationSpec",
    "SpinOrientationSpec",
    "Timeline",
    "apply_asset_alignment",
    "orientation_to_tuple",
    "prepare_orientation",
    "smoothing_time_from_step_fraction",
]
