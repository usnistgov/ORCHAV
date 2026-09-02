"""Actor pose preparation.

``prepare_scenario`` is the normal entry point.  The resulting immutable
``PreparedScenario`` is shared by generator runtime, preview, and figure
adapters; renderer-specific Euler/radian conversion belongs at those adapters.
``state.ActorStateManager`` owns the indexed cache consumed by propagation.
"""

from .errors import PosePreparationError, PreparationIssue
from .mobility import (
    derive_group_member_mobility,
    group_offset_from_world_position,
    prepare_mobility,
    prepare_sampled_mobility,
)
from .orientation import apply_asset_alignment, prepare_orientation
from .preparation import prepare_mobility_with_resources, prepare_scenario
from .quaternion import Quaternion
from .types import (
    ActorRole,
    Position3,
    PreparedActorPose,
    PreparedGroupPose,
    PreparedMobility,
    PreparedOrientation,
    PreparedScenario,
    Timeline,
    Velocity3,
)

__all__ = [
    "ActorRole",
    "PosePreparationError",
    "Position3",
    "PreparationIssue",
    "PreparedActorPose",
    "PreparedGroupPose",
    "PreparedMobility",
    "PreparedOrientation",
    "PreparedScenario",
    "Quaternion",
    "Timeline",
    "Velocity3",
    "apply_asset_alignment",
    "derive_group_member_mobility",
    "group_offset_from_world_position",
    "prepare_mobility",
    "prepare_mobility_with_resources",
    "prepare_orientation",
    "prepare_sampled_mobility",
    "prepare_scenario",
]
