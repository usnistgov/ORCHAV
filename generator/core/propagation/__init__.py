"""Propagation frame computation and frame-state helpers.

Start with ``compute_ray_tracing_step``. It reads one frame from
``SimulationObjects.actor_state_manager``, applies optional live overrides,
mutates the live Sionna scene objects, runs the path solver, and returns frame
data.

Support modules separate the terms that otherwise look similar:
``actor_state_application`` applies per-frame TX/RX/target state to live scene
objects, ``snapshots`` builds immutable NumPy arrays stored in frame data, and
``frozen_paths`` freezes Sionna path tensors when a frame is cached.
"""

from .actor_state_application import (
    apply_target_scale_overrides,
    apply_target_state_to_scene,
    should_update_mesh_for_step,
)
from .frozen_paths import freeze_frame_paths
from .live_overrides import (
    LiveActorCategory,
    LiveActorOverride,
    LiveOverrideMap,
    category_from_value,
    normalize_live_overrides,
)
from .raytracing import compute_ray_tracing_step

__all__ = [
    "LiveActorOverride",
    "LiveActorCategory",
    "LiveOverrideMap",
    "apply_target_scale_overrides",
    "apply_target_state_to_scene",
    "category_from_value",
    "compute_ray_tracing_step",
    "freeze_frame_paths",
    "normalize_live_overrides",
    "should_update_mesh_for_step",
]
