"""Synchronize external camera moves with pygfx's damped input controller.

pygfx 0.17 caches camera state for the lifetime of an input action. Updating
the camera alone therefore loses Follow translation on the next native tick.
Keep this small, tested compatibility boundary here; do not cancel wheel/drag
actions when following an actor. There is no public state-rebase API in 0.17.
"""

from __future__ import annotations


def rebase_controller(controller, camera, *, cancel=False):
    """Rebase pending input on the applied pose, optionally ending old gestures."""
    if controller is None:
        return
    actions = getattr(controller, "_actions", None)
    if cancel and actions is not None:
        actions.clear()
    if hasattr(controller, "_last_cam_state"):
        controller._last_cam_state = camera.get_state()
