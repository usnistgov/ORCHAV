from __future__ import annotations

from visualizer.src.sensing.update_request import SensingUpdateRequest


def test_effective_display_step_uses_state_step_when_missing() -> None:
    request = SensingUpdateRequest(
        display_frame={},
        state_frame={},
        state_step=12,
        display_step=None,
    )
    assert request.effective_display_step == 12
