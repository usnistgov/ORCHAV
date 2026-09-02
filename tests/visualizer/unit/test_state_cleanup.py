"""Focused AppState invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from visualizer.src.state import create_initial_state


class TestAppStateStillFrozen:
    """Verify AppState is still frozen after field removal."""

    def test_still_frozen(self):
        state = create_initial_state()
        with pytest.raises(FrozenInstanceError):
            state.step = 99

    def test_remaining_standalone_fields_intact(self):
        state = create_initial_state()
        assert state.standalone_beamforming_mode == "standalone"
        assert state.standalone_antenna_rows == 1
        assert state.standalone_carrier_frequency_ghz == 28.0
        assert state.standalone_steering_strategy == "svd"
