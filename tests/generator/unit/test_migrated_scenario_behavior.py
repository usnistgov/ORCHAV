"""Behavior checks for release-facing scenarios migrated to child roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from generator.core.scenario_actors import prepare_scenario
from shared.scenarios import load_scenario_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BEAMFORMING_ROOT = PROJECT_ROOT / "scenarios/visualizer/beamforming"


def test_circular_beamforming_receiver_starts_east_and_moves_counterclockwise() -> None:
    """The migrated degree/direction fields must match the authored v1 position."""
    scenario = load_scenario_configuration(
        BEAMFORMING_ROOT / "circular_rx",
        project_root=PROJECT_ROOT,
    )

    receiver = prepare_scenario(scenario).actor("OrbitRX")

    assert receiver.positions_m[0] == pytest.approx((12.0, 0.0, 1.5))
    assert receiver.positions_m[1][1] > 0.0
    assert receiver.positions_m[-1] == pytest.approx(receiver.positions_m[0])


def test_multi_device_beamforming_orbit_starts_north_and_moves_clockwise() -> None:
    """Pair-selection geometry must retain its declared start anchor and direction."""
    scenario = load_scenario_configuration(
        BEAMFORMING_ROOT / "multi_device",
        project_root=PROJECT_ROOT,
    )

    prepared = prepare_scenario(scenario)
    receiver = prepared.actor("OrbitRX")

    assert receiver.positions_m[0] == pytest.approx((0.0, 12.0, 1.5))
    assert receiver.positions_m[1][0] > 0.0
    assert receiver.positions_m[-1] == pytest.approx(receiver.positions_m[0])
    assert prepared.actor("EastTX").orientation.euler_deg[0][0] == pytest.approx(
        180.0,
        abs=1e-9,
    )
