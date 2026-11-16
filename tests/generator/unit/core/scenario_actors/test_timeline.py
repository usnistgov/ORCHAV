"""Authored sampling clocks remain identical across scene and source motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import load_simulation_config
from generator.core.scenario_actors.types import Timeline
from shared.scenarios.actors import TimelineSpec


@pytest.mark.parametrize("steps,rate", [(150, 30.0), (263, 120.0), (10240, 2000.0), (64, 1000.0)])
def test_authored_rate_matches_source_sample_clock_exactly(steps, rate):
    spec = TimelineSpec(steps=steps, sample_rate_hz=rate)
    clock = Timeline(steps, spec.duration_s, sample_rate_hz=rate)
    assert clock.timestamps_s == tuple(np.arange(steps, dtype=np.float64) / rate)
    assert clock.timestamps_s[-1] == spec.duration_s
    cfg = load_simulation_config(SimpleNamespace(timeline=spec))
    assert cfg.scene_sample_rate_hz == rate
    assert cfg.duration == spec.duration_s


@pytest.mark.parametrize(
    "steps,duration", [(1, 0.0), (1, 3.0), (4, 0.0), (150, 149 / 30), (18, 3.7)]
)
def test_duration_only_clock_keeps_existing_arithmetic(steps, duration):
    expected = (
        (0.0,)
        if steps == 1
        else tuple((duration / (steps - 1)) * i for i in range(steps - 1)) + (duration,)
    )
    assert Timeline(steps, duration).timestamps_s == expected


def test_single_sample_authored_rate_has_zero_timestamp():
    assert Timeline(1, 0.0, sample_rate_hz=30).timestamps_s == (0.0,)


@pytest.mark.parametrize("rate", [0, -30, float("inf"), float("nan"), True])
def test_invalid_authored_rate_rejected(rate):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        Timeline(150, 149 / 30, sample_rate_hz=rate)


def test_conflicting_duration_rejected_without_snapping():
    with pytest.raises(ValueError, match="conflicts"):
        Timeline(150, 5.0, sample_rate_hz=30)
