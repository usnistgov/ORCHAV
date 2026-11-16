"""Endpoint-inclusive clocks resolve from one unambiguous authored cadence."""

from __future__ import annotations

import math

import pytest

from shared.scenarios.actors import TimelineSpec


@pytest.mark.parametrize("count, rate", [(1, 1000), (2, 1000), (512, 1000), (1001, 30000 / 1001)])
def test_sample_rate_resolves_endpoint(count, rate):
    timeline = TimelineSpec(steps=count, sample_rate_hz=rate)
    assert timeline.duration_s == (count - 1) / rate
    assert timeline.model_dump()["duration_s"] == timeline.duration_s


@pytest.mark.parametrize("rate", [0, -1, math.nan, math.inf, True, False])
def test_invalid_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        TimelineSpec(steps=512, sample_rate_hz=rate)


def test_missing_clock_is_rejected():
    with pytest.raises(ValueError, match="duration_s or sample_rate_hz"):
        TimelineSpec.model_validate({"steps": 512})


@pytest.mark.parametrize("count, duration", [(512, 0.512), (1, 0.001), (2, 0.0)])
def test_conflicting_duration_does_not_silently_choose_a_clock(count, duration):
    with pytest.raises(ValueError, match="conflicts with sample_rate_hz"):
        TimelineSpec(steps=count, sample_rate_hz=1000, duration_s=duration)


def test_redundant_duration_tolerates_only_roundoff_and_resolves_exact_rate():
    timeline = TimelineSpec(steps=512, sample_rate_hz=1000, duration_s=math.nextafter(0.511, 1))
    assert timeline.duration_s == 0.511


@pytest.mark.parametrize("count, duration", [(512, 0.512), (1, 10.0), (2, 0.0)])
def test_duration_only_scenarios_preserve_the_existing_contract(count, duration):
    timeline = TimelineSpec(steps=count, duration_s=duration)
    assert timeline.duration_s == duration
    assert timeline.sample_rate_hz is None


def test_unrepresentable_duration_is_rejected():
    with pytest.raises(ValueError, match="finite duration_s"):
        TimelineSpec(steps=10**400, sample_rate_hz=1.0)


def test_schema_allows_rate_without_a_redundant_duration():
    assert TimelineSpec.model_json_schema()["required"] == ["steps"]
