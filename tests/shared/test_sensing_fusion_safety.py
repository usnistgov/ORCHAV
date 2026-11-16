"""Epoch exception safety and geometry regressions from the fusion review."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from shared.sensing.fusion.core import MonostaticFusion, Observation, Sensor, measurement_model


def report(sensor, position, time=0):
    z, _ = measurement_model(np.r_[position, np.zeros(3)], sensor)
    return Observation(sensor.sensor_id, "report", time, *z)


@pytest.mark.parametrize("reverse_names", [False, True])
def test_unrelated_polar_track_does_not_prevent_birth(reverse_names):
    a = Sensor("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    b = Sensor("b", (5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    if reverse_names:
        a, b = replace(a, sensor_id="z"), replace(b, sensor_id="a")
    epoch = MonostaticFusion([a, b]).step(0, [report(a, [5, 0, 1]), report(b, [10, 0, 1])])
    positions = sorted(t["position_m"] for t in epoch["tracks"])
    np.testing.assert_allclose(positions, [[5, 0, 1], [10, 0, 1]], atol=1e-8)


def test_late_failure_preserves_all_state_and_retry(monkeypatch):
    a = Sensor("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    processor = MonostaticFusion([a])
    processor.step(0, [report(a, [5, 0, 1])])
    before = deepcopy(processor.__dict__)
    original = MonostaticFusion._update

    def failing(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("deliberate late failure")

    with monkeypatch.context() as patch:
        patch.setattr(MonostaticFusion, "_update", failing)
        with pytest.raises(RuntimeError, match="late failure"):
            processor.step(0.1, [report(a, [5.01, 0, 1], 0.1)])
    assert processor.time_s == before["time_s"]
    assert processor.next_id == before["next_id"]
    assert len(processor.tracks) == len(before["tracks"])
    for actual, expected in zip(processor.tracks, before["tracks"]):
        np.testing.assert_array_equal(actual.state, expected.state)
        np.testing.assert_array_equal(actual.covariance, expected.covariance)
        assert actual.sources == expected.sources
        assert actual.hit_epochs == expected.hit_epochs
        assert actual.missed_epochs == expected.missed_epochs
    control = MonostaticFusion([a])
    control.step(0, [report(a, [5, 0, 1])])
    assert processor.step(0.1, [report(a, [5.01, 0, 1], 0.1)]) == control.step(
        0.1, [report(a, [5.01, 0, 1], 0.1)]
    )
