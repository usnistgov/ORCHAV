from __future__ import annotations

import pytest

from generator.core.pipeline.schedule import FixedIntervalScheduler


def test_fixed_interval_scheduler_marks_every_step_as_acquisition() -> None:
    scheduler = FixedIntervalScheduler(first_step=0, last_step=4, interval_length=1)

    events = [scheduler.event_for_step(i) for i in range(5)]
    assert all(e.is_acquisition_step for e in events)
    assert all(e.is_interval_end for e in events)
    assert [e.interval_index for e in events] == [0, 1, 2, 3, 4]


def test_fixed_interval_scheduler_tracks_interval_boundaries() -> None:
    scheduler = FixedIntervalScheduler(first_step=0, last_step=9, interval_length=4)

    e0 = scheduler.event_for_step(0)
    e3 = scheduler.event_for_step(3)
    e4 = scheduler.event_for_step(4)
    e7 = scheduler.event_for_step(7)
    e9 = scheduler.event_for_step(9)

    assert e0.is_acquisition_step is True
    assert e0.is_interval_end is False
    assert e0.interval_index == 0
    assert e0.sample_index_in_interval == 0

    assert e3.is_interval_end is True
    assert e3.interval_size == 4

    assert e4.is_acquisition_step is True
    assert e4.interval_index == 1
    assert e4.sample_index_in_interval == 0

    assert e7.is_interval_end is True
    assert e7.interval_size == 4

    # Last interval is truncated (8..9)
    assert e9.is_interval_end is True
    assert e9.interval_size == 2
    assert e9.interval_end_step == 9


@pytest.mark.parametrize("step_idx", [-1, 10])
def test_fixed_interval_scheduler_rejects_out_of_range_steps(step_idx: int) -> None:
    scheduler = FixedIntervalScheduler(first_step=0, last_step=9, interval_length=4)

    with pytest.raises(ValueError, match="outside scheduled range"):
        scheduler.event_for_step(step_idx)


def test_fixed_interval_scheduler_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="last_step"):
        FixedIntervalScheduler(first_step=5, last_step=4, interval_length=1)


def test_fixed_interval_scheduler_rejects_invalid_interval_length() -> None:
    with pytest.raises(ValueError, match="interval_length"):
        FixedIntervalScheduler(first_step=0, last_step=4, interval_length=0)
