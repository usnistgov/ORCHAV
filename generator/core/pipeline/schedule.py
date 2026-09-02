"""Step scheduling for output frames that share ray-tracing acquisitions.

The file pipeline may write an output frame for every simulation step while
running the expensive path solver only at fixed acquisition boundaries. This
module provides the small scheduling contract used by file output and derived
per-frame metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StepScheduleEvent:
    """Metadata for one output step in a fixed acquisition cadence.

    ``is_acquisition_step`` tells the pipeline to run a real path solve.
    Non-acquisition steps reuse the most recent solve while still writing an
    output frame with updated entity metadata.
    """

    step_idx: int
    relative_step: int
    interval_length: int
    interval_index: int
    sample_index_in_interval: int
    interval_start_step: int
    interval_end_step: int
    interval_size: int
    is_acquisition_step: bool
    is_interval_end: bool


class StepScheduler(ABC):
    """Map each output step to its ray-tracing acquisition interval."""

    @abstractmethod
    def event_for_step(self, step_idx: int) -> StepScheduleEvent:
        """Return schedule metadata for one output step."""


class FixedIntervalScheduler(StepScheduler):
    """Deterministic scheduler for fixed-cadence RT acquisition."""

    def __init__(self, *, first_step: int, last_step: int, interval_length: int):
        self.first_step = int(first_step)
        self.last_step = int(last_step)
        self.interval_length = int(interval_length)
        if self.last_step < self.first_step:
            raise ValueError("last_step must be greater than or equal to first_step")
        if self.interval_length < 1:
            raise ValueError("interval_length must be at least 1")

    def event_for_step(self, step_idx: int) -> StepScheduleEvent:
        step = int(step_idx)
        if step < self.first_step or step > self.last_step:
            raise ValueError(
                f"step_idx {step} is outside scheduled range "
                f"[{self.first_step}, {self.last_step}]"
            )
        relative = step - self.first_step
        interval_length = self.interval_length
        interval_index = relative // interval_length
        sample_index = relative % interval_length
        interval_start = self.first_step + interval_index * interval_length
        interval_end = min(interval_start + interval_length - 1, self.last_step)
        interval_size = interval_end - interval_start + 1

        is_acquisition_step = interval_length <= 1 or sample_index == 0
        is_interval_end = step == interval_end

        return StepScheduleEvent(
            step_idx=step,
            relative_step=relative,
            interval_length=interval_length,
            interval_index=int(interval_index),
            sample_index_in_interval=int(sample_index),
            interval_start_step=int(interval_start),
            interval_end_step=int(interval_end),
            interval_size=int(interval_size),
            is_acquisition_step=bool(is_acquisition_step),
            is_interval_end=bool(is_interval_end),
        )
