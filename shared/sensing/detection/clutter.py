"""Streaming baseline-removal helpers for sensing power vectors.

``SlidingBaselineRemover`` maintains a short history of previous samples and
subtracts their mean from the current sample. It is intentionally stateful and
small so post-processing code can apply moving clutter suppression without
depending on a larger pipeline object.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np


@dataclass
class BaselineRemovalResult:
    """Result of subtracting a sliding baseline from one sample."""

    cleaned: np.ndarray
    baseline: np.ndarray
    frames_used: int
    processing: dict[str, Any]


class SlidingBaselineRemover:
    """Maintain a causal previous-sample baseline and subtract it.

    After warm-up with a window of ``K`` updates, the linear (unclipped)
    operation is ``y[n] = x[n] - (1/K) sum(x[n-k], k=1..K)``.  It has a
    direct current-sample term and therefore zero algorithmic output latency;
    its phase/group delay is frequency-dependent rather than a single delay.
    During warm-up the mean uses every available prior sample, and the first
    sample passes through because no baseline exists.  ``clip_zero=True`` adds
    a nonlinear projection after subtraction, so an LTI transfer-function
    claim no longer applies to the complete operation.
    """

    def __init__(self, window: int, *, clip_zero: bool = False) -> None:
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or window <= 0
        ):
            raise ValueError("window must be a positive integer")
        self.window = int(window)
        self.clip_zero = clip_zero
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.window)

    def process(self, sample: np.ndarray) -> BaselineRemovalResult:
        arr = np.asarray(sample)
        if not np.isfinite(arr).all():
            raise ValueError("Baseline samples must be finite")
        if self.clip_zero and np.iscomplexobj(arr):
            raise ValueError("clip_zero requires real-valued samples")
        if self._buffer and arr.shape != self._buffer[0].shape:
            raise ValueError("Sample shape must remain fixed within baseline history")
        frames_used = len(self._buffer)
        if frames_used:
            stacked = np.stack(self._buffer, axis=0)
            baseline = stacked.mean(axis=0)
        else:
            baseline = np.zeros_like(arr)

        cleaned = arr - baseline
        if self.clip_zero:
            cleaned = np.clip(cleaned, 0.0, None)

        self._buffer.append(arr.copy())
        return BaselineRemovalResult(
            cleaned=cleaned,
            baseline=baseline,
            frames_used=frames_used,
            processing=self.processing_contract(frames_used=frames_used),
        )

    def processing_contract(self, *, frames_used: int | None = None) -> dict[str, Any]:
        """Return exact delay, transfer, transient, and clipping semantics."""

        used = len(self._buffer) if frames_used is None else int(frames_used)
        if used < 0 or used > self.window:
            raise ValueError("frames_used must lie between zero and the configured window")
        return {
            "operator": "causal_previous_sample_mean_subtraction",
            "steady_state_equation": "y[n]=x[n]-(1/K)*sum(x[n-k],k=1..K)",
            "steady_state_transfer_function": ("H(z)=1-(1/K)*sum(z**(-k),k=1..K)"),
            "window_samples": self.window,
            "history_delay_samples": [1, self.window],
            "output_latency_samples": 0,
            "group_delay_policy": "frequency_dependent_no_single_group_delay",
            "transient_policy": "mean_all_available_prior_samples_first_sample_passthrough",
            "transient_length_updates": self.window,
            "history_samples_used": used,
            "steady_state": used == self.window,
            "post_subtraction_clip_zero": self.clip_zero,
            "complete_operator_lti": not self.clip_zero and used == self.window,
        }
