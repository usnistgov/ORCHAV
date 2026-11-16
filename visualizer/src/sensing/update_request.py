"""Typed update payload for the pop-out sensing dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from shared.extensions.sensing import assert_sensing_contract, assert_supported_sensing_schema

if TYPE_CHECKING:
    from shared.sensing.replay import ReplayEvent

    from .dashboard_data import RecordingHistory
    from .motion_overview import MotionOverview
    from .prepared_angle import PreparedAngle
    from .processing_inspection import ProcessingInspection
    from .saved_tracking import SavedTracking
    from .stage_inspector import DetectionStageBundle


class _DisplayDataToken:
    """Validation state owned by one private dashboard request.

    Cached sensing containers are read-only by ownership; numeric products use
    immutable backing buffers. Replacing the sensing mapping never transfers
    ownership, even when dataclasses.replace preserves this token.
    """

    def __init__(self, sensing: dict[str, Any]) -> None:
        self.sensing = sensing
        self.require_complete: bool | None = None


@dataclass(frozen=True)
class SensingUpdateRequest:
    """Typed payload for SensingWindow updates."""

    display_frame: dict[str, Any]
    state_frame: dict[str, Any]
    state_step: Optional[int]
    display_step: Optional[int] = None
    display_from_cache: bool = False
    coherent_cir_time_steps: Optional[int] = None
    rp_result: Any = None
    rd_result: Any = None
    variant: str = "auto"
    range_mode: str = "monostatic"
    carrier_freq_hz: float = 3.5e9
    prf_hz: float = 30.0
    sampling_frequency: float = 1e9
    cir_history: Optional[list[np.ndarray]] = None
    cir_history_times_s: Optional[list[float]] = None
    replay_event: Optional[ReplayEvent] = None
    detection_stage_bundle: Optional[DetectionStageBundle] = None
    recording_managed: bool = False
    # A saved result supplies the recording's product contract, not a selected
    # event for the current scene (for example, before the first complete CPI).
    recording_overview_only: bool = False
    recording_history: RecordingHistory | None = None
    motion_overview: MotionOverview | None = None
    tracking_history: RecordingHistory | None = None
    saved_tracking: SavedTracking | None = None
    saved_tracking_error: str | None = None
    available_product_kinds: frozenset[str] | None = None
    # Sparse STFT publication is a recording capability, independent of the
    # selected result's products and whichever overview family is resident.
    recording_stft_variants: frozenset[str] | None = None
    selective_products: bool = False
    processing_inspection: ProcessingInspection | None = None
    source_identity: tuple = ()
    prepared_evaluation: tuple[dict, dict] | None = None
    prepared_angular: PreparedAngle | None = None
    prepared_views: frozenset[str] = frozenset()
    # Assigned only to owned dashboard payloads; cached retention is bounded.
    # Held-frame updates retain this token while rebinding their scene context.
    display_data_token: object | None = None

    def validate_display_frame(self) -> None:
        """Validate each owned result once, and caller-owned payloads on every use."""
        sensing = self.display_frame.get("sensing")
        assert_supported_sensing_schema(sensing)
        token = self.display_data_token
        require_complete = not self.selective_products
        owned_token = (
            token if isinstance(token, _DisplayDataToken) and token.sensing is sensing else None
        )
        if (
            owned_token is not None
            and owned_token.require_complete is not None
            and (owned_token.require_complete or not require_complete)
        ):
            return
        assert_sensing_contract(
            self.display_frame,
            require_schema_version=True,
            require_described_products=require_complete,
        )
        if owned_token is not None:
            owned_token.require_complete = require_complete

    @cached_property
    def evaluation(self) -> dict:
        """Reuse motion references and default scores for this immutable request.

        dataclasses.replace creates a fresh cache, so replacing a frame cannot
        carry an evaluation prepared for an older recording/result.
        """
        if self.prepared_evaluation is not None:
            frame, evidence = self.prepared_evaluation
            if frame is self.display_frame:
                return evidence
        from .dashboard_evaluation import evaluation_event

        return evaluation_event(self.display_frame.get("sensing") or {}, self.display_frame)

    @property
    def effective_display_step(self) -> Optional[int]:
        """Return the displayed frame index, falling back to the animation state step."""
        if self.display_step is not None:
            return self.display_step
        return self.state_step
