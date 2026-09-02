"""Disabled hook for reserved sensing-compatible generator extensions.

The public generator keeps this state object so the offline pipeline has one
import path, but v0.1 does not generate sensing payloads. Existing HDF5 frames
may still carry an optional ``StandardMPCFrame.sensing`` extension payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNSUPPORTED_SENSING_GENERATION = (
    "Sensing generation is reserved for a future ORCHAV extension; this public "
    "release can read existing HDF5 frames with optional sensing extension "
    "payloads, but it does not generate them."
)


@dataclass(frozen=True)
class SensingPipelineState:
    """No-op state used when public scenarios do not enable sensing generation."""

    enabled: bool = False

    @classmethod
    def from_config(
        cls,
        scenario_configuration: Any,
        *,
        frame_dt_s: float,
        gt_dt_s: float,
        cir_steps: int,
        first_step: int,
    ) -> "SensingPipelineState":
        """Create disabled state or reject explicitly enabled sensing generation."""
        del frame_dt_s, gt_dt_s, cir_steps, first_step
        sensing_cfg = getattr(scenario_configuration, "sensing", None) or {}
        if isinstance(sensing_cfg, dict):
            enabled = bool(sensing_cfg.get("enabled", False))
        else:
            enabled = getattr(sensing_cfg, "enabled", False) is True
        if enabled:
            raise RuntimeError(_UNSUPPORTED_SENSING_GENERATION)
        return cls(enabled=False)

    def process_frame(
        self,
        frame_data: dict[str, Any],
        *,
        step_idx: int,
        is_rt_step: bool,
        schedule_event: Any,
    ) -> None:
        """Leave public generator frames unchanged."""
        del frame_data, step_idx, is_rt_step, schedule_event

    def generate_summary(
        self,
        scenario_configuration: Any,
        *,
        output_root: Any = None,
        strict: bool = False,
    ) -> None:
        """Public builds do not generate sensing summary artifacts."""
        del scenario_configuration, output_root, strict


__all__ = ["SensingPipelineState"]
