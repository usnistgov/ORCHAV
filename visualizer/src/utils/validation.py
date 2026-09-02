"""Validation helpers for configuration and state management."""

from __future__ import annotations

MIN_STEP = 0
DEFAULT_COLOR_MODE = "reflection_order"
AVAILABLE_COLOR_MODES: list[str] = [
    "reflection_order",
    "mpc_type",
    "delay",
    "path_loss",
    "material",
    "reconstruction_type",
]


def validate_step(step: int, max_steps: int) -> int:
    """Clamp the requested step to the valid range."""
    return max(MIN_STEP, min(step, max_steps - 1))


def validate_color_mode(color_mode: str) -> str:
    """Ensure the provided color mode is supported."""
    if color_mode in AVAILABLE_COLOR_MODES:
        return color_mode
    return DEFAULT_COLOR_MODE
