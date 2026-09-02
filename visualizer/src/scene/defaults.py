"""Shared defaults for the 3D scene viewport."""

from __future__ import annotations

DEFAULT_SCENE_BACKGROUND_PRESET = "Dark Gray"
DEFAULT_SCENE_BACKGROUND_COLOR = [0.2, 0.2, 0.2]
DEFAULT_SCENE_BACKGROUND_COLOR_RGBA = [0.2, 0.2, 0.2, 1.0]

# Scene-appearance controls use broad safety bounds rather than presentation-
# oriented slider ranges. Values must remain finite and positive where the
# renderer constructs geometry from them.
DEFAULT_NODE_MARKER_SIZE_M = 0.3
NODE_MARKER_SIZE_BOUNDS_M = (0.001, 1_000_000.0)

DEFAULT_LABEL_FONT_SIZE = 0.3
LABEL_FONT_SIZE_BOUNDS = (0.01, 100_000.0)

DEFAULT_LABEL_OFFSET_M = (1.5, 0.0, 1.0)
LABEL_OFFSET_BOUNDS_M = (-1_000_000.0, 1_000_000.0)

DEFAULT_ORIENTATION_SCALE_M = 3.0
ORIENTATION_SCALE_BOUNDS_M = (0.001, 1_000_000.0)

DEFAULT_MPC_LINE_WIDTH_PX = 2.0
MPC_LINE_WIDTH_BOUNDS_PX = (0.1, 1_000.0)

DEFAULT_MPC_POINT_SIZE_PX = 5.0
MPC_POINT_SIZE_BOUNDS_PX = (0.1, 1_000.0)

DEFAULT_TRAJECTORY_LINE_WIDTH_PX = 3.0
TRAJECTORY_LINE_WIDTH_BOUNDS_PX = (0.1, 1_000.0)

DEFAULT_TRAJECTORY_POINT_SIZE_PX = 6.0
TRAJECTORY_POINT_SIZE_BOUNDS_PX = (0.1, 1_000.0)

SCENE_BACKGROUND_PRESETS = {
    "Dark Gray": DEFAULT_SCENE_BACKGROUND_COLOR,
    "White": [1.0, 1.0, 1.0],
    "Light": [0.85, 0.85, 0.85],
    "Slate": [0.25, 0.28, 0.32],
    "Midnight": [0.05, 0.06, 0.10],
}
