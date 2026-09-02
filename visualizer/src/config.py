"""
Configuration module for ORCHAV.

This module consolidates configuration constants and default values used throughout
the visualizer. It provides a single source of truth for configuration settings.
"""

from typing import Tuple

# Performance Configuration

# Target 60 FPS for smooth UI updates
UPDATE_DEBOUNCE_MS = 16  # ~60 FPS (1000ms / 60fps ≈ 16.67ms)

# Cache sizes
MAX_VIEW_MODEL_CACHE_SIZE = 1000  # Maximum number of cached ViewModels
MAX_FRAME_CACHE_SIZE = 50  # Maximum number of cached raw frames
DEFAULT_CANON_CACHE_MB = 2048  # Default canonical data cache budget (MB)

# Rendering Configuration

# Default marker sizes for TX/RX nodes
TX_MARKER_SIZE = 0.3  # meters
RX_MARKER_SIZE = 0.3  # meters

# MPC visualization
MPC_LINE_WIDTH = 1.0
MPC_POINT_SIZE = 2.0
BOUNCE_POINT_SIZE = 3.0

# Color Configuration

# Default colors (RGB, 0-1 range)
DEFAULT_TX_COLOR = (1.0, 0.0, 0.0)  # Red
DEFAULT_RX_COLOR = (0.0, 0.0, 1.0)  # Blue
DEFAULT_BOUNCE_COLOR = (1.0, 1.0, 0.0)  # Yellow
DEFAULT_BACKGROUND_COLOR = (0.1, 0.1, 0.1)  # Dark gray

# Color maps for different visualization modes
REFLECTION_ORDER_COLORS = {
    0: (0.0, 1.0, 0.0),  # Green (direct path/LOS)
    1: (1.0, 1.0, 0.0),  # Yellow (1st reflection)
    2: (1.0, 0.5, 0.0),  # Orange (2nd reflection)
    3: (1.0, 0.0, 0.0),  # Red (3rd reflection)
    4: (0.5, 0.0, 0.5),  # Purple (4th reflection)
}

INTERACTION_TYPE_COLORS = {
    0: (1.0, 0.84, 0.0),  # Gold (LoS)
    1: (0.29, 0.56, 0.89),  # Blue (specular)
    2: (0.2, 0.85, 0.2),  # Lime green (diffuse)
    4: (0.61, 0.35, 0.71),  # Purple (refraction)
    8: (0.95, 0.55, 0.20),  # Orange (diffraction)
}

# UI Configuration

# Window dimensions
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080

# Panel dimensions
CONTROL_PANEL_WIDTH = 400
CONTROL_PANEL_MIN_WIDTH = 300

# Font sizes
LABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
STATS_FONT_SIZE = 9

# Data Configuration

# Default filter values
DEFAULT_ALLOWED_ORDERS = [0, 1, 2, 3, 4]  # Allow up to 4th order reflections
DEFAULT_ALLOWED_TYPES = [1, 2, 4, 8]  # All interaction types

# Default visualization modes

# Animation Configuration

# Animation timing
DEFAULT_FPS = 30
MIN_FPS = 1
MAX_FPS = 60

# Preloading
DEFAULT_PRELOAD_FRAMES = 10  # Number of frames to preload ahead

# File Paths Configuration

# These are defaults and can be overridden by scenario AppConfig.
DEFAULT_SCENE_PATH = "scene.xml"
DEFAULT_DATA_PATH = "simulation_data.h5"

# Validation Constants

# Bounds checking
MAX_REFLECTION_ORDER = 10  # Reasonable maximum for reflection order

# Helper Functions


def get_color_for_reflection_order(order: int) -> Tuple[float, float, float]:
    """Get color for a given reflection order.

    Args:
        order: Reflection order (0 = LOS, 1+ = reflections)

    Returns:
        RGB color tuple (0-1 range)
    """
    if order in REFLECTION_ORDER_COLORS:
        return REFLECTION_ORDER_COLORS[order]

    # For higher orders, interpolate or use a default
    return (0.5, 0.5, 0.5)  # Gray for unknown orders


def get_color_for_interaction_type(interaction_type: int) -> Tuple[float, float, float]:
    """Get color for a given interaction type.

    Args:
        interaction_type: Interaction type (1=specular, 2=diffuse, 4=refraction, 8=diffraction)

    Returns:
        RGB color tuple (0-1 range)
    """
    if interaction_type in INTERACTION_TYPE_COLORS:
        return INTERACTION_TYPE_COLORS[interaction_type]

    return (0.5, 0.5, 0.5)  # Gray for unknown types
