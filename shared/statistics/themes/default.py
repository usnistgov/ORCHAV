"""
Default (Light) Theme.

This is the standard theme with colors optimized for light backgrounds.
Colors are chosen for good visibility on white/light plot backgrounds.
"""

from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_LOS,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
)

from .base import Theme

DEFAULT_THEME = Theme(
    name="default",
    # -------------------------------------------------------------------------
    # Reflection order colors (0 = LoS, 1-5 = orders, 6+ = high order)
    # These match the visualizer reflection-order palette for consistency.
    # -------------------------------------------------------------------------
    reflection_order=[
        [0.0, 1.0, 0.53],  # 0 (LoS) - bright green
        [1.0, 0.84, 0.0],  # 1st order - yellow
        [1.0, 0.55, 0.0],  # 2nd order - orange
        [1.0, 0.27, 0.27],  # 3rd order - red
        [0.6, 0.2, 0.8],  # 4th order - purple
        [0.25, 0.41, 0.88],  # 5th order - blue
        [0.18, 0.18, 0.18],  # 6+ orders - dark gray
    ],
    # -------------------------------------------------------------------------
    # Interaction type colors
    # Keys: 0=LoS, 1=Specular, 2=Diffuse, 4=Refraction, 8=Diffraction
    # -------------------------------------------------------------------------
    interaction_type={
        SIONNA_INTERACTION_LOS: [1.0, 0.84, 0.0],  # LoS - gold
        SIONNA_INTERACTION_SPECULAR: [0.29, 0.56, 0.89],  # Specular reflection - blue
        SIONNA_INTERACTION_DIFFUSE: [0.2, 0.85, 0.2],  # Diffuse scattering - lime green
        SIONNA_INTERACTION_REFRACTION: [0.61, 0.35, 0.71],  # Refraction - purple
        SIONNA_INTERACTION_DIFFRACTION: [0.95, 0.55, 0.20],  # Diffraction - orange
    },
    # -------------------------------------------------------------------------
    # Data source colors (for comparison plots)
    # -------------------------------------------------------------------------
    measurement=[0.2, 0.6, 0.2],  # Green - measurement data
    sionna_filtered=[0.8, 0.2, 0.2],  # Red - Sionna filtered
    sionna_unfiltered=[0.6, 0.6, 0.6],  # Gray - Sionna unfiltered
    # -------------------------------------------------------------------------
    # Figure/plot colors (for light background)
    # -------------------------------------------------------------------------
    plot_background=[1.0, 1.0, 1.0],  # White background
    text_color=[0.0, 0.0, 0.0],  # Black text
    grid_color=[0.8, 0.8, 0.8],  # Light gray grid
    # -------------------------------------------------------------------------
    # Colormaps
    # -------------------------------------------------------------------------
    continuous_colormap="RdYlGn_r",  # Red-Yellow-Green reversed (red=bad, green=good)
    categorical_colormap="theme_default",  # Use semantic theme colors (green=LoS, etc.)
)
