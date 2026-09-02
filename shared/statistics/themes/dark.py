"""
Dark Theme.

Theme optimized for dark backgrounds. Only background/text/grid colors change;
MPC visualization colors (reflection order, interaction type) remain the same
as the default theme for consistency.
"""

from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_LOS,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
)

from .base import Theme

DARK_THEME = Theme(
    name="dark",
    # -------------------------------------------------------------------------
    # Reflection order colors - SAME as default for consistency
    # Users can change MPC colors via the categorical colormap dropdown
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
    # Interaction type colors - SAME as default for consistency
    # -------------------------------------------------------------------------
    interaction_type={
        SIONNA_INTERACTION_LOS: [1.0, 0.84, 0.0],  # LoS - gold
        SIONNA_INTERACTION_SPECULAR: [0.29, 0.56, 0.89],  # Specular reflection - blue
        SIONNA_INTERACTION_DIFFUSE: [0.2, 0.85, 0.2],  # Diffuse scattering - lime green
        SIONNA_INTERACTION_REFRACTION: [0.61, 0.35, 0.71],  # Refraction - purple
        SIONNA_INTERACTION_DIFFRACTION: [0.95, 0.55, 0.20],  # Diffraction - orange
    },
    # -------------------------------------------------------------------------
    # Data source colors - SAME as default for consistency
    # -------------------------------------------------------------------------
    measurement=[0.2, 0.6, 0.2],  # Green - measurement data
    sionna_filtered=[0.8, 0.2, 0.2],  # Red - Sionna filtered
    sionna_unfiltered=[0.6, 0.6, 0.6],  # Gray - Sionna unfiltered
    # -------------------------------------------------------------------------
    # Figure/plot colors (for dark background) - THESE ARE THE ONLY CHANGES
    # -------------------------------------------------------------------------
    plot_background=[0.12, 0.12, 0.12],  # Dark gray background
    text_color=[0.9, 0.9, 0.9],  # Light gray text
    grid_color=[0.3, 0.3, 0.3],  # Dark gray grid
    # -------------------------------------------------------------------------
    # Colormaps - same as default
    # -------------------------------------------------------------------------
    continuous_colormap="RdYlGn_r",  # Same as default
    categorical_colormap="theme_default",  # Use semantic theme colors
)
