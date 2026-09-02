"""Base theme definition for shared statistics figures.

``Theme`` stores semantic colors for MPC paths, comparison plots, and generated
figures. It is intentionally data-only: UI widget styling stays in the
visualizer, while shared figure code reads these colors and colormap names.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Theme:
    """Normalized RGB colors and colormaps for shared visual outputs.

    This theme controls colors for:
    - Renderer-neutral MPC paths (reflection order, interaction type)
    - Matplotlib/pyqtgraph plots (backgrounds, text, grids)
    - Comparison plots (measurement vs simulation)
    - Colormaps for heatmaps and categorical data

    RGB triplets use the closed interval ``[0, 1]``. Interaction code ``0`` is
    a semantic plotting category for LoS paths, not a physical bounce stored in
    ``StandardMPCFrame.interactions``.
    """

    name: str

    # -------------------------------------------------------------------------
    # MPC Coloring
    # -------------------------------------------------------------------------

    # Reflection order colors (indices 0-6+)
    # Index 0 = LoS/direct path, 1-5 = reflection orders, 6+ = clamped to 6
    reflection_order: List[List[float]] = field(default_factory=list)

    # Interaction type colors
    # Keys: 0=LoS, 1=Specular, 2=Diffuse, 4=Refraction, 8=Diffraction
    interaction_type: Dict[int, List[float]] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Data Source Colors (for comparison plots)
    # -------------------------------------------------------------------------

    measurement: List[float] = field(default_factory=lambda: [0.2, 0.6, 0.2])
    sionna_filtered: List[float] = field(default_factory=lambda: [0.8, 0.2, 0.2])
    sionna_unfiltered: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.6])

    # -------------------------------------------------------------------------
    # Figure/Plot Colors (matplotlib, pyqtgraph)
    # -------------------------------------------------------------------------

    plot_background: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    text_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grid_color: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])

    # -------------------------------------------------------------------------
    # Colormaps
    # -------------------------------------------------------------------------

    # For heatmaps, coverage maps, continuous data
    continuous_colormap: str = "viridis"

    # For bar charts, line plots, categorical data
    categorical_colormap: str = "tab10"

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_order_color(self, order: int) -> List[float]:
        """
        Get color for reflection order, clamping orders >= 6 to index 6.

        Args:
            order: Reflection order (0 = LoS, 1 = first reflection, etc.)

        Returns:
            RGB color as [R, G, B] with values in [0, 1]
        """
        if not self.reflection_order:
            return [0.5, 0.5, 0.5]  # Default gray
        clamped_order = max(0, min(order, len(self.reflection_order) - 1))
        return self.reflection_order[clamped_order]

    def get_type_color(self, itype: int) -> List[float]:
        """
        Get color for interaction type.

        Args:
            itype: Interaction type code (0=LoS, 1=Specular, 2=Diffuse,
                   4=Refraction, 8=Diffraction)

        Returns:
            RGB color as [R, G, B] with values in [0, 1]
        """
        return self.interaction_type.get(itype, [0.5, 0.5, 0.5])

    def get_continuous_cmap(self):
        """
        Get matplotlib colormap for continuous data.

        Returns:
            matplotlib.colors.Colormap instance
        """
        import matplotlib.pyplot as plt

        return plt.get_cmap(self.continuous_colormap)

    def get_categorical_colors(self, n: int) -> List[List[float]]:
        """
        Get n distinct colors from categorical colormap.

        Args:
            n: Number of colors needed

        Returns:
            List of RGB colors, each as [R, G, B] with values in [0, 1]
        """
        # Handle theme_default: use reflection_order colors from theme
        if self.categorical_colormap == "theme_default":
            if self.reflection_order:
                # Cycle through reflection order colors
                return [self.reflection_order[i % len(self.reflection_order)] for i in range(n)]
            # Fallback to tab10 if no reflection_order defined
            cmap_name = "tab10"
        else:
            cmap_name = self.categorical_colormap

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
        return [list(cmap(i / max(n - 1, 1))[:3]) for i in range(n)]

    def to_hex(self, rgb: List[float]) -> str:
        """
        Convert RGB color to hex string.

        Args:
            rgb: [R, G, B] with values in [0, 1]

        Returns:
            Hex color string like '#FF5500'
        """
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# Available colormaps for user selection
AVAILABLE_CONTINUOUS_COLORMAPS = [
    # Perceptually uniform (recommended for scientific visualization)
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    # Diverging (good for data with meaningful center point)
    "coolwarm",
    "RdBu",
    "RdYlGn",
    "RdYlGn_r",  # Reversed (red=bad, green=good) - good for signal strength
    "seismic",
    # Sequential (traditional)
    "hot",
    "YlOrRd",
    "Blues",
    "Greens",
]

AVAILABLE_CATEGORICAL_COLORMAPS = [
    "theme_default",  # Use semantic theme colors (green=LoS, etc.)
    "tab10",  # 10 distinct colors
    "tab20",  # 20 colors
    "Set1",  # Bold, saturated colors
    "Set2",  # Softer colors
    "Set3",  # Pastel colors
    "Paired",  # Pairs of similar colors
    "Accent",  # Accent colors
    "Dark2",  # Dark, distinct colors
    "viridis",  # Perceptually uniform
    "plasma",  # Purple-orange-yellow
]
