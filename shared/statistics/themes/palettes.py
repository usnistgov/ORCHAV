"""Optional presentation palettes shared by paths, legends and figures."""

DARK_PRESENTATION_COLORMAP = "orchav_dark"
GLOW_COLORMAP = "orchav_glow"
PAPER_COLORMAP = "orchav_paper"
PRESENTATION_COLORMAPS = {
    DARK_PRESENTATION_COLORMAP: [(0.12, 0.94, 0.77), (0.40, 0.84, 1.0), (1.0, 0.72, 0.27)],
    GLOW_COLORMAP: [(0.05, 1.0, 1.0), (0.60, 0.25, 1.0), (1.0, 0.10, 0.60)],
    PAPER_COLORMAP: [(0.0, 0.39, 0.29), (0.0, 0.32, 0.63), (0.69, 0.20, 0.015)],
}


def register_presentation_colormap(name: str = DARK_PRESENTATION_COLORMAP) -> None:
    """Register one shared presentation ramp on first use.

    This is a presentation palette, not a perceptually uniform scientific
    scale. Callers retain their existing data normalization and units.
    """
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if name not in colormaps:
        colormaps.register(
            LinearSegmentedColormap.from_list(
                name,
                PRESENTATION_COLORMAPS[name],
            )
        )
