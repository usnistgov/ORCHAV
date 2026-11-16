"""MPC display recipes shared by desktop and notebook views."""

from shared.statistics.themes.palettes import (
    DARK_PRESENTATION_COLORMAP,
    GLOW_COLORMAP,
    PAPER_COLORMAP,
)

PRESETS = {
    "Standard": dict(
        color_mode="reflection_order",
        continuous_colormap="RdYlGn_r",
        categorical_colormap="theme_default",
        line_width=2.0,
        point_size=5.0,
        bounce_points=True,
    ),
    "Paper": dict(
        continuous_colormap=PAPER_COLORMAP,
        categorical_colormap="tab10",
        line_width=1.5,
        point_size=3.0,
        bounce_points=True,
    ),
    "Dark": dict(
        color_mode="delay",
        continuous_colormap=DARK_PRESENTATION_COLORMAP,
        categorical_colormap="theme_default",
        line_width=2.2,
        point_size=4.0,
        bounce_points=False,
    ),
    "Glow": dict(
        color_mode="delay",
        continuous_colormap=GLOW_COLORMAP,
        categorical_colormap="theme_default",
        line_width=2.2,
        point_size=4.0,
        bounce_points=False,
    ),
}

SCENE_MPC_PRESETS = {
    "Detailed": "Standard",
    "Paper": "Paper",
    "Plan": "Standard",
    "Dark": "Dark",
    "Glow": "Glow",
}

DESCRIPTIONS = {
    "Standard": "Reflection-order colors with small interaction markers.",
    "Paper": "Dark colors for white backgrounds; keeps your current coloring metric.",
    "Dark": "Delay colors from turquoise through ice blue to gold, with clear path lines.",
    "Glow": "Delay colors from cyan through violet to pink. Use the Glow scene preset for bloom.",
}
