"""MPC display recipes for global looks and independent path adjustments."""

import math
from dataclasses import replace

from PySide6.QtCore import QObject, Signal

from shared.statistics.themes import theme_manager
from shared.statistics.themes.palettes import (
    PRESENTATION_COLORMAPS,
    register_presentation_colormap,
)

from ..materials.mpc_styles import DESCRIPTIONS, PRESETS, SCENE_MPC_PRESETS  # noqa: F401
from ..scene.defaults import MPC_LINE_WIDTH_BOUNDS_PX, MPC_POINT_SIZE_BOUNDS_PX


class MpcAppearanceService(QObject):
    """Capture and restore actual display values without changing path selection."""

    changed = Signal()

    def __init__(self, visualizer):
        super().__init__()
        self.viz = visualizer

    def capture(self) -> dict:
        v, r = self.viz, self.viz.renderer
        return dict(
            version=1,
            color_mode=v.app_state.color_mode,
            continuous_colormap=theme_manager.current.continuous_colormap,
            categorical_colormap=theme_manager.current.categorical_colormap,
            line_width=r.get_line_width(),
            point_size=r.get_point_size(),
            bounce_points=v.app_state.mpc_visibility.bounce_points,
        )

    @staticmethod
    def validate(data: dict) -> dict:
        from matplotlib import colormaps

        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Invalid MPC appearance version")
        if data.get("color_mode") not in (
            "reflection_order",
            "mpc_type",
            "delay",
            "path_loss",
            "material",
            "reconstruction_type",
        ):
            raise ValueError("Invalid MPC coloring mode")
        for key in ("continuous_colormap", "categorical_colormap"):
            name = data.get(key)
            if not isinstance(name, str):
                raise ValueError(f"Invalid MPC palette: {key}")
            if name in PRESENTATION_COLORMAPS:
                register_presentation_colormap(name)
            if name not in colormaps and not (
                key == "categorical_colormap" and name == "theme_default"
            ):
                raise ValueError(f"Unknown MPC palette: {name}")
        for key, bounds in (
            ("line_width", MPC_LINE_WIDTH_BOUNDS_PX),
            ("point_size", MPC_POINT_SIZE_BOUNDS_PX),
        ):
            value = data.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not bounds[0] <= value <= bounds[1]
            ):
                raise ValueError(f"Invalid MPC size: {key}")
        if not isinstance(data.get("bounce_points"), bool):
            raise ValueError("Invalid MPC interaction visibility")
        return dict(data)

    def apply(self, name: str) -> None:
        """Apply a named path style, retaining the frame, visible paths and filters."""
        if name not in PRESETS:
            raise ValueError(f"Unknown MPC appearance preset: {name}")
        self.restore({**self.capture(), **PRESETS[name]})

    def restore(self, data: dict) -> None:
        data = self.validate(data)
        v, r = self.viz, self.viz.renderer
        r.set_line_width(data["line_width"])
        r.set_point_size(data["point_size"])
        for kind in ("continuous", "categorical"):
            name = data[f"{kind}_colormap"]
            if name != getattr(theme_manager.current, f"{kind}_colormap"):
                theme_manager.set_colormap(kind, name)
        v.set_state(
            color_mode=data["color_mode"],
            mpc_visibility=replace(v.app_state.mpc_visibility, bounce_points=data["bounce_points"]),
        )
        # Palette listeners may publish before the coloring mode is restored.
        # Coalesce one final packet after every path-style field is authoritative.
        schedule = getattr(v, "schedule_update", None)
        if callable(schedule) and not getattr(v, "_shutdown_started", False):
            schedule()
        self.changed.emit()

    def current_name(self) -> str:
        """Identify an unchanged built-in style without storing a stale UI label."""
        data = self.capture()
        return next(
            (
                name
                for name, recipe in PRESETS.items()
                if all(data[key] == value for key, value in recipe.items())
            ),
            "Custom",
        )
