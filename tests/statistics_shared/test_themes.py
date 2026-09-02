"""
Unit tests for shared.statistics themes module.

These tests verify theme structure, manager singleton, and listener notification.
"""

import pytest

from shared.statistics import theme_manager
from shared.statistics.themes import (
    AVAILABLE_CATEGORICAL_COLORMAPS,
    AVAILABLE_CONTINUOUS_COLORMAPS,
    DARK_THEME,
    DEFAULT_THEME,
    Theme,
    ThemeManager,
)


class TestThemeStructure:
    """Tests for Theme dataclass structure."""

    def test_default_theme_has_name(self):
        """Default theme should have a name."""
        assert DEFAULT_THEME.name == "default"

    def test_dark_theme_has_name(self):
        """Dark theme should have a name."""
        assert DARK_THEME.name == "dark"

    def test_reflection_order_has_7_colors(self):
        """Reflection order palette should have 7 colors (0-6+)."""
        assert len(DEFAULT_THEME.reflection_order) == 7
        assert len(DARK_THEME.reflection_order) == 7

    def test_reflection_order_colors_are_rgb(self):
        """Each reflection order color should be [R, G, B]."""
        for color in DEFAULT_THEME.reflection_order:
            assert len(color) == 3
            for component in color:
                assert 0.0 <= component <= 1.0

    def test_interaction_type_has_expected_keys(self):
        """Interaction type dict should have standard keys."""
        expected_keys = {0, 1, 2, 4, 8}  # LoS, Specular, Diffuse, Refraction, Diffraction
        assert set(DEFAULT_THEME.interaction_type.keys()) == expected_keys
        assert set(DARK_THEME.interaction_type.keys()) == expected_keys

    def test_interaction_type_colors_are_rgb(self):
        """Each interaction type color should be [R, G, B]."""
        for itype, color in DEFAULT_THEME.interaction_type.items():
            assert len(color) == 3
            for component in color:
                assert 0.0 <= component <= 1.0

    def test_data_source_colors_are_rgb(self):
        """Data source colors should be [R, G, B]."""
        for color in [
            DEFAULT_THEME.measurement,
            DEFAULT_THEME.sionna_filtered,
            DEFAULT_THEME.sionna_unfiltered,
        ]:
            assert len(color) == 3

    def test_plot_colors_are_rgb(self):
        """Plot background/text/grid colors should be [R, G, B]."""
        for color in [
            DEFAULT_THEME.plot_background,
            DEFAULT_THEME.text_color,
            DEFAULT_THEME.grid_color,
        ]:
            assert len(color) == 3

    def test_colormaps_are_strings(self):
        """Colormap names should be strings."""
        assert isinstance(DEFAULT_THEME.continuous_colormap, str)
        assert isinstance(DEFAULT_THEME.categorical_colormap, str)


class TestThemeMethods:
    """Tests for Theme helper methods."""

    def test_get_order_color_valid_index(self):
        """get_order_color should return correct color for valid index."""
        color = DEFAULT_THEME.get_order_color(0)
        assert color == DEFAULT_THEME.reflection_order[0]

    def test_get_order_color_clamps_high_index(self):
        """get_order_color should clamp to 6 for high indices."""
        color_6 = DEFAULT_THEME.get_order_color(6)
        color_10 = DEFAULT_THEME.get_order_color(10)
        color_100 = DEFAULT_THEME.get_order_color(100)
        assert color_6 == color_10 == color_100

    def test_get_order_color_clamps_negative_index(self):
        """get_order_color should clamp negative orders to LoS."""
        color = DEFAULT_THEME.get_order_color(-1)
        assert color == DEFAULT_THEME.reflection_order[0]

    def test_get_type_color_valid_type(self):
        """get_type_color should return correct color for valid type."""
        color = DEFAULT_THEME.get_type_color(0)  # LoS
        assert color == DEFAULT_THEME.interaction_type[0]

    def test_get_type_color_invalid_type_returns_gray(self):
        """get_type_color should return gray for unknown type."""
        color = DEFAULT_THEME.get_type_color(999)  # Invalid type
        assert color == [0.5, 0.5, 0.5]

    def test_to_hex_conversion(self):
        """to_hex should convert RGB to hex string."""
        result = DEFAULT_THEME.to_hex([1.0, 0.0, 0.0])
        assert result == "#ff0000"

        result = DEFAULT_THEME.to_hex([0.0, 1.0, 0.0])
        assert result == "#00ff00"

        result = DEFAULT_THEME.to_hex([0.0, 0.0, 1.0])
        assert result == "#0000ff"

    def test_get_categorical_colors(self):
        """get_categorical_colors should return n distinct colors."""
        colors = DEFAULT_THEME.get_categorical_colors(5)
        assert len(colors) == 5
        for color in colors:
            assert len(color) == 3


class TestThemeManager:
    """Tests for ThemeManager singleton."""

    def test_singleton_pattern(self):
        """ThemeManager should be a singleton."""
        manager1 = ThemeManager()
        manager2 = ThemeManager()
        assert manager1 is manager2

    def test_global_instance(self):
        """theme_manager should be a ThemeManager instance."""
        assert isinstance(theme_manager, ThemeManager)

    def test_default_theme_is_default(self):
        """Initial theme should be DEFAULT_THEME."""
        # Reset to default
        theme_manager.set_theme("default")
        assert theme_manager.current.name == "default"

    def test_available_themes(self):
        """Should list available themes."""
        themes = theme_manager.available_themes
        assert "default" in themes
        assert "dark" in themes

    def test_set_theme_dark(self):
        """Should be able to switch to dark theme."""
        theme_manager.set_theme("dark")
        assert theme_manager.current.name == "dark"
        # Reset
        theme_manager.set_theme("default")

    def test_set_theme_invalid_raises(self):
        """Setting unknown theme should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown theme"):
            theme_manager.set_theme("nonexistent")

    def test_listener_notification(self):
        """Listeners should be notified on theme change."""
        notifications = []

        def listener(theme):
            notifications.append(theme.name)

        theme_manager.add_listener(listener)
        try:
            theme_manager.set_theme("dark")
            theme_manager.set_theme("default")
            assert notifications == ["dark", "default"]
        finally:
            theme_manager.remove_listener(listener)

    def test_remove_listener(self):
        """Removed listeners should not be notified."""
        notifications = []

        def listener(theme):
            notifications.append(theme.name)

        theme_manager.add_listener(listener)
        theme_manager.remove_listener(listener)
        theme_manager.set_theme("dark")
        theme_manager.set_theme("default")
        assert notifications == []

    def test_set_colormap_continuous(self):
        """Should be able to change continuous colormap."""
        original = theme_manager.current.continuous_colormap
        theme_manager.set_colormap("continuous", "plasma")
        assert theme_manager.current.continuous_colormap == "plasma"
        # Reset
        theme_manager.set_colormap("continuous", original)

    def test_set_colormap_categorical(self):
        """Should be able to change categorical colormap."""
        original = theme_manager.current.categorical_colormap
        theme_manager.set_colormap("categorical", "Set1")
        assert theme_manager.current.categorical_colormap == "Set1"
        # Reset
        theme_manager.set_colormap("categorical", original)

    def test_set_colormap_invalid_type_raises(self):
        """Setting unknown colormap type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown colormap type"):
            theme_manager.set_colormap("invalid", "viridis")

    def test_colormap_change_notifies_listeners(self):
        """Colormap change should notify listeners."""
        notifications = []

        def listener(theme):
            notifications.append(theme.continuous_colormap)

        theme_manager.add_listener(listener)
        try:
            theme_manager.set_colormap("continuous", "inferno")
            assert "inferno" in notifications
        finally:
            theme_manager.remove_listener(listener)
            theme_manager.set_colormap("continuous", "viridis")

    def test_register_custom_theme(self):
        """Should be able to register custom themes."""
        custom = Theme(
            name="custom",
            reflection_order=[[1, 0, 0]] * 7,
            interaction_type={0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 4: [1, 1, 0], 8: [1, 0, 1]},
        )
        theme_manager.register_theme(custom)
        assert "custom" in theme_manager.available_themes
        theme_manager.set_theme("custom")
        assert theme_manager.current.name == "custom"
        # Reset
        theme_manager.set_theme("default")


class TestAvailableColormaps:
    """Tests for available colormap lists."""

    def test_continuous_colormaps_not_empty(self):
        """Should have some continuous colormaps."""
        assert len(AVAILABLE_CONTINUOUS_COLORMAPS) > 0

    def test_categorical_colormaps_not_empty(self):
        """Should have some categorical colormaps."""
        assert len(AVAILABLE_CATEGORICAL_COLORMAPS) > 0

    def test_viridis_in_continuous(self):
        """viridis should be in continuous colormaps."""
        assert "viridis" in AVAILABLE_CONTINUOUS_COLORMAPS

    def test_tab10_in_categorical(self):
        """tab10 should be in categorical colormaps."""
        assert "tab10" in AVAILABLE_CATEGORICAL_COLORMAPS
