"""Concise lighting choices must preserve complete saved light recipes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.renderers.pygfx.lighting_profiles import (
    CUSTOM_PROFILE,
    INSPECTION_PROFILE,
    OUTDOOR_REALISTIC_PROFILE,
    PARIS_DAYLIGHT_PROFILE,
    PYGFX_LIGHTING_PROFILE_LABELS,
    SIONNA_RT_PREVIEW_PROFILE,
)
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer


@pytest.fixture
def renderer():
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._lighting_profile_name = INSPECTION_PROFILE
    renderer._suppress_lighting_profile_custom = False
    renderer._key_light = None
    renderer._ibl_manager = SimpleNamespace(set_intensity=Mock())
    renderer._apply_light_intensities = Mock()
    renderer._apply_world_light_transforms = Mock()
    renderer.request_redraw = Mock()
    return renderer


def test_standard_choices_are_distinct_setups_with_stable_identifiers(renderer):
    assert renderer.get_available_lighting_profiles() == {
        INSPECTION_PROFILE: "Even lighting",
        PARIS_DAYLIGHT_PROFILE: "Directional light",
        OUTDOOR_REALISTIC_PROFILE: "Strong contrast",
        CUSTOM_PROFILE: "Custom",
    }
    choices = renderer.get_available_lighting_profiles()
    choices.pop(INSPECTION_PROFILE)
    assert INSPECTION_PROFILE in renderer.get_available_lighting_profiles()
    assert SIONNA_RT_PREVIEW_PROFILE in PYGFX_LIGHTING_PROFILE_LABELS
    renderer.request_redraw.assert_not_called()


def test_preview_recipe_still_applies_and_is_represented_when_active(renderer):
    assert renderer.set_lighting_profile(SIONNA_RT_PREVIEW_PROFILE)
    expected = {
        "lighting_profile": SIONNA_RT_PREVIEW_PROFILE,
        "ambient_intensity": 0.35,
        "headlight_enabled": False,
        "headlight_intensity": 1.2,
        "key_azimuth_deg": 31.0,
        "key_elevation_deg": -28.0,
        "key_intensity": 3.8,
        "fill_azimuth_deg": 53.0,
        "fill_elevation_deg": -45.0,
        "fill_intensity": 0.25,
        "shadows_enabled": True,
    }
    assert renderer.get_light_rig_state() == expected
    assert renderer.get_ibl_intensity() == 5000
    assert (
        renderer.get_available_lighting_profiles()[SIONNA_RT_PREVIEW_PROFILE]
        == "RT preview lighting"
    )

    assert renderer.set_lighting_profile(PARIS_DAYLIGHT_PROFILE)
    assert SIONNA_RT_PREVIEW_PROFILE not in renderer.get_available_lighting_profiles()
    assert renderer.restore_light_rig_state(expected, ibl=5000)
    assert renderer.get_light_rig_state() == expected
    assert renderer.get_ibl_intensity() == 5000
    assert SIONNA_RT_PREVIEW_PROFILE in renderer.get_available_lighting_profiles()
    renderer._ibl_manager.set_intensity.assert_called_with(pytest.approx(5000 / 30000))


def test_manual_light_edit_returns_to_custom_without_resetting_the_rig(renderer):
    assert renderer.set_lighting_profile(SIONNA_RT_PREVIEW_PROFILE)
    before = renderer.get_light_rig_state()
    assert renderer.set_key_light_intensity(2.7)
    assert renderer.get_light_rig_state() == {
        **before,
        "lighting_profile": CUSTOM_PROFILE,
        "key_intensity": 2.7,
    }
    assert SIONNA_RT_PREVIEW_PROFILE not in renderer.get_available_lighting_profiles()
    assert renderer.get_ibl_intensity() == 5000
