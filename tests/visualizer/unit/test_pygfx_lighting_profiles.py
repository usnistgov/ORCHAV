"""Unit tests for pygfx lighting profiles without requiring a live GPU canvas."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from visualizer.src.renderers.pygfx.lighting_profiles import (
    CUSTOM_PROFILE,
    INSPECTION_PROFILE,
    OUTDOOR_REALISTIC_PROFILE,
    PYGFX_LIGHTING_PROFILES,
    SIONNA_RT_PREVIEW_PROFILE,
)
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer


class _FakeIBLManager:
    def __init__(self) -> None:
        self.pygfx_intensity = None

    def set_intensity(self, value: float) -> None:
        self.pygfx_intensity = float(value)


class _FakeLight:
    def __init__(self) -> None:
        self.intensity = 0.0
        self.cast_shadow = True
        self.local = SimpleNamespace(position=None)
        self.look_at_target = None

    def look_at(self, target) -> None:
        self.look_at_target = target


def _profile_renderer() -> PygfxRenderer:
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._ambient_light = _FakeLight()
    renderer._key_light = _FakeLight()
    renderer._fill_light = _FakeLight()
    renderer._head_light = _FakeLight()
    renderer._ibl_manager = _FakeIBLManager()
    renderer._lighting_profile_name = INSPECTION_PROFILE
    renderer._suppress_lighting_profile_custom = False
    renderer._base_ambient_intensity = 1.5
    renderer._base_key_intensity = 3.0
    renderer._base_fill_intensity = 1.0
    renderer._headlight_enabled = True
    renderer._headlight_intensity = 1.2
    renderer._key_light_azimuth_deg = -123.0
    renderer._key_light_elevation_deg = -48.0
    renderer._fill_light_azimuth_deg = 53.0
    renderer._fill_light_elevation_deg = -45.0
    renderer._ibl_intensity = 2000.0
    renderer._shadows_enabled = True
    renderer.redraw_count = 0
    renderer.request_redraw = lambda: setattr(renderer, "redraw_count", renderer.redraw_count + 1)
    return renderer


def test_lighting_profile_registry_contains_expected_profiles():
    assert PYGFX_LIGHTING_PROFILES[INSPECTION_PROFILE].headlight_enabled is True
    assert PYGFX_LIGHTING_PROFILES[SIONNA_RT_PREVIEW_PROFILE].headlight_enabled is False
    assert PYGFX_LIGHTING_PROFILES[SIONNA_RT_PREVIEW_PROFILE].ambient_intensity == pytest.approx(
        0.35
    )
    assert PYGFX_LIGHTING_PROFILES[SIONNA_RT_PREVIEW_PROFILE].fill_intensity == pytest.approx(0.25)
    assert PYGFX_LIGHTING_PROFILES[SIONNA_RT_PREVIEW_PROFILE].ibl_intensity == pytest.approx(5000.0)
    assert PYGFX_LIGHTING_PROFILES[OUTDOOR_REALISTIC_PROFILE].headlight_enabled is False
    assert PYGFX_LIGHTING_PROFILES[OUTDOOR_REALISTIC_PROFILE].ambient_intensity == pytest.approx(
        0.03
    )
    assert PYGFX_LIGHTING_PROFILES[OUTDOOR_REALISTIC_PROFILE].fill_intensity == pytest.approx(0.0)


def test_set_lighting_profile_applies_outdoor_realistic_state():
    renderer = _profile_renderer()

    assert renderer.set_lighting_profile(OUTDOOR_REALISTIC_PROFILE) is True

    state = renderer.get_light_rig_state()
    assert renderer.get_lighting_profile() == OUTDOOR_REALISTIC_PROFILE
    assert state["headlight_enabled"] is False
    assert state["ambient_intensity"] == pytest.approx(0.03)
    assert state["key_intensity"] == pytest.approx(4.5)
    assert state["fill_intensity"] == pytest.approx(0.0)
    assert state["key_azimuth_deg"] == pytest.approx(31.0)
    assert state["key_elevation_deg"] == pytest.approx(-17.0)
    assert state["shadows_enabled"] is True
    assert renderer._ibl_intensity == pytest.approx(0.0)
    assert renderer._ibl_manager.pygfx_intensity == pytest.approx(0.0)
    assert renderer._head_light.intensity == pytest.approx(0.0)
    assert renderer._key_light.cast_shadow is True


def test_unknown_lighting_profile_is_rejected():
    renderer = _profile_renderer()

    assert renderer.set_lighting_profile("missing") is False
    assert renderer.get_lighting_profile() == INSPECTION_PROFILE


def test_manual_lighting_edit_marks_profile_custom():
    renderer = _profile_renderer()
    assert renderer.set_lighting_profile(OUTDOOR_REALISTIC_PROFILE) is True

    assert renderer.set_key_light_intensity(2.25) is True

    assert renderer.get_lighting_profile() == CUSTOM_PROFILE
    assert renderer.get_light_rig_state()["key_intensity"] == pytest.approx(2.25)
