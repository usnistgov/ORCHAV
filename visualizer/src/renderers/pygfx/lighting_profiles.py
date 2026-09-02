"""Runtime lighting profiles for the pygfx renderer.

These presets describe pygfx's direct-light rig and environment intensity.
They are renderer state, not session state, and intentionally do not model
Open3D/Filament lighting because Open3D exposes different native controls.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lighting import DEFAULT_PYGFX_IBL_INTENSITY

INSPECTION_PROFILE = "inspection"
SIONNA_RT_PREVIEW_PROFILE = "sionna_rt_preview"
OUTDOOR_REALISTIC_PROFILE = "outdoor_realistic"
CUSTOM_PROFILE = "custom"


@dataclass(frozen=True)
class PygfxLightingProfile:
    """A complete pygfx light-rig state that can be applied atomically."""

    key: str
    label: str
    ambient_intensity: float
    key_intensity: float
    fill_intensity: float
    headlight_enabled: bool
    headlight_intensity: float
    key_azimuth_deg: float
    key_elevation_deg: float
    fill_azimuth_deg: float
    fill_elevation_deg: float
    ibl_intensity: float
    shadows_enabled: bool


PYGFX_LIGHTING_PROFILES: dict[str, PygfxLightingProfile] = {
    INSPECTION_PROFILE: PygfxLightingProfile(
        key=INSPECTION_PROFILE,
        label="Inspection",
        ambient_intensity=1.5,
        key_intensity=3.0,
        fill_intensity=1.0,
        headlight_enabled=True,
        headlight_intensity=1.2,
        key_azimuth_deg=-123.0,
        key_elevation_deg=-48.0,
        fill_azimuth_deg=53.0,
        fill_elevation_deg=-45.0,
        ibl_intensity=DEFAULT_PYGFX_IBL_INTENSITY,
        shadows_enabled=True,
    ),
    SIONNA_RT_PREVIEW_PROFILE: PygfxLightingProfile(
        key=SIONNA_RT_PREVIEW_PROFILE,
        label="Sionna RT Preview",
        ambient_intensity=0.35,
        key_intensity=3.8,
        fill_intensity=0.25,
        headlight_enabled=False,
        headlight_intensity=1.2,
        key_azimuth_deg=31.0,
        key_elevation_deg=-28.0,
        fill_azimuth_deg=53.0,
        fill_elevation_deg=-45.0,
        ibl_intensity=5000.0,
        shadows_enabled=True,
    ),
    OUTDOOR_REALISTIC_PROFILE: PygfxLightingProfile(
        key=OUTDOOR_REALISTIC_PROFILE,
        label="Outdoor Realistic",
        ambient_intensity=0.03,
        key_intensity=4.5,
        fill_intensity=0.0,
        headlight_enabled=False,
        headlight_intensity=1.2,
        key_azimuth_deg=31.0,
        key_elevation_deg=-17.0,
        fill_azimuth_deg=53.0,
        fill_elevation_deg=-45.0,
        ibl_intensity=0.0,
        shadows_enabled=True,
    ),
}

PYGFX_LIGHTING_PROFILE_LABELS: dict[str, str] = {
    **{key: profile.label for key, profile in PYGFX_LIGHTING_PROFILES.items()},
    CUSTOM_PROFILE: "Custom",
}
