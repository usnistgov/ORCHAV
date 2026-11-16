"""Runtime lighting profiles for the pygfx renderer.

These presets describe pygfx's direct-light rig and environment intensity.
Appearance snapshots preserve the selected identifier and complete light state.
Open3D/Filament lighting uses its own native controls.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lighting import DEFAULT_PYGFX_IBL_INTENSITY

INSPECTION_PROFILE = "inspection"
SIONNA_RT_PREVIEW_PROFILE = "sionna_rt_preview"
OUTDOOR_REALISTIC_PROFILE = "outdoor_realistic"
# Keep the experiment's identifier compatible with existing callers/captures.
PARIS_DAYLIGHT_PROFILE = "paris_daylight"
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
    # None keeps ORCHAV's existing fixed bias. A texel offset scales with the
    # world area covered by the shadow map, avoiding a city-sized offset indoors.
    shadow_bias_texels: float | None = None


PYGFX_LIGHTING_PROFILES: dict[str, PygfxLightingProfile] = {
    PARIS_DAYLIGHT_PROFILE: PygfxLightingProfile(
        key=PARIS_DAYLIGHT_PROFILE,
        label="Directional light",
        ambient_intensity=0.45,
        key_intensity=3.0,
        fill_intensity=0.25,
        headlight_enabled=False,
        headlight_intensity=1.2,
        key_azimuth_deg=31.0,
        key_elevation_deg=-42.0,
        fill_azimuth_deg=-149.0,
        fill_elevation_deg=-60.0,
        ibl_intensity=2000.0,
        shadows_enabled=True,
        shadow_bias_texels=7.0,
    ),
    INSPECTION_PROFILE: PygfxLightingProfile(
        key=INSPECTION_PROFILE,
        label="Even lighting",
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
        label="RT preview lighting",
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
        label="Strong contrast",
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

PYGFX_PRIMARY_LIGHTING_PROFILES = (
    INSPECTION_PROFILE,
    PARIS_DAYLIGHT_PROFILE,
    OUTDOOR_REALISTIC_PROFILE,
)

PYGFX_LIGHTING_PROFILE_LABELS: dict[str, str] = {
    **{key: profile.label for key, profile in PYGFX_LIGHTING_PROFILES.items()},
    CUSTOM_PROFILE: "Custom",
}


def environment_lighting_profile(name: str) -> PygfxLightingProfile | None:
    """Return the catalog sky rig used by desktop and notebook views."""
    from ...lighting.assets import environment_catalog

    asset = environment_catalog().get(name)
    if asset is None:
        return None
    values = asset.lighting
    profile = PygfxLightingProfile(
        key=CUSTOM_PROFILE,
        label="Custom",
        ambient_intensity=values["ambient"],
        key_intensity=values["key"],
        fill_intensity=values["fill"],
        ibl_intensity=values["ibl"],
        key_azimuth_deg=values["key_azimuth"],
        key_elevation_deg=values["key_elevation"],
        fill_azimuth_deg=53,
        fill_elevation_deg=-45,
        headlight_enabled=False,
        headlight_intensity=1.2,
        shadows_enabled=True,
    )
    return profile
