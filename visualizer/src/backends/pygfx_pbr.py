"""Backend-local material policy for pygfx 0.17.

HDR environment texels and their mipmaps stay in linear radiance. pygfx 0.17's
PBR and skybox shaders otherwise decode every environment as sRGB, regardless
of the texture's colorspace. Only these material subclasses specialize that
sampling step; albedo textures and scientific vertex colors are untouched.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def needs_physical_material(material: Any) -> bool:
    """Use the fast 4% dielectric shader unless the payload needs more control."""
    return bool(material.has_advanced_pbr or float(material.reflectance) != 0.5)


def apply_display_reflectance(native: Any, reflectance: float) -> None:
    """Map display reflectance to Filament's dielectric F0 = 0.16 * r**2.

    r in [0, 1] maps to F0 in [0, 0.16] and IOR in [1, 7/3]. The default
    r=0.5 has IOR=1.5, matching Standard's fixed F0=0.04. Metalness continues
    to blend toward albedo, independently of this dielectric parameter.
    """
    r = max(0.0, min(1.0, float(reflectance)))
    if hasattr(native, "ior"):
        root_f0 = 0.4 * r
        native.ior = (1.0 + root_f0) / (1.0 - root_f0)
    if hasattr(native, "specular_intensity"):
        native.specular_intensity = 1.0


def pbr_material_classes(gfx: Any) -> tuple[Any, Any, Any]:
    """Return local material classes, retaining lightweight test-double support."""
    if getattr(gfx, "__name__", None) == "pygfx":
        return _native_pbr_classes()
    return (
        getattr(gfx, "MeshStandardMaterial", None),
        getattr(gfx, "MeshPhysicalMaterial", None),
        getattr(gfx, "BackgroundSkyboxMaterial", None),
    )


def _replace_shader_fragment(source: str, old: str, new: str, count: int) -> str:
    """Fail visibly if an upstream shader changes this compatibility contract."""
    if source.count(old) != count:
        raise RuntimeError("Unsupported pygfx environment shader; expected pygfx 0.17")
    return source.replace(old, new)


@lru_cache(maxsize=1)
def _native_pbr_classes() -> tuple[Any, Any, Any]:
    import pygfx as gfx
    from pygfx.renderers.wgpu import load_wgsl, register_wgpu_render_function
    from pygfx.renderers.wgpu.shaders.backgroundshader import BackgroundShader
    from pygfx.renderers.wgpu.shaders.meshshader import MeshPhysicalShader, MeshStandardShader

    pbr_code = _replace_shader_fragment(
        load_wgsl("light_pbr.wgsl"),
        "srgb2physical(envMapColor_srgb.rgb)",
        "orchav_environment_radiance(envMapColor_srgb.rgb)",
        2,
    )
    pbr_code = """
fn orchav_environment_radiance(sampled: vec3<f32>) -> vec3<f32> {
    $$ if orchav_linear_environment
        return sampled;
    $$ else
        return srgb2physical(sampled);
    $$ endif
}
""" + pbr_code
    mesh_code = _replace_shader_fragment(
        load_wgsl("mesh.wgsl"),
        "{$ include 'pygfx.light_pbr.wgsl' $}",
        pbr_code,
        1,
    )
    background_code = _replace_shader_fragment(
        load_wgsl("background.wgsl"),
        "let physical_color = srgb2physical(final_color.rgb);",
        """$$ if orchav_linear_environment
        let physical_color = final_color.rgb;
    $$ else
        let physical_color = srgb2physical(final_color.rgb);
    $$ endif""",
        1,
    )

    class LinearEnvironmentStandardMaterial(gfx.MeshStandardMaterial):
        """Standard shading with colorspace-aware HDR environment sampling."""

    class LinearEnvironmentPhysicalMaterial(gfx.MeshPhysicalMaterial):
        """Physical shading with colorspace-aware HDR environment sampling."""

    class LinearEnvironmentSkyboxMaterial(gfx.BackgroundSkyboxMaterial):
        """Skybox shading with colorspace-aware HDR environment sampling."""

    class LinearEnvironmentShaderMixin:
        def get_bindings(self, wobject, shared, scene):
            bindings = super().get_bindings(wobject, shared, scene)
            env_map = getattr(wobject.material, "env_map", None) or getattr(
                scene, "environment", None
            )
            self["orchav_linear_environment"] = bool(
                env_map is not None and env_map.texture.colorspace == "physical"
            )
            return bindings

        def get_code(self):
            return mesh_code

    @register_wgpu_render_function(gfx.WorldObject, LinearEnvironmentStandardMaterial)
    class LinearEnvironmentStandardShader(LinearEnvironmentShaderMixin, MeshStandardShader):
        pass

    @register_wgpu_render_function(gfx.WorldObject, LinearEnvironmentPhysicalMaterial)
    class LinearEnvironmentPhysicalShader(LinearEnvironmentShaderMixin, MeshPhysicalShader):
        pass

    @register_wgpu_render_function(gfx.Background, LinearEnvironmentSkyboxMaterial)
    class LinearEnvironmentSkyboxShader(BackgroundShader):
        def get_bindings(self, wobject, shared, scene):
            bindings = super().get_bindings(wobject, shared, scene)
            texture_map = wobject.material.map
            self["orchav_linear_environment"] = bool(
                texture_map is not None and texture_map.texture.colorspace == "physical"
            )
            return bindings

        def get_code(self):
            return background_code

    return (
        LinearEnvironmentStandardMaterial,
        LinearEnvironmentPhysicalMaterial,
        LinearEnvironmentSkyboxMaterial,
    )
