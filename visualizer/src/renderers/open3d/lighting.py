"""Material and transparency controls for Open3D.

This mixin owns Filament material records, texture-map loading, and
transparency updates for named Open3D geometry. It remains backend-specific
because Open3D exposes material knobs that are not available in every renderer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...materials.catalog import (
    ITU_TO_PBR,
    PBR_ADVANCED_FIELDS,
    PBR_WAVE_A_FIELDS,
    PBR_WAVE_B_FIELDS,
    effective_emissive_color,
    effective_texture_path,
    material_preset,
    pbr_props_to_kwargs,
    textures_disabled,
)
from ...materials.texture_policy import apply_texture_policy_to_props, warn_for_texture_policy

logger = get_logger("orchav.renderer_open3d.lighting")

__all__ = (
    "ITU_TO_PBR",
    "PBR_ADVANCED_FIELDS",
    "PBR_USE_BASE_PREFIX",
    "PBR_WAVE_A_FIELDS",
    "PBR_WAVE_B_FIELDS",
    "MaterialLightingMixin",
    "effective_emissive_color",
    "effective_texture_path",
    "material_preset",
    "pbr_props_to_kwargs",
    "textures_disabled",
)

_test_mat = rendering.MaterialRecord()
PBR_USE_BASE_PREFIX: bool = hasattr(_test_mat, "base_roughness")
"""True when Open3D 0.19+ API is available (``base_roughness`` vs ``roughness``)."""
del _test_mat


class MaterialLightingMixin:
    """Mixin providing Open3D PBR materials and transparency updates.

    Mixed into ``Open3DRenderer`` which supplies all ``self.*`` attributes
    (``_o3d_vis``, ``_geometry_names``, ``_pbr_materials``, etc.).

    Sun direction, sun enable/disable, and IBL enable/disable are **not**
    managed here because O3DVisualizer's "Sun Follows Camera" (enabled by
    default, no Python binding) overrides them on every mouse event.  Users
    control these via the O3D Settings panel ("Show O3D Settings" checkbox).
    """

    @staticmethod
    def _display_pbr_factors(
        *,
        name: str,
        roughness: float,
        metallic: float,
        reflectance: float,
        has_albedo: bool,
    ) -> tuple[float, float, float]:
        """Return Open3D display factors for scalar PBR materials.

        Filament's default outdoor IBL can turn low-roughness, fully metallic
        targets into near-white mirrors when no albedo map is present.  Keep
        the neutral base color legible for untextured target assets while
        leaving textured assets and static scene materials physically closer to
        the requested values.
        """
        roughness_clamped = max(0.0, min(1.0, roughness))
        metallic_clamped = max(0.0, min(1.0, metallic))
        reflectance_clamped = max(0.0, min(1.0, reflectance))
        if (
            str(name).startswith("target:")
            and not has_albedo
            and metallic_clamped >= 0.9
            and roughness_clamped <= 0.25
            and reflectance_clamped >= 0.7
        ):
            return (
                max(roughness_clamped, 0.55),
                min(metallic_clamped, 0.45),
                min(reflectance_clamped, 0.35),
            )
        return roughness_clamped, metallic_clamped, reflectance_clamped

    def _load_texture_cached(self, texture_path: str) -> Optional["o3d.geometry.Image"]:
        """Return a native image backed by the shared decoded texture asset."""
        from ...materials.texture_assets import load_decoded_texture, texture_asset_identity

        identity_result = texture_asset_identity(texture_path)
        if identity_result is not None:
            identity, resolved_path = identity_result
            img = self._texture_image_cache.get(identity)
            if img is not None:
                return img
            texture_path = str(resolved_path)

        asset = load_decoded_texture(texture_path)
        if asset is None:
            return None
        img = self._texture_image_cache.get(asset.identity)
        if img is not None:
            return img
        previous_identity = self._texture_source_identities.get(asset.path)
        if previous_identity is not None and previous_identity != asset.identity:
            self._texture_image_cache.pop(previous_identity, None)
        try:
            img = o3d.geometry.Image(np.array(asset.rgba, copy=True))
            self._texture_image_cache[asset.identity] = img
            self._texture_source_identities[asset.path] = asset.identity
            logger.debug("Created and cached Open3D texture '%s'", asset.path)
            return img
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Failed to create Open3D texture '%s': %s", asset.path, exc)
            return None

    @staticmethod
    def _apply_advanced_pbr(
        material: "rendering.MaterialRecord",
        *,
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.0,
        anisotropy: float = 0.0,
        emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        emissive_intensity: float = 0.0,
        transmission: float = 0.0,
        glass_thickness: float = 0.0,
        absorption_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Apply advanced PBR properties to an Open3D MaterialRecord.

        Wave A (clearcoat, anisotropy, emissive) and Wave B (transmission,
        glass_thickness, absorption_color) are both applied here.
        Filament's ``emissive_color`` is an RGBA-like tuple whose magnitude
        implies intensity, so RGB is premultiplied by ``emissive_intensity``.

        ``glass_thickness`` maps to Filament's ``material.thickness``
        attribute. The distinct parameter name disambiguates it from
        the XML-derived physical-material thickness metadata that flows
        through scene I/O and is shown in the materials panel as a
        read-only label.

        ``absorption_color`` tints transmitted light as it passes through
        the volumetric glass medium — Filament defaults this to white
        (no tint), so it is set only when the caller passes a non-white
        value to avoid overwriting the default for plain materials.

        **Perf note:** every property write below is guarded on a
        non-default value. The first iteration of this helper wrote all
        five scalar properties unconditionally, which regressed the
        Open3D renderer by ~260 ms/frame on dense city-scale scenes because
        ``material.transmission = 0.0`` and ``material.thickness = 0.0``
        override Filament's defaults of 1.0 and engage a heavier shader
        path on every non-glass material. Keeping the defaults in place
        for materials that don't opt in recovers that cost while
        preserving advanced PBR for materials that do.
        """
        if clearcoat > 0.0:
            material.base_clearcoat = max(0.0, min(1.0, float(clearcoat)))
        if clearcoat_roughness > 0.0:
            material.base_clearcoat_roughness = max(0.0, min(1.0, float(clearcoat_roughness)))
        if anisotropy > 0.0:
            material.base_anisotropy = max(0.0, min(1.0, float(anisotropy)))
        if emissive_intensity > 0.0 and any(c > 0.0 for c in emissive_color):
            er, eg, eb = emissive_color
            scale = float(emissive_intensity)
            material.emissive_color = [er * scale, eg * scale, eb * scale, 1.0]
        if transmission > 0.0:
            material.transmission = max(0.0, min(1.0, float(transmission)))
        if glass_thickness > 0.0:
            material.thickness = max(0.0, float(glass_thickness))
        if any(abs(c - 1.0) > 1e-6 for c in absorption_color):
            ar, ag, ab = absorption_color
            material.absorption_color = [
                max(0.0, min(1.0, float(ar))),
                max(0.0, min(1.0, float(ag))),
                max(0.0, min(1.0, float(ab))),
            ]

    def _apply_pbr_maps(
        self,
        material: "rendering.MaterialRecord",
        *,
        normal_map_path: Optional[str] = None,
        roughness_map_path: Optional[str] = None,
        ao_map_path: Optional[str] = None,
        metallic_map_path: Optional[str] = None,
    ) -> None:
        """Load optional PBR detail map channels into a Filament material."""
        if normal_map_path is not None:
            img = self._load_texture_cached(normal_map_path)
            if img is not None:
                material.normal_img = img
        if roughness_map_path is not None:
            img = self._load_texture_cached(roughness_map_path)
            if img is not None:
                material.roughness_img = img
        if ao_map_path is not None:
            img = self._load_texture_cached(ao_map_path)
            if img is not None:
                material.ao_img = img
        if metallic_map_path is not None:
            img = self._load_texture_cached(metallic_map_path)
            if img is not None:
                material.metallic_img = img

    @staticmethod
    def _select_shader(*, alpha: float, shader_variant: Optional[str]) -> str:
        """Pick the Filament shader variant for a material.

        Precedence (transparency wins over SSR):
        1. ``alpha < 1.0`` → ``defaultLitTransparency``. Glass and water
           opt into both SSR and alpha; without this branch first they
           would lose their alpha-blended look.
        2. ``shader_variant == "defaultLitSSR"`` → ``defaultLitSSR``.
        3. Otherwise → ``defaultLit``.
        """
        if alpha < 1.0:
            return "defaultLitTransparency"
        if shader_variant == "defaultLitSSR":
            return "defaultLitSSR"
        return "defaultLit"

    def modify_geometry_material_pbr(
        self,
        name: str,
        color: list[float],
        roughness: float = 0.5,
        metallic: float = 0.0,
        reflectance: float = 0.5,
        alpha: float = 1.0,
        color_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0),
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.0,
        anisotropy: float = 0.0,
        emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        emissive_intensity: float = 0.0,
        transmission: float = 0.0,
        glass_thickness: float = 0.0,
        absorption_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        texture_path: Optional[str] = None,
        normal_map_path: Optional[str] = None,
        normal_map_strength: float = 1.0,
        roughness_map_path: Optional[str] = None,
        ao_map_path: Optional[str] = None,
        metallic_map_path: Optional[str] = None,
        uv_scale_meters: float = 2.0,
        uv_repeat_scale: Optional[tuple[float, float]] = None,
        shader_variant: Optional[str] = None,
    ) -> bool:
        """Modify material properties without re-uploading geometry."""
        if self._o3d_vis is None:
            return False

        if name not in self._geometry_names:
            logger.debug("Cannot modify '%s' - not in scene", name)
            return False

        try:
            texture_props = {
                "color": color,
                "alpha": alpha,
                "texture_path": texture_path,
                "normal_map_path": normal_map_path,
                "roughness_map_path": roughness_map_path,
                "ao_map_path": ao_map_path,
                "metallic_map_path": metallic_map_path,
            }
            texture_policy = apply_texture_policy_to_props(
                texture_props,
                color=color,
                alpha=alpha,
                context=name,
            )[1]
            warn_for_texture_policy(texture_policy, log=logger)
            texture_path = texture_policy.active_maps["texture_path"]
            normal_map_path = texture_policy.active_maps["normal_map_path"]
            roughness_map_path = texture_policy.active_maps["roughness_map_path"]
            ao_map_path = texture_policy.active_maps["ao_map_path"]
            metallic_map_path = texture_policy.active_maps["metallic_map_path"]

            material = rendering.MaterialRecord()

            material.shader = self._select_shader(alpha=alpha, shader_variant=shader_variant)

            albedo_img = None
            if texture_path is not None:
                albedo_img = self._load_texture_cached(texture_path)
            if albedo_img is not None:
                material.base_color = list(texture_policy.renderer_base_color)
                material.albedo_img = albedo_img
            else:
                material.base_color = [color[0], color[1], color[2], alpha]
            material.base_color = [
                float(material.base_color[index]) * float(color_multiplier[index])
                for index in range(3)
            ] + [float(material.base_color[3])]

            roughness_clamped, metallic_clamped, reflectance_clamped = self._display_pbr_factors(
                name=name,
                roughness=roughness,
                metallic=metallic,
                reflectance=reflectance,
                has_albedo=albedo_img is not None,
            )

            if PBR_USE_BASE_PREFIX:
                material.base_roughness = roughness_clamped
                material.base_metallic = metallic_clamped
                material.base_reflectance = reflectance_clamped
            else:
                material.roughness = roughness_clamped
                material.metallic = metallic_clamped
                material.reflectance = reflectance_clamped

            self._apply_advanced_pbr(
                material,
                clearcoat=clearcoat,
                clearcoat_roughness=clearcoat_roughness,
                anisotropy=anisotropy,
                emissive_color=emissive_color,
                emissive_intensity=emissive_intensity,
                transmission=transmission,
                glass_thickness=glass_thickness,
                absorption_color=absorption_color,
            )

            self._apply_pbr_maps(
                material,
                normal_map_path=normal_map_path,
                roughness_map_path=roughness_map_path,
                ao_map_path=ao_map_path,
                metallic_map_path=metallic_map_path,
            )

            self._o3d_vis.modify_geometry_material(name, material)
            self._pbr_materials[name] = material
            self._post_redraw()
            if not self._frame_update_in_progress:
                self._request_visibility_settle_redraw(f"PBR material '{name}'")

            logger.debug(
                "Modified PBR material for '%s' " "(roughness=%.2f, metallic=%.2f, alpha=%.2f)",
                name,
                roughness,
                metallic,
                alpha,
            )
            return True

        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to modify PBR material '%s': %s", name, exc)
            return False

    def set_coverage_transparency(self, alpha: float) -> bool:
        """Set transparency for the coverage mesh.

        Uses white base_color so that per-vertex colors are preserved;
        only the alpha channel controls opacity.
        """
        if self._o3d_vis is None:
            return False

        try:
            mat = rendering.MaterialRecord()
            if alpha < 1.0:
                mat.shader = "defaultLitTransparency"
            else:
                mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 1.0, 1.0, float(alpha)]

            if self.COVERAGE_MESH_NAME in self._geometry_names:
                self._o3d_vis.modify_geometry_material(self.COVERAGE_MESH_NAME, mat)
                self._post_redraw()
                return True
            else:
                self._coverage_alpha = alpha
                return True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to set coverage transparency: %s", exc)
            return False

    def set_mesh_transparency(
        self,
        name: str,
        alpha: float,
        color: Optional[list] = None,
        defer_redraw: bool = False,
        roughness: float = 0.5,
        metallic: float = 0.0,
        reflectance: float = 0.5,
    ) -> bool:
        """Set transparency for a named mesh while preserving PBR factors."""
        if self._o3d_vis is None:
            return False

        if name not in self._geometry_names:
            return False

        try:
            if color is None:
                color = [0.7, 0.7, 0.7]

            material = rendering.MaterialRecord()
            if alpha < 1.0:
                material.shader = "defaultLitTransparency"
            else:
                material.shader = "defaultLit"
            material.base_color = [color[0], color[1], color[2], float(alpha)]

            roughness_c = max(0.0, min(1.0, roughness))
            metallic_c = max(0.0, min(1.0, metallic))
            reflectance_c = max(0.0, min(1.0, reflectance))
            if PBR_USE_BASE_PREFIX:
                material.base_roughness = roughness_c
                material.base_metallic = metallic_c
                material.base_reflectance = reflectance_c
            else:
                material.roughness = roughness_c
                material.metallic = metallic_c
                material.reflectance = reflectance_c

            self._o3d_vis.modify_geometry_material(name, material)

            if not defer_redraw:
                self._post_redraw()
                if not self._frame_update_in_progress:
                    try:
                        gui.Application.instance.run_one_tick()
                    except (RuntimeError, ValueError) as exc:
                        logger.debug("Failed to tick GUI after transparency update: %s", exc)
            return True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to set mesh transparency: %s", exc)
            return False

    def set_geometry_transparency(
        self,
        geometry: o3d.geometry.Geometry,
        alpha: float,
        color: Optional[list] = None,
        defer_redraw: bool = False,
        roughness: float = 0.5,
        metallic: float = 0.0,
        reflectance: float = 0.5,
    ) -> bool:
        """Set transparency for a geometry object by resolving its renderer name."""
        geom_id = id(geometry)
        name = self._geometry_id_to_name.get(geom_id)
        if name is None:
            return False
        return self.set_mesh_transparency(
            name, alpha, color, defer_redraw, roughness, metallic, reflectance
        )
