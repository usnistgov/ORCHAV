"""pygfx material construction and material-state helpers.

The mixin maps renderer-neutral ``MaterialPayload`` values onto pygfx material
classes while preserving ORCHAV's shared visual-state contract. It also owns
backend-specific compatibility policy for unlit color inspection, vertex-color
retention, clipping planes, texture-map binding, transparency, and PBR fields
that pygfx cannot currently represent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ...backends.pygfx_scene_helpers import (
    _is_pygfx_unlit_mode_enabled,
    apply_texture_policy_to_material_payload,
)
from ...types.render_payloads import MaterialPayload, SurfaceColorSource

__all__ = ["PygfxMaterialMixin"]

logger = logging.getLogger(__name__)


class PygfxMaterialMixin:
    """Material construction and update behavior for ``PygfxRenderer``.

    Geometry upload creates simple default materials. Later material updates
    are normalized through ``MaterialPayload`` so scene presets, UI controls,
    and backend-local object updates all follow one policy.
    """

    def set_color_fidelity_mode(self, enabled: bool) -> bool:
        """Toggle pygfx mesh materials between lit PBR and unlit color inspection."""
        enabled = bool(enabled)
        self._unlit_mode_enabled = enabled
        ok = True
        with self.batch_updates():
            for name, mat_payload in list(self._materials.items()):
                if name not in self._objects or self._kinds.get(name) != "mesh":
                    continue
                ok = self.set_named_material(name, mat_payload) and ok
        return ok

    def get_color_fidelity_mode(self) -> bool:
        """Return whether unlit color-inspection materials are active."""
        return bool(self._unlit_mode_enabled)

    def _material_apply_signature(
        self,
        name: str,
        material: MaterialPayload,
        obj_mat: Any,
        *,
        has_color_buffer: bool,
    ) -> tuple[Any, ...]:
        """Return the state tuple that makes a material update observable."""
        return (
            material,
            self._kinds.get(name),
            self._geometry_color_sources.get(name, SurfaceColorSource.MATERIAL),
            bool(has_color_buffer),
            type(obj_mat).__name__,
            bool(getattr(self, "_unlit_mode_enabled", False)),
            bool(getattr(self, "_ibl_loaded", False)),
            tuple(getattr(self, "_clipping_planes", ()) or ()),
        )

    def set_named_material(self, name: str, material: MaterialPayload | dict[str, Any]) -> bool:
        """Apply one renderer-neutral material to an existing pygfx object.

        Unsupported native fields remain best-effort, but the applied-material
        cache advances only after every supported required field succeeds. A
        failed native update therefore remains retryable.
        """
        obj = self._objects.get(name)
        if obj is None:
            return False
        material_apply_signatures = getattr(self, "_material_apply_signatures", None)
        if material_apply_signatures is None:
            material_apply_signatures = {}
            self._material_apply_signatures = material_apply_signatures

        t_total_start = time.perf_counter()
        t_coerce_start = t_total_start
        requested_mat = self._coerce_material(material)
        mat = apply_texture_policy_to_material_payload(requested_mat, context=name)
        if self._kinds.get(name) == "mesh" and not self._geometry_has_uvs(name):
            mat = replace(
                mat,
                base_color=(
                    float(requested_mat.base_color[0]),
                    float(requested_mat.base_color[1]),
                    float(requested_mat.base_color[2]),
                    float(mat.base_color[3]),
                ),
                texture_path=None,
                normal_map_path=None,
                roughness_map_path=None,
                ao_map_path=None,
                metallic_map_path=None,
            )
        if not bool(getattr(self, "_rf_xray_applying_overlay", False)) and name in getattr(
            self, "_rf_xray_overlaid_names", set()
        ):
            base_materials = getattr(self, "_rf_xray_base_materials", None)
            if isinstance(base_materials, dict):
                base_materials[name] = mat
                return True
        self._record_frame_update_metric(
            "set_named_material_coerce_ms",
            (time.perf_counter() - t_coerce_start) * 1000.0,
        )
        obj_mat = getattr(obj, "material", None)
        if obj_mat is None:
            material_apply_signatures.pop(name, None)
            return False
        has_color_buffer = self._object_has_color_buffer(obj)
        signature = self._material_apply_signature(
            name,
            mat,
            obj_mat,
            has_color_buffer=has_color_buffer,
        )
        if self._kinds.get(name) == "orientation_frame":
            ok = self._apply_orientation_frame_material(name, obj, mat)
            if ok:
                self._materials[name] = mat
                material_apply_signatures[name] = signature
                self._record_frame_update_metric("set_named_material_texture_bind_ms", 0.0)
                self._record_frame_update_metric(
                    "set_named_material_total_ms",
                    (time.perf_counter() - t_total_start) * 1000.0,
                )
                self.request_redraw()
                return True

            material_apply_signatures.pop(name, None)
            self._record_frame_update_metric("set_named_material_texture_bind_ms", 0.0)
            self._record_frame_update_metric(
                "set_named_material_total_ms",
                (time.perf_counter() - t_total_start) * 1000.0,
            )
            return False

        if material_apply_signatures.get(name) == signature:
            # The signature is written only after a successful native update,
            # so it is safe to repair a missing/stale applied-material cache.
            self._materials[name] = mat
            self._record_frame_update_metric("set_named_material_texture_bind_ms", 0.0)
            self._record_frame_update_metric(
                "set_named_material_total_ms",
                (time.perf_counter() - t_total_start) * 1000.0,
            )
            return True

        gfx = self._gfx

        # Unlit mode: swap to MeshBasicMaterial so base_color displays
        # unmodulated (chip = displayed color), matching Open3D's
        # defaultUnlit behavior. The env var seeds startup state; the Render
        # panel toggles this at runtime through set_color_fidelity_mode().
        basic_cls = getattr(gfx, "MeshBasicMaterial", None)
        physical_cls = getattr(gfx, "MeshPhysicalMaterial", None)
        standard_cls = getattr(gfx, "MeshStandardMaterial", None)
        unlit_mode_enabled = bool(
            getattr(self, "_unlit_mode_enabled", _is_pygfx_unlit_mode_enabled())
        )
        payload_requests_unlit = str(mat.shader).strip().lower() == "unlit"
        is_mesh = self._kinds.get(name) == "mesh"
        is_exact_basic = basic_cls is not None and type(obj_mat) is basic_cls
        use_unlit_material = is_mesh and (unlit_mode_enabled or payload_requests_unlit)
        if use_unlit_material and basic_cls is not None and not is_exact_basic:
            try:
                new_mat = basic_cls(color=mat.base_color)
                obj.material = new_mat
                obj_mat = new_mat
            except Exception as exc:
                logger.debug(
                    "PygfxRenderer: MeshBasicMaterial swap failed for '%s': %s",
                    name,
                    exc,
                )
                material_apply_signatures.pop(name, None)
                return False
        elif not use_unlit_material and basic_cls is not None and is_exact_basic and is_mesh:
            try:
                if mat.has_advanced_pbr and physical_cls is not None:
                    new_mat = physical_cls(pick_write=True)
                elif standard_cls is not None:
                    new_mat = standard_cls(pick_write=True)
                else:
                    new_mat = gfx.MeshPhongMaterial(color=mat.base_color, shininess=20)
                obj.material = new_mat
                obj_mat = new_mat
            except Exception as exc:
                logger.debug(
                    "PygfxRenderer: lit material restore failed for '%s': %s",
                    name,
                    exc,
                )
                material_apply_signatures.pop(name, None)
                return False

        # If a Wave A slider promoted this mesh to MeshPhysicalMaterial, return
        # to the fast MeshStandardMaterial path once those advanced fields are
        # neutral again. Otherwise a clearcoat 0 -> N -> 0 round-trip keeps a
        # different shader than the initial scene build.
        if (
            not unlit_mode_enabled
            and not mat.has_advanced_pbr
            and physical_cls is not None
            and isinstance(obj_mat, physical_cls)
            and is_mesh
        ):
            try:
                if standard_cls is not None:
                    new_mat = standard_cls(pick_write=True)
                else:
                    new_mat = gfx.MeshPhongMaterial(color=mat.base_color, shininess=20)
                obj.material = new_mat
                obj_mat = new_mat
            except Exception as exc:
                logger.debug(
                    "PygfxRenderer: MeshStandardMaterial downgrade failed for '%s': %s",
                    name,
                    exc,
                )
                material_apply_signatures.pop(name, None)
                return False

        # If the payload requests advanced PBR (Wave A) and the current
        # material is not already MeshPhysicalMaterial, swap it. This is the
        # only place that incurs the heavier-shader cost; meshes with default
        # advanced PBR keep MeshStandardMaterial for the fast path.
        if (
            not unlit_mode_enabled
            and mat.has_advanced_pbr
            and physical_cls is not None
            and not isinstance(obj_mat, physical_cls)
            and is_mesh
        ):
            try:
                new_mat = physical_cls(pick_write=True)
                obj.material = new_mat
                obj_mat = new_mat
            except Exception as exc:
                logger.debug(
                    "PygfxRenderer: MeshPhysicalMaterial swap failed for '%s': %s",
                    name,
                    exc,
                )
                material_apply_signatures.pop(name, None)
                return False

        # Open3D-only fields that pygfx cannot currently apply on the tested
        # 0.16.0 / 0.31.0 stack:
        #  - transmission / glass_thickness: no equivalent in MeshPhysicalMaterial
        #  - anisotropy: upstream WGSL bug (getTangentFrame missing without normal map)
        if mat.transmission > 0.0 or mat.glass_thickness > 0.0 or mat.anisotropy > 0.0:
            logger.debug(
                "PygfxRenderer: dropping Open3D-only PBR (transmission/"
                "glass_thickness/anisotropy) for '%s'",
                name,
            )

        r, g, b, a = self._normalize_rgba(mat.base_color)
        multiplier = tuple(float(value) for value in mat.color_multiplier)
        r, g, b = r * multiplier[0], g * multiplier[1], b * multiplier[2]
        mesh_uses_vertex_colors = bool(
            self._geometry_color_sources.get(name) is SurfaceColorSource.VERTEX and has_color_buffer
        )
        texture_bind_ms = 0.0
        try:
            # Preserve vertex colors for colored point/line overlays. Without
            # this, material updates turn sensing detections/tracks white even
            # though their payloads carry per-vertex colors.
            if hasattr(obj_mat, "color_mode"):
                try:
                    if mesh_uses_vertex_colors or (
                        self._kinds.get(name) in {"points", "lines"} and has_color_buffer
                    ):
                        obj_mat.color_mode = "vertex"
                    elif self._kinds.get(name) == "mesh" and mat.texture_path is not None:
                        obj_mat.color_mode = "auto"
                    elif self._kinds.get(name) == "mesh":
                        obj_mat.color_mode = "uniform"
                    else:
                        obj_mat.color_mode = "auto"
                except (AttributeError, ValueError):
                    pass
            if hasattr(obj_mat, "color"):
                obj_mat.color = (r, g, b, a)
            if hasattr(obj_mat, "opacity"):
                obj_mat.opacity = a
            if self._apply_material_alpha_state(name, obj_mat, mat) is False:
                raise RuntimeError("required alpha/depth material state was rejected")
            if hasattr(obj_mat, "roughness"):
                obj_mat.roughness = max(0.0, min(1.0, mat.roughness))
            reflectance = max(0.0, min(1.0, float(mat.reflectance)))
            for attr in ("reflectivity", "specular_intensity"):
                if hasattr(obj_mat, attr):
                    try:
                        setattr(obj_mat, attr, reflectance)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
            # With IBL active, full metallic range is safe because surfaces have
            # environment reflections. Without IBL, clamp to 0.9 to avoid black.
            max_metallic = 1.0 if self._ibl_loaded else 0.9
            metallic_clamped = max(0.0, min(max_metallic, mat.metallic))
            if hasattr(obj_mat, "metalness"):
                obj_mat.metalness = metallic_clamped
            if hasattr(obj_mat, "metallic"):
                obj_mat.metallic = metallic_clamped
            # Advanced PBR is only meaningful on MeshPhysicalMaterial.
            # Anisotropy is intentionally skipped on pygfx: the tested 0.16.0
            # stack can compile an invalid WGSL shader when anisotropy > 0
            # and no normal map is bound. The
            # Open3D renderer applies anisotropy via base_anisotropy as usual.
            if hasattr(obj_mat, "clearcoat"):
                obj_mat.clearcoat = max(0.0, min(1.0, mat.clearcoat))
            if hasattr(obj_mat, "clearcoat_roughness"):
                obj_mat.clearcoat_roughness = max(0.0, min(1.0, mat.clearcoat_roughness))
            if hasattr(obj_mat, "emissive"):
                er, eg, eb = mat.emissive_color
                try:
                    obj_mat.emissive = (er, eg, eb, 1.0)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            if hasattr(obj_mat, "emissive_intensity"):
                try:
                    obj_mat.emissive_intensity = max(0.0, mat.emissive_intensity)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            # ``thickness`` here refers to the **line thickness** for line
            # materials. pygfx's MeshPhysicalMaterial still has no equivalent
            # of Filament's volumetric glass thickness on the tested stack, so
            # this branch only fires for line/segment materials.
            if (
                hasattr(obj_mat, "thickness")
                and mat.line_width is not None
                and self._kinds.get(name) != "mesh"
            ):
                obj_mat.thickness = float(mat.line_width)
            if hasattr(obj_mat, "size") and mat.point_size is not None:
                obj_mat.size = float(mat.point_size)
            # Apply albedo texture if provided; explicitly clear when the
            # payload has no path so stale textures from a previous material
            # don't persist.  Without the explicit None-assignment, a mesh
            # that was textured on one frame keeps sampling that texture
            # forever even after the material is updated with texture_path=None,
            # which manifests as the mesh rendering fully invisible when the
            # underlying geometry's UVs no longer match after a mesh switch.
            if hasattr(obj_mat, "map"):
                tex = None
                if mat.texture_path is not None:
                    t_tex_start = time.perf_counter()
                    tex = self._load_texture_binding(
                        mat.texture_path,
                        uv_repeat_scale=mat.uv_repeat_scale,
                    )
                    texture_bind_ms += (time.perf_counter() - t_tex_start) * 1000.0
                obj_mat.map = tex
            # Tier 0 item 4 — PBR texture maps (parity across both renderers).
            # pygfx MeshStandardMaterial / MeshPhysicalMaterial accept
            # normal_map, roughness_map, ao_map, metalness_map.  Also cleared
            # explicitly on None-assignment to match the albedo path.
            if hasattr(obj_mat, "normal_map"):
                tex = None
                if mat.normal_map_path is not None:
                    t_tex_start = time.perf_counter()
                    tex = self._load_texture_binding(
                        mat.normal_map_path,
                        uv_repeat_scale=mat.uv_repeat_scale,
                    )
                    texture_bind_ms += (time.perf_counter() - t_tex_start) * 1000.0
                obj_mat.normal_map = tex
                if hasattr(obj_mat, "normal_scale"):
                    normal_strength = (
                        max(0.0, min(4.0, float(mat.normal_map_strength)))
                        if tex is not None
                        else 1.0
                    )
                    obj_mat.normal_scale = (normal_strength, normal_strength)
            if hasattr(obj_mat, "roughness_map"):
                tex = None
                if mat.roughness_map_path is not None:
                    t_tex_start = time.perf_counter()
                    tex = self._load_texture_binding(
                        mat.roughness_map_path,
                        uv_repeat_scale=mat.uv_repeat_scale,
                    )
                    texture_bind_ms += (time.perf_counter() - t_tex_start) * 1000.0
                obj_mat.roughness_map = tex
            if hasattr(obj_mat, "ao_map"):
                tex = None
                if mat.ao_map_path is not None:
                    t_tex_start = time.perf_counter()
                    tex = self._load_texture_binding(
                        mat.ao_map_path,
                        uv_repeat_scale=mat.uv_repeat_scale,
                    )
                    texture_bind_ms += (time.perf_counter() - t_tex_start) * 1000.0
                obj_mat.ao_map = tex
            if hasattr(obj_mat, "metalness_map"):
                tex = None
                if mat.metallic_map_path is not None:
                    t_tex_start = time.perf_counter()
                    tex = self._load_texture_binding(
                        mat.metallic_map_path,
                        uv_repeat_scale=mat.uv_repeat_scale,
                    )
                    texture_bind_ms += (time.perf_counter() - t_tex_start) * 1000.0
                obj_mat.metalness_map = tex
            if self._ibl_loaded:
                self._ibl_manager.apply_to_material(obj_mat)
            self._apply_clipping_to_material(obj_mat)
            applied_signature = self._material_apply_signature(
                name,
                mat,
                getattr(obj, "material", obj_mat),
                has_color_buffer=has_color_buffer,
            )
            material_apply_signatures[name] = applied_signature
            self._materials[name] = mat
            self._record_frame_update_metric("set_named_material_texture_bind_ms", texture_bind_ms)
            self._record_frame_update_metric(
                "set_named_material_total_ms",
                (time.perf_counter() - t_total_start) * 1000.0,
            )
            self.request_redraw()
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._record_frame_update_metric("set_named_material_texture_bind_ms", texture_bind_ms)
            self._record_frame_update_metric(
                "set_named_material_total_ms",
                (time.perf_counter() - t_total_start) * 1000.0,
            )
            material_apply_signatures.pop(name, None)
            logger.debug("PygfxRenderer: set_named_material failed for '%s': %s", name, exc)
            return False

    def _apply_orientation_frame_material(
        self,
        name: str,
        obj: Any,
        material: MaterialPayload,
    ) -> bool:
        """Apply non-color material state to a native pygfx axis helper.

        ``gfx.AxesHelper`` stores its shaft colors in the line geometry and
        its arrow colors on child meshes. Applying ORCHAV's neutral white
        material as a generic object material turns the shafts white, so this
        path preserves vertex-color mode and only syncs state that does not
        replace the RGB axes.
        """
        obj_mat = getattr(obj, "material", None)
        if obj_mat is None:
            return False

        try:
            if hasattr(obj_mat, "color_mode"):
                obj_mat.color_mode = "vertex"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        try:
            if hasattr(obj_mat, "thickness") and material.line_width is not None:
                obj_mat.thickness = float(material.line_width)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        try:
            if hasattr(obj_mat, "pick_write"):
                obj_mat.pick_write = False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        try:
            alpha = self._normalize_rgba(material.base_color)[3]
            if hasattr(obj_mat, "opacity"):
                obj_mat.opacity = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        self._apply_clipping_to_material(obj_mat)
        return True

    @staticmethod
    def _material_requests_transparency(material: MaterialPayload) -> bool:
        """Return whether alpha or shader name requests transparent rendering."""
        shader = str(material.shader).lower()
        return float(material.base_color[3]) < 0.999 or "transparen" in shader

    def _apply_material_alpha_state(
        self,
        name: str,
        obj_mat: Any,
        material: MaterialPayload,
    ) -> bool:
        """Map transparency intent and report supported-property failures."""
        if self._kinds.get(name) != "mesh":
            return True

        transparent = self._material_requests_transparency(material)
        alpha_mode = "weighted_blend" if transparent else "auto"
        ok = True

        for attr, value in (
            ("alpha_mode", alpha_mode),
            ("depth_write", not transparent),
            ("depth_test", True),
        ):
            try:
                if not hasattr(obj_mat, attr):
                    continue
                setattr(obj_mat, attr, value)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                ok = False
                logger.debug(
                    "PygfxRenderer: material property '%s' failed for '%s': %s",
                    attr,
                    name,
                    exc,
                )
        return ok

    def _apply_coverage_material_state(self, alpha: float, *, request_redraw: bool) -> bool:
        """Apply pygfx coverage overlay material settings without losing vertex colors."""
        obj = self._objects.get(self.COVERAGE_MESH_NAME)
        if obj is None:
            return False

        alpha = max(0.0, min(1.0, float(alpha)))
        mat = getattr(obj, "material", None)
        if mat is None:
            return False

        gfx = self._gfx
        basic_cls = getattr(gfx, "MeshBasicMaterial", None)
        if basic_cls is not None and mat.__class__ is not basic_cls:
            next_mat = None
            kwargs: dict[str, Any] = {"color": (1.0, 1.0, 1.0, alpha)}
            try:
                kwargs["color_mode"] = "vertex"
                next_mat = basic_cls(**kwargs)
            except TypeError:
                kwargs.pop("color_mode", None)
                try:
                    next_mat = basic_cls(**kwargs)
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug(
                        "PygfxRenderer: coverage MeshBasicMaterial fallback failed: %s",
                        exc,
                    )
            except (AttributeError, RuntimeError, ValueError) as exc:
                logger.debug("PygfxRenderer: coverage MeshBasicMaterial swap failed: %s", exc)
            if next_mat is None:
                return False
            try:
                obj.material = next_mat
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("PygfxRenderer: coverage material assignment failed: %s", exc)
                return False
            mat = next_mat

        for attr, value in (
            ("color_mode", "vertex"),
            ("side", "both"),
            ("opacity", alpha),
            ("color", (1.0, 1.0, 1.0, alpha)),
            ("env_map", None),
            ("env_map_intensity", 0.0),
        ):
            try:
                if not hasattr(mat, attr):
                    continue
                setattr(mat, attr, value)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "PygfxRenderer: coverage material property '%s' failed: %s",
                    attr,
                    exc,
                )
                return False

        tracked_materials = getattr(getattr(self, "_ibl_manager", None), "_tracked_materials", None)
        if tracked_materials is not None:
            try:
                tracked_materials[:] = [
                    tracked for tracked in tracked_materials if tracked is not mat
                ]
            except (AttributeError, TypeError, ValueError):
                pass

        payload = MaterialPayload(
            base_color=(1.0, 1.0, 1.0, alpha),
            roughness=0.8,
            metallic=0.0,
            shader="transparent" if alpha < 0.999 else "unlit",
        )
        if self._apply_material_alpha_state(self.COVERAGE_MESH_NAME, mat, payload) is False:
            return False
        self._apply_clipping_to_material(mat)

        if request_redraw:
            self.request_redraw()
        return True

    def _load_texture_cached(self, texture_path: str) -> Any:
        """Return a renderer-native texture backed by shared decoded pixels."""
        from ...materials.texture_assets import load_decoded_texture, texture_asset_identity

        identity_result = texture_asset_identity(texture_path)
        if identity_result is not None:
            identity, resolved_path = identity_result
            cached = self._texture_cache.get(identity)
            if cached is not None:
                return cached
            texture_path = str(resolved_path)

        asset = load_decoded_texture(texture_path)
        if asset is None:
            return None
        cached = self._texture_cache.get(asset.identity)
        if cached is not None:
            return cached

        previous_identity = self._texture_source_identities.get(asset.path)
        if previous_identity is not None and previous_identity != asset.identity:
            self._texture_cache.pop(previous_identity, None)
        try:
            tex = self._gfx.Texture(np.array(asset.rgba, copy=True), dim=2)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.debug("PygfxRenderer: failed to create texture %s: %s", asset.path, exc)
            return None
        self._texture_cache[asset.identity] = tex
        self._texture_source_identities[asset.path] = asset.identity
        logger.debug(
            "PygfxRenderer: created native texture %s (%dx%d)",
            asset.path,
            asset.rgba.shape[1],
            asset.rgba.shape[0],
        )
        return tex

    @staticmethod
    def _normalize_uv_repeat_scale(
        uv_repeat_scale: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        """Normalize texture repeat values for pygfx TextureMap scale."""
        if uv_repeat_scale is None:
            return None
        try:
            sx = float(uv_repeat_scale[0])
            sy = float(uv_repeat_scale[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not np.isfinite(sx) or not np.isfinite(sy):
            return None
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return None
        return (sx, sy)

    def _load_texture_binding(
        self,
        texture_path: str,
        *,
        uv_repeat_scale: Optional[tuple[float, float]] = None,
    ) -> Any:
        """Return either a cached texture or a scaled TextureMap wrapper."""
        tex = self._load_texture_cached(texture_path)
        if tex is None:
            return None
        repeat_scale = self._normalize_uv_repeat_scale(uv_repeat_scale)
        if repeat_scale is None:
            return tex
        texture_map_cls = getattr(self._gfx, "TextureMap", None)
        if texture_map_cls is None:
            return tex
        try:
            texture_map = texture_map_cls(tex)
            if hasattr(texture_map, "scale"):
                texture_map.scale = repeat_scale
            if hasattr(texture_map, "update_matrix"):
                texture_map.update_matrix()
            return texture_map
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "PygfxRenderer: failed to create TextureMap for %s: %s",
                texture_path,
                exc,
            )
            return tex

    def _apply_clipping_to_material(self, mat: Any) -> None:
        """Push the current cutaway planes onto *mat* if it supports clipping.

        No-op when no planes are active or the material lacks the attribute.
        Called from every material builder and from set_clipping_planes()
        when planes change.
        """
        if mat is None or not hasattr(mat, "clipping_planes"):
            return
        try:
            mat.clipping_planes = list(self._clipping_planes)
        except Exception as exc:
            logger.debug("Failed to apply clipping planes to material: %s", exc)

    def set_clipping_planes(self, planes: tuple[tuple[float, float, float, float], ...]) -> None:
        """Set cutaway clipping planes on every tracked material.

        Each plane is an ``(nx, ny, nz, d)`` tuple interpreted by pygfx as
        keeping the half-space ``dot(world_pos, (nx, ny, nz)) >= d``.
        Passing ``()`` disables clipping.
        """
        self._clipping_planes = tuple(tuple(float(v) for v in plane) for plane in planes)
        for obj in self._objects.values():
            mat = getattr(obj, "material", None)
            if mat is not None:
                self._apply_clipping_to_material(mat)
        self._update_status_chip_overlay()
        self.request_redraw()

    def _build_mesh_material(self, has_vertex_colors: bool = False) -> Any:
        """Build the default mesh material before any payload material is applied."""
        gfx = self._gfx
        kwargs: dict[str, Any] = {
            "color": (0.8, 0.8, 0.8, 1.0),
            "roughness": 0.6,
            "metalness": 0.0,
            "pick_write": True,
        }
        if has_vertex_colors:
            kwargs["color_mode"] = "vertex"
        try:
            mat = gfx.MeshStandardMaterial(**kwargs)
        except TypeError:
            kwargs.pop("color_mode", None)
            try:
                mat = gfx.MeshStandardMaterial(**kwargs)
            except Exception:
                mat = gfx.MeshPhongMaterial(color=(0.8, 0.8, 0.8, 1.0), shininess=20)
                self._apply_clipping_to_material(mat)
                return mat
        except Exception:
            mat = gfx.MeshPhongMaterial(color=(0.8, 0.8, 0.8, 1.0), shininess=20)
            self._apply_clipping_to_material(mat)
            return mat
        if self._ibl_loaded:
            self._ibl_manager.apply_to_material(mat)
        if self._wireframe_enabled and hasattr(mat, "wireframe"):
            mat.wireframe = True
        self._apply_clipping_to_material(mat)
        return mat

    def _build_line_material(
        self,
        has_vertex_colors: bool = False,
        *,
        line_strip: bool = False,
        pick_write: bool = True,
    ) -> Any:
        """Build a line or line-segment material for neutral line payloads."""
        gfx = self._gfx
        kwargs: dict[str, Any] = {
            "color": (0.6, 0.6, 0.6, 1.0),
            "thickness": 2.0,
            "pick_write": bool(pick_write),
        }
        if has_vertex_colors:
            kwargs["color_mode"] = "vertex"
        if line_strip and hasattr(gfx, "LineMaterial"):
            try:
                mat = gfx.LineMaterial(**kwargs)
            except TypeError:
                kwargs.pop("color_mode", None)
                mat = gfx.LineMaterial(**kwargs)
            self._apply_clipping_to_material(mat)
            return mat

        try:
            mat = gfx.LineSegmentMaterial(**kwargs)
        except (TypeError, Exception):
            # Older pygfx may not support color_mode; fall back without it.
            kwargs.pop("color_mode", None)
            try:
                mat = gfx.LineSegmentMaterial(**kwargs)
            except Exception:
                mat = gfx.LineMaterial(**kwargs)
        self._apply_clipping_to_material(mat)
        return mat

    def _build_points_material(self, has_vertex_colors: bool = False) -> Any:
        """Build the default point material for point-cloud payloads."""
        gfx = self._gfx
        kwargs: dict[str, Any] = {
            "color": (0.9, 0.9, 0.9, 1.0),
            "size": 5.0,
            "pick_write": True,
        }
        if has_vertex_colors:
            kwargs["color_mode"] = "vertex"
        try:
            mat = gfx.PointsMaterial(**kwargs)
        except TypeError:
            kwargs.pop("color_mode", None)
            mat = gfx.PointsMaterial(**kwargs)
        self._apply_clipping_to_material(mat)
        return mat

    @staticmethod
    def _object_has_color_buffer(obj: Any) -> bool:
        """Return whether a pygfx object has usable vertex color data."""
        geom = getattr(obj, "geometry", None)
        colors = getattr(geom, "colors", None)
        data = getattr(colors, "data", None)
        if data is None:
            return colors is not None
        try:
            return int(np.asarray(data).size) > 0
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _sync_color_mode(obj: Any, has_colors: bool) -> None:
        """Ensure the material's color_mode matches whether vertex colors are present."""
        mat = getattr(obj, "material", None)
        if mat is None or not hasattr(mat, "color_mode"):
            return
        desired = "vertex" if has_colors else "auto"
        if mat.color_mode != desired:
            try:
                mat.color_mode = desired
            except (AttributeError, ValueError):
                pass

    def _coerce_material(self, material: MaterialPayload | dict[str, Any]) -> MaterialPayload:
        """Normalize dict or payload material input to ``MaterialPayload``."""
        if isinstance(material, MaterialPayload):
            return MaterialPayload(
                base_color=self._normalize_rgba(material.base_color),
                color_multiplier=tuple(material.color_multiplier),  # type: ignore[arg-type]
                roughness=float(material.roughness),
                metallic=float(material.metallic),
                reflectance=float(material.reflectance),
                shader=str(material.shader),
                line_width=material.line_width,
                point_size=material.point_size,
                texture_path=material.texture_path,
                clearcoat=float(material.clearcoat),
                clearcoat_roughness=float(material.clearcoat_roughness),
                anisotropy=float(material.anisotropy),
                emissive_color=tuple(material.emissive_color),  # type: ignore[arg-type]
                emissive_intensity=float(material.emissive_intensity),
                transmission=float(material.transmission),
                glass_thickness=float(material.glass_thickness),
                absorption_color=tuple(material.absorption_color),  # type: ignore[arg-type]
                normal_map_path=material.normal_map_path,
                normal_map_strength=float(material.normal_map_strength),
                roughness_map_path=material.roughness_map_path,
                ao_map_path=material.ao_map_path,
                metallic_map_path=material.metallic_map_path,
                uv_scale_meters=float(material.uv_scale_meters),
                uv_repeat_scale=(
                    tuple(float(v) for v in material.uv_repeat_scale)
                    if material.uv_repeat_scale is not None
                    else None
                ),
                shader_variant=material.shader_variant,
            )

        color = material.get("base_color", material.get("color", [1.0, 1.0, 1.0, 1.0]))
        base_color = self._normalize_rgba(color, alpha=material.get("alpha"))

        emissive = material.get("emissive_color", (0.0, 0.0, 0.0))
        emissive_tuple: tuple[float, float, float] = (
            float(emissive[0]),
            float(emissive[1]),
            float(emissive[2]),
        )
        absorption = material.get("absorption_color", (1.0, 1.0, 1.0))
        absorption_tuple: tuple[float, float, float] = (
            float(absorption[0]),
            float(absorption[1]),
            float(absorption[2]),
        )

        def _opt_str(key: str) -> Optional[str]:
            """Return optional texture-map paths as strings for payload fields."""
            val = material.get(key)
            return str(val) if val is not None else None

        uv_repeat = material.get("uv_repeat_scale")
        uv_repeat_scale: Optional[tuple[float, float]] = None
        if uv_repeat is not None:
            try:
                uv_repeat_scale = (float(uv_repeat[0]), float(uv_repeat[1]))
            except (TypeError, ValueError, IndexError):
                uv_repeat_scale = None

        return MaterialPayload(
            base_color=base_color,
            roughness=float(material.get("roughness", 0.5)),
            metallic=float(material.get("metallic", 0.0)),
            reflectance=float(material.get("reflectance", 0.5)),
            shader=str(material.get("shader", "lit")),
            line_width=(
                float(material["line_width"]) if material.get("line_width") is not None else None
            ),
            point_size=(
                float(material["point_size"]) if material.get("point_size") is not None else None
            ),
            texture_path=_opt_str("texture_path"),
            clearcoat=float(material.get("clearcoat", 0.0)),
            clearcoat_roughness=float(material.get("clearcoat_roughness", 0.0)),
            anisotropy=float(material.get("anisotropy", 0.0)),
            emissive_color=emissive_tuple,
            emissive_intensity=float(material.get("emissive_intensity", 0.0)),
            transmission=float(material.get("transmission", 0.0)),
            glass_thickness=float(material.get("glass_thickness", 0.0)),
            absorption_color=absorption_tuple,
            normal_map_path=_opt_str("normal_map_path"),
            normal_map_strength=float(material.get("normal_map_strength", 1.0)),
            roughness_map_path=_opt_str("roughness_map_path"),
            ao_map_path=_opt_str("ao_map_path"),
            metallic_map_path=_opt_str("metallic_map_path"),
            uv_scale_meters=float(material.get("uv_scale_meters", 2.0)),
            uv_repeat_scale=uv_repeat_scale,
            shader_variant=_opt_str("shader_variant"),
        )

    def _iter_materials_by_kind(self, kind: str) -> list[Any]:
        """Return native pygfx materials for currently tracked objects of *kind*."""
        mats: list[Any] = []
        for name, obj_kind in self._kinds.items():
            if obj_kind != kind:
                continue
            obj = self._objects.get(name)
            if obj is None:
                continue
            mat = getattr(obj, "material", None)
            if mat is not None:
                mats.append(mat)
        return mats

    def modify_geometry_material_pbr(self, name: str, **kwargs: Any) -> bool:
        """Update one geometry from the Open3D-shaped PBR keyword API."""
        if name not in self._name_to_handle:
            return False
        # Drop texture-map paths when the geometry has no UV coordinates.
        # Without UVs the shader samples undefined texcoords and the entity can
        # render fully invisible after a same-layout geometry buffer update.
        mesh_has_uvs = self._geometry_has_uvs(name)
        tex_path = kwargs.get("texture_path") if mesh_has_uvs else None
        normal_path = kwargs.get("normal_map_path") if mesh_has_uvs else None
        roughness_path = kwargs.get("roughness_map_path") if mesh_has_uvs else None
        ao_path = kwargs.get("ao_map_path") if mesh_has_uvs else None
        metallic_path = kwargs.get("metallic_map_path") if mesh_has_uvs else None
        mat_payload = self._coerce_material(
            {
                "color": kwargs.get("color", [1.0, 1.0, 1.0]),
                "alpha": kwargs.get("alpha", 1.0),
                "roughness": kwargs.get("roughness", 0.5),
                "metallic": kwargs.get("metallic", 0.0),
                "reflectance": kwargs.get("reflectance", 0.5),
                # Wave A — parity advanced PBR
                "clearcoat": kwargs.get("clearcoat", 0.0),
                "clearcoat_roughness": kwargs.get("clearcoat_roughness", 0.0),
                "anisotropy": kwargs.get("anisotropy", 0.0),
                "emissive_color": kwargs.get("emissive_color", (0.0, 0.0, 0.0)),
                "emissive_intensity": kwargs.get("emissive_intensity", 0.0),
                # Wave B — discarded by pygfx, accepted for parity dispatch
                "transmission": kwargs.get("transmission", 0.0),
                "glass_thickness": kwargs.get("glass_thickness", 0.0),
                "absorption_color": kwargs.get("absorption_color", (1.0, 1.0, 1.0)),
                # Tier 0 item 4 — PBR texture maps + SSR shader variant
                "texture_path": tex_path,
                "normal_map_path": normal_path,
                "normal_map_strength": kwargs.get("normal_map_strength", 1.0),
                "roughness_map_path": roughness_path,
                "ao_map_path": ao_path,
                "metallic_map_path": metallic_path,
                "uv_scale_meters": kwargs.get("uv_scale_meters", 2.0),
                "uv_repeat_scale": kwargs.get("uv_repeat_scale"),
                "shader_variant": kwargs.get("shader_variant"),
            }
        )
        return self.set_named_material(name, mat_payload)

    def _geometry_has_uvs(self, name: str) -> bool:
        """Return whether the named gfx.Geometry has a texcoords buffer."""
        tracked = getattr(self, "_geometry_texcoords_available", {}).get(name)
        if tracked is not None:
            return bool(tracked)
        obj = self._objects.get(name)
        if obj is None:
            return False
        geom = getattr(obj, "geometry", None)
        if geom is None:
            return False
        texcoords = getattr(geom, "texcoords", None)
        return texcoords is not None

    def set_coverage_transparency(self, alpha: float) -> bool:
        """Update coverage mesh opacity without re-uploading geometry."""
        return self._apply_coverage_material_state(alpha, request_redraw=True)

    def set_geometry_transparency(
        self,
        geometry: Any,
        alpha: float,
        color: Any = None,
        defer_redraw: bool = False,
        roughness: float = 0.5,
        metallic: float = 0.0,
        reflectance: float = 0.5,
    ) -> bool:
        """Apply alpha to a named or external geometry without changing geometry."""
        if isinstance(geometry, str):
            name = geometry
        else:
            name = self._external_name_for_geometry(geometry)
        if name is None or name not in self._name_to_handle:
            return False

        current = self._materials.get(name)
        if color is None and current is not None:
            base = current.base_color
        else:
            base = color if color is not None else (1.0, 1.0, 1.0, 1.0)
        r, g, b, _ = self._normalize_rgba(base)
        mat_payload = MaterialPayload(
            base_color=(r, g, b, float(alpha)),
            roughness=float(roughness),
            metallic=float(metallic),
            reflectance=float(reflectance),
            shader=current.shader if current is not None else "lit",
            line_width=current.line_width if current is not None else None,
            point_size=current.point_size if current is not None else None,
            texture_path=current.texture_path if current is not None else None,
        )
        ok = self.set_named_material(name, mat_payload)
        if ok and not defer_redraw:
            self.request_redraw()
        return ok

    @staticmethod
    def _normalize_rgba(
        color: Any, alpha: Optional[float] = None
    ) -> tuple[float, float, float, float]:
        """Coerce color-like input and optional alpha to an RGBA tuple."""
        arr = np.asarray(color, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            a = float(arr[3]) if alpha is None else float(alpha)
            return (float(arr[0]), float(arr[1]), float(arr[2]), a)
        if arr.size == 3:
            a = 1.0 if alpha is None else float(alpha)
            return (float(arr[0]), float(arr[1]), float(arr[2]), a)
        if arr.size == 1:
            v = float(arr[0])
            a = 1.0 if alpha is None else float(alpha)
            return (v, v, v, a)
        a = 1.0 if alpha is None else float(alpha)
        return (1.0, 1.0, 1.0, a)
