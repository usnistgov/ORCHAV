"""Reversible, scenario-authored blue-hour presentation for native pygfx.

Only optical light/material properties change. Local lights are shadowless and
bounded; target lights inherit the existing target transform. No extra timer,
mesh uploads, RF geometry, frame reads, or background work are introduced.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.scenarios.model import DuskViewModel


@dataclass
class _Emission:
    material: object
    original: tuple
    texture_map: object
    target: bool
    applied_map: object = None
    progress: float = -1.0


def _smooth_phase(progress: float, start: float, end: float) -> float:
    """A bounded ramp with zero slope at the authored phase boundaries."""
    value = min(1.0, max(0.0, (progress - start) / (end - start)))
    return value * value * (3.0 - 2.0 * value)


def _sunset_phases(progress: float) -> tuple[float, float, float]:
    """Daylight/sky fades first, architectural lights next, vehicle lamps last."""
    return (
        _smooth_phase(progress, 0.0, 0.78),
        _smooth_phase(progress, 0.28, 0.82),
        _smooth_phase(progress, 0.62, 1.0),
    )


def _palette_pixels(target: dict) -> np.ndarray:
    """Create emission only in the explicitly authored source-material slots."""
    pixels = np.zeros((1, target["palette_slots"], 3), dtype=np.uint8)
    for index, color in target["emissive_slots"].items():
        pixels[0, int(index)] = np.round(np.asarray(color) * 255).astype(np.uint8)
    return pixels


def _window_pixels(albedo: np.ndarray) -> np.ndarray | None:
    """Light selected existing panes of the versioned Haussmann atlas.

    Selecting the exact pane color preserves mullions, balconies and cornices,
    instead of repainting rectangular windows over the authored surface. The
    atlas repeats bays; this is illustrative emission, not surveyed occupancy.
    """
    if albedo.shape != (4096, 1536, 3) and albedo.shape != (4096, 1536, 4):
        return None
    mask = np.all(albedo[:, :, :3] == (74, 92, 99), axis=2)
    # Roof, stone and source-color slots contain no window emission.
    mask[3072:] = False
    pixels = np.zeros((4096, 1536, 3), dtype=np.uint8)
    pixels[mask] = (255, 175, 74)
    return pixels


class PygfxDuskMixin:
    """Backend-local light recipe with exact restoration on disable/reset."""

    def configure_dusk(self, recipe: dict | None) -> bool:
        """Register a validated recipe without enabling it or touching lighting."""
        validated = None if recipe is None else DuskViewModel.model_validate(recipe).model_dump()
        self._clear_dusk()
        self._dusk_recipe = validated
        self._dusk_candidates = {}
        self._dusk_object_count = -1
        self._dusk_texture_cache = {}
        return validated is not None

    def get_dusk_state(self) -> dict:
        """Small UI/benchmark status; counts describe realized native lights."""
        bindings = getattr(self, "_dusk_target_bindings", {})
        return {
            "available": getattr(self, "_dusk_recipe", None) is not None,
            "enabled": bool(getattr(self, "_dusk_enabled", False)),
            "progress": float(getattr(self, "_dusk_progress", 0.0)),
            "label": (getattr(self, "_dusk_recipe", None) or {}).get("label", "Blue hour"),
            "facade_lights": len(getattr(self, "_dusk_facade_lights", ())),
            "headlights": len(bindings),
            "visible_headlights": sum(
                self._dusk_visible(source) for source, _ in bindings.values()
            ),
            "emissive_materials": len(getattr(self, "_dusk_emissions", {})),
        }

    def set_dusk_enabled(self, enabled: bool) -> bool:
        """Select the complete blue-hour look, or restore the original look."""
        if enabled:
            return self.set_dusk_progress(1.0)
        if getattr(self, "_dusk_enabled", False):
            self._clear_dusk()
        return True

    def set_dusk_progress(self, progress: float, *, request_redraw: bool = True) -> bool:
        """Set sunset phase without a timer or rebuilding native materials.

        Zero keeps the rig prepared while reproducing the original optical
        appearance. One is the complete blue-hour look. Playback can coalesce
        these uniform changes into its next scene draw by passing False.
        """
        progress = float(progress)
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("Sunset progress must be finite and between 0 and 1")
        prepared = False
        if not getattr(self, "_dusk_enabled", False):
            if not self._prepare_dusk():
                return False
            prepared = True
        if not prepared and progress == self._dusk_progress:
            return True
        self._dusk_progress = progress
        self._dusk_phases = _sunset_phases(progress)
        self._apply_dusk_progress()
        self._sync_dusk()
        if request_redraw:
            self._request_canvas_draw()
        return True

    def _prepare_dusk(self) -> bool:
        """Allocate the bounded rig once, leaving the original backdrop intact."""
        if not getattr(self, "_dusk_recipe", None) or self._scene is None:
            return False
        self._dusk_saved = {
            "lights": deepcopy(self.get_light_rig_state()),
            "ibl": self.get_ibl_intensity(),
            "clear_color": tuple(self._clear_color),
            "skybox": self.get_skybox_visible(),
            "backgrounds": tuple(
                (obj, bool(obj.visible))
                for obj in self._scene.children
                if isinstance(obj, self._gfx.Background)
            ),
            "colors": {
                name: tuple(getattr(self, name).color)
                for name in ("_ambient_light", "_key_light", "_fill_light")
                if getattr(self, name, None) is not None
            },
        }
        # Keep the original solid/HDR background resident. A second background
        # blends over it before opaque geometry, with no HDR reload, backdrop
        # replacement or shader change as its opacity moves from zero to one.
        self._dusk_background = self._gfx.Background.from_color(
            (0.19, 0.23, 0.34, 1), (0.035, 0.065, 0.16, 1)
        )
        self._dusk_background.material.render_queue = 1001
        self._dusk_background.material.opacity = 0.0
        self._scene.add(self._dusk_background)
        self._dusk_facade_lights = []
        self._dusk_target_bindings = {}
        self._dusk_emissions = {}
        for item in self._dusk_recipe["facade_lights"]:
            light = self._dusk_light(item)
            self._scene.add(light)
            self._dusk_facade_lights.append(light)
        self._dusk_enabled = True
        self._dusk_progress = 0.0
        self._dusk_phases = (0.0, 0.0, 0.0)
        self._dusk_object_count = -1
        return True

    def _apply_dusk_progress(self) -> None:
        """Update resident uniforms; leave shadow poses and render cadence alone."""
        daylight, architecture, headlights = self._dusk_phases
        saved = self._dusk_saved
        rig = saved["lights"]
        for field, destination in (
            ("ambient_intensity", 0.55),
            ("key_intensity", 0.42),
            ("fill_intensity", 0.22),
        ):
            setattr(self, f"_base_{field}", rig[field] * (1 - daylight) + destination * daylight)
        self._headlight_enabled = bool(rig["headlight_enabled"]) and daylight < 1
        self._headlight_intensity = rig["headlight_intensity"] * (1 - daylight)
        if daylight == 1:
            # Preserve the stored camera-light setting while its switch is off,
            # matching the instantaneous blue-hour endpoint.
            self._headlight_intensity = rig["headlight_intensity"]
        self._apply_light_intensities()
        for name, destination in (
            ("_ambient_light", (0.48, 0.62, 1.0, 1.0)),
            ("_key_light", (0.56, 0.69, 1.0, 1.0)),
            ("_fill_light", (0.31, 0.45, 0.83, 1.0)),
        ):
            if name in saved["colors"]:
                getattr(self, name).color = tuple(
                    value * (1 - daylight) + target * daylight
                    for value, target in zip(saved["colors"][name], destination, strict=True)
                )
        self._ibl_intensity = saved["ibl"] * (1 - daylight) + 700.0 * daylight
        self._ibl_manager.set_intensity(self._ibl_intensity / 30000.0)
        self._lighting_profile_name = (
            rig["lighting_profile"] if self._dusk_progress == 0 else "custom"
        )
        self._dusk_background.material.opacity = daylight
        # At the opaque endpoint the underlying backdrop contributes nothing.
        # Keep its object/texture resident, but avoid drawing it unnecessarily.
        for background, visible in saved["backgrounds"]:
            wanted = visible and daylight < 1.0
            if background.visible != wanted:
                background.visible = wanted
        for light, item in zip(
            self._dusk_facade_lights, self._dusk_recipe["facade_lights"], strict=True
        ):
            light.intensity = item["intensity"] * architecture
        for item in self._dusk_recipe["targets"]:
            binding = self._dusk_target_bindings.get(item["object_id"])
            if binding is not None:
                binding[1].intensity = item["light"]["intensity"] * headlights

    def _dusk_light(self, item: dict):
        light = self._gfx.SpotLight(
            color=item["color"],
            intensity=0.0,
            cast_shadow=False,
            distance=item["distance"],
            decay=0,
            angle=np.deg2rad(item["angle_deg"]),
            penumbra=0.65,
        )
        light.local.position = item["position"]
        # pygfx's target is a world-space object. For target-local headlights,
        # both light and target are parented to the moving source mesh.
        light.target.local.position = item["lookat"]
        return light

    @staticmethod
    def _dusk_visible(source) -> bool:
        while source is not None:
            if not source.visible:
                return False
            source = source.parent
        return True

    def _discover_dusk_materials(self) -> None:
        """Discover eligible atlas materials only on scene population changes."""
        candidates = {}
        targets = {item["object_id"]: item for item in self._dusk_recipe["targets"]}
        for name, payload in getattr(self, "_materials", {}).items():
            texture = str(getattr(payload, "texture_path", "") or "")
            if name in targets:
                candidates[name] = targets[name]
            elif self._dusk_recipe["window_accents"] and "haussmann-v6" in Path(texture).parts:
                candidates[name] = None
        self._dusk_candidates = candidates
        self._dusk_object_count = len(self._objects)

    def _sync_dusk(self) -> None:
        """Bind replacement materials/actors; off is a single boolean check."""
        if not getattr(self, "_dusk_enabled", False):
            return
        if len(self._objects) != self._dusk_object_count:
            self._discover_dusk_materials()
        for item in self._dusk_recipe["targets"]:
            name = item["object_id"]
            source = self._objects.get(name)
            previous = self._dusk_target_bindings.get(name)
            if previous is not None and source is not previous[0]:
                _, light = self._dusk_target_bindings.pop(name)
                for child in (light, light.target):
                    if child.parent is not None:
                        child.parent.remove(child)
            if (
                source is not None
                and source.parent is not None
                and name not in self._dusk_target_bindings
            ):
                light = self._dusk_light(item["light"])
                light.intensity = item["light"]["intensity"] * self._dusk_phases[2]
                source.add(light, light.target)
                self._dusk_target_bindings[name] = (source, light)
        live_stores = set()
        for name, target in self._dusk_candidates.items():
            source = self._objects.get(name)
            material = getattr(source, "material", None)
            if material is None or not hasattr(material, "emissive_map"):
                continue
            texture_map = getattr(material, "map", None)
            store_id = id(material._store)
            live_stores.add(store_id)
            emission = self._dusk_emissions.get(store_id)
            payload = getattr(self, "_materials", {}).get(name)
            atlas_active = (
                target is not None
                or "haussmann-v6" in Path(str(getattr(payload, "texture_path", "") or "")).parts
            )
            # Untextured/material-color/wireframe looks retain their intended
            # surface appearance. Existing materials can be reused across modes.
            if (
                texture_map is None
                or bool(getattr(material, "wireframe", False))
                or material.opacity < 1.0
                or material.color.a < 1.0
                or not material.depth_write
                or not atlas_active
            ):
                if emission is not None:
                    material.emissive, material.emissive_intensity, material.emissive_map = (
                        emission.original
                    )
                    del self._dusk_emissions[store_id]
                continue
            if emission is not None and material.emissive_map is emission.applied_map:
                self._apply_dusk_emission(emission)
                continue
            key = ("palette", target["object_id"]) if target else ("windows",)
            emission_map = self._dusk_texture_cache.get(key)
            if emission_map is None:
                pixels = (
                    _palette_pixels(target) if target else _window_pixels(texture_map.texture.data)
                )
                if pixels is None:
                    continue
                emission_map = self._gfx.TextureMap(
                    self._gfx.Texture(pixels, dim=2), filter="nearest" if target else "linear"
                )
                self._dusk_texture_cache[key] = emission_map
            original = (
                tuple(material.emissive),
                material.emissive_intensity,
                material.emissive_map,
            )
            emission = _Emission(material, original, emission_map, target is not None)
            self._dusk_emissions[store_id] = emission
            self._apply_dusk_emission(emission)
        # Sequence replacement must not retain every previously displayed
        # material. Restore and release retired stores immediately.
        for store_id in self._dusk_emissions.keys() - live_stores:
            emission = self._dusk_emissions.pop(store_id)
            material = emission.material
            material.emissive, material.emissive_intensity, material.emissive_map = (
                emission.original
            )

    def _apply_dusk_emission(self, emission: _Emission) -> None:
        """Fade prebound emission; preserve an already-emissive daytime material."""
        progress = self._dusk_progress
        if emission.progress == progress:
            return
        material = emission.material
        color, intensity, texture_map = emission.original
        existing_emission = intensity != 0 and any(value != 0 for value in color[:3])
        if existing_emission and progress <= 0.16:
            # Keep an authored daytime emitter exact at zero. Fade it to zero
            # before the one-time map handoff, which occurs before windows or
            # vehicle lamps start illuminating. No extra shader layer is needed.
            material.emissive = color
            material.emissive_intensity = intensity * (1 - _smooth_phase(progress, 0.0, 0.16))
            material.emissive_map = texture_map
        else:
            # Paris source emission is zero. Bind its authored map immediately
            # at preparation (intensity zero), so the first visible glow only
            # updates a uniform instead of compiling another shader mid-fade.
            material.emissive = (1, 1, 1)
            material.emissive_intensity = (
                1.2 * self._dusk_phases[2] if emission.target else 0.65 * self._dusk_phases[1]
            )
            material.emissive_map = emission.texture_map
        emission.applied_map = material.emissive_map
        emission.progress = progress

    def _clear_dusk(self, restore: bool = True) -> None:
        """Release all local lights and restore saved native material stores."""
        self._dusk_enabled = False
        self._dusk_progress = 0.0
        self._dusk_phases = (0.0, 0.0, 0.0)
        for emission in getattr(self, "_dusk_emissions", {}).values():
            material = emission.material
            material.emissive, material.emissive_intensity, material.emissive_map = (
                emission.original
            )
        self._dusk_emissions = {}
        for _, light in getattr(self, "_dusk_target_bindings", {}).values():
            for child in (light, light.target):
                if child.parent is not None:
                    child.parent.remove(child)
        self._dusk_target_bindings = {}
        for light in getattr(self, "_dusk_facade_lights", ()):
            if light.parent is not None:
                light.parent.remove(light)
        self._dusk_facade_lights = []
        background = getattr(self, "_dusk_background", None)
        if background is not None and background.parent is not None:
            background.parent.remove(background)
        self._dusk_background = None
        saved = getattr(self, "_dusk_saved", None)
        self._dusk_saved = None
        if restore and saved is not None:
            self.restore_light_rig_state(saved["lights"], ibl=saved["ibl"])
            for name, color in saved["colors"].items():
                getattr(self, name).color = color
            for background, visible in saved["backgrounds"]:
                background.visible = visible
            # The original solid/HDR backdrop has remained attached throughout.
            # Removing our gradient is sufficient; do not rebuild it on exit.
            self._request_canvas_draw()
