"""Apply and persist global pygfx appearance without touching experiment state.

Presets compose existing material, scene and lighting owners. They are explicit
GUI actions, never frame callbacks. Source PBR overrides remain independent.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..materials.appearance_recipes import appearance_recipe
from ..materials.texture_policy import textures_globally_enabled, textures_locked_off
from ..materials.view_styles import PRESET_ALIASES, PRESETS
from ..renderers.effect_settings import DEFAULT_BLOOM, validate_bloom
from ..renderers.protocol import renderer_capabilities
from ..scene.defaults import SCENE_BACKGROUND_PRESETS
from .mpc_appearance_service import MpcAppearanceService


class AppearancePresetService(QObject):
    """One owner for named global looks and optional workspace snapshots."""

    changed = Signal()

    def __init__(self, visualizer, directory: Path | None = None):
        super().__init__()
        self.viz = visualizer
        self.directory = directory or Path.home() / ".orchav" / "appearance"
        self.surface = "original"
        self.unlit = False
        self.applying = False
        self.active_name = "Custom"
        self.last_applied = None
        self.previous = None
        self.notice = ""

    def capture(self, *, include_mpc: bool = False) -> dict:
        """Read scene state, optionally including paths for legacy snapshots.

        Workspace and Demo owners capture paths separately; manual path edits
        remain independent of the active scene preset's Custom status.
        """
        v, r = self.viz, self.viz.renderer
        data = dict(
            version=1,
            surface=self.surface,
            unlit=self.unlit,
            textures=textures_globally_enabled(),
            wireframe=r.get_wireframe(),
            background=list(getattr(v, "current_background_color", (0.2, 0.2, 0.2))),
            alpha=float(getattr(v, "current_building_alpha", 1.0)),
            outlines=bool(getattr(v, "outlines_enabled", False)),
            edge_width=r.get_edge_line_width(),
            building_appearance=getattr(v, "building_appearance", "original"),
            facade_theme=getattr(v, "facade_theme", "mixed"),
            road_appearance=getattr(v, "road_appearance", "original"),
            authored_surface_textures=bool(getattr(v, "authored_surface_textures", True)),
            sky=r.get_ibl_name(),
            skybox=r.get_skybox_visible(),
            ibl=float(r.get_ibl_intensity() or 0),
            lights=r.get_light_rig_state(),
            aa=r.get_antialiasing_mode(),
            mipmaps=r.get_texture_mipmaps_enabled(),
            color_fidelity=r.get_color_fidelity_mode(),
            shadow_quality=r.get_shadow_quality(),
            shadow_radius=r.get_shadow_quality_state()["focus_radius"],
            shadow_bias=r.get_shadow_quality_state()["bias_texels"],
            depth_precision=r.get_depth_precision_enabled(),
            bloom=r.get_bloom_settings(),
            plan_view=bool(
                getattr(
                    getattr(getattr(v, "scene_service", None), "plan_view", None), "enabled", False
                )
            ),
        )
        mpc = getattr(v, "mpc_appearance_service", None)
        if include_mpc and mpc is not None:
            data["mpc_appearance"] = mpc.capture()
        return data

    @staticmethod
    def validate(data: dict) -> dict:
        """Reject malformed/custom future recipes before mutating the display."""
        from ..materials.scene_variation import THEMES

        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported appearance preset version")
        data = json.loads(json.dumps(data, allow_nan=False))
        if "mpc_appearance" in data:
            data["mpc_appearance"] = MpcAppearanceService.validate(data["mpc_appearance"])
        if "depth_precision" in data and not isinstance(data["depth_precision"], bool):
            raise ValueError("Invalid appearance flag: depth_precision")
        if "plan_view" in data and type(data["plan_view"]) is not bool:
            raise ValueError("Invalid appearance flag: plan_view")
        choices = dict(
            surface={"original", "geometry", "radio", "dark", "glow"},
            building_appearance={"original", "procedural", "varied"},
            road_appearance={"original", "marked"},
            facade_theme=set(THEMES),
            aa={"off", "fxaa", "ddaa", "ppaa"},
            shadow_quality={"standard", "scene", "focus"},
        )
        for key, values in choices.items():
            if not isinstance(data.get(key), str) or data[key] not in values:
                raise ValueError(f"Invalid appearance setting: {key}")
        if "bloom" in data:
            data["bloom"] = validate_bloom(data["bloom"])
        if "edge_width" in data:
            value = data["edge_width"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.1 <= value <= 1000
            ):
                raise ValueError("Invalid appearance edge width")
        # Saved looks keep AA enabled when their retired selection is restored.
        if data["aa"] in {"ddaa", "ppaa"}:
            data["aa"] = "fxaa"
        for key in (
            "unlit",
            "textures",
            "wireframe",
            "outlines",
            "authored_surface_textures",
            "skybox",
            "mipmaps",
            "color_fidelity",
        ):
            if not isinstance(data.get(key), bool):
                raise ValueError(f"Invalid appearance flag: {key}")
        for key, lo, hi in (
            ("alpha", 0, 1),
            ("ibl", 0, 100000),
            ("shadow_radius", 0.1, 1e5),
            ("shadow_bias", 0, 8),
        ):
            value = data.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not lo <= value <= hi
            ):
                raise ValueError(f"Invalid appearance value: {key}")
        bg = data.get("background")
        if (
            not isinstance(bg, list)
            or len(bg) != 3
            or any(not isinstance(x, (float, int)) or not 0 <= x <= 1 for x in bg)
        ):
            raise ValueError("Invalid appearance background")
        if data.get("sky") is not None and not isinstance(data["sky"], str):
            raise ValueError("Invalid appearance sky")
        lights = data.get("lights")
        if not isinstance(lights, dict):
            raise ValueError("Missing appearance light rig")
        for key in (
            "ambient_intensity",
            "headlight_intensity",
            "key_intensity",
            "fill_intensity",
            "key_azimuth_deg",
            "key_elevation_deg",
            "fill_azimuth_deg",
            "fill_elevation_deg",
        ):
            val = lights.get(key)
            if not isinstance(val, (float, int)) or not math.isfinite(val):
                raise ValueError(f"Invalid light value: {key}")
            if key.endswith("intensity") and not 0 <= val <= 100000:
                raise ValueError(f"Invalid light intensity: {key}")
        for key in ("headlight_enabled", "shadows_enabled"):
            if not isinstance(lights.get(key), bool):
                raise ValueError(f"Invalid light flag: {key}")
        return data

    recipe = staticmethod(appearance_recipe)

    def apply(self, name: str) -> None:
        """Apply a scene look, preserving the user's path settings."""
        name = PRESET_ALIASES.get(name, name)
        if name not in PRESETS:
            data = self.load(name)
            # Older saved looks included paths. Normal appearance selection is
            # now scene-only; direct snapshot restoration still accepts them.
            data.pop("mpc_appearance", None)
            self.restore(data, name=name)
            return
        data, notice = self.recipe(
            name,
            self.capture(),
            varied_buildings=any(e.get("_osm_building_parts") for e in self.viz.mesh_entries),
        )
        self.restore(data, name=name, matching_sky=name == "Detailed", notice=notice)

    def restore(self, data: dict, *, name: str = "Custom", matching_sky=False, notice="") -> None:
        """Replay a validated global look, batching renderer updates."""
        data = self.validate(data)
        if data["textures"] and textures_locked_off():
            raise ValueError(
                "Textures are locked off by --disable-textures. Relaunch without that explicit lock to use this preset."
            )
        v, r = self.viz, self.viz.renderer
        plan = getattr(getattr(v, "scene_service", None), "plan_view", None)
        if data.get("plan_view", False) and not (plan is not None and plan.available):
            # Saved looks and workspaces can outlive the scene they came from.
            data["plan_view"] = False
            if name == "Plan":
                name = "Custom"
            notice = "Plan needs a saved street graph; the 3D environment is shown."
        # Appearance changes take over from the temporary scenario rig. Restore
        # native light colours and the ordinary backdrop before saving "previous"
        # or applying a preset; otherwise blue-hour state leaks into both looks.
        if renderer_capabilities(r).dusk_lighting:
            r.set_dusk_enabled(False)
        self.previous = self.capture()
        self.notice = notice
        self.applying = True
        try:
            with r.batch_updates():
                self.surface, self.unlit = data["surface"], data["unlit"]
                # Stage global material choices, then publish through one owner.
                for key in (
                    "building_appearance",
                    "facade_theme",
                    "road_appearance",
                    "authored_surface_textures",
                ):
                    setattr(v, key, data[key])
                v.current_building_alpha = data["alpha"]
                pbr = v.material_pbr_service
                pbr.invalidate_material_resolution_cache()
                changed_textures = textures_globally_enabled() != data["textures"]
                ok = pbr.set_texture_maps_enabled(data["textures"])
                if not changed_textures:
                    ok = (
                        v.object_appearance_service.refresh_entry_appearance_batch(
                            list(v.mesh_entries) + list(v.target_entries)
                        )
                        and ok
                    )
                if not ok:
                    raise RuntimeError(
                        "Some surface materials could not be refreshed; select the preset again to retry."
                    )
                # Workspace/runtime synchronization also reads these display
                # controls from AppState; keep both owners in agreement.
                v.set_state(antialiasing_mode=data["aa"], texture_mipmaps=data["mipmaps"])
                v.scene_appearance_service.set_edge_visibility(data["outlines"])
                if "edge_width" in data:
                    r.set_edge_line_width(data["edge_width"])
                v.current_background_preset = next(
                    (
                        key
                        for key, color in SCENE_BACKGROUND_PRESETS.items()
                        if color == data["background"]
                    ),
                    "Custom",
                )
                v.scene_appearance_service.set_background_color(data["background"])
                loaded = self.apply_renderer_settings(r, data, matching_sky=matching_sky)
                if not loaded:
                    self.notice = "Saved sky is unavailable; the current environment was retained. Download it under Lighting."
                mpc = getattr(v, "mpc_appearance_service", None)
                if mpc is not None and "mpc_appearance" in data:
                    mpc.restore(data["mpc_appearance"])
                plan = getattr(getattr(v, "scene_service", None), "plan_view", None)
                if plan is not None:
                    plan.set_enabled(data.get("plan_view", False))
            self.active_name = name
            self.last_applied = self.capture()
        except (RuntimeError, ValueError, OSError):
            self.active_name = "Custom"
            self.last_applied = None
            raise
        finally:
            self.applying = False
            self.changed.emit()

    @staticmethod
    def apply_renderer_settings(r, data: dict, *, matching_sky=False) -> bool:
        """Apply the renderer portion of a look to either isolated viewport."""
        r.set_wireframe(data["wireframe"])
        r.set_color_fidelity_mode(data["color_fidelity"])
        r.set_texture_mipmaps_enabled(data["mipmaps"])
        r.set_antialiasing_mode(data["aa"])
        r.set_bloom_settings(data.get("bloom", DEFAULT_BLOOM))
        loaded = True
        if data["sky"] and data["sky"] != r.get_ibl_name():
            loaded = r.set_ibl(data["sky"])
        r.restore_light_rig_state(data["lights"], ibl=data["ibl"])
        r.set_shadow_quality(data["shadow_quality"])
        r.set_shadow_focus_radius(data["shadow_radius"])
        r.set_shadow_bias_texels(data["shadow_bias"])
        if "depth_precision" in data:
            r.set_depth_precision_enabled(data["depth_precision"])
        if matching_sky and data["skybox"] and loaded:
            r.apply_environment_lighting(data["sky"])
        r.show_skybox(data["skybox"] and loaded)
        return loaded

    def mark_if_custom(self) -> None:
        """Called after explicit UI edits, not during frame rendering."""
        if (
            not self.applying
            and self.last_applied is not None
            and self.capture() != self.last_applied
        ):
            self.active_name = "Custom"
            self.notice = ""
            self.changed.emit()

    def _path(self, name: str) -> Path:
        if (
            not name
            or name != name.strip()
            or len(name) > 80
            or any(not (c.isalnum() or c in " _-") for c in name)
        ):
            raise ValueError(
                "Use letters, numbers, spaces, underscores or hyphens for the preset name."
            )
        if name in PRESETS or name in PRESET_ALIASES or name == "Custom":
            raise ValueError("Choose a name different from the built-in presets.")
        return self.directory / f"{name}.json"

    def names(self) -> list[str]:
        return sorted(
            p.stem
            for p in self.directory.glob("*.json")
            if p.stem not in PRESETS and p.stem not in PRESET_ALIASES and p.stem != "Custom"
        )

    def load(self, name: str) -> dict:
        path = self._path(name)
        if path.stat().st_size > 65536:
            raise ValueError("Appearance preset is too large")
        return self.validate(json.loads(path.read_text()))

    def save(self, name: str) -> None:
        """Atomically save a small JSON recipe outside the checkout."""
        path = self._path(name)
        data = self.validate(self.capture())
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as f:
                tmp = f.name
                json.dump(data, f, indent=2, allow_nan=False)
            os.replace(tmp, path)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        self.active_name, self.last_applied = name, self.capture()
        self.changed.emit()
