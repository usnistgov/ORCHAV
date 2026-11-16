"""Per-view pygfx presentation using the desktop recipes and annotation layout."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from shared.statistics.themes import theme_manager

from ..backends.pygfx_scene_helpers import payload_to_pygfx_lines, payload_to_pygfx_mesh
from ..materials.appearance_recipes import appearance_recipe
from ..materials.mpc_styles import PRESETS as MPC_PRESETS
from ..materials.mpc_styles import SCENE_MPC_PRESETS
from ..materials.texture_policy import textures_locked_off, view_texture_maps
from ..materials.view_styles import PRESET_ALIASES, dark_presentation_ground, styled_surface
from ..model.render_objects import RenderObject
from ..renderers.pygfx.canvas import build_pygfx_effect_passes
from ..renderers.pygfx.labels import PygfxLabelMixin, text_label_from_payload
from ..renderers.pygfx.lighting_profiles import environment_lighting_profile
from ..renderers.pygfx.scene_controls import PygfxSceneControlsMixin
from ..renderers.pygfx.shadows import PygfxShadowMixin
from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
from ..scene.geometry_payload_factory import extract_wireframe_payload
from ..services.label_callout_service import label_with_callout_style
from ..services.material_context import MaterialContext
from ..types.label_style import LabelCalloutStyle
from ..types.render_payloads import MaterialPayload, SurfaceColorSource, TextLabelPayload
from ..utils.colors import MpcColorPalette

APPEARANCES = ("Detailed", "Paper", "Dark", "Glow")
TX_COLOR = (1.0, 0.2, 0.2, 1.0)
RX_COLOR = (0.2, 0.2, 1.0, 1.0)


class NotebookPresentation(PygfxLabelMixin, PygfxShadowMixin):
    """Own label, material, and lighting state for one canvas or image."""

    def __init__(
        self,
        gfx: Any,
        appearance: str | None,
        label_style: str | dict | LabelCalloutStyle,
        tx_color=TX_COLOR,
        rx_color=RX_COLOR,
        *,
        varied_buildings: bool = False,
    ) -> None:
        self._gfx = gfx
        self._objects: dict[str, Any] = {}
        self._label_backgrounds: dict[str, Any] = {}
        self._key_light = None
        self._shadow_quality = "scene"
        self.ibl_manager = None
        self.settings = None
        self.notice = ""
        self.mpc = dict(MPC_PRESETS["Standard"])
        if appearance is not None:
            appearance = PRESET_ALIASES.get(appearance, appearance)
            if appearance not in APPEARANCES:
                raise ValueError(
                    f"Notebook appearance must be one of {APPEARANCES}; got {appearance!r}"
                )
            self.settings, self.notice = appearance_recipe(
                appearance, {}, varied_buildings=varied_buildings
            )
            if self.settings["textures"] and textures_locked_off():
                raise ValueError(
                    "Detailed requires textures; unset ORCHAV_DISABLE_TEXTURES or choose another appearance"
                )
            self.mpc.update(MPC_PRESETS[SCENE_MPC_PRESETS[appearance]])
        else:
            theme = theme_manager.current
            self.mpc.update(
                continuous_colormap=theme.continuous_colormap,
                categorical_colormap=theme.categorical_colormap,
            )
        self.palette = MpcColorPalette.from_colormaps(
            self.mpc["continuous_colormap"], self.mpc["categorical_colormap"]
        )
        if isinstance(label_style, str):
            if label_style not in ("plain", "callout", "card"):
                raise ValueError("label_style must be 'plain', 'callout', or 'card'")
            label_style = {"mode": label_style}
        self.label_style = LabelCalloutStyle.from_dict(label_style)
        self.node_colors = {"tx": self._color(tx_color), "rx": self._color(rx_color)}

    @staticmethod
    def _color(value) -> tuple:
        color = np.asarray(value, dtype=float)
        if (
            color.shape not in ((3,), (4,))
            or not np.isfinite(color).all()
            or np.any((color < 0) | (color > 1))
        ):
            raise ValueError("TX/RX colors must contain three or four values between zero and one")
        return tuple(color) if len(color) == 4 else (*color, 1.0)

    @property
    def background(self) -> tuple:
        if self.settings is None:
            return tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA)
        return (*self.settings["background"], 1.0)

    @property
    def textures(self) -> bool | None:
        return None if self.settings is None else self.settings["textures"]

    def material_context(self) -> MaterialContext:
        return MaterialContext.from_defaults(self.settings or {})

    def setup_lights(self, scene) -> None:
        """Install the resolved rig; fit its shared shadow camera before drawing."""
        gfx = self._gfx
        if self.settings is None:
            scene.add(gfx.AmbientLight(intensity=0.4))
            light = gfx.DirectionalLight(intensity=0.8)
            light.local.position = (50, 50, 100)
            scene.add(light)
            return
        settings = self.settings
        lights = settings["lights"]
        if settings["skybox"] and self.ibl_manager is not None:
            profile = environment_lighting_profile(settings["sky"])
            if profile is not None:
                lights = vars(profile)
            self.ibl_manager.set_skybox_visible(True, scene)
        scene.add(gfx.AmbientLight(intensity=lights["ambient_intensity"]))
        self._shadows_enabled = lights["shadows_enabled"]
        self._key_light_azimuth_deg = lights["key_azimuth_deg"]
        self._key_light_elevation_deg = lights["key_elevation_deg"]
        for kind, radius in (("key", 80.0), ("fill", 70.0)):
            light = gfx.DirectionalLight(intensity=lights[f"{kind}_intensity"])
            scene.add(light)
            PygfxSceneControlsMixin._set_directional_light_from_angles(
                light=light,
                azimuth_deg=lights[f"{kind}_azimuth_deg"],
                elevation_deg=lights[f"{kind}_elevation_deg"],
                radius=radius,
            )
            if kind == "key":
                self._key_light = light
                light.cast_shadow = self._shadows_enabled

    def add_mesh(self, group, name, payload, material, entry, *, target=False) -> None:
        """Apply shared display materials while retaining authored geometry."""
        if self.settings is not None:
            material, source = styled_surface(
                material,
                payload.color_source,
                self.settings["surface"],
                self.settings["unlit"],
                entry.get("material_type", "default"),
                target,
                dark_presentation_ground(entry, payload),
            )
            if source == SurfaceColorSource.MATERIAL:
                payload = replace(payload, color_source=source, vertex_colors=None)
        with view_texture_maps(self.textures):
            mesh = payload_to_pygfx_mesh(self._gfx, payload, material, ibl_manager=self.ibl_manager)
        mesh.cast_shadow = mesh.receive_shadow = True
        self._objects[name] = mesh
        group.add(mesh)
        if self.settings is not None and self.settings["outlines"]:
            edges = extract_wireframe_payload(payload)
            if len(edges.lines):
                line = payload_to_pygfx_lines(
                    self._gfx, edges, line_width=self.settings["edge_width"]
                )
                line.material.color = (0.15, 0.15, 0.15, 1.0)
                group.add(line)

    def clear_dynamic(self) -> None:
        """Release the previous frame's native annotations and target registry."""
        self._objects = {
            name: obj for name, obj in self._objects.items() if name.startswith("scene_")
        }
        self._label_backgrounds.clear()

    def add_node_labels(self, group, tx_positions, rx_positions) -> None:
        for kind, positions in (("tx", tx_positions), ("rx", rx_positions)):
            for index, position in enumerate(positions):
                transform = np.eye(4)
                transform[:3, 3] = position
                if self.label_style.mode == "plain":
                    transform[2, 3] += 2.0
                obj = label_with_callout_style(
                    RenderObject(
                        id=f"node:{kind}_{index}",
                        payload=TextLabelPayload(f"{kind.upper()}{index + 1}"),
                        material=MaterialPayload(base_color=self.node_colors[kind]),
                        transform=transform,
                        metadata={"layout_anchor": position.tolist(), "node_kind": kind},
                    ),
                    self.label_style,
                )
                native = text_label_from_payload(self._gfx, obj.payload)
                native.material.color = obj.material_payload.base_color
                native.material.depth_test = obj.metadata.get("depth_test", True)
                native.material.depth_write = obj.metadata.get("depth_write", False)
                native.material.alpha_mode = obj.metadata.get("alpha_mode", "blend")
                native.render_order = obj.metadata.get("render_order", 0)
                native.local.position = obj.transform_matrix[:3, 3]
                self._objects[obj.id] = native
                group.add(native)
                self._apply_label_metadata(obj, native)

    def configure_renderer(self, renderer) -> None:
        if self.settings is not None:
            renderer.effect_passes = build_pygfx_effect_passes(
                self._gfx,
                clear_color=self.background,
                aa_mode=self.settings["aa"],
                depth_pass_enabled=False,
                bloom=self.settings["bloom"],
            )

    def prepare_draw(self, canvas, camera) -> None:
        self._canvas, self._camera = canvas, camera
        self._update_label_backgrounds()
        self._update_shadow_quality()

    @staticmethod
    def _is_scene_mesh_name(name: str) -> bool:
        return name.startswith("scene_")

    @staticmethod
    def _is_target_mesh_name(name: str) -> bool:
        return name.startswith("target_")
