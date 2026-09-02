"""Runtime PBR override service for visualizer material types.

``MaterialPBRService`` resolves the renderer-facing material payload from
catalog defaults, XML/authored entry properties, user overrides, texture policy,
and global transparency controls. It also applies updates to named renderer
objects while keeping scene and target entry identity stable. Renderer
synchronization is deliberately owned by ``ObjectAppearanceService``.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from shared.logging import get_logger

from ..materials.appearance import (
    MaterialGroupSummary,
    VisualMaterialBinding,
    VisualMaterialSource,
)
from ..materials.catalog import ResolvedMaterial, material_preset, resolve_pbr_material
from ..materials.presets import BUILTIN_MATERIAL_PRESETS
from ..materials.texture_policy import (
    TEXTURE_MAP_KEYS,
    clear_texture_path_validation_cache,
    textures_globally_enabled,
)
from ..model import RenderObjectState
from ..types.render_payloads import MeshPayload, SurfaceColorSource
from .object_identity import (
    ensure_scene_entry_identity,
    ensure_target_entry_identity,
    make_scene_entry_geometry_name,
    make_target_entry_geometry_name,
)

if TYPE_CHECKING:
    from ..pipeline.core import Visualizer

logger = get_logger("orchav.material_pbr_service")


class MaterialPBRService:
    """Manages PBR material properties with runtime overrides."""

    # Built-in material presets (material-agnostic - can be applied to any material)
    # Each preset defines visual properties that will be applied to the selected material
    PRESETS = BUILTIN_MATERIAL_PRESETS
    _INVALID = object()
    _MAX_RESOLVED_MATERIALS = 512

    _TEXTURED_PRESET_MATERIAL_TYPES = {
        "Concrete": "concrete",
        "Asphalt": "asphalt",
        "Brick": "brick",
        "Grass": "grass",
        "NIST CTL Floor": "ground_nist_ctl_floor",
    }
    _TEXTURE_PROPERTY_KEYS = (
        "texture_path",
        "normal_map_path",
        "normal_map_strength",
        "roughness_map_path",
        "ao_map_path",
        "metallic_map_path",
        "uv_scale_meters",
        "uv_repeat_scale",
        "shader_variant",
    )
    _TEXTURE_MAP_PROP_KEYS = TEXTURE_MAP_KEYS
    _UNIT_INTERVAL_PROPERTIES = {
        "roughness",
        "metallic",
        "reflectance",
        "alpha",
        "clearcoat",
        "clearcoat_roughness",
        "anisotropy",
        "transmission",
    }
    _NON_NEGATIVE_PROPERTIES = {
        "emissive_intensity",
        "glass_thickness",
        "normal_map_strength",
        "uv_scale_meters",
    }
    _SUPPORTED_PROPERTIES = {
        *_UNIT_INTERVAL_PROPERTIES,
        *_NON_NEGATIVE_PROPERTIES,
        "color",
        "emissive_color",
        "absorption_color",
        "texture_path",
        "normal_map_path",
        "roughness_map_path",
        "ao_map_path",
        "metallic_map_path",
        "uv_repeat_scale",
        "shader_variant",
    }
    _PROPERTY_DEFAULTS: Dict[str, Any] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
        "alpha": 1.0,
        "clearcoat": 0.0,
        "clearcoat_roughness": 0.0,
        "anisotropy": 0.0,
        "emissive_color": [0.0, 0.0, 0.0],
        "emissive_intensity": 0.0,
        "transmission": 0.0,
        "glass_thickness": 0.0,
        "absorption_color": [1.0, 1.0, 1.0],
        "normal_map_strength": 1.0,
        "uv_scale_meters": 2.0,
    }

    def __init__(self, visualizer: "Visualizer"):
        """Initialize runtime PBR overrides and the user preset directory."""
        self.visualizer = visualizer
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self._resolved_materials: OrderedDict[tuple[Any, ...], ResolvedMaterial] = OrderedDict()
        self.preset_dir = Path.home() / ".orchav" / "material_presets"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MaterialPBRService: Initialized (preset dir: {self.preset_dir})")

    @classmethod
    def _semantic_value(cls, value: Any) -> Any:
        """Return a hashable, sequence-shape-neutral material value."""
        if isinstance(value, Mapping):
            return tuple(
                sorted((str(key), cls._semantic_value(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._semantic_value(item) for item in value)
        if isinstance(value, set):
            return frozenset(cls._semantic_value(item) for item in value)
        if isinstance(value, Path):
            return str(value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @classmethod
    def _semantically_equal(cls, left: Any, right: Any) -> bool:
        """Compare normalized material values independent of list/tuple shape."""
        return cls._semantic_value(left) == cls._semantic_value(right)

    def _merge_override_values(
        self,
        material_type: str,
        values: Mapping[str, Any],
    ) -> bool:
        """Merge only semantic changes into one material override group."""
        overrides = self.overrides.setdefault(material_type, {})
        changed = False
        for key, value in values.items():
            if key in overrides and self._semantically_equal(overrides[key], value):
                continue
            overrides[key] = value
            changed = True
        return changed

    def _resolve_material(
        self,
        props: Mapping[str, Any],
        *,
        context: str,
    ) -> ResolvedMaterial:
        """Return the shared immutable resolution for one semantic material."""
        signature = (
            context,
            textures_globally_enabled(),
            self._semantic_value(props),
        )
        cached = self._resolved_materials.get(signature)
        if cached is not None:
            self._resolved_materials.move_to_end(signature)
            return cached
        resolved = resolve_pbr_material(
            props.get("color", [0.7, 0.7, 0.7]),
            props,
            context=context,
        )
        self._resolved_materials[signature] = resolved
        self._resolved_materials.move_to_end(signature)
        while len(self._resolved_materials) > self._MAX_RESOLVED_MATERIALS:
            self._resolved_materials.popitem(last=False)
        return resolved

    def invalidate_material_resolution_cache(self) -> None:
        """Drop shared resolutions after external texture assets change in place."""
        self._resolved_materials.clear()
        clear_texture_path_validation_cache()

    def set_property(
        self, material_type: str, property_name: str, value: "float | list[float]"
    ) -> bool:
        """Set a PBR property for a material type.

        Args:
            material_type: Material type string (e.g., "glass", "concrete")
            property_name: Property name ("roughness", "metallic", "reflectance", "alpha",
                          "color")
            value: Property value - float (0.0-1.0) or list of 3 floats for "color"

        Returns:
            True if property was set and rendering updated
        """
        normalized = self._validate_property(property_name, value)
        if normalized is self._INVALID:
            logger.warning("MaterialPBRService: Invalid %s value: %r", property_name, value)
            return False
        material_overrides = self.overrides.setdefault(material_type, {})
        override_changed = property_name not in material_overrides or not self._semantically_equal(
            material_overrides[property_name], normalized
        )
        if override_changed:
            material_overrides[property_name] = normalized
        binding_changed = self._bind_matching_entries_manual(
            material_type,
            {property_name: normalized},
        )
        if not override_changed and not binding_changed:
            logger.debug(
                "MaterialPBRService: Coalesced unchanged %s.%s",
                material_type,
                property_name,
            )
            return True
        logger.debug("MaterialPBRService: Set %s.%s = %r", material_type, property_name, normalized)

        self._update_material_rendering(material_type)
        return True

    @classmethod
    def _validate_property(cls, property_name: str, value: Any) -> Any:
        """Return a normalized safe PBR value, or the invalid sentinel."""
        if property_name not in cls._SUPPORTED_PROPERTIES:
            return cls._INVALID
        if property_name in {
            "texture_path",
            "normal_map_path",
            "roughness_map_path",
            "ao_map_path",
            "metallic_map_path",
            "shader_variant",
        }:
            if value is None:
                return None
            if not isinstance(value, (str, Path)):
                return cls._INVALID
            normalized_path = str(value).strip()
            return normalized_path or cls._INVALID
        if property_name == "uv_repeat_scale":
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return cls._INVALID
            try:
                scale = tuple(float(channel) for channel in value[:2])
            except (TypeError, ValueError):
                return cls._INVALID
            if not all(math.isfinite(channel) and channel > 0.0 for channel in scale):
                return cls._INVALID
            return scale
        if property_name in {"color", "emissive_color", "absorption_color"}:
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                return cls._INVALID
            try:
                channels = [float(channel) for channel in value[:3]]
            except (TypeError, ValueError):
                return cls._INVALID
            if not all(math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in channels):
                return cls._INVALID
            return channels
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return cls._INVALID
        if not math.isfinite(scalar):
            return cls._INVALID
        if property_name in cls._UNIT_INTERVAL_PROPERTIES and not 0.0 <= scalar <= 1.0:
            return cls._INVALID
        if property_name in cls._NON_NEGATIVE_PROPERTIES and scalar < 0.0:
            return cls._INVALID
        if property_name == "normal_map_strength" and scalar > 4.0:
            return cls._INVALID
        if property_name == "uv_scale_meters" and scalar <= 0.0:
            return cls._INVALID
        return scalar

    def _all_entries(self) -> list[Dict[str, Any]]:
        """Return persistent scene and target entries outside the frame/MPC path."""
        return list(getattr(self.visualizer, "mesh_entries", []) or []) + list(
            getattr(self.visualizer, "target_entries", []) or []
        )

    def get_visual_binding(self, entry: Dict[str, Any]) -> VisualMaterialBinding:
        """Return or establish the entry's EM-independent visual binding."""
        binding = entry.get("_visual_material_binding")
        if isinstance(binding, VisualMaterialBinding):
            return binding
        profile_service = getattr(self.visualizer, "visual_profile_service", None)
        resolve_binding = getattr(profile_service, "resolve_binding", None)
        if callable(resolve_binding):
            profile_binding = resolve_binding(
                str(entry.get("name") or ""),
                str(entry.get("material_type") or "default"),
                self._entry_kind(entry),
            )
            if isinstance(profile_binding, VisualMaterialBinding):
                entry["_visual_material_binding"] = profile_binding
                return profile_binding
        return VisualMaterialBinding()

    def set_visual_binding(
        self,
        entry: Dict[str, Any],
        binding: VisualMaterialBinding,
    ) -> None:
        """Assign an explicit profile/manual binding without changing EM identity."""
        entry["_visual_material_binding"] = binding

    def clear_visual_binding(self, entry: Dict[str, Any]) -> None:
        """Return one entry to EM-derived visual defaults."""
        entry["_visual_material_binding"] = VisualMaterialBinding()

    def get_visual_material_key(self, entry: Dict[str, Any]) -> str:
        """Return the stable visual-material key used by controls and overlays."""
        return self._binding_material_key(entry)

    def _binding_material_key(
        self,
        entry: Dict[str, Any],
        binding: VisualMaterialBinding | None = None,
    ) -> str:
        """Return the visual material key used by group overrides and controls."""
        resolved = binding or self.get_visual_binding(entry)
        if resolved.source is VisualMaterialSource.FOLLOW_EM:
            return str(entry.get("material_type") or "default")
        return str(resolved.material_type or entry.get("material_type") or "default")

    def _bind_matching_entries_manual(
        self,
        material_type: str,
        changed: Dict[str, Any],
        *,
        preset: str | None = None,
    ) -> bool:
        """Make a visual group override explicit on each current member."""
        changed_any = False
        for entry in self._all_entries():
            binding = self.get_visual_binding(entry)
            if self._binding_material_key(entry, binding) != material_type:
                continue
            if binding.source is VisualMaterialSource.MANUAL:
                overrides = dict(binding.overrides)
            else:
                # Freeze the complete current visual assignment before the
                # entry becomes EM-independent. This retains authored albedo
                # and non-color maps under later EM Material ID changes.
                overrides = self.get_entry_base_properties(entry)
            entry_changes = dict(changed)
            if preset is None and "color" in entry_changes:
                color_source = self._entry_surface_color_source(entry)
                active_color_is_external = (
                    self.resolve_entry_material(entry).texture_policy.active_albedo_path is not None
                )
                if color_source is SurfaceColorSource.VERTEX or active_color_is_external:
                    # A group color edit affects only uniform-color members.
                    entry_changes.pop("color", None)
            if not entry_changes:
                continue
            overrides.update(entry_changes)
            updated_binding = VisualMaterialBinding(
                source=VisualMaterialSource.MANUAL,
                material_type=(
                    binding.material_type
                    if binding.source is not VisualMaterialSource.FOLLOW_EM
                    else material_type
                ),
                preset=preset if preset is not None else binding.preset,
                overrides=overrides,
            )
            if updated_binding == binding:
                continue
            entry["_visual_material_binding"] = updated_binding
            changed_any = True
        return changed_any

    def get_property(
        self, material_type: str, property_name: str, base_props: Dict[str, Any]
    ) -> float:
        """Get the effective value of a property after applying overrides.

        Args:
            material_type: Material type string
            property_name: Property name
            base_props: Base properties dict from XML or ITU_TO_PBR

        Returns:
            Effective property value (with UI override if set)
        """
        if material_type in self.overrides and property_name in self.overrides[material_type]:
            return self.overrides[material_type][property_name]
        return base_props.get(property_name, 0.0)

    def get_effective_properties(
        self, material_type: str, base_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get final properties after applying UI overrides.

        Args:
            material_type: Material type string
            base_props: Base properties dict (from XML or ITU_TO_PBR)

        Returns:
            Properties dict with UI overrides applied
        """
        props = self._effective_unresolved_properties(material_type, base_props)
        return self._resolve_material(props, context=material_type).properties_copy(
            mark_texture_policy=True
        )

    def _effective_unresolved_properties(
        self,
        material_type: str,
        base_props: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge and validate authored/UI properties before texture policy."""
        props = dict(base_props)
        if material_type in self.overrides:
            props.update(self.overrides[material_type])
        return self._validated_properties(props, context=material_type)

    @classmethod
    def _validated_properties(
        cls,
        props: Dict[str, Any],
        *,
        context: str,
    ) -> Dict[str, Any]:
        """Return finite, renderer-safe PBR properties with stable fallbacks."""
        validated = dict(props)
        for property_name in cls._SUPPORTED_PROPERTIES:
            if property_name not in validated:
                continue
            normalized = cls._validate_property(property_name, validated[property_name])
            if normalized is not cls._INVALID:
                validated[property_name] = normalized
                continue
            logger.warning(
                "MaterialPBRService: Ignoring invalid authored %s.%s value %r",
                context,
                property_name,
                validated[property_name],
            )
            if property_name in cls._PROPERTY_DEFAULTS:
                fallback = cls._PROPERTY_DEFAULTS[property_name]
                validated[property_name] = (
                    list(fallback) if isinstance(fallback, list) else fallback
                )
            else:
                validated[property_name] = None
        return validated

    @staticmethod
    def _effective_global_alpha(base_alpha: Any, current_alpha: Any) -> float:
        """Keep preset alpha unless the global transparency slider is active."""
        try:
            current = float(current_alpha)
        except (TypeError, ValueError):
            current = 1.0
        try:
            base = float(base_alpha)
        except (TypeError, ValueError):
            base = 1.0
        if not math.isfinite(current) or not 0.0 <= current <= 1.0:
            current = 1.0
        if not math.isfinite(base) or not 0.0 <= base <= 1.0:
            base = 1.0
        if abs(current - 1.0) > 1e-9:
            return current
        return base

    def _entry_kind(self, entry: Dict[str, Any]) -> str:
        """Classify an entry as scene mesh or target for alpha and naming policy."""
        entry_type = str(entry.get("entry_type") or "")
        if entry_type:
            return entry_type
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        entry_kind = getattr(appearance, "entry_kind", None)
        if callable(entry_kind):
            resolved_kind = entry_kind(entry)
            if resolved_kind:
                return str(resolved_kind)
        for target_entry in getattr(self.visualizer, "target_entries", []):
            if target_entry is entry:
                return "target"
        return "mesh"

    @classmethod
    def _fill_discovered_texture_defaults(
        cls,
        props: Dict[str, Any],
        *defaults: Mapping[str, Any] | None,
    ) -> None:
        """Fill empty FOLLOW_EM map slots from ordered discovery results."""
        for key in cls._TEXTURE_MAP_PROP_KEYS:
            if props.get(key):
                continue
            for source in defaults:
                if not isinstance(source, Mapping):
                    continue
                discovered = source.get(key)
                if discovered:
                    props[key] = discovered
                    break

    def get_entry_base_properties(
        self,
        entry: Dict[str, Any],
        *,
        discovered_texture_defaults: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return authored properties plus optional FOLLOW_EM texture defaults."""
        binding = self.get_visual_binding(entry)
        em_material_type = str(entry.get("material_type") or "default")
        if binding.source is VisualMaterialSource.FOLLOW_EM:
            props = dict(material_preset(em_material_type))
            props.update(dict(entry.get("pbr_properties", {}) or {}))
            authored_fields = {
                "color": entry.get("pbr_color", entry.get("color")),
                "roughness": entry.get("pbr_roughness"),
                "metallic": entry.get("pbr_metallic"),
                "reflectance": entry.get("pbr_reflectance"),
                "alpha": entry.get("pbr_alpha"),
                "texture_path": entry.get("texture_path"),
            }
            props.update(
                {key: value for key, value in authored_fields.items() if value is not None}
            )
            self._fill_discovered_texture_defaults(
                props,
                discovered_texture_defaults,
                entry.get("_resolved_texture_maps"),
            )
        else:
            if binding.preset and binding.preset in self.PRESETS:
                props = {
                    key: value
                    for key, value in self.PRESETS[binding.preset].items()
                    if key != "description"
                }
            else:
                props = dict(material_preset(binding.material_type or em_material_type))
            props.update(dict(binding.overrides))
        if not props:
            props = {
                "color": entry.get("pbr_color", entry.get("color", [0.7, 0.7, 0.7])),
                "roughness": entry.get("pbr_roughness", 0.5),
                "metallic": entry.get("pbr_metallic", 0.0),
                "reflectance": entry.get("pbr_reflectance", 0.5),
                "alpha": entry.get("pbr_alpha", 1.0),
            }
        else:
            props.setdefault("color", entry.get("pbr_color", entry.get("color", [0.7, 0.7, 0.7])))
            props.setdefault("roughness", entry.get("pbr_roughness", 0.5))
            props.setdefault("metallic", entry.get("pbr_metallic", 0.0))
            props.setdefault("reflectance", entry.get("pbr_reflectance", 0.5))
            props.setdefault("alpha", entry.get("pbr_alpha", 1.0))
        return props

    @staticmethod
    def _entry_surface_color_source(entry: Dict[str, Any]) -> SurfaceColorSource:
        """Read explicit neutral color ownership from an entry mesh."""
        mesh = entry.get("mesh")
        payload = mesh.payload if isinstance(mesh, RenderObjectState) else mesh
        if isinstance(payload, MeshPayload):
            return payload.color_source
        return SurfaceColorSource.MATERIAL

    @staticmethod
    def _entry_has_uvs(entry: Dict[str, Any]) -> bool:
        """Return whether an entry can sample UV-compatible texture maps."""
        mesh = entry.get("mesh")
        payload = mesh.payload if isinstance(mesh, RenderObjectState) else mesh
        return bool(
            isinstance(payload, MeshPayload)
            and payload.triangle_uvs is not None
            and len(payload.triangle_uvs) > 0
        )

    def get_effective_entry_properties(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve renderer-ready visual properties for one entry.

        This is the shared appearance path used by material-type controls,
        object material reassignment refreshes, and target highlight restore.
        """
        return self.resolve_entry_material(entry).properties_copy(mark_texture_policy=True)

    def resolve_entry_material(
        self,
        entry: Dict[str, Any],
        *,
        discovered_texture_defaults: Mapping[str, Any] | None = None,
    ) -> ResolvedMaterial:
        """Return one immutable material with optional FOLLOW_EM texture discovery."""
        binding = self.get_visual_binding(entry)
        material_type = self._binding_material_key(entry, binding)
        base_props = self.get_entry_base_properties(
            entry,
            discovered_texture_defaults=discovered_texture_defaults,
        )
        props = self._effective_unresolved_properties(material_type, base_props)
        kind = self._entry_kind(entry)
        alpha_key = "current_target_alpha" if kind == "target" else "current_building_alpha"
        props["alpha"] = self._effective_global_alpha(
            props.get("alpha", 1.0),
            getattr(self.visualizer, alpha_key, 1.0),
        )
        color_source = self._entry_surface_color_source(entry)
        if color_source is SurfaceColorSource.VERTEX:
            props["color"] = [1.0, 1.0, 1.0]
            props["texture_path"] = None
            if not self._entry_has_uvs(entry):
                for key in self._TEXTURE_MAP_PROP_KEYS:
                    props[key] = None
            return self._resolve_material(props, context=material_type)

        resolved = self._resolve_material(props, context=material_type)
        if resolved.texture_policy.active_albedo_path is not None:
            # A group color edit is not a hidden texture tint. External
            # albedo keeps its authored factor while non-color PBR still
            # follows the selected visual assignment.
            authored_color = list(base_props.get("color", [1.0, 1.0, 1.0]))
            if not self._semantically_equal(props.get("color"), authored_color):
                props["color"] = authored_color
                resolved = self._resolve_material(props, context=material_type)
        return resolved

    def summarize_material_group(self, material_type: str) -> MaterialGroupSummary:
        """Aggregate editable and authored color ownership for panel controls."""
        members = [
            entry
            for entry in self._all_entries()
            if self._binding_material_key(entry) == material_type
        ]
        uniform = 0
        external_albedo = 0
        vertex = 0
        representative: Dict[str, Any] = {}
        for entry in members:
            props = self.resolve_entry_material(entry).properties_copy()
            if not representative:
                representative = dict(props)
            if self._entry_surface_color_source(entry) is SurfaceColorSource.VERTEX:
                vertex += 1
            elif props.get("texture_path"):
                external_albedo += 1
            else:
                uniform += 1
        return MaterialGroupSummary(
            member_count=len(members),
            uniform_color_members=uniform,
            external_albedo_members=external_albedo,
            vertex_color_members=vertex,
            representative_properties=representative,
        )

    @staticmethod
    def _find_entry_index(entries: Any, entry: Dict[str, Any]) -> Optional[int]:
        """Find an entry by identity, mesh identity, or stable display name."""
        mesh = entry.get("mesh")
        name = entry.get("name")
        for index, candidate in enumerate(entries or []):
            if candidate is entry:
                return index
            if mesh is not None and candidate.get("mesh") is mesh:
                return index
            if name and candidate.get("name") == name:
                return index
        return None

    def get_entry_geometry_name(self, entry: Dict[str, Any]) -> str:
        """Return the stable renderer geometry name for an entry mesh."""
        existing = entry.get("geometry_name")
        if existing:
            return str(existing)
        if self._entry_kind(entry) == "target":
            appearance = getattr(self.visualizer, "object_appearance_service", None)
            entry_index = getattr(appearance, "entry_index", None)
            if callable(entry_index):
                resolved_index = entry_index(entry, entry_type="target")
                index = resolved_index if resolved_index >= 0 else None
            else:
                index = self._find_entry_index(
                    getattr(self.visualizer, "target_entries", []),
                    entry,
                )
            ensure_target_entry_identity(entry, index)
            refresh_index = getattr(appearance, "refresh_entry_index", None)
            if callable(refresh_index):
                refresh_index(entry, entry_type="target")
            return make_target_entry_geometry_name(entry, "mesh")
        ensure_scene_entry_identity(entry)
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh_index = getattr(appearance, "refresh_entry_index", None)
        if callable(refresh_index):
            refresh_index(entry, entry_type="mesh")
        return make_scene_entry_geometry_name(entry, "mesh")

    def reset_material(self, material_type: str) -> bool:
        """Remove manual overrides and restore the underlying profile or EM binding.

        Args:
            material_type: Material type to reset

        Returns:
            True if material was reset and rendering updated
        """
        entries = self._all_entries()
        affected_entries: list[Dict[str, Any]] = []
        binding_changed = False
        override_changed = self.overrides.pop(material_type, None) is not None
        for entry in entries:
            binding = self.get_visual_binding(entry)
            if self._binding_material_key(entry, binding) != material_type:
                continue
            if override_changed or binding.source is VisualMaterialSource.MANUAL:
                affected_entries.append(entry)
            if binding.source is VisualMaterialSource.MANUAL:
                entry.pop("_visual_material_binding", None)
                binding_changed = True

        if not override_changed and not binding_changed:
            return False
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
        if affected_entries and callable(refresh_batch):
            refresh_batch(affected_entries)
        logger.info("MaterialPBRService: Reset manual overrides for '%s'", material_type)
        return True

    def clear_visual_assignment(self, material_type: str) -> bool:
        """Return one visual-material group to explicit Follow EM bindings.

        Scenario profiles and manual assignments can exist without a matching
        group override. Capture members before clearing their bindings because
        an EM-independent visual key may change as soon as Follow EM is stored.
        """
        affected_entries: list[Dict[str, Any]] = []
        binding_changed = False
        for entry in self._all_entries():
            binding = self.get_visual_binding(entry)
            if self._binding_material_key(entry, binding) != material_type:
                continue
            affected_entries.append(entry)
            if binding.source is VisualMaterialSource.FOLLOW_EM:
                continue
            self.clear_visual_binding(entry)
            binding_changed = True

        override_changed = self.overrides.pop(material_type, None) is not None
        if not binding_changed and not override_changed:
            return False

        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
        if affected_entries and callable(refresh_batch):
            refresh_batch(affected_entries)
        logger.info("MaterialPBRService: Cleared visual assignment for '%s'", material_type)
        return True

    def reset_all(self) -> bool:
        """Reset all manual overrides to their underlying profile or EM bindings."""
        material_types = set(self.overrides)
        affected_entries: list[Dict[str, Any]] = []
        binding_changed = False
        for entry in self._all_entries():
            binding = self.get_visual_binding(entry)
            material_key = self._binding_material_key(entry, binding)
            if material_key in material_types or binding.source is VisualMaterialSource.MANUAL:
                affected_entries.append(entry)
            if binding.source is VisualMaterialSource.MANUAL:
                entry.pop("_visual_material_binding", None)
                binding_changed = True
        if not material_types and not binding_changed:
            return False

        self.overrides.clear()
        logger.info("MaterialPBRService: Reset all material overrides")

        # Republish only affected semantic materials. A full scene render would
        # retire and upload unrelated objects; SceneService owns focused,
        # failure-aware regrouping for material-signature changes.
        appearance = getattr(self.visualizer, "object_appearance_service", None)
        refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
        if affected_entries and callable(refresh_batch):
            refresh_batch(affected_entries)
        return True

    def apply_preset(self, preset_name: str, material_type: Optional[str] = None) -> bool:
        """Apply a built-in preset to a material type.

        Args:
            preset_name: Name of the preset to apply
            material_type: Material type to apply preset to. If None, applies to all
                          materials in the scene.

        Returns:
            True if preset was applied successfully
        """
        if preset_name not in self.PRESETS:
            logger.warning(f"MaterialPBRService: Unknown preset '{preset_name}'")
            return False

        preset_data = self.PRESETS[preset_name]

        # Extract just the property values (exclude 'description').  Also
        # clear every PBR texture/advanced field not owned by the new preset.
        # Otherwise applying NIST CTL Floor and then Brick would keep the
        # presentation-floor clearcoat/anisotropy on the brick material.
        # Picking "Gold" on a concrete wall must also look like gold, not
        # gold-tinted concrete.
        properties = {
            k: v
            for k, v in preset_data.items()
            if k
            in [
                "roughness",
                "metallic",
                "reflectance",
                "alpha",
                "color",
                "clearcoat",
                "clearcoat_roughness",
                "anisotropy",
                "emissive_color",
                "emissive_intensity",
                "transmission",
                "glass_thickness",
                "absorption_color",
            ]
        }
        properties["texture_path"] = None
        properties["normal_map_path"] = None
        properties["normal_map_strength"] = 1.0
        properties["roughness_map_path"] = None
        properties["ao_map_path"] = None
        properties["metallic_map_path"] = None
        properties["uv_scale_meters"] = 2.0
        properties["uv_repeat_scale"] = None
        properties["shader_variant"] = None
        properties.setdefault("clearcoat", 0.0)
        properties.setdefault("clearcoat_roughness", 0.0)
        properties.setdefault("anisotropy", 0.0)
        properties.setdefault("emissive_color", (0.0, 0.0, 0.0))
        properties.setdefault("emissive_intensity", 0.0)
        properties.setdefault("transmission", 0.0)
        properties.setdefault("glass_thickness", 0.0)
        properties.setdefault("absorption_color", (1.0, 1.0, 1.0))
        texture_material_type = self._TEXTURED_PRESET_MATERIAL_TYPES.get(preset_name)
        if texture_material_type:
            from ..materials.catalog import material_preset

            pbr_texture_pack = material_preset(texture_material_type)
            for key in self._TEXTURE_PROPERTY_KEYS:
                if key in pbr_texture_pack:
                    properties[key] = pbr_texture_pack.get(key)

        if material_type:
            override_changed = self._merge_override_values(material_type, properties)
            binding_changed = self._bind_matching_entries_manual(
                material_type,
                properties,
                preset=preset_name,
            )
            if override_changed or binding_changed:
                logger.info(
                    "MaterialPBRService: Applied preset '%s' to '%s'",
                    preset_name,
                    material_type,
                )
                self._update_material_rendering(material_type)
            else:
                logger.debug(
                    "MaterialPBRService: Coalesced unchanged preset '%s' on '%s'",
                    preset_name,
                    material_type,
                )
        else:
            material_types = self.get_material_types_in_scene()
            changed_types: set[str] = set()
            for mat_type in material_types:
                override_changed = self._merge_override_values(mat_type, properties)
                binding_changed = self._bind_matching_entries_manual(
                    mat_type,
                    properties,
                    preset=preset_name,
                )
                if override_changed or binding_changed:
                    changed_types.add(mat_type)
            if changed_types:
                logger.info(
                    "MaterialPBRService: Applied preset '%s' to all materials",
                    preset_name,
                )
                appearance = getattr(self.visualizer, "object_appearance_service", None)
                refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
                if callable(refresh_batch):
                    refresh_batch(
                        [
                            entry
                            for entry in self._all_entries()
                            if self._binding_material_key(entry) in changed_types
                        ]
                    )

        return True

    def get_preset_properties(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get the properties dict for a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            Properties dict (without 'description'), or None if not found
        """
        if preset_name not in self.PRESETS:
            return None

        preset_data = self.PRESETS[preset_name]
        return {k: v for k, v in preset_data.items() if k != "description"}

    def save_preset(self, name: str) -> bool:
        """Save current material overrides as a user preset.

        Args:
            name: Name for the preset

        Returns:
            True if preset was saved successfully
        """
        if not self.overrides:
            logger.warning("MaterialPBRService: No overrides to save")
            return False

        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        preset_path = self.preset_dir / f"{safe_name}.json"

        try:
            preset_data = {"name": name, "overrides": self.overrides}
            preset_path.write_text(json.dumps(preset_data, indent=2))
            logger.info(f"MaterialPBRService: Saved preset '{name}' to {preset_path}")
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"MaterialPBRService: Failed to save preset: {exc}")
            return False

    def load_preset(self, name: str) -> bool:
        """Load a user preset.

        Args:
            name: Name of the preset to load

        Returns:
            True if preset was loaded successfully
        """
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        preset_path = self.preset_dir / f"{safe_name}.json"

        if not preset_path.exists():
            logger.warning(f"MaterialPBRService: Preset '{name}' not found")
            return False

        try:
            preset_data = json.loads(preset_path.read_text())
            raw_overrides = preset_data.get("overrides", {})
            if not isinstance(raw_overrides, dict):
                raise ValueError("preset overrides must be an object")
            validated_overrides: Dict[str, Dict[str, Any]] = {}
            for material_type, properties in raw_overrides.items():
                if not isinstance(material_type, str) or not isinstance(properties, dict):
                    raise ValueError("preset material entries must be property objects")
                validated_properties: Dict[str, Any] = {}
                for property_name, value in properties.items():
                    if property_name not in self._SUPPORTED_PROPERTIES:
                        raise ValueError(f"unsupported preset property: {property_name}")
                    normalized = self._validate_property(property_name, value)
                    if normalized is self._INVALID:
                        raise ValueError(
                            f"invalid preset property: {material_type}.{property_name}"
                        )
                    validated_properties[property_name] = normalized
                validated_overrides[material_type] = validated_properties
            entries = self._all_entries()
            old_material_types = set(self.overrides)
            affected_entries: list[Dict[str, Any]] = []
            affected_ids: set[int] = set()

            def _mark_affected(entry: Dict[str, Any]) -> None:
                identity = id(entry)
                if identity not in affected_ids:
                    affected_ids.add(identity)
                    affected_entries.append(entry)

            # Loading a saved preset replaces the complete manual override
            # state. Rebase existing MANUAL entries first so properties and
            # material groups absent from the file cannot leak forward.
            for entry in entries:
                binding = self.get_visual_binding(entry)
                material_key = self._binding_material_key(entry, binding)
                if material_key in old_material_types or material_key in validated_overrides:
                    _mark_affected(entry)
                if binding.source is VisualMaterialSource.MANUAL:
                    entry.pop("_visual_material_binding", None)
                    _mark_affected(entry)

            self.overrides = validated_overrides
            for material_type, properties in validated_overrides.items():
                self._bind_matching_entries_manual(material_type, properties)
            for entry in entries:
                if self._binding_material_key(entry) in validated_overrides:
                    _mark_affected(entry)
            logger.info(f"MaterialPBRService: Loaded preset '{name}'")

            appearance = getattr(self.visualizer, "object_appearance_service", None)
            refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
            if affected_entries and callable(refresh_batch):
                refresh_batch(affected_entries)

            return True
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"MaterialPBRService: Failed to load preset: {exc}")
            return False

    def list_user_presets(self) -> list[str]:
        """Get list of user-defined preset names.

        Returns:
            List of preset names
        """
        try:
            presets = []
            for path in self.preset_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    presets.append(data.get("name", path.stem))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
            return sorted(presets)
        except OSError as exc:
            logger.error(f"MaterialPBRService: Failed to list presets: {exc}")
            return []

    def _update_material_rendering(self, material_type: str) -> bool:
        """Notify the appearance coordinator that one visual group changed."""
        viz = self.visualizer
        matching_entries = [
            entry
            for entry in self._all_entries()
            if self._binding_material_key(entry) == material_type
        ]
        if not matching_entries:
            return False
        appearance = getattr(viz, "object_appearance_service", None)
        refresh_batch = getattr(appearance, "refresh_entry_appearance_batch", None)
        synced = bool(callable(refresh_batch) and refresh_batch(matching_entries))
        return synced

    def get_material_types_in_scene(self) -> list[str]:
        """Get list of material types present in the current scene.

        Returns:
            List of unique material type strings from both scene meshes and targets.
        """
        material_types: set[str] = set()

        for entry in self._all_entries():
            binding = self.get_visual_binding(entry)
            if not entry.get("material_type") and binding.source is VisualMaterialSource.FOLLOW_EM:
                continue
            material_type = self._binding_material_key(entry)
            if material_type:
                material_types.add(material_type)

        return sorted(material_types)

    def cleanup(self) -> None:
        """Clean up service resources."""
        self.overrides.clear()
        self._resolved_materials.clear()
        logger.info("MaterialPBRService: Cleaned up")
