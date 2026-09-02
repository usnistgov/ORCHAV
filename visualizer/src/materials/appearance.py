"""Pure visual-material binding and effective-appearance contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..types.render_payloads import MaterialPayload, SurfaceColorSource


class MaterialDisplayMode(str, Enum):
    """Transient overlay applied to every object using a material key."""

    NORMAL = "normal"
    HIDDEN = "hidden"
    HIGHLIGHTED = "highlighted"

    @classmethod
    def coerce(cls, value: Any) -> "MaterialDisplayMode":
        """Normalize an enum or exact serialized display-mode value."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Invalid material display mode: {value}") from exc


class VisualMaterialSource(str, Enum):
    """Describe why an entry uses its current visual material."""

    FOLLOW_EM = "follow_em"
    PROFILE = "profile"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class VisualMaterialBinding:
    """Stable relationship between EM identity and visual PBR appearance."""

    source: VisualMaterialSource = VisualMaterialSource.FOLLOW_EM
    material_type: str | None = None
    preset: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize enum and detach mutable override mappings."""
        object.__setattr__(self, "source", VisualMaterialSource(self.source))
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        if self.source is VisualMaterialSource.FOLLOW_EM and (
            self.material_type is not None or self.preset is not None or self.overrides
        ):
            raise ValueError("FOLLOW_EM bindings cannot carry an explicit visual assignment")


@dataclass(frozen=True, slots=True)
class AppearanceIntent:
    """All independent inputs needed to resolve one persistent object's appearance."""

    manual_visible: bool = True
    runtime_visible: bool = True
    frame_visible: bool = True
    pov_visible: bool = True
    global_visible: bool = True
    manual_highlight: bool = False
    selected: bool = False
    material_mode: MaterialDisplayMode = MaterialDisplayMode.NORMAL
    material: MaterialPayload = field(default_factory=MaterialPayload)
    color_source: SurfaceColorSource = SurfaceColorSource.MATERIAL

    def __post_init__(self) -> None:
        """Normalize enum-like constructor inputs."""
        object.__setattr__(self, "material_mode", MaterialDisplayMode.coerce(self.material_mode))
        object.__setattr__(self, "color_source", SurfaceColorSource(self.color_source))


@dataclass(frozen=True, slots=True)
class ResolvedAppearance:
    """Final renderer-facing visibility, highlight, and material snapshot."""

    visible: bool
    highlighted: bool
    material: MaterialPayload
    color_source: SurfaceColorSource


@dataclass(frozen=True, slots=True)
class MaterialGroupSummary:
    """Aggregate color ownership for one visual-material control group."""

    member_count: int
    uniform_color_members: int
    external_albedo_members: int
    vertex_color_members: int
    representative_properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def color_editable(self) -> bool:
        """Return whether a color edit can affect at least one uniform member."""
        return self.uniform_color_members > 0

    @property
    def color_edit_note(self) -> str:
        """Explain which members retain authored RGB under a group edit."""
        authored = self.external_albedo_members + self.vertex_color_members
        if authored == 0:
            return "Color applies to every member in this visual-material group."
        if self.uniform_color_members == 0:
            if self.external_albedo_members and not self.vertex_color_members:
                return "Color is locked because an authored albedo texture owns RGB."
            if self.vertex_color_members and not self.external_albedo_members:
                return "Color is locked because intrinsic vertex colors own RGB."
            return "Color is locked because authored albedo textures or vertex colors own RGB."
        return (
            f"Color applies to {self.uniform_color_members} uniform member(s); "
            f"{authored} textured or vertex-colored member(s) retain authored RGB."
        )


def _highlight_material(
    material: MaterialPayload,
    color_source: SurfaceColorSource,
) -> MaterialPayload:
    """Return the temporary red inspection appearance without replacing authored RGB."""
    owns_external_rgb = (
        color_source is SurfaceColorSource.VERTEX or material.texture_path is not None
    )
    if owns_external_rgb:
        return replace(material, color_multiplier=(1.0, 0.3, 0.3))
    alpha = material.base_color[3]
    return replace(
        material,
        base_color=(1.0, 0.0, 0.0, alpha),
        color_multiplier=(1.0, 1.0, 1.0),
    )


def resolve_appearance(intent: AppearanceIntent) -> ResolvedAppearance:
    """Resolve appearance precedence without reading or mutating application state."""
    mode = MaterialDisplayMode.coerce(intent.material_mode)
    visible = bool(
        intent.manual_visible
        and intent.runtime_visible
        and intent.frame_visible
        and intent.pov_visible
        and intent.global_visible
        and mode is not MaterialDisplayMode.HIDDEN
    )
    highlighted = bool(
        visible
        and (intent.manual_highlight or intent.selected or mode is MaterialDisplayMode.HIGHLIGHTED)
    )
    material = (
        _highlight_material(intent.material, intent.color_source)
        if highlighted
        else replace(intent.material, color_multiplier=(1.0, 1.0, 1.0))
    )
    return ResolvedAppearance(
        visible=visible,
        highlighted=highlighted,
        material=material,
        color_source=intent.color_source,
    )
