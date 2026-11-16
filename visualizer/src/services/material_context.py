"""Explicit renderer-independent inputs to the shared material service."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MaterialContext:
    """Appearance settings captured before a scene preparation worker starts."""

    building_appearance: str = "original"
    facade_theme: str = "mixed"
    road_appearance: str = "original"
    authored_surface_textures: bool = True
    current_building_alpha: float = 1.0
    current_target_alpha: float = 1.0

    @classmethod
    def from_defaults(cls, defaults: Mapping[str, Any]) -> "MaterialContext":
        return cls(
            **{name: defaults[name] for name in cls.__dataclass_fields__ if name in defaults}
        )
