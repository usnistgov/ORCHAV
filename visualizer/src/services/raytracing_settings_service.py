"""Shared raytracing solver settings for live controls and preview solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from generator.core.configuration import QUALITY_PRESETS
from shared.scenarios.defaults import DEFAULT_RAYTRACING_QUALITY_PRESET

from .base import BaseService

DEFAULT_RAYTRACING_PRESET = DEFAULT_RAYTRACING_QUALITY_PRESET

_DRAG_MAX_DEPTH_CAP = 2
_DRAG_SAMPLES_CAP = 4096
_DRAG_PATHS_CAP = 30000
_DRAG_SAMPLES_DIVISOR = 100
_DRAG_PATHS_DIVISOR = 10
_MIN_POSITIVE_BUDGET = 1


@dataclass(frozen=True)
class RaytracingSettings:
    """Normalized Sionna RT solver settings used by visualizer controls."""

    max_depth: int
    samples_per_src: int
    max_num_paths_per_src: int
    seed: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    synthetic_array: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> "RaytracingSettings":
        """Build normalized settings from a preset/custom mapping."""
        source = dict(defaults or QUALITY_PRESETS[DEFAULT_RAYTRACING_PRESET])
        source.update(dict(values or {}))
        return cls(
            max_depth=_coerce_int(source.get("max_depth"), default=4, minimum=1),
            samples_per_src=_coerce_int(
                source.get("samples_per_src"),
                default=100_000,
                minimum=_MIN_POSITIVE_BUDGET,
            ),
            max_num_paths_per_src=_coerce_int(
                source.get("max_num_paths_per_src"),
                default=100_000,
                minimum=_MIN_POSITIVE_BUDGET,
            ),
            seed=_coerce_int(source.get("seed"), default=42, minimum=0),
            los=bool(source.get("los", True)),
            specular_reflection=bool(source.get("specular_reflection", True)),
            diffuse_reflection=bool(source.get("diffuse_reflection", True)),
            refraction=bool(source.get("refraction", False)),
            diffraction=bool(source.get("diffraction", False)),
            synthetic_array=bool(source.get("synthetic_array", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/protobuf-friendly settings dictionary."""
        return {
            "max_depth": int(self.max_depth),
            "samples_per_src": int(self.samples_per_src),
            "max_num_paths_per_src": int(self.max_num_paths_per_src),
            "seed": int(self.seed),
            "los": bool(self.los),
            "specular_reflection": bool(self.specular_reflection),
            "diffuse_reflection": bool(self.diffuse_reflection),
            "refraction": bool(self.refraction),
            "diffraction": bool(self.diffraction),
            "synthetic_array": bool(self.synthetic_array),
        }

    def scaled_for_drag(self) -> "RaytracingSettings":
        """Return responsive drag settings derived from the release budget."""
        return RaytracingSettings(
            max_depth=max(1, min(int(self.max_depth), _DRAG_MAX_DEPTH_CAP)),
            samples_per_src=_scaled_budget(
                self.samples_per_src,
                divisor=_DRAG_SAMPLES_DIVISOR,
                cap=_DRAG_SAMPLES_CAP,
            ),
            max_num_paths_per_src=_scaled_budget(
                self.max_num_paths_per_src,
                divisor=_DRAG_PATHS_DIVISOR,
                cap=_DRAG_PATHS_CAP,
            ),
            seed=self.seed,
            los=self.los,
            specular_reflection=self.specular_reflection,
            diffuse_reflection=self.diffuse_reflection,
            refraction=self.refraction,
            diffraction=self.diffraction,
            synthetic_array=self.synthetic_array,
        )


class RaytracingSettingsService(BaseService):
    """Own the current solver settings shared by raytracing UI and preview."""

    def __init__(self, *, default_preset: str = DEFAULT_RAYTRACING_PRESET) -> None:
        """Initialize from a known generator preset."""
        super().__init__()
        self._preset_name = DEFAULT_RAYTRACING_PRESET
        self._release_settings = RaytracingSettings.from_mapping(
            QUALITY_PRESETS[DEFAULT_RAYTRACING_PRESET]
        )
        self.set_preset(default_preset)

    @property
    def current_preset(self) -> str:
        """Return the current preset label, including ``custom``."""
        return self._preset_name

    def set_preset(self, preset_name: str) -> dict[str, Any]:
        """Select a named generator preset and return normalized settings."""
        preset = str(preset_name or DEFAULT_RAYTRACING_PRESET)
        if preset not in QUALITY_PRESETS:
            preset = DEFAULT_RAYTRACING_PRESET
        self._preset_name = preset
        self._release_settings = RaytracingSettings.from_mapping(QUALITY_PRESETS[preset])
        return self.release_settings()

    def set_custom(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Select custom settings and return their normalized values."""
        self._preset_name = "custom"
        self._release_settings = RaytracingSettings.from_mapping(settings)
        return self.release_settings()

    def release_settings(self) -> dict[str, Any]:
        """Return exact settings for release/recompute preview solves."""
        return self._release_settings.to_dict()

    def drag_settings(self) -> dict[str, Any]:
        """Return lower-cost settings for continuous drag preview solves."""
        return self._release_settings.scaled_for_drag().to_dict()


def _coerce_int(value: Any, *, default: int, minimum: int) -> int:
    """Coerce an integer field while enforcing a lower bound."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), parsed)


def _scaled_budget(value: int, *, divisor: int, cap: int) -> int:
    """Scale a solver budget without exceeding the selected release value."""
    release = max(_MIN_POSITIVE_BUDGET, int(value))
    scaled = max(_MIN_POSITIVE_BUDGET, release // max(1, int(divisor)))
    return max(_MIN_POSITIVE_BUDGET, min(release, int(cap), scaled))
