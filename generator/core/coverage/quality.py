#!/usr/bin/env python3
"""Coverage quality profile helper.

Coverage uses Sionna RT's ``RadioMapSolver``, whose sample-count parameter is
``samples_per_tx``. The rest of the generator's ray-tracing presets use
``samples_per_src``. This module keeps that vocabulary bridge and the override
precedence in one place:

1. Start from the coverage quality preset.
2. Let ``coverage.quality.preset`` or ``coverage.solver.preset`` choose a
   coverage-specific preset.
3. Merge custom coverage overrides, with ``coverage.solver.custom`` taking
   precedence over ``coverage.quality.custom``.
"""

from typing import Any

from generator.core.configuration.defaults import DEFAULT_COVERAGE_QUALITY_PRESET


class CoverageQuality:
    """Derive RadioMapSolver arguments from quality presets and overrides.

    Coverage quality is intentionally separate from per-frame ray-tracing
    quality. A scenario may need a coarse radio map with high-fidelity frames,
    or a dense coverage solve with cheaper frame ray tracing.

    ``coverage.solver.preset`` overrides the simulation quality preset when
    present. ``coverage.solver.custom`` is merged on top of
    ``coverage.quality.custom``.
    """

    def __init__(
        self,
        profile: dict[str, Any],
        preset_override: str | None = None,
        custom: dict[str, Any] | None = None,
    ):
        self.profile = dict(profile or {})
        self.preset_override = preset_override
        self.custom = dict(custom or {})

    @classmethod
    def from_context(cls, simulation_config, scenario_context=None):
        """Build coverage quality settings from simulation and scenario context."""
        coverage_config = getattr(simulation_config, "coverage", None)
        preset = getattr(coverage_config, "quality", DEFAULT_COVERAGE_QUALITY_PRESET)
        custom = None
        if scenario_context is not None:
            cov_cfg = getattr(scenario_context, "coverage_cfg", None)
            if cov_cfg:
                solver_cfg = cov_cfg.get("solver", {}) or {}
                quality_cfg = cov_cfg.get("quality", {}) or {}
                if solver_cfg.get("preset"):
                    preset = str(solver_cfg["preset"])
                elif quality_cfg.get("preset"):
                    preset = str(quality_cfg["preset"])
                custom = {
                    **(quality_cfg.get("custom", {}) or {}),
                    **(solver_cfg.get("custom", {}) or {}),
                }
                if not custom:
                    custom = None
        profile = simulation_config.QUALITY_PRESETS.get(
            preset,
            simulation_config.QUALITY_PRESETS[DEFAULT_COVERAGE_QUALITY_PRESET],
        )
        return cls(profile=profile, preset_override=preset, custom=custom)

    def to_radio_map_args(self) -> dict[str, Any]:
        """Return keyword arguments accepted by Sionna RT ``RadioMapSolver``."""
        args = {
            "samples_per_tx": int(self.profile.get("samples_per_src", 100000)),
            "max_depth": int(self.profile.get("max_depth", 3)),
            "los": bool(self.profile.get("los", True)),
            "specular_reflection": bool(self.profile.get("specular_reflection", True)),
            "diffuse_reflection": bool(self.profile.get("diffuse_reflection", True)),
            "refraction": bool(self.profile.get("refraction", True)),
            "diffraction": bool(self.profile.get("diffraction", False)),
        }
        # Accept both RadioMapSolver and ray-tracing preset spellings so coverage
        # custom blocks can reuse the generator's quality vocabulary.
        if self.custom:
            if "samples_per_tx" in self.custom:
                args["samples_per_tx"] = int(self.custom["samples_per_tx"])
            elif "samples_per_src" in self.custom:
                args["samples_per_tx"] = int(self.custom["samples_per_src"])
            if "max_depth" in self.custom:
                args["max_depth"] = int(self.custom["max_depth"])
            for k in [
                "specular_reflection",
                "diffuse_reflection",
                "refraction",
                "diffraction",
                "los",
            ]:
                if k in self.custom:
                    args[k] = bool(self.custom[k])
        return args
