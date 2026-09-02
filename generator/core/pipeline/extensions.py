"""Optional material-tuning hook for the generator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from shared.logging import get_logger

logger = get_logger(__name__)


class MaterialTuningAdapter(Protocol):
    """Adapter protocol for optional material-parameter tuning support.

    Adapters receive prepared ``SimulationObjects`` from ``RayTracingService``.
    They should not construct their own scene or timeline state.
    """

    @property
    def available(self) -> bool:
        """Return whether runtime support is available."""
        ...

    def build_config(self, scenario_configuration: Any) -> Any:
        """Build a material-tuning config object from scenario data."""

    def run(self, simulation_objects: Any, material_tuning_config: Any) -> None:
        """Run material tuning against prepared simulation objects."""


@dataclass
class _DisabledMaterialTuningConfig:
    enabled: bool = False


class _NoMaterialTuningAdapter:
    """Default adapter used when no extension is installed."""

    available = False

    def build_config(self, scenario_configuration: Any) -> _DisabledMaterialTuningConfig:
        raw_cfg = getattr(scenario_configuration, "material_tuning_cfg", None)
        if isinstance(raw_cfg, dict) and raw_cfg.get("enabled"):
            raise RuntimeError(
                "Material tuning is not available in this ORCHAV installation. "
                "Install or enable the required extension before using this scenario."
            )
        return _DisabledMaterialTuningConfig()

    def run(self, simulation_objects: Any, material_tuning_config: Any) -> None:
        raise RuntimeError("Material tuning is not available in this ORCHAV installation.")


_material_tuning_adapter: MaterialTuningAdapter = _NoMaterialTuningAdapter()
_PRIVATE_LOADED = False


def register_material_tuning_adapter(adapter: MaterialTuningAdapter) -> None:
    """Register the active material-parameter tuning adapter."""

    global _material_tuning_adapter
    _material_tuning_adapter = adapter


def load_private_pipeline_extensions() -> None:
    """Load private pipeline extension registrations when available."""

    global _PRIVATE_LOADED
    if _PRIVATE_LOADED:
        return
    module_name = f"{__package__}._private_extensions"
    try:
        private_extensions = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        _PRIVATE_LOADED = True
        return

    try:
        private_extensions.register()
    except Exception as exc:  # noqa: BLE001 - extension boundary; pragma: no cover
        logger.debug("Optional pipeline extensions failed to load: %s", exc)
    _PRIVATE_LOADED = True


def get_material_tuning_adapter() -> MaterialTuningAdapter:
    """Return the active material-parameter tuning adapter."""

    load_private_pipeline_extensions()
    return _material_tuning_adapter
