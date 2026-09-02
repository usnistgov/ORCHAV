"""Repository-local optional scenario extension hooks.

The core scenario schema stays limited to documented ORCHAV fields. Optional
feature packages can register additional top-level sections without expanding
the default scenario contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from shared.logging import get_logger

logger = get_logger(__name__)

ValidateHook = Callable[[Any], None]
ParseHook = Callable[[dict[str, Any], Path], Any]
ApplyHook = Callable[[Any, Any], None]
PathCheckHook = Callable[[dict[str, Any], Path], list[str]]
SceneValidateHook = Callable[[dict[str, Any]], None]
SceneParseHook = Callable[[dict[str, Any], Path, Path], tuple[str, str, Path | None]]


@dataclass(frozen=True)
class ScenarioExtension:
    """Hooks for one optional top-level scenario section."""

    key: str
    validate: ValidateHook | None = None
    parse: ParseHook | None = None
    apply: ApplyHook | None = None
    check_paths: PathCheckHook | None = None


@dataclass(frozen=True)
class SceneSourceExtension:
    """Hooks for one optional scene source."""

    source: str
    parse: SceneParseHook
    validate: SceneValidateHook | None = None


_EXTENSIONS: dict[str, ScenarioExtension] = {}
_SCENE_SOURCES: dict[str, SceneSourceExtension] = {}
_PRIVATE_LOADED = False


def register_scenario_extension(extension: ScenarioExtension) -> None:
    """Register or replace one optional scenario extension."""

    key = str(extension.key).strip()
    if not key:
        raise ValueError("Scenario extension key must be non-empty")
    _EXTENSIONS[key] = extension


def register_scene_source_extension(extension: SceneSourceExtension) -> None:
    """Register or replace one optional scene-source resolver."""

    source = str(extension.source).strip()
    if not source:
        raise ValueError("Scene source key must be non-empty")
    _SCENE_SOURCES[source] = extension


def load_private_scenario_extensions() -> None:
    """Attempt repository-local registrations once and tolerate their absence."""

    global _PRIVATE_LOADED
    if _PRIVATE_LOADED:
        return
    _PRIVATE_LOADED = True
    try:
        from . import _private_extensions
    except ImportError:
        return
    try:
        _private_extensions.register()
    except Exception as exc:  # noqa: BLE001 - extension boundary; pragma: no cover
        logger.debug("Optional scenario extensions failed to load: %s", exc)


def registered_extension_keys() -> set[str]:
    """Return registered top-level extension keys."""

    load_private_scenario_extensions()
    return set(_EXTENSIONS)


def registered_scene_source_keys() -> set[str]:
    """Return registered optional scene-source keys."""

    load_private_scenario_extensions()
    return set(_SCENE_SOURCES)


def validate_registered_extensions(data: dict[str, Any]) -> None:
    """Validate registered extension sections present in raw scenario data."""

    load_private_scenario_extensions()
    for key, extension in _EXTENSIONS.items():
        if key not in data or extension.validate is None:
            continue
        extension.validate(data[key])


def strip_registered_extensions(data: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy without registered extension sections."""

    load_private_scenario_extensions()
    return {key: value for key, value in data.items() if key not in _EXTENSIONS}


def parse_registered_extensions(data: dict[str, Any], scenario_root: Path) -> dict[str, Any]:
    """Parse registered extension sections for a loaded scenario."""

    load_private_scenario_extensions()
    parsed: dict[str, Any] = {}
    for key, extension in _EXTENSIONS.items():
        if extension.parse is not None:
            parsed[key] = extension.parse(data, scenario_root)
        elif key in data:
            parsed[key] = data[key]
    return parsed


def apply_registered_extensions(config: Any, parsed: dict[str, Any]) -> None:
    """Apply parsed extension data to a ScenarioConfiguration object."""

    load_private_scenario_extensions()
    for key, value in parsed.items():
        extension = _EXTENSIONS.get(key)
        if extension is not None and extension.apply is not None:
            extension.apply(config, value)


def check_registered_paths(data: dict[str, Any], scenario_dir: Path) -> list[str]:
    """Run path checks contributed by registered extensions."""

    load_private_scenario_extensions()
    warnings: list[str] = []
    for extension in _EXTENSIONS.values():
        if extension.check_paths is not None:
            warnings.extend(extension.check_paths(data, scenario_dir))
    return warnings


def validate_registered_scene_source(scene: dict[str, Any]) -> None:
    """Validate an optional scene-source extension block."""

    load_private_scenario_extensions()
    source = str(scene.get("source", "") or "").strip()
    extension = _SCENE_SOURCES.get(source)
    if extension is not None and extension.validate is not None:
        extension.validate(scene)


def strip_registered_scene_source_sections(data: dict[str, Any]) -> dict[str, Any]:
    """Return data with the registered scene-source payload removed."""

    load_private_scenario_extensions()
    scene = data.get("scene")
    if not isinstance(scene, dict):
        return data
    source = str(scene.get("source", "") or "").strip()
    if source not in _SCENE_SOURCES or source not in scene:
        return data

    stripped = dict(data)
    stripped_scene = dict(scene)
    stripped_scene.pop(source, None)
    stripped["scene"] = stripped_scene
    return stripped


def parse_registered_scene_source(
    scene: dict[str, Any],
    scenario_root: Path,
    project_root: Path,
) -> tuple[str, str, Path | None] | None:
    """Parse a scene handled by a registered source extension."""

    load_private_scenario_extensions()
    source = str(scene.get("source", "") or "").strip()
    extension = _SCENE_SOURCES.get(source)
    if extension is None:
        return None
    return extension.parse(scene, scenario_root, project_root)
