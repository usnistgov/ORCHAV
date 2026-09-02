"""CLI validation for scenario YAML files.

``orchav-validate`` exercises the shared scenario loading boundary without
running generation. It parses YAML, validates the shared Pydantic schema,
checks referenced input paths, and can dump a normalized view of the validated
scenario configuration.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from shared.logging import get_logger
from shared.scenarios.defaults import DEFAULT_DEBUG_LEVEL, DEFAULT_RAYTRACING_QUALITY_PRESET
from shared.scenarios.extensions import (
    check_registered_paths,
    registered_extension_keys,
    registered_scene_source_keys,
)
from shared.scenarios.frame_paths import DEFAULT_FRAMES_DIRECTORY
from shared.scenarios.paths import find_project_root, normalize_path, resolve_actor_resource
from shared.scenarios.yaml import load_scenario_yaml, validate_scenario_data

logger = get_logger(__name__)

_TARGET_MESH_SUFFIXES = frozenset({".obj", ".ply", ".stl", ".glb", ".gltf"})
_LOG_LEVEL_NAMES = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _check_referenced_paths(
    data: dict,
    scenario_dir: Path,
    *,
    check_frame_paths: bool = False,
) -> List[str]:
    """Check that file paths referenced in the YAML actually exist.

    Validates paths for scene XML files (local source), HDF5 frame directories,
    and other file references relative to the scenario directory.

    Args:
        data: Parsed scenario YAML dictionary.
        scenario_dir: Directory containing the scenario YAML file.
        check_frame_paths: Whether to require `data.files.directory`
            to exist. Leave false for pre-generation scenario validation.

    Returns:
        List of warning strings for paths that do not exist.
    """
    warnings: List[str] = []
    project_root = find_project_root(scenario_dir)

    # Check scene XML for local source
    scene = data.get("scene", {}) or {}
    if scene.get("source") == "local":
        scene_id = scene.get("id", "")
        if scene_id:
            xml_path = normalize_path(
                scene_id,
                base=scenario_dir,
                project_root=project_root,
            )
            if not xml_path.exists():
                warnings.append(
                    f"scene.id references '{scene_id}' (source=local) "
                    f"but '{xml_path}' does not exist"
                )

    # Check data files directory only when explicitly requested. For generator
    # scenarios this directory is an output path and is expected to be absent
    # before the scenario has been run.
    data_cfg = data.get("data", {}) or {}
    if check_frame_paths and data_cfg.get("mode", "files") == "files":
        files_cfg = data_cfg.get("files", {}) or {}
        directory = Path(files_cfg.get("directory", DEFAULT_FRAMES_DIRECTORY))
        frames_dir = normalize_path(
            directory,
            base=scenario_dir,
            project_root=project_root,
        )
        if not frames_dir.exists():
            warnings.append(
                f"data.files.directory references directory '{directory}' "
                f"but '{frames_dir}' does not exist"
            )

    warnings.extend(check_registered_paths(data, scenario_dir))
    warnings.extend(_check_actor_resource_paths(data, scenario_dir))
    return warnings


def _check_actor_resource_paths(data: dict, scenario_dir: Path) -> List[str]:
    """Return actionable warnings for actor resources unavailable on disk."""

    warnings: List[str] = []
    project_root = find_project_root(scenario_dir)

    def resolved(raw: object, *, confine_to: Path | None = None) -> Path:
        return resolve_actor_resource(
            str(raw),
            scenario_root=scenario_dir,
            project_root=project_root,
            confine_to=confine_to,
        )

    def check_file(raw: object, field: str) -> None:
        resource = resolved(raw)
        if not resource.is_file():
            warnings.append(
                f"{field} references {str(raw)!r}, resolved to '{resource}', " "but no file exists"
            )

    def check_mobility(mobility: object, field: str) -> None:
        if not isinstance(mobility, dict):
            return
        mobility_type = mobility.get("type")
        if mobility_type == "mesh_sequence":
            check_file(mobility.get("positions_path", ""), f"{field}.positions_path")
        elif mobility_type == "network_route":
            graph_path = mobility.get("graph_path") or "street_network.graphml"
            check_file(graph_path, f"{field}.graph_path")

    actors = data.get("actors", {}) or {}
    if isinstance(actors, dict):
        for role in ("tx", "rx", "targets"):
            entries = actors.get(role, ()) or ()
            if not isinstance(entries, list):
                continue
            for index, actor in enumerate(entries):
                if not isinstance(actor, dict):
                    continue
                check_mobility(actor.get("mobility"), f"actors.{role}[{index}].mobility")
                if role != "targets":
                    continue
                asset = actor.get("asset")
                if not isinstance(asset, dict):
                    continue
                source = asset.get("source")
                if source == "catalog":
                    field = f"actors.targets[{index}].asset.id"
                    catalog_root = (project_root / "libraries" / "targets").resolve()
                    try:
                        resource = resolved(asset.get("id", ""), confine_to=catalog_root)
                    except ValueError as exc:
                        warnings.append(f"{field} is invalid: {exc}")
                        continue
                    pattern = None
                else:
                    field = f"actors.targets[{index}].asset.path"
                    resource = resolved(asset.get("path", ""))
                    pattern = asset.get("pattern", "*.ply") if source == "directory" else None

                if source == "file":
                    if not resource.is_file():
                        warnings.append(
                            f"{field} references {str(asset.get('path', ''))!r}, "
                            f"resolved to '{resource}', but no file exists"
                        )
                    elif resource.suffix.lower() not in _TARGET_MESH_SUFFIXES:
                        warnings.append(
                            f"{field} resolved to '{resource}', but its suffix is not a "
                            "supported target mesh format"
                        )
                    continue
                if not resource.is_dir():
                    warnings.append(f"{field} resolved to '{resource}', but no directory exists")
                    continue
                if pattern is None:
                    matches = (
                        path
                        for path in resource.iterdir()
                        if path.is_file() and path.suffix.lower() in _TARGET_MESH_SUFFIXES
                    )
                else:
                    try:
                        matches = (
                            path
                            for path in resource.glob(str(pattern))
                            if path.is_file() and path.suffix.lower() in _TARGET_MESH_SUFFIXES
                        )
                    except (OSError, ValueError) as exc:
                        warnings.append(
                            f"actors.targets[{index}].asset.pattern={pattern!r} "
                            f"cannot be checked in '{resource}': {exc}"
                        )
                        continue
                try:
                    has_match = next(matches, None) is not None
                except OSError as exc:
                    warnings.append(f"{field} cannot be inspected at '{resource}': {exc}")
                    continue
                if not has_match:
                    expected = "a supported mesh" if pattern is None else f"pattern {pattern!r}"
                    warnings.append(
                        f"{field} resolved to '{resource}', but it contains no file matching "
                        f"{expected}"
                    )

    groups = data.get("groups", ()) or ()
    if isinstance(groups, list):
        for index, group in enumerate(groups):
            if isinstance(group, dict):
                check_mobility(group.get("mobility"), f"groups[{index}].mobility")
    return warnings


def validate_scenario(scenario_path: Path, *, strict: bool = False) -> int:
    """Validate a scenario YAML file without running the simulation.

    Performs three levels of validation: YAML parsing, shared schema checks,
    and optional path checks for files referenced by the scenario.

    Args:
        scenario_path: Path to scenario YAML file or directory containing one.
        strict: Require generated frame directories and treat every path
            warning as a validation failure.

    Returns:
        0 on success, 1 on failure.
    """
    # Resolve the actual YAML file path
    if scenario_path.is_dir():
        yaml_path = scenario_path / "scenario.yaml"
    else:
        yaml_path = scenario_path

    scenario_dir = yaml_path.parent

    try:
        data = load_scenario_yaml(scenario_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except yaml.YAMLError as e:
        logger.error("YAML syntax error in %s:\n%s", yaml_path, e)
        return 1
    except ValueError as e:
        logger.error("Validation error: %s", e)
        return 1

    try:
        validate_scenario_data(data)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    path_warnings = _check_referenced_paths(
        data,
        scenario_dir,
        check_frame_paths=strict,
    )
    for warning in path_warnings:
        logger.warning("Path check: %s", warning)

    if path_warnings and strict:
        logger.error(
            "%s: FAILED strict validation (%d path warning(s); "
            "YAML syntax and scenario schema passed)",
            yaml_path,
            len(path_warnings),
        )
    elif path_warnings:
        logger.info(
            "%s: OK with %d path warning(s) "
            "(YAML syntax and scenario schema passed; referenced paths need review)",
            yaml_path,
            len(path_warnings),
        )
    else:
        checked = ["YAML syntax", "scenario schema", "referenced input paths"]
        data_cfg = data.get("data", {}) or {}
        if strict and data_cfg.get("mode", "files") == "files":
            checked.append("file-mode output directory")
        strict_suffix = "; strict" if strict else ""
        logger.info("%s: OK (%s%s)", yaml_path, ", ".join(checked), strict_suffix)

    return 1 if strict and path_warnings else 0


def _without_empty_mappings(value: Any) -> Any:
    """Recursively omit unset values and empty mappings from a YAML value."""

    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned = _without_empty_mappings(item)
            if cleaned == {}:
                continue
            normalized[str(key)] = cleaned
        return normalized
    if isinstance(value, list):
        return [_without_empty_mappings(item) for item in value if item is not None]
    return value


def _without_none(value: Any) -> Any:
    """Recursively omit unset values while preserving authored empty containers."""

    if isinstance(value, dict):
        return {str(key): _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value if item is not None]
    return value


def _normalized_scenario_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a concise user-facing schema view of validated scenario data.

    Pydantic supplies deterministic schema defaults, while this projection
    removes inactive optional sections and provider settings for data modes
    that are not selected. It intentionally does not resolve scene-dependent
    runtime choices such as inherited antenna arrays.
    """

    scenario = validate_scenario_data(data)
    normalized = scenario.model_dump(mode="json", exclude_none=True)

    actors = normalized.get("actors")
    if isinstance(actors, dict):
        for role in ("tx", "rx", "targets"):
            if not actors.get(role):
                actors.pop(role, None)
        if not actors:
            normalized.pop("actors", None)

    if not normalized.get("groups"):
        normalized.pop("groups", None)

    for key in ("raytracing", "coverage", "sensing", "generator_summary"):
        section = normalized.get(key)
        if isinstance(section, dict) and section.get("enabled") is False:
            normalized.pop(key, None)

    raytracing = normalized.get("raytracing")
    if isinstance(raytracing, dict):
        quality = raytracing.setdefault("quality", {})
        if isinstance(quality, dict):
            quality.setdefault("preset", DEFAULT_RAYTRACING_QUALITY_PRESET)
        path_filter = raytracing.get("path_filter")
        if (
            raytracing.get("export_path_metrics") is False
            and isinstance(path_filter, dict)
            and any(
                path_filter.get(key) is not None
                for key in (
                    "relative_threshold_db",
                    "max_path_loss_db",
                    "max_paths_per_pair",
                )
            )
        ):
            raytracing["export_path_metrics"] = True

        # MaterialOverrideModel validates coefficient-only mappings by adding
        # a lambertian pattern, but the runtime intentionally consumes the
        # authored mapping and applies only the coefficient. Preserve that
        # validated authored shape so copying the dump does not change runtime
        # behavior.
        raw_raytracing = data.get("raytracing")
        if isinstance(raw_raytracing, dict) and raw_raytracing.get("materials") is not None:
            raytracing["materials"] = _without_none(raw_raytracing["materials"])

    configured_debug_level = normalized.get("debug_level")
    if configured_debug_level is None:
        normalized["debug_level"] = DEFAULT_DEBUG_LEVEL
    else:
        normalized_debug_level = str(configured_debug_level).upper()
        normalized["debug_level"] = (
            normalized_debug_level
            if normalized_debug_level in _LOG_LEVEL_NAMES
            else DEFAULT_DEBUG_LEVEL
        )

    data_config = normalized.get("data")
    if isinstance(data_config, dict):
        active_mode = data_config.get("mode", "files")
        for mode, section_key in (
            ("files", "files"),
            ("live_grpc", "live_grpc"),
            ("remote_hdf5", "remote_hdf5"),
        ):
            if active_mode != mode:
                data_config.pop(section_key, None)

    cleaned: Dict[str, Any] = _without_empty_mappings(normalized)

    # Registered extension payloads are validated outside ScenarioModel. Keep
    # them in the normalized document rather than silently dropping them,
    # including an authored empty mapping when the extension accepts one.
    for key in registered_extension_keys():
        if key in data:
            cleaned[key] = _without_none(data[key])

    raw_scene = data.get("scene")
    normalized_scene = cleaned.get("scene")
    if isinstance(raw_scene, dict) and isinstance(normalized_scene, dict):
        source = str(raw_scene.get("source", "") or "").strip()
        if source in registered_scene_source_keys() and source in raw_scene:
            normalized_scene[source] = _without_none(raw_scene[source])

    return cleaned


def dump_config(scenario_path: Path, *, explicit_start: bool = False) -> int:
    """Validate a scenario and dump its normalized configuration as YAML.

    Args:
        scenario_path: Path to scenario YAML file or directory containing one.
        explicit_start: Prefix the output with a YAML document marker.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        data = load_scenario_yaml(scenario_path)
        serializable = _normalized_scenario_config(data)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except (yaml.YAMLError, ValueError) as e:
        logger.error("Failed to load scenario: %s", e)
        return 1

    yaml.safe_dump(
        serializable,
        sys.stdout,
        default_flow_style=False,
        explicit_start=explicit_start,
        sort_keys=False,
    )
    return 0


def main() -> None:
    """Entry point for the orchav-validate CLI."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="orchav-validate",
        description="Validate ORCHAV scenario YAML files without running simulations.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="SCENARIO",
        help="Path(s) to scenario YAML file(s) or directories containing scenario.yaml",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat path warnings as errors, including missing frame directories "
            "referenced by data.files.directory"
        ),
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Validate scenario and dump normalized configuration YAML to stdout",
    )
    args = parser.parse_args()

    if args.dump_config:
        exit_code = 0
        dumped_count = 0
        for path in args.paths:
            result = dump_config(path.resolve(), explicit_start=dumped_count > 0)
            if result != 0:
                exit_code = 1
            else:
                dumped_count += 1
        sys.exit(exit_code)

    exit_code = 0
    for path in args.paths:
        result = validate_scenario(path.resolve(), strict=args.strict)
        if result != 0:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
