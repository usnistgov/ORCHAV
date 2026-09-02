"""Scenario and application configuration loading for the visualizer."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from shared.grpc_transport import format_grpc_endpoint, parse_grpc_endpoint
from shared.logging import get_logger
from shared.scenarios.actors import ActorsSpec, GroupSpec, TimelineSpec
from shared.scenarios.defaults import DEFAULT_DEBUG_LEVEL
from shared.scenarios.paths import find_project_root as find_scenario_project_root
from shared.scenarios.paths import normalize_path
from shared.scenarios.yaml import validate_scenario_data

logger = get_logger("orchav")

# Defaults are environment-overridable so scenario YAML can omit local endpoints.
DEFAULT_GRPC_SIONNA = os.environ.get("ORCHAV_GRPC_SIONNA", "grpc://localhost:50051")
DEFAULT_REMOTE_HDF5_SERVER = os.environ.get("ORCHAV_REMOTE_HDF5_SERVER", "localhost:50052")
_BUILTIN_DATA_MODES = frozenset({"files", "live_grpc", "remote_hdf5"})


def _endpoint_with_port(endpoint: str, port: int, *, include_scheme: bool) -> str:
    """Replace only the port in a validated gRPC endpoint."""
    normalized_port = int(port)
    if not 1 <= normalized_port <= 65535:
        raise ValueError("gRPC port override must be between 1 and 65535")
    host, _ = parse_grpc_endpoint(endpoint)
    address = format_grpc_endpoint(host, normalized_port)
    return f"grpc://{address}" if include_scheme else address


@dataclass
class AppConfig:
    """Application configuration with sensible defaults."""

    scenes: Path
    targets: Path
    scenarios: Path
    output: Path
    live_grpc: dict[str, str]

    @classmethod
    def get_defaults(cls, project_root: Optional[Path] = None) -> AppConfig:
        """Return default configuration with absolute paths."""
        if project_root is None:
            from shared.scenarios.paths import find_project_root

            project_root = find_project_root(Path.cwd())

        return cls(
            scenes=project_root / "libraries" / "scenes",
            targets=project_root / "libraries" / "targets",
            scenarios=project_root / "scenarios",
            output=project_root / "output",
            live_grpc={"sionna": DEFAULT_GRPC_SIONNA},
        )


@dataclass
class Scenario:
    """Validated scenario data adapted for visualizer services."""

    root: Path
    scene_spec: dict[str, Any]
    data_mode: str
    data_spec: dict[str, Any]
    view_defaults: dict[str, Any]
    timeline: TimelineSpec
    scene_id: str = "default"
    scene_source: str = "library"
    visualizer_cfg: dict[str, Any] = field(default_factory=dict)
    live_grpc_endpoints: dict[str, str] = field(default_factory=dict)
    debug_level: str = DEFAULT_DEBUG_LEVEL
    actors: ActorsSpec = field(default_factory=ActorsSpec)
    groups: tuple[GroupSpec, ...] = ()
    sensing: dict[str, Any] = None  # type: ignore[assignment]
    raytracing: dict[str, Any] = None  # type: ignore[assignment]

    @property
    def frames_dir(self) -> Path:
        """Return the configured file-frame directory resolved from the scenario."""
        files_spec = self.data_spec.get("files")
        directory = (
            files_spec.get("directory", "frames") if isinstance(files_spec, dict) else "frames"
        )
        return normalize_path(
            str(directory),
            base=self.root,
            project_root=find_scenario_project_root(self.root),
        )


def find_project_root() -> Path:
    """Return the project root using the shared scenario path policy."""
    from shared.scenarios.paths import find_project_root as _find_project_root

    return _find_project_root(Path.cwd())


def load_app_config(path: Optional[Path] = None) -> AppConfig:
    """Load application configuration from TOML or return defaults."""
    if path is None:
        project_root = find_project_root()
        path = project_root / "config" / "app.toml"
        logger.info("Looking for config at: %s", path)

    if not path.exists():
        logger.info("No config file found, using defaults")
        config = AppConfig.get_defaults()
    else:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            paths = data.get("paths", {})
            live_grpc = data.get("live_grpc", {})

            config = AppConfig(
                scenes=Path(paths.get("scenes", "libraries/scenes")),
                targets=Path(paths.get("targets", "libraries/targets")),
                scenarios=Path(paths.get("scenarios", "scenarios")),
                output=Path(paths.get("output", "output")),
                live_grpc=live_grpc,
            )
            logger.info("Loaded config from: %s", path)

        except (OSError, ValueError, KeyError):
            logger.warning(
                "Failed to load config from %s, using defaults",
                path,
                exc_info=True,
            )
            config = AppConfig.get_defaults()

    # App config stores project-relative paths; runtime services use absolutes.
    project_root = find_project_root()
    config.scenes = (project_root / config.scenes).resolve()
    config.targets = (project_root / config.targets).resolve()
    config.scenarios = (project_root / config.scenarios).resolve()
    config.output = (project_root / config.output).resolve()

    return config


def load_scenario(
    path: Path,
    app: AppConfig,
    *,
    data_mode_override: str | None = None,
    grpc_port_override: int | None = None,
) -> Scenario:
    """Load a scenario and apply optional process-local data-source overrides."""
    if path.is_dir():
        yaml_path = path / "scenario.yaml"
    else:
        yaml_path = path

    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {yaml_path}")

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Scenario YAML root must be a mapping")

        scenario_model = validate_scenario_data(data)

        scene_spec = data.get("scene", {}) if isinstance(data.get("scene"), dict) else {}
        if scenario_model.scene is not None:
            scene_spec = dict(scene_spec)
            scene_spec["source"] = scenario_model.scene.source
            scene_source = scenario_model.scene.source
        else:
            scene_source = "library"
        scene_id = scene_spec.get("id")
        if scene_source == "local" and isinstance(scene_id, str) and "${PROJECT_ROOT}" in scene_id:
            scene_spec = dict(scene_spec)
            scene_spec["id"] = str(
                normalize_path(
                    scene_id,
                    base=yaml_path.parent,
                    project_root=find_scenario_project_root(yaml_path.parent),
                )
            )
        raw_data_spec = data.get("data", {})
        data_spec = dict(raw_data_spec) if isinstance(raw_data_spec, dict) else {}
        data_spec.setdefault("mode", scenario_model.data.mode)
        if data_mode_override is not None:
            if data_mode_override not in _BUILTIN_DATA_MODES:
                raise ValueError(f"Unsupported visualizer data mode: {data_mode_override}")
            data_spec["mode"] = data_mode_override
        if data_spec["mode"] == "files":
            files_spec = scenario_model.data.files.model_dump()
            if files_spec["format"] == "hdf5":
                files_spec["format"] = "h5"
            data_spec["files"] = files_spec
        view_defaults = (
            data.get("view_defaults", {}) if isinstance(data.get("view_defaults"), dict) else {}
        )

        visualizer_cfg_raw = data.get("visualizer", {})
        visualizer_cfg = visualizer_cfg_raw if isinstance(visualizer_cfg_raw, dict) else {}

        live_grpc_endpoints: dict[str, str] = {}
        if isinstance(app.live_grpc, dict):
            for key, value in app.live_grpc.items():
                if value:
                    live_grpc_endpoints[str(key)] = str(value)

        root_overrides = data.get("live_grpc_endpoints", {})
        if isinstance(root_overrides, dict):
            for key, value in root_overrides.items():
                if value:
                    live_grpc_endpoints[str(key)] = str(value)

        data_overrides = data_spec.get("live_grpc_endpoints", {})
        if isinstance(data_overrides, dict):
            for key, value in data_overrides.items():
                if value:
                    live_grpc_endpoints[str(key)] = str(value)

        live_grpc_spec_value = data_spec.get("live_grpc")
        live_grpc_spec = live_grpc_spec_value if isinstance(live_grpc_spec_value, dict) else {}
        live_grpc_endpoint = live_grpc_spec.get("endpoint")
        if live_grpc_endpoint:
            live_grpc_endpoints["sionna"] = str(live_grpc_endpoint)

        if grpc_port_override is not None:
            if data_spec["mode"] == "live_grpc":
                endpoint = (
                    live_grpc_spec.get("endpoint")
                    or live_grpc_endpoints.get("sionna")
                    or DEFAULT_GRPC_SIONNA
                )
                endpoint = _endpoint_with_port(
                    str(endpoint),
                    grpc_port_override,
                    include_scheme=True,
                )
                live_grpc_spec = dict(live_grpc_spec)
                live_grpc_spec["endpoint"] = endpoint
                data_spec["live_grpc"] = live_grpc_spec
                live_grpc_endpoints["sionna"] = endpoint
            elif data_spec["mode"] == "remote_hdf5":
                remote_value = data_spec.get("remote_hdf5")
                remote_spec = dict(remote_value) if isinstance(remote_value, dict) else {}
                server = remote_spec.get("server") or DEFAULT_REMOTE_HDF5_SERVER
                remote_spec["server"] = _endpoint_with_port(
                    str(server),
                    grpc_port_override,
                    include_scheme=False,
                )
                data_spec["remote_hdf5"] = remote_spec
            else:
                raise ValueError("gRPC port override requires live_grpc or remote_hdf5 data mode")

        scenario = Scenario(
            root=path if path.is_dir() else path.parent,
            scene_spec=scene_spec,
            data_mode=str(data_spec["mode"]),
            data_spec=data_spec,
            view_defaults=view_defaults,
            scene_id=str(scene_spec.get("id", "default")),
            scene_source=scene_source,
            visualizer_cfg=visualizer_cfg,
            live_grpc_endpoints=live_grpc_endpoints,
            debug_level=data.get("debug_level", DEFAULT_DEBUG_LEVEL),
            actors=scenario_model.actors,
            groups=scenario_model.groups,
            timeline=scenario_model.timeline,
            sensing=data.get("sensing", {}) or {},
            raytracing=data.get("raytracing", {}) or {},
        )

        logger.info("Loaded scenario from: %s", yaml_path)
        return scenario

    except (OSError, ValueError, KeyError) as e:
        logger.error("Failed to load scenario from %s: %s", yaml_path, e)
        raise


def resolve_scene_meshes(scene_spec: dict[str, Any], app: AppConfig) -> list[Path]:
    """Resolve mesh files referenced by a scene specification."""
    source = scene_spec.get("source", "library")
    mesh_extensions = [".obj", ".ply", ".stl", ".glb", ".gltf"]

    if source == "library":
        scene_id = scene_spec.get("id", "default")
        scene_path = app.scenes / scene_id
    elif source == "local":
        scene_path = Path(scene_spec.get("path", "."))
    elif source == "sionna":
        from shared.scenarios.parsers import resolve_sionna_scene_xml

        scene_id = scene_spec.get("id", "default")
        xml_path = resolve_sionna_scene_xml(scene_id)
        if xml_path is None:
            logger.info("Sionna built-in scene requested but package not found")
            return []
        scene_path = xml_path.parent
    else:
        raise ValueError(f"Unknown scene source: {source}")

    if not scene_path.exists():
        logger.warning("Scene path does not exist: %s", scene_path)
        return []

    mesh_files = []
    meshes_dir = scene_path / "meshes"

    if meshes_dir.exists():
        for ext in mesh_extensions:
            mesh_files.extend(meshes_dir.rglob(f"*{ext}"))

    for ext in mesh_extensions:
        mesh_files.extend(scene_path.glob(f"*{ext}"))

    mesh_files = sorted(set(mesh_files))
    logger.debug("Found %d mesh files in scene: %s", len(mesh_files), scene_path)

    return mesh_files
