"""Load scenario YAML into the shared ``ScenarioConfiguration`` contract.

The loader owns schema validation, path resolution, and extraction of the
scenario sections that multiple applications need. Generator-specific runtime
dataclasses are built later in ``generator.core.configuration``; visualizer
settings are consumed by visualizer code. Keeping this layer shared prevents
each application from reparsing YAML with slightly different rules.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging import get_logger
from shared.scenarios.actors import ActorsSpec, GroupSpec, TimelineSpec
from shared.scenarios.model import ScenarioModel
from shared.scenarios.paths import create_path_policy, normalize_path

from .app_paths import load_live_grpc_endpoints
from .extensions import apply_registered_extensions, parse_registered_extensions
from .parsers import (
    parse_coverage_config,
    parse_data_config,
    parse_generator_summary_config,
    parse_raytracing_config,
    parse_scene_config,
)
from .yaml import load_scenario_yaml, validate_scenario_data

logger = get_logger(__name__)


@dataclass
class ScenarioConfiguration:
    """Validated scenario data plus paths resolved relative to the scenario.

    Actor poses are exposed through immutable shared specifications.  Other
    sections remain dictionaries for application-specific adapters.
    """

    root: Path  # Resolved scenario root directory
    project_root: Path  # Resolved ORCHAV project root used by the path policy
    scene_xml: Optional[Path]  # Path to scene XML file
    targets_dir: Path  # Directory containing target meshes
    frames_directory: str  # Authored data.files.directory value
    frames_dir: Path  # Resolved configured frame location
    frames_pattern: str  # Basename-only read filter (e.g., "mpc_frames_*.h5")
    frames_format: str  # Format for frames (h5)
    chunk_size: int  # Frames per output chunk
    compression: str  # HDF5 compression profile (lzf/balanced, gzip-4/compact, none)
    scene_id: str  # Scene identifier
    scene_source: str  # Built-in or registered scene-source key
    data_mode: str  # Data mode: 'files', 'live_grpc', or 'remote_hdf5'
    view_defaults: Dict[str, Any]  # View configuration defaults
    timeline: TimelineSpec
    actors: ActorsSpec
    groups: tuple[GroupSpec, ...]
    raytracing: Dict[str, Any]  # Raytracing configuration
    coverage_cfg: Dict[str, Any]  # Coverage configuration
    generator_summary: Dict[str, Any]  # Generator summary config (optional)
    summary_yaml_hash: str  # Normalized authored YAML identity for summary reuse
    visualizer_cfg: Dict[str, Any]  # Visualizer overrides (panel toggles, defaults)
    debug_level: Optional[str]  # Logging level (e.g., DEBUG, INFO, WARNING) if explicitly set
    live_grpc_endpoints: Dict[str, Any]  # Endpoints loaded from app.toml [live_grpc]
    scene_source_config: Optional[Dict[str, Any]] = None


def load_scenario(scenario_path: Path) -> ScenarioModel:
    """Load one scenario YAML document into the immutable public model."""
    data = load_scenario_yaml(Path(scenario_path))
    return validate_scenario_data(data)


def _summary_yaml_hash(scenario_data: Dict[str, Any]) -> str:
    """Hash normalized authored YAML, excluding the one-shot summary force flag."""
    normalized = deepcopy(scenario_data)
    summary = normalized.get("generator_summary")
    if isinstance(summary, dict):
        summary.pop("force", None)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_scenario_configuration(
    scenario_path: Path,
    project_root: Optional[Path] = None,
    *,
    yaml_path: Optional[Path] = None,
) -> ScenarioConfiguration:
    """
    Load and resolve all paths for a scenario.

    Args:
        scenario_path: Path to scenario folder or .yaml file.
        project_root: Optional project root path (will be auto-detected if None).
        yaml_path: Explicit YAML file to load instead of the default
            ``scenario_path / "scenario.yaml"``.  The scenario root is still
            derived from *scenario_path* (or its parent if it points to a file),
            so scenario-relative asset and reader paths resolve from that
            directory.

    Returns:
        ScenarioConfiguration with validated models and resolved paths.

    Raises:
        FileNotFoundError: If scenario file not found
        ValueError: If scenario configuration is invalid
    """
    if yaml_path is not None:
        resolved_yaml = Path(yaml_path)
    elif scenario_path.is_dir():
        resolved_yaml = scenario_path / "scenario.yaml"
    else:
        resolved_yaml = scenario_path

    policy = create_path_policy(resolved_yaml, project_root_override=project_root)
    project_root_path = policy.project_root
    scenario_data = load_scenario_yaml(resolved_yaml)
    scenario = validate_scenario_data(scenario_data)

    if scenario_path.is_dir():
        scenario_root = scenario_path.resolve()
    else:
        scenario_root = scenario_path.parent.resolve()

    live_grpc_endpoints = load_live_grpc_endpoints(project_root_path)
    root_live_grpc = scenario_data.get("live_grpc_endpoints", {})
    if isinstance(root_live_grpc, dict):
        live_grpc_endpoints.update({str(k): str(v) for k, v in root_live_grpc.items() if v})
    data_spec = scenario_data.get("data", {}) or {}
    data_live_grpc_endpoints = data_spec.get("live_grpc_endpoints", {})
    if isinstance(data_live_grpc_endpoints, dict):
        live_grpc_endpoints.update(
            {str(k): str(v) for k, v in data_live_grpc_endpoints.items() if v}
        )
    data_live_grpc = data_spec.get("live_grpc", {})
    if isinstance(data_live_grpc, dict) and data_live_grpc.get("endpoint"):
        live_grpc_endpoints["sionna"] = str(data_live_grpc["endpoint"])

    scene_id, scene_source, scene_xml = parse_scene_config(
        scenario_data, scenario_root, project_root_path
    )

    targets_dir = normalize_path("libraries/targets", base=project_root_path)
    data_cfg = parse_data_config(
        scenario_data,
        scenario_root,
        project_root=project_root_path,
    )

    raytracing = parse_raytracing_config(scenario_data)
    generator_summary = parse_generator_summary_config(scenario_data, scenario_root)
    coverage_cfg = parse_coverage_config(scenario_data, scenario_root)
    extension_cfg = parse_registered_extensions(scenario_data, scenario_root)

    _dl = scenario_data.get("debug_level", None)
    debug_level = str(_dl).upper() if _dl is not None else None

    config = ScenarioConfiguration(
        root=scenario_root,
        project_root=project_root_path.resolve(),
        scene_xml=scene_xml,
        targets_dir=targets_dir,
        frames_directory=data_cfg["frames_directory"],
        frames_dir=data_cfg["frames_dir"],
        frames_pattern=data_cfg["frames_pattern"],
        frames_format=data_cfg["frames_format"],
        chunk_size=data_cfg["chunk_size"],
        compression=data_cfg["compression"],
        scene_id=scene_id,
        scene_source=scene_source,
        data_mode=data_cfg["mode"],
        view_defaults=scenario_data.get("view_defaults", {}),
        timeline=scenario.timeline,
        actors=scenario.actors,
        groups=scenario.groups,
        generator_summary=generator_summary,
        summary_yaml_hash=_summary_yaml_hash(scenario_data),
        visualizer_cfg=scenario_data.get("visualizer", {}) or {},
        raytracing=raytracing,
        coverage_cfg=coverage_cfg,
        debug_level=debug_level,
        live_grpc_endpoints=live_grpc_endpoints,
        scene_source_config=(
            (scenario_data.get("scene", {}) or {}).get(scene_source)
            if scene_source not in {"library", "local", "sionna"}
            else None
        ),
    )
    apply_registered_extensions(config, extension_cfg)
    return config
