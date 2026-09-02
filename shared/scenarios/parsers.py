"""Derive path-resolved runtime sections from validated scenario YAML.

Pydantic models in ``shared.scenarios.model`` validate authored values. These
parsers resolve paths and produce the small dictionaries consumed by generator
and visualizer adapters.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.logging import get_logger
from shared.scenarios.defaults import DEFAULT_COVERAGE_TX_MODE
from shared.scenarios.extensions import parse_registered_scene_source
from shared.scenarios.frame_paths import (
    DEFAULT_FRAMES_DIRECTORY,
    DEFAULT_FRAMES_PATTERN,
    validate_frames_directory,
    validate_frames_pattern,
)
from shared.scenarios.paths import normalize_path

logger = get_logger(__name__)

# HDF5 output defaults
DEFAULT_CHUNK_SIZE = 100  # Frames per output chunk
DEFAULT_COMPRESSION = "lzf"  # Interactive HDF5 compression profile


def available_sionna_scene_ids() -> Tuple[str, ...]:
    """Return installed Sionna RT scene ids without importing Sionna.

    A built-in scene is catalogued only when it follows Sionna's canonical
    ``scenes/<id>/<id>.xml`` layout.  Discovering the installed files keeps UI
    selectors aligned with the exact Sionna version in the active environment
    and avoids the costly side effects of importing :mod:`sionna`.
    """
    spec = importlib.util.find_spec("sionna")
    if spec is None or spec.origin is None:
        return ()

    scenes_root = Path(spec.origin).parent / "rt" / "scenes"
    try:
        return tuple(
            sorted(
                path.name
                for path in scenes_root.iterdir()
                if path.is_dir() and (path / f"{path.name}.xml").is_file()
            )
        )
    except OSError:
        return ()


def resolve_sionna_scene_xml(scene_id: str) -> Optional[Path]:
    """Resolve the XML path for a Sionna RT built-in scene.

    Locates the installed sionna package via importlib without importing it
    (avoids triggering TensorFlow initialization), then constructs the path
    to the scene XML file using the standard Sionna RT directory layout.

    Args:
        scene_id: Sionna built-in scene name (e.g., ``"floor_wall"``, ``"etoile"``).

    Returns:
        Absolute path to the scene XML file, or ``None`` if the sionna
        package is not installed or the scene file does not exist.
    """
    spec = importlib.util.find_spec("sionna")
    if spec is None or spec.origin is None:
        logger.debug("sionna package not found on this system")
        return None

    sionna_root = Path(spec.origin).parent
    xml_path = sionna_root / "rt" / "scenes" / scene_id / f"{scene_id}.xml"
    if xml_path.exists():
        return xml_path

    logger.warning("Sionna scene XML not found at expected path: %s", xml_path)
    return None


def parse_scene_config(
    scenario_data: Dict[str, Any],
    scenario_root: Path,
    project_root: Path,
) -> Tuple[str, str, Optional[Path]]:
    """
    Parse scene configuration section.

    Args:
        scenario_data: Full scenario YAML data
        scenario_root: Scenario root directory
        project_root: Project root directory

    Returns:
        Tuple of (scene_id, scene_source, scene_xml_path)
    """
    scene_spec = scenario_data.get("scene") or {}
    scene_id = scene_spec.get("id", "default")
    scene_source = str(scene_spec.get("source", "library") or "").strip()

    # Resolve scene XML path using path policy
    scene_xml = None
    if scene_source == "local":
        # Local XML file - resolve relative to scenario root
        scene_xml = normalize_path(
            scene_id,
            base=scenario_root,
            project_root=project_root,
        )
    elif scene_source == "library":
        # Library scene - resolve relative to project root
        scene_xml = normalize_path(f"libraries/scenes/{scene_id}", base=project_root)
    elif scene_source == "sionna":
        # Sionna built-in scene - resolve XML from installed package
        scene_xml = resolve_sionna_scene_xml(scene_id)
    else:
        parsed = parse_registered_scene_source(scene_spec, scenario_root, project_root)
        if parsed is None:
            raise ValueError(
                f"Unsupported scene source: {scene_source}. "
                "Supported built-in sources are: library, local, sionna."
            )
        scene_id, scene_source, scene_xml = parsed

    return scene_id, scene_source, scene_xml


def parse_data_config(
    scenario_data: Dict[str, Any],
    scenario_root: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse frame reading and publication settings.

    ``directory`` and ``pattern`` select frames for readers. ``chunk_size`` and
    ``compression`` configure publication when the generator writes its fixed
    ``<scenario>/frames`` output.

    Args:
        scenario_data: Full scenario YAML data
        scenario_root: Scenario root directory.
        project_root: Root substituted for ``${PROJECT_ROOT}`` in read paths.

    Returns:
        Dictionary with data configuration
    """
    data_spec = scenario_data.get("data", {}) or {}
    files_spec = data_spec.get("files", {}) or {}
    frames_directory = validate_frames_directory(
        files_spec.get("directory", DEFAULT_FRAMES_DIRECTORY)
    )
    frames_pattern = validate_frames_pattern(files_spec.get("pattern", DEFAULT_FRAMES_PATTERN))
    frames_dir = normalize_path(
        frames_directory,
        base=scenario_root,
        project_root=project_root,
    )
    frames_format = files_spec.get("format", "h5")

    if frames_format not in ["h5", "hdf5"]:
        raise ValueError(
            f"Unsupported frame format: {frames_format}. Only HDF5 format is supported."
        )

    if frames_format in ["h5", "hdf5"]:
        frames_format = "h5"

    chunk_size = int(files_spec.get("chunk_size", DEFAULT_CHUNK_SIZE))
    if chunk_size <= 0:
        raise ValueError("data.files.chunk_size must be a positive integer")
    compression = files_spec.get("compression", DEFAULT_COMPRESSION)

    return {
        "mode": data_spec.get("mode", "files"),
        "frames_directory": frames_directory,
        "frames_dir": frames_dir,
        "frames_pattern": frames_pattern,
        "frames_format": frames_format,
        "chunk_size": chunk_size,
        "compression": compression,
    }


def parse_raytracing_config(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse raytracing configuration section.

    Args:
        scenario_data: Full scenario YAML data

    Returns:
        Dictionary with raytracing configuration
    """
    rt_spec = scenario_data.get("raytracing", {}) or {}

    export_path_metrics = bool(rt_spec.get("export_path_metrics", False))
    path_filter = rt_spec.get("path_filter", None)

    # Path filtering requires path metrics (path_loss_db) to determine which
    # paths are weak.  Auto-enable metrics when any filter threshold is set.
    if path_filter and not export_path_metrics:
        pf = path_filter if isinstance(path_filter, dict) else {}
        has_filter = (
            pf.get("relative_threshold_db") is not None
            or pf.get("max_path_loss_db") is not None
            or pf.get("max_paths_per_pair") is not None
        )
        if has_filter:
            export_path_metrics = True

    return {
        "enabled": bool(rt_spec.get("enabled", False)),
        "geometry_only": bool(rt_spec.get("geometry_only", False)),
        "start_step": rt_spec.get("start_step", None),
        "view": rt_spec.get("view", None),
        "export_path_metrics": export_path_metrics,
        "carrier_frequency_hz": rt_spec.get("carrier_frequency_hz", None),
        "bandwidth_hz": rt_spec.get("bandwidth_hz", None),
        "temperature_k": rt_spec.get("temperature_k", None),
        "quality": {
            "preset": (rt_spec.get("quality", {}) or {}).get("preset", None),
            "custom": (rt_spec.get("quality", {}) or {}).get("custom", {}) or {},
        },
        # Path filtering configuration (reduces file size/write time for diffuse reflection)
        "path_filter": path_filter,
        # Antenna array configuration
        "antenna": rt_spec.get("antenna", None),
        # Per-material property overrides
        "materials": rt_spec.get("materials", None),
        # Scene material default policy
        "scene_materials": rt_spec.get("scene_materials", None),
        # Mesh animation rate (decoupled from RT step rate)
        "mesh_update_interval_s": rt_spec.get("mesh_update_interval_s", None),
        # Sionna-native CIR evolution
        "cir_time_steps": rt_spec.get("cir_time_steps", None),
        "cir_sampling_frequency_hz": rt_spec.get("cir_sampling_frequency_hz", None),
    }


def parse_generator_summary_config(
    scenario_data: Dict[str, Any],
    scenario_root: Path,
) -> Dict[str, Any]:
    """
    Parse generator summary configuration section.

    Args:
        scenario_data: Full scenario YAML data
        scenario_root: Scenario root directory
    Returns:
        Dictionary with generator summary configuration
    """
    summary_spec = scenario_data.get("generator_summary", {}) or {}
    # Defaults
    summary_enabled = bool(summary_spec.get("enabled", False))
    summary_create = summary_spec.get("create", []) or []
    summary_output = summary_spec.get("output", {}) or {}
    summary_dir = "summary"
    summary_dir_path = normalize_path(summary_dir, base=scenario_root)
    summary_format = summary_output.get("format", "png")

    topo_rel = Path(summary_dir) / "topology"
    vel_rel = Path(summary_dir) / "velocity"
    ang_rel = Path(summary_dir) / "angular"
    category_dirs = {
        "topology": normalize_path(str(topo_rel), base=scenario_root),
        "velocity": normalize_path(str(vel_rel), base=scenario_root),
        "angular": normalize_path(str(ang_rel), base=scenario_root),
    }

    # Extract visualization config (optional)
    summary_visualization = summary_spec.get("visualization", {}) or {}

    return {
        "enabled": summary_enabled,
        "force": bool(summary_spec.get("force", False)),
        "create": summary_create,
        "output": {
            "dir": summary_dir_path,
            "format": summary_format,
            "dirs": category_dirs,
        },
        "visualization": summary_visualization if summary_visualization else None,
    }


def parse_coverage_config(
    scenario_data: Dict[str, Any],
    scenario_root: Path,
) -> Dict[str, Any]:
    """
    Parse coverage configuration section.

    Args:
        scenario_data: Full scenario YAML data
        scenario_root: Scenario root directory
    Returns:
        Dictionary with coverage configuration
    """
    cov_spec = scenario_data.get("coverage", {}) or {}
    grid_spec = cov_spec.get("grid", {}) or {}
    solver_spec = cov_spec.get("solver", {}) or {}
    metrics_spec = cov_spec.get("metrics", {}) or {}
    tx_spec = cov_spec.get("tx", {}) or {}
    cov_save = cov_spec.get("save", {}) or {}
    # Nested save blocks: {data: {...}, figure: {...}}
    cov_save_data = cov_save.get("data", {}) or {}
    cov_save_figure = cov_save.get("figure", {}) or {}

    save_path_rel = "coverage/coverage_maps.h5"
    save_path = normalize_path(save_path_rel, base=scenario_root)
    save_compression = cov_save.get("compression", "lzf")

    data_dir_rel = "coverage"
    data_dir = normalize_path(data_dir_rel, base=scenario_root)
    fig_dir_rel = "summary/coverage"
    fig_dir = normalize_path(fig_dir_rel, base=scenario_root)

    cov_res = grid_spec.get("resolution_m", None)
    if cov_res is not None:
        # Enforce exactly two elements (reject 3D specifications)
        if not isinstance(cov_res, (list, tuple)) or len(cov_res) != 2:
            raise ValueError("coverage.grid.resolution_m must be a 2-element list [dx, dy]")
        try:
            dx = float(cov_res[0])
            dy = float(cov_res[1])
            if dx <= 0.0 or dy <= 0.0:
                raise ValueError
            cov_res = (dx, dy)
        except (ValueError, TypeError):
            raise ValueError("coverage.grid.resolution_m must contain positive numbers")

    bbox_xy = grid_spec.get("bbox_xy", "auto")
    if bbox_xy != "auto" and bbox_xy is not None:
        try:
            if len(bbox_xy) == 2 and all(len(pair) == 2 for pair in bbox_xy):
                bbox_xy = tuple(tuple(float(v) for v in pair) for pair in bbox_xy)
            elif len(bbox_xy) == 4:
                bbox_xy = (
                    (float(bbox_xy[0]), float(bbox_xy[1])),
                    (float(bbox_xy[2]), float(bbox_xy[3])),
                )
            else:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                "coverage.grid.bbox_xy must be 'auto' or [[x_min, x_max], [y_min, y_max]]"
            )

    cov_heights = grid_spec.get("heights_m", None)
    if isinstance(cov_heights, (int, float)):
        cov_heights = [float(cov_heights)]

    # Figure rendering options (optional)
    fig_overlay_scene = False
    fig_overlay_alpha = 0.3
    fig_overlay_style = "outline"
    fig_interpolation = "nearest"
    fig_metrics = []
    fig_metric_filename = "coverage_metrics"
    fig_columns = 3
    fig_show_tx = True
    fig_distribution = {
        "enabled": False,
        "metrics": [],
        "filename": "coverage_distributions",
        "bins": 40,
    }
    if "save" in cov_spec:
        fig_block = (cov_spec.get("save", {}) or {}).get("figure", {}) or {}
        fig_overlay_scene = bool(fig_block.get("overlay_scene", False))
        fig_overlay_alpha = float(fig_block.get("overlay_alpha", 0.3))
        fig_overlay_style = str(fig_block.get("overlay_style", "outline"))
        fig_interpolation = str(fig_block.get("interpolation", "nearest"))
        fig_metrics = list(fig_block.get("metrics", []) or [])
        fig_metric_filename = str(fig_block.get("metric_filename", "coverage_metrics"))
        fig_columns = int(fig_block.get("columns", 3))
        fig_show_tx = bool(fig_block.get("show_tx", True))
        distribution_block = fig_block.get("distribution", {}) or {}
        fig_distribution = {
            "enabled": bool(distribution_block.get("enabled", False)),
            "metrics": list(distribution_block.get("metrics", []) or []),
            "filename": str(distribution_block.get("filename", "coverage_distributions")),
            "bins": int(distribution_block.get("bins", 40)),
        }

    return {
        "enabled": bool(cov_spec.get("enabled", False)),
        "bbox_xy": bbox_xy,
        "resolution_m": cov_res,
        "heights_m": cov_heights,
        "stride": cov_spec.get("stride", 1),
        "metrics": {
            "store": metrics_spec.get("store", ["path_gain_linear"]),
            "derived": metrics_spec.get(
                "derived",
                ["path_loss_db", "rss_dbm", "sinr_db", "serving_tx", "tx_margin_db"],
            ),
        },
        "tx_mode": tx_spec.get("mode", DEFAULT_COVERAGE_TX_MODE),
        "tx_selected": tx_spec.get("selected", None),
        "solver": {
            "preset": solver_spec.get("preset", None),
            "custom": solver_spec.get("custom", {}) or {},
            "samples_per_tx": solver_spec.get("samples_per_tx", None),
            "max_depth": solver_spec.get("max_depth", None),
            "los": solver_spec.get("los", None),
            "specular_reflection": solver_spec.get("specular_reflection", None),
            "diffuse_reflection": solver_spec.get("diffuse_reflection", None),
            "refraction": solver_spec.get("refraction", None),
            "diffraction": solver_spec.get("diffraction", None),
            "seed": solver_spec.get("seed", None),
            "rr_depth": solver_spec.get("rr_depth", None),
            "rr_prob": solver_spec.get("rr_prob", None),
            "stop_threshold": solver_spec.get("stop_threshold", None),
        },
        "precoding_vec": tx_spec.get("precoding_vec", None),
        "save": {
            "path": save_path,
            "compression": save_compression,
            "data": {
                "enabled": bool(cov_save_data.get("enabled", True)),
                "dir": data_dir,
                "format": "h5",
                "filename": "coverage_maps",
            },
            "figure": {
                "enabled": bool(cov_save_figure.get("enabled", False)),
                "dir": fig_dir,
                "format": cov_save_figure.get("format", "png"),
                "filename": cov_save_figure.get("filename", "coverage_maps"),
                "overlay_scene": fig_overlay_scene,
                "overlay_alpha": fig_overlay_alpha,
                "overlay_style": fig_overlay_style,
                "interpolation": fig_interpolation,
                "metrics": fig_metrics,
                "metric_filename": fig_metric_filename,
                "columns": fig_columns,
                "show_tx": fig_show_tx,
                "distribution": fig_distribution,
            },
        },
    }
