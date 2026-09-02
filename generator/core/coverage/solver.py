#!/usr/bin/env python3
"""Coverage map solver orchestration.

This module is the execution path behind
``generator.core.coverage.compute_coverage_map``. It bridges ORCHAV scenario
config to Sionna RT's ``RadioMapSolver``, solves a stack of 2D radio maps at
requested heights, converts per-TX path gain into RF metric layers, and returns
a schema-v2 payload for the HDF5 writer.

Main flow:

1. :func:`compute_coverage_map` validates solver availability and resolves
   bounds/quality policy.
2. :func:`_run_radio_map_solver` builds the grid and ``RadioMapSolver`` kwargs.
3. :func:`_solve_radio_map_heights` runs one radio-map solve per height.
4. :func:`_build_metric_data` and :func:`_build_coverage_payload` prepare stored
   arrays and metadata.
"""

import inspect
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.coverage.schema import (
    COVERAGE_HDF5_SCHEMA_VERSION,
    COVERAGE_NO_SERVING_TX,
    sum_coverage_interference_w,
)
from shared.logging import get_logger

from ..configuration import CoverageConfig, SimulationConfig
from ..utils import to_float, to_numpy
from .bounds import CoverageBoundsError, resolve_coverage_bbox
from .metrics import DB_POWER_SCALE, _db_from_linear, _path_loss_db, _tx_power_dbm
from .quality import CoverageQuality

logger = get_logger(__name__)

BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
DEFAULT_COVERAGE_BANDWIDTH_HZ = 2e9
DEFAULT_COVERAGE_TEMPERATURE_K = 293.0


@dataclass
class _CoverageGridSpec:
    """Grid geometry passed to ``RadioMapSolver`` for a stack of height slices."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    dx: float
    dy: float
    z_heights: np.ndarray
    center: Any
    orientation: Any
    size: Any
    cell_size: Any


@dataclass(frozen=True)
class _DeviceMetadata:
    """Device positions and RF noise parameters needed for derived metrics."""

    tx_positions: np.ndarray
    rx_positions: np.ndarray
    tx_power_dbm: np.ndarray
    bandwidth_hz: float
    temperature_k: float
    noise_power_w: float


@dataclass(frozen=True)
class _CoverageMetricData:
    """Metric requests, compact writer inputs, and the selected display layer.

    ``metrics_store`` and ``metrics_derived`` preserve the user's logical
    requests for HDF5 metadata. ``stored_values`` and ``derived`` are the smaller
    in-memory arrays needed by the compact writer and quick-look figures.
    """

    metric_name: str
    tx_mode: str
    metrics_store: list[str]
    metrics_derived: list[str]
    stored_values: dict[str, np.ndarray]
    derived: dict[str, np.ndarray]
    values_3d: np.ndarray
    display_metric_name: str
    value_min: float
    value_max: float


def _normalise_radio_map_tensor(value: Any, num_tx: int, *, name: str) -> np.ndarray:
    """Validate Sionna's ``(num_tx, ny, nx)`` radio-map tensor contract.

    Keeping the documented axis order explicit prevents equally sized TX and
    spatial axes from being silently reinterpreted. Singleton spatial axes stay
    present because no general-purpose dimension squeezing is performed.
    """
    if value is None:
        raise ValueError(f"Radio map missing {name}")
    if num_tx < 1:
        raise ValueError(f"Radio map {name} requires at least one transmitter")

    arr = np.asarray(to_numpy(value), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != num_tx:
        raise ValueError(
            f"Unexpected {name} shape: {arr.shape}; expected "
            f"({num_tx}, ny, nx) from Sionna RadioMapSolver"
        )
    return arr


def _derive_coverage_layers(
    path_gain_linear: np.ndarray,
    tx_power_dbm: np.ndarray,
    noise_power_w: float,
    derived_requested: list[str],
    metric_name: str,
    tx_mode: str = "per_tx",
    include_metric_name: bool = True,
) -> dict[str, np.ndarray]:
    """Derive stored and display layers from Sionna per-TX path gain.

    ``RadioMapSolver`` returns path gain. ORCHAV stores that raw layer and
    derives the coverage views users usually inspect: RSS, SINR, path loss,
    serving transmitter, and best-vs-second transmitter margin. The input shape
    is ``(step, height, tx, ny, nx)``. Only immediately requested layers are
    materialized; the HDF5 writer advertises other derivable metrics through
    shared recipe metadata instead of storing every dense array.
    """
    gain = np.asarray(path_gain_linear, dtype=np.float32)
    # Convert configured TX powers from dBm to watts before applying path gain.
    power_w = (
        DB_POWER_SCALE ** ((tx_power_dbm.astype(np.float64) - 30.0) / DB_POWER_SCALE)
    ).astype(np.float32)
    rss_w = gain * power_w.reshape((1, 1, -1, 1, 1))
    best_tx_index = np.nanargmax(np.where(np.isfinite(rss_w), rss_w, -np.inf), axis=2).astype(
        np.int16
    )
    best_power = np.take_along_axis(rss_w, best_tx_index[:, :, np.newaxis, :, :], axis=2)[:, :, 0]
    valid_best = np.isfinite(best_power)
    # Keep uncovered cells distinct from transmitter index 0 in the integer
    # categorical layer.
    serving_tx = np.where(valid_best, best_tx_index, COVERAGE_NO_SERVING_TX).astype(np.int16)

    if rss_w.shape[2] > 1:
        sorted_power = np.sort(np.where(np.isfinite(rss_w), rss_w, -np.inf), axis=2)
        second_power = sorted_power[:, :, -2, :, :]
    else:
        second_power = np.full_like(best_power, np.nan)

    interference = sum_coverage_interference_w(rss_w, best_tx_index)
    denominator = interference + float(noise_power_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinr_linear = best_power / denominator

    derived: dict[str, np.ndarray] = {}
    requested = set(derived_requested or [])
    if include_metric_name:
        requested.add(metric_name)
    if "path_gain_db" in requested:
        derived["path_gain_db"] = _db_from_linear(gain)
    if (
        "path_loss_db" in requested
        or "best_path_loss_db" in requested
        or metric_name in {"path_loss_db", "best_path_loss_db"}
    ):
        best_gain = np.take_along_axis(gain, best_tx_index[:, :, np.newaxis, :, :], axis=2)[:, :, 0]
        derived["best_path_loss_db"] = _path_loss_db(best_gain)
        if "path_loss_db" in requested or metric_name == "path_loss_db":
            derived["path_loss_db"] = _path_loss_db(gain)
    if (
        "rss_dbm" in requested
        or "best_rss_dbm" in requested
        or metric_name in {"rss_w", "rss_dbm", "best_rss_dbm"}
    ):
        derived["rss_dbm"] = _db_from_linear(np.maximum(rss_w, 1e-30)) + 30.0
        derived["best_rss_dbm"] = _db_from_linear(np.maximum(best_power, 1e-30)) + 30.0
    if "sum_rss_dbm" in requested or metric_name == "sum_rss_dbm" or tx_mode == "sum_power":
        summed_rss_w = np.nansum(rss_w, axis=2)
        # No finite TX sample means no coverage, not zero received power.
        summed_rss_w = np.where(np.any(np.isfinite(rss_w), axis=2), summed_rss_w, np.nan)
        derived["sum_rss_dbm"] = _db_from_linear(np.maximum(summed_rss_w, 1e-30)) + 30.0
    if "sinr_db" in requested or metric_name in {"sinr_linear", "sinr_db"}:
        derived["sinr_db"] = _db_from_linear(np.maximum(sinr_linear, 1e-30))
    if "serving_tx" in requested:
        derived["serving_tx"] = serving_tx
    if "tx_margin_db" in requested:
        best_db = _db_from_linear(np.maximum(best_power, 1e-30))
        second_db = _db_from_linear(np.maximum(second_power, 1e-30))
        derived["tx_margin_db"] = (best_db - second_db).astype(np.float32)
    if "rss_w" in requested:
        derived["rss_w"] = rss_w.astype(np.float32)
    if "sinr_linear" in requested:
        derived["sinr_linear"] = sinr_linear.astype(np.float32)
    return derived


def _apply_solver_settings_overrides(
    quality_settings: dict[str, Any],
    solver_settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply direct ``coverage.solver`` overrides on top of preset settings."""
    settings = dict(quality_settings)
    samples = solver_settings.get("samples_per_tx", solver_settings.get("samples_per_src"))
    if samples is not None:
        settings["samples_per_tx"] = int(samples)
    if solver_settings.get("max_depth") is not None:
        settings["max_depth"] = int(solver_settings["max_depth"])
    for key in (
        "los",
        "specular_reflection",
        "diffuse_reflection",
        "refraction",
        "diffraction",
    ):
        if solver_settings.get(key) is not None:
            settings[key] = bool(solver_settings[key])
    return settings


def _format_progress_time(seconds: float) -> str:
    """Format elapsed/ETA seconds for stderr progress messages."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"


def _write_coverage_progress(
    done: int,
    total: int,
    *,
    start_time: float,
    current_height: float | None = None,
) -> None:
    """Write a human-readable height-slice progress line to stderr."""
    total = max(1, total)
    done = max(0, min(done, total))
    elapsed = time.time() - start_time
    avg = elapsed / done if done > 0 else 0.0
    remaining = avg * max(0, total - done)
    pct = 100.0 * done / total
    bar_width = 30
    fill = int(bar_width * done / total)
    bar = "#" * fill + "." * (bar_width - fill)
    height_msg = "" if current_height is None else f" | height {current_height:.2f}m"
    message = (
        f"Coverage [{bar}] {done}/{total} {pct:5.1f}%"
        f"{height_msg}"
        f" | elapsed {_format_progress_time(elapsed)}"
        f" | eta {_format_progress_time(remaining)}"
    )
    try:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
    except OSError:
        pass


def _write_coverage_status(message: str) -> None:
    """Write a coverage status line to stderr without failing the solve."""
    try:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
    except OSError:
        pass


def _resolve_solver_quality_settings(
    quality_settings: dict[str, Any] | None,
    coverage_config: CoverageConfig,
    simulation_config: SimulationConfig,
) -> dict[str, Any]:
    """Return RadioMapSolver quality settings after coverage-specific overrides."""
    if quality_settings is None:
        try:
            qs = simulation_config.get_coverage_quality_settings()
        except (AttributeError, TypeError):
            qs = {
                "max_depth": 3,
                "samples_per_tx": 100000,
                "specular_reflection": True,
                "diffuse_reflection": True,
                "refraction": True,
            }
        quality_settings = {
            "samples_per_tx": int(qs.get("samples_per_tx", 100000)),
            "max_depth": int(qs.get("max_depth", 3)),
            "los": True,
            "specular_reflection": bool(qs.get("specular_reflection", True)),
            "diffuse_reflection": bool(qs.get("diffuse_reflection", True)),
            "refraction": bool(qs.get("refraction", True)),
            "diffraction": bool(qs.get("diffraction", False)),
        }

    solver_settings = getattr(coverage_config, "solver_settings", None) or {}
    return _apply_solver_settings_overrides(quality_settings, solver_settings)


def _build_coverage_grid_spec(
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    coverage_config: CoverageConfig,
    mi: Any,
) -> _CoverageGridSpec:
    """Translate ORCHAV bbox/resolution settings into Mitsuba grid objects."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
    dx, dy = coverage_config.resolution
    if coverage_config.heights is None:
        z_heights = np.array([0.5 * (z_min + z_max)])
    else:
        z_heights = np.array(coverage_config.heights, dtype=float)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return _CoverageGridSpec(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        dx=dx,
        dy=dy,
        z_heights=z_heights,
        center=mi.Point3f(float(x_center), float(y_center), 0.0),
        orientation=mi.Point3f(0, 0, 0),
        size=mi.Point2f(float(x_max - x_min), float(y_max - y_min)),
        cell_size=mi.Point2f(
            float(dx * coverage_config.stride), float(dy * coverage_config.stride)
        ),
    )


def _ensure_scene_arrays(scene: Any) -> None:
    """Attach default Sionna arrays when a scene lacks coverage-ready arrays."""
    from ..scenario_entities.antenna_arrays import create_planar_array

    if not getattr(scene, "tx_array", None):
        scene.tx_array = create_planar_array()
    if not getattr(scene, "rx_array", None):
        scene.rx_array = create_planar_array()


def _solver_supported_params(solver: Any) -> set[str]:
    """Inspect the installed Sionna solver so optional kwargs stay compatible."""
    return set(inspect.signature(solver.__call__).parameters)


def _build_base_solver_kwargs(
    scene: Any,
    grid: _CoverageGridSpec,
    coverage_config: CoverageConfig,
    quality_settings: dict[str, Any],
    solver_params: set[str],
    mi: Any,
) -> dict[str, Any]:
    """Build kwargs shared by every height solve for one coverage grid."""
    solver_kwargs = {
        "scene": scene,
        "orientation": grid.orientation,
        "size": grid.size,
        "cell_size": grid.cell_size,
        "samples_per_tx": quality_settings["samples_per_tx"],
        "max_depth": quality_settings["max_depth"],
        "los": quality_settings.get("los", True),
        "specular_reflection": quality_settings["specular_reflection"],
        "diffuse_reflection": quality_settings["diffuse_reflection"],
        "refraction": quality_settings["refraction"],
    }
    if "diffraction" in solver_params:
        solver_kwargs["diffraction"] = bool(quality_settings["diffraction"])

    rr_depth = getattr(coverage_config, "rr_depth", None)
    if rr_depth is not None:
        solver_kwargs["rr_depth"] = int(rr_depth)
    rr_prob = getattr(coverage_config, "rr_prob", None)
    if rr_prob is not None:
        solver_kwargs["rr_prob"] = float(rr_prob)
    stop_threshold = getattr(coverage_config, "stop_threshold", None)
    if stop_threshold is not None:
        solver_kwargs["stop_threshold"] = float(stop_threshold)
    seed = getattr(coverage_config, "seed", None)
    if seed is not None and "seed" in solver_params:
        solver_kwargs["seed"] = int(seed)

    precoding_vec = _build_precoding_vec(getattr(coverage_config, "precoding_vec", None), mi)
    if precoding_vec is not None:
        solver_kwargs["precoding_vec"] = precoding_vec
    return solver_kwargs


def _build_precoding_vec(precoding_vec: Any, mi: Any) -> Any | None:
    """Build the real/imaginary Sionna precoding tuple when configured."""
    if precoding_vec is None:
        return None
    try:
        import drjit as dr

        real_vals = [float(v) for v in precoding_vec]
        precoding_real = mi.TensorXf(real_vals)
        precoding_imag = dr.zeros(mi.TensorXf, len(real_vals))
        return precoding_real, precoding_imag
    except (ImportError, ValueError, TypeError) as exc:
        logger.warning("Could not build precoding vector: %s", exc)
        return None


def _log_solver_height_parameters(
    grid: _CoverageGridSpec,
    quality_settings: dict[str, Any],
) -> None:
    """Log solver parameters for one height slice at debug level."""
    logger.debug("Coverage parameters:")
    logger.debug(f"  Center: {grid.center}")
    logger.debug(f"  Size: {grid.size}")
    logger.debug(f"  Cell size: {grid.cell_size}")
    logger.debug(f"  Samples per TX: {quality_settings['samples_per_tx']}")
    logger.debug(f"  Max depth: {quality_settings['max_depth']}")
    logger.debug(f"  LOS: {quality_settings.get('los', True)}")
    logger.debug(f"  Specular reflection: {quality_settings['specular_reflection']}")
    logger.debug(f"  Diffuse reflection: {quality_settings['diffuse_reflection']}")
    logger.debug(f"  Refraction: {quality_settings['refraction']}")
    logger.debug(f"  Diffraction: {quality_settings['diffraction']}")


def _extract_path_gain_slice(
    radio_map: Any,
    num_tx: int,
    z_height: float,
) -> np.ndarray | None:
    """Extract and clean the per-TX path-gain tensor from a Sionna radio map."""
    try:
        gain = _normalise_radio_map_tensor(
            getattr(radio_map, "path_gain", None), num_tx, name="path_gain"
        )
        return np.where(np.isfinite(gain) & (gain > 0.0), gain, np.nan)
    except ValueError:
        logger.exception(
            "Failed to extract path gain data from radio map for height %.2f",
            z_height,
        )
        return None


def _solve_radio_map_heights(
    solver: Any,
    base_solver_kwargs: dict[str, Any],
    grid: _CoverageGridSpec,
    quality_settings: dict[str, Any],
    num_tx: int,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Run ``RadioMapSolver`` once per requested height and stack the results."""
    path_gain_slices: list[np.ndarray] = []
    grid_info: dict[str, int] | None = None
    total_heights = len(grid.z_heights)
    progress_start = time.time()
    if total_heights > 0:
        _write_coverage_status(f"Coverage: solving {total_heights} height slice(s)")

    for z_idx, z_height in enumerate(grid.z_heights):
        _write_coverage_status(
            f"Coverage: solving height {z_idx + 1}/{total_heights} ({z_height:.2f}m)"
        )
        logger.info(f"Computing coverage for height {z_height:.2f}m ({z_idx + 1}/{total_heights})")
        grid.center.z = float(z_height)
        _log_solver_height_parameters(grid, quality_settings)

        radio_map = solver(**{**base_solver_kwargs, "center": grid.center})
        gain = _extract_path_gain_slice(radio_map, num_tx, float(z_height))
        if gain is None:
            if grid_info is None:
                return None
            gain = np.full((num_tx, grid_info["ny"], grid_info["nx"]), np.nan)

        if grid_info is None:
            actual_ny, actual_nx = gain.shape[-2:]
            grid_info = {"nx": actual_nx, "ny": actual_ny, "nz": total_heights}
            logger.debug("Grid dimensions: %d x %d x %d", actual_nx, actual_ny, total_heights)

        path_gain_slices.append(gain.astype(np.float32, copy=False))
        _write_coverage_progress(
            z_idx + 1,
            total_heights,
            start_time=progress_start,
            current_height=float(z_height),
        )

    if not path_gain_slices or grid_info is None:
        raise ValueError("No coverage height slices were computed")
    path_gain_linear = np.stack(path_gain_slices, axis=0).astype(np.float32, copy=False)
    return path_gain_linear[np.newaxis, ...], grid_info


def _extract_device_metadata(
    tx_list: list[Any],
    rx_list: list[Any],
    simulation_config: SimulationConfig,
    num_tx: int,
) -> _DeviceMetadata:
    """Collect TX/RX positions and thermal-noise inputs for metric derivation."""
    tx_positions = np.array(
        [
            [to_float(tx.position.x), to_float(tx.position.y), to_float(tx.position.z)]
            for tx in tx_list
        ],
        dtype=np.float32,
    )
    rx_positions = np.array(
        [
            [to_float(rx.position.x), to_float(rx.position.y), to_float(rx.position.z)]
            for rx in rx_list
        ],
        dtype=np.float32,
    ).reshape((len(rx_list), 3))

    tx_power_dbm = np.array([_tx_power_dbm(tx) for tx in tx_list], dtype=np.float32)
    if tx_power_dbm.size == 0:
        tx_power_dbm = np.zeros((num_tx,), dtype=np.float32)
    bandwidth_hz = float(
        getattr(simulation_config, "bandwidth_hz", DEFAULT_COVERAGE_BANDWIDTH_HZ)
        or DEFAULT_COVERAGE_BANDWIDTH_HZ
    )
    temperature_k = float(
        getattr(simulation_config, "temperature_k", DEFAULT_COVERAGE_TEMPERATURE_K)
        or DEFAULT_COVERAGE_TEMPERATURE_K
    )
    noise_power_w = BOLTZMANN_CONSTANT_J_PER_K * temperature_k * bandwidth_hz
    return _DeviceMetadata(
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_power_dbm=tx_power_dbm,
        bandwidth_hz=bandwidth_hz,
        temperature_k=temperature_k,
        noise_power_w=noise_power_w,
    )


def _build_metric_data(
    path_gain_linear: np.ndarray,
    coverage_config: CoverageConfig,
    device_metadata: _DeviceMetadata,
    tx_names: list[str],
) -> _CoverageMetricData:
    """Derive the minimal metric arrays needed before HDF5 serialization.

    The scenario can ask for many logical metrics, but storing all of them would
    duplicate information already present in ``path_gain_linear``. This function
    keeps the user's requested metric lists for metadata while only materializing
    layers needed for immediate display and compact categorical output.
    """
    metric_name = str(getattr(coverage_config, "metric", "path_loss_db"))
    tx_mode = str(getattr(coverage_config, "tx_mode", "per_tx"))
    metrics_store = list(getattr(coverage_config, "metrics_store", None) or ["path_gain_linear"])
    metrics_derived = list(
        getattr(coverage_config, "metrics_derived", None)
        or ["path_loss_db", "rss_dbm", "sinr_db", "serving_tx", "tx_margin_db"]
    )
    # These requests select the compact top-level display layer, not the full
    # logical metric menu. The writer reconstructs that larger menu from
    # ``metrics_store`` and ``metrics_derived``.
    display_requests: list[str] = []

    def request_display_metric(name: str) -> None:
        if name not in display_requests:
            display_requests.append(name)

    if "serving_tx" in metrics_derived:
        request_display_metric("serving_tx")
    if tx_mode == "margin":
        request_display_metric("tx_margin_db")
    elif tx_mode == "sum_power":
        request_display_metric("sum_rss_dbm")
    elif tx_mode == "selected":
        request_display_metric("path_loss_db")
    else:
        request_display_metric("best_path_loss_db")
    if metric_name in {
        "best_path_loss_db",
        "best_rss_dbm",
        "sum_rss_dbm",
        "sinr_linear",
        "sinr_db",
        "tx_margin_db",
    }:
        request_display_metric(metric_name)
    derived_requested = list(
        dict.fromkeys(
            [
                *display_requests,
            ]
        )
    )
    derived = _derive_coverage_layers(
        path_gain_linear,
        device_metadata.tx_power_dbm,
        device_metadata.noise_power_w,
        derived_requested,
        metric_name,
        tx_mode=tx_mode,
        include_metric_name=False,
    )
    stored_values = {"path_gain_linear": path_gain_linear}

    display_metric_name, values_3d = _select_display_values(
        path_gain_linear,
        coverage_config,
        derived,
        metric_name,
        tx_mode,
        tx_names,
    )
    valid_values = values_3d[np.isfinite(values_3d)]
    if len(valid_values) > 0:
        value_min = float(np.min(valid_values))
        value_max = float(np.max(valid_values))
    else:
        value_min = value_max = 0.0

    return _CoverageMetricData(
        metric_name=metric_name,
        tx_mode=tx_mode,
        metrics_store=metrics_store,
        metrics_derived=metrics_derived,
        stored_values=stored_values,
        derived=derived,
        values_3d=values_3d.astype(np.float32, copy=False),
        display_metric_name=display_metric_name,
        value_min=value_min,
        value_max=value_max,
    )


def _select_display_values(
    path_gain_linear: np.ndarray,
    coverage_config: CoverageConfig,
    derived: dict[str, np.ndarray],
    metric_name: str,
    tx_mode: str,
    tx_names: list[str],
) -> tuple[str, np.ndarray]:
    """Select the 3D metric layer exposed as top-level coverage values."""
    display_metric_name = "best_path_loss_db" if "best_path_loss_db" in derived else metric_name
    if tx_mode == "margin" and "tx_margin_db" in derived:
        display_metric_name = "tx_margin_db"
    elif tx_mode == "sum_power" and "sum_rss_dbm" in derived:
        display_metric_name = "sum_rss_dbm"
    elif tx_mode == "selected" and "path_loss_db" in derived:
        selected = getattr(coverage_config, "tx_selected", None)
        selected_name = None
        if isinstance(selected, int) and 0 <= selected < len(tx_names):
            selected_name = tx_names[selected]
        elif isinstance(selected, str) and selected in tx_names:
            selected_name = selected
        if selected_name is not None:
            display_metric_name = f"path_loss_db/{selected_name}"

    values_3d = derived.get(display_metric_name)
    if values_3d is None:
        if "/" in display_metric_name and "path_loss_db" in derived:
            metric_base, tx_name = display_metric_name.split("/", 1)
            tx_index = tx_names.index(tx_name) if tx_name in tx_names else 0
            values_3d = derived[metric_base][:, :, tx_index, :, :]
        else:
            values_3d = derived.get("best_path_loss_db")
    if values_3d is None:
        values_3d = _path_loss_db(np.nanmax(path_gain_linear, axis=2))
        display_metric_name = "path_loss_db"
    return display_metric_name, np.asarray(values_3d[0], dtype=np.float32)


def _build_coverage_payload(
    grid: _CoverageGridSpec,
    grid_info: dict[str, int],
    path_gain_linear: np.ndarray,
    coverage_config: CoverageConfig,
    quality_settings: dict[str, Any],
    device_metadata: _DeviceMetadata,
    metric_data: _CoverageMetricData,
    tx_names: list[str],
    rx_names: list[str],
) -> dict[str, Any]:
    """Assemble the schema-v2 coverage payload consumed by the HDF5 writer."""
    coverage_height = grid.z_heights[0] if len(grid.z_heights) > 0 else grid.z_min
    grid_origin = (grid.x_min, grid.y_min, coverage_height)
    grid_spacing = (grid.dx * coverage_config.stride, grid.dy * coverage_config.stride)
    grid_shape = (grid_info["nx"], grid_info["ny"], grid_info["nz"])

    return {
        "schema_version": COVERAGE_HDF5_SCHEMA_VERSION,
        "grid_origin": np.array(grid_origin, dtype=np.float32),
        "grid_spacing": np.array(grid_spacing, dtype=np.float32),
        "grid_shape": np.array(grid_shape, dtype=np.int32),
        "heights": np.asarray(grid.z_heights, dtype=np.float32),
        "path_gain_linear": path_gain_linear,
        "stored_values": metric_data.stored_values,
        "derived": metric_data.derived,
        "values_3d": metric_data.values_3d,
        "metric_name": metric_data.display_metric_name,
        "tx_positions": device_metadata.tx_positions,
        "rx_positions": device_metadata.rx_positions,
        "tx_names": tx_names,
        "rx_names": rx_names,
        "tx_power_dbm": device_metadata.tx_power_dbm,
        "value_min": metric_data.value_min,
        "value_max": metric_data.value_max,
        "metadata": {
            "schema_version": COVERAGE_HDF5_SCHEMA_VERSION,
            "tx_mode": metric_data.tx_mode,
            "tx_index": getattr(coverage_config, "tx_index", None),
            "tx_selected": getattr(coverage_config, "tx_selected", None),
            "requested_metric": metric_data.metric_name,
            "quality": coverage_config.quality,
            "solver": {
                **quality_settings,
                "seed": getattr(coverage_config, "seed", None),
                "rr_depth": getattr(coverage_config, "rr_depth", None),
                "rr_prob": getattr(coverage_config, "rr_prob", None),
                "stop_threshold": getattr(coverage_config, "stop_threshold", None),
            },
            "heights": np.asarray(grid.z_heights, dtype=np.float32),
            "resolution": coverage_config.resolution,
            "metrics_store": metric_data.metrics_store,
            "metrics_derived": metric_data.metrics_derived,
            "noise_power_w": float(device_metadata.noise_power_w),
            "bandwidth_hz": device_metadata.bandwidth_hz,
            "temperature_k": device_metadata.temperature_k,
            "bbox_xy": (
                (float(grid.x_min), float(grid.x_max)),
                (float(grid.y_min), float(grid.y_max)),
            ),
        },
    }


def compute_coverage_map(
    scene: Any,
    tx_list: list[Any],
    rx_list: list[Any],
    target_objects: list[Any],
    coverage_config: CoverageConfig,
    simulation_config: SimulationConfig,
    scenario_context: Any | None = None,
) -> dict[str, Any] | None:
    """Compute a schema-v2 coverage map with Sionna RT's RadioMapSolver."""
    try:
        import sionna.rt as srt

        if not hasattr(srt, "RadioMapSolver"):
            logger.error("This Sionna RT build has no RadioMapSolver")
            return None
        logger.info("Computing coverage map...")
        logger.info(f"Metric: {coverage_config.metric}")
        logger.info(f"Resolution: {coverage_config.resolution}")
        logger.info(f"Stride: {coverage_config.stride}")
        bbox = resolve_coverage_bbox(
            coverage_config,
            scene=scene,
            scenario_context=scenario_context,
        )
        cq = CoverageQuality.from_context(simulation_config, scenario_context=scenario_context)
        quality_args = cq.to_radio_map_args()
        coverage_data = _run_radio_map_solver(
            scene,
            tx_list,
            rx_list,
            target_objects,
            coverage_config,
            simulation_config,
            bbox=bbox,
            quality_settings=quality_args,
        )
        if coverage_data is None:
            logger.error("Coverage map computation failed")
            return None
        logger.info("Coverage map computed successfully")
        return coverage_data
    except CoverageBoundsError:
        raise
    except (ValueError, TypeError, AttributeError, RuntimeError):
        logger.exception("Error computing coverage map")
        return None


def _run_radio_map_solver(
    scene,
    tx_list,
    rx_list,
    target_objects,
    coverage_config,
    simulation_config,
    *,
    bbox,
    quality_settings=None,
):
    """
    Compute coverage map tensors using Sionna RT's RadioMapSolver.

    The facade resolves scenario policy such as bbox and quality
    precedence before calling this solver runner. ``target_objects`` is kept in
    the signature for pipeline consistency; targets are already present in the
    loaded Sionna scene used by the solver.
    """
    progress_started = False
    try:
        import mitsuba as mi
        from sionna.rt import RadioMapSolver

        logger.info("Computing coverage map")
        logger.info(f"Metric: {coverage_config.metric}")
        logger.info(f"Resolution: {coverage_config.resolution}")
        logger.info(f"Stride: {coverage_config.stride}")

        quality_settings = _resolve_solver_quality_settings(
            quality_settings,
            coverage_config,
            simulation_config,
        )
        grid = _build_coverage_grid_spec(bbox, coverage_config, mi)
        logger.info(f"Heights: {grid.z_heights}")

        solver = RadioMapSolver()
        solver_params = _solver_supported_params(solver)
        _ensure_scene_arrays(scene)

        logger.debug(f"Radio map center: {grid.center}")
        logger.debug(f"Radio map size: {grid.size}")
        logger.debug(f"Cell size: {grid.cell_size}")

        num_tx = max(1, len(tx_list))
        progress_started = len(grid.z_heights) > 0
        base_solver_kwargs = _build_base_solver_kwargs(
            scene,
            grid,
            coverage_config,
            quality_settings,
            solver_params,
            mi,
        )

        solve_result = _solve_radio_map_heights(
            solver,
            base_solver_kwargs,
            grid,
            quality_settings,
            num_tx,
        )
        if solve_result is None:
            return None
        path_gain_linear, grid_info = solve_result

        logger.info(f"Final grid: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")
        logger.info(f"Total points: {grid_info['nx'] * grid_info['ny'] * grid_info['nz']}")

        valid_count = np.sum(np.isfinite(path_gain_linear))
        total_count = path_gain_linear.size
        logger.info(
            f"Valid coverage values: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)"
        )

        device_metadata = _extract_device_metadata(tx_list, rx_list, simulation_config, num_tx)
        tx_names = [str(getattr(tx, "name", f"TX{i+1}")) for i, tx in enumerate(tx_list)]
        rx_names = [str(getattr(rx, "name", f"RX{i+1}")) for i, rx in enumerate(rx_list)]
        metric_data = _build_metric_data(
            path_gain_linear,
            coverage_config,
            device_metadata,
            tx_names,
        )
        coverage_data = _build_coverage_payload(
            grid,
            grid_info,
            path_gain_linear,
            coverage_config,
            quality_settings,
            device_metadata,
            metric_data,
            tx_names,
            rx_names,
        )

        logger.info("Coverage map computed successfully")
        return coverage_data

    except (ValueError, TypeError, AttributeError, RuntimeError):
        if progress_started:
            try:
                sys.stderr.write("\n")
                sys.stderr.flush()
            except OSError:
                pass
        logger.exception("Error computing coverage map")
        return None
