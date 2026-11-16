"""Figure generation for sensing post-processing results.

Provides summary figures (RD maps, detection timelines, threshold-sweep curves,
micro-Doppler spectrograms) from saved HDF5 sensing data.

All exported entry points accept plain paths and dicts so callers are not
forced to import the full ``PostProcessingConfig``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from shared.sensing.delay import (
    DelayDomainSamples,
    append_delay_domain_samples,
    extract_delay_domain_samples,
    require_compatible_delay_sequence,
)
from shared.sensing.evaluation import ScenarioEvaluationResult, ThresholdSweepPoint
from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth
from shared.sensing.gt import extract_stored_gt_entry
from shared.sensing.loader import load_frames_sensing, load_saved_results
from shared.sensing.signal_processing import (
    compute_micro_doppler,
    compute_range_collapsed_micro_doppler,
    compute_range_tracked_micro_doppler,
    detection_velocity_display,
    doppler_axis_to_velocity_m_s,
)

# Use shared logger if available, else standard logging
try:
    from shared.logging import get_logger

    logger = get_logger("shared.sensing.plotting.figures")
except ImportError:
    logger = logging.getLogger("shared.sensing.plotting.figures")

DEFAULT_FIGURE_EXCLUSIONS = {"detection_summary", "detection_stages", "tracking_decisions"}
_SENSING_INSTALL_COMMAND = "python -m pip install orchav"


@cache
def _load_sensing_viz() -> Any:
    """Load Matplotlib-backed sensing plots only for figure-producing calls."""

    try:
        import shared.sensing.plotting.viz as sensing_viz
    except ModuleNotFoundError as exc:
        if exc.name is None or not (exc.name == "matplotlib" or exc.name.startswith("matplotlib.")):
            raise
        raise RuntimeError(
            "Sensing figure generation requires Matplotlib. "
            f"Install ORCHAV with: {_SENSING_INSTALL_COMMAND}"
        ) from exc
    return sensing_viz


def _threshold_annotation_indices(
    mean_false_alarms: np.ndarray,
    detection_probability: np.ndarray,
) -> list[int]:
    """Select sparse threshold labels without repeating identical plot points."""

    if mean_false_alarms.shape != detection_probability.shape:
        raise ValueError("Threshold annotation coordinate arrays must have matching shapes")
    step = max(1, len(mean_false_alarms) // 5)
    selected: list[int] = []
    seen_coordinates: set[tuple[float, float]] = set()
    for index in range(0, len(mean_false_alarms), step):
        coordinate = (
            float(mean_false_alarms[index]),
            float(detection_probability[index]),
        )
        if not all(np.isfinite(coordinate)) or coordinate in seen_coordinates:
            continue
        seen_coordinates.add(coordinate)
        selected.append(index)
    return selected


# ---------------------------------------------------------------------------
# Vertex-velocity overlay (PLY mesh analysis)
# ---------------------------------------------------------------------------


def compute_vertex_velocity_overlay(
    mesh_dir: Path,
    *,
    mesh_fps: float,
    radar_pos: np.ndarray,
    target_pos: np.ndarray,
    bulk_velocity: float = 0.0,
    carrier_hz: float = 28.0e9,
    prf_hz: float = 2000.0,
    smooth: bool = True,
    smooth_kernel_size: int = 0,
) -> dict[str, Any]:
    """Compute per-vertex radial velocity statistics from consecutive PLY meshes.

    Args:
        mesh_dir: Directory containing sequentially-named PLY files.
        mesh_fps: Frame rate of the mesh sequence (Hz).
        radar_pos: 3-D position of the radar.
        target_pos: 3-D position of the target centre.
        bulk_velocity: Translational radial velocity to add (m/s), positive
            when approaching the radar.
        carrier_hz: Carrier frequency (Hz) for Doppler context.
        prf_hz: Pulse repetition frequency (Hz) for Doppler context.
        smooth: Apply median filter to vmin/vmax/vmean to remove interpolation
            spikes from upsampled target meshes. Enabled by default.
        smooth_kernel_size: Median filter kernel size (must be odd).  When 0
            (default), the kernel is auto-sized to ~5 ms worth of frames
            (``max(int(mesh_fps * 0.005) | 1, 3)``), capped at 51.

    Returns:
        Dict with ``time_s``, ``v_min``, ``v_max``, ``v_mean``,
        ``carrier_hz``, ``prf_hz``, and ``radial_velocity_convention``.
        All radial velocities are positive approaching, matching the Doppler
        axis used by ORCHAV sensing figures.
    """
    import trimesh

    mesh_dir = Path(mesh_dir)
    ply_files = sorted(mesh_dir.glob("*.ply"))
    if len(ply_files) < 2:
        raise FileNotFoundError(f"Need >= 2 PLY files in {mesh_dir}, found {len(ply_files)}")
    los = np.asarray(target_pos, dtype=np.float64) - np.asarray(radar_pos, dtype=np.float64)
    los_hat = los / np.linalg.norm(los)
    dt = 1.0 / mesh_fps
    prev_verts = np.array(trimesh.load(ply_files[0], process=False).vertices, dtype=np.float64)
    v_min_list: list[float] = []
    v_max_list: list[float] = []
    v_mean_list: list[float] = []
    for pf in ply_files[1:]:
        cur_verts = np.array(trimesh.load(pf, process=False).vertices, dtype=np.float64)
        velocity = (cur_verts - prev_verts) / dt
        # LOS points away from the radar; approach is decreasing path length.
        v_radial = -(velocity @ los_hat) + bulk_velocity
        v_min_list.append(float(np.min(v_radial)))
        v_max_list.append(float(np.max(v_radial)))
        v_mean_list.append(float(np.mean(v_radial)))
        prev_verts = cur_verts

    v_min_arr = np.array(v_min_list)
    v_max_arr = np.array(v_max_list)
    v_mean_arr = np.array(v_mean_list)

    if smooth and len(v_min_arr) > 3:
        from scipy.ndimage import median_filter

        if smooth_kernel_size <= 0:
            k = max(int(mesh_fps * 0.005) | 1, 3)
            k = min(k, 51)
            if k % 2 == 0:
                k += 1
        else:
            k = smooth_kernel_size
        v_min_arr = median_filter(v_min_arr, size=k)
        v_max_arr = median_filter(v_max_arr, size=k)
        v_mean_arr = median_filter(v_mean_arr, size=k)

    time_s = np.arange(len(v_min_arr)) * dt
    return {
        "time_s": time_s,
        "v_min": v_min_arr,
        "v_max": v_max_arr,
        "v_mean": v_mean_arr,
        "carrier_hz": carrier_hz,
        "prf_hz": prf_hz,
        "radial_velocity_convention": "positive_approaching",
    }


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------


def _extract_scene_overlay(
    frames: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    """Extract scene floor-plane bounding box from stored HDF5 vertices.

    The ``vertices`` dataset in HDF5 contains MPC path interaction points
    (reflection/scattering vertices on scene surfaces). Their x/y bounding
    box gives the scene extents for a wall outline overlay.

    Returns ``(x_min, x_max, y_min, y_max)`` or ``None``.
    """
    import h5py

    for frame in frames:
        path = frame.get("_path")
        if path is None:
            continue
        try:
            with h5py.File(str(path), "r") as f:
                if "vertices" not in f:
                    continue
                verts = np.asarray(f["vertices"][0], dtype=np.float64)
                flat = verts.reshape(-1, 3)
                mask = np.all(np.isfinite(flat), axis=1)
                flat = flat[mask]
                if flat.shape[0] == 0:
                    continue
                return (
                    float(flat[:, 0].min()),
                    float(flat[:, 0].max()),
                    float(flat[:, 1].min()),
                    float(flat[:, 1].max()),
                )
        except (OSError, KeyError):
            continue
    return None


def generate_figures(
    output_dir: Path,
    n_frames: int,
    *,
    frames: list[dict[str, Any]] | None = None,
    all_results: list[dict[str, Any]] | None = None,
    eval_result: ScenarioEvaluationResult | None = None,
    pruned_ghost_track_ids: set[int] | None = None,
    scene_overlay: bool = False,
    figure_names: set[str] | None = None,
    angular_evaluation_gates: tuple[float, float] | None = None,
) -> None:
    """Generate summary figures from post-processed results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return
    sensing_viz = _load_sensing_viz()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    def wants(name: str) -> bool:
        if figure_names is None:
            return name not in DEFAULT_FIGURE_EXCLUSIONS
        return name in figure_names

    # Extract scene floor plan for overlay (if requested)
    bounds: tuple | None = None
    if scene_overlay and frames:
        bounds = _extract_scene_overlay(frames)

    persisted_results = load_saved_results(output_dir)
    result_candidates = persisted_results or all_results or []
    render_results: list[dict[str, Any]] = []
    for result_index, result in enumerate(result_candidates):
        typed_sensing = result.get("_typed_sensing") if isinstance(result, Mapping) else None
        if not isinstance(typed_sensing, Mapping):
            logger.warning(
                "Skipping sensing render result %d: complete schema-v4 typed payload is required",
                result_index,
            )
            continue
        if result.get("rd_map") is None:
            continue
        render_results.append(result)

    # Collect all detections across frames
    all_detections: list[dict] = []
    last_rd_map = None
    last_range_axis = None
    last_doppler_axis = None
    last_render_result: dict[str, Any] | None = None
    figure_range_mode: str | None = None
    frame_lookup: dict[int, dict[str, Any]] = {}
    for idx, frame in enumerate(frames or []):
        frame_lookup[idx] = frame
        frame_lookup[int(frame.get("_frame_index", idx))] = frame

    for idx, result in enumerate(render_results):
        frame_index = int(result.get("_frame_index", idx))
        for det in deepcopy(result.get("detections") or []):
            det["frame"] = frame_index
            all_detections.append(det)
        last_rd_map = np.asarray(result["rd_map"])
        last_range_axis = np.asarray(result["range_axis"])
        last_doppler_axis = np.asarray(result["doppler_axis"])
        last_render_result = result
        mode = result.get("_range_mode")
        if not isinstance(mode, str) or not mode:
            raise ValueError(
                f"Typed sensing render result {frame_index} lacks exact config_resolved.range_mode"
            )
        figure_range_mode = mode

    # Optional saved-decision views share their interpretation with the dashboard.
    if last_render_result is not None:
        sensing = last_render_result["_typed_sensing"]
        if wants("detection_stages"):
            from shared.sensing.plotting.inspection import detection_stages_figure
            from shared.sensing.stage_inspection import load_detection_stage_bundle

            figure = detection_stages_figure(load_detection_stage_bundle(sensing))
            path = fig_dir / "detection_stages.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            generated.append(path)
        if wants("tracking_decisions"):
            from shared.sensing.plotting.inspection import tracking_decisions_figure
            from shared.sensing.tracking_inspection import tracking_decisions

            figure = tracking_decisions_figure(*tracking_decisions(sensing))
            path = fig_dir / "tracking_decisions.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            generated.append(path)

    # --- 1. RD map (last frame) ---
    if last_rd_map is not None and last_range_axis is not None and last_doppler_axis is not None:
        if last_render_result is None:
            raise RuntimeError("Typed sensing render state was lost")
        last_meta = last_render_result.get("config")
        if not isinstance(last_meta, Mapping):
            raise ValueError("Typed sensing render result lacks exact config_resolved")
        if figure_range_mode is None:
            raise ValueError("Typed sensing render result lacks an exact range mode")
        typed_sensing = last_render_result.get("_typed_sensing")
        if not isinstance(typed_sensing, Mapping):
            raise ValueError("Typed sensing render result lacks its schema-v4 payload")
        if wants("range_doppler"):
            try:
                p = sensing_viz.create_range_doppler_figure(
                    typed_sensing,
                    output_path=fig_dir / "range_doppler.png",
                    zoom_to_data=True,
                )
                generated.append(p)
            except (ValueError, KeyError) as exc:
                logger.warning("RD figure failed: %s", exc)

        # Range profile (uses same sensing_data)
        if wants("range_profile"):
            try:
                p = sensing_viz.create_range_profile_figure(
                    typed_sensing, fig_dir / "range_profile.png", zoom_to_data=True
                )
                generated.append(p)
            except (ValueError, KeyError) as exc:
                logger.info("Range-profile figure skipped: %s", exc)

        requested_angular = {
            name
            for name in (
                "range_azimuth",
                "doppler_azimuth",
                "angular_overview",
                "azimuth_elevation",
                "range_elevation",
                "detections_3d",
            )
            if wants(name)
        }
        if requested_angular:
            logger.info(
                "Typed angular summary figures skipped (%s): the legacy array wrappers infer "
                "coordinates/config and cannot consume schema-v4 products; use a typed "
                "range_doppler_angle consumer",
                ", ".join(sorted(requested_angular)),
            )

    if wants("angular_tracking") or wants("angular_errors"):
        from shared.sensing.angular import angular_event, create_angular_summary

        angular_events = [
            event
            for result in render_results
            if (event := angular_event(result["_typed_sensing"])) is not None
        ]
        if angular_events:
            if angular_evaluation_gates is not None:
                for event in angular_events:
                    event["range_gate_m"], event["velocity_gate_m_s"] = angular_evaluation_gates
            generated.extend(create_angular_summary(angular_events, fig_dir))
        else:
            logger.info("Angular GT figures skipped: no receiver-local angular truth was retained")

    # --- 2. Detection timeline ---
    gt_history: list[dict[str, Any]] = []
    needs_gt_history = (
        wants("detection_timeline")
        or wants("tracking")
        or wants("tracking_xy")
        or wants("tracking_xyz")
    )
    if needs_gt_history and render_results:
        try:
            gt_history = _build_gt_history(
                all_detections,
                render_results=render_results,
            )
        except ValueError as exc:
            logger.warning("Canonical GT history unavailable: %s", exc)
        if wants("detection_timeline") and gt_history:
            try:
                p = sensing_viz.create_gt_evolution_figure(
                    gt_history, fig_dir / "detection_timeline.png"
                )
                generated.append(p)
            except (ValueError, KeyError) as exc:
                logger.warning("GT evolution figure failed: %s", exc)

    # --- 2b. Tracking figure (Kalman tracks vs GT + raw detections) ---
    if (
        gt_history
        and all_detections
        and (wants("tracking") or wants("tracking_xy") or wants("tracking_xyz"))
    ):
        track_history = _build_tracking_history(all_detections, frame_lookup=frame_lookup)
        if track_history:
            # Mark short-lived tracks as ghost candidates (visible in figure)
            short_lived_ids = {tid for tid, hist in track_history.items() if len(hist) < 5}
            all_pruned = (pruned_ghost_track_ids or set()) | short_lived_ids
            if wants("tracking"):
                try:
                    p = sensing_viz.create_tracking_figure(
                        gt_history,
                        track_history,
                        fig_dir / "tracking.png",
                        pruned_track_ids=all_pruned if all_pruned else None,
                    )
                    generated.append(p)
                except (ValueError, KeyError) as exc:
                    logger.warning("Tracking figure failed: %s", exc)
            if wants("tracking_xy") or wants("tracking_xyz"):
                gt_xy_history: dict[str, list[dict[str, Any]]] = {}
                radar_history: list[dict[str, Any]] = []
                try:
                    gt_xy_history, radar_history = _build_xy_ground_truth_history(frames or [])
                    if wants("tracking_xy"):
                        p = sensing_viz.create_xy_tracking_figure(
                            track_history,
                            fig_dir / "tracking_xy.png",
                            raw_detections=_build_world_xy_detections(
                                all_detections,
                                frame_lookup=frame_lookup,
                            ),
                            gt_tracks=gt_xy_history if gt_xy_history else None,
                            radar_positions=radar_history if radar_history else None,
                            pruned_track_ids=all_pruned if all_pruned else None,
                            scene_bounds=bounds,
                        )
                        generated.append(p)
                except (ValueError, KeyError) as exc:
                    logger.warning("X/Y tracking figure failed: %s", exc)

            # --- 2c. 3D tracking figure (when z data is available) ---
            has_z = any(
                np.isfinite(e.get("z_m", float("nan")))
                for entries in track_history.values()
                for e in entries
            )
            if has_z and wants("tracking_xyz"):
                try:
                    sensing_viz.create_xyz_tracking_figure(
                        track_history,
                        gt_tracks=gt_xy_history if gt_xy_history else None,
                        pruned_track_ids=all_pruned if all_pruned else None,
                        scene_bounds=bounds,
                        output_path=fig_dir / "tracking_xyz.png",
                    )
                    generated.append(fig_dir / "tracking_xyz.png")
                except (ValueError, KeyError) as exc:
                    logger.warning("3D tracking figure failed: %s", exc)

    # --- 3. Detection summary (TP/FP/FN + RMSE) ---
    if eval_result is not None and (wants("detection_summary") or wants("detection_events")):
        assoc_history = _build_association_history(eval_result)
        if wants("detection_summary"):
            try:
                p = sensing_viz.create_detection_summary_figure(
                    assoc_history, fig_dir / "detection_summary.png"
                )
                generated.append(p)
            except (ValueError, KeyError) as exc:
                logger.warning("Detection summary figure failed: %s", exc)
        if wants("detection_events"):
            try:
                p = sensing_viz.create_detection_events_figure(
                    assoc_history, fig_dir / "detection_events.png"
                )
                generated.append(p)
            except (ValueError, KeyError) as exc:
                logger.warning("Detection events figure failed: %s", exc)

    # --- 4. Running Pd and mean false-alarm count over evaluated events ---
    if wants("cumulative_detection_performance") and eval_result is not None:
        p = _plot_cumulative_detection_performance(fig_dir, eval_result, plt)
        if p:
            generated.append(p)

    # --- 5. Error distribution ---
    if (
        wants("error_distribution")
        and eval_result is not None
        and (eval_result.all_range_errors or eval_result.all_velocity_errors)
    ):
        p = _plot_error_distribution(fig_dir, eval_result, plt)
        if p:
            generated.append(p)

    # --- 6. Detections-per-frame histogram ---
    if wants("detections_per_frame") and all_detections and render_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Use actual sensing frame indices (not contiguous 0..N — coherent CPI
        # frames can be sparse, e.g. every 64th step).
        sensing_frame_indices = sorted(
            set(int(r.get("_frame_index", idx)) for idx, r in enumerate(render_results))
        )
        det_count_map: dict[int, int] = {}
        for d in all_detections:
            fi = d["frame"]
            det_count_map[fi] = det_count_map.get(fi, 0) + 1
        counts = [det_count_map.get(fi, 0) for fi in sensing_frame_indices]
        ax.bar(
            range(len(sensing_frame_indices)),
            counts,
            width=1.0,
            color="steelblue",
            alpha=0.8,
        )
        mean_det = np.mean(counts) if counts else 0.0
        ax.set_xlabel("Sensing Frame")
        ax.set_ylabel("Detections")
        ax.set_title(
            f"Detections per Frame (mean={mean_det:.2f}, {len(sensing_frame_indices)} frames)"
        )
        p = fig_dir / "detections_per_frame.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)

    logger.info("Figures: generated %d files in %s", len(generated), fig_dir)


def _build_gt_history(
    all_detections: list[dict],
    *,
    render_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build identity/time-keyed GT history from complete schema-v4 results."""
    # Index detections by frame
    dets_by_frame: dict[int, list[dict]] = {}
    for d in all_detections:
        dets_by_frame.setdefault(d["frame"], []).append(d)

    gt_history: list[dict[str, Any]] = []
    for pos_idx, result in enumerate(render_results):
        actual_frame = int(result.get("_frame_index", pos_idx))
        typed_sensing = result.get("_typed_sensing")
        if not isinstance(typed_sensing, Mapping):
            raise ValueError(
                f"Sensing result {actual_frame} has no complete typed schema-v4 payload"
            )
        temporal = typed_sensing.get("temporal")
        truth_temporal = temporal.get("truth") if isinstance(temporal, Mapping) else None
        raw_reference_time_s = (
            truth_temporal.get("reference_time_s") if isinstance(truth_temporal, Mapping) else None
        )
        try:
            reference_time_s = float(raw_reference_time_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sensing result {actual_frame} truth lacks an exact finite reference_time_s"
            ) from exc
        if not np.isfinite(reference_time_s):
            raise ValueError(
                f"Sensing result {actual_frame} truth lacks an exact finite reference_time_s"
            )
        stored_gt = extract_recorded_evaluation_truth(typed_sensing)
        if stored_gt is None:
            raise ValueError(f"Sensing result {actual_frame} has no canonical truth snapshot")
        if stored_gt.get("_evaluation_unavailable_reason"):
            continue
        frame_dets = dets_by_frame.get(actual_frame, [])
        gt_history.append(
            {
                "step": actual_frame,
                "reference_time_s": reference_time_s,
                "target_ids": stored_gt["target_ids"],
                "target_ranges": stored_gt["target_ranges"],
                "target_speeds": stored_gt["target_speeds"],
                "target_velocity_valid": stored_gt["target_velocity_valid"],
                "target_velocity_sources": stored_gt["target_velocity_sources"],
                "target_velocity_reference_dt_s": stored_gt["target_velocity_reference_dt_s"],
                "target_velocity_unavailable_reasons": stored_gt[
                    "target_velocity_unavailable_reasons"
                ],
                **{
                    key: stored_gt[key]
                    for key in (
                        "evaluation_reference_protocol",
                        "target_velocity_references",
                        "target_velocity_reference_evidence",
                        "recorded_target_speeds",
                    )
                    if key in stored_gt
                },
                "det_ranges": [d["range_m"] for d in frame_dets],
                "det_velocities": [d["velocity_m_s"] for d in frame_dets],
            }
        )
    return gt_history


def build_tracking_history(
    all_detections: list[dict],
    *,
    frame_lookup: dict[int, dict[str, Any]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Build per-track history from detection results.

    Returns:
        ``{track_id: [{"frame": int, "range_m": float, "velocity_m_s": float,
        "confirmed": bool, "raw_range": float, "raw_velocity": float}, ...]}``
    """
    tracks: dict[int, list[dict[str, Any]]] = {}
    for d in all_detections:
        tid = d.get("track_id")
        if tid is None:
            continue
        local_track_x = float(d.get("track_x_m", d.get("x_m", np.nan)))
        local_track_y = float(d.get("track_y_m", d.get("y_m", np.nan)))
        local_raw_x = float(d.get("x_m", np.nan))
        local_raw_y = float(d.get("y_m", np.nan))
        world_track_x, world_track_y = np.nan, np.nan
        world_raw_x, world_raw_y = np.nan, np.nan
        radar_z = 0.0
        if frame_lookup is not None:
            pose = _extract_sensor_pose(frame_lookup.get(int(d.get("frame", 0)), {}))
            if pose is not None:
                origin_xy, yaw_deg, radar_z = pose
                if np.isfinite(local_track_x) and np.isfinite(local_track_y):
                    world_track_x, world_track_y = _local_xy_to_world(
                        local_track_x, local_track_y, origin_xy, yaw_deg
                    )
                if np.isfinite(local_raw_x) and np.isfinite(local_raw_y):
                    world_raw_x, world_raw_y = _local_xy_to_world(
                        local_raw_x, local_raw_y, origin_xy, yaw_deg
                    )
        entry = {
            "frame": d.get("frame", 0),
            "range_m": float(d.get("track_range_m", d.get("range_m", 0))),
            "velocity_m_s": float(d.get("track_velocity_m_s", d.get("velocity_m_s", 0))),
            "x_m": float(d.get("track_x_m", d.get("x_m", np.nan))),
            "y_m": float(d.get("track_y_m", d.get("y_m", np.nan))),
            "z_m": float(d.get("track_z_m", d.get("z_m", np.nan))) + radar_z,
            "world_x_m": float(world_track_x),
            "world_y_m": float(world_track_y),
            "confirmed": bool(d.get("track_confirmed", True)),
            "raw_range": float(d.get("range_m", 0)),
            "raw_velocity": float(d.get("velocity_m_s", 0)),
            "raw_x": float(d.get("x_m", np.nan)),
            "raw_y": float(d.get("y_m", np.nan)),
            "raw_world_x": float(world_raw_x),
            "raw_world_y": float(world_raw_y),
        }
        tracks.setdefault(int(tid), []).append(entry)
    # Sort each track's history by frame
    for tid in tracks:
        tracks[tid].sort(key=lambda e: e["frame"])
    return tracks


_build_tracking_history = build_tracking_history


def _extract_sensor_pose(
    frame: dict[str, Any],
) -> tuple[np.ndarray, float, float] | None:
    """Return the typed sensing pair's RX origin, yaw in degrees, and height.

    Returns:
        Tuple of ``(origin_xy_m, yaw_deg, origin_z_m)`` or ``None``.
    """
    sensing = frame.get("_typed_sensing")
    if sensing is None:
        sensing = frame.get("sensing")
    if not isinstance(sensing, Mapping):
        return None
    pair_axis = sensing.get("pair_axis")
    if not isinstance(pair_axis, list) or len(pair_axis) != 1:
        raise ValueError("Sensing figure pose requires exactly one typed pair_axis row")
    pair = pair_axis[0]
    if not isinstance(pair, Mapping) or pair.get("pair_index") != 0:
        raise ValueError("Sensing figure pose requires typed pair_axis pair_index 0")
    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping):
        raise ValueError("Sensing figure pair_axis has no temporal acquisition identity")
    for field in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
        if pair.get(field) != acquisition.get(field):
            raise ValueError(
                f"Sensing figure pair_axis {field} disagrees with temporal acquisition"
            )
    frame_pair_index = pair.get("frame_pair_index")
    if (
        not isinstance(frame_pair_index, int)
        or isinstance(frame_pair_index, bool)
        or frame_pair_index < 0
    ):
        raise ValueError("Sensing figure pair_axis requires a nonnegative frame_pair_index")

    positions = np.asarray(frame.get("rx_positions"), dtype=np.float64)
    orientations = np.asarray(frame.get("rx_orientations"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("Sensing figure RX positions must have shape [frame_pair, 3]")
    if orientations.ndim != 2 or orientations.shape[1:] != (3,):
        raise ValueError("Sensing figure RX orientations must have shape [frame_pair, 3]")
    if frame_pair_index >= len(positions) or frame_pair_index >= len(orientations):
        raise ValueError("Sensing figure pair index is outside the RX pose arrays")
    position = positions[frame_pair_index]
    orientation = orientations[frame_pair_index]
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise ValueError("Sensing figure RX pose must contain finite coordinates")
    return position[:2], float(np.degrees(orientation[0])), float(position[2])


def _local_xy_to_world(
    local_x_m: float,
    local_y_m: float,
    origin_xy_m: np.ndarray,
    yaw_deg: float,
) -> tuple[float, float]:
    """Rotate and translate sensor-local x/y into world x/y."""
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    world_x = float(origin_xy_m[0] + cos_yaw * local_x_m - sin_yaw * local_y_m)
    world_y = float(origin_xy_m[1] + sin_yaw * local_x_m + cos_yaw * local_y_m)
    return world_x, world_y


def _build_world_xy_detections(
    all_detections: list[dict[str, Any]],
    *,
    frame_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Transform per-detection local x/y into world x/y when radar pose is available."""
    world_detections: list[dict[str, Any]] = []
    for detection in all_detections:
        local_x = float(detection.get("x_m", np.nan))
        local_y = float(detection.get("y_m", np.nan))
        if not (np.isfinite(local_x) and np.isfinite(local_y)):
            continue
        pose = _extract_sensor_pose(frame_lookup.get(int(detection.get("frame", 0)), {}))
        if pose is None:
            continue
        origin_xy, yaw_deg, _origin_z = pose
        world_x, world_y = _local_xy_to_world(local_x, local_y, origin_xy, yaw_deg)
        world_detections.append(
            {
                "frame": int(detection.get("frame", 0)),
                "x_m": world_x,
                "y_m": world_y,
            }
        )
    return world_detections


def _build_xy_ground_truth_history(
    frames: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Build ID-keyed world-plane histories from exact schema-v4 truth and pair pose."""
    gt_tracks: dict[str, list[dict[str, Any]]] = {}
    radar_history: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        frame_index = int(frame.get("_frame_index", idx))
        pose = _extract_sensor_pose(frame)
        if pose is not None:
            origin_xy, _, _origin_z = pose
            radar_history.append(
                {
                    "frame": frame_index,
                    "x_m": float(origin_xy[0]),
                    "y_m": float(origin_xy[1]),
                }
            )
        sensing = frame.get("_typed_sensing")
        if sensing is None:
            sensing = frame.get("sensing")
        truth = sensing.get("truth") if isinstance(sensing, Mapping) else None
        targets = truth.get("targets") if isinstance(truth, Mapping) else None
        if targets is None:
            continue
        if not isinstance(targets, list):
            raise ValueError("Sensing figure truth.targets must be a list")
        seen_target_ids: set[str] = set()
        for target in targets:
            if not isinstance(target, Mapping):
                raise ValueError("Sensing figure truth target must be a mapping")
            target_id = target.get("target_id")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("Sensing figure truth target requires an exact target_id")
            if target_id in seen_target_ids:
                raise ValueError(f"Sensing figure truth repeats target_id {target_id!r}")
            seen_target_ids.add(target_id)
            if target.get("lifecycle") != "active" or target.get("valid") is not True:
                continue
            position = np.asarray(target.get("position_m"), dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(
                    f"Sensing figure truth target {target_id!r} lacks a finite position_m"
                )
            gt_tracks.setdefault(target_id, []).append(
                {
                    "frame": frame_index,
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                }
            )
    return gt_tracks, radar_history


def _build_association_history(
    eval_result: ScenarioEvaluationResult,
) -> list[dict[str, Any]]:
    """Build association_history list for ``create_detection_summary_figure``."""
    return [
        {
            "tp": eval_result.per_event_tp[i],
            "fp": eval_result.per_event_fp[i],
            "fn": eval_result.per_event_fn[i],
            "range_rmse": (
                np.nan
                if eval_result.per_event_range_rmse[i] is None
                else eval_result.per_event_range_rmse[i]
            ),
            "velocity_rmse": (
                np.nan
                if eval_result.per_event_velocity_rmse[i] is None
                else eval_result.per_event_velocity_rmse[i]
            ),
        }
        for i in range(eval_result.evaluated_event_count)
    ]


def _plot_cumulative_detection_performance(
    fig_dir: Path,
    eval_result: ScenarioEvaluationResult,
    plt: Any,
) -> Path | None:
    """Plot running Pd and mean false alarms per evaluated event."""
    n = eval_result.evaluated_event_count
    if n < 2:
        return None

    tp_cum = np.cumsum(eval_result.per_event_tp)
    fp_cum = np.cumsum(eval_result.per_event_fp)
    fn_cum = np.cumsum(eval_result.per_event_fn)
    positives_cum = tp_cum + fn_cum

    pd_running = np.full(n, np.nan, dtype=np.float64)
    np.divide(tp_cum, positives_cum, out=pd_running, where=positives_cum > 0)
    events_arange = np.arange(1, n + 1)
    mean_false_alarms_running = fp_cum / events_arange

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(n), pd_running, color="#2ca02c", linewidth=1.2, label="Pd")
    ax1.set_xlabel("Evaluated event")
    ax1.set_ylabel("Detection Probability", color="#2ca02c")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        range(n),
        mean_false_alarms_running,
        color="#d62728",
        linewidth=1.2,
        label="Mean false alarms/event",
    )
    ax2.set_ylabel("Mean False Alarms per Evaluated Event", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax1.set_title("Cumulative Detection Performance")
    p = fig_dir / "cumulative_detection_performance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_error_distribution(
    fig_dir: Path,
    eval_result: ScenarioEvaluationResult,
    plt: Any,
) -> Path | None:
    """Histogram of range and velocity errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if eval_result.all_range_errors:
        ax1.hist(
            eval_result.all_range_errors, bins=30, color="#1f77b4", alpha=0.8, edgecolor="white"
        )
        ax1.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax1.set_xlabel("Range Error (m)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Range Errors (RMSE={eval_result.cumulative_range_rmse:.2f} m)")
    else:
        ax1.text(0.5, 0.5, "No matched pairs", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Range Errors")

    if eval_result.all_velocity_errors:
        ax2.hist(
            eval_result.all_velocity_errors, bins=30, color="#9467bd", alpha=0.8, edgecolor="white"
        )
        ax2.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax2.set_xlabel("Velocity Error (m/s)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Velocity Errors (RMSE={eval_result.cumulative_velocity_rmse:.2f} m/s)")
    else:
        ax2.text(0.5, 0.5, "No matched pairs", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Velocity Errors")

    fig.tight_layout()
    p = fig_dir / "error_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def generate_threshold_sweep_figures(
    output_dir: Path,
    sweep_results: list[ThresholdSweepPoint],
) -> None:
    """Generate threshold-sweep curves with explicitly named denominators."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping threshold-sweep figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    mean_false_alarms = np.array(
        [
            (
                np.nan
                if pt.mean_false_alarms_per_evaluated_event is None
                else pt.mean_false_alarms_per_evaluated_event
            )
            for pt in sweep_results
        ],
        dtype=np.float64,
    )
    pd = np.array(
        [np.nan if point.pd is None else point.pd for point in sweep_results],
        dtype=np.float64,
    )
    miss_rate = np.array(
        [np.nan if point.miss_rate is None else point.miss_rate for point in sweep_results],
        dtype=np.float64,
    )
    thresholds = np.array([pt.threshold_scale for pt in sweep_results])
    total_dets = np.array([pt.total_detections for pt in sweep_results])

    # 1. Detection probability vs mean false-alarm count. The horizontal
    # quantity is a count per evaluated event, not a false-alarm probability.
    fig, ax = plt.subplots(figsize=(7, 5))
    valid_detection = np.isfinite(mean_false_alarms) & np.isfinite(pd)
    if valid_detection.any():
        ax.plot(
            mean_false_alarms[valid_detection],
            pd[valid_detection],
            "o-",
            color="#1f77b4",
            markersize=4,
            linewidth=1.2,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Detection probability undefined\n(no target opportunities)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    # Annotate a few points with threshold values
    for i in _threshold_annotation_indices(mean_false_alarms, pd):
        ax.annotate(
            f"T={thresholds[i]:.1f}",
            (mean_false_alarms[i], pd[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            color="gray",
        )
    ax.set_xlabel("Mean False Alarms per Evaluated Event")
    ax.set_ylabel("Detection Probability")
    ax.set_title("Detection Performance (CFAR Threshold Sweep)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    p = fig_dir / "detection_probability_vs_mean_false_alarms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(p)

    # 2. Miss rate vs mean false-alarm count (log scale where defined).
    fig, ax = plt.subplots(figsize=(7, 5))
    finite_miss = np.isfinite(mean_false_alarms) & np.isfinite(miss_rate)
    valid = finite_miss & (mean_false_alarms > 0) & (miss_rate > 0)
    if valid.any():
        ax.plot(
            mean_false_alarms[valid],
            miss_rate[valid],
            "o-",
            color="#d62728",
            markersize=4,
            linewidth=1.2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif finite_miss.any():
        ax.plot(
            mean_false_alarms[finite_miss],
            miss_rate[finite_miss],
            "o-",
            color="#d62728",
            markersize=4,
            linewidth=1.2,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Miss rate undefined\n(no target opportunities)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Mean False Alarms per Evaluated Event")
    ax.set_ylabel("Miss Rate")
    ax.set_title("Miss-Rate Performance (CFAR Threshold Sweep)")
    ax.grid(True, alpha=0.3, which="both")
    p = fig_dir / "miss_rate_vs_mean_false_alarms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(p)

    # 3. Threshold vs detections with detection probability on the secondary axis.
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(thresholds, total_dets, "s-", color="#1f77b4", markersize=4, linewidth=1.2)
    ax1.set_xlabel("CFAR Threshold Scale")
    ax1.set_ylabel("Total Detections", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    valid_pd = np.isfinite(pd)
    if valid_pd.any():
        ax2.plot(
            thresholds[valid_pd],
            pd[valid_pd],
            "o-",
            color="#2ca02c",
            markersize=4,
            linewidth=1.2,
        )
    ax2.set_ylabel("Detection Probability", color="#2ca02c")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title("Detections and Detection Probability by CFAR Threshold")
    ax1.grid(True, alpha=0.3)
    p = fig_dir / "threshold_vs_detections.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(p)

    logger.info("Threshold-sweep figures: generated %d files in %s", len(generated), fig_dir)


def _exact_rd_animation_contract(
    render_results: Sequence[Mapping[str, Any]],
) -> tuple[float, str]:
    """Return the one exact carrier/range contract shared by RD animation rows."""

    if not render_results:
        raise ValueError("RD animation requires at least one typed sensing result")
    reference: tuple[float, str] | None = None
    for result_index, result in enumerate(render_results):
        config = result.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"RD animation result {result_index} lacks exact config_resolved")
        carrier_hz = config.get("carrier_freq_hz")
        range_mode = config.get("range_mode")
        if (
            isinstance(carrier_hz, bool)
            or not isinstance(carrier_hz, (int, float))
            or not np.isfinite(float(carrier_hz))
            or float(carrier_hz) <= 0.0
        ):
            raise ValueError(
                f"RD animation result {result_index} lacks a positive exact carrier_freq_hz"
            )
        if range_mode not in {"monostatic", "bistatic"}:
            raise ValueError(
                f"RD animation result {result_index} lacks an exact monostatic/bistatic "
                "range_mode"
            )
        result_mode = result.get("_range_mode")
        if result_mode is not None and result_mode != range_mode:
            raise ValueError(
                f"RD animation result {result_index} range-mode mirror disagrees with "
                "config_resolved"
            )
        candidate = (float(carrier_hz), str(range_mode))
        if reference is None:
            reference = candidate
        elif candidate != reference:
            raise ValueError(
                "RD animation results do not share one exact carrier frequency and range mode"
            )
    if reference is None:  # pragma: no cover - guarded by the non-empty check above.
        raise RuntimeError("RD animation contract resolution lost its reference result")
    return reference


def generate_rd_gif(
    output_dir: Path,
    n_frames: int,
    *,
    all_results: list[dict[str, Any]] | None = None,
    fps: int = 10,
    interpolation: str = "nearest",
    stride: int = 1,
    per_frame_vmax: bool = False,
    show_detections: bool = True,
) -> None:
    """Render each frame's RD map and compile into an animated GIF.

    Args:
        stride: Subsample every N-th frame (1 = all frames).
        per_frame_vmax: If True, each frame uses its own max power for
            the colorscale instead of the global max across all frames.
            This reveals per-frame structure (clutter, weak targets) that
            would otherwise be hidden by the dominant close-range frame.
        show_detections: If True (default), overlay CFAR detection markers.
            Set to False to generate clean RD map GIFs without markers.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        logger.warning("matplotlib or Pillow not available, skipping GIF")
        return
    sensing_viz = _load_sensing_viz()

    import io

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    render_results = [
        r for r in (all_results or load_saved_results(output_dir)) if r.get("rd_map") is not None
    ]
    if not render_results:
        logger.warning("No RD maps found for GIF generation")
        return

    # Collect RD maps (subsampled by stride) and detect global power range
    rd_maps: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    carrier_hz, gif_range_mode = _exact_rd_animation_contract(render_results)
    all_detections: dict[int, list[dict]] = {}
    velocity_display = detection_velocity_display(range_mode=gif_range_mode)

    for idx, result in enumerate(render_results):
        frame_index = int(result.get("_frame_index", idx))
        all_detections[frame_index] = deepcopy(result.get("detections") or [])
        if frame_index % stride != 0:
            continue
        rd_maps.append(
            (
                frame_index,
                np.asarray(result["rd_map"]),
                np.asarray(result["range_axis"]),
                np.asarray(result["doppler_axis"]),
            )
        )

    if not rd_maps:
        logger.warning("No RD maps found for GIF generation")
        return

    # Global power range for consistent colorscale
    all_db = [10 * np.log10(np.maximum(rd, 1e-30)).max() for _, rd, _, _ in rd_maps]
    global_vmax = max(all_db)
    global_vmin = global_vmax - 60

    vmax_label = "per-frame" if per_frame_vmax else "global"
    use_fast_path = not show_detections and interpolation == "nearest"
    mode_label = "fast (PIL)" if use_fast_path else "matplotlib"
    logger.info(
        "Rendering %d RD map frames for GIF (%s, stride=%d, %d total, vmax=%s)...",
        len(rd_maps),
        mode_label,
        stride,
        n_frames,
        vmax_label,
    )
    pil_frames: list[Image.Image] = []

    if use_fast_path:
        # Fast path: bypass matplotlib, map dB values directly through a
        # viridis LUT.  ~10-50x faster than the matplotlib path.
        from matplotlib import colormaps

        cmap = colormaps["viridis"]
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        for _frame_idx, rd_map, _unused_range_axis, _unused_doppler_axis in rd_maps:
            rd_db = 10 * np.log10(np.maximum(rd_map, 1e-30))
            if per_frame_vmax:
                fmax = float(rd_db.max())
                fmin = fmax - 60
            else:
                fmax = global_vmax
                fmin = global_vmin
            normed = np.clip((rd_db - fmin) / max(fmax - fmin, 1e-12), 0, 1)
            indices = (normed * 255).astype(np.uint8)
            # Flip vertically to match origin="lower" convention
            rgb = lut[indices[::-1]]
            pil_frames.append(Image.fromarray(rgb, "RGB"))
    else:
        # Matplotlib path: supports detection overlays and interpolation modes.
        # Recreate figure each frame to avoid colorbar accumulation.
        for frame_idx, rd_map, range_axis, doppler_axis in rd_maps:
            fig, ax = plt.subplots(figsize=(8, 5))
            rd_db = 10 * np.log10(np.maximum(rd_map, 1e-30))
            vel_axis = doppler_axis_to_velocity_m_s(doppler_axis, carrier_hz)
            extent = [
                float(range_axis[0]),
                float(range_axis[-1]),
                float(vel_axis[0]),
                float(vel_axis[-1]),
            ]

            if per_frame_vmax:
                frame_vmax = float(rd_db.max())
                frame_vmin = frame_vmax - 60
            else:
                frame_vmax = global_vmax
                frame_vmin = global_vmin

            im = ax.imshow(
                rd_db,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=frame_vmin,
                vmax=frame_vmax,
                interpolation=interpolation,
            )

            if show_detections:
                dets = all_detections.get(frame_idx, [])
                if dets:
                    det_r = [d["range_m"] for d in dets]
                    det_v = [d["velocity_m_s"] for d in dets]
                    ax.scatter(
                        det_r, det_v, marker="x", c="red", s=80, linewidths=2, label="Detection"
                    )

            ax.set_xlabel("Range (m)")
            ax.set_ylabel(velocity_display.axis_label)
            if not sensing_viz._suppress_titles():
                ax.set_title(f"Range-Doppler Map — Frame {frame_idx}/{n_frames - 1}")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            pil_frames.append(Image.open(buf).convert("RGB"))
            plt.close(fig)

    interp_suffix = f"_{interpolation}" if interpolation != "nearest" else ""
    vmax_suffix = "_perframe" if per_frame_vmax else ""
    det_suffix = "_nodet" if not show_detections else ""
    gif_path = fig_dir / f"rd_evolution{interp_suffix}{vmax_suffix}{det_suffix}.gif"
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    logger.info("GIF saved: %s (%d frames, %d fps)", gif_path, len(pil_frames), fps)


def generate_rd_gif_3d(
    output_dir: Path,
    n_frames: int,
    *,
    all_results: list[dict[str, Any]] | None = None,
    fps: int = 10,
) -> None:
    """Render each frame's RD map as a 3D surface and compile into a GIF."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from PIL import Image
    except ImportError:
        logger.warning("matplotlib or Pillow not available, skipping 3D GIF")
        return
    sensing_viz = _load_sensing_viz()

    import io

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    render_results = [
        r for r in (all_results or load_saved_results(output_dir)) if r.get("rd_map") is not None
    ]
    if not render_results:
        logger.warning("No RD maps found for 3D GIF generation")
        return

    # Collect all RD maps
    rd_maps: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    carrier_hz, gif_range_mode = _exact_rd_animation_contract(render_results)
    velocity_display = detection_velocity_display(range_mode=gif_range_mode)

    # Global power range
    for idx, result in enumerate(render_results):
        frame_index = int(result.get("_frame_index", idx))
        rd_maps.append(
            (
                frame_index,
                np.asarray(result["rd_map"]),
                np.asarray(result["range_axis"]),
                np.asarray(result["doppler_axis"]),
            )
        )

    all_db = [10 * np.log10(np.maximum(rd, 1e-30)).max() for _, rd, _, _ in rd_maps]
    vmax = max(all_db)
    vmin = vmax - 60

    logger.info("Rendering %d 3D RD map frames for GIF...", len(rd_maps))
    pil_frames: list[Image.Image] = []

    for frame_idx, rd_map, range_axis, doppler_axis in rd_maps:
        rd_db = 10 * np.log10(np.maximum(rd_map, 1e-30))
        rd_db_clipped = np.clip(rd_db, vmin, vmax)
        vel_axis = doppler_axis_to_velocity_m_s(doppler_axis, carrier_hz)
        R, V = np.meshgrid(range_axis, vel_axis)

        # Normalize for colormap
        norm = (rd_db_clipped - vmin) / (vmax - vmin)
        colors = cm.viridis(norm)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(R, V, rd_db_clipped, facecolors=colors, rstride=1, cstride=1, shade=False)
        ax.set_xlabel("Range (m)")
        ax.set_ylabel(velocity_display.axis_label)
        ax.set_zlabel("Power (dB)")
        ax.set_zlim(vmin, vmax)
        if not sensing_viz._suppress_titles():
            ax.set_title(f"Range-Doppler 3D — Frame {frame_idx}/{n_frames - 1}")
        ax.view_init(elev=35, azim=-60)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("RGB"))

    gif_path = fig_dir / "rd_evolution_3d.gif"
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    logger.info("3D GIF saved: %s (%d frames, %d fps)", gif_path, len(pil_frames), fps)


def load_cir_sequence(
    frames_dir: Path,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load the canonical typed delay-domain sequence from persisted frames.

    Exact product coordinates are authoritative. Overlapping typed slow-time
    windows are de-duplicated by epoch only when the repeated complex rows are
    identical; compatibility arrays, strategy names, frame IDs, and configured
    cadence are never used to select or time the samples.

    Returns:
        Tuple of (fast_time_samples list, metadata dict from valid frames).
    """
    fast_time_samples: list[np.ndarray] = []
    sample_times_s: list[float] = []
    timed_acquisition_counts: list[tuple[float, int]] = []
    config: dict[str, Any] = {}
    sequence_reference: DelayDomainSamples | None = None
    frames = load_frames_sensing(Path(frames_dir))

    for frame in frames:
        typed_sensing = frame.get("_typed_sensing")
        if not isinstance(typed_sensing, Mapping):
            continue
        frame_config = frame.get("config")
        if not isinstance(frame_config, Mapping):
            raise ValueError(
                f"Sensing frame {frame.get('_frame_index')} lacks exact config_resolved"
            )
        if not config:
            config = deepcopy(dict(frame_config))
        elif config != dict(frame_config):
            raise ValueError("Delay-domain sequence mixes different exact resolved configs")
        product = _delay_samples_for_analysis(typed_sensing)
        if product is None:
            continue
        if sequence_reference is None:
            sequence_reference = product
        else:
            require_compatible_delay_sequence(sequence_reference, product)
        first_new_index = len(sample_times_s)
        appended = append_delay_domain_samples(
            product,
            fast_time_samples,
            sample_times_s,
        )
        if appended:
            timed_acquisition_counts.append((sample_times_s[first_new_index], appended))

    config["sample_times_s"] = sample_times_s
    config["sample_time_source"] = "typed_delay_domain_product_coordinates"
    if sequence_reference is not None:
        config["range_axis_m"] = np.array(sequence_reference.range_axis_m, copy=True)
        config["pair_id"] = sequence_reference.pair_id
    config["acquisition_sample_counts"] = [
        count for _, count in sorted(timed_acquisition_counts, key=lambda item: item[0])
    ]
    return fast_time_samples, config


def _delay_samples_for_analysis(sensing: Mapping[str, Any]) -> DelayDomainSamples | None:
    """Explicit micro-Doppler analysis can reconstruct the canonical CFR input."""
    if not isinstance(sensing.get("temporal", {}).get("result"), Mapping):
        return None
    product = extract_delay_domain_samples(sensing)
    if product is None and any(
        d.get("kind") == "baseband_frequency_observation"
        for d in sensing.get("products", {}).values()
    ):
        from shared.sensing.cfr_replay import reconstruct_delay_observation

        product = extract_delay_domain_samples(reconstruct_delay_observation(sensing))
    return product


def _uniform_sample_interval_s(metadata: Mapping[str, Any], sample_count: int) -> float:
    """Return the uniform interval declared by physical sample timestamps.

    Ordinary FFT-based STFT assumes uniform slow-time sampling.  This helper
    deliberately has no PRF or frame-index fallback: callers must explicitly
    resample a nonuniform sequence before using that algorithm.
    """

    raw_times = metadata.get("sample_times_s")
    if not isinstance(raw_times, Sequence) or isinstance(raw_times, (str, bytes)):
        raise ValueError(
            "Micro-Doppler requires schema-v4 physical sample timestamps in "
            "metadata['sample_times_s']; regenerate the sensing data"
        )
    if len(raw_times) != sample_count:
        raise ValueError(
            "Micro-Doppler CIR/timestamp count mismatch: "
            f"{sample_count} CIR rows and {len(raw_times)} physical timestamps"
        )
    if sample_count < 2:
        raise ValueError(
            "Micro-Doppler needs at least two physical sample timestamps to establish cadence"
        )

    try:
        times = np.asarray(raw_times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Micro-Doppler sample timestamps must be finite numbers") from exc
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("Micro-Doppler sample timestamps must be a finite one-dimensional list")
    intervals = np.diff(times)
    if not np.all(intervals > 0.0):
        raise ValueError("Micro-Doppler physical sample timestamps must be strictly increasing")

    interval_s = float(np.median(intervals))
    tolerance_s = max(1e-12, abs(interval_s) * 1e-9)
    if not np.allclose(intervals, interval_s, rtol=1e-6, atol=tolerance_s):
        raise ValueError(
            "Nonuniform physical sensing cadence is incompatible with ordinary STFT "
            "micro-Doppler; resample onto a uniform physical-time grid before analysis"
        )
    return interval_s


def load_gt_entries_for_tracking(
    frames_dir: Path,
) -> list[dict[str, Any]]:
    """Join canonical truth snapshots to delay rows by exact physical epoch.

    A single truth snapshot is never repeated across a multi-row acquisition.
    Every retained delay sample must have one and only one schema-v4 truth
    snapshot at the identical ``reference_time_s``.
    """

    sequence_samples: list[np.ndarray] = []
    sample_times_s: list[float] = []
    sequence_reference: DelayDomainSamples | None = None
    truth_by_time: dict[float, dict[str, Any]] = {}
    for frame in load_frames_sensing(Path(frames_dir)):
        typed_sensing = frame.get("_typed_sensing")
        if not isinstance(typed_sensing, Mapping):
            continue
        stored_gt = extract_stored_gt_entry(typed_sensing)
        if stored_gt is not None:
            temporal = typed_sensing.get("temporal")
            truth_temporal = temporal.get("truth") if isinstance(temporal, Mapping) else None
            raw_time = (
                truth_temporal.get("reference_time_s")
                if isinstance(truth_temporal, Mapping)
                else None
            )
            try:
                truth_time_s = float(raw_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sensing frame {frame.get('_frame_index')} truth lacks a finite "
                    "reference_time_s"
                ) from exc
            if not np.isfinite(truth_time_s):
                raise ValueError(
                    f"Sensing frame {frame.get('_frame_index')} truth lacks a finite "
                    "reference_time_s"
                )
            entry = {
                "reference_time_s": truth_time_s,
                "target_ids": deepcopy(stored_gt["target_ids"]),
                "target_ranges": deepcopy(stored_gt["target_ranges"]),
                "target_speeds": deepcopy(stored_gt["target_speeds"]),
            }
            previous = truth_by_time.get(truth_time_s)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"Multiple conflicting truth snapshots exist at {truth_time_s!r} s"
                )
            truth_by_time[truth_time_s] = entry

        product = _delay_samples_for_analysis(typed_sensing)
        if product is None:
            continue
        if sequence_reference is None:
            sequence_reference = product
        else:
            require_compatible_delay_sequence(sequence_reference, product)
        append_delay_domain_samples(
            product,
            sequence_samples,
            sample_times_s,
        )

    missing_times = [time_s for time_s in sample_times_s if time_s not in truth_by_time]
    if missing_times:
        preview = ", ".join(repr(value) for value in missing_times[:3])
        raise ValueError(
            "Truth-assisted range tracking has no exact truth snapshot for physical sample "
            f"time(s) {preview}; truth expansion is forbidden"
        )
    return [deepcopy(truth_by_time[time_s]) for time_s in sample_times_s]


def generate_micro_doppler_figures(
    frames_dir: Path,
    output_dir: Path,
    *,
    config: Any,
    target_range_m: float | None = None,
    target_id: str | None = None,
    velocity_overlay: dict[str, Any] | None = None,
    stft_window_override: int | None = None,
    source_sequence: tuple[list[np.ndarray], dict[str, Any]] | None = None,
) -> None:
    """Generate micro-Doppler spectrograms from stored CIR data.

    Reads per-frame CIR vectors from HDF5 chunk files in *frames_dir*,
    computes single-bin and range-collapsed micro-Doppler spectrograms,
    and saves figures to *output_dir/figures/*.
    """
    if source_sequence is None:
        logger.info("Loading CIR sequence from %s ...", frames_dir)
        fast_time_samples, src_config = load_cir_sequence(frames_dir)
    else:
        fast_time_samples, src_config = source_sequence
    if not fast_time_samples:
        logger.warning("No CIR data found; skipping micro-Doppler generation")
        return
    sensing_viz = _load_sensing_viz()

    logger.info("Loaded %d CIR samples for micro-Doppler", len(fast_time_samples))

    raw_range_axis = src_config.get("range_axis_m")
    try:
        range_axis = np.asarray(raw_range_axis, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Micro-Doppler generation requires exact typed range_axis_m coordinates"
        ) from exc
    first_shape = np.asarray(fast_time_samples[0]).shape
    if len(first_shape) != 1:
        raise ValueError("Micro-Doppler delay samples must be one-dimensional")
    range_bin_count = int(first_shape[0])
    if (
        range_axis.ndim != 1
        or range_axis.size != range_bin_count
        or not np.all(np.isfinite(range_axis))
        or (range_axis.size > 1 and not np.all(np.diff(range_axis) > 0.0))
    ):
        raise ValueError(
            "Micro-Doppler exact range_axis_m must be finite, strictly increasing, and "
            "aligned one-to-one with delay bins"
        )
    if any(np.asarray(sample).shape != (range_bin_count,) for sample in fast_time_samples):
        raise ValueError("Micro-Doppler delay rows do not share the exact range_axis_m shape")
    try:
        fft_size_doppler = int(src_config["fft_size_doppler"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Micro-Doppler generation requires exact config_resolved.fft_size_doppler"
        ) from exc
    if fft_size_doppler <= 0:
        raise ValueError("config_resolved.fft_size_doppler must be positive")

    # Ordinary STFT requires a uniform physical sampling interval.  Use only
    # persisted schema-v4 timestamps, never configured PRF or storage IDs.
    frame_dt_s = _uniform_sample_interval_s(src_config, len(fast_time_samples))
    prf_hz = 1.0 / frame_dt_s
    configured_prf = src_config.get("prf_hz")
    if configured_prf is None:
        configured_prf_hz = None
    else:
        try:
            configured_prf_hz = float(configured_prf)
        except (TypeError, ValueError):
            configured_prf_hz = None
    if configured_prf_hz is not None and not np.isclose(
        configured_prf_hz,
        prf_hz,
        rtol=1e-6,
        atol=max(1e-12, prf_hz * 1e-9),
    ):
        logger.info(
            "Using physical sensing cadence %.9g Hz for micro-Doppler instead of "
            "configured PRF %.9g Hz",
            prf_hz,
            configured_prf_hz,
        )

    acquisition_counts = src_config.get("acquisition_sample_counts")
    cpi_duration_s: float | None = None
    if isinstance(acquisition_counts, Sequence) and not isinstance(
        acquisition_counts, (str, bytes)
    ):
        try:
            positive_counts = [int(value) for value in acquisition_counts]
        except (TypeError, ValueError) as exc:
            raise ValueError("acquisition_sample_counts must contain positive integers") from exc
        if any(value <= 0 for value in positive_counts):
            raise ValueError("acquisition_sample_counts must contain positive integers")
        if positive_counts and len(set(positive_counts)) == 1 and positive_counts[0] > 1:
            cpi_duration_s = float(positive_counts[0]) * frame_dt_s

    # STFT parameters
    stft_win = stft_window_override if stft_window_override else max(fft_size_doppler, 32)
    # Coherent CPI: each CPI is independent (new RT call at boundary).
    # No overlap — hop equals window to avoid mixing cross-CPI data.
    if cpi_duration_s is not None:
        stft_hop = stft_win
    else:
        stft_hop = max(stft_win // 4, 1)

    # Select range bin
    if target_range_m is not None:
        range_bin = int(np.argmin(np.abs(range_axis - target_range_m)))
        logger.info("Using user-specified range bin %d (%.2f m)", range_bin, target_range_m)
    else:
        # Blind selection is observation-derived only: strongest mean power on
        # the exact producer-authored range axis, with no truth or center bin.
        fast_arr = np.abs(np.stack(fast_time_samples, axis=0))
        mean_power = np.nanmean(fast_arr, axis=0)
        if not np.any(np.isfinite(mean_power)):
            raise ValueError("Blind micro-Doppler selection has no finite observed power")
        range_bin = int(np.nanargmax(mean_power))
        actual_range = float(range_axis[range_bin])
        logger.info(
            "Blind-selected range bin %d (%.2f m, strongest observed mean power)",
            range_bin,
            actual_range,
        )

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Single-bin micro-Doppler
    md_result = compute_micro_doppler(
        range_bin,
        fast_time_sequence=fast_time_samples,
        prf_hz=prf_hz,
        frame_duration_s=frame_dt_s,
        stft_window_frames=stft_win,
        stft_hop_frames=stft_hop,
        range_axis_m=range_axis,
        mti_enabled=config.mti_enabled,
    )
    if md_result is not None:
        p = sensing_viz.create_micro_doppler_figure(
            md_result,
            output_path=fig_dir / "micro_doppler.png",
            velocity_overlay=velocity_overlay,
        )
        generated.append(Path(p))

        p = sensing_viz.create_micro_doppler_figure(
            md_result,
            output_path=fig_dir / "micro_doppler_interp.png",
            interpolation="bilinear",
            velocity_overlay=velocity_overlay,
        )
        generated.append(Path(p))

        if cpi_duration_s is not None:
            p = sensing_viz.create_micro_doppler_figure(
                md_result,
                output_path=fig_dir / "micro_doppler_cpi_markers.png",
                cpi_duration_s=cpi_duration_s,
                velocity_overlay=velocity_overlay,
            )
            generated.append(Path(p))
    else:
        logger.warning("Single-bin micro-Doppler computation returned None")

    # 2. Range-collapsed micro-Doppler
    md_collapsed = compute_range_collapsed_micro_doppler(
        fast_time_sequence=fast_time_samples,
        prf_hz=prf_hz,
        frame_duration_s=frame_dt_s,
        stft_window_frames=stft_win,
        stft_hop_frames=stft_hop,
        range_axis_m=range_axis,
        mti_enabled=config.mti_enabled,
    )
    if md_collapsed is not None:
        p = sensing_viz.create_micro_doppler_figure(
            md_collapsed,
            output_path=fig_dir / "micro_doppler_collapsed.png",
            velocity_overlay=velocity_overlay,
        )
        generated.append(Path(p))

        p = sensing_viz.create_micro_doppler_figure(
            md_collapsed,
            output_path=fig_dir / "micro_doppler_collapsed_interp.png",
            interpolation="bilinear",
            velocity_overlay=velocity_overlay,
        )
        generated.append(Path(p))

        if cpi_duration_s is not None:
            p = sensing_viz.create_micro_doppler_figure(
                md_collapsed,
                output_path=fig_dir / "micro_doppler_collapsed_cpi_markers.png",
                interpolation="bilinear",
                cpi_duration_s=cpi_duration_s,
                velocity_overlay=velocity_overlay,
            )
            generated.append(Path(p))
    else:
        logger.warning("Range-collapsed micro-Doppler computation returned None")

    # 3. Range-tracked micro-Doppler (follows target across range bins using GT)
    if not isinstance(target_id, str) or not target_id:
        logger.info("Range-tracked micro-Doppler disabled: an explicit target_id was not supplied")
    else:
        try:
            gt_entries = load_gt_entries_for_tracking(frames_dir)
            md_tracked = compute_range_tracked_micro_doppler(
                gt_entries,
                target_id=target_id,
                fast_time_sequence=fast_time_samples,
                sample_times_s=src_config["sample_times_s"],
                range_axis_m=range_axis,
                stft_window_frames=stft_win,
                stft_hop_frames=stft_hop,
                mti_enabled=config.mti_enabled,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Range-tracked micro-Doppler disabled: %s", exc)
        else:
            if md_tracked is not None:
                p = sensing_viz.create_micro_doppler_figure(
                    md_tracked,
                    output_path=fig_dir / "micro_doppler_tracked.png",
                    velocity_overlay=velocity_overlay,
                )
                generated.append(Path(p))

                p = sensing_viz.create_micro_doppler_figure(
                    md_tracked,
                    output_path=fig_dir / "micro_doppler_tracked_interp.png",
                    interpolation="bilinear",
                    velocity_overlay=velocity_overlay,
                )
                generated.append(Path(p))

                if cpi_duration_s is not None:
                    p = sensing_viz.create_micro_doppler_figure(
                        md_tracked,
                        output_path=fig_dir / "micro_doppler_tracked_cpi_markers.png",
                        cpi_duration_s=cpi_duration_s,
                        velocity_overlay=velocity_overlay,
                    )
                    generated.append(Path(p))
            else:
                logger.warning("Range-tracked micro-Doppler computation returned None")

    if generated:
        logger.info(
            "Micro-Doppler figures saved to %s: %s",
            fig_dir,
            ", ".join(p.name for p in generated),
        )
    else:
        logger.warning("No micro-Doppler figures were generated")


def generate_noise_sweep_figures(
    output_dir: Path,
    sweep_results: Sequence[Mapping[str, Any]],
    *,
    rd_examples: Sequence[Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Generate seed-level noise robustness summaries; see sensing_noise_figures."""
    from shared.sensing.plotting.noise import generate_noise_sweep_figures as generate

    return generate(output_dir, sweep_results, rd_examples=rd_examples)
