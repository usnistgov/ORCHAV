#!/usr/bin/env python3
"""Dispatch sensing summary figures requested by scenario configuration.

This module stays separate from the normal generator summary dispatcher because
it imports sensing-only visualization and signal-processing helpers. It is
loaded lazily by ``generator.figures.summary`` when a sensing-enabled run asks
for post-run diagnostics.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shared.logging import get_logger
from shared.sensing.angle import select_range_doppler_angle_product
from shared.sensing.plotting.diagram import (
    build_sensing_diagram_model,
    create_sensing_acquisition_processing_figure,
    write_sensing_diagram_manifest,
)
from shared.sensing.plotting.style import sensing_figure_style
from shared.sensing.plotting.viz import (
    create_azimuth_elevation_figure,
    create_detection_summary_figure,
    create_doppler_azimuth_figure,
    create_gt_evolution_figure,
    create_micro_doppler_figure,
    create_range_azimuth_figure,
    create_range_doppler_figure,
    create_range_elevation_figure,
    create_range_profile_figure,
    create_tracking_figure,
)

logger = get_logger(__name__)


def _uniform_physical_sample_interval_s(
    sample_times_s: Sequence[float] | None,
    *,
    sample_count: int,
    label: str,
) -> float:
    """Validate and return a uniform schema-v4 physical sample interval."""

    if sample_times_s is None:
        raise ValueError(
            f"{label} requires schema-v4 physical sample timestamps; "
            "scenario frame spacing and configured PRF are not valid substitutes"
        )
    if len(sample_times_s) != sample_count:
        raise ValueError(
            f"{label} has {sample_count} samples but {len(sample_times_s)} physical timestamps"
        )
    if sample_count < 2:
        raise ValueError(f"{label} needs at least two physical timestamps to establish cadence")
    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError(f"{label} timestamps must be a finite one-dimensional sequence")
    intervals = np.diff(times)
    if not np.all(intervals > 0.0):
        raise ValueError(f"{label} timestamps must be strictly increasing")
    interval_s = float(np.median(intervals))
    if not np.allclose(
        intervals,
        interval_s,
        rtol=1e-6,
        atol=max(1e-12, abs(interval_s) * 1e-9),
    ):
        raise ValueError(
            f"{label} has nonuniform physical cadence; resample onto a uniform "
            "physical-time grid before ordinary FFT/STFT analysis"
        )
    return interval_s


@sensing_figure_style
def maybe_generate_sensing_summary(
    last_sensing_data: dict[str, Any],
    gt_history: list[dict[str, Any]],
    gt_state: dict[str, Any] | None,
    scenario_context: Any,
    fast_time_samples: list | None = None,
    analytical_paths: list | None = None,
    association_history: list[dict[str, Any]] | None = None,
    frame_dt_s: float = 1e-3,
    *,
    target_id: str | None = None,
    fast_time_sample_times_s: list[float] | None = None,
    analytical_path_sample_times_s: list[float] | None = None,
    sensing_event_history: Sequence[Mapping[str, Any]] | None = None,
    doppler_history: Sequence[Mapping[str, Any]] | None = None,
    cartesian_history: Sequence[Mapping[str, Any]] | None = None,
    tracking_history: dict[int, list[dict[str, Any]]] | None = None,
    signal_diagnostics: bool = False,
    range_limits_m: tuple[float, float] | None = None,
    velocity_limits_m_s: tuple[float, float] | None = None,
    power_limits_db: tuple[float, float] | None = None,
    result_label: str = "last frame",
    output_root: Path | None = None,
    strict: bool = False,
) -> list[Path] | None:
    """Generate sensing figures when the ``sensing`` summary is selected.

    The offline pipeline calls this after committing frames and directs output
    to its private summary staging tree. Direct callers may instead supply the
    configured summary directory.

    Args:
        last_sensing_data: Sensing dict from the last computed frame.
        result_label: Caption for single-result maps; saved-frame selections can
            override the default last-frame description.
        gt_history: Per-frame GT accumulation (target_ranges, target_speeds,
            det_ranges, det_velocities).
        gt_state: Last-frame GT state dict for the RD overlay (target_ranges,
            target_speeds), or ``None``.
        scenario_context: ScenarioConfiguration with ``generator_summary``.
        fast_time_samples: Per-frame CIR/range-profile samples for
            micro-Doppler computation (list of 1-D numpy arrays).
        fast_time_sample_times_s: Schema-v4 physical timestamp aligned with
            each fast-time sample.
        analytical_paths: Retained call compatibility; unused by the summary.
        analytical_path_sample_times_s: Retained call compatibility; unused by the summary.
        sensing_event_history: Bounded metadata-only acquisition snapshots used
            to show actual first/next CPI or sliding-window membership.
        association_history: Per-frame detection association metrics
            (tp, fp, fn, range_rmse, velocity_rmse).
        frame_dt_s: Legacy scenario frame spacing retained for call
            compatibility. Sensing analysis never uses it as sample cadence.
        target_id: Retained call compatibility; ordinary summaries do not select targets.
        doppler_history: Compact power reductions of the saved primary RD products.
        cartesian_history: Recorded receiver-local position events, independent of truth.
        range_limits_m: Optional range bounds for zoomed views; full views stay complete.
        velocity_limits_m_s: Optional velocity bounds for the zoomed RD map.
        power_limits_db: Optional common RD color limits, including the full map.
        tracking_history: Recorded track estimates keyed by their saved identity.
        signal_diagnostics: Explicitly request one range-collapsed STFT view, without
            additional clutter removal. This is a processing experiment, not the saved
            sensing recipe, and is disabled in ordinary generator summaries.
        output_root: Summary tree root. The figures are written below its
            ``sensing`` child.
        strict: Raise after attempting the requested figures if any one fails.

    Returns:
        List of generated file paths, or ``None`` when summary is disabled.
    """
    gs = getattr(scenario_context, "generator_summary", None)
    if not gs or not gs.get("enabled", False):
        return None
    if "sensing" not in (gs.get("create", []) or []):
        logger.debug("Sensing summary skipped because it was not selected")
        return None

    options = gs.get("sensing") or {}
    selected = set(
        options.get(
            "figures",
            [
                "range_profile",
                "range_doppler",
                "doppler",
                "truth",
                "detection",
                "angles",
            ],
        )
    )
    rd_views = options.get("range_doppler_views", ["zoom_to_data", "full"])
    rp_views = options.get("range_profile_views", ["zoom_to_data", "full"])
    use_physical_time = options.get("history_time_axis", "scene_frame") == "time"
    out_spec = gs.get("output", {}) or {}
    out_dir = output_root if output_root is not None else out_spec.get("dir", None)
    out_fmt = (out_spec.get("format", "png") or "png").lower()

    if not out_dir:
        logger.debug("Sensing summary skipped: no output dir configured")
        return None

    sensing_dir = Path(out_dir) / "sensing"
    sensing_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    failures: list[tuple[str, Exception]] = []

    def _figure_failed(label: str, exc: Exception) -> None:
        logger.warning("%s failed: %s", label, exc)
        if strict:
            failures.append((label, exc))

    def _figure_skipped(label: str, reason: str) -> None:
        logger.info("%s skipped: %s", label, reason)

    inspections = list(
        dict.fromkeys(
            (gs.get("inspection_figures") or []) + (options.get("inspection_figures") or [])
        )
    )
    for name in inspections:
        try:
            from shared.sensing.plotting.inspection import (
                detection_stages_figure,
                tracking_decisions_figure,
            )
            from shared.sensing.stage_inspection import load_detection_stage_bundle
            from shared.sensing.tracking_inspection import tracking_decisions

            if name == "detection_stages":
                figure = detection_stages_figure(load_detection_stage_bundle(last_sensing_data))
            elif name == "tracking_decisions":
                figure = tracking_decisions_figure(*tracking_decisions(last_sensing_data))
            else:
                raise ValueError(f"Unknown inspection figure: {name}")
            path = sensing_dir / f"{name}.{out_fmt}"
            figure.savefig(path, dpi=150)
            figure.clear()
            generated.append(path)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            _figure_failed(name, exc)

    # Continue after an individual failure so direct best-effort callers retain
    # successful diagnostics. Strict pipeline callers reject the staged tree.
    if "acquisition" in selected:
        try:
            model = build_sensing_diagram_model(
                last_sensing_data,
                sensing_event_history=sensing_event_history,
            )
            p = create_sensing_acquisition_processing_figure(
                last_sensing_data,
                sensing_event_history=sensing_event_history,
                output_path=sensing_dir / f"acquisition_processing.{out_fmt}",
            )
            generated.append(Path(p))
            manifest_path = write_sensing_diagram_manifest(
                model,
                sensing_dir / "acquisition_processing.json",
            )
            generated.append(manifest_path)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
            _figure_failed("Sensing acquisition/processing diagram", e)

    if "range_profile" in selected:
        for view in rp_views:
            try:
                name = "range_profile_full" if view == "full" else "range_profile"
                p = create_range_profile_figure(
                    last_sensing_data,
                    output_path=sensing_dir / f"{name}.{out_fmt}",
                    zoom_to_data=view == "zoom_to_data",
                    range_xlim=range_limits_m if view == "zoom_to_data" else None,
                )
                generated.append(Path(p))
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
                _figure_failed("Sensing range-profile figure", e)

    # 2. Range–Doppler map
    if "range_doppler" in selected and "zoom_to_data" in rd_views:
        try:
            p = create_range_doppler_figure(
                last_sensing_data,
                output_path=sensing_dir / f"range_doppler.{out_fmt}",
                zoom_to_data=True,
                range_xlim=range_limits_m,
                velocity_ylim=velocity_limits_m_s,
                power_limits_db=power_limits_db,
                result_label=result_label,
            )
            generated.append(Path(p))
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
            _figure_failed("Sensing range-Doppler figure", e)

    # 2b. The same saved array over its full range/velocity grid.
    if "range_doppler" in selected and "full" in rd_views:
        try:
            p = create_range_doppler_figure(
                last_sensing_data,
                output_path=sensing_dir / f"range_doppler_full.{out_fmt}",
                zoom_to_data=False,
                power_limits_db=power_limits_db,
                result_label=result_label,
            )
            generated.append(Path(p))
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
            _figure_failed("Sensing full range-Doppler figure", e)

    # Angular power projections use the saved product, independently of truth.
    if "angles" in selected:
        try:
            angular_product = select_range_doppler_angle_product(last_sensing_data)
            if angular_product is not None:
                axes = {
                    "range": angular_product.range_axis_m,
                    "doppler": angular_product.doppler_axis_hz,
                    "azimuth": angular_product.azimuth_broadside_angle_deg,
                    "elevation": angular_product.elevation_broadside_angle_deg,
                }
                angular_views = [
                    ("range_azimuth", create_range_azimuth_figure, ("range", "azimuth")),
                    ("doppler_azimuth", create_doppler_azimuth_figure, ("doppler", "azimuth")),
                    (
                        "azimuth_elevation",
                        create_azimuth_elevation_figure,
                        ("azimuth", "elevation"),
                    ),
                    ("range_elevation", create_range_elevation_figure, ("range", "elevation")),
                ]
                for name, renderer, coordinates in angular_views:
                    if any(np.count_nonzero(np.isfinite(axes[key])) < 2 for key in coordinates):
                        _figure_skipped(name, "fewer than two physical bins on a display axis")
                        continue
                    overlay = (
                        {"gt_state": gt_state}
                        if name in {"range_azimuth", "doppler_azimuth"}
                        else {}
                    )
                    generated.append(
                        Path(
                            renderer(
                                last_sensing_data,
                                output_path=sensing_dir / f"{name}.{out_fmt}",
                                result_label=result_label,
                                **overlay,
                            )
                        )
                    )
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
            _figure_failed("Sensing angular-power figures", exc)

    # 3. GT evolution
    if gt_history and ({"truth", "detection_history", "angles"} & selected):
        for kind, filename in (
            ("truth", "gt_evolution"),
            ("detection_history", "detection_history"),
        ):
            if kind not in selected:
                continue
            try:
                p = create_gt_evolution_figure(
                    gt_history,
                    output_path=sensing_dir / f"{filename}.{out_fmt}",
                    use_physical_time=use_physical_time,
                )
                generated.append(Path(p))
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
                _figure_failed("Sensing GT-evolution figure", e)
        angular_events = [
            entry["angular_summary"] for entry in gt_history if entry.get("angular_summary")
        ]
        if angular_events and "angles" in selected:
            try:
                from shared.sensing.angular import create_angular_summary

                generated.extend(create_angular_summary(angular_events, sensing_dir))
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
                _figure_failed("Sensing angular-GT figures", e)
    else:
        _figure_skipped("Sensing GT-evolution figure", "no ground-truth history was recorded")

    if "tracking" in selected:
        if tracking_history:
            try:
                p = create_tracking_figure(
                    gt_history,
                    tracking_history,
                    sensing_dir / f"tracking.{out_fmt}",
                    use_physical_time=use_physical_time,
                )
                generated.append(p)
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
                _figure_failed("Saved tracking history", exc)
        else:
            _figure_skipped("Saved tracking history", "no recorded track identities")

    cartesian_selected = selected & {"tracking_xy", "tracking_xyz", "tracking_cartesian"}
    if cartesian_selected:
        cartesian_events = (
            list(cartesian_history)
            if cartesian_history is not None
            else [
                entry["cartesian_summary"]
                for entry in gt_history
                if entry.get("cartesian_summary") is not None
            ]
        )
        if any(event["tracks"] for event in cartesian_events):
            try:
                from shared.sensing.plotting.cartesian import create_cartesian_tracking_figures

                generated.extend(
                    create_cartesian_tracking_figures(
                        cartesian_events,
                        sensing_dir,
                        figures=cartesian_selected,
                        output_format=out_fmt,
                    )
                )
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
                _figure_failed("Cartesian tracking figures", exc)
        else:
            _figure_skipped("Cartesian tracking figures", "no recorded Cartesian observations")

    # Reuse the primary product: never add another mean-removal or FFT stage.
    if "doppler" in selected:
        try:
            from shared.sensing.plotting.doppler_summary import (
                doppler_summary_event,
                plot_doppler_history,
            )

            history = list(doppler_history or ())
            if not history:
                event = doppler_summary_event(last_sensing_data)
                if event is not None:
                    history.append(event)
            if history:
                p = plot_doppler_history(history, sensing_dir / f"doppler_history.{out_fmt}")
                generated.append(p)
            else:
                _figure_skipped("Saved Doppler history", "no primary range–Doppler product")
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
            _figure_failed("Saved Doppler history", exc)

    # One optional textbook STFT, for callers deliberately requesting a separate
    # analysis. No truth-following, interpolation variants or extra MTI. The
    # normal time–Doppler view above uses saved primary CPI results instead.
    if signal_diagnostics and fast_time_samples:
        try:
            from shared.sensing.delay import extract_delay_domain_samples
            from shared.sensing.signal_processing import compute_range_collapsed_micro_doppler

            cfg = last_sensing_data["config_resolved"]
            delay = extract_delay_domain_samples(last_sensing_data)
            if delay is None:
                raise ValueError("STFT diagnostic requires a typed delay-domain product")
            dt = _uniform_physical_sample_interval_s(
                fast_time_sample_times_s,
                sample_count=len(fast_time_samples),
                label="Micro-Doppler diagnostic",
            )
            window_length = int(cfg["fft_size_doppler"])
            result = compute_range_collapsed_micro_doppler(
                fast_time_sequence=fast_time_samples,
                prf_hz=1.0 / dt,
                frame_duration_s=dt,
                stft_window_frames=window_length,
                stft_hop_frames=max(1, window_length // 4),
                window=cfg["processing_recipe"]["window_doppler"],
                range_axis_m=delay.range_axis_m,
                range_mode=cfg["range_mode"],
                mti_enabled=False,
            )
            if result is not None:
                result.time_axis_s += float(fast_time_sample_times_s[0])
                p = create_micro_doppler_figure(
                    result,
                    output_path=sensing_dir / f"micro_doppler.{out_fmt}",
                    plot_title="Micro-Doppler · separate STFT, no clutter suppression",
                )
                generated.append(Path(p))
            else:
                _figure_skipped("Micro-Doppler", "too few samples for one STFT window")
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
            _figure_failed("Micro-Doppler diagnostic", exc)

    if association_history and "detection" in selected:
        try:
            p = create_detection_summary_figure(
                association_history,
                output_path=sensing_dir / f"detection_summary.{out_fmt}",
            )
            generated.append(Path(p))
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as e:
            _figure_failed("Sensing detection-summary figure", e)

        # Persist aggregate detection metrics alongside images so benchmarks can
        # compare runs without parsing plotted pixels.
        try:
            import json as _json

            versioned_reference = any(
                "evaluation_reference_protocol" in h for h in association_history
            )
            evaluated = (
                [h for h in association_history if h.get("evaluated", True)]
                if versioned_reference
                else association_history
            )
            _tp = sum(h["tp"] for h in evaluated)
            _fp = sum(h["fp"] for h in evaluated)
            _fn = sum(h["fn"] for h in evaluated)
            _n = len(evaluated)
            _r_rmse_vals = [
                h["range_rmse"] for h in evaluated if h["tp"] > 0 and h["range_rmse"] is not None
            ]
            _v_rmse_vals = [
                h["velocity_rmse"]
                for h in evaluated
                if h["tp"] > 0 and h["velocity_rmse"] is not None
            ]
            undefined = None if versioned_reference else 0.0
            metrics = {
                "frames": _n,
                "total_tp": int(_tp),
                "total_fp": int(_fp),
                "total_fn": int(_fn),
                "tp_rate": round(float(_tp / (_tp + _fn)), 4) if (_tp + _fn) else undefined,
                "fp_per_frame": round(float(_fp / _n), 3) if _n else undefined,
                "mean_range_rmse_m": (
                    round(float(np.mean(_r_rmse_vals)), 4) if _r_rmse_vals else undefined
                ),
                "mean_velocity_rmse_m_s": (
                    round(float(np.mean(_v_rmse_vals)), 4) if _v_rmse_vals else undefined
                ),
            }
            if versioned_reference:
                metrics.update(
                    evaluation_reference_protocols=sorted(
                        {
                            h["evaluation_reference_protocol"]
                            for h in association_history
                            if "evaluation_reference_protocol" in h
                        }
                    ),
                    evaluated_frames=_n,
                    unavailable_reference_frames=len(association_history) - _n,
                    velocity_matched_pair_count=sum(
                        h.get("velocity_matched_pair_count", 0) for h in evaluated
                    ),
                    velocity_excluded_matched_pair_count=sum(
                        h.get("velocity_excluded_matched_pair_count", 0) for h in evaluated
                    ),
                    velocity_unavailable_opportunity_count=sum(
                        h.get("velocity_unavailable_opportunity_count", 0) for h in evaluated
                    ),
                    range_error_frame_count=len(_r_rmse_vals),
                    velocity_error_frame_count=len(_v_rmse_vals),
                )
            metrics_path = sensing_dir / "detection_metrics.json"
            metrics_path.write_text(_json.dumps(metrics, indent=2) + "\n")
            logger.info("Detection metrics saved: %s", metrics_path)
        except (ValueError, TypeError, ZeroDivisionError, OSError) as e:
            _figure_failed("Detection metrics JSON", e)

    logger.info("Sensing summary: generated %d files in %s", len(generated), sensing_dir)
    if failures:
        labels = ", ".join(label for label, _exc in failures)
        raise RuntimeError(f"Sensing summary failed for: {labels}") from failures[0][1]
    return generated
