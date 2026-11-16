"""Compare sensing recipes on one immutable saved acquisition.

This small Python/CLI workflow reuses the postprocessor's typed input, temporal
lineage and writer boundaries. It never invokes ray tracing. Researchers can
provide a processor factory while retaining the same save/evaluate interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from shared.sensing.cfr_replay import processing_config_dict
from shared.sensing.evaluation import compute_mot_metrics, evaluate_scenario
from shared.sensing.loader import _open_provider, load_frames_sensing, load_saved_results
from shared.sensing.plotting.figures import generate_figures
from shared.sensing.postprocessor import (
    PostProcessingConfig,
    SensingPostProcessor,
    _build_effective_processing_recipe,
    _build_reprocessed_temporal,
    _join_evaluation_events,
    _select_typed_detection_input,
    _source_display_binding,
    _source_scene_frame_step,
    _source_sensing_payload,
    _temporal_stream_identity,
    _typed_range_doppler_arrays,
    validate_typed_processing_config,
)
from shared.sensing.writer import ColumnarResultWriter, _json_default

logger = logging.getLogger(__name__)


def _fingerprints(path: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix in {".json", ".h5"}
    }


def compare_recipes(
    source: Path,
    recipes: Mapping[str, PostProcessingConfig],
    output: Path,
    *,
    processor_factory: Callable[
        [PostProcessingConfig], SensingPostProcessor
    ] = SensingPostProcessor,
    range_gate_m: float = 0.6,
    velocity_gate_m_s: float = 0.25,
    figures: bool = True,
    inspection_figures: bool = False,
    measure_costs: bool = False,
) -> dict:
    """Save, score and compare recipes, loading the common source only once.

    Inputs must have complete stored RD results from one clock and sensor pair.
    Processing receives exact saved bin coordinates and result times. Optional
    custom fields in a PostProcessingConfig subclass participate in recipe IDs.
    The reference processor does not use truth to form detections or tracks;
    truth retained in the source metadata is used for evaluation and plots.

    Timing is warm processing wall time, excluding file IO, scoring and plots.
    It is illustrative single-run timing, not a calibrated performance benchmark.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (range_gate_m, velocity_gate_m_s)):
        raise ValueError("Evaluation gates must be finite and positive")
    if not recipes or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) for name in recipes
    ):
        raise ValueError("Recipes require distinct simple directory names")
    if any(cfg.ghost_track_correlation_threshold is not None for cfg in recipes.values()):
        raise ValueError("Batch ghost-track pruning is not supported by this compact workflow")
    for config in recipes.values():
        validate_typed_processing_config(config)
    benchmark_start = perf_counter()
    costs: dict[str, float] = {}

    def add_cost(name: str, start: float) -> None:
        if measure_costs:
            costs[name] = costs.get(name, 0.0) + perf_counter() - start

    frames_dir, manifest, provider = _open_provider(Path(source))
    add_cost("source_open", benchmark_start)
    try:
        output = Path(output).resolve()
        if output.is_relative_to(frames_dir.resolve()) or frames_dir.resolve().is_relative_to(
            output
        ):
            raise ValueError("Source frames and comparison output must not contain each other")
        cost_start = perf_counter()
        fingerprints = _fingerprints(frames_dir)
        add_cost("source_hashing", cost_start)
        cost_start = perf_counter()
        frames = load_frames_sensing(frames_dir)
        add_cost("input_loading", cost_start)
        selected = []
        for frame in frames:
            product = _select_typed_detection_input(frame, rebuild_rd=False)
            if product is not None:
                if product.kind != "range_doppler" or not frame["_typed_sensing"]["temporal"].get(
                    "result"
                ):
                    raise ValueError("Comparison requires complete saved RD results")
                selected.append((frame, product))
        if not selected:
            raise ValueError("No complete stored RD results")
        if len({_temporal_stream_identity(frame) for frame, _ in selected}) != 1:
            raise ValueError("Do not share tracker state across clocks or sensor pairs")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        summary = {
            "source_frame_set_id": manifest.frame_set_id,
            "source_generation_id": manifest.generation_id,
            "result_count": len(selected),
            "evaluation_gates": {"range_m": range_gate_m, "velocity_m_s": velocity_gate_m_s},
            "timing_scope": "one warm processing pass; excludes IO, scoring, plotting; not a benchmark",
            "recipes": {},
        }
        saved = {}
        for name, cfg in recipes.items():
            destination = output / name
            destination.mkdir()
            (destination / "recipe.json").write_text(
                json.dumps(processing_config_dict(cfg), indent=2) + "\n"
            )
            # Warm dependency imports and one detector call, then discard state.
            first, first_product = selected[0]
            cost_start = perf_counter()
            warm = processor_factory(cfg)
            cfr_rebuild = cfg.range_response is not None or cfg.range_ifft_size is not None
            if cfr_rebuild:
                warm.process_from_frequency_observation(
                    _source_sensing_payload(first),
                    update_time_s=float(
                        first["_typed_sensing"]["temporal"]["result"]["reference_time_s"]
                    ),
                )
            else:
                power, cube = _typed_range_doppler_arrays(first_product)
                warm.process_from_rd_map(
                    power,
                    _source_sensing_payload(first),
                    range_axis_m=first_product.range_axis_m,
                    doppler_axis_hz=first_product.doppler_axis_hz,
                    rd_cube=cube,
                    update_time_s=float(
                        first["_typed_sensing"]["temporal"]["result"]["reference_time_s"]
                    ),
                )
            add_cost("warmup", cost_start)
            processor = processor_factory(cfg)
            run_id = str(uuid4())
            writer = ColumnarResultWriter(destination)
            results = []
            elapsed = 0.0
            try:
                for frame, product in selected:
                    meta = _source_sensing_payload(frame)
                    if cfr_rebuild:
                        product = _select_typed_detection_input(
                            frame, rebuild_rd=False, rebuild_cfr=True
                        )

                    temporal = _build_reprocessed_temporal(
                        frame,
                        processing_run_id=run_id,
                        postprocessing_config=cfg,
                        input_kind=product.data_key,
                    )
                    start = perf_counter()
                    if cfr_rebuild:
                        result = processor.process_from_frequency_observation(
                            meta, update_time_s=float(temporal["result"]["reference_time_s"])
                        )
                    else:
                        power, cube = _typed_range_doppler_arrays(product)
                        result = processor.process_from_rd_map(
                            power,
                            meta,
                            range_axis_m=product.range_axis_m,
                            doppler_axis_hz=product.doppler_axis_hz,
                            rd_cube=cube,
                            update_time_s=float(temporal["result"]["reference_time_s"]),
                        )
                    duration = perf_counter() - start
                    elapsed += duration
                    if measure_costs:
                        costs["processing"] = costs.get("processing", 0.0) + duration
                    cost_start = perf_counter()
                    result.update(
                        temporal=temporal,
                        processing_recipe_resolved=_build_effective_processing_recipe(
                            frame, postprocessing_config=cfg, input_kind=product.data_key
                        ),
                        _frame_index=frame["_frame_index"],
                        _input_kind=product.data_key,
                    )
                    writer.add_frame(
                        result,
                        frame_index=frame["_frame_index"],
                        source_frame=provider.load_frame(_source_scene_frame_step(frame)),
                        source_sensing=meta,
                        postprocessing_config=cfg,
                        input_kind=product.data_key,
                        source_display_binding=_source_display_binding(frame),
                    )
                    results.append(result)
                    add_cost("serialization", cost_start)
                cost_start = perf_counter()
                writer.finalize()
                from shared.sensing.result_scenario import write_result_scenario

                write_result_scenario(source, destination)
                add_cost("serialization", cost_start)
            except BaseException:
                writer.abort()
                raise
            cost_start = perf_counter()
            joined = _join_evaluation_events(frames, results)
            truth = [(e.gt_ranges, e.gt_velocities) for e in joined.events]
            detections = [
                (
                    np.array([d["range_m"] for d in e.detections]),
                    np.array([d["velocity_m_s"] for d in e.detections]),
                )
                for e in joined.events
            ]
            metrics = evaluate_scenario(
                truth,
                detections,
                range_gate=range_gate_m,
                velocity_gate=velocity_gate_m_s,
                per_event_velocity_valid=(
                    [
                        (
                            event.gt_velocity_valid
                            if event.gt_velocity_valid is not None
                            else [True] * len(event.gt_ranges)
                        )
                        for event in joined.events
                    ]
                    if any(event.gt_velocity_valid is not None for event in joined.events)
                    else None
                ),
            )
            record = {"processing_seconds": elapsed, "metrics": asdict(metrics), "output": "frames"}
            if any(d.get("track_id") is not None for r in results for d in r["detections"]):
                velocity_complete = all(
                    e.gt_velocity_valid is None or all(e.gt_velocity_valid) for e in joined.events
                )
                if velocity_complete:
                    record["tracking"] = asdict(
                        compute_mot_metrics(
                            [(e.target_ids, e.gt_ranges, e.gt_velocities) for e in joined.events],
                            [list(e.detections) for e in joined.events],
                            range_gate=range_gate_m,
                            velocity_gate=velocity_gate_m_s,
                        )
                    )
                else:
                    record["tracking_unavailable_reason"] = (
                        "MOT metrics require valid velocity truth for every evaluated target; "
                        "saved tracks and range-only detection metrics remain available."
                    )
            summary["recipes"][name] = record
            (destination / "evaluation.json").write_text(
                json.dumps(record, indent=2, default=_json_default) + "\n"
            )
            add_cost("evaluation", cost_start)
            cost_start = perf_counter()
            saved[name] = load_saved_results(destination)
            add_cost("result_readback", cost_start)
            cost_start = perf_counter()
            if figures:
                generate_figures(
                    destination,
                    len(results),
                    frames=frames,
                    all_results=saved[name],
                    eval_result=metrics,
                    figure_names={
                        "tracking",
                        "tracking_xy",
                        "angular_tracking",
                        "angular_errors",
                        "detection_events",
                    }
                    | ({"detection_stages", "tracking_decisions"} if inspection_figures else set()),
                    angular_evaluation_gates=(range_gate_m, velocity_gate_m_s),
                )
            add_cost("figures", cost_start)
            logger.info(
                "%s: TP=%d FP=%d FN=%d", name, metrics.total_tp, metrics.total_fp, metrics.total_fn
            )
        cost_start = perf_counter()
        if fingerprints != _fingerprints(frames_dir):
            raise RuntimeError("Source acquisition changed during comparison")
        add_cost("source_hashing", cost_start)
        summary["source_unchanged"] = True
        (output / "source_hashes.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
        (output / "comparison.json").write_text(
            json.dumps(summary, indent=2, default=_json_default) + "\n"
        )
        cost_start = perf_counter()
        if figures:
            _comparison_figure(saved, summary, output / "comparison.png")
            _lifecycle_figure(saved, output / "track_lifecycle.png")
        add_cost("figures", cost_start)
        if measure_costs:
            total = perf_counter() - benchmark_start
            costs["other"] = max(0.0, total - sum(costs.values()))
            summary["phase_costs_s"] = costs
            summary["measured_total_s"] = total
            (output / "comparison.json").write_text(
                json.dumps(summary, indent=2, default=_json_default) + "\n"
            )
        return summary
    finally:
        provider.close()


def _comparison_figure(saved: dict, summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth

    fig, axes = plt.subplots(
        2,
        len(saved),
        figsize=(max(9, 4 * len(saved)), 7),
        squeeze=False,
        layout="constrained",
        sharex=True,
        sharey="row",
    )
    for column, (name, results) in enumerate(saved.items()):
        gt, tracks = {}, {}
        raw = []
        for result in results:
            s = result["_typed_sensing"]
            t = s["temporal"]["result"]["reference_time_s"]
            truth = extract_recorded_evaluation_truth(s)
            if truth and not truth.get("_evaluation_unavailable_reason"):
                for ident, r, v in zip(
                    truth["target_ids"], truth["target_ranges"], truth["target_speeds"], strict=True
                ):
                    gt.setdefault(ident, []).append((t, r, np.nan if v is None else v))
            for d in s["detections"]:
                raw.append((t, d["range_m"], d["velocity_m_s"]))
                if "track_id" in d:
                    tracks.setdefault(d["track_id"], []).append(
                        (
                            t,
                            d.get("track_range_m", d["range_m"]),
                            d.get("track_velocity_m_s", d["velocity_m_s"]),
                        )
                    )
        for row, label in enumerate(("Range (m)", "Radial closing velocity (m/s)")):
            ax = axes[row, column]
            for ident, points in gt.items():
                a = np.array(points)
                ax.plot(a[:, 0], a[:, row + 1], "--", lw=2, label=f"{ident} GT")
            if raw:
                a = np.array(raw)
                ax.scatter(
                    a[:, 0], a[:, row + 1], s=10, c="0.65", marker="x", label="Output detections"
                )
            for ident, points in tracks.items():
                a = np.array(points)
                # Missing updates stay gaps: trackers do not emit predictions
                # during coasting in this interface.
                ax.scatter(a[:, 0], a[:, row + 1], s=9, label=f"Track {ident}")
            ax.set(xlabel="Time (s)", ylabel=label)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, ncols=2)
        record = summary["recipes"][name]
        m = record["metrics"]
        axes[0, column].set_title(
            f"{name}\nTP {m['total_tp']} · FP {m['total_fp']} · FN {m['total_fn']}\nProcessing {record['processing_seconds']*1000:.1f} ms",
            fontsize=10,
        )
    fig.suptitle(
        "Same saved acquisition · all output detections and tracks · identical evaluation gates"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _lifecycle_figure(saved: dict, path: Path) -> None:
    """Show observed IDs separately from internal active/confirmed counts."""
    import matplotlib.pyplot as plt

    tracked = {
        name: results
        for name, results in saved.items()
        if any(
            r["_typed_sensing"]
            .get("detection_filter_metadata", {})
            .get("tracker", {})
            .get("enabled")
            for r in results
        )
    }
    if not tracked:
        return
    fig, axes = plt.subplots(
        2,
        len(tracked),
        figsize=(max(9, 4 * len(tracked)), 6),
        squeeze=False,
        layout="constrained",
        sharex=True,
    )
    for column, (name, results) in enumerate(tracked.items()):
        times, active, confirmed, emitted = [], [], [], []
        tracks = {}
        for result in results:
            s = result["_typed_sensing"]
            t = s["temporal"]["result"]["reference_time_s"]
            meta = s.get("detection_filter_metadata", {}).get("tracker", {})
            times.append(t)
            active.append(meta.get("active_tracks", np.nan))
            confirmed.append(meta.get("confirmed_tracks", np.nan))
            emitted.append(len(s["detections"]))
            if meta.get("observation_delivery_experiment", {}).get("withheld"):
                for ax in axes[:, column]:
                    ax.axvline(t, color="0.8", alpha=0.8, lw=6)
            for detection in s["detections"]:
                if "track_id" in detection:
                    tracks.setdefault(detection["track_id"], []).append(t)
        for identifier, ts in tracks.items():
            axes[0, column].scatter(ts, [identifier] * len(ts), s=16)
        axes[0, column].set(title=name, ylabel="Emitted track ID")
        if tracks:
            from matplotlib.ticker import MaxNLocator

            axes[0, column].set_ylim(min(tracks) - 0.5, max(tracks) + 0.5)
            axes[0, column].yaxis.set_major_locator(MaxNLocator(integer=True))
        for values, label, style in (
            (active, "Active", "-"),
            (confirmed, "Confirmed", "--"),
            (emitted, "Output reports", ":"),
        ):
            axes[1, column].step(times, values, where="post", label=label, linestyle=style)
        axes[1, column].set(xlabel="Time (s)", ylabel="Count")
        axes[1, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.grid(alpha=0.25)
    fig.suptitle(
        "Track lifecycle: stored state counts and emitted observations\nGray stripes mark explicitly withheld updates, when configured",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_recipe_file(source: Path, recipe_file: Path, output: Path) -> dict:
    """Run named recipes with shared defaults from a versioned JSON file."""
    spec = json.loads(recipe_file.read_text())
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Recipe bundle requires schema_version: 1")
    unknown = set(spec) - {"schema_version", "defaults", "recipes", "evaluation", "summary"}
    if unknown:
        raise ValueError(f"Unknown recipe bundle fields: {', '.join(sorted(unknown))}")
    defaults = spec.get("defaults", {})
    entries = spec.get("recipes")
    evaluation = spec.get("evaluation", {})
    if not isinstance(defaults, dict) or not isinstance(entries, dict) or not entries:
        raise ValueError("Recipe defaults and nonempty recipes must be JSON objects")
    if not isinstance(evaluation, dict) or set(evaluation) - {
        "range_gate_m",
        "velocity_gate_m_s",
        "figures",
        "inspection_figures",
        "measure_costs",
    }:
        raise ValueError("Invalid recipe evaluation options")
    summary = spec.get("summary", False)
    if not isinstance(summary, bool):
        raise ValueError("Recipe summary must be true or false")
    recipes = {}
    for name, overrides in entries.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Recipe {name!r} must be a JSON object")
        recipes[name] = PostProcessingConfig(**{**defaults, **overrides})
    result = compare_recipes(source, recipes, output, **evaluation)
    if summary:
        from generator.figures.summary.saved_sensing import generate_saved_sensing_summary

        for name in recipes:
            destination = output / name
            generate_saved_sensing_summary(destination, destination / "summary")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_recipe_file(args.source, args.recipes, args.output_dir)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
