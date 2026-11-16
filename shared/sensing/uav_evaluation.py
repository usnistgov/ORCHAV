"""Saved-result UAV evaluation aligned with TR 38.765 V20.0.0, clauses 4.1/4.2.

This is an evaluation profile, not a channel/waveform implementation or a 3GPP
compliance certificate. A drop is one declared saved result epoch. Independent
run groups, rather than overlapping CPIs, are resampled for uncertainty intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from shared.sensing.fusion.core import gated_assignment
from shared.sensing.fusion.result import FusionResult

PROFILE = "3gpp_tr_38_765_v20_0_0_uav"
REFERENCE = "https://www.3gpp.org/ftp/Specs/archive/38_series/38.765/38765-k00.zip"
OBJECTIVES = {
    "miss_probability": 0.05,
    "false_alarm_type_1": 0.05,
    "false_alarm_type_2": 0.05,
    "horizontal_p90_m": 10.0,
    "vertical_p90_m": 10.0,
    "radial_velocity_p90_m_s": 5.0,
    "vector_velocity_p90_m_s": 5.0,
}


@dataclass(frozen=True)
class UAVProfile:
    """Frozen scoring choices, independent of detector and tracker gates.

    The ROI is an inclusive world-XYZ box applied to both truth and outputs.
    All active targets in it remain eligible, including occluded targets. No
    visibility is inferred. Error objectives never change the matching gate.
    """

    position_gate_m: float = 30.0
    roi_min_m: tuple[float, float, float] | None = None
    roi_max_m: tuple[float, float, float] | None = None
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 20260923

    def __post_init__(self):
        if not np.isfinite(self.position_gate_m) or self.position_gate_m <= 0:
            raise ValueError("Position association gate must be positive and finite")
        if (self.roi_min_m is None) != (self.roi_max_m is None):
            raise ValueError("Supply both ROI bounds or neither")
        if self.roi_min_m is not None:
            lo, hi = np.asarray(self.roi_min_m), np.asarray(self.roi_max_m)
            if (
                lo.shape != (3,)
                or hi.shape != (3,)
                or not np.all(np.isfinite([lo, hi]))
                or np.any(lo >= hi)
            ):
                raise ValueError("ROI requires finite ordered XYZ bounds")
            object.__setattr__(self, "roi_min_m", tuple(float(v) for v in lo))
            object.__setattr__(self, "roi_max_m", tuple(float(v) for v in hi))
        if type(self.bootstrap_samples) is not int or not 0 <= self.bootstrap_samples <= 10000:
            raise ValueError("Bootstrap sample count must be an integer in [0, 10000]")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**32:
            raise ValueError("Bootstrap seed must be an unsigned 32-bit integer")


def _vector(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} requires a finite world XYZ vector")
    return result


def _inside(position, config):
    p = _vector(position, "Position")
    return config.roi_min_m is None or bool(
        np.all(p >= config.roi_min_m) and np.all(p <= config.roi_max_m)
    )


def _mean(values):
    return float(np.mean(values)) if values else None


def _statistics(rows):
    positive = [r for r in rows if r["truth_count"]]
    negative = [r for r in rows if not r["truth_count"]]
    nonempty = [r for r in positive if r["output_count"]]
    errors = [e for r in rows for e in r["errors"]]
    result = {
        "miss_probability": _mean([r["fn"] / r["truth_count"] for r in positive]),
        "false_alarm_type_1": _mean([float(r["output_count"] > 0) for r in negative]),
        "false_alarm_type_2": _mean([r["fp"] / r["output_count"] for r in nonempty]),
    }
    for field, name in (
        ("horizontal_m", "horizontal_p90_m"),
        ("vertical_m", "vertical_p90_m"),
        ("radial_velocity_m_s", "radial_velocity_p90_m_s"),
        ("vector_velocity_m_s", "vector_velocity_p90_m_s"),
    ):
        values = [e[field] for e in errors if e.get(field) is not None]
        result[name] = float(np.quantile(values, 0.9)) if values else None
    return result


def evaluate_events(events, config=UAVProfile()):
    """Score one cohort with a common one-to-one 3D association at each epoch.

    Each event names run_id/event_id/time_s, truth (None when unavailable), and
    outputs. Rows contain world positions, optional world velocities and optional
    closing radial speeds. Unusable outputs exclude the epoch rather than quietly
    disappearing from its false-alarm denominator. Empty truth is evaluable.
    """
    rows, skipped, seen, targets = [], Counter(), set(), defaultdict(list)
    previous_times = {}
    run_groups = {}
    for event in events:
        identity = (event["run_id"], event["event_id"])
        if identity in seen:
            raise ValueError(f"Repeated evaluation event: {identity}")
        seen.add(identity)
        time = float(event["time_s"])
        if not all(isinstance(v, str) and v for v in identity) or not np.isfinite(time):
            raise ValueError("Events need explicit run/event identities and finite time")
        if time <= previous_times.get(event["run_id"], -np.inf):
            raise ValueError("Epoch times must increase within each run")
        previous_times[event["run_id"]] = time
        group = event.get("independent_run_group")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("Independent trial groups must be explicit nonempty strings")
        if event["run_id"] in run_groups and run_groups[event["run_id"]] != group:
            raise ValueError("A run must belong to one consistent independent trial group")
        run_groups[event["run_id"]] = group
        if event["truth"] is None:
            skipped["truth_unavailable_or_unaligned"] += 1
            continue
        try:
            truth = [t for t in event["truth"] if _inside(t["position_m"], config)]
            outputs = [o for o in event["outputs"] if _inside(o["position_m"], config)]
        except (TypeError, ValueError, KeyError):
            skipped["world_position_unavailable"] += 1
            continue
        ids = [t["target_id"] for t in truth]
        if any(not isinstance(target_id, str) or not target_id for target_id in ids):
            raise ValueError("Truth target IDs must be nonempty semantic strings")
        if len(ids) != len(set(ids)):
            raise ValueError("Truth target IDs must be unique within an epoch")
        costs = np.array(
            [
                [np.linalg.norm(np.subtract(t["position_m"], o["position_m"])) for o in outputs]
                for t in truth
            ]
        ).reshape(len(truth), len(outputs))
        pairs = gated_assignment(costs, config.position_gate_m)
        matched = dict(pairs)
        errors = []
        for i, t in enumerate(truth):
            j = matched.get(i)
            o = outputs[j] if j is not None else None
            targets[(event["run_id"], t["target_id"])].append(
                {
                    "time_s": time,
                    "matched": o is not None,
                    "track_id": o.get("track_id") if o else None,
                    "predicted_only": bool(o and o.get("predicted_only", False)),
                }
            )
            if o is None:
                continue
            delta = np.subtract(o["position_m"], t["position_m"])
            error = {
                "target_id": t["target_id"],
                "horizontal_m": float(np.linalg.norm(delta[:2])),
                "vertical_m": abs(float(delta[2])),
            }
            if t.get("velocity_m_s") is not None and o.get("velocity_m_s") is not None:
                error["vector_velocity_m_s"] = float(
                    np.linalg.norm(
                        _vector(o["velocity_m_s"], "Output velocity")
                        - _vector(t["velocity_m_s"], "Truth velocity")
                    )
                )
            if (
                t.get("radial_velocity_m_s") is not None
                and o.get("radial_velocity_m_s") is not None
            ):
                values = [t["radial_velocity_m_s"], o["radial_velocity_m_s"]]
                if not np.all(np.isfinite(values)):
                    raise ValueError("Recorded radial speeds must be finite")
                error["radial_velocity_m_s"] = abs(float(values[1] - values[0]))
            errors.append(error)
        rows.append(
            {
                "run_id": event["run_id"],
                "event_id": event["event_id"],
                "time_s": time,
                "independent_run_group": event.get("independent_run_group"),
                "truth_count": len(truth),
                "output_count": len(outputs),
                "tp": len(pairs),
                "fp": len(outputs) - len(pairs),
                "fn": len(truth) - len(pairs),
                "errors": errors,
            }
        )
    values = _statistics(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["independent_run_group"] is not None:
            groups[row["independent_run_group"]].append(row)
    intervals = {k: None for k in values}
    if (
        len(groups) >= 2
        and all(r["independent_run_group"] is not None for r in rows)
        and config.bootstrap_samples
    ):
        rng = np.random.default_rng(config.bootstrap_seed)
        blocks = list(groups.values())
        boot = defaultdict(list)
        for _ in range(config.bootstrap_samples):
            sample = [
                r for index in rng.integers(len(blocks), size=len(blocks)) for r in blocks[index]
            ]
            for key, value in _statistics(sample).items():
                if value is not None:
                    boot[key].append(value)
        for key, samples in boot.items():
            intervals[key] = {
                "bounds": np.quantile(samples, [0.025, 0.975]).tolist(),
                "defined_resamples": len(samples),
            }
    outcomes = []
    for (run, target), history in sorted(targets.items()):
        history.sort(key=lambda r: r["time_s"])
        hits = [h for h in history if h["matched"]]
        previous_id, switches, losses, reacquisitions, found, lost = None, 0, 0, 0, False, False
        for h in history:
            if h["matched"]:
                if h["track_id"] is not None:
                    switches += int(previous_id is not None and previous_id != h["track_id"])
                    previous_id = h["track_id"]
                reacquisitions += int(lost)
                found, lost = True, False
            elif found and not lost:
                losses += 1
                lost = True
        outcomes.append(
            {
                "run_id": run,
                "target_id": target,
                "opportunities": len(history),
                "coverage": len(hits) / len(history),
                "losses": losses,
                "reacquisitions": reacquisitions,
                "identity_switches": switches,
                "first_match_delay_s": hits[0]["time_s"] - history[0]["time_s"] if hits else None,
                "matched_prediction_only_epochs": sum(h["predicted_only"] for h in hits),
            }
        )
    total = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")}
    positive = [r for r in rows if r["truth_count"]]
    return {
        "metrics": {
            k: {
                "value": v,
                "objective": OBJECTIVES[k],
                "status": (
                    "not_measured"
                    if v is None
                    else "meets_observed_objective" if v <= OBJECTIVES[k] else "exceeds_objective"
                ),
                "run_bootstrap_95_interval": intervals[k],
            }
            for k, v in values.items()
        },
        "counts": {
            **total,
            "evaluated_epochs": len(rows),
            "positive_epochs": len(positive),
            "negative_epochs": len(rows) - len(positive),
            "positive_epochs_with_outputs": sum(r["output_count"] > 0 for r in positive),
            "independent_run_groups": len(groups),
            "matched_coverage": (
                total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else None
            ),
        },
        "skipped": dict(skipped),
        "targets": outcomes,
        "epochs": rows,
        "uncertainty_note": "Run-block bootstrap only when every epoch has a declared independent_run_group and at least two groups exist. Deterministic cases are not independent random trials. Observed objectives are not compliance certification.",
    }


def events_from_result(result, run_id, cohort):
    """Use the canonical reader; distinguish raw sensor detections and track states."""
    if not cohort.startswith(("detections:", "tracks:", "updated_tracks:")):
        raise ValueError("Unknown output cohort")
    reader = result if isinstance(result, FusionResult) else FusionResult(result)
    if cohort.startswith("detections:"):
        sensor = cohort.split(":", 1)[1]
        history = reader["observations"][sensor]
        origin = np.asarray(reader["sources"][sensor]["sensor"]["position_m"])
    else:
        mode = cohort.split(":", 1)[1]
        history = reader["histories"][mode]
        origin = None
    events = []
    for index, (epoch, gt) in enumerate(zip(history, reader["truth_for_evaluation"], strict=True)):
        if epoch["time_s"] != gt["time_s"]:
            raise ValueError("Result and truth epochs differ")
        truth = [dict(t) for t in gt["targets"]] if gt.get("available", True) else None
        if origin is not None:
            outputs = [
                {
                    "position_m": d.get("world_position_m"),
                    "radial_velocity_m_s": (
                        d.get("closing_velocity_m_s")
                        if "closing_velocity_m_s" in d.get("components", [])
                        else None
                    ),
                }
                for d in epoch["detections"]
            ]
            for t in truth or []:
                if t.get("velocity_m_s") is not None:
                    delta = np.subtract(t["position_m"], origin)
                    if np.linalg.norm(delta) > 0:
                        t["radial_velocity_m_s"] = -float(
                            delta @ t["velocity_m_s"] / np.linalg.norm(delta)
                        )
        else:
            outputs = [
                dict(t)
                for t in epoch["tracks"]
                if t["confirmed"] and (cohort.startswith("tracks:") or not t["predicted_only"])
            ]
        events.append(
            {
                "run_id": run_id,
                "event_id": str(index),
                "time_s": epoch["time_s"],
                "truth": truth,
                "outputs": outputs,
            }
        )
    return events


def evaluate_manifest(manifest_path, output):
    """Publish a separate portable report; source bundles remain unchanged."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("profile") != PROFILE:
        raise ValueError("Unsupported UAV evaluation manifest")
    config = UAVProfile(**manifest["evaluation"])
    if output.exists():
        raise FileExistsError(f"Preserve existing evaluation: {output}")
    readers, identities, result_hashes = [], set(), set()
    for trial in manifest["trials"]:
        if trial["run_id"] in identities:
            raise ValueError("Each independent trial needs a distinct run_id")
        identities.add(trial["run_id"])
        path = (manifest_path.parent / trial["result"]).resolve()
        reader = FusionResult(path)
        digest = hashlib.sha256(reader.path.read_bytes()).hexdigest()
        if digest in result_hashes:
            raise ValueError("The same result cannot represent multiple independent trials")
        result_hashes.add(digest)
        readers.append(({**trial, "result_sha256": digest}, reader))
    cohorts = manifest["cohorts"]
    if not readers or not cohorts or len(cohorts) != len(set(cohorts)):
        raise ValueError("Supply trials and distinct output cohorts")
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "reference": REFERENCE,
        "evaluation": asdict(config),
        "cohorts": {},
        "cases": {},
        "sources": [],
        "assumptions": manifest.get("assumptions", []),
        "not_measured": [
            "NR waveform compliance",
            "calibrated UAV RCS",
            "end-to-end service latency",
            "sensing resolution",
            "classification",
        ],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    for cohort in cohorts:
        if not cohort.startswith(("detections:", "tracks:", "updated_tracks:")):
            raise ValueError("Unknown output cohort")
        events = [
            {**e, "independent_run_group": trial.get("independent_run_group")}
            for trial, reader in readers
            for e in events_from_result(reader, trial["run_id"], cohort)
        ]
        report["cohorts"][cohort] = evaluate_events(events, config)
        for case in sorted({trial.get("case", "unspecified") for trial, _ in readers}):
            run_ids = {
                trial["run_id"] for trial, _ in readers if trial.get("case", "unspecified") == case
            }
            report["cases"].setdefault(case, {})[cohort] = evaluate_events(
                [e for e in events if e["run_id"] in run_ids], config
            )
    for trial, reader in readers:
        times = np.asarray([e["time_s"] for e in reader["truth_for_evaluation"]])
        if hashlib.sha256(reader.path.read_bytes()).hexdigest() != trial["result_sha256"]:
            raise RuntimeError("Saved result changed during evaluation")
        report["sources"].append(
            {
                **trial,
                "fusion_run_id": reader.inspection_id,
                "code_identity": reader.get("code_identity"),
                "native_sources": reader["sources"],
                "result_intervals_s": np.diff(times).tolist(),
                "offline_tracking_seconds": reader.get("processing_seconds"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    staging = Path(tempfile.mkdtemp(prefix=".uav-evaluation-", dir=output.parent))
    try:
        (staging / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        write_report(report, staging)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def write_report(report, output):
    """Standalone error CDFs and a readable report with counts beside accuracy."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    fields = ("horizontal_m", "vertical_m", "radial_velocity_m_s", "vector_velocity_m_s")
    for name, data in report["cohorts"].items():
        errors = [e for row in data["epochs"] for e in row["errors"]]
        for axis, key in zip(axes, fields, strict=True):
            values = np.sort([e[key] for e in errors if key in e])
            if len(values):
                axis.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    label=f"{name} (n={len(values)})",
                )
    for axis, label, objective in zip(
        axes,
        (
            "Horizontal error (m)",
            "Altitude error (m)",
            "Radial velocity error (m/s)",
            "World velocity-vector error (m/s)",
        ),
        (10, 10, 5, 5),
        strict=True,
    ):
        axis.axvline(objective, color="grey", linestyle="--")
        axis.axhline(0.9, color="grey", linestyle=":")
        axis.set(xlabel=label, ylabel="Matched-output empirical CDF", ylim=(0, 1.02))
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.savefig(output / "accuracy_cdf.png", dpi=140)
    sections = [
        "<h1>UAV sensing evaluation</h1><p>Saved results only; detector not rerun. "
        "Observed objectives do not certify 3GPP compliance. Accuracy is conditional on matching.</p>"
    ]
    for name, data in report["cohorts"].items():
        counts = data["counts"]
        coverage = (
            "not measured"
            if counts["matched_coverage"] is None
            else f"{100*counts['matched_coverage']:.1f}%"
        )
        sections.append(
            f"<h2>{html.escape(name)}</h2><p><b>TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}</b> · matched coverage {coverage}</p>"
        )
        sections.append(
            f"<p>{counts['positive_epochs']} target-present epochs, {counts['negative_epochs']} target-absent epochs, {counts['positive_epochs_with_outputs']} positive epochs with outputs. {counts['independent_run_groups']} declared independent groups.</p>"
        )
        sections.append(
            "<table><tr><th>Metric</th><th>Measured</th><th>Objective</th><th>Status</th></tr>"
        )
        for metric, row in data["metrics"].items():
            probability = metric.startswith(("miss_", "false_"))
            scale, unit = (100, "%") if probability else (1, "")
            value = "not measured" if row["value"] is None else f"{scale*row['value']:.5g}{unit}"
            sections.append(
                f"<tr><td>{metric.replace('_',' ')}</td><td>{value}</td><td>{scale*row['objective']:g}{unit}</td><td>{row['status'].replace('_',' ')}</td></tr>"
            )
        sections.append("</table>")
        sections.append(f"<p>Excluded: {html.escape(str(data['skipped']))}</p>")
    sections.append(
        '<img src="accuracy_cdf.png" alt="Matched position and velocity error distributions">'
    )
    sections.append("<details><summary>Per-case counts and target outcomes</summary>")
    for case, cohorts in report.get("cases", {}).items():
        sections.append(
            f"<h3>{html.escape(case)}</h3><table><tr><th>Outputs</th><th>TP</th><th>FP</th><th>FN</th><th>Coverage</th></tr>"
        )
        for name, data in cohorts.items():
            c = data["counts"]
            coverage = "—" if c["matched_coverage"] is None else f"{100*c['matched_coverage']:.1f}%"
            sections.append(
                f"<tr><td>{html.escape(name)}</td><td>{c['tp']}</td><td>{c['fp']}</td><td>{c['fn']}</td><td>{coverage}</td></tr>"
            )
        sections.append("</table>")
        sections.append(
            "<pre>"
            + html.escape(
                json.dumps({name: data["targets"] for name, data in cohorts.items()}, indent=2)
            )
            + "</pre>"
        )
    sections.append("</details>")
    sections.append(
        "<h2>Scope and assumptions</h2><ul>"
        + "".join(f"<li>{html.escape(str(a))}</li>" for a in report["assumptions"])
        + "</ul>"
    )
    sections.append("<p>Not measured: " + html.escape(", ".join(report["not_measured"])) + ".</p>")
    sections.append(
        "<p>Run-block confidence intervals are absent unless all epochs belong to at least two declared independent trial groups. See JSON for intervals and defined resample counts.</p>"
    )
    sections.append(
        '<p><a href="evaluation.json">Full metrics, denominators, intervals, target outcomes and source identities</a></p>'
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>UAV sensing evaluation</title><style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}td,th{padding:.45em;text-align:left;border-bottom:1px solid #ccc}img{max-width:100%}</style>'
        + "\n".join(sections)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evaluate_manifest(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
