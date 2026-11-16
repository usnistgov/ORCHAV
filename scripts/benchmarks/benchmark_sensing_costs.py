#!/usr/bin/env python3
"""Measure CPU saved-input processing costs without altering the acquisition."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from shared.sensing.experiment import compare_recipes
from shared.sensing.postprocessor import PostProcessingConfig


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Saved scenario or frames directory; repeat for scene comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two measured repetitions.")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = PostProcessingConfig(
        cfar_threshold_scale=2, cfar_min_snr_db=6, min_detection_range_m=1, max_detection_range_m=30
    )
    tracker = replace(cfg, tracker_type="kalman", tracker_m=2)
    detailed = replace(tracker, include_stage_artifacts=True, include_tracking_diagnostics=True)
    configurations = [
        ("detector", cfg, False),
        ("Kalman", tracker, False),
        ("records", detailed, False),
        ("records + figures", detailed, True),
    ]
    records = []
    for scene_index, source in enumerate(args.source):
        for config_index, (label, recipe, figures) in enumerate(configurations):
            runs = []
            # Discard the first complete run. Imports and first-use caches are not
            # mixed into same-process repeated-work timings.
            for repeat in range(args.repeats + 1):
                output = args.output / f"{scene_index}_{config_index}_{repeat}"
                result = compare_recipes(
                    source,
                    {"recipe": recipe},
                    output,
                    figures=figures,
                    inspection_figures=figures,
                    measure_costs=True,
                )
                if repeat:
                    runs.append(
                        {
                            "costs_s": result["phase_costs_s"],
                            "total_s": result["measured_total_s"],
                            "metrics": result["recipes"]["recipe"]["metrics"],
                        }
                    )
            totals = [r["total_s"] for r in runs]
            middle = sorted(range(len(runs)), key=lambda i: totals[i])[len(runs) // 2]
            records.append(
                {
                    "scene": source.name,
                    "configuration": label,
                    "source_frame_set_id": result["source_frame_set_id"],
                    "source_generation_id": result["source_generation_id"],
                    "results": result["result_count"],
                    "runs": runs,
                    "representative": middle,
                    "median_s": float(np.median(totals)),
                    "p95_s": float(np.percentile(totals, 95)),
                    "source_hashes": json.loads((output / "source_hashes.json").read_text()),
                }
            )
            print(source.name, label, records[-1]["median_s"], flush=True)
    root = Path(__file__).resolve().parents[2]
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "shared", "scripts/benchmarks"], cwd=root
    )
    report = {
        "scope": "Same-process repeats after discarded full run; OS disk cache uncontrolled. "
        "CPU saved RD only: no acquisition, ray tracing, FFT rebuild or GUI. "
        "Serialization includes loading source display frames; warmup is one detector call. "
        "Measured total excludes Python imports, provider close and final timing JSON write.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "threads": os.environ.get("OPENBLAS_NUM_THREADS", "not set"),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "h5py": importlib.metadata.version("h5py"),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tracked_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": records,
    }
    (args.output / "costs.json").write_text(json.dumps(report, indent=2) + "\n")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, max(5, len(records) * 0.38)), layout="constrained")
    ax = fig.subplots()
    phases = [
        "source_open",
        "source_hashing",
        "input_loading",
        "warmup",
        "processing",
        "serialization",
        "evaluation",
        "result_readback",
        "figures",
        "other",
    ]
    left = np.zeros(len(records))
    for phase in phases:
        values = [r["runs"][r["representative"]]["costs_s"].get(phase, 0) for r in records]
        ax.barh(range(len(records)), values, left=left, label=phase.replace("_", " "))
        left += values
    ax.set_yticks(
        range(len(records)),
        [f"{r['scene']} ({r['results']} CPIs) · {r['configuration']}" for r in records],
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Wall time (s)",
        title="Saved RD processing costs · representative median-total run\nSame process, warmed caches · no ray tracing or GUI",
    )
    ax.legend(ncols=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(args.output / "costs.png", dpi=160)


if __name__ == "__main__":
    run()
