"""Acquire a cube experiment in a fresh directory and audit its saved CFR.

Run as ``python -m scripts.analysis.validate_cube_radar --scenario ...
--output ... [--pulses 256]``. The CUDA variant is selected before Sionna is
imported. Generation uses the checkout's ``python -m generator`` entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import runpy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml


def run(scenario: Path, output: Path, pulses: int | None = None) -> dict:
    import mitsuba as mi

    mi.set_variant("cuda_ad_mono_polarized")
    print("Mitsuba variant:", mi.variant(), flush=True)
    from generator.core.services import raytracing_service

    root = Path(__file__).resolve().parents[2]
    source = scenario / "scenario.yaml" if scenario.is_dir() else scenario
    data = yaml.safe_load(source.read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data["scene"] = {"source": "local", "id": str(root / "libraries/scenes" / data["scene"]["id"])}
    for target in data["actors"]["targets"]:
        target["asset"]["path"] = str(root / target["asset"]["path"])
    if pulses is not None:
        data["timeline"]["steps"] = pulses
    (output / "scenario.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    timings = {"acquisition_s": [], "processing_s": []}
    factory = raytracing_service._load_sensing_processor_factory
    create = factory()
    original_rt = raytracing_service.compute_ray_tracing_step

    def timed(kind, function, *args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        timings[kind].append(perf_counter() - start)
        return result

    def instrumented(*args, **kwargs):
        processor = create(*args, **kwargs)
        process = processor.process
        processor.process = lambda *a, **kw: timed("processing_s", process, *a, **kw)
        return processor

    raytracing_service._load_sensing_processor_factory = lambda: instrumented
    raytracing_service.compute_ray_tracing_step = lambda *a, **kw: timed(
        "acquisition_s", original_rt, *a, **kw
    )
    old_argv = sys.argv
    start = perf_counter()
    try:
        sys.argv = ["generator", str(output)]
        try:
            runpy.run_module("generator", run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv
        raytracing_service._load_sensing_processor_factory = factory
        raytracing_service.compute_ray_tracing_step = original_rt
    elapsed = perf_counter() - start
    generation_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    from shared.frames.providers import Hdf5Provider
    from shared.sensing.cfr_replay import rebuild_frequency_observation

    provider = Hdf5Provider(output)
    records = []
    incomplete = 0
    worst_rd = worst_profile = 0.0
    prior_observation = None
    overlap_checks = 0
    try:
        for step in provider.list_frames():
            sensing = provider.load_frame(step).sensing
            if not sensing:
                continue
            temporal = sensing["temporal"]
            if not temporal.get("result"):
                incomplete += 1
                assert "range_doppler_by_pair" not in sensing
                continue
            rebuilt = rebuild_frequency_observation(sensing)
            native = sensing["range_doppler_by_pair"][0]
            error = np.max(abs(native - rebuilt["rd_cube"])) / max(np.max(abs(native)), 1e-30)
            profile = sensing["range_profile_by_pair"][0]
            rp_error = np.max(abs(profile - rebuilt["range_profile"])) / max(np.max(profile), 1e-30)
            worst_rd = max(worst_rd, float(error))
            worst_profile = max(worst_profile, float(rp_error))
            observation = sensing["cfr_by_pair_rx_element"]
            sequential = sensing["processing_strategy"] == "sequential_stft"
            if sequential and prior_observation is not None:
                np.testing.assert_array_equal(prior_observation[:, :, 1:], observation[:, :, :-1])
                overlap_checks += 1
            prior_observation = observation
            records.append(
                {
                    "frame": step,
                    "reference_time_s": temporal["result"]["reference_time_s"],
                    "detections": sensing.get("detections", []),
                }
            )
    finally:
        provider.close()
    assert worst_rd < 1e-12 and worst_profile < 1e-12
    report = {
        "variant": mi.variant(),
        "scenario": str(scenario),
        "source_yaml_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pulses": data["timeline"]["steps"],
        "incomplete_acquisitions": incomplete,
        "complete_windows": len(records),
        "exact_overlap_checks": overlap_checks,
        "max_relative_replay_error": worst_rd,
        "max_relative_profile_error": worst_profile,
        "generation_wall_s": elapsed,
        "generation_peak_rss_mib": generation_peak_rss,
        "timing_scope": "Acquisition is scene update and RT; processing includes CIR extraction for sequential pulses. Generation wall time also includes initialization, persistence and summaries.",
        "timings": {
            key: {"count": len(v), "total_s": sum(v), "mean_s": float(np.mean(v))}
            for key, v in timings.items()
        },
        "frame_bytes": sum(p.stat().st_size for p in (output / "frames").rglob("*") if p.is_file()),
        "results": records,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pulses", type=int)
    args = parser.parse_args()
    run(args.scenario, args.output, args.pulses)
