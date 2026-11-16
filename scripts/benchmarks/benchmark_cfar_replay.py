"""Compare CFAR implementations on identical saved CFR acquisitions.

Run with --recording <scenario> (repeatable), --reference-source <old cfar.py>,
--output <fresh directory>. No ray tracing or receiver-sample changes occur.
The reference is explicitly supplied trusted Python source, loaded only here.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import io
import json
import math
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

import shared.sensing.detection.cfar as cfar
from shared.sensing.loader import load_frames_sensing
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor


def saved_processor(sensing, *, artifacts=True):
    """Reconstruct the saved native detector recipe, with tracking disabled."""
    recipe = sensing["config_resolved"]["processing_recipe"]
    supported = {f.name for f in fields(PostProcessingConfig)}
    params = {k: v for k, v in recipe.items() if k in supported}
    for dimension in ("doppler", "range"):
        for region in ("guard", "ref"):
            params[f"cfar_{region}_cells_{dimension}"] = recipe[
                f"cfar_rd_{region}_cells_{dimension}"
            ]
    params.update(
        range_ifft_size=recipe["fft_size_range"],
        nms_radius_doppler=max(2, recipe["cfar_rd_guard_cells_doppler"]),
        nms_radius_range=max(4, recipe["cfar_rd_guard_cells_range"]),
        tracker_type="none",
        include_stage_artifacts=artifacts,
    )
    return SensingPostProcessor(PostProcessingConfig(**params))


def digest(value):
    """Compare every result field, including arrays, NaNs and detection stages."""
    if is_dataclass(value):
        return {f.name: digest(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return [str(value.dtype), value.shape, hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest(v) for v in value]
    return value


def assert_identical(a, b):
    assert json.dumps(digest(a), sort_keys=True) == json.dumps(digest(b), sort_keys=True)


@contextmanager
def implementation(kernel):
    previous = cfar._cfar_2d_ca_vectorized
    cfar._cfar_2d_ca_vectorized = kernel
    try:
        yield
    finally:
        cfar._cfar_2d_ca_vectorized = previous


def distribution(values):
    return {
        "seconds": values,
        "median_s": float(np.median(values)),
        "min_s": min(values),
        "max_s": max(values),
        "p25_s": float(np.percentile(values, 25)),
        "p75_s": float(np.percentile(values, 75)),
    }


def compare_runs(function, kernels, repeats):
    outputs = {}
    for name, kernel in kernels.items():
        with implementation(kernel):
            outputs[name] = function()  # Warm imports and allocation paths.
    assert_identical(outputs["reference"], outputs["current"])
    timings = {name: [] for name in kernels}
    for repeat in range(repeats):
        # Alternate order to reduce systematic thermal/cache-order bias.
        order = list(kernels) if repeat % 2 == 0 else list(reversed(kernels))
        for name in order:
            with implementation(kernels[name]):
                start = time.perf_counter()
                result = function()
                timings[name].append(time.perf_counter() - start)
            assert_identical(result, outputs[name])
    report = {name: distribution(values) for name, values in timings.items()}
    report["speedup"] = report["reference"]["median_s"] / report["current"]["median_s"]
    report["bit_identical"] = True
    return report


def profile(function, output):
    profiler = cProfile.Profile()
    profiler.runcall(function)
    profiler.dump_stats(str(output.with_suffix(".prof")))
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(45)
    output.with_suffix(".txt").write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three timed repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "_cfar_benchmark_reference", args.reference_source
    )
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    kernels = {
        "reference": reference._cfar_2d_ca_vectorized,
        "current": cfar._cfar_2d_ca_vectorized,
    }
    report = {
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.reference_source, Path(cfar.__file__), Path(__file__))
        },
        "repetitions": args.repeats,
        "recordings": [],
        "maps": {},
    }
    for index, recording in enumerate(args.recording):
        frames = load_frames_sensing(recording)
        sensing = frames[0]["_typed_sensing"]
        processor = saved_processor(sensing)

        def replay():
            return processor.process_from_frequency_observation(sensing)

        result = compare_runs(replay, kernels, args.repeats)
        result["recording"] = str(recording.resolve())
        result["input_sha256"] = hashlib.sha256(
            sensing["cfr_by_pair_rx_element"].tobytes()
        ).hexdigest()
        result["shape"] = list(sensing["cfr_by_pair_rx_element"].shape)
        for frame in frames:
            s = frame["_typed_sensing"]
            with implementation(kernels["reference"]):
                before = saved_processor(s).process_from_frequency_observation(s)
            after = saved_processor(s).process_from_frequency_observation(s)
            assert_identical(before, after)
        result["all_cpis_identical"] = len(frames)
        # Allocation tracing is separate from timed benchmark repetitions.
        tracemalloc.start()
        replay()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["current_traced_peak_bytes"] = peak
        profile(replay, args.output / f"recording_{index}_current")
        calls = []
        old = cfar._local_training_sums
        fsum = math.fsum
        exact_calls = [0]

        def exact(values):
            exact_calls[0] += 1
            return fsum(values)

        def local(arr, locations, **kwargs):
            start = time.perf_counter()
            value = old(arr, locations, **kwargs)
            calls.append({"cells": len(locations), "seconds": time.perf_counter() - start})
            return value

        try:
            cfar._local_training_sums = local
            math.fsum = exact
            replay()
        finally:
            cfar._local_training_sums = old
            math.fsum = fsum
        result["local_sums"] = calls
        result["ambiguous_rounding_fsum_calls"] = exact_calls[0]
        without = saved_processor(sensing, artifacts=False)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            without.process_from_frequency_observation(sensing)
            times.append(time.perf_counter() - start)
        result["current_without_artifacts"] = distribution(times)
        report["recordings"].append(result)
        print(recording, result["speedup"], "all arrays/stages/detections identical", flush=True)

    rng = np.random.default_rng(854)
    maps = {"ordinary": rng.exponential(size=(64, 256))}
    maps["low_power"] = maps["ordinary"] * 1e-240
    maps["low_sidelobes"] = maps["ordinary"] * 1e-100
    maps["low_sidelobes"][30, 45] = 1.0
    maps["sparse"] = np.zeros((64, 256))
    maps["sparse"][30, 45] = 1.0
    maps["sparse"][30, 60] = 1e-100
    for name, power in maps.items():

        def detect():
            r = cfar.cfar_2d(power, guard_cells=(1, 4), reference_cells=(4, 12), threshold_scale=16)
            return vars(r)

        report["maps"][name] = compare_runs(detect, kernels, args.repeats)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
