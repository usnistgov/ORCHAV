"""Repeated fixed-input CPU stages and optional saved-recording replay costs.

Use --code-root with an archived baseline tree for a default-route comparison.
The deterministic numerical workloads exclude acquisition. Live acquisition
timings are recorded independently by finite_bandwidth_cube/generate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    import numpy as np

    from generator.core.configuration import AntennaConfig, SensingConfig
    from generator.core.sensing.acquisition import resolve_sensing_execution_values
    from generator.core.sensing.processor import create_sensing_processor

    os.environ["ORCHAV_SENSING_PROFILE"] = "1"
    records = []
    for name, e, p, t, n in (
        ("small", 1, 4, 16, 64),
        ("medium", 1, 128, 64, 256),
        ("array", 8, 64, 64, 256),
    ):
        rng = np.random.default_rng(20260925)
        times = np.arange(t) / 1000
        gains = (rng.standard_normal((e, p)) + 1j * rng.standard_normal((e, p))) / np.sqrt(2 * p)
        a = gains[:, :, None] * np.exp(
            2j * np.pi * rng.uniform(-400, 400, p)[None, :, None] * times
        )
        tau = rng.uniform(5, n - 5, p) / 100e6
        identity = hashlib.sha256(
            a.astype("<c16").tobytes() + tau.astype("<f8").tobytes()
        ).hexdigest()
        simulation = SimpleNamespace(
            simulation_config=SimpleNamespace(rx_antenna=AntennaConfig(num_rows=1, num_cols=e))
        )
        for model in (("point_bin_floor",) if args.point_only else ("point_bin_floor", "cfr_ifft")):
            cfg = SensingConfig(
                enabled=True,
                processing_mode="coherent_cpi",
                bandwidth=100e6,
                prf_hz=1000,
                fft_size_range=n,
                fft_size_doppler=t,
                window_range="boxcar",
                window_doppler="hann",
                raw_payload_policy="diagnostic",
                output_detections=True,
            )
            if model == "cfr_ifft":
                from shared.sensing.range_response import RangeResponseConfig

                cfg.range_response = RangeResponseConfig(model, n)
                cfg.__post_init__()
            execution = resolve_sensing_execution_values(
                cfg,
                cir_time_steps=t,
                cir_sampling_frequency_hz=1000,
                frame_dt_s=0.001,
                rx_elements=e,
            )
            processor = create_sensing_processor(cfg, 28e9, execution)

            def run():
                return processor.process(
                    simulation,
                    0,
                    None,
                    cir_stack=(a[None, :, None, None], tau[None, None]),
                    source_time_s=0.0,
                )

            run()
            measurements = []
            for _ in range(args.repeats):
                start = perf_counter()
                result = run()
                measurements.append(
                    {
                        "total_s": perf_counter() - start,
                        "stages_ms": result["timing"]["stage_profile_ms"],
                    }
                )
            tracemalloc.start()
            before = tracemalloc.get_traced_memory()[0]
            result = run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            records.append(
                {
                    "workload": name,
                    "model": model,
                    "elements": e,
                    "paths": p,
                    "slow_time": t,
                    "frequencies": n,
                    "input_sha256": identity,
                    "runs": measurements,
                    "median_s": float(np.median([m["total_s"] for m in measurements])),
                    "additional_traced_peak_bytes": peak - before,
                    "retained_dense_array_bytes": sum(
                        v.nbytes for v in result.values() if isinstance(v, np.ndarray)
                    ),
                }
            )
    replay_records = []
    for index, source in enumerate(args.source):
        from shared.sensing.experiment import compare_recipes
        from shared.sensing.loader import load_frames_sensing
        from shared.sensing.postprocessor import PostProcessingConfig

        frame = load_frames_sensing(source / "frames")[0]["_typed_sensing"]
        response = frame["config_resolved"].get("range_response", {})
        overrides = {}
        if response.get("model") == "cfr_ifft":
            from shared.sensing.range_response import RangeResponseConfig

            overrides["range_response"] = RangeResponseConfig.parse(response)
        cfg = PostProcessingConfig(include_stage_artifacts=True, **overrides)
        for repeat in range(4):
            result = compare_recipes(
                source,
                {"boxcar": cfg},
                args.output / f"replay_{index}_{repeat}",
                figures=False,
                measure_costs=True,
            )
            if repeat:
                replay_records.append(
                    {
                        "source_index": index,
                        "repeat": repeat,
                        "source_frame_set_id": result["source_frame_set_id"],
                        "result_count": result["result_count"],
                        "phases_s": result["phase_costs_s"],
                        "total_s": result["measured_total_s"],
                        "source_unchanged": result["source_unchanged"],
                    }
                )
    report = {
        "scope": "Warm CPU fixed-input projection, FFT and detector; no RT. One warmup discarded. Traced NumPy/Python peak is measured in a separate untimed call; output included, input excluded. RSS covers all imports and allocator history. Replay OS caches uncontrolled; serialization is not an fsync durability test.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
        },
        "code_module": sys.modules["generator.core.sensing.processor"].__file__,
        "workloads": records,
        "replay": replay_records,
    }
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
