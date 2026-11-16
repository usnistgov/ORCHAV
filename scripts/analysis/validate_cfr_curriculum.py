"""Compare authored sensing recipes on identical native CIR inputs.

Run with explicit current and baseline YAML files and a fresh output directory.
The primary recording uses the current recipe; a second recording retains the
baseline products from the same native acquisition. No path reconstruction from
point bins or ground-truth-assisted direction estimation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def stage_configuration(path: Path, destination: Path) -> None:
    """Relocate inputs only; keep all scientific and output settings intact."""
    data = yaml.safe_load(path.read_text())
    if data["scene"]["source"] == "library":
        data["scene"] = {
            "source": "local",
            "id": str(ROOT / "libraries/scenes" / data["scene"]["id"]),
        }
    for actor in data["actors"].get("targets", []):
        asset = actor["asset"]
        if "path" in asset:
            asset["path"] = str((ROOT / asset["path"]).resolve())
    destination.mkdir()
    (destination / "scenario.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def run(current: Path, baseline: Path, output: Path, variant: str) -> dict:
    import mitsuba as mi

    mi.set_variant(variant)
    print("Mitsuba variant:", mi.variant(), flush=True)
    from generator import build_simulation_config, perform_pipeline
    from generator.core.sensing.acquisition import resolve_sensing_execution_values
    from generator.core.services import raytracing_service
    from shared.extensions.sensing import canonical_sensing_hash, finalize_sensing_v4_publication
    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.frames.providers import Hdf5Provider
    from shared.scenarios import load_scenario_configuration
    from shared.sensing.products import materialize_product_specs

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, path in (("cfr_ifft", current), ("point_bin_floor", baseline)):
        stage_configuration(path, output / name)
    primary_scenario = load_scenario_configuration(output / "cfr_ifft")
    primary_sim = build_simulation_config(primary_scenario)
    baseline_sim = build_simulation_config(load_scenario_configuration(output / "point_bin_floor"))
    # Only sensing settings may differ in a controlled shared-path comparison.
    primary_yaml = yaml.safe_load((output / "cfr_ifft/scenario.yaml").read_text())
    baseline_yaml = yaml.safe_load((output / "point_bin_floor/scenario.yaml").read_text())
    primary_yaml.pop("sensing")
    baseline_yaml.pop("sensing")
    if primary_yaml != baseline_yaml:
        raise ValueError("Shared-path comparison requires identical scene/acquisition settings")
    assert primary_sim.sensing.range_response.model == "cfr_ifft"
    assert baseline_sim.sensing.range_response.model == "point_bin_floor"
    factory = raytracing_service._load_sensing_processor_factory
    create_processor = factory()
    original_rt = raytracing_service.compute_ray_tracing_step
    original_cir = raytracing_service.RayTracingService._compute_cir_stack
    records, references = [], {}
    timings = {
        key: []
        for key in (
            "scene_update_and_rt_s",
            "cir_extraction_s",
            "point_processing_s",
            "cfr_processing_s",
        )
    }
    (output / "path_inputs").mkdir()

    def timed(name, function, *args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        timings[name].append(perf_counter() - start)
        return result

    def fanout(config, carrier_frequency_hz, execution_config):
        primary = create_processor(config, carrier_frequency_hz, execution_config)
        reference_execution = resolve_sensing_execution_values(
            baseline_sim.sensing,
            cir_time_steps=primary_sim.cir_time_steps,
            cir_sampling_frequency_hz=primary_sim.cir_sampling_frequency_hz,
            frame_dt_s=primary_sim.duration / (primary_sim.num_steps - 1),
            rx_elements=execution_config.topology.rx_elements,
        )
        reference = create_processor(
            baseline_sim.sensing, carrier_frequency_hz, reference_execution
        )

        class Fanout:
            def process(self, simulation_objects, frame_idx, **kwargs):
                a, tau = kwargs["cir_stack"]
                digest = hashlib.sha256()
                for array in (a, tau):
                    array = np.asarray(array)
                    digest.update(str((array.shape, array.dtype.str)).encode())
                    digest.update(array.tobytes())
                digest.update(float(kwargs["source_time_s"]).hex().encode())
                identity = digest.hexdigest()
                np.savez_compressed(
                    output / "path_inputs" / f"{frame_idx}.npz",
                    a=a,
                    tau=tau,
                    time_s=kwargs["source_time_s"],
                )
                point = timed(
                    "point_processing_s", reference.process, simulation_objects, frame_idx, **kwargs
                )
                cfr = timed(
                    "cfr_processing_s", primary.process, simulation_objects, frame_idx, **kwargs
                )
                for block in (point, cfr):
                    block["shared_path_input_sha256"] = identity
                references[frame_idx] = point
                records.append(
                    {
                        "source_step": frame_idx,
                        "sha256": identity,
                        "coefficient_shape": list(a.shape),
                        "delay_shape": list(tau.shape),
                        "valid_paths": int(np.count_nonzero(np.asarray(tau) >= 0)),
                        "point_stages_ms": point.get("timing", {}).get("stage_profile_ms"),
                        "cfr_stages_ms": cfr.get("timing", {}).get("stage_profile_ms"),
                    }
                )
                print(
                    "CPI",
                    frame_idx,
                    "detections:",
                    len(point.get("detections", [])),
                    "->",
                    len(cfr.get("detections", [])),
                    flush=True,
                )
                return cfr

        return Fanout()

    raytracing_service._load_sensing_processor_factory = lambda: fanout
    raytracing_service.compute_ray_tracing_step = lambda *a, **kw: timed(
        "scene_update_and_rt_s", original_rt, *a, **kw
    )
    raytracing_service.RayTracingService._compute_cir_stack = lambda *a, **kw: timed(
        "cir_extraction_s", original_cir, *a, **kw
    )
    start = perf_counter()
    try:
        perform_pipeline(
            simulation_config=primary_sim,
            scenario_configuration=primary_scenario,
            show_progress=False,
        )
    finally:
        raytracing_service._load_sensing_processor_factory = factory
        raytracing_service.compute_ray_tracing_step = original_rt
        raytracing_service.RayTracingService._compute_cir_stack = original_cir
    combined_runtime = perf_counter() - start
    provider = Hdf5Provider(output / "cfr_ifft")
    writer = FrameSetWriter.create_new(
        output / "point_bin_floor/frames", chunk_size=64, compression="lzf"
    )
    run_id = str(uuid4())
    try:
        for step in provider.list_frames():
            frame = provider.load_frame(step)
            original = frame.sensing
            if original and original.get("temporal", {}).get("result"):
                point = references[int(original["source_frame_step"])]
                for key in (
                    "target_catalog",
                    "target_state",
                    "truth",
                    "gt",
                    "gt_sensing",
                    "gt_state",
                ):
                    if key in original:
                        point[key] = deepcopy(original[key])
                point["schema_version"] = 4
                point["temporal"] = deepcopy(original["temporal"])
                point["temporal"]["result"].update(
                    processing_run_id=run_id,
                    processing_recipe_id=canonical_sensing_hash(
                        point["config_resolved"]["processing_recipe"]
                    ),
                )
                materialize_product_specs(point)
                finalize_sensing_v4_publication(point)
                frame = replace(frame, sensing=point)
            writer.append(frame)
        writer.finalize(
            provenance={
                "experiment": "curriculum_shared_path_comparison",
                "source": str(output / "cfr_ifft"),
            }
        )
    except BaseException:
        writer.abort()
        raise
    finally:
        provider.close()
    report = {
        "variant": mi.variant(),
        "inputs": {"current": str(current), "baseline": str(baseline)},
        "input_sha256": {
            "current": hashlib.sha256(current.read_bytes()).hexdigest(),
            "baseline": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        },
        "acquisitions": records,
        "timings": timings,
        "combined_generation_s": combined_runtime,
        "combined_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "timing_scope": "One native acquisition, two processors; includes path capture and primary serialization. RSS includes both processors. Not independent per-model total-runtime estimates.",
        "frame_bytes": {
            name: sum(
                p.stat().st_size for p in (output / name / "frames").rglob("*") if p.is_file()
            )
            for name in ("point_bin_floor", "cfr_ifft")
        },
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", type=Path, required=True, help="Current CFR/IFFT scenario YAML"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Preserved point-bin scenario YAML"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh directory; never overwrites recordings"
    )
    parser.add_argument(
        "--variant",
        choices=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
        default="cuda_ad_mono_polarized",
    )
    args = parser.parse_args()
    run(args.scenario, args.baseline, args.output, args.variant)


if __name__ == "__main__":
    main()
