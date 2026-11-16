"""Fuse saved monostatic detections using an explicit synchronized experiment.

Run ``python -m shared.sensing.fusion.experiment experiment.json --output-dir ...``.
Original packed frames remain immutable. This consumer writes a separate versioned
JSON result, not a single-pair frame falsely labelled as a multi-sensor product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from shared.extensions.sensing import assert_sensing_contract, assert_supported_sensing_schema
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.sensing.fusion.core import FusionConfig, MonostaticFusion, Sensor
from shared.sensing.fusion.quality import apply_quality, native_observation
from shared.sensing.fusion.result import (
    code_identity,
    validate_alignment,
    write_result,
)
from shared.sensing.loader import _open_provider


def fingerprints(directory: Path) -> dict[str, str]:
    """Stream hashes of the authoritative frame chunks and manifests."""
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in {".json", ".h5"}:
            with path.open("rb") as stream:
                result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def load_sensor(source: Path, sensor_id: str, *, result_frame_ids=None) -> dict:
    """Read raw detections and exact source-epoch poses; keep GT separate.

    A caller with an authored acquisition index may select result storage
    frames. Source hashing and exact scene-epoch binding still cover the native
    recording; omitted frames are not treated as empty detection epochs.
    Tracked sources must retain the complete epoch-bound pre-tracking reports;
    confirmed outputs and saved Cartesian states are never fusion observations.
    """
    directory, manifest, provider = _open_provider(source)
    started = perf_counter()
    before = fingerprints(directory)
    hashing_seconds = perf_counter() - started
    read_started = perf_counter()
    epochs, truth, lineage = [], [], []
    sensor = None
    stream_identity = None
    request = FrameReadRequest(
        components=frozenset({FrameComponent.DEVICES, FrameComponent.PROVENANCE})
    )
    # The packed provider always supplies metadata. Selecting its descriptor
    # dictionary avoids materializing the much larger CIR/RD/angle arrays.
    detection_request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING, FrameComponent.PROVENANCE}),
        sensing_products=frozenset({"products"}),
    )
    try:
        frame_ids = manifest.frame_ids if result_frame_ids is None else tuple(result_frame_ids)
        if result_frame_ids is not None:
            available = set(manifest.frame_ids)
            if (
                not frame_ids
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or i not in available
                    for i in frame_ids
                )
                or len(set(frame_ids)) != len(frame_ids)
            ):
                raise ValueError("Result storage frame IDs must be distinct members of the source")
        for projection in provider.iter_frame_projections(frame_ids, detection_request):
            sensing = projection.frame.sensing
            if not sensing or not sensing.get("temporal", {}).get("result"):
                continue
            assert_supported_sensing_schema(sensing)
            assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
            if "detections" not in sensing:
                raise ValueError(
                    "Fusion requires retained detections, including explicit empty lists"
                )
            frame_id = int(projection.frame.frame_index)
            temporal, cfg = sensing["temporal"], sensing["config_resolved"]
            acquisition = temporal["acquisition"]
            clock = temporal["clock"]
            if cfg["range_mode"] != "monostatic":
                raise ValueError("Fusion consumes monostatic detections")
            tracker = sensing.get("detection_filter_metadata", {}).get("tracker", {})
            tracking_enabled = cfg["track_confirmation_enabled"] or tracker.get("enabled", False)
            detection_collection = "detections"
            if tracking_enabled:
                # Confirmation can suppress complete epochs and modify the output
                # collection. Fusion needs every retained angular observation,
                # not local track states or optional decision-record inputs.
                detection_collection = "detections_before_tracking"
                if not isinstance(sensing.get(detection_collection), list) or not isinstance(
                    sensing.get("detections_before_tracking_metadata"), Mapping
                ):
                    raise ValueError(
                        "Fusion requires explicit detections_before_tracking with metadata "
                        "when the source tracker is enabled"
                    )
            if (
                clock["domain"] != "scenario"
                or clock["unit"] != "s"
                or clock["origin"] != "scenario_start"
                or clock["origin_time_s"] != 0
            ):
                raise ValueError("Baseline requires scenario-start clocks in seconds")
            array = sensing["rx_array"]
            if min(array["rows"], array["columns"]) <= 1:
                raise ValueError(
                    "3-D fusion requires resolved azimuth and elevation (a planar RX array)"
                )
            identity = (clock["clock_id"], acquisition["pair_id"])
            if stream_identity is not None and identity != stream_identity:
                raise ValueError("A sensor source must contain one clock/pair stream")
            stream_identity = identity
            time_s = float(temporal["result"]["reference_time_s"])
            # Use the result's display binding, never GT to find a sensor pose.
            binding = temporal["display_binding"]
            scene = provider.load_frame_projection(binding["scene_frame_id"], request).frame
            if scene.timestamp_s is None or not np.isclose(
                scene.timestamp_s, time_s, rtol=0, atol=1e-9
            ):
                raise ValueError("Sensor pose must be stored at the exact result epoch")
            rx_name, tx_name = acquisition["rx_sensor_id"], acquisition["tx_sensor_id"]
            if scene.rx_names.count(rx_name) != 1 or scene.tx_names.count(tx_name) != 1:
                raise ValueError("Missing or ambiguous sensor identity in scene")
            ri, ti = scene.rx_names.index(rx_name), scene.tx_names.index(tx_name)
            if not np.allclose(scene.rx_positions[ri], scene.tx_positions[ti], rtol=0, atol=1e-6):
                raise ValueError("Monostatic fusion requires colocated TX/RX")
            current = Sensor(
                sensor_id, tuple(scene.rx_positions[ri]), tuple(scene.rx_orientations[ri])
            )
            if sensor is not None and current != sensor:
                raise ValueError("This baseline requires stationary sensors")
            sensor = current
            observations = []
            for i, detection in enumerate(sensing[detection_collection]):
                observations.append(
                    native_observation(
                        sensor_id,
                        frame_id,
                        i,
                        time_s,
                        detection,
                        sensing,
                        {
                            "frame_set_id": manifest.frame_set_id,
                            "generation_id": manifest.generation_id,
                            "scene_frame_id": binding["scene_frame_id"],
                            "acquisition_id": acquisition["acquisition_id"],
                            "processing_recipe_id": temporal["result"]["processing_recipe_id"],
                            "processing_run_id": temporal["result"]["processing_run_id"],
                            **(
                                {"detection_collection": detection_collection}
                                if tracking_enabled
                                else {}
                            ),
                        },
                    )
                )
            epochs.append({"time_s": time_s, "observations": observations})
            lineage.append(
                {
                    "time_s": time_s,
                    "storage_frame_id": frame_id,
                    "scene_frame_id": binding["scene_frame_id"],
                    "acquisition_id": acquisition["acquisition_id"],
                    "processing_recipe_id": temporal["result"]["processing_recipe_id"],
                    **({"detection_collection": detection_collection} if tracking_enabled else {}),
                }
            )
            # This separate list is only consumed after estimation, by evaluation.
            truth_epoch = temporal.get("truth", {}).get("reference_time_s")
            truth.append(
                {
                    "time_s": time_s,
                    "available": (
                        truth_epoch == time_s
                        and "truth" in sensing
                        and all(
                            t["valid"]
                            for t in sensing.get("truth", {}).get("targets", [])
                            if t["lifecycle"] == "active"
                        )
                    ),
                    "targets": (
                        [
                            {
                                "target_id": t["target_id"],
                                "position_m": t["position_m"],
                                "velocity_m_s": t["velocity_m_s"],
                            }
                            for t in sensing.get("truth", {}).get("targets", [])
                            if t["valid"] and t["lifecycle"] == "active"
                        ]
                        if truth_epoch == time_s
                        else []
                    ),
                }
            )
        if not epochs:
            raise ValueError("No saved sensing results")
        times = np.array([e["time_s"] for e in epochs])
        if not np.all(np.diff(times) > 0):
            raise ValueError("Sensor result times must increase strictly")
        loading_seconds = perf_counter() - read_started
        check_started = perf_counter()
        if fingerprints(directory) != before:
            raise RuntimeError("Source changed while being read")
        hashing_seconds += perf_counter() - check_started
        return {
            "sensor": sensor,
            "epochs": epochs,
            "truth": truth,
            "directory": directory,
            "performance": {"hashing_seconds": hashing_seconds, "loading_seconds": loading_seconds},
            "source": {
                "frame_set_id": manifest.frame_set_id,
                "generation_id": manifest.generation_id,
                "native_clock_id": stream_identity[0],
                "pair_id": stream_identity[1],
                "hashes": before,
                "lineage": lineage,
            },
        }
    finally:
        provider.close()


def run_experiment(recipe_file: Path, output: Path, *, figures=True) -> dict:
    """Save a verified replay; spatial campaigns can skip per-trial figure exports."""
    recipe_file, output = Path(recipe_file).resolve(), Path(output).resolve()
    recipe = json.loads(recipe_file.read_text())
    clock = recipe["clock"]
    if (
        recipe["schema_version"] not in (1, 2)
        or clock.get("assumption") != "synchronized"
        or clock.get("unit") != "s"
        or not clock.get("id")
    ):
        raise ValueError("Declare a named synchronized common clock in seconds")
    specs = recipe["sensors"]
    if not specs or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) or name == "fused" for name in specs
    ):
        raise ValueError("Supply at least one named sensor source (fused is reserved)")
    if any(spec.get("time_offset_s") != 0 for spec in specs.values()):
        raise ValueError("This synchronized baseline requires explicit zero time offsets")
    world_frame = validate_alignment(recipe)
    compatibility = (
        {
            "confirmation_mode": "legacy_cumulative",
            "gate_mode": "legacy_numeric",
            "quality_policy": "legacy",
            "birth_policy": "legacy_distance_velocity",
        }
        if recipe["schema_version"] == 1
        else {}
    )
    config = FusionConfig(**{**compatibility, **recipe.get("fusion", {})})
    evaluation_gate = float(recipe.get("evaluation", {}).get("position_gate_m", 0.75))
    if not np.isfinite(evaluation_gate) or evaluation_gate <= 0:
        raise ValueError("Evaluation position gate must be positive and finite")
    sources = {
        name: (recipe_file.parent / spec["source"]).resolve() for name, spec in specs.items()
    }
    if any(
        output.is_relative_to(source) or source.is_relative_to(output)
        for source in sources.values()
    ):
        raise ValueError("Fusion output and source directories must not contain each other")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    loaded = {name: load_sensor(source, name) for name, source in sources.items()}
    for name, data in loaded.items():
        if "measurement_covariance" in specs[name]:
            data["sensor"] = replace(
                data["sensor"],
                measurement_covariance=tuple(
                    tuple(row) for row in specs[name]["measurement_covariance"]
                ),
            )
        for epoch in data["epochs"]:
            epoch["observations"] = [apply_quality(o, config) for o in epoch["observations"]]
    if len({s["source"]["frame_set_id"] for s in loaded.values()}) != len(loaded):
        raise ValueError("Do not double-count a recording under two sensor names")
    first = next(iter(loaded.values()))
    times = np.array([e["time_s"] for e in first["epochs"]])
    for data in loaded.values():
        other = np.array([e["time_s"] for e in data["epochs"]])
        if other.shape != times.shape or not np.allclose(other, times, rtol=0, atol=1e-9):
            raise ValueError(
                "Synchronized fusion requires equal result schedules, including empty detections"
            )
    histories, timings = {}, {}
    for mode in (*sorted(loaded), "fused"):
        names = sorted(loaded) if mode == "fused" else [mode]
        processor = MonostaticFusion([loaded[n]["sensor"] for n in names], config)
        history = []
        start = perf_counter()
        for i, time_s in enumerate(times):
            reports = [o for n in names for o in loaded[n]["epochs"][i]["observations"]]
            history.append(processor.step(float(time_s), reports))
        timings[mode] = perf_counter() - start
        histories[mode] = history
    # Check truth only after all estimation. Never feed identity or kinematics
    # from this evaluation channel back into birth, association or tracking.
    truth = first["truth"]
    for data in loaded.values():
        if data["truth"] != truth:
            raise ValueError(
                "Evaluation truth differs between sensor recordings; verify the common scene/motion"
            )
    verification_started = perf_counter()
    unchanged = all(fingerprints(d["directory"]) == d["source"]["hashes"] for d in loaded.values())
    if not unchanged:
        raise RuntimeError("Source changed during fusion")
    result = {
        "schema_version": 2,
        "run_id": str(uuid.uuid4()),
        "world_frame": world_frame,
        "code_identity": code_identity(Path(__file__).resolve().parents[3]),
        "config_sha256": hashlib.sha256(
            json.dumps(
                {
                    "fusion": asdict(config),
                    "sensors": specs,
                    "clock": clock,
                    "world_frame": world_frame,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "kind": "synchronized_monostatic_detection_fusion",
        "clock": clock,
        "coordinate_frame": "world_cartesian_m",
        "state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "config": asdict(config),
        "evaluation_config": {"position_gate_m": evaluation_gate},
        "sources_unchanged": unchanged,
        "sources": {
            n: {
                **d["source"],
                "sensor": asdict(d["sensor"]),
                "time_offset_s": 0.0,
                "source": os.path.relpath(sources[n], output),
                "world_frame_id": world_frame["id"],
                "alignment": "explicit_shared_world",
            }
            for n, d in loaded.items()
        },
        "histories": histories,
        "truth_for_evaluation": truth,
        "observations": {
            n: [
                {
                    "time_s": e["time_s"],
                    "detections": [
                        {
                            **asdict(o),
                            "world_position_m": (
                                o.world_position(d["sensor"]).tolist() if o.has_position else None
                            ),
                        }
                        for o in e["observations"]
                    ],
                }
                for e in d["epochs"]
            ]
            for n, d in loaded.items()
        },
        "processing_seconds": timings,
        "performance": {
            "sources": {n: d.get("performance", {}) for n, d in loaded.items()},
            "final_source_verification_seconds": perf_counter() - verification_started,
        },
        "timing_scope": "single CPU pass; excludes loading, detection, evaluation and plotting; not a benchmark",
    }
    saved_recipe = {
        **recipe,
        "fusion": asdict(config),
        "evaluation": {"position_gate_m": evaluation_gate},
        "sensors": {
            name: {**specs[name], "source": os.path.relpath(source, output)}
            for name, source in sources.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "experiment.json").write_text(json.dumps(saved_recipe, indent=2) + "\n")
        start = perf_counter()
        reader = write_result(result, staging)
        serialization_seconds = perf_counter() - start
        evaluation = None
        if figures:
            from shared.sensing.fusion.figures import generate_fusion_figures

            evaluation = generate_fusion_figures(reader, staging)
        (staging / "performance.json").write_text(
            json.dumps(
                {
                    **result["performance"],
                    "fusion_seconds": timings,
                    "serialization_and_validation_seconds": serialization_seconds,
                },
                indent=2,
            )
            + "\n"
        )
        # Destination must be new; every bundle member is complete before publication.
        if output.exists():
            raise FileExistsError(f"Choose a new output directory: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"output": str(output), "sources_unchanged": unchanged, "evaluation": evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(run_experiment(args.experiment, args.output_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
