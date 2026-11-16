"""Shared native acquisition and verified collection for authored volume examples.

Scenario scripts own geometry and trial configuration. These helpers preserve
completed recordings and score one target per CPI without changing detection.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scenarios.sensing.volume_progress import VolumeProgress
from scenarios.sensing.volume_storage import keeps_native, release_native_frames, retention_mode
from shared.sensing.fusion.experiment import load_sensor
from shared.sensing.fusion.result import code_identity
from shared.sensing.volume.evaluation import saved_reports


def write_json(path, value):
    """Publish one complete metadata file without exposing partial JSON."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_trials(root, worker, workers, limit):
    """Run a disjoint worker partition; restart only unfinished empty trials."""
    spec = json.loads((root / "volume_recipe.json").read_text())
    pending = [t for t in spec["trials"] if t["status"] == "pending"][worker::workers]
    selected = pending[:limit] if limit is not None else pending
    completed = sum((root / t["native"]).parent.joinpath("outcome.json").exists() for t in selected)
    excluded = sum(t["status"] == "excluded_geometry" for t in spec["trials"])
    label = spec.get("condition", root.name)
    if workers > 1:
        label = f"{label} / worker {worker + 1}/{workers}"
    print(
        f"\nVolume: {label}\n"
        f"  {len(selected):,} selected trials | {completed:,} already saved | "
        f"{len(selected) - completed:,} remaining | {excluded:,} geometry exclusions\n"
        f"  Output: {root}",
        flush=True,
    )
    with VolumeProgress(label, len(selected), completed=completed) as progress:
        for trial in selected:
            native = root / trial["native"]
            if (native.parent / "outcome.json").exists():
                release_native_frames(root, spec, trial)
                continue
            progress.start_trial(trial, spec.get("cpi_samples", 128))
            outcome = _acquire_trial(root, spec, trial, progress)
            progress.complete_trial(outcome)


def _acquire_trial(root, spec, trial, progress):
    """Generate, verify and save one trial before advancing campaign progress."""
    from generator.__main__ import _run_from_yaml

    native = root / trial["native"]
    result_path = native.parent / "outcome.json"
    if (
        hashlib.sha256((native / "scenario.yaml").read_bytes()).hexdigest()
        != trial["configuration_sha256"]
    ):
        raise ValueError(f"Trial configuration changed: {native}")
    start = time.perf_counter()
    if not (native / "frames").exists():
        _run_from_yaml(native, on_step_complete=progress.step_completed)
    progress.set_stage("Verifying frames / scoring detections")
    manifest = json.loads((native / "frames/frames_manifest.json").read_text())
    if manifest.get("frame_ids") != list(range(spec.get("cpi_samples", 128))):
        raise ValueError(f"Incomplete native acquisition: {native}")
    data = load_sensor(native, spec.get("sensor_id", "Port"))
    if len(data["epochs"]) != 1 or not data["truth"][0]["available"]:
        raise ValueError("Expected exactly one complete CPI and valid independent truth")
    truth = data["truth"][0]["targets"][0]
    if not np.allclose(truth["position_m"], trial["position_m"], atol=1e-5) or not np.allclose(
        truth["velocity_m_s"], trial["velocity_m_s"], atol=1e-5
    ):
        raise ValueError("Generated truth disagrees with the frozen trial")
    detections = data["epochs"][0]["observations"]
    distances = [
        float(np.linalg.norm(o.world_position(data["sensor"]) - truth["position_m"]))
        for o in detections
    ]
    match = min(range(len(distances)), key=distances.__getitem__) if distances else None
    hit = match is not None and distances[match] <= spec["position_gate_m"]
    delta = np.subtract(trial["position_m"], spec["sensor_position_m"])
    radial = -float(np.dot(trial["velocity_m_s"], delta / np.linalg.norm(delta)))
    outcome = {
        **trial,
        "status": "measured",
        "hit": bool(hit),
        "tp": int(hit),
        "fp": len(detections) - int(hit),
        "fn": int(not hit),
        "radial_closing_velocity_m_s": radial,
        "matched_error_m": distances[match] if hit else None,
        "matched_detection_id": detections[match].detection_id if hit else None,
        "nearest_report_distance_m": distances[match] if match is not None else None,
        "source_hashes": data["source"]["hashes"],
        "reports": saved_reports(data),
        "wall_seconds": time.perf_counter() - start,
    }
    if retention_mode(spec) == "volume":
        outcome["native_evidence"] = "included" if keeps_native(spec, trial) else "not_included"
    write_json(result_path, outcome)
    release_native_frames(root, spec, trial)
    return outcome


def collect(root):
    """Publish partial progress explicitly; missing trials never become misses."""
    spec = json.loads((root / "volume_recipe.json").read_text())
    rows = []
    for trial in spec["trials"]:
        path = root / trial["native"] / "../outcome.json" if "native" in trial else None
        row = json.loads(path.read_text()) if path is not None and path.is_file() else trial
        for field in (name for name in trial if name != "status"):
            if row.get(field) != trial.get(field):
                raise ValueError(f"Outcome disagrees with frozen trial {trial['id']}: {field}")
        rows.append(row)
    data = {
        **spec,
        "trials": rows,
        "world_frame": spec["world_frame"],
        "scene_configuration_sha256": hashlib.sha256(
            (root / spec["scene_scenario"]).read_bytes()
        ).hexdigest(),
        "collector_identity": code_identity(Path(__file__).resolve().parents[2]),
        "collector_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    from shared.sensing.volume.reader import SensingVolume

    candidate = root / "sensing_volume.pending.json"
    write_json(candidate, data)
    reader = SensingVolume(candidate)
    reader.verify_scene()
    for key, trial in reader.cells.items():
        if trial["status"] == "measured":
            if trial.get("native_evidence") != "not_included":
                reader.verify_native(key)
            elif (
                hashlib.sha256((root / trial["native"] / "scenario.yaml").read_bytes()).hexdigest()
                != trial["configuration_sha256"]
            ):
                raise ValueError("Compact trial configuration changed")
    if all("reports" in t for t in rows if t["status"] == "measured"):
        from shared.sensing.volume.evaluation import VolumeEvaluation

        # Full and compact retention must reproduce the same saved associations.
        VolumeEvaluation(reader, {t["id"]: t["reports"] for t in rows if t["status"] == "measured"})
    candidate.replace(root / "sensing_volume.json")
    print(
        f"Collected {sum(t['status']=='measured' for t in rows)}/{len(rows)} cells/headings",
        flush=True,
    )
