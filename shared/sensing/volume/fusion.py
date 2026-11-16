"""Bind a spatial trial to a saved fusion replay, without inventing radar measurements."""

from __future__ import annotations

import hashlib

import numpy as np

BUNDLE_FILES = ("fusion_results.json", "fusion_decisions.jsonl", "experiment.json")


def matched_contributions(model, heading, snapshot):
    """Map the saved contributors of the currently matched estimate in each cell.

    Extra estimates never change the matched estimate's attribution. Multiple
    contributors are grouped together for volumes with more than two sensors;
    each selected estimate still retains its exact sensor list for inspection.
    """
    reader = model.reader
    names = tuple(reader.metadata["sensor_positions_m"])
    labels = [f"{name.title()} only" for name in names]
    labels += [" + ".join(n.title() for n in names) if len(names) == 2 else "Multiple sensors"]
    labels += ["No matched estimate", "Contributors not recorded"]
    single_colors = ("#5797cb", "#e49e56", "#a68bcc", "#d57c9d", "#71a7a5")
    colors = [single_colors[i % len(single_colors)] for i in range(len(names))]
    colors += ["#47a883", "#777777", "#bbbbbb"]
    no_match, unknown = len(names) + 1, len(names) + 2
    values = np.full(reader.shape, np.nan, dtype=np.float32)
    for key, trial in reader.cells.items():
        if key[0] != heading or trial["status"] != "measured":
            continue
        cell = key[1:]
        selected = int(snapshot["selected_report"][cell])
        if selected < 0:
            values[cell] = no_match
            continue
        report_id = model.report_ids[selected]
        report = next(r for r in model.rows[trial["id"]] if r["detection_id"] == report_id)
        support = report.get("sensor_support")
        if not isinstance(support, list) or not support or any(n not in names for n in support):
            values[cell] = unknown
        elif len(set(support)) == 1:
            values[cell] = names.index(support[0])
        else:
            values[cell] = len(names)
    if not np.any(values == unknown):
        labels.pop()
        colors.pop()
    return dict(values=values, labels=tuple(labels), colors=tuple(colors))


def bundle_hashes(directory):
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in BUNDLE_FILES
    }


def fused_outcome(result, trial, gate):
    """Evaluate current fused estimates only after truth-independent processing."""
    history = result["histories"]["fused"]
    truth = result["truth_for_evaluation"]
    if len(history) != 1 or len(truth) != 1 or not truth[0]["available"]:
        raise ValueError("A fusion volume cell requires one epoch and independent truth")
    targets = truth[0]["targets"]
    if (
        len(targets) != 1
        or not np.allclose(targets[0]["position_m"], trial["position_m"], atol=1e-5)
        or not np.allclose(targets[0]["velocity_m_s"], trial["velocity_m_s"], atol=1e-5)
    ):
        raise ValueError("Fusion truth disagrees with its authored spatial trial")
    reports = [
        {
            "detection_id": f"track:{track['track_id']}",
            "position_m": track["position_m"],
            "velocity_m_s": track["velocity_m_s"],
            "sensor_support": track["current_sensor_support"],
            "range_m": None,
            "closing_velocity_m_s": None,
        }
        for track in history[0]["tracks"]
        if track["confirmed"] and not track["predicted_only"]
    ]
    distances = [
        float(np.linalg.norm(np.subtract(r["position_m"], trial["position_m"]))) for r in reports
    ]
    nearest = min(range(len(reports)), key=distances.__getitem__) if reports else None
    hit = nearest is not None and distances[nearest] <= gate
    directory = result.path.parent
    return {
        **trial,
        "status": "measured",
        "hit": bool(hit),
        "tp": int(hit),
        "fn": int(not hit),
        "fp": len(reports) - int(hit),
        "radial_closing_velocity_m_s": None,
        "matched_error_m": distances[nearest] if hit else None,
        "matched_detection_id": reports[nearest]["detection_id"] if hit else None,
        "nearest_report_distance_m": distances[nearest] if nearest is not None else None,
        "reports": reports,
        "source_hashes": bundle_hashes(directory),
        "configuration_sha256": hashlib.sha256(
            (directory / "experiment.json").read_bytes()
        ).hexdigest(),
    }


def validate_fusion_metadata(data):
    sensors = data.get("sensor_positions_m", {})
    reference = data.get("reference_sensor_id")
    if (
        not isinstance(sensors, dict)
        or len(sensors) < 2
        or not isinstance(reference, str)
        or reference not in sensors
        or any(not isinstance(name, str) or not name for name in sensors)
    ):
        raise ValueError("A fusion volume requires named sensors and a display reference")
    for position in sensors.values():
        value = np.asarray(position, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Invalid fusion sensor position")
    if not np.array_equal(sensors[reference], data["sensor_position_m"]):
        raise ValueError("Fusion display reference disagrees with the declared sensor")


def verify_fusion_trial(reader, key):
    from shared.sensing.fusion.result import FusionResult

    trial = reader.cells[key]
    source = (reader.path.parent / trial["native"]).resolve()
    if bundle_hashes(source) != trial["source_hashes"]:
        raise ValueError("Fusion cell replay hash mismatch")
    result = FusionResult(source)
    if not all(result.verify_sources().values()):
        raise ValueError("Fusion cell native source hash mismatch")
    if result["config"] != reader.metadata.get("fusion_config"):
        raise ValueError("Fusion replay settings disagree with the volume")
    if result["world_frame"] != {**reader.metadata["world_frame"], "angle_unit": "rad"}:
        raise ValueError("Fusion replay world frame disagrees with the volume")
    if set(result["sources"]) != set(reader.metadata["sensor_positions_m"]):
        raise ValueError("Fusion replay sensor inventory disagrees with the volume")
    for name, position in reader.metadata["sensor_positions_m"].items():
        if not np.allclose(result["sources"][name]["sensor"]["position_m"], position):
            raise ValueError("Fusion replay sensor pose disagrees with the volume")
    expected = fused_outcome(result, trial, reader.metadata["position_gate_m"])
    for field in (
        "hit",
        "tp",
        "fn",
        "fp",
        "reports",
        "matched_detection_id",
        "matched_error_m",
        "configuration_sha256",
    ):
        if trial.get(field) != expected[field]:
            raise ValueError(f"Saved fusion outcome disagrees with its replay: {field}")
    return source
