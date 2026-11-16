"""Compare saved single-target volume reports without rerunning a detector.

Definitions bind labelled sources to manifest hashes and an authored motion.
Calculation is independent of Qt, rendering, and the RF runtime. Matplotlib is
imported only by the optional plot/export operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from shared.sensing.volume.evaluation import VolumeEvaluation, load_evaluation
from shared.sensing.volume.reader import MAX_MANIFEST_BYTES, SensingVolume

MAX_SOURCES = 32


def source_definition(path, *, label=None, heading=None):
    """Bind a volume and selected authored motion to its current content hash."""
    reader = SensingVolume(path)
    if heading is None:
        if len(reader.headings) != 1:
            raise ValueError("Select a heading explicitly for a multi-heading volume")
        heading = reader.headings[0]["name"]
    _heading_index(reader, heading)
    return dict(
        path=str(reader.path),
        sha256=reader.identity,
        label=label or reader.metadata.get("condition") or reader.path.parent.name,
        heading=heading,
    )


def _heading_index(reader, name):
    matches = [i for i, h in enumerate(reader.headings) if h["name"] == name]
    if len(matches) != 1:
        raise ValueError("Selected heading must identify exactly one authored motion")
    return matches[0]


def validate_definition(definition):
    """Validate bounded source selection and explicit scoring/support policies."""
    if definition.get("schema_version") != 1:
        raise ValueError("Unsupported volume comparison definition")
    sources = definition.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError(f"Choose between 1 and {MAX_SOURCES} volumes")
    labels = []
    for source in sources:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key].strip()
            for key in ("path", "sha256", "label", "heading")
        ):
            raise ValueError("Each source needs a path, SHA-256, label and heading")
        if len(source["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in source["sha256"]
        ):
            raise ValueError("Source SHA-256 must be a lowercase hexadecimal digest")
        labels.append(source["label"])
    if len(set(labels)) != len(labels):
        raise ValueError("Comparison source labels must be unique")
    gate = definition.get("position_gate_m")
    if (
        isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or not np.isfinite(gate)
        or gate <= 0
    ):
        raise ValueError("Position gate must be finite and positive")
    if definition.get("support") not in {"common_measured", "per_source"}:
        raise ValueError("Choose common_measured or per_source support")


def load_definition(path):
    """Resolve source paths relative to a saved comparison definition."""
    path = Path(path).resolve()
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Comparison definition exceeds 64 MiB")
    definition = json.loads(path.read_bytes())
    validate_definition(definition)
    for source in definition["sources"]:
        source["path"] = str((path.parent / source["path"]).resolve())
    return definition


def save_definition(path, definition):
    """Save relocatable paths and exact source bindings, never source recordings."""
    validate_definition(definition)
    path = Path(path).resolve()
    copy = json.loads(json.dumps(definition))
    for source in copy["sources"]:
        try:
            source["path"] = os.path.relpath(Path(source["path"]).resolve(), path.parent)
        except ValueError:
            # Different Windows drive letters cannot be represented relatively.
            source["path"] = str(Path(source["path"]).resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(copy, indent=2, allow_nan=False) + "\n")


def compare_volumes(definition):
    """Rematch saved reports at one inclusive 3D gate and summarize each source.

    Common support is the intersection of measured cells on identical grids and
    selected velocity vectors. Per-source support permits different grids in the
    same world frame. Pending/excluded cells never become misses. A matched cell
    contributes one report; every other saved report remains an extra.
    """
    validate_definition(definition)
    loaded = []
    for source in definition["sources"]:
        reader = SensingVolume(source["path"])
        if reader.identity != source["sha256"]:
            raise ValueError(f"Volume identity changed: {source['label']}")
        heading = _heading_index(reader, source["heading"])
        trials = [t for t in reader.cells.values() if t["status"] == "measured"]
        model = (
            VolumeEvaluation(reader, {t["id"]: t["reports"] for t in trials})
            if all("reports" in t for t in trials)
            else load_evaluation(reader)
        )
        snapshot = model.evaluate(heading, position_gate_m=definition["position_gate_m"])
        loaded.append((source, reader, model, heading, snapshot))
    base, base_heading = loaded[0][1], loaded[0][3]
    for _, reader, _, heading, _ in loaded[1:]:
        if reader.metadata["world_frame"] != base.metadata["world_frame"]:
            raise ValueError("Comparisons require the same declared world frame")
        if not np.array_equal(
            reader.headings[heading]["velocity_m_s"], base.headings[base_heading]["velocity_m_s"]
        ):
            raise ValueError("Comparisons require the same selected target velocity")
        if definition["support"] == "common_measured" and any(
            not np.array_equal(reader.axes[key], base.axes[key]) for key in "xyz"
        ):
            raise ValueError("Common measured support requires identical grid axes")
    common = None
    if definition["support"] == "common_measured":
        common = np.logical_and.reduce([np.isfinite(item[4]["outcomes"]) for item in loaded])
    conditions = []
    for source, reader, model, heading, snapshot in loaded:
        measured = np.isfinite(snapshot["outcomes"])
        mask = measured if common is None else common
        hit = (snapshot["outcomes"] == 1) & mask
        counts = snapshot["report_counts"]
        errors = snapshot["nearest_error_m"][hit]
        rows = []
        for (ih, iz, iy, ix), trial in sorted(reader.cells.items()):
            if ih != heading:
                continue
            index = (iz, iy, ix)
            included = bool(mask[index])
            error = snapshot["nearest_error_m"][index]
            rows.append(
                dict(
                    trial_id=trial["id"],
                    index_zyx=list(index),
                    position_m=trial["position_m"],
                    status=trial["status"],
                    included=included,
                    report_count=int(counts[index]) if measured[index] else None,
                    matched=bool(hit[index]) if included else None,
                    nearest_error_m=float(error) if np.isfinite(error) else None,
                )
            )
        summary = dict(
            measured_locations=int(measured.sum()),
            denominator=int(mask.sum()),
            matched_locations=int(hit.sum()),
            extra_reports=int(counts[mask].sum() - hit.sum()),
            empty_outputs=int((mask & (counts == 0)).sum()),
            out_of_gate_outputs=int((mask & (counts > 0) & ~hit).sum()),
            reports_without_position=int(
                (mask & (counts > 0) & ~np.isfinite(snapshot["nearest_error_m"])).sum()
            ),
            authored_grid_locations=int(np.prod(reader.shape)),
            unknown_or_excluded_locations=int(np.prod(reader.shape) - measured.sum()),
            measured_outside_support=int((measured & ~mask).sum()),
            matched_fraction=float(hit.sum() / mask.sum()) if mask.any() else None,
            matched_rmse_m=float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            matched_median_error_m=float(np.median(errors)) if errors.size else None,
            matched_p90_error_m=float(np.quantile(errors, 0.9)) if errors.size else None,
        )
        conditions.append(
            dict(
                source=dict(source),
                summary=summary,
                matched_errors_m=np.sort(errors).tolist(),
                cells=rows,
                sensor_position_m=reader.metadata["sensor_position_m"],
                sensor_yaw_deg=reader.metadata.get("sensor_yaw_deg"),
                target_velocity_m_s=reader.headings[heading]["velocity_m_s"],
                original_acquisition_identity=reader.metadata.get("collector_identity"),
                original_volume_sha256=reader.metadata.get(
                    "original_volume_sha256", reader.identity
                ),
                report_snapshot_sha256=hashlib.sha256(
                    json.dumps(
                        model.rows, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest(),
            )
        )
    return dict(
        schema_version=1,
        kind="saved_volume_comparison",
        definition=definition,
        world_frame=base.metadata["world_frame"],
        conditions=conditions,
        units=dict(
            position="m", error="m", velocity="m/s", counts="reports or locations as labelled"
        ),
        policy=dict(
            matching="Nearest saved world-position report within inclusive Euclidean gate; at most one match per single-target location",
            support=definition["support"],
            unknown="Pending, excluded and absent cells are outside the denominator",
            detector="Saved reports unchanged; no detector replay",
            error="Conditional on matched locations; no errors imputed for misses",
        ),
    )


def comparison_figure(result, figure=None):
    """Compose the same count and conditional-error plots for GUI and exports."""
    from matplotlib.figure import Figure

    figure = figure or Figure(figsize=(10, 4.5), layout="constrained")
    figure.clear()
    left, right = figure.subplots(1, 2)
    conditions = result["conditions"]
    x = np.arange(len(conditions))
    labels = [c["source"]["label"] for c in conditions]
    for shift, field, label in (
        (-0.27, "matched_locations", "Matched locations"),
        (0, "extra_reports", "Extra reports"),
        (0.27, "empty_outputs", "Empty outputs"),
    ):
        left.bar(x + shift, [c["summary"][field] for c in conditions], 0.25, label=label)
    left.set_xticks(x, labels, rotation=18, ha="right")
    left.set_ylabel("Count")
    left.legend(fontsize=8)
    left.grid(axis="y", alpha=0.25)
    for condition in conditions:
        errors = condition["matched_errors_m"]
        if errors:
            right.step(
                errors,
                np.arange(1, len(errors) + 1) / len(errors),
                where="post",
                label=condition["source"]["label"],
            )
    right.set(
        xlabel="Matched position error (m)",
        ylabel="Conditional cumulative fraction",
        ylim=(0, 1.03),
        xlim=(0, result["definition"]["position_gate_m"]),
    )
    right.grid(alpha=0.25)
    if right.lines:
        right.legend(fontsize=8)
    denominators = ", ".join(str(c["summary"]["denominator"]) for c in conditions)
    figure.suptitle(
        f"Saved reports · {result['definition']['position_gate_m']:g} m gate · denominators: {denominators}"
    )
    return figure


def export_comparison(result, directory, *, plots=True):
    """Export a comparison bundle into a new or empty directory."""
    directory = Path(directory)
    if directory.is_dir() and any(directory.iterdir()):
        raise FileExistsError(
            f"Export directory is not empty: {directory}; choose a new or empty directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_definition(directory / "definition.json", result["definition"])
    with (directory / "summary.csv").open("w", newline="") as stream:
        fields = ["label", *result["conditions"][0]["summary"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            dict(label=c["source"]["label"], **c["summary"]) for c in result["conditions"]
        )
    if plots:
        figure = comparison_figure(result)
        for extension in ("png", "pdf", "svg"):
            figure.savefig(directory / f"comparison.{extension}", dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate", type=float, help="Explicit 3D rematching gate in metres")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    definition = load_definition(args.definition)
    if args.gate is not None:
        definition["position_gate_m"] = args.gate
    export_comparison(compare_volumes(definition), args.output, plots=not args.no_plots)


if __name__ == "__main__":
    main()
