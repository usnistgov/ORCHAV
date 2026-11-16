"""Audit saved volume associations, native provenance and CPU inspection costs.

No acquisition or detection is performed. Native verification is explicit because
it reads retained files; ordinary inspection is measured separately from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.sensing.volume.evaluation import VolumeEvaluation, load_evaluation, saved_reports
from shared.sensing.volume.inspection import cell_support, detail_grid, report_details, slice_data
from shared.sensing.volume.reader import SensingVolume


def audit(path, native="none"):
    start = perf_counter()
    reader = SensingVolume(path)
    reader.verify_scene()
    read_seconds = perf_counter() - start
    start = perf_counter()
    model = load_evaluation(reader)
    evaluation_seconds = perf_counter() - start
    totals = dict(measured=0, matches=0, extra_reports=0, no_reports=0, matches_outside_cell=0)
    examples = {}
    scores = []
    for heading in range(len(reader.headings)):
        snapshot = model.evaluate(heading, position_gate_m=reader.metadata["position_gate_m"])
        scores.append({k: v for k, v in snapshot.items() if isinstance(v, int)})
        categories = detail_grid(snapshot)
        assert np.isfinite(categories).sum() == snapshot["measured"]
        for orientation in ("xy", "xz", "yz"):
            axis = {"xy": 0, "xz": 1, "yz": 2}[orientation]
            for index in range(reader.shape[axis]):
                plane, *_ = slice_data(snapshot["outcomes"], reader.axes, orientation, index)
                np.testing.assert_array_equal(
                    plane, np.take(snapshot["outcomes"], index, axis=axis)
                )
        for key, trial in reader.cells.items():
            if key[0] != heading:
                continue
            if trial["status"] != "measured":
                assert not np.isfinite(snapshot["outcomes"][key[1:]])
                continue
            rows = model.rows[trial["id"]]
            errors = [
                (
                    math.dist(row["position_m"], trial["position_m"])
                    if row["position_m"] is not None
                    else math.inf
                )
                for row in rows
            ]
            selected = min(range(len(errors)), key=errors.__getitem__) if errors else None
            hit = selected is not None and errors[selected] <= reader.metadata["position_gate_m"]
            assert hit == trial["hit"] == bool(snapshot["outcomes"][key[1:]])
            details = report_details(model, key, snapshot)
            assert sum(row["matched"] for row in details) == int(hit)
            if hit:
                assert details[selected]["matched"]
            totals["measured"] += 1
            totals["matches"] += int(hit)
            totals["extra_reports"] += len(rows) - int(hit)
            totals["no_reports"] += int(not rows)
            outside = hit and not details[selected]["inside_sample_cell"]
            totals["matches_outside_cell"] += int(outside)
            label = (
                "match_outside_cell"
                if outside
                else "match" if hit else "miss_with_reports" if rows else "no_reports"
            )
            examples.setdefault(label, list(key))
            if len(rows) > int(hit):
                examples.setdefault("extra_reports", list(key))
            centre, spacing, lower, upper = cell_support(reader, key)
            np.testing.assert_array_equal(centre, trial["position_m"])
            np.testing.assert_allclose(upper - lower, spacing)
    # Gate changes classify the same report inventory. They must not mutate it.
    row_hash = hashlib.sha256(json.dumps(model.rows, sort_keys=True).encode()).hexdigest()
    start = perf_counter()
    for _ in range(5):
        for gate in (0.25, 0.75, 1.5):
            for heading in range(len(reader.headings)):
                result = model.evaluate(heading, position_gate_m=gate)
                assert result["tp"] + result["fn"] == result["measured"]
                assert result["tp"] + result["fp"] == int(result["report_counts"].sum())
    rescore_seconds = (perf_counter() - start) / (15 * len(reader.headings))
    assert row_hash == hashlib.sha256(json.dumps(model.rows, sort_keys=True).encode()).hexdigest()
    eligible = [
        key
        for key, row in reader.cells.items()
        if row["status"] == "measured" and row.get("native_evidence") != "not_included"
    ]
    keys = (
        eligible
        if native == "all"
        else (
            [tuple(k) for k in examples.values() if tuple(k) in eligible]
            if native == "sample"
            else []
        )
    )
    keys = list(dict.fromkeys(keys))
    native_rows = {}
    verified_bytes = 0
    native_start = perf_counter()
    for index, key in enumerate(keys):
        from shared.sensing.fusion.experiment import load_sensor

        trial = reader.cells[key]
        source = reader.verify_native(key)
        samples = reader.metadata.get("cpi_samples")
        data = load_sensor(
            source,
            reader.metadata.get("sensor_id", "Volume"),
            result_frame_ids=(samples - 1,) if samples else None,
        )
        assert data["source"]["hashes"] == trial["source_hashes"]
        truth = data["truth"][0]["targets"][0]
        np.testing.assert_allclose(truth["position_m"], trial["position_m"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(truth["velocity_m_s"], trial["velocity_m_s"], rtol=0, atol=1e-5)
        rows = saved_reports(data)
        assert rows == model.rows[trial["id"]], f"Native report discrepancy: {trial['id']}"
        native_rows[trial["id"]] = rows
        verified_bytes += sum(
            (data["directory"] / name).stat().st_size for name in trial["source_hashes"]
        )
        if (index + 1) % 100 == 0:
            print(
                f"Verified {index + 1}/{len(keys)} native trials in {reader.path.parent}",
                flush=True,
            )
    if native == "all" and len(keys) == totals["measured"]:
        VolumeEvaluation(reader, native_rows)
    assert hashlib.sha256(reader.path.read_bytes()).hexdigest() == reader.identity
    return dict(
        source=str(reader.path),
        manifest_sha256=reader.identity,
        totals=totals,
        headings=scores,
        representative_indices=examples,
        native_verification=native,
        verified_native_trials=len(keys),
        verified_native_bytes=verified_bytes,
        seconds=dict(
            manifest_and_scene=read_seconds,
            reports=evaluation_seconds,
            mean_rescore=rescore_seconds,
            native_verification=perf_counter() - native_start,
        ),
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        ray_tracing_imported=any(
            name == "sionna.rt" or name.startswith("sionna.rt.") for name in sys.modules
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volumes", nargs="+", type=Path)
    parser.add_argument("--native", choices=("none", "sample", "all"), default="none")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [audit(path, args.native) for path in args.volumes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
