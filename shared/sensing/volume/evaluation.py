"""CPU-only rescoring of saved single-target volume trials.

The native detector is never rerun. Position matching and range/radial matching
are distinct requirements; report presence and nearest-report error do not assert
that a report originated from the target. One truth target permits at most one TP.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path

import numpy as np

from shared.sensing.volume.reader import MAX_MANIFEST_BYTES, SensingVolume

logger = logging.getLogger(__name__)


def saved_reports(data):
    """Compact raw native measurements, before optional tracker quality policies."""
    if len(data["epochs"]) != 1:
        raise ValueError("Volume evaluation requires one complete CPI per trial")
    return [
        {
            "detection_id": observation.detection_id,
            "position_m": (
                observation.world_position(data["sensor"]).tolist()
                if observation.has_position
                else None
            ),
            "range_m": observation.range_m,
            "closing_velocity_m_s": observation.closing_velocity_m_s,
        }
        for observation in data["epochs"][0]["observations"]
    ]


def _digest(rows):
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_evaluation(reader, *, cancel=None, progress=None):
    """Build or reuse a manifest-bound derived report cache, with cancellation.

    Building verifies native hashes and truth bindings. Reuse reads that saved
    snapshot without rescanning native files; opening any native trial still
    verifies its source. A new manifest invalidates the cache. Never change the
    original recordings or the saved classification manifest.
    """
    cancel = cancel or threading.Event()
    cache = reader.path.parent / "evaluation_reports.json"
    if cache.is_file() and cache.stat().st_size <= MAX_MANIFEST_BYTES:
        try:
            payload = json.loads(cache.read_bytes())
            if (
                isinstance(payload, dict)
                and payload.get("schema_version") == 1
                and payload.get("volume_sha256") == reader.identity
                and payload.get("reports_sha256") == _digest(payload["reports"])
            ):
                if cancel.is_set():
                    raise InterruptedError("Evaluation loading cancelled")
                return VolumeEvaluation(reader, payload["reports"])
        except (ValueError, KeyError, TypeError):
            # This is a disposable cache, not authoritative detector evidence.
            # Rebuild from source instead of publishing partially valid data.
            pass
    trials = [t for t in reader.cells.values() if t["status"] == "measured"]
    rows = {}
    for index, trial in enumerate(trials):
        if cancel.is_set():
            raise InterruptedError("Evaluation loading cancelled")
        if "reports" in trial:
            reports = trial["reports"]
        else:
            from shared.sensing.fusion.experiment import load_sensor

            native = (reader.path.parent / trial["native"]).resolve()
            if (
                hashlib.sha256((native / "scenario.yaml").read_bytes()).hexdigest()
                != trial["configuration_sha256"]
            ):
                raise ValueError("Native trial configuration changed")
            # These authored trials store one completed CPI on its final slow-
            # time frame. Its display binding still selects the reference epoch.
            samples = reader.metadata.get("cpi_samples")
            if samples is not None and (
                isinstance(samples, bool) or not isinstance(samples, int) or samples < 1
            ):
                raise ValueError("Volume CPI sample count must be a positive integer")
            data = load_sensor(
                native,
                reader.metadata.get("sensor_id", "Volume"),
                result_frame_ids=(samples - 1,) if samples else None,
            )
            if data["source"]["hashes"] != trial["source_hashes"]:
                raise ValueError("Native trial hashes disagree with the volume")
            if (
                len(data["truth"]) != 1
                or not data["truth"][0]["available"]
                or len(data["truth"][0]["targets"]) != 1
            ):
                raise ValueError("Expected one independently recorded truth target")
            truth = data["truth"][0]["targets"][0]
            if not np.allclose(
                truth["position_m"], trial["position_m"], atol=1e-5
            ) or not np.allclose(truth["velocity_m_s"], trial["velocity_m_s"], atol=1e-5):
                raise ValueError("Native truth disagrees with the volume reference")
            if not np.allclose(data["sensor"].position_m, reader.metadata["sensor_position_m"]):
                raise ValueError("Native sensor pose disagrees with the volume")
            reports = saved_reports(data)
        rows[trial["id"]] = reports
        if progress is not None:
            progress(index + 1, len(trials))
    result = VolumeEvaluation(reader, rows)
    if cancel.is_set():
        raise InterruptedError("Evaluation loading cancelled")
    payload = {
        "schema_version": 1,
        "volume_sha256": reader.identity,
        "reports_sha256": _digest(rows),
        "reports": rows,
    }
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) > MAX_MANIFEST_BYTES:
        raise ValueError("Evaluation report cache exceeds 64 MiB")
    temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(encoded)
        if cancel.is_set():
            raise InterruptedError("Evaluation loading cancelled")
        temporary.replace(cache)
    except OSError as exc:
        if not isinstance(exc, PermissionError) and exc.errno not in {
            errno.EACCES,
            errno.EPERM,
            errno.EROFS,
        }:
            raise
        # Portable recordings can be read-only. Live inspection still works;
        # only persistence is unavailable, so a later opening reads again.
        logger.info("Evaluation cache is read-only; retaining reports in memory: %s", cache)
    finally:
        temporary.unlink(missing_ok=True)
    return result


class VolumeEvaluation:
    """Vectorized single-truth associations, preserving duplicate false reports."""

    def __init__(self, reader, rows):
        self.reader = reader
        self.rows = rows
        self.shape = (len(reader.headings), *reader.shape)
        self.size = int(np.prod(self.shape))
        self.measured = np.zeros(self.size, dtype=bool)
        self.report_counts = np.zeros(self.size, dtype=np.int64)
        cells, position_errors, range_errors, velocity_errors, report_ids = [], [], [], [], []
        expected = {t["id"] for t in reader.cells.values() if t["status"] == "measured"}
        if not isinstance(rows, dict) or rows.keys() != expected:
            raise ValueError("Report cache trial inventory disagrees with the volume")
        sensor = np.asarray(reader.metadata["sensor_position_m"])
        for key, trial in reader.cells.items():
            if trial["status"] != "measured":
                continue
            cell = int(np.ravel_multi_index(key, self.shape))
            reports = rows[trial["id"]]
            if not isinstance(reports, list) or len(reports) != trial["tp"] + trial["fp"]:
                raise ValueError("Saved detector report count disagrees with the volume")
            self.measured[cell] = True
            self.report_counts[cell] = len(reports)
            truth_position = np.asarray(trial["position_m"])
            truth_range = np.linalg.norm(truth_position - sensor)
            ids = set()
            for report in reports:
                report_id = report["detection_id"]
                if not isinstance(report_id, str) or not report_id or report_id in ids:
                    raise ValueError("Report IDs must be nonempty and unique within a trial")
                ids.add(report_id)
                position = report["position_m"]
                if position is not None:
                    position = np.asarray(position, dtype=float)
                    if position.shape != (3,) or not np.isfinite(position).all():
                        raise ValueError("Invalid report world position")
                for field in ("range_m", "closing_velocity_m_s"):
                    value = report[field]
                    if value is not None and (isinstance(value, bool) or not np.isfinite(value)):
                        raise ValueError("Report components must be finite or explicitly absent")
                if report["range_m"] is not None and report["range_m"] <= 0:
                    raise ValueError("Report range must be positive")
                cells.append(cell)
                report_ids.append(report_id)
                position_errors.append(
                    np.linalg.norm(position - truth_position) if position is not None else np.inf
                )
                range_errors.append(
                    abs(report["range_m"] - truth_range)
                    if report["range_m"] is not None
                    else np.inf
                )
                velocity_errors.append(
                    abs(report["closing_velocity_m_s"] - trial["radial_closing_velocity_m_s"])
                    if report["closing_velocity_m_s"] is not None
                    else np.inf
                )
        self.report_cells = np.asarray(cells, dtype=np.int64)
        self.position_errors = np.asarray(position_errors, dtype=float)
        self.range_errors = np.asarray(range_errors, dtype=float)
        self.velocity_errors = np.asarray(velocity_errors, dtype=float)
        self.report_ids = report_ids
        self.nearest_error = np.full(self.size, np.inf)
        np.minimum.at(self.nearest_error, self.report_cells, self.position_errors)
        self.nearest_error[~np.isfinite(self.nearest_error)] = np.nan
        # Cache/embedded reports must reproduce the saved baseline before any
        # interactive gate is applied. Counts alone cannot detect wrong values.
        for key, trial in reader.cells.items():
            if trial["status"] != "measured":
                continue
            nearest = self.nearest_error[np.ravel_multi_index(key, self.shape)]
            if bool(nearest <= reader.metadata["position_gate_m"]) != bool(trial["hit"]):
                raise ValueError("Saved reports disagree with the recorded matching outcome")
            error = trial.get("matched_error_m")
            if error is not None and not np.isclose(nearest, error, rtol=1e-7, atol=1e-7):
                raise ValueError("Saved reports disagree with the recorded position error")
            if "matched_detection_id" in trial:
                recorded = trial["matched_detection_id"]
                reports = rows[trial["id"]]
                eligible = [r for r in reports if r["position_m"] is not None]
                nearest_report = min(
                    eligible,
                    key=lambda r: np.linalg.norm(np.asarray(r["position_m"]) - trial["position_m"]),
                    default=None,
                )
                expected_id = nearest_report["detection_id"] if trial["hit"] else None
                if recorded != expected_id:
                    raise ValueError("Saved reports disagree with the recorded matched report ID")

    def evaluate(
        self, heading, *, mode="position", position_gate_m=1, range_gate_m=1, velocity_gate_m_s=1
    ):
        """Apply inclusive gates; match at most one report per independently tested target.

        Range/radial gates and normalized Euclidean cost match the shared
        dashboard evaluator's single-target case. Missing radial values cannot
        satisfy a radial gate. Extra reports remain FP even in a matched cell.
        """
        if not 0 <= heading < len(self.reader.headings):
            raise IndexError("Unknown volume heading")
        for gate in (position_gate_m, range_gate_m, velocity_gate_m_s):
            if isinstance(gate, bool) or not np.isfinite(gate) or gate <= 0:
                raise ValueError("Evaluation gates must be finite and positive")
        if mode == "position":
            scores = self.position_errors
            valid = scores <= position_gate_m
        elif mode == "range_velocity":
            if self.reader.is_fusion:
                raise ValueError(
                    "Fused estimates use 3D position matching, not single-sensor radial gates"
                )
            scores = np.hypot(
                self.range_errors / range_gate_m, self.velocity_errors / velocity_gate_m_s
            )
            valid = (self.range_errors <= range_gate_m) & (
                self.velocity_errors <= velocity_gate_m_s
            )
        else:
            raise ValueError("Choose position or range_velocity matching")
        best = np.full(self.size, np.inf)
        np.minimum.at(best, self.report_cells[valid], scores[valid])
        candidates = valid & (scores == best[self.report_cells])
        selected = np.full(self.size, len(scores), dtype=np.int64)
        np.minimum.at(selected, self.report_cells[candidates], np.flatnonzero(candidates))
        selected[selected == len(scores)] = -1
        hits = selected >= 0
        outcomes = np.where(self.measured, hits.astype(float), np.nan).reshape(self.shape)[heading]
        count = self.report_counts.reshape(self.shape)[heading]
        measured = self.measured.reshape(self.shape)[heading]
        hit_grid = hits.reshape(self.shape)[heading]
        return {
            "outcomes": outcomes,
            "report_counts": count,
            "nearest_error_m": self.nearest_error.reshape(self.shape)[heading],
            "selected_report": selected.reshape(self.shape)[heading],
            "tp": int(hit_grid.sum()),
            "fp": int(count.sum() - hit_grid.sum()),
            "fn": int(measured.sum() - hit_grid.sum()),
            "measured": int(measured.sum()),
            "no_reports": int((measured & (count == 0)).sum()),
            "reports_without_position": int(
                (
                    measured
                    & (count > 0)
                    & ~np.isfinite(self.nearest_error.reshape(self.shape)[heading])
                ).sum()
            ),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare cached reports for live volume evaluation"
    )
    parser.add_argument("volume", type=Path)
    args = parser.parse_args()
    volume = SensingVolume(args.volume)
    volume.verify_scene()
    result = load_evaluation(
        volume,
        progress=lambda done, total: (
            print(f"Reports {done}/{total}", flush=True)  # noqa: T201 - explicit CLI progress
            if done % 250 == 0 or done == total
            else None
        ),
    )
    for heading in range(len(volume.headings)):
        result_row = result.evaluate(heading, position_gate_m=volume.metadata["position_gate_m"])
        print({k: v for k, v in result_row.items() if isinstance(v, int)})  # noqa: T201 - CLI
