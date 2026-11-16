"""Bind separately saved world tracks to their exact native scenario."""

from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shared.sensing.fusion.result import FusionResult

from .dashboard_data import HistoryRecord, RecordingHistory
from .tracking_presentation import tracking_snapshot


@dataclass(frozen=True)
class SavedTracking:
    """Verified saved states, separate from the native detector's output rows."""

    reader: FusionResult
    sensor_id: str
    history: RecordingHistory


class SavedTrackingLoader:
    """Worker-owned optional binding; invalidate on bundle or native file changes."""

    def __init__(self, source, result_path=None):
        self.source = Path(source).resolve() if source else None
        if self.source and self.source.suffix in {".yaml", ".yml"}:
            self.source = self.source.parent
        self._signature = None
        self._saved = None
        self._error = None
        self.result_path = Path(result_path).resolve() if result_path else None

    def load(self, cancelled):
        if self.source is None:
            return None, None
        if cancelled.is_set():
            raise CancelledError()
        path = self.result_path or self.source.parent / "tracking/fusion_results.json"
        if not path.is_file():
            self._signature = self._saved = self._error = None
            return None, None
        try:
            from shared.sensing.fusion.experiment import fingerprints
            from shared.sensing.loader import _resolve_frames_dir

            from .fusion_inspection import source_signature

            directory = _resolve_frames_dir(self.source)
            signature = (source_signature(path.parent), source_signature(directory))
            if signature != self._signature:
                self._saved = self._error = None
                reader = FusionResult(path)
                names = tuple(reader.get("sources", {}))
                if len(names) != 1 or reader.source_path(names[0]) != self.source:
                    raise ValueError("Saved tracking is not bound to this single recording")
                name = names[0]
                from .recording_cache import RecordingCache

                # Remember successful full verification against the exact bundle
                # and native file stamps. A changed/missing file forces rehashing.
                paths = (
                    path,
                    *sorted(
                        p
                        for p in directory.iterdir()
                        if p.is_file() and p.suffix in {".json", ".h5"}
                    ),
                )
                cache = RecordingCache(paths)
                verified = cache.load("saved-tracking-verification", ("v1",))
                if verified != reader["sources"][name]["hashes"]:
                    if fingerprints(directory) != reader["sources"][name]["hashes"]:
                        raise ValueError("Native source hash mismatch")
                    if cancelled.is_set():
                        raise CancelledError()
                    cache.save(
                        "saved-tracking-verification", reader["sources"][name]["hashes"], ("v1",)
                    )
                if cancelled.is_set():
                    raise CancelledError()
                if signature != (source_signature(path.parent), source_signature(directory)):
                    raise RuntimeError("Saved tracking source changed while loading")
                self._saved = SavedTracking(reader, name, saved_tracking_history(reader, name))
                self._signature = signature
            return self._saved, self._error
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self._signature = self._saved = None
            self._error = f"Saved tracking unavailable: {exc}"
            return None, self._error


def adjacent_single_sensor_result(source):
    """Discover only the conventional sibling bundle bound to this recording.

    Discovery checks the explicit source binding. The inspection worker verifies
    the native hashes before exposing the result; proximity alone is insufficient.
    """
    if not source:
        return None
    source = Path(source).resolve()
    if source.suffix in {".yaml", ".yml"}:
        source = source.parent
    path = source.parent / "tracking/fusion_results.json"
    if not path.is_file():
        return None
    reader = FusionResult(path)
    names = tuple(reader.get("sources", {}))
    if len(names) != 1 or reader.source_path(names[0]) != source:
        return None
    return path


def saved_tracking_history(reader, sensor_id):
    """Adapt confirmed world states for plotting without modifying detections.

    Radial closing velocity is projected from the saved world velocity relative
    to the explicitly named sensor. It is not world speed or a measured Doppler.
    """
    sensor = np.asarray(reader["sources"][sensor_id]["sensor"]["position_m"])
    truths = reader.get("truth_for_evaluation", [])
    lineage = reader["sources"][sensor_id].get("lineage", [])
    records = []
    snapshot_binding = SimpleNamespace(reader=reader, sensor_id=sensor_id)
    for i, time_s in enumerate(reader.times):
        tracks = []
        for track in reader.epoch(i)["tracks"]:
            if not track["confirmed"]:
                continue
            delta = np.subtract(track["position_m"], sensor)
            distance = float(np.linalg.norm(delta))
            radial = -float(np.dot(track["velocity_m_s"], delta / distance)) if distance else None
            tracks.append(
                {
                    "track_id": track["track_id"],
                    "world_track_position": track["position_m"],
                    "range_m": distance,
                    "velocity_m_s": radial,
                    "predicted_only": track["predicted_only"],
                    "track_confirmed": track["confirmed"],
                    "source_kind": "saved_world_track",
                }
            )
        truth = truths[i] if i < len(truths) else {}
        valid_truth = (
            truth.get("available", False) and abs(truth.get("time_s", float("inf")) - time_s) < 1e-9
        )
        targets = truth.get("targets", []) if valid_truth else []
        gt = {"target_ids": [], "target_ranges": [], "target_speeds": []} if valid_truth else None
        positions = []
        for target in targets:
            delta = np.subtract(target["position_m"], sensor)
            distance = float(np.linalg.norm(delta))
            gt["target_ids"].append(str(target["target_id"]))
            gt["target_ranges"].append(distance)
            gt["target_speeds"].append(
                -float(np.dot(target["velocity_m_s"], delta / distance)) if distance else None
            )
            positions.append((str(target["target_id"]), tuple(target["position_m"])))
        binding = lineage[i] if i < len(lineage) else {}
        snapshot = tracking_snapshot(None, saved_tracking=snapshot_binding, time_s=float(time_s))
        records.append(
            HistoryRecord(
                time_s=float(time_s),
                identity=("saved_world_tracks", reader.inspection_id, sensor_id, str(i)),
                spectrum=None,
                doppler_hz=None,
                range_m=None,
                range_power=None,
                detections=tuple(tracks),
                truth=gt,
                truth_positions=tuple(positions),
                scene_frame_id=binding.get("scene_frame_id"),
                storage_id=binding.get("storage_frame_id"),
                tracking_states=tuple(state for state in snapshot.states if state.get("confirmed")),
            )
        )
    return RecordingHistory(tuple(records))
