"""Saved fusion inspection using native dashboard readers and neutral world layers."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Event
from types import SimpleNamespace
from zlib import crc32

import numpy as np

from shared.sensing.fusion.core import COMPONENTS, Sensor, measurement_model
from shared.sensing.fusion.result import NOT_RECORDED, FusionResult
from shared.sensing.loader import _open_provider, _resolve_frames_dir

from ..model import RenderObject, Transform
from ..services.sensing_service import SensingSceneLayer, SensingService
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    PointCloudPayload,
    TextLabelPayload,
)
from .dashboard_data import (
    DashboardSession,
    HistoryRecord,
    RecordingHistory,
    load_dashboard_request,
)
from .display_style import color_rgba, identity_color


def native_observation_request(request, lineage):
    """Display the exact saved observation rows named by verified fusion lineage.

    This is a shallow presentation projection: products and persisted detector/
    tracker lists remain untouched. Its collection marker keeps row selections
    distinct from confirmed native outputs with the same numerical index.
    """
    collection = lineage.get("detection_collection", "detections")
    if collection == "detections":
        return request
    if collection != "detections_before_tracking":
        raise ValueError(f"Unsupported native detection collection: {collection}")
    from shared.extensions.sensing import assert_sensing_contract

    sensing = request.display_frame["sensing"]
    assert_sensing_contract(request.display_frame, require_schema_version=True)
    temporal = sensing["temporal"]
    if (
        lineage.get("acquisition_id") != temporal["acquisition"]["acquisition_id"]
        or lineage.get("time_s") != temporal["result"]["reference_time_s"]
        or lineage.get("processing_recipe_id") != temporal["result"]["processing_recipe_id"]
        or not isinstance(sensing.get(collection), list)
        or "detections_before_tracking_metadata" not in sensing
    ):
        raise ValueError("Native pre-tracking reports do not match saved source lineage")
    return replace(
        request,
        display_frame={
            **request.display_frame,
            "sensing": {
                **sensing,
                "detections": sensing[collection],
                "_display_detection_collection": collection,
            },
        },
        # Native detection histories index confirmed output rows. They cannot
        # identify rows of this different collection; saved fusion tracks remain
        # separately available through tracking_history.
        recording_history=None,
        prepared_evaluation=None,
    )


def native_result_request(request):
    """Give detection plots the exact selected native result, including its truth.

    Fusion owns the timeline and does not request a separate recording history.
    Without this record, the dashboard waits forever for that missing history
    and clears the range/radial plots despite having loaded the measurements.
    Build this after selecting the saved detection collection, so pre-tracking
    reports can never be replaced by confirmed-output rows.
    """
    sensing = request.display_frame["sensing"]
    evidence = request.evaluation
    truth = evidence["truth"]
    positions = tuple(
        (t["target_id"], tuple(t["position_m"]))
        for t in (sensing.get("truth") or {}).get("targets", [])
        if truth is not None and t.get("valid") and t.get("lifecycle") == "active"
    )
    record = HistoryRecord(
        time_s=evidence["time_s"],
        identity=evidence["identity"],
        spectrum=None,
        doppler_hz=None,
        range_m=None,
        range_power=None,
        detections=tuple(sensing.get("detections", [])),
        truth=truth,
        truth_positions=positions,
        evaluation=evidence,
        scene_frame_id=request.effective_display_step,
    )
    return replace(request, recording_history=RecordingHistory((record,)))


@dataclass(frozen=True)
class FusionSelection:
    run_id: str
    track_id: int
    epoch_index: int
    mode: str = "fused"


def _track_trails(result, index, mode):
    """Connect consecutive saved states of each identity, only through this epoch."""
    histories = {}
    for epoch_index, epoch in enumerate(result["histories"][mode][: index + 1]):
        for track in epoch["tracks"]:
            histories.setdefault(track["track_id"], []).append((epoch_index, track))
    layers = []
    for track_id, rows in histories.items():
        edges = [(i, i + 1) for i in range(len(rows) - 1) if rows[i + 1][0] == rows[i][0] + 1]
        if not edges:
            continue
        last_index, _ = rows[-1]
        current = last_index == index
        layers.append(
            SensingSceneLayer(
                name=f"fusion_trail_{mode}_{track_id}",
                payload=LineSetPayload(
                    points=np.asarray([track["position_m"] for _, track in rows], dtype=np.float32),
                    lines=np.asarray(edges, dtype=np.int32),
                ),
                material=MaterialPayload(
                    base_color=color_rgba(identity_color(track_id), 0.6 if current else 0.25),
                    line_width=2.5,
                ),
                metadata={
                    "fusion_run_id": result.inspection_id,
                    "sensing_role": "track",
                    "track_id": track_id,
                    "fusion_selection": FusionSelection(
                        result.inspection_id, track_id, last_index, mode
                    ),
                    "current": current,
                    "epoch_indices": tuple(epoch for epoch, _ in rows),
                    "predicted_only": tuple(track["predicted_only"] for _, track in rows),
                },
            )
        )
    return layers


def world_layers(result, index, visible, selected=None):
    """Visibility changes only render snapshots, never fusion state or decisions."""
    layers = []
    run = result.inspection_id

    def point(name, position, color, size, **metadata):
        layers.append(
            SensingSceneLayer(
                name=f"fusion_{name}",
                payload=PointCloudPayload(
                    points=np.asarray(position, dtype=np.float32).reshape(-1, 3)
                ),
                material=MaterialPayload(base_color=color_rgba(color), point_size=size),
                metadata={"fusion_run_id": run, **metadata},
            )
        )

    for name, source in result.get("sources", {}).items():
        color = identity_color(crc32(name.encode()))
        if "sensors" in visible:
            point(
                f"sensor_{name}", source["sensor"]["position_m"], color, 18, sensing_role="sensor"
            )
        if name in visible:
            observations = result.get("observations", {}).get(name, [])
            if index < len(observations):
                for i, detection in enumerate(observations[index]["detections"]):
                    if detection.get("world_position_m") is not None:
                        point(
                            f"detection_{name}_{i}",
                            detection["world_position_m"],
                            color,
                            8,
                            sensing_role="evidence",
                            native_source=detection.get("native_source", NOT_RECORDED),
                        )
    for mode in result["histories"]:
        if mode not in visible and f"tracks:{mode}" not in visible:
            continue
        if mode != "fused" and f"tracks:{mode}" not in visible:
            continue
        if "trails" in visible:
            layers.extend(_track_trails(result, index, mode))
        for track in result.epoch(index, mode)["tracks"]:
            selection = FusionSelection(run, track["track_id"], index, mode)
            point(
                f"track_{mode}_{track['track_id']}",
                track["position_m"],
                identity_color(track["track_id"]),
                20 if selection == selected else 12,
                sensing_role="track",
                track_id=track["track_id"],
                fusion_selection=selection,
            )
    if "truth" in visible:
        truth = result.get("truth_for_evaluation", [])
        if index < len(truth) and truth[index].get("available", True):
            for target in truth[index]["targets"]:
                point(
                    f"truth_{target['target_id']}",
                    target["position_m"],
                    identity_color(target["target_id"], truth=True),
                    15,
                    sensing_role="truth",
                )
    return layers


def world_sensor_labels(result, visible):
    """Name every saved radar pose, including sources outside the open native scene."""
    if "sensors" not in visible:
        return []
    labels = []
    for name, source in result.get("sources", {}).items():
        position = np.asarray(source["sensor"]["position_m"], dtype=float)
        labels.append(
            RenderObject(
                id=f"fusion_sensor_label_{name}",
                payload=TextLabelPayload(text=f"{name} radar", font_size=0.3, screen_space=True),
                material=MaterialPayload(
                    base_color=color_rgba(identity_color(crc32(name.encode())))
                ),
                transform=Transform.from_translation(position + [0, 0, 0.5]),
                metadata={
                    "fusion_run_id": result.inspection_id,
                    "sensing_role": "sensor",
                    "layout_anchor": position.tolist(),
                    "locator_role": "radio",
                },
            )
        )
    return labels


def track_details(result, selection, sensor_id):
    track = next(
        t
        for t in result.epoch(selection.epoch_index, selection.mode)["tracks"]
        if t["track_id"] == selection.track_id
    )
    details = dict(track)
    details["world_speed_m_s"] = float(np.linalg.norm(track["velocity_m_s"]))
    sensor = Sensor(**result["sources"][sensor_id]["sensor"])
    try:
        radial, _ = measurement_model(
            np.r_[track["position_m"], track["velocity_m_s"]], sensor, (COMPONENTS[3],)
        )
        details[f"{sensor_id}_predicted_radial_closing_velocity_m_s"] = float(radial[0])
    except ValueError:
        details[f"{sensor_id}_predicted_radial_closing_velocity_m_s"] = "undefined at sensor"
    details["confirmation_history"] = [
        {
            "time_s": e["time_s"],
            "confirmed": t["confirmed"],
            "predicted_only": t["predicted_only"],
            "recent_hits": t.get("recent_hits", NOT_RECORDED),
            "sensors": t.get("current_sensor_support", NOT_RECORDED),
        }
        for e in result["histories"][selection.mode][: selection.epoch_index + 1]
        for t in e["tracks"]
        if t["track_id"] == selection.track_id
    ]
    return details


def source_signature(directory):
    """Detect source replacement/in-place changes around a cached or fresh read."""
    result = []
    for path in sorted(directory.iterdir()):
        if path.suffix in {".json", ".h5"}:
            stat = path.stat()
            result.append(
                (
                    path.name,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
    return tuple(result)


class FusionInspectionSession:
    """Single worker owns one native provider and a two-request application cache.

    Qt polls immutable completions. Source switches retire the previous reader on
    its owning worker; cancellation never races HDF5 close against an active read.
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fusion-inspection")
        self.cancelled = Event()
        self.provider = None
        self.service = None
        self.source = None
        self.signature = None
        self.cache = OrderedDict()
        self.closed = False

    def open(self, path):
        return self.executor.submit(FusionResult, path)

    def request(self, result, index, sensor_id, view="analysis", variant="auto"):
        return self.executor.submit(self._native, result, index, sensor_id, view, variant)

    def _native(self, result, index, sensor_id, view, variant):
        if self.cancelled.is_set():
            raise CancelledError()
        source = result.source_path(sensor_id)
        if source is None or not source.exists():
            raise FileNotFoundError(f"Native recording unavailable: {sensor_id}")
        directory = _resolve_frames_dir(source)
        signature = source_signature(directory)
        key = (result.inspection_id, sensor_id, str(source), signature, index, view, variant)
        if source != self.source or signature != self.signature:
            self._close_provider()
            from shared.sensing.fusion.experiment import fingerprints

            if fingerprints(directory) != result["sources"][sensor_id]["hashes"]:
                raise ValueError(f"Native source hash mismatch: {sensor_id}")
            _, _, self.provider = _open_provider(source)
            self.source, self.signature = source, signature
            self.service = SensingService(
                SimpleNamespace(frame_source=SimpleNamespace(provider=self.provider))
            )
            self.service.configure_background_reader(self.cancelled, lambda *args: None)
            self.cache.clear()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        lineage = result["sources"][sensor_id].get("lineage", [])
        if index >= len(lineage):
            raise ValueError("Native epoch binding not recorded")
        request = load_dashboard_request(
            self.service, lineage[index]["scene_frame_id"], view, variant, False, "auto"
        )
        if self.cancelled.is_set():
            raise CancelledError()
        if source_signature(directory) != signature:
            raise RuntimeError("Native source changed while loading")
        if request is not None:
            # Exact-cell inspection requires the same provider-occurrence binding
            # as the ordinary dashboard, after source integrity has been verified.
            request = replace(
                request, source_identity=DashboardSession.provider_identity(self.provider)
            )
            request = native_observation_request(request, lineage[index])
            if view in {"analysis", "current"}:
                request = native_result_request(request)
        self.cache[key] = request
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return request

    def _close_provider(self):
        if self.provider is not None:
            self.provider.close()
        self.provider = self.service = self.source = self.signature = None
        self.cache.clear()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancelled.set()
        # Cleanup follows any in-flight read; no callback touches a retired Qt object.
        self.executor.submit(self._close_provider)
        self.executor.shutdown(wait=False)
