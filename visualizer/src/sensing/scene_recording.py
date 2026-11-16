"""Persistent, renderer-neutral preparation of all native 3D sensing views.

Scientific validation and replay construction run once on a worker. The disk
artifact stores per-result geometry and identities, never GPU objects or dense
receiver arrays. Playback assembles history slices from these prepared records.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import asdict, replace
from threading import Event
from types import SimpleNamespace

import numpy as np

from shared.sensing.replay import ReplaySceneBinding

from ..io.packed_frame_payload import visualizer_frame_provider
from ..types.render_payloads import LineSetPayload, MaterialPayload, PointCloudPayload
from .path_context import analytical_source_record
from .recording_cache import RecordingCache

# Bump when preparation semantics or serialized fields change.
SCENE_CACHE_KEY = ("scene-recording-v1",)
_VIEW_CACHE_BYTES = 64 * 1024 * 1024
_VIEW_CACHE_ENTRIES = 512
_POSE_FIELDS = ("rx_positions", "rx_orientations", "tx_positions", "tx_orientations")


def provider_token(provider):
    info = provider.info
    return (id(provider), info.generation_id, info.frame_set_id, id(provider.frame_ids_snapshot))


def _array(value, width, *, dtype=np.float32):
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        array = array.reshape(0, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError("Invalid prepared scene array shape")
    if not np.all(np.isfinite(array)):
        raise ValueError("Non-finite prepared scene geometry")
    array.setflags(write=False)
    return array


def _encode_layer(layer):
    if layer is None:
        return None
    payload = layer.payload
    return {
        "name": layer.name,
        "points": payload.points,
        "colors": payload.colors,
        "lines": payload.lines if isinstance(payload, LineSetPayload) else None,
        "line_strip": payload.line_strip if isinstance(payload, LineSetPayload) else False,
        "material": asdict(layer.material) if layer.material is not None else None,
        "metadata": layer.metadata,
    }


def _decode_layer(data):
    from ..services.sensing_service import SensingSceneLayer

    if data is None:
        return None
    if not isinstance(data["name"], str) or not data["name"].startswith("sensing_"):
        raise ValueError("Invalid prepared sensing layer name")
    points = _array(data["points"], 3)
    colors = None if data["colors"] is None else _array(data["colors"], 4)
    if colors is not None and len(colors) != len(points):
        raise ValueError("Prepared colors do not match points")
    if data["lines"] is None:
        payload = PointCloudPayload(points=points, colors=colors)
    else:
        lines = _array(data["lines"], 2, dtype=np.int32)
        if lines.size and (lines.min() < 0 or lines.max() >= len(points)):
            raise ValueError("Prepared line index outside point buffer")
        payload = LineSetPayload(
            points=points, lines=lines, colors=colors, line_strip=bool(data["line_strip"])
        )
    material = dict(data["material"]) if data["material"] is not None else None
    if material is not None:
        for name in ("base_color", "color_multiplier", "emissive_color", "absorption_color"):
            material[name] = tuple(material[name])
    return SensingSceneLayer(
        data["name"],
        payload,
        MaterialPayload(**material) if material is not None else None,
        data["metadata"],
    )


def _encode_record(frame):
    record = {
        key: frame[key]
        for key in (*_POSE_FIELDS, "timestamp_s", "sensing", "_sensing_display_provenance")
    }
    record["current"] = _encode_layer(frame["_sensing_scene_current"])
    record["truth"] = [_encode_layer(layer) for layer in frame["_sensing_scene_truth"]]
    record["evidence"] = frame["_sensing_scene_evidence"]
    record["tracks"] = frame["_sensing_scene_tracks"]
    return record


def _decode_record(data):
    from ..services.sensing_service import SensingService

    frame = {key: _array(data[key], 3, dtype=np.float64) for key in _POSE_FIELDS}
    frame.update(
        timestamp_s=float(data["timestamp_s"]),
        sensing=data["sensing"],
        _sensing_display_provenance=data["_sensing_display_provenance"],
    )
    temporal = frame["sensing"]["temporal"]
    SensingService._bound_frame_pair_index(frame)
    SensingService._replay_layer_metadata(frame)
    identity = frame["_sensing_display_provenance"].get("replay_event_identity")
    if identity is not None:
        ReplaySceneBinding(**identity["scene_binding"])
    if not isinstance(frame["sensing"]["detections"], list) or any(
        not isinstance(row, dict) for row in frame["sensing"]["detections"]
    ):
        raise ValueError("Invalid prepared detection rows")
    if not np.isfinite(frame["timestamp_s"]) or not np.isfinite(
        temporal["result"]["reference_time_s"]
    ):
        raise ValueError("Invalid prepared result timestamp")
    for field in ("storage_frame_id", "bound_scene_frame_id"):
        value = frame["_sensing_display_provenance"][field]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Invalid prepared scene binding")
    frame["_sensing_scene_prepared"] = True
    frame["_sensing_scene_current"] = _decode_layer(data["current"])
    frame["_sensing_scene_truth"] = [_decode_layer(layer) for layer in data["truth"]]
    points, colors = (_array(data["evidence"][0], 3), _array(data["evidence"][1], 4))
    if len(points) != len(colors):
        raise ValueError("Invalid prepared evidence colors")
    frame["_sensing_scene_evidence"] = points, colors
    tracks = {}
    for tid, row in data["tracks"].items():
        sample = dict(row)
        world = _array([sample["world"]], 3)
        if world.shape != (1, 3):
            raise ValueError("Invalid prepared track point")
        sample["world"] = world[0]
        if sample["arrow"] is not None:
            sample["arrow"] = (
                _array(sample["arrow"][0], 3),
                _array(sample["arrow"][1], 2, dtype=np.int32),
            )
            points, lines = sample["arrow"]
            if lines.size and (lines.min() < 0 or lines.max() >= len(points)):
                raise ValueError("Invalid prepared arrow indices")
        tracks[int(tid)] = sample
    frame["_sensing_scene_tracks"] = tracks
    return frame


def _prepare_record(service, frame, scene_id):
    sensing = frame["sensing"]
    range_mode = service._overlay_range_mode(sensing)
    detections = service._prepare_detections(sensing.get("detections") or [], range_mode=range_mode)
    origin, rotation = service._resolve_radar_pose(frame)
    current = service._build_current_detection_layer(
        detections,
        origin,
        rotation,
        has_xy=any(service._has_local_cartesian(det) for det in detections),
        range_mode=range_mode,
    )
    track_layers = service._build_track_layers(
        frame, history=[(scene_id, frame)], current_detections=detections, show_velocity_arrows=True
    )
    arrows = {
        layer.metadata["track_id"]: (layer.payload.points, layer.payload.lines)
        for layer in track_layers
        if layer.name.startswith(service.TRACK_ARROW_PREFIX)
    }
    tracks = {
        layer.metadata["track_id"]: {
            "step": scene_id,
            "world": layer.payload.points[0],
            "confirmed": layer.metadata["confirmed"],
            "arrow": arrows.get(layer.metadata["track_id"]),
        }
        for layer in track_layers
        if layer.name.startswith(service.TRACK_POINT_PREFIX)
    }
    evidence = (
        (current.payload.points, current.payload.colors)
        if (current is not None and isinstance(current.payload, PointCloudPayload))
        else (np.empty((0, 3), np.float32), np.empty((0, 4), np.float32))
    )
    # The replay identity already covers the pulse timeline. Compact scene
    # records omit it, so detach only the metadata that is actually retained.
    temporal = deepcopy(
        {
            **sensing["temporal"],
            "acquisition": {
                key: value
                for key, value in sensing["temporal"]["acquisition"].items()
                if key != "samples"
            },
        }
    )
    compact = {key: frame[key] for key in _POSE_FIELDS}
    compact.update(
        timestamp_s=service._scene_time_s(frame),
        sensing={
            "detections": deepcopy(sensing.get("detections") or []),
            "temporal": temporal,
            "pair_axis": deepcopy(sensing["pair_axis"]),
            "config_resolved": {"range_mode": range_mode},
        },
        _sensing_display_provenance=deepcopy(frame["_sensing_display_provenance"]),
        _sensing_scene_prepared=True,
        _sensing_scene_current=current,
        _sensing_scene_truth=service._build_truth_layers(sensing),
        _sensing_scene_tracks=tracks,
        _sensing_scene_evidence=evidence,
    )
    return compact


class PreparedSceneRecording:
    """All source-derived scene records; history choices do only display assembly."""

    def __init__(self, records, analytical_sources, token, *, disk_hit=False):
        from ..services.sensing_service import SensingService

        self.token, self.disk_hit = token, disk_hit
        self.analytical_sources = {row["source_scene_id"]: row for row in analytical_sources}
        self.records = sorted(
            records,
            key=lambda frame: SensingService._selection_key(
                frame, frame["_sensing_display_provenance"]["storage_frame_id"]
            ),
        )
        self.times = [
            frame["sensing"]["temporal"]["result"]["reference_time_s"] for frame in self.records
        ]
        self.exact = {}
        self.selectable = []
        for index, frame in enumerate(self.records):
            if frame["sensing"]["detections"]:
                self.exact[frame["_sensing_display_provenance"]["bound_scene_frame_id"]] = index
                self.selectable.append(index)
        self.selectable_times = [self.times[i] for i in self.selectable]
        self._views = OrderedDict()
        self._view_bytes = 0

    def matches_provider(self, provider):
        return provider is not None and provider_token(provider) == self.token

    def resolve(
        self,
        service,
        scene,
        *,
        step,
        mode,
        history_frames,
        use_full_history,
        show_velocity_arrows,
        pin_freshest,
    ):
        if not self.matches_provider(visualizer_frame_provider(service.visualizer)):
            return NotImplemented
        timestamp = service._scene_time_s(scene)
        if timestamp is None:
            return NotImplemented
        pinned = pin_freshest or mode in {"tracks", "evidence", "full"}
        mode = str(mode or "detections").strip().lower()
        if mode not in {"detections", "tracks", "evidence", "full"}:
            mode = "detections"
        if pinned:
            position = bisect_right(self.selectable_times, timestamp) - 1
            index = self.selectable[position] if position >= 0 else -1
        else:
            index = self.exact.get(step, -1)
        if index < 0:
            return None
        record = self.records[index]
        bound = record["_sensing_display_provenance"]["bound_scene_frame_id"]
        if bound == step and (
            not np.isclose(timestamp, record["timestamp_s"], rtol=0, atol=1e-12)
            or any(not np.array_equal(scene.get(key), record[key]) for key in _POSE_FIELDS)
        ):
            return NotImplemented  # Interactive pose edits need their own validated projection.
        key = (
            index,
            mode,
            None if use_full_history else max(1, history_frames),
            show_velocity_arrows,
        )
        layers = self._views.get(key)
        if layers is None:
            unique = {}
            for candidate in (
                self.records[: bisect_right(self.times, self.times[index])]
                if mode != "detections"
                else ()
            ):
                data = candidate["sensing"]
                if data["detections"]:
                    identity = (
                        data["temporal"]["result"]["processing_run_id"],
                        data["temporal"]["acquisition"]["acquisition_id"],
                    )
                    unique[identity] = candidate
            history = sorted(
                unique.values(),
                key=lambda f: service._selection_key(
                    f, f["_sensing_display_provenance"]["storage_frame_id"]
                ),
            )
            if not use_full_history:
                history = history[-max(1, history_frames) :]
            layers = service._build_scene_overlays_from_validated(
                record,
                step=bound,
                mode=mode,
                history_frames=history_frames,
                use_full_history=use_full_history,
                show_velocity_arrows=show_velocity_arrows,
                history=[
                    (f["_sensing_display_provenance"]["bound_scene_frame_id"], f) for f in history
                ],
            )
            size = sum(
                layer.payload.points.nbytes
                + (layer.payload.colors.nbytes if layer.payload.colors is not None else 0)
                + (layer.payload.lines.nbytes if isinstance(layer.payload, LineSetPayload) else 0)
                for layer in layers
            )
            self._views[key] = (layers, size)
            self._view_bytes += size
            while self._views and (
                len(self._views) > _VIEW_CACHE_ENTRIES or self._view_bytes > _VIEW_CACHE_BYTES
            ):
                _, (_, removed) = self._views.popitem(last=False)
                self._view_bytes -= removed
        else:
            self._views.move_to_end(key)
            layers = layers[0]
        stale = bound != step
        identity = record["_sensing_display_provenance"].get("replay_event_identity")
        if identity is not None:
            binding = ReplaySceneBinding(**identity["scene_binding"])
            binding = replace(
                binding,
                current_scene_frame_id=step,
                current_scene_time_s=timestamp,
                current_scene_result_offset_s=timestamp - self.times[index],
                stale_age_s=max(0.0, timestamp - self.times[index]) if stale else 0.0,
                pinned=pinned,
                state="stale" if stale else "fresh",
                fresh=not stale,
                stale=stale,
            )
            identity = {**identity, "scene_binding": binding.to_manifest()}
            layers = [
                replace(
                    layer, metadata={**(layer.metadata or {}), "replay_event_identity": identity}
                )
                for layer in layers
            ]
        annotations = {
            "sensing": {
                "detections": record["sensing"]["detections"],
                "temporal": service._annotation_temporal(record["sensing"]),
            },
            "stale": stale,
            "result_time_s": self.times[index],
            "scene_time_s": timestamp,
        }
        return annotations, layers


def prepare_scene_recording(provider, cancelled: Event, progress=lambda *_: None):
    """Load a persistent artifact or validate and prepare it once off the UI thread."""
    from ..services.sensing_service import SensingService

    token = provider_token(provider)
    cache = RecordingCache.for_provider(provider)

    def check(stage, done, total):
        if cancelled.is_set():
            raise CancelledError()
        if provider_token(provider) != token:
            raise RuntimeError("Recording changed during 3D sensing preparation")
        progress(stage, done, total)

    check("Loading prepared 3D sensing", 0, 1)
    cached = cache.load("scene", SCENE_CACHE_KEY) if cache is not None else None
    if cached is not None:
        try:
            records = [_decode_record(row) for row in cached["records"]]
            prepared = PreparedSceneRecording(
                records, cached["analytical_sources"], token, disk_hit=True
            )
            check("3D sensing ready", len(records), len(records))
            return prepared
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass  # Disposable cache; the source remains authoritative.
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    service.configure_background_reader(cancelled, progress)
    if not provider.list_frames():
        return PreparedSceneRecording([], [], token)
    ids = service.candidate_storage_frame_ids(0)
    expected = service._refresh_metadata_index_token(ids)
    indexed = service._indexed_sensing_results(ids, expected)
    records, analytical = [], []
    for index, (storage, metadata) in enumerate(indexed):
        check("Preparing 3D sensing", index, len(indexed))
        binding = metadata["sensing"]["temporal"]["display_binding"]
        scene_id = binding.get("scene_frame_id")
        if scene_id is None or not provider.has_frame(scene_id):
            continue
        product = service._load_sensing_candidate(storage, ("detections",))
        if product is None:
            raise ValueError(f"Could not read sensing result {storage}")
        sensing = product["sensing"]
        record = analytical_source_record(sensing)
        if record is not None:
            analytical.append(record)
        scene = service._load_scene_pose_frame(scene_id)
        if scene is None:
            raise ValueError(f"Could not read bound scene {scene_id}")
        # Empty detections are valid results too. Resolve identity independently
        # of whether a particular product has any rows.
        selection = service.resolve_display_sensing_frame(
            scene,
            scene_frame_id=scene_id,
            storage_frame_id=scene_id,
            product_names=(),
            candidate_storage_ids=(storage,),
            load_candidate=lambda _: product,
        )
        if selection is None:
            raise ValueError(f"Could not bind sensing result {storage}")
        records.append(_prepare_record(service, selection.display_frame, scene_id))
    check("Saving prepared 3D sensing", len(records), len(indexed))
    if cache is not None:
        saved = cache.save(
            "scene",
            {"records": [_encode_record(f) for f in records], "analytical_sources": analytical},
            SCENE_CACHE_KEY,
        )
        if cache._building and not saved:
            raise OSError("Could not save the prepared sensing scene")
    check("3D sensing ready", len(records), len(indexed))
    return PreparedSceneRecording(records, analytical, token)
