"""Render verified saved world states through the normal sensing layer controls."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

import numpy as np

from ..model import RenderObject, Transform
from ..services.sensing_overlay_sync import SensingOverlaySync
from ..services.sensing_service import SensingSceneLayer
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    PointCloudPayload,
    TextLabelPayload,
)
from .display_style import color_rgba, identity_color
from .fusion_inspection import FusionSelection
from .saved_tracking import SavedTrackingLoader


def saved_track_layers(binding, index, *, history_count, arrows, labels, service, at_epoch=True):
    """Keep observed and predicted states explicit; never synthesize detector rows."""
    reader = binding.reader
    start = 0 if history_count is None else max(0, index + 1 - history_count)
    histories = {}
    for epoch_index in range(start, index + 1):
        for track in reader.epoch(epoch_index)["tracks"]:
            histories.setdefault(track["track_id"], []).append((epoch_index, track))
    layers, text = [], []
    for tid, rows in histories.items():
        last_index, track = rows[-1]
        current = last_index == index and at_epoch
        rgba = color_rgba(identity_color(tid), 1.0 if current else 0.3)
        selection = FusionSelection(reader.inspection_id, tid, last_index)
        metadata = {
            "track_id": tid,
            "fusion_selection": selection,
            "current": current,
            "confirmed": track["confirmed"],
            "predicted_only": track["predicted_only"],
        }
        points = np.asarray([row[1]["position_m"] for row in rows], dtype=np.float32)
        layers.append(
            SensingSceneLayer(
                name=f"sensing_saved_track_point_{tid}",
                payload=PointCloudPayload(points=points[-1:]),
                material=MaterialPayload(
                    base_color=rgba, point_size=12 if track["confirmed"] else 8
                ),
                metadata=metadata,
            )
        )
        # Do not connect across missing epochs: a gap is not an observed trajectory.
        edges = [(i, i + 1) for i in range(len(rows) - 1) if rows[i + 1][0] == rows[i][0] + 1]
        if edges:
            layers.append(
                SensingSceneLayer(
                    name=f"sensing_saved_track_trail_{tid}",
                    payload=LineSetPayload(points=points, lines=np.asarray(edges, dtype=np.int32)),
                    material=MaterialPayload(
                        base_color=rgba[:3] + (0.6 if current else 0.25,), line_width=2.5
                    ),
                    metadata=metadata,
                )
            )
        if arrows and current:
            arrow = service._build_velocity_arrow_payload(
                points[-1], np.asarray(track["velocity_m_s"])
            )
            if arrow is not None:
                layers.append(
                    SensingSceneLayer(
                        name=f"sensing_saved_track_arrow_{tid}",
                        payload=LineSetPayload(points=arrow[0], lines=arrow[1]),
                        material=MaterialPayload(base_color=rgba, line_width=3.0),
                        metadata=metadata,
                    )
                )
        if labels:
            status = (
                "coasting"
                if track["predicted_only"]
                else ("confirmed" if track["confirmed"] else "tentative")
            )
            text.append(
                RenderObject(
                    id=f"sensing_saved_track_label_{tid}",
                    payload=TextLabelPayload(
                        text=f"Track {tid} · {status}" + (" · history" if not current else ""),
                        font_size=0.3,
                        screen_space=True,
                    ),
                    material=MaterialPayload(base_color=rgba),
                    transform=Transform.from_translation(points[-1] + [0, 0, 0.6]),
                    metadata={"layout_anchor": points[-1].tolist(), "pickable": False},
                )
            )
    return layers, text


class SavedTrackingScene:
    """One cancellable reader per source; scene display does not require a dashboard."""

    def __init__(self):
        self.source_key = None
        self.binding = None
        self.pending = None
        self.executor = None
        self.cancelled = Event()
        self.overlay = SensingOverlaySync()
        self.layers = []
        self.error = None
        self.status = ""
        self._render_key = None
        self._snapshots = ()
        self._loaded = False

    def install(self, binding, error, source_key, result_path=None):
        """Adopt startup-prepared tracking without a second source verification."""
        self.close()
        self.source_key = (*source_key, str(result_path))
        self.binding, self.error = binding, error
        self._loaded = True
        self._render_key = None

    def prepare(self, source, source_key, step, state, service, result_path=None):
        from .runtime import state_value

        source_key = (*source_key, str(result_path))
        if source_key != self.source_key:
            self.close()
            self.source_key = source_key
            self.binding = self.error = None
            self._render_key = None
        active = state_value(state, "show_detections") and state_value(state, "scene_mode") in {
            "tracks",
            "full",
        }
        if not active:
            self.layers = []
            self.status = ""
            return ()
        if self.executor is None and not self._loaded:
            self.cancelled = Event()
            self.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="saved-track-scene"
            )
            self.pending = self.executor.submit(
                SavedTrackingLoader(source, result_path).load, self.cancelled
            )
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            self._loaded = True
            try:
                self.binding, self.error = future.result()
            except CancelledError:
                self.binding = None
        self.layers = []
        if self.binding is None:
            self.status = self.error or (
                "Loading saved tracking…" if self.pending is not None else ""
            )
            return ()
        lineage = self.binding.reader["sources"][self.binding.sensor_id].get("lineage", [])
        # Keep previous states as explicitly historical evidence between results;
        # never extrapolate a saved state to the intermediate scene pose.
        eligible = [
            (row["scene_frame_id"], i)
            for i, row in enumerate(lineage)
            if row["scene_frame_id"] <= step
        ]
        index = max(eligible)[1] if eligible else None
        if index is None:
            self.status = "No saved tracking epoch precedes this scene."
            return ()
        at_epoch = lineage[index]["scene_frame_id"] == step
        self.status = f"Saved 3D tracks at {self.binding.reader.times[index]:.6g} s" + (
            "" if at_epoch else " · historical states; no extrapolation"
        )
        history = (
            None
            if state_value(state, "full_history")
            else int(state_value(state, "history_frames"))
        )
        arrows = bool(state_value(state, "show_velocity_arrows"))
        labels = bool(state_value(state, "show_track_labels"))
        key = (id(self.binding), index, history, arrows, labels, at_epoch)
        if key != self._render_key:
            layers, text = saved_track_layers(
                self.binding,
                index,
                history_count=history,
                arrows=arrows,
                labels=labels,
                service=service,
                at_epoch=at_epoch,
            )
            self._cached_layers = layers
            self._snapshots = tuple([layer.to_render_object() for layer in layers] + text)
            self._render_key = key
        self.layers = self._cached_layers
        return self._snapshots

    def close(self):
        self.cancelled.set()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self.pending = None
        self._loaded = False
