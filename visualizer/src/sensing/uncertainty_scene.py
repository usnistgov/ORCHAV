"""Opt-in, read-only model covariance overlays for exact saved tracker epochs."""

from functools import lru_cache

import numpy as np

from ..model import RenderObject, Transform
from ..services.sensing_overlay_sync import SensingOverlaySync
from ..types.render_payloads import LineSetPayload, MaterialPayload, MeshPayload, PointCloudPayload
from .display_style import color_rgba, identity_color
from .tracking_presentation import tracking_snapshot
from .tracking_uncertainty import covariance_ellipsoid, native_epoch_valid


@lru_cache(maxsize=1)
def unit_ellipsoid_geometry():
    """One small shared sphere and three principal-plane outlines, cached once."""
    longitude = np.linspace(0, 2 * np.pi, 33)
    latitude = np.linspace(0, np.pi, 17)
    vertices = np.array(
        [
            [np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]
            for p in latitude
            for t in longitude
        ],
        dtype=np.float32,
    )
    triangles = []
    for i in range(16):
        for j in range(32):
            a = i * 33 + j
            triangles.extend(((a, a + 33, a + 1), (a + 1, a + 33, a + 34)))
    mesh = MeshPayload(
        vertices=vertices,
        triangles=np.asarray(triangles, dtype=np.int32),
        normals=vertices,
        cache_key="tracking-model-unit-ellipsoid-v1",
    )
    angle = np.linspace(0, 2 * np.pi, 65)
    points = np.concatenate(
        [
            np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle))),
            np.column_stack((np.cos(angle), np.zeros_like(angle), np.sin(angle))),
            np.column_stack((np.zeros_like(angle), np.cos(angle), np.sin(angle))),
        ]
    ).astype(np.float32)
    edges = np.array(
        [(65 * k + i, 65 * k + i + 1) for k in range(3) for i in range(64)], dtype=np.int32
    )
    return mesh, LineSetPayload(points=points, lines=edges)


class TrackingUncertaintyScene:
    """No timers, readers or inference; consume already accepted saved data only."""

    MAX_TRACKS = 100

    def __init__(self):
        self.overlay = SensingOverlaySync()
        self.frame = None
        self.binding = None
        self.time_s = None
        self.source = None
        self.status = "Tracking uncertainty off"
        self._key = None
        self._snapshots = ()

    def clear(self, renderer):
        self.frame = self.binding = self.time_s = self.source = self._key = None
        self._snapshots = ()
        self.status = "Tracking uncertainty off"
        return self.overlay.clear(renderer)

    def sync(
        self,
        renderer,
        frame,
        *,
        binding=None,
        time_s=None,
        source=None,
        selection=None,
        sigma=1.0,
        selected_only=True,
    ):
        """Bind one result and source; a replacement clears all previous geometry."""
        self.frame, self.binding, self.time_s, self.source = frame, binding, time_s, source
        return self.refresh(renderer, selection=selection, sigma=sigma, selected_only=selected_only)

    def refresh(self, renderer, *, selection=None, sigma=1.0, selected_only=True):
        from ..services.sensing_service import SensingService
        from .fusion_inspection import FusionSelection
        from .selection import selection_scope

        if getattr(renderer, "renderer_id", None) != "pygfx":
            self.status = "Tracking uncertainty requires the pygfx renderer"
            return self.overlay.clear(renderer)
        frame = self.frame or {}
        sensing = frame.get("sensing") or {}
        try:
            if self.binding is None:
                valid, reason = native_epoch_valid(sensing, self.time_s)
                if not valid:
                    raise ValueError(reason)
            snapshot = tracking_snapshot(sensing, saved_tracking=self.binding, time_s=self.time_s)
            if not snapshot.complete_states:
                raise ValueError("No complete saved Cartesian tracker covariance")
            track_id = None
            # With no selection, inspect every eligible track. A stale or
            # unrelated selection still goes through the source checks below.
            selected_only = selected_only and selection is not None
            if selected_only:
                if self.binding is not None:
                    if (
                        not isinstance(selection, FusionSelection)
                        or selection.run_id != self.binding.reader.inspection_id
                    ):
                        raise ValueError("Select a track from this saved tracking run")
                elif (
                    selection is None
                    or getattr(selection, "scope", None) != selection_scope(sensing)
                    or (getattr(selection, "source", ()) and selection.source != self.source)
                ):
                    raise ValueError("Select a track from this recording and result")
                if self.binding is None and selection.detection_index is not None:
                    acquisition = sensing.get("temporal", {}).get("acquisition", {})
                    if selection.acquisition_id != acquisition.get(
                        "acquisition_id"
                    ) or selection.detection_collection != sensing.get(
                        "_display_detection_collection", "detections"
                    ):
                        raise ValueError(
                            "Selected detection belongs to another acquisition or collection"
                        )
                track_id = selection.track_id
                if track_id is None:
                    raise ValueError("Selected detection has no associated saved track")
            states = [s for s in snapshot.states if not selected_only or s["track_id"] == track_id]
            # No file reads: epoch states and pose come from the already bound frame.
            pose = None
            if any(s.get("position_frame") == "receiver local" for s in states):
                pose = SensingService._resolve_radar_pose(frame)
            key = (
                id(self.frame),
                id(self.binding),
                self.source,
                self.time_s,
                sigma,
                selected_only,
                track_id,
            )
            if key == self._key:
                return self.overlay.sync(renderer, self._snapshots)
            objects, reasons = [], []
            eligible = []
            for state in states:
                ellipsoid, reason = covariance_ellipsoid(
                    state,
                    sigma_scale=sigma,
                    receiver_position=None if pose is None else pose[0],
                    receiver_rotation=None if pose is None else pose[1],
                )
                if ellipsoid is None:
                    reasons.append(reason)
                else:
                    eligible.append((state, ellipsoid))
            eligible.sort(key=lambda item: str(item[0]["track_id"]))
            for state, ellipsoid in eligible[: self.MAX_TRACKS]:
                objects.extend(self._objects(state, ellipsoid))
            self._snapshots = tuple(objects)
            self._key = key
            count = min(len(eligible), self.MAX_TRACKS)
            self.status = f"{count} model covariance ellipsoid(s) · {sigma:g}σ principal axes; not measured error or containment probability"
            if len(eligible) > self.MAX_TRACKS:
                self.status += (
                    f" · display capped at {self.MAX_TRACKS} of {len(eligible)} eligible tracks"
                )
            if any(np.any(item[1].radii == 0) for item in eligible[: self.MAX_TRACKS]):
                self.status += " · zero-volume covariance shown as exact outlines/points"
            if reasons:
                self.status += f" · {len(reasons)} unavailable: {reasons[0]}"
            if not states:
                self.status = "No saved track at this exact result epoch"
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            self._snapshots = ()
            self._key = None
            self.status = f"Tracking uncertainty unavailable: {exc}"
        return self.overlay.sync(renderer, self._snapshots)

    @staticmethod
    def _objects(state, ellipsoid):
        mesh, lines = unit_ellipsoid_geometry()
        matrix = np.eye(4)
        matrix[:3, :3] = ellipsoid.basis @ np.diag(ellipsoid.radii)
        matrix[:3, 3] = ellipsoid.center
        transform = Transform(matrix)
        color = color_rgba(identity_color(state["track_id"]))[:3]
        tentative = not bool(state.get("confirmed", True))
        coasting = state.get("predicted_only") is True
        opacity = 0.10 if tentative else (0.12 if coasting else 0.18)
        name = f"sensing_uncertainty_{state['track_id']}"
        metadata = {
            "pickable": False,
            "model_covariance": True,
            "status": state.get("status"),
            "source_key": state.get("source_key"),
        }
        center = RenderObject(
            name + "_center",
            PointCloudPayload(points=np.asarray([ellipsoid.center], dtype=np.float32)),
            MaterialPayload(base_color=(*color, 1.0), point_size=7),
            metadata=metadata,
        )
        if np.any(ellipsoid.radii == 0):
            # A singular model covariance has no volume. Project its exact
            # outline coordinates rather than give a mesh a singular transform
            # or manufacture a small positive variance to make it renderable.
            if not np.any(ellipsoid.radii > 0):
                return (center,)
            projected = lines.points @ matrix[:3, :3].T + ellipsoid.center
            flat_outline = LineSetPayload(
                points=np.asarray(projected, dtype=np.float32), lines=lines.lines
            )
            return (
                RenderObject(
                    name + "_outline",
                    flat_outline,
                    MaterialPayload(base_color=(*color, 0.55), line_width=1.3),
                    metadata=metadata,
                ),
                center,
            )
        return (
            RenderObject(
                name,
                mesh,
                MaterialPayload(base_color=(*color, opacity), shader="unlit"),
                transform,
                metadata=metadata,
            ),
            RenderObject(
                name + "_outline",
                lines,
                MaterialPayload(base_color=(*color, 0.55), line_width=1.3),
                transform,
                metadata=metadata,
            ),
            center,
        )
