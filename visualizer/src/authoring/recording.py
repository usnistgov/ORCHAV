"""Read-only actor inspection from authoritative, selectively loaded frames."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from shared.frames.contracts import FrameReadRequest
from shared.frames.hdf5 import HDF5FormatHandler
from shared.scenarios.paths import resolve_actor_resource

from ..io.frame_sources import FileSource
from ..services.trajectory_load_service import TrajectoryLoadCoordinator
from .domain import ActorRole
from .frame_geometry import FrameGeometry


class RecordingPreview:
    """Keep three device/target projections, without loading paths or sensing arrays."""

    def __init__(self, scenario, root: Path, assets):
        self.scenario = scenario
        self.assets = assets
        self.root = root
        mapping = scenario.source_snapshot.to_mapping()
        directory = mapping.get("data", {}).get("files", {}).get("directory", "frames")
        self.directory = directory
        self.reader = HDF5FormatHandler(root, frames_subdir=directory)
        self.steps = self.reader.frame_ids_snapshot or ()
        if not self.steps:
            self.reader.close()
            raise ValueError(f"Recorded poses require a frame manifest under {root / directory}")
        self._frames = OrderedDict()
        self.trajectory_positions = {}
        self.trajectory_loader = None
        self.trajectory_error = ""
        self._closed = False

    def start_trajectories(self, changed):
        """Share normal playback's progressive position scan across both views.

        Start on the GUI thread so coordinator publications are delivered there.
        The scan owns a separate reader and closes it on completion/cancellation;
        it never contends with the displayed frame's bounded projection cache.
        """
        if self._closed or self.trajectory_loader is not None:
            return
        self.trajectory_loader = TrajectoryLoadCoordinator()

        def updated(snapshot):
            if self._closed:
                return
            try:
                first = self.frame(0)
            except (OSError, ValueError, KeyError) as exc:
                failed(str(exc))
                return
            for actor in self.scenario.actors:
                if actor.role is ActorRole.TARGET:
                    track = snapshot.target_positions.get(actor.name, ())
                else:
                    names = getattr(first, f"{actor.role.value}_names")
                    tracks = getattr(snapshot, f"{actor.role.value}_positions")
                    track = tracks.get(names.index(actor.name), ()) if actor.name in names else ()
                # The shared scan publishes a coarse overview before filling in
                # intermediate samples. Connect them in recorded frame order.
                self.trajectory_positions[actor.id] = tuple(p[1:] for p in sorted(track))
            changed()

        def failed(message):
            if not self._closed:
                self.trajectory_error = message
                changed()

        self.trajectory_loader.snapshot_updated.connect(updated)
        self.trajectory_loader.loading_complete.connect(updated)
        self.trajectory_loader.error_occurred.connect(failed)
        self.trajectory_loader.load(
            FileSource(self.root, self.directory, "hdf5"), close_source=True
        )

    def frame(self, step: int):
        index = self.steps[min(max(0, step), len(self.steps) - 1)]
        if index not in self._frames:
            request = FrameReadRequest(components=frozenset({"devices", "targets"}))
            self._frames[index] = self.reader.load_frame_projection(index, request).frame
            while len(self._frames) > 3:
                self._frames.popitem(last=False)
        self._frames.move_to_end(index)
        return self._frames[index]

    def sample(self, actor, step: int):
        from .compiler import ActorSamples

        frame = self.frame(step)
        if actor.role is ActorRole.TARGET:
            item = next((m for m in frame.targets_metadata if m.get("name") == actor.name), None)
            if item is None:
                raise ValueError(f"Recorded frame {step} has no actor {actor.name!r}")
            position = item["current_position"]
            orientation = item.get("orientation", (0, 0, 0))
        else:
            prefix = actor.role.value
            names = getattr(frame, f"{prefix}_names")
            index = names.index(actor.name)
            position = getattr(frame, f"{prefix}_positions")[index]
            orientation = getattr(frame, f"{prefix}_orientations")[index]
        return ActorSamples(
            (tuple(float(v) for v in position),),
            (tuple(float(v) for v in np.degrees(orientation)),),
            has_physical_velocity=False,
        )

    def target_frames(self, actor):
        """Select each mesh by the recorded filename, never by timeline modulo."""

        def prepare(step):
            frame = self.frame(step)
            item = next(m for m in frame.targets_metadata if m.get("name") == actor.name)
            target = actor.target
            if target is None:
                return self.assets.build_target_assets(actor, (), self.sample(actor, step))[0]
            directory = resolve_actor_resource(
                target.mesh_directory,
                scenario_root=self.root,
                project_root=self.assets.project_root,
            )
            filename = str(item["mesh_file"])
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory):
                raise ValueError(
                    f"Recorded mesh for {actor.name!r} escapes {directory}: {filename}"
                )
            displayed = actor.with_changes(
                target=replace(
                    target, scale=float(item.get("scale", target.scale)), mesh_animation=False
                )
            )
            return replace(
                self.assets.build_target_assets(displayed, (path,), self.sample(actor, step))[0],
                step_index=step,
                metadata_source="recorded_frame",
            )

        result = FrameGeometry(len(self.steps), prepare)
        result[0]
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.trajectory_loader is not None:
            self.trajectory_loader.shutdown(timeout=0)
            self.trajectory_loader.deleteLater()
        self.reader.close()
        self._frames.clear()
        self.trajectory_positions.clear()
