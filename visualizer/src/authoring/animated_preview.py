"""Bounded inspection of manifest-owned motion at exact physical timestamps."""

from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

from generator.core.scenario_actors.types import Timeline
from shared.animated_target import (
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    load_animated_target,
)
from shared.scenarios.paths import resolve_actor_resource

from .frame_geometry import FrameGeometry


class AnimatedPreview:
    """Keep the full position track, with at most three bound geometry samples."""

    def __init__(self, actor, timeline, root, assets, *, progress=None):
        target = actor.target
        self.actor = actor
        self.assets = assets
        self.timeline = Timeline(timeline.steps, timeline.duration_s)
        path = resolve_actor_resource(
            target.path, scenario_root=root, project_root=assets.project_root
        )
        self.asset = load_animated_target(path)
        playback = target.playback
        self.context = AnimatedTargetBindingContext(
            target_id=actor.name,
            scenario_clock_id=playback.scenario_clock_id,
            motion_clock_id=playback.motion_clock_id,
            pulse_clock_id=None,
            playback_rate=playback.playback_rate,
            playback_offset_s=playback.playback_offset_s,
            retiming_policy="affine_scenario_to_motion",
        )
        self._samples = OrderedDict()
        centers = []
        for start in range(0, timeline.steps, 128):
            if progress is not None:
                progress(f"Preparing {actor.name} trajectory: {start}/{timeline.steps} positions")
            # Share canonical placement without constructing normals, mesh
            # payloads or binding hashes for timestamps that are not displayed.
            times = tuple(
                self._binding_time(step) for step in range(start, min(start + 128, timeline.steps))
            )
            centers.extend(
                tuple(float(v) for v in center)
                for center in self.asset.sample_centers(self.context, times)
            )
        self.trajectory_positions = tuple(centers)
        self.binding(0)
        if progress is not None:
            progress(
                f"Prepared {actor.name} trajectory: {timeline.steps}/{timeline.steps} positions"
            )

    def _binding_time(self, step):
        t = self.timeline.timestamps_s[step]
        return AnimatedTargetBindingTime(
            sample_id=f"{self.context.scenario_clock_id}:time:{float(t).hex()}",
            scenario_time_s=t,
            motion_time_s=self.context.playback_rate * t + self.context.playback_offset_s,
        )

    def binding(self, step):
        if step not in self._samples:
            self._samples[step] = self.asset.bind(self.context, (self._binding_time(step),))
            while len(self._samples) > 3:
                self._samples.popitem(last=False)
        self._samples.move_to_end(step)
        return self._samples[step]

    def sample(self, step):
        from .compiler import ActorSamples

        return ActorSamples(
            (self.trajectory_positions[step],),
            ((0.0, 0.0, 0.0),),
            has_physical_velocity=False,
        )

    def target_frames(self):
        def prepare(step):
            animation = SimpleNamespace(
                asset=self.asset,
                binding=self.binding(step),
                centers_m=self.sample(step).positions,
            )
            return replace(
                self.assets.build_animated_target_assets(self.actor, animation)[0],
                step_index=step,
            )

        return FrameGeometry(self.timeline.steps, prepare)
