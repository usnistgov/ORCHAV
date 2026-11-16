"""Use the main transport for saved fusion epochs without native recordings."""

from PySide6.QtCore import QSignalBlocker

from ..playback import timestamp_interval_ms


class FusionPlayback:
    """Present saved world states; never synthesize frame or measurement data."""

    def __init__(self, window):
        self.window = window
        self.visualizer = window.visualizer
        self.steps = tuple(range(len(window.reader.times)))
        viz = self.visualizer
        self.previous_count = viz.total_animation_steps
        self.previous_step = viz.animation_step
        if viz.animation_running:
            viz.animation_controller.toggle_animation()
        viz.result_playback = self
        self._navigation(len(self.steps))
        viz.update_frame(window.timeline.value())

    def _navigation(self, count):
        viz = self.visualizer
        blockers = [QSignalBlocker(w) for w in (viz.step_slider, viz.frame_input)]
        try:
            viz.total_animation_steps = count
            viz.ui_manager.update_total_steps(count)
        finally:
            for blocker in blockers:
                blocker.unblock()

    def interval_ms(self, current, following):
        times = self.window.reader.times
        return timestamp_interval_ms(float(times[current]), float(times[following]))

    def present(self, index):
        window = self.window
        if index not in self.steps:
            return False
        window._following_scene = True
        try:
            if window.timeline.value() == index:
                window._draw_world()
            else:
                window.timeline.setValue(index)
        finally:
            window._following_scene = False
        return True

    def close(self):
        """Return ownership before a new scenario or normal recording takes over."""
        viz = self.visualizer
        if getattr(viz, "result_playback", None) is not self:
            return
        if viz.animation_running:
            viz.animation_controller.toggle_animation()
        viz.result_playback = None
        self._navigation(self.previous_count)
        viz.animation_step = self.previous_step
        viz.set_state(step=self.previous_step)
        viz._sync_animation_navigation_widgets(viz.get_animation_step_index(self.previous_step))
