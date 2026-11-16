"""Temporary presentation automation for an already loaded local scene.

Camera motion and recorded-frame playback retain their existing owners. This
coordinator cycles visual looks and already installed environments, and restores
its temporary settings before handing control back to the user.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from time import perf_counter

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)
from shiboken6 import isValid

from shared.logging import get_logger
from shared.statistics.themes import theme_manager

from ..lighting.assets import cached_environment, environment_catalog
from ..materials.texture_policy import textures_locked_off
from ..materials.view_styles import PRESETS
from .demo_tour import scenario_tour
from .mpc_appearance_service import PRESETS as MPC_PRESETS
from .mpc_appearance_service import SCENE_MPC_PRESETS

logger = get_logger("orchav.demo")

DEFAULT_APPEARANCE_CYCLE = ("Detailed", "Paper", "Dark", "Glow")


class DemoService(QObject):
    """Coordinate reversible presentation effects and independent camera control."""

    changed = Signal()

    def __init__(self, visualizer):
        super().__init__(visualizer if isinstance(visualizer, QObject) else None)
        self.viz = visualizer
        self.active = False
        self.status = "Ready. Frame a view, then start the demo."
        self._snapshot = None
        self._camera_tools = None
        self._orbit_owned = False
        self._tour_owned = False
        self._targets_owned = False
        self._target_interval = 5.0
        self._next_target = 0.0
        self._exhibition = None
        self._presentation_view = None
        self.camera_paused = False
        self._orbit_speed = 3.0
        self._tour_elapsed = 0.0
        self._look_index = 0
        self._look_interval = 5.0
        self._next_look = 0.0
        self._include_wireframe = False
        self._cycle_lighting = False
        self._appearance_cycle = ()
        self._environment_cycle = ()
        self._environment_index = 0
        self._scenario_sunset = False
        self._sunset_duration = 5.0
        self._sunset_started = None
        self._sunset_progress = 0.0
        self._showcase = None
        self._sensing_story = None
        self._showcase_phase = None
        self._showcase_returned = False
        self.last_switch_duration_s = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.tick)

    def start(
        self,
        *,
        orbit: bool = True,
        tour: bool = False,
        cycle_targets: bool = False,
        exhibition: bool = False,
        exhibition_content: str = "actor",
        clean_view: bool = False,
        captions: bool = False,
        sensing_story: bool = False,
        playback: bool = True,
        cycle_lighting: bool = True,
        include_wireframe: bool = False,
        cycle_ibl: bool = False,
        cycle_appearance: bool = False,
        appearance_presets: Sequence[str] | None = None,
        ibl_environments: Sequence[str] | None = None,
        showcase: bool = False,
        presentation: str | None = None,
        scenario_dusk: bool = False,
        scenario_sunset: bool = False,
        orbit_speed: float = 3.0,
        look_interval: float = 5.0,
        target_interval: float = 5.0,
        sunset_duration: float = 5.0,
    ) -> bool:
        """Start from the current view, cycling selected presets and installed skies.

        Omitted selections use the default cycles. Explicit selections retain
        their order, follow the current look/sky, and must contain a usable item.
        """
        if self.active:
            return True
        sensing_exhibition = exhibition and exhibition_content == "sensing"
        if exhibition_content not in {"actor", "sensing"}:
            return self._unavailable("Choose valid exhibition views.")
        if sensing_exhibition and not self.available_options().get("sensing_presentation", False):
            return self._unavailable(
                "Sensing presentation needs a prepared local sensing recording with a saved range–velocity map."
            )
        if sensing_story:
            from .demo_sensing_story import DemoSensingStory

            if not DemoSensingStory.available(self.viz):
                return self._unavailable("Sensing story needs a prepared local sensing recording.")
            orbit = tour = cycle_targets = showcase = False
            cycle_lighting = cycle_ibl = cycle_appearance = False
            scenario_dusk = scenario_sunset = False
            presentation = None
            playback = True
        if showcase:
            # One frozen-frame tour, then ordinary forward playback. The caller's
            # independent cycle options remain available for continuous demos.
            playback = cycle_appearance = True
            cycle_lighting = cycle_ibl = scenario_dusk = scenario_sunset = False
            presentation = None
        # The fixed presentation look owns its lighting for the entire demo.
        if presentation is not None and presentation not in PRESETS:
            return self._unavailable("Choose a valid presentation preset.")
        if presentation is not None:
            cycle_lighting = cycle_ibl = cycle_appearance = False
            scenario_dusk = scenario_sunset = False
        v, renderer = self.viz, getattr(self.viz, "renderer", None)
        if getattr(renderer, "renderer_id", None) != "pygfx":
            return self._unavailable("Demo mode requires the pygfx renderer.")
        if not getattr(v, "ready", False) or getattr(v, "scenario", None) is None:
            return self._unavailable("Open a scene before starting the demo.")
        if getattr(getattr(v, "live_preview_service", None), "enabled", False):
            return self._unavailable("Finish actor editing before starting the demo.")
        camera_motion = orbit or tour or cycle_targets
        if cycle_targets and (orbit or tour):
            return self._unavailable("Choose target cycling, orbit or a camera tour separately.")
        if cycle_targets and not self._target_rows():
            return self._unavailable("No recorded targets are available to follow.")
        if (
            camera_motion
            and getattr(getattr(v, "app_state", None), "camera_mode", "overview") != "overview"
        ):
            return self._unavailable("Choose Overview camera mode to use demo camera motion.")
        if not (
            camera_motion
            or playback
            or cycle_lighting
            or cycle_ibl
            or cycle_appearance
            or scenario_dusk
            or scenario_sunset
            or exhibition
            or clean_view
            or captions
            or presentation is not None
        ):
            return self._unavailable("Enable camera motion, playback or a look cycle.")
        if not (
            0.2 <= orbit_speed <= 12
            and 1 <= look_interval <= 120
            and 1 <= target_interval <= 600
            and 1 <= sunset_duration <= 600
        ):
            return self._unavailable("Choose valid camera and lighting intervals.")
        if playback and getattr(v.scenario, "data_mode", "files") != "files":
            return self._unavailable(
                "Demo playback uses local recorded frames. Turn playback off for this source."
            )
        available_steps = getattr(v, "get_available_animation_steps", None)
        if playback and callable(available_steps) and not available_steps():
            return self._unavailable(
                "No recorded frames are available. Turn playback off to show the scene."
            )
        if showcase and (not callable(available_steps) or len(available_steps()) < 2):
            return self._unavailable("Showcase needs at least two recorded frames.")
        try:
            tour_data = scenario_tour(v.scenario) if tour else None
        except ValueError as exc:
            return self._unavailable(str(exc))
        if tour and tour_data is None:
            return self._unavailable("This scenario has no authored demo camera tour.")
        workspace = getattr(v, "_visualization_workspace", None)
        if clean_view and exhibition:
            return self._unavailable("Choose the clean view or exhibition layout separately.")
        if (clean_view or captions) and not self.available_options()["clean_view"]:
            return self._unavailable("Presentation view requires the embedded scene viewport.")
        if exhibition and (
            getattr(workspace, "splitter", None) is None or workspace.splitter.count() < 2
        ):
            return self._unavailable("Exhibition layout requires the embedded scene viewport.")

        tools = v.camera_controller.presentation()
        if tools is not None and tools.exporting:
            return self._unavailable("Finish camera export before starting the demo.")
        if camera_motion and (tools is None or renderer.get_camera_state() is None):
            return self._unavailable("The camera is not ready for demo motion.")
        if camera_motion and any((tools.autorotate, tools.sequence, tools.transition)):
            return self._unavailable("Stop the current camera animation before starting the demo.")
        tools = tools if camera_motion else None

        loop = getattr(v, "loop_cb", None)
        appearance = getattr(v, "appearance_preset_service", None)
        if (cycle_appearance or presentation is not None) and appearance is None:
            return self._unavailable("Appearance presets are unavailable for this scene.")
        mpc = getattr(v, "mpc_appearance_service", None)
        if presentation is not None and mpc is None:
            return self._unavailable("MPC appearance presets are unavailable for this scene.")
        manage_mpc = mpc is not None and (cycle_appearance or presentation is not None)
        appearances = ()
        if cycle_appearance:
            names = (
                (tuple(PRESETS) if showcase else DEFAULT_APPEARANCE_CYCLE)
                if appearance_presets is None
                else tuple(appearance_presets)
            )
            if any(name not in PRESETS for name in names):
                return self._unavailable("Choose valid appearance presets for the cycle.")
            appearances = tuple(
                name
                for name in dict.fromkeys(names)
                if not (PRESETS[name]["textures"] and textures_locked_off())
            )
            if not appearances:
                return self._unavailable("Select at least one available appearance preset.")
        environments = []
        if cycle_ibl:
            current_sky = renderer.get_ibl_name()
            if current_sky is None:
                return self._unavailable("The renderer's environment is not ready for sky cycling.")
            available = dict(self.available_environments())
            names = tuple(available) if ibl_environments is None else tuple(ibl_environments)
            if any(name not in available and name != current_sky for name in names):
                return self._unavailable(
                    "A selected sky is no longer installed. Choose installed skies in Demo."
                )
            environments = [(current_sky, "current sky")]
            environments.extend(
                (name, available[name]) for name in dict.fromkeys(names) if name != current_sky
            )
            if len(environments) < 2:
                return self._unavailable(
                    "Install another sky in Rendering to cycle environments, or turn sky cycling off."
                    if ibl_environments is None
                    else "Select at least one installed sky other than the current environment."
                )
        visual_cycle = cycle_lighting or cycle_ibl or cycle_appearance
        manage_appearance = visual_cycle or presentation is not None
        dusk = getattr(renderer, "get_dusk_state", lambda: {})()
        if (scenario_dusk or scenario_sunset) and not dusk.get("available", False):
            return self._unavailable("This scenario has no authored blue-hour lighting.")
        manage_dusk = dusk.get("available", False) and (
            manage_appearance or scenario_dusk or scenario_sunset
        )
        dusk_progress = float(dusk.get("progress", 1.0 if dusk.get("enabled", False) else 0.0))
        if manage_dusk and dusk.get("enabled", False):
            # Capture the ordinary appearance beneath the recipe. Capturing
            # dusk itself would make it the next recipe's restoration baseline.
            renderer.set_dusk_enabled(False)
        try:
            self._snapshot = {
                "renderer": renderer,
                "scenario": v.scenario,
                "source": getattr(v, "current_scenario_path", None),
                "source_epoch": (
                    getattr(v, "_mpc_presented_source_epoch", 0)
                    if showcase or sensing_story or sensing_exhibition
                    else None
                ),
                "playback": playback,
                "running": bool(getattr(v, "animation_running", False)),
                "loop": loop.isChecked() if loop is not None else None,
                "play_direction": (
                    getattr(v, "play_direction", 1) if showcase or sensing_story else None
                ),
                "history": list(tools.history) if tools is not None else None,
                "target_follow": (
                    {
                        "selection": deepcopy(v.target_focus_dropdown.currentData()),
                        "mode": v.app_state.camera_mode,
                        "framing": {
                            key: deepcopy(getattr(tools, key))
                            for key in ("follow_heading", "follow_offset", "follow_world_yaw")
                        },
                    }
                    if cycle_targets
                    else None
                ),
                "views": deepcopy(tools.views) if tour else None,
                "shots": deepcopy(tools.shots) if tour else None,
                "lights": deepcopy(renderer.get_light_rig_state()) if manage_appearance else None,
                "ibl": renderer.get_ibl_intensity() if manage_appearance else None,
                "wireframe": renderer.get_wireframe() if manage_appearance else None,
                "sky": renderer.get_ibl_name() if cycle_ibl else None,
                "skybox": renderer.get_skybox_visible() if cycle_ibl else None,
                "shadow_quality": renderer.get_shadow_quality() if cycle_ibl else None,
                "cycle_ibl": cycle_ibl,
                "dusk": (
                    {
                        "enabled": dusk.get("enabled", False),
                        "progress": dusk_progress,
                        "current_look": scenario_dusk
                        or scenario_sunset
                        or dusk.get("enabled", False),
                        "current_progress": 1.0 if scenario_dusk else dusk_progress,
                    }
                    if manage_dusk
                    else None
                ),
                "appearance": appearance,
                "full_appearance": (
                    deepcopy(appearance.capture())
                    if cycle_appearance or presentation is not None
                    else None
                ),
                "presentation": presentation,
                "presentation_state": (
                    {
                        key: deepcopy(getattr(v.app_state, key))
                        for key in (
                            "color_mode",
                            "topk_render_enabled",
                            "topk_render_max_paths",
                            "mpc_visibility",
                            "antialiasing_mode",
                            "texture_mipmaps",
                        )
                    }
                    if manage_mpc
                    else None
                ),
                "mpc": mpc,
                "mpc_appearance": mpc.capture() if manage_mpc else None,
                "owned_palettes": {},
                "appearance_metadata": (
                    {
                        key: deepcopy(getattr(appearance, key))
                        for key in ("active_name", "previous", "last_applied", "notice")
                    }
                    if appearance is not None and manage_appearance
                    else None
                ),
            }
        finally:
            if manage_dusk and dusk.get("enabled", False):
                self._restore_dusk_level(renderer, True, dusk_progress)
        self._camera_tools = tools
        self._orbit_owned = False
        self._tour_owned = False
        self._targets_owned = False
        self.camera_paused = False
        self._orbit_speed = orbit_speed
        self._tour_elapsed = 0.0
        self._target_interval = target_interval
        self._look_index = 0
        self._look_interval = look_interval
        self._next_look = perf_counter() + look_interval
        self._include_wireframe = include_wireframe
        self._cycle_lighting = cycle_lighting and not (cycle_ibl or cycle_appearance)
        self._appearance_cycle = appearances
        self._environment_cycle = tuple(environments)
        self._environment_index = 0
        self._scenario_sunset = bool(scenario_sunset)
        self._sunset_duration = sunset_duration
        self._sunset_started = None
        self.last_switch_duration_s = 0.0
        self._showcase_phase = "presets" if showcase else None
        self._showcase_returned = False
        self.active = True
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        try:
            if showcase:
                from .demo_mpc_timeline import DemoMpcTimeline

                self._set_playback(False)
                self._showcase = DemoMpcTimeline(v)
            if presentation is not None:
                self._apply_presentation(self._snapshot, presentation)
            else:
                self._apply_demo_dusk(self._snapshot)
            if tour:
                self._tour_owned = True
                tools.views = tour_data["views"]
                tools.shots = tour_data["shots"]
                tools.start_sequence(loop=True)
            elif cycle_targets:
                self._targets_owned = True
                tools.follow_heading = "fixed"
                v.set_state(camera_mode="follow")
                self._cycle_target(perf_counter(), initial=True)
            elif tools is not None:
                self._orbit_owned = True
                tools.start_orbit(renderer.get_camera_state().lookat, speed=orbit_speed)
            if exhibition:
                from .demo_layout import ExhibitionLayout

                if sensing_exhibition:
                    from .demo_sensing_layout import SensingExhibitionLayout

                    self._exhibition = SensingExhibitionLayout(
                        v, lambda reason: self.stop(reason=reason)
                    )
                else:
                    self._exhibition = ExhibitionLayout(v)
                self._exhibition.enter()
                if not self.active:
                    return False
            if clean_view or captions or sensing_exhibition:
                from .demo_presentation import DemoPresentation

                self._presentation_view = DemoPresentation(v, self)
                self._presentation_view.enter(
                    clean_view=clean_view,
                    captions=captions,
                    compact=sensing_exhibition,
                    playback=sensing_exhibition and playback,
                )
            if playback and not showcase:
                if loop is not None:
                    loop.setChecked(True)
                if sensing_story:
                    v.play_direction = 1
                self._set_playback(True)
            # Publish a full dwell interval after camera/layout/light setup.
            now = perf_counter()
            self._next_look = now + self._look_dwell()
            self._next_target = now + target_interval
            if self._sunset_started is not None:
                self._sunset_started = now
            if self._showcase is not None:
                self._showcase.restart(now, self._look_dwell())
                self._timer.setInterval(33)
            if sensing_story:
                self._sensing_story = DemoSensingStory(v, interval=look_interval)
                self._sensing_story.start(now)
            self._timer.start()
            label = f" · {tour_data['label']}" if tour else ""
            look = presentation if presentation is not None else f"current look{self._dusk_label()}"
            self.status = f"Demo running{label} · {look} · drag to explore; Esc stops Demo."
            if showcase:
                self.status = "Showcase · current look · slowed MPC travel, then recorded playback."
            if sensing_story:
                self.status = "Sensing story · saved detections and tracks · Esc stops Demo."
            self.changed.emit()
            return True
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.warning("Could not start demo: %s", exc)
            self.stop(reason=f"Demo stopped: {exc}")
            return False

    def available_environments(self) -> tuple[tuple[str, str], ...]:
        """List verified installed skies other than the current one, without downloads."""
        renderer = getattr(self.viz, "renderer", None)
        current = getattr(renderer, "get_ibl_name", lambda: None)()
        return tuple(
            (asset.id, asset.label)
            for asset in environment_catalog().values()
            if asset.id != current and cached_environment(asset) is not None
        )

    def available_options(self) -> dict[str, bool]:
        """Describe usable controls for the loaded scene without changing any owner."""
        v, renderer = self.viz, getattr(self.viz, "renderer", None)
        ready = bool(
            getattr(renderer, "renderer_id", None) == "pygfx"
            and getattr(v, "ready", False)
            and getattr(v, "scenario", None) is not None
            and not getattr(getattr(v, "live_preview_service", None), "enabled", False)
        )
        controller = getattr(v, "camera_controller", None)
        tools = controller.presentation() if ready and controller is not None else None
        if tools is not None and tools.exporting:
            ready = False
        camera = bool(
            ready
            and tools is not None
            and getattr(v.app_state, "camera_mode", "overview") == "overview"
            and renderer.get_camera_state() is not None
            and not any((tools.autorotate, tools.sequence, tools.transition))
        )
        try:
            tour = bool(camera and scenario_tour(v.scenario))
        except ValueError:
            tour = False
        steps = getattr(v, "get_available_animation_steps", lambda: ())() if ready else ()
        playback = bool(
            ready and getattr(v.scenario, "data_mode", "files") == "files" and len(steps) > 1
        )
        appearance = ready and getattr(v, "appearance_preset_service", None) is not None
        workspace = getattr(v, "_visualization_workspace", None)
        splitter = getattr(workspace, "splitter", None)
        embedded = bool(ready and splitter is not None and splitter.count() >= 2)
        host = getattr(workspace, "viewport_host", None)
        clean = bool(embedded and isinstance(host, QWidget) and host.layout() is not None)
        from .demo_mpc_timeline import DemoMpcTimeline
        from .demo_sensing_layout import SensingExhibitionLayout
        from .demo_sensing_story import DemoSensingStory

        sensing_ready = bool(playback and DemoSensingStory.available(v))
        panel = getattr(getattr(v, "ui_manager", None), "panels", {}).get("sensing")
        dashboard = getattr(panel, "_sensing_window", None)

        return dict(
            ready=ready,
            orbit=camera,
            tour=tour,
            targets=bool(camera and self._target_rows()),
            playback=playback,
            showcase=bool(playback and appearance and DemoMpcTimeline.available(v)),
            sensing_story=sensing_ready,
            sensing_presentation=bool(
                sensing_ready
                and embedded
                and appearance
                and not textures_locked_off()
                and SensingExhibitionLayout.supports_dashboard(dashboard)
            ),
            appearance=bool(appearance),
            ibl=bool(
                ready and renderer.get_ibl_name() is not None and self.available_environments()
            ),
            lighting=ready,
            dusk=bool(ready and getattr(renderer, "get_dusk_state", lambda: {})().get("available")),
            exhibition=embedded,
            clean_view=clean,
            captions=clean,
        )

    def pause_camera(self) -> None:
        """Release the view for a gesture while playback and presentation effects continue."""
        if self.active and self._sensing_story is not None and not self.camera_paused:
            self._sensing_story.pause_camera()
            self.camera_paused = True
            self.changed.emit()
            return
        if (
            not self.active
            or self.camera_paused
            or self._showcase_phase == "playback"
            or not (self._orbit_owned or self._tour_owned or self._targets_owned)
        ):
            return
        tools = self._camera_tools
        if self._tour_owned and tools.sequence is not None:
            self._tour_elapsed = max(0.0, perf_counter() - tools.sequence[0])
        tools.stop()
        if self._targets_owned:
            self.viz.set_state(camera_mode="overview")
        self.camera_paused = True
        self.changed.emit()

    def resume_camera(self) -> bool:
        """Resume only Demo's camera automation after explicit user input."""
        if not self.active or not self.camera_paused:
            return False
        tools = self._camera_tools
        try:
            if self._sensing_story is not None:
                self._sensing_story.resume_camera()
            elif self._tour_owned:
                tools.start_sequence(loop=True)
                tools.sequence = (perf_counter() - self._tour_elapsed, True)
            elif self._targets_owned:
                self.viz.set_state(camera_mode="follow")
                self._cycle_target(perf_counter(), initial=True)
            elif self._orbit_owned:
                pose = self.viz.renderer.get_camera_state()
                if pose is None:
                    raise ValueError("The camera is not ready")
                tools.start_orbit(pose.lookat, speed=self._orbit_speed)
            self.camera_paused = False
            self.changed.emit()
            return True
        except (AttributeError, RuntimeError, ValueError) as exc:
            self.stop(reason=f"Demo stopped: {exc}")
            return False

    def caption_text(self) -> str:
        """Describe the actual Demo phase and display choices without implying new data."""
        if not self.active:
            return self.status
        if self._sensing_story is not None:
            text = self._sensing_story.caption
            if self.camera_paused:
                text += " · Manual camera · Resume camera to show detection reveals"
            return text
        if self._showcase_phase == "presets":
            phase = "Frozen recorded frame · slowed illustrative MPC pulses"
        elif getattr(self.viz, "animation_running", False):
            phase = "Recorded playback" if self._snapshot["playback"] else "Scene playback"
        else:
            phase = "Current scene"
        appearance = getattr(self.viz, "appearance_preset_service", None)
        look = getattr(appearance, "active_name", "Current")
        descriptions = {
            "Detailed": "Detailed surfaces",
            "Paper": "Radio-material view",
            "Plan": "Street-plan view",
            "Dark": "Dark presentation",
            "Glow": "Glow presentation",
        }
        mode = getattr(getattr(self.viz, "app_state", None), "color_mode", None)
        colors = {
            "reflection_order": "reflection order",
            "mpc_type": "path type",
            "delay": "delay",
            "path_loss": "path loss",
            "material": "material",
            "reconstruction_type": "reconstruction type",
        }
        parts = [phase, descriptions.get(look, "Your scene appearance")]
        if getattr(self._exhibition, "interactive_figures", False):
            parts.append("Range–velocity and range profile · saved detections")
        visibility = getattr(getattr(self.viz, "app_state", None), "mpc_visibility", None)
        if mode in colors and getattr(visibility, "effective_paths", False):
            parts.append(f"Paths colored by {colors[mode]}")
            state = self.viz.app_state
            if getattr(state, "topk_render_enabled", False):
                parts.append(f"Up to {state.topk_render_max_paths:,} strongest paths")
        if self.camera_paused:
            parts.append("Manual camera · Demo continues")
        return " · ".join(parts)

    def _unavailable(self, reason: str) -> bool:
        self.status = reason
        self.changed.emit()
        return False

    def _set_playback(self, running: bool) -> None:
        if bool(getattr(self.viz, "animation_running", False)) != running:
            self.viz.animation_controller.toggle_animation()

    def _start_showcase_playback(self) -> None:
        """Retire frozen paths and camera motion before advancing recorded data."""
        self._showcase.close()
        self._showcase = None
        if self._camera_tools is not None:
            self._camera_tools.stop()
        if self._targets_owned:
            self.viz.set_state(camera_mode="overview")
        self._showcase_phase = "playback"
        self.camera_paused = False
        loop = getattr(self.viz, "loop_cb", None)
        if loop is not None:
            loop.setChecked(False)
        self.viz.animation_controller.reset_animation()
        self.viz.animation_controller.toggle_animation(direction=1)
        self._timer.setInterval(250)
        self.status = "Showcase · recorded playback · drag to explore; Esc stops Demo."
        self.changed.emit()

    def _target_rows(self) -> list[int]:
        """Use the normal focus catalog, which carries stable recorded identities."""
        combo = getattr(self.viz, "target_focus_dropdown", None)
        if combo is None:
            return []
        return [
            row
            for row in range(combo.count())
            if isinstance(combo.itemData(row), dict) and combo.itemData(row).get("type") == "target"
        ]

    def _cycle_target(self, now: float, *, initial: bool = False) -> None:
        rows = self._target_rows()
        if not rows:
            raise ValueError("No recorded targets are available to follow")
        combo = self.viz.target_focus_dropdown
        current = combo.currentIndex()
        row = (
            rows[(rows.index(current) + int(not initial)) % len(rows)]
            if current in rows
            else rows[0]
        )
        with QSignalBlocker(combo):
            combo.setCurrentIndex(row)
        camera = self.viz.camera_controller
        camera.remember_focus_selection()
        self._camera_tools.prepare()
        # Existing fixed-position Follow updates the look-at on every recorded
        # frame, including the destination of an in-progress smooth transition.
        # Switching actors never applies a saved pose or translates the eye.
        camera.focus_on_target()
        self._next_target = now + self._target_interval

    def _look_dwell(self) -> float:
        """Give a sunset its complete roundtrip before showing a technical look."""
        return (
            max(self._look_interval, self._sunset_duration)
            if self._sunset_started is not None
            else self._look_interval
        )

    def tick(self, now: float | None = None) -> None:
        """Advance slow look changes; never drive or skip scientific frames."""
        snapshot = self._snapshot
        if not self.active or snapshot is None:
            return
        if (
            getattr(self.viz, "_shutdown_started", False)
            or self.viz.renderer is not snapshot["renderer"]
            or self.viz.scenario is not snapshot["scenario"]
            or getattr(self.viz, "current_scenario_path", None) != snapshot["source"]
            or (
                snapshot["source_epoch"] is not None
                and getattr(self.viz, "_mpc_presented_source_epoch", 0) != snapshot["source_epoch"]
            )
        ):
            self.stop(reason="Demo stopped: the scene changed.", restore_playback=False)
            return
        if self._showcase_phase == "playback":
            if not self.viz.animation_running:
                self.stop(reason="Showcase complete.", restore_playback=False)
            return
        if self._sensing_story is not None:
            caption = self._sensing_story.caption
            try:
                self._sensing_story.tick(now)
            except (RuntimeError, ValueError, OSError) as exc:
                self.stop(reason=f"Sensing story stopped: {exc}")
                return
            if caption != self._sensing_story.caption:
                self.changed.emit()
            return
        if not self.camera_paused and (
            (self._orbit_owned and self._camera_tools.autorotate is None)
            or (self._tour_owned and self._camera_tools.sequence is None)
        ):
            self.pause_camera()
        now = perf_counter() if now is None else now
        if self._showcase is not None:
            try:
                self._showcase.sample(now)
                if self._showcase_returned and now >= self._next_look:
                    self._start_showcase_playback()
                    return
            except (RuntimeError, ValueError, OSError) as exc:
                self.stop(reason=f"Showcase stopped: {exc}")
                return
        if self._targets_owned and not self.camera_paused:
            if self.viz.app_state.camera_mode != "follow":
                self.pause_camera()
            elif now >= self._next_target:
                try:
                    self._cycle_target(now)
                except (RuntimeError, ValueError) as exc:
                    self.stop(reason=f"Demo stopped: {exc}")
                    return
        if snapshot["presentation"] is not None:
            return
        if snapshot["lights"] is None or now < self._next_look:
            try:
                self._advance_sunset(snapshot, now)
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Demo sunset transition failed: %s", exc)
                self.stop(reason=f"Demo stopped: {exc}")
            return
        started = perf_counter()
        try:
            labels = []
            if self._sunset_started is not None:
                # Complete the return to daylight before applying the next
                # look, even if Qt delivered this timer after its deadline.
                self._advance_sunset(snapshot, self._sunset_started + self._sunset_duration)
            if snapshot["dusk"] is not None:
                snapshot["renderer"].set_dusk_enabled(False)
            if self._appearance_cycle:
                self._look_index = (self._look_index + 1) % (len(self._appearance_cycle) + 1)
                appearance = snapshot["appearance"]
                if self._look_index == 0:
                    appearance.restore(
                        snapshot["full_appearance"],
                        name=snapshot["appearance_metadata"]["active_name"],
                    )
                    if snapshot["mpc_appearance"] is not None:
                        snapshot["mpc"].restore(snapshot["mpc_appearance"])
                        self.viz.set_state(**snapshot["presentation_state"])
                        snapshot["owned_palettes"] = {}
                    labels.append("current look")
                else:
                    name = self._appearance_cycle[self._look_index - 1]
                    self._apply_presentation(snapshot, name)
                    labels.append(name)
            elif self._cycle_lighting:
                labels.append(self._apply_simple_look(snapshot))
            if self._environment_cycle:
                self._environment_index = (self._environment_index + 1) % len(
                    self._environment_cycle
                )
                labels.append(self._apply_environment(snapshot))
            self._apply_demo_dusk(snapshot)
            dusk_label = self._dusk_label()
            if dusk_label:
                labels.append(dusk_label.removeprefix(" · "))
            self.last_switch_duration_s = perf_counter() - started
            # A slow switch never creates a queue of overdue switches. Allow
            # the next look its requested dwell time after publication.
            self._next_look = now + self.last_switch_duration_s + self._look_dwell()
            if self._showcase is not None:
                self._showcase_returned = self._look_index == 0
                self._showcase.restart(now + self.last_switch_duration_s, self._look_dwell())
                self._timer.setInterval(33)
            if self._sunset_started is not None:
                self._sunset_started = now + self.last_switch_duration_s
            self.status = f"Demo running · {' · '.join(labels)} · drag to explore; Esc stops Demo."
            if self._showcase is not None:
                self.status = (
                    f"Showcase · {' · '.join(labels)} · slowed MPC travel, then recorded playback."
                )
            self.changed.emit()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Demo look change failed: %s", exc)
            self.stop(reason=f"Demo stopped: {exc}")

    def _apply_demo_dusk(self, snapshot: dict) -> None:
        """Keep the authored light recipe out of technical appearance views."""
        self._sunset_started = None
        self._timer.setInterval(250)
        dusk = snapshot["dusk"]
        if dusk is None:
            return
        renderer = snapshot["renderer"]
        enabled = dusk["current_look"] and self._look_index == 0 and not renderer.get_wireframe()
        if enabled and self._scenario_sunset:
            renderer.set_dusk_enabled(False)
            if not renderer.set_dusk_progress(0.0):
                raise RuntimeError("The scenario's sunset lighting is no longer available")
            self._sunset_started = perf_counter()
            self._sunset_progress = 0.0
            self._timer.setInterval(250)
        elif not self._restore_dusk_level(renderer, enabled, dusk["current_progress"]):
            raise RuntimeError("The scenario's blue-hour lighting is no longer available")

    @staticmethod
    def _restore_dusk_level(renderer, enabled: bool, progress: float) -> bool:
        if enabled and progress < 1.0:
            return renderer.set_dusk_progress(progress)
        return renderer.set_dusk_enabled(enabled)

    def _advance_sunset(self, snapshot: dict, now: float) -> None:
        """Cycle daylight, blue hour and daylight without rebuilding the native rig."""
        if self._sunset_started is None:
            return
        phase = max(0.0, (now - self._sunset_started) / self._sunset_duration)
        phase = min(phase, 1.0) if snapshot["lights"] is not None else phase % 1.0
        if phase < 0.1:
            progress, transitioning = 0.0, False
        elif phase < 0.5:
            progress, transitioning = (phase - 0.1) / 0.4, True
        elif phase < 0.6:
            progress, transitioning = 1.0, False
        else:
            progress, transitioning = max(0.0, (1.0 - phase) / 0.4), phase < 1.0
        self._timer.setInterval(33 if transitioning else 250)
        if progress != self._sunset_progress:
            # Playback/orbit already presents the scene. Change uniforms for
            # that draw instead of scheduling competing animation frames.
            # Tours can hold still, so a tour alone still needs fade redraws.
            scene_is_animating = bool(getattr(self.viz, "animation_running", False)) or (
                self._orbit_owned and self._camera_tools.autorotate is not None
            )
            if not snapshot["renderer"].set_dusk_progress(
                progress, request_redraw=not scene_is_animating
            ):
                raise RuntimeError("The scenario's sunset lighting is no longer available")
            self._sunset_progress = progress

    def _dusk_label(self) -> str:
        if self._sunset_started is not None:
            return " · sunset cycle"
        renderer = self._snapshot["renderer"] if self._snapshot else None
        state = getattr(renderer, "get_dusk_state", lambda: {})()
        return " · blue hour" if state.get("enabled", False) else ""

    def _apply_simple_look(self, snapshot: dict) -> str:
        """Switch only direct lights or optional wireframe, without material refresh."""
        self._look_index = (self._look_index + 1) % (4 if self._include_wireframe else 3)
        renderer = snapshot["renderer"]
        lights = dict(snapshot["lights"])
        ibl = snapshot["ibl"]
        wireframe = snapshot["wireframe"]
        label = "current look"
        if self._look_index in (1, 2):
            side = self._look_index == 2
            label = "side light" if side else "soft studio"
            lights.update(
                lighting_profile="custom",
                headlight_enabled=False,
                ambient_intensity=0.35 if side else 0.7,
                key_intensity=2.2 if side else 1.5,
                fill_intensity=0.3 if side else 0.8,
                key_azimuth_deg=float(lights["key_azimuth_deg"]) + (65 if side else 0),
            )
            ibl = 0.0
        elif self._look_index == 3:
            label, wireframe = "wireframe", True
        renderer.restore_light_rig_state(lights, ibl=ibl)
        if renderer.get_wireframe() != wireframe:
            renderer.set_wireframe(wireframe)
        return label

    def _apply_presentation(self, snapshot: dict, name: str) -> None:
        """Pair scene and path looks; only Demo applies the presentation path cap."""
        if snapshot["mpc_appearance"] is not None:
            mpc_name = SCENE_MPC_PRESETS[name]
            snapshot["owned_palettes"] = {
                kind: MPC_PRESETS[mpc_name][f"{kind}_colormap"]
                for kind in ("continuous", "categorical")
            }
        snapshot["appearance"].apply(name)
        if snapshot["mpc_appearance"] is None:
            return
        snapshot["mpc"].apply(mpc_name)
        state = snapshot["presentation_state"]
        capped = name in {"Dark", "Glow"}
        self.viz.set_state(
            topk_render_enabled=True if capped else state["topk_render_enabled"],
            topk_render_max_paths=20 if capped else state["topk_render_max_paths"],
            mpc_visibility=replace(self.viz.app_state.mpc_visibility, enabled=True, paths=True),
        )

    def _apply_environment(self, snapshot: dict) -> str:
        """Apply only installed skies; their catalog lighting accompanies the backdrop."""
        renderer = snapshot["renderer"]
        name, label = self._environment_cycle[self._environment_index]
        if renderer.get_ibl_name() != name and not renderer.set_ibl(name):
            raise RuntimeError(f"The installed sky {label!r} is no longer available")
        if self._environment_index == 0:
            renderer.restore_light_rig_state(snapshot["lights"], ibl=snapshot["ibl"])
            renderer.show_skybox(snapshot["skybox"])
            renderer.set_shadow_quality(snapshot["shadow_quality"])
        else:
            renderer.apply_environment_lighting(name)
            renderer.show_skybox(True)
        return label

    def stop(
        self,
        *,
        reason: str = "Manual control restored.",
        restore_playback: bool = True,
        preserve_playback: bool = False,
        preserve_loop: bool = False,
    ) -> None:
        """Restore temporary settings while leaving the visible camera pose in place."""
        if not self.active:
            return
        self.active = False
        self._timer.stop()
        self._timer.setInterval(250)
        self._sunset_started = None
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        snapshot, self._snapshot = self._snapshot, None
        tools, self._camera_tools = self._camera_tools, None
        failures = []
        if self._showcase is not None:
            try:
                self._showcase.close()
            except RuntimeError as exc:
                failures.append(str(exc))
                logger.warning("Could not clear demo MPC pulses: %s", exc)
        self._showcase = None
        self._showcase_phase = None
        if (self._orbit_owned or self._tour_owned or self._targets_owned) and tools is not None:
            try:
                tools.stop()
            except RuntimeError as exc:
                failures.append(str(exc))
                logger.warning("Could not stop demo camera motion: %s", exc)
            finally:
                tools.history = snapshot["history"]
                if self._tour_owned:
                    tools.views, tools.shots = snapshot["views"], snapshot["shots"]
        self._orbit_owned = False
        self._tour_owned = False
        same_scene = (
            getattr(self.viz, "renderer", None) is snapshot["renderer"]
            and getattr(self.viz, "scenario", None) is snapshot["scenario"]
            and getattr(self.viz, "current_scenario_path", None) == snapshot["source"]
            and (
                snapshot["source_epoch"] is None
                or getattr(self.viz, "_mpc_presented_source_epoch", 0) == snapshot["source_epoch"]
            )
        )
        story, self._sensing_story = self._sensing_story, None
        if story is not None:
            try:
                story.close(restore=same_scene)
            except (RuntimeError, ValueError, OSError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore sensing presentation: %s", exc)
        if self._targets_owned and same_scene:
            try:
                follow = snapshot["target_follow"]
                self.viz.set_state(camera_mode=follow["mode"])
                combo = self.viz.target_focus_dropdown
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(max(0, combo.findData(follow["selection"])))
                self.viz.camera_controller.remember_focus_selection()
                for key, value in follow["framing"].items():
                    setattr(tools, key, value)
                tools.follow_position = None
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo target following: %s", exc)
        self._targets_owned = False
        self.camera_paused = False
        presentation_view, self._presentation_view = self._presentation_view, None
        if presentation_view is not None:
            try:
                presentation_view.restore(restore_docks=same_scene)
            except RuntimeError as exc:
                failures.append(str(exc))
                logger.warning("Could not restore presentation view: %s", exc)
        exhibition, self._exhibition = self._exhibition, None
        if exhibition is not None:
            try:
                exhibition.restore(restore_docks=same_scene)
            except RuntimeError as exc:
                failures.append(str(exc))
                logger.warning("Could not restore exhibition layout: %s", exc)
        # Palettes are global. Release only values still owned by this Demo,
        # including after a replacement scene or a failed appearance restoration.
        for kind, owned in snapshot["owned_palettes"].items():
            if getattr(theme_manager.current, f"{kind}_colormap") == owned:
                theme_manager.set_colormap(kind, snapshot["mpc_appearance"][f"{kind}_colormap"])
        if same_scene and snapshot["dusk"] is not None:
            try:
                snapshot["renderer"].set_dusk_enabled(False)
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not remove demo blue-hour lighting: %s", exc)
        if same_scene and snapshot["lights"] is not None:
            if snapshot["full_appearance"] is not None:
                try:
                    if snapshot["appearance"].capture() != snapshot["full_appearance"]:
                        snapshot["appearance"].restore(
                            snapshot["full_appearance"],
                            name=snapshot["appearance_metadata"]["active_name"],
                        )
                except (RuntimeError, ValueError, OSError) as exc:
                    failures.append(str(exc))
                    logger.warning("Could not restore demo preset: %s", exc)
            if snapshot["cycle_ibl"]:
                try:
                    renderer = snapshot["renderer"]
                    if renderer.get_ibl_name() != snapshot["sky"]:
                        if not renderer.set_ibl(snapshot["sky"]):
                            raise RuntimeError("The original sky could not be restored")
                    renderer.show_skybox(snapshot["skybox"])
                    renderer.set_shadow_quality(snapshot["shadow_quality"])
                except (RuntimeError, ValueError, OSError) as exc:
                    failures.append(str(exc))
                    logger.warning("Could not restore demo sky: %s", exc)
            try:
                snapshot["renderer"].restore_light_rig_state(
                    snapshot["lights"], ibl=snapshot["ibl"]
                )
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo appearance: %s", exc)
            try:
                if snapshot["renderer"].get_wireframe() != snapshot["wireframe"]:
                    snapshot["renderer"].set_wireframe(snapshot["wireframe"])
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo wireframe state: %s", exc)
        if same_scene and snapshot["mpc_appearance"] is not None:
            try:
                snapshot["mpc"].restore(snapshot["mpc_appearance"])
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo MPC appearance: %s", exc)
        if same_scene and snapshot["presentation_state"] is not None:
            try:
                # Restore AppState after the appearance owner has synchronized
                # renderer controls, including any original automatic defaults.
                self.viz.set_state(**snapshot["presentation_state"])
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo path settings: %s", exc)
        if same_scene and snapshot["dusk"] is not None:
            try:
                if not self._restore_dusk_level(
                    snapshot["renderer"], snapshot["dusk"]["enabled"], snapshot["dusk"]["progress"]
                ):
                    raise RuntimeError("The original blue-hour lighting could not be restored")
            except (RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo blue-hour lighting: %s", exc)
        if snapshot["playback"] and same_scene:
            try:
                if not preserve_playback:
                    if snapshot["play_direction"] is not None:
                        self.viz.play_direction = snapshot["play_direction"]
                    self._set_playback(snapshot["running"] if restore_playback else False)
                loop = getattr(self.viz, "loop_cb", None)
                if not preserve_loop and loop is not None and snapshot["loop"] is not None:
                    loop.setChecked(snapshot["loop"])
            except (AttributeError, RuntimeError) as exc:
                failures.append(str(exc))
                logger.warning("Could not restore demo playback: %s", exc)
        if same_scene and snapshot["appearance_metadata"] is not None:
            appearance = snapshot["appearance"]
            for key, value in snapshot["appearance_metadata"].items():
                setattr(appearance, key, value)
            # Refresh the normal appearance and Rendering widgets from the
            # restored owners; a delayed custom-check may have run during demo.
            appearance.changed.emit()
        # Palette callbacks can publish an intermediate packet before AppState
        # restoration. Coalesce one final same-frame publication after all owners
        # and playback have returned to their original state.
        if (
            same_scene
            and snapshot["mpc_appearance"] is not None
            and not getattr(self.viz, "_shutdown_started", False)
        ):
            self.viz.schedule_update()
        self.status = (
            reason if not failures else f"Demo stopped; restoration failed: {'; '.join(failures)}"
        )
        self.changed.emit()

    def eventFilter(self, watched, event):
        """Allow scene exploration; restore owned state before edits to ordinary controls."""
        # PySide can expose an unrelated or expired wrapper during widget changes.
        # Inspecting it as an event can enter an invalid native method.
        if not isinstance(event, QEvent) or not isValid(event):
            return False
        if self.active and event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
            QEvent.Type.ShortcutOverride,
            QEvent.Type.TouchBegin,
        ):
            if isinstance(watched, QWidget) and (
                watched is self.viz
                or (isinstance(self.viz, QWidget) and self.viz.isAncestorOf(watched))
            ):
                key_event = event.type() in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride)
                if key_event and event.key() == Qt.Key.Key_Escape:
                    self.stop()
                    return True
                if watched.property("demoAction"):
                    return False
                if self._presentation_view is not None and self._presentation_view.owns_widget(
                    watched
                ):
                    return False
                workspace = getattr(self.viz, "_visualization_workspace", None)
                host = getattr(workspace, "viewport_host", None)
                viewport_input = isinstance(host, QWidget) and (
                    watched is host or host.isAncestorOf(watched)
                )
                camera_key = key_event and event.key() in (
                    Qt.Key.Key_Left,
                    Qt.Key.Key_Right,
                    Qt.Key.Key_Up,
                    Qt.Key.Key_Down,
                    Qt.Key.Key_W,
                    Qt.Key.Key_A,
                    Qt.Key.Key_S,
                    Qt.Key.Key_D,
                    Qt.Key.Key_Alt,
                    Qt.Key.Key_Shift,
                    Qt.Key.Key_Control,
                    Qt.Key.Key_Meta,
                )
                if viewport_input and (not key_event or camera_key):
                    self.pause_camera()
                    return False
                retiring = self._exhibition is not None and self._exhibition.owns_widget(watched)
                if retiring and getattr(self._exhibition, "interactive_figures", False):
                    return False
                loop = getattr(self.viz, "loop_cb", None)
                transport = (
                    getattr(self.viz, "play_btn", None),
                    getattr(self.viz, "reverse_play_btn", None),
                )
                space = (
                    event.type() in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride)
                    and event.key() == Qt.Key.Key_Space
                    and not isinstance(
                        watched,
                        (
                            QAbstractButton,
                            QAbstractSpinBox,
                            QComboBox,
                            QLineEdit,
                            QPlainTextEdit,
                            QTextEdit,
                        ),
                    )
                )
                # A toggle acts on its visible state. Restoring that state
                # before forwarding the click would invert the user's intent.
                self.stop(
                    preserve_playback=not retiring and (watched in transport or space),
                    preserve_loop=watched is loop,
                )
                # Demo-created docks return their plot widgets and close on
                # takeover. Do not forward a gesture to their retired canvas.
                return retiring
        return False

    def close(self) -> None:
        """Stop before panel or renderer disposal, without restarting playback."""
        self.stop(restore_playback=False)
