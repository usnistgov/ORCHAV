"""Pygfx camera presentation tools, separate from acquisition/playback timing.

One coordinator owns transitions, autorotation and view sequences. Native
pygfx input remains responsible for mouse navigation. Saved tools are part of
the existing workspace camera payload; a sequence can also be shared as JSON.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget

from shared.logging import get_logger

from ..renderers.camera_ops import camera_state_for_pov, compute_pov_forward_up
from ..types.camera_state import CameraState
from .camera_motion import interpolate, orbit, translated, validate_pose

logger = get_logger("orchav.camera_presentation")


class CameraPresentation(QObject):
    """Coordinate optional camera motion without modifying recorded observations."""

    status_changed = Signal(str)
    views_changed = Signal()

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.viz = owner.visualizer
        self.renderer = self.viz.renderer
        self.duration = 0.8
        self.smooth = True
        self.follow_heading = "world"
        self.follow_offset = None  # [distance behind, left, height], in metres
        self.follow_world_yaw = None
        self.true_pov = True
        self.pov_offset = [0.0, 0.0, 0.0]
        self.pov_fov = 60.0
        self.horizon_lock = True
        self.views = {}
        self.shots = []
        self.history = []
        self.applying = False
        self.exporting = False
        self.pending = False
        self.transition = None
        self.autorotate = None
        self.sequence = None
        self.last_yaw = None
        self.follow_position = None
        self.message = "Ready"
        self.source = getattr(self.viz, "current_scenario_path", None)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.escape = None
        if isinstance(self.viz, QWidget):
            self.escape = QShortcut(QKeySequence("Escape"), self.viz)
            self.escape.activated.connect(self.interrupt)

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, text: str) -> None:
        self._message = text
        self.status_changed.emit(text)

    def bind_input(self):
        """Observe real viewport input; never consume the native navigation event."""
        self.renderer.set_camera_input_callback(self.input_event)
        if self.escape is not None:
            self.escape.setEnabled(True)

    def input_event(self, event):
        """Let input interrupt camera presentation, keeping ordinary Follow active."""
        if self.exporting:
            return
        if event.type == "key_down" and getattr(event, "key", "") not in (
            "Escape",
            "w",
            "a",
            "s",
            "d",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
        ):
            return
        self.interrupt()

    def interrupt(self):
        """Stop from either viewport input or the control window's Escape shortcut."""
        if not self.exporting:
            self.stop()
            if getattr(self.viz.app_state, "camera_mode", None) == "pov":
                self.free_camera()

    def free_camera(self):
        """Release actor-driven pose updates without restoring an older POV pose."""
        self.owner.restore_pov_entity_visibility(update_renderer=False)
        self.owner.clear_pre_pov_camera_state()
        self.viz.set_state(camera_mode="overview", pov_hidden_node=None)

    def prepare(self):
        """Remember the current view and animate the next deliberate camera action."""
        if self.exporting:
            return
        self.source = getattr(self.viz, "current_scenario_path", None)
        self.bind_input()
        self.remember_follow_framing()
        self.stop()
        state = self.renderer.get_camera_state()
        if state is not None and self.source is not None:
            self.history.append(state)
            self.history = self.history[-20:]
        # Scene loading applies its initial view before committing source identity.
        # Apply that view immediately; only interactive navigation is animated.
        self.pending = self.source is not None
        self.last_yaw = None
        self.follow_position = None
        self._cancel_native_motion()

    def _cancel_native_motion(self):
        self.renderer.cancel_camera_gestures()

    def intercept(self, state: CameraState) -> bool:
        """Queue a deliberate pose change or update the destination of a live transition."""
        if self.applying:
            return False
        if self.exporting:
            return True
        if self.pending:
            self.pending = False
            start = self.renderer.get_camera_state()
            if self.smooth and self.duration > 0 and start is not None:
                self.transition = [start, state, perf_counter()]
                self.timer.start()
                return True
        if self.transition is not None:
            self.transition[1] = state
            return True
        return False

    def apply(self, state: CameraState):
        """Apply one presentation sample without recursively starting a transition."""
        self.applying = True
        try:
            return self.renderer.set_camera_state(validate_pose(state))
        finally:
            self.applying = False

    def stop(self):
        """Stop at the visible pose; stopping never rewinds camera or data time."""
        self.timer.stop()
        self.pending = False
        self.transition = self.autorotate = self.sequence = None
        self.message = "Stopped"

    def reset_for_recording(self):
        """Retire source-bound motion and notify visible named-view controls."""
        self.stop()
        self.history.clear()
        self.views.clear()
        self.shots.clear()
        self.last_yaw = None
        self.follow_position = None
        self.views_changed.emit()

    def previous(self):
        """Return to the preceding deliberate view change."""
        if not self.history:
            return
        state = self.history.pop()
        self.stop()
        self.free_camera()
        self._cancel_native_motion()
        self.pending = True
        self.renderer.set_camera_state(state)

    def tick(self, now=None):
        """Advance camera animation in wall time, including while playback is paused."""
        if self.exporting:
            return
        if getattr(self.viz, "current_scenario_path", None) != self.source:
            self.stop()
            self.history.clear()
            self.source = getattr(self.viz, "current_scenario_path", None)
            self.message = "Camera motion stopped because the recording changed"
            return
        now = perf_counter() if now is None else now
        try:
            if self.transition is not None:
                start, end, began = self.transition
                fraction = min(1, (now - began) / max(self.duration, 1e-9))
                state = interpolate(start, end, fraction)
                if (
                    self.follow_heading == "fixed"
                    and getattr(self.viz.app_state, "camera_mode", None) == "follow"
                ):
                    # Quaternion interpolation can bank between level endpoints.
                    # Keep tracking level even when another switch interrupts it.
                    state = replace(state, up=(0.0, 0.0, 1.0))
                self.apply(state)
                if fraction >= 1:
                    self.transition = None
            elif self.autorotate is not None:
                state, center, speed, began, selection = self.autorotate
                if selection is not None:
                    position = self.resolve(selection)
                    if position is None:
                        raise ValueError("Orbit actor is unavailable in this frame")
                    state = translated(state, np.asarray(position) - center)
                    center = position
                self.apply(orbit(state, center, speed * (now - began)))
            elif self.sequence is not None:
                began, loop = self.sequence
                elapsed = now - began
                length = self.sequence_duration()
                self.apply(self.sequence_pose(elapsed % length if loop else elapsed))
                if not loop and elapsed >= length:
                    self.sequence = None
            if not any((self.transition, self.autorotate, self.sequence)):
                self.timer.stop()
                self.message = "Ready"
        except (ValueError, KeyError, IndexError) as exc:
            self.stop()
            self.message = str(exc)
            logger.warning("Camera animation stopped: %s", exc)

    def selection(self):
        dropdown = getattr(self.viz, "target_focus_dropdown", None)
        return deepcopy(dropdown.currentData()) if dropdown is not None else None

    def resolve(self, selection):
        """Resolve an actor by saved selection identity, without changing the GUI."""
        query = self.owner.scene_query
        kind = (selection or {}).get("type")
        if kind == "target":
            return query.resolve_target_selection_position(selection)
        if kind == "tx":
            return query.resolve_tx_position(int(selection["index"]))
        if kind == "rx":
            return query.resolve_rx_position(int(selection["index"]))
        if kind == "auto":
            return query.resolve_auto_focus_position()
        return None

    def start_orbit(self, center, speed=8.0, *, actor=False):
        """Orbit a fixed scene point or translate the orbit with the selected actor."""
        self.prepare()
        self.pending = False
        selection = self.selection() if actor else None
        if actor:
            center = self.resolve(selection)
        if center is None:
            raise ValueError("Select an available orbit center")
        state = self.renderer.get_camera_state()
        if state is None:
            raise ValueError("Camera is not ready")
        center = np.asarray(center, dtype=float)
        state = CameraState(state.eye, tuple(center), state.up, state.fov_deg)
        validate_pose(state)
        self.free_camera()
        self.autorotate = (state, center, float(speed), perf_counter(), selection)
        self.timer.start()
        self.message = "Autorotating — mouse input or Escape stops"

    def focus(self, position, *, radius=None):
        """Focus an actor, pair or inspected cell with a remembered return view."""
        if position is None:
            raise ValueError("No position is selected")
        self.prepare()
        self.free_camera()
        current = self.renderer.get_camera_state()
        if current is None:
            raise ValueError("Camera is not ready")
        offset = np.asarray(current.eye) - current.lookat
        if radius is not None:
            offset *= max(float(radius), 1.0) / np.linalg.norm(offset)
        self.renderer.set_camera_state(
            CameraState(
                tuple(np.asarray(position) + offset), tuple(position), current.up, current.fov_deg
            )
        )

    def save_view(self, name: str, *, relative=False):
        """Store a named view, optionally translated with one stable actor identity."""
        name = name.strip()
        if not name:
            raise ValueError("Give the view a name")
        state = validate_pose(self.renderer.get_camera_state())
        selection = self.selection() if relative else None
        if relative:
            position = self.resolve(selection)
            if position is None or selection.get("type") == "auto":
                raise ValueError("Choose a specific actor for an actor-relative view")
            state = translated(state, -np.asarray(position))
        self.views[name] = {"pose": state.to_dict(), "actor": selection}

    def view_pose(self, name):
        view = self.views[name]
        state = validate_pose(CameraState.from_dict(view["pose"]))
        if view["actor"] is not None:
            position = self.resolve(view["actor"])
            if position is None:
                raise ValueError(f"Actor for '{name}' is unavailable")
            state = translated(state, position)
        return state

    def recall_view(self, name):
        state = self.view_pose(name)
        self.prepare()
        self.free_camera()
        self.renderer.set_camera_state(state)

    def sequence_duration(self):
        return sum(s["hold"] + (s["move"] if i else 0) for i, s in enumerate(self.shots))

    def sequence_pose(self, seconds):
        """Evaluate the same camera path for live preview and deterministic export."""
        if not self.shots:
            raise ValueError("Add saved views to the sequence first")
        remaining = max(0.0, float(seconds))
        previous = self.view_pose(self.shots[0]["view"])
        for i, shot in enumerate(self.shots):
            current = self.view_pose(shot["view"])
            if i:
                if remaining < shot["move"]:
                    return interpolate(previous, current, remaining / shot["move"])
                remaining -= shot["move"]
            if remaining < shot["hold"]:
                return current
            remaining -= shot["hold"]
            previous = current
        return previous

    def start_sequence(self, loop=False):
        if self.sequence_duration() <= 0:
            raise ValueError("Add a sequence with a nonzero duration")
        for shot in self.shots:
            self.view_pose(shot["view"])
        self.prepare()
        self.pending = False
        self.free_camera()
        self.sequence = (perf_counter(), bool(loop))
        self.timer.start()
        self.message = "Playing camera sequence — data playback is independent"

    def pov_pose(self, position, orientation, axis):
        """Apply local actor offsets, adjustable lens and optional world-up horizon."""
        yaw, pitch, roll = list(orientation)[:3]
        forward, up = compute_pov_forward_up(yaw, pitch, roll, axis="forward")
        left = np.cross(up, forward)
        offset = np.column_stack([forward, left, up]) @ (
            [0.0, 0.0, 0.0] if self.true_pov else self.pov_offset
        )
        state = camera_state_for_pov(
            np.asarray(position) + offset, orientation, axis=axis, fov=self.pov_fov
        )
        if state is not None and self.horizon_lock:
            state = CameraState(state.eye, state.lookat, (0.0, 0.0, 1.0), state.fov_deg)
        return state

    def apply_framing(self, *, offset_changed=False):
        """Apply an edited active-mode setting now, even during a paused transition.

        A framing edit takes precedence over an in-flight camera move. Suppress
        viewport focus transfer so typing another digit in a spin box still works.
        """
        if self.exporting:
            raise ValueError("Wait for camera export to finish before changing framing")
        mode = getattr(self.viz.app_state, "camera_mode", None)
        if mode not in {"follow", "pov"}:
            raise ValueError("Select Follow or POV in the Camera panel to edit framing")
        self.stop()
        self._cancel_native_motion()
        self.applying = True
        try:
            if mode == "pov":
                success = self.owner.set_pov_camera()
            elif self.follow_heading in {"chase", "motion", "fixed"} or offset_changed:
                position = self.owner.scene_query.get_focus_position()
                if position is None:
                    raise ValueError("The selected actor has no position in this frame")
                self.follow_position = None
                if self.follow_heading != "motion":
                    self.last_yaw = None
                if offset_changed and self.follow_heading != "fixed":
                    current = self.renderer.get_camera_state()
                    state = self.follow_offset_pose(current, np.asarray(position))
                    success = self.renderer.set_camera_state(state)
                    self.follow_position = np.asarray(position).copy()
                else:
                    success = self.follow(np.asarray(position))
            else:
                # World heading keeps the current pose, including mouse framing.
                self.last_yaw = None
                self.follow_world_yaw = None
                self.remember_follow_framing()
                success = True
        finally:
            self.applying = False
        if not success:
            raise ValueError("Camera framing could not be applied to the selected actor")
        self.message = f"{mode.upper() if mode == 'pov' else 'Follow'} framing applied"

    def remember_follow_framing(self):
        """Read current mouse framing without driving the camera or cancelling input."""
        if self.follow_position is None or self.follow_heading == "fixed":
            return
        state = self.transition[1] if self.transition else self.renderer.get_camera_state()
        if state is None:
            return
        delta = np.asarray(state.eye) - self.follow_position
        if self.follow_heading in {"chase", "motion"}:
            yaw = self.last_yaw
            if yaw is None:
                return
        else:
            if self.follow_world_yaw is None:
                self.follow_world_yaw = float(np.arctan2(-delta[1], -delta[0]))
            yaw = self.follow_world_yaw
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0])
        self.follow_offset = [float(-delta @ forward), float(delta @ left), float(delta[2])]

    def follow_yaw(self, state):
        """Resolve chase heading from recorded orientation or horizontal motion."""
        if self.follow_heading == "motion":
            direction = self.owner.scene_query.get_focus_motion_direction()
            if direction is not None and np.linalg.norm(direction[:2]) > 1e-6:
                return float(np.arctan2(direction[1], direction[0]))
            if self.last_yaw is not None:
                return self.last_yaw
            # No travel direction exists for a stationary/vertical-only actor.
            # Start with the current viewing heading until horizontal motion is known.
            direction = np.subtract(state.lookat, state.eye)
            yaw = float(np.arctan2(direction[1], direction[0]))
            if self.follow_offset is not None:
                distance, side, _ = self.follow_offset
                yaw += float(np.arctan2(side, distance))
            return yaw
        _, orientation, _ = self.owner.scene_query.get_entity_position_orientation_and_info()
        return float(orientation[0]) if orientation is not None else None

    def follow_offset_pose(self, state, position):
        """Place the camera using actor-heading or fixed-world follow offsets."""
        if state is None:
            raise ValueError("Camera is not ready")
        if self.follow_heading in {"chase", "motion"}:
            yaw = self.follow_yaw(state)
            if yaw is None:
                raise ValueError("Behind-actor framing requires a recorded orientation")
            self.last_yaw = yaw
        else:
            if self.follow_world_yaw is None:
                delta = np.asarray(state.lookat) - state.eye
                self.follow_world_yaw = float(np.arctan2(delta[1], delta[0]))
            yaw = self.follow_world_yaw
        if self.follow_offset is None:
            return self.behind_pose(state, position, yaw)
        distance, side, height = self.follow_offset
        offset = np.array(
            [
                -distance * np.cos(yaw) - side * np.sin(yaw),
                -distance * np.sin(yaw) + side * np.cos(yaw),
                height,
            ]
        )
        if np.linalg.norm(offset) < 0.01:
            raise ValueError("Follow offsets must place the camera at least 0.01 m from the actor")
        return CameraState(tuple(position + offset), tuple(position), (0, 0, 1), state.fov_deg)

    @staticmethod
    def behind_pose(state, position, yaw):
        """Start a chase behind the actor at the current distance and a raised angle."""
        distance = max(3.0, np.linalg.norm(np.subtract(state.eye, state.lookat)))
        offset = np.array([-np.cos(yaw), -np.sin(yaw), 0.35])
        offset *= distance / np.linalg.norm(offset)
        return CameraState(
            tuple(position + offset), tuple(position), (0.0, 0.0, 1.0), state.fov_deg
        )

    def follow(self, position):
        """Move with the actor while keeping ongoing user zoom, pan and orbit."""
        if self.exporting:
            return True
        previous = self.follow_position
        state = self.transition[1] if self.transition else self.renderer.get_camera_state()
        if state is None:
            return False
        if self.follow_heading == "fixed":
            # Look-at tracking changes orientation only, never camera position.
            if np.linalg.norm(np.asarray(state.eye) - position) < 1e-6:
                self.message = "Actor coincides with the fixed camera; keeping the last direction"
                return True
            result = self.renderer.set_camera_state(
                CameraState(state.eye, tuple(position), (0.0, 0.0, 1.0), state.fov_deg)
            )
            if result:
                self.follow_position = np.asarray(position).copy()
            return result
        if previous is None:
            delta = position - np.asarray(state.lookat)
        else:
            delta = position - previous
        state = translated(state, delta)
        if self.follow_heading in {"chase", "motion"}:
            yaw = self.follow_yaw(state)
            if yaw is not None:
                if previous is None:
                    # Actor switches reset follow_position. Re-establish a rear
                    # view relative to this actor, not the predecessor's heading.
                    state = self.follow_offset_pose(state, position)
                elif self.last_yaw is not None:
                    state = orbit(state, position, np.degrees(yaw - self.last_yaw))
                self.last_yaw = yaw
            elif previous is None:
                raise ValueError("Behind-actor framing requires a recorded orientation")
        result = self.renderer.set_camera_state(state)
        if result:
            self.follow_position = np.asarray(position).copy()
            self.remember_follow_framing()
        return result

    def payload(self):
        """Portable camera tools only; no scientific arrays or runtime timers."""
        self.remember_follow_framing()
        return deepcopy(
            {
                "version": 1,
                "smooth": self.smooth,
                "duration": self.duration,
                "follow_heading": self.follow_heading,
                "follow_offset": self.follow_offset,
                "follow_world_yaw": self.follow_world_yaw,
                "true_pov": self.true_pov,
                "pov_offset": self.pov_offset,
                "pov_fov": self.pov_fov,
                "horizon_lock": self.horizon_lock,
                "views": self.views,
                "shots": self.shots,
            }
        )

    def restore(self, payload):
        """Validate the complete payload before replacing settings or saved views."""
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported camera tools format")
        data = deepcopy(payload)
        duration = float(data["duration"])
        offset = np.asarray(data["pov_offset"], dtype=float)
        if not 0 <= duration <= 30 or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Invalid camera timing or offset")
        # Older camera JSON stored offsets without a separate true-POV switch.
        data.setdefault("true_pov", not bool(np.any(offset)))
        follow_offset = data.setdefault("follow_offset", None)
        world_yaw = data.setdefault("follow_world_yaw", None)
        if follow_offset is not None:
            follow_offset = np.asarray(follow_offset, dtype=float)
            if (
                follow_offset.shape != (3,)
                or not np.isfinite(follow_offset).all()
                or np.linalg.norm(follow_offset) < 0.01
            ):
                raise ValueError("Invalid follow offsets")
            data["follow_offset"] = follow_offset.tolist()
        if world_yaw is not None:
            data["follow_world_yaw"] = float(world_yaw)
            if not np.isfinite(data["follow_world_yaw"]):
                raise ValueError("Invalid world follow heading")
        if not 1 <= float(data["pov_fov"]) <= 179 or data["follow_heading"] not in {
            "world",
            "chase",
            "motion",
            "fixed",
        }:
            raise ValueError("Invalid camera framing")
        if not isinstance(data["views"], dict) or not isinstance(data["shots"], list):
            raise ValueError("Invalid camera views or sequence")
        if not all(isinstance(data[key], bool) for key in ("smooth", "horizon_lock", "true_pov")):
            raise ValueError("Camera toggles must be booleans")
        data["duration"] = duration
        data["pov_offset"] = offset.tolist()
        data["pov_fov"] = float(data["pov_fov"])
        for name, view in data["views"].items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid view name")
            validate_pose(CameraState.from_dict(view["pose"]))
            actor = view["actor"]
            if actor is not None and (
                not isinstance(actor, dict) or actor.get("type") not in {"tx", "rx", "target"}
            ):
                raise ValueError("Invalid saved actor")
            if actor is not None and actor["type"] in {"tx", "rx"}:
                if not isinstance(actor.get("index"), int) or actor["index"] < 0:
                    raise ValueError("Invalid saved radio actor index")
        for shot in data["shots"]:
            if shot["view"] not in data["views"] or not all(
                np.isfinite(float(shot[k])) and 0 <= float(shot[k]) <= 3600
                for k in ("hold", "move")
            ):
                raise ValueError("Invalid camera sequence shot")
            shot["hold"], shot["move"] = float(shot["hold"]), float(shot["move"])
        self.stop()
        for key in (
            "smooth",
            "duration",
            "follow_heading",
            "follow_offset",
            "follow_world_yaw",
            "true_pov",
            "pov_offset",
            "pov_fov",
            "horizon_lock",
            "views",
            "shots",
        ):
            setattr(self, key, data[key])
        self.views_changed.emit()

    def close(self):
        self.stop()
        self.renderer.set_camera_input_callback(None)
        if self.escape is not None:
            self.escape.setEnabled(False)

    def suspend_for_export(self):
        """Freeze live timers and native navigation during deterministic export."""
        self.bind_input()
        enabled = self.renderer.set_camera_input_enabled(False)
        snapshot = (
            self.renderer.get_camera_state(),
            self.transition,
            self.autorotate,
            self.sequence,
            self.pending,
            perf_counter(),
            enabled,
            self.message,
        )
        self.stop()
        self.exporting = True
        self._cancel_native_motion()
        return snapshot

    def resume_after_export(self, snapshot):
        """Restore the visible pose and resume camera timers where they paused."""
        state, transition, autorotate, sequence, pending, began, enabled, message = snapshot
        self.apply(state)
        self.exporting = False
        self.renderer.set_camera_input_enabled(enabled)
        delay = perf_counter() - began
        if transition is not None:
            transition = [transition[0], transition[1], transition[2] + delay]
        if autorotate is not None:
            autorotate = (*autorotate[:3], autorotate[3] + delay, autorotate[4])
        if sequence is not None:
            sequence = (sequence[0] + delay, sequence[1])
        self.transition, self.autorotate, self.sequence = transition, autorotate, sequence
        self.pending = pending
        self.message = message
        if any((transition, autorotate, sequence)):
            self.timer.start()
