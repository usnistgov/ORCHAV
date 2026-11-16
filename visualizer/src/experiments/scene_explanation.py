"""Bounded display of exact report bins and metadata-backed receive responses."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QTimer
from shiboken6 import isValid

from ..sensing.selection import selection_scope
from ..sensing.volume_antenna import receive_metadata_signature, response_object
from .report_geometry import (
    receive_pattern_for_request,
    selected_beam_index,
)


class SceneExplanationController:
    """Reuse the selected-result overlay lifecycle; never open another reader."""

    def __init__(self, viz):
        self.viz = viz
        self.dashboard = None
        self.accepted_step = None
        self.overlay = None
        self.response_key = None
        self.desired_response = None
        self.pending = None
        self.executor = None
        self.poll_timer = None
        self.response_cache = {}
        self.failed_response = None
        self.cinematic = None

    def _layer(self):
        if self.overlay is None:
            from ..renderers.pygfx.experiments import ExperimentOverlay

            self.overlay = ExperimentOverlay(self.viz.renderer)
        return self.overlay

    def clear(self):
        if self.cinematic is not None:
            self.cinematic.clear()
        if self.overlay is not None and self.overlay.layers:
            self.overlay.clear()
        self.response_key = self.desired_response = None
        self._stop_polling()
        if self.pending is not None and self.pending[1].cancel():
            self.pending = None

    def _stop_polling(self):
        if self.poll_timer is not None and isValid(self.poll_timer):
            self.poll_timer.stop()
        else:
            self.poll_timer = None

    def sync(self, dashboard, accepted_step):
        self.dashboard, self.accepted_step = dashboard, accepted_step
        controls = getattr(dashboard, "scene_explanation", None)
        if controls is None or not isValid(controls):
            self.clear()
            return
        from ..services.paused_presentation import frames_advancing

        controls.set_playing(frames_advancing(self.viz))
        request = dashboard._last_request
        selection = dashboard._linked_selection
        controls.bind_context(request, selection)
        current_pattern = None
        try:
            if not dashboard.isVisible() or not controls.isVisible() or request is None:
                self.clear()
                return
            if getattr(self.viz.renderer, "renderer_id", None) != "pygfx":
                raise ValueError("3D explanations require the pygfx renderer")
            panel = getattr(getattr(self.viz, "ui_manager", None), "panels", {}).get("sensing")
            session = getattr(panel, "_dashboard_session", None)
            if session is not None and session.source_identity != request.source_identity:
                raise ValueError("This result belongs to a retired recording")
            if controls.response.currentData() != "off":
                current_pattern = self._response_at_current_pose(request, accepted_step, panel)
                if current_pattern is None:
                    raise ValueError(
                        "Waiting for matching current RX identity, pose and array metadata"
                    )
            if request.effective_display_step != accepted_step:
                if controls.response.currentData() not in {"broadside", "scan"}:
                    raise ValueError("Waiting for the saved acquisition's scene")
                # A recorded array response is independent of report time when
                # the current sensor has exactly the same identity and pose.
                # Report geometry and selected-report beams retain exact epochs.
                controls.stop(notify=False)
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            self.clear()
            controls.stop(notify=False)
            controls.note.setText(str(exc))
            controls.response_note.setText(str(exc))
            return
        self._sync_geometry(request, selection, controls)
        self._sync_response(request, selection, controls, pattern=current_pattern)

    def _response_at_current_pose(self, request, accepted_step, panel):
        """Reuse a recorded array only at an explicitly matching accepted RX pose.

        Accepted geometry, cached device inventory and already indexed metadata
        prove the current array. Missing evidence never triggers a reader.
        """
        session = getattr(panel, "_dashboard_session", None)
        service = getattr(self.viz, "extension_services", {}).get("experiments")
        state = getattr(self.viz, "app_state", None)
        token = getattr(service, "token", None)
        if (
            panel is None
            or getattr(panel, "_sensing_window", None) is not self.dashboard
            or session is None
            or session.source_identity != request.source_identity
            or session.provider_identity(session.provider) != request.source_identity
            or getattr(service, "step", None) != accepted_step
            or not token
            or token[0] != getattr(self.viz, "_mpc_presented_source_epoch", None)
        ):
            return None
        pattern = receive_pattern_for_request(request)
        sensing = request.display_frame["sensing"]
        current = session.service.cached_dashboard_result_metadata(
            accepted_step, selection_scope(sensing)
        )
        cache = getattr(self.viz, "cache_service", None)
        frame = cache.get_frame(accepted_step) if cache is not None else None
        signature = receive_metadata_signature(sensing)
        if (
            current is None
            or frame is None
            or signature is None
            or current.get("_receive_array_signature") != signature
        ):
            return None
        acquisition = sensing["temporal"]["acquisition"]
        pair_index = sensing["pair_axis"][0]["frame_pair_index"]
        pairs = np.asarray(frame.get("tx_rx_pairs"))
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
            return None
        if not 0 <= pair_index < len(pairs):
            return None
        for column, role in enumerate(("tx", "rx")):
            index = int(pairs[pair_index, column])
            names = tuple(getattr(state, f"{role}_device_names", ()))
            recorded = tuple(frame.get(f"{role}_names", ()))
            sensor_id = acquisition[f"{role}_sensor_id"]
            if (
                not 0 <= index < min(len(names), len(recorded))
                or names[index] != sensor_id
                or recorded[index] != sensor_id
                or names.count(sensor_id) != 1
                or recorded.count(sensor_id) != 1
                or getattr(state, f"selected_{role}", None) not in {"all", index}
            ):
                return None
        rx_index = int(pairs[pair_index, 1])
        view = getattr(service, "view_model", None)
        for field, expected in (
            ("rx_positions", pattern.position_m),
            ("rx_orientations", pattern.orientation_rad),
        ):
            values = np.asarray(getattr(view, field, None))
            if (
                values.ndim != 2
                or values.shape[1] != 3
                or rx_index >= len(values)
                or not np.array_equal(values[rx_index], expected)
            ):
                return None
        return pattern

    def _sync_geometry(self, request, selection, controls):
        phase = controls.phase.currentData()
        if phase == "off":
            if self.cinematic is not None:
                self.cinematic.clear()
            if controls.play.isEnabled():
                controls.note.setText(
                    "Select a saved report, then reveal range, coarse angle and position."
                )
            return
        if self.cinematic is None:
            from .cinematic_reveal import CinematicReveal

            self.cinematic = CinematicReveal(self.viz, controls)
        try:
            if selection is None:
                raise ValueError("Select a saved report or track first")
            self.cinematic.sync(request, selection, self.dashboard)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.cinematic.clear()
            controls.stop(notify=False)
            controls.note.setText(f"Cinematic reveal unavailable: {exc}")

    def _sync_response(self, request, selection, controls, *, pattern=None):
        mode = controls.response.currentData()
        if mode == "off":
            self.desired_response = None
            if self.response_key is not None and self.overlay is not None:
                self.overlay.remove("response")
            self.response_key = None
            self._stop_polling()
            return
        try:
            if pattern is None:
                pattern = receive_pattern_for_request(request)
            beam = selected_beam_index(request, selection, pattern) if mode == "beam" else None
            if mode == "scan" and not pattern.scan_enabled:
                raise ValueError("This recording has no coarse receive scan")
            key = (pattern, mode, controls.radius.value(), beam)
            self.desired_response = key
            if key != self.response_key and self.response_key is not None:
                self._layer().remove("response")
                self.response_key = None
            if self.failed_response is not None and self.failed_response[0] == key:
                controls.response_note.setText(
                    f"RX response unavailable: {self.failed_response[1]}"
                )
                return
            if key in self.response_cache:
                if key != self.response_key:
                    self._layer().receive_response(self.response_cache[key])
                    self.response_key = key
                    description = {
                        "broadside": "Broadside beam",
                        "scan": f"Maximum of {len(pattern.beam_directions)} coarse beams",
                        "beam": f"Saved coarse beam {beam}",
                    }[mode]
                    controls.response_note.setText(
                        f"{pattern.rows} × {pattern.columns} RX · {description}. "
                        "Relative array power −40…0 dB; arbitrary drawing radius. "
                        "Front/back lobes retained. Not received power or detection coverage."
                    )
                return
            controls.response_note.setText("Preparing the recorded RX response…")
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rx-response")
            if self.poll_timer is None or not isValid(self.poll_timer):
                self.poll_timer = QTimer(self.dashboard)
                self.poll_timer.setInterval(40)
                self.poll_timer.timeout.connect(self._poll)
            if self.pending is None:
                self.pending = (
                    key,
                    self.executor.submit(
                        response_object,
                        pattern,
                        mode,
                        controls.radius.value(),
                        0.4,
                        beam_index=beam,
                    ),
                )
            self.poll_timer.start()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.desired_response = self.response_key = None
            if self.overlay is not None:
                self.overlay.remove("response")
            self._stop_polling()
            controls.response_note.setText(f"RX response unavailable: {exc}")

    def _poll(self):
        if self.pending is None or not self.pending[1].done():
            return
        key, future = self.pending
        self.pending = None
        self._stop_polling()
        try:
            result = future.result()
        except (ValueError, TypeError, RuntimeError) as exc:
            self.failed_response = (key, str(exc))
            if self.dashboard is not None and key == self.desired_response:
                self.dashboard.scene_explanation.response_note.setText(
                    f"RX response unavailable: {exc}"
                )
            elif self.dashboard is not None:
                self.sync(self.dashboard, self.accepted_step)
            return
        if len(self.response_cache) >= 3:
            self.response_cache.pop(next(iter(self.response_cache)))
        self.response_cache[key] = result
        # Revalidate source, accepted scene and controls before publishing. An
        # obsolete request can warm this bounded cache but cannot draw its mesh.
        if self.dashboard is not None:
            self.sync(self.dashboard, self.accepted_step)

    def close(self):
        if self.cinematic is not None:
            self.cinematic.close()
            self.cinematic = None
        controls = getattr(self.dashboard, "scene_explanation", None)
        if controls is not None and isValid(controls):
            controls.stop(notify=False)
        self.clear()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = None
        self.pending = None
        self.dashboard = None
        self.response_cache.clear()
        self.failed_response = None
