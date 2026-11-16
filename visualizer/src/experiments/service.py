"""Accepted-frame lifecycle and latest-only analysis for visual experiments."""

from __future__ import annotations

import logging
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Signal
from PySide6.QtWidgets import QWidget

from ..metrics.mpc_path_catalog import MpcPathCatalog
from .path_analysis import (
    SPEED_OF_LIGHT_M_PER_NS,
    DelayAngleHistogram,
    PathAnalysisData,
    arrival_gate,
    delay_angle_histogram,
    geometric_progress,
    select_paths,
)

if TYPE_CHECKING:
    from ..pipeline.core import FrameRenderPacket, ViewModel
    from ..renderers.pygfx.experiments import ExperimentOverlay
    from .window import ExperimentWindow

_AnalysisResult = tuple[PathAnalysisData, DelayAngleHistogram, float]
_FrameToken = tuple[int, int, int]

logger = logging.getLogger(__name__)


class _Bridge(QObject):
    completed = Signal(int, object, object)


def _analyze(catalog, settings):
    tx, rx = settings["pair"]
    data = select_paths(
        catalog, tx_id=tx, rx_id=rx, scope=settings["scope"], weight_mode=settings["weight_mode"]
    )
    if settings["weight_mode"] == "power" and not data.power_available:
        data = select_paths(
            catalog, tx_id=tx, rx_id=rx, scope=settings["scope"], weight_mode="count"
        )
    angle = settings["angle"]
    available = data.available_angles
    if available and angle not in available:
        angle = available[0]
    histogram = delay_angle_histogram(
        data,
        angle=angle,
        angular_bin_width_deg=settings["angular_bin_width_deg"],
        delay_bin_width_ns=settings["delay_bin_width_ns"],
    )
    progress = geometric_progress(catalog, data.path_ids, 0.0, max_paths=1500)
    finite = progress.geometric_arrival_ns[progress.valid]
    clock_end = float(np.max(finite)) if len(finite) else 1.0
    return data, histogram, clock_end


class ExperimentService:
    """An optional view; accepted canonical arrays and source identities stay read-only."""

    def __init__(self, viz):
        self.viz = viz
        from .range_shell import RangeShellController

        self.range_shell = RangeShellController(viz)
        self.window: ExperimentWindow | None = None
        self.overlay: ExperimentOverlay | None = None
        self.catalog: MpcPathCatalog | None = None
        self._catalog_signature: tuple[_FrameToken, int, int, bool] | None = None
        self.packet: FrameRenderPacket | None = None
        self.view_model: ViewModel | None = None
        self.token: _FrameToken | None = None
        self.step: int | None = None
        self.data: PathAnalysisData | None = None
        self.histogram: DelayAngleHistogram | None = None
        self.path_ids: np.ndarray = np.empty(0, np.int32)
        self._serial = 0
        self._future: Future[_AnalysisResult] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-experiment")
        self._bridge = _Bridge()
        self._bridge.completed.connect(self._completed)
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(90)
        self._debounce.timeout.connect(self._submit)
        self._clock = QTimer()
        self._clock.setInterval(40)
        self._clock.timeout.connect(self._tick)
        self._cinematic_geometry = None
        self._cinematic_label_key = None
        self._cinematic_color_key = None
        self._cinematic_rgb = None
        self._selected_population = np.empty(0, np.int32)
        self._clock_time = 0.0
        self._clock_end = 1.0
        self._clock_last_wall = None
        self._presentation_speed = 1.0
        self._cell_gate: tuple | None = None
        self._cell_path_selection: tuple | None = None
        self._reset_gate = True
        self._retired = False
        self._explorer = None

    @property
    def active(self):
        return self.window is not None and not self.window.isHidden() and not self._retired

    def _bind_explorer(self):
        """Borrow the table's catalog and single-path selection while analysis is open."""
        if self._explorer is not None and not self._explorer._finished:
            return
        from ..services.mpc_explorer_service import ensure_mpc_explorer

        self._explorer = ensure_mpc_explorer(self.viz)
        self._explorer.retain_analysis()
        self._explorer.scopeChanged.connect(self._scope_changed)
        self._explorer.selectedDetailsChanged.connect(self.window.set_selected_path)
        self._explorer.matchingPathsCleared.connect(self.clear_overlays)
        self._explorer.closed.connect(self._workspace_closed)
        with QSignalBlocker(self.window.scope):
            self.window.scope.setCurrentIndex(self.window.scope.findData(self._explorer.scope))
        self.window.set_selected_path(getattr(self._explorer._selection_port, "details", None))

    def _workspace_closed(self):
        """Retire callbacks before the shared session's Qt objects are deleted."""
        self.pause()
        if self.window is not None:
            self.window.close()

    def _scope_changed(self, scope):
        if self.window is None or self.window.scope.currentData() == scope:
            return
        with QSignalBlocker(self.window.scope):
            self.window.scope.setCurrentIndex(self.window.scope.findData(scope))
        self._options_changed()

    def show_paths(self):
        """Raise the shared table without losing the current plot selection."""
        self._bind_explorer()
        self._explorer.open()

    def open(self):
        if self._retired:
            return
        if self.window is None:
            from ..renderers.pygfx.experiments import ExperimentOverlay
            from .window import ExperimentWindow

            self.window = ExperimentWindow(self.viz if isinstance(self.viz, QWidget) else None)
            self.overlay = ExperimentOverlay(self.viz.renderer)
            self.window.options_changed.connect(self._options_changed)
            self.window.gate_changed.connect(self.gate)
            self.window.bin_selected.connect(self.select_bin)
            self.window.clock_changed.connect(self.clock)
            self.window.play_changed.connect(self.play)
            self.window.cinematic_changed.connect(self._cinematic_changed)
            self.window.interaction_mode_changed.connect(self._interaction_mode_changed)
            self.window.presentation_speed_changed.connect(self._presentation_speed_changed)
            self.window.presentation_style_changed.connect(self._interaction_mode_changed)
            self.window.restart_requested.connect(self._restart_presentation)
            self.window.interaction_step_requested.connect(self._step_interaction)
            self.window.paths_requested.connect(self.show_paths)
            self.window.cleared.connect(self.clear_overlays)
            self.window.closed.connect(self.pause)
        from ..services.paused_presentation import frames_advancing

        self.set_presentation_playing(frames_advancing(self.viz))
        self.window.show()
        self._bind_explorer()
        table = self._explorer.window
        if (
            table is not None
            and not table.isHidden()
            and not table.isFloating()
            and self.window.isFloating()
        ):
            self.window.join(table, tabbed=True)
        self.window.raise_()
        self.window.activateWindow()
        self.viz.force_update_next_frame = True
        self.viz.schedule_update()
        if self.packet is not None:
            self.present(self.step, self.view_model, self.packet)

    def present(self, step, view_model, packet):
        """Called exclusively after the primary renderer accepts the frame."""
        canonical = getattr(packet, "canonical_data", None)
        token = (int(getattr(self.viz, "_mpc_presented_source_epoch", 0)), int(step), id(canonical))
        changed = token != self.token
        self.packet, self.view_model, self.step = packet, view_model, int(step)
        self.range_shell.present(int(step))
        self.token = token
        if not self.active:
            return
        assert self.window is not None
        self._bind_explorer()
        self._explorer._on_presented_frame(token[0], token[1], view_model, packet)
        visibility = getattr(view_model, "mpc_visibility", None)
        signature = (
            token,
            id(getattr(view_model, "path_mask", None)),
            id(getattr(packet, "segment_mask", None)),
            bool(getattr(visibility, "effective_paths", True)),
        )
        if signature == self._catalog_signature and self.catalog is not None:
            self.refresh_cinematic_style()
            return
        self._catalog_signature = signature
        self._serial += 1
        self._debounce.stop()
        self.catalog = None
        self.clear_overlays()
        self.data = None
        self.histogram = None
        self.window.clear_analysis()
        if changed:
            self._reset_gate = True
            self._cell_gate = None
            self.window.selection_context.hide()
        if canonical is None:
            self.catalog = None
            self.window.set_status("This accepted frame has no canonical path data")
            return
        self.catalog = self._explorer.catalog
        if self.catalog is None:
            self._catalog_signature = None
            self.window.set_status("Path Explorer could not index this frame")
            return
        pair_ids = np.unique(np.column_stack((self.catalog.tx_ids, self.catalog.rx_ids)), axis=0)
        pairs = [(int(tx), int(rx)) for tx, rx in pair_ids if tx >= 0 and rx >= 0]
        self.window.set_pairs(pairs)
        self.window.set_frame_identity(
            f"Accepted frame {step} · source epoch {token[0]} · frame-local path identities"
        )
        self._debounce.start()

    def _options_changed(self):
        assert self.window is not None
        if self._explorer is not None:
            self._explorer._on_scope_changed(self.window.scope.currentData())
        self._serial += 1
        self._cell_gate = None
        self.window.selection_context.hide()
        self._reset_gate = True
        self.data = None
        self.histogram = None
        self.window.clear_analysis()
        self.clear_overlays()
        self._debounce.start()

    def _submit(self):
        if not self.active or self.catalog is None:
            return
        assert self.window is not None
        settings = self.window.settings()
        if settings["pair"] is None:
            self.window.set_status("No valid TX/RX pair in this frame")
            return
        self._serial += 1
        serial = self._serial
        if self._future is not None:
            self._future.cancel()
        self.window.set_status("Analyzing selected pair and scope…")
        self._future = self._executor.submit(_analyze, self.catalog, settings)

        def done(future):
            if future.cancelled():
                return
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            if not self._retired:
                try:
                    self._bridge.completed.emit(serial, result, error)
                except RuntimeError:
                    pass

        self._future.add_done_callback(done)

    def _completed(self, serial, result, error):
        if not self.active or serial != self._serial:
            return
        assert self.window is not None
        if error is not None:
            self.clear_overlays()
            self.window.set_status(f"Analysis unavailable: {error}")
            logger.warning("Visual experiment analysis failed: %s", error)
            return
        self.data, self.histogram, self._clock_end = result
        reset = self._reset_gate
        self._reset_gate = False
        self.window.set_analysis(
            self.data, self.histogram, reset_gate=reset, clock_end=self._clock_end
        )
        pending, self._cell_gate = self._cell_gate, None
        if pending is not None:
            token, lower, upper, dashboard_ref, selection = pending
            dashboard = dashboard_ref()
            if (
                token == self.token
                and dashboard is not None
                and dashboard._linked_selection == selection
                and dashboard._last_request is not None
            ):
                self.window.gate.setRegion((lower, upper))
                self._cell_path_selection = pending
                self.window.selection_context.show()
                candidates = arrival_gate(self.data, lower, upper)
                self._publish_matching(
                    candidates.path_ids,
                    f"Sensing delay-matched candidates · [{lower:.3f}, {upper:.3f}) ns; Doppler not tested",
                )
                dashboard.cell_explanation.paths_note.setText(
                    f"{candidates.path_count} delay-matched paths · Doppler not tested. "
                    "Exact per-path RD contributions are unavailable."
                )
            else:
                self.clear_overlays()

    def _publish_matching(self, ids, reason):
        """Publish every matching ID; only the scene overlay has a display cap."""
        description = f"Matching paths: {len(ids):,} · {reason}"
        self.window.set_matching_paths(description)
        if self._explorer is not None and self.token is not None:
            from ..services.mpc_explorer_service import MpcPresentedFrameToken

            self._explorer.set_matching_paths(MpcPresentedFrameToken(*self.token), ids, description)

    def _emphasize(self, ids, reason):
        assert self.window is not None and self.overlay is not None and self.catalog is not None
        self.play(False)
        self.window.play.setChecked(False)
        self._clock_time = 0.0
        self._selected_population = np.asarray(ids, dtype=np.int32)
        self.overlay.remove("paths")
        self.overlay.remove("pulses")
        self.overlay.remove("directions")
        self._cinematic_changed()
        self._publish_matching(ids, reason)
        return len(self.path_ids)

    def gate(self, lower, upper):
        if self.data is None or not self.active:
            return
        assert self.window is not None
        self._release_cell_paths()
        chosen = arrival_gate(self.data, min(lower, upper), max(lower, upper))
        count = self._emphasize(
            chosen.path_ids, f"Delay gate [{min(lower, upper):.3f}, {max(lower, upper):.3f}] ns"
        )
        quantiles = ", ".join(
            f"{v:.3g}" if np.isfinite(v) else "unavailable" for v in chosen.percentiles_ns
        )
        self.window.set_status(
            "",
            f"{chosen.path_count} paths in delay gate · {chosen.weight_share:.1%} of pair/scope weight · gate τ10/τ50/τ90: {quantiles} ns.\nScene shows {count}/{chosen.path_count} paths (metrics use all eligible rows). "
            + " ".join(self.data.disclosures),
        )

    def select_bin(self, delay, angle):
        if self.histogram is None or not self.active:
            return
        assert self.window is not None and self.overlay is not None
        self._release_cell_paths()
        self.play(False)
        ids = self.histogram.path_ids_for_bin(delay, angle)
        low, high = self.histogram.delay_interval_ns(delay)
        left, right = self.histogram.angle_interval_deg(angle)
        count = self._emphasize(
            ids, f"Delay–angle cell · [{low:.3f}, {high:.3f}) ns; [{left:.1f}, {right:.1f})°"
        )
        self.overlay.directions(
            self.catalog, self.path_ids, departure=self.window.settings()["angle"].startswith("aod")
        )
        self.window.set_status(
            "",
            f"Selected numerical cell: delay [{low:.3f}, {high:.3f}) ns; angle [{left:.1f}, {right:.1f})°. {len(ids)} contributing paths; {count} drawn.\n"
            + " ".join(self.histogram.disclosures),
        )

    def clock(self, time_ns):
        if self.catalog is None or not self.active:
            return
        assert self.window is not None and self.overlay is not None
        self._clock_time = float(time_ns)
        # A seek while playing starts a fresh wall-time interval.
        if self._clock.isActive():
            self._clock_last_wall = time.monotonic()
        from ..services.paused_presentation import frames_advancing

        if frames_advancing(self.viz):
            self.set_presentation_playing(True)
            return
        if self._cinematic_geometry is not None:
            sample = self._cinematic_geometry.sample(
                self._clock_time,
                interaction_mode=self.window.interaction_mode.currentData(),
                trail_mode=self.window.trail_mode.currentData(),
                dwell_seconds=self.window.interaction_dwell.value(),
            )
            self._apply_cinematic_appearance(sample)
        self.window.set_clock(self._clock_time)

    def _apply_cinematic_appearance(self, sample):
        from .cinematic_appearance import interaction_colors, interaction_label_specs

        geometry = self._cinematic_geometry
        state = getattr(self.viz, "app_state", None)
        core = getattr(self.viz, "mpc_core", None)
        mode = getattr(state, "color_mode", "reflection_order")
        distinct = bool(getattr(state, "use_distinct_material_colors", False))
        match = self.window.match_mpc_colors.isChecked()
        show_types = bool(getattr(state, "show_mpc_type_markers", False))
        key = (
            id(geometry),
            id(self.packet),
            mode,
            distinct,
            match,
            id(getattr(core, "_order_palette", None)),
            id(getattr(core, "_type_palette", None)),
        )
        if key != self._cinematic_color_key:
            if match:
                self._cinematic_rgb, note = interaction_colors(
                    self.catalog,
                    geometry.interaction_point_indices,
                    core,
                    mode,
                    distinct_materials=distinct,
                )
            else:
                self._cinematic_rgb = None
                note = "Warm interaction highlight; RGB is presentation-only"
            self._cinematic_color_key = key
            self._cinematic_color_note = note
        if self._cinematic_rgb is not None:
            sample.interaction_colors[:, :3] = self._cinematic_rgb
        label_key = (id(geometry), self.window.interaction_labels.isChecked(), show_types)
        if label_key != self._cinematic_label_key:
            self._cinematic_labels = interaction_label_specs(
                geometry, enabled=label_key[1], show_types=show_types
            )
            self._cinematic_label_key = label_key
        labels = self._cinematic_labels
        note = self._cinematic_color_note
        if labels:
            note += f" · Labels {len(labels)}/{len(geometry.vertices)}; P path, I ordinal; on-top text (not visibility)"
        note += (
            " · Recorded type glyphs" if show_types else " · Plain markers (MPC type option off)"
        )
        self.window.set_interaction_note(note)
        self.overlay.cinematic_pulses(
            sample,
            interaction_types=geometry.interaction_types,
            type_markers=show_types,
            label_specs=labels,
        )

    def refresh_cinematic_style(self):
        """React to accepted palette or existing type-marker UI changes while paused."""
        if self.active and self._cinematic_geometry is not None:
            self._cinematic_color_key = None
            self._interaction_mode_changed()

    def set_presentation_playing(self, playing):
        if self.window is None:
            return
        self.window.play.setEnabled(not playing)
        self.window.cinematic_limit.setEnabled(not playing)
        self.window.presentation_options.setEnabled(not playing)
        self.window.interaction_mode.setEnabled(not playing)
        self.window.presentation_speed.setEnabled(not playing)
        if playing:
            self.play(False)
            self.window.play.setChecked(False)
            self._cinematic_geometry = None
            if self.overlay is not None:
                self.overlay.clear_cinematic()
            self.window.cinematic_note.setText(
                "Pause scene playback to animate the selected paths."
            )
        elif self._cinematic_geometry is None:
            self.window.cinematic_note.setText("Select paths, then press Play geometric clock.")

    def _cinematic_changed(self):
        if self.window is None or self.overlay is None:
            return
        resume = self._clock.isActive()
        self.play(False)
        self.window.play.setChecked(False)
        self.overlay.clear_cinematic()
        self._cinematic_geometry = None
        self._cinematic_label_key = None
        self._cinematic_color_key = None
        self._cinematic_rgb = None
        self.path_ids = np.empty(0, np.int32)
        from ..services.paused_presentation import frames_advancing
        from .cinematic_pulses import CinematicPulseGeometry, cinematic_subset

        if frames_advancing(self.viz):
            self.set_presentation_playing(True)
            return
        if self.catalog is None or not len(self._selected_population):
            self.window.cinematic_note.setText("Select a delay gate or delay–angle cell first.")
            return
        ids, note = cinematic_subset(
            self.catalog, self._selected_population, self.window.cinematic_limit.currentData()
        )
        self._cinematic_geometry = CinematicPulseGeometry(self.catalog, ids)
        self.path_ids = self._cinematic_geometry.ids
        self.window.cinematic_note.setText(note)
        self.overlay.remove("pulses")
        self.overlay.cinematic_routes(self.catalog, self._cinematic_geometry.ids)
        self.overlay.set_cinematic_halo_opacity(self.window.halo_opacity.value())
        self._clock_end = self._cinematic_geometry.extent
        self.window.set_clock_extent(self._clock_end)
        self._clock_time = 0.0
        self.clock(0.0)
        self.window.play.setChecked(resume)

    def _interaction_mode_changed(self):
        # Existing vertex/color buffers are reused; no route rebuild or timer.
        if self.window is not None and self._cinematic_geometry is not None:
            if self._clock.isActive():
                self._tick()
            else:
                self.clock(self._clock_time)
            self.overlay.set_cinematic_halo_opacity(self.window.halo_opacity.value())

    def _restart_presentation(self):
        if self.window is None or self._cinematic_geometry is None:
            return
        self.clock(0.0)
        self.window.play.setChecked(True)
        self.play(True)

    def _step_interaction(self, direction):
        if self.window is None or self._cinematic_geometry is None:
            return
        moment = self._cinematic_geometry.interaction_time(self._clock_time, direction)
        self.window.play.setChecked(False)
        self.play(False)
        if moment is not None:
            self.clock(moment)

    def _presentation_speed_changed(self, speed):
        if speed not in (0.25, 0.5, 1.0, 2.0):
            raise ValueError("Unsupported presentation speed")
        # Account for elapsed time at the old rate before changing speed.
        if (
            self._clock.isActive()
            and self.window is not None
            and self._cinematic_geometry is not None
        ):
            self._tick()
        self._presentation_speed = float(speed)

    def play(self, enabled):
        if enabled and self.active:
            from ..services.paused_presentation import frames_advancing

            if frames_advancing(self.viz):
                self.set_presentation_playing(True)
                return
            if self._cinematic_geometry is None:
                self._cinematic_changed()
                if self._cinematic_geometry is None:
                    return
                with QSignalBlocker(self.window.play):
                    self.window.play.setChecked(True)
            if self._clock_time >= self._clock_end:
                self._clock_time = 0.0
            self._clock_last_wall = time.monotonic()
            self._clock.start()
        else:
            self._clock.stop()
            self._clock_last_wall = None

    def _tick(self):
        if not self.active:
            self._clock.stop()
            return
        assert self.window is not None
        now = time.monotonic()
        elapsed = 0.0 if self._clock_last_wall is None else max(0.0, now - self._clock_last_wall)
        advance = self._clock_end * elapsed * self._presentation_speed / 5.0
        target = self._clock_time + advance
        mode = self.window.presentation_end.currentData()
        if mode == "loop" and target >= self._clock_end:
            # Stateless sampling at wrapped time naturally clears reached markers.
            self.clock(target % max(self._clock_end, 1e-9))
        else:
            self.clock(min(self._clock_end, target))
        self._clock_last_wall = now
        if mode != "loop" and self._clock_time >= self._clock_end:
            self._clock.stop()
            self.window.play.setChecked(False)
            if mode == "once":
                self.overlay.clear_cinematic_pulses()

    def sync_sensing_selection(self, dashboard):
        """Update the shell and retire paths belonging to a previous cell."""
        self.range_shell.sync(dashboard)
        owned = self._cell_gate or self._cell_path_selection
        if owned is None or owned[3]() is not dashboard:
            return
        if dashboard._last_request is not None and dashboard._linked_selection == owned[4]:
            return
        # Cancel both queued and running work: a late result must not restore
        # paths after the user has cleared or changed the originating cell.
        self._serial += 1
        self._debounce.stop()
        if self._future is not None:
            self._future.cancel()
        self.clear_overlays()
        self.data = None
        self.histogram = None
        self._reset_gate = True
        if self.window is not None:
            self.window.clear_analysis()
            self.window.set_status(
                "Sensing cell selection cleared or changed. Inspect another cell or choose a path pair."
            )

    def _release_cell_paths(self):
        """A manual path gate or angle selection becomes independent of sensing."""
        self._cell_path_selection = None
        if self.window is not None:
            self.window.selection_context.hide()

    def open_cell_paths(self, dashboard):
        """Open a delay gate on the exact saved scene/pair, never assert RD attribution."""
        from .range_geometry import build_range_locus

        request, selection = dashboard._last_request, dashboard._linked_selection
        if request is None or selection is None or request.effective_display_step != self.step:
            raise ValueError("Wait for the selected result's scene to be displayed")
        panel = getattr(getattr(self.viz, "ui_manager", None), "panels", {}).get("sensing")
        session = getattr(panel, "_dashboard_session", None)
        if session is not None and session.source_identity != request.source_identity:
            raise ValueError("The selected result belongs to another recording")
        locus = build_range_locus(request, selection)
        if len(locus.support_m) != 2:
            raise ValueError("This single range coordinate has no saved bin boundaries")
        if self.packet is None or self.packet.canonical_data is None:
            raise ValueError("No propagation paths are saved for this scene")
        pair = request.display_frame["sensing"]["pair_axis"][0]
        tx, rx = pair.get("tx_device_index"), pair.get("rx_device_index")
        if type(tx) is not int or type(rx) is not int or min(tx, rx) < 0:
            raise ValueError("The recording does not identify the path TX/RX device indices")
        scale = (2.0 if request.range_mode == "monostatic" else 1.0) / SPEED_OF_LIGHT_M_PER_NS
        lower, upper = np.asarray(locus.support_m) * scale
        self.open()
        assert self.window is not None
        index = next(
            (
                i
                for i in range(self.window.pair.count())
                if self.window.pair.itemData(i) == (tx, rx)
            ),
            -1,
        )
        if index < 0:
            raise ValueError("No saved paths for the selected sensor pair")
        with QSignalBlocker(self.window.pair), QSignalBlocker(self.window.scope):
            self.window.pair.setCurrentIndex(index)
            self.window.scope.setCurrentIndex(self.window.scope.findData("all"))
        self._options_changed()
        self._cell_gate = (
            self.token,
            float(lower),
            float(np.nextafter(upper, lower)),
            weakref.ref(dashboard),
            selection,
        )
        self.window.tabs.setCurrentIndex(0)
        self.window.selection_context.setText(
            f"Opened from Sensing: range bin {selection.cell.range_bin} at "
            f"{selection.cell.ranges[selection.cell.range_bin]:.3f} m · delay [{lower:.3f}, {upper:.3f}) ns. "
            "Delay-matched candidates only; Doppler, beamforming and interference are not tested. "
            "Other delays can contribute through finite-bandwidth spreading."
        )
        self.window.selection_context.show()
        dashboard.cell_explanation.paths_note.setText("Loading delay-matched paths…")

    def clear_overlays(self):
        self._cinematic_geometry = None
        self._cinematic_label_key = None
        self._cinematic_color_key = None
        self._cinematic_rgb = None
        self._selected_population = np.empty(0, np.int32)
        self._cell_gate = None
        self._cell_path_selection = None
        self._clock.stop()
        self._clock_last_wall = None
        self._clock_time = 0.0
        self.path_ids = np.empty(0, np.int32)
        if self.overlay is not None:
            self.overlay.clear()
        if self._explorer is not None:
            self._explorer.clear_matching_paths()
        if self.window is not None:
            self.window.selection_context.hide()
            self.window.play.setChecked(False)
            self.window.set_matching_paths("")
            self.window.set_clock(0.0)
            self.window.set_status("")

    def pause(self):
        self._serial += 1
        self._debounce.stop()
        self._cell_gate = None
        if self._future is not None:
            self._future.cancel()
        self.clear_overlays()
        self.catalog = None
        self._catalog_signature = None
        self.data = None
        self.histogram = None
        explorer, self._explorer = self._explorer, None
        if self.window is not None:
            self.window.clear_analysis()
        if explorer is not None:
            explorer.scopeChanged.disconnect(self._scope_changed)
            explorer.selectedDetailsChanged.disconnect(self.window.set_selected_path)
            explorer.matchingPathsCleared.disconnect(self.clear_overlays)
            explorer.closed.disconnect(self._workspace_closed)
            explorer.release_analysis()

    def close(self):
        if self._retired:
            return
        self.pause()
        self.range_shell.close()
        self._retired = True
        # Disconnect the service's own callbacks before releasing their receiver.
        for signal, callback in (
            (self._bridge.completed, self._completed),
            (self._debounce.timeout, self._submit),
            (self._clock.timeout, self._tick),
        ):
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                # A connection may already have been retired during shutdown.
                pass
        if self.window is not None:
            self.window.close()
            self.window.retire_plots()
            self.window.deleteLater()
            self.window = None
        self.packet = None
        self.view_model = None
        self._executor.shutdown(wait=False, cancel_futures=True)
