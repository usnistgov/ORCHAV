"""Fusion controls beside one sensor's native RD/angular dashboard and world view."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..panels.collapsible_section import CollapsibleSection
from ..services.sensing_overlay_sync import SensingOverlaySync
from ..widgets.wrapping_layout import WrappingLayout
from .fusion_comparison import FusionComparison
from .fusion_inspection import (
    FusionInspectionSession,
    FusionSelection,
    world_layers,
    world_sensor_labels,
)
from .sensing_window import SensingDashboard


class FusionWindow(SensingDashboard):
    """Reuse native plot/selection presentation; all fusion decisions are read-only."""

    _SOURCE_RECORDING_WORKSPACES = {
        "summary": "Run summary",
        "micro-doppler": "Motion",
    }

    def _set_view_mode(self, mode):
        """Keep restored native views within the selected fusion-result scope."""
        if mode in self._SOURCE_RECORDING_WORKSPACES or mode == "evaluation":
            mode = "current"
        super()._set_view_mode(mode)

    def _sync_workspace_navigation(self):
        """Reserve whole-recording analysis for the sensor's source workspace."""
        super()._sync_workspace_navigation()
        workspace = getattr(self, "workspace_box", None)
        if workspace is None:
            return
        # Native evidence follows one exact fusion lineage entry. Product
        # availability alone does not supply its whole-recording history or
        # summary, so keep these choices disabled after each native update.
        for mode, label in self._SOURCE_RECORDING_WORKSPACES.items():
            reason = f"Open this sensor's source recording to view its {label}."
            for combo in (workspace, self._view_mode_box):
                index = combo.findData(mode)
                if index >= 0:
                    item = combo.model().item(index)
                    item.setEnabled(False)
                    item.setToolTip(reason)

    def __init__(self, visualizer, path):
        from .qt_lifetime import ensure_gui_garbage_collection

        ensure_gui_garbage_collection()
        super().__init__()
        self.setWindowTitle("ORCHAV Sensing — Fusion inspection")
        self._initial_view_selected = True
        self._set_view_mode("analysis")
        # Fusion owns the common timeline; do not show a second native index.
        self.previous_result_button.hide()
        self.next_result_button.hide()
        self.navigation_label.hide()
        self.visualizer = visualizer
        self.result_path = str(Path(path).resolve())
        self.reader = None
        self.world_aligned = True
        self.session = FusionInspectionSession()
        self.opening = self.session.open(path)
        self.pending = None
        self.pending_key = self.completed_key = None
        self.selected = None
        self.overlay = SensingOverlaySync()
        self.closed = False
        self.result_playback = None
        self.paused = False
        self._following_scene = False
        self.visibility = {}
        self.tracking_histories = {}
        # Reuse the native dashboard as optional evidence beneath the common
        # timeline. This is a panel widget, never an extra application window.
        self.native_products = QWidget()
        self.native_products.setLayout(self.layout())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        body = self
        self.fusion_title = QLabel("Saved multi-sensor fusion")
        header = QHBoxLayout()
        header.addWidget(self.fusion_title, 1)
        close_result = QPushButton("Close result")
        close_result.clicked.connect(self.close)
        header.addWidget(close_result)
        layout.addLayout(header)
        self.fusion_status = QLabel("Loading fusion result…", body)
        self.fusion_status.setWordWrap(True)
        layout.addWidget(self.fusion_status)
        self.world_notice = QLabel(body)
        self.world_notice.setWordWrap(True)
        layout.addWidget(self.world_notice)
        self.world_notice.hide()
        self.fusion_time = QLabel(body)
        layout.addWidget(self.fusion_time)
        self.fusion_navigation = QWidget()
        navigation = QHBoxLayout(self.fusion_navigation)
        navigation.setContentsMargins(0, 0, 0, 0)
        previous, following = QPushButton("Previous", body), QPushButton("Next", body)
        previous.clicked.connect(lambda: self.timeline.setValue(self.timeline.value() - 1))
        following.clicked.connect(lambda: self.timeline.setValue(self.timeline.value() + 1))
        navigation.addWidget(previous)
        self.timeline = QSlider(Qt.Horizontal, body)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._epoch_changed)
        navigation.addWidget(self.timeline, 1)
        navigation.addWidget(following)
        layout.addWidget(self.fusion_navigation)
        self.sensor_choice = QComboBox(body)
        self.sensor_choice.currentTextChanged.connect(self._source_changed)
        overlays = CollapsibleSection("Scene overlays", start_open=False)
        self.visibility_layout = WrappingLayout()
        overlays.content_layout().addLayout(self.visibility_layout)
        layout.addWidget(overlays)
        self.track_summary = QLabel("Loading saved tracks…")
        layout.addWidget(self.track_summary)
        self.track_choice = QComboBox(body)
        self.track_choice.setToolTip("Select an estimate to compare its accepted sensor reports.")
        self.track_choice.currentIndexChanged.connect(self._track_selected)
        layout.addWidget(self.track_choice)
        self.comparison = FusionComparison()
        layout.addWidget(self.comparison)
        self.pause_button = QPushButton("Cancel native loading", body)
        self.pause_button.clicked.connect(self._pause_loading)
        self.native_section = CollapsibleSection(
            "Sensor products", start_open=False, expand_content=True
        )
        native_layout = self.native_section.content_layout()
        source_row = WrappingLayout()
        source_row.addWidget(QLabel("Sensor"))
        source_row.addWidget(self.sensor_choice)
        source_row.addWidget(self.pause_button)
        native_layout.addLayout(source_row)
        native_layout.addWidget(self.native_products, 1)
        layout.addWidget(self.native_section, 1)
        self.native_section.toggled.connect(self._source_changed)
        self.uav_report_button = QPushButton("Open UAV evaluation report", body)
        self.uav_report_button.setToolTip(
            "Saved detections and tracks: TP/FP/FN, 3GPP metric definitions, accuracy and coverage. "
            "Opens the separate report; does not rerun the detector."
        )
        self.uav_report_button.hide()
        self.uav_report_button.clicked.connect(self._open_uav_report)
        layout.addWidget(self.uav_report_button)
        self.view_changed.connect(self._source_changed)
        self.result_seek_requested.connect(
            lambda result: self.timeline.setValue(self.reader.epoch_index(result.time_s))
        )
        self.result_step_requested.connect(
            lambda direction: self.timeline.setValue(self.timeline.value() + direction)
        )
        self.selection_changed.connect(self._native_selected)
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        # Cleanup also runs for parent-driven deleteLater, with no widget access.
        self.destroyed.connect(self.session.close)

    def _finish_open(self, result):
        single_sensor = len(result.get("sources", {})) == 1
        if single_sensor:
            self.setWindowTitle("ORCHAV Sensing — Saved 3D tracking inspection")
            self.fusion_title.setText("Saved 3D tracking")
        self.uav_report_button.setVisible(self._uav_report_path().is_file())
        current = getattr(self.visualizer, "current_scenario_path", None)
        current = Path(current).resolve() if current else None
        if current is not None and current.suffix in {".yaml", ".yml"}:
            current = current.parent
        sources = [result.source_path(name) for name in result.get("sources", {})]
        if current not in sources:
            source = next(
                (path for path in sources if path and (path / "scenario.yaml").is_file()), None
            )
            opener = getattr(self.visualizer, "open_scenario", None)
            if source is not None and callable(opener):
                outcome = opener(str(source), autorun_initial_frame=False)
                self.world_aligned = bool(getattr(outcome, "succeeded", False))
            else:
                self.world_aligned = current is None
        self.world_notice.setVisible(not self.world_aligned)
        if not self.world_aligned:
            self.world_notice.setText(
                "World overlay unavailable: the open scene is not a bound fusion source."
            )
        renderer = getattr(self.visualizer, "renderer", None)
        if (
            renderer is not None
            and self.world_aligned
            and not getattr(self.visualizer, "vis_initialized", True)
        ):
            from ..app.renderer_lifecycle import boot_visualizer_empty

            boot_visualizer_empty(self.visualizer)
        self.reader = result
        from .saved_tracking import saved_tracking_history

        self.tracking_histories = {
            name: saved_tracking_history(result, name) for name in result.get("sources", {})
        }
        saved = getattr(self.visualizer, "_fusion_inspection_state", {})
        saved = saved if saved.get("path") == self.result_path else {}
        self.timeline.setRange(0, max(0, len(result.times) - 1))
        self.timeline.setEnabled(bool(len(result.times)))
        self.fusion_navigation.setVisible(len(result.times) > 1)
        self.sensor_choice.addItems(list(result.get("sources", {})))
        options = [
            ("sensors", "Sensor poses", True),
            ("fused", "3D tracks" if single_sensor else "Fused tracks", True),
            ("trails", "Track trails", True),
            ("truth", "Ground truth", False),
        ]
        options += [(n, f"{n} detections", True) for n in result.get("sources", {})]
        if not single_sensor:
            options += [(f"tracks:{n}", f"{n} tracks", False) for n in result.get("sources", {})]
        for key, title, default in options:
            widget = QCheckBox(title, self)
            if key == "trails":
                widget.setToolTip(
                    "Show saved track states up to this epoch, including saved coasting states. "
                    "Missing epochs remain gaps."
                )
            widget.setChecked(saved.get("visible", {}).get(key, default))
            self.visibility[key] = widget
            self.visibility_layout.addWidget(widget)
            widget.toggled.connect(self._draw_world)
        if saved.get("sensor") in result.get("sources", {}):
            self.sensor_choice.setCurrentText(saved["sensor"])
        self.timeline.setValue(min(saved.get("index", 0), max(0, len(result.times) - 1)))
        self._epoch_changed()
        if self.world_aligned and renderer is not None:
            import numpy as np

            points = [
                source["sensor"]["position_m"] for source in result.get("sources", {}).values()
            ]
            points += (
                [t["position_m"] for t in result.epoch(self.timeline.value())["tracks"]]
                if len(result.times)
                else []
            )
            if points:
                points = np.asarray(points)
                camera = getattr(self.visualizer, "camera_controller", None)
                tools = camera.presentation() if camera is not None else None
                if tools is not None:
                    tools.prepare()
                    tools.free_camera()
                renderer.set_overview_camera(
                    "top",
                    (
                        (points.min(axis=0) + points.max(axis=0)) / 2,
                        np.maximum(np.ptp(points, axis=0), 1.0),
                    ),
                )
        context = getattr(self, "volume_context", None)
        if context and self.world_aligned:
            # A cell inspection should expose the saved contributions directly,
            # including the estimate selected in the volume's report table.
            selected_report = context.get("selected_report")
            for index in range(self.track_choice.count()):
                track_id = self.track_choice.itemData(index)
                if selected_report == f"track:{track_id}":
                    self.track_choice.setCurrentIndex(index)
                    track = next(
                        t
                        for t in result.epoch(self.timeline.value())["tracks"]
                        if t["track_id"] == track_id
                    )
                    support = track.get("current_sensor_support", [])
                    if support:
                        self.sensor_choice.setCurrentText(support[0])
            self.native_section.expand()
            from .volume_return import install_return

            native = self.visualizer.current_scenario_path
            native = Path(native)
            install_return(
                self.visualizer,
                context,
                native.parent if native.is_file() else native,
                native_sources=sources,
            )
            # Opening the source recording restores the main workflow tabs.
            # Return to the requested fusion workspace after that scene change.
            from .volume_workspace import reveal_sensing_panel

            reveal_sensing_panel(self.visualizer)
            self.visualizer.ui_manager.panels["sensing"].recording_tabs.setCurrentWidget(self)
        self.fusion_status.setText("Select an estimate to compare its sensor reports.")
        if (
            self.world_aligned
            and not getattr(self.visualizer, "current_scenario_path", None)
            and len(result.times)
            and hasattr(self.visualizer, "update_frame")
        ):
            from .fusion_playback import FusionPlayback

            self.result_playback = FusionPlayback(self)
            self.world_notice.setText(
                "Saved world tracks · native scene recordings are unavailable."
            )
            self.world_notice.show()

    def _uav_report_path(self):
        path = Path(self.result_path)
        directory = path if path.is_dir() else path.parent
        return directory / "uav_evaluation" / "index.html"

    def _open_uav_report(self):
        path = self._uav_report_path()
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _epoch_changed(self, *_args):
        if self.reader is None or not len(self.reader.times):
            return
        index = self.timeline.value()
        time = float(self.reader.times[index])
        self.fusion_time.setText(
            f"Common time {time:.6f} s · epoch {index+1}/{len(self.reader.times)}"
        )
        self.completed_key = None
        self.track_choice.blockSignals(True)
        self.track_choice.clear()
        tracks = self.reader.epoch(index)["tracks"]
        self.track_summary.setText(
            f"{len(tracks)} saved tracks · {len(self.reader.get('sources', {}))} sensors"
            if tracks
            else "No saved tracks at this epoch. Use the timeline to inspect another result."
        )
        for track in tracks:
            state = (
                "coasting"
                if track["predicted_only"]
                else ("confirmed" if track["confirmed"] else "tentative")
            )
            support = " + ".join(n.title() for n in track.get("current_sensor_support", []))
            self.track_choice.addItem(
                f"Track {track['track_id']} · {state}" + (f" · {support}" if support else ""),
                track["track_id"],
            )
            if self.selected and track["track_id"] == self.selected.track_id:
                self.track_choice.setCurrentIndex(self.track_choice.count() - 1)
        self.track_choice.blockSignals(False)
        self._track_selected()
        self._draw_world()
        if not self._following_scene:
            self._seek_scene(index)

    def _seek_scene(self, index):
        if (
            self.result_playback is not None
            and getattr(self.visualizer, "result_playback", None) is self.result_playback
        ):
            if self.visualizer.animation_running:
                self.visualizer.animation_controller.toggle_animation()
            self.visualizer.update_frame(index)
            return
        # The world scene remains bound to its own recording, regardless of which
        # native sensor is selected in the dashboard.
        current = getattr(self.visualizer, "current_scenario_path", None)
        if current is None:
            return
        current = Path(current).resolve()
        current = current.parent if current.suffix in {".yaml", ".yml"} else current
        for name, source in self.reader.get("sources", {}).items():
            if self.reader.source_path(name) != current:
                continue
            lineage = source.get("lineage", [])
            if index >= len(lineage):
                return
            steps = self.visualizer.get_available_animation_steps()
            scene = lineage[index]["scene_frame_id"]
            if scene in steps:
                if self.visualizer.animation_running:
                    self.visualizer.animation_controller.toggle_animation()
                self.visualizer.ui_controller.handle_frame_input_changed(steps.index(scene) + 1)
            return

    def follow_scene(self, step):
        """Snap normal scene navigation to an authored common result epoch."""
        if self.reader is None or not len(self.reader.times):
            return
        current = getattr(self.visualizer, "current_scenario_path", None)
        if current is None:
            return
        current = Path(current).resolve()
        current = current.parent if current.suffix in {".yaml", ".yml"} else current
        name = next(
            (n for n in self.reader.get("sources", {}) if self.reader.source_path(n) == current),
            None,
        )
        if name is None:
            self.world_aligned = False
            self.overlay.clear(self.visualizer.renderer)
            self.world_notice.setText(
                "World overlay unavailable: the open scene is not a bound fusion source."
            )
            self.world_notice.show()
            return
        self.world_aligned = True
        self.world_notice.clear()
        self.world_notice.hide()
        lineage = self.reader["sources"][name].get("lineage", [])
        if lineage:
            index = min(range(len(lineage)), key=lambda i: abs(lineage[i]["scene_frame_id"] - step))
            if index != self.timeline.value():
                # Following scene playback must never seek back into it: that
                # feedback used to pause playback on the very next frame.
                self._following_scene = True
                try:
                    self.timeline.setValue(index)
                finally:
                    self._following_scene = False

    def _source_changed(self, *_args):
        self.completed_key = None

    def _draw_world(self, *_args):
        if self.reader is None or not len(self.reader.times):
            return
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None or not self.world_aligned:
            return
        visible = {k for k, w in self.visibility.items() if w.isChecked()}
        layers = world_layers(
            self.reader,
            self.timeline.value(),
            visible,
            self.selected,
        )
        from ..services.label_callout_service import objects_with_callouts

        labels = objects_with_callouts(world_sensor_labels(self.reader, visible), self.visualizer)
        self.overlay.sync(renderer, [*(layer.to_render_object() for layer in layers), *labels])
        # The renderer's existing sensing adapter owns click bindings.
        from .runtime import bind_fusion_layers

        bind_fusion_layers(renderer, layers)
        request = getattr(renderer, "request_redraw", None)
        if callable(request):
            request()

    def select_fusion(self, selection):
        if self.reader is None or selection.run_id != self.reader.inspection_id:
            return
        self.selected = selection
        self.timeline.setValue(selection.epoch_index)
        with QSignalBlocker(self.track_choice):
            self.track_choice.setCurrentIndex(self.track_choice.findData(selection.track_id))
        self.comparison.update_result(self.reader, self.selected)
        self._draw_world()

    def _track_selected(self, *_args):
        track_id = self.track_choice.currentData()
        self.selected = (
            FusionSelection(self.reader.inspection_id, track_id, self.timeline.value())
            if track_id is not None
            else None
        )
        self.comparison.update_result(self.reader, self.selected)
        self._draw_world()
        self._highlight_native()

    def select_track(self, track_id):
        """Select saved world-state IDs without assigning IDs to native detections."""
        if self.reader is not None:
            self.select_fusion(
                FusionSelection(self.reader.inspection_id, int(track_id), self.timeline.value())
            )
            self._highlight_native()

    def _highlight_native(self):
        if self.selected is None or self._last_request is None:
            return
        track = next(
            (
                t
                for t in self.reader.epoch(self.timeline.value())["tracks"]
                if t["track_id"] == self.selected.track_id
            ),
            None,
        )
        if track is None:
            return
        for source in track["sources"]:
            if source["sensor_id"] == self.sensor_choice.currentText():
                native = source.get("native_source", {})
                sensing = self._last_request.display_frame["sensing"]
                if (
                    sensing["temporal"]["acquisition"]["acquisition_id"]
                    == native.get("acquisition_id")
                    and native.get("detection_collection", "detections")
                    == sensing.get("_display_detection_collection", "detections")
                    and isinstance(native.get("detection_index"), int)
                    and 0 <= native["detection_index"] < len(sensing.get("detections", []))
                ):
                    from .selection import select_estimate

                    self.set_linked_selection(
                        select_estimate(sensing, detection_index=native["detection_index"])
                    )
                return

    def _native_selected(self, selection):
        if selection is None or self.reader is None:
            return
        for track in self.reader.epoch(self.timeline.value())["tracks"]:
            for source in track["sources"]:
                native = source.get("native_source", {})
                if (
                    source["sensor_id"] == self.sensor_choice.currentText()
                    and native.get("acquisition_id") == selection.acquisition_id
                    and native.get("detection_index") == selection.detection_index
                    and native.get("detection_collection", "detections")
                    == selection.detection_collection
                ):
                    self.select_fusion(
                        FusionSelection(
                            self.reader.inspection_id, track["track_id"], self.timeline.value()
                        )
                    )
                    return

    def _pause_loading(self):
        self.paused = not self.paused
        self.session.close()
        self.pending = None
        if not self.paused:
            self.session = FusionInspectionSession()
            self.destroyed.connect(self.session.close)
            self.completed_key = None
        self.pause_button.setText(
            "Resume native loading" if self.paused else "Cancel native loading"
        )

    def _poll(self):
        if self.closed:
            return
        if self.opening is not None and self.opening.done():
            future, self.opening = self.opening, None
            try:
                self._finish_open(future.result())
            except Exception as exc:
                self.fusion_status.setText(f"Cannot open result: {exc}")
                return
        if (
            self.reader is None
            or self.paused
            or not len(self.reader.times)
            or not self.native_section.is_expanded()
        ):
            return
        key = (
            self.timeline.value(),
            self.sensor_choice.currentText(),
            self.request_view_mode(),
            self.product_variant(),
        )
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            if self.pending_key == key:
                try:
                    request = future.result()
                    self.reset_recording(preserve_options=True)
                    if request is not None:
                        request = replace(
                            request, tracking_history=self.tracking_histories.get(key[1])
                        )
                        self.update_from_request(request)
                        self._highlight_native()
                        self.fusion_status.setText(
                            f"{key[1]} native products · world speed and radial velocity are distinct"
                        )
                    else:
                        message = f"{key[1]}: no native products saved for this epoch. Tracks remain available above."
                        self.fusion_status.setText(message)
                        self.set_unavailable(message, scene_frame_id=None)
                except (CancelledError, OSError, ValueError, KeyError, RuntimeError) as exc:
                    self.reset_recording(preserve_options=True)
                    message = f"Native sensor products unavailable: {exc}. Saved tracks remain available above."
                    self.fusion_status.setText(message)
                    self.set_unavailable(message, scene_frame_id=None)
                self.completed_key = key
        if self.pending is None and key != self.completed_key:
            self.pending_key = key
            self.pending = self.session.request(self.reader, *key)
            self.fusion_status.setText(f"Loading {key[1]} native products…")

    def snapshot(self):
        return {
            "path": self.result_path,
            "index": self.timeline.value(),
            "sensor": self.sensor_choice.currentText(),
            "visible": {k: w.isChecked() for k, w in self.visibility.items()},
        }

    def closeEvent(self, event):
        self.closed = True
        if self.result_playback is not None:
            self.result_playback.close()
        self.timer.stop()
        self.session.close()
        self.visualizer._fusion_inspection_state = self.snapshot()
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is not None:
            self.overlay.clear(renderer)
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("sensing")
        if panel is not None:
            panel.detach_fusion(self)
        if not getattr(self.visualizer, "_shutdown_started", False):
            schedule = getattr(self.visualizer, "schedule_update", None)
            if callable(schedule):
                self.visualizer.force_update_next_frame = True
                schedule()
        super().closeEvent(event)


def open_fusion_window(visualizer, path, *, volume_context=None):
    """Open a result from CLI or the sensing panel with identical behavior."""
    from shared.sensing.fusion.result import FusionResult

    result = FusionResult(path)
    volume = getattr(visualizer, "sensing_volume_window", None)
    if volume is not None and not volume.closed:
        volume.close()
    old = getattr(visualizer, "fusion_window", None)
    if old is not None:
        from shiboken6 import isValid

        if isValid(old):
            old.close()
    names = tuple(result.get("sources", {}))
    if len(names) == 1:
        source = result.source_path(names[0])
        if source is not None and (source / "scenario.yaml").is_file():
            outcome = visualizer.open_scenario(str(source), autorun_initial_frame=True)
            if outcome.succeeded:
                visualizer.ui_manager.ensure_panel("sensing")
                panel = visualizer.ui_manager.panels["sensing"]
                panel._retire_dashboard_session()
                panel.saved_result_path = str(result.path)
                panel._on_open_dashboard()
                return panel._sensing_window
    window = FusionWindow(visualizer, path)
    window.volume_context = volume_context
    visualizer.fusion_window = window
    visualizer.ui_manager.ensure_panel("sensing")
    visualizer.ui_manager.panels["sensing"].attach_fusion(window)
    from .volume_workspace import reveal_sensing_panel

    reveal_sensing_panel(visualizer)
    window.show()
    return window
