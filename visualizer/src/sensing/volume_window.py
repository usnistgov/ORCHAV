"""Optional conditional-outcome volume controls for the main physical scene."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
)

from shared.sensing.volume.evaluation import load_evaluation
from shared.sensing.volume.inspection import PairedVolume, sensor_forward, validate_context
from shared.sensing.volume.reader import SensingVolume

from ..renderers.protocol import volume_rendering_unavailable_reason
from ..services.sensing_overlay_sync import SensingOverlaySync
from ..types.camera_state import CameraState
from .volume_workflow import VolumeWorkflow


class SensingVolumeWindow(QWidget):
    """One embeddable workspace; the main scene owns its volume and camera."""

    def __init__(self, visualizer, path, *, view_state=None, context=None, scene_is_open=False):
        from .qt_lifetime import ensure_gui_garbage_collection

        ensure_gui_garbage_collection()
        super().__init__(visualizer)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("ORCHAV — Sensing volume")
        self.resize(530, 750)
        self.visualizer = visualizer
        self.view_state = view_state
        self.context = context
        self.scene_is_open = scene_is_open
        self.selected_key = None
        self.pair_path = None
        self.pair_pending = None
        self.pair_failed = False
        self.reader = None
        self.closed = False
        self.opening_scene = False
        self.overlay = SensingOverlaySync()
        self.templates = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-volume")
        self.opening = (
            self.executor.submit(self._load, path, context)
            if context is not None
            else self.executor.submit(self._load, path)
        )
        self.native_pending = None
        self.native_pending_key = None
        self.selection_generation = 0
        self.last_selection = None
        self.evaluation = None
        self.evaluation_pending = None
        self.evaluation_cancel = Event()
        self.evaluation_progress = (0, 0)
        self.evaluation_snapshot = None
        self.evaluation_signature = None
        from .volume_controls import build_controls

        build_controls(self)
        self.workflow = VolumeWorkflow(self, self.inspection_controls)
        select_volume = getattr(visualizer.renderer, "set_volume_cell_selection_callback", None)
        if callable(select_volume):
            select_volume(
                self.workflow.select_volume_cell,
                object_ids=("sensing:outcome_volume", "sensing:outcome_volume:misses"),
            )
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self.destroyed.connect(self.evaluation_cancel.set)
        self.destroyed.connect(partial(self.executor.shutdown, wait=False, cancel_futures=True))

    def _reset_evaluation(self):
        if self.reader is not None:
            self.evaluation_controls.configure(self.reader.metadata)

    def _show_label_settings(self):
        manager = self.visualizer.ui_manager
        controls = manager.panels["nodes"].widgets["label_style_controls"]
        manager.set_panel_visible("nodes", True)
        manager.reveal_control("nodes", controls.mode)

    def _prepare_camera(self):
        """Use the application's camera-motion preferences for volume navigation."""
        camera = getattr(self.visualizer, "camera_controller", None)
        tools = camera.presentation() if camera is not None else None
        if tools is not None:
            tools.prepare()
            tools.free_camera()

    def _slice_coordinate_changed(self, index):
        if self.reader is None or index < 0:
            return
        normal = {"xy": "z", "xz": "y", "yz": "x"}[
            self.inspection_controls.orientation.currentData()
        ]
        {"x": self.x, "y": self.y, "z": self.altitude}[normal].setCurrentIndex(index)

    def _sync_slice_coordinate(self, normal):
        source = {"x": self.x, "y": self.y, "z": self.altitude}[normal]
        with QSignalBlocker(self.slice_coordinate):
            if self.slice_coordinate.property("axis") != normal:
                self.slice_coordinate.clear()
                for i in range(source.count()):
                    self.slice_coordinate.addItem(source.itemText(i), source.itemData(i))
                self.slice_coordinate.setProperty("axis", normal)
            self.slice_coordinate.setCurrentIndex(source.currentIndex())

    def _condition_changed(self, index):
        path = self.conditions.itemData(index)
        if self.reader is not None and path and Path(path) != self.reader.path:
            open_sensing_volume(self.visualizer, path, view_state=self._inspection_view_state())

    def _pair_source_changed(self, index):
        self.pair_path = self.pair_source.itemData(index)
        if not hasattr(self, "workflow"):
            return
        self.pair_generation = getattr(self, "pair_generation", 0) + 1
        self.workflow.pair = self.workflow.pair_snapshot = self.workflow.pair_signature = None
        self.pair_failed = False
        self.templates.clear()
        self._draw()

    @staticmethod
    def _load(path, context=None):
        reader = SensingVolume(path)
        reader.verify_scene()
        if context is not None:
            validate_context(reader, context)
        reader.inspection_forward = sensor_forward(reader)
        return reader

    def _finish_open(self, reader):
        self.opening_scene = True
        if self.context is not None:
            key = validate_context(reader, self.context)
            self.view_state = dict(
                self.context["view"],
                heading=reader.headings[key[0]]["name"],
                camera=(
                    CameraState.from_dict(self.context["camera"])
                    if self.context.get("camera")
                    else None
                ),
            )
            self.workflow.selected_report = self.context.get("selected_report")
        try:
            if not self.scene_is_open:
                self.visualizer._opening_sensing_volume_scene = True
                try:
                    outcome = self.visualizer.open_scenario(
                        str(reader.scene_path),
                        autorun_initial_frame=True,
                        data_mode_override="files",
                    )
                finally:
                    self.visualizer._opening_sensing_volume_scene = False
                if not outcome.succeeded:
                    raise ValueError("Cannot open the volume's bound physical scene")
            from .volume_scene import initialize_scene_radios

            initialize_scene_radios(self.visualizer)
            # Populate before assigning the reader so Qt signals cannot draw partial choices.
            self.heading.addItems([h["name"] for h in reader.headings])
            self.heading.setEnabled(len(reader.headings) > 1)
            self.heading.setVisible(len(reader.headings) > 1)
            self.single_heading.setVisible(len(reader.headings) == 1)
            self.single_heading.setText(reader.headings[0]["name"])
            self.motion_label.setText(reader.metadata.get("motion_label", "Target motion"))
            from .volume_sources import condition_sources

            sources = condition_sources(reader)
            with QSignalBlocker(self.conditions), QSignalBlocker(self.pair_source):
                for source in sources:
                    self.conditions.addItem(source.parent.name, str(source))
                    if source != reader.path:
                        self.pair_source.addItem(source.parent.name, str(source))
                self.conditions.setCurrentIndex(
                    self.conditions.findData(str(getattr(reader, "path", "")))
                )
            self.conditions.setVisible(len(sources) > 1)
            self.pair_path = self.pair_source.currentData()
            saved_pair = (self.context or {}).get("pair") or {}
            if saved_pair.get("volume"):
                self.pair_path = saved_pair["volume"]
                with QSignalBlocker(self.pair_source):
                    index = self.pair_source.findData(self.pair_path)
                    if index < 0:
                        self.pair_source.addItem(Path(self.pair_path).parent.name, self.pair_path)
                        index = self.pair_source.count() - 1
                    self.pair_source.setCurrentIndex(index)
            if self.pair_path:
                self.quantity.addItem("Paired change", "pair")
            if reader.metadata.get("geometry_description"):
                self.quantity.addItem("Direct path blockage (geometry)", "geometry")
            if reader.is_fusion:
                self.quantity.addItem("Sensor contributions", "contributions")
                self.selection_hint.setText(
                    "Click a cell to inspect its fused estimates and sensor contributions."
                )
            condition = reader.metadata.get("condition")
            if condition:
                self.setWindowTitle(f"ORCHAV — Detection volume · {condition}")
            for name, combo in (("x", self.x), ("y", self.y), ("z", self.altitude)):
                for value in reader.axes[name]:
                    combo.addItem(f"{value:g} m", float(value))
                combo.setCurrentIndex(len(reader.axes[name]) // 2)
            if self.view_state:
                selected = self.heading.findText(self.view_state.get("heading", ""))
                if self.context is not None:
                    selected = validate_context(reader, self.context)[0]
                elif self.view_state.get("heading_identity"):
                    wanted = self.view_state["heading_identity"]
                    selected = next(
                        (
                            i
                            for i, heading in enumerate(reader.headings)
                            if heading["name"] == wanted["name"]
                            and heading["velocity_m_s"] == wanted["velocity_m_s"]
                        ),
                        -1,
                    )
                if selected >= 0:
                    self.heading.setCurrentIndex(selected)
                # A new condition may use a different grid or target motion.
                # Preserve inspection only when the original physical cell exists.
                selection_available = selected >= 0
                for name in ("quantity", "mode", "altitude", "x", "y"):
                    combo = getattr(self, name)
                    index = combo.findData(self.view_state.get(name))
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    elif name in ("altitude", "x", "y"):
                        selection_available = False
                self.opacity.setValue(self.view_state["opacity"])
                self.inspect.setChecked(self.view_state["inspect"] and selection_available)
            self.reader = reader
            self.geometry_section.setVisible(not reader.is_fusion)
            self.contribution_plot.setVisible(reader.is_fusion)
            if reader.is_fusion:
                for quantity, title in (
                    ("extras", "Unmatched estimates per trial"),
                    ("reports", "Nearest estimate position error"),
                ):
                    self.quantity.setItemText(self.quantity.findData(quantity), title)
                for predicate in ("blocked", "all_blocked", "clear"):
                    combo = self.inspection_controls.focus
                    index = combo.findData(predicate)
                    if index >= 0:
                        combo.removeItem(index)
                self.inspection_controls.table.horizontalHeaderItem(0).setText("Estimate")
                self.inspection_controls.table.setColumnHidden(7, False)
                for column in (4, 5):
                    self.inspection_controls.table.setColumnHidden(column, True)
                self.inspection_controls.count_max.setSuffix(" estimates/trial")
            self.visible.setChecked((self.view_state or {}).get("visible", True))
            self.cell_edges.setChecked((self.view_state or {}).get("cell_edges", True))
            self.uniform_colour.setChecked(
                self.view_state.get("uniform_colour", False)
                if self.view_state is not None
                else True
            )
            self.readable_sensor.setChecked((self.view_state or {}).get("readable_sensor", True))
            self.dim_context.setChecked((self.view_state or {}).get("dim_context", False))
            self.hide_volume.setChecked((self.view_state or {}).get("hide_volume", False))
            self.highlight_cell.setChecked((self.view_state or {}).get("highlight_cell", True))
            if self.inspect.isChecked():
                self.selected_key = self._cell_key()
            self.unavailable_edges.setChecked(
                (self.view_state or {}).get("unavailable_edges", False)
            )
            self.inspection_controls.restore((self.view_state or {}).get("inspection", {}))
            self.antenna.restore((self.view_state or {}).get("antenna", {}))
            if not self.view_state or not self.view_state.get("antenna"):
                span = max(float(np.ptp(reader.axes[n])) for n in "xyz")
                self.antenna.radius.setValue(max(1, span * 0.6))
            self.source_note.setText(
                f"Saved sensing volume · {reader.metadata.get('condition', 'experiment')}"
            )
            self.source_note.setToolTip(str(getattr(reader, "path", reader.scene_path)))
            with QSignalBlocker(self.cutaway):
                self.cutaway.configure(reader.axes, (self.view_state or {}).get("cutaway"))
            self.evaluation_controls.configure(
                reader.metadata, (self.view_state or {}).get("evaluation")
            )
            self.evaluation_controls.setEnabled(True)
            self._load_evaluation()
            if self.context and self.context.get("pair"):
                self._load_pair()
            self.controls.setEnabled(True)
            self.compare_saved.setEnabled(True)
            self.tabs.setCurrentIndex(
                max(0, min(self.tabs.count() - 1, (self.view_state or {}).get("active_tab", 0)))
            )
            self._draw()
            if hasattr(self.visualizer, "ui_manager"):
                from .volume_workspace import reveal_sensing_panel

                reveal_sensing_panel(self.visualizer)
            if self.view_state and self.view_state.get("camera") is not None:
                self._prepare_camera()
                self.visualizer.renderer.set_camera_state(self.view_state["camera"])
            else:
                self._frame_grid()
        finally:
            self.opening_scene = False

    def _cell_key(self):
        return (
            self.heading.currentIndex(),
            self.altitude.currentIndex(),
            self.y.currentIndex(),
            self.x.currentIndex(),
        )

    def _draw(self, *_args):
        if self.reader is None:
            return
        if hasattr(self, "workflow"):
            key = self._cell_key()
            self.selected_key = key if self.inspect.isChecked() else None
            if self.selected_key is None:
                self.workflow.selected_report = None
            selection = (key, self.inspect.isChecked())
            if selection != self.last_selection:
                if not self.workflow.stepping:
                    self.workflow.sweep.stop()
                self.last_selection = selection
                self.selection_generation += 1
            self.workflow.draw()
            if self.antenna.pending is not None:
                self.timer.start()

    def _load_pair(self):
        if (
            self.reader is None
            or self.pair_pending is not None
            or self.workflow.pair is not None
            or self.pair_failed
            or not self.pair_path
        ):
            return
        reader = self.reader
        cancel = self.evaluation_cancel
        path = self.pair_path
        self.pending_pair_generation = getattr(self, "pair_generation", 0)

        def load_pair():
            other = self._load(path)
            pair = PairedVolume(reader, other, strict_experiment=False)
            return pair, load_evaluation(other, cancel=cancel)

        self.status.setText("Validating paired experiment and compact reports…")
        self.pair_pending = self.executor.submit(load_pair)
        self.timer.start()

    def _frame_grid(self):
        if self.reader is None:
            return
        lo = np.array([self.reader.axes[n][0] for n in ("x", "y", "z")])
        hi = np.array([self.reader.axes[n][-1] for n in ("x", "y", "z")])
        lo = np.minimum(lo, self.reader.metadata["sensor_position_m"])
        hi = np.maximum(hi, self.reader.metadata["sensor_position_m"])
        for position in self.reader.metadata.get("sensor_positions_m", {}).values():
            lo, hi = np.minimum(lo, position), np.maximum(hi, position)
        self._prepare_camera()
        self.visualizer.renderer.set_overview_camera("isometric", ((lo + hi) / 2, (hi - lo) * 1.4))

    def _open_native(self, arm="a"):
        if (
            self.reader is not None
            and self.selected_key is not None
            and self.native_pending is None
        ):
            self.native_pending_key = self._cell_key()
            self.native_context = self.workflow.context()
            self.native_pending_generation = self.selection_generation
            self.workflow.sweep.stop()
            self.native_arm = "B" if arm == "b" else "A"
            reader, key = self.reader, self.native_pending_key
            if arm == "b":
                if self.workflow.pair is None:
                    return
                pair = self.workflow.pair[0]
                reader, key = pair.b, (pair.headings[key[0]], *key[1:])
            self.native_is_fusion = reader.is_fusion
            self.native_pending = self.executor.submit(reader.verify_native, key)
            self.timer.start()
            self.inspection_controls.native_b.setEnabled(False)
            self.open_native.setEnabled(False)
            self.native_note.setText("Opening the saved trial…")
            self.native_note.show()

    def _compare_saved_volumes(self):
        from .volume_comparison import VolumeComparisonDialog

        visualizer = self.visualizer
        dialog = VolumeComparisonDialog(
            visualizer,
            initial_volume=self.reader.path,
            gate=self.evaluation_controls.position_gate.value(),
        )

        def inspect_source(path):
            # The original control window may have closed on a previous source
            # switch. Read the currently owned volume, never a stale Qt object.
            current = getattr(visualizer, "sensing_volume_window", None)
            state = current._inspection_view_state() if current and not current.closed else {}
            if state:
                state["evaluation"] = dict(
                    state["evaluation"], basis="position", position_gate=dialog.gate.value()
                )
                source = next(s for s in dialog.definition()["sources"] if s["path"] == path)
                state["heading"] = source["heading"]
                state.pop("heading_identity", None)
            open_sensing_volume(visualizer, path, view_state=state or None)

        dialog.inspect_requested.connect(inspect_source)
        dialog.show()
        return dialog

    def _inspection_view_state(self):
        """Keep the same physical view when inspecting another saved condition."""
        state = {
            name: getattr(self, name).currentData()
            for name in ("quantity", "mode", "altitude", "x", "y")
        }
        state.update(
            opacity=self.opacity.value(),
            uniform_colour=self.uniform_colour.isChecked(),
            readable_sensor=self.readable_sensor.isChecked(),
            inspect=self.inspect.isChecked(),
            dim_context=self.dim_context.isChecked(),
            hide_volume=self.hide_volume.isChecked(),
            highlight_cell=self.highlight_cell.isChecked(),
            camera=self.visualizer.renderer.get_camera_state(),
            evaluation=self.evaluation_controls.state(),
            cutaway=self.cutaway.state(),
            heading=self.heading.currentText(),
            heading_identity=self.reader.headings[self.heading.currentIndex()],
            visible=self.visible.isChecked(),
            cell_edges=self.cell_edges.isChecked(),
            unavailable_edges=self.unavailable_edges.isChecked(),
            inspection=self.inspection_controls.state(),
            antenna=self.antenna.state(),
            active_tab=self.tabs.currentIndex(),
        )
        return state

    def _load_evaluation(self):
        if (
            self.reader is None
            or self.evaluation_pending is not None
            or self.evaluation is not None
        ):
            return
        self.evaluation_controls.progress.setText("Reading saved detections in the background…")
        self.evaluation_controls.retry.hide()

        def progress(done, total):
            self.evaluation_progress = (done, total)

        self.evaluation_pending = self.executor.submit(
            load_evaluation, self.reader, cancel=self.evaluation_cancel, progress=progress
        )
        self.timer.start()

    def _poll(self):
        if self.closed or self.opening_scene:
            return
        try:
            if self.opening is not None and self.opening.done():
                future, self.opening = self.opening, None
                self._finish_open(future.result())
            if self.reader is not None:
                current = getattr(self.visualizer, "current_scenario_path", None)
                current = Path(current).resolve() if current else None
                if current is not None and current.is_dir():
                    current /= "scenario.yaml"
                if current != self.reader.scene_path:
                    self.close()
                    return
            self.antenna.poll()
            if self.evaluation_pending is not None:
                if self.evaluation_pending.done():
                    future, self.evaluation_pending = self.evaluation_pending, None
                    try:
                        self.evaluation = future.result()
                        self.evaluation_controls.set_ready()
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        self.evaluation_controls.progress.setText(f"Evaluation unavailable: {exc}.")
                        self.evaluation_controls.retry.show()
                elif self.evaluation_progress[1]:
                    done, total = self.evaluation_progress
                    self.evaluation_controls.progress.setText(
                        f"Reading saved reports: {done}/{total}. The detector is not running."
                    )
            if self.pair_pending is not None and self.pair_pending.done():
                future, self.pair_pending = self.pair_pending, None
                if self.pending_pair_generation != getattr(self, "pair_generation", 0):
                    self._draw()
                    return
                try:
                    self.workflow.pair = future.result()
                    expected = (self.context or {}).get("pair")
                    if expected and expected["volume_sha256"] != self.workflow.pair[0].b.identity:
                        self.workflow.pair = None
                        raise ValueError("Bookmark paired source identity changed")
                    self._draw()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.pair_failed = True
                    self.status.setText(f"Paired comparison unavailable: {exc}")
            if self.native_pending is not None and self.native_pending.done():
                future, self.native_pending = self.native_pending, None
                if (
                    self.native_pending_key == self._cell_key()
                    and getattr(self, "native_pending_generation", self.selection_generation)
                    == self.selection_generation
                ):
                    source = future.result()
                    self.native_context = self.workflow.context()
                    if getattr(self, "native_is_fusion", False):
                        from .fusion_window import open_fusion_window

                        open_fusion_window(
                            self.visualizer, source, volume_context=self.native_context
                        )
                        return
                    self.opening_scene = True
                    try:
                        result = self.visualizer.open_scenario(
                            str(source),
                            autorun_initial_frame=True,
                            data_mode_override="files",
                        )
                    finally:
                        self.opening_scene = False
                    if result.succeeded:
                        from .volume_return import install_return

                        install_return(
                            self.visualizer, self.native_context, source, arm=self.native_arm
                        )
                        self.close()
                        # The dashboard follows the producer's scene binding;
                        # a CPI's storage frame is not its physical scene epoch.
                        self.visualizer.ui_manager.ensure_panel("sensing")
                        self.visualizer.ui_manager.panels["sensing"]._on_open_dashboard()
                    else:
                        from .volume_return import install_return

                        current = Path(self.visualizer.current_scenario_path)
                        install_return(
                            self.visualizer,
                            self.native_context,
                            current.parent if current.is_file() else current,
                        )
                        self.status.setText(
                            "Native opening failed; Return to volume can restore the saved context."
                        )
                else:
                    self._draw()
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            self.status.setText(f"Volume inspection unavailable: {exc}")
            self.native_note.setText(str(exc))
            self.native_note.show()
            if self.reader is not None and self.native_pending is None:
                self.open_native.setEnabled(
                    self.reader.native_availability(self._cell_key()) == "available"
                )
        if all(
            f is None
            for f in (
                self.opening,
                self.evaluation_pending,
                self.pair_pending,
                self.native_pending,
                self.antenna.pending,
            )
        ):
            self.timer.stop()

    def closeEvent(self, event):
        self.closed = True
        select_volume = getattr(
            self.visualizer.renderer, "set_volume_cell_selection_callback", None
        )
        if callable(select_volume):
            select_volume(
                None, object_ids=("sensing:outcome_volume", "sensing:outcome_volume:misses")
            )
        self.evaluation_cancel.set()
        self.timer.stop()
        self.workflow.sweep.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.overlay.clear(self.visualizer.renderer)
        self.visualizer.renderer.set_analysis_legend(None)
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("sensing")
        if panel is not None and getattr(self.visualizer, "sensing_volume_window", None) is self:
            panel._volume_closed()
        super().closeEvent(event)


def open_sensing_volume(visualizer, path, *, view_state=None, context=None, scene_is_open=False):
    """Open an optional volume; unsupported backends keep ordinary sensing intact."""
    reason = volume_rendering_unavailable_reason(visualizer.renderer)
    if reason:
        raise ValueError(reason)
    fusion = getattr(visualizer, "fusion_window", None)
    if fusion is not None and not fusion.closed:
        fusion.close()
    coverage = getattr(visualizer, "coverage_volume_window", None)
    if coverage is not None and not coverage.closed:
        embedded = getattr(coverage, "embedded", False)
        coverage.close()
        if embedded:
            visualizer.ui_controller.handle_coverage_toggled(False)
    old = getattr(visualizer, "sensing_volume_window", None)
    if old is not None:
        from shiboken6 import isValid

        if isValid(old):
            old.close()
    visualizer.ui_manager.ensure_panel("sensing")
    window = SensingVolumeWindow(
        visualizer, path, view_state=view_state, context=context, scene_is_open=scene_is_open
    )
    visualizer.sensing_volume_window = window
    panel = visualizer.ui_manager.panels["sensing"]
    panel.attach_volume(window)
    from .volume_workspace import reveal_sensing_panel

    reveal_sensing_panel(visualizer)
    window.show()
    return window
