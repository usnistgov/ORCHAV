"""Optional coverage generation dialog, separate from saved-map display controls."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..authoring.generation import GenerationJob
from ..widgets.form_layout import AlignedFormLayout
from .generation import coverage_mapping, estimate_work, parse_heights, prepare_run


class CoverageGenerationWindow(QDialog):
    """Prepare in a worker and poll the existing cancellable generator job."""

    def __init__(self, visualizer):
        super().__init__(visualizer)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("ORCHAV — Generate coverage")
        self.resize(560, 570)
        self.visualizer = visualizer
        source = Path(visualizer.current_scenario_path)
        self.source = source / "scenario.yaml" if source.is_dir() else source
        self.mapping = yaml.safe_load(self.source.read_text())
        self.job = None
        self.pending = None
        self.result_path = None
        self.cancelled = False
        self.log_offsets = [0, 0]
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coverage-prepare")
        self.poll = QTimer(self)
        self.poll.setInterval(100)
        self.poll.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Compute radio coverage at the saved scenario's initial poses (t = 0). "
            "Uses its scene, transmitters, antennas and materials. This runs Sionna RT. "
            "A new result directory preserves the source recording."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.inputs = QWidget()
        form = AlignedFormLayout(self.inputs)
        cov = self.mapping.get("coverage") or {}
        grid = cov.get("grid") or {}
        self.dx = self._real(0.01, 10000, (grid.get("resolution_m") or [5, 5])[0])
        self.dy = self._real(0.01, 10000, (grid.get("resolution_m") or [5, 5])[1])
        form.addRow("Cell width X (m)", self.dx)
        form.addRow("Cell width Y (m)", self.dy)
        height_values = grid.get("heights_m") or [1.5]
        text = ", ".join(str(v) for v in height_values)
        if len(height_values) > 2:
            step = height_values[1] - height_values[0]
            if step > 0 and all(
                abs((b - a) - step) < 1e-6 for a, b in zip(height_values, height_values[1:])
            ):
                text = f"{height_values[0]:g}:{height_values[-1]:g}:{step:g}"
        self.heights = QLineEdit(text)
        self.heights.setToolTip("Increasing metres: 1.5, 10, 30 or regular sweep 10:70:5")
        form.addRow("Heights (m; list or start:stop:step)", self.heights)
        self.destination = QLineEdit(
            str(
                self.source.parent.parent
                / (
                    self.source.parent.name
                    + "_coverage_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            )
        )
        row = QHBoxLayout()
        row.addWidget(self.destination)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        form.addRow("New result directory", row)
        advanced = QCheckBox("Region and solver settings")
        form.addRow(advanced)
        self.details = QWidget()
        detail = AlignedFormLayout(self.details)
        bbox = grid.get("bbox_xy", "auto")
        self.auto = QCheckBox("Use scene bounds")
        self.auto.setChecked(bbox == "auto")
        detail.addRow(self.auto)
        if bbox == "auto":
            bbox = [[-100, 100], [-100, 100]]
        self.bounds = [self._real(-1e7, 1e7, v) for pair in bbox for v in pair]
        for label, spin in zip(
            ("X minimum (m)", "X maximum (m)", "Y minimum (m)", "Y maximum (m)"), self.bounds
        ):
            detail.addRow(label, spin)
            spin.setEnabled(not self.auto.isChecked())
        self.auto.toggled.connect(lambda active: [v.setEnabled(not active) for v in self.bounds])
        solver = cov.get("solver") or {}
        self.samples = QSpinBox()
        self.samples.setRange(1, 100_000_000)
        self.samples.setValue(solver.get("samples_per_tx", 1_000_000))
        detail.addRow("Samples per TX per height", self.samples)
        self.depth = QSpinBox()
        self.depth.setRange(0, 20)
        self.depth.setValue(solver.get("max_depth", 3))
        detail.addRow("Maximum interaction depth", self.depth)
        self.mechanisms = {}
        for key, label, default in (
            ("los", "Line of sight", True),
            ("specular_reflection", "Specular reflection", True),
            ("diffuse_reflection", "Diffuse reflection", False),
            ("refraction", "Transmission / refraction", False),
            ("diffraction", "Diffraction (backend dependent)", False),
        ):
            box = QCheckBox(label)
            box.setChecked(solver.get(key, default))
            detail.addRow(box)
            self.mechanisms[key] = box
        self.details.hide()
        advanced.toggled.connect(self.details.setVisible)
        form.addRow(self.details)
        layout.addWidget(self.inputs)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.hide()
        log_toggle = QCheckBox("Job details")
        log_toggle.toggled.connect(self.log.setVisible)
        layout.addWidget(log_toggle)
        layout.addWidget(self.log)
        buttons = QHBoxLayout()
        self.generate = QPushButton("Generate new coverage")
        self.generate.clicked.connect(self._start)
        self.cancel = QPushButton("Cancel job")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel)
        self.open_result = QPushButton("Open result")
        self.open_result.setEnabled(False)
        self.open_result.clicked.connect(self._open)
        for button in (self.generate, self.cancel, self.open_result):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        for spin in (self.dx, self.dy, self.samples, self.depth, *self.bounds):
            spin.valueChanged.connect(self._preview)
        self.heights.textChanged.connect(self._preview)
        self.destination.textChanged.connect(self._preview)
        self.auto.toggled.connect(self._preview)
        for box in self.mechanisms.values():
            box.toggled.connect(self._preview)
        self._preview()

    @staticmethod
    def _real(low, high, value):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def _settings(self):
        bounds = [v.value() for v in self.bounds]
        return dict(
            heights=parse_heights(self.heights.text()),
            resolution=[self.dx.value(), self.dy.value()],
            bbox="auto" if self.auto.isChecked() else [bounds[:2], bounds[2:]],
            samples=self.samples.value(),
            depth=self.depth.value(),
            mechanisms={key: box.isChecked() for key, box in self.mechanisms.items()},
        )

    def _preview(self):
        if self.pending is not None or self.job is not None:
            return
        try:
            dest = Path(self.destination.text()).expanduser()
            if not self.destination.text().strip() or dest.exists() or dest.is_symlink():
                raise ValueError("Choose a new, non-existing result directory")
            self.status.setText(estimate_work(coverage_mapping(self.mapping, **self._settings())))
            self.generate.setEnabled(True)
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))
            self.generate.setEnabled(False)

    def _browse(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose parent for a new result directory")
        if parent:
            self.destination.setText(str(Path(parent) / Path(self.destination.text()).name))

    def _start(self):
        if self.pending is not None or self.job is not None:
            return
        try:
            settings = self._settings()
            estimate_work(coverage_mapping(self.mapping, **settings))
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))
            return
        self.cancelled = False
        self.result_path = None
        self.log_offsets = [0, 0]
        self.log.clear()
        self.inputs.setEnabled(False)
        self.generate.setEnabled(False)
        self.open_result.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Preparing an independent scenario and its assets…")
        self.pending = self.executor.submit(
            prepare_run, self.source, Path(self.destination.text()), settings
        )
        self.poll.start()

    def _poll(self):
        if self.pending is not None:
            if not self.pending.done():
                return
            pending, self.pending = self.pending, None
            try:
                path = pending.result()
                if self.cancelled:
                    self._finish("Cancelled during preparation; no solver launched.")
                    return
                self.job = GenerationJob(path, 0)
                self.job.start()
                self.status.setText("Starting Sionna RT; loading the scene…")
            except Exception as exc:  # display worker/launch failures; never publish success
                self._finish(f"Coverage generation failed: {exc}")
                return
        if self.job is None:
            return
        for i, lines in enumerate((self.job.stdout_log, self.job.stderr_log)):
            for line in lines[self.log_offsets[i] :]:
                self.log.appendPlainText(line)
                match = re.search(r"Coverage: solving height (\d+)/(\d+)", line)
                if match and not self.cancelled:
                    height, total = map(int, match.groups())
                    self.progress.setRange(0, total)
                    self.progress.setValue(height - 1)
                    self.status.setText(f"Solving height {height}/{total}…")
            self.log_offsets[i] = len(lines)
        result = self.job.result
        if result is not None:
            path = result.paths.scenario_yaml.parent
            success = (
                result.succeeded
                and not self.cancelled
                and (path / "coverage/coverage_maps.h5").is_file()
            )
            if success:
                self.result_path = path
            message = (
                f"Coverage saved to {path}. Open result to inspect it."
                if success
                else (
                    "Cancelled. Source recording preserved; partial output is not opened."
                    if self.cancelled
                    else f"Coverage generation failed: {result.error_message or 'no verified coverage output'}"
                )
            )
            self.job = None
            self._finish(message)

    def _finish(self, message):
        self.poll.stop()
        self.inputs.setEnabled(True)
        self.cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(int(self.result_path is not None))
        self.open_result.setEnabled(self.result_path is not None)
        self.generate.setEnabled(False)
        self.status.setText(message)

    def _cancel(self):
        self.cancelled = True
        self.cancel.setEnabled(False)
        self.status.setText("Cancelling…")
        if self.job is not None:
            self.job.cancel()

    def _open(self):
        if self.result_path is None:
            return
        viz = self.visualizer
        previous = getattr(viz, "coverage_volume_window", None)
        view = (
            {
                name: getattr(previous, name).currentData()
                for name in ("mode", "orientation", "cut_axis")
            }
            if previous is not None and not previous.closed
            else {}
        )
        outcome = viz.open_scenario(str(self.result_path))
        if not outcome.succeeded:
            self.status.setText(outcome.message or "Coverage result could not be opened.")
            return
        viz.ui_controller.handle_coverage_toggled(True)
        current = getattr(viz, "coverage_volume_window", None)
        if current is not None and not current.closed:
            # Keep the display choice, but populate coordinates from the new
            # grid and start with the full region rather than an old cutaway.
            for name, value in view.items():
                combo = getattr(current, name)
                index = combo.findData(value)
                if index >= 0 and combo.model().item(index).isEnabled():
                    combo.setCurrentIndex(index)
        viz.ui_manager.show_panel("coverage")
        self.close()

    def closeEvent(self, event):
        self.poll.stop()
        if self.job is not None:
            self.job.cancel()
        # No worker callback touches Qt; a closed preparation cannot launch a job.
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def open_coverage_generation(visualizer):
    """Keep one live generation dialog per application window."""
    from shiboken6 import isValid

    current = getattr(visualizer, "coverage_generation_window", None)
    if current is not None and isValid(current):
        current.show()
        current.raise_()
        return current
    window = CoverageGenerationWindow(visualizer)
    visualizer.coverage_generation_window = window
    window.show()
    return window
