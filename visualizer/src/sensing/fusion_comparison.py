"""Compare the exact reports accepted into a saved estimate with its final position."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from shared.sensing.fusion.core import gated_assignment

from ..widgets.inline_plot import InlinePlotGrid


def comparison_data(reader, selection):
    """Use saved source IDs for attribution; truth is only an evaluation reference.

    A single-target trial has an unambiguous reference even for an extra estimate.
    With multiple targets, use one-to-one gated matching rather than silently
    attributing every estimate to its nearest truth.
    """
    if selection is None:
        return None
    index = selection.epoch_index
    tracks = reader.epoch(index, selection.mode)["tracks"]
    track = next((t for t in tracks if t["track_id"] == selection.track_id), None)
    if track is None:
        return None
    truth = reader.get("truth_for_evaluation", [])
    targets = []
    if index < len(truth):
        epoch = truth[index]
        if epoch.get("available", True) and epoch.get("time_s") == reader.times[index]:
            targets = epoch.get("targets", [])
    target = targets[0] if len(targets) == 1 else None
    if len(targets) > 1:
        confirmed = [t for t in tracks if t["confirmed"]]
        costs = np.array(
            [
                [np.linalg.norm(np.subtract(t["position_m"], g["position_m"])) for t in confirmed]
                for g in targets
            ]
        ).reshape(len(targets), len(confirmed))
        gate = reader.get("evaluation_config", {}).get("position_gate_m", 0.75)
        target = next(
            (
                targets[i]
                for i, j in gated_assignment(costs, gate)
                if confirmed[j]["track_id"] == selection.track_id
            ),
            None,
        )
    colors = ("#5797cb", "#e49e56", "#a68bcc", "#d57c9d", "#71a7a5")
    rows = []
    for i, name in enumerate(reader.get("sources", {})):
        accepted = [s for s in track["sources"] if s["sensor_id"] == name]
        epochs = reader.get("observations", {}).get(name, [])
        observations = epochs[index].get("detections", []) if index < len(epochs) else []
        report = next(
            (
                d
                for d in observations
                if any(d["detection_id"] == s["detection_id"] for s in accepted)
            ),
            None,
        )
        position = report.get("world_position_m") if report is not None else None
        rows.append(
            dict(
                label=f"{name.title()} report",
                color=colors[i % len(colors)],
                position=position,
                reason="Position unavailable" if accepted else "No contribution",
            )
        )
    rows.append(
        dict(
            label=(
                "Fused estimate"
                if selection.mode == "fused"
                else f"{selection.mode.title()} estimate"
            ),
            color="#47a883",
            position=track["position_m"],
            reason="",
        )
    )
    for row in rows:
        row["error_m"] = (
            float(np.linalg.norm(np.subtract(row["position"], target["position_m"])))
            if row["position"] is not None and target is not None
            else None
        )
    return dict(rows=rows, target=target, coasting=track["predicted_only"])


class FusionComparison(QWidget):
    """A spatial comparison and errors, with no invented sensor influence weights."""

    def __init__(self):
        super().__init__()
        self.data = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sensor reports and fused estimate"), 1)
        self.plane = QComboBox()
        self.plane.addItems(["XY", "XZ", "YZ"])
        self.plane.setToolTip(
            "Projection of saved world positions. Errors always use all three axes."
        )
        self.plane.currentIndexChanged.connect(self._draw)
        row.addWidget(self.plane)
        layout.addLayout(row)
        self.charts = InlinePlotGrid(
            [
                ("Position comparison", "X (m)", "Y (m)"),
                ("3D position error", "Error (m)", ""),
            ]
        )
        self.charts.plots[0].setMinimumHeight(260)
        self.charts.plots[1].setMinimumHeight(190)
        for plot in self.charts.plots:
            for axis in ("left", "bottom"):
                plot.getAxis(axis).enableAutoSIPrefix(False)
        layout.addWidget(self.charts)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

    def update_result(self, reader, selection):
        self.data = comparison_data(reader, selection)
        self._draw()

    def _draw(self):
        self.charts.clear()
        spatial, errors = self.charts.plots
        if self.data is None:
            for plot in self.charts.plots:
                plot.set_empty_message("No estimate selected")
            self.note.clear()
            return
        data = self.data
        plane = self.plane.currentText()
        a, b = ["XYZ".index(axis) for axis in plane]
        spatial.configure(xlabel=f"{plane[0]} (m)", ylabel=f"{plane[1]} (m)")
        points = []
        for i, row in enumerate(data["rows"]):
            if row["position"] is None:
                continue
            point = np.asarray(row["position"])[[a, b]]
            points.append(point)
            curve = spatial.curve(
                [point[0]],
                [point[1]],
                name=row["label"],
                color=row["color"],
                symbol="d" if i == len(data["rows"]) - 1 else "o",
            )
            curve.setSymbolSize(17 if i == len(data["rows"]) - 1 else 10)
            if i == len(data["rows"]) - 1:
                estimate_marker = curve
        target = data["target"]
        if target is not None:
            point = np.asarray(target["position_m"])[[a, b]]
            points.append(point)
            marker = spatial.curve(
                [point[0]], [point[1]], name="Ground truth", color=None, symbol="+"
            )
            marker.setSymbolSize(18)
        spatial.apply_theme()
        estimate_marker.setSymbolBrush(None)
        spatial.setAspectLocked(True)
        if points:
            points = np.asarray(points)
            centre = (points.min(axis=0) + points.max(axis=0)) / 2
            span = max(float(np.ptp(points, axis=0).max()) * 1.5, 0.5)
            spatial.setRange(
                xRange=(centre[0] - span / 2, centre[0] + span / 2),
                yRange=(centre[1] - span / 2, centre[1] + span / 2),
                padding=0,
            )
        rows = data["rows"]
        positions = np.arange(len(rows))[::-1]
        errors.getAxis("left").setTicks(
            [[(y, r["label"]) for y, r in zip(positions, rows, strict=True)]]
        )
        errors.getAxis("left").setWidth(
            max(self.fontMetrics().horizontalAdvance(r["label"]) for r in rows) + 16
        )
        errors.setYRange(-0.6, len(rows) - 0.4, padding=0)
        maximum = max((r["error_m"] or 0 for r in rows), default=0)
        errors.setXRange(0, max(maximum * 1.4, 0.1), padding=0)
        for y, row in zip(positions, rows, strict=True):
            error = row["error_m"]
            if error is not None:
                errors.addItem(
                    pg.BarGraphItem(
                        x0=0, y=[y], width=[error], height=0.55, brush=row["color"], pen=None
                    )
                )
                errors.text(f"{error:.3f}", error, y)
            else:
                errors.text(
                    row["reason"] if row["position"] is None else "No truth reference", 0, y
                )
        self.note.setText(
            "Coasting estimate; no current sensor reports."
            if data["coasting"]
            else (
                "Accepted reports for this estimate; ground truth is used only to measure error."
                if target is not None
                else "Accepted reports for this estimate; no matched ground truth is available."
            )
        )
