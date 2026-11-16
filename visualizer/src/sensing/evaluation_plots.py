"""Live evaluation figures over immutable saved outputs, without detector calls."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shared.sensing.evaluation_truth import evaluation_velocity_mask

from ..app.plot_theme import apply_pyqtgraph_plot_theme
from ..widgets.dashboard_plot import DashboardPlot
from .dashboard_evaluation import match_event


class EvaluationPlots(QTabWidget):
    """Separate a selected-result assignment from interval/recording totals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(210)
        accuracy = QWidget()
        layout = QVBoxLayout(accuracy)
        self.target_scope = QLabel()
        self.target_scope.setWordWrap(True)
        layout.addWidget(self.target_scope)
        self.targets = QTableWidget(0, 7)
        self.targets.setHorizontalHeaderLabels(
            [
                "Target",
                "Matched / eligible",
                "Coverage",
                "Missed",
                "Range RMSE\n(m)",
                "Radial RMSE\n(m/s)",
                "Velocity reference",
            ]
        )
        self.targets.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.targets.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.targets.verticalHeader().hide()
        self.targets.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.targets.horizontalHeader().setStretchLastSection(True)
        self.targets.setAlternatingRowColors(True)
        layout.addWidget(self.targets, 1)
        self.reference = QLabel()
        self.reference.setTextFormat(Qt.TextFormat.PlainText)
        self.reference.setWordWrap(True)
        self.reference.setToolTip(
            "For a frozen-pose coherent solve driven by root translation, GT uses "
            "the saved root velocity projected onto the recorded radar geometry, assuming "
            "fixed sensors. Other motion models retain their recorded GT velocity. "
            "This is a bulk-motion reference, not limb micro-Doppler truth. "
            "The original recorded target-center diagnostic is preserved."
        )
        layout.addWidget(self.reference)
        self.addTab(accuracy, "Per-target accuracy")
        self.matches = DashboardPlot()
        self.counts = DashboardPlot()
        for plot in (self.matches, self.counts):
            apply_pyqtgraph_plot_theme(plot)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
        self.addTab(self.matches, "Selected-result matches")
        self.addTab(self.counts, "Recording / interval counts")
        self.matches.setLabel("bottom", "Range", units="m")
        self.matches.setLabel("left", "Radial velocity", units="m/s")
        self.matches.add_external_legend(labelTextSize="8pt")
        self.links = self.matches.plot(pen=pg.mkPen("#76838d", width=1))
        self.gates = self.matches.plot(
            pen=pg.mkPen("#76838d", style=Qt.PenStyle.DashLine), name="GT gate"
        )
        self.markers = {}
        for key, name, symbol, color in (
            ("gt", "Matched GT", "+", "#4388c9"),
            ("tp", "TP detection", "o", "#229979"),
            ("fp", "FP detection", "t", "#cf5267"),
            ("fn", "FN target", "x", "#c78924"),
        ):
            self.markers[key] = self.matches.plot(
                pen=None,
                symbol=symbol,
                symbolSize=11,
                symbolPen=pg.mkPen(color, width=2),
                symbolBrush=pg.mkBrush(color),
                name=name,
            )
        self.counts.setLabel("left", "Count")
        self.counts.getAxis("bottom").setTicks([[(0, "TP"), (1, "FP"), (2, "FN")]])
        self.bars = pg.BarGraphItem(
            x=[0, 1, 2],
            height=[0, 0, 0],
            width=0.6,
            brushes=["#229979", "#cf5267", "#c78924"],
            pen=None,
        )
        self.counts.addItem(self.bars)
        self.count_labels = []
        for color in ("#229979", "#cf5267", "#c78924"):
            label = pg.TextItem(color=color, anchor=(0.5, 1))
            self.counts.addItem(label)
            self.count_labels.append(label)
        self.counts.setXRange(-0.7, 2.7, padding=0)

    def update_references(self, event):
        """Show the chosen reference beside the untouched saved-center diagnostic."""
        truth = (event or {}).get("truth") or {}
        ids = truth.get("target_ids", [])
        speeds = truth.get("target_speeds", [])
        recorded = truth.get("recorded_target_speeds", speeds)
        references = truth.get(
            "target_velocity_references", ["Recorded target-center velocity"] * len(ids)
        )
        rows = []
        for tid, value, original, reference in zip(ids, speeds, recorded, references, strict=True):
            if reference == "Recorded target-center velocity":
                continue

            def number(value):
                return "unavailable" if value is None else f"{value:.3f} m/s"

            recorded_label = (
                "recorded observable"
                if evaluation_velocity_mask(truth) is not None
                else "recorded center"
            )
            rows.append(f"{tid}: {reference} {number(value)}; {recorded_label} {number(original)}")
        prefix = f"Selected result {event['time_s']:.6g} s · " if rows else ""
        self.reference.setText(prefix + "\n".join(rows))
        self.reference.setVisible(bool(rows))

    def update_targets(self, summary, scope):
        """Present matched-only accuracy with each target's missed opportunities."""
        rows = summary.get("per_target", {}) if not summary.get("reason") else {}
        self.targets.setRowCount(len(rows))
        for index, (tid, row) in enumerate(rows.items()):
            values = [
                tid,
                f"{row['matched']} / {row['eligible']}",
                f"{100 * row['coverage']:.1f}%",
                str(row["missed"]),
                "—" if row["range_rmse"] is None else f"{row['range_rmse']:.3f}",
                "—" if row["velocity_rmse"] is None else f"{row['velocity_rmse']:.3f}",
                row["velocity_reference"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(
                    f"{tid} · {row['matched']} range pairs and "
                    f"{row.get('velocity_matched', row['matched'])} velocity pairs contribute to RMSE. "
                    f"Velocity reference: {row['velocity_reference']}."
                )
                self.targets.setItem(index, column, item)
        self.target_scope.setText(
            f"{scope} · {summary['reason']}"
            if summary.get("reason")
            else f"{scope} · RMSE uses matched pairs only; coverage includes misses. "
            "False positives are counted in the totals."
        )

    def update_matches(self, event, summary, range_gate, velocity_gate, confirmed_only):
        """Redraw assignments at the selected epoch using the same scoring gates."""
        for item in (*self.markers.values(), self.links, self.gates):
            item.setData([], [])
        matched = match_event(event, range_gate, velocity_gate, confirmed_only=confirmed_only)
        if matched is None:
            self.matches.setTitle("Matching unavailable")
            self.matches.setToolTip(summary.get("reason") or "No saved result")
            return
        self.matches.setToolTip(
            "Dashed rectangles show ± each evaluation gate around GT. "
            "Lines show one-to-one assignments, not target motion. "
            "Saved detections are unchanged."
        )
        truth, detections, assignment = matched
        velocity_mask = evaluation_velocity_mask(truth)
        points = {key: [] for key in self.markers}
        links, gates = [], []
        for index, (r, v) in enumerate(
            zip(truth["target_ranges"], truth["target_speeds"], strict=True)
        ):
            det_index = assignment.gt_to_det[index]
            if velocity_mask is not None and not velocity_mask[index]:
                # A valid range-only match has no point or gate in this 2D
                # velocity chart; zero would invent an unrecorded reference.
                continue
            points["fn" if det_index is None else "gt"].append((r, v))
            gates.extend(
                [
                    (r - range_gate, v - velocity_gate),
                    (r + range_gate, v - velocity_gate),
                    (r + range_gate, v + velocity_gate),
                    (r - range_gate, v + velocity_gate),
                    (r - range_gate, v - velocity_gate),
                    (np.nan, np.nan),
                ]
            )
            if det_index is not None:
                det = detections[det_index]
                links.extend([(r, v), (det["range_m"], det["velocity_m_s"]), (np.nan, np.nan)])
        for index, det in enumerate(detections):
            key = "fp" if assignment.det_to_gt[index] is None else "tp"
            points[key].append((det["range_m"], det["velocity_m_s"]))
        for curve, values in [
            (self.links, links),
            (self.gates, gates),
            *((self.markers[key], values) for key, values in points.items()),
        ]:
            if values:
                xy = np.asarray(values)
                curve.setData(xy[:, 0], xy[:, 1], connect="finite")
        self.matches.setTitle(
            f"Selected result {event['time_s']:.6g} s · "
            f"TP {assignment.true_positives} / FP {assignment.false_positives} / FN {assignment.false_negatives}"
            + (
                f" · {assignment.velocity_excluded_matched_pair_count} range-only matches"
                if assignment.velocity_excluded_matched_pair_count
                else ""
            )
        )

    def update_counts(self, summary, scope):
        """Clear unavailable totals instead of representing them as zero errors."""
        self.update_targets(summary, scope)
        available = not summary.get("reason")
        values = [summary[f"total_{key}"] for key in ("tp", "fp", "fn")] if available else [0, 0, 0]
        self.bars.setOpts(height=values)
        for index, (label, value) in enumerate(zip(self.count_labels, values, strict=True)):
            label.setText(str(value) if available else "")
            label.setPos(index, value)
        self.counts.setYRange(0, max(1, *values) * 1.2, padding=0)
        self.counts.setTitle(
            f"{scope} · {summary['evaluated_event_count']} evaluated results"
            if available
            else f"{scope} · counts unavailable"
        )
        self.counts.setToolTip(
            summary.get("reason") or "Counts over unique saved results, independent of playback."
        )
