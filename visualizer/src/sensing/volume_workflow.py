"""Coordinate compact evidence, view state and display-only volume navigation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from weakref import proxy

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QTimer
from PySide6.QtWidgets import QFileDialog

from shared.sensing.volume.fusion import matched_contributions
from shared.sensing.volume.inspection import (
    DETAIL_COLORS,
    DETAIL_LABELS,
    PAIR_COLORS,
    PAIR_LABELS,
    cell_support,
    context_snapshot,
    detail_grid,
    extras_grid,
    focus_mask,
    load_bookmark,
    report_details,
    save_bookmark,
)

from ..model import RenderObject
from ..utils.cell_edges import selected_cell_objects
from .volume_cell_edges import cell_edge_objects
from .volume_cutaway import cut_volume
from .volume_evidence import geometry_objects, trial_objects
from .volume_presentation import (
    error_volume_template,
    inspection_template,
    volume_legend,
    volume_template,
)


class VolumeWorkflow(QObject):
    """Own only bounded presentation caches and small navigation state."""

    def __init__(self, window, controls):
        super().__init__(window)
        self.window, self.controls = proxy(window), controls
        self.selected_report = None
        self.selection_origin = "Source grid selection"
        self.selection_origin_key = None
        self.row_signature = None
        self.rows = []
        self.mask = None
        self.pair = None
        self.pair_snapshot = None
        self.pair_signature = None
        self.contributions = None
        self.stepping = False
        self.sweep = QTimer(self)
        self.sweep.timeout.connect(self._advance)
        controls.changed.connect(window._draw)
        controls.report_selected.connect(self.select_report)
        controls.action.connect(self.action)
        controls.slice_view.selected.connect(self.select_slice)

    def criteria(self):
        c = self.window.evaluation_controls
        return dict(
            mode=c.basis.currentData(),
            position_gate_m=c.position_gate.value(),
            range_gate_m=c.range_gate.value(),
            velocity_gate_m_s=c.velocity_gate.value(),
        )

    def context(self):
        w = self.window
        view = {n: getattr(w, n).currentData() for n in ("quantity", "mode", "altitude", "x", "y")}
        view.update(
            opacity=w.opacity.value(),
            uniform_colour=w.uniform_colour.isChecked(),
            readable_sensor=w.readable_sensor.isChecked(),
            dim_context=w.dim_context.isChecked(),
            hide_volume=w.hide_volume.isChecked(),
            highlight_cell=w.highlight_cell.isChecked(),
            visible=w.visible.isChecked(),
            cell_edges=w.cell_edges.isChecked(),
            unavailable_edges=w.unavailable_edges.isChecked(),
            inspect=w.inspect.isChecked(),
            evaluation=w.evaluation_controls.state(),
            cutaway=w.cutaway.state(),
            inspection=self.controls.state(),
            antenna=w.antenna.state(),
        )
        camera = w.visualizer.renderer.get_camera_state()
        pair = (
            None
            if self.pair is None
            else dict(
                volume_sha256=self.pair[0].b.identity,
                changed_variable=self.pair[0].changed_variable,
                volume=str(self.pair[0].b.path),
            )
        )
        return context_snapshot(
            w.reader,
            w.heading.currentIndex(),
            w._cell_key(),
            view,
            camera.to_dict() if camera else None,
            self.selected_report,
            pair,
        )

    def draw(self):
        w, c = self.window, self.controls
        if w.closed or w.reader is None:
            return
        r, h = w.reader, w.heading.currentIndex()
        volume_visible = w.visible.isChecked() and not (
            w.inspect.isChecked() and w.hide_volume.isChecked()
        )
        quantity = w.quantity.currentData()
        with QSignalBlocker(w.evaluation_controls):
            w.evaluation_controls.set_quantity(quantity)
        signature = (r.identity, h, *self.criteria().values(), w.evaluation is not None)
        if signature != w.evaluation_signature:
            w.evaluation_signature = signature
            w.evaluation_snapshot = (
                w.evaluation.evaluate(h, **self.criteria()) if w.evaluation else None
            )
            self.contributions = (
                matched_contributions(w.evaluation, h, w.evaluation_snapshot)
                if r.is_fusion and w.evaluation is not None
                else None
            )
            w.templates.clear()
        snapshot = w.evaluation_snapshot
        if (
            quantity in {"details", "extras", "reports", "pair", "contributions"}
            and snapshot is None
        ):
            # A new source/criterion is pending: retire old fills and support
            # edges together instead of leaving a stale acquisition visible.
            w.overlay.clear(w.visualizer.renderer)
            w.visualizer.renderer.set_analysis_legend(None)
            w.status.setText("Loading saved reports for compact inspection… Detector unchanged.")
            c.slice_view.hide()
            w.cell_note.setText("Loading the current source quantity…")
            return
        if quantity == "pair" and self.pair is None:
            w.overlay.clear(w.visualizer.renderer)
            w.visualizer.renderer.set_analysis_legend(None)
            w.open_native.setEnabled(False)
            c.slice_view.hide()
            w.cell_note.setText("Loading paired source quantity…")
            w._load_pair()
            return
        if self.pair is not None and signature != self.pair_signature:
            self.pair_signature = signature
            self.pair_snapshot = (
                self.pair[0].evaluate(w.evaluation, self.pair[1], h, **self.criteria())
                if snapshot
                else None
            )
        basis = w.evaluation_controls.basis.currentData() if snapshot else "saved"
        outcomes = snapshot["outcomes"] if snapshot else r.outcome_grid(h)
        details = detail_grid(snapshot) if snapshot else None
        self.mask = (
            focus_mask(
                r, h, snapshot, c.focus.currentData(), c.threshold.value(), c.minimum.value()
            )
            if snapshot
            else np.isfinite(outcomes)
        )
        if details is not None:
            for i, check in enumerate(c.category_filters):
                if not check.isChecked():
                    self.mask &= details != i
        with QSignalBlocker(w.mode):
            restricted = quantity != "outcomes"
            for i in range(w.mode.count()):
                available = not restricted or w.mode.itemData(i) in {"slice", "both"}
                w.mode.model().item(i).setEnabled(available)
                w.mode.view().setRowHidden(
                    i, not available or w.mode.itemData(i) in {"hit", "miss"}
                )
            w.mode.setItemText(
                w.mode.findData("both"),
                "3D volume",
            )
            if restricted and w.mode.currentData() not in {"slice", "both"}:
                w.mode.setCurrentIndex(w.mode.findData("both"))
        display = w.mode.currentData()
        w.uniform_colour.setEnabled(display not in {"slice", "mip"})
        can_sweep = display == "slice" or (
            w.cutaway.active.isChecked() and w.cutaway.mode.currentData() == "plane"
        )
        if not can_sweep:
            self.sweep.stop()
        w.cutaway.setVisible(display != "slice")
        w.slice_coordinate.setVisible(display == "slice")
        w.slice_coordinate_label.setVisible(display == "slice")
        c.sweep_controls.setVisible(can_sweep)
        c.sweep_controls.setEnabled(can_sweep)
        c.buttons["play"].setText(
            "Pause sweep"
            if self.sweep.isActive()
            else "Sweep slice" if display == "slice" else "Sweep 3D"
        )
        c.orientation.setVisible(display == "slice")
        c.orientation_label.setVisible(display == "slice")
        cut = w.cutaway.cut(display)
        orientation = c.orientation.currentData()
        normal = {"xy": "z", "xz": "y", "yz": "x"}[orientation]
        w.mode.setItemText(w.mode.findData("slice"), "Slice")
        w._sync_slice_coordinate(normal)
        coordinate = {"x": w.x, "y": w.y, "z": w.altitude}[normal].currentData()
        plane = tuple(1 if n == normal else 0 for n in ("x", "y", "z")) + (-coordinate,)
        regions = (
            ("hit", "miss")
            if quantity == "outcomes" and display == "both"
            else (
                (display,) if quantity == "outcomes" and display in {"hit", "miss"} else (quantity,)
            )
        )
        state = c.state()
        cache_common = (
            signature,
            quantity,
            cut,
            state["focus"],
            state["threshold"],
            state["minimum"],
            tuple(state["categories"]),
            state["count_max"],
            w.evaluation_controls.error_max.value(),
        )
        objects = []
        slice_signature = (
            signature,
            quantity,
            state["count_max"],
            w.evaluation_controls.error_max.value(),
            id(self.pair_snapshot),
        )
        if getattr(self, "_source_slice_signature", None) != slice_signature:
            # Slice views show the complete source quantity, not the first ISO region
            # or its filtered/cut occupancy mask. Source indices remain shared.
            if quantity == "reports":
                slice_payload, _ = error_volume_template(
                    r, h, snapshot, w.evaluation_controls.error_max.value()
                )
            elif quantity == "details":
                slice_payload, _ = inspection_template(r, h, details, colors=DETAIL_COLORS)
            elif quantity == "contributions":
                slice_payload, _ = inspection_template(
                    r, h, self.contributions["values"], colors=self.contributions["colors"]
                )
            elif quantity == "extras":
                slice_payload, _ = inspection_template(
                    r, h, extras_grid(snapshot), maximum=c.count_max.value()
                )
            elif quantity == "pair":
                slice_payload, _ = inspection_template(
                    r, h, self.pair_snapshot["values"], colors=PAIR_COLORS
                )
            elif quantity == "geometry":
                slice_payload, _ = inspection_template(
                    r, h, r.geometry_grid(h), colors=("#1accf2", "#a666f2")
                )
            else:
                slice_payload, _ = volume_template(r, h, "outcomes", outcomes=outcomes)
            self._source_slice_payload = slice_payload
            self._source_slice_signature = slice_signature
        slice_payload = self._source_slice_payload
        visible_count = 0
        self.visible_mask = np.zeros(outcomes.shape, dtype=bool)
        for index, region in enumerate(regions):
            key = (cache_common, region)
            if key not in w.templates:
                if len(w.templates) >= 2:
                    w.templates.pop(next(iter(w.templates)))
                if quantity == "reports":
                    template, transform = error_volume_template(
                        r, h, snapshot, w.evaluation_controls.error_max.value()
                    )
                elif quantity == "details":
                    template, transform = inspection_template(r, h, details, colors=DETAIL_COLORS)
                elif quantity == "contributions":
                    template, transform = inspection_template(
                        r, h, self.contributions["values"], colors=self.contributions["colors"]
                    )
                elif quantity == "extras":
                    template, transform = inspection_template(
                        r, h, extras_grid(snapshot), maximum=c.count_max.value()
                    )
                elif quantity == "pair":
                    template, transform = inspection_template(
                        r, h, self.pair_snapshot["values"], colors=PAIR_COLORS
                    )
                else:
                    template, transform = volume_template(
                        r, h, "blocked" if quantity == "geometry" else region, outcomes=outcomes
                    )
                values = (
                    np.where(self.mask, template.values, 0)
                    if quantity != "geometry" or snapshot is not None
                    else template.values
                )
                template = cut_volume(replace(template, values=values), cut)
                w.templates[key] = (template, transform)
            template, transform = w.templates[key]
            payload = replace(
                template,
                mode="slice" if display == "slice" else "mip" if display == "mip" else "iso",
                plane=plane,
                opacity=w.opacity.value()
                * (0.3 if w.inspect.isChecked() and w.dim_context.isChecked() else 1),
                lighting=not w.uniform_colour.isChecked(),
                surface_grid=w.cell_edges.isChecked(),
            )
            objects.append(
                RenderObject(
                    id="sensing:outcome_volume" + (":misses" if index else ""),
                    payload=payload,
                    transform=transform,
                    visibility=volume_visible,
                    metadata={"type": "sensing_volume"},
                )
            )
            support = payload.values > 0
            if display == "slice":
                plane_index = [slice(None)] * 3
                plane_index[{"x": 2, "y": 1, "z": 0}[normal]] = w._cell_key()[
                    {"z": 1, "y": 2, "x": 3}[normal]
                ]
                support = support[tuple(plane_index)]
                self.visible_mask[tuple(plane_index)] |= support
            else:
                self.visible_mask |= support
            visible_count += int(support.sum())
        edge_key = (
            r.identity,
            h,
            w.cell_edges.isChecked(),
            w.unavailable_edges.isChecked(),
            display,
            normal,
            w._cell_key()[{"z": 1, "y": 2, "x": 3}[normal]] if display == "slice" else None,
            cut,
            volume_visible,
        )
        if getattr(self, "edge_key", None) != edge_key:
            self.edge_key = edge_key
            self.edge_objects = cell_edge_objects(
                r,
                h,
                enabled=w.cell_edges.isChecked(),
                include_unavailable=w.unavailable_edges.isChecked(),
                display=display,
                normal=normal,
                index=w._cell_key()[{"z": 1, "y": 2, "x": 3}[normal]],
                cut=cut,
                visible=volume_visible,
            )
        edges, edge_notes = self.edge_objects
        w.cell_edges.setToolTip("\n".join(edge_notes))
        objects.extend(edges)
        sensor = np.asarray(r.metadata["sensor_position_m"], np.float32)
        forward = getattr(r, "inspection_forward", None)
        c.set_geometry_context(
            sensor,
            r.headings[h]["velocity_m_s"],
            r.axes,
            forward_available=forward is not None,
        )
        objects += geometry_objects(
            r,
            h,
            forward,
            show_forward=c.forward.isChecked() and not r.is_fusion,
            show_motion=c.motion.isChecked() and not r.is_fusion,
            show_zero=c.zero.isEnabled() and c.zero.isChecked() and not r.is_fusion,
        )
        if not r.is_fusion:
            objects += w.antenna.objects(r, w.executor)
        inspected = w.inspect.isChecked()
        if inspected and volume_visible and w.highlight_cell.isChecked():
            key = (r.identity, w._cell_key(), normal if display == "slice" else None)
            if getattr(self, "highlight_key", None) != key:
                self.highlight_key = key
                self.highlight_objects = selected_cell_objects(
                    r.axes,
                    w._cell_key()[1:][::-1],
                    normal=key[-1],
                    prefix="sensing:volume:selected_cell",
                )
            objects.extend(self.highlight_objects)
        w.inspector.setVisible(inspected)
        w.inspect_section.setVisible(inspected)
        w.selection_hint.setVisible(not inspected)
        for name in ("previous", "next"):
            c.buttons[name].setEnabled(inspected)
        for axis, combo in (("x", w.x), ("y", w.y), ("z", w.altitude)):
            combo.setVisible(True)
            combo.setEnabled(inspected)
            w.coordinate_labels[axis].setVisible(True)
        c.selected.setVisible(inspected)
        trial = r.cells.get(w._cell_key())
        measured = trial is not None and trial["status"] == "measured"
        row_signature = (signature, w._cell_key(), inspected)
        if self.row_signature != row_signature:
            self.row_signature = row_signature
            self.rows = report_details(w.evaluation, w._cell_key(), snapshot) if snapshot else []
            for row in self.rows:
                if row["matched"]:
                    row["association"] = "Matched"
                elif row["position_m"] is None:
                    row["association"] = "Extra · no 3D position"
                elif self.criteria()["mode"] == "position":
                    row["association"] = (
                        "Extra · within gate"
                        if row["error_m"] <= self.criteria()["position_gate_m"]
                        else "Extra · outside gate"
                    )
                else:
                    row["association"] = "Extra"
            if (
                inspected
                and snapshot is not None
                and self.selected_report not in [row["detection_id"] for row in self.rows]
            ):
                self.selected_report = next(
                    (row["detection_id"] for row in self.rows if row["matched"]), None
                )
            c.set_reports(self.rows, self.selected_report)
        if inspected:
            objects += trial_objects(
                r,
                w._cell_key(),
                self.rows,
                self.selected_report,
                vectors=c.vectors.isChecked(),
                through=c.through.isChecked(),
            )
        c.report_note.setText(
            (
                "Saved reports not loaded."
                if snapshot is None
                else (
                    "No saved detector reports."
                    if measured and not self.rows
                    else f"{len(self.rows)} saved reports; {sum(row['position_m'] is None for row in self.rows)} without 3D position."
                )
            )
            + "\nTested target: reference position. Cyan: current match; amber: other reports. Error vectors compare positions, not propagation paths."
            + (
                " Diagnostic depth override is ON."
                if c.through.isChecked()
                else " Normal scene occlusion."
            )
        )
        selected_row = next(
            (row for row in self.rows if row["detection_id"] == self.selected_report), None
        )
        w.contributor_note.setVisible(inspected and r.is_fusion)
        if r.is_fusion:
            support = selected_row.get("sensor_support", []) if selected_row else []
            w.contributor_note.setText(
                "Contributors: " + " + ".join(name.title() for name in support)
                if support
                else (
                    "Contributors not recorded"
                    if selected_row
                    else "Select an estimate to see its sensor contributions."
                )
            )
        if inspected and selected_row and selected_row["inside_sample_cell"] is not None:
            location = "inside" if selected_row["inside_sample_cell"] else "outside"
            c.report_note.setText(
                f"Selected report: {location} the sampling cell. Matching uses the evaluation gate.\n"
                + c.report_note.text()
            )
        native = r.native_availability(w._cell_key())
        w.open_native.setEnabled(inspected and native == "available" and w.native_pending is None)
        w.open_native.setToolTip(
            (
                "Inspect the selected fused estimate's sensor contributions and each sensor's saved products. Return to volume restores your selection."
                if r.is_fusion
                else "Open this cell's saved scene, playback and sensing plots. Return to volume restores your selection."
            )
            if native == "available"
            else native
        )
        w.open_native.setText(
            "Inspect fusion & sensors"
            if r.is_fusion
            else "Open A trial in visualizer" if quantity == "pair" else "Open trial in visualizer"
        )
        w.native_note.setText(
            "Opening the saved trial…"
            if w.native_pending is not None
            else native if inspected and native != "available" else ""
        )
        w.native_note.setVisible(bool(w.native_note.text()))
        c.native_b.setVisible(quantity == "pair")
        if quantity == "pair":
            other = self.pair[0].b
            other_key = (self.pair[0].headings[h], *w._cell_key()[1:])
            availability = other.native_availability(other_key)
            c.native_b.setEnabled(availability == "available" and w.native_pending is None)
            c.native_b.setToolTip(availability)

        if measured:
            cell = w._cell_key()[1:]
            tp = int(snapshot["outcomes"][cell]) if snapshot else trial["tp"]
            fp = int(snapshot["report_counts"][cell]) - tp if snapshot else trial["fp"]
            text = f"{trial['id']} · TP {tp} / FP {fp} / FN {1 - tp}\n{tp + fp} saved detector reports from this one acquisition."
            if "direct_path_blocked" in trial:
                text += f"\nDirect centre path: {'blocked' if trial['direct_path_blocked'] else 'clear'}."
            if "blocked_target_probes" in trial:
                text += f" {trial['blocked_target_probes']}/{trial['target_probe_count']} probes blocked."
            if trial.get("radial_closing_velocity_m_s") is not None:
                text += f"\nExpected direct radial closing velocity: {trial['radial_closing_velocity_m_s']:.3f} m/s."
            if native != "available":
                text += "\n" + native
            if not self.mask[cell]:
                text += "\nSelected position is outside the current focus."
            if quantity == "pair":
                peer = self.pair_snapshot["b"]
                v = peer["outcomes"][cell]
                text += "\nB: " + (
                    f"TP {int(v)} / FP {int(peer['report_counts'][cell] - v)} / FN {1 - int(v)}"
                    if np.isfinite(v)
                    else "no measured outcome"
                )
            w.cell_note.setText(text)
        else:
            w.cell_note.setText(
                "No measured outcome: " + (trial["status"] if trial else "not sampled")
            )
        centre = ", ".join(
            f"{axis.upper()} {r.axes[axis][w._cell_key()[position]]:g}"
            for axis, position in (("x", 3), ("y", 2), ("z", 1))
        )
        sensor_text = ", ".join(f"{value:g}" for value in sensor)
        cell = w._cell_key()[1:]
        _, spacing, lower, upper = cell_support(r, w._cell_key())
        bounds_text = "; ".join(
            f"{n.upper()} {lo:g} m to {hi:g} m"
            for n, lo, hi in zip("xyz", lower, upper, strict=True)
        )
        size_text = " × ".join(f"{value:g} m" for value in spacing)
        raw = self._raw_cell_value(quantity, outcomes, details, snapshot, cell)
        identity = r.metadata.get("transmitter_id", r.metadata.get("sensor_id", "not recorded"))
        availability = (
            trial.get("reason", trial.get("status", "not sampled")) if trial else "not sampled"
        )
        w.cell_note.setText(
            f"{self.selection_origin if self.selection_origin_key == w._cell_key() else 'Source grid selection'}\nSource index XYZ: {cell[2]}, {cell[1]}, {cell[0]}\n"
            f"Centre: {centre} m\nRaw source value: {raw}\n"
            f"Sampling cell: {size_text}\nBounds: {bounds_text}\n"
            "Cell bounds describe sampling, not the matching gate or position uncertainty.\n"
            f"Transmitter / sensor ID: {identity}; position ({sensor_text}) m\n"
            f"Heading: {r.headings[h]['name']}\nAvailability: {availability}\n" + w.cell_note.text()
        )
        if not inspected:
            w.cell_note.setText(
                "No cell selected. Click a volume cell or enable coordinate selection."
            )
        w.cell_summary.setText(
            f"{centre} m · {availability} · {len(self.rows)} saved reports"
            if inspected
            else "Click a volume cell to inspect its saved trial."
        )
        w.clear_selection.setEnabled(inspected)
        if snapshot:
            w.status.setText(
                f"TP {snapshot['tp']} / FP {snapshot['fp']} / FN {snapshot['fn']} · {snapshot['measured']} measured positions"
            )
            w.summary_detail.setText(
                f"All {snapshot['measured']:,} measured cells. View filters and cuts do not change these charts."
            )
            w.summary_plots.update_snapshot(snapshot)
            if self.contributions is not None:
                w.contribution_plot.update_contributions(self.contributions)
        else:
            w.status.setText("Saved baseline outcomes; report details are loading.")
        c.focus_note.setText(
            f"{visible_count:,} cells shown; {int(self.mask.sum()):,} match the filter."
            + (
                " Saved category exclusions are active; Reset clears them."
                if not all(check.isChecked() for check in c.category_filters)
                else ""
            )
        )
        for name, widget in (("blocked", c.focus), ("clear", c.focus), ("all_blocked", c.focus)):
            available = any(
                ("target_probe_count" if name == "all_blocked" else "direct_path_blocked") in t
                for t in r.cells.values()
            )
            item = widget.model().item(widget.findData(name))
            if item is not None:
                item.setEnabled(available)
        for widget in (c.threshold, c.threshold_label):
            widget.setVisible(c.focus.currentData() == "error")
        for widget in (c.minimum, c.minimum_label):
            widget.setVisible(c.focus.currentData() == "count")
        w.count_scale_row.setVisible(quantity == "extras")
        legend = volume_legend(
            r,
            h,
            quantity if quantity in {"reports", "geometry"} else basis,
            display,
            w.evaluation_controls.state(),
        )
        lines = list(legend.lines)
        lines.extend(w.antenna.legend_lines())
        if display == "slice":
            lines.append(f"{orientation.upper()} · {normal.upper()} = {coordinate:g} m")
        elif display == "mip":
            lines.append("Maximum along view")
        if quantity in {"details", "pair"}:
            labels, colors = (
                (DETAIL_LABELS, DETAIL_COLORS)
                if quantity == "details"
                else (PAIR_LABELS, PAIR_COLORS)
            )
            legend = replace(
                legend,
                title="Outcome detail" if quantity == "details" else "Paired change",
                swatches=tuple(zip(labels, colors, strict=True)),
            )
        if quantity == "extras":
            legend = replace(
                legend,
                title=(
                    "Unmatched estimates per trial"
                    if r.is_fusion
                    else "Unmatched reports per trial"
                ),
                swatches=(),
                colorbar_label=(
                    "Estimates/trial (capped)" if r.is_fusion else "Reports/trial (capped)"
                ),
                colorbar_range=(0, c.count_max.value()),
            )
        if quantity == "contributions":
            legend = replace(
                legend,
                title="Contributors to matched estimate",
                swatches=tuple(
                    zip(self.contributions["labels"], self.contributions["colors"], strict=True)
                ),
            )
        if quantity == "pair":
            lines.append(
                f"A: {r.metadata.get('condition', r.path.parent.name)} · B: {self.pair[0].b.metadata.get('condition', self.pair[0].b.path.parent.name)}"
            )
        if w.cell_edges.isChecked() and w.unavailable_edges.isChecked():
            lines.append("Dashed edges: unmeasured")
        condition = r.metadata.get("condition")
        legend = replace(
            legend,
            title=f"{condition} · {legend.title}" if condition else legend.title,
            lines=tuple(lines),
        )
        w.legend.setText(legend.title + "\n" + " · ".join(label for label, _ in legend.swatches))
        w.display_note.setText("\n".join(lines))
        w.visualizer.renderer.set_analysis_legend(legend if volume_visible else None)
        from ..services.label_callout_service import objects_with_callouts

        objects = objects_with_callouts(objects, w.visualizer)
        if not w.overlay.sync(w.visualizer.renderer, objects):
            w.status.setText("Volume rendering failed; source data unchanged.")
        w.visualizer.renderer.request_redraw()
        c.slice_view.setVisible(inspected and c.slice_plot.isChecked())
        if c.slice_plot.isChecked():
            c.slice_view.update_slice(
                slice_payload, r, orientation, w._cell_key(), show_selection=inspected
            )
        self.sweep.setInterval(max(50, round(1000 / c.rate.value())))
        c.sweep_note.setText(
            ("Playing" if self.sweep.isActive() else "Paused")
            + f" · spatial {normal.upper()} planes, not simulation time."
        )
        w.tabs.updateGeometry()

    def _raw_cell_value(self, quantity, outcomes, details, snapshot, cell):
        """Describe retained scientific data, never the palette/occupancy code."""
        w = self.window
        if quantity == "reports":
            value, units = snapshot["nearest_error_m"][cell], "m (nearest report position error)"
        elif quantity == "extras":
            value, units = extras_grid(snapshot)[cell], "unmatched reports / trial"
        elif quantity == "details":
            value, units = details[cell], "(outcome category)"
            if np.isfinite(value):
                return f"{int(value)} — {DETAIL_LABELS[int(value)]}"
        elif quantity == "contributions":
            value = self.contributions["values"][cell]
            return (
                self.contributions["labels"][int(value)]
                if np.isfinite(value)
                else "unavailable / not retained"
            )
        elif quantity == "geometry":
            value, units = (
                w.reader.geometry_grid(w.heading.currentIndex())[cell],
                "(direct-path blockage category)",
            )
        elif quantity == "pair":
            value, units = self.pair_snapshot["values"][cell], "(paired category)"
        else:
            value, units = outcomes[cell], "(1 matched, 0 acquired miss; dimensionless)"
        if quantity == "reports" and not np.isfinite(value):
            if np.isfinite(snapshot["outcomes"][cell]):
                return (
                    "not defined — no saved detector reports"
                    if snapshot["report_counts"][cell] == 0
                    else "not defined — saved reports have no 3D position"
                )
        return f"{value:g} {units}" if np.isfinite(value) else "unavailable / not retained"

    def select_volume_cell(self, object_id, indices, mode):
        """Select the source cell returned by native visible-hit picking."""
        w = self.window
        if (
            w.closed
            or w.reader is None
            or object_id not in {"sensing:outcome_volume", "sensing:outcome_volume:misses"}
        ):
            return
        self.sweep.stop()
        for widget, index in zip((w.x, w.y, w.altitude), indices, strict=True):
            if not 0 <= index < widget.count():
                return
        for widget, index in zip((w.x, w.y, w.altitude), indices, strict=True):
            with QSignalBlocker(widget):
                widget.setCurrentIndex(index)
        self.selection_origin = {
            "mip": "Ray-maximum hit (MIP; not every cell along the ray)",
            "iso": "Visible ISO surface hit (not a hidden cell)",
            "slice": "Visible slice cell",
        }[mode]
        with QSignalBlocker(w.inspect):
            w.inspect.setChecked(True)
        w.inspect_section.expand()
        self.selection_origin_key = w._cell_key()
        w._draw()

        # Bring the opening/focus actions into view after a scene pick, even
        # when the panel is narrow or the Analysis tab is active.
        QTimer.singleShot(0, w.inspector, lambda: w.scroll.ensureWidgetVisible(w.inspector))

    def select_report(self, identifier):
        self.sweep.stop()
        self.selected_report = identifier
        self.window._draw()

    def select_slice(self, orientation, horizontal, vertical):
        self.selection_origin = f"{orientation.upper()} source slice cell"
        self.sweep.stop()
        w = self.window
        h, v = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}[orientation]
        for name, value in ((h, horizontal), (v, vertical)):
            widget = {"x": w.x, "y": w.y, "z": w.altitude}[name]
            with QSignalBlocker(widget):
                widget.setCurrentIndex(int(np.argmin(abs(w.reader.axes[name] - value))))
        with QSignalBlocker(w.inspect):
            w.inspect.setChecked(True)
        w.inspect_section.expand()
        self.selection_origin_key = w._cell_key()
        w._draw()

    def _advance(self):
        self.step_plane(1)

    def step_plane(self, direction):
        w = self.window
        if w.mode.currentData() == "slice":
            normal = {"xy": "z", "xz": "y", "yz": "x"}[self.controls.orientation.currentData()]
            combo = {"x": w.x, "y": w.y, "z": w.altitude}[normal]
        else:
            if not w.cutaway.active.isChecked() or w.cutaway.mode.currentData() != "plane":
                return
            combo = w.cutaway.boundary
        index = combo.currentIndex() + direction
        if not 0 <= index < combo.count():
            if self.controls.loop.isChecked():
                index %= combo.count()
            else:
                self.sweep.stop()
                w._draw()
                return
        self.stepping = True
        try:
            combo.setCurrentIndex(index)
        finally:
            self.stepping = False

    def action(self, name):
        w, c = self.window, self.controls
        if w.closed:
            return
        if w.reader is None:
            return
        try:
            if name == "native_b":
                w._open_native("b")
            elif name in {"plane_previous", "plane_next"}:
                self.step_plane(-1 if name == "plane_previous" else 1)
            elif name == "play":
                if self.sweep.isActive():
                    self.sweep.stop()
                else:
                    if w.mode.currentData() != "slice" and (
                        not w.cutaway.active.isChecked() or w.cutaway.mode.currentData() != "plane"
                    ):
                        return
                    self.sweep.start(round(1000 / c.rate.value()))
                w._draw()
            elif name in {"previous", "next"}:
                if not w.inspect.isChecked() or self.mask is None:
                    return
                indices = np.flatnonzero(self.visible_mask)
                if not len(indices):
                    w.cell_note.setText("No positions match the focus.")
                    return
                current = np.ravel_multi_index(w._cell_key()[1:], w.reader.shape)
                index = (
                    np.searchsorted(indices, current, side="right")
                    if name == "next"
                    else np.searchsorted(indices, current) - 1
                ) % len(indices)
                z, y, x = np.unravel_index(indices[index], w.reader.shape)
                for widget, value in ((w.altitude, z), (w.y, y), (w.x, x)):
                    with QSignalBlocker(widget):
                        widget.setCurrentIndex(int(value))
                with QSignalBlocker(w.inspect):
                    w.inspect.setChecked(True)
                w._draw()
            elif name in {"frame_reports", "side_reports", "frame_focus", "locate_report"}:
                if name == "frame_focus":
                    if self.mask is None or not self.mask.any():
                        return
                    indices = np.argwhere(self.mask)
                    points = np.column_stack(
                        [w.reader.axes[n][indices[:, i]] for n, i in [("x", 2), ("y", 1), ("z", 0)]]
                    )
                else:
                    trial = w.reader.cells.get(w._cell_key())
                    if not trial:
                        return
                    rows = self.rows
                    if name == "locate_report":
                        rows = [
                            row
                            for row in rows
                            if row["detection_id"] == self.selected_report
                            and row["position_m"] is not None
                        ]
                        if not rows:
                            w.cell_note.setText("Select a report with a saved 3D position first.")
                            return
                    points = np.array(
                        [trial["position_m"]]
                        + [row["position_m"] for row in rows if row["position_m"] is not None]
                    )
                lo, hi = points.min(axis=0), points.max(axis=0)
                w._prepare_camera()
                w.visualizer.renderer.set_overview_camera(
                    "side" if name == "side_reports" else "isometric",
                    ((lo + hi) / 2, np.maximum(hi - lo, 2) * 1.4),
                )
            elif name == "bookmark_load":
                path, _ = QFileDialog.getOpenFileName(
                    self.parent(), "Open volume view bookmark", "", "JSON (*.json)"
                )
                if path:
                    from .volume_window import open_sensing_volume

                    state = load_bookmark(path)
                    open_sensing_volume(w.visualizer, state["volume"], context=state)
            elif name in {"bookmark_save", "capture"}:
                suffix = "png" if name == "capture" else "json"
                path, _ = QFileDialog.getSaveFileName(
                    self.parent(),
                    "Save image" if name == "capture" else "Save volume view bookmark",
                    "inspection." + suffix,
                    f"{suffix.upper()} (*.{suffix})",
                )
                if not path:
                    return
                if name == "capture":
                    if not w.visualizer.renderer.export_screenshot(path, include_hud=True):
                        raise ValueError("Renderer screenshot failed")
                    path = str(Path(path).with_suffix(".json"))
                save_bookmark(path, self.context())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            w.status.setText(f"Inspection action unavailable: {exc}")
