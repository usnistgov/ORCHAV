"""Compact saved-cell values and links to recorded detector/path evidence."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ..widgets.wrapping_layout import WrappingLayout
from .dashboard_diagnostics import cell_explanation
from .selection import catalog_identity, selection_scope
from .stage_inspector import DETECTION_STAGE_KINDS, CellSelection


def _db(value: float | None) -> str:
    if value is None or not np.isfinite(value) or value < 0:
        return "unavailable"
    return f"{10 * np.log10(value):.1f} dB" if value else "zero power"


class CellExplanation(QWidget):
    """Explain numerical coordinates without inferring targets or ray attribution."""

    paths_requested = Signal()
    detector_requested = Signal()

    def __init__(self, owner):
        super().__init__(owner)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.values = QLabel()
        self.decision = QLabel()
        self.paths_note = QLabel()
        for label in (self.values, self.decision, self.paths_note):
            label.setWordWrap(True)
            layout.addWidget(label)
        buttons = WrappingLayout()
        self.detector_button = QPushButton("Inspect detector stages")
        self.detector_button.clicked.connect(self.detector_requested.emit)
        self.path_button = QPushButton("Inspect delay-matched paths…")
        self.path_button.setToolTip("Inspect paths within this bin's delay interval.")
        self.path_button.clicked.connect(self.paths_requested.emit)
        buttons.addWidget(self.detector_button)
        buttons.addWidget(self.path_button)
        layout.addLayout(buttons)
        self.locus = None
        self.hide()

    def update_cell(self, request, selection, *, velocity: bool = True):
        self.locus = None
        cell = selection.cell if selection is not None else None
        self.setVisible(request is not None and cell is not None)
        if request is None or cell is None:
            return
        sensing = request.display_frame["sensing"]
        temporal = sensing["temporal"]
        current = (
            selection.scope == selection_scope(sensing)
            and selection.source == request.source_identity == cell.source
            and selection.acquisition_id == temporal["acquisition"]["acquisition_id"]
            and cell.recipe == temporal["result"]["processing_recipe_id"]
            and cell.catalog_identity == catalog_identity(sensing)
            and cell.variant == request.variant
        )
        if not current:
            self.hide()
            return
        r, d = cell.range_bin, cell.doppler_bin
        range_value = cell.ranges[r]
        distance_label = "Total path length" if request.range_mode == "bistatic" else "Range"
        fields = [f"{distance_label} {range_value:.3f} m"]
        if d is not None and cell.doppler:
            hz = cell.doppler[d]
            if velocity:
                speed = hz * 299792458.0 / (2 * request.carrier_freq_hz)
                name = (
                    "Equivalent velocity"
                    if request.range_mode == "bistatic"
                    else "Closing velocity"
                )
                fields.append(f"{name} {speed:+.3f} m/s")
            fields.append(f"Doppler {hz:+.3g} Hz")
        rd = request.rd_result
        if (
            rd is not None
            and d is not None
            and np.array_equal(rd.range_axis_m, cell.ranges)
            and np.array_equal(rd.doppler_axis_hz, cell.doppler)
        ):
            fields.append(f"Displayed power {_db(float(rd.power_linear[d, r]))}")
        self.values.setText(" · ".join(fields))
        bundle = request.detection_stage_bundle
        compatible = (
            bundle is not None
            and bundle.available
            and d is not None
            and bundle.event.acquisition_id == selection.acquisition_id
            and (bundle.event.clock_id, bundle.event.processing_run_id, bundle.event.pair_id)
            == selection.scope
            and bundle.event.processing_recipe_id == cell.recipe
            and np.array_equal(bundle.range_axis_m, cell.ranges)
            and np.array_equal(bundle.doppler_axis_hz, cell.doppler)
        )
        if compatible:
            evidence = bundle.inspect(CellSelection(bundle.event.pair_id, d, r))
            self.decision.setText(
                cell_explanation(evidence).split(" Threshold margin:", 1)[0]
                + f" Detector input {_db(evidence.detection_input_power)}; threshold {_db(evidence.cfar_threshold_power)}."
            )
            self.decision.setToolTip(
                cell_explanation(evidence)
                + "\nThe saved detector may use a different beam from the displayed RD spectrum."
            )
        else:
            self.decision.setText(
                "Detector details not loaded"
                if bundle is None
                else "Detector evidence unavailable for this cell."
            )
            self.decision.setToolTip(bundle.reason or "" if bundle is not None else "")
        saved_kinds = request.available_product_kinds or {
            descriptor.get("kind") for descriptor in sensing.get("products", {}).values()
        }
        self.detector_button.setEnabled(
            compatible or (bundle is None and bool(saved_kinds.intersection(DETECTION_STAGE_KINDS)))
        )
        try:
            from ..experiments.range_geometry import build_range_locus

            self.locus = build_range_locus(request, selection)
        except (ValueError, KeyError, TypeError, AttributeError):
            pass  # Coordinates may still be useful without a usable scene/path binding.
        self.paths_note.setText("Path selection: delay interval")
        self.path_button.setEnabled(self.locus is not None and len(self.locus.support_m) == 2)
