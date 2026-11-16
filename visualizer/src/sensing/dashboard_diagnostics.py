"""Plain-language explanations and display transforms of saved detector evidence."""

from dataclasses import dataclass

import numpy as np

from .stage_inspector import CellInspection, DetectionStageBundle


@dataclass(frozen=True)
class ThresholdMargin:
    value_db: float | None
    note: str
    comparison_passed: bool | None = None


def threshold_margin(power, threshold, eligible, saved=None) -> ThresholdMargin:
    """Inspect the linear test without a display floor or invented denominator."""
    if eligible is not True:
        return ThresholdMargin(
            None, "not tested" if eligible is False else "eligibility unavailable"
        )
    if power is None or threshold is None:
        return ThresholdMargin(None, "input or threshold not recorded")
    if not np.isfinite(power) or not np.isfinite(threshold) or min(power, threshold) < 0:
        return ThresholdMargin(None, "invalid/nonfinite input or threshold")
    passed = bool(power > threshold)
    if power == threshold == 0:
        value, note = None, "undefined ratio 0/0; strict comparison fails"
    elif threshold == 0:
        value, note = np.inf, "zero threshold; infinite ratio, not calibrated confidence"
    elif power == 0:
        value, note = -np.inf, "zero input power"
    else:
        value = float(10 * (np.log10(power) - np.log10(threshold)))
        note = "equality fails strict >" if power == threshold else "linear input / saved threshold"
    if saved is not None and passed != saved:
        note += "; CONSISTENCY ERROR: strict comparison disagrees with saved CFAR mask"
    return ThresholdMargin(value, note, passed)


def stage_counts(bundle: DetectionStageBundle, reports: int) -> str:
    """Count only the saved pair grid; this is not a count across all beams."""
    counts = []
    for kind in (
        "range_doppler_cfar_mask",
        "range_doppler_nms_mask",
        "range_doppler_cluster_labels",
    ):
        product = bundle.product(kind)
        if not product.available:
            counts.append("not recorded")
        elif kind.endswith("labels"):
            counts.append(str(len(np.unique(product.values[product.values >= 0]))))
        else:
            counts.append(str(np.count_nonzero(product.values)))
    return (
        f"Saved pair grid (possibly a winning-beam projection): CFAR-positive cells {counts[0]} · "
        f"NMS peaks {counts[1]} · clusters {counts[2]}. Published reports in this result: {reports}. "
        "Projected counts do not count every evaluated beam."
    )


STAGE_HELP = {
    "range_doppler_detection_input_power": "Power tested by the detector. The slice compares this input with its saved threshold at one Doppler bin.",
    "range_doppler_cfar_noise_power": "Local background estimate from the recorded CFAR training cells; this is not a calibrated receiver noise measurement.",
    "range_doppler_cfar_threshold_power": "Saved detection threshold, including the configured peak-relative power floor. That floor can make the threshold constant. An eligible cell passes when its input is strictly above this threshold.",
    "range_doppler_cfar_training_cell_count": "Number of cells used for the local background estimate. A fixed CFAR stencil gives the same count in every eligible interior cell; excluded borders can be zero. This is not a target count.",
    "range_doppler_cfar_eligible_mask": "1: tested by CFAR. 0: excluded before testing, for example at the grid border or by a DC exclusion.",
    "range_doppler_cfar_mask": "1: passed the saved CFAR test. Passing cells can still be removed by peak suppression, clustering, or tracking.",
    "range_doppler_nms_mask": "1: retained by non-maximum suppression (NMS), which removes neighboring weaker peaks.",
    "range_doppler_snr_db": "Local signal/background ratio in dB, not calibrated measurement uncertainty. Cells without a valid saved ratio are blank.",
    "range_doppler_snr_valid_mask": "1: the saved signal/background ratio is valid. This can include every eligible cell, even where there is no detection. 0: unavailable; it is not a zero-dB measurement.",
    "range_doppler_cluster_labels": "Cells with the same integer belong to one cluster. Blank cells are unclustered; an ID is a label, not signal strength.",
}


def stage_image(bundle: DetectionStageBundle, kind: str) -> tuple[np.ndarray | None, str]:
    """Prepare a display copy; never alter saved arrays or recreate a detector stage."""
    product = bundle.product(kind)
    if not product.available:
        return None, "Not recorded"
    values = np.asarray(product.values[0], dtype=float).copy()
    if kind.endswith("_power"):
        if kind == "range_doppler_cfar_threshold_power":
            eligible = bundle.product("range_doppler_cfar_eligible_mask")
            if not eligible.available:
                return None, "Eligibility not recorded"
            values[~eligible.values[0].astype(bool)] = np.nan
        return 10 * np.log10(np.maximum(values, 1e-30)), "Power (dB)"
    if kind == "range_doppler_snr_db":
        valid = bundle.product("range_doppler_snr_valid_mask")
        if not valid.available:
            return None, "Validity mask not recorded"
        values[~np.asarray(valid.values[0], dtype=bool)] = np.nan
        return values, "Local ratio (dB)"
    if kind == "range_doppler_cluster_labels":
        values[values < 0] = np.nan
        return values, "Cluster ID"
    return values, "Decision (0 / 1)" if kind.endswith("_mask") else "Training cells"


def stage_unavailable_text(bundle: DetectionStageBundle | None, reason: str) -> str:
    if reason == "detector_stages_not_recorded" or (
        bundle is not None
        and bundle.products
        and all(p.reason == "descriptor_absent" for p in bundle.products)
    ):
        return (
            "Detector-stage details were not recorded. Saved spectra and detections remain "
            "available. Future acquisitions can enable persist_detection_stage_artifacts; "
            "reprocessing to a separate output can also save stages."
        )
    if bundle is None and reason == "detection_stage_bundle unavailable":
        return "Detector-stage details are not loaded. Detector details loads them when available."
    if reason in {"detector_stage_values_not_loaded", "detector_stage_projection_unavailable"}:
        return "Detector-stage products are declared, but their saved values could not be read."
    if bundle is not None and bundle.disposition.value == "invalid":
        return "Recorded detector-stage data are inconsistent. Expand details for the source error."
    return "Detector-stage details unavailable: " + reason.replace("_", " ")


def cell_explanation(cell: CellInspection) -> str:
    """Explain only decisions present in the saved products, without rerunning CFAR."""
    if cell.cfar_eligible is False:
        outcome = "Excluded from CFAR testing by the recorded eligibility mask."
    elif cell.cfar_passed is False:
        outcome = "Rejected by the recorded CFAR decision."
    elif cell.nms_retained is False:
        outcome = "Not retained by peak suppression."
    elif (
        cell.detection_link is not None
        and cell.detection_link.range_bin == cell.selection.range_bin
        and cell.detection_link.doppler_bin == cell.selection.doppler_bin
    ):
        outcome = "Retained as a published detection."
    elif cell.cluster_id is not None:
        outcome = (
            f"Assigned to cluster {cell.cluster_id}; this cell is not a published representative."
        )
    else:
        outcome = "No published detection is linked to this cell; reason not retained."
    margin = threshold_margin(
        cell.detection_input_power, cell.cfar_threshold_power, cell.cfar_eligible, cell.cfar_passed
    )
    margin_value = f"{margin.value_db:+.3g} dB" if margin.value_db is not None else "unavailable"
    outcome += f" Threshold margin: {margin_value} ({margin.note})."
    values = []
    for label, value in (
        ("input power", cell.detection_input_power),
        ("background", cell.cfar_noise_power),
        ("threshold", cell.cfar_threshold_power),
    ):
        if value is not None:
            values.append(f"{label} {value:.4g}")
    if cell.snr_db is not None:
        values.append(f"local signal/background ratio {cell.snr_db:.2f} dB")
    return outcome + ("\n" + " · ".join(values) if values else "")
