"""Deterministic, persisted-data-only exports for detector-stage inspection.

The pure builders in this module serialize a :class:`DetectionStageBundle`
without rerunning CFAR, NMS, clustering, or any other sensing computation.
The file publisher builds every payload first and atomically replaces each
fixed-name destination through a sibling temporary file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .stage_inspector import (
    DETECTION_STAGE_KINDS,
    CellInspection,
    CellSelection,
    DetectionLink,
    DetectionStageBundle,
    DetectionStageProduct,
    LineageTrace,
)

STAGE_EXPORT_VERSION = 1
_DEFAULT_STEM = "detection_stages"
_SAFE_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

_DETECTION_COLUMNS = (
    "acquisition_id",
    "processing_run_id",
    "pair_id",
    "cluster_id",
    "doppler_bin",
    "range_bin",
    "doppler_hz",
    "range_m",
)

_CELL_COLUMNS = (
    "clock_id",
    "acquisition_id",
    "processing_run_id",
    "processing_recipe_id",
    "pair_id",
    "reference_time_s",
    "selection_cluster_id",
    "doppler_bin",
    "range_bin",
    "doppler_hz",
    "range_m",
    "detection_input_power",
    "cfar_noise_power",
    "cfar_threshold_power",
    "cfar_training_cell_count",
    "cfar_eligible",
    "cfar_passed",
    "nms_retained",
    "snr_db",
    "snr_valid",
    "cell_cluster_id",
    "eligibility_disposition",
    "cfar_disposition",
    "nms_disposition",
    "cluster_disposition",
    "detection_disposition",
    "linked_cluster_id",
    "linked_doppler_bin",
    "linked_range_bin",
    "inspection_disposition",
    "reason",
)


@dataclass(frozen=True, slots=True)
class StageExportArtifact:
    """One atomically published export and its content identity."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StageExportResult:
    """Paths and SHA-256 identities for a complete detector-stage export."""

    manifest: StageExportArtifact
    detections_csv: StageExportArtifact
    cell_csv: StageExportArtifact
    diagnostic_png: StageExportArtifact

    @property
    def artifacts(self) -> tuple[StageExportArtifact, ...]:
        """Return artifacts in their stable publication order."""

        return (
            self.manifest,
            self.detections_csv,
            self.cell_csv,
            self.diagnostic_png,
        )


def _json_safe(value: Any) -> Any:
    """Return recursively sorted JSON data with nonfinite numbers as null."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key]) for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Stage export cannot serialize {type(value).__name__}")


def _normalize_stage_kinds(
    bundle: DetectionStageBundle,
    stage_kinds: Sequence[str] | None,
) -> tuple[str, ...]:
    if isinstance(stage_kinds, str):
        raise TypeError("Diagnostic stage kinds must be a sequence, not one string")
    raw_kinds = DETECTION_STAGE_KINDS if stage_kinds is None else tuple(stage_kinds)
    if any(not isinstance(kind, str) or not kind for kind in raw_kinds):
        raise ValueError("Diagnostic stage kinds must be non-empty strings")
    requested = set(raw_kinds)
    known = {product.kind for product in bundle.products}
    unknown = sorted(requested - known)
    if unknown:
        raise KeyError(f"Unknown diagnostic stage kinds: {unknown!r}")
    return tuple(kind for kind in DETECTION_STAGE_KINDS if kind in requested)


def _lineage_manifest(trace: LineageTrace | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "nodes": [
            {
                "data_loaded": node.data_loaded,
                "descriptor_available": node.descriptor_available,
                "kind": node.kind,
                "product_id": node.product_id,
                "source_product_ids": list(node.source_product_ids),
            }
            for node in trace.nodes
        ],
        "root_product_id": trace.root_product_id,
        "unresolved_source_product_ids": list(trace.unresolved_source_product_ids),
    }


def _product_manifest(product: DetectionStageProduct) -> dict[str, Any]:
    return {
        "available": product.available,
        "data_key": product.data_key,
        "disposition": product.disposition.value,
        "dtype": product.values.dtype.str if product.values is not None else None,
        "evidence_role": product.evidence_role,
        "kind": product.kind,
        "lineage": _lineage_manifest(product.lineage_trace),
        "product_id": product.product_id,
        "reason": product.reason,
        "representation": product.representation,
        "shape": list(product.values.shape) if product.values is not None else None,
    }


def _detection_manifest(link: DetectionLink) -> dict[str, Any]:
    return {
        "acquisition_id": link.acquisition_id,
        "cluster_id": link.cluster_id,
        "doppler_bin": link.doppler_bin,
        "pair_id": link.pair_id,
        "processing_run_id": link.processing_run_id,
        "range_bin": link.range_bin,
        "stable_key": list(link.stable_key),
    }


def _inspection_manifest(inspection: CellInspection) -> dict[str, Any]:
    link = inspection.detection_link
    return {
        "cell_cluster_id": inspection.cluster_id,
        "cfar_eligible": inspection.cfar_eligible,
        "cfar_noise_power": inspection.cfar_noise_power,
        "cfar_passed": inspection.cfar_passed,
        "cfar_disposition": inspection.cfar_disposition,
        "cfar_threshold_power": inspection.cfar_threshold_power,
        "cfar_training_cell_count": inspection.cfar_training_cell_count,
        "cluster_disposition": inspection.cluster_disposition,
        "detection_disposition": inspection.detection_disposition,
        "detection_input_power": inspection.detection_input_power,
        "detection_link": _detection_manifest(link) if link is not None else None,
        "disposition": inspection.disposition.value,
        "doppler_hz": inspection.doppler_hz,
        "eligibility_disposition": inspection.eligibility_disposition,
        "nms_disposition": inspection.nms_disposition,
        "nms_retained": inspection.nms_retained,
        "range_m": inspection.range_m,
        "reason": inspection.reason,
        "selection": {
            "cluster_id": inspection.selection.cluster_id,
            "doppler_bin": inspection.selection.doppler_bin,
            "pair_id": inspection.selection.pair_id,
            "range_bin": inspection.selection.range_bin,
        },
        "snr_db": inspection.snr_db,
        "snr_valid": inspection.snr_valid,
    }


def build_stage_export_manifest(
    bundle: DetectionStageBundle,
    selection: CellSelection | None = None,
    *,
    stage_kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the sorted JSON-safe manifest for one immutable bundle."""

    selected_kinds = _normalize_stage_kinds(bundle, stage_kinds)
    event = bundle.event
    time_span = bundle.time_span
    inspection = bundle.inspect(selection) if selection is not None else None
    manifest = {
        "axes": {
            "doppler_frequency": {
                "size": (
                    int(bundle.doppler_axis_hz.size) if bundle.doppler_axis_hz is not None else None
                ),
                "unit": "Hz",
                "values": (
                    bundle.doppler_axis_hz.tolist() if bundle.doppler_axis_hz is not None else None
                ),
            },
            "pair": {"pair_id": event.pair_id if event is not None else None},
            "range": {
                "size": (
                    int(bundle.range_axis_m.size) if bundle.range_axis_m is not None else None
                ),
                "unit": "m",
                "values": (
                    bundle.range_axis_m.tolist() if bundle.range_axis_m is not None else None
                ),
            },
        },
        "detections": [
            _detection_manifest(link)
            for link in sorted(bundle.detection_links, key=lambda item: item.stable_key)
        ],
        "diagnostic_stage_kinds": list(selected_kinds),
        "disposition": bundle.disposition.value,
        "event": (
            {
                "acquisition_id": event.acquisition_id,
                "clock_id": event.clock_id,
                "pair_id": event.pair_id,
                "processing_recipe_id": event.processing_recipe_id,
                "processing_run_id": event.processing_run_id,
                "reference_time_s": event.reference_time_s,
            }
            if event is not None
            else None
        ),
        "products": [
            _product_manifest(product)
            for product in sorted(
                bundle.products,
                key=lambda item: (item.kind, item.product_id or ""),
            )
        ],
        "reason": bundle.reason,
        "selected_cell": _inspection_manifest(inspection) if inspection is not None else None,
        "stage_export_version": STAGE_EXPORT_VERSION,
        "time_span": (
            {
                "clock_id": time_span.clock_id,
                "end_time_s": time_span.end_time_s,
                "reference_time_policy": time_span.reference_time_policy,
                "reference_time_s": time_span.reference_time_s,
                "start_time_s": time_span.start_time_s,
            }
            if time_span is not None
            else None
        ),
    }
    normalized = _json_safe(manifest)
    assert isinstance(normalized, dict)
    return normalized


def stage_manifest_bytes(
    bundle: DetectionStageBundle,
    selection: CellSelection | None = None,
    *,
    stage_kinds: Sequence[str] | None = None,
) -> bytes:
    """Return canonical UTF-8 JSON with no non-standard numeric tokens."""

    manifest = build_stage_export_manifest(bundle, selection, stage_kinds=stage_kinds)
    return (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return _csv_scalar(value.value)
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return format(parsed, ".17g") if math.isfinite(parsed) else ""
    return str(value)


def _csv_bytes(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("Stage export CSV row does not match its fixed column contract")
        writer.writerow([_csv_scalar(value) for value in row])
    return stream.getvalue().encode("utf-8")


def detections_csv_bytes(bundle: DetectionStageBundle) -> bytes:
    """Return detections in stable occurrence/cluster/bin order."""

    rows: list[tuple[Any, ...]] = []
    for link in sorted(bundle.detection_links, key=lambda item: item.stable_key):
        doppler_hz = None
        range_m = None
        if bundle.doppler_axis_hz is not None and link.doppler_bin < bundle.doppler_axis_hz.size:
            doppler_hz = bundle.doppler_axis_hz[link.doppler_bin]
        if bundle.range_axis_m is not None and link.range_bin < bundle.range_axis_m.size:
            range_m = bundle.range_axis_m[link.range_bin]
        rows.append(
            (
                link.acquisition_id,
                link.processing_run_id,
                link.pair_id,
                link.cluster_id,
                link.doppler_bin,
                link.range_bin,
                doppler_hz,
                range_m,
            )
        )
    return _csv_bytes(_DETECTION_COLUMNS, rows)


def _cell_row(bundle: DetectionStageBundle, inspection: CellInspection) -> tuple[Any, ...]:
    event = bundle.event
    link = inspection.detection_link
    return (
        event.clock_id if event is not None else None,
        event.acquisition_id if event is not None else None,
        event.processing_run_id if event is not None else None,
        event.processing_recipe_id if event is not None else None,
        event.pair_id if event is not None else inspection.selection.pair_id,
        event.reference_time_s if event is not None else None,
        inspection.selection.cluster_id,
        inspection.selection.doppler_bin,
        inspection.selection.range_bin,
        inspection.doppler_hz,
        inspection.range_m,
        inspection.detection_input_power,
        inspection.cfar_noise_power,
        inspection.cfar_threshold_power,
        inspection.cfar_training_cell_count,
        inspection.cfar_eligible,
        inspection.cfar_passed,
        inspection.nms_retained,
        inspection.snr_db,
        inspection.snr_valid,
        inspection.cluster_id,
        inspection.eligibility_disposition,
        inspection.cfar_disposition,
        inspection.nms_disposition,
        inspection.cluster_disposition,
        inspection.detection_disposition,
        link.cluster_id if link is not None else None,
        link.doppler_bin if link is not None else None,
        link.range_bin if link is not None else None,
        inspection.disposition.value,
        inspection.reason,
    )


def selected_cell_csv_bytes(
    bundle: DetectionStageBundle,
    selection: CellSelection | None = None,
) -> bytes:
    """Return a header-only CSV or one exact selected-cell row."""

    rows: list[tuple[Any, ...]] = []
    if selection is not None:
        rows.append(_cell_row(bundle, bundle.inspect(selection)))
    return _csv_bytes(_CELL_COLUMNS, rows)


def _stage_image_values(
    bundle: DetectionStageBundle,
    product: DetectionStageProduct,
) -> np.ma.MaskedArray[Any, Any]:
    assert product.values is not None
    values = np.asarray(product.values[0])
    invalid = ~np.isfinite(values.astype(np.float64, copy=False))
    if product.kind == "range_doppler_snr_db":
        validity = bundle.product("range_doppler_snr_valid_mask")
        if validity.available and validity.values is not None:
            invalid = invalid | ~np.asarray(validity.values[0], dtype=np.bool_)
        else:
            invalid = np.ones(values.shape, dtype=np.bool_)
    return np.ma.array(values, mask=invalid, copy=True)


def _plot_extent(axis: np.ndarray) -> tuple[float, float]:
    first = float(axis[0])
    last = float(axis[-1])
    if first != last:
        return first, last
    # A singleton coordinate has no persisted bin width.  Use a fixed visual
    # half-width while retaining and labeling the exact center coordinate.
    return first - 0.5, first + 0.5


def diagnostic_png_bytes(
    bundle: DetectionStageBundle,
    selection: CellSelection | None = None,
    *,
    stage_kinds: Sequence[str] | None = None,
) -> bytes:
    """Render fixed-layout diagnostics directly from persisted stage arrays."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    selected_kinds = _normalize_stage_kinds(bundle, stage_kinds)
    inspection = bundle.inspect(selection) if selection is not None else None
    panel_count = max(1, len(selected_kinds))
    columns = min(2, panel_count)
    rows = int(math.ceil(panel_count / columns))
    figure = Figure(figsize=(10.0, 3.4 * rows), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    flat_axes = tuple(axis for row_axes in axes for axis in row_axes)
    figure.subplots_adjust(
        left=0.08,
        right=0.94,
        bottom=0.09,
        top=0.91,
        hspace=0.38,
        wspace=0.32,
    )

    event = bundle.event
    event_label = (
        f"run {event.processing_run_id[:8]} · t={event.reference_time_s:.9g} s"
        if event is not None
        else bundle.disposition.value
    )
    figure.suptitle(f"Persisted detector stages · {event_label}", fontsize=11)

    if not selected_kinds:
        axis = flat_axes[0]
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"No detector stages selected\n{bundle.disposition.value}: {bundle.reason or ''}",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        for axis, kind in zip(flat_axes, selected_kinds, strict=False):
            product = bundle.product(kind)
            axis.set_title(kind, fontsize=9)
            if (
                not product.available
                or product.values is None
                or bundle.range_axis_m is None
                or bundle.doppler_axis_hz is None
                or bundle.range_axis_m.size == 0
                or bundle.doppler_axis_hz.size == 0
            ):
                axis.axis("off")
                axis.text(
                    0.5,
                    0.5,
                    f"{product.disposition.value}\n{product.reason or ''}",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue
            image_values = _stage_image_values(bundle, product)
            range_limits = _plot_extent(bundle.range_axis_m)
            doppler_limits = _plot_extent(bundle.doppler_axis_hz)
            image = axis.imshow(
                image_values,
                aspect="auto",
                cmap=("gray_r" if product.representation == "boolean_mask" else "viridis"),
                extent=(
                    range_limits[0],
                    range_limits[1],
                    doppler_limits[0],
                    doppler_limits[1],
                ),
                interpolation="nearest",
                origin="lower",
                vmin=0.0 if product.representation == "boolean_mask" else None,
                vmax=1.0 if product.representation == "boolean_mask" else None,
            )
            axis.set_xlim(*range_limits)
            axis.set_ylim(*doppler_limits)
            axis.set_xlabel("Range (m)")
            axis.set_ylabel("Doppler frequency (Hz)")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            if (
                inspection is not None
                and inspection.range_m is not None
                and inspection.doppler_hz is not None
            ):
                axis.plot(
                    inspection.range_m,
                    inspection.doppler_hz,
                    color="#ff2d2d",
                    marker="x",
                    markeredgewidth=1.6,
                    markersize=9,
                    linestyle="none",
                )
        for axis in flat_axes[len(selected_kinds) :]:
            axis.axis("off")

    stream = io.BytesIO()
    try:
        figure.savefig(
            stream,
            format="png",
            dpi=100,
            facecolor="white",
            edgecolor="white",
            metadata={"Software": "ORCHAV deterministic sensing stage export"},
        )
        return stream.getvalue()
    finally:
        figure.clear()


def _safe_stem(stem: str) -> str:
    if not isinstance(stem, str) or _SAFE_STEM_RE.fullmatch(stem) is None:
        raise ValueError(
            "Stage export stem must begin with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return stem


def _atomic_write(path: Path, payload: bytes) -> StageExportArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return StageExportArtifact(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def export_detection_stage_bundle(
    bundle: DetectionStageBundle,
    output_directory: str | Path,
    selection: CellSelection | None = None,
    *,
    stage_kinds: Sequence[str] | None = None,
    stem: str = _DEFAULT_STEM,
) -> StageExportResult:
    """Build and atomically publish deterministic detector-stage artifacts."""

    normalized_stem = _safe_stem(stem)
    selected_kinds = _normalize_stage_kinds(bundle, stage_kinds)
    payloads = (
        (
            "manifest",
            f"{normalized_stem}_manifest.json",
            stage_manifest_bytes(bundle, selection, stage_kinds=selected_kinds),
        ),
        ("detections_csv", f"{normalized_stem}_detections.csv", detections_csv_bytes(bundle)),
        ("cell_csv", f"{normalized_stem}_cell.csv", selected_cell_csv_bytes(bundle, selection)),
        (
            "diagnostic_png",
            f"{normalized_stem}_diagnostic.png",
            diagnostic_png_bytes(bundle, selection, stage_kinds=selected_kinds),
        ),
    )
    root = Path(output_directory).expanduser().resolve()
    published = {
        name: _atomic_write(root / filename, payload) for name, filename, payload in payloads
    }
    return StageExportResult(
        manifest=published["manifest"],
        detections_csv=published["detections_csv"],
        cell_csv=published["cell_csv"],
        diagnostic_png=published["diagnostic_png"],
    )


__all__ = [
    "STAGE_EXPORT_VERSION",
    "StageExportArtifact",
    "StageExportResult",
    "build_stage_export_manifest",
    "detections_csv_bytes",
    "diagnostic_png_bytes",
    "export_detection_stage_bundle",
    "selected_cell_csv_bytes",
    "stage_manifest_bytes",
]
