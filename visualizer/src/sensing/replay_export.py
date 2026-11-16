"""Deterministic, renderer-neutral exports for canonical sensing replay events.

The exporter accepts the immutable shared replay contract and has no Qt,
renderer, frame-provider, or service dependency.  Interactive and headless
callers therefore publish the same bytes.  A complete package is prepared in
a sibling staging directory and atomically renamed into place only after every
artifact has been generated and flushed successfully.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from shared.animated_target_workflow import (
    AnimatedTargetWorkflowError,
    _promote_directory_noreplace,
)
from shared.sensing.replay import ReplayABComparison, ReplayEvent

from .stage_export import (
    build_stage_export_manifest,
    detections_csv_bytes,
    diagnostic_png_bytes,
    selected_cell_csv_bytes,
)
from .stage_inspector import (
    CellSelection,
    DetectionStageBundle,
    InspectionDisposition,
)

REPLAY_EXPORT_VERSION = 2

_ABSOLUTE_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_LAYERS = ("estimate", "truth", "oracle")
_STAGE_ARTIFACT_NAMES = (
    "detection_stages_manifest.json",
    "detection_stages_detections.csv",
    "detection_stages_cell.csv",
    "detection_stages_diagnostic.png",
)


class ReplayExportError(RuntimeError):
    """Base error for a replay package that cannot be safely published."""


class ReplayExportBlockedError(ReplayExportError):
    """Raised when credibility preflight blocks export without an override."""


@dataclass(frozen=True, slots=True)
class ReplayExportArtifact:
    """One published artifact and its deterministic content identity."""

    name: str
    media_type: str
    sha256: str
    byte_size: int
    evidence_layer: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayExportResult:
    """Published replay package paths and identities for caller integration."""

    output_directory: Path
    event_manifest_id: str
    event_view_manifest_id: str
    comparison_id: str | None
    blocked_override_used: bool
    artifacts: tuple[ReplayExportArtifact, ...]


def _json_bytes(payload: Any) -> bytes:
    """Return canonical UTF-8 JSON with one stable trailing newline."""

    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_cell(payload: Any) -> str:
    """Return compact canonical JSON for one CSV cell."""

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _float_cell(value: float) -> str:
    """Format an IEEE-754 value losslessly and independently of locale."""

    return format(float(value), ".17g")


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    """Return RFC-style CSV with stable UTF-8 and LF line endings."""

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(tuple(header))
    writer.writerows(tuple(tuple(value for value in row) for row in rows))
    return stream.getvalue().encode("utf-8")


def _pulse_csv(event: ReplayEvent) -> bytes:
    pulses = sorted(
        event.pulses,
        key=lambda pulse: (
            pulse.sample_index,
            pulse.sample_time_s,
            pulse.source_reference_id,
        ),
    )
    rows = [
        (
            pulse.sample_index,
            _float_cell(pulse.sample_time_s),
            "true" if pulse.valid else "false",
            "true" if pulse.phase_coherent else "false",
            pulse.source_generation_id,
            "int" if isinstance(pulse.source_scene_frame_id, int) else "str",
            pulse.source_scene_frame_id,
            pulse.sample_offset,
            pulse.source_frame_set_id or "",
            pulse.solve_id or "",
            pulse.source_reference_id,
            _json_cell(pulse.record.to_dict()),
        )
        for pulse in pulses
    ]
    return _csv_bytes(
        (
            "sample_index",
            "sample_time_s",
            "valid",
            "phase_coherent",
            "source_generation_id",
            "source_scene_frame_id_type",
            "source_scene_frame_id",
            "sample_offset",
            "source_frame_set_id",
            "solve_id",
            "source_reference_id",
            "record_json",
        ),
        rows,
    )


def _product_csv(event: ReplayEvent, layer: str) -> bytes:
    products = sorted(
        (product for product in event.products if product.evidence_layer == layer),
        key=lambda product: (
            product.kind,
            product.variant,
            product.data_key,
            product.product_id,
        ),
    )
    rows = [
        (
            product.evidence_layer,
            product.kind,
            product.variant,
            product.data_key,
            product.product_id,
            product.representation,
            product.unit,
            product.scale_semantics,
            _json_cell([axis.to_dict() for axis in product.axes]),
            _json_cell(product.lineage.to_dict()),
            _json_cell(product.time_span.to_dict()),
        )
        for product in products
    ]
    return _csv_bytes(
        (
            "evidence_layer",
            "kind",
            "variant",
            "data_key",
            "product_id",
            "representation",
            "unit",
            "scale_semantics",
            "axes_json",
            "lineage_json",
            "time_span_json",
        ),
        rows,
    )


def _evidence_csv(event: ReplayEvent, layer: str) -> bytes:
    references = sorted(
        (reference for reference in event.evidence if reference.layer == layer),
        key=lambda reference: (
            reference.kind,
            reference.identity,
            reference.ordinal,
            reference.reference_id,
        ),
    )
    rows = [
        (
            reference.layer,
            reference.kind,
            reference.identity,
            reference.ordinal,
            reference.payload_hash,
            reference.reference_id,
        )
        for reference in references
    ]
    return _csv_bytes(
        (
            "evidence_layer",
            "kind",
            "identity",
            "ordinal",
            "payload_hash",
            "reference_id",
        ),
        rows,
    )


def _short(value: str) -> str:
    return value[:8]


def _summary_png(event: ReplayEvent, comparison: ReplayABComparison | None) -> bytes:
    """Render one compact timeline/identity figure through Agg only."""

    figure = Figure(figsize=(8.0, 3.0), dpi=120, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.08, 0.28, 0.88, 0.35))

    pulses = sorted(
        event.pulses,
        key=lambda pulse: (
            pulse.sample_index,
            pulse.sample_time_s,
            pulse.source_reference_id,
        ),
    )
    pulse_times = [pulse.sample_time_s for pulse in pulses]
    valid_times = [pulse.sample_time_s for pulse in pulses if pulse.valid]
    invalid_times = [pulse.sample_time_s for pulse in pulses if not pulse.valid]
    if pulse_times:
        axis.plot(
            [pulse_times[0], pulse_times[-1]],
            [0.0, 0.0],
            color="#1976d2",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
    if valid_times:
        axis.scatter(
            valid_times,
            [0.0] * len(valid_times),
            color="#2e7d32",
            edgecolors="#1b5e20",
            marker="o",
            s=32,
            zorder=3,
        )
    if invalid_times:
        axis.scatter(
            invalid_times,
            [0.0] * len(invalid_times),
            color="#c62828",
            marker="x",
            linewidths=1.8,
            s=42,
            zorder=4,
        )

    axis.axvline(
        event.key.reference_time_s,
        color="#ef6c00",
        linewidth=1.8,
        label="result reference",
        zorder=2,
    )
    axis.axvline(
        event.scene_binding.current_scene_time_s,
        color="#6a1b9a",
        linewidth=1.6,
        linestyle="--",
        label="current scene",
        zorder=2,
    )
    visible_times = pulse_times + [
        event.key.reference_time_s,
        event.scene_binding.current_scene_time_s,
    ]
    time_min = min(visible_times)
    time_max = max(visible_times)
    span_s = time_max - time_min
    padding_s = span_s * 0.06 if span_s > 0.0 else max(abs(time_min) * 1e-6, 0.5)
    axis.set_xlim(time_min - padding_s, time_max + padding_s)
    axis.set_ylim(-0.5, 0.5)
    axis.set_yticks([])
    axis.set_xlabel("Scenario-clock time (s)", fontfamily="DejaVu Sans", fontsize=8)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#999999")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        frameon=False,
        prop={"family": "DejaVu Sans", "size": 7},
    )

    key = event.key
    identity_line = (
        f"acq={_short(key.acquisition_id)}  pair={_short(key.pair_id)}  "
        f"run={_short(key.processing_run_id)}  recipe={_short(event.recipe.processing_recipe_id)}"
    )
    disposition_line = (
        f"scene={event.scene_binding.state}  stale_age={event.scene_binding.stale_age_s:.6g}s  "
        f"preflight={event.preflight.disposition}  pulses={len(event.pulses)}  "
        f"products={len(event.products)}  estimate/truth/oracle="
        f"{len(event.estimates)}/{len(event.truth)}/{len(event.oracle)}"
    )
    if comparison is not None:
        disposition_line += f"  A/B={_short(comparison.comparison_id)}"
    figure.text(
        0.5,
        0.91,
        "ORCHAV sensing replay",
        ha="center",
        va="center",
        color="#222222",
        fontfamily="DejaVu Sans",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.82,
        identity_line,
        ha="center",
        va="center",
        color="#333333",
        fontfamily="DejaVu Sans Mono",
        fontsize=8,
    )
    figure.text(
        0.5,
        0.10,
        disposition_line,
        ha="center",
        va="center",
        color="#444444",
        fontfamily="DejaVu Sans",
        fontsize=7,
    )

    output = io.BytesIO()
    canvas.print_png(
        output,
        metadata={
            "Software": "ORCHAV deterministic replay exporter",
            "Title": "ORCHAV sensing replay summary",
        },
        pil_kwargs={"compress_level": 9, "optimize": False},
    )
    figure.clear()
    return output.getvalue()


def _assert_no_absolute_paths(value: Any, *, path: str = "$") -> None:
    """Reject private absolute paths instead of modifying an exact manifest."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_absolute_paths(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_absolute_paths(child, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if (
        value.startswith("/")
        or value.startswith("\\\\")
        or value.lower().startswith("file://")
        or _ABSOLUTE_WINDOWS_PATH_RE.match(value) is not None
    ):
        raise ReplayExportError(f"replay export forbids absolute path at {path}")


def _validate_comparison(event: ReplayEvent, comparison: ReplayABComparison) -> None:
    """Reassert the same-source guarantees carried by an A/B comparison."""

    left = comparison.left_key
    right = comparison.right_key
    for field_name in ("clock_id", "acquisition_id", "pair_id"):
        if getattr(left, field_name) != getattr(right, field_name):
            raise ReplayExportError(f"A/B export requires the same {field_name}")
    if left.processing_run_id == right.processing_run_id:
        raise ReplayExportError("A/B export requires different processing runs")
    if event.key not in (left, right):
        raise ReplayExportError("A/B comparison does not include the exported replay event")
    if comparison.source_samples_id != event.source_samples_id:
        raise ReplayExportError("A/B export requires exactly the event's source samples")
    if not comparison.root_source_product_ids:
        raise ReplayExportError("A/B export requires root source-product identities")
    if comparison.root_source_product_ids != event.root_source_product_ids:
        raise ReplayExportError("A/B export requires exactly the event's root source products")


def _replay_stage_identity(event: ReplayEvent) -> tuple[str, str, str, str, str, float]:
    """Return the exact replay occurrence identity shared with detector stages."""

    return (
        event.key.clock_id,
        event.key.acquisition_id,
        event.key.processing_run_id,
        event.recipe.processing_recipe_id,
        event.key.pair_id,
        event.key.reference_time_s,
    )


def _validate_stage_inputs(
    event: ReplayEvent,
    stage_bundle: DetectionStageBundle | None,
    stage_selection: CellSelection | None,
    stage_kinds: Sequence[str] | None,
) -> None:
    """Fail before publication when optional detector-stage inputs are incoherent."""

    if stage_bundle is None:
        if stage_selection is not None or stage_kinds is not None:
            raise ReplayExportError("detector-stage selection requires stage_bundle")
        return
    if not isinstance(stage_bundle, DetectionStageBundle):
        raise TypeError("stage_bundle must be a DetectionStageBundle or None")
    if stage_selection is not None and not isinstance(stage_selection, CellSelection):
        raise TypeError("stage_selection must be a CellSelection or None")
    if stage_bundle.disposition is InspectionDisposition.INVALID:
        raise ReplayExportError(
            "detector-stage bundle is invalid: "
            + (stage_bundle.reason or "unknown contract failure")
        )
    stage_event = stage_bundle.event
    if stage_event is None:
        raise ReplayExportError("detector-stage bundle has no event identity")
    stage_identity = (
        stage_event.clock_id,
        stage_event.acquisition_id,
        stage_event.processing_run_id,
        stage_event.processing_recipe_id,
        stage_event.pair_id,
        stage_event.reference_time_s,
    )
    if stage_identity != _replay_stage_identity(event):
        fields = ("clock", "acquisition", "run", "recipe", "pair", "reference_time_s")
        mismatched = [
            field
            for field, replay_value, stage_value in zip(
                fields,
                _replay_stage_identity(event),
                stage_identity,
                strict=True,
            )
            if replay_value != stage_value
        ]
        raise ReplayExportError(
            "detector-stage bundle event identity does not match replay event: "
            + ", ".join(mismatched)
        )
    if (
        stage_selection is not None
        and stage_bundle.disposition is not InspectionDisposition.AVAILABLE
    ):
        raise ReplayExportError("detector-stage cell selection requires an available stage bundle")


def _stage_payloads(
    bundle: DetectionStageBundle,
    selection: CellSelection | None,
    stage_kinds: Sequence[str] | None,
) -> tuple[list[tuple[str, str, bytes, str | None]], dict[str, Any]]:
    """Build every detector-stage payload before the package staging transaction."""

    manifest = build_stage_export_manifest(
        bundle,
        selection,
        stage_kinds=stage_kinds,
    )
    _assert_no_absolute_paths(manifest, path="$.detection_stages")
    selected_kinds = manifest["diagnostic_stage_kinds"]
    assert isinstance(selected_kinds, list)
    payloads: list[tuple[str, str, bytes, str | None]] = [
        (
            _STAGE_ARTIFACT_NAMES[0],
            "application/json",
            _json_bytes(manifest),
            None,
        ),
        (
            _STAGE_ARTIFACT_NAMES[1],
            "text/csv",
            detections_csv_bytes(bundle),
            None,
        ),
        (
            _STAGE_ARTIFACT_NAMES[2],
            "text/csv",
            selected_cell_csv_bytes(bundle, selection),
            None,
        ),
        (
            _STAGE_ARTIFACT_NAMES[3],
            "image/png",
            diagnostic_png_bytes(bundle, selection, stage_kinds=tuple(selected_kinds)),
            None,
        ),
    ]
    selection_manifest = manifest.get("selected_cell")
    selected_cell = None
    if isinstance(selection_manifest, Mapping):
        selected_cell = selection_manifest.get("selection")
    metadata = {
        "bundle_disposition": bundle.disposition.value,
        "included": True,
        "selected_cell": selected_cell,
        "selected_stage_kinds": selected_kinds,
    }
    return payloads, metadata


def _artifact(
    name: str,
    media_type: str,
    payload: bytes,
    *,
    evidence_layer: str | None = None,
) -> ReplayExportArtifact:
    return ReplayExportArtifact(
        name=name,
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        evidence_layer=evidence_layer,
    )


def _artifact_manifest(artifact: ReplayExportArtifact) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "byte_size": artifact.byte_size,
        "media_type": artifact.media_type,
        "name": artifact.name,
        "sha256": artifact.sha256,
    }
    if artifact.evidence_layer is not None:
        payload["evidence_layer"] = artifact.evidence_layer
    return payload


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def export_replay_session(
    event: ReplayEvent,
    output_directory: str | os.PathLike[str],
    *,
    comparison: ReplayABComparison | None = None,
    allow_blocked: bool = False,
    stage_bundle: DetectionStageBundle | None = None,
    stage_selection: CellSelection | None = None,
    stage_kinds: Sequence[str] | None = None,
) -> ReplayExportResult:
    """Atomically publish one deterministic replay/session export package.

    The destination must not already exist.  Blocked credibility preflight is
    refused before any directory is created unless ``allow_blocked=True`` is
    explicit.  Optional persisted detector stages are validated against the
    replay occurrence, hashed, and published in the same directory rename as
    every replay artifact.  The override and stage inclusion are recorded in
    ``export_manifest.json``.
    """

    if not isinstance(event, ReplayEvent):
        raise TypeError("event must be a shared.sensing.replay.ReplayEvent")
    if not isinstance(allow_blocked, bool):
        raise TypeError("allow_blocked must be a bool")
    if comparison is not None and not isinstance(comparison, ReplayABComparison):
        raise TypeError("comparison must be a ReplayABComparison or None")

    blocked = event.preflight.disposition == "blocked"
    if blocked and not allow_blocked:
        raise ReplayExportBlockedError(
            "replay credibility preflight is blocked; pass allow_blocked=True to export explicitly"
        )
    if comparison is not None:
        _validate_comparison(event, comparison)
    _validate_stage_inputs(event, stage_bundle, stage_selection, stage_kinds)

    event_manifest = event.to_manifest()
    comparison_manifest = comparison.to_manifest() if comparison is not None else None
    _assert_no_absolute_paths(event_manifest, path="$.replay")
    if comparison_manifest is not None:
        _assert_no_absolute_paths(comparison_manifest, path="$.comparison")

    payloads: list[tuple[str, str, bytes, str | None]] = [
        ("replay_manifest.json", "application/json", _json_bytes(event_manifest), None),
        ("pulse_timeline.csv", "text/csv", _pulse_csv(event), None),
    ]
    for layer in _LAYERS:
        payloads.append(
            (
                f"products_{layer}.csv",
                "text/csv",
                _product_csv(event, layer),
                layer,
            )
        )
    for layer in _LAYERS:
        payloads.append(
            (
                f"evidence_{layer}.csv",
                "text/csv",
                _evidence_csv(event, layer),
                layer,
            )
        )
    payloads.append(
        (
            "replay_summary.png",
            "image/png",
            _summary_png(event, comparison),
            None,
        )
    )
    if comparison_manifest is not None:
        payloads.append(
            (
                "ab_comparison.json",
                "application/json",
                _json_bytes(comparison_manifest),
                None,
            )
        )

    stage_metadata: dict[str, Any] = {"included": False}
    if stage_bundle is not None:
        detector_payloads, stage_metadata = _stage_payloads(
            stage_bundle,
            stage_selection,
            stage_kinds,
        )
        payloads.extend(detector_payloads)

    content_artifacts = tuple(
        _artifact(name, media_type, payload, evidence_layer=layer)
        for name, media_type, payload, layer in payloads
    )
    export_manifest = {
        "artifacts": [_artifact_manifest(artifact) for artifact in content_artifacts],
        "blocked_override_used": bool(blocked and allow_blocked),
        "comparison_id": comparison.comparison_id if comparison is not None else None,
        "detection_stages": stage_metadata,
        "event_manifest_id": event.manifest_id,
        "event_view_manifest_id": event.view_manifest_id,
        "preflight_disposition": event.preflight.disposition,
        "replay_export_version": REPLAY_EXPORT_VERSION,
    }
    export_manifest_bytes = _json_bytes(export_manifest)
    export_artifact = _artifact(
        "export_manifest.json",
        "application/json",
        export_manifest_bytes,
    )
    all_payloads = payloads + [
        ("export_manifest.json", "application/json", export_manifest_bytes, None)
    ]
    all_artifacts = content_artifacts + (export_artifact,)

    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"replay export destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        for name, _media_type, payload, _layer in all_payloads:
            _write_fsynced(staging / name, payload)
        try:
            _promote_directory_noreplace(staging, destination)
        except AnimatedTargetWorkflowError as exc:
            raise ReplayExportError(f"replay publication failed: {exc}") from exc
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return ReplayExportResult(
        output_directory=destination,
        event_manifest_id=event.manifest_id,
        event_view_manifest_id=event.view_manifest_id,
        comparison_id=comparison.comparison_id if comparison is not None else None,
        blocked_override_used=bool(blocked and allow_blocked),
        artifacts=all_artifacts,
    )


__all__ = [
    "REPLAY_EXPORT_VERSION",
    "ReplayExportArtifact",
    "ReplayExportBlockedError",
    "ReplayExportError",
    "ReplayExportResult",
    "export_replay_session",
]
