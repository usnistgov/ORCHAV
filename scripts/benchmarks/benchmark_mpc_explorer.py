#!/usr/bin/env python3
"""Benchmark the MPC Explorer's CPU-side indexing and selection path.

This is a synthetic data/model benchmark, not a renderer FPS benchmark. It
builds one valid canonical frame per requested path count, then measures the
same catalog, NumPy query, Qt model, and transient selection components used by
the Explorer. Generated reports belong in a scratch directory, not in the
repository.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402

from visualizer.src.metrics.mpc_canon import CanonicalStepData  # noqa: E402
from visualizer.src.metrics.mpc_path_catalog import (  # noqa: E402
    MpcPathCatalog,
    MpcPathScope,
)
from visualizer.src.metrics.mpc_path_query import (  # noqa: E402
    MpcGrouping,
    MpcPathQueryEngine,
    MpcQuerySpec,
    MpcSortPreset,
    query_spec_for_preset,
)
from visualizer.src.model.mpc_explorer_model import (  # noqa: E402
    MpcExplorerTableModel,
)
from visualizer.src.renderers.protocol import (  # noqa: E402
    RendererCapabilities,
)
from visualizer.src.services.mpc_selection_service import (  # noqa: E402
    MpcSelectionService,
)

_T = TypeVar("_T")
_MIB = 1024.0 * 1024.0
DEFAULT_PATH_COUNTS = (100_000, 1_000_000, 2_000_000)


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    """Canonical arrays and presentation masks for one synthetic frame."""

    canonical: CanonicalStepData
    filtered_path_mask: np.ndarray
    rendered_segment_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class Measurement:
    """Repeated latency samples plus one warmed peak-allocation sample."""

    repeats: int
    min_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    peak_additional_mib: float


class _SelectionRenderer:
    """Small renderer port used to time the production selection service."""

    capabilities = RendererCapabilities(mpc_path_inspection=True)

    def __init__(self) -> None:
        self.callback: Any = None
        self.snapshot: Any = None
        self.overlay_updates = 0
        self.flow_updates = 0
        self.bulk_mpc_revision = 0

    def set_mpc_path_selection_callback(self, callback: Any) -> None:
        self.callback = callback

    def set_mpc_path_inspection(self, snapshot: Any) -> bool:
        self.snapshot = snapshot
        self.overlay_updates += 1
        return True

    def clear_mpc_path_inspection(self) -> None:
        self.snapshot = None

    def update_mpc_path_flow(self, phase: float) -> bool:
        if not 0.0 <= float(phase) < 1.0:
            raise ValueError("flow phase must be normalized")
        self.flow_updates += 1
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MPC Explorer catalog, vectorized query/sort, Qt model, "
            "and selected-path latency. This does not measure renderer FPS."
        )
    )
    parser.add_argument(
        "--path-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_PATH_COUNTS),
        help="Synthetic path populations (default: 100000 1000000 2000000).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Untraced latency repetitions per operation (default: 3).",
    )
    parser.add_argument(
        "--visible-rows",
        type=int,
        default=200,
        help="Rows sampled for the visible-cell formatting probe (default: 200).",
    )
    parser.add_argument(
        "--closed-iterations",
        type=int,
        default=1_000_000,
        help="Accepted-frame callback-guard iterations (default: 1000000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional scratch JSON path. The report is always printed to stdout.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or unsafe benchmark sizes."""
    if not args.path_counts or any(count <= 0 for count in args.path_counts):
        raise SystemExit("--path-counts must contain positive integers")
    max_paths = (np.iinfo(np.int32).max - 4) // 2
    if any(count > max_paths for count in args.path_counts):
        raise SystemExit(f"path counts must not exceed {max_paths:,}")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.visible_rows <= 0:
        raise SystemExit("--visible-rows must be positive")
    if args.closed_iterations <= 0:
        raise SystemExit("--closed-iterations must be positive")


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without another dependency."""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measure(
    operation: Callable[[], _T],
    *,
    repeats: int,
) -> tuple[_T, Measurement]:
    """Measure untraced latency and a separate warmed additional-memory peak."""
    durations_ms: list[float] = []
    previous_result: Any = None
    for _ in range(int(repeats)):
        gc.collect()
        started = time.perf_counter()
        result = operation()
        durations_ms.append((time.perf_counter() - started) * 1_000.0)
        previous_result = result
    del previous_result

    gc.collect()
    tracemalloc.start()
    baseline_bytes = tracemalloc.get_traced_memory()[0]
    traced_result = operation()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return traced_result, Measurement(
        repeats=len(durations_ms),
        min_ms=min(durations_ms),
        median_ms=statistics.median(durations_ms),
        p95_ms=_percentile(durations_ms, 0.95),
        max_ms=max(durations_ms),
        peak_additional_mib=max(0, peak_bytes - baseline_bytes) / _MIB,
    )


def _measure_once(operation: Callable[[], _T]) -> tuple[_T, Measurement]:
    """Measure a state-changing operation once, including its peak allocation."""
    gc.collect()
    tracemalloc.start()
    baseline_bytes = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    result = operation()
    duration_ms = (time.perf_counter() - started) * 1_000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, Measurement(
        repeats=1,
        min_ms=duration_ms,
        median_ms=duration_ms,
        p95_ms=duration_ms,
        max_ms=duration_ms,
        peak_additional_mib=max(0, peak_bytes - baseline_bytes) / _MIB,
    )


def _measurement_dict(measurement: Measurement) -> dict[str, int | float]:
    """Round one measurement for a stable JSON report."""
    values = asdict(measurement)
    return {
        "repeats": int(values["repeats"]),
        "min_ms": round(float(values["min_ms"]), 4),
        "median_ms": round(float(values["median_ms"]), 4),
        "p95_ms": round(float(values["p95_ms"]), 4),
        "max_ms": round(float(values["max_ms"]), 4),
        "peak_additional_mib": round(float(values["peak_additional_mib"]), 4),
    }


def make_synthetic_frame(path_count: int) -> SyntheticFrame:
    """Build a valid deterministic frame with one four-bounce inspection path."""
    count = int(path_count)
    path_ids = np.arange(count, dtype=np.int32)
    starts = path_ids * np.int32(2)
    point_count = count * 2 + 4
    segment_count = count + 4

    path_tx = np.remainder(path_ids, 64).astype(np.int16, copy=False)
    path_rx = np.remainder(path_ids // 64, 64).astype(np.int16, copy=False)
    path_delays = np.remainder(path_ids * np.int32(37), 10_000).astype(np.float32) * 0.01
    path_losses = np.remainder(path_ids * np.int32(17), 12_000).astype(np.float32) * 0.01 + 30.0
    path_orders = np.zeros((count,), dtype=np.uint8)
    path_orders[-1] = 4

    points = np.zeros((point_count, 3), dtype=np.float32)
    if count > 1:
        points[starts[:-1] + 1, 0] = 100.0
    last_start = int(starts[-1])
    points[last_start : last_start + 6] = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (18.0, 8.0, 1.0),
            (39.0, -5.0, 3.0),
            (61.0, 10.0, 2.0),
            (82.0, -3.0, 1.0),
            (100.0, 0.0, 0.0),
        ),
        dtype=np.float32,
    )

    lines = np.empty((segment_count, 2), dtype=np.int32)
    if count > 1:
        lines[: count - 1, 0] = starts[:-1]
        lines[: count - 1, 1] = starts[:-1] + 1
    lines[count - 1 :] = np.column_stack(
        (
            np.arange(last_start, last_start + 5, dtype=np.int32),
            np.arange(last_start + 1, last_start + 6, dtype=np.int32),
        )
    )

    point_path_ids = np.empty((point_count,), dtype=np.int32)
    point_tx = np.empty((point_count,), dtype=np.int16)
    point_rx = np.empty((point_count,), dtype=np.int16)
    point_delay = np.empty((point_count,), dtype=np.float32)
    point_loss = np.empty((point_count,), dtype=np.float32)
    if count > 1:
        regular_slice = slice(0, last_start)
        point_path_ids[regular_slice] = np.repeat(path_ids[:-1], 2)
        point_tx[regular_slice] = np.repeat(path_tx[:-1], 2)
        point_rx[regular_slice] = np.repeat(path_rx[:-1], 2)
        point_delay[regular_slice] = np.repeat(path_delays[:-1], 2)
        point_loss[regular_slice] = np.repeat(path_losses[:-1], 2)
    point_path_ids[last_start:] = path_ids[-1]
    point_tx[last_start:] = path_tx[-1]
    point_rx[last_start:] = path_rx[-1]
    point_delay[last_start:] = path_delays[-1]
    point_loss[last_start:] = path_losses[-1]

    point_order = np.zeros((point_count,), dtype=np.uint8)
    point_order[last_start:] = 4
    point_interactions = np.zeros((point_count,), dtype=np.uint8)
    point_interactions[last_start + 1 : last_start + 5] = np.asarray(
        (1, 2, 4, 8),
        dtype=np.uint8,
    )
    point_materials = np.zeros((point_count,), dtype=np.int16)
    point_materials[last_start + 1 : last_start + 5] = np.asarray(
        (1, 2, 3, 4),
        dtype=np.int16,
    )

    segment_path_ids = np.empty((segment_count,), dtype=np.int32)
    if count > 1:
        segment_path_ids[: count - 1] = path_ids[:-1]
    segment_path_ids[count - 1 :] = path_ids[-1]
    segment_interactions = np.zeros((segment_count,), dtype=np.uint8)
    segment_interactions[count - 1 :] = np.asarray((1, 1, 2, 4, 8), dtype=np.uint8)

    canonical = CanonicalStepData(
        points=points,
        lines=lines,
        order=point_order,
        itype=point_interactions,
        delay=point_delay,
        loss=point_loss,
        tx_id=point_tx,
        rx_id=point_rx,
        path_id=point_path_ids,
        path_start_indices=starts,
        path_orders=path_orders,
        path_delays=path_delays,
        path_losses=path_losses,
        path_tx=path_tx,
        path_rx=path_rx,
        path_delay_is_estimated=np.zeros((count,), dtype=bool),
        path_loss_is_estimated=np.zeros((count,), dtype=bool),
        segment_path_id=segment_path_ids,
        segment_itype=segment_interactions,
        material_ids=point_materials,
        material_id_to_name={
            0: "",
            1: "Concrete",
            2: "Glass",
            3: "Metal",
            4: "Wood",
        },
        delay_min=float(np.min(path_delays)),
        delay_max=float(np.max(path_delays)),
        loss_min=float(np.min(path_losses)),
        loss_max=float(np.max(path_losses)),
    )
    filtered_path_mask = np.remainder(path_ids, 3) != 0
    rendered_segment_mask = filtered_path_mask[segment_path_ids] & (
        np.remainder(segment_path_ids, 5) != 0
    )
    return SyntheticFrame(
        canonical=canonical,
        filtered_path_mask=filtered_path_mask,
        rendered_segment_mask=rendered_segment_mask,
    )


def _canonical_array_bytes(canonical: CanonicalStepData) -> int:
    """Return unique ndarray bytes retained by the synthetic canonical frame."""
    seen: set[int] = set()
    total = 0
    for field_info in fields(canonical):
        value = getattr(canonical, field_info.name)
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _make_catalog(frame: SyntheticFrame) -> MpcPathCatalog:
    """Construct the production borrowed-array catalog."""
    return MpcPathCatalog(
        frame.canonical,
        filtered_path_mask=frame.filtered_path_mask,
        rendered_segment_mask=frame.rendered_segment_mask,
    )


def _prepare_model(
    catalog: MpcPathCatalog,
    result: Any,
    *,
    generation: int,
) -> MpcExplorerTableModel:
    """Create and populate one production table model."""
    model = MpcExplorerTableModel(fetch_batch_size=50_000)
    model.begin_generation(catalog, generation=generation)
    if not model.apply_query_result(result):
        raise RuntimeError("table model rejected current benchmark result")
    return model


def _format_visible_cells(model: MpcExplorerTableModel, row_count: int) -> int:
    """Format a viewport-sized row sample and return a deterministic checksum."""
    available = model.rowCount()
    sample_count = min(int(row_count), available)
    if sample_count <= 0:
        return 0
    rows = np.linspace(0, available - 1, sample_count, dtype=np.int32)
    checksum = 0
    for row in rows:
        row_index = int(row)
        for column in range(model.columnCount()):
            value = model.data(
                model.index(row_index, column),
                Qt.ItemDataRole.DisplayRole,
            )
            checksum += len(str(value))
    return checksum


def _closed_guard(iterations: int) -> dict[str, int | float]:
    """Measure the sole closed-state accepted-frame branch."""
    callback: Any = None
    callback_invocations = 0
    started = time.perf_counter_ns()
    for _ in range(int(iterations)):
        if callback is not None:
            callback_invocations += 1
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "iterations": int(iterations),
        "elapsed_ms": round(elapsed_ns / 1_000_000.0, 4),
        "ns_per_guard": round(elapsed_ns / float(iterations), 4),
        "callback_invocations": callback_invocations,
        "catalogs_constructed": 0,
        "query_requests": 0,
        "models_constructed": 0,
        "selection_timers_started": 0,
    }


def benchmark_population(
    path_count: int,
    *,
    repeats: int,
    visible_rows: int,
) -> dict[str, Any]:
    """Benchmark one synthetic path population."""
    print(f"building synthetic frame for {path_count:,} paths", file=sys.stderr)
    frame = make_synthetic_frame(path_count)
    fixture_mib = (
        _canonical_array_bytes(frame.canonical)
        + frame.filtered_path_mask.nbytes
        + frame.rendered_segment_mask.nbytes
    ) / _MIB

    catalog, catalog_measurement = _measure(
        lambda: _make_catalog(frame),
        repeats=repeats,
    )
    engine = MpcPathQueryEngine(catalog)
    filter_spec = MpcQuerySpec(
        scope=MpcPathScope.FILTERED,
        grouping=MpcGrouping.NONE,
        path_loss_max_db=90.0,
        delay_min_ns=10.0,
        include_status=False,
    )

    # These private phase methods are called deliberately: execute() combines
    # them, while this diagnostic needs distinct vectorized-mask and
    # permutation timings.
    filter_result, filter_measurement = _measure(
        lambda: engine._filtered_path_ids(filter_spec),  # noqa: SLF001
        repeats=repeats,
    )
    filtered_ids, scope_count = filter_result

    sort_spec = query_spec_for_preset(
        MpcSortPreset.TX_RX_STRONGEST,
        scope=MpcPathScope.FILTERED,
        include_status=False,
    )
    sorted_ids, sort_measurement = _measure(
        lambda: engine._sort_path_ids(filtered_ids, sort_spec),  # noqa: SLF001
        repeats=repeats,
    )
    if sorted_ids.dtype != np.int32:
        raise RuntimeError("query engine returned a non-int32 permutation")

    rendered_mask, rendered_scope_measurement = _measure_once(
        lambda: catalog.scope_mask(MpcPathScope.RENDERED)
    )
    full_spec = query_spec_for_preset(
        MpcSortPreset.TX_RX_STRONGEST,
        scope=MpcPathScope.ALL,
        include_status=True,
    )
    query_result, full_query_measurement = _measure(
        lambda: engine.execute(full_spec, generation=1),
        repeats=repeats,
    )

    model, model_measurement = _measure(
        lambda: _prepare_model(catalog, query_result, generation=1),
        repeats=repeats,
    )
    _, fetch_measurement = _measure_once(
        lambda: model.fetchMore() if model.canFetchMore() else None
    )
    visible_checksum, visible_measurement = _measure(
        lambda: _format_visible_cells(model, visible_rows),
        repeats=repeats,
    )

    renderer = _SelectionRenderer()
    # A running visualizer already owns the interaction palette used by its
    # bulk MPC layer. Supplying it here keeps the measurement focused on path
    # extraction/snapshot construction rather than first-import startup cost.
    palette = np.tile(
        np.asarray((0.2, 0.8, 1.0, 1.0), dtype=np.float32),
        (9, 1),
    )
    selection_service = MpcSelectionService(
        SimpleNamespace(
            renderer=renderer,
            mpc_core=SimpleNamespace(_type_palette=palette),
        )
    )
    render_packet = object()
    selection_service.set_presented_frame(
        ("synthetic", int(path_count)),
        catalog,
        render_packet,
    )
    selected_path_id = int(path_count) - 1

    def select_path() -> int:
        selection_service.clear_selection(reason="benchmark repeat")
        revision_before = renderer.bulk_mpc_revision
        selection_service.select_path(
            ("synthetic", int(path_count)),
            selected_path_id,
            origin="table",
        )
        if renderer.snapshot is None:
            raise RuntimeError("selection service did not create an overlay snapshot")
        if renderer.bulk_mpc_revision != revision_before:
            raise RuntimeError("selection modified the bulk MPC revision")
        return int(renderer.snapshot.points.shape[0])

    selected_point_count, selection_measurement = _measure(
        select_path,
        repeats=repeats,
    )

    flow_tick_count = 1_000

    def advance_flow() -> int:
        for _ in range(flow_tick_count):
            selection_service._on_flow_timeout()  # noqa: SLF001
        return renderer.flow_updates

    _, flow_measurement = _measure(advance_flow, repeats=repeats)
    selection_service.shutdown()

    result = {
        "path_count": int(path_count),
        "segment_count": int(catalog.segment_count),
        "synthetic_frame_mib": round(fixture_mib, 4),
        "catalog_construction": _measurement_dict(catalog_measurement),
        "vectorized_filter": {
            **_measurement_dict(filter_measurement),
            "scope_path_count": int(scope_count),
            "matching_path_count": int(filtered_ids.size),
        },
        "compound_sort": {
            **_measurement_dict(sort_measurement),
            "sorted_path_count": int(sorted_ids.size),
        },
        "rendered_scope_materialization": {
            **_measurement_dict(rendered_scope_measurement),
            "rendered_path_count": int(np.count_nonzero(rendered_mask)),
        },
        "full_default_query_and_sort": {
            **_measurement_dict(full_query_measurement),
            "matching_path_count": int(query_result.matching_path_count),
            "engine_elapsed_ms_last": round(float(query_result.elapsed_ms), 4),
        },
        "initial_table_population": {
            **_measurement_dict(model_measurement),
            "initial_loaded_rows": min(50_000, int(query_result.matching_path_count)),
            "total_rows": int(model.total_row_count),
        },
        "fetch_next_table_batch": {
            **_measurement_dict(fetch_measurement),
            "loaded_rows_after": int(model.loaded_row_count),
        },
        "visible_cell_formatting": {
            **_measurement_dict(visible_measurement),
            "sampled_rows": min(int(visible_rows), int(model.rowCount())),
            "column_count": int(model.columnCount()),
            "checksum": int(visible_checksum),
        },
        "selection_to_overlay": {
            **_measurement_dict(selection_measurement),
            "selected_path_id": selected_path_id,
            "selected_point_count": int(selected_point_count),
            "selected_bounce_count": max(0, int(selected_point_count) - 2),
            "bulk_revision_unchanged": renderer.bulk_mpc_revision == 0,
        },
        "flow_tick_fake_renderer": {
            **_measurement_dict(flow_measurement),
            "ticks_per_measurement": flow_tick_count,
            "median_us_per_tick": round(
                flow_measurement.median_ms * 1_000.0 / flow_tick_count,
                4,
            ),
            "bulk_revision_unchanged": renderer.bulk_mpc_revision == 0,
        },
    }
    return result


def _version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested populations and return a JSON-serializable report."""
    QCoreApplication.instance() or QCoreApplication([])
    populations = [
        benchmark_population(
            count,
            repeats=args.repeats,
            visible_rows=args.visible_rows,
        )
        for count in args.path_counts
    ]
    return {
        "schema_version": 1,
        "benchmark": "mpc_explorer_cpu",
        "measurement_scope": (
            "Synthetic CPU-side catalog/query/model/selection timings; not renderer FPS"
        ),
        "memory_method": (
            "tracemalloc peak above each operation baseline; synthetic frame memory "
            "is reported separately and process RSS is not measured"
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyside6": _version("PySide6"),
            "pygfx": _version("pygfx"),
            "pid": os.getpid(),
        },
        "closed_explorer_guard": _closed_guard(args.closed_iterations),
        "populations": populations,
    }


def main() -> None:
    """Run the benchmark and print/write its JSON report."""
    args = parse_args()
    _validate_args(args)
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {output_path}", file=sys.stderr)
    print(payload)


if __name__ == "__main__":
    main()
