"""Renderer-neutral evidence schema for sensing replay performance samples.

This module deliberately owns no Qt widgets, renderer objects, HDF5 policy, or
timing loop. Callers measure one operation, attach service load-count deltas and
process RSS boundaries, then aggregate the immutable samples here. Keeping the
measurement loop separate makes the same evidence contract usable by headless
service benchmarks and later end-to-end dashboard benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

SENSING_REPLAY_BENCHMARK_SCHEMA = "orchav.sensing-replay-benchmark.v1"


def _package_version(distribution: str) -> str | None:
    """Return one installed distribution version without importing its runtime."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git_output(repository_root: Path, *arguments: str) -> str:
    """Run one read-only git query and return normalized stdout."""
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git query failed"
        raise RuntimeError(detail)
    return result.stdout.replace("\r\n", "\n")


@dataclass(frozen=True, slots=True)
class SensingReplayBenchmarkEnvironment:
    """Stable host, dependency, source, and optional artifact fingerprint."""

    python_version: str
    python_implementation: str
    operating_system: str
    operating_system_release: str
    machine: str
    numpy_version: str | None
    h5py_version: str | None
    git_commit: str
    git_dirty: bool
    git_status_sha256: str
    artifact_path: str | None = None
    artifact_manifest_id: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-safe mapping with explicit optional artifact identity."""
        return {
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_path": self.artifact_path,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_status_sha256": self.git_status_sha256,
            "h5py_version": self.h5py_version,
            "machine": self.machine,
            "numpy_version": self.numpy_version,
            "operating_system": self.operating_system,
            "operating_system_release": self.operating_system_release,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
        }


def capture_sensing_replay_environment(
    repository_root: str | Path,
    *,
    artifact_path: str | Path | None = None,
    artifact_manifest_id: str | None = None,
) -> SensingReplayBenchmarkEnvironment:
    """Capture reproducibility metadata without starting Qt or a renderer."""
    root = Path(repository_root)
    commit = _git_output(root, "rev-parse", "HEAD").strip()
    if not commit:
        raise RuntimeError("git commit fingerprint is empty")
    status = _git_output(root, "status", "--short", "--untracked-files=all")
    artifact = None if artifact_path is None else Path(artifact_path).as_posix()
    if artifact_manifest_id is not None and not artifact_manifest_id.strip():
        raise ValueError("artifact_manifest_id must be non-empty when provided")
    return SensingReplayBenchmarkEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine(),
        numpy_version=_package_version("numpy"),
        h5py_version=_package_version("h5py"),
        git_commit=commit,
        git_dirty=bool(status.strip()),
        git_status_sha256=hashlib.sha256(status.encode("utf-8")).hexdigest(),
        artifact_path=artifact,
        artifact_manifest_id=artifact_manifest_id,
    )


def _non_negative_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SensingReplayBenchmarkSample:
    """One raw operation measurement with RSS boundaries and load-count deltas."""

    operation: str
    repetition: int
    duration_ms: float
    rss_before_bytes: int
    rss_peak_bytes: int
    rss_after_bytes: int
    load_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        if not operation:
            raise ValueError("operation must be non-empty")
        repetition = _non_negative_integer(self.repetition, label="repetition")
        if isinstance(self.duration_ms, bool):
            raise ValueError("duration_ms must be a finite non-negative number")
        duration_ms = float(self.duration_ms)
        if not math.isfinite(duration_ms) or duration_ms < 0.0:
            raise ValueError("duration_ms must be a finite non-negative number")
        rss_before = _non_negative_integer(self.rss_before_bytes, label="rss_before_bytes")
        rss_peak = _non_negative_integer(self.rss_peak_bytes, label="rss_peak_bytes")
        rss_after = _non_negative_integer(self.rss_after_bytes, label="rss_after_bytes")
        if rss_peak < max(rss_before, rss_after):
            raise ValueError("rss_peak_bytes must include both RSS boundary observations")

        counts: dict[str, int] = {}
        for raw_name, raw_count in self.load_counts.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("load-count names must be non-empty")
            if name in counts:
                raise ValueError(f"duplicate load-count name: {name}")
            counts[name] = _non_negative_integer(raw_count, label=f"load_counts[{name!r}]")

        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "repetition", repetition)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "rss_before_bytes", rss_before)
        object.__setattr__(self, "rss_peak_bytes", rss_peak)
        object.__setattr__(self, "rss_after_bytes", rss_after)
        object.__setattr__(self, "load_counts", MappingProxyType(dict(sorted(counts.items()))))

    def to_manifest(self) -> dict[str, Any]:
        """Return the exact raw observation without rounding or timestamps."""
        return {
            "duration_ms": self.duration_ms,
            "load_counts": dict(self.load_counts),
            "operation": self.operation,
            "repetition": self.repetition,
            "rss_after_bytes": self.rss_after_bytes,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
        }


@dataclass(frozen=True, slots=True)
class SensingReplayCacheObservation:
    """One post-selection observation of every replay-relevant cache bound.

    The service-owned dense caches and the visualizer raw-frame cache have
    different lifetimes.  Recording both after every selection makes a
    multi-result campaign capable of proving bounded occupancy instead of
    inferring it from one cold/warm result pair.
    """

    mode: str
    worker_repetition: int
    selection_index: int
    result_id: int
    metadata_entries: int
    selected_product_entries: int
    selected_product_capacity: int
    detection_stage_entries: int
    detection_stage_capacity: int
    raw_cache_entries: int
    raw_cache_capacity: int
    raw_cache_evictions: int

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        if not mode:
            raise ValueError("mode must be non-empty")
        worker_repetition = _non_negative_integer(
            self.worker_repetition,
            label="worker_repetition",
        )
        selection_index = _non_negative_integer(
            self.selection_index,
            label="selection_index",
        )
        result_id = _non_negative_integer(self.result_id, label="result_id")
        metadata_entries = _non_negative_integer(
            self.metadata_entries,
            label="metadata_entries",
        )
        selected_entries = _non_negative_integer(
            self.selected_product_entries,
            label="selected_product_entries",
        )
        selected_capacity = _non_negative_integer(
            self.selected_product_capacity,
            label="selected_product_capacity",
        )
        stage_entries = _non_negative_integer(
            self.detection_stage_entries,
            label="detection_stage_entries",
        )
        stage_capacity = _non_negative_integer(
            self.detection_stage_capacity,
            label="detection_stage_capacity",
        )
        raw_entries = _non_negative_integer(
            self.raw_cache_entries,
            label="raw_cache_entries",
        )
        raw_capacity = _non_negative_integer(
            self.raw_cache_capacity,
            label="raw_cache_capacity",
        )
        raw_evictions = _non_negative_integer(
            self.raw_cache_evictions,
            label="raw_cache_evictions",
        )
        for label, entries, capacity in (
            ("selected product", selected_entries, selected_capacity),
            ("detection stage", stage_entries, stage_capacity),
            ("raw frame", raw_entries, raw_capacity),
        ):
            if capacity <= 0:
                raise ValueError(f"{label} cache capacity must be positive")
            if entries > capacity:
                raise ValueError(f"{label} cache entries {entries} exceed capacity {capacity}")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "worker_repetition", worker_repetition)
        object.__setattr__(self, "selection_index", selection_index)
        object.__setattr__(self, "result_id", result_id)
        object.__setattr__(self, "metadata_entries", metadata_entries)
        object.__setattr__(self, "selected_product_entries", selected_entries)
        object.__setattr__(self, "selected_product_capacity", selected_capacity)
        object.__setattr__(self, "detection_stage_entries", stage_entries)
        object.__setattr__(self, "detection_stage_capacity", stage_capacity)
        object.__setattr__(self, "raw_cache_entries", raw_entries)
        object.__setattr__(self, "raw_cache_capacity", raw_capacity)
        object.__setattr__(self, "raw_cache_evictions", raw_evictions)

    def to_manifest(self) -> dict[str, int | str]:
        """Return a stable JSON-safe cache observation."""
        return {
            "detection_stage_capacity": self.detection_stage_capacity,
            "detection_stage_entries": self.detection_stage_entries,
            "metadata_entries": self.metadata_entries,
            "mode": self.mode,
            "raw_cache_capacity": self.raw_cache_capacity,
            "raw_cache_entries": self.raw_cache_entries,
            "raw_cache_evictions": self.raw_cache_evictions,
            "result_id": self.result_id,
            "selected_product_capacity": self.selected_product_capacity,
            "selected_product_entries": self.selected_product_entries,
            "selection_index": self.selection_index,
            "worker_repetition": self.worker_repetition,
        }


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    """Deterministic descriptive statistics for repeated scalar observations."""

    count: int
    mean: float
    p50: float
    p95: float
    maximum: float
    stddev: float

    def to_manifest(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "max": self.maximum,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "stddev": self.stddev,
        }


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Iterable[int | float]) -> NumericDistribution:
    numeric = tuple(float(value) for value in values)
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("distribution values must be non-empty and finite")
    return NumericDistribution(
        count=len(numeric),
        mean=statistics.fmean(numeric),
        p50=_percentile(numeric, 50.0),
        p95=_percentile(numeric, 95.0),
        maximum=max(numeric),
        stddev=statistics.pstdev(numeric),
    )


def _operation_summary(samples: tuple[SensingReplayBenchmarkSample, ...]) -> dict[str, Any]:
    count_names = sorted({name for sample in samples for name in sample.load_counts})
    return {
        "duration_ms": _distribution(sample.duration_ms for sample in samples).to_manifest(),
        "load_counts": {
            name: _distribution(sample.load_counts.get(name, 0) for sample in samples).to_manifest()
            for name in count_names
        },
        "rss_bytes": {
            "after": _distribution(sample.rss_after_bytes for sample in samples).to_manifest(),
            "before": _distribution(sample.rss_before_bytes for sample in samples).to_manifest(),
            "peak": _distribution(sample.rss_peak_bytes for sample in samples).to_manifest(),
            "peak_delta": _distribution(
                sample.rss_peak_bytes - sample.rss_before_bytes for sample in samples
            ).to_manifest(),
        },
    }


@dataclass(frozen=True, slots=True)
class SensingReplayBenchmarkReport:
    """Deterministic raw and aggregate evidence for one benchmark environment."""

    environment: SensingReplayBenchmarkEnvironment
    samples: tuple[SensingReplayBenchmarkSample, ...]
    operation_summaries: Mapping[str, Mapping[str, Any]]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_manifest(),
            "operations": {
                name: dict(summary) for name, summary in self.operation_summaries.items()
            },
            "samples": [sample.to_manifest() for sample in self.samples],
            "schema": SENSING_REPLAY_BENCHMARK_SCHEMA,
        }

    def to_json(self) -> str:
        """Serialize byte-stably; non-finite measurements are rejected earlier."""
        return json.dumps(
            self.to_manifest(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def aggregate_sensing_replay_samples(
    environment: SensingReplayBenchmarkEnvironment,
    samples: Iterable[SensingReplayBenchmarkSample],
) -> SensingReplayBenchmarkReport:
    """Validate, sort, and aggregate raw samples without wall-clock thresholds."""
    ordered = tuple(sorted(samples, key=lambda sample: (sample.operation, sample.repetition)))
    if not ordered:
        raise ValueError("at least one sensing replay benchmark sample is required")
    identities = [(sample.operation, sample.repetition) for sample in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError("operation/repetition identities must be unique")

    operation_summaries = {
        operation: _operation_summary(
            tuple(sample for sample in ordered if sample.operation == operation)
        )
        for operation in sorted({sample.operation for sample in ordered})
    }
    return SensingReplayBenchmarkReport(
        environment=environment,
        samples=ordered,
        operation_summaries=MappingProxyType(operation_summaries),
    )


def telemetry_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Return a fail-closed non-negative counter delta for one measured operation."""
    names = sorted(set(before) | set(after))
    delta: dict[str, int] = {}
    for name in names:
        start = _non_negative_integer(before.get(name, 0), label=f"before[{name!r}]")
        end = _non_negative_integer(after.get(name, 0), label=f"after[{name!r}]")
        if end < start:
            raise ValueError(f"telemetry counter {name!r} decreased without an explicit reset")
        delta[name] = end - start
    return delta
