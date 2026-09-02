"""Shared constants and derivation helpers for generated coverage-map files.

``coverage_schema_version`` is the HDF5 coverage-file schema, not the
scenario YAML schema. Version 2 is the first stabilized coverage layout;
it requires grouped grid, TX/RX, values, derived, solver, and metadata
sections.

The storage layout is compact: files materialize the canonical
``path_gain_linear`` tensor and small classification layers such as
``serving_tx``. Metrics such as path loss, RSS, SINR, and TX margin are
available through recipes that use path gain plus TX/noise metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

COVERAGE_HDF5_SCHEMA_VERSION = 2
# Layout marker for the compact v2 contract. Readers reject schema-v2 files
# whose groups exist but whose metric storage semantics are different.
COVERAGE_HDF5_STORAGE_LAYOUT = "canonical_derived_v2"
# Coverage derived during frame generation carries both identities so the
# visualizer can reject a map published for a different frame set.
COVERAGE_FRAME_GENERATION_ID_ATTR = "frame_generation_id"
COVERAGE_FRAME_SET_ID_ATTR = "frame_set_id"
# Sentinel stored in ``serving_tx`` when no transmitter has finite signal at a
# grid cell. Valid transmitter indices remain zero-based.
COVERAGE_NO_SERVING_TX = -1
COVERAGE_CANONICAL_VALUE_METRICS = ("path_gain_linear",)
COVERAGE_COMPACT_DERIVED_METRICS = ("serving_tx",)

COVERAGE_PER_TX_METRICS = frozenset(
    {
        "path_gain_linear",
        "path_gain_db",
        "path_loss_db",
        "rss_w",
        "rss_dbm",
    }
)
# These scalar metrics also accept an optional TX selector. The bare spelling
# remains the best-server result; ``metric/TX`` evaluates that transmitter as
# the signal and every other transmitter as interference.
COVERAGE_OPTIONAL_TX_METRICS = frozenset({"sinr_linear", "sinr_db"})
COVERAGE_TX_SELECTOR_METRICS = COVERAGE_PER_TX_METRICS | COVERAGE_OPTIONAL_TX_METRICS
COVERAGE_SCALAR_METRICS = frozenset(
    {
        "best_path_loss_db",
        "best_rss_dbm",
        "sum_rss_dbm",
        "serving_path_gain_linear",
        "sinr_linear",
        "sinr_db",
        "serving_tx",
        "tx_margin_db",
    }
)
COVERAGE_DERIVABLE_METRICS = COVERAGE_PER_TX_METRICS | COVERAGE_SCALAR_METRICS


def decode_hdf5_attr(value: Any, default: str = "") -> str:
    """Return an HDF5 attribute as text."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def validate_coverage_hdf5_contract(
    schema_version: int,
    storage_layout: Any,
) -> None:
    """Validate the schema-v2 coverage layout marker.

    The numeric schema and storage-layout marker together identify the coverage
    file contract that readers can derive metrics from.
    """
    if schema_version != COVERAGE_HDF5_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported coverage schema version {schema_version}; regenerate coverage"
        )
    layout = decode_hdf5_attr(storage_layout)
    if layout != COVERAGE_HDF5_STORAGE_LAYOUT:
        raise ValueError(
            "Unsupported coverage v2 storage layout; regenerate coverage maps "
            f"with layout '{COVERAGE_HDF5_STORAGE_LAYOUT}'"
        )


def coverage_metric_base(metric_name: str) -> tuple[str, str | None]:
    """Split ``metric/TX`` syntax into base metric and optional selector."""
    value = str(metric_name)
    if "/" in value:
        base, selector = value.split("/", 1)
        return base, selector
    return value, None


def normalise_coverage_tx_names(tx_names: Sequence[str], tx_count: int) -> list[str]:
    """Return TX names with generated fallbacks for unnamed transmitters."""
    names = [str(name) for name in tx_names]
    if len(names) < tx_count:
        names.extend(f"TX{idx + 1}" for idx in range(len(names), tx_count))
    return names[:tx_count]


def resolve_coverage_tx_index(
    selector: str | None,
    tx_names: Sequence[str],
    tx_count: int,
) -> int:
    """Resolve a TX selector from ``metric/TX`` syntax.

    Names are preferred when present. Bare numeric selectors are zero-based;
    ``txN`` labels retain their conventional one-based spelling.
    """
    if tx_count <= 0:
        raise ValueError("coverage metric has no transmitters")
    if selector is None:
        if tx_count == 1:
            return 0
        raise ValueError("per-TX coverage metric requires a TX selector")

    selector = str(selector)
    names = normalise_coverage_tx_names(tx_names, tx_count)
    if selector in names:
        return names.index(selector)
    if selector.startswith("tx"):
        suffix = selector[2:]
        if suffix.isdigit():
            index = int(suffix) - 1
            if 0 <= index < tx_count:
                return index
    if selector.isdigit():
        index = int(selector)
        if 0 <= index < tx_count:
            return index
    raise ValueError(f"unknown TX selector for coverage metric: {selector}")


def db_from_linear(values: np.ndarray) -> np.ndarray:
    """Convert linear power/gain values to dB."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10.0 * np.log10(np.asarray(values, dtype=np.float32))).astype(np.float32)


def path_loss_db_from_gain(path_gain_linear: np.ndarray) -> np.ndarray:
    """Convert path gain to path loss in dB."""
    return (-db_from_linear(path_gain_linear)).astype(np.float32)


def tx_power_dbm_to_watts(tx_power_dbm: np.ndarray, tx_count: int) -> np.ndarray:
    """Return TX powers in watts, padding missing metadata with 0 dBm."""
    powers = np.asarray(tx_power_dbm, dtype=np.float64).reshape(-1)
    if powers.size < tx_count:
        powers = np.pad(powers, (0, tx_count - powers.size), constant_values=0.0)
    powers = powers[:tx_count]
    return (10.0 ** ((powers - 30.0) / 10.0)).astype(np.float32)


def normalise_coverage_metric_array(values: np.ndarray) -> np.ndarray:
    """Return a materialized coverage metric as ``(height, y, x)``."""
    arr = np.asarray(values)
    if arr.ndim == 4:
        return np.asarray(arr[0])
    if arr.ndim == 3:
        return arr
    raise ValueError(f"coverage metric array has unsupported shape {arr.shape}")


def normalise_path_gain_tensor(path_gain_linear: np.ndarray) -> np.ndarray:
    """Return canonical path gain as ``(step, height, tx, y, x)``.

    In-memory coverage helpers may provide one frame as ``(height, tx, y, x)``.
    HDF5 writers and readers normalize that shape here so all derivation formulas
    can share one convention.
    """
    gain = np.asarray(path_gain_linear, dtype=np.float32)
    if gain.ndim == 4:
        gain = gain[np.newaxis, ...]
    if gain.ndim != 5:
        raise ValueError(f"path_gain_linear has unsupported shape {gain.shape}")
    return gain


def sum_coverage_interference_w(
    rss_w: np.ndarray,
    desired_tx_index: int | np.ndarray,
) -> np.ndarray:
    """Sum received power from every transmitter except the desired one.

    Adding only interferers avoids subtracting the desired signal from a much
    larger total, which can discard weak interference in float32 coverage data.
    ``desired_tx_index`` may be one TX index or a per-cell serving-TX array.
    """
    powers = np.asarray(rss_w, dtype=np.float32)
    if powers.ndim != 5:
        raise ValueError(
            "rss_w must have shape (step, height, transmitter, y, x), " f"got {powers.shape}"
        )
    desired = np.asarray(desired_tx_index)
    expected_shape = powers.shape[:2] + powers.shape[3:]
    if desired.ndim != 0 and desired.shape != expected_shape:
        raise ValueError(
            "desired_tx_index must be scalar or have shape "
            f"{expected_shape}, got {desired.shape}"
        )

    interference = np.zeros(expected_shape, dtype=np.float32)
    for tx_index in range(powers.shape[2]):
        contribution = powers[:, :, tx_index, :, :]
        include = (desired != tx_index) & np.isfinite(contribution)
        interference += np.where(include, contribution, 0.0)
    return interference


def derive_coverage_metric_layer(
    path_gain_linear: np.ndarray,
    metric_name: str,
    *,
    tx_power_dbm: np.ndarray,
    noise_power_w: float,
    tx_names: Sequence[str] = (),
    materialized: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Derive one coverage metric layer as ``(height, y, x)``.

    ``path_gain_linear`` is the canonical dense stored layer. All other common
    coverage metrics are derived from it and the TX/noise metadata. Per-TX
    metrics use ``metric/TXName`` or ``metric/index`` selectors. Bare SINR
    selects the strongest server, while selected SINR evaluates the named TX
    against all other TXs plus thermal noise. Other scalar metrics such as
    ``best_path_loss_db`` collapse the TX axis.
    """
    materialized = materialized or {}
    if metric_name in materialized:
        return normalise_coverage_metric_array(np.asarray(materialized[metric_name]))

    base_name, selector = coverage_metric_base(metric_name)
    gain = normalise_path_gain_tensor(path_gain_linear)
    tx_count = int(gain.shape[2])
    names = normalise_coverage_tx_names(tx_names, tx_count)
    power_w = tx_power_dbm_to_watts(tx_power_dbm, tx_count)
    rss_w = gain * power_w.reshape((1, 1, tx_count, 1, 1))

    if base_name in COVERAGE_PER_TX_METRICS:
        tx_index = resolve_coverage_tx_index(selector, names, tx_count)
        if base_name == "path_gain_linear":
            return gain[0, :, tx_index, :, :].astype(np.float32, copy=False)
        if base_name == "path_gain_db":
            return db_from_linear(gain[0, :, tx_index, :, :])
        if base_name == "path_loss_db":
            return path_loss_db_from_gain(gain[0, :, tx_index, :, :])
        if base_name == "rss_w":
            return rss_w[0, :, tx_index, :, :].astype(np.float32, copy=False)
        if base_name == "rss_dbm":
            return db_from_linear(np.maximum(rss_w[0, :, tx_index, :, :], 1e-30)) + 30.0

    if base_name in COVERAGE_OPTIONAL_TX_METRICS and selector is not None:
        tx_index = resolve_coverage_tx_index(selector, names, tx_count)
        selected_power = rss_w[:, :, tx_index, :, :]
        interference = sum_coverage_interference_w(rss_w, tx_index)
        with np.errstate(divide="ignore", invalid="ignore"):
            selected_sinr = selected_power / (interference + float(noise_power_w))
        if base_name == "sinr_linear":
            return selected_sinr[0].astype(np.float32, copy=False)
        return db_from_linear(np.maximum(selected_sinr[0], 1e-30))

    # Treat NaN cells as unservable when ranking transmitters, but keep the raw
    # NaNs in the physical arrays so downstream range statistics still reflect
    # missing coverage.
    finite_rss = np.where(np.isfinite(rss_w), rss_w, -np.inf)
    best_tx_index = np.argmax(finite_rss, axis=2).astype(np.int16)
    best_power = np.take_along_axis(rss_w, best_tx_index[:, :, np.newaxis, :, :], axis=2)[:, :, 0]
    best_gain = np.take_along_axis(gain, best_tx_index[:, :, np.newaxis, :, :], axis=2)[:, :, 0]
    valid_best = np.isfinite(best_power)
    serving_tx = np.where(valid_best, best_tx_index, COVERAGE_NO_SERVING_TX).astype(np.int16)

    if base_name == "best_path_loss_db":
        return path_loss_db_from_gain(best_gain[0])
    if base_name == "serving_path_gain_linear":
        return best_gain[0].astype(np.float32, copy=False)
    if base_name == "best_rss_dbm":
        return db_from_linear(np.maximum(best_power[0], 1e-30)) + 30.0
    if base_name == "sum_rss_dbm":
        summed_rss_w = np.nansum(rss_w, axis=2)
        # No finite TX sample means no coverage, not zero received power.
        summed_rss_w = np.where(np.any(np.isfinite(rss_w), axis=2), summed_rss_w, np.nan)
        return db_from_linear(np.maximum(summed_rss_w[0], 1e-30)) + 30.0
    if base_name == "serving_tx":
        return serving_tx[0]

    interference = sum_coverage_interference_w(rss_w, best_tx_index)
    denominator = interference + float(noise_power_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinr_linear = best_power / denominator

    if base_name == "sinr_linear":
        return sinr_linear[0].astype(np.float32, copy=False)
    if base_name == "sinr_db":
        return db_from_linear(np.maximum(sinr_linear[0], 1e-30))
    if base_name == "tx_margin_db":
        if tx_count > 1:
            sorted_power = np.sort(finite_rss, axis=2)
            second_power = sorted_power[:, :, -2, :, :]
        else:
            second_power = np.full_like(best_power, np.nan)
        best_db = db_from_linear(np.maximum(best_power[0], 1e-30))
        second_db = db_from_linear(np.maximum(second_power[0], 1e-30))
        return (best_db - second_db).astype(np.float32)

    raise ValueError(f"unsupported coverage metric: {metric_name}")


def coverage_metric_recipe_metadata() -> dict[str, str]:
    """Return short derivation recipes for HDF5 metadata and docs."""
    return {
        "path_gain_linear": "canonical stored Sionna RT path gain per TX",
        "path_gain_db": "10*log10(path_gain_linear) per TX",
        "path_loss_db": "-10*log10(path_gain_linear) per TX",
        "serving_path_gain_linear": "path_gain_linear for the TX with strongest rss_w",
        "rss_w": "path_gain_linear * tx_power_w per TX",
        "rss_dbm": "10*log10(rss_w) + 30 per TX",
        "best_path_loss_db": "path_loss_db for the TX with strongest rss_w",
        "best_rss_dbm": "rss_dbm for the TX with strongest rss_w",
        "sum_rss_dbm": "10*log10(sum(rss_w over TX)) + 30",
        "sinr_linear": (
            "bare: best_rss_w / (sum(other_rss_w) + noise_power_w); "
            "metric/TX: rss_w[TX] / (sum(rss_w[other TXs]) + noise_power_w)"
        ),
        "sinr_db": "10*log10(sinr_linear), with the same bare or metric/TX selection",
        "serving_tx": "zero-based TX index with strongest rss_w; -1 means no finite signal",
        "tx_margin_db": "best_rss_db - second_best_rss_db",
    }


def coverage_available_metrics(
    *,
    tx_names: Sequence[str],
    tx_count: int,
    metrics_store: Iterable[str],
    metrics_derived: Iterable[str],
    primary_metric: str,
    materialized_values: Iterable[str] = (),
    materialized_derived: Iterable[str] = (),
) -> list[str]:
    """Return the logical metric menu for a compact coverage file.

    ``materialized_*`` identifies datasets physically present in HDF5. The
    returned list may be larger because it includes recipe-derived metrics that
    can be reconstructed from canonical path gain and TX/noise metadata.
    """
    names = normalise_coverage_tx_names(tx_names, tx_count)
    metrics: list[str] = []

    def add(name: str) -> None:
        if name not in metrics:
            metrics.append(name)

    def add_per_tx(base_name: str) -> None:
        for tx_name in names:
            add(f"{base_name}/{tx_name}")

    requested = [*metrics_store, *metrics_derived, primary_metric, *materialized_values]
    requested.extend(materialized_derived)
    add_per_tx("path_gain_linear")
    if tx_count > 1:
        add("serving_path_gain_linear")
    for metric in requested:
        base_name, selector = coverage_metric_base(metric)
        if base_name in COVERAGE_PER_TX_METRICS:
            if selector is not None:
                add(metric)
            else:
                add_per_tx(base_name)
            if base_name == "path_loss_db":
                add("best_path_loss_db")
            elif base_name == "rss_dbm":
                add("best_rss_dbm")
        elif base_name in COVERAGE_SCALAR_METRICS:
            add(base_name)
            if base_name in COVERAGE_OPTIONAL_TX_METRICS:
                if selector is None:
                    add_per_tx(base_name)
                else:
                    add(metric)
    return metrics
