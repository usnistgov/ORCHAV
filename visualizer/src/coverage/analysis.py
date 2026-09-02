"""Coverage-map analysis helpers used by the visualizer UI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from shared.coverage.schema import COVERAGE_NO_SERVING_TX, coverage_metric_base

try:
    from contourpy import contour_generator as _contour_generator
except ImportError:  # pragma: no cover - exercised only in minimal installations
    _contour_generator = None

CoverageComparator = Literal["<=", ">="]
CoverageColorScale = Literal["linear", "logarithmic", "categorical"]

_MAX_COVERAGE_CURVE_POINTS = 5000

_SERVING_TX_COLOR_HEX = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#7f7f7f",
)

_COVERAGE_METRIC_INFO: dict[str, tuple[str, str, CoverageComparator | None]] = {
    "path_gain_linear": ("Path gain", "linear", ">="),
    "serving_path_gain_linear": ("Serving path gain", "linear", ">="),
    "path_gain_db": ("Path gain", "dB", ">="),
    "path_loss_db": ("Path loss", "dB", "<="),
    "best_path_loss_db": ("Best path loss", "dB", "<="),
    "rss_w": ("Received power", "W", ">="),
    "rss_dbm": ("Received power", "dBm", ">="),
    "best_rss_dbm": ("Best received power", "dBm", ">="),
    "sum_rss_dbm": ("Aggregate received power", "dBm", ">="),
    "sinr_linear": ("SINR", "linear", ">="),
    "sinr_db": ("SINR", "dB", ">="),
    "serving_tx": ("Serving TX", "index", None),
    "tx_margin_db": ("TX margin", "dB", ">="),
}

_LOGARITHMIC_COLOR_METRICS = frozenset(
    {
        "path_gain_linear",
        "serving_path_gain_linear",
        "rss_w",
        "sinr_linear",
    }
)


@dataclass(frozen=True)
class CoverageThresholdSummary:
    """Summary for a coverage threshold on one metric and height slice."""

    metric_name: str
    label: str
    unit: str
    threshold: float
    comparator: CoverageComparator
    total_cells: int
    valid_cells: int
    covered_cells: int
    cell_area_m2: float

    @property
    def valid_area_m2(self) -> float:
        """Area represented by finite cells in the selected coverage slice."""
        return self.valid_cells * self.cell_area_m2

    @property
    def covered_area_m2(self) -> float:
        """Area represented by cells that satisfy the metric comparator."""
        return self.covered_cells * self.cell_area_m2

    @property
    def covered_percent_valid(self) -> float:
        """Covered percentage among finite cells only."""
        if self.valid_cells <= 0:
            return 0.0
        return 100.0 * self.covered_cells / self.valid_cells

    @property
    def covered_percent_total(self) -> float:
        """Covered percentage across the full slice including invalid cells."""
        if self.total_cells <= 0:
            return 0.0
        return 100.0 * self.covered_cells / self.total_cells


@dataclass(frozen=True)
class CoverageSliceSummary:
    """Descriptive statistics for one selected scalar coverage slice."""

    metric_name: str
    label: str
    unit: str
    total_cells: int
    valid_cells: int
    cell_area_m2: float
    percentile_10: float | None
    percentile_50: float | None
    percentile_90: float | None

    @property
    def no_data_cells(self) -> int:
        """Number of cells without a usable scalar value."""
        return max(0, self.total_cells - self.valid_cells)

    @property
    def valid_percent(self) -> float:
        """Percentage of the full slice with usable scalar values."""
        if self.total_cells <= 0:
            return 0.0
        return 100.0 * self.valid_cells / self.total_cells

    @property
    def no_data_percent(self) -> float:
        """Percentage of the full slice without a usable scalar value."""
        if self.total_cells <= 0:
            return 0.0
        return 100.0 * self.no_data_cells / self.total_cells


@dataclass(frozen=True)
class ServingTxCoverageSummary:
    """Serving-transmitter allocation for one selected coverage slice."""

    total_cells: int
    tx_names: tuple[str, ...]
    served_cells: tuple[int, ...]
    cell_area_m2: float

    @property
    def no_service_cells(self) -> int:
        """Number of cells not assigned to a valid transmitter."""
        return max(0, self.total_cells - sum(self.served_cells))

    @property
    def no_service_percent(self) -> float:
        """Percentage of the full slice without a serving transmitter."""
        if self.total_cells <= 0:
            return 0.0
        return 100.0 * self.no_service_cells / self.total_cells

    @property
    def no_service_area_m2(self) -> float:
        """Physical area without a serving transmitter."""
        return self.no_service_cells * self.cell_area_m2

    def served_percent(self, tx_index: int) -> float:
        """Percentage of the full slice served by one transmitter."""
        if self.total_cells <= 0:
            return 0.0
        return 100.0 * self.served_cells[tx_index] / self.total_cells

    def served_area_m2(self, tx_index: int) -> float:
        """Physical area served by one transmitter."""
        return self.served_cells[tx_index] * self.cell_area_m2


@dataclass(frozen=True)
class CoverageScalarPlotData:
    """Chart-ready values for one raw scalar coverage slice."""

    summary: CoverageSliceSummary
    comparator: CoverageComparator
    color_scale: CoverageColorScale
    valid_values: np.ndarray
    thresholds: np.ndarray
    qualifying_percent_total: np.ndarray


def coverage_metric_label(metric_name: Any) -> tuple[str, str]:
    """Return a human label and unit for a coverage metric key."""
    raw = str(metric_name or "").strip()
    base, _, suffix = raw.partition("/")
    label, unit, _ = _COVERAGE_METRIC_INFO.get(
        base,
        (base.replace("_", " ").strip().title() or "Coverage", "", ">="),
    )
    if suffix:
        label = f"{label} ({suffix})"
    return label, unit


def coverage_metric_comparator(metric_name: Any) -> CoverageComparator | None:
    """Return the default coverage-threshold comparator for a metric."""
    base = str(metric_name or "").strip().partition("/")[0]
    return _COVERAGE_METRIC_INFO.get(base, ("", "", ">="))[2]


def coverage_metric_colormap(metric_name: Any) -> str | None:
    """Return the semantic scalar colormap name for a coverage metric.

    Lower-is-better metrics run from green to red as values increase, while
    higher-is-better metrics run from red to green. Categorical metrics do not
    use a scalar colormap.
    """
    comparator = coverage_metric_comparator(metric_name)
    if comparator is None:
        return None
    return "RdYlGn_r" if comparator == "<=" else "RdYlGn"


def coverage_metric_color_scale(metric_name: Any) -> CoverageColorScale:
    """Return the color normalization appropriate for one coverage metric."""
    base, _ = coverage_metric_base(str(metric_name or ""))
    if base == "serving_tx":
        return "categorical"
    if base in _LOGARITHMIC_COLOR_METRICS:
        return "logarithmic"
    return "linear"


def coverage_metric_valid_mask(values: Any, metric_name: Any) -> np.ndarray:
    """Return cells that carry a value usable by the metric's scalar scale.

    Logarithmic RF quantities require strictly positive values. Other scalar
    metrics accept every finite value, including valid negative dB values.
    """
    array = np.asarray(values)
    valid = np.isfinite(array)
    if coverage_metric_color_scale(metric_name) == "logarithmic":
        valid &= array > 0
    return valid


def supports_coverage_threshold(metric_name: Any) -> bool:
    """Return True when a metric is scalar and meaningful for thresholding."""
    return coverage_metric_comparator(metric_name) is not None


def is_serving_tx_metric(metric_name: Any) -> bool:
    """Return True for the categorical serving-transmitter coverage layer."""
    base, _ = coverage_metric_base(str(metric_name or ""))
    return base == "serving_tx"


def serving_tx_labels(tx_names: Sequence[str], tx_count: int | None = None) -> list[str]:
    """Return TX labels for serving-TX categories 0..N-1."""
    count = len(tx_names) if tx_count is None else max(0, int(tx_count))
    labels = [str(name) for name in tx_names]
    if len(labels) < count:
        labels.extend(f"TX{idx + 1}" for idx in range(len(labels), count))
    return labels[:count]


def serving_tx_color_hex(tx_index: int) -> str:
    """Return a stable categorical color for a serving-TX index."""
    return _SERVING_TX_COLOR_HEX[int(tx_index) % len(_SERVING_TX_COLOR_HEX)]


def serving_tx_color_rgb(tx_index: int) -> tuple[float, float, float]:
    """Return a normalized RGB categorical color for a serving-TX index."""
    color = serving_tx_color_hex(tx_index).lstrip("#")
    return tuple(int(color[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4))


def serving_tx_valid_mask(values: np.ndarray, tx_count: int) -> np.ndarray:
    """Return cells with a valid serving-TX index, excluding no-service cells."""
    arr = np.asarray(values, dtype=np.float32)
    upper_bound = max(0, int(tx_count))
    return np.isfinite(arr) & (arr != COVERAGE_NO_SERVING_TX) & (arr >= 0) & (arr < upper_bound)


def format_coverage_value(value: Any, unit: str) -> str:
    """Format a coverage scalar for compact UI display."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    if unit == "index":
        return f"{numeric:.0f}"
    if unit in {"linear", "W"}:
        return f"{numeric:.3g}"
    return f"{numeric:.1f}"


def coverage_values_for_height(
    coverage_data: dict[str, Any],
    height_index: int,
) -> np.ndarray:
    """Return the active 2D coverage slice for one logical height.

    File-backed coverage keeps only the active height in memory, so its array
    index is zero even when ``height_index`` identifies a later logical height.
    In-memory coverage can retain every height and is indexed normally.
    """
    values = coverage_data.get("values_3d")
    if values is None:
        values = coverage_data.get("values")
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim >= 3:
        if coverage_data.get("coverage_file") and arr.shape[0] == 1:
            return np.asarray(arr[0], dtype=np.float32)
        idx = max(0, min(int(height_index), arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return arr.reshape(1, -1)


def _coverage_cell_area_m2(coverage_data: dict[str, Any]) -> float:
    """Return the absolute XY area represented by one coverage cell."""
    spacing = np.asarray(coverage_data.get("grid_spacing", [1.0, 1.0]), dtype=np.float32)
    dx = float(spacing[0]) if spacing.size >= 1 and np.isfinite(spacing[0]) else 1.0
    dy = float(spacing[1]) if spacing.size >= 2 and np.isfinite(spacing[1]) else 1.0
    return abs(dx * dy)


def compute_coverage_slice_summary(
    coverage_data: dict[str, Any],
    *,
    height_index: int,
) -> CoverageSliceSummary:
    """Compute percentiles and data availability for one scalar slice."""
    metric_name = str(coverage_data.get("metric_name", "coverage"))
    if is_serving_tx_metric(metric_name):
        raise ValueError("serving_tx requires a categorical coverage summary")

    values = coverage_values_for_height(coverage_data, height_index)
    valid_mask = coverage_metric_valid_mask(values, metric_name)
    valid_values = np.asarray(values[valid_mask], dtype=np.float64)
    if valid_values.size:
        percentiles = np.percentile(valid_values, [10.0, 50.0, 90.0])
        percentile_10, percentile_50, percentile_90 = (float(value) for value in percentiles)
    else:
        percentile_10 = percentile_50 = percentile_90 = None

    label, unit = coverage_metric_label(metric_name)
    return CoverageSliceSummary(
        metric_name=metric_name,
        label=label,
        unit=unit,
        total_cells=int(values.size),
        valid_cells=int(valid_values.size),
        cell_area_m2=_coverage_cell_area_m2(coverage_data),
        percentile_10=percentile_10,
        percentile_50=percentile_50,
        percentile_90=percentile_90,
    )


def compute_coverage_scalar_plot_data(
    coverage_data: dict[str, Any],
    *,
    height_index: int,
    max_curve_points: int = _MAX_COVERAGE_CURVE_POINTS,
) -> CoverageScalarPlotData:
    """Prepare a bounded success curve and distribution values for one slice.

    The success curve uses the metric's physical direction: lower path loss is
    successful at ``value <= threshold`` while gain, power, and SINR use
    ``value >= threshold``. Percentages use every grid cell as the denominator,
    so the curve also exposes area where the metric has no usable value.
    """
    metric_name = str(coverage_data.get("metric_name", "coverage"))
    comparator = coverage_metric_comparator(metric_name)
    if comparator is None:
        raise ValueError(f"Coverage metric {metric_name!r} is not scalar")

    summary = compute_coverage_slice_summary(
        coverage_data,
        height_index=height_index,
    )
    values = coverage_values_for_height(coverage_data, height_index)
    valid = np.asarray(
        values[coverage_metric_valid_mask(values, metric_name)],
        dtype=np.float64,
    )

    if valid.size == 0:
        thresholds = np.empty(0, dtype=np.float64)
        qualifying = np.empty(0, dtype=np.float64)
    else:
        thresholds, counts = np.unique(valid, return_counts=True)
        cumulative = np.cumsum(counts, dtype=np.int64)
        if comparator == "<=":
            qualifying_counts = cumulative
        else:
            qualifying_counts = valid.size - np.concatenate(
                (np.zeros(1, dtype=np.int64), cumulative[:-1])
            )
        qualifying = 100.0 * qualifying_counts / max(summary.total_cells, 1)

        point_limit = max(2, int(max_curve_points))
        if thresholds.size > point_limit:
            indices = np.linspace(
                0,
                thresholds.size - 1,
                point_limit,
                dtype=np.intp,
            )
            thresholds = thresholds[indices]
            qualifying = qualifying[indices]

    return CoverageScalarPlotData(
        summary=summary,
        comparator=comparator,
        color_scale=coverage_metric_color_scale(metric_name),
        valid_values=valid,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        qualifying_percent_total=np.asarray(qualifying, dtype=np.float64),
    )


def compute_serving_tx_coverage_summary(
    coverage_data: dict[str, Any],
    *,
    height_index: int,
) -> ServingTxCoverageSummary:
    """Compute per-transmitter and no-service shares for one slice."""
    metric_name = str(coverage_data.get("metric_name", "coverage"))
    if not is_serving_tx_metric(metric_name):
        raise ValueError("serving-transmitter summary requires the serving_tx metric")

    values = coverage_values_for_height(coverage_data, height_index)
    raw_names = coverage_data.get("tx_names")
    if raw_names is None:
        raw_names = []
    if hasattr(raw_names, "tolist"):
        raw_names = raw_names.tolist()
    names = [str(name) for name in raw_names]
    declared_counts = [len(names)]
    for key in ("serving_tx_count", "tx_count"):
        try:
            count = max(0, int(coverage_data.get(key, 0)))
        except (TypeError, ValueError):
            continue
        if count:
            declared_counts.append(count)
    numeric_values = np.asarray(values, dtype=np.float64)
    tx_count = max(declared_counts)
    if tx_count == 0:
        finite_indices = numeric_values[
            np.isfinite(numeric_values)
            & (numeric_values >= 0)
            & (numeric_values == np.floor(numeric_values))
        ]
        if finite_indices.size:
            tx_count = int(np.max(finite_indices)) + 1
    tx_names = tuple(serving_tx_labels(names, tx_count))
    served_cells = tuple(
        int(np.count_nonzero(numeric_values == tx_index)) for tx_index in range(tx_count)
    )
    return ServingTxCoverageSummary(
        total_cells=int(values.size),
        tx_names=tx_names,
        served_cells=served_cells,
        cell_area_m2=_coverage_cell_area_m2(coverage_data),
    )


def default_coverage_threshold(
    coverage_data: dict[str, Any],
    height_index: int = 0,
) -> float:
    """Choose a robust default threshold from the displayed height slice."""
    values = coverage_values_for_height(coverage_data, height_index)
    metric_name = coverage_data.get("metric_name")
    valid = values[coverage_metric_valid_mask(values, metric_name)]
    if valid.size:
        return float(np.median(valid))
    try:
        vmin = float(coverage_data.get("value_min", 0.0))
        vmax = float(coverage_data.get("value_max", 1.0))
    except (TypeError, ValueError):
        return 0.0
    bounds = np.asarray([vmin, vmax], dtype=np.float64)
    valid_bounds = bounds[coverage_metric_valid_mask(bounds, metric_name)]
    if valid_bounds.size == 0:
        return 0.0
    if valid_bounds.size == 2 and coverage_metric_color_scale(metric_name) == "logarithmic":
        return float(np.sqrt(valid_bounds[0] * valid_bounds[1]))
    return float(np.mean(valid_bounds))


def compute_coverage_threshold_summary(
    coverage_data: dict[str, Any],
    *,
    height_index: int,
    threshold: float,
) -> CoverageThresholdSummary:
    """Compute coverage area and percentage for the active metric slice."""
    metric_name = str(coverage_data.get("metric_name", "coverage"))
    comparator = coverage_metric_comparator(metric_name)
    if comparator is None:
        raise ValueError(f"Coverage metric {metric_name!r} is not thresholdable")

    values = coverage_values_for_height(coverage_data, height_index)
    valid_mask = coverage_metric_valid_mask(values, metric_name)
    if comparator == "<=":
        covered_mask = valid_mask & (values <= float(threshold))
    else:
        covered_mask = valid_mask & (values >= float(threshold))

    label, unit = coverage_metric_label(metric_name)
    return CoverageThresholdSummary(
        metric_name=metric_name,
        label=label,
        unit=unit,
        threshold=float(threshold),
        comparator=comparator,
        total_cells=int(values.size),
        valid_cells=int(valid_mask.sum()),
        covered_cells=int(covered_mask.sum()),
        cell_area_m2=_coverage_cell_area_m2(coverage_data),
    )


def compute_coverage_threshold_mask(
    values_2d: np.ndarray,
    *,
    metric_name: Any,
    threshold: float,
) -> np.ndarray:
    """Return a boolean pass/fail mask for one coverage height slice."""
    comparator = coverage_metric_comparator(metric_name)
    if comparator is None:
        raise ValueError(f"Coverage metric {metric_name!r} is not thresholdable")
    values = np.asarray(values_2d, dtype=np.float32)
    valid_mask = coverage_metric_valid_mask(values, metric_name)
    if comparator == "<=":
        return valid_mask & (values <= float(threshold))
    return valid_mask & (values >= float(threshold))


def apply_coverage_threshold_mask_to_colors(
    colors_2d: np.ndarray,
    threshold_mask: np.ndarray,
    *,
    fail_blend: float = 0.68,
) -> np.ndarray:
    """Dim failing cells while keeping passing cells in the normal colormap."""
    colors = np.asarray(colors_2d, dtype=np.float32).copy()
    mask = np.asarray(threshold_mask, dtype=bool)
    if colors.ndim != 3 or colors.shape[:2] != mask.shape:
        raise ValueError("coverage colors and threshold mask shape mismatch")

    fail = ~mask
    if np.any(fail):
        gray = np.mean(colors[fail], axis=1, keepdims=True)
        colors[fail] = colors[fail] * (1.0 - fail_blend) + gray * fail_blend
        colors[fail] *= 0.55
    return np.clip(colors, 0.0, 1.0)


def build_coverage_isoline(
    values_2d: np.ndarray,
    *,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    z_level: float,
    level: float,
    z_offset: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one coverage isoline as indexed world-space line segments."""
    values = np.asarray(values_2d, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return _empty_coverage_isoline()

    origin = np.asarray(grid_origin, dtype=np.float64).reshape(-1)
    spacing = np.asarray(grid_spacing, dtype=np.float64).reshape(-1)
    ox = float(origin[0]) if origin.size >= 1 else 0.0
    oy = float(origin[1]) if origin.size >= 2 else 0.0
    dx = float(spacing[0]) if spacing.size >= 1 and np.isfinite(spacing[0]) else 1.0
    dy = float(spacing[1]) if spacing.size >= 2 and np.isfinite(spacing[1]) else 1.0
    z = float(z_level) + float(z_offset)
    level = float(level)
    ny, nx = values.shape
    x_coordinates = ox + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_coordinates = oy + (np.arange(ny, dtype=np.float64) + 0.5) * dy

    if not np.isfinite(level):
        return _empty_coverage_isoline()

    if _contour_generator is not None:
        try:
            generator = _contour_generator(
                x=x_coordinates,
                y=y_coordinates,
                z=np.ma.masked_invalid(values, copy=False),
                line_type="ChunkCombinedOffset",
            )
            return _coverage_contour_chunks_to_payload(generator.lines(level), z=z)
        except (TypeError, ValueError):
            # The fallback supports degenerate or descending grid coordinates
            # that contourpy cannot process.
            pass

    return _build_coverage_isoline_fallback(
        values,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        z=z,
        level=level,
    )


def _empty_coverage_isoline() -> tuple[np.ndarray, np.ndarray]:
    """Return an empty isoline payload with stable renderer-facing dtypes."""
    return (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 2), dtype=np.int32),
    )


def _coverage_contour_chunks_to_payload(
    chunks: tuple[Sequence[np.ndarray | None], Sequence[np.ndarray | None]],
    *,
    z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert contourpy's combined paths to the visualizer line payload."""
    raw_point_chunks, raw_offset_chunks = chunks
    point_chunks: list[np.ndarray] = []
    line_chunks: list[np.ndarray] = []
    point_offset = 0
    for raw_points, raw_offsets in zip(raw_point_chunks, raw_offset_chunks):
        if raw_points is None or raw_offsets is None:
            continue
        path = np.asarray(raw_points, dtype=np.float64)
        offsets = np.asarray(raw_offsets, dtype=np.int64).reshape(-1)
        if (
            path.ndim != 2
            or path.shape[0] < 2
            or path.shape[1] < 2
            or not np.isfinite(path[:, :2]).all()
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != path.shape[0]
            or np.any(np.diff(offsets) < 2)
        ):
            continue

        count = path.shape[0]
        points = np.empty((count, 3), dtype=np.float64)
        points[:, :2] = path[:, :2]
        points[:, 2] = z
        starts = np.arange(count - 1, dtype=np.int64)
        if offsets.size > 2:
            keep = np.ones(starts.shape, dtype=bool)
            keep[offsets[1:-1] - 1] = False
            starts = starts[keep]
        starts += point_offset
        lines = np.column_stack((starts, starts + 1)).astype(np.int32, copy=False)
        point_chunks.append(points)
        line_chunks.append(lines)
        point_offset += count

    if not point_chunks:
        return _empty_coverage_isoline()
    return np.vstack(point_chunks), np.vstack(line_chunks)


def _build_coverage_isoline_fallback(
    values: np.ndarray,
    *,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z: float,
    level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a contour when the optional compiled contour engine is unavailable."""
    points: list[tuple[float, float, float]] = []
    lines: list[tuple[int, int]] = []

    def interpolate(
        p0: tuple[float, float],
        p1: tuple[float, float],
        value0: float,
        value1: float,
    ) -> tuple[float, float, float]:
        fraction = (level - value0) / (value1 - value0)
        x = p0[0] + fraction * (p1[0] - p0[0])
        y = p0[1] + fraction * (p1[1] - p0[1])
        return float(x), float(y), z

    ny, nx = values.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            corner_values = (
                float(values[j, i]),
                float(values[j, i + 1]),
                float(values[j + 1, i + 1]),
                float(values[j + 1, i]),
            )
            if not np.isfinite(corner_values).all():
                continue

            high = tuple(value > level for value in corner_values)
            edge_corners = ((0, 1), (1, 2), (2, 3), (3, 0))
            crossed_edges = [
                edge_index
                for edge_index, (first, second) in enumerate(edge_corners)
                if high[first] != high[second]
            ]
            if len(crossed_edges) not in {2, 4}:
                continue

            x0, x1 = float(x_coordinates[i]), float(x_coordinates[i + 1])
            y0, y1 = float(y_coordinates[j]), float(y_coordinates[j + 1])
            corner_points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            intersections = {
                edge: interpolate(
                    corner_points[edge_corners[edge][0]],
                    corner_points[edge_corners[edge][1]],
                    corner_values[edge_corners[edge][0]],
                    corner_values[edge_corners[edge][1]],
                )
                for edge in crossed_edges
            }

            if len(crossed_edges) == 2:
                edge_pairs = [(crossed_edges[0], crossed_edges[1])]
            else:
                center_is_high = sum(corner_values) > 4.0 * level
                use_top_right_pairing = center_is_high == high[0]
                if use_top_right_pairing:
                    edge_pairs = [(0, 1), (2, 3)]
                else:
                    edge_pairs = [(0, 3), (1, 2)]

            for first_edge, second_edge in edge_pairs:
                first = intersections[first_edge]
                second = intersections[second_edge]
                if np.allclose(first, second, rtol=0.0, atol=1e-12):
                    continue
                start = len(points)
                points.extend((first, second))
                lines.append((start, start + 1))

    if not points:
        return _empty_coverage_isoline()
    return np.asarray(points, dtype=np.float64), np.asarray(lines, dtype=np.int32)


def format_coverage_threshold_summary(summary: CoverageThresholdSummary) -> str:
    """Format a coverage-threshold summary for the Coverage panel."""
    threshold = format_coverage_value(summary.threshold, summary.unit)
    return (
        f"{summary.comparator} {threshold}: "
        f"{summary.covered_cells}/{summary.valid_cells} valid cells "
        f"({summary.covered_percent_valid:.1f}% valid; "
        f"{summary.covered_percent_total:.1f}% total), "
        f"{summary.covered_area_m2:.1f} m^2"
    )


def format_coverage_slice_summary(summary: CoverageSliceSummary) -> str:
    """Format scalar availability and percentile statistics for the panel."""
    percentiles = "/".join(
        format_coverage_value(value, summary.unit)
        for value in (
            summary.percentile_10,
            summary.percentile_50,
            summary.percentile_90,
        )
    )
    unit_suffix = f" {summary.unit}" if summary.unit and summary.unit != "index" else ""
    return (
        f"Valid: {summary.valid_cells}/{summary.total_cells} cells "
        f"({summary.valid_percent:.1f}%); no data: {summary.no_data_percent:.1f}% | "
        f"P10/P50/P90: {percentiles}{unit_suffix}"
    )


def format_serving_tx_coverage_summary(summary: ServingTxCoverageSummary) -> str:
    """Format categorical serving-area shares for the Coverage panel."""
    total = summary.total_cells
    entries = [
        (
            f"No service: {summary.no_service_cells}/{total} cells "
            f"({summary.no_service_percent:.1f}%, "
            f"{summary.no_service_area_m2:.1f} m^2)"
        )
    ]
    entries.extend(
        (
            f"{name}: {summary.served_cells[index]}/{total} cells "
            f"({summary.served_percent(index):.1f}%, "
            f"{summary.served_area_m2(index):.1f} m^2)"
        )
        for index, name in enumerate(summary.tx_names)
    )
    return " | ".join(entries)
