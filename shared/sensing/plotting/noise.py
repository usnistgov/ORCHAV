"""Reusable noise-sweep summaries from one metric record per independent seed.

Importing this module does not require Matplotlib. The input and aggregation
contract is documented on :func:`generate_noise_sweep_figures`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_METRICS = (
    ("recall", "Raw target-opportunity recall", (0, 1.05)),
    ("false_reports_per_cpi", "Raw false reports / evaluated CPI", None),
    ("range_rmse_m", "Conditional range RMSE (m)", None),
    ("velocity_rmse_m_s", "Conditional radial-velocity RMSE (m/s)", None),
    ("matched_count", "Matched target opportunities / seed", None),
)
_TRACKING_METRICS = (
    ("tracked_fraction", "Tracked target-opportunity fraction", (0, 1.05)),
    ("id_switches", "Identity switches / seed", None),
    ("max_gap_cpis", "Longest missed-target gap (CPIs)", None),
)


def _number(value: Any) -> float:
    return np.nan if value is None else float(value)


def _prepare_rows(sweep_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate seed-level input and preserve undefined conditional errors."""
    rows = []
    seen = set()
    for result in sweep_results:
        row = dict(result)
        row["regime"] = str(row.get("regime", "AWGN"))
        if "seed" not in row or "sample_snr_db" not in row:
            raise ValueError("Each noise-sweep row requires seed and sample_snr_db")
        snr = row["sample_snr_db"]
        if snr is not None and not np.isfinite(float(snr)):
            raise ValueError("sample_snr_db must be finite, or None for the clean control")
        row["sample_snr_db"] = None if snr is None else float(snr)
        key = (row["regime"], row["sample_snr_db"], row["seed"])
        if key in seen:
            raise ValueError("Expected one aggregate row per regime, SNR and independent seed")
        seen.add(key)
        if _number(row.get("matched_count")) == 0:
            row["range_rmse_m"] = None
            row["velocity_rmse_m_s"] = None
        if "confirmed_false_reports_per_cpi" in row:
            row.setdefault("tracked_false_reports_per_cpi", row["confirmed_false_reports_per_cpi"])
        rows.append(row)
    return rows


def _seed_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[float, ...]:
    """Return mean, min, max and finite seed count; never a frame-pooled CI."""
    values = np.asarray([_number(row.get(metric)) for row in rows])
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan, 0
    return float(values.mean()), float(values.min()), float(values.max()), int(values.size)


def generate_noise_sweep_figures(
    output_dir: Path,
    sweep_results: Sequence[Mapping[str, Any]],
    *,
    rd_examples: Sequence[Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Save overview, conditional-error, optional tracking and common-scale RD plots.

    Each row aggregates all evaluated CPIs within one independent noise seed
    and operating point. Required fields are ``seed`` and ``sample_snr_db``
    (finite dB, or ``None`` for a clean control). ``regime`` defaults to
    ``AWGN``. Duplicate ``(regime, sample_snr_db, seed)`` keys are rejected.
    Use one fixed clean-sample mean-power reference across the sweep. This is
    input complex-sample SNR, not post-FFT cell SNR or a CFAR threshold. Label
    noise-only controls explicitly when their variance uses a target recording's
    reference power, since they have no actual target-signal SNR.

    Metric fields are:

    * ``recall``: matched / eligible target opportunities.
    * ``false_reports_per_cpi``: unmatched detections / evaluated CPIs, a count
      rate rather than cell-level false-alarm probability.
    * ``range_rmse_m`` and ``velocity_rmse_m_s``: matched-only errors, accompanied
      by their support in ``matched_count``.
    * ``tracked_recall`` and ``tracked_false_reports_per_cpi``: optional metrics
      for confirmed reports. ``confirmed_false_reports_per_cpi`` is an alias
      for the latter.
    * ``tracked_fraction``, ``id_switches`` and ``max_gap_cpis``: optional
      continuity metrics, computed consistently across seeds.
    * ``per_target``: optional target-ID mapping to ``recall`` and
      ``tracked_recall`` values.

    Use ``None`` for undefined metrics. Zero matched count masks conditional
    errors. Dots show seed values, lines their arithmetic mean, and bands their
    observed minimum and maximum. These are descriptive variability, not
    confidence intervals. One seed provides no variability estimate. Seeds
    have equal weight, so mean conditional RMSE is not pooled match-level RMSE.

    Each optional RD example declares ``label``, nonnegative magnitude-squared
    ``power`` shaped ``(len(velocity_m_s), len(range_m))``, and those physical
    axes. Choose and identify representative seeds/CPIs yourself. Maps are not
    selected, averaged or transposed by this helper. Optional ``truth_range_m`` /
    ``truth_velocity_m_s`` and ``detected_range_m`` / ``detected_velocity_m_s``
    pairs supply finite one-dimensional marker arrays of equal length. Empty
    arrays omit the corresponding markers and legend. Truth centers can differ
    from reflecting surfaces.

    Maps share one maximum-power reference and a -80 to 0 dB scale. Optional
    ``display_range_m`` and ``display_velocity_m_s`` are finite increasing
    bounds affecting display only. The common reference still uses complete
    maps. At most three panels are placed in each row. Use compatible processing
    normalization across examples.

    Returns PNG paths under ``output_dir/figures``: ``noise_sweep_detection``,
    ``noise_sweep_errors`` and, when supported by inputs, ``noise_sweep_tracking``,
    ``noise_sweep_per_target`` and ``noise_sweep_rd_examples`` (all ``.png``).
    An empty sweep returns no files. Matplotlib is imported only for plotting.
    """
    rows = _prepare_rows(sweep_results)
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shared.sensing.plotting.style import PALETTE, STYLE

    output = Path(output_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    snrs = sorted({r["sample_snr_db"] for r in rows if r["sample_snr_db"] is not None})
    step = max(5.0, (snrs[-1] - snrs[0]) / max(len(snrs) - 1, 1)) if snrs else 5.0
    clean_x = snrs[-1] + step if snrs else 0.0
    has_clean = any(r["sample_snr_db"] is None for r in rows)

    def draw(ax: Any, metric: str, label: str, limits: Any, records: Any) -> None:
        available = False
        for regime_index, regime in enumerate(dict.fromkeys(r["regime"] for r in records)):
            groups = {}
            for row in records:
                if row["regime"] == regime:
                    x = clean_x if row["sample_snr_db"] is None else row["sample_snr_db"]
                    groups.setdefault(x, []).append(row)
            xs = sorted(groups)
            stats = np.array([_seed_summary(groups[x], metric) for x in xs])
            if not np.isfinite(stats[:, 0]).any():
                continue
            available = True
            # Missing values remain NaN so a zero-match condition breaks the line.
            (line,) = ax.plot(
                xs,
                stats[:, 0],
                "o-",
                label={
                    "clean": "Clean",
                    "signal_plus_noise": "Signal + noise",
                    "noise_only": "Noise only (reference variance)",
                }.get(regime.split(" · target ")[0], regime.split(" · target ")[0])
                + (" · " + regime.split(" · target ", 1)[1] if " · target " in regime else ""),
                color=PALETTE[regime_index % len(PALETTE)],
            )
            ax.fill_between(xs, stats[:, 1], stats[:, 2], color=line.get_color(), alpha=0.15)
            for x, group in groups.items():
                values = [_number(row.get(metric)) for row in group]
                ax.scatter([x] * len(values), values, s=12, alpha=0.45, color=line.get_color())
        if not available:
            ax.text(
                0.5, 0.5, "Undefined: no eligible observations", ha="center", transform=ax.transAxes
            )
        ax.set_ylabel(label)
        ax.set_xlabel("Input SNR, global clean-sample reference (dB)")
        ax.grid(True)
        if has_clean:
            ax.set_xticks(snrs + [clean_x], [f"{x:g}" for x in snrs] + ["Clean"])
        if limits:
            ax.set_ylim(*limits)
        else:
            ax.set_ylim(bottom=0)
        if available:
            ax.legend()

    def save_metrics(filename: str, title: str, metrics: Any, records: Any = None) -> None:
        columns = 2 if len(metrics) == 4 else len(metrics)
        nrows = 2 if len(metrics) == 4 else 1
        fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 4.4 * nrows), squeeze=False)
        for ax, (metric, label, limits) in zip(axes.flat, metrics, strict=True):
            draw(ax, metric, label, limits, rows if records is None else records)
        fig.suptitle(title)
        fig.text(
            0.5,
            0.01,
            "Lines: seed means · bands: observed seed min–max · dots: independent seeds (not a CI)",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        path = output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    with matplotlib.rc_context(STYLE):
        detection_metrics = list(_METRICS[:2])
        if any("tracked_recall" in row for row in rows):
            detection_metrics.insert(
                1, ("tracked_recall", "Tracked target-opportunity recall", (0, 1.05))
            )
        if any("tracked_false_reports_per_cpi" in row for row in rows):
            detection_metrics.append(
                ("tracked_false_reports_per_cpi", "Confirmed false reports / evaluated CPI", None)
            )
        save_metrics(
            "noise_sweep_detection.png",
            "Noise robustness: detection and false reports",
            detection_metrics,
        )
        save_metrics(
            "noise_sweep_errors.png", "Matched-only errors and their sample support", _METRICS[2:]
        )
        tracking = [metric for metric in _TRACKING_METRICS if any(metric[0] in row for row in rows)]
        if tracking:
            save_metrics("noise_sweep_tracking.png", "Tracking continuity under noise", tracking)
        target_rows = []
        for row in rows:
            for target_id, metrics in row.get("per_target", {}).items():
                target_rows.append(
                    dict(row) | metrics | {"regime": f"{row['regime']} · target {target_id}"}
                )
        if target_rows:
            save_metrics(
                "noise_sweep_per_target.png",
                "Per-target noise robustness",
                [
                    ("recall", "Raw target-opportunity recall", (0, 1.05)),
                    ("tracked_recall", "Confirmed target-opportunity recall", (0, 1.05)),
                ],
                target_rows,
            )
        if rd_examples:
            generated.append(_save_rd_examples(output, rd_examples, plt))
    return generated


def _save_rd_examples(output: Path, examples: Sequence[Mapping[str, Any]], plt: Any) -> Path:
    """Plot magnitude-squared RD power relative to one shared peak reference."""
    prepared = []
    for example in examples:
        power = np.asarray(example["power"], dtype=float)
        ranges = np.asarray(example["range_m"], dtype=float)
        velocities = np.asarray(example["velocity_m_s"], dtype=float)
        if power.shape != (velocities.size, ranges.size) or power.ndim != 2:
            raise ValueError("RD power must have shape (len(velocity_m_s), len(range_m))")
        if not power.size or not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("RD power must be finite, nonnegative and nonempty")
        if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(velocities)):
            raise ValueError("RD axes must be finite")
        for key in ("display_range_m", "display_velocity_m_s"):
            if key in example:
                limits = np.asarray(example[key], dtype=float)
                if (
                    limits.shape != (2,)
                    or not np.all(np.isfinite(limits))
                    or limits[0] >= limits[1]
                ):
                    raise ValueError(f"{key} must be two finite increasing display limits")
        for first, second in (
            ("truth_range_m", "truth_velocity_m_s"),
            ("detected_range_m", "detected_velocity_m_s"),
        ):
            if (first in example) != (second in example):
                raise ValueError(f"{first} and {second} must be supplied together")
            if first in example:
                x, y = np.asarray(example[first], float), np.asarray(example[second], float)
                if (
                    x.ndim != 1
                    or x.shape != y.shape
                    or not np.all(np.isfinite(x))
                    or not np.all(np.isfinite(y))
                ):
                    raise ValueError(
                        f"{first} and {second} must be equal finite one-dimensional coordinates"
                    )
        prepared.append((example, power, ranges, velocities))
    peak = max(float(power.max()) for _, power, _, _ in prepared)
    peak = max(peak, np.finfo(float).tiny)
    columns = min(3, len(prepared))
    rows = (len(prepared) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4.6 * rows), squeeze=False, constrained_layout=True
    )
    used_axes = list(axes.flat)[: len(prepared)]
    for unused in list(axes.flat)[len(prepared) :]:
        unused.set_visible(False)
    for ax, (example, power, ranges, velocities) in zip(used_axes, prepared, strict=True):
        db = 10 * np.log10(np.maximum(power / peak, 1e-8))
        im = ax.pcolormesh(ranges, velocities, db, vmin=-80, vmax=0, shading="auto")
        ax.set(title=str(example["label"]), xlabel="Range (m)", ylabel="Radial velocity (m/s)")
        if "display_range_m" in example:
            ax.set_xlim(example["display_range_m"])
        if "display_velocity_m_s" in example:
            ax.set_ylim(example["display_velocity_m_s"])
        if (
            "truth_range_m" in example
            and "truth_velocity_m_s" in example
            and np.asarray(example["truth_range_m"]).size
        ):
            ax.scatter(
                example["truth_range_m"],
                example["truth_velocity_m_s"],
                marker="x",
                color="white",
                label="Actor center",
            )
        if "detected_range_m" in example and np.asarray(example["detected_range_m"]).size:
            ax.scatter(
                example["detected_range_m"],
                example["detected_velocity_m_s"],
                marker="o",
                facecolors="none",
                edgecolors="#e34828",
                linewidths=1.8,
                s=65,
                label="Raw report",
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.colorbar(im, ax=used_axes, label="RD power (dB relative to shared peak)", shrink=0.85)
    fig.suptitle(
        "Representative range–Doppler maps · identical power scale"
        + (
            " · display region only"
            if any("display_range_m" in ex or "display_velocity_m_s" in ex for ex, *_ in prepared)
            else ""
        )
    )
    path = output / "noise_sweep_rd_examples.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
