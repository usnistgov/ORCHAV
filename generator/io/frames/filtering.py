"""Select Sionna paths before compact frame geometry is materialized.

The generator first identifies valid Sionna path indices for each TX/RX pair.
This module applies power and Top-K policies to those indices and to the small
per-path metric vectors. Geometry and material arrays are expanded only for the
retained paths, so discarded paths do not cross the materialization boundary.
"""

from pathlib import Path
from typing import Any

import numpy as np

from shared.logging import get_logger

logger = get_logger(__name__)

SECONDS_TO_NANOSECONDS = 1e9

# Default path filtering configuration. ``None`` disables each optional stage.
DEFAULT_PATH_FILTER_CONFIG = {
    # Keep paths within this many dB of the strongest path.
    "relative_threshold_db": None,
    # Keep paths whose path loss does not exceed this dB limit.
    "max_path_loss_db": None,
    # Keep at most this many strongest paths per TX-RX pair.
    "max_paths_per_pair": None,
    # Log filtering statistics when a filter is active.
    "log_filtering_stats": True,
    # Generate a before/after diagnostic plot.
    "generate_diagnostic": False,
}


def select_path_indices_by_power(
    valid_path_indices_by_pair: list[np.ndarray],
    metrics_by_pair: dict[str, list[np.ndarray]] | None,
    filter_config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    bandwidth_hz: float | None = None,
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]] | None]:
    """Return retained raw path indices and aligned metric vectors.

    ``valid_path_indices_by_pair`` maps each compact per-pair metric row back
    to its raw Sionna path index. Selection preserves that original order even
    when Top-K ranking is based on path loss.
    """
    config = filter_config or DEFAULT_PATH_FILTER_CONFIG
    relative_threshold_db = config.get("relative_threshold_db")
    max_path_loss_db = config.get("max_path_loss_db")
    max_paths_per_pair = config.get("max_paths_per_pair")
    log_stats = config.get("log_filtering_stats", True)
    generate_diagnostic = config.get("generate_diagnostic", False)

    metrics = metrics_by_pair or {}

    if relative_threshold_db is None and max_path_loss_db is None and max_paths_per_pair is None:
        return valid_path_indices_by_pair, metrics_by_pair

    path_loss_arrays = metrics.get("all_pair_path_loss_db", [])
    if not path_loss_arrays:
        logger.warning(
            "Path filtering requested but no path_loss_db available. Skipping filtering."
        )
        return valid_path_indices_by_pair, metrics_by_pair

    metrics_before = None
    if generate_diagnostic and output_dir is not None:
        metrics_before = {
            key: [np.array(values, copy=True) for values in columns]
            for key, columns in metrics.items()
        }

    retained_indices: list[np.ndarray] = []
    filtered_metrics: dict[str, list[np.ndarray]] = {key: [] for key in metrics}

    total_paths_before = 0
    total_paths_after = 0
    num_pairs = len(valid_path_indices_by_pair)

    for pair_idx in range(num_pairs):
        raw_indices = np.asarray(valid_path_indices_by_pair[pair_idx], dtype=np.int64)
        path_loss = (
            np.asarray(path_loss_arrays[pair_idx])
            if pair_idx < len(path_loss_arrays)
            else np.empty((0,), dtype=np.float32)
        )
        num_paths = len(raw_indices)
        total_paths_before += num_paths

        if num_paths == 0 or len(path_loss) == 0:
            retained_indices.append(raw_indices)
            total_paths_after += num_paths
            for key in filtered_metrics:
                if pair_idx < len(metrics[key]):
                    filtered_metrics[key].append(np.asarray(metrics[key][pair_idx]))
                else:
                    filtered_metrics[key].append(np.empty((0,), dtype=np.float32))
            continue

        keep_mask = np.ones(num_paths, dtype=bool)
        valid_pl_mask = (
            ~np.isnan(path_loss) if len(path_loss) == num_paths else np.ones(num_paths, dtype=bool)
        )
        num_valid_pl = int(np.sum(valid_pl_mask))

        if num_valid_pl == 0:
            logger.warning(
                "[FILTER] All path_loss values are NaN for pair %d. " "Keeping all %d paths.",
                pair_idx,
                num_paths,
            )
        else:
            if max_path_loss_db is not None and len(path_loss) == num_paths:
                keep_mask &= (path_loss <= max_path_loss_db) & valid_pl_mask
            if relative_threshold_db is not None and len(path_loss) == num_paths:
                kept_losses = path_loss[keep_mask & valid_pl_mask]
                if len(kept_losses) > 0:
                    min_loss = np.nanmin(kept_losses)
                    threshold = min_loss + relative_threshold_db
                    keep_mask &= (path_loss <= threshold) & valid_pl_mask

        if max_paths_per_pair is not None and np.sum(keep_mask) > max_paths_per_pair:
            kept_indices = np.where(keep_mask)[0]
            if len(path_loss) == num_paths:
                kept_losses = path_loss[kept_indices]
                sorted_order = np.argsort(kept_losses)
                top_k_indices = kept_indices[sorted_order[:max_paths_per_pair]]
                keep_mask = np.zeros(num_paths, dtype=bool)
                keep_mask[top_k_indices] = True
            else:
                logger.warning(
                    "Top-K filter: path_loss not available (got %d, need %d). "
                    "Taking first %d paths instead.",
                    len(path_loss),
                    num_paths,
                    max_paths_per_pair,
                )
                keep_mask = np.zeros(num_paths, dtype=bool)
                keep_mask[kept_indices[:max_paths_per_pair]] = True

        num_kept = int(np.sum(keep_mask))
        total_paths_after += num_kept
        retained_indices.append(raw_indices[keep_mask])
        for key in filtered_metrics:
            if pair_idx < len(metrics[key]):
                arr = np.asarray(metrics[key][pair_idx])
                if len(arr) == num_paths:
                    filtered_metrics[key].append(arr[keep_mask])
                else:
                    filtered_metrics[key].append(arr)
            else:
                filtered_metrics[key].append(np.empty((0,), dtype=np.float32))

    if log_stats:
        if total_paths_before > 0:
            reduction_pct = 100.0 * (1.0 - total_paths_after / total_paths_before)
            logger.info(
                "[PATH FILTER] Reduced paths from %s to %s (%.1f%% reduction)",
                f"{total_paths_before:,}",
                f"{total_paths_after:,}",
                reduction_pct,
            )
            if max_path_loss_db is not None:
                logger.info("  - Absolute threshold: path_loss <= %s dB", max_path_loss_db)
            if relative_threshold_db is not None:
                logger.info("  - Relative threshold: %s dB from strongest", relative_threshold_db)
            if max_paths_per_pair is not None:
                logger.info("  - Max paths per pair: %s", f"{max_paths_per_pair:,}")
            avg_paths = total_paths_after / num_pairs if num_pairs > 0 else 0
            logger.info("  - Average paths per pair after filtering: %.1f", avg_paths)
        else:
            logger.warning("[PATH FILTER] No paths to filter (total_paths_before=0)")

    if generate_diagnostic and output_dir is not None and metrics_before is not None:
        try:
            diagnostic_path = Path(output_dir) / "path_filter_diagnostic.png"
            generate_path_filter_diagnostic(
                all_pair_metrics_before=metrics_before,
                all_pair_metrics_after=filtered_metrics,
                filter_config=config,
                output_path=diagnostic_path,
                pair_idx=0,  # Visualize first pair
                bandwidth_hz=bandwidth_hz,
            )
        except (ValueError, TypeError, IndexError, OSError):
            logger.warning("Failed to generate path filter diagnostic", exc_info=True)

    return retained_indices, filtered_metrics


def generate_path_filter_diagnostic(
    all_pair_metrics_before: dict[str, Any],
    all_pair_metrics_after: dict[str, Any],
    filter_config: dict[str, Any],
    output_path: Path,
    pair_idx: int = 0,
    bandwidth_hz: float | None = None,
) -> None:
    """Write a path-filtering diagnostic plot for one TX-RX pair."""
    import matplotlib.pyplot as plt

    relative_threshold_db = filter_config.get("relative_threshold_db")
    max_path_loss_db = filter_config.get("max_path_loss_db")
    max_paths_per_pair = filter_config.get("max_paths_per_pair")

    pl_before_list = all_pair_metrics_before.get("all_pair_path_loss_db", [])
    pl_after_list = all_pair_metrics_after.get("all_pair_path_loss_db", [])
    delays_before_list = all_pair_metrics_before.get("all_pair_delays_ns", [])
    delays_after_list = all_pair_metrics_after.get("all_pair_delays_ns", [])

    logger.debug(
        f"[DIAG] pl_before_list type: {type(pl_before_list)}, len: {len(pl_before_list) if hasattr(pl_before_list, '__len__') else 'N/A'}"
    )
    logger.debug(
        f"[DIAG] pl_after_list type: {type(pl_after_list)}, len: {len(pl_after_list) if hasattr(pl_after_list, '__len__') else 'N/A'}"
    )
    logger.debug(f"[DIAG] pair_idx: {pair_idx}, type: {type(pair_idx)}")
    if pl_before_list and len(pl_before_list) > 0:
        logger.debug(f"[DIAG] pl_before_list[0] type: {type(pl_before_list[0])}")
    if pl_after_list and len(pl_after_list) > 0:
        logger.debug(f"[DIAG] pl_after_list[0] type: {type(pl_after_list[0])}")

    if not isinstance(pl_before_list, list):
        pl_before_list = []
    if not isinstance(pl_after_list, list):
        pl_after_list = []
    if not isinstance(delays_before_list, list):
        delays_before_list = []
    if not isinstance(delays_after_list, list):
        delays_after_list = []

    if int(pair_idx) >= len(pl_before_list) or int(pair_idx) >= len(pl_after_list):
        logger.warning(f"Pair index {pair_idx} out of range for path filter diagnostic")
        return

    pl_before = np.asarray(pl_before_list[pair_idx], dtype=np.float64)
    pl_after = np.asarray(pl_after_list[pair_idx], dtype=np.float64)
    delays_before = (
        np.asarray(delays_before_list[pair_idx], dtype=np.float64)
        if pair_idx < len(delays_before_list)
        else np.array([])
    )
    delays_after = (
        np.asarray(delays_after_list[pair_idx], dtype=np.float64)
        if pair_idx < len(delays_after_list)
        else np.array([])
    )

    valid_before = ~np.isnan(pl_before)
    valid_after = ~np.isnan(pl_after) if len(pl_after) > 0 else np.array([], dtype=bool)
    pl_before_valid = pl_before[valid_before] if len(pl_before) > 0 else np.array([])
    pl_after_valid = pl_after[valid_after] if len(pl_after) > 0 else np.array([])

    n_before = len(pl_before_valid)
    n_after = len(pl_after_valid)

    if n_before == 0:
        logger.warning("No valid paths before filtering for diagnostic")
        return

    has_delays = len(delays_before) > 0 and len(delays_after) > 0
    n_rows = 2 if has_delays else 1
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    ax_pl_before = axes[0, 0]
    ax_pl_after = axes[0, 1]

    pl_min = min(np.nanmin(pl_before_valid), np.nanmin(pl_after_valid) if n_after > 0 else np.inf)
    pl_max = np.nanmax(pl_before_valid)
    bins = np.linspace(pl_min - 5, pl_max + 5, 60)

    ax_pl_before.hist(
        pl_before_valid, bins=bins, color="blue", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax_pl_before.axvline(
        np.nanmin(pl_before_valid),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Strongest: {np.nanmin(pl_before_valid):.1f} dB",
    )

    if max_path_loss_db is not None:
        ax_pl_before.axvline(
            max_path_loss_db,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Absolute: {max_path_loss_db} dB",
        )
    if relative_threshold_db is not None:
        rel_threshold = np.nanmin(pl_before_valid) + relative_threshold_db
        ax_pl_before.axvline(
            rel_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Relative: +{relative_threshold_db} dB = {rel_threshold:.1f} dB",
        )

    ax_pl_before.set_xlabel("Path Loss (dB)")
    ax_pl_before.set_ylabel("Count")
    ax_pl_before.set_title(f"BEFORE Filtering\n{n_before:,} paths", fontweight="bold")
    ax_pl_before.legend(loc="upper right", fontsize=9)
    ax_pl_before.grid(True, alpha=0.3)

    if n_after > 0:
        ax_pl_after.hist(
            pl_after_valid, bins=bins, color="green", alpha=0.7, edgecolor="black", linewidth=0.5
        )
        ax_pl_after.axvline(
            np.nanmin(pl_after_valid),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Strongest: {np.nanmin(pl_after_valid):.1f} dB",
        )
    else:
        ax_pl_after.text(
            0.5,
            0.5,
            "No paths remaining",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax_pl_after.transAxes,
        )

    ax_pl_after.set_xlabel("Path Loss (dB)")
    ax_pl_after.set_ylabel("Count")
    reduction_pct = 100.0 * (1.0 - n_after / n_before) if n_before > 0 else 0
    ax_pl_after.set_title(
        f"AFTER Filtering\n{n_after:,} paths ({reduction_pct:.1f}% removed)", fontweight="bold"
    )
    ax_pl_after.legend(loc="upper right", fontsize=9)
    ax_pl_after.grid(True, alpha=0.3)
    ax_pl_after.set_xlim(ax_pl_before.get_xlim())

    if has_delays:
        ax_pdp_before = axes[1, 0]
        ax_pdp_after = axes[1, 1]

        # Path loss in dB maps to linear power as 10 ** (-loss / 10).
        power_before = 10.0 ** (-pl_before[valid_before] / 10.0)
        delays_before_valid = (
            delays_before[valid_before] if len(delays_before) == len(pl_before) else delays_before
        )

        power_after = 10.0 ** (-pl_after[valid_after] / 10.0) if n_after > 0 else np.array([])
        delays_after_valid = (
            delays_after[valid_after]
            if len(delays_after) == len(pl_after) and n_after > 0
            else np.array([])
        )

        try:
            bw_hz = float(bandwidth_hz) if bandwidth_hz is not None else 0
        except (ValueError, TypeError):
            bw_hz = 0
        if bw_hz > 0:
            bin_width_ns = SECONDS_TO_NANOSECONDS / bw_hz
        else:
            bin_width_ns = 1.0  # Use 1 ns when channel bandwidth is unavailable.

        max_delay = max(
            np.nanmax(delays_before_valid),
            np.nanmax(delays_after_valid) if len(delays_after_valid) > 0 else 0,
        )
        delay_bins = np.arange(0, max_delay + bin_width_ns, bin_width_ns)

        # A PDP bin contains the sum of all path powers in that delay interval.
        pdp_before = np.zeros(len(delay_bins) - 1)
        for d, p in zip(delays_before_valid, power_before):
            if np.isfinite(d) and np.isfinite(p):
                bin_idx = int(d / bin_width_ns)
                if 0 <= bin_idx < len(pdp_before):
                    pdp_before[bin_idx] += p

        pdp_after = np.zeros(len(delay_bins) - 1)
        if len(delays_after_valid) > 0:
            for d, p in zip(delays_after_valid, power_after):
                if np.isfinite(d) and np.isfinite(p):
                    bin_idx = int(d / bin_width_ns)
                    if 0 <= bin_idx < len(pdp_after):
                        pdp_after[bin_idx] += p

        pdp_before_db = 10.0 * np.log10(pdp_before + 1e-20)
        pdp_after_db = 10.0 * np.log10(pdp_after + 1e-20)

        delay_centers = (delay_bins[:-1] + delay_bins[1:]) / 2

        ax_pdp_before.stem(delay_centers, pdp_before_db, linefmt="b-", markerfmt="bo", basefmt="k-")
        ax_pdp_before.set_xlabel("Delay (ns)")
        ax_pdp_before.set_ylabel("Power (dB)")
        ax_pdp_before.set_title(
            f"PDP Before Filtering\n(bin width: {bin_width_ns:.2f} ns)", fontweight="bold"
        )
        ax_pdp_before.grid(True, alpha=0.3)
        ax_pdp_before.set_ylim(
            [
                (
                    pdp_before_db[pdp_before_db > -200].min() - 10
                    if np.any(pdp_before_db > -200)
                    else -200
                ),
                pdp_before_db.max() + 5,
            ]
        )

        ax_pdp_after.stem(delay_centers, pdp_after_db, linefmt="g-", markerfmt="go", basefmt="k-")
        ax_pdp_after.set_xlabel("Delay (ns)")
        ax_pdp_after.set_ylabel("Power (dB)")
        ax_pdp_after.set_title(
            f"PDP After Filtering\n(bin width: {bin_width_ns:.2f} ns)", fontweight="bold"
        )
        ax_pdp_after.grid(True, alpha=0.3)
        ax_pdp_after.set_xlim(ax_pdp_before.get_xlim())
        ax_pdp_after.set_ylim(ax_pdp_before.get_ylim())

    filter_settings = []
    if max_path_loss_db is not None:
        filter_settings.append(f"Absolute: {max_path_loss_db} dB")
    if relative_threshold_db is not None:
        filter_settings.append(f"Relative: {relative_threshold_db} dB")
    if max_paths_per_pair is not None:
        filter_settings.append(f"Top-K: {max_paths_per_pair}")
    filter_str = " | ".join(filter_settings) if filter_settings else "No filtering"

    fig.suptitle(
        f"Path Filter Diagnostic (Pair {pair_idx})\n{filter_str}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Path filter diagnostic saved to: {output_path}")
