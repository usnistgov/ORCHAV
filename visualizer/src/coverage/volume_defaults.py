"""Translate an ordinary scalar coverage view into volume controls."""

import numpy as np

from shared.coverage.schema import coverage_metric_base, resolve_coverage_tx_index
from shared.coverage.volume import ALL_TRANSMITTERS

from .analysis import coverage_metric_colormap


def initial_volume_controls(visualizer):
    """Inherit supported data selections without substituting a different metric.

    SINR, serving identity and other unsupported metrics keep explicit fallback
    controls and an explanation. Linear power/gain use the equivalent dB scale.
    A loaded bookmark overrides these opening defaults.
    """
    source = visualizer.coverage_data
    names = source["tx_names"]
    metric = source.get("metric_name", "")
    base, selector = coverage_metric_base(metric)
    result = dict(
        transmitter=names[0],
        quantity="path_gain_db",
        low=-130.0,
        high=-50.0,
        colormap="viridis",
        threshold=-95.0,
        opacity=0.65,
        height_index=None,
        note="",
    )
    result["height_index"] = getattr(
        visualizer, "coverage_height_index", source.get("_active_height_index")
    )
    result["opacity"] = float(getattr(visualizer, "coverage_opacity", 0.65))
    if base in ("best_rss_dbm", "best_path_loss_db"):
        result["transmitter"] = ALL_TRANSMITTERS if len(names) > 1 else names[0]
        quantity = base.removeprefix("best_")
    elif base in ("rss_dbm", "path_gain_db", "path_loss_db", "path_gain_linear", "rss_w"):
        result["transmitter"] = names[resolve_coverage_tx_index(selector, names, len(names))]
        quantity = {"path_gain_linear": "path_gain_db", "rss_w": "rss_dbm"}.get(base, base)
    else:
        if metric:
            result["note"] = (
                f"The ordinary metric '{metric}' is not available in this volume inspector. "
                f"Showing path gain for {names[0]}; choose a supported quantity below."
            )
        return result
    result.update(quantity=quantity, colormap=coverage_metric_colormap(quantity))
    if quantity == "path_loss_db":
        result.update(low=50.0, high=130.0, threshold=95.0)
    bounds = np.asarray([source.get("value_min", np.nan), source.get("value_max", np.nan)])
    if base in ("path_gain_linear", "rss_w"):
        with np.errstate(divide="ignore", invalid="ignore"):
            bounds = 10 * np.log10(bounds) + (30 if base == "rss_w" else 0)
        result["note"] = "The ordinary linear power/gain scale is shown in equivalent dB units."
    if np.isfinite(bounds).all() and bounds[0] < bounds[1]:
        result.update(low=float(bounds[0]), high=float(bounds[1]))
    return result
