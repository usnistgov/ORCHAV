"""Canonical sensing figure functions (shared by generator and postprocessor).

Produces PNG plots using the matplotlib Agg backend:
  1. Range profile (power vs range) with CFAR detection markers
  2. Range-Doppler heatmap with CFAR detections and GT overlay
  3. GT evolution (range + radial speed vs frame index)
  4. Micro-Doppler spectrogram (time vs Doppler)
  5. Detection summary (TP/FP/FN + RMSE per frame)

Style: DPI 120, ``bbox_inches="tight"``, Agg backend.

Set ``SENSING_FIGURES_NO_TITLE=1`` in the environment to suppress per-axes
titles (useful when the caller supplies an external caption, e.g. in LaTeX
documents).
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from shared.extensions.sensing import assert_supported_sensing_schema
from shared.logging import get_logger
from shared.sensing.angle import RangeDopplerAngleProduct, select_range_doppler_angle_product
from shared.sensing.arrays import broadside_azimuth_projection
from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth
from shared.sensing.plotting.style import (
    AMBER,
    BLUE,
    ORANGE,
    PURPLE,
    RED,
    TEAL,
    sensing_figure_style,
)
from shared.sensing.products import select_product_by_contract
from shared.sensing.signal_processing import (
    MicroDopplerResult,
    detection_velocity_display,
    doppler_axis_to_velocity_m_s,
)

logger = get_logger(__name__)

SUMMARY_FIGURE_DPI = 160


def _require_numeric_axis(
    coordinate: np.ndarray | list[Any],
    *,
    name: str,
    size: int,
) -> np.ndarray:
    """Return one exact producer-authored coordinate vector or fail closed."""

    axis = np.asarray(coordinate)
    if axis.ndim != 1 or axis.shape != (size,) or not np.issubdtype(axis.dtype, np.number):
        raise ValueError(
            f"Typed sensing axis {name!r} must be a one-dimensional numeric array "
            f"with shape {(size,)}, got {axis.shape}"
        )
    if np.iscomplexobj(axis) or not np.all(np.isfinite(axis)):
        raise ValueError(f"Typed sensing axis {name!r} must contain finite real coordinates")
    axis = axis.astype(np.float64, copy=False)
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"Typed sensing axis {name!r} must be strictly increasing")
    return axis


def _require_single_pair_product(
    sensing_data: Mapping[str, Any],
    *,
    kind: str,
    axis_names: tuple[str, ...],
    axis_units: tuple[str, ...],
    representations: tuple[str, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], Mapping[str, Any]]:
    """Select and validate one typed schema-v4 product without key or shape inference."""

    assert_supported_sensing_schema(sensing_data)
    _data_key, values, coordinates, descriptor = select_product_by_contract(
        sensing_data,
        kind=kind,
        variant="primary",
        axis_names=axis_names,
        representations=representations,
    )
    axes = descriptor.get("axes")
    if not isinstance(axes, list):
        raise ValueError(f"Typed sensing product {kind!r} has no ordered axis descriptors")
    declared_units = tuple(axis.get("unit") if isinstance(axis, Mapping) else None for axis in axes)
    if declared_units != axis_units:
        raise ValueError(
            f"Typed sensing product {kind!r} must declare axis units {axis_units!r}, "
            f"got {declared_units!r}"
        )
    if len(coordinates) != len(axis_names):
        raise ValueError(f"Typed sensing product {kind!r} coordinate count is inconsistent")

    pair_axis = coordinates[0]
    pair_ids = descriptor.get("pair_ids")
    if (
        not isinstance(pair_axis, list)
        or len(pair_axis) != 1
        or values.ndim != len(axis_names)
        or values.shape[0] != 1
        or not isinstance(pair_ids, list)
        or len(pair_ids) != 1
    ):
        raise ValueError(
            f"Typed sensing product {kind!r} requires exactly one explicit pair; "
            "multi-pair summary plotting is unsupported"
        )
    pair = pair_axis[0]
    temporal = sensing_data.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(pair, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError(f"Typed sensing product {kind!r} has no explicit acquisition pair")
    for field in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
        if pair.get(field) != acquisition.get(field):
            raise ValueError(
                f"Typed sensing product {kind!r} pair-axis {field} disagrees with "
                "temporal acquisition identity"
            )
    if pair_ids[0] != pair.get("pair_id"):
        raise ValueError(
            f"Typed sensing product {kind!r} descriptor pair_ids disagree with pair_axis"
        )

    numeric_coordinates = tuple(
        _require_numeric_axis(coordinate, name=name, size=int(values.shape[index]))
        for index, (coordinate, name) in enumerate(
            zip(coordinates[1:], axis_names[1:], strict=True), start=1
        )
    )
    return np.asarray(values[0]), numeric_coordinates, descriptor


def _require_resolved_plot_config(sensing_data: Mapping[str, Any]) -> tuple[str, float]:
    """Return range mode and carrier from the canonical resolved configuration."""

    config = sensing_data.get("config_resolved")
    if not isinstance(config, Mapping):
        raise ValueError("Schema-v4 sensing plotting requires config_resolved")
    range_mode = config.get("range_mode")
    if not isinstance(range_mode, str) or not range_mode:
        raise ValueError("Schema-v4 sensing plotting requires config_resolved.range_mode")
    carrier_frequency = config.get("carrier_freq_hz")
    try:
        carrier_hz = float(carrier_frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Schema-v4 sensing plotting requires finite config_resolved.carrier_freq_hz"
        ) from exc
    if not np.isfinite(carrier_hz) or carrier_hz <= 0.0:
        raise ValueError(
            "Schema-v4 sensing plotting requires positive finite " "config_resolved.carrier_freq_hz"
        )
    return range_mode, carrier_hz


def _range_axis_label(range_mode: str) -> str:
    """Return the physical label associated with the persisted range coordinates."""

    if range_mode.lower().startswith("mono"):
        return "Monostatic Range (m)"
    return "Bistatic Path Length (m)"


def _canonical_truth_points(sensing_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return identity-keyed target overlays from the canonical truth snapshot."""

    stored = extract_recorded_evaluation_truth(sensing_data)
    if stored is None or stored.get("_evaluation_unavailable_reason"):
        return []
    target_ids = stored["target_ids"]
    target_ranges = stored["target_ranges"]
    target_speeds = stored["target_speeds"]
    velocity_valid = stored["target_velocity_valid"]
    if not (len(target_ids) == len(target_ranges) == len(target_speeds) == len(velocity_valid)):
        raise ValueError("Canonical sensing truth target fields are not aligned by target_id")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Canonical sensing truth contains duplicate target_id values")
    return [
        {"target_id": target_id, "range_m": target_range, "velocity_m_s": float(target_speed)}
        for target_id, target_range, target_speed, valid in zip(
            target_ids,
            target_ranges,
            target_speeds,
            velocity_valid,
            strict=True,
        )
        if valid
    ]


def _suppress_titles() -> bool:
    """Return True when ``SENSING_FIGURES_NO_TITLE`` requests title suppression."""

    return os.environ.get("SENSING_FIGURES_NO_TITLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _swap_world_xy_axes() -> bool:
    """Return True when ``SENSING_FIGURES_SWAP_XY`` requests the world-xy axes to be swapped.

    Default is off: the plot shows World X on the horizontal axis and World Y
    on the vertical axis. Opt in when the trajectory of interest spans the Y
    dimension more than the X dimension, to get a wide-short figure that fits
    a single-column paper layout.
    """

    return os.environ.get("SENSING_FIGURES_SWAP_XY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------------
# 1. Range profile
# ---------------------------------------------------------------------------


@sensing_figure_style
def create_range_profile_figure(
    sensing_data: Dict[str, Any],
    output_path: Path,
    *,
    range_xlim: tuple[float, float] | None = None,
    zoom_to_data: bool = False,
) -> Path:
    """Plot one canonical schema-v4 range-profile product and its exact coordinates.

    Args:
        sensing_data: Complete schema-v4 sensing dictionary from the generator.
        output_path: Destination PNG path.
        range_xlim: Optional display limits; the stored samples are unchanged.
        zoom_to_data: Show the nonzero response and reported ranges using the
            same signal bounds as the range-Doppler summary.

    Returns:
        The written file path.
    """
    output_path = Path(output_path)

    range_mode, _carrier_hz = _require_resolved_plot_config(sensing_data)
    profile, coordinates, descriptor = _require_single_pair_product(
        sensing_data,
        kind="range_profile",
        axis_names=("pair", "range"),
        axis_units=("1", "m"),
        representations=("power_linear",),
    )
    if descriptor.get("representation") != "power_linear":
        raise ValueError("Range-profile summary requires representation 'power_linear'")
    reported_ranges = sorted(
        {
            float(det["range_m"])
            for det in sensing_data.get("detections") or []
            if det.get("range_m") is not None and np.isfinite(float(det["range_m"]))
        }
    )
    return create_range_profile_plot(
        profile,
        coordinates[0],
        output_path,
        range_mode=range_mode,
        reported_ranges=reported_ranges,
        range_resolution_m=sensing_data["config_resolved"].get("range_resolution_m"),
        range_xlim=range_xlim,
        zoom_to_data=zoom_to_data,
    )


@sensing_figure_style
def create_range_profile_plot(
    profile: np.ndarray,
    range_axis_m: np.ndarray,
    output_path: Path,
    *,
    range_mode: str,
    description: str = "last acquisition sample",
    reported_ranges: Sequence[float] = (),
    range_resolution_m: float | None = None,
    range_xlim: tuple[float, float] | None = None,
    zoom_to_data: bool = False,
) -> Path:
    """Draw the standard range profile from explicit real power and coordinates.

    The schema-v4 wrapper validates saved acquisition identity before calling
    this renderer. Analytical controls may call it directly with their actual
    coordinates and an explicit description, without inventing an acquisition,
    carrier or detector report. No estimate is inferred from the response peak.
    """
    output_path = Path(output_path)
    profile = np.asarray(profile)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("Range-profile power must be a nonempty one-dimensional array")
    range_axis = _require_numeric_axis(range_axis_m, name="range", size=profile.size)
    reported_ranges = sorted(set(float(value) for value in reported_ranges))
    if not np.all(np.isfinite(reported_ranges)):
        raise ValueError("Reported ranges must be finite")
    if range_resolution_m is not None and (
        not np.isfinite(range_resolution_m) or range_resolution_m <= 0
    ):
        raise ValueError("Range resolution must be finite and positive")
    if np.iscomplexobj(profile) or not np.issubdtype(profile.dtype, np.number):
        raise ValueError("Range-profile power_linear values must be real numeric samples")
    profile = profile.astype(np.float64, copy=False)
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise ValueError("Range-profile power_linear values must be finite and nonnegative")
    range_label = _range_axis_label(range_mode)
    power_db = 10.0 * np.log10(np.maximum(profile, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range_axis, power_db, ".-", linewidth=0.8, markersize=3, color=BLUE)
    ax.set_xlabel(range_label)
    ax.set_ylabel("Power (dB)")
    if not _suppress_titles():
        ax.set_title(f"Range response · {description}")
    if np.any(profile > 0):
        peak_db = float(power_db.max())
        ax.set_ylim(peak_db - 80, peak_db + 3)
    ax.grid(True, alpha=0.3)

    for index, value in enumerate(reported_ranges):
        label = f"Reported range: {value:.3f} m" if index < 6 else None
        if index == 6:
            label = f"{len(reported_ranges) - 6} additional reported ranges"
        ax.axvline(
            value,
            color=ORANGE,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            label=label,
        )
    if reported_ranges:
        ax.legend(fontsize=8, loc="upper right")

    if range_xlim is not None:
        limits = np.asarray(range_xlim, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            plt.close(fig)
            raise ValueError("range_xlim must contain two finite, strictly increasing values")
    elif zoom_to_data and np.any(profile > 0):
        from shared.sensing.plotting.limits import range_doppler_zoom_limits

        limits = range_doppler_zoom_limits(
            power_db[np.newaxis, :],
            range_axis,
            np.asarray([0.0]),
            extra_points=[(value, 0.0) for value in reported_ranges],
        )
        range_xlim = limits[0] if limits is not None else None
    if range_xlim is None:
        range_xlim = (float(range_axis[0]), float(range_axis[-1]))
        if range_xlim[0] == range_xlim[1]:
            range_xlim = (range_xlim[0] - 0.5, range_xlim[1] + 0.5)
    ax.set_xlim(range_xlim)

    # Acquisition metadata stays outside the signal plotting area.
    range_res = range_resolution_m
    if range_res:
        ax.text(
            0.0,
            1.025,
            f"Range resolution: {range_res:.2f} m",
            transform=ax.transAxes,
            fontsize=8,
            color="#5d687b",
            verticalalignment="bottom",
        )
        if not _suppress_titles():
            ax.set_title(f"Range response · {description}", pad=28)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Range profile figure saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Range–Doppler map
# ---------------------------------------------------------------------------


@sensing_figure_style
def create_range_doppler_figure(
    sensing_data: Dict[str, Any],
    output_path: Path,
    *,
    range_xlim: tuple[float, float] | None = None,
    velocity_ylim: tuple[float, float] | None = None,
    interpolation: str = "nearest",
    power_limits_db: tuple[float, float] | None = None,
    zoom_to_data: bool | None = None,
    result_label: str = "last frame",
) -> Path:
    """Plot one canonical typed RD product with exact coordinates and truth IDs."""

    range_mode, carrier_freq = _require_resolved_plot_config(sensing_data)
    rd_values, coordinates, descriptor = _require_single_pair_product(
        sensing_data,
        kind="range_doppler",
        axis_names=("pair", "doppler_frequency", "range"),
        axis_units=("1", "Hz", "m"),
        representations=("complex_voltage", "power_linear"),
    )
    representation = descriptor.get("representation")
    if representation == "complex_voltage":
        if not np.iscomplexobj(rd_values):
            raise ValueError("complex_voltage RD products must use a complex numeric dtype")
        rd_power = np.abs(rd_values) ** 2
    elif representation == "power_linear":
        if np.iscomplexobj(rd_values) or not np.issubdtype(rd_values.dtype, np.number):
            raise ValueError("power_linear RD products must use a real numeric dtype")
        rd_power = rd_values.astype(np.float64, copy=False)
    else:  # Defensive: selection already restricts the representation.
        raise ValueError(f"Unsupported typed RD representation {representation!r}")
    if not np.all(np.isfinite(rd_power)) or np.any(rd_power < 0.0):
        raise ValueError("RD power values must be finite and nonnegative")

    doppler_axis, range_axis = coordinates
    return create_range_doppler_array_figure(
        power_linear=rd_power,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        range_mode=range_mode,
        carrier_freq_hz=carrier_freq,
        output_path=output_path,
        detections=sensing_data.get("detections") or (),
        target_truth=_canonical_truth_points(sensing_data),
        range_xlim=range_xlim,
        velocity_ylim=velocity_ylim,
        interpolation=interpolation,
        power_limits_db=power_limits_db,
        zoom_to_data=zoom_to_data,
        range_axis_label=_range_axis_label(range_mode),
        plot_title=f"Range–Velocity Map ({result_label})",
        display_config=sensing_data["config_resolved"],
    )


@sensing_figure_style
def create_range_doppler_array_figure(
    *,
    power_linear: np.ndarray,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    range_mode: str,
    carrier_freq_hz: float,
    output_path: Path,
    detections: Sequence[Mapping[str, Any]] = (),
    target_truth: Sequence[Mapping[str, Any]] = (),
    range_xlim: tuple[float, float] | None = None,
    velocity_ylim: tuple[float, float] | None = None,
    interpolation: str = "nearest",
    power_limits_db: tuple[float, float] | None = None,
    zoom_to_data: bool | None = None,
    range_axis_label: str | None = None,
    plot_title: str = "Range–Velocity Map",
    display_config: Mapping[str, Any] | None = None,
) -> Path:
    """Plot explicitly represented RD power with exact caller-supplied coordinates.

    This low-level API performs no dtype inference and reconstructs no axes.
    Callers must supply linear power, exact range/Doppler coordinates, and
    identity-keyed truth points if an overlay is desired. ``power_limits_db``
    optionally fixes a shared display scale in 10*log10(stored linear power),
    not dBm or calibrated received power. Omit it for the peak-minus-60 dB
    scale. Values outside the chosen scale are clipped for display only.
    """

    output_path = Path(output_path)
    rd_map = np.asarray(power_linear)
    if rd_map.ndim != 2 or np.iscomplexobj(rd_map) or not np.issubdtype(rd_map.dtype, np.number):
        raise ValueError("power_linear must be a two-dimensional real numeric array")
    rd_map = rd_map.astype(np.float64, copy=False)
    if not np.all(np.isfinite(rd_map)) or np.any(rd_map < 0.0):
        raise ValueError("power_linear must contain finite nonnegative values")
    doppler_axis = _require_numeric_axis(
        doppler_axis_hz,
        name="doppler_frequency",
        size=rd_map.shape[0],
    )
    range_axis = _require_numeric_axis(range_axis_m, name="range", size=rd_map.shape[1])
    try:
        carrier_freq = float(carrier_freq_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("carrier_freq_hz must be a positive finite number") from exc
    if not np.isfinite(carrier_freq) or carrier_freq <= 0.0:
        raise ValueError("carrier_freq_hz must be a positive finite number")
    if not isinstance(range_mode, str) or not range_mode:
        raise ValueError("range_mode must be an explicit non-empty string")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("RD interpolation must be 'nearest' or 'bilinear'")

    rd_db = 10.0 * np.log10(np.maximum(rd_map, 1e-30))
    if power_limits_db is None:
        vmin, vmax = float(rd_db.max()) - 60.0, float(rd_db.max())
    else:
        limits = np.asarray(power_limits_db, dtype=np.float64)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            raise ValueError("power_limits_db must contain two finite, strictly increasing values")
        vmin, vmax = (float(value) for value in limits)
    velocity_display = detection_velocity_display(range_mode=range_mode)
    velocity_axis = doppler_axis_to_velocity_m_s(doppler_axis, carrier_freq)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        range_axis,
        velocity_axis,
        rd_db,
        shading="nearest" if interpolation == "nearest" else "gouraud",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlabel(range_axis_label or _range_axis_label(range_mode))
    ax.set_ylabel(velocity_display.axis_label)
    fig.colorbar(im, ax=ax, label="Power (dB)")

    detection_points = [
        (float(detection["range_m"]), float(detection["velocity_m_s"]))
        for detection in detections
        if "range_m" in detection and "velocity_m_s" in detection
    ]
    if detection_points:
        ax.scatter(
            [point[0] for point in detection_points],
            [point[1] for point in detection_points],
            marker="x",
            color=ORANGE,
            s=40,
            linewidths=1.3,
            label="Detection",
            zorder=5,
        )

    seen_target_ids: set[str] = set()
    for target in target_truth:
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id or target_id in seen_target_ids:
            raise ValueError("target_truth requires unique non-empty target_id values")
        seen_target_ids.add(target_id)
        try:
            target_range = float(target["range_m"])
            target_speed = float(target["velocity_m_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"target_truth[{target_id!r}] requires finite range_m and velocity_m_s"
            ) from exc
        if not np.isfinite(target_range) or not np.isfinite(target_speed):
            raise ValueError(
                f"target_truth[{target_id!r}] requires finite range_m and velocity_m_s"
            )
        ax.scatter(
            [target_range],
            [target_speed],
            marker="o",
            facecolors="none",
            edgecolors="#22b6a2",
            s=76,
            linewidths=1.5,
            label=f"GT {target_id}",
            zorder=6,
        )

    has_legend = bool(detection_points or seen_target_ids)
    legend = None
    if has_legend:
        legend = ax.legend(
            fontsize=8,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.07),
            ncol=3,
            borderaxespad=0,
            frameon=False,
        )

    config = display_config or {}
    range_res = config.get("range_resolution_m")
    vel_res = config.get("velocity_resolution_m_s")
    parts = []
    if range_res:
        parts.append(f"Range resolution: {range_res:.2f} m")
    if vel_res:
        parts.append(f"Velocity-bin spacing: {vel_res:.3f} m/s")
    if velocity_display.is_approximate:
        parts.append("Bistatic detections use approx. velocity")
    if parts:
        ax.text(
            0.0,
            1.025,
            "  ·  ".join(parts),
            transform=ax.transAxes,
            fontsize=8,
            color="#5d687b",
            verticalalignment="bottom",
        )
    if not _suppress_titles():
        title_pad = 28 if parts else 12
        if legend is not None:
            # Legend height grows with the number of targets. Measure its actual
            # extent before tight_layout reserves space above the plotting area.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_top = legend.get_window_extent(renderer).y1
            axes_top = ax.get_window_extent(renderer).y1
            title_pad = max(title_pad, (legend_top - axes_top) * 72 / fig.dpi + 8)
        ax.set_title(plot_title, pad=title_pad)

    # None retains explicit view hints for existing figure callers. Exported
    # summaries request either the shared GUI zoom or the full saved grid.
    if zoom_to_data is True:
        from shared.sensing.plotting.limits import range_doppler_zoom_limits

        points = detection_points + [
            (float(t["range_m"]), float(t["velocity_m_s"])) for t in target_truth
        ]
        limits = range_doppler_zoom_limits(rd_db, range_axis, velocity_axis, extra_points=points)
        if limits is not None:
            if range_xlim is None:
                range_xlim = limits[0]
            if velocity_ylim is None:
                velocity_ylim = limits[1]
    elif zoom_to_data is False:
        if range_axis.size > 1:
            edges = _coordinate_edges(range_axis, name="range")
            range_xlim = (
                max(0.0, float(edges[0])) if range_axis[0] >= 0 else float(edges[0]),
                float(edges[-1]),
            )
        if velocity_axis.size > 1:
            edges = _coordinate_edges(velocity_axis, name="velocity")
            velocity_ylim = (float(edges[0]), float(edges[-1]))

    if range_xlim is None and zoom_to_data is None:
        cfg_xlim = config.get("display_range_xlim")
        if cfg_xlim and len(cfg_xlim) == 2:
            range_xlim = (float(cfg_xlim[0]), float(cfg_xlim[1]))
    if range_xlim is not None:
        ax.set_xlim(range_xlim)

    if velocity_ylim is None and zoom_to_data is None:
        cfg_ylim = config.get("display_velocity_ylim")
        if cfg_ylim and len(cfg_ylim) == 2:
            velocity_ylim = (float(cfg_ylim[0]), float(cfg_ylim[1]))
    if velocity_ylim is not None:
        ax.set_ylim(velocity_ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Range–Doppler figure saved: %s", output_path)
    return output_path


def _resolve_angular_inputs(
    sensing_data: Dict[str, Any],
) -> tuple[
    RangeDopplerAngleProduct,
    Mapping[str, Any],
    Mapping[str, Any],
    np.ndarray,
    str,
]:
    """Resolve one exact typed angular product and its display configuration."""

    if "schema_version" in sensing_data:
        native_sensing: Mapping[str, Any] = sensing_data
    else:
        candidate = sensing_data.get("_typed_sensing")
        if not isinstance(candidate, Mapping):
            raise ValueError(
                "Angular plotting requires a native schema-v4 sensing payload; "
                "legacy angular aliases are unsupported"
            )
        native_sensing = candidate
    assert_supported_sensing_schema(native_sensing)
    product = sensing_data.get("_angular_product")
    if not isinstance(product, RangeDopplerAngleProduct):
        product = select_range_doppler_angle_product(native_sensing)
    if product is None:
        raise ValueError("No typed primary range_doppler_angle product is available")

    range_mode, carrier_hz = _require_resolved_plot_config(native_sensing)
    expected_range_semantics = (
        "monostatic_range" if range_mode.lower().startswith("mono") else "bistatic_path_length"
    )
    if product.range_coordinate_semantics != expected_range_semantics:
        raise ValueError(
            "Typed angular range-coordinate semantics disagree with config_resolved.range_mode"
        )
    velocity_axis = doppler_axis_to_velocity_m_s(product.doppler_axis_hz, carrier_hz)
    config = native_sensing["config_resolved"]
    assert isinstance(config, Mapping)
    return product, native_sensing, config, velocity_axis, range_mode


def _coordinate_edges(coordinates: np.ndarray, *, name: str) -> np.ndarray:
    """Convert exact monotonic bin centers to plotting edges without uniformity assumptions."""

    values = np.asarray(coordinates, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or not (np.all(np.diff(values) > 0.0) or np.all(np.diff(values) < 0.0))
    ):
        raise ValueError(f"Typed angular {name} plotting requires at least two ordered bins")
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            np.asarray([values[0] - (midpoints[0] - values[0])]),
            midpoints,
            np.asarray([values[-1] + (values[-1] - midpoints[-1])]),
        )
    )


def _plot_angular_power_mesh(
    ax: Any,
    power_linear: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    cmap: str,
    dynamic_range_db: float = 60.0,
    x_is_angle: bool = False,
    y_is_angle: bool = False,
) -> Any:
    """Plot one exact-coordinate power projection and return its color artist."""

    power = np.asarray(power_linear, dtype=np.float64)
    if power.shape != (len(y_axis), len(x_axis)):
        raise ValueError(
            f"Typed angular projection shape {power.shape} does not match exact "
            f"{y_name}/{x_name} axes {(len(y_axis), len(x_axis))}"
        )
    # Compact spatial FFT grids retain bins without a physical degree coordinate.
    # Omit those rows/columns from the display, preserving the saved full tensor.
    x_valid, y_valid = np.isfinite(x_axis), np.isfinite(y_axis)
    x_axis, y_axis = x_axis[x_valid], y_axis[y_valid]
    power = power[np.ix_(y_valid, x_valid)]
    if power.size == 0:
        raise ValueError("Typed angular projection has no physical display bins")
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    vmax = float(np.max(power_db))
    artist = ax.pcolormesh(
        _coordinate_edges(x_axis, name=x_name),
        _coordinate_edges(y_axis, name=y_name),
        power_db,
        cmap=cmap,
        shading="flat",
        vmin=vmax - dynamic_range_db,
        vmax=vmax,
    )
    # Exact bin edges can extrapolate past the physical broadside domain.
    # Clip only the view; keep the authored coordinates and power unchanged.
    if x_is_angle:
        ax.set_xlim(-90.0, 90.0)
    if y_is_angle:
        ax.set_ylim(-90.0, 90.0)
    return artist


def _broadside_detection_azimuths(
    product: RangeDopplerAngleProduct, detections: list[dict[str, Any]]
) -> np.ndarray:
    """Place spherical reports on the heatmap's broadside projection axis."""
    azimuths = np.asarray([d.get("aoa_az_deg", np.nan) for d in detections], dtype=float)
    elevations = (
        np.asarray([d.get("aoa_el_deg", np.nan) for d in detections], dtype=float)
        if product.power_linear.shape[0] > 1
        else np.zeros(len(detections))
    )
    return broadside_azimuth_projection(azimuths, elevations)


def _broadside_truth_azimuths(
    product: RangeDopplerAngleProduct, gt_state: Mapping[str, Any]
) -> np.ndarray:
    """Project known truth directions, leaving missing UPA elevation unavailable."""
    azimuths = next(
        (
            gt_state[key]
            for key in ("target_azimuths", "target_azimuths_deg", "target_aoa_az_deg")
            if gt_state.get(key) is not None and len(gt_state[key])
        ),
        [],
    )
    elevations = next(
        (
            gt_state[key]
            for key in ("target_elevations", "target_elevations_deg", "target_aoa_el_deg")
            if gt_state.get(key) is not None and len(gt_state[key])
        ),
        [],
    )
    if product.power_linear.shape[0] == 1:
        elevations = np.zeros(len(azimuths))
    elif len(elevations) != len(azimuths):
        # A rectangular-array projection cannot infer missing truth elevation.
        return np.full(len(azimuths), np.nan)
    return broadside_azimuth_projection(azimuths, elevations)


@sensing_figure_style
def create_range_azimuth_figure(
    sensing_data: Dict[str, Any],
    gt_state: Optional[Dict[str, Any]],
    output_path: Path,
    *,
    interpolation: str = "nearest",
    result_label: str = "last frame",
) -> Path:
    """Create a range-azimuth heatmap from the exact typed 5-D product."""
    output_path = Path(output_path)
    product, _native, config, _velocity_axis, _range_mode = _resolve_angular_inputs(sensing_data)
    if interpolation != "nearest":
        raise ValueError("Typed angular meshes support exact-bin nearest rendering only")
    # [E, A, D, R] -> [A, R]. Max projection preserves narrow spatial peaks.
    raz_power = np.max(product.forward_power_linear, axis=(0, 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = _plot_angular_power_mesh(
        ax,
        raz_power,
        product.range_axis_m,
        product.azimuth_broadside_angle_deg,
        x_name="range",
        y_name="azimuth",
        cmap="viridis",
        y_is_angle=True,
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Broadside azimuth (deg)")
    ax.set_title(f"Range-Broadside Azimuth Map ({result_label})")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    detections = sensing_data.get("detections") or []
    det_ranges = [d["range_m"] for d in detections if "range_m" in d]
    det_azimuths = _broadside_detection_azimuths(product, detections)
    if det_ranges:
        ax.scatter(
            det_ranges,
            det_azimuths[: len(det_ranges)],
            marker="x",
            color=ORANGE,
            s=40,
            linewidths=1.3,
            label="Detection",
            zorder=5,
        )

    if gt_state:
        gt_ranges = gt_state.get("target_ranges", [])
        gt_azimuths = _broadside_truth_azimuths(product, gt_state)
        if gt_ranges:
            for idx, gt_range in enumerate(gt_ranges):
                ax.axvline(
                    gt_range,
                    color="white",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label="GT range" if idx == 0 else None,
                )
        for idx, gt_azimuth in enumerate(gt_azimuths):
            if not np.isfinite(gt_azimuth):
                continue
            ax.axhline(
                gt_azimuth,
                color="cyan",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                label="GT broadside azimuth" if idx == 0 else None,
            )

    if detections or gt_state:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=8, loc="upper right")

    range_res = config.get("range_resolution_m")
    if range_res:
        ax.text(
            0.02,
            0.95,
            f"Range res: {range_res:.2f} m",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Range-Azimuth figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_doppler_azimuth_figure(
    sensing_data: Dict[str, Any],
    gt_state: Optional[Dict[str, Any]],
    output_path: Path,
    *,
    interpolation: str = "nearest",
    result_label: str = "last frame",
) -> Path:
    """Create a Doppler-azimuth heatmap from the exact typed 5-D product."""
    output_path = Path(output_path)
    product, _native, _config, velocity_axis, range_mode = _resolve_angular_inputs(sensing_data)
    if interpolation != "nearest":
        raise ValueError("Typed angular meshes support exact-bin nearest rendering only")
    # [E, A, D, R] -> [A, D].
    daz_power = np.max(product.forward_power_linear, axis=(0, 3))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = _plot_angular_power_mesh(
        ax,
        daz_power,
        velocity_axis,
        product.azimuth_broadside_angle_deg,
        x_name="Doppler velocity",
        y_name="azimuth",
        cmap="magma",
        y_is_angle=True,
    )
    ax.set_xlabel(detection_velocity_display(range_mode=range_mode).axis_label)
    ax.set_ylabel("Broadside azimuth (deg)")
    ax.set_title(f"Doppler-Broadside Azimuth Map ({result_label})")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    detections = sensing_data.get("detections") or []
    det_velocities = [d.get("velocity_m_s", 0.0) for d in detections]
    det_azimuths = _broadside_detection_azimuths(product, detections)
    if det_velocities:
        ax.scatter(
            det_velocities,
            det_azimuths[: len(det_velocities)],
            marker="x",
            color=ORANGE,
            s=40,
            linewidths=1.3,
            label="Detection",
            zorder=5,
        )

    if gt_state:
        gt_speeds = gt_state.get("target_speeds", [])
        gt_azimuths = _broadside_truth_azimuths(product, gt_state)
        if len(gt_speeds) and len(gt_azimuths):
            for idx, (gt_speed, gt_azimuth) in enumerate(zip(gt_speeds, gt_azimuths)):
                if not np.isfinite(gt_azimuth):
                    continue
                ax.scatter(
                    gt_speed,
                    gt_azimuth,
                    marker="o",
                    facecolors="none",
                    edgecolors="cyan",
                    s=120,
                    linewidths=2.0,
                    label="GT target" if idx == 0 else None,
                    zorder=6,
                )

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Doppler-Azimuth figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_azimuth_elevation_figure(
    sensing_data: Dict[str, Any],
    output_path: Path,
    *,
    interpolation: str = "nearest",
    result_label: str = "last frame",
) -> Path:
    """Create an azimuth-elevation heatmap from the exact typed 5-D product."""
    output_path = Path(output_path)
    product, _native, _config, _velocity_axis, _range_mode = _resolve_angular_inputs(sensing_data)
    if interpolation != "nearest":
        raise ValueError("Typed angular meshes support exact-bin nearest rendering only")
    # [E, A, D, R] -> [E, A].
    angle_power = np.max(product.forward_power_linear, axis=(2, 3))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = _plot_angular_power_mesh(
        ax,
        angle_power,
        product.azimuth_broadside_angle_deg,
        product.elevation_broadside_angle_deg,
        x_name="azimuth",
        y_name="elevation",
        cmap="viridis",
        dynamic_range_db=40.0,
        x_is_angle=True,
        y_is_angle=True,
    )
    ax.set_xlabel("Broadside azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f"Azimuth-Elevation Power Map ({result_label})")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    detections = sensing_data.get("detections") or []
    det_az = _broadside_detection_azimuths(product, detections)
    det_el = [d.get("aoa_el_deg", float("nan")) for d in detections]
    valid_det = [(az, el) for az, el in zip(det_az, det_el) if np.isfinite(az) and np.isfinite(el)]
    if valid_det:
        ax.scatter(
            [v[0] for v in valid_det],
            [v[1] for v in valid_det],
            marker="x",
            color=ORANGE,
            s=40,
            linewidths=1.3,
            label="Detection",
            zorder=5,
        )
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Azimuth-Elevation figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_range_elevation_figure(
    sensing_data: Dict[str, Any],
    output_path: Path,
    *,
    interpolation: str = "nearest",
    result_label: str = "last frame",
) -> Path:
    """Create a range-elevation heatmap from the exact typed 5-D product."""
    output_path = Path(output_path)
    product, _native, _config, _velocity_axis, _range_mode = _resolve_angular_inputs(sensing_data)
    if interpolation != "nearest":
        raise ValueError("Typed angular meshes support exact-bin nearest rendering only")
    # [E, A, D, R] -> [E, R].
    range_elevation_power = np.max(product.forward_power_linear, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = _plot_angular_power_mesh(
        ax,
        range_elevation_power,
        product.range_axis_m,
        product.elevation_broadside_angle_deg,
        x_name="range",
        y_name="elevation",
        cmap="viridis",
        dynamic_range_db=40.0,
        y_is_angle=True,
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f"Range-Elevation Power Map ({result_label})")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Range-Elevation figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_xyz_detection_figure(
    sensing_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Create a 3D scatter plot of detections in sensor-frame (x, y, z).

    Requires detections with ``x_m``, ``y_m``, and ``z_m`` fields (populated
    by ``augment_detections_with_sensor_frame_xy`` when elevation data is
    available).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_path = Path(output_path)
    detections = sensing_data.get("detections") or []

    xs, ys, zs, powers = [], [], [], []
    for d in detections:
        x = d.get("x_m", float("nan"))
        y = d.get("y_m", float("nan"))
        z = d.get("z_m", float("nan"))
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x)
            ys.append(y)
            zs.append(z)
            powers.append(d.get("snr_db", 0.0))

    if not xs:
        raise ValueError("No detections with finite x/y/z for 3D figure")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        xs,
        ys,
        zs,
        c=powers,
        cmap="plasma",
        s=40,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="SNR (dB)", shrink=0.6, pad=0.1)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Detection Positions (sensor frame)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("3D detection figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_angular_overview_figure(
    sensing_data: Dict[str, Any],
    gt_state: Optional[Dict[str, Any]],
    output_path: Path,
    *,
    interpolation: str = "nearest",
) -> Path:
    """Create a typed 3-panel R-D, R-Az, and D-Az overview."""
    output_path = Path(output_path)
    product, _native, _config, velocity_axis, range_mode = _resolve_angular_inputs(sensing_data)
    if interpolation != "nearest":
        raise ValueError("Typed angular meshes support exact-bin nearest rendering only")

    power = product.forward_power_linear
    rd_power = np.max(power, axis=(0, 1))
    range_azimuth_power = np.max(power, axis=(0, 2))
    doppler_azimuth_power = np.max(power, axis=(0, 3))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        (
            axes[0],
            rd_power,
            product.range_axis_m,
            velocity_axis,
            "Range (m)",
            detection_velocity_display(range_mode=range_mode).axis_label,
            "R-D",
            "viridis",
            False,
        ),
        (
            axes[1],
            range_azimuth_power,
            product.range_axis_m,
            product.azimuth_broadside_angle_deg,
            "Range (m)",
            "Broadside azimuth (deg)",
            "R-Az",
            "viridis",
            True,
        ),
        (
            axes[2],
            doppler_azimuth_power,
            velocity_axis,
            product.azimuth_broadside_angle_deg,
            detection_velocity_display(range_mode=range_mode).axis_label,
            "Broadside azimuth (deg)",
            "D-Az",
            "magma",
            True,
        ),
    ]

    for ax, image, x_axis, y_axis, xlabel, ylabel, title, cmap, y_is_angle in panels:
        im = _plot_angular_power_mesh(
            ax,
            image,
            x_axis,
            y_axis,
            x_name=xlabel,
            y_name=ylabel,
            cmap=cmap,
            y_is_angle=y_is_angle,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    detections = sensing_data.get("detections") or []
    if detections:
        det_ranges = [d["range_m"] for d in detections if "range_m" in d]
        det_velocities = [d.get("velocity_m_s", 0.0) for d in detections]
        det_azimuths = [d.get("aoa_az_deg", float("nan")) for d in detections]
        axes[0].scatter(
            det_ranges, det_velocities[: len(det_ranges)], marker="x", color=ORANGE, s=40
        )
        axes[1].scatter(det_ranges, det_azimuths[: len(det_ranges)], marker="x", color=ORANGE, s=40)
        axes[2].scatter(
            det_velocities[: len(det_azimuths)],
            det_azimuths,
            marker="x",
            color=ORANGE,
            s=40,
        )

    if gt_state:
        gt_ranges = gt_state.get("target_ranges", [])
        gt_speeds = gt_state.get("target_speeds", [])
        gt_azimuths = (
            gt_state.get("target_azimuths")
            or gt_state.get("target_azimuths_deg")
            or gt_state.get("target_aoa_az_deg")
            or []
        )
        for gt_range, gt_speed in zip(gt_ranges, gt_speeds):
            axes[0].scatter(
                gt_range,
                gt_speed,
                marker="o",
                facecolors="none",
                edgecolors="white",
                s=90,
                linewidths=1.8,
            )
        for gt_range, gt_azimuth in zip(gt_ranges, gt_azimuths):
            axes[1].scatter(
                gt_range,
                gt_azimuth,
                marker="o",
                facecolors="none",
                edgecolors="white",
                s=90,
                linewidths=1.8,
            )
        for gt_speed, gt_azimuth in zip(gt_speeds, gt_azimuths):
            axes[2].scatter(
                gt_speed,
                gt_azimuth,
                marker="o",
                facecolors="none",
                edgecolors="white",
                s=90,
                linewidths=1.8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Angular overview figure saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. GT evolution (range + radial speed vs frame)
# ---------------------------------------------------------------------------


@sensing_figure_style
def create_gt_evolution_figure(
    gt_history: List[Dict[str, Any]],
    output_path: Path,
    *,
    use_physical_time: bool = False,
) -> Path:
    """Create GT range and radial speed evolution plots with detection overlay.

    Args:
        gt_history: Per-frame list of dicts, each with keys:
            ``step`` (int, optional simulation frame index),
            ``target_ids`` (list[str]), ``target_ranges`` (list[float]),
            ``target_speeds`` (list[float | None]) and aligned
            ``target_velocity_valid`` (list[bool]),
            ``det_ranges`` (list[float]), ``det_velocities`` (list[float]).
        output_path: Destination PNG path.
        use_physical_time: Use the exact persisted result reference time rather
            than the scene frame number; every history row must provide it.

    Returns:
        The written file path.
    """
    output_path = Path(output_path)

    if not gt_history:
        raise ValueError("gt_history is empty")

    frame_indices = _history_coordinates(gt_history, use_physical_time=use_physical_time)

    target_order, target_series = _identity_keyed_tracking_truth_series(
        gt_history, use_physical_time=use_physical_time
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, (ax_range, ax_speed) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- GT range and speed curves per target ---
    for tidx, target_id in enumerate(target_order):
        color = palette[tidx % len(palette)]
        series = target_series[target_id]
        ax_range.plot(
            series["frames"],
            series["ranges"],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            label=target_id,
            clip_on=False,
        )
        ax_speed.plot(
            series["frames"],
            series["speeds"],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            label=target_id,
            clip_on=False,
        )

    # --- Detection overlay (scatter) ---
    # Collect detection ranges and velocities per frame
    det_frame_r: List[float] = []
    det_vals_r: List[float] = []
    det_frame_v: List[float] = []
    det_vals_v: List[float] = []
    for fidx, entry in enumerate(gt_history):
        for dr in entry.get("det_ranges", []):
            det_frame_r.append(float(frame_indices[fidx]))
            det_vals_r.append(dr)
        for dv in entry.get("det_velocities", []):
            det_frame_v.append(float(frame_indices[fidx]))
            det_vals_v.append(dv)

    if det_vals_r:
        ax_range.scatter(
            det_frame_r,
            det_vals_r,
            marker="x",
            color=ORANGE,
            s=12,
            linewidths=0.7,
            alpha=0.6,
            label="Detection",
            zorder=4,
            clip_on=False,
        )
    if det_vals_v:
        ax_speed.scatter(
            det_frame_v,
            det_vals_v,
            marker="x",
            color=ORANGE,
            s=12,
            linewidths=0.7,
            alpha=0.6,
            label="Detection",
            zorder=4,
            clip_on=False,
        )

    ax_range.set_xlabel("Result reference time (s)" if use_physical_time else "Scene frame")
    ax_range.set_ylabel("Range (m)")
    ax_range.set_title("Range · reference and detections")
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(fontsize=7, loc="best")

    ax_speed.set_xlabel("Result reference time (s)" if use_physical_time else "Scene frame")
    ax_speed.set_ylabel("Radial velocity (m/s)")
    ax_speed.set_title("Radial velocity · reference and detections")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=7, loc="best")

    # Include zero as a common reference without hiding negative velocities or
    # changing any plotted sample. Preserve automatic padding at the data end.
    for axis, key, detected in (
        (ax_range, "ranges", det_vals_r),
        (ax_speed, "speeds", det_vals_v),
    ):
        values = np.asarray(
            [value for series in target_series.values() for value in series[key]] + detected,
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            if np.min(finite) >= 0:
                axis.set_ylim(bottom=0)
            if np.max(finite) <= 0:
                axis.set_ylim(top=0 if np.min(finite) < 0 else 0.1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("GT evolution figure saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Micro-Doppler spectrogram
# ---------------------------------------------------------------------------


@sensing_figure_style
def create_micro_doppler_figure(
    result: MicroDopplerResult,
    output_path: Path,
    *,
    interpolation: str = "nearest",
    plot_title: str | None = None,
    dynamic_range_db: float = 60.0,
    cpi_duration_s: float | None = None,
    velocity_overlay: dict | None = None,
    show_containment_bars: bool | None = None,
    containment_n: int = 3,
) -> Path:
    """Create a micro-Doppler spectrogram (time vs Doppler frequency).

    Args:
        result: Output of :func:`shared.sensing.signal_processing.compute_micro_doppler`.
        output_path: Destination PNG path.
        dynamic_range_db: Display dynamic range in dB below the peak.
        cpi_duration_s: Optional coherent processing interval used to draw
            vertical CPI guide lines.
        velocity_overlay: Optional vertex velocity ground truth to overlay.
            Dict with ``time_s`` (1-D array), ``v_min`` (1-D array),
            ``v_max`` (1-D array), and optionally ``carrier_hz`` and
            ``prf_hz`` for wrapping into the unambiguous Doppler range.
        show_containment_bars: Show vertical bars at best/worst containment
            frames (where spectrogram power is most/least bounded by the
            vertex velocity envelope).  ``None`` (default) enables bars
            automatically when ``velocity_overlay`` is provided.
        containment_n: Number of best and worst containment frames to mark
            (default: 3).

    Returns:
        The written file path.
    """
    output_path = Path(output_path)

    power_db = 10.0 * np.log10(np.maximum(result.spectrogram, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 5))
    frequency_edges = _coordinate_edges(np.asarray(result.doppler_axis_hz), name="doppler")
    times = np.asarray(result.time_axis_s)
    if times.size == 1:
        duration_s = 1.0 / float(result.doppler_axis_hz[1] - result.doppler_axis_hz[0])
        time_edges = [float(times[0]) - duration_s / 2, float(times[0]) + duration_s / 2]
    else:
        time_edges = _coordinate_edges(times, name="time")
    extent = [time_edges[0], time_edges[-1], frequency_edges[0], frequency_edges[-1]]
    dyn_db = max(float(dynamic_range_db), 1.0)
    im = ax.imshow(
        power_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation=interpolation,
        vmin=power_db.max() - dyn_db,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler (Hz)")
    if result.collapsed:
        start = result.range_bin_start if result.range_bin_start is not None else 0
        end = result.range_bin_end if result.range_bin_end is not None else 0
        title = (
            "Range-Collapsed Micro-Doppler Spectrogram "
            f"(bins {start}-{end}, mean {result.range_m:.2f} m)"
        )
    elif result.range_tracking and result.tracked_range_bins is not None:
        tracked_min = int(np.min(result.tracked_range_bins))
        tracked_max = int(np.max(result.tracked_range_bins))
        title = (
            "Range-Tracked Micro-Doppler Spectrogram "
            f"(bins {tracked_min}-{tracked_max}, mean {result.range_m:.2f} m)"
        )
    else:
        title = f"Micro-Doppler Spectrogram (range bin {result.range_bin}, {result.range_m:.2f} m)"
    if not _suppress_titles():
        ax.set_title(plot_title or title)
    fig.colorbar(im, ax=ax, label="Power (dB)")

    if cpi_duration_s is not None and cpi_duration_s > 0.0:
        time_start = float(result.time_axis_s[0])
        time_stop = float(result.time_axis_s[-1])
        cpi_markers = np.arange(cpi_duration_s, time_stop + cpi_duration_s * 0.5, cpi_duration_s)
        label_assigned = False
        for marker_time in cpi_markers:
            if marker_time <= time_start or marker_time >= time_stop:
                continue
            ax.axvline(
                marker_time,
                color="white",
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
                label="CPI boundary" if not label_assigned else None,
                zorder=6,
            )
            label_assigned = True
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=8, loc="upper right")

    # Vertex velocity ground-truth overlay
    if velocity_overlay is not None:
        _C0 = 299_792_458.0
        ov_t = np.asarray(velocity_overlay["time_s"])
        carrier_hz = float(velocity_overlay.get("carrier_hz", 28e9))
        lam = _C0 / carrier_hz
        prf = float(velocity_overlay.get("prf_hz", result.doppler_axis_hz[-1] * 2))
        d_max = prf / 2.0  # max unambiguous Doppler (Hz)
        spec_end = float(result.time_axis_s[-1])
        # Tile the overlay to cover the full spectrogram duration
        # (mesh libraries are cyclic — one gait cycle may be shorter
        # than the simulation).
        ov_duration = float(ov_t[-1]) + float(ov_t[1] - ov_t[0]) if len(ov_t) > 1 else 1.0
        default_styles = {
            "v_min": {
                "color": "cyan",
                "label": "min $v_r$ (GT)",
                "linestyle": "-",
                "linewidth": 0.9,
                "alpha": 0.75,
            },
            "v_max": {
                "color": "lime",
                "label": "max $v_r$ (GT)",
                "linestyle": "-",
                "linewidth": 0.9,
                "alpha": 0.75,
            },
            "v_mean": {
                "color": "white",
                "label": "mean $v_r$ (GT)",
                "linestyle": "--",
                "linewidth": 0.8,
                "alpha": 0.75,
            },
        }
        style_overrides = velocity_overlay.get("styles", {})
        for v_key in ("v_min", "v_max", "v_mean"):
            if v_key not in velocity_overlay:
                continue
            style = dict(default_styles[v_key])
            style.update(style_overrides.get(v_key, {}))
            v_ms = np.asarray(velocity_overlay[v_key])
            # Convert velocity (m/s) to Doppler (Hz): f_d = 2*v/lambda
            d_hz = 2.0 * v_ms / lam
            # Wrap into unambiguous Doppler range [-d_max, d_max)
            d_wrapped = ((d_hz + d_max) % (2 * d_max)) - d_max
            # Tile to cover full spectrogram
            n_tiles = max(1, int(np.ceil(spec_end / ov_duration)))
            tiled_t = np.concatenate([ov_t + i * ov_duration for i in range(n_tiles)])
            tiled_d = np.tile(d_wrapped, n_tiles)
            # Clip to spectrogram range
            mask = tiled_t <= spec_end
            ax.plot(
                tiled_t[mask],
                tiled_d[mask],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=style["label"],
            )
        ax.legend(fontsize=7, loc="upper right")

        # Containment bars: mark frames where spectrogram power is best/worst
        # bounded by the vertex velocity envelope.
        do_containment = show_containment_bars if show_containment_bars is not None else True
        if do_containment and containment_n > 0:
            _C0_c = 299_792_458.0
            carrier_c = float(velocity_overlay.get("carrier_hz", 28e9))
            lam_c = _C0_c / carrier_c
            ov_t_c = np.asarray(velocity_overlay["time_s"])
            v_min_c = np.asarray(velocity_overlay["v_min"])
            v_max_c = np.asarray(velocity_overlay["v_max"])
            spec_t = result.time_axis_s
            d_axis = result.doppler_axis_hz
            vel_axis = d_axis * lam_c / 2.0

            vmin_interp = np.interp(spec_t, ov_t_c, v_min_c)
            vmax_interp = np.interp(spec_t, ov_t_c, v_max_c)

            containment = np.zeros(len(spec_t))
            for t_i in range(len(spec_t)):
                frame_power = result.spectrogram[t_i]
                total = np.sum(frame_power)
                if total <= 0:
                    continue
                in_bounds = (vel_axis >= vmin_interp[t_i]) & (vel_axis <= vmax_interp[t_i])
                containment[t_i] = np.sum(frame_power[in_bounds]) / total

            n_bars = min(containment_n, len(spec_t) // 2)
            sorted_idx = np.argsort(containment)
            worst_idx = sorted_idx[:n_bars]
            best_idx = sorted_idx[-n_bars:]

            for idx in worst_idx:
                ax.axvline(
                    spec_t[idx],
                    color=ORANGE,
                    linewidth=1.2,
                    alpha=0.7,
                    linestyle=":",
                    zorder=7,
                    label=(f"worst ({containment[idx]*100:.0f}%)" if idx == worst_idx[0] else None),
                )
            for idx in best_idx:
                ax.axvline(
                    spec_t[idx],
                    color="lime",
                    linewidth=1.2,
                    alpha=0.7,
                    linestyle=":",
                    zorder=7,
                    label=f"best ({containment[idx]*100:.0f}%)" if idx == best_idx[0] else None,
                )
            ax.legend(fontsize=7, loc="upper right")

    elif show_containment_bars is True:
        logger.warning("Containment bars require velocity_overlay — skipping")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Micro-Doppler figure saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 5. Detection summary (TP/FP/FN + RMSE over time)
# ---------------------------------------------------------------------------


@sensing_figure_style
def create_detection_summary_figure(
    association_history: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Create a per-frame detection quality bar chart with RMSE subplot.

    Args:
        association_history: Per-frame list of dicts with keys
            ``tp``, ``fp``, ``fn``, ``range_rmse``, ``velocity_rmse``.
        output_path: Destination PNG path.

    Returns:
        The written file path.
    """
    output_path = Path(output_path)

    n = len(association_history)
    frames = np.arange(n)
    tp = np.array([h["tp"] for h in association_history])
    fp = np.array([h["fp"] for h in association_history])
    fn = np.array([h["fn"] for h in association_history])
    r_rmse = np.array([h["range_rmse"] for h in association_history], dtype=float)
    v_rmse = np.array([h["velocity_rmse"] for h in association_history], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    ax_det, ax_rate, ax_range, ax_velocity = axes.flat
    width = 0.75
    for values, bottom, label, color in (
        (tp, np.zeros(n), "Matched", TEAL),
        (fp, tp, "Unmatched reports", RED),
        (fn, tp + fp, "Missed references", AMBER),
    ):
        ax_det.bar(frames, values, width=width, bottom=bottom, color=color, label=label)
    ax_det.set_title("Association outcomes")
    ax_det.set_ylim(0, max(1, float(np.max(tp + fp + fn))) * 1.5)
    ax_det.set_ylabel("Count")
    ax_det.legend(loc="best", fontsize=8)
    for numerator, denominator, label, color in (
        (tp, tp + fn, "Recall", BLUE),
        (tp, tp + fp, "Precision", PURPLE),
    ):
        rate = np.divide(
            numerator * 100.0, denominator, out=np.full(n, np.nan), where=denominator > 0
        )
        ax_rate.plot(
            frames,
            rate,
            marker="o",
            markersize=6 if label == "Recall" else 3,
            markerfacecolor="white" if label == "Recall" else color,
            linestyle="--" if label == "Recall" else "-",
            color=color,
            label=label,
        )
    ax_rate.set_title("Detection coverage")
    ax_rate.set_ylabel("Percent")
    ax_rate.set_ylim(-3, 103)
    ax_rate.legend(loc="best", fontsize=8)
    for ax, values, title, unit, color in (
        (ax_range, r_rmse, "Range error · matched pairs", "RMSE (m)", BLUE),
        (ax_velocity, v_rmse, "Radial velocity error · matched pairs", "RMSE (m/s)", PURPLE),
    ):
        # A frame without a match has no localization error, not zero error.
        values = np.where(tp > 0, values, np.nan)
        ax.plot(frames, values, marker="o", markersize=3, color=color)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel("Sensing update")
        finite = values[np.isfinite(values)]
        ax.set_ylim(0, max(float(finite.max()) * 1.15, 0.01) if finite.size else 1)
    from matplotlib.ticker import MaxNLocator

    ax_det.yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Detection summary figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_detection_events_figure(
    association_history: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Create a compact event-style detection validation figure.

    This complements ``create_detection_summary_figure``.  The stacked bar
    version is useful for long runs, but with only a few CPI frames it can look
    like a solid block.  This view uses three event lanes and marker size/text
    to show TP, FP, and FN counts without filling the plot area.
    """
    output_path = Path(output_path)

    n = len(association_history)
    frames = np.arange(n)
    tp = np.array([h["tp"] for h in association_history], dtype=float)
    fp = np.array([h["fp"] for h in association_history], dtype=float)
    fn = np.array([h["fn"] for h in association_history], dtype=float)
    r_rmse = np.array([h["range_rmse"] for h in association_history], dtype=float)
    v_rmse = np.array([h["velocity_rmse"] for h in association_history], dtype=float)

    fig, (ax_evt, ax_rmse) = plt.subplots(
        2,
        1,
        figsize=(10, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    lanes = [
        ("TP", tp, 2.0, "#2ca02c"),
        ("FP", fp, 1.0, "#d62728"),
        ("FN", fn, 0.0, "#ff7f0e"),
    ]
    annotate_counts = n <= 30
    for label, values, y, color in lanes:
        active = values > 0
        if np.any(active):
            sizes = 70.0 + 45.0 * values[active]
            ax_evt.scatter(
                frames[active],
                np.full(np.count_nonzero(active), y),
                s=sizes,
                marker="o",
                facecolor=color,
                edgecolor="#202020",
                linewidth=0.8,
                alpha=0.88,
                label=label,
                zorder=3,
            )
            if annotate_counts:
                for x, count in zip(frames[active], values[active]):
                    ax_evt.text(
                        x,
                        y,
                        str(int(count)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        weight="bold",
                        zorder=4,
                    )
        else:
            # Keep the legend stable even when a class has no events.
            ax_evt.scatter(
                [], [], s=70, marker="o", facecolor=color, edgecolor="#202020", label=label
            )

    for y in (0.0, 1.0, 2.0):
        ax_evt.axhline(y, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax_evt.set_yticks([0.0, 1.0, 2.0])
    ax_evt.set_yticklabels(["FN", "FP", "TP"])
    ax_evt.set_ylim(-0.55, 2.55)
    ax_evt.set_ylabel("Event")
    ax_evt.set_title(
        f"Detection Events (TP={int(np.sum(tp))}, FP={int(np.sum(fp))}, FN={int(np.sum(fn))})"
    )
    ax_evt.legend(fontsize=8, loc="upper right", ncol=3)
    ax_evt.grid(True, alpha=0.25, axis="x")

    valid_r = np.isfinite(r_rmse)
    valid_v = np.isfinite(v_rmse)
    if np.any(valid_r):
        ax_rmse.scatter(
            frames[valid_r],
            r_rmse[valid_r],
            color=BLUE,
            edgecolor="#202020",
            linewidth=0.5,
            s=42,
            label="Range RMSE (m)",
        )
        ax_rmse.plot(frames[valid_r], r_rmse[valid_r], color=BLUE, linewidth=0.9, alpha=0.7)
    if np.any(valid_v):
        ax_rmse.scatter(
            frames[valid_v],
            v_rmse[valid_v],
            color=PURPLE,
            edgecolor="#202020",
            linewidth=0.5,
            s=42,
            label="Velocity RMSE (m/s)",
        )
        ax_rmse.plot(frames[valid_v], v_rmse[valid_v], color=PURPLE, linewidth=0.9, alpha=0.7)
    ax_rmse.set_xlabel("Evaluated Event")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend(fontsize=8, loc="upper right")
    ax_rmse.grid(True, alpha=0.3)

    if n > 0:
        ax_evt.set_xlim(-0.5, max(0.5, n - 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Detection events figure saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 6. Tracking figure (Kalman tracks vs GT + raw detections)
# ---------------------------------------------------------------------------


def _history_coordinates(
    history: Sequence[Mapping[str, Any]], *, use_physical_time: bool
) -> np.ndarray:
    """Select authored physical timestamps or the historical scene-frame axis."""
    field = "reference_time_s" if use_physical_time else "step"
    values = np.asarray(
        [
            entry.get(field, np.nan if use_physical_time else index)
            for index, entry in enumerate(history)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Sensing history requires a finite {field} for every result")
    if use_physical_time and np.any(np.diff(values) <= 0):
        raise ValueError("Sensing history result reference times must increase strictly")
    return values


def _identity_keyed_tracking_truth_series(
    gt_history: Sequence[Mapping[str, Any]],
    *,
    use_physical_time: bool = False,
) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    """Build reorder-safe truth curves with explicit lifecycle gaps.

    Every target series spans every evaluated event.  An absent target is
    represented by NaN at that event, which makes Matplotlib break the line
    instead of interpolating across an inactive lifecycle interval.
    """

    event_frames: list[float] = []
    event_values: list[dict[str, tuple[float, float]]] = []
    target_order: list[str] = []
    seen_targets: set[str] = set()
    coordinates = _history_coordinates(gt_history, use_physical_time=use_physical_time)

    for event_index, entry in enumerate(gt_history):
        raw_frame = entry.get("step", event_index)
        if isinstance(raw_frame, (bool, np.bool_)):
            raise ValueError("Tracking truth step must be a finite integer")
        try:
            frame = float(raw_frame)
        except (TypeError, ValueError) as exc:
            raise ValueError("Tracking truth step must be a finite integer") from exc
        if not np.isfinite(frame) or not frame.is_integer():
            raise ValueError("Tracking truth step must be a finite integer")

        target_ids = entry.get("target_ids")
        target_ranges = entry.get("target_ranges")
        target_speeds = entry.get("target_speeds")
        velocity_valid = entry.get("target_velocity_valid")
        if not all(
            isinstance(values, list)
            for values in (target_ids, target_ranges, target_speeds, velocity_valid)
        ):
            raise ValueError(
                "Tracking truth requires target_ids, ranges, speeds, and velocity validity lists"
            )
        if not (len(target_ids) == len(target_ranges) == len(target_speeds) == len(velocity_valid)):
            raise ValueError("Tracking truth fields must align exactly by target_id")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("Tracking truth events cannot contain duplicate target_ids")

        values_by_id: dict[str, tuple[float, float]] = {}
        for target_id, raw_range, raw_speed, speed_valid in zip(
            target_ids,
            target_ranges,
            target_speeds,
            velocity_valid,
            strict=True,
        ):
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("Tracking truth target_ids must be non-empty strings")
            if isinstance(raw_range, (bool, np.bool_)):
                raise ValueError("Tracking truth ranges must be finite and non-negative")
            try:
                range_m = float(raw_range)
            except (TypeError, ValueError) as exc:
                raise ValueError("Tracking truth ranges must be finite and non-negative") from exc
            if not np.isfinite(range_m) or range_m < 0.0:
                raise ValueError("Tracking truth ranges must be finite and non-negative")
            if not isinstance(speed_valid, (bool, np.bool_)):
                raise ValueError("Tracking truth velocity validity must contain booleans")
            if speed_valid:
                if isinstance(raw_speed, (bool, np.bool_)):
                    raise ValueError("Valid tracking truth speeds must be finite")
                try:
                    speed_m_s = float(raw_speed)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Valid tracking truth speeds must be finite") from exc
                if not np.isfinite(speed_m_s):
                    raise ValueError("Valid tracking truth speeds must be finite")
            else:
                if raw_speed is not None:
                    raise ValueError("Unavailable tracking truth speed must be null")
                speed_m_s = float("nan")
            values_by_id[target_id] = (range_m, speed_m_s)
            if target_id not in seen_targets:
                seen_targets.add(target_id)
                target_order.append(target_id)
        event_frames.append(float(coordinates[event_index]))
        event_values.append(values_by_id)

    series: dict[str, dict[str, list[float]]] = {}
    for target_id in target_order:
        series[target_id] = {
            "frames": list(event_frames),
            "ranges": [
                values.get(target_id, (float("nan"), float("nan")))[0] for values in event_values
            ],
            "speeds": [
                values.get(target_id, (float("nan"), float("nan")))[1] for values in event_values
            ],
        }
    return target_order, series


@sensing_figure_style
def create_tracking_figure(
    gt_history: List[Dict[str, Any]],
    track_history: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
    *,
    pruned_track_ids: set[int] | None = None,
    use_physical_time: bool = False,
) -> Path:
    """Create a tracking visualization showing GT, raw detections, and Kalman tracks.

    Physical range starts at zero. Velocity starts at zero when every finite
    plotted truth, detection, and track-state value is non-negative.

    Args:
        gt_history: Per-frame list from ``_build_gt_history`` with ``step``,
            ``target_ranges``, ``target_speeds``, ``det_ranges``, ``det_velocities``.
        track_history: Per-track dict: ``{track_id: [{"frame": int, "range_m": float,
            "velocity_m_s": float, "confirmed": bool, "raw_range": float,
            "raw_velocity": float}, ...]}``
        pruned_track_ids: Track IDs that were pruned as ghosts by the
            correlation filter.  Rendered as red dashed lines with "X" markers.
        output_path: Destination PNG path.
        use_physical_time: Use exact result reference epochs from ``gt_history``
            for detections and tracks, joined by their recorded frame identity.

    Returns:
        The written file path.
    """
    output_path = Path(output_path)

    if not gt_history:
        raise ValueError("gt_history is empty")
    coordinates = _history_coordinates(gt_history, use_physical_time=use_physical_time)
    coordinate_by_frame = {
        int(entry.get("step", index)): float(coordinates[index])
        for index, entry in enumerate(gt_history)
    }

    gt_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    track_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    fig, (ax_range, ax_speed) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Layer 1: Raw detections (grey dots — noisy input) ---
    raw_frames_r: List[int] = []
    raw_vals_r: List[float] = []
    raw_frames_v: List[int] = []
    raw_vals_v: List[float] = []
    for entry in gt_history:
        frame = int(entry.get("step", 0))
        step = coordinate_by_frame[frame] if use_physical_time else frame
        for dr in entry.get("det_ranges", []):
            raw_frames_r.append(step)
            raw_vals_r.append(dr)
        for dv in entry.get("det_velocities", []):
            raw_frames_v.append(step)
            raw_vals_v.append(dv)

    if raw_vals_r:
        ax_range.scatter(
            raw_frames_r,
            raw_vals_r,
            marker=".",
            color="#cccccc",
            s=6,
            alpha=0.4,
            zorder=1,
            label="Reported detections",
        )
    if raw_vals_v:
        ax_speed.scatter(
            raw_frames_v,
            raw_vals_v,
            marker=".",
            color="#cccccc",
            s=6,
            alpha=0.4,
            zorder=1,
            label="Reported detections",
        )

    # --- Layer 2: GT curves ---
    target_order, target_series = _identity_keyed_tracking_truth_series(
        gt_history, use_physical_time=use_physical_time
    )
    for target_index, target_id in enumerate(target_order):
        series = target_series[target_id]
        color = gt_palette[target_index % len(gt_palette)]
        ax_range.plot(
            series["frames"],
            series["ranges"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=f"GT {target_id}",
            zorder=3,
        )
        ax_speed.plot(
            series["frames"],
            series["speeds"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=f"GT {target_id}",
            zorder=3,
        )

    # --- Layer 3: Kalman tracks (colored lines per track_id) ---
    _pruned = pruned_track_ids or set()
    for idx, (tid, history) in enumerate(sorted(track_history.items())):
        if not history:
            continue
        is_pruned = tid in _pruned
        color = "#d62728" if is_pruned else track_palette[idx % len(track_palette)]
        frames_t = [
            coordinate_by_frame[int(h["frame"])] if use_physical_time else h["frame"]
            for h in history
        ]
        ranges_t = [h["range_m"] for h in history]
        vels_t = [h["velocity_m_s"] for h in history]
        confirmed = [h.get("confirmed", True) for h in history]

        # Draw as line segments
        for j in range(len(frames_t) - 1):
            if is_pruned:
                ls, lw, alpha = ":", 1.0, 0.4
            elif confirmed[j]:
                ls, lw, alpha = "-", 1.8, 0.9
            else:
                ls, lw, alpha = "--", 1.0, 0.5
            ax_range.plot(
                [frames_t[j], frames_t[j + 1]],
                [ranges_t[j], ranges_t[j + 1]],
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                zorder=2 if is_pruned else 4,
            )
            ax_speed.plot(
                [frames_t[j], frames_t[j + 1]],
                [vels_t[j], vels_t[j + 1]],
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                zorder=2 if is_pruned else 4,
            )

        # Label at start of track
        marker = "x" if is_pruned else "o"
        suffix = " (ghost)" if is_pruned else ""
        label = f"Track {tid}{suffix}" if idx < 8 else None
        for axis, values in ((ax_range, ranges_t), (ax_speed, vels_t)):
            axis.scatter(
                [frames_t[0]],
                [values[0]],
                marker=marker,
                color=color,
                s=40 if is_pruned else 30,
                zorder=2 if is_pruned else 5,
                label=label,
            )

    ax_range.set_xlabel("Result reference time (s)" if use_physical_time else "Scene frame")
    ax_range.set_ylabel("Range (m)")
    ax_range.set_title(
        "Tracking: Range vs Time" if use_physical_time else "Tracking: Range vs Frame"
    )
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(fontsize=6, loc="best", ncol=2)

    ax_speed.set_xlabel("Result reference time (s)" if use_physical_time else "Scene frame")
    ax_speed.set_ylabel("Radial velocity (m/s)")
    ax_speed.set_title(
        "Tracking: Velocity vs Time" if use_physical_time else "Tracking: Velocity vs Frame"
    )
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=6, loc="best", ncol=2)

    # Recompute the upper padding against zero so nearly constant histories
    # retain space for markers after their displayed span expands.
    range_top = ax_range.get_ylim()[1]
    if np.isfinite(ax_range.dataLim.ymax):
        range_top = max(range_top, ax_range.dataLim.ymax * (1 + ax_range.get_ymargin()))
    ax_range.set_ylim(0, range_top)
    velocities = np.asarray(
        [value for series in target_series.values() for value in series["speeds"]]
        + raw_vals_v
        + [entry["velocity_m_s"] for history in track_history.values() for entry in history],
        dtype=float,
    )
    finite_velocities = velocities[np.isfinite(velocities)]
    if finite_velocities.size and np.all(finite_velocities >= 0):
        velocity_top = max(
            ax_speed.get_ylim()[1], np.max(finite_velocities) * (1 + ax_speed.get_ymargin())
        )
        ax_speed.set_ylim(0, velocity_top)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Tracking figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_xy_tracking_figure(
    track_history: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
    *,
    raw_detections: Optional[List[Dict[str, Any]]] = None,
    gt_tracks: Optional[Dict[str | int, List[Dict[str, Any]]]] = None,
    radar_positions: Optional[List[Dict[str, Any]]] = None,
    pruned_track_ids: set[int] | None = None,
    scene_bounds: tuple[float, float, float, float] | None = None,
    coordinate_frame: str = "auto",
    show_detections: bool = False,
) -> Path:
    """Create an x/y view of detections, GT, and track-linked trajectories.

    When GT and radar positions are provided, the figure is rendered in world
    coordinates so it can be read like a topology view. Callers may explicitly
    select receiver-local coordinates and supply saved Cartesian track states.
    The historical automatic frame selection and report visibility stay default.
    """
    output_path = Path(output_path)

    if coordinate_frame not in {"auto", "world", "receiver_local"}:
        raise ValueError("Choose auto, world or receiver_local tracking coordinates")
    use_world_xy = coordinate_frame == "world" or (
        coordinate_frame == "auto" and bool(gt_tracks or radar_positions)
    )

    def _track_xy(entry: Dict[str, Any]) -> tuple[float, float]:
        if use_world_xy:
            return (
                float(entry.get("world_x_m", entry.get("x_m", np.nan))),
                float(entry.get("world_y_m", entry.get("y_m", np.nan))),
            )
        return float(entry.get("x_m", np.nan)), float(entry.get("y_m", np.nan))

    def _raw_xy(entry: Dict[str, Any]) -> tuple[float, float]:
        return float(entry.get("x_m", np.nan)), float(entry.get("y_m", np.nan))

    has_track_xy = any(
        any(np.isfinite(_track_xy(h)[0]) and np.isfinite(_track_xy(h)[1]) for h in history)
        for history in track_history.values()
    )
    has_raw_xy = any(
        np.isfinite(_raw_xy(det)[0]) and np.isfinite(_raw_xy(det)[1])
        for det in (raw_detections or [])
    )
    has_gt_xy = any(
        any(
            np.isfinite(h.get("x_m", np.nan)) and np.isfinite(h.get("y_m", np.nan)) for h in history
        )
        for history in (gt_tracks or {}).values()
    )
    if not has_track_xy and not has_raw_xy and not has_gt_xy:
        raise ValueError("No finite x/y detections available for x/y tracking figure")

    track_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    gt_palette = ["#111111", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    _pruned = pruned_track_ids or set()
    primary_track_ids: set[int] = set()
    if use_world_xy:
        for tid, history in track_history.items():
            if tid in _pruned:
                continue
            finite_count = sum(
                1 for h in history if np.isfinite(_track_xy(h)[0]) and np.isfinite(_track_xy(h)[1])
            )
            if finite_count > 0:
                primary_track_ids.add(tid)

    # Paper-style mode: drop tracks whose points all fall outside the GT
    # bounding box (padded by ``_PAPER_BBOX_PAD_M``). These are typically
    # wall-multipath ghosts that stretch the axes and clutter the legend.
    _PAPER_BBOX_PAD_M = 2.0
    paper_mode = _suppress_titles()
    paper_bbox: tuple[float, float, float, float] | None = None
    if paper_mode and use_world_xy and gt_tracks:
        gt_x_all: list[float] = []
        gt_y_all: list[float] = []
        for history in gt_tracks.values():
            for h in history:
                gx = h.get("x_m", np.nan)
                gy = h.get("y_m", np.nan)
                if np.isfinite(gx) and np.isfinite(gy):
                    gt_x_all.append(float(gx))
                    gt_y_all.append(float(gy))
        if gt_x_all and gt_y_all:
            paper_bbox = (
                min(gt_x_all) - _PAPER_BBOX_PAD_M,
                max(gt_x_all) + _PAPER_BBOX_PAD_M,
                min(gt_y_all) - _PAPER_BBOX_PAD_M,
                max(gt_y_all) + _PAPER_BBOX_PAD_M,
            )
    if paper_bbox is not None:
        bx_min, bx_max, by_min, by_max = paper_bbox
        filtered_primary: set[int] = set()
        for tid in primary_track_ids:
            for h in track_history.get(tid, []):
                tx, ty = _track_xy(h)
                if (
                    np.isfinite(tx)
                    and np.isfinite(ty)
                    and bx_min <= tx <= bx_max
                    and by_min <= ty <= by_max
                ):
                    filtered_primary.add(tid)
                    break
        primary_track_ids = filtered_primary

    # Optional: put the longer trajectory axis on the horizontal axis so the
    # figure fits a single-column paper layout. Controlled by the
    # ``SENSING_FIGURES_SWAP_XY`` env var; default behaviour is unchanged.
    swap_world_xy = use_world_xy and _swap_world_xy_axes()
    figsize = (7.5, 4.5) if swap_world_xy else (7.5, 7.0)
    fig, ax = plt.subplots(figsize=figsize)

    # Existing callers omit raw reports. Cartesian state summaries can show
    # them explicitly so measured positions remain distinct from filtered states.
    if show_detections:
        points = [
            _raw_xy(detection)
            for detection in (raw_detections or [])
            if np.isfinite(_raw_xy(detection)).all()
        ]
        if points:
            ax.scatter(
                *np.asarray(points).T,
                marker="x",
                color="#999999",
                s=18,
                alpha=0.55,
                label="Reported detections",
                zorder=2,
            )
    del raw_detections

    gt_start_labeled = False
    gt_end_labeled = False
    if gt_tracks:
        for idx, (target_id, history) in enumerate(sorted(gt_tracks.items())):
            xy_history = [
                (float(h["x_m"]), float(h["y_m"]))
                for h in history
                if np.isfinite(h.get("x_m", np.nan)) and np.isfinite(h.get("y_m", np.nan))
            ]
            if not xy_history:
                continue
            x_vals = [xy[0] for xy in xy_history]
            y_vals = [xy[1] for xy in xy_history]
            color = gt_palette[idx % len(gt_palette)]
            line_xy = (
                [(h.get("x_m", np.nan), h.get("y_m", np.nan)) for h in history]
                if coordinate_frame == "receiver_local"
                else xy_history
            )
            # Draw GT on top of the per-track scatter with a dashed line so the
            # trajectory remains visible through the detection cloud.
            ax.plot(
                [xy[0] for xy in line_xy],
                [xy[1] for xy in line_xy],
                color=color,
                linestyle="--",
                linewidth=2.2,
                alpha=0.95,
                zorder=10,
                label=f"GT target {target_id}",
            )
            ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                marker="s",
                facecolors="white",
                edgecolors=color,
                s=52,
                linewidths=1.6,
                zorder=11,
                label="GT start" if not gt_start_labeled else None,
            )
            gt_start_labeled = True
            ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                marker="D",
                facecolors=color,
                edgecolors="white",
                s=56,
                linewidths=1.2,
                zorder=11,
                label="GT end" if not gt_end_labeled else None,
            )
            gt_end_labeled = True

    if radar_positions:
        radar_x = [
            float(entry["x_m"])
            for entry in radar_positions
            if np.isfinite(entry.get("x_m", np.nan)) and np.isfinite(entry.get("y_m", np.nan))
        ]
        radar_y = [
            float(entry["y_m"])
            for entry in radar_positions
            if np.isfinite(entry.get("x_m", np.nan)) and np.isfinite(entry.get("y_m", np.nan))
        ]
        if radar_x:
            ax.scatter(
                [radar_x[0]],
                [radar_y[0]],
                marker="*",
                color="black",
                s=120,
                zorder=6,
                label="Radar",
            )
    elif not use_world_xy:
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            color="black",
            s=110,
            zorder=6,
            label="Radar",
        )

    for idx, (tid, history) in enumerate(sorted(track_history.items())):
        xy_history = [
            _track_xy(h)
            for h in history
            if np.isfinite(_track_xy(h)[0]) and np.isfinite(_track_xy(h)[1])
        ]
        if not xy_history:
            continue
        x_vals = [xy[0] for xy in xy_history]
        y_vals = [xy[1] for xy in xy_history]
        is_pruned = tid in _pruned
        color = "#d62728" if is_pruned else track_palette[idx % len(track_palette)]
        linestyle = ":" if is_pruned else "-"
        alpha = 0.45 if is_pruned else 0.85
        suffix = " (ghost)" if is_pruned else ""
        label = f"Track {tid}{suffix}" if idx < 8 else None
        if use_world_xy:
            # Skip ghost / non-primary tracks in world-xy mode: their orphan
            # "x" markers cluttered the figure without adding information.
            if is_pruned or tid not in primary_track_ids:
                continue
            # In world-XY mode the paper figure needs an actual trajectory, not
            # just a cloud of detections. Plot the per-track path and overlay
            # the assigned detections as points.
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if not is_pruned else 1.1,
                alpha=alpha,
                zorder=3 if not is_pruned else 2,
                label=label,
            )
            ax.scatter(
                x_vals,
                y_vals,
                marker="o",
                color=color,
                s=24 if len(x_vals) > 1 else 36,
                alpha=0.75,
                zorder=4,
            )
        else:
            # Sensor-plane mode: line carries the legend entry, start marker
            # is drawn without a label to avoid duplicating the track in the
            # legend.
            line_xy = (
                [_track_xy(h) for h in history]
                if coordinate_frame == "receiver_local"
                else xy_history
            )
            ax.plot(
                [xy[0] for xy in line_xy],
                [xy[1] for xy in line_xy],
                color=color,
                linestyle=linestyle,
                linewidth=1.8 if not is_pruned else 1.1,
                alpha=alpha,
                zorder=3 if not is_pruned else 2,
                label=label,
            )
            ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                marker="x" if is_pruned else "o",
                color=color,
                s=36 if not is_pruned else 44,
                zorder=5 if not is_pruned else 4,
            )

    if swap_world_xy:
        # Swap the plotted data in place: every scatter PathCollection and
        # every Line2D on this axis has its (x, y) transposed.
        for artist in list(ax.collections):
            offs = artist.get_offsets()
            if offs is not None and len(offs):
                artist.set_offsets(np.column_stack([offs[:, 1], offs[:, 0]]))
        for line in list(ax.get_lines()):
            xd = np.asarray(line.get_xdata())
            yd = np.asarray(line.get_ydata())
            line.set_data(yd, xd)
        ax.set_xlabel("World Y (m)")
        ax.set_ylabel("World X (m)")
    else:
        local_label = "Receiver-local" if coordinate_frame == "receiver_local" else "Sensor-Local"
        ax.set_xlabel("World X (m)" if use_world_xy else f"{local_label} X (m)")
        ax.set_ylabel("World Y (m)" if use_world_xy else f"{local_label} Y (m)")
    if not _suppress_titles():
        ax.set_title(
            "Receiver-local X/Y tracks"
            if coordinate_frame == "receiver_local"
            else (
                "World X/Y: GT, Detections, and Track Assignments"
                if use_world_xy
                else "Sensor-Plane Detections / Track Assignments"
            )
        )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Draw scene wall outline
    if scene_bounds is not None:
        x_min, x_max, y_min, y_max = scene_bounds
        wall_x = [x_min, x_max, x_max, x_min, x_min]
        wall_y = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(wall_x, wall_y, color="#888888", linewidth=1.5, alpha=0.6, zorder=0)

    # Paper-style mode: explicitly pin axis limits to the union of GT points,
    # retained tracks, and radar position (padded). Prevents the auto-scaled
    # "equal" aspect from stretching the figure because of any remaining
    # out-of-scene samples.
    if paper_mode and use_world_xy:
        data_x: list[float] = []
        data_y: list[float] = []
        if gt_tracks:
            for history in gt_tracks.values():
                for h in history:
                    gx = h.get("x_m", np.nan)
                    gy = h.get("y_m", np.nan)
                    if np.isfinite(gx) and np.isfinite(gy):
                        data_x.append(float(gx))
                        data_y.append(float(gy))
        for tid in primary_track_ids:
            for h in track_history.get(tid, []):
                tx, ty = _track_xy(h)
                if np.isfinite(tx) and np.isfinite(ty):
                    data_x.append(float(tx))
                    data_y.append(float(ty))
        if radar_positions:
            for entry in radar_positions:
                rx = entry.get("x_m", np.nan)
                ry = entry.get("y_m", np.nan)
                if np.isfinite(rx) and np.isfinite(ry):
                    data_x.append(float(rx))
                    data_y.append(float(ry))
        if data_x and data_y:
            if swap_world_xy:
                ax.set_xlim(min(data_y) - 1.0, max(data_y) + 1.0)
                ax.set_ylim(min(data_x) - 1.0, max(data_x) + 1.0)
            else:
                ax.set_xlim(min(data_x) - 1.0, max(data_x) + 1.0)
                ax.set_ylim(min(data_y) - 1.0, max(data_y) + 1.0)

    # Collapse any remaining duplicate labels (e.g. same label used twice
    # when a track is drawn by both a line and a start marker).
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("X/Y tracking figure saved: %s", output_path)
    return output_path


@sensing_figure_style
def create_xyz_tracking_figure(
    track_history: Dict[int, List[Dict[str, Any]]],
    *,
    gt_tracks: Optional[Dict[str | int, List[Dict[str, Any]]]] = None,
    pruned_track_ids: set[int] | None = None,
    scene_bounds: tuple[float, float, float, float] | None = None,
    output_path: str | Path | None = None,
    coordinate_frame: str = "world",
    raw_detections: Optional[List[Dict[str, Any]]] = None,
    connect_tracks: bool = False,
) -> "plt.Figure | None":
    """Create a 3D scatter plot of tracked positions in world X/Y/Z.

    Plots each track as colored dots in 3D space. Ground-truth trajectories
    are rendered as solid lines. Ghost (pruned) tracks use X markers.

    Args:
        track_history: Per-track history dicts with ``x_m``, ``y_m``, ``z_m``.
        gt_tracks: Optional ground-truth trajectories with ``x_m``, ``y_m``,
            ``z_m`` per frame.
        pruned_track_ids: Track IDs identified as ghosts (rendered with X
            markers and muted colors).
        output_path: If provided, save the figure to this path and close it.

    Returns:
        The matplotlib Figure, or ``None`` if no 3D data is available. If
        *output_path* is given the figure is saved and closed automatically.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _pruned = pruned_track_ids or set()
    if coordinate_frame not in {"world", "receiver_local"}:
        raise ValueError("Choose world or receiver_local tracking coordinates")

    def track_position(entry):
        return (
            (
                float(entry.get("world_x_m", entry.get("x_m", np.nan)))
                if coordinate_frame == "world"
                else float(entry.get("x_m", np.nan))
            ),
            (
                float(entry.get("world_y_m", entry.get("y_m", np.nan)))
                if coordinate_frame == "world"
                else float(entry.get("y_m", np.nan))
            ),
            float(entry.get("z_m", np.nan)),
        )

    # Check that at least some track entries have finite z
    has_z = any(
        np.isfinite(e.get("z_m", float("nan")))
        for entries in track_history.values()
        for e in entries
    )
    if not has_z:
        return None

    track_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    gt_palette = ["#111111", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    reports = [
        [d.get(f"{coordinate}_m", np.nan) for coordinate in "xyz"] for d in (raw_detections or [])
    ]
    if reports:
        finite_reports = np.asarray(reports, dtype=float)
        finite_reports = finite_reports[np.isfinite(finite_reports).all(axis=1)]
        if finite_reports.size:
            ax.scatter(
                *finite_reports.T,
                marker="x",
                color="#999999",
                alpha=0.55,
                s=18,
                label="Reported detections",
            )

    # Plot GT trajectories as solid lines (world coordinates)
    if gt_tracks:
        for idx, (target_id, history) in enumerate(sorted(gt_tracks.items())):
            xyz = [
                (
                    float(h.get("x_m", np.nan)),
                    float(h.get("y_m", np.nan)),
                    float(h.get("z_m", np.nan)),
                )
                for h in history
                if np.isfinite(h.get("x_m", np.nan))
                and np.isfinite(h.get("y_m", np.nan))
                and np.isfinite(h.get("z_m", np.nan))
            ]
            if not xyz:
                continue
            xs = [p[0] for p in xyz]
            ys = [p[1] for p in xyz]
            zs = [p[2] for p in xyz]
            color = gt_palette[idx % len(gt_palette)]
            if coordinate_frame == "receiver_local":
                xs, ys, zs = np.asarray(
                    [[h.get(f"{axis}_m", np.nan) for axis in "xyz"] for h in history]
                ).T
            ax.plot(
                xs,
                ys,
                zs,
                color=color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.9,
                label=f"GT target {target_id}",
            )

    # Plot tracked positions (prefer world coordinates for x/y, z from track)
    for idx, (tid, history) in enumerate(sorted(track_history.items())):
        xyz = [
            track_position(entry) for entry in history if np.isfinite(track_position(entry)).all()
        ]
        if not xyz:
            continue
        xs = [p[0] for p in xyz]
        ys = [p[1] for p in xyz]
        zs = [p[2] for p in xyz]
        is_pruned = tid in _pruned
        color = "#d62728" if is_pruned else track_palette[idx % len(track_palette)]
        marker = "x" if is_pruned else "o"
        alpha = 0.4 if is_pruned else 0.8
        suffix = " (ghost)" if is_pruned else ""
        label = f"Track {tid}{suffix}" if idx < 8 else None
        if connect_tracks:
            trajectory = np.asarray([track_position(entry) for entry in history])
            ax.plot(*trajectory.T, color=color, linewidth=1.6, alpha=alpha)
        ax.scatter(
            xs,
            ys,
            zs,
            marker=marker,
            color=color,
            s=28 if not is_pruned else 36,
            alpha=alpha,
            label=label,
        )

    # Draw scene floor outline if bounds provided
    if scene_bounds is not None:
        sx_min, sx_max, sy_min, sy_max = scene_bounds
        wall_x = [sx_min, sx_max, sx_max, sx_min, sx_min]
        wall_y = [sy_min, sy_min, sy_max, sy_max, sy_min]
        wall_z = [0.0] * 5
        ax.plot(wall_x, wall_y, wall_z, color="#888888", linewidth=1.5, alpha=0.5)

    local = coordinate_frame == "receiver_local"
    ax.set_xlabel("Receiver-local X (m)" if local else "World X (m)")
    ax.set_ylabel("Receiver-local Y (m)" if local else "World Y (m)")
    ax.set_zlabel("Receiver-local Z (m)" if local else "Z (m)")
    ax.set_title("3D tracking: receiver-local X/Y/Z" if local else "3D Tracking: World X/Y/Z")
    ax.legend(fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        # Axes3D's tight bounds can omit the long local-coordinate Z label.
        # Include all three labels explicitly without changing legacy exports.
        extra = (
            {"bbox_extra_artists": [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]}
            if local
            else {}
        )
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight", **extra)
        plt.close(fig)
        logger.info("3D tracking figure saved: %s", output_path)
        return None
    return fig
