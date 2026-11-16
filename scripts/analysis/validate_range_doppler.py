"""CPU qualification of the radial cube's saved CFR and the shared RD pipeline.

Run from the repository root with ``python -m scripts.analysis.validate_range_doppler
--output <directory>``.
The independent reference uses explicit Fourier sums, not numpy.fft. No frame,
scenario, detector setting, or processing default is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from shared.extensions.sensing import build_sensing_temporal_contract, canonical_sensing_hash
from shared.sensing.cfr_replay import rebuild_frequency_observation
from shared.sensing.products import build_product_descriptor, dense_axis
from shared.sensing.range_response import C0, RangeResponseConfig

CARRIER = 28e9
BANDWIDTH = 500e6
FREQUENCIES = 128
SAMPLES = 64
SAMPLE_RATE = 1000.0
RANGE_FFT = 512
WINDOWS = (("boxcar", "boxcar"), ("hann", "boxcar"), ("boxcar", "hann"), ("hann", "hann"))


def point_observation(*, velocity=1.8, range0=5.0, cpi=0, amplitude=1.0, frozen_delay=False):
    """Independent absolute-frequency phase model, with optional native approximation.

    ``frozen_delay`` holds the envelope at the solve epoch while evolving only
    carrier Doppler, matching the recorded one-solve Sionna acquisition model.
    The default evolves R(t) at every absolute frequency, including migration.
    """
    times = (cpi * SAMPLES + np.arange(SAMPLES)) / SAMPLE_RATE
    offsets = (np.arange(FREQUENCIES) - FREQUENCIES // 2) * BANDWIDTH / FREQUENCIES
    ranges = range0 - velocity * times
    if frozen_delay:
        phase = CARRIER * ranges[:, None] + offsets[None, :] * ranges[0]
    else:
        phase = (CARRIER + offsets[None, :]) * ranges[:, None]
    return amplitude * np.exp(-4j * np.pi * phase / C0), offsets, times


def observation_payload(observation, frequencies, times, *, clutter="legacy"):
    """Describe just the unwindowed observation required by the shared replay API."""
    recipe = {"window_doppler": "boxcar", "coherent_clutter_mode": clutter}
    generation = str(uuid5(NAMESPACE_URL, "orchav-rd-validation"))
    pair = canonical_sensing_hash({"tx_sensor_id": "tx", "rx_sensor_id": "rx"})
    temporal = build_sensing_temporal_contract(
        clock_id=generation,
        acquisition_id=str(uuid5(NAMESPACE_URL, f"point-{times[0]:.6f}")),
        acquisition_config_id=canonical_sensing_hash({"fs": SAMPLE_RATE, "n": SAMPLES}),
        pair_id=pair,
        tx_sensor_id="tx",
        rx_sensor_id="rx",
        samples=[
            dict(
                sample_index=i,
                sample_time_s=float(t),
                valid=True,
                phase_coherent=True,
                source={
                    "scene_frame_id": 0,
                    "sample_offset": i,
                    "source_generation_id": generation,
                },
            )
            for i, t in enumerate(times)
        ],
        processing_run_id=generation,
        processing_recipe_id=canonical_sensing_hash(recipe),
        reference_time_s=float(times[0]),
        reference_time_policy="solve_start",
        display_scene_frame_id=0,
        display_policy="source_scene_exact",
        display_reason="analytical channel; no scene or ray tracer",
    )
    sensing = {
        "temporal": temporal,
        "config_resolved": dict(
            range_response=RangeResponseConfig("cfr_ifft", FREQUENCIES).to_dict(),
            bandwidth=BANDWIDTH,
            carrier_freq_hz=CARRIER,
            prf_hz=SAMPLE_RATE,
            fft_size_range=RANGE_FFT,
            fft_size_doppler=len(times),
            range_mode="monostatic",
            coherent_clutter_mode=clutter,
            processing_recipe=recipe,
        ),
        "frequency_axis_hz": frequencies,
        "slow_time_axis_s": times,
        "pair_coordinate": np.array([0]),
        "element_coordinate": np.array([0]),
        "observation": observation[None, None],
    }
    axes = [
        dense_axis(name, size=size, unit=unit, coordinate_key=key, coordinate_semantics=semantics)
        for name, size, unit, key, semantics in (
            ("pair", 1, "1", "pair_coordinate", "pair_index"),
            ("rx_element", 1, "1", "element_coordinate", "element_index"),
            ("slow_time", len(times), "s", "slow_time_axis_s", "physical_sample_time"),
            ("frequency", len(frequencies), "Hz", "frequency_axis_hz", "offset_from_carrier"),
        )
    ]
    sensing["products"] = {
        "observation": build_product_descriptor(
            sensing,
            data_key="observation",
            kind="baseband_frequency_observation",
            variant="primary",
            axes=axes,
            representation="complex_voltage",
            unit="1",
            scale_semantics="relative_unitless",
            pair_ids=(pair,),
            processing=dict(
                window_applied=False, carrier_frequency_hz=CARRIER, noise={"variance": 0.0}
            ),
        )
    }
    return sensing


def process(
    sensing, range_window="boxcar", doppler_window="boxcar", *, clean=False, size=RANGE_FFT
):
    """Use the identical shared CFR replay path used for retained cube observations."""
    return rebuild_frequency_observation(
        sensing,
        response=RangeResponseConfig("cfr_ifft", FREQUENCIES, range_window),
        fft_size=size,
        window_doppler=doppler_window,
        mti_enabled=clean,
    )


def direct_response(
    observation, range_window="boxcar", doppler_window="boxcar", *, size=RANGE_FFT, clean=False
):
    """Explicit double Fourier sum with independently expressed window normalization."""
    nt, nf = observation.shape
    k, n = np.arange(nf), np.arange(nt)
    wr = np.ones(nf) if range_window == "boxcar" else np.sin(np.pi * k / nf) ** 2
    wd = np.ones(nt) if doppler_window == "boxcar" else np.sin(np.pi * n / (nt - 1)) ** 2
    delay = (
        (observation * wr)
        @ np.exp(2j * np.pi * (k - nf // 2)[:, None] * np.arange(size) / size)
        / wr.sum()
    )
    source = delay - delay.mean(axis=0) if clean else delay
    ell = np.arange(-(nt // 2), nt - nt // 2)
    rd = (np.exp(-2j * np.pi * ell[:, None] * n / nt) * wd) @ source / np.sqrt(np.sum(wd**2))
    return delay, rd


def relative_error(actual, expected):
    return float(
        np.max(abs(actual - expected)) / max(float(np.max(abs(expected))), np.finfo(float).tiny)
    )


def db(power):
    """Unclipped dB for analysis; exact zero remains -inf."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(power)


def json_ready(value):
    """Use JSON null for undefined/nonfinite diagnostics, never invent a dB floor."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def lobe_metrics(axis, power):
    """Sampled -3 dB width and highest sidelobe outside nearest local minima.

    Crossing coordinates use linear interpolation solely to measure widths;
    neither the scientific arrays nor plots acquire interpolated samples.
    The axis must cover the mainlobe and both first minima without wrapping.
    """
    p = np.asarray(power) / np.max(power)
    peak = int(np.argmax(p))
    minima = np.flatnonzero((p[1:-1] < p[:-2]) & (p[1:-1] <= p[2:])) + 1
    left = minima[minima < peak]
    right = minima[minima > peak]
    lo, hi = (int(left[-1]) if len(left) else 0), (int(right[0]) if len(right) else len(p) - 1)
    below_left = np.flatnonzero(p[:peak] < 0.5)
    below_right = np.flatnonzero(p[peak + 1 :] < 0.5) + peak + 1
    width = None
    if len(below_left) and len(below_right):
        a, b = int(below_left[-1]), int(below_right[0])
        left_x = axis[a] + (0.5 - p[a]) / (p[a + 1] - p[a]) * (axis[a + 1] - axis[a])
        right_x = axis[b - 1] + (0.5 - p[b - 1]) / (p[b] - p[b - 1]) * (axis[b] - axis[b - 1])
        width = float(right_x - left_x)
    sidelobes = np.r_[p[:lo], p[hi + 1 :]]
    return dict(
        half_power_width=width,
        sampled_peak_sidelobe_db=float(db(sidelobes.max())) if len(sidelobes) else None,
        first_minima=[float(axis[lo]), float(axis[hi])],
    )


def measurements(result):
    power = abs(result["rd_cube"]) ** 2
    row, col = np.unravel_index(np.argmax(power), power.shape)
    ranges, doppler = result["range_axis_m"], result["doppler_axis_hz"]
    profile = result["range_profile"]
    ridge = power[row]
    zero = int(np.argmin(abs(doppler)))
    normalized_profile, normalized_ridge = profile / profile.max(), ridge / ridge.max()
    # Dense DTFT is a diagnostic measurement of the same finite record. The
    # 64-bin spectrum undersamples sidelobe minima, so measuring widths/PSL
    # from its polygon alone would be misleading for off-grid tones.
    nt = power.shape[0]
    n = np.arange(nt)
    ell = np.arange(-(nt // 2), nt - nt // 2)
    weighted_signal = (
        np.exp(2j * np.pi * n[:, None] * ell / nt) @ result["rd_cube"][:, col] / np.sqrt(nt)
    )
    dense_doppler = np.linspace(-SAMPLE_RATE / 2, SAMPLE_RATE / 2, 8192, endpoint=False)
    dense_power = (
        abs(np.exp(-2j * np.pi * dense_doppler[:, None] * n / SAMPLE_RATE) @ weighted_signal) ** 2
    )
    period = C0 * FREQUENCIES / (2 * BANDWIDTH)
    distance = abs((ranges - ranges[col] + period / 2) % period - period / 2)
    opposite = int(np.argmax(distance))
    return dict(
        range_peak_m=float(ranges[col]),
        velocity_peak_m_s=float(doppler[row] * C0 / (2 * CARRIER)),
        doppler_peak_hz=float(doppler[row]),
        peak_power=float(power[row, col]),
        zero_at_target_relative_db=float(db(power[zero, col] / power[row, col])),
        zero_row_max_relative_db=float(db(power[zero].max() / power[row, col])),
        range_lobe=lobe_metrics(ranges, ridge),
        doppler_lobe=lobe_metrics(dense_doppler, dense_power),
        far_range_peak_relative_db=float(db(ridge[distance > 2].max() / ridge.max())),
        profile_ridge_normalized_linf=float(np.max(abs(normalized_profile - normalized_ridge))),
        profile_zero_count=int(np.count_nonzero(profile == 0)),
        profile_min_relative_db=float(db(profile.min() / profile.max())),
        boundary_envelope_rise_db=float(
            db(profile[-16:].max() / profile[opposite - 8 : opposite + 8].max())
        ),
    )


def run_analytical():
    records = []
    for label, velocity, range0, frozen in (
        ("moving_exact", 1.8, 5.0, False),
        ("moving_frozen", 1.8, 5.0, True),
        ("stationary_on_grid", 0.0, 16 * C0 / (2 * BANDWIDTH), False),
        ("receding", -1.8, 5.0, False),
        ("doppler_on_grid", 20 * SAMPLE_RATE / SAMPLES * C0 / (2 * CARRIER), 5.0, True),
    ):
        for cpi in range(3):
            h, frequencies, times = point_observation(
                velocity=velocity, range0=range0, cpi=cpi, frozen_delay=frozen
            )
            source = observation_payload(h, frequencies, times)
            for rw, dw in WINDOWS:
                result = process(source, rw, dw)
                delay, rd = direct_response(h, rw, dw)
                records.append(
                    dict(
                        case=label,
                        cpi=cpi,
                        range_window=rw,
                        doppler_window=dw,
                        delay_relative_error=relative_error(result["delay"][0], delay),
                        rd_relative_error=relative_error(result["rd_cube"], rd),
                        **measurements(result),
                    )
                )
    return records


def cube_record(sensing, frame):
    """Audit the reference recording; reject incompatible acquisition settings."""
    cfg = sensing["config_resolved"]
    if any(
        cfg[key] != expected
        for key, expected in (
            ("bandwidth", BANDWIDTH),
            ("carrier_freq_hz", CARRIER),
            ("prf_hz", SAMPLE_RATE),
            ("fft_size_range", RANGE_FFT),
            ("range_mode", "monostatic"),
        )
    ) or sensing["cfr_by_pair_rx_element"].shape != (1, 1, SAMPLES, FREQUENCIES):
        raise ValueError(
            "This experiment requires the reference 28 GHz/500 MHz/64x128 SISO recording"
        )
    observation = sensing["cfr_by_pair_rx_element"][0, 0]
    rebuilt = process(sensing, "boxcar", "hann", clean=True)
    raw = rebuilt["rd_raw"]
    clean = rebuilt["rd_cube"]
    expected_delay, expected_raw = direct_response(observation, "boxcar", "hann")
    _, expected_clean = direct_response(observation, "boxcar", "hann", clean=True)
    samples = sensing["temporal"]["acquisition"]["samples"]
    windows = {f"{rw}/{dw}": measurements(process(sensing, rw, dw)) for rw, dw in WINDOWS}
    weighted_source = deepcopy(sensing)
    weighted_source["config_resolved"]["coherent_clutter_mode"] = "windowed_mean"
    weighted = process(weighted_source, "boxcar", "hann", clean=True)
    times = np.array([r["sample_time_s"] for r in samples])
    # Fit the dominant carrier tone to the aggregate observation, not individual
    # paths. Residual DC is a bound/diagnostic, not proof of static-path absence.
    carrier_channel = observation[:, FREQUENCIES // 2]
    tone = np.angle(np.vdot(carrier_channel[:-1], carrier_channel[1:])) * SAMPLE_RATE / (2 * np.pi)
    phase_fit = np.polyfit(times - times[0], np.unwrap(np.angle(carrier_channel)), 1)
    phase_residual = np.unwrap(np.angle(carrier_channel)) - np.polyval(phase_fit, times - times[0])
    mu = rebuilt["delay"][0].mean(axis=0)
    w = np.hanning(SAMPLES)
    dc_term = mu * w.sum() / np.sqrt(np.sum(w * w))
    zero = SAMPLES // 2
    dc_prediction = raw[zero] - dc_term
    bounce_counts = np.diff(frame.bounce_offsets)
    if not np.all(bounce_counts == 1) or len(sensing["truth"]["targets"]) != 1:
        raise ValueError("Path-velocity diagnostic requires one target and single-bounce paths")
    phase_spatial = np.asarray(frame.bounce_xyz_m) - np.asarray(frame.tx_positions)[0]
    directions = phase_spatial / np.linalg.norm(phase_spatial, axis=1)[:, None]
    target_velocity = np.asarray(sensing["truth"]["targets"][0]["velocity_m_s"])
    path_velocities = -directions @ target_velocity
    delay_ranges = np.asarray(frame.delays_ns) * 1e-9 * C0 / 2
    return dict(
        start_s=float(times[0]),
        end_s=float(times[-1]),
        samples=len(times),
        acquisition_id=sensing["temporal"]["acquisition"]["acquisition_id"],
        reference_time_s=sensing["temporal"]["result"]["reference_time_s"],
        cfr_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
        saved_delay_relative_error=relative_error(
            rebuilt["delay"], sensing["slow_time_by_pair_rx_element"][0]
        ),
        saved_raw_relative_error=relative_error(raw, sensing["range_doppler_raw_by_pair"][0]),
        saved_processed_relative_error=relative_error(clean, sensing["range_doppler_by_pair"][0]),
        saved_profile_relative_error=relative_error(
            rebuilt["range_profile"], sensing["range_profile_by_pair"][0]
        ),
        direct_delay_relative_error=relative_error(expected_delay, rebuilt["delay"][0]),
        direct_raw_relative_error=relative_error(expected_raw, raw),
        direct_processed_relative_error=relative_error(expected_clean, clean),
        windows=windows,
        processed=measurements(rebuilt),
        weighted_mean_diagnostic=measurements(weighted),
        zero_prediction_relative_error=relative_error(dc_prediction, clean[zero]),
        mean_projection_term_relative_db=float(
            db(np.max(abs(dc_term) ** 2) / np.max(abs(clean) ** 2))
        ),
        single_bounce_velocity_min_max_m_s=[
            float(path_velocities.min()),
            float(path_velocities.max()),
        ],
        path_count=int(len(bounce_counts)),
        zero_bounce_paths=int(np.count_nonzero(bounce_counts == 0)),
        bounce_count_min_max=[int(bounce_counts.min()), int(bounce_counts.max())],
        path_range_min_max_m=[float(delay_ranges.min()), float(delay_ranges.max())],
        fitted_carrier_doppler_hz=float(tone),
        phase_fit_max_residual_rad=float(np.max(abs(phase_residual))),
        detections=sensing.get("detections", []),
        truth=sensing.get("truth"),
        saved_profile_descriptor=sensing["products"]["range_profile_by_pair"],
        configured_windows={
            "frequency": cfg["range_response"]["frequency_window"],
            "doppler": cfg["doppler_taper"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Optional qualified 64x128 CFR recording with a 512-point range FFT; omit for an analytical-only run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from shared.frames.contracts import FrameComponent, FrameReadRequest
    from shared.frames.providers import Hdf5Provider

    records = []
    first = None
    if args.scenario is not None:
        provider = Hdf5Provider(args.scenario)
        try:
            for frame_id in (63, 127, 191):
                frame = provider.load_frame_projection(
                    frame_id, FrameReadRequest(components=frozenset(FrameComponent))
                ).frame
                records.append(cube_record(frame.sensing, frame))
                if frame_id == 63:
                    first = frame.sensing
        finally:
            provider.close()
    results = dict(
        configuration=dict(
            carrier_hz=CARRIER,
            bandwidth_hz=BANDWIDTH,
            frequency_samples=FREQUENCIES,
            frequency_spacing_hz=BANDWIDTH / FREQUENCIES,
            slow_time_samples=SAMPLES,
            slow_time_sampling_hz=SAMPLE_RATE,
            range_ifft_size=RANGE_FFT,
            scenario=str(args.scenario) if args.scenario else None,
            numpy_version=version("numpy"),
            range_lobe_measurement="512 saved samples; interpolated half-power crossings",
            doppler_lobe_measurement="8192-frequency DTFT of the same 64-sample record",
        ),
        analytical=run_analytical(),
        cube=records,
    )
    (args.output / "measurements.json").write_text(
        json.dumps(json_ready(results), indent=2, allow_nan=False) + "\n"
    )
    make_figures(first, args.output)
    print(
        json.dumps(
            dict(
                cube_events=len(records),
                analytical_cases=len(results["analytical"]),
                max_analytical_rd_error=max(r["rd_relative_error"] for r in results["analytical"]),
            ),
            indent=2,
        )
    )


def make_figures(source, output):
    """All comparisons use unchanged samples and shared dB limits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    h, f, t = point_observation()
    point = observation_payload(h, f, t)
    for name, sensing in (("point", point),) + ((("cube", source),) if source is not None else ()):
        results = [process(sensing, rw, dw) for rw, dw in WINDOWS]
        peak = max(float(db(abs(r["rd_cube"]) ** 2).max()) for r in results)
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, result, (rw, dw) in zip(axs.flat, results, WINDOWS, strict=True):
            im = ax.pcolormesh(
                result["range_axis_m"],
                result["doppler_axis_hz"] * C0 / (2 * CARRIER),
                db(abs(result["rd_cube"]) ** 2),
                shading="nearest",
                cmap="viridis",
                norm=Normalize(peak - 90, peak),
            )
            ax.set(
                xlim=(0, 8),
                ylim=(-2.7, 2.7),
                xlabel="Range (m)",
                ylabel="Closing velocity (m/s)",
                title=f"Frequency: {rw}; slow time: {dw}",
            )
        fig.colorbar(im, ax=axs, label="Power (dB re 1); shared scale; mean removal OFF")
        fig.savefig(output / f"{name}_windowing.png", dpi=150)
        plt.close(fig)
    if source is None:
        return
    original = process(source, "boxcar", "hann", clean=True)
    raw = process(source, "boxcar", "hann")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    peak = float(db(abs(original["rd_cube"]) ** 2).max())
    for ax, result, title in zip(
        axs,
        (original, raw),
        ("Saved primary: mean removal then Hann", "Saved raw: Hann only"),
        strict=True,
    ):
        im = ax.pcolormesh(
            result["range_axis_m"],
            result["doppler_axis_hz"] * C0 / (2 * CARRIER),
            db(abs(result["rd_cube"]) ** 2),
            shading="nearest",
            cmap="viridis",
            vmin=peak - 90,
            vmax=peak,
        )
        ax.set(xlim=(0, 8), xlabel="Range (m)", ylabel="Closing velocity (m/s)", title=title)
    fig.colorbar(im, ax=axs, label="Power (dB re 1)")
    fig.savefig(output / "cube_original_vs_raw.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ranges, power = original["range_axis_m"], original["range_profile"]
    edges = np.r_[
        ranges[0] - (ranges[1] - ranges[0]) / 2,
        (ranges[:-1] + ranges[1:]) / 2,
        ranges[-1] + (ranges[-1] - ranges[-2]) / 2,
    ]
    for row, limits in enumerate(((0, 38.4), (3, 7))):
        axs[row, 0].stairs(db(power), edges, baseline=None, lw=1)
        axs[row, 1].plot(ranges, db(power), lw=1, marker="." if row else None, ms=3)
        for col, title in enumerate(("Original steps", "Line through saved samples")):
            axs[row, col].set(
                xlim=limits, xlabel="Range (m)", ylabel="Power (dB re 1)", title=title
            )
            axs[row, col].grid(alpha=0.2)
    fig.savefig(output / "profile_before_after.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for name, sensing in (
        ("Cube", source),
        ("Moving point (exact)", point),
        ("Moving point (frozen delay)", observation_payload(*point_observation(frozen_delay=True))),
    ):
        result = process(sensing, "boxcar", "hann")
        p = abs(result["rd_cube"]) ** 2
        row, col = np.unravel_index(np.argmax(p), p.shape)
        # Align range peaks only for the shape comparison; never for calibration.
        axs[0].plot(
            result["range_axis_m"] - result["range_axis_m"][col],
            db(p[row] / p[row, col]),
            lw=1,
            label=name,
        )
        axs[1].plot(result["doppler_axis_hz"], db(p[:, col] / p[row, col]), lw=1, label=name)
        axs[2].plot(
            result["range_axis_m"],
            db(result["range_profile"] / result["range_profile"].max()),
            lw=1,
            label=name,
        )
    axs[0].set(
        xlim=(-2, 2),
        ylim=(-80, 3),
        xlabel="Range relative to sampled peak (m)",
        ylabel="Relative power (dB)",
    )
    axs[1].set(ylim=(-120, 3), xlabel="Doppler (Hz)")
    axs[2].set(ylim=(-100, 3), xlabel="Absolute range (m)")
    for ax in axs:
        ax.grid(alpha=0.2)
    axs[2].legend(fontsize=8)
    fig.savefig(output / "point_cube_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
