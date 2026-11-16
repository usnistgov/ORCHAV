"""Reproducible CPU qualification of range/Doppler windows on identical inputs.

Run ``python -m scripts.analysis.validate_sensing_hann --output <directory>``.
Add ``--recordings`` for read-only replay of the existing tutorial acquisitions.
The dense spectra measure the finite record, not extra physical resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from scripts.analysis.validate_range_doppler import (
    BANDWIDTH,
    C0,
    CARRIER,
    FREQUENCIES,
    RANGE_FFT,
    SAMPLE_RATE,
    SAMPLES,
    WINDOWS,
    db,
    direct_response,
    json_ready,
    observation_payload,
    point_observation,
    process,
    relative_error,
)
from shared.sensing import RangeDopplerResult
from shared.sensing.cfr_replay import rebuild_frequency_observation
from shared.sensing.detection.pipeline import DetectionPipelineConfig, run_detection_pipeline
from shared.sensing.range_response import RangeResponseConfig

AMPLITUDE = 0.6 + 0.8j
RANGE_SCALE = C0 / (2 * BANDWIDTH)
VELOCITY_SCALE = C0 * SAMPLE_RATE / (2 * CARRIER * SAMPLES)
# Fixed before the experiments. No GT is supplied to the detector. This is a
# deterministic sidelobe/sensitivity experiment, not a noise-ensemble Pfa test.
DETECTOR = DetectionPipelineConfig(
    cfar_guard_cells=(2, 8),
    cfar_reference_cells=(4, 16),
    cfar_threshold_scale=5.0,
    min_snr_db=10.0,
    nms_suppression_radius=(2, 4),
    cluster_range_gate_m=0.2,
    cluster_velocity_gate_m_s=0.1,
    cfar_max_dynamic_range_db=40.0,
    min_detection_range_m=1.0,
    max_detection_range_m=12.0,
)
CASES = {
    "stationary": [(5.0, 0.0, AMPLITUDE)],
    "approaching": [(5.0, 1.0, AMPLITUDE)],
    "receding": [(5.0, -1.0, AMPLITUDE)],
    "two_separated": [(5.0, 1.0, AMPLITUDE), (8.0, -1.0, AMPLITUDE)],
    "strong_weak": [(5.0, 0.0, AMPLITUDE), (6.05, 0.0, 0.03 * AMPLITUDE)],
    "unresolved_pair": [(5.0, 0.0, AMPLITUDE), (5.1, 0.0, 0.5 * AMPLITUDE)],
}


def synthetic(targets, *, migration=False):
    """Positive velocity means closing; frozen envelope matches native one-solve CPI."""
    observations = []
    for range_m, velocity, amplitude in targets:
        h, frequencies, times = point_observation(
            range0=range_m, velocity=velocity, amplitude=amplitude, frozen_delay=not migration
        )
        observations.append(h)
    observation = np.sum(observations, axis=0)
    return observation_payload(observation, frequencies, times, clutter="none")


def detect(result):
    """Use production CFAR/NMS/clustering on the exact plotted linear RD power."""
    rd = result["rd_cube"]
    return run_detection_pipeline(
        RangeDopplerResult(
            power_linear=abs(rd) ** 2,
            cube=rd,
            range_axis_m=result["range_axis_m"],
            doppler_axis_hz=result["doppler_axis_hz"],
        ),
        config=DETECTOR,
        wavelength_m=C0 / CARRIER,
    ).detections


def match_detections(detections, targets):
    """One-to-one evaluation gates; never used by the detector or window processing."""
    unused = set(range(len(detections)))
    matches = []
    for target_index, (range_m, velocity, _) in enumerate(targets):
        candidates = [
            i
            for i in unused
            if abs(detections[i]["range_m"] - range_m) <= 0.15
            and abs(detections[i]["velocity_m_s"] - velocity) <= 0.13
        ]
        if candidates:
            i = min(candidates, key=lambda j: abs(detections[j]["range_m"] - range_m))
            unused.remove(i)
            matches.append(dict(target=target_index, detection=i))
    return dict(tp=len(matches), fn=len(targets) - len(matches), fp=len(unused), matches=matches)


def score_recording(detections, truth):
    """One-to-one center association at fixed 0.60 m / 0.25 m/s review gates."""
    from scipy.optimize import linear_sum_assignment

    targets = [t for t in (truth or {}).get("targets", []) if t.get("valid")]
    costs = np.full((len(targets), len(detections)), 1e6)
    errors = {}
    for i, target in enumerate(targets):
        observable = target["pair_observables"][0]
        for j, detection in enumerate(detections):
            dr = detection["range_m"] - observable["monostatic_equivalent_range_m"]
            dv = detection["velocity_m_s"] - observable["monostatic_equivalent_velocity_m_s"]
            if abs(dr) <= 0.6 and abs(dv) <= 0.25:
                costs[i, j] = (dr / 0.6) ** 2 + (dv / 0.25) ** 2
                errors[i, j] = dict(
                    target=target["target_id"], range_error_m=dr, velocity_error_m_s=dv
                )
    i, j = linear_sum_assignment(costs)
    matches = [errors[a, b] for a, b in zip(i, j) if costs[a, b] < 1e6]
    return dict(
        tp=len(matches),
        fn=len(targets) - len(matches),
        fp=len(detections) - len(matches),
        matches=matches,
    )


def lobe_metrics(coordinates, power, *, center, period, first_null):
    """PSL=max side/peak; ISL=sum side/sum main over one circular ambiguity period.

    Mainlobe is bounded by the analytic first nulls, not a fixed bin count or a
    threshold. Half-power width interpolates adjacent densely sampled powers.
    """
    offset = (np.asarray(coordinates) - center + period / 2) % period - period / 2
    order = np.argsort(offset)
    x, p = offset[order], np.asarray(power)[order]
    p = p / p.max()
    main = abs(x) < first_null
    high = np.flatnonzero((p >= 0.5) & main)
    lo, hi = int(high[0]), int(high[-1])
    left = x[lo - 1] + (0.5 - p[lo - 1]) / (p[lo] - p[lo - 1]) * (x[lo] - x[lo - 1])
    right = x[hi] + (0.5 - p[hi]) / (p[hi + 1] - p[hi]) * (x[hi + 1] - x[hi])
    return dict(
        psl_db=float(db(p[~main].max())),
        isl_db=float(db(p[~main].sum() / p[main].sum())),
        half_power_width=float(right - left),
        null_to_null_width=2 * first_null,
        minimum_relative_db=float(db(p.min())),
    )


def dense_profiles(source, rw, dw, *, range_m=5.0, velocity=0.0):
    """Oversample the actual production range transform and finite Doppler record."""
    result = process(source, rw, dw, size=FREQUENCIES * 64)
    r = result["range_axis_m"]
    j = int(np.argmin(abs(r - range_m)))
    # Recover the exact already-windowed temporal sequence from production RD.
    weighted = np.fft.ifft(np.fft.ifftshift(result["rd_cube"][:, j])) * np.sqrt(SAMPLES)
    size = SAMPLES * 256
    spectrum = np.fft.fftshift(np.fft.fft(weighted, n=size)) / np.sqrt(SAMPLES)
    v = np.fft.fftshift(np.fft.fftfreq(size, 1 / SAMPLE_RATE)) * C0 / (2 * CARRIER)
    rp = result["range_profile"]
    dp = abs(spectrum) ** 2
    return dict(
        range_axis=r,
        range_power=rp,
        velocity_axis=v,
        doppler_power=dp,
        range=lobe_metrics(
            r,
            rp,
            center=range_m,
            period=FREQUENCIES * RANGE_SCALE,
            first_null=RANGE_SCALE * (2 if rw == "hann" else 1),
        ),
        doppler=lobe_metrics(
            v,
            dp,
            center=velocity,
            period=SAMPLES * VELOCITY_SCALE,
            first_null=VELOCITY_SCALE * (2 * SAMPLES / (SAMPLES - 1) if dw == "hann" else 1),
        ),
    )


def analytical(output):
    rows, plotted = [], {}
    for case, targets in CASES.items():
        source = synthetic(targets)
        for rw, dw in WINDOWS:
            result = process(source, rw, dw)
            reference_delay, reference_rd = direct_response(source["observation"][0, 0], rw, dw)
            power = abs(result["rd_cube"]) ** 2
            row, col = np.unravel_index(power.argmax(), power.shape)
            detections = detect(result)
            record = dict(
                case=case,
                range_window=rw,
                doppler_window=dw,
                input_sha256=hashlib.sha256(source["observation"].tobytes()).hexdigest(),
                delay_relative_error=relative_error(result["delay"][0], reference_delay),
                rd_relative_error=relative_error(result["rd_cube"], reference_rd),
                peak_range_m=float(result["range_axis_m"][col]),
                peak_velocity_m_s=float(result["doppler_axis_hz"][row] * C0 / (2 * CARRIER)),
                sampled_peak_power=float(power[row, col]),
                last_profile_peak_power=float(result["range_profile"].max()),
                detections=detections,
                detection_score=match_detections(detections, targets),
                exact_zero_bins=int(np.count_nonzero(power == 0)),
            )
            if len(targets) == 1:
                record["range_error_m"] = record["peak_range_m"] - targets[0][0]
                record["velocity_error_m_s"] = record["peak_velocity_m_s"] - targets[0][1]
                dense = dense_profiles(
                    source, rw, dw, range_m=targets[0][0], velocity=targets[0][1]
                )
                record.update(range=dense["range"], doppler=dense["doppler"])
                plotted[(case + "_dense", rw, dw)] = dense
            if case == "stationary":
                w = np.hanning(SAMPLES) if dw == "hann" else np.ones(SAMPLES)
                record["doppler_coherent_power_gain"] = float(w.sum() ** 2 / np.sum(w * w))
                record["profile_zero_slice_normalized_error"] = relative_error(
                    power[SAMPLES // 2] / power.max(),
                    result["range_profile"] / result["range_profile"].max(),
                )
                plotted[("dense", rw, dw)] = dense
            rows.append(record)
            plotted[(case, rw, dw)] = result
    # Explicit range-migration control is distinct from the frozen-envelope model.
    migration = []
    for velocity in (1.0, -1.0):
        source = synthetic([(5.0, velocity, AMPLITUDE)], migration=True)
        for rw, dw in WINDOWS:
            result = process(source, rw, dw)
            _, reference = direct_response(source["observation"][0, 0], rw, dw)
            i, j = np.unravel_index(abs(result["rd_cube"]).argmax(), result["rd_cube"].shape)
            migration.append(
                dict(
                    velocity=velocity,
                    range_window=rw,
                    doppler_window=dw,
                    midpoint_range_m=5 - velocity * (SAMPLES - 1) / (2 * SAMPLE_RATE),
                    peak_range_m=float(result["range_axis_m"][j]),
                    peak_velocity_m_s=float(result["doppler_axis_hz"][i] * C0 / (2 * CARRIER)),
                    rd_relative_error=relative_error(result["rd_cube"], reference),
                )
            )
    make_analytical_figures(plotted, output)
    return dict(
        parameters=dict(
            carrier_hz=CARRIER,
            bandwidth_hz=BANDWIDTH,
            frequency_samples=FREQUENCIES,
            slow_time_samples=SAMPLES,
            sample_rate_hz=SAMPLE_RATE,
            range_ifft_size=RANGE_FFT,
            range_scale_m=RANGE_SCALE,
            velocity_bin_m_s=VELOCITY_SCALE,
            amplitude=[AMPLITUDE.real, AMPLITUDE.imag],
        ),
        detector=asdict(DETECTOR),
        cases=rows,
        migration=migration,
    )


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_maps(results, titles, output, name, *, xlim=None, ylim=None):
    """Compare all stored bins by default, with one absolute power scale."""
    plt = plotting()
    from matplotlib.colors import Normalize

    maximum = max(float(db(abs(r["rd_cube"]) ** 2).max()) for r in results)
    # Exact zero power is -inf dB. Matplotlib masks it; use the bottom color
    # instead of transparent white, without flooring the scientific array.
    cmap = plt.get_cmap("magma")
    cmap = cmap.with_extremes(bad=cmap(0.0))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, result, title in zip(axes.flat, results, titles, strict=True):
        v = result["doppler_axis_hz"] * C0 / (2 * result.get("carrier_hz", CARRIER))
        im = ax.pcolormesh(
            result["range_axis_m"],
            v,
            db(abs(result["rd_cube"]) ** 2),
            shading="nearest",
            norm=Normalize(maximum - 80, maximum),
            cmap=cmap,
            rasterized=True,
        )
        ax.set(
            xlim=xlim, ylim=ylim, title=title, xlabel="Range (m)", ylabel="Closing velocity (m/s)"
        )
    fig.colorbar(im, ax=axes, label="Power (dB re 1 channel unit); common absolute scale")
    fig.savefig(output / name, dpi=150)
    plt.close(fig)


def make_analytical_figures(results, output):
    plt = plotting()
    titles = [f"{chr(65+i)}: range {rw}, Doppler {dw}" for i, (rw, dw) in enumerate(WINDOWS)]
    for case in CASES:
        draw_maps([results[(case, rw, dw)] for rw, dw in WINDOWS], titles, output, f"{case}_rd.png")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for rw, dw in WINDOWS:
        d = results[("dense", rw, dw)]
        label = f"range {rw}, Doppler {dw}"
        axs[0].plot(d["range_axis"], db(d["range_power"] / d["range_power"].max()), label=label)
        axs[1].plot(
            d["velocity_axis"], db(d["doppler_power"] / d["doppler_power"].max()), label=label
        )
    axs[0].set(xlim=(3, 7), ylim=(-80, 1), xlabel="Range (m)", ylabel="Peak-normalized power (dBc)")
    axs[1].set(xlim=(-0.7, 0.7), ylim=(-80, 1), xlabel="Closing velocity (m/s)")
    for ax in axs:
        ax.grid(alpha=0.25)
    axs[1].legend(fontsize=8)
    fig.savefig(output / "window_profiles.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for col, case in enumerate(("stationary", "approaching", "receding")):
        for rw, dw in WINDOWS:
            d = results[(case + "_dense", rw, dw)]
            label = f"range {rw}, Doppler {dw}"
            axs[0, col].plot(
                d["range_axis"], db(d["range_power"] / d["range_power"].max()), label=label
            )
            axs[1, col].plot(
                d["velocity_axis"], db(d["doppler_power"] / d["doppler_power"].max()), label=label
            )
        velocity = CASES[case][0][1]
        axs[0, col].set(xlim=(3, 7), ylim=(-80, 2), title=case, xlabel="Range (m)")
        axs[1, col].set(
            xlim=(velocity - 0.6, velocity + 0.6), ylim=(-80, 2), xlabel="Closing velocity (m/s)"
        )
        for row in (0, 1):
            axs[row, col].grid(alpha=0.25)
    axs[0, 0].set_ylabel("Peak-normalized power (dBc)")
    axs[1, 0].set_ylabel("Peak-normalized power (dBc)")
    axs[0, 2].legend(fontsize=7)
    fig.savefig(output / "stationary_moving_profiles.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for case, ax in zip(("strong_weak", "unresolved_pair"), axs, strict=True):
        for rw in ("boxcar", "hann"):
            result = results[(case, rw, "hann")]
            ax.plot(result["range_axis_m"], db(result["range_profile"]), label=rw)
            for detection in detect(result):
                x = detection["range_m"]
                if 4 < x < 7:
                    ax.axvline(x, alpha=0.25, ls=":" if rw == "hann" else "--")
        for r, _, a in CASES[case]:
            ax.plot(r, float(db(abs(a) ** 2)), "kx")
        ax.set(
            xlim=(4, 7),
            ylim=(-70, 5),
            title=case,
            xlabel="Range (m)",
            ylabel="Last-sample power (dB re 1)",
        )
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(output / "strong_weak_and_resolution.png", dpi=160)
    plt.close(fig)


RECORDINGS = (
    "foundations/stationary_specular_cube",
    "foundations/radial_velocity_cube",
    "targets_motion/human_target_tracking",
    "detection_tracking/multi_target_tracking",
    "applications/paris_opera_urban_tracking",
)


def recordings(output, overrides=None):
    from scripts.benchmarks.benchmark_cfar_replay import saved_processor
    from shared.sensing.loader import load_frames_sensing

    rows = []
    for name in RECORDINGS:
        path = (overrides or {}).get(name, Path("scenarios/sensing") / name)
        if not (path / "frames" / "frames_manifest.json").exists():
            rows.append(dict(scenario=name, unavailable=True))
            continue
        frames = load_frames_sensing(path)
        case_rows = []
        for index, frame in enumerate(frames):
            sensing = frame["_typed_sensing"]
            cfg = sensing["config_resolved"]
            stored_response = RangeResponseConfig.parse(cfg["range_response"])
            base_processor = saved_processor(sensing)
            for rw in ("boxcar", "hann"):
                response = replace(stored_response, frequency_window=rw)
                processor = type(base_processor)(
                    replace(base_processor.config, range_response=response)
                )
                result = processor.process_from_frequency_observation(sensing)
                rebuilt = result["cfr_rebuilt"]
                saved_window = rw == stored_response.frequency_window
                record = dict(
                    event=index,
                    range_window=rw,
                    doppler_window=rebuilt["metadata"]["doppler_taper"],
                    input_sha256=hashlib.sha256(
                        sensing["cfr_by_pair_rx_element"].tobytes()
                    ).hexdigest(),
                    detections=result["detections"],
                    truth=sensing.get("truth"),
                    saved_detections=sensing.get("detections", []),
                    saved_rd_relative_error=(
                        relative_error(rebuilt["rd_cube"], sensing["range_doppler_by_pair"][0])
                        if saved_window
                        else None
                    ),
                    saved_profile_relative_error=(
                        relative_error(
                            rebuilt["range_profile"], sensing["range_profile_by_pair"][0]
                        )
                        if saved_window
                        else None
                    ),
                    mean_removal_applied=rebuilt["metadata"]["slow_time_mean_removal_applied"],
                    raw_primary_difference=relative_error(rebuilt["rd_cube"], rebuilt["rd_raw"]),
                    center_score=score_recording(result["detections"], sensing.get("truth")),
                )
                if name.endswith("stationary_specular_cube"):
                    power = abs(rebuilt["rd_cube"]) ** 2
                    zero = int(np.argmin(abs(rebuilt["doppler_axis_hz"])))
                    record["zero_slice_profile_normalized_error"] = relative_error(
                        power[zero] / power[zero].max(),
                        rebuilt["range_profile"] / rebuilt["range_profile"].max(),
                    )
                    h = sensing["cfr_by_pair_rx_element"][0, 0]
                    f = sensing["frequency_axis_hz"]
                    delay = -np.angle(np.vdot(h[0, :-1], h[0, 1:])) / (2 * np.pi * (f[1] - f[0]))
                    expected = h[0, 0] * np.exp(-2j * np.pi * (f - f[0]) * delay)
                    record["fitted_reflector_range_m"] = float(delay * C0 / 2)
                    record["single_path_cfr_relative_error"] = relative_error(h, expected)
                    record["slow_time_relative_variation"] = float(
                        abs(np.diff(h, axis=0)).max() / abs(h).max()
                    )
                case_rows.append(record)
            if index == 0:
                result_set = [
                    rebuild_frequency_observation(
                        sensing,
                        response=replace(stored_response, frequency_window=rw),
                        window_doppler=dw,
                    )
                    for rw, dw in WINDOWS
                ]
                for result in result_set:
                    result["carrier_hz"] = cfg["carrier_freq_hz"]
                draw_maps(
                    result_set,
                    [f"range {rw}, Doppler {dw}" for rw, dw in WINDOWS],
                    output,
                    name.rsplit("/", 1)[-1] + "_windows.png",
                )
                if name.endswith("stationary_specular_cube"):
                    stationary_figure(sensing, result_set, output)
        rows.append(
            dict(scenario=name, source=str(path), events=len(frames), comparisons=case_rows)
        )
    return rows


def stationary_figure(sensing, results, output):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for rw, result in zip(("boxcar", "hann"), (results[2], results[3]), strict=True):
        r = result["range_axis_m"]
        p = result["range_profile"]
        ridge = abs(result["rd_cube"][len(result["doppler_axis_hz"]) // 2]) ** 2
        axes[0].plot(r, db(p / p.max()), label=rw + " last sample")
        axes[0].plot(r, db(ridge / ridge.max()), ls="--", label=rw + " zero-Doppler slice")
        axes[1].plot(r, db(p), label=rw + " last sample")
        axes[1].plot(r, db(ridge), ls="--", label=rw + " coherent zero-Doppler")
    axes[0].set(xlim=(0, 38.4), ylim=(-100, 2), ylabel="Peak-normalized power (dBc)")
    axes[1].set(xlim=(3.5, 5.5), ylabel="Power (dB re 1 channel unit)")
    for ax in axes:
        ax.set_xlabel("Range (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(output / "stationary_stripe_profile.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings", action="store_true")
    parser.add_argument(
        "--recording",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Use an existing acquisition for a named tutorial, without generating frames",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = analytical(args.output)
    report["environment"] = {}
    for name in ("numpy", "scipy", "matplotlib", "PySide6", "sionna-rt"):
        try:
            report["environment"][name] = version(name)
        except PackageNotFoundError:
            report["environment"][name] = "not installed (optional for the CPU experiment)"
    report["source_sha256"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "shared/sensing/range_response.py",
            "shared/sensing/cfr_replay.py",
            "shared/sensing/signal_processing.py",
            "shared/sensing/detection/pipeline.py",
            "generator/core/sensing/strategies/coherent_cpi_fft.py",
            "scripts/analysis/validate_sensing_hann.py",
        )
    }
    if args.recordings:
        overrides = {
            name: Path(path) for name, path in (item.split("=", 1) for item in args.recording)
        }
        report["recordings"] = recordings(args.output, overrides)
    (args.output / "measurements.json").write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {len(report['cases'])} analytical comparisons to {args.output}")


if __name__ == "__main__":
    main()
