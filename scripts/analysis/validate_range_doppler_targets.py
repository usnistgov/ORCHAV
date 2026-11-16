"""Live constant-RCS, diffuse/specular cube, and point-bin comparison.

Run with a Sionna RT 2.2 environment:
python -m scripts.analysis.validate_range_doppler_targets --output <directory>
Use --skip-native to repeat the mesh controls on older Sionna installations.
No scenario inputs, production defaults, or saved frames are modified.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

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
    measurements,
    observation_payload,
    point_observation,
    process,
    relative_error,
)


def acquire(case, cpi=0):
    """Use ORCHAV's native target adapter or the unchanged material cube mesh."""
    import drjit as dr
    import mitsuba as mi
    import sionna.rt as rt

    from generator.core.propagation.native_rcs import compute_native_paths
    from generator.core.target.rcs import NativeTargetManager
    from generator.core.target.representation import NativeTargetConfig
    from shared.scenarios.actors import ConstantRCSSpec, NativeBoxGeometrySpec

    dr.set_thread_count(2)
    scene = rt.load_scene()
    scene.frequency = CARRIER
    for name in ("tx_array", "rx_array"):
        setattr(
            scene,
            name,
            rt.PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V",
            ),
        )
    scene.add(rt.Transmitter("tx", position=[0, 0, 0.5]))
    scene.add(rt.Receiver("rx", position=[0, 0, 0.5]))
    radial = case == "constant_radial"
    velocity = np.array([0.0, -1.8 if radial else -0.35 / 0.191, 0.0])
    position = np.array([0.0 if radial else 1.0, 5.0, 0.5]) + velocity * cpi * SAMPLES / SAMPLE_RATE
    kwargs = dict(
        scene=scene,
        max_depth=2,
        samples_per_src=100000,
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=case == "cube_diffuse",
        refraction=False,
        seed=42,
        max_num_paths_per_src=500000,
    )
    if case.startswith("constant"):
        config = NativeTargetConfig(
            name="target",
            representation="constant_rcs",
            geometry=NativeBoxGeometrySpec(length_m=1, width_m=1, height_m=1),
            constant_rcs=ConstantRCSSpec(sigma_m2=1),
            _initial_position=tuple(position),
            _initial_velocity=tuple(velocity),
        )
        manager = NativeTargetManager(config, scene)
        manager.create_target()
        simulation = SimpleNamespace(
            scene=scene,
            tx_list=list(scene.transmitters.values()),
            rx_list=list(scene.receivers.values()),
            target_managers=[manager],
            settings={
                "rcs": dict(
                    channel="target_only",
                    max_depth=1,
                    samples_per_sp=32,
                    buffer_size_per_sp=32,
                    los=True,
                    specular_reflection=False,
                    refraction=False,
                    seed=42,
                )
            },
            path_solver=rt.PathSolver(),
            rcs_solver=None,
        )
        paths, native_metadata = compute_native_paths(simulation, kwargs)
    else:
        material = rt.ITURadioMaterial(
            "metal",
            "metal",
            thickness=0.01,
            scattering_coefficient=0.99,
            scattering_pattern="lambertian",
        )
        cube = rt.SceneObject(
            name="cube",
            fname=str(Path(__file__).resolve().parents[2] / "libraries/targets/cube/cube.ply"),
            radio_material=material,
        )
        scene.edit(add=cube)
        cube.position = mi.Point3f(position.tolist())
        cube.orientation = [
            (
                float(np.arctan2(position[1], position[0]))
                if case == "cube_specular_aligned"
                else -np.pi / 2
            ),
            0,
            0,
        ]
        cube.velocity = mi.Vector3f(velocity.tolist())
        paths = rt.PathSolver()(**kwargs)
        native_metadata = None
    a, tau = paths.cir(
        sampling_frequency=SAMPLE_RATE,
        num_time_steps=SAMPLES,
        normalize_delays=False,
        out_type="numpy",
    )
    valid = tau[0, 0] >= 0
    a = a[0, 0, 0, 0, valid].astype(np.complex128)
    tau = tau[0, 0, valid].astype(float)
    offsets = (np.arange(FREQUENCIES) - FREQUENCIES // 2) * BANDWIDTH / FREQUENCIES
    native_cfr = paths.cfr(
        mi.Float(offsets),
        sampling_frequency=SAMPLE_RATE,
        num_time_steps=SAMPLES,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )[0, 0, 0, 0]
    return dict(
        a=a,
        tau=tau,
        native_cfr=native_cfr,
        position=position,
        velocity=velocity,
        case=case,
        cpi=cpi,
        native_metadata=native_metadata,
        backend=mi.variant(),
        versions={p: version(p) for p in ("sionna-rt", "mitsuba", "drjit")},
    )


def native_processing(a, tau, model, *, cpi=0):
    """Feed the identical path coefficients into each production strategy branch."""
    from generator.core.configuration import AntennaConfig, SensingConfig
    from generator.core.sensing.acquisition import resolve_sensing_execution_values
    from generator.core.sensing.processor import create_sensing_processor
    from shared.sensing.range_response import RangeResponseConfig

    config = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        bandwidth=BANDWIDTH,
        prf_hz=SAMPLE_RATE,
        fft_size_range=RANGE_FFT,
        fft_size_doppler=SAMPLES,
        window_range="boxcar",
        window_doppler="hann",
        range_mode="monostatic",
        raw_payload_policy="always",
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        range_response=(
            RangeResponseConfig(model, FREQUENCIES)
            if model == "cfr_ifft"
            else RangeResponseConfig()
        ),
    )
    execution = resolve_sensing_execution_values(
        config,
        cir_time_steps=SAMPLES,
        cir_sampling_frequency_hz=SAMPLE_RATE,
        frame_dt_s=1 / SAMPLE_RATE,
        rx_elements=1,
    )
    processor = create_sensing_processor(config, CARRIER, execution)
    simulation = SimpleNamespace(simulation_config=SimpleNamespace(rx_antenna=AntennaConfig()))
    return processor.process(
        simulation,
        cpi * SAMPLES,
        None,
        cir_stack=(a, tau),
        source_time_s=cpi * SAMPLES / SAMPLE_RATE,
    )


def analyze(acquisition):
    """Measure complex parity, power calibration, windows and the floor projection."""
    a, tau, cpi = (acquisition[k] for k in ("a", "tau", "cpi"))
    record = {k: v for k, v in acquisition.items() if k not in {"a", "tau", "native_cfr"}}
    record["path_count"] = len(tau)
    if not len(tau):
        record["interpretation"] = "No target echo; no normalized sidelobe/DC ratios are defined"
        return record, None
    cfr = native_processing(a, tau, "cfr_ifft", cpi=cpi)
    floor = native_processing(a, tau, "point_bin_floor", cpi=cpi)
    h = cfr["cfr_by_pair_rx_element"][0, 0]
    times = (cpi * SAMPLES + np.arange(SAMPLES)) / SAMPLE_RATE
    source = observation_payload(h, cfr["frequency_axis_hz"], times)
    replay = process(source, "boxcar", "hann", clean=True)
    _, reference = direct_response(h, "boxcar", "hann", clean=True)
    record.update(
        path_ranges_m=tau * C0 / 2,
        cfr_native_backend_relative_error=relative_error(h, acquisition["native_cfr"]),
        replay_generator_relative_error=relative_error(
            replay["rd_cube"], cfr["range_doppler_by_pair"][0]
        ),
        direct_generator_relative_error=relative_error(reference, cfr["range_doppler_by_pair"][0]),
        windows={f"{rw}/{dw}": measurements(process(source, rw, dw)) for rw, dw in WINDOWS},
        primary=measurements(replay),
    )
    floor_power = abs(floor["range_doppler_by_pair"][0]) ** 2
    row, col = np.unravel_index(np.argmax(floor_power), floor_power.shape)
    expected_floor = np.zeros((SAMPLES, RANGE_FFT), complex)
    for coeff, delay in zip(a, tau, strict=True):
        expected_floor[:, int(np.floor(delay * BANDWIDTH))] += coeff
    record["point_bin_floor"] = dict(
        sample_spacing_m=float(np.diff(floor["range_axis_m"])[0]),
        last_range_m=float(floor["range_axis_m"][-1]),
        occupied_bins=int(np.count_nonzero(floor["range_profile_by_pair"][0])),
        range_peak_m=float(floor["range_axis_m"][col]),
        velocity_peak_m_s=float(floor["doppler_axis_hz"][row] * C0 / (2 * CARRIER)),
        zero_at_peak_db=float(db(floor_power[SAMPLES // 2, col] / floor_power[row, col])),
        projection_relative_error=relative_error(expected_floor, cir_from_result(floor)),
        profile_relative_error=relative_error(
            abs(expected_floor[-1]) ** 2, floor["range_profile_by_pair"][0]
        ),
    )
    if len(tau) == 1 and acquisition["case"].startswith("constant"):
        # Remove only one global complex gain for the shape comparison. Delay
        # and Doppler are independently derived from the configured geometry.
        displacement = acquisition["position"] - np.array([0, 0, 0.5])
        r = float(np.linalg.norm(displacement))
        v = float(-np.dot(displacement / r, acquisition["velocity"]))
        expected_power = (C0 / CARRIER) ** 2 / ((4 * np.pi) ** 3 * r**4)
        offsets = cfr["frequency_axis_hz"]
        model = np.exp(-4j * np.pi * (CARRIER + offsets)[None, :] * r / C0) * np.exp(
            4j * np.pi * CARRIER * v * (times - times[0])[:, None] / C0
        )
        gain = np.vdot(model, h) / np.vdot(model, model)
        record["single_point_reference"] = dict(
            range_m=r,
            closing_velocity_m_s=v,
            complex_shape_relative_error=relative_error(h, gain * model),
            radar_equation_power=expected_power,
            observed_path_power=float(abs(a[0, 0]) ** 2),
            radar_equation_relative_error=float(abs(abs(a[0, 0]) ** 2 / expected_power - 1)),
        )
    return record, dict(cfr=cfr, floor=floor, source=source)


def cir_from_result(sensing):
    return sensing["slow_time_by_pair_rx_element"][0, 0]


def figures(all_results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = {name: result for name, result in all_results.items() if result is not None}
    fig, axs = plt.subplots(
        len(all_results),
        3,
        figsize=(15, 3.2 * len(all_results)),
        constrained_layout=True,
        squeeze=False,
    )
    peak = max(
        float(db(abs(r["cfr"]["range_doppler_by_pair"][0]) ** 2).max()) for r in available.values()
    )
    for axes, (name, result) in zip(axs, all_results.items(), strict=True):
        if result is None:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No echo\nNo valid propagation paths",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            axes[0].set_title(name)
            continue
        for ax, model in zip(axes[:2], ("cfr", "floor"), strict=True):
            s = result[model]
            ax.pcolormesh(
                s["range_axis_m"],
                s["doppler_axis_hz"] * C0 / (2 * CARRIER),
                db(abs(s["range_doppler_by_pair"][0]) ** 2),
                shading="nearest",
                vmin=peak - 90,
                vmax=peak,
            )
            ax.set(
                xlim=(0, 8),
                ylim=(-2.7, 2.7),
                xlabel="Range (m)",
                ylabel="Closing velocity (m/s)",
                title=f"{name}: {model}",
            )
        for model in ("cfr", "floor"):
            s = result[model]
            power = s["range_profile_by_pair"][0]
            # Exact zero is absent on a logarithmic axis. Mark occupied floor
            # bins rather than connecting zeros at an invented noise level.
            mask = power > 0
            axes[2].plot(
                s["range_axis_m"][mask],
                db(power[mask]),
                label=model,
                marker="." if model == "floor" else None,
                lw=1,
                linestyle="none" if model == "floor" else "-",
            )
        axes[2].set(
            xlim=(0, 38.4),
            xlabel="Range (m)",
            ylabel="Profile power (dB re 1)",
            title="Last acquisition sample",
        )
        axes[2].legend()
    fig.suptitle(
        "Same paths, different projections · primary RD: mean removal then Hann\nRD common color scale: "
        + f"{peak-90:.1f} to {peak:.1f} dB re 1; no amplitude equalization"
    )
    fig.savefig(output / "target_projection_comparison.png", dpi=140)
    plt.close(fig)
    if "constant_radial" in available:
        source = available["constant_radial"]["source"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        results = [process(source, rw, dw) for rw, dw in WINDOWS]
        peak = max(float(db(abs(r["rd_cube"]) ** 2).max()) for r in results)
        for ax, result, (rw, dw) in zip(axs.flat, results, WINDOWS, strict=True):
            im = ax.pcolormesh(
                result["range_axis_m"],
                result["doppler_axis_hz"] * C0 / (2 * CARRIER),
                db(abs(result["rd_cube"]) ** 2),
                shading="nearest",
                vmin=peak - 90,
                vmax=peak,
            )
            ax.set(
                xlim=(0, 8),
                xlabel="Range (m)",
                ylabel="Closing velocity (m/s)",
                title=f"Constant RCS: {rw}/{dw}",
            )
        fig.colorbar(im, ax=axs, label="Power (dB re 1); mean removal OFF")
        fig.savefig(output / "native_rcs_windowing.png", dpi=150)
        plt.close(fig)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ideal = observation_payload(*point_observation(frozen_delay=True))
        for label, payload, style in (
            ("Native 1 m² RCS", source, "-"),
            ("Analytical frozen-delay point", ideal, "--"),
        ):
            for ax in axs[0]:
                r = process(payload)
                ax.plot(
                    r["range_axis_m"],
                    db(r["range_profile"] / r["range_profile"].max()),
                    style,
                    lw=1,
                    label=label,
                )
            for window, clean, ax in (
                ("boxcar", False, axs[1, 0]),
                ("hann", False, axs[1, 1]),
                ("hann", True, axs[1, 1]),
            ):
                r = process(payload, doppler_window=window, clean=clean)
                p = abs(r["rd_cube"]) ** 2
                column = np.argmax(p.max(axis=0))
                ax.plot(
                    r["doppler_axis_hz"],
                    db(p[:, column] / p[:, column].max()),
                    style,
                    lw=1,
                    label=label + (" · mean removal" if clean else " · raw"),
                )
        axs[0, 0].set(
            xlim=(0, 38.4),
            ylim=(-85, 3),
            xlabel="Range (m)",
            ylabel="Relative power (dB)",
            title="Range periodicity",
        )
        axs[0, 1].set(
            xlim=(3.5, 6.5), ylim=(-65, 3), xlabel="Range (m)", title="Range mainlobe and sidelobes"
        )
        for ax, title in zip(
            axs[1], ("Rectangular slow time", "Hann: with/without mean removal"), strict=True
        ):
            ax.set(ylim=(-125, 3), xlabel="Doppler (Hz)", ylabel="Relative power (dB)", title=title)
        for ax in axs.flat:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.suptitle("Native RCS versus analytical point · peak-normalized shape comparison")
        fig.savefig(output / "native_rcs_reference.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ["cube_diffuse", "cube_specular", "cube_specular_aligned"]
    if not args.skip_native:
        cases = ["constant_radial", "constant_cube_pose", *cases]
    records = []
    first = {}
    for case in cases:
        for cpi in range(3):
            acquisition = acquire(case, cpi)
            record, result = analyze(acquisition)
            records.append(record)
            if cpi == 0:
                first[case] = result
            np.savez_compressed(
                args.output / f"{case}_{cpi}_paths.npz", a=acquisition["a"], tau=acquisition["tau"]
            )
            print(case, cpi, "paths", record["path_count"], flush=True)
    (args.output / "target_measurements.json").write_text(
        json.dumps(json_ready(records), indent=2, allow_nan=False) + "\n"
    )
    figures(first, args.output)


if __name__ == "__main__":
    main()
