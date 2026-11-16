"""Optional independent Sionna PHY 2.2 comparison; no production dependency.

Run from the checkout with a full Sionna 2.2 environment::

    python -m scripts.analysis.validate_sionna22_finite_bandwidth --output <directory>

Add ``--live`` for a CUDA RT 2.2 acquisition of a constant-RCS control and a
specular mesh. Select a free GPU through CUDA_VISIBLE_DEVICES. An isolated
``pip install --no-deps --target <directory> sionna==2.2.0`` installation can be
selected with PYTHONPATH without altering an existing ORCHAV environment.
Only the requested output directory is written; existing results are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from scripts.analysis.validate_range_doppler import observation_payload
from shared.sensing.cfr_replay import rebuild_frequency_observation
from shared.sensing.products import build_product_descriptor, dense_axis
from shared.sensing.range_response import (
    C0,
    RangeResponseConfig,
    frequency_grid,
    paths_to_frequency,
)

CARRIER = 28e9
BANDWIDTH = 500e6
FREQUENCIES = 128
SAMPLES = 64
SAMPLE_RATE = 1000.0
TIMES = np.arange(SAMPLES) / SAMPLE_RATE
WINDOWS = (("boxcar", "boxcar"), ("hann", "boxcar"), ("boxcar", "hann"), ("hann", "hann"))
DOUBLE_RELATIVE_TOLERANCE = 5e-12


@dataclass
class PathCase:
    """One common baseband CIR, with carrier and Doppler already applied."""

    name: str
    coefficients: np.ndarray  # [receive element, path, slow time]
    delays: np.ndarray  # shared delays [path]
    validity: np.ndarray
    dopplers: np.ndarray


def point_case(name, ranges, dopplers, gains=None, elements=1):
    delays = 2 * np.asarray(ranges) / C0
    gains = np.ones(len(delays), complex) if gains is None else np.asarray(gains, complex)
    # Exactly once: carrier phase is part of a; only offsets enter either CFR API.
    a = gains[:, None] * np.exp(-2j * np.pi * CARRIER * delays[:, None])
    a = a * np.exp(2j * np.pi * np.asarray(dopplers)[:, None] * TIMES)
    a = np.exp(1j * np.arange(elements) * 0.7)[:, None, None] * a[None]
    return PathCase(name, a, delays, np.ones(len(delays), bool), np.asarray(dopplers))


def deterministic_cases():
    resolution = C0 / (2 * BANDWIDTH)
    period = FREQUENCIES * resolution
    moving = 2 * CARRIER * 1.8 / C0
    return [
        point_case("stationary_on_grid", [16 * resolution], [0]),
        point_case("stationary_fractional", [5], [0]),
        point_case("moving_off_grid", [5], [moving]),
        point_case("two_distinct_paths", [4, 8], [moving, -moving / 2], [1, 0.65j]),
        point_case("two_close_coherent", [5, 5.12], [moving, moving], [1, 0.65j]),
        point_case("four_receive_elements", [5], [moving], elements=4),
        point_case("ambiguity_boundary", [period - 0.09], [-112.3]),
        point_case("two_equal_coherent", [5, 5], [moving, moving], [1, 0.65j]),
        point_case("moving_on_grid", [16 * resolution], [12 * SAMPLE_RATE / SAMPLES]),
    ]


def reference_api():
    """Import only on demand, with an actionable version error."""
    import sionna
    import torch
    from sionna.phy import channel

    required = ("ofdm_to_time_channel", "time_to_doppler_channel", "ofdm_to_delay_doppler_channel")
    if sionna.__version__ != "2.2.0" or any(not hasattr(channel, f) for f in required):
        raise RuntimeError(
            f"Full Sionna PHY 2.2.0 is required; imported {sionna.__version__} "
            f"from {sionna.__file__}. Installing sionna-rt==2.2.0 alone is insufficient."
        )
    torch.set_num_threads(1)
    return torch, channel


def compare(actual, expected):
    """Full-array error; phase is measured only above 1e-10 of the peak."""
    error = np.abs(actual - expected)
    peak = max(float(np.max(abs(expected))), np.finfo(float).tiny)
    mask = abs(expected) > peak * 1e-10
    return {
        "max_absolute_complex_error": float(error.max()),
        "normalized_max_complex_error": float(error.max() / peak),
        "relative_l2_complex_error": float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        "max_magnitude_error": float(np.max(abs(abs(actual) - abs(expected)))),
        "max_phase_error_above_relative_1e_10_rad": float(
            np.max(abs(np.angle(actual[mask] * expected[mask].conj())))
        ),
    }


def require_close(actual, expected, *, relative=DOUBLE_RELATIVE_TOLERANCE):
    metrics = compare(actual, expected)
    assert metrics["normalized_max_complex_error"] < relative, metrics
    return metrics


def project_case(case, api):
    """Identical validity and common double-precision inputs for both projectors."""
    torch, channel = api
    config = RangeResponseConfig("cfr_ifft", FREQUENCIES)
    orchav, _ = paths_to_frequency(
        case.coefficients,
        case.delays,
        case.validity,
        TIMES,
        bandwidth=BANDWIDTH,
        config=config,
    )
    # Both APIs see the same valid paths; invalid padding is not a physical path.
    a = torch.tensor(case.coefficients[:, case.validity], dtype=torch.complex128)
    a = a[None, None, :, None, None]
    tau = torch.tensor(case.delays[case.validity], dtype=torch.float64)[None, None, None]
    offsets = torch.tensor(frequency_grid(BANDWIDTH, FREQUENCIES), dtype=torch.float64)
    reference = channel.cir_to_ofdm_channel(offsets, a, tau, normalize=False)
    reference = reference.numpy()[0, 0, :, 0, 0]
    return orchav, reference, require_close(orchav, reference)


def replay(observation, range_window, doppler_window, fft_size):
    """Use the real saved-product replay entry point, including receive elements."""
    sensing = observation_payload(
        observation[0], frequency_grid(BANDWIDTH, FREQUENCIES), TIMES, clutter="none"
    )
    sensing["observation"] = observation[None]
    sensing["element_coordinate"] = np.arange(observation.shape[0])
    old = sensing["products"]["observation"]
    axes = old["axes"]
    axes[1] = dense_axis(
        "rx_element",
        size=observation.shape[0],
        unit="1",
        coordinate_key="element_coordinate",
        coordinate_semantics="element_index",
    )
    sensing["products"]["observation"] = build_product_descriptor(
        sensing,
        data_key="observation",
        kind="baseband_frequency_observation",
        axes=axes,
        representation="complex_voltage",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=old["pair_ids"],
        processing=old["processing"],
    )
    return rebuild_frequency_observation(
        sensing,
        response=RangeResponseConfig("cfr_ifft", FREQUENCIES, range_window),
        fft_size=fft_size,
        window_doppler=doppler_window,
        coherent_clutter_mode="none",
    )


def matched_windows(range_window, doppler_window):
    """Independent explicit definitions: periodic frequency, symmetric slow time."""
    wr = (
        np.ones(FREQUENCIES)
        if range_window == "boxcar"
        else np.sin(np.pi * np.arange(FREQUENCIES) / FREQUENCIES) ** 2
    )
    wd = (
        np.ones(SAMPLES)
        if doppler_window == "boxcar"
        else np.sin(np.pi * np.arange(SAMPLES) / (SAMPLES - 1)) ** 2
    )
    return wr, wd


def reference_transform(observation, range_window, doppler_window, fft_size, api):
    """Sionna native transform of an explicitly zero-extended centered grid.

    The 384 added zeros are transform padding, never measured subcarriers.
    Delta-f stays B/128. Sionna divides by M*S; ORCHAV by sum(wr)*sqrt(sum(wd**2)).
    """
    torch, channel = api
    wr, wd = matched_windows(range_window, doppler_window)
    centered = np.zeros((*observation.shape[:-1], fft_size), complex)
    start = fft_size // 2 - FREQUENCIES // 2
    centered[..., start : start + FREQUENCIES] = observation * wr * wd[None, :, None]
    native = channel.ofdm_to_delay_doppler_channel(torch.tensor(centered)).numpy()
    scale = fft_size * SAMPLES / (wr.sum() * np.sqrt(np.sum(wd**2)))
    return native * scale, native, scale, centered


def direct_transform(observation, range_window, doppler_window, fft_size):
    """Independent dense Fourier sums; neither FFT library is used."""
    wr, wd = matched_windows(range_window, doppler_window)
    q = np.arange(FREQUENCIES) - FREQUENCIES // 2
    delay_kernel = np.exp(2j * np.pi * q[:, None] * np.arange(fft_size) / fft_size)
    delay = (observation * wr) @ delay_kernel / wr.sum()
    doppler_indices = np.arange(-SAMPLES // 2, SAMPLES // 2)
    kernel = np.exp(-2j * np.pi * doppler_indices[:, None] * np.arange(SAMPLES) / SAMPLES)
    result = (kernel * wd) @ delay / np.sqrt(np.sum(wd**2))
    return delay, result


def qualify_transform(observation, range_window, doppler_window, fft_size, api):
    result = replay(observation, range_window, doppler_window, fft_size)
    actual = result["element_rd"]
    reference, native, scale, centered = reference_transform(
        observation, range_window, doppler_window, fft_size, api
    )
    direct_delay, direct = direct_transform(observation, range_window, doppler_window, fft_size)
    torch, channel = api
    # Check the two named component utilities as well as their composition.
    time_channel = channel.ofdm_to_time_channel(torch.tensor(centered))
    composed = channel.time_to_doppler_channel(time_channel).numpy()
    require_close(native, composed)
    require_close(result["delay"], direct_delay)
    metrics = require_close(actual, reference)
    direct_metrics = require_close(actual, direct)
    peaks = [list(np.unravel_index(np.argmax(abs(a)), a.shape)) for a in actual]
    reference_peaks = [list(np.unravel_index(np.argmax(abs(a)), a.shape)) for a in reference]
    assert peaks == reference_peaks
    expected_range = np.arange(fft_size) * C0 * FREQUENCIES / (2 * fft_size * BANDWIDTH)
    np.testing.assert_allclose(result["range_axis_m"], expected_range, atol=1e-14, rtol=1e-15)
    _, frequency_axis = channel.time_frequency_vector(
        SAMPLES, 1 / SAMPLE_RATE, precision="double", device="cpu"
    )
    np.testing.assert_allclose(
        result["doppler_axis_hz"], frequency_axis.numpy(), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        channel.subcarrier_frequencies(
            FREQUENCIES, BANDWIDTH / FREQUENCIES, precision="double", device="cpu"
        ).numpy(),
        frequency_grid(BANDWIDTH, FREQUENCIES),
        rtol=0,
        atol=0,
    )
    raw_power = float(np.sum(abs(native) ** 2))
    parseval = float(np.sum(abs(centered) ** 2) / (fft_size * SAMPLES))
    assert abs(raw_power / parseval - 1) < 3e-14
    preservation = None
    if fft_size == 512:
        unpadded = replay(observation, range_window, doppler_window, FREQUENCIES)
        preservation = require_close(actual[..., ::4], unpadded["element_rd"])
    return {
        "range_window": range_window,
        "doppler_window": doppler_window,
        "range_ifft": fft_size,
        "sionna_to_orchav_complex_scale": float(scale),
        "complex_comparison": metrics,
        "direct_fourier_comparison": direct_metrics,
        "original_samples_preserved": preservation,
        "peaks_element_doppler_range": peaks,
        "peak_ranges_m": [float(result["range_axis_m"][r]) for _, r in peaks],
        "peak_velocities_m_s": [
            float(result["doppler_axis_hz"][d] * C0 / (2 * CARRIER)) for d, _ in peaks
        ],
        "orchav_total_power": float(np.sum(abs(actual) ** 2)),
        "sionna_total_power": raw_power,
        "sionna_parseval_expected_power": parseval,
    }


def sinc_comparison(api):
    """Characterize a different model without asserting false equivalence."""
    torch, channel = api
    aligned_delay = 16 / BANDWIDTH
    aligned_h = np.exp(-2j * np.pi * frequency_grid(BANDWIDTH, FREQUENCIES) * aligned_delay)
    aligned_taps = (
        channel.cir_to_time_channel(
            BANDWIDTH,
            torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128),
            torch.tensor([[[[aligned_delay]]]], dtype=torch.float64),
            0,
            FREQUENCIES - 1,
            normalize=False,
        )
        .numpy()
        .reshape(-1)
    )
    aligned_periodic = channel.ofdm_to_time_channel(torch.tensor(aligned_h[None])).numpy()[0]
    aligned_comparison = require_close(aligned_periodic, aligned_taps)
    rows = []
    for n in (128, 512, 2048):
        fractional_bin = 16.35
        lags = np.arange(n)
        tau = fractional_bin / BANDWIDTH
        h = np.exp(-2j * np.pi * frequency_grid(BANDWIDTH, n) * tau)[None, :]
        periodic = channel.ofdm_to_time_channel(torch.tensor(h)).numpy()[0]
        taps = (
            channel.cir_to_time_channel(
                BANDWIDTH,
                torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128),
                torch.tensor([[[[tau]]]], dtype=torch.float64),
                0,
                n - 1,
                normalize=False,
            )
            .numpy()
            .reshape(-1)
        )
        x = lags - fractional_bin
        sinc_expected = np.sinc(x)
        dirichlet_expected = np.exp(-1j * np.pi * x / n) * np.sinc(x) / np.sinc(x / n)
        require_close(taps, sinc_expected)
        require_close(periodic, dirichlet_expected)
        main = abs(x) <= 1
        rows.append(
            {
                "frequency_observations": n,
                "mainlobe_max_complex_difference": float(np.max(abs(periodic[main] - taps[main]))),
                "mainlobe_max_magnitude_difference": float(
                    np.max(abs(abs(periodic[main]) - abs(taps[main])))
                ),
                "full_interval_max_difference": float(np.max(abs(periodic - taps))),
                "finite_grid_theory": compare(periodic, dirichlet_expected),
                "sinc_theory": compare(taps, sinc_expected),
            }
        )
    # Boundary behavior is particularly different: the IDFT is periodic.
    tau = (FREQUENCIES - 0.3) / BANDWIDTH
    h = np.exp(-2j * np.pi * frequency_grid(BANDWIDTH, FREQUENCIES) * tau)[None, :]
    edge = channel.ofdm_to_time_channel(torch.tensor(h)).numpy()[0]
    return {
        "on_grid_comparison": aligned_comparison,
        "convergence_fixed_bandwidth": rows,
        "boundary_delay_bins": FREQUENCIES - 0.3,
        "periodic_magnitude_at_zero": float(abs(edge[0])),
        "sinc_magnitude_at_zero": float(abs(np.sinc(-BANDWIDTH * tau))),
    }


def acquire_live(kind):
    """One solve per model; all comparisons use the very same returned Paths."""
    import mitsuba as mi

    if mi.variant() != "cuda_ad_mono_polarized":
        mi.set_variant("cuda_ad_mono_polarized")
    import drjit as dr
    import sionna.rt as rt

    dr.set_thread_count(2)
    scene = rt.load_scene()
    scene.frequency = CARRIER
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", position=[0, 0, 0]))
    scene.add(rt.Receiver("rx", position=[0, 0, 0]))
    start = time.perf_counter()
    if kind == "constant_rcs":
        from sionna.rt.rcs import ConstantCPM, RCSSolver, ScatteringModel, SensingTarget

        def rcs(k_i, k_s, seed):
            return dr.ones(mi.Float, dr.width(k_i))

        target = SensingTarget(
            name="control",
            scattering_model=ScatteringModel([0, 0, 0], rcs=rcs, cpm=ConstantCPM()),
            length=1,
            width=1,
            height=1,
            position=[0, 5, 0],
            velocity=[0, -1.8, 0],
        )
        scene.add(target)
        paths = RCSSolver()(
            scene,
            max_depth=1,
            samples_per_sp=32,
            buffer_size_per_sp=32,
            los=True,
            specular_reflection=False,
            refraction=False,
            seed=42,
        )
        expected_range = 5.0
    else:
        mesh = Path(__file__).resolve().parents[2] / "libraries/targets/cube/cube.ply"
        target = rt.SceneObject(
            name="mesh",
            fname=str(mesh),
            radio_material=rt.ITURadioMaterial("metal", "metal", thickness=0.01),
        )
        scene.edit(add=target)
        target.position = [0, 5, 0]
        target.velocity = [0, -1.8, 0]
        paths = rt.PathSolver()(
            scene,
            max_depth=1,
            samples_per_src=10000,
            los=False,
            specular_reflection=True,
            diffuse_reflection=False,
            seed=42,
        )
        expected_range = 4.5
    a, tau = paths.cir(
        sampling_frequency=SAMPLE_RATE,
        num_time_steps=SAMPLES,
        normalize_delays=False,
        out_type="numpy",
    )
    coefficients = a[0, :, 0, 0].astype(np.complex128)
    delays = tau[0, 0].astype(float)
    valid = delays >= 0
    assert np.any(valid), f"No valid {kind} paths"
    case = PathCase(kind, coefficients, delays, valid, np.asarray(paths.doppler)[0, 0])
    cfr = paths.cfr(
        mi.Float(frequency_grid(BANDWIDTH, FREQUENCIES)),
        sampling_frequency=SAMPLE_RATE,
        num_time_steps=SAMPLES,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )[0, :, 0, 0]
    taps = paths.taps(
        BANDWIDTH,
        0,
        FREQUENCIES - 1,
        sampling_frequency=SAMPLE_RATE,
        num_time_steps=SAMPLES,
        normalize=False,
        normalize_delays=False,
        out_type="numpy",
    )[0, :, 0, 0]
    np.testing.assert_allclose(delays[valid] * C0 / 2, expected_range, atol=2e-5, rtol=0)
    return (
        case,
        cfr,
        taps,
        {
            "variant": mi.variant(),
            "solve_and_read_seconds": time.perf_counter() - start,
            "valid_paths": int(valid.sum()),
            "expected_surface_or_point_range_m": expected_range,
        },
    )


def qualify_live(kind, api, output):
    case, native_cfr, native_taps, metadata = acquire_live(kind)
    observation, _, cfr_metrics = project_case(case, api)
    # Native Dr.Jit computations are float32; double precision cannot recover those bits.
    backend_metrics = require_close(observation, native_cfr, relative=3e-5)
    torch, channel = api
    a = torch.tensor(case.coefficients[:, case.validity])[None, None, :, None, None]
    tau = torch.tensor(case.delays[case.validity])[None, None, None]
    taps = channel.cir_to_time_channel(
        BANDWIDTH, a, tau, 0, FREQUENCIES - 1, normalize=False
    ).numpy()[0, 0, :, 0, 0]
    taps_metrics = require_close(native_taps, taps, relative=3e-5)
    rows = [
        qualify_transform(observation, wr, wd, size, api)
        for size in (128, 512)
        for wr, wd in WINDOWS
    ]
    np.savez_compressed(
        output / f"{kind}_common_inputs.npz",
        coefficients=case.coefficients,
        delays_s=case.delays,
        validity=case.validity,
        sample_times_s=TIMES,
        native_cfr=native_cfr,
        native_taps=native_taps,
    )
    return {
        "case": kind,
        **metadata,
        "double_cfr_comparison": cfr_metrics,
        "native_float32_cfr_comparison": backend_metrics,
        "native_float32_sinc_taps_comparison": taps_metrics,
        "transforms": rows,
    }


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_figure(output, api):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = deterministic_cases()[2]
    h, _, _ = project_case(case, api)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (wr, wd) in zip(axes.flat, WINDOWS):
        actual = replay(h, wr, wd, 512)
        reference, _, _, _ = reference_transform(h, wr, wd, 512, api)
        i = int(np.argmax(np.max(abs(actual["element_rd"][0]), axis=-1)))
        p = 10 * np.log10(np.maximum(abs(actual["element_rd"][0, i]) ** 2, 1e-30))
        p_ref = 10 * np.log10(np.maximum(abs(reference[0, i]) ** 2, 1e-30))
        ax.plot(actual["range_axis_m"], p, lw=1, label="ORCHAV")
        ax.plot(actual["range_axis_m"][::8], p_ref[::8], ".", ms=3, label="Sionna × derived scale")
        ax.set(
            xlim=(0, 15),
            ylim=(-90, 20),
            xlabel="Monostatic range (m)",
            ylabel="Power (dB re 1)",
            title=f"Frequency: {wr}; slow time: {wd}",
        )
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Identical off-grid moving-point input; slice at strongest Doppler bin")
    fig.savefig(output / "matched_spectra.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live",
        action="store_true",
        help="CUDA RT 2.2 controls, requires an exposed NVIDIA driver",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise FileExistsError(
            "Choose a fresh output directory; existing scientific results are preserved"
        )
    # Set the RT variant before any Sionna import, when RT validation is requested.
    if args.live:
        if version("sionna-rt") != "2.2.0":
            raise RuntimeError("The live control requires sionna-rt==2.2.0")
        import mitsuba as mi

        mi.set_variant("cuda_ad_mono_polarized")
        print(f"Mitsuba variant: {mi.variant()}", flush=True)
    api = reference_api()
    start = time.perf_counter()
    results = {
        "cases": [],
        "live": [],
        "commands": {"argv": sys.argv, "python": sys.executable},
        "versions": {"python": platform.python_version()},
        "configuration": {
            "carrier_hz": CARRIER,
            "bandwidth_hz": BANDWIDTH,
            "frequency_observations": FREQUENCIES,
            "slow_time_samples": SAMPLES,
            "slow_time_sampling_hz": SAMPLE_RATE,
            "coherent_clutter_mode": "none",
            "phy_transform_device": "cpu",
            "frequency_grid": "q*B/N, q=-N/2,...,N/2-1; upper endpoint excluded",
            "range_window": "periodic Hann sin(pi*k/N)^2 or rectangular",
            "doppler_window": "symmetric Hann sin(pi*n/(S-1))^2 or rectangular",
            "paths": "Frozen delays within one CPI; coefficients contain carrier phase and Doppler",
        },
    }
    for package in ("sionna", "sionna-rt", "torch", "numpy", "mitsuba", "drjit"):
        try:
            results["versions"][package] = version(package)
        except PackageNotFoundError:
            results["versions"][package] = None
    source_files = [
        Path(__file__),
        Path(inspect.getfile(api[1].ofdm_to_delay_doppler_channel)),
        Path(inspect.getfile(paths_to_frequency)),
        Path(inspect.getfile(rebuild_frequency_observation)),
    ]
    results["source_sha256"] = {str(p.resolve()): file_hash(p) for p in source_files}
    inputs = {}
    for case in deterministic_cases():
        h, _, cfr_metrics = project_case(case, api)
        rows = [
            qualify_transform(h, wr, wd, size, api) for size in (128, 512) for wr, wd in WINDOWS
        ]
        results["cases"].append(
            {"case": case.name, "cfr_comparison": cfr_metrics, "transforms": rows}
        )
        inputs.update(
            {f"{case.name}_coefficients": case.coefficients, f"{case.name}_delays_s": case.delays}
        )
        print(f"PASS {case.name}: CFR and 8 complex delay–Doppler comparisons", flush=True)
    np.savez_compressed(
        args.output / "deterministic_common_inputs.npz",
        **inputs,
        sample_times_s=TIMES,
        frequency_offsets_hz=frequency_grid(BANDWIDTH, FREQUENCIES),
    )
    results["sinc_comparison"] = sinc_comparison(api)
    if args.live:
        for kind in ("constant_rcs", "specular_mesh"):
            results["live"].append(qualify_live(kind, api, args.output))
            print(f"PASS CUDA {kind}", flush=True)
    results["runtime_seconds"] = time.perf_counter() - start
    make_figure(args.output, api)
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2, default=lambda x: x.item()) + "\n"
    )
    print(f"Saved quantitative results to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
