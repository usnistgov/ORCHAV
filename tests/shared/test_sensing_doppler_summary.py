"""Summary figures consume the saved processing result without a second MTI."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from shared.sensing.plotting.doppler_summary import doppler_summary_event, plot_doppler_history
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload


def _payload(mode, moving=True, representation="complex_voltage"):
    n = 64
    t = np.arange(n) / 1000
    samples = np.ones((n, 3), dtype=complex)
    if moving:
        samples[:, 1] += np.exp(2j * np.pi * 336.2 * t)
    w = np.hanning(n)
    cleaned = samples.copy()
    if mode == "windowed_mean":
        cleaned -= np.sum(w[:, None] * samples, axis=0) / w.sum()
    elif mode == "legacy" and np.any(np.diff(samples, axis=0)):
        cleaned -= samples.mean(axis=0)
    rd = np.fft.fftshift(np.fft.fft(cleaned * w[:, None], axis=0, norm="ortho"), axes=0)
    key = (
        "range_doppler_by_pair"
        if representation == "complex_voltage"
        else "range_doppler_power_by_pair"
    )
    values = rd if representation == "complex_voltage" else abs(rd) ** 2
    payload = build_visualizer_sensing_payload(
        {
            key: values[None],
            "range_axis_m": np.arange(3.0),
            "doppler_axis_hz": np.fft.fftshift(np.fft.fftfreq(n, 0.001)),
            "config_resolved": {"coherent_clutter_mode": mode},
        }
    )
    return payload, rd


@pytest.mark.parametrize("mode", ["none", "legacy", "windowed_mean"])
@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("representation", ["complex_voltage", "power_linear"])
def test_summary_preserves_saved_clutter_operator(mode, moving, representation):
    payload, rd = _payload(mode, moving, representation)
    before = deepcopy(payload)
    event = doppler_summary_event(payload)
    np.testing.assert_array_equal(event["power_linear"], np.sum(abs(rd) ** 2, axis=1))
    assert event["time_s"] == payload["temporal"]["result"]["reference_time_s"]
    for key in payload:
        if isinstance(payload[key], np.ndarray):
            np.testing.assert_array_equal(payload[key], before[key])
    if not moving and mode in ("none", "legacy"):
        assert event["power_linear"][32] > 0
    if mode == "windowed_mean":
        assert event["power_linear"][32] < 1e-28


def test_summary_history_owns_reduction_without_retaining_rd():
    payload, rd = _payload("none")
    event = doppler_summary_event(payload)
    original = event["power_linear"].copy()
    payload["range_doppler_by_pair"].fill(0)
    np.testing.assert_array_equal(event["power_linear"], original)
    assert event["power_linear"].size == rd.shape[0]
    assert doppler_summary_event({"products": {}}) is None


@pytest.mark.parametrize("mode", ["none", "windowed_mean"])
@pytest.mark.parametrize("count", [1, 3])
def test_plot_saved_spectra_handles_one_cpi_and_zero_power(tmp_path, count, mode):
    payload, _ = _payload(mode, moving=False)
    first = doppler_summary_event(payload)
    history = [{**first, "time_s": i * 0.064} for i in range(count)]
    path = plot_doppler_history(history, tmp_path / "spectrum.png")
    assert path.stat().st_size > 1000


def test_default_generator_summary_never_runs_additional_signal_processing(tmp_path, monkeypatch):
    from generator.figures.summary import sensing as summary
    from shared.sensing import signal_processing as algorithms

    def forbidden(*args, **kwargs):
        raise AssertionError("A saved-product summary must not reprocess the signal")

    for name in (
        "compute_micro_doppler",
        "compute_range_collapsed_micro_doppler",
        "compute_range_tracked_micro_doppler",
        "compute_range_tracked_range_doppler",
        "compute_analytical_micro_doppler",
    ):
        monkeypatch.setattr(algorithms, name, forbidden)
    for name in (
        "create_range_profile_figure",
        "create_range_doppler_figure",
        "create_sensing_acquisition_processing_figure",
    ):
        monkeypatch.setattr(summary, name, lambda *a, output_path, **k: output_path)
    monkeypatch.setattr(summary, "build_sensing_diagram_model", lambda *a, **k: object())
    monkeypatch.setattr(
        summary, "write_sensing_diagram_manifest", lambda _, output_path: output_path
    )
    payload, _ = _payload("none", moving=False)
    paths = summary.maybe_generate_sensing_summary(
        payload,
        [],
        None,
        SimpleNamespace(generator_summary={"enabled": True, "create": ["sensing"]}),
        output_root=tmp_path,
        strict=True,
        fast_time_samples=[np.ones(3)] * 64,
        analytical_paths=[{}] * 64,
    )
    assert tmp_path / "sensing/doppler_history.png" in paths
    assert not any("micro_doppler" in p.name for p in paths)
    assert not any("acquisition_processing" in p.name for p in paths)


def test_nested_figure_selection_only_renders_requested_saved_product(tmp_path, monkeypatch):
    from generator.figures.summary import sensing as summary

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrequested figure was generated")

    for name in (
        "create_range_profile_figure",
        "create_range_doppler_figure",
        "create_sensing_acquisition_processing_figure",
    ):
        monkeypatch.setattr(summary, name, forbidden)
    payload, _ = _payload("none")
    paths = summary.maybe_generate_sensing_summary(
        payload,
        [],
        None,
        SimpleNamespace(
            generator_summary={
                "enabled": True,
                "create": ["sensing"],
                "sensing": {"figures": ["doppler"]},
            }
        ),
        output_root=tmp_path,
        strict=True,
    )
    assert [p.name for p in paths] == ["doppler_history.png"]


def test_detection_summary_keeps_missing_errors_as_gaps_and_restores_theme(tmp_path, monkeypatch):
    import matplotlib as mpl

    from shared.sensing.plotting import viz as sensing_viz

    original = dict(mpl.rcParams)
    captured = []
    close = sensing_viz.plt.close
    monkeypatch.setattr(sensing_viz.plt, "close", lambda fig: captured.append(fig))
    sensing_viz.create_detection_summary_figure(
        [
            dict(tp=1, fp=0, fn=0, range_rmse=0.1, velocity_rmse=0.2),
            dict(tp=0, fp=1, fn=1, range_rmse=None, velocity_rmse=None),
        ],
        tmp_path / "quality.png",
    )
    fig = captured[-1]
    for ax, expected in zip(fig.axes[2:], [0.1, 0.2]):
        values = ax.lines[0].get_ydata()
        assert values[0] == expected and np.isnan(values[1])
    assert fig.axes[2].get_ylabel() == "RMSE (m)"
    assert fig.axes[3].get_ylabel() == "RMSE (m/s)"
    assert dict(mpl.rcParams) == original
    close(fig)


def test_acquisition_diagram_requires_explicit_selection(tmp_path, monkeypatch):
    from generator.figures.summary import sensing as summary
    from shared.scenarios.model import SensingSummaryModel

    assert "acquisition" not in SensingSummaryModel().figures
    captured = []
    monkeypatch.setattr(summary, "build_sensing_diagram_model", lambda *a, **k: object())

    def capture(*a, output_path, **k):
        captured.append(output_path)
        return output_path

    monkeypatch.setattr(summary, "create_sensing_acquisition_processing_figure", capture)
    monkeypatch.setattr(summary, "write_sensing_diagram_manifest", lambda _, path: path)
    paths = summary.maybe_generate_sensing_summary(
        {},
        [],
        None,
        SimpleNamespace(
            generator_summary={
                "enabled": True,
                "create": ["sensing"],
                "sensing": {"figures": ["acquisition"]},
            }
        ),
        output_root=tmp_path,
        strict=True,
    )
    assert [p.name for p in paths] == ["acquisition_processing.png", "acquisition_processing.json"]
    assert len(captured) == 1


def _comparison_payload(start=0.0, *, midpoint=False, scale=1.0):
    from tests.visualizer.fixtures.sensing_v3 import materialize_visualizer_product_specs

    times = start + np.arange(4) * 0.001
    values = scale * (np.arange(12).reshape(1, 4, 3) + 1j)
    payload = build_visualizer_sensing_payload(
        {
            "range_doppler_by_pair": values,
            "range_axis_m": np.arange(3.0),
            "doppler_axis_hz": np.array([-500.0, -250.0, 0.0, 250.0]),
        }
    )
    acquisition = payload["temporal"]["acquisition"]
    acquisition.update(
        samples=[
            {
                "sample_index": i,
                "sample_time_s": float(time),
                "valid": True,
                "phase_coherent": True,
                "source": {},
            }
            for i, time in enumerate(times)
        ],
        expected_sample_count=4,
        observed_sample_count=4,
        valid_sample_count=4,
        completeness="complete",
        start_time_s=float(times[0]),
        end_time_s=float(times[-1]),
    )
    payload["temporal"]["result"].update(
        reference_time_s=float(np.mean(times)) if midpoint else start,
        reference_time_policy="cpi_center" if midpoint else "solve_start",
    )
    materialize_visualizer_product_specs(payload)
    for descriptor in payload["products"].values():
        if descriptor["kind"] == "range_doppler":
            descriptor["processing"] = {
                "range_processing": "delay_domain_cir_bins",
                "fft_normalization": "unitary_1_over_sqrt_n",
                "window_normalization": "unit_energy",
            }
    return payload


def _load_comparison(monkeypatch, payloads, label="Recording"):
    from pathlib import Path

    from shared.sensing import loader as sensing_loader
    from shared.sensing.plotting.doppler_summary import load_doppler_recording

    monkeypatch.setattr(
        sensing_loader,
        "iter_frames_sensing",
        lambda source: iter(
            {"_typed_sensing": p, "_frame_index": i} for i, p in enumerate(payloads)
        ),
    )
    return load_doppler_recording(Path(label), label)


def test_generic_comparison_only_reduces_saved_power_and_preserves_epochs(tmp_path, monkeypatch):
    from shared.sensing import signal_processing as sensing
    from shared.sensing.plotting import viz as sensing_viz
    from shared.sensing.plotting.doppler_summary import plot_doppler_comparison

    a = [_comparison_payload(0), _comparison_payload(0.004)]
    b = [_comparison_payload(i * 0.001, midpoint=True) for i in range(5)]
    first = _load_comparison(monkeypatch, a, "Frozen geometry")
    second = _load_comparison(monkeypatch, b, "Independent pulses")
    expected = np.sum(np.abs(a[0]["range_doppler_by_pair"][0]) ** 2, axis=1)
    np.testing.assert_array_equal(first["events"][0]["power_linear"], expected)
    assert first["events"][0]["time_s"] == 0
    assert second["events"][0]["time_s"] == 0.0015
    assert first["events"][0]["midpoint_s"] == second["events"][0]["midpoint_s"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison must not transform, window or remove clutter again")

    for name in ("fft", "ifft", "fft2"):
        monkeypatch.setattr(np.fft, name, forbidden)
    monkeypatch.setattr(sensing, "remove_window_weighted_mean", forbidden)
    captured = []
    close = sensing_viz.plt.close
    monkeypatch.setattr(sensing_viz.plt, "close", lambda fig: captured.append(fig))
    receipt = plot_doppler_comparison(
        [first, second],
        tmp_path / "comparison.png",
        axis="velocity",
        time_reference="midpoint",
        match_windows=True,
    )
    assert receipt["matching_window_indices"] == [0, 4]
    assert receipt["common_power_reference"] == max(expected)
    fig = captured[-1]
    assert len(fig.axes) == 4  # three maps and one common colorbar
    assert all("Closing Velocity" in ax.get_ylabel() for ax in fig.axes[:3])
    assert all(ax.collections[0].norm.vmax == 0 for ax in fig.axes[:3])
    close(fig)


def test_window_matching_checks_every_epoch_not_midpoint_or_endpoints():
    from shared.sensing.plotting.doppler_summary import matching_window_indices

    expected = {"sample_times_s": np.array([0, 0.001, 0.002, 0.003])}
    other = {"sample_times_s": np.array([0, 0.0009, 0.0021, 0.003])}
    with pytest.raises(ValueError, match="exact complete"):
        matching_window_indices([expected], [other])
    with pytest.raises(ValueError, match="exact complete"):
        matching_window_indices([expected], [expected, expected])
    rounded = {"sample_times_s": np.nextafter(expected["sample_times_s"], np.inf)}
    assert matching_window_indices([expected], [rounded]) == [0]
    shifted = {"sample_times_s": expected["sample_times_s"] + 1e-8}
    with pytest.raises(ValueError, match="exact complete"):
        matching_window_indices([expected], [shifted])


@pytest.mark.parametrize("mutation", ["nonuniform", "partial", "invalid"])
def test_comparison_rejects_invalid_declared_complete_windows(monkeypatch, mutation):
    payload = _comparison_payload()
    acquisition = payload["temporal"]["acquisition"]
    if mutation == "nonuniform":
        acquisition["samples"][1]["sample_time_s"] += 0.0001
    elif mutation == "partial":
        acquisition["observed_sample_count"] = 3
    else:
        acquisition["samples"][1]["valid"] = False
    with pytest.raises(ValueError, match="complete"):
        _load_comparison(monkeypatch, [payload])


def test_independent_solve_phase_flags_remain_provenance_not_a_summary_rejection(monkeypatch):
    payload = _comparison_payload()
    for sample in payload["temporal"]["acquisition"]["samples"]:
        sample["phase_coherent"] = False
    recording = _load_comparison(monkeypatch, [payload])
    assert recording["events"][0]["sample_phase_coherent"] == [False] * 4


def test_comparison_rejects_undeclared_power_normalization(monkeypatch):
    payload = _comparison_payload()
    for descriptor in payload["products"].values():
        descriptor.pop("processing", None)
    with pytest.raises(ValueError, match="declared signal normalization"):
        _load_comparison(monkeypatch, [payload])


def test_comparison_skips_legitimate_warmup_without_losing_window_samples(monkeypatch):
    warmup = {
        "products": {},
        "temporal": {"acquisition": {"completeness": "incomplete_unprocessed"}},
    }
    payload = _comparison_payload()
    recording = _load_comparison(monkeypatch, [warmup, payload])
    assert recording["skipped_incomplete"] == 1
    assert len(recording["events"]) == 1
    assert len(recording["events"][0]["sample_times_s"]) == 4


@pytest.mark.parametrize(
    "field",
    [
        "carrier",
        "range_mode",
        "range_grid",
        "doppler_grid",
        "normalization",
        "window_length",
        "cadence",
    ],
)
def test_comparison_rejects_incompatible_coordinates_and_calibration(monkeypatch, field):
    from shared.sensing.plotting.doppler_summary import validate_doppler_comparison

    reference = _load_comparison(monkeypatch, [_comparison_payload()])
    other = deepcopy(reference)
    if field == "carrier":
        other["calibration"]["carrier_frequency_hz"] *= 2
    elif field == "range_mode":
        other["calibration"]["range_mode"] = "bistatic"
    elif field == "normalization":
        other["calibration"]["processing"]["fft_normalization"] = "different"
    elif field == "range_grid":
        other["range_axis_m"] += 0.5
    elif field == "doppler_grid":
        other["doppler_axis_hz"] += 10
    elif field == "window_length":
        other["events"][0]["sample_times_s"] = np.arange(8) * 0.001
    else:
        other["events"][0]["sample_interval_s"] *= 2
    with pytest.raises(ValueError, match="incompatible"):
        validate_doppler_comparison([reference, other])


def test_velocity_axis_uses_saved_carrier_and_bistatic_label(tmp_path, monkeypatch):
    from shared.sensing.plotting import viz as sensing_viz
    from shared.sensing.plotting.doppler_summary import plot_doppler_comparison

    recording = _load_comparison(monkeypatch, [_comparison_payload()])
    recording["calibration"]["carrier_frequency_hz"] = 60e9
    recording["calibration"]["range_mode"] = "bistatic"
    captured = []
    close = sensing_viz.plt.close
    monkeypatch.setattr(sensing_viz.plt, "close", lambda fig: captured.append(fig))
    plot_doppler_comparison([recording], tmp_path / "velocity.png", axis="velocity")
    ax = captured[-1].axes[0]
    assert ax.get_ylabel() == "Monostatic-equivalent Velocity (m/s)"
    edges = ax.collections[0].get_coordinates()[:, 0, 1]
    centers = (edges[:-1] + edges[1:]) / 2
    np.testing.assert_allclose(centers, recording["doppler_axis_hz"] * 299792458 / 60e9 / 2)
    close(captured[-1])


@pytest.mark.parametrize("layout", ["combined", "separate"])
def test_comparison_cli_exports_default_scale_and_exact_provenance(tmp_path, monkeypatch, layout):
    import hashlib
    import json

    from shared.sensing.plotting import doppler_summary as summary

    recording = _load_comparison(monkeypatch, [_comparison_payload()])
    source = tmp_path / "recording"
    frames = source / "frames"
    frames.mkdir(parents=True)
    chunk = frames / "nested" / "chunk.h5"
    chunk.parent.mkdir()
    chunk.write_bytes(b"saved observation source")
    manifest = frames / "frames_manifest.json"
    manifest.write_text(json.dumps({"chunks": [{"file": "nested/chunk.h5"}]}))
    recording["source"] = source
    monkeypatch.setattr(summary, "load_doppler_recording", lambda path, label: recording)
    output = tmp_path / "summary"
    command = ["--source", "Example", str(source), "--output", str(output)]
    summary.main(command + (["--layout", layout] if layout == "separate" else []))
    receipt = json.loads((output / "receipt.json").read_text())
    image = receipt if layout == "combined" else receipt["outputs"][0]
    assert (output / image["image"]).is_file()
    if layout == "separate":
        image_receipt = json.loads((output / "source_1_doppler.json").read_text())
        assert image_receipt["sha256"] == image["sha256"]
        assert (
            image_receipt["comparison_receipt_sha256"]
            == hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest()
        )
    assert receipt["dynamic_range_db"] == 40
    assert receipt["time_reference"] == "result"
    saved = receipt["recordings"][0]
    window = saved["windows"][0]
    assert window["saved_result_time_s"] == 0
    assert window["saved_reference_policy"] == "solve_start"
    assert window["midpoint_s"] == 0.0015
    assert window["sample_times_s"] == [0, 0.001, 0.002, 0.003]
    assert (
        saved["source_hashes"]["nested/chunk.h5"] == hashlib.sha256(chunk.read_bytes()).hexdigest()
    )
    with pytest.raises(SystemExit):
        summary.main(["--source", "Example", str(source), "--output", str(output)])


def test_comparison_closes_stream_when_invalid_window_is_rejected(monkeypatch):
    from pathlib import Path

    from shared.sensing import loader as sensing_loader
    from shared.sensing.plotting.doppler_summary import load_doppler_recording

    payload = _comparison_payload()
    payload["temporal"]["acquisition"]["valid_sample_count"] = 0
    closed = []

    def frames(source):
        try:
            yield {"_typed_sensing": payload, "_frame_index": 0}
        finally:
            closed.append(True)

    monkeypatch.setattr(sensing_loader, "iter_frames_sensing", frames)
    with pytest.raises(ValueError, match="complete valid"):
        load_doppler_recording(Path("unused"), "Example")
    assert closed == [True]


def test_separate_comparison_reuses_standard_summary_with_common_axes_and_scale(
    tmp_path, monkeypatch
):
    from shared.sensing.plotting import doppler_summary as summary
    from shared.sensing.plotting import viz as sensing_viz

    first = _load_comparison(
        monkeypatch, [_comparison_payload(0), _comparison_payload(0.004)], "Coherent"
    )
    second = _load_comparison(
        monkeypatch,
        [_comparison_payload(i * 0.001, midpoint=True, scale=0.5) for i in range(5)],
        "Sequential",
    )
    calls = []
    rendered = []
    original = summary.plot_doppler_history
    close = sensing_viz.plt.close

    def standard(history, path, **kwargs):
        calls.append((history, kwargs))
        return original(history, path, **kwargs)

    monkeypatch.setattr(summary, "plot_doppler_history", standard)
    monkeypatch.setattr(sensing_viz.plt, "close", lambda fig: rendered.append(fig))
    receipt = summary.plot_doppler_comparison(
        [first, second],
        tmp_path,
        axis="velocity",
        time_reference="midpoint",
        match_windows=True,
        layout="separate",
    )
    assert len(calls) == 3
    assert [len(history) for history, _ in calls] == [2, 5, 2]
    assert calls[2][0][0] is second["events"][0]
    assert calls[2][0][1] is second["events"][4]
    assert receipt["matching_window_indices"] == [0, 4]
    assert [output["image"] for output in receipt["outputs"]] == [
        "source_1_doppler.png",
        "source_2_doppler.png",
        "matched_doppler.png",
    ]
    axes = [figure.axes[0] for figure in rendered]
    assert len(axes) == 3
    assert all(
        ax.get_xlim() == axes[0].get_xlim() and ax.get_ylim() == axes[0].get_ylim() for ax in axes
    )
    assert all(
        ax.collections[0].norm.vmin == -40 and ax.collections[0].norm.vmax == 0 for ax in axes
    )
    assert all("midpoint" in ax.get_xlabel() for ax in axes)
    assert all(
        options["power_reference"] == receipt["common_power_reference"] for _, options in calls
    )
    np.testing.assert_allclose(np.max(axes[1].collections[0].get_array()), 10 * np.log10(0.25))
    for figure in rendered:
        close(figure)


def _source_bound_replay_payload(start, observation):
    from shared.extensions.sensing import canonical_processing_recipe_id

    payload = _comparison_payload(start)
    response = {
        "frequency_window": "hann",
        "range_processing": "finite_bandwidth_delay_response",
        "fft_normalization": "unitary_1_over_sqrt_n",
        "window_normalization": "unit_energy",
        "source_product_id": f"source-{observation}",
        "contributing_element_path_count": observation + 1,
        "noise": {
            "variance": 0.0,
            "seed": 42,
            "realization_index": observation,
            "observation_sha256": str(observation),
        },
    }
    config = payload["config_resolved"]
    config["range_response_processing"] = deepcopy(response)
    config["processing_recipe"] = {
        "tool": "shared.sensing_postprocessor",
        "input_kind": "cfr_by_pair_rx_element",
        "effective_overrides": {"frequency_window": "hann", "clutter_mode": "none"},
        "inherited_numeric_model": {"range_response_processing": deepcopy(response)},
    }
    recipe_id = canonical_processing_recipe_id(config["processing_recipe"])
    payload["temporal"]["result"]["processing_recipe_id"] = recipe_id
    for descriptor in payload["products"].values():
        if descriptor["kind"] == "range_doppler":
            descriptor["lineage"]["processing_recipe_id"] = recipe_id
            descriptor["processing"] = {
                "tool": "shared.sensing_postprocessor",
                "input_kind": "cfr_by_pair_rx_element",
                "recipe_id": recipe_id,
            }
    return payload


def test_source_bound_replay_ids_keep_equal_settings_in_standard_history(tmp_path):
    from shared.sensing.plotting.doppler_summary import doppler_processing_matches

    payloads = [_source_bound_replay_payload(0, 0), _source_bound_replay_payload(0.004, 64)]
    before = deepcopy(payloads)
    events = [doppler_summary_event(payload) for payload in payloads]
    assert events[0]["processing_recipe_id"] != events[1]["processing_recipe_id"]
    assert doppler_processing_matches(*events)
    plot_doppler_history(events, tmp_path / "replay.png")
    for payload, original in zip(payloads, before):
        assert payload["config_resolved"] == original["config_resolved"]
        assert payload["temporal"] == original["temporal"]
        assert payload["products"] == original["products"]


@pytest.mark.parametrize(
    "changed", ["frequency_window", "clutter_mode", "noise_variance", "unknown_setting"]
)
def test_replay_display_identity_does_not_ignore_changed_settings(tmp_path, changed):
    from shared.extensions.sensing import canonical_processing_recipe_id

    first = _source_bound_replay_payload(0, 0)
    second = _source_bound_replay_payload(0.004, 64)
    recipe = second["config_resolved"]["processing_recipe"]
    if changed == "noise_variance":
        recipe["inherited_numeric_model"]["range_response_processing"]["noise"]["variance"] = 0.1
    else:
        recipe["effective_overrides"][changed] = "different"
    recipe_id = canonical_processing_recipe_id(recipe)
    second["temporal"]["result"]["processing_recipe_id"] = recipe_id
    for descriptor in second["products"].values():
        if descriptor["kind"] == "range_doppler":
            descriptor["lineage"]["processing_recipe_id"] = recipe_id
            descriptor["processing"]["recipe_id"] = recipe_id
    events = [doppler_summary_event(payload) for payload in (first, second)]
    with pytest.raises(ValueError, match="recipe"):
        plot_doppler_history(events, tmp_path / "invalid.png")


def test_source_bound_replay_requires_hashed_recipe_to_match_descriptor():
    payload = _source_bound_replay_payload(0, 0)
    payload["config_resolved"]["processing_recipe"]["effective_overrides"][
        "frequency_window"
    ] = "boxcar"
    with pytest.raises(ValueError, match="do not agree"):
        doppler_summary_event(payload)
