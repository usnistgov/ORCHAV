"""Physical-time, power and saved-evidence contracts for motion comparison."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from shared.sensing.motion_analysis import (
    SCHEMA,
    MotionRecipe,
    analyze_comparison,
    compute_motion_spectra,
    export_comparison,
    load_definition,
    load_export,
    load_motion_recording,
)


def fixture_comparison():
    times = np.arange(64) / 32
    values = (2 + np.exp(2j * np.pi * 4 * times))[:, None]
    recipe = MotionRecipe(2, 4, window_samples=16, hop_samples=8, window="boxcar")
    result = compute_motion_spectra(values, times, [3.0], recipe)
    result.update(label="Independent tone", provenance={"fixture": True})
    return {
        "schema": SCHEMA,
        "definition": {
            "schema": SCHEMA,
            "sources": [{"label": "Independent tone", "frames": "fixture"}],
            "recipe": result["metadata"]["recipe"],
        },
        "results": [result],
    }


def test_direct_transform_and_parseval():
    result = fixture_comparison()["results"][0]
    frequencies = result["doppler_hz"]
    # Boxcar DFT of DC amplitude 2 and unit amplitude on-grid tone; df=2Hz.
    np.testing.assert_allclose(result["retained_psd"][frequencies == 0], 2)
    np.testing.assert_allclose(result["retained_psd"][frequencies == 4], 0.5)
    np.testing.assert_allclose(result["removed_psd"].sum(axis=0) * 2, 1)
    assert result["metadata"]["mean_removal_loss_db"] == pytest.approx(10 * np.log10(5))
    np.testing.assert_array_equal(result["time_s"], np.arange(7) / 4 + 15 / 64)


def test_hann_no_delay_taper_matches_independent_complex_sum():
    rng = np.random.default_rng(117)
    values = rng.normal(size=(40, 4)) + 1j * rng.normal(size=(40, 4))
    times, ranges = 1.5 + np.arange(40) / 128, [1.0, 2.0, 3.0, 4.0]
    recipe = MotionRecipe(2, 3, window_samples=15, hop_samples=5)
    result = compute_motion_spectra(values, times, ranges, recipe)
    frequency = np.fft.fftshift(np.fft.fftfreq(15, 1 / 128))
    matrix = np.exp(-2j * np.pi * frequency[:, None] * np.arange(15)[None, :] / 128)
    for column in range(6):
        value = values[column * 5 : column * 5 + 15, 1:3]
        for mode in ("retained", "removed"):
            x = value - value.mean(axis=0) if mode == "removed" else value
            expected = abs(matrix @ (x * np.hanning(15)[:, None])) ** 2
            np.testing.assert_allclose(
                result[mode + "_psd"][:, column],
                expected.sum(axis=1) / (128 * np.sum(np.hanning(15) ** 2)),
                rtol=1e-13,
                atol=1e-14,
            )


@pytest.mark.parametrize(
    "change,message",
    [
        ("gap", "uniform"),
        ("nan", "finite"),
        ("real", "Complex"),
        ("range", "one range"),
        ("short", "complete"),
        ("reverse", "increase"),
    ],
)
def test_invalid_or_missing_observations_are_not_zero_channels(change, message):
    x, t, ranges = np.ones((32, 2), dtype=complex), np.arange(32) / 20, np.array([1.0, 2.0])
    recipe = MotionRecipe(0, 5, window_samples=16, hop_samples=8)
    if change == "gap":
        t[17:] += 0.01
    if change == "nan":
        x[0, 0] = np.nan
    if change == "real":
        x = x.real
    if change == "range":
        ranges += 100
    if change == "short":
        recipe = MotionRecipe(0, 5, time_end_s=0.1, window_samples=16, hop_samples=8)
    if change == "reverse":
        t = t[::-1]
    with pytest.raises(ValueError, match=message):
        compute_motion_spectra(x, t, ranges, recipe)


def test_valid_empty_channel_and_zero_residual_remain_valid():
    recipe = MotionRecipe(0, 5, window_samples=16, hop_samples=8)
    for amplitude in (0, 3):
        result = compute_motion_spectra(
            np.full((32, 1), amplitude + 0j), np.arange(32), [2.0], recipe
        )
        assert not result["removed_psd"].any()
        assert result["metadata"]["mean_removal_loss_db"] is None


def test_roundtrip_hash_and_existing_output_protection(tmp_path):
    comparison = fixture_comparison()
    output = export_comparison(comparison, tmp_path / "new", plots=False)
    loaded = load_export(output)
    np.testing.assert_array_equal(
        loaded["results"][0]["retained_psd"], comparison["results"][0]["retained_psd"]
    )
    with pytest.raises(FileExistsError):
        export_comparison(comparison, output, plots=False)
    with (output / "spectra.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        load_export(output)


def test_definition_preserves_names_and_resolves_small_shared_references(tmp_path):
    source = {"label": "not Paris", "frames": "../native/frames", "root_motion": "root.json"}
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"schema": SCHEMA, "sources": [source], "recipe": {}}))
    definition = load_definition(path)
    assert definition["sources"][0]["label"] == "not Paris"
    assert definition["sources"][0]["frames"] == str((tmp_path / "../native/frames").resolve())
    assert definition["sources"][0]["root_motion"] == str(tmp_path / "root.json")


def test_alternate_count_names_and_incompatible_carrier(tmp_path, monkeypatch):
    def load(path, _progress=None):
        return (
            np.ones((32, 1), complex),
            np.arange(32) / 50,
            np.array([3.0]),
            dict(
                carrier_hz=28e9 if path.name != "wrong" else 3.5e9,
                range_mode="monostatic",
                scale_semantics="relative_unitless",
                combine={"policy": "siso_identity"},
            ),
        )

    monkeypatch.setattr("shared.sensing.motion_analysis.load_motion_recording", load)
    definition = {
        "sources": [{"label": name, "frames": name} for name in ("one", "second", "third")],
        "recipe": {"window_samples": 16, "hop_samples": 8},
    }
    result = analyze_comparison(definition)
    assert [r["label"] for r in result["results"]] == ["one", "second", "third"]
    definition["sources"][1]["frames"] = "wrong"
    with pytest.raises(ValueError, match="matching carrier"):
        analyze_comparison(definition)


def test_native_loader_keeps_valid_early_samples_without_a_processed_result(tmp_path, monkeypatch):
    from tests.shared.test_sensing_delay import _delay_payload

    payloads = []
    for index in range(4):
        s = _delay_payload(
            np.full((1, 1, 3), index + 1j),
            kind="delay_domain_cir",
            sample_times_s=[index / 10],
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
        s["config_resolved"] = dict(
            processing_mode="sequential_stft", carrier_freq_hz=24e9, range_mode="monostatic"
        )
        from shared.sensing.products import build_product_descriptor

        desc = s["products"]["authored_cir"]
        s["products"]["authored_cir"] = build_product_descriptor(
            s,
            **{
                key: desc[key]
                for key in (
                    "data_key",
                    "kind",
                    "axes",
                    "representation",
                    "unit",
                    "scale_semantics",
                    "pair_ids",
                    "combine",
                )
            },
            processing={"delay_projection": "point_bin_floor"},
        )
        s["temporal"].pop("result", None)
        payloads.append(s)

    class Provider:
        def __init__(self, *a, **k):
            pass

        def load_frame_projection(self, index, request):
            return SimpleNamespace(frame=SimpleNamespace(sensing=payloads[index]))

        def iter_frame_projections(self, ids, request):
            for i in ids:
                yield self.load_frame_projection(i, request)

        def close(self):
            pass

    (tmp_path / "frames_manifest.json").write_text("{}")
    monkeypatch.setattr("shared.frames.providers.Hdf5Provider", Provider)
    monkeypatch.setattr(
        "shared.frames.manifest.load_frame_manifest",
        lambda p: SimpleNamespace(
            frame_ids=list(range(4)), chunks=[], generation_id="test", frame_set_id="test"
        ),
    )
    x, times, _, provenance = load_motion_recording(tmp_path)
    np.testing.assert_array_equal(times, [0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(x[:, 0], np.arange(4) + 1j)
    assert provenance["product_count"] == 4


def test_numeric_import_does_not_load_rf_or_plotting():
    import subprocess
    import sys

    code = "import shared.sensing.motion_analysis, sys; assert not any(n in sys.modules for n in ['sionna', 'mitsuba', 'torch', 'matplotlib', 'PySide6'])"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"recipe": {"window": "unknown"}},
        {"recipe": {"window_samples": 70000}},
        {"display": {"normalization": "unknown"}},
    ],
)
def test_saved_definition_rejects_unsupported_recipe_before_ui_clamping(tmp_path, updates):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"schema": SCHEMA, "sources": [], **updates}))
    with pytest.raises(ValueError):
        load_definition(path)


@pytest.mark.parametrize(
    "change", ["shape", "negative", "nonfinite", "frequency", "time", "recipe", "loss"]
)
def test_portable_export_validates_numeric_structure_even_if_hash_is_updated(tmp_path, change):
    from shared.sensing.motion_analysis import file_identity

    directory = export_comparison(fixture_comparison(), tmp_path / "export", plots=False)
    metadata = json.loads((directory / "comparison.json").read_text())
    with np.load(directory / "spectra.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    key = "source_0_retained_psd"
    if change == "shape":
        arrays[key] = arrays[key][:, :-1]
    elif change == "negative":
        arrays[key][0, 0] = -1
    elif change == "nonfinite":
        arrays[key][0, 0] = np.nan
    elif change == "frequency":
        arrays["source_0_doppler_hz"][0] += 0.5
    elif change == "time":
        arrays["source_0_time_s"][0] += 0.5
    elif change == "recipe":
        metadata["results"][0]["metadata"]["recipe"]["window_samples"] = 18
    elif change == "loss":
        metadata["results"][0]["metadata"]["mean_removal_loss_db"] += 1
    np.savez_compressed(directory / "spectra.npz", **arrays)
    metadata["arrays_sha256"] = file_identity(directory / "spectra.npz")["sha256"]
    (directory / "comparison.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_export(directory)
