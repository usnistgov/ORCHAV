"""Grouped authoring must preserve fully resolved scientific recipes."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from generator.core.configuration.loader import load_simulation_config
from shared.scenarios.loader import load_scenario_configuration
from shared.scenarios.model import ScenarioModel, SensingModel, SensingSummaryModel
from shared.scenarios.sensing_groups import (
    GROUPS,
    normalize_scenario_sensing,
    normalize_sensing_groups,
)

ROOT = Path(__file__).resolve().parents[2]


def leaves(spec, prefix=()):
    for key, value in spec.items():
        if isinstance(value, dict):
            yield from leaves(value, (*prefix, key))
        else:
            yield (*prefix, key), value


def grouped(flat):
    result = {"enabled": flat.get("enabled", False)}
    used = set()
    for path, destination in leaves(GROUPS):
        if destination in used:
            continue
        value = flat
        for part in destination.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            node = result
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
            used.add(destination)
    for key in ("display_range_xlim", "display_velocity_ylim"):
        if key in flat:
            result[key] = flat[key]
    return result


def test_all_existing_sensing_fields_have_grouped_ownership():
    fields = {dest.split(".")[0] for _, dest in leaves(GROUPS)}
    assert fields | {"enabled", "display_range_xlim", "display_velocity_ylim"} == set(
        SensingModel.model_fields
    )


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in (ROOT / "scenarios/sensing").rglob("scenario.yaml")
        if p.parent.joinpath("README.md").is_file()
        and not {"runs", "analysis", ".orchav"}.intersection(p.relative_to(ROOT).parts)
    ],
)
def test_authored_recipe_grouping_is_numerically_equivalent(path):
    raw = yaml.safe_load(path.read_text())
    old = raw.get("sensing")
    if not old or any(k in old for k in GROUPS) or "preset" in old:
        return
    new = grouped(old)
    # samples_per_cpi is coherent-only; sequential/oracle FFT sizes belong to Doppler.
    if old.get("processing_mode") != "coherent_cpi":
        count = new.get("acquisition", {}).pop("samples_per_cpi", None)
        if count is not None:
            new.setdefault("processing", {}).setdefault("doppler", {})["fft_size"] = count
    a = SensingModel.model_validate(old).model_dump(exclude_none=True)
    b = SensingModel.model_validate(new).model_dump(exclude_none=True)
    assert a == b
    rt = raw.get("raytracing", {})

    def runtime(values):
        return asdict(
            load_simulation_config(SimpleNamespace(sensing=values, raytracing=rt)).sensing
        )

    assert runtime(a) == runtime(b)
    assert raw.get("sensing") == old


def test_preset_matches_the_stationary_scientific_recipe(tmp_path):
    path = ROOT / "scenarios/sensing/cube_radar/stationary"
    scenario = load_scenario_configuration(path)
    actual = load_simulation_config(scenario)
    assert actual.cir_time_steps == 64
    assert actual.cir_sampling_frequency_hz == 1000
    expected = {
        "enabled": True,
        "processing_mode": "coherent_cpi",
        "bandwidth": 500e6,
        "range_mode": "monostatic",
        "prf_hz": 1000.0,
        "fft_size_range": 128,
        "fft_size_doppler": 64,
        "window_range": "boxcar",
        "range_response": {
            "model": "cfr_ifft",
            "num_frequency_bins": 128,
            "frequency_window": "hann",
        },
        "cfar_threshold_scale": 2.0,
        "cfar_min_snr_db": 3.0,
        "cfar_max_dynamic_range_db": 12.0,
        "cfar_rd_ref_cells_range": 6,
        "output_range_profile": True,
        "recording_profile": "maps",
        "coherent_clutter_mode": "none",
        "persist_detection_stage_artifacts": True,
        "cluster_range_gate_m": 0.5,
        "cluster_velocity_gate_m_s": 0.3,
    }
    reference = load_simulation_config(
        SimpleNamespace(sensing=expected, raytracing=scenario.raytracing)
    )
    assert asdict(actual.sensing) == asdict(reference.sensing)


@pytest.mark.parametrize(
    "data,match",
    [
        ({"tracking": {"enable": True}}, "Unknown sensing setting"),
        ({"detection": 1}, "must be a mapping"),
        ({"prf_hz": 1000, "acquisition": {"sample_rate_hz": 2000}}, "duplicates"),
        (
            {"processing": {"doppler": {"fft_size": 64}}, "acquisition": {"samples_per_cpi": 64}},
            "duplicates",
        ),
        (
            {
                "range_response": {"frequency_window": "hann"},
                "processing": {"range": {"frequency_window": "boxcar"}},
            },
            "duplicates",
        ),
        ({"preset": "not_a_preset"}, "Unknown sensing preset"),
    ],
)
def test_rejects_ambiguous_or_misspelled_grouped_settings(data, match):
    with pytest.raises((ValueError, ValidationError), match=match):
        SensingModel.model_validate(data)


def test_coherent_clock_aliases_and_no_mutation():
    raw = {
        "sensing": {
            "enabled": True,
            "acquisition": {"mode": "coherent_cpi", "samples_per_cpi": 32, "sample_rate_hz": 2000},
        }
    }
    before = deepcopy(raw)
    value = normalize_scenario_sensing(raw)
    assert value["raytracing"] == {"cir_time_steps": 32, "cir_sampling_frequency_hz": 2000}
    assert raw == before
    with pytest.raises(ValueError, match="conflicts"):
        normalize_scenario_sensing({**raw, "raytracing": {"cir_time_steps": 64}})
    with pytest.raises(ValueError, match="requires mode"):
        normalize_scenario_sensing({"sensing": {"acquisition": {"samples_per_cpi": 64}}})


def test_legacy_defaults_not_changed():
    assert normalize_sensing_groups({"enabled": True}) == {"enabled": True}
    config = load_simulation_config(SimpleNamespace(sensing={"enabled": True}, raytracing={}))
    assert config.sensing.range_response.model == "point_bin_floor"
    assert config.sensing.coherent_clutter_mode == "legacy"


@pytest.mark.parametrize(
    "settings",
    [
        {"preset": "coherent_hann"},
        {"acquisition": {"mode": "coherent_cpi"}},
    ],
)
def test_new_coherent_authoring_defaults_to_weighted_clutter_suppression(settings):
    assert normalize_sensing_groups(settings)["coherent_clutter_mode"] == "windowed_mean"
    settings["processing"] = {"clutter": {"coherent_mode": "none"}}
    assert normalize_sensing_groups(settings)["coherent_clutter_mode"] == "none"


@pytest.mark.parametrize("mode", ["legacy", "unweighted_mean"])
def test_unweighted_authoring_name_preserves_the_saved_operator(mode):
    settings = {"processing": {"clutter": {"coherent_mode": mode}}}
    assert normalize_sensing_groups(settings)["coherent_clutter_mode"] == "legacy"
    assert SensingModel.model_validate(settings).coherent_clutter_mode == "legacy"


def test_noise_and_view_groups():
    data = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0},
        "sensing": {
            "enabled": True,
            "preset": "coherent_hann",
            "acquisition": {"frequency_samples": 128},
            "noise": {"frequency": {"variance": 0.01, "seed": 37}},
        },
        "view_defaults": {"sensing": {"range_xlim": [1, 8]}},
        "generator_summary": {"sensing": {"figures": ["doppler"], "range_doppler_views": ["full"]}},
    }
    model = ScenarioModel.model_validate(data)
    assert model.sensing.display_range_xlim == [1, 8]
    assert model.sensing.range_response.frequency_noise.variance == 0.01
    assert model.sensing.range_response.frequency_noise.seed == 37
    with pytest.raises(ValidationError):
        ScenarioModel.model_validate(
            {**data, "generator_summary": {"sensing": {"figures": ["typo"]}}}
        )


def test_document_roundtrip_and_template_overrides():
    from generator.core.configuration.loader import resolve_sensing_document

    path = ROOT / "scenarios/sensing/cube_radar/azimuth/scenario.yaml"
    data = yaml.safe_load(path.read_text())
    model = ScenarioModel.model_validate(data)
    ScenarioModel.model_validate(model.model_dump(exclude_none=True))
    flat = resolve_sensing_document(data)
    flat["sensing"]["display_range_xlim"] = [0, 20]
    flat["sensing"]["cfar_threshold_scale"] = 7
    config = load_simulation_config(SimpleNamespace(**flat))
    assert config.sensing.cfar_threshold_scale == 7
    assert config.sensing.display_range_xlim == [0, 20]


def test_summary_display_options_preserve_defaults_and_document_roundtrip():
    defaults = SensingSummaryModel()
    assert defaults.history_time_axis == "scene_frame"
    assert defaults.range_profile_views == ["zoom_to_data", "full"]
    assert defaults.range_doppler_views == ["zoom_to_data", "full"]

    document = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0},
        "generator_summary": {
            "sensing": {
                "history_time_axis": "time",
                "range_profile_views": ["full"],
                "range_doppler_views": ["zoom_to_data"],
            }
        },
    }
    model = ScenarioModel.model_validate(document)
    serialized = yaml.safe_load(yaml.safe_dump(model.model_dump(exclude_none=True)))
    roundtrip = ScenarioModel.model_validate(serialized)
    options = roundtrip.generator_summary.sensing
    assert options.history_time_axis == "time"
    assert options.range_profile_views == ["full"]
    assert options.range_doppler_views == ["zoom_to_data"]


@pytest.mark.parametrize(
    "options",
    [
        {"history_time_axis": "frame"},
        {"range_profile_views": ["zoom"]},
        {"range_doppler_views": ["zoom"]},
    ],
)
def test_summary_display_options_reject_unknown_values(options):
    with pytest.raises(ValidationError):
        SensingSummaryModel.model_validate(options)


def test_simplifier_preserves_implicit_unweighted_clutter():
    from scripts.analysis.simplify_sensing_yaml import fingerprint, simplify

    data = {
        "schema_version": 2,
        "timeline": {"steps": 64, "sample_rate_hz": 1000},
        "raytracing": {"cir_time_steps": 64, "cir_sampling_frequency_hz": 1000},
        "sensing": {
            "enabled": True,
            "processing_mode": "coherent_cpi",
            "prf_hz": 1000,
            "fft_size_doppler": 64,
            "window_range": "boxcar",
            "range_response": {"model": "cfr_ifft", "num_frequency_bins": 128},
        },
    }
    grouped_data, before = simplify(data)
    assert grouped_data["sensing"]["processing"]["clutter"]["coherent_mode"] == "unweighted_mean"
    assert fingerprint(grouped_data) == before


def test_processing_preset_does_not_supply_acquisition_parameters():
    from shared.scenarios.sensing_groups import COHERENT_HANN

    assert not {"bandwidth", "prf_hz", "fft_size_doppler", "range_mode"}.intersection(COHERENT_HANN)
    assert "num_frequency_bins" not in COHERENT_HANN["range_response"]
    raw = {"sensing": {"preset": "coherent_hann"}}
    assert "raytracing" not in normalize_scenario_sensing(raw)
    raw["sensing"]["acquisition"] = {
        "bandwidth_hz": 200e6,
        "frequency_samples": 256,
        "sample_rate_hz": 4000,
        "samples_per_cpi": 32,
    }
    result = normalize_scenario_sensing(raw)
    assert result["sensing"]["bandwidth"] == 200e6
    assert result["sensing"]["range_response"]["num_frequency_bins"] == 256
    assert result["raytracing"] == {"cir_time_steps": 32, "cir_sampling_frequency_hz": 4000}


def test_redundant_authoring_preserves_solver_settings_and_explicit_panel_hide():
    from scripts.analysis.simplify_sensing_yaml import (
        effective_fingerprint,
        omit_redundant_authoring,
    )

    raw = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0},
        "sensing": {"enabled": True},
        "raytracing": {"quality": {"custom": {"max_depth": 1, "samples_per_src": 10000}}},
        "data": {"files": {"format": "hdf5"}},
        "visualizer": {"panels": {"sensing": {"enabled": True}}},
    }
    result = omit_redundant_authoring(raw)
    assert "data" not in result
    assert "visualizer" not in result
    assert result["raytracing"] == raw["raytracing"]
    assert effective_fingerprint(result) == effective_fingerprint(raw)
    result["raytracing"]["quality"]["custom"]["max_depth"] = 2
    assert effective_fingerprint(result) != effective_fingerprint(raw)
    raw["visualizer"]["panels"]["sensing"]["enabled"] = False
    assert omit_redundant_authoring(raw)["visualizer"] == raw["visualizer"]
