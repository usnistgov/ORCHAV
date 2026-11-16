"""Runnable sensing examples expose summaries without changing observation settings."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from generator import build_simulation_config
from shared.scenarios import load_scenario_configuration

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "scripts/release/sensing_catalog.json"
if CATALOG_PATH.is_file():
    catalog = json.loads(CATALOG_PATH.read_text())
    PUBLIC = [
        entry["path"]
        for entry in catalog["scenarios"]
        if entry["status"].startswith("public_")
        and entry["path"].startswith("scenarios/sensing/")
        and entry["path"].endswith("scenario.yaml")
    ]
else:
    # The curated public checkout omits its private source catalog. Only authored
    # examples have a sibling README; prepared/output configurations do not.
    PUBLIC = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scenarios/sensing").rglob("scenario.yaml"))
        if (path.parent / "README.md").is_file()
    ]


@pytest.mark.parametrize("path", PUBLIC)
def test_public_sensing_examples_enable_native_products_and_their_tracker(path):
    scenario = load_scenario_configuration((ROOT / path).parent)
    sensing = build_simulation_config(scenario).sensing
    summary = scenario.generator_summary
    assert sensing.enabled
    assert summary["enabled"] and "sensing" in summary["create"]
    figures = set(summary["sensing"]["figures"])
    # Tracking examples may emphasize trajectories instead of the two 1D
    # projections, but every example retains the native range-Doppler product.
    assert "range_doppler" in figures
    if sensing.track_confirmation_enabled:
        if sensing.tracker_type == "cartesian3d":
            assert "tracking_xyz" in figures
        elif sensing.tracker_type == "cartesian":
            assert "tracking_xy" in figures
        else:
            assert "tracking" in figures
    else:
        assert {"range_profile", "doppler"} <= figures
    inspections = summary["sensing"].get("inspection_figures", [])
    if "detection_stages" in inspections:
        assert sensing.persist_detection_stage_artifacts
    if "tracking_decisions" in inspections:
        assert sensing.persist_tracking_diagnostics and sensing.track_confirmation_enabled


@pytest.mark.parametrize("retention", ["full", "volume"])
def test_wall_campaign_keeps_one_native_summary_per_condition(tmp_path, retention):
    from scenarios.sensing.applications.wall_detection_volume.generate import prepare

    output = tmp_path / "campaign"
    prepare(output, grid="tutorial", retention=retention)
    for condition in ("wall", "open"):
        root = output / condition
        spec = json.loads((root / "volume_recipe.json").read_text())
        pending = [trial for trial in spec["trials"] if trial["status"] == "pending"]
        assert spec["summary_trial_ids"] == [pending[0]["id"]]
        if retention == "volume":
            assert spec["summary_trial_ids"] == spec["retention"]["reference_trial_ids"]
        for index, trial in enumerate(pending):
            document = yaml.safe_load((root / trial["native"] / "scenario.yaml").read_text())
            assert document["generator_summary"]["enabled"] is (index == 0)
            assert trial["native_summary"] is (index == 0)
            assert "sensing" in document["generator_summary"]["create"]
            # The per-trial summary decision cannot change propagation/processing.
            assert (
                document["sensing"]
                == yaml.safe_load((root / pending[0]["native"] / "scenario.yaml").read_text())[
                    "sensing"
                ]
            )


def test_reference_summary_skips_excluded_cells_and_preserves_inputs():
    from scenarios.sensing.volume_storage import configure_trial_summary

    config = {
        "sensing": {"enabled": True, "preset": "coherent_hann"},
        "generator_summary": {"enabled": True, "create": ["sensing"]},
    }
    excluded = [{"id": "blocked", "status": "excluded_geometry"}]
    original = deepcopy(config)
    assert configure_trial_summary(config, excluded)
    assert config == original
    second = deepcopy(original)
    assert not configure_trial_summary(second, [*excluded, {"id": "first", "status": "pending"}])
    assert second["sensing"] == original["sensing"]
    assert second["generator_summary"] == {**original["generator_summary"], "enabled": False}


def test_cube_reprocessing_uses_the_shared_recipe_file(tmp_path, monkeypatch):
    import sys

    from shared.sensing import experiment as sensing_experiment
    from shared.sensing import postprocessor

    source, output = tmp_path / "source", tmp_path / "outputs"
    (source / "frames").mkdir(parents=True)
    (source / "frames/frames_manifest.json").write_text("{}")
    calls = []

    monkeypatch.setattr(sensing_experiment, "run_recipe_file", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchav-sensing",
            str(source / "frames"),
            "--recipes",
            str(ROOT / "scenarios/sensing/cube_radar/filtering.json"),
            "--output-dir",
            str(output),
        ],
    )
    postprocessor.main()
    assert calls == [
        (source / "frames", ROOT / "scenarios/sensing/cube_radar/filtering.json", output)
    ]
