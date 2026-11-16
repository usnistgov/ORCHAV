"""Saved-input comparisons must retain source identity and isolate tracker state."""

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shared.sensing.experiment import compare_recipes
from shared.sensing.loader import load_saved_results
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from tests.shared.test_sensing_postprocessor import (
    _make_standard_frame,
    _with_single_target_truth,
    _write_packed_frame_set,
)

ROOT = Path(__file__).resolve().parents[2]


def _policy_recording(root: Path, policy: str) -> Path:
    """Persist complete synthetic truth/reference evidence beside one saved RD map."""
    from generator.core.scenario_actors.animated_asset import prepare_animated_asset
    from generator.core.scenario_actors.types import Timeline
    from shared.animated_target_workflow import _write_synthetic_asset
    from shared.sensing.targets import build_target_catalog
    from tests.shared.test_sensing_instantaneous_evaluation import (
        interval_payload,
        reference_payload,
    )

    power = np.ones((32, 64), dtype=np.float64)
    power[9 if policy == "root" else 16, 16] = 10.0
    sensing = _with_single_target_truth(
        frame_idx=64,
        sensing={
            "source_sample_time_s": 0.064,
            "detections": [],
            "range_doppler_map": power,
            "range_axis_m": np.arange(64, dtype=np.float64) / 4.0,
            "doppler_axis_hz": np.fft.fftshift(np.fft.fftfreq(32, d=0.001)),
        },
        target_range_m=4.0,
        target_speed_m_s=0.5,
    )
    epoch = sensing["temporal"]["result"]["reference_time_s"]
    radar = np.zeros(3)
    if policy == "root":
        _write_synthetic_asset(root / "asset")
        prepared = prepare_animated_asset(
            {
                "path": "asset/animated_target.json",
                "playback": {
                    "scenario_clock_id": "scenario:reference-test",
                    "motion_clock_id": "motion:reference-test",
                    "playback_rate": 1.0,
                    "playback_offset_s": 0.0,
                },
            },
            Timeline(129, 0.128),
            target_name="fixture_target",
            base_dir=root,
        )
        position = np.asarray(prepared.centers_m[64])
        radar = position - [4.0, 0.0, 0.0]
        reference = reference_payload(
            epoch=epoch,
            tx=radar,
            rx=radar,
            position=position,
            target_velocity=(-0.5, 0, 0),
        )
        catalog = build_target_catalog(
            [
                {
                    **sensing["target_catalog"]["targets"][0],
                    "animated_asset": prepared.catalog_metadata(),
                }
            ]
        )
        sensing["target_catalog"] = catalog
        for block in (sensing["target_state"], sensing["truth"]):
            block["catalog_revision"] = catalog["catalog_revision"]
            block["targets"][0].update(
                position_m=position.tolist(),
                velocity_m_s=[-0.5, 0, 0],
                animated_asset=prepared.sample_metadata(64),
            )
    else:
        reference = interval_payload()
        evidence = reference["truth"]["targets"][0]["pair_observables"][0][
            "velocity_reference_evidence"
        ]
        evidence.update(evaluation_epoch_s=epoch, target_position_m=[4.0, 0.0, 0.0])
        evidence["support"].update(
            start_time_s=epoch - 0.064,
            end_time_s=epoch,
            reference_time_s=epoch - 0.032,
        )
    observable = sensing["truth"]["targets"][0]["pair_observables"][0]
    replacement = deepcopy(reference["truth"]["targets"][0]["pair_observables"][0])
    replacement["pair_id"] = observable["pair_id"]
    replacement.update(
        bistatic_path_length_m=8.0,
        monostatic_equivalent_range_m=4.0,
        undefined_reason="Synthetic comparison omits angular reference observables",
    )
    evidence = replacement["velocity_reference_evidence"]
    evidence.update(pair_id=observable["pair_id"], target_id="fixture_target")
    sensing["truth"]["targets"][0]["pair_observables"] = [replacement]
    if policy != "legacy":
        sensing["truth"]["velocity_reference_policy"] = reference["truth"][
            "velocity_reference_policy"
        ]
    else:
        replacement.pop("velocity_reference_evidence")
    frame = replace(
        _make_standard_frame(frame_idx=64, sensing=sensing),
        tx_positions=radar[None, :],
        rx_positions=radar[None, :],
    )
    _write_packed_frame_set(root / "frames", frame)
    return root


def _synthetic_recording(root: Path) -> Path:
    """Write one typed saved power map without external recordings or RT."""
    power = np.ones((32, 64), dtype=np.float64)
    power[16, 16] = 10.0
    sensing = _with_single_target_truth(
        frame_idx=0,
        sensing={
            "range_doppler_map": power,
            "range_axis_m": np.arange(64, dtype=np.float64) / 4.0,
            "doppler_axis_hz": np.fft.fftshift(np.fft.fftfreq(32, d=0.001)),
        },
        target_range_m=4.0,
        target_speed_m_s=0.0,
    )
    _write_packed_frame_set(root / "frames", _make_standard_frame(frame_idx=0, sensing=sensing))
    return root


def test_tracker_factory_initializes_once_even_without_tracker():
    calls = []
    processor = SensingPostProcessor(tracker_factory=lambda config: calls.append(config))
    processor._ensure_tracker()
    processor._ensure_tracker()
    assert calls == [processor.config]


def test_saved_input_comparison_preserves_acquisition_and_distinguishes_recipes(tmp_path):
    source = _synthetic_recording(tmp_path / "source")
    configs = {
        name: PostProcessingConfig(
            cfar_threshold_scale=threshold,
            cfar_min_snr_db=6.0,
            min_detection_range_m=1.0,
            max_detection_range_m=8.0,
        )
        for name, threshold in [("sensitive", 3.0), ("conservative", 20.0)]
    }
    result = compare_recipes(source, configs, tmp_path / "output", figures=False)
    assert result["source_unchanged"]
    assert result["result_count"] == 1
    a = load_saved_results(tmp_path / "output" / "sensitive")[0]["_typed_sensing"]
    b = load_saved_results(tmp_path / "output" / "conservative")[0]["_typed_sensing"]
    assert a["temporal"]["acquisition"] == b["temporal"]["acquisition"]
    assert (
        a["temporal"]["result"]["processing_recipe_id"]
        != b["temporal"]["result"]["processing_recipe_id"]
    )
    assert a["truth"] == b["truth"]
    assert len(a["detections"]) == 1
    assert len(b["detections"]) == 0
    with pytest.raises(FileExistsError):
        compare_recipes(source, configs, tmp_path / "output", figures=False)


def test_recipe_cli_applies_defaults_overrides_and_summarizes_saved_results(tmp_path, monkeypatch):
    from generator.figures.summary import saved_sensing
    from shared.sensing.postprocessor import main

    source = _synthetic_recording(tmp_path / "source")
    output = tmp_path / "comparison"
    recipe = tmp_path / "recipes.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"cfar_threshold_scale": 3.0, "max_detection_range_m": 2.0},
                "recipes": {"clipped": {}, "visible": {"max_detection_range_m": None}},
                "evaluation": {"figures": False},
                "summary": True,
            }
        )
    )
    summaries = []

    def summarize(recording, destination):
        assert destination == recording / "summary"
        summaries.append((recording.name, len(load_saved_results(recording)[0]["detections"])))

    monkeypatch.setattr(saved_sensing, "generate_saved_sensing_summary", summarize)
    monkeypatch.setattr(
        sys, "argv", ["orchav-sensing", str(source), "--recipes", str(recipe), "-o", str(output)]
    )
    main()
    assert summaries == [("clipped", 0), ("visible", 1)]
    assert json.loads((output / "comparison.json").read_text())["source_unchanged"]
    assert json.loads((output / "visible/recipe.json").read_text())["cfar_threshold_scale"] == 3.0


@pytest.mark.parametrize("extra", [[], ["-o", "result", "--cfar-threshold", "7"]])
def test_recipe_cli_rejects_missing_output_and_conflicting_flags(tmp_path, monkeypatch, extra):
    from shared.sensing.postprocessor import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["orchav-sensing", "source", "--recipes", "r.json", *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "spec",
    [
        {"schema_version": 2, "recipes": {"one": {}}},
        {"schema_version": 1, "recipes": {"one": {}}, "typo": True},
        {"schema_version": 1, "recipes": {"one": {"unknown_setting": 3}}},
    ],
)
def test_invalid_recipe_file_does_not_create_outputs(tmp_path, spec):
    from shared.sensing.experiment import run_recipe_file

    recipe = tmp_path / "recipes.json"
    recipe.write_text(json.dumps(spec))
    with pytest.raises((ValueError, TypeError)):
        run_recipe_file(tmp_path / "missing-source", recipe, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("policy", ["root", "interval", "legacy"])
def test_comparison_scores_saved_reference_policy_and_velocity_support(tmp_path, policy):
    from shared.sensing.evaluation import INSTANTANEOUS_REFERENCE_PROTOCOL

    source = _policy_recording(tmp_path / "source", policy)
    result = compare_recipes(
        source,
        {
            "one": PostProcessingConfig(
                cfar_threshold_scale=3.0,
                cfar_min_snr_db=6.0,
                min_detection_range_m=1.0,
                max_detection_range_m=8.0,
            )
        },
        tmp_path / "output",
        figures=False,
    )
    metrics = result["recipes"]["one"]["metrics"]
    assert result["source_unchanged"]
    if policy == "legacy":
        assert (metrics["total_tp"], metrics["total_fp"], metrics["total_fn"]) == (0, 1, 1)
        assert metrics["evaluation_reference_protocol"] != INSTANTANEOUS_REFERENCE_PROTOCOL
    else:
        assert (metrics["total_tp"], metrics["total_fp"], metrics["total_fn"]) == (1, 0, 0)
        assert metrics["evaluation_reference_protocol"] == INSTANTANEOUS_REFERENCE_PROTOCOL
        if policy == "root":
            assert metrics["velocity_matched_pair_count"] == 1
            assert metrics["cumulative_velocity_rmse"] < 0.05
        else:
            assert metrics["velocity_matched_pair_count"] == 0
            assert metrics["velocity_excluded_matched_pair_count"] == 1
            assert metrics["velocity_unavailable_opportunity_count"] == 1
            assert metrics["cumulative_velocity_rmse"] is None


def test_comparison_figure_uses_saved_evaluation_reference_and_preserves_input(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth
    from shared.sensing.experiment import _comparison_figure

    source = _policy_recording(tmp_path / "source", "root")
    results = load_saved_results(source)
    sensing = results[0]["_typed_sensing"]
    original_truth = deepcopy(sensing["truth"])
    reference = extract_recorded_evaluation_truth(sensing)["target_speeds"][0]
    figures = []
    original_subplots = plt.subplots

    def capture(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture)
    _comparison_figure(
        {"one": results},
        {
            "recipes": {
                "one": {
                    "metrics": {"total_tp": 1, "total_fp": 0, "total_fn": 0},
                    "processing_seconds": 0.0,
                }
            }
        },
        tmp_path / "comparison.png",
    )
    velocity_axes = figures[0][1][1, 0]
    np.testing.assert_array_equal(velocity_axes.lines[0].get_ydata(), [reference])
    assert sensing["truth"] == original_truth


@pytest.mark.parametrize("range_only", [False, True])
def test_tracked_comparison_retains_reports_and_declares_unavailable_mot(tmp_path, range_only):
    import json
    from dataclasses import asdict

    from shared.sensing.evaluation import compute_mot_metrics

    source = (
        _policy_recording(tmp_path / "source", "interval")
        if range_only
        else _synthetic_recording(tmp_path / "source")
    )
    output = tmp_path / "output"
    summary = compare_recipes(
        source,
        {
            "tracked": PostProcessingConfig(
                cfar_threshold_scale=3.0,
                cfar_min_snr_db=6.0,
                min_detection_range_m=1.0,
                max_detection_range_m=8.0,
                tracker_type="kalman",
                tracker_m=1,
                tracker_n=1,
            )
        },
        output,
        figures=False,
    )
    saved = load_saved_results(output / "tracked")[0]["_typed_sensing"]
    report = saved["detections"][0]
    assert report["track_id"] == 1 and report["track_confirmed"]
    assert summary["source_unchanged"]
    record = summary["recipes"]["tracked"]
    assert (
        record["metrics"]["total_tp"],
        record["metrics"]["total_fp"],
        record["metrics"]["total_fn"],
    ) == (
        1,
        0,
        0,
    )
    if range_only:
        assert "tracking" not in record
        assert "velocity truth" in record["tracking_unavailable_reason"]
        assert record["metrics"]["velocity_matched_pair_count"] == 0
        assert record["metrics"]["cumulative_velocity_rmse"] is None
        persisted = json.loads((output / "tracked" / "evaluation.json").read_text())
        assert persisted == record
        assert (
            "tracking"
            not in json.loads((output / "comparison.json").read_text())["recipes"]["tracked"]
        )
    else:
        expected = compute_mot_metrics(
            [(["fixture_target"], np.array([4.0]), np.array([0.0]))],
            [saved["detections"]],
            range_gate=0.6,
            velocity_gate=0.25,
        )
        assert record["tracking"] == asdict(expected)
        assert "tracking_unavailable_reason" not in record


@pytest.mark.parametrize("gate", [0.0, -1.0, float("nan"), float("inf")])
def test_comparison_rejects_invalid_gates_before_opening_input(tmp_path, gate):
    with pytest.raises(ValueError, match="gates"):
        compare_recipes(
            tmp_path, {"a": PostProcessingConfig()}, tmp_path / "out", range_gate_m=gate
        )


@pytest.mark.parametrize("option", ["vote_fusion_enabled", "angle_peak_detection_enabled"])
def test_cfr_comparison_rejects_deferred_algorithms_before_creating_output(tmp_path, option):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="typed.*postprocessing not implemented"):
        compare_recipes(
            ROOT / "scenarios/sensing/foundations/portable_replay",
            {"unsupported": PostProcessingConfig(range_ifft_size=256, **{option: True})},
            output,
            figures=False,
        )
    assert not output.exists()
