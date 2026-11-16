"""Saved-input fusion keeps clocks, sources, estimation and truth separate."""

import json
from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.fusion.core import Observation, Sensor, measurement_model
from shared.sensing.fusion.experiment import load_sensor, run_experiment


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    sources = {}
    for name, position, angle in [
        ("West", (0.0, 0.0, 1.0), 0.0),
        ("South", (5.0, -5.0, 1.0), np.pi / 2),
    ]:
        sensor = Sensor(name, position, (angle, 0.0, 0.0))
        epochs, truth = [], []
        for i in range(5):
            t = i * 0.1
            p, v = np.array([5.0 + t, 0.3 * t, 1.0]), np.array([1.0, 0.3, 0.0])
            z, _ = measurement_model(np.r_[p, v], sensor)
            epochs.append({"time_s": t, "observations": [Observation(name, str(i), t, *z)]})
            truth.append(
                {
                    "time_s": t,
                    "targets": [
                        {"target_id": "cube", "position_m": p.tolist(), "velocity_m_s": v.tolist()}
                    ],
                }
            )
        sources[name] = {
            "sensor": sensor,
            "epochs": epochs,
            "truth": truth,
            "directory": tmp_path / name,
            "source": {
                "frame_set_id": name,
                "native_clock_id": name,
                "pair_id": name,
                "hashes": {},
            },
        }
    monkeypatch.setattr(
        "shared.sensing.fusion.experiment.load_sensor", lambda path, name: deepcopy(sources[name])
    )
    monkeypatch.setattr("shared.sensing.fusion.experiment.fingerprints", lambda path: {})
    recipe = {
        "schema_version": 1,
        "clock": {"id": "common", "assumption": "synchronized", "unit": "s"},
        "world_frame": {
            "id": "test_world",
            "origin": "scene_origin",
            "axes": ["x", "y", "z"],
            "handedness": "right",
            "length_unit": "m",
            "angle_unit": "rad",
        },
        "sensors": {
            name: {
                "source": name,
                "time_offset_s": 0.0,
                "world_frame_id": "test_world",
                "alignment": "explicit_shared_world",
            }
            for name in sources
        },
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(recipe))
    return path, sources, recipe


def test_saved_experiment_reproducible_and_truth_independent(experiment, tmp_path):
    path, sources, _ = experiment
    run_experiment(path, tmp_path / "a")
    first = json.loads((tmp_path / "a/fusion_results.json").read_text())
    for data in sources.values():
        for epoch in data["truth"]:
            epoch["targets"][0]["position_m"] = [99.0, 88.0, 77.0]
    run_experiment(path, tmp_path / "b")
    second = json.loads((tmp_path / "b/fusion_results.json").read_text())
    assert first["histories"] == second["histories"]
    assert first["sources_unchanged"]
    saved_recipe = json.loads((tmp_path / "a/experiment.json").read_text())
    assert (
        tmp_path / "a" / saved_recipe["sensors"]["West"]["source"]
    ).resolve() == tmp_path / "West"
    assert (tmp_path / "a/figures/velocity_components.png").is_file()
    assert (tmp_path / "a/figures/track_availability.png").is_file()
    for mode in ("West", "South", "fused"):
        assert (tmp_path / "a/figures" / f"tracking_xyz_{mode}.png").is_file()
    score = json.loads((tmp_path / "b/evaluation.json").read_text())
    assert score["modes"]["fused"]["matched_outputs"] == 0
    with pytest.raises(FileExistsError):
        run_experiment(path, tmp_path / "a")


def test_spatial_campaign_can_save_complete_replay_without_figures(experiment, tmp_path):
    from shared.sensing.fusion.result import FusionResult

    path, _, _ = experiment
    outcome = run_experiment(path, tmp_path / "spatial", figures=False)
    reader = FusionResult(tmp_path / "spatial")
    assert outcome["evaluation"] is None and outcome["sources_unchanged"]
    assert len(reader.times) == 5
    assert reader.decisions(0, "fused")
    assert not (tmp_path / "spatial/figures").exists()


def test_one_sensor_baseline_uses_the_same_reader_and_tracker(experiment, tmp_path):
    from shared.sensing.fusion.result import FusionResult
    from shared.sensing.uav_evaluation import PROFILE, evaluate_manifest

    path, _, recipe = experiment
    recipe["sensors"].pop("South")
    path.write_text(json.dumps(recipe))
    run_experiment(path, tmp_path / "one")
    reader = FusionResult(tmp_path / "one")
    assert reader["histories"]["fused"] == reader["histories"]["West"]
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "evaluation": {},
        "cohorts": ["detections:West", "tracks:fused"],
        "trials": [{"run_id": "one", "result": "one", "case": "fixture"}],
    }
    source = tmp_path / "uav.json"
    source.write_text(json.dumps(manifest))
    report = evaluate_manifest(source, tmp_path / "report")
    assert report["cohorts"]["detections:West"]["counts"]["tp"] == 5
    assert (tmp_path / "report/index.html").is_file()
    assert report["cases"]["fixture"]["tracks:fused"]["counts"]["fn"] > 0
    manifest["trials"].append({"run_id": "copy", "result": "one"})
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same result"):
        evaluate_manifest(source, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


@pytest.mark.parametrize(
    "failure", ["offset", "schedule", "duplicate", "truth", "unknown_clock", "overlap"]
)
def test_invalid_bindings_rejected(experiment, tmp_path, failure):
    path, sources, recipe = experiment
    output = tmp_path / "output"
    if failure == "offset":
        recipe["sensors"]["South"]["time_offset_s"] = 0.01
    elif failure == "schedule":
        sources["South"]["epochs"].pop()
    elif failure == "duplicate":
        sources["South"]["source"]["frame_set_id"] = "West"
    elif failure == "truth":
        sources["South"]["truth"][0]["targets"][0]["position_m"] = [1, 2, 3]
    elif failure == "unknown_clock":
        recipe["clock"]["assumption"] = "infer"
    else:
        output = tmp_path / "West/output"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        run_experiment(path, output)
    assert not output.exists()


def test_missing_truth_does_not_count_detections_as_false_positives(experiment, tmp_path):
    path, sources, _ = experiment
    for data in sources.values():
        for epoch in data["truth"]:
            epoch.update(available=False, targets=[])
    run_experiment(path, tmp_path / "no_truth")
    evaluation = json.loads((tmp_path / "no_truth/evaluation.json").read_text())
    assert evaluation["modes"]["fused"]["truth_unavailable_epochs"] == 5
    assert evaluation["modes"]["fused"]["unmatched_confirmed_outputs"] == 0
    assert evaluation["modes"]["fused"]["position_rmse_m"] is None
    assert evaluation["modes"]["fused"]["all_targets_coverage"] is None


def test_simultaneous_coverage_requires_every_present_target(experiment, tmp_path):
    from shared.sensing.fusion.figures import evaluate_fusion

    path, _, _ = experiment
    run_experiment(path, tmp_path / "coverage", figures=False)
    result = json.loads((tmp_path / "coverage/fusion_results.json").read_text())
    truth = result["truth_for_evaluation"]
    truth[0].update(available=False, targets=[])
    truth[1]["targets"] = []  # Empty scenes are not target-coverage opportunities.
    for epoch in truth[3:]:
        epoch["targets"].append(
            {"target_id": "second", "position_m": [50, 50, 50], "velocity_m_s": [0, 0, 0]}
        )
    last = result["histories"]["fused"][-1]
    second = deepcopy(last["tracks"][0])
    second.update(track_id=999, position_m=[50, 50, 50], confirmed=True)
    extra = deepcopy(second)
    extra.update(track_id=1000, position_m=[100, 100, 100])
    last["tracks"].extend([second, extra])

    score = evaluate_fusion(result)["modes"]["fused"]
    assert score["target_present_epochs"] == 3
    assert score["all_targets_matched_epochs"] == 2
    assert score["all_targets_coverage"] == pytest.approx(2 / 3)
    assert score["matched_outputs"] == 4  # One, one, then both targets.
    assert score["unmatched_confirmed_outputs"] == 1  # Count extras independently.


def test_evaluation_gate_changes_only_scoring(experiment, tmp_path):
    path, sources, recipe = experiment
    for data in sources.values():
        for epoch in data["truth"]:
            epoch["targets"][0]["position_m"][2] += 1.0
    run_experiment(path, tmp_path / "tight")
    recipe["evaluation"] = {"position_gate_m": 3.0}
    path.write_text(json.dumps(recipe))
    run_experiment(path, tmp_path / "wide")
    a = json.loads((tmp_path / "tight/fusion_results.json").read_text())
    b = json.loads((tmp_path / "wide/fusion_results.json").read_text())
    assert a["histories"] == b["histories"]
    assert (
        json.loads((tmp_path / "tight/evaluation.json").read_text())["modes"]["fused"][
            "matched_outputs"
        ]
        == 0
    )
    assert (
        json.loads((tmp_path / "wide/evaluation.json").read_text())["modes"]["fused"][
            "matched_outputs"
        ]
        == 3
    )


def test_alignment_never_inferred_from_matching_truth(experiment, tmp_path):
    path, _, recipe = experiment
    recipe.pop("world_frame")
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="world_frame"):
        run_experiment(path, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_atomic_bundle_and_lazy_decision_integrity(experiment, tmp_path, monkeypatch):
    from shared.sensing.fusion.result import FusionResult

    path, _, _ = experiment
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("figure failure")

        patch.setattr("shared.sensing.fusion.figures.generate_fusion_figures", fail)
        with pytest.raises(RuntimeError, match="figure failure"):
            run_experiment(path, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-*"))
    run_experiment(path, tmp_path / "valid")
    result = FusionResult(tmp_path / "valid")
    assert result.epoch_index(0.1999) == 2
    assert result.decisions(1)["decisions"]
    assert result.metadata["code_identity"]["implementation_sha256"]
    decision_file = tmp_path / "valid/fusion_decisions.jsonl"
    decision_file.write_bytes(b"x" * decision_file.stat().st_size)
    # Opening compact states never reads the detailed decision file.
    result = FusionResult(tmp_path / "valid")
    assert len(result.times) == 5
    with pytest.raises(ValueError, match="integrity"):
        result.decisions(1)


def test_v1_reader_marks_unrecorded_information(experiment, tmp_path):
    from shared.sensing.fusion.result import FusionResult

    path, _, _ = experiment
    run_experiment(path, tmp_path / "a")
    payload = json.loads((tmp_path / "a/fusion_results.json").read_text())
    payload["schema_version"] = 1
    for key in ("run_id", "world_frame", "code_identity", "decision_index"):
        payload.pop(key)
    result = FusionResult(payload)
    assert result.metadata["world_frame"] == "not recorded"
    assert result.decisions(0)["decisions"] == "not recorded"


def test_saved_state_validation(experiment, tmp_path):
    from shared.sensing.fusion.result import FusionResult

    path, _, _ = experiment
    run_experiment(path, tmp_path / "a")
    payload = json.loads((tmp_path / "a/fusion_results.json").read_text())
    payload["histories"]["fused"][0]["tracks"][0]["covariance"][0][0] = -10
    with pytest.raises(ValueError, match="covariance"):
        FusionResult(payload)


@pytest.mark.parametrize("selection", [[], [False], [1, 1], [99], [1.0]])
def test_selected_result_frames_are_strict_and_close_provider(tmp_path, monkeypatch, selection):
    from types import SimpleNamespace

    from shared.sensing.fusion import experiment as module

    closed = []
    provider = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        module,
        "_open_provider",
        lambda source: (
            tmp_path,
            SimpleNamespace(frame_ids=[0, 1]),
            provider,
        ),
    )
    with pytest.raises(ValueError, match="Result storage frame IDs"):
        module.load_sensor(tmp_path, "Radar", result_frame_ids=selection)
    assert closed == [True]


def test_selected_results_do_not_read_unrequested_detection_frames(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from shared.sensing.fusion import experiment as module

    selected, closed = [], []

    def projections(frame_ids, request):
        selected.extend(frame_ids)
        return iter([SimpleNamespace(frame=SimpleNamespace(sensing={}))])

    provider = SimpleNamespace(
        iter_frame_projections=projections, close=lambda: closed.append(True)
    )
    monkeypatch.setattr(
        module,
        "_open_provider",
        lambda source: (
            tmp_path,
            SimpleNamespace(frame_ids=[0, 1]),
            provider,
        ),
    )
    with pytest.raises(ValueError, match="No saved sensing results"):
        module.load_sensor(tmp_path, "Radar", result_frame_ids=[1])
    assert selected == [1] and closed == [True]


@pytest.fixture
def native_sensor_payloads(tmp_path, monkeypatch):
    """Exercise the real v4 reader contract without loading signal tensors."""
    from types import SimpleNamespace

    from shared.sensing.fusion import experiment as module
    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload

    payloads = []
    for index in range(5):
        time_s = index * 0.1
        report = {
            "range_m": 10.0 + time_s,
            "velocity_m_s": -1.0,
            "aoa_az_deg": 10.0,
            "aoa_el_deg": 5.0,
            "x_m": 9.8106,
            "y_m": 1.7299,
            "z_m": 0.8716,
            "range_bin": 10,
            "doppler_bin": 0,
            "snr_db": 17.5,
        }
        payload = build_visualizer_sensing_payload(
            {
                "config_resolved": {
                    "track_confirmation_enabled": False,
                    "processing_mode": "coherent_cpi",
                    "acquisition": {
                        "sample_source": "coherent_one_solve",
                        "sample_period_s": 0.001,
                        "sample_rate_hz": 1000.0,
                        "phase_contract": "sionna_intra_solve_phase_coherent",
                    },
                    "processing": {
                        "mode": "coherent_cpi",
                        "strategy": "coherent_cpi_fft",
                        "product_semantics": "cir_point_bin_slow_time_fft",
                    },
                    "topology": {"rx_elements": 4, "rx_array_processing": "element_preserving"},
                },
                "processing_strategy": "coherent_cpi_fft",
                "rx_array": {"rows": 2, "columns": 2},
                "detections": [] if index == 3 else [report],
                "detection_filter_metadata": {"tracker": {"enabled": False}},
            },
            scene_frame_id=index,
            sample_time_s=time_s,
        )
        payloads.append(payload)

    def projections(frame_ids, request):
        assert request.sensing_products == frozenset({"products"})
        for index in frame_ids:
            yield SimpleNamespace(frame=SimpleNamespace(frame_index=index, sensing=payloads[index]))

    def scene(index, request):
        return SimpleNamespace(
            frame=SimpleNamespace(
                timestamp_s=index * 0.1,
                tx_names=["visualizer-tx"],
                rx_names=["visualizer-rx"],
                tx_positions=[[0.0, 0.0, 1.0]],
                rx_positions=[[0.0, 0.0, 1.0]],
                rx_orientations=[[0.0, 0.0, 0.0]],
            )
        )

    provider = SimpleNamespace(
        iter_frame_projections=projections,
        load_frame_projection=scene,
        close=lambda: None,
    )
    manifest = SimpleNamespace(frame_ids=range(5), frame_set_id="fixture", generation_id="fixture")
    monkeypatch.setattr(module, "_open_provider", lambda source: (tmp_path, manifest, provider))
    return payloads


def _retain_native_tracker_inputs(payloads):
    for index, sensing in enumerate(payloads):
        sensing["config_resolved"]["track_confirmation_enabled"] = True
        sensing["config_resolved"]["tracker_type"] = "cartesian3d"
        sensing["config"] = deepcopy(sensing["config_resolved"])
        sensing["detection_filter_metadata"]["tracker"]["enabled"] = True
        sensing["detections_before_tracking"] = deepcopy(sensing["detections"])
        sensing["detections_before_tracking_metadata"] = {
            "schema_version": 1,
            "stage": "after_angle_refinement_before_tracking",
            "coordinate_frame": "receiver_local",
            "tracking_applied": False,
            "reference_time_s": index * 0.1,
        }
        if index < 2:
            sensing["detections"] = []
        else:
            for detection in sensing["detections"]:
                detection.update(track_id=99, track_x_m=900.0, track_y_m=800.0, track_z_m=700.0)


def test_native_cartesian_tracking_preserves_fusion_observations_and_estimates(
    native_sensor_payloads, tmp_path
):
    from shared.sensing.fusion.core import MonostaticFusion

    baseline = load_sensor(tmp_path, "Radar")
    _retain_native_tracker_inputs(native_sensor_payloads)
    tracked = load_sensor(tmp_path, "Radar")
    assert [len(epoch["observations"]) for epoch in tracked["epochs"]] == [1, 1, 1, 0, 1]
    old_filter = MonostaticFusion([baseline["sensor"]])
    new_filter = MonostaticFusion([tracked["sensor"]])
    for old, new in zip(baseline["epochs"], tracked["epochs"], strict=True):
        assert old["time_s"] == new["time_s"]
        for original, retained in zip(old["observations"], new["observations"], strict=True):
            evidence = deepcopy(retained.native_source)
            assert evidence.pop("detection_collection") == "detections_before_tracking"
            assert evidence == original.native_source
            np.testing.assert_array_equal(retained.vector, original.vector)
            assert retained.components == original.components
        a = old_filter.step(old["time_s"], old["observations"])
        b = new_filter.step(new["time_s"], new["observations"])
        for prior, current in zip(a["tracks"], b["tracks"], strict=True):
            for key in ("track_id", "position_m", "velocity_m_s", "covariance", "confirmed"):
                assert prior[key] == current[key]
    assert all(
        row["detection_collection"] == "detections_before_tracking"
        for row in tracked["source"]["lineage"]
    )


@pytest.mark.parametrize("configuration_flag", [False, True])
def test_fusion_rejects_tracked_results_without_explicit_raw_observations(
    native_sensor_payloads, tmp_path, configuration_flag
):
    for payload in native_sensor_payloads:
        payload["config_resolved"]["track_confirmation_enabled"] = configuration_flag
        payload["config"] = deepcopy(payload["config_resolved"])
        payload["detection_filter_metadata"]["tracker"]["enabled"] = True
    with pytest.raises(ValueError, match="requires explicit detections_before_tracking"):
        load_sensor(tmp_path, "Radar")


@pytest.mark.parametrize("failure", ["missing_metadata", "wrong_epoch", "tracked_inputs"])
def test_fusion_rejects_invalid_retained_observation_contract(
    native_sensor_payloads, tmp_path, failure
):
    _retain_native_tracker_inputs(native_sensor_payloads)
    first = native_sensor_payloads[0]
    if failure == "missing_metadata":
        first.pop("detections_before_tracking_metadata")
    elif failure == "wrong_epoch":
        first["detections_before_tracking_metadata"]["reference_time_s"] = 0.5
    else:
        first["detections_before_tracking"][0]["track_x_m"] = 123.0
    with pytest.raises(ValueError, match="detections_before_tracking"):
        load_sensor(tmp_path, "Radar")
