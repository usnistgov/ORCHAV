"""Standard denominators, missing evidence and association cannot hide failures."""

import json
from copy import deepcopy

import pytest

from shared.sensing.uav_evaluation import UAVProfile, evaluate_events, events_from_result


def target(x=0, name="a", z=0):
    return {"target_id": name, "position_m": [x, 0, z], "velocity_m_s": [1, 0, 0]}


def event(i, truth, outputs, run="a", **extra):
    return {
        "run_id": run,
        "event_id": str(i),
        "time_s": float(i),
        "truth": truth,
        "outputs": outputs,
        **extra,
    }


def value(report, metric):
    return report["metrics"][metric]["value"]


def test_standard_denominators_are_macro_and_empty_outputs_are_excluded_from_type2():
    rows = [
        event(0, [target()], []),
        event(1, [target(), target(100, "b")], [target(), target(200)]),
        event(2, [], []),
        event(3, [], [target(), target(100)]),
    ]
    r = evaluate_events(rows)
    assert value(r, "miss_probability") == 0.75  # (1 + 1/2) / 2, not 2/3
    assert value(r, "false_alarm_type_1") == 0.5  # one occupied negative epoch, not two outputs
    assert (
        value(r, "false_alarm_type_2") == 0.5
    )  # empty positive epoch is not a zero false alarm rate
    assert r["counts"]["tp"] == 1 and r["counts"]["fp"] == 3 and r["counts"]["fn"] == 2
    assert r["counts"]["matched_coverage"] == pytest.approx(1 / 3)


def test_undefined_is_not_zero_and_missing_truth_is_not_an_empty_control():
    r = evaluate_events([event(0, [target()], []), event(1, None, [target()])])
    assert value(r, "miss_probability") == 1
    for k in ("false_alarm_type_1", "false_alarm_type_2", "horizontal_p90_m"):
        assert value(r, k) is None
    assert r["skipped"] == {"truth_unavailable_or_unaligned": 1}


def test_one_assignment_drives_counts_and_all_errors():
    outputs = [{"position_m": [3, 4, 12], "velocity_m_s": [1, 4, 3]}, {"position_m": [0, 0, 14]}]
    r = evaluate_events([event(0, [target()], outputs)], UAVProfile(position_gate_m=13.5))
    assert r["counts"]["tp"] == 1 and r["counts"]["fp"] == 1
    assert value(r, "horizontal_p90_m") == 5 and value(r, "vertical_p90_m") == 12
    assert value(r, "vector_velocity_p90_m_s") == 5
    assert value(r, "radial_velocity_p90_m_s") is None


def test_roi_and_bad_world_positions_are_explicit():
    config = UAVProfile(roi_min_m=(-10, -10, -10), roi_max_m=(10, 10, 10))
    r = evaluate_events(
        [
            event(0, [target(10), target(50, "b")], [target(10), target(50)]),
            event(1, [target()], [{"position_m": None}]),
        ],
        config,
    )
    assert r["counts"]["tp"] == 1 and r["counts"]["fp"] == r["counts"]["fn"] == 0
    assert r["skipped"] == {"world_position_unavailable": 1}


def test_bootstrap_resamples_declared_runs_and_is_reproducible():
    rows = [
        event(i, [target()], [] if run == "b" else [target()], run, independent_run_group=run)
        for run in ("a", "b")
        for i in range(4)
    ]
    r = evaluate_events(rows, UAVProfile(bootstrap_samples=100))
    assert r == evaluate_events(rows, UAVProfile(bootstrap_samples=100))
    assert r["metrics"]["miss_probability"]["run_bootstrap_95_interval"]["bounds"] == [0, 1]
    for row in rows:
        row.pop("independent_run_group")
    assert evaluate_events(rows)["metrics"]["miss_probability"]["run_bootstrap_95_interval"] is None


def test_confirmation_coasting_losses_and_identity_switches_remain_visible():
    rows = [
        event(i, [target()], o)
        for i, o in enumerate(
            (
                [],
                [dict(target(), track_id=1)],
                [dict(target(), track_id=1, predicted_only=True)],
                [],
                [dict(target(), track_id=2)],
            )
        )
    ]
    r = evaluate_events(rows)["targets"][0]
    assert r["first_match_delay_s"] == 1 and r["matched_prediction_only_epochs"] == 1
    assert r["losses"] == 1 and r["reacquisitions"] == 1 and r["identity_switches"] == 1


def test_saved_result_adapter_uses_world_pose_and_positive_closing_velocity():
    track = dict(target(0, z=10), track_id=4, confirmed=True, predicted_only=True)
    r = {
        "schema_version": 1,
        "histories": {"fused": [{"time_s": 0.0, "tracks": [track]}]},
        "sources": {"Port": {"sensor": {"position_m": [-10, 0, 10]}}},
        "observations": {
            "Port": [
                {
                    "time_s": 0.0,
                    "detections": [
                        {
                            "world_position_m": [0, 0, 10],
                            "closing_velocity_m_s": -1.0,
                            "components": [
                                "range_m",
                                "azimuth_rad",
                                "elevation_rad",
                                "closing_velocity_m_s",
                            ],
                        }
                    ],
                }
            ]
        },
        "truth_for_evaluation": [{"time_s": 0.0, "available": True, "targets": [target(0, z=10)]}],
    }
    original = deepcopy(r)
    e = events_from_result(r, "trial", "detections:Port")
    assert value(evaluate_events(e), "radial_velocity_p90_m_s") == 0
    assert len(events_from_result(r, "trial", "tracks:fused")[0]["outputs"]) == 1
    assert not events_from_result(r, "trial", "updated_tracks:fused")[0]["outputs"]
    r["observations"]["Port"][0]["detections"][0]["components"].remove("closing_velocity_m_s")
    assert (
        value(
            evaluate_events(events_from_result(r, "trial", "detections:Port")),
            "radial_velocity_p90_m_s",
        )
        is None
    )
    assert original["truth_for_evaluation"] == r["truth_for_evaluation"]


@pytest.mark.parametrize(
    "rows",
    [
        [event(0, [], []), event(0, [], [])],
        [event(2, [], []), event(1, [], [])],
        [event(0, [target(), target()], [])],
    ],
)
def test_malformed_epoch_identity_rejected(rows):
    with pytest.raises(ValueError):
        evaluate_events(rows)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_gate_m": 0},
        {"roi_min_m": (0, 0, 0)},
        {"roi_min_m": (0, 0, 0), "roi_max_m": (0, 1, 1)},
        {"bootstrap_samples": 1.5},
        {"bootstrap_seed": -1},
    ],
)
def test_invalid_profile(kwargs):
    with pytest.raises(ValueError):
        UAVProfile(**kwargs)


def test_legacy_reader_and_atomic_report_failure(tmp_path, monkeypatch):
    from shared.sensing import uav_evaluation as module

    result = {
        "schema_version": 1,
        "sources": {},
        "histories": {"fused": [{"time_s": 0.0, "tracks": []}]},
        "truth_for_evaluation": [{"time_s": 0.0, "targets": []}],
    }
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(result))
    original = source.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": module.PROFILE,
                "evaluation": {},
                "cohorts": ["tracks:fused"],
                "trials": [{"run_id": "legacy", "result": "legacy.json"}],
            }
        )
    )

    def failed_report(*args):
        raise RuntimeError("deliberate late serialization failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "write_report", failed_report)
        with pytest.raises(RuntimeError, match="deliberate late"):
            module.evaluate_manifest(manifest, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".uav-evaluation-*"))
    assert source.read_bytes() == original
    report = module.evaluate_manifest(manifest, tmp_path / "report")
    assert report["sources"][0]["fusion_run_id"].startswith("legacy:")
    assert value(report["cohorts"]["tracks:fused"], "false_alarm_type_1") == 0
