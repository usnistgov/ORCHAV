"""Scientific conventions and failure boundaries of monostatic fusion."""

from dataclasses import replace

import numpy as np
import pytest

from shared.sensing.fusion.core import (
    FusionConfig,
    MonostaticFusion,
    Observation,
    Sensor,
    gated_assignment,
    measurement_model,
)
from shared.sensing.fusion.figures import evaluate_fusion


def observation(sensor, position, velocity, time=0, identity="report"):
    z, _ = measurement_model(np.r_[position, velocity], sensor)
    return Observation(sensor.sensor_id, identity, time, *z)


@pytest.fixture
def sensors():
    return [
        Sensor("west", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        Sensor("south", (5.0, -5.0, 1.0), (np.pi / 2, 0.0, 0.0)),
    ]


def test_jacobian_matches_finite_differences_with_rotated_sensor():
    sensor = Sensor("tilted", (1.0, -2.0, 3.0), (0.8, -0.2, 0.15))
    state = np.array([6.0, 2.0, 1.0, -0.7, 0.4, 0.05])
    _, jac = measurement_model(state, sensor)
    eps = 1e-5
    numerical = np.column_stack(
        [
            (
                measurement_model(state + eps * np.eye(6)[i], sensor)[0]
                - measurement_model(state - eps * np.eye(6)[i], sensor)[0]
            )
            / (2 * eps)
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(jac, numerical, atol=1e-8)
    report = observation(sensor, state[:3], state[3:])
    np.testing.assert_allclose(report.world_position(sensor), state[:3], atol=1e-12)


def test_positive_closing_convention(sensors):
    report = observation(sensors[0], [5, 0, 1], [-2, 0, 0])
    assert report.closing_velocity_m_s == 2
    state = MonostaticFusion([sensors[0]], FusionConfig(confirmation_epochs=1)).step(0, [report])[
        "tracks"
    ][0]
    assert state["velocity_m_s"][0] < -1.99


def test_two_radars_constrain_complementary_velocity_components(sensors):
    position, velocity = np.array([5.0, 0.0, 1.0]), np.array([0.7, 0.4, 0.0])
    reports = [observation(s, position, velocity) for s in sensors]
    fusion = MonostaticFusion(sensors, FusionConfig(confirmation_epochs=1))
    tracks = fusion.step(0, reports)["tracks"]
    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0]["velocity_m_s"], velocity, atol=0.002)
    assert len(tracks[0]["sources"]) == 2
    for s, report in zip(sensors, reports):
        single = MonostaticFusion([s]).step(0, [report])["tracks"][0]
        assert np.linalg.norm(np.subtract(single["velocity_m_s"], velocity)) > 0.39


def test_two_targets_associate_without_truth_and_confirm_once_per_epoch(sensors):
    processor = MonostaticFusion(sensors)
    for i in range(5):
        t = i * 0.064
        observations = []
        for n, (p, v) in enumerate(
            [([5.0, 0.0, 1.0], [0.7, 0.4, 0]), ([7.0, 3.0, 1.0], [-0.5, -0.2, 0])]
        ):
            for s in sensors:
                observations.append(observation(s, np.array(p) + t * np.array(v), v, t, str(n)))
        tracks = processor.step(t, observations[::-1])["tracks"]
        assert len(tracks) == 2
        assert all(len(tr["sources"]) == 2 for tr in tracks)
        assert all(tr["hit_epochs"] == i + 1 for tr in tracks)
        assert all(tr["confirmed"] == (i >= 2) for tr in tracks)
        assert all(np.linalg.eigvalsh(tr["covariance"]).min() > 0 for tr in tracks)


def test_empty_updates_coast_and_retire(sensors):
    config = FusionConfig(confirmation_epochs=1, max_missed_epochs=1)
    processor = MonostaticFusion(sensors, config)
    reports = [observation(s, [5, 0, 1], [0.7, 0.4, 0]) for s in sensors]
    initial = processor.step(0, reports)["tracks"][0]
    coast = processor.step(0.5, [])["tracks"][0]
    assert coast["predicted_only"] and not coast["sources"]
    np.testing.assert_allclose(
        coast["position_m"],
        np.array(initial["position_m"]) + 0.5 * np.array(initial["velocity_m_s"]),
    )
    retired = processor.step(1, [])
    assert retired["tracks"] == [] and retired["retired_track_ids"] == [1]


def test_parallel_views_do_not_invent_transverse_velocity_information():
    sensors = [
        Sensor("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        Sensor("b", (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ]
    reports = [observation(s, [5, 0, 0], [1, 2, 0]) for s in sensors]
    result = MonostaticFusion(sensors).step(0, reports)["tracks"][0]
    assert abs(result["velocity_m_s"][1]) < 1e-10
    assert result["covariance"][4][4] > 24
    assert result["covariance"][3][3] < 0.01


def test_input_order_deterministic(sensors):
    reports = [observation(s, [5, 0, 1], [1, 0.2, 0]) for s in sensors]
    assert MonostaticFusion(sensors).step(0, reports) == MonostaticFusion(sensors[::-1]).step(
        0, reports[::-1]
    )


def test_neighbouring_peaks_record_suppressed_birth_but_distinct_motion_survives(sensors):
    s = sensors[0]
    processor = MonostaticFusion([s])
    reports = [
        observation(s, [5, 0, 1], [1, 0, 0], identity="0"),
        observation(s, [5.3, 0, 1], [1, 0, 0], identity="1"),
        observation(s, [5.1, 0, 1], [-1, 0, 0], identity="2"),
    ]
    epoch = processor.step(0, reports)
    assert len(epoch["tracks"]) == 2
    assert len(epoch["suppressed_births"]) == 1
    assert epoch["suppressed_births"][0]["detection_id"] == "1"
    assert len(epoch["tracks"][0]["sources"]) == 1


def test_invalid_epochs_and_double_counting_rejected(sensors):
    processor = MonostaticFusion(sensors)
    report = observation(sensors[0], [5, 0, 1], [1, 0, 0])
    for reports in [
        [report, report],
        [replace(report, sensor_id="unknown")],
        [replace(report, time_s=0.1)],
    ]:
        with pytest.raises(ValueError, match="Detections"):
            processor.step(0, reports)
    processor.step(0, [report])
    with pytest.raises(ValueError, match="increasing"):
        processor.step(0, [])


def test_assignment_maximizes_valid_matches():
    # Greedy nearest association would take (0,0), losing the second match.
    assert gated_assignment(np.array([[0.1, 0.2], [0.2, np.inf]]), 1) == [(0, 1), (1, 0)]
    assert gated_assignment(np.array([[np.inf]]), 1) == []


def test_evaluation_counts_unmatched_and_missed(sensors):
    processor = MonostaticFusion(sensors, FusionConfig(confirmation_epochs=1))
    history = [processor.step(0, [observation(sensors[0], [5, 0, 1], [1, 0, 0])])]
    result = {
        "histories": {"fused": history},
        "truth_for_evaluation": [
            {
                "time_s": 0,
                "targets": [
                    {"target_id": "elsewhere", "position_m": [9, 3, 1], "velocity_m_s": [0, 0, 0]}
                ],
            }
        ],
    }
    score = evaluate_fusion(result)["modes"]["fused"]
    assert score["matched_outputs"] == 0
    assert score["missed_target_epochs"] == score["unmatched_confirmed_outputs"] == 1
    assert score["position_rmse_m"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"range_std_m": 0},
        {"azimuth_std_deg": float("nan")},
        {"confirmation_epochs": 0},
        {"max_missed_epochs": -1},
    ],
)
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        FusionConfig(**changes)
