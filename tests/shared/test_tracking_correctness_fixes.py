"""Independent exclusion, lifecycle, optional-dependency, and stress regressions."""

import builtins
import itertools
import time

import numpy as np
import pytest

from shared.sensing import tracker as t


def oracle(weights, miss):
    ids = list(weights)
    total = 0.0
    marginal = {i: {None: 0.0, **{j: 0.0 for j in weights[i]}} for i in ids}
    for event in itertools.product(*[[None, *weights[i]] for i in ids]):
        observed = [d for d in event if d is not None]
        if len(set(observed)) != len(observed):
            continue
        mass = np.prod([miss if d is None else weights[i][d] for i, d in zip(ids, event)])
        total += mass
        for i, d in zip(ids, event):
            marginal[i][d] += mass
    return {i: {d: w / total for d, w in x.items()} for i, x in marginal.items()}


@pytest.mark.parametrize("seed", range(20))
def test_joint_matches_independent_product_oracle(seed):
    rng = np.random.default_rng(seed)
    w = {
        i: {j: float(np.exp(rng.normal())) for j in range(4) if rng.random() > 0.3}
        for i in range(3)
    }
    got, _ = t._joint_association_marginals(
        {i: {j: np.log(x) for j, x in row.items()} for i, row in w.items()}, np.log(0.1)
    )
    ref = oracle(w, 0.1)
    for i in w:
        for j in ref[i]:
            assert got[i][j] == pytest.approx(ref[i][j], abs=2e-14)
    for j in range(4):
        assert sum(row.get(j, 0.0) for row in got.values()) <= 1.0 + 1e-13


def test_extreme_log_weights_and_permutation():
    w = {10: {1: -1000.0, 2: 1000.0}, 20: {1: -1100.0, 2: 1001.0}, 30: {9: 2.0}}
    a, _ = t._joint_association_marginals(w, -100.0)
    b, _ = t._joint_association_marginals(dict(reversed(list(w.items()))), -100.0)
    for i in w:
        assert sum(a[i].values()) == pytest.approx(1.0, abs=1e-12)
        for j in a[i]:
            assert np.isfinite(a[i][j])
            assert a[i][j] == pytest.approx(b[i][j])
    assert sum(row.get(2, 0.0) for row in a.values()) <= 1.0 + 1e-12


def test_dense_ten_by_twenty_budget_is_explicit_and_bounded():
    start = time.perf_counter()
    with pytest.raises(t.JointAssociationLimitError, match="max_joint_nodes=1000"):
        t._joint_association_marginals(
            {i: dict.fromkeys(range(20), 0.0) for i in range(10)}, -2.0, max_nodes=1000
        )
    assert time.perf_counter() - start < 5.0


def det(r, v=0.0):
    return dict(range_m=float(r), velocity_m_s=float(v), x_m=float(r), y_m=0.0)


def test_budget_failure_can_retry_timestamp_without_mutation():
    tracker = t.JPDAFTracker(t.TrackerConfig(m=1, n=1, range_gate_m=5), max_joint_nodes=1)
    tracker.filter_detections([det(10)], update_time_s=0)
    x, p = tracker._tracks[1].x.copy(), tracker._tracks[1].P.copy()
    with pytest.raises(t.JointAssociationLimitError):
        tracker.filter_detections([det(10)], update_time_s=1)
    np.testing.assert_array_equal(tracker._tracks[1].x, x)
    np.testing.assert_array_equal(tracker._tracks[1].P, p)
    assert tracker._last_update_time_s == 0
    tracker._max_joint_nodes = 100
    tracker.filter_detections([det(10)], update_time_s=1)
    tracker.filter_detections([det(10)], update_time_s=2)
    assert tracker._last_update_time_s == 2


@pytest.mark.parametrize(
    "cls", [t.MofNTracker, t.KalmanTracker, t.CartesianKalmanTracker, t.JPDAFTracker]
)
def test_confirmation_is_rolling_not_cumulative(cls):
    tracker = cls(t.TrackerConfig(m=2, n=3, max_missed_frames=5, output_tentative=True))
    tracker.filter_detections([det(10)], update_time_s=0)
    tracker.filter_detections([], update_time_s=1)
    tracker.filter_detections([], update_time_s=2)
    out, _ = tracker.filter_detections([det(10)], update_time_s=3)
    assert len(out) == 1
    assert not out[0]["track_confirmed"]
    out, _ = tracker.filter_detections([det(10)], update_time_s=4)
    assert out[0]["track_confirmed"]


def test_persistent_resolvable_in_gate_birth_and_no_blind_duplicate():
    cfg = t.TrackerConfig(m=1, n=1, range_gate_m=3, velocity_gate_m_s=1, max_missed_frames=5)
    tracker = t.JPDAFTracker(
        cfg, measurement_noise_range=0.01, acceleration_psd_m2_s3=0, gate_threshold=1000
    )
    tracker.filter_detections([det(10)], update_time_s=0)
    tracker.filter_detections([det(10), det(10.5)], update_time_s=0.1)
    assert len(tracker._tracks) == 1
    tracker.filter_detections([det(10), det(10.5)], update_time_s=0.2)
    assert len(tracker._tracks) == 1
    tracker.filter_detections([det(10), det(10.5)], update_time_s=0.3)
    assert len(tracker._tracks) == 2
    duplicate = t.JPDAFTracker(cfg, measurement_noise_range=0.01, acceleration_psd_m2_s3=0)
    duplicate.filter_detections([det(10)], update_time_s=0)
    for k in range(1, 20):
        duplicate.filter_detections([det(10), det(10.01)], update_time_s=k * 0.1)
    assert len(duplicate._tracks) == 1


def test_retirement_releases_detection_for_birth():
    tracker = t.JPDAFTracker(t.TrackerConfig(m=1, n=1, max_missed_frames=0), pd=0.001, beta_fa=100)
    tracker.filter_detections([det(10)], update_time_s=0)
    out, _ = tracker.filter_detections([det(10)], update_time_s=1)
    assert [x["track_id"] for x in out] == [2]


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (4, 7), (7, 4), (8, 8), (0, 4)])
def test_numpy_assignment_matches_scipy(shape):
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(12)
    for _ in range(20):
        a = rng.normal(size=shape)
        r, c = t._linear_sum_assignment_numpy(a)
        rr, cc = scipy.linear_sum_assignment(a)
        assert a[r, c].sum() == pytest.approx(a[rr, cc].sum())


def test_no_scipy_keeps_optimal_assignment(monkeypatch):
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("deliberate optional-dependency test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    r, c = t._linear_sum_assignment(np.array([[1.0, 2.0], [2.0, 100.0]]))
    assert list(zip(r, c)) == [(0, 1), (1, 0)]


def test_long_crossing_clutter_missed_returns_covariance_and_cardinality():
    rng = np.random.default_rng(84)
    tracker = t.JPDAFTracker(
        t.TrackerConfig(
            m=2,
            n=4,
            max_missed_frames=8,
            range_gate_m=2,
            velocity_gate_m_s=1,
            output_tentative=True,
        ),
        measurement_noise_range=0.04,
        measurement_noise_velocity=0.01,
    )
    max_tracks = 0
    for k in range(240):
        ts = k * 0.1
        reports = [
            det(30 + v * (ts - 12) + rng.normal(0, 0.08), -v + rng.normal(0, 0.03))
            for i, v in enumerate([-0.5, 0.5])
            if (k + i) % 17 not in (0, 1)
        ]
        if k % 13 == 0:
            reports.append(det(60 + rng.uniform(0, 10), rng.uniform(-2, 2)))
        out, meta = tracker.filter_detections(reports, update_time_s=ts)
        assert len({x["track_id"] for x in out}) == len(out)
        assert len(out) <= len(reports)
        max_tracks = max(max_tracks, meta["active_tracks"])
        for tr in tracker._tracks.values():
            assert np.isfinite(tr.x).all()
            np.testing.assert_allclose(tr.P, tr.P.T, atol=1e-12)
            assert np.linalg.eigvalsh(tr.P).min() >= -1e-12
    assert max_tracks <= 5


def test_numpy_assignment_exhaustive_binary_three_by_three():
    permutations = list(itertools.permutations(range(3)))
    for values in itertools.product((0.0, 1.0), repeat=9):
        matrix = np.asarray(values).reshape(3, 3)
        rows, cols = t._linear_sum_assignment_numpy(matrix)
        expected = min(sum(matrix[i, p[i]] for i in range(3)) for p in permutations)
        assert matrix[rows, cols].sum() == expected


def test_symmetric_exact_jpda_does_not_promise_no_coalescence():
    tracker = t.JPDAFTracker(
        t.TrackerConfig(m=1, n=1, max_missed_frames=20, range_gate_m=3), acceleration_psd_m2_s3=0
    )
    tracker.filter_detections([det(10), det(10)], update_time_s=0)
    for k in range(1, 12):
        tracker.filter_detections([det(9.8), det(10.2)], update_time_s=k * 0.1)
        np.testing.assert_allclose(tracker._tracks[1].x, tracker._tracks[2].x, atol=1e-14)


def test_ghost_pruning_negative_control_is_explicitly_heuristic():
    # Two real stationary targets with equal speeds can meet containment criteria.
    # Preserve/document the opt-in heuristic rather than claiming provenance.
    frames = [
        {"_frame_index": k, "detections": [dict(det(10), track_id=1), dict(det(20), track_id=2)]}
        for k in range(8)
    ]
    _, meta = t.prune_correlated_ghost_tracks(frames)
    assert meta["ghost_tracks"] == [2]


def test_large_sparse_component_is_iterative_and_budget_failure_transactional():
    tracker = t.JPDAFTracker(
        t.TrackerConfig(m=1, n=1, max_missed_frames=4), max_joint_nodes=100_000
    )
    tracker.filter_detections([det(10, 1) for _ in range(1000)], update_time_s=0)
    prior = {
        i: (tr.x.copy(), tr.P.copy(), list(tr.hit_history)) for i, tr in tracker._tracks.items()
    }
    with pytest.raises(t.JointAssociationLimitError, match="max_joint_nodes=100000"):
        tracker.filter_detections([det(9, 1)], update_time_s=1)
    assert tracker._last_update_time_s == 0
    assert tracker._F is None and tracker._Q is None
    for i, tr in tracker._tracks.items():
        np.testing.assert_array_equal(tr.x, prior[i][0])
        np.testing.assert_array_equal(tr.P, prior[i][1])
        assert list(tr.hit_history) == prior[i][2]
    tracker._max_joint_nodes = 1_000_000
    tracker.filter_detections([det(9, 1)], update_time_s=1)
    assert tracker._last_update_time_s == 1
    tracker.filter_detections([], update_time_s=2)
    assert tracker._last_update_time_s == 2


def test_unexpected_enumeration_exception_rolls_back(monkeypatch):
    tracker = t.JPDAFTracker()
    tracker.filter_detections([det(10, 1)], update_time_s=0)
    prior = tracker._tracks[1].x.copy()
    real = t._joint_association_marginals

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(t, "_joint_association_marginals", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        tracker.filter_detections([det(9, 1)], update_time_s=1)
    assert tracker._last_update_time_s == 0
    np.testing.assert_array_equal(tracker._tracks[1].x, prior)
    monkeypatch.setattr(t, "_joint_association_marginals", real)
    tracker.filter_detections([det(9, 1)], update_time_s=1)


@pytest.mark.parametrize("separation,expected", [(1.0, 1), (0.0, 2)])
def test_original_close_target_birth_depends_on_explicit_policy(separation, expected):
    tracker = t.JPDAFTracker(
        t.TrackerConfig(m=1, n=1, range_gate_m=3, max_missed_frames=5),
        birth_separation_chi2=separation,
    )
    tracker.filter_detections([det(10)], update_time_s=0)
    for k in range(1, 12):
        _, meta = tracker.filter_detections([det(10), det(10.2)], update_time_s=k * 0.1)
    assert len(tracker._tracks) == expected
    assert meta["birth_separation_chi2"] == separation


def test_zero_birth_separation_still_blocks_exact_duplicates_and_transient_clutter():
    tracker = t.JPDAFTracker(
        t.TrackerConfig(m=1, n=1, range_gate_m=3, max_missed_frames=5), birth_separation_chi2=0.0
    )
    tracker.filter_detections([det(10)], update_time_s=0)
    for k in range(1, 12):
        # A transient in-gate clutter report is separated, but never persists.
        observations = [det(10), det(10 if k % 3 else 10.2)]
        tracker.filter_detections(observations, update_time_s=k * 0.1)
    assert len(tracker._tracks) == 1


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), True, "bad"])
def test_birth_separation_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError, match="birth_separation_chi2"):
        t.JPDAFTracker(birth_separation_chi2=bad)


TRACKER_VARIANTS = [
    t.MofNTracker,
    t.KalmanTracker,
    t.CartesianKalmanTracker,
    lambda cfg: t.CartesianKalmanTracker(cfg, include_z=True),
    t.JPDAFTracker,
]


def lifecycle_detection():
    return {**det(10), "z_m": 0.0}


@pytest.mark.parametrize("factory", TRACKER_VARIANTS)
@pytest.mark.parametrize("mode", ["latched", "continuous"])
def test_confirmation_policy_coast_reacquire_retire_and_rebirth(factory, mode):
    tracker = factory(t.TrackerConfig(m=2, n=3, max_missed_frames=3, confirmation_mode=mode))
    observation = lifecycle_detection()
    assert tracker.filter_detections([observation], update_time_s=0)[0] == []
    out, metadata = tracker.filter_detections([observation], update_time_s=1)
    track_id = out[0]["track_id"]
    assert out[0]["track_confirmed"]
    assert metadata["confirmation_mode"] == mode
    for timestamp in [2, 3, 4]:
        out, metadata = tracker.filter_detections([], update_time_s=timestamp)
        assert out == []  # Coasting must never fabricate detection rows.
        assert metadata["active_tracks"] == 1
    assert metadata["confirmed_tracks"] == (1 if mode == "latched" else 0)
    out, _ = tracker.filter_detections([observation], update_time_s=5)
    if mode == "latched":
        assert out[0]["track_id"] == track_id
        assert out[0]["track_confirmed"]
    else:
        assert out == []
    out, _ = tracker.filter_detections([observation], update_time_s=6)
    assert out[0]["track_id"] == track_id
    for timestamp in [7, 8, 9, 10]:
        out, metadata = tracker.filter_detections([], update_time_s=timestamp)
        assert out == []
    assert metadata["active_tracks"] == metadata["confirmed_tracks"] == 0
    assert tracker.filter_detections([observation], update_time_s=11)[0] == []
    out, _ = tracker.filter_detections([observation], update_time_s=12)
    assert out[0]["track_confirmed"]
    assert out[0]["track_id"] != track_id


@pytest.mark.parametrize("factory", TRACKER_VARIANTS)
def test_latched_default_does_not_accumulate_sparse_tentative_hits(factory):
    tracker = factory(t.TrackerConfig(m=2, n=3, max_missed_frames=4, output_tentative=True))
    for timestamp in range(7):
        detections = [lifecycle_detection()] if timestamp % 3 == 0 else []
        out, metadata = tracker.filter_detections(detections, update_time_s=timestamp)
        assert metadata["confirmation_mode"] == "latched"
        assert metadata["confirmed_tracks"] == 0
        assert all(not d["track_confirmed"] for d in out)


@pytest.mark.parametrize("factory", TRACKER_VARIANTS)
def test_latched_m_one_and_minimum_lifetime_remain_independent(factory):
    tracker = factory(t.TrackerConfig(m=1, n=1, max_missed_frames=2, min_track_lifetime=2))
    out, metadata = tracker.filter_detections([lifecycle_detection()], update_time_s=0)
    assert out == []
    assert metadata["confirmed_tracks"] == 1
    assert tracker.filter_detections([], update_time_s=1)[0] == []
    out, _ = tracker.filter_detections([lifecycle_detection()], update_time_s=2)
    assert out[0]["track_confirmed"]


def test_confirmation_policy_rejects_invalid_mode_and_preserves_positionals():
    import inspect

    with pytest.raises(ValueError, match="confirmation_mode"):
        t.TrackerConfig(confirmation_mode="cumulative")
    assert (
        inspect.signature(t.TrackerConfig).parameters["confirmation_mode"].kind
        == inspect.Parameter.KEYWORD_ONLY
    )
    assert t.TrackerConfig(2, 3, 4, 0.6, 0.3).max_missed_frames == 4


@pytest.mark.parametrize("mode", ["latched", "continuous"])
@pytest.mark.parametrize("tracker_type", ["m_of_n", "kalman", "cartesian", "cartesian3d", "jpdaf"])
def test_confirmation_policy_configuration_entrypoints(mode, tracker_type):
    from generator.core.configuration.models import SensingConfig
    from shared.scenarios.sensing_groups import normalize_sensing_groups
    from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor

    flat = normalize_sensing_groups({"tracking": {"confirmation": {"mode": mode}}})
    config = SensingConfig(**flat)
    assert config.track_confirmation_mode == mode
    from generator.core.configuration.loader import _apply_sensing
    from generator.core.configuration.models import SimulationConfig
    from shared.scenarios.model import SensingModel

    assert (
        SensingModel.model_validate(
            {"tracking": {"confirmation": {"mode": mode}}}
        ).track_confirmation_mode
        == mode
    )
    runtime = SimulationConfig()
    _apply_sensing(runtime, {"enabled": True, **flat}, {})
    assert runtime.sensing.track_confirmation_mode == mode
    tracker = SensingPostProcessor._create_tracker(
        PostProcessingConfig(tracker_type=tracker_type, tracker_confirmation_mode=mode)
    )
    assert tracker.filter_detections([], update_time_s=0)[1]["confirmation_mode"] == mode
