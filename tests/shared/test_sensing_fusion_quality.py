"""Component quality, confirmation, and covariance behavior with held-out geometry."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2

from shared.sensing.fusion.core import (
    COMPONENTS,
    FusionConfig,
    MonostaticFusion,
    Observation,
    Sensor,
    measurement_model,
)
from shared.sensing.fusion.quality import apply_quality


def sensor(name="a", position=(0.0, 0.0, 0.0)):
    return Sensor(name, position, (0.0, 0.0, 0.0))


def obs(s, p=(5.0, 0.0, 1.0), v=(0.0, 0.0, 0.0), t=0.0, identity="0"):
    z, _ = measurement_model(np.r_[p, v], s)
    return Observation(s.sensor_id, identity, t, *z)


def test_spatial_only_birth_preserves_unobserved_velocity():
    s = sensor()
    o = replace(obs(s), closing_velocity_m_s=None, components=COMPONENTS[:3])
    tr = MonostaticFusion([s]).step(0.0, [o])["tracks"][0]
    np.testing.assert_array_equal(tr["velocity_m_s"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.array(tr["covariance"])[3:, 3:], np.eye(3) * 25)
    assert tr["hit_epochs"] == 1


def test_velocity_only_cannot_birth_but_can_update_at_pole():
    s = sensor()
    o = Observation("a", "0", 0.0, None, None, None, 1.0, components=(COMPONENTS[3],))
    result = MonostaticFusion([s]).step(0.0, [o])
    assert not result["tracks"] and result["unsupported_births"]
    predicted, jac = measurement_model(np.array([0.0, 0.0, 5.0, 0.0, 0.0, -1.0]), s, o.components)
    np.testing.assert_allclose(predicted, [1.0])
    np.testing.assert_allclose(jac[0, 3:], [0.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "cov",
    [np.eye(3), np.diag([1, 1, 1, -1]), np.ones((4, 4)), np.eye(4) * np.nan, [[1, 2], [0, 1]]],
)
def test_invalid_covariance_strict(cov):
    with pytest.raises(ValueError, match="covariance"):
        replace(obs(sensor()), covariance=cov)


def test_reduced_gate_preserves_probability():
    cfg = FusionConfig()
    for k in (1, 2, 3, 4):
        assert chi2.cdf(cfg.gate(k), k) == pytest.approx(chi2.cdf(cfg.innovation_gate, 4))
    assert FusionConfig(gate_mode="legacy_numeric").gate(2) == 25


@pytest.mark.parametrize(
    "bin_index,omitted", [(61, False), (62, True), (63, True), (65, True), (66, True), (67, False)]
)
def test_dc_policy_uses_bins_not_target_truth_or_speed(bin_index, omitted):
    o = replace(
        obs(sensor(), v=(0.001, 0.0, 0.0)),
        native_source={
            "dc_exclusion_bin_count": 3,
            "fft_size_doppler": 128,
            "doppler_bin": bin_index,
            "dc_exclusion_convention": "fftshift_centered_total_width",
        },
    )
    changed = apply_quality(o, FusionConfig())
    assert (COMPONENTS[3] not in changed.components) == omitted
    assert changed.has_position
    assert apply_quality(o, FusionConfig(quality_policy="legacy")).components == COMPONENTS


def test_rolling_sparse_hits_do_not_confirm_legacy_does():
    s = sensor()
    rolling = MonostaticFusion([s])
    legacy = MonostaticFusion([s], FusionConfig(confirmation_mode="legacy_cumulative"))
    for i in range(9):
        reports = [obs(s, t=i)] if i in (0, 4, 8) else []
        r = rolling.step(i, reports)
        historical = legacy.step(i, reports)
    assert not r["tracks"][0]["confirmed"] and historical["tracks"][0]["confirmed"]


def test_confirmation_latches_and_missing_sources_are_distinct():
    s = sensor()
    f = MonostaticFusion([s])
    for i in range(3):
        result = f.step(i, [obs(s, t=i)])
    assert result["tracks"][0]["confirmed"]
    for i in range(3, 6):
        result = f.step(i, [], available_sensors=[])
        assert result["tracks"][0]["confirmed"] and result["tracks"][0]["predicted_only"]
        assert result["source_status"] == {"a": "missing"}
    assert f.step(6, [])["retired_track_ids"] == [1]
    assert f.step(7, [obs(s, t=7)])["tracks"][0]["track_id"] == 2


def test_optional_corroboration_and_once_per_epoch_hits():
    a, b = sensor(), sensor("b", (0.0, -5.0, 0.0))
    f = MonostaticFusion([a, b], FusionConfig(minimum_recent_sensors=2))
    for i in range(3):
        tr = f.step(i, [obs(a, t=i)])["tracks"][0]
    assert not tr["confirmed"]
    tr = f.step(3, [obs(a, t=3), obs(b, t=3)])["tracks"][0]
    assert tr["confirmed"] and tr["hit_epochs"] == 4
    assert tr["current_sensor_support"] == ["a", "b"]


def test_unequal_quality_downweights_biased_sensor():
    a, b = sensor(), sensor("b", (-1.0, 0.0, 0.0))
    ra, rb = obs(a, p=(5, 0, 0), v=(1, 0, 0)), obs(b, p=(5, 0, 0), v=(1, 0, 0))
    rb = replace(rb, closing_velocity_m_s=-2.0)
    noisy = replace(b, measurement_covariance=tuple(map(tuple, np.diag([0.04, 0.01, 0.01, 100.0]))))
    cfg = FusionConfig(innovation_gate=1000.0)
    tr = MonostaticFusion([a, noisy], cfg).step(0.0, [ra, rb])["tracks"][0]
    assert abs(tr["velocity_m_s"][0] - 1.0) < 0.01
    assert np.linalg.eigvalsh(tr["covariance"]).min() > 0


def test_three_sensors_noise_ensemble_and_parallel_uncertainty():
    rng = np.random.default_rng(872)
    sensors = [sensor(), sensor("b", (-1.0, 0.0, 0.0)), sensor("c", (-2.0, 0.01, 0.0))]
    errors = []
    for _ in range(40):
        reports = []
        for s in sensors:
            o = obs(s, p=(8.0, 0.0, 0.0), v=(1.0, 2.0, 0.0))
            reports.append(
                replace(o, closing_velocity_m_s=o.closing_velocity_m_s + rng.normal(0, 0.1))
            )
        tr = MonostaticFusion(sensors).step(0.0, reports)["tracks"][0]
        assert tr["covariance"][4][4] > 20
        errors.append(tr["velocity_m_s"][0] - 1.0)
    assert np.sqrt(np.mean(np.square(errors))) < 0.1


def test_close_resolved_targets_survive_measurement_consistency_birth_gate():
    s = sensor()
    cfg = FusionConfig(range_std_m=0.01, azimuth_std_deg=0.1, elevation_std_deg=0.1)
    reports = [obs(s, p=(5.0, 0.0, 0.0), identity="a"), obs(s, p=(5.0, 0.3, 0.0), identity="b")]
    result = MonostaticFusion([s], cfg).step(0.0, reports)
    assert len(result["tracks"]) == 2
    assert not result["suppressed_births"]


def test_crossing_targets_with_opposite_radial_motion_and_maneuver():
    sensors = [sensor(), sensor("b", (5.0, -5.0, 0.0)), sensor("c", (10.0, 0.0, 0.0))]
    config = FusionConfig(position_gate_m=1.0, birth_exclusion_m=0.2)
    processor = MonostaticFusion(sensors, config)
    ids = None
    for i in range(31):
        t = i * 0.1
        reports = []
        for k, (p, v) in enumerate(
            [((4.0 + t, 0.1, 1.0), (1.0, 0.0, 0.0)), ((7.0 - t, -0.1, 1.0), (-1.0, 0.0, 0.0))]
        ):
            for s in sensors:
                if s.sensor_id == "c" and 10 <= i < 15:
                    continue
                reports.append(obs(s, p=p, v=v, t=t, identity=str(k)))
        epoch = processor.step(t, reports)
        assert len(epoch["tracks"]) == 2
        current = [
            tr["track_id"] for tr in sorted(epoch["tracks"], key=lambda tr: tr["velocity_m_s"][0])
        ]
        ids = ids or current
        assert current == ids
        for track in epoch["tracks"]:
            assert len({source["sensor_id"] for source in track["sources"]}) == len(
                track["sources"]
            )
    assert all(t["confirmed"] for t in epoch["tracks"])


def test_sensor_relabeling_preserves_physical_associations_but_not_exact_ekf_state():
    a, b = sensor(), sensor("b", (5.0, -5.0, 0.0))
    reports = [
        obs(a, p=(5.2, 0.2, 1.1), v=(0.7, 0.4, 0.05)),
        obs(b, p=(4.9, -0.1, 0.95), v=(0.7, 0.4, 0.05)),
    ]
    first = MonostaticFusion([a, b]).step(0.0, reports)
    renamed = MonostaticFusion([replace(a, sensor_id="z"), replace(b, sensor_id="a")]).step(
        0.0, [replace(reports[0], sensor_id="z"), replace(reports[1], sensor_id="a")]
    )
    assert len(first["tracks"]) == len(renamed["tracks"]) == 1
    assert len(first["tracks"][0]["sources"]) == len(renamed["tracks"][0]["sources"]) == 2
    assert (
        np.linalg.norm(
            np.subtract(first["tracks"][0]["position_m"], renamed["tracks"][0]["position_m"])
        )
        < 0.05
    )


def test_all_components_omitted_are_recorded_without_zero_constraints():
    s = sensor()
    observation = Observation(
        "a",
        "unusable",
        0.0,
        None,
        None,
        None,
        0.0,
        components=(COMPONENTS[3],),
        native_source={
            "dc_exclusion_bin_count": 3,
            "fft_size_doppler": 128,
            "doppler_bin": 66,
            "dc_exclusion_convention": "fftshift_centered_total_width",
        },
    )
    filtered = apply_quality(observation, FusionConfig())
    assert not filtered.components
    epoch = MonostaticFusion([s]).step(0.0, [filtered])
    assert epoch["tracks"] == []
    assert epoch["unsupported_births"][0]["reason"] == "no_usable_components"
