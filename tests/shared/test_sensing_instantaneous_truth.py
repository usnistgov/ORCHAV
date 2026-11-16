"""Independent physical and persistence checks for explicit velocity references."""

import json
from copy import deepcopy

import numpy as np
import pytest

from shared.frames.json_codec import dumps_frame_json
from shared.sensing.gt import extract_stored_gt_entry
from shared.sensing.signal_processing import compute_gt_entry
from shared.sensing.targets import (
    INSTANTANEOUS_VELOCITY_REFERENCE_POLICY as POLICY,
)
from shared.sensing.targets import (
    build_target_catalog,
    build_target_truth,
    validate_target_truth,
    validate_velocity_reference_evidence,
)

PAIR = "a" * 64


def _entry(**overrides):
    arguments = dict(
        tx_pos=np.zeros(3),
        rx_pos=np.zeros(3),
        target_positions=[np.array([5.0, 0.0, 0.0])],
        range_mode="monostatic",
        target_velocities=np.array([[-1.0, 0.0, 0.0]]),
        target_ids=["Cube"],
        reference_time_s=0.128,
        velocity_reference_policy=POLICY,
        tx_velocity_m_s=np.zeros(3),
        rx_velocity_m_s=np.zeros(3),
        pair_id=PAIR,
    )
    arguments.update(overrides)
    return compute_gt_entry(**arguments)


def _payload(entry):
    evidence = entry.target_velocity_reference_evidence[0]
    return {
        "config_resolved": {"range_mode": "monostatic"},
        "temporal": {
            "acquisition": {"pair_id": PAIR},
            "truth": {"reference_time_s": entry.reference_time_s},
        },
        "truth": {
            "velocity_reference_policy": POLICY,
            "targets": [
                {
                    "target_id": "Cube",
                    "lifecycle": "active",
                    "valid": True,
                    "position_m": evidence["target_position_m"],
                    "pair_observables": [
                        {
                            "pair_id": PAIR,
                            "monostatic_equivalent_range_m": entry.target_ranges[0],
                            "monostatic_equivalent_velocity_m_s": entry.target_speeds[0],
                            "velocity_valid": entry.target_velocity_valid[0],
                            "velocity_derivation_source": entry.target_velocity_sources[0],
                            "velocity_reference_dt_s": entry.target_velocity_reference_dt_s[0],
                            "undefined_reason": entry.target_velocity_unavailable_reasons[0],
                            "velocity_reference_evidence": evidence,
                        }
                    ],
                }
            ],
        },
    }


@pytest.mark.parametrize("range_mode", ["monostatic", "bistatic"])
def test_all_moving_endpoints_match_independent_centered_path_derivative(range_mode):
    tx = np.array([0.3, -0.4, 0.8])
    rx = np.array([6.0, 1.0, 0.2])
    target = np.array([4.0, 3.0, 2.0])
    tx_velocity = np.array([0.3, 0.1, -0.15])
    rx_velocity = np.array([-0.2, 0.5, 0.3])
    target_velocity = np.array([0.7, -0.2, 0.4])

    def path_length(offset):
        position = target + offset * target_velocity
        return np.linalg.norm(position - tx - offset * tx_velocity) + np.linalg.norm(
            position - rx - offset * rx_velocity
        )

    step = 1e-5
    expected_velocity = -(path_length(step) - path_length(-step)) / (4 * step)
    entry = _entry(
        tx_pos=tx,
        rx_pos=rx,
        target_positions=[target],
        target_velocities=target_velocity[None],
        tx_velocity_m_s=tx_velocity,
        rx_velocity_m_s=rx_velocity,
        range_mode=range_mode,
        prev_target_ranges=[99.0],
        dt=0.064,
    )
    assert entry.target_speeds == pytest.approx([expected_velocity], abs=5e-11)
    assert entry.target_ranges == pytest.approx(
        [path_length(0) / (2 if range_mode == "monostatic" else 1)]
    )
    assert entry.target_path_closing_rates_m_s == pytest.approx([2 * expected_velocity])
    assert entry.target_velocity_reference_evidence[0]["instantaneous_eligible"] is True


def test_common_translation_of_every_endpoint_has_zero_doppler():
    velocity = np.array([1.1, -0.8, 0.3])
    entry = _entry(
        tx_pos=np.array([1.0, 0.0, 0.0]),
        rx_pos=np.array([-2.0, 3.0, 0.0]),
        target_velocities=velocity[None],
        tx_velocity_m_s=velocity,
        rx_velocity_m_s=velocity,
    )
    assert entry.target_speeds == [0.0]


def test_stationary_target_and_moving_monostatic_radar_use_relative_velocity():
    entry = _entry(
        target_velocities=np.zeros((1, 3)),
        tx_velocity_m_s=np.array([1.0, 0.0, 0.0]),
        rx_velocity_m_s=np.array([1.0, 0.0, 0.0]),
    )
    assert entry.target_speeds == [1.0]


def test_accelerating_target_selects_current_velocity_and_preserves_legacy_result():
    epoch = 0.128
    duration = 0.064
    omega = 4 * np.pi

    def position(t):
        return 5 - 0.15 * np.sin(omega * t)

    closing_velocity = 0.15 * omega * np.cos(omega * epoch)
    arguments = dict(
        tx_pos=np.array([0.0, 0.0, 0.5]),
        rx_pos=np.array([0.0, 0.0, 0.5]),
        target_positions=[np.array([0.0, position(epoch), 0.5])],
        range_mode="monostatic",
        target_velocities=np.array([[0.0, -closing_velocity, 0.0]]),
        target_ids=["Cube"],
        reference_time_s=epoch,
        prev_target_ranges=[position(epoch - duration)],
        dt=duration,
    )
    legacy = compute_gt_entry(**arguments)
    current = compute_gt_entry(
        **arguments,
        velocity_reference_policy=POLICY,
        tx_velocity_m_s=np.zeros(3),
        rx_velocity_m_s=np.zeros(3),
        pair_id=PAIR,
    )
    interval = (position(epoch - duration) - position(epoch)) / duration
    assert legacy.target_speeds == pytest.approx([interval])
    assert legacy.target_velocity_reference_evidence == []
    assert current.target_speeds == pytest.approx([closing_velocity])
    assert abs(interval - closing_velocity) > 0.7
    assert current.target_velocity_reference_dt_s == [None]
    assert current.target_velocity_reference_evidence[0]["support"] == {
        "kind": "instantaneous",
        "reference_time_s": epoch,
        "start_time_s": epoch,
        "end_time_s": epoch,
    }


@pytest.mark.parametrize("missing", ["tx_velocity_m_s", "rx_velocity_m_s", "target_velocities"])
def test_missing_velocity_retains_explicit_interval_fallback_without_instantaneous_eligibility(
    missing,
):
    entry = _entry(**{missing: None}, prev_target_ranges=[5.064], dt=0.064)
    evidence = entry.target_velocity_reference_evidence[0]
    assert entry.target_speeds == pytest.approx([1.0])
    assert entry.target_velocity_sources == ["exact_epoch_finite_difference"]
    assert evidence["support"] == {
        "kind": "interval_mean",
        "reference_time_s": 0.096,
        "start_time_s": 0.064,
        "end_time_s": 0.128,
    }
    assert evidence["instantaneous_eligible"] is False
    assert evidence["fallback_reason"].startswith("missing_instantaneous_velocity:")
    extracted = extract_stored_gt_entry(_payload(entry))
    assert extracted["target_speeds"] == pytest.approx([1.0])
    assert extracted["target_velocity_instantaneous_eligible"] == [False]


def test_unknown_sensor_velocity_is_not_silently_zero_at_first_epoch():
    entry = _entry(tx_velocity_m_s=None)
    evidence = entry.target_velocity_reference_evidence[0]
    assert entry.target_speeds == [None]
    assert entry.target_velocity_valid == [False]
    assert evidence["tx_velocity_m_s"] is None
    assert evidence["support"] == dict(
        kind="unavailable", reference_time_s=None, start_time_s=None, end_time_s=None
    )


def test_undefined_instantaneous_geometry_can_retain_only_interval_evidence():
    entry = _entry(target_positions=[np.zeros(3)], prev_target_ranges=[0.1], dt=0.1)
    evidence = entry.target_velocity_reference_evidence[0]
    assert entry.target_speeds == pytest.approx([1.0])
    assert evidence["instantaneous_eligible"] is False
    assert evidence["fallback_reason"] == "instantaneous_projection_undefined_geometry"


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({"reference_time_s": None}, "reference_time_s"),
        ({"pair_id": None}, "pair_id"),
        ({"pair_id": "unknown"}, "pair_id"),
        ({"target_ids": None}, "target_ids"),
        ({"target_ids": ["target_0"]}, "semantic target_id"),
        ({"target_ids": ["Cube", "Cube"]}, "one-to-one"),
        ({"tx_velocity_m_s": [np.nan, 0, 0]}, "finite vector"),
        ({"rx_velocity_m_s": [0, 0]}, "finite vector"),
        ({"velocity_reference_policy": "invented"}, "velocity_reference_policy"),
    ],
)
def test_new_policy_rejects_incomplete_identity_and_malformed_state(arguments, match):
    with pytest.raises(ValueError, match=match):
        _entry(**arguments)


def test_serialized_evidence_is_copied_and_extraction_preserves_provenance():
    velocity = np.array([-1.0, 0.0, 0.0])
    entry = _entry(target_velocities=velocity[None])
    evidence = entry.target_velocity_reference_evidence[0]
    velocity[:] = 42
    assert evidence["target_velocity_m_s"] == [-1.0, 0.0, 0.0]
    assert json.loads(dumps_frame_json(evidence)) == evidence
    payload = _payload(entry)
    before = deepcopy(payload)
    extracted = extract_stored_gt_entry(payload)
    assert extracted["target_velocity_instantaneous_eligible"] == [True]
    assert extracted["target_velocity_reference_evidence"] == [evidence]
    extracted["target_velocity_reference_evidence"][0]["tx_velocity_m_s"][0] = 99
    assert payload == before


def test_reference_rows_keep_exact_target_order_without_positional_rebinding():
    entry = _entry(
        target_positions=[np.array([3.0, 0, 0]), np.array([4.0, 0, 0])],
        target_velocities=np.array([[-2.0, 0, 0], [1.0, 0, 0]]),
        target_ids=["bravo", "alpha"],
    )
    assert entry.target_speeds == [2.0, -1.0]
    assert [row["target_id"] for row in entry.target_velocity_reference_evidence] == [
        "bravo",
        "alpha",
    ]


@pytest.mark.parametrize(
    "change", ["pair", "target", "epoch", "support", "eligible", "position", "missing", "policy"]
)
def test_extraction_rejects_rebound_or_missing_reference_evidence(change):
    payload = _payload(_entry())
    observable = payload["truth"]["targets"][0]["pair_observables"][0]
    evidence = observable["velocity_reference_evidence"]
    if change == "pair":
        evidence["pair_id"] = "b" * 64
    elif change == "target":
        evidence["target_id"] = "different"
    elif change == "epoch":
        evidence["evaluation_epoch_s"] = 0.129
    elif change == "support":
        evidence["support"]["reference_time_s"] = 0.1
    elif change == "eligible":
        evidence["instantaneous_eligible"] = False
    elif change == "position":
        evidence["target_position_m"] = [4.9, 0.0, 0.0]
    elif change == "missing":
        observable.pop("velocity_reference_evidence")
    else:
        payload["truth"].pop("velocity_reference_policy")
    with pytest.raises(ValueError, match="velocity_reference_evidence"):
        extract_stored_gt_entry(payload)


def test_evidence_validator_rejects_an_instantaneous_claim_with_missing_sensor_velocity():
    evidence = deepcopy(_entry().target_velocity_reference_evidence[0])
    evidence["rx_velocity_m_s"] = None
    errors = validate_velocity_reference_evidence(evidence, pair_id=PAIR, target_id="Cube")
    assert any("requires all endpoint velocities" in error for error in errors)


def test_interval_duration_must_agree_with_recorded_derivation():
    payload = _payload(_entry(tx_velocity_m_s=None, prev_target_ranges=[5.064], dt=0.064))
    payload["truth"]["targets"][0]["pair_observables"][0]["velocity_reference_dt_s"] = 0.1
    with pytest.raises(ValueError, match="disagrees with interval support"):
        extract_stored_gt_entry(payload)


def test_truth_builder_preserves_policy_and_rejects_missing_evidence():
    entry = _entry()
    target = _payload(entry)["truth"]["targets"][0]
    target.update(orientation_rad=[0.0, 0.0, 0.0], velocity_m_s=[-1.0, 0.0, 0.0])
    target["pair_observables"][0].update(
        bistatic_path_length_m=10.0,
        path_closing_rate_m_s=2.0,
        bistatic_bisector_velocity_m_s=1.0,
    )
    catalog = build_target_catalog(
        [{"target_id": "Cube", "display_name": "Cube", "identity_source": "scenario_actor_name"}]
    )
    truth = build_target_truth(catalog, targets=[target], velocity_reference_policy=POLICY)
    assert truth["velocity_reference_policy"] == POLICY
    assert validate_target_truth(catalog, truth) == []
    truth["targets"][0]["pair_observables"][0].pop("velocity_reference_evidence")
    assert any(
        "velocity_reference_evidence" in error for error in validate_target_truth(catalog, truth)
    )


def test_empty_new_policy_truth_retains_policy_without_fabricating_targets():
    catalog = build_target_catalog([])
    truth = build_target_truth(catalog, targets=[], velocity_reference_policy=POLICY)
    payload = _payload(_entry())
    payload["truth"] = truth
    entry = extract_stored_gt_entry(payload)
    assert entry["target_ids"] == []
    assert entry["target_velocity_reference_evidence"] == []
    assert entry["target_velocity_instantaneous_eligible"] == []
