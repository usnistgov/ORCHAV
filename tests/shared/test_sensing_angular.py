"""Angular reference conventions and unbiased RD-conditioned diagnostics."""

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.angular import (
    angular_event,
    angular_residuals,
    angular_truth_fields,
    create_angular_summary,
    receiver_local_angles,
)


def test_receiver_orientation_and_translation():
    assert receiver_local_angles([2, 3, 1], [2, 2, 1], [np.pi / 2, 0, 0]) == pytest.approx((0, 0))
    assert receiver_local_angles([1, 0, 1], [0, 0, 0], [0, 0, 0]) == pytest.approx((0, 45))
    # Positive pitch turns local +X toward world -Z.
    assert receiver_local_angles([1, 0, -1], [0, 0, 0], [0, np.pi / 4, 0]) == pytest.approx((0, 0))
    assert receiver_local_angles([1, 0, 1], [0, 0, 0], [0, 0, np.pi / 2]) == pytest.approx((45, 0))


@pytest.mark.parametrize("position", [[0, 0, 0], [0, 0, 1], [np.nan, 0, 1]])
def test_undefined_angles_are_not_invented(position):
    with pytest.raises(ValueError):
        receiver_local_angles(position, [0, 0, 0], [0, 0, 0])
    assert angular_truth_fields({}, position) == {}


def payload():
    return {
        "config_resolved": {
            "range_mode": "monostatic",
            "eval_association_range_gate_m": 0.6,
            "eval_association_velocity_gate_m_s": 0.25,
        },
        "rx_array": {"rows": 1, "columns": 16},
        "temporal": {
            "truth": {"reference_time_s": 1.0},
            "result": {"reference_time_s": 1.0},
            "acquisition": {"pair_id": "pair"},
        },
        "truth": {
            "targets": [
                {
                    "target_id": "cube",
                    "valid": True,
                    "lifecycle": "active",
                    "pair_observables": [
                        {
                            "pair_id": "pair",
                            "angular_reference_frame": "rx_array_local",
                            "receiver_azimuth_deg": 179.0,
                            "receiver_elevation_deg": 3.0,
                            "monostatic_equivalent_range_m": 5.0,
                            "monostatic_equivalent_velocity_m_s": 1.0,
                            "velocity_valid": True,
                        }
                    ],
                }
            ]
        },
        "detections": [
            {"range_m": 5.0, "velocity_m_s": 1.0, "aoa_az_deg": -179.0, "aoa_el_deg": 0.0},
            {"range_m": 12.0, "velocity_m_s": 2.0, "aoa_az_deg": 179.0},
        ],
    }


def test_angles_never_gate_matching_and_unresolved_axis_is_explicit(tmp_path):
    s = payload()
    e = angular_event(s)
    assert e["azimuth_resolved"] and not e["elevation_resolved"]
    r = angular_residuals([e])
    assert r["matches"][0]["azimuth_error_deg"] == 2.0
    assert r["matches"][0]["elevation_error_deg"] == -3.0
    assert r["unmatched_reports"] == 1
    s["detections"][0]["aoa_az_deg"] = 40.0
    assert len(angular_residuals([angular_event(s)])["matches"]) == 1
    assert len(create_angular_summary([e], tmp_path)) == 3


def test_epoch_pair_and_old_recording_must_not_be_guessed():
    s = payload()
    for modify in [
        lambda p: p["temporal"]["truth"].update(reference_time_s=2.0),
        lambda p: p["temporal"]["acquisition"].update(pair_id="other"),
        lambda p: p["truth"]["targets"][0]["pair_observables"][0].pop("angular_reference_frame"),
        lambda p: p["rx_array"].update(columns=1),
    ]:
        copy = deepcopy(s)
        modify(copy)
        assert angular_event(copy) is None
