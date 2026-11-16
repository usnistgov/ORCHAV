"""Select physical velocity references from model evidence, never detector fit."""

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.evaluation_truth import ROOT_REFERENCE, extract_evaluation_truth
from tests.shared.test_sensing_gt import PAIR_ID, _active, _payload


def animated_frame():
    target = _active("walker", path_m=10, velocity_m_s=0.75)
    target.update(
        position_m=[5, 0, 0],
        animated_asset={
            "binding_id": "walk",
            "asset_revision_sha256": "revision",
            "root_velocity_m_s": [1, 0, 0],
        },
    )
    sensing = _payload(targets=[target])
    sensing["config_resolved"]["acquisition"] = {"sample_source": "coherent_one_solve"}
    sensing["temporal"]["acquisition"].update(tx_sensor_id="tx", rx_sensor_id="rx")
    sensing["pair_axis"] = [
        dict(
            pair_index=0, frame_pair_index=1, pair_id=PAIR_ID, tx_sensor_id="tx", rx_sensor_id="rx"
        )
    ]
    sensing["target_catalog"] = {
        "targets": [
            {
                "target_id": "walker",
                "animated_asset": {
                    "binding_id": "walk",
                    "asset_revision_sha256": "revision",
                    "rt_velocity_source": "piecewise_linear_root_translation",
                    "coherent_geometry_policy": "frozen_at_solve_epoch",
                },
            }
        ]
    }
    return dict(
        sensing=sensing,
        tx_positions=[[100, 0, 0], [0, 0, 0]],
        rx_positions=[[100, 0, 0], [0, 0, 0]],
    )


def test_root_reference_uses_exact_pair_positive_closing_and_preserves_source():
    frame = animated_frame()
    before = deepcopy(frame)
    entry = extract_evaluation_truth(frame["sensing"], frame)
    assert entry["target_speeds"] == [-1]
    assert entry["recorded_target_speeds"] == [0.75]
    assert entry["target_velocity_references"] == [ROOT_REFERENCE]
    assert frame == before
    frame["sensing"]["truth"]["targets"][0]["animated_asset"]["root_velocity_m_s"] = [-1, 0, 0]
    assert extract_evaluation_truth(frame["sensing"], frame)["target_speeds"] == [1]


def test_bistatic_velocity_is_half_path_closing_rate_even_for_full_path_range():
    frame = animated_frame()
    frame["sensing"]["config_resolved"]["range_mode"] = "bistatic"
    frame["rx_positions"][1] = [5, 5, 0]
    result = extract_evaluation_truth(frame["sensing"], frame)
    assert result["target_ranges"] == [10]
    assert result["target_speeds"] == [-0.5]


@pytest.mark.parametrize(
    "change", ["binding", "revision", "missing_root", "nan_root", "pair", "pose"]
)
def test_invalid_projection_evidence_is_unavailable_not_a_fallback(change):
    frame = animated_frame()
    state = frame["sensing"]["truth"]["targets"][0]["animated_asset"]
    if change == "binding":
        state["binding_id"] = "different"
    elif change == "revision":
        state["asset_revision_sha256"] = "different"
    elif change == "missing_root":
        state.pop("root_velocity_m_s")
    elif change == "nan_root":
        state["root_velocity_m_s"] = [np.nan, 0, 0]
    elif change == "pair":
        frame["sensing"]["pair_axis"][0]["tx_sensor_id"] = "different"
    else:
        frame.pop("tx_positions")
    result = extract_evaluation_truth(frame["sensing"], frame)
    assert result["target_speeds"] == [None]
    assert result["target_velocity_valid"] == [False]
    assert result["target_velocity_unavailable_reasons"][0]
    assert result["recorded_target_speeds"] == [0.75]


@pytest.mark.parametrize("change", ["sample_source", "velocity_source", "geometry_policy", "rigid"])
def test_other_motion_models_keep_the_recorded_observable(change):
    frame = animated_frame()
    sensing = frame["sensing"]
    declaration = sensing["target_catalog"]["targets"][0]["animated_asset"]
    if change == "sample_source":
        sensing["config_resolved"]["acquisition"]["sample_source"] = "multi_solve"
    elif change == "velocity_source":
        declaration["rt_velocity_source"] = "articulated"
    elif change == "geometry_policy":
        declaration["coherent_geometry_policy"] = "changing_pose"
    else:
        sensing.pop("target_catalog")
    assert extract_evaluation_truth(sensing, frame)["target_speeds"] == [0.75]
