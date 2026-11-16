from __future__ import annotations

import pytest

from shared.sensing.gt import extract_stored_gt_entry, select_gt_channel

PAIR_ID = "a" * 64


def _payload(
    *,
    range_mode: str = "monostatic",
    targets: list[dict] | None = None,
) -> dict:
    return {
        "config_resolved": {"range_mode": range_mode},
        "temporal": {"acquisition": {"pair_id": PAIR_ID}},
        "truth": {"targets": targets or []},
    }


def _active(
    target_id: str,
    *,
    path_m: float,
    velocity_m_s: float | None,
    velocity_valid: bool = True,
) -> dict:
    return {
        "target_id": target_id,
        "lifecycle": "active",
        "valid": True,
        "pair_observables": [
            {
                "pair_id": PAIR_ID,
                "bistatic_path_length_m": path_m,
                "monostatic_equivalent_range_m": 0.5 * path_m,
                "monostatic_equivalent_velocity_m_s": velocity_m_s,
                "velocity_valid": velocity_valid,
                "velocity_derivation_source": (
                    "analytic_velocity_projection" if velocity_valid else "unavailable"
                ),
                "velocity_reference_dt_s": None,
                "undefined_reason": None if velocity_valid else "truth_velocity_unavailable",
            }
        ],
    }


def test_extract_stored_gt_entry_uses_exact_target_and_pair_identity() -> None:
    entry = extract_stored_gt_entry(
        _payload(
            targets=[
                _active("bravo", path_m=16.0, velocity_m_s=-1.0),
                _active("alpha", path_m=10.0, velocity_m_s=2.0),
            ]
        ),
        expected_targets=2,
    )
    assert entry == {
        "target_ids": ["bravo", "alpha"],
        "target_ranges": [8.0, 5.0],
        "target_speeds": [-1.0, 2.0],
        "target_velocity_valid": [True, True],
        "target_velocity_sources": [
            "analytic_velocity_projection",
            "analytic_velocity_projection",
        ],
        "target_velocity_reference_dt_s": [None, None],
        "target_velocity_unavailable_reasons": [None, None],
    }


def test_extract_stored_gt_entry_uses_bistatic_path_length_when_declared() -> None:
    entry = extract_stored_gt_entry(
        _payload(
            range_mode="bistatic",
            targets=[_active("alpha", path_m=10.0, velocity_m_s=2.0)],
        )
    )
    assert entry is not None
    assert entry["target_ranges"] == [10.0]


def test_extract_stored_gt_entry_preserves_empty_and_skips_inactive_truth() -> None:
    inactive = {
        "target_id": "inactive",
        "lifecycle": "inactive",
        "valid": False,
        "pair_observables": [],
    }
    assert extract_stored_gt_entry(_payload(targets=[inactive]), expected_targets=0) == {
        "target_ids": [],
        "target_ranges": [],
        "target_speeds": [],
        "target_velocity_valid": [],
        "target_velocity_sources": [],
        "target_velocity_reference_dt_s": [],
        "target_velocity_unavailable_reasons": [],
    }


def test_extract_stored_gt_entry_preserves_unavailable_velocity_as_null() -> None:
    entry = extract_stored_gt_entry(
        _payload(
            targets=[
                _active(
                    "alpha",
                    path_m=10.0,
                    velocity_m_s=None,
                    velocity_valid=False,
                )
            ]
        )
    )

    assert entry is not None
    assert entry["target_speeds"] == [None]
    assert entry["target_velocity_valid"] == [False]
    assert entry["target_velocity_sources"] == ["unavailable"]
    assert entry["target_velocity_unavailable_reasons"] == ["truth_velocity_unavailable"]


def test_extract_stored_gt_entry_rejects_pair_and_count_mismatch() -> None:
    wrong_pair = _active("alpha", path_m=10.0, velocity_m_s=2.0)
    wrong_pair["pair_observables"][0]["pair_id"] = "b" * 64
    with pytest.raises(ValueError, match="exactly one observable"):
        extract_stored_gt_entry(_payload(targets=[wrong_pair]))

    with pytest.raises(ValueError, match="active-target count"):
        extract_stored_gt_entry(
            _payload(targets=[_active("alpha", path_m=10.0, velocity_m_s=2.0)]),
            expected_targets=2,
        )


def test_extract_stored_gt_entry_rejects_non_finite_values() -> None:
    target = _active("alpha", path_m=10.0, velocity_m_s=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        extract_stored_gt_entry(_payload(targets=[target]))


@pytest.mark.parametrize("range_mode", ["mono", "MONOSTATIC", "monopoly", ""])
def test_extract_stored_gt_entry_requires_exact_range_mode(range_mode) -> None:
    with pytest.raises(ValueError, match="exactly 'monostatic' or 'bistatic'"):
        extract_stored_gt_entry(
            _payload(
                range_mode=range_mode,
                targets=[_active("alpha", path_m=10.0, velocity_m_s=2.0)],
            )
        )


@pytest.mark.parametrize("bad_range", [-1.0, True])
def test_extract_stored_gt_entry_rejects_nonphysical_range(bad_range) -> None:
    target = _active("alpha", path_m=10.0, velocity_m_s=2.0)
    target["pair_observables"][0]["monostatic_equivalent_range_m"] = bad_range

    with pytest.raises(ValueError, match="finite number|non-negative"):
        extract_stored_gt_entry(_payload(targets=[target]))


def test_extract_stored_gt_entry_never_falls_back_to_outer_config() -> None:
    payload = _payload(targets=[_active("alpha", path_m=10.0, velocity_m_s=2.0)])
    payload["config"] = payload.pop("config_resolved")

    with pytest.raises(ValueError, match="config_resolved.*not a fallback"):
        extract_stored_gt_entry(payload)


def test_select_gt_channel_accepts_only_canonical_truth() -> None:
    assert select_gt_channel({"truth": {}, "gt_sensing": {}}) == "truth"
    assert select_gt_channel({"gt": {}, "gt_sensing": {}}) is None


def test_select_gt_channel_returns_none_without_truth() -> None:
    assert select_gt_channel({"detections": []}) is None
