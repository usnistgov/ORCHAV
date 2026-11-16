"""Identity, lifecycle, and reorder tests for sensing targets."""

from __future__ import annotations

from copy import deepcopy

import pytest

from shared.sensing.targets import (
    build_target_catalog,
    build_target_state,
    build_target_truth,
    target_rows_by_id,
    validate_target_catalog,
    validate_target_lifecycle_sequence,
    validate_target_state,
)

PAIR_ID = "a" * 64


def _catalog() -> dict[str, object]:
    return build_target_catalog(
        [
            {
                "target_id": "walker-B",
                "display_name": "Walker B",
                "identity_source": "scenario_actor_name",
                "asset_ref": "walker_b.ply",
            },
            {
                "target_id": "walker_A",
                "display_name": "Walker A",
                "identity_source": "scenario_actor_name",
                "asset_ref": "walker_a.ply",
            },
        ]
    )


def _active(target_id: str, x: float) -> dict[str, object]:
    return {
        "target_id": target_id,
        "lifecycle": "active",
        "valid": True,
        "position_m": [x, 0.0, 1.0],
        "orientation_rad": [0.0, 0.0, 0.0],
        "velocity_m_s": [1.0, 0.0, 0.0],
    }


def _inactive(target_id: str, lifecycle: str = "inactive") -> dict[str, object]:
    return {
        "target_id": target_id,
        "lifecycle": lifecycle,
        "valid": False,
        "position_m": None,
        "orientation_rad": None,
        "velocity_m_s": None,
    }


def _truth_active(target_id: str, x: float, range_m: float) -> dict[str, object]:
    row = _active(target_id, x)
    row["pair_observables"] = [
        {
            "pair_id": PAIR_ID,
            "bistatic_path_length_m": 2.0 * range_m,
            "monostatic_equivalent_range_m": range_m,
            "path_closing_rate_m_s": -2.0,
            "monostatic_equivalent_velocity_m_s": -1.0,
            "bistatic_bisector_velocity_m_s": -1.0,
            "undefined_reason": None,
        }
    ]
    return row


def test_catalog_is_content_addressed_and_order_independent() -> None:
    first = _catalog()
    second = build_target_catalog(list(reversed(first["targets"])))
    assert first == second
    assert first["catalog_revision"].startswith("sha256:")
    assert [target["target_id"] for target in first["targets"]] == [
        "walker-B",
        "walker_A",
    ]


def test_exact_ids_allow_normalized_collisions_but_forbid_index_fallbacks() -> None:
    catalog = build_target_catalog(
        [
            {
                "target_id": "Target A",
                "display_name": "First",
                "identity_source": "authored",
            },
            {
                "target_id": "target_a",
                "display_name": "Second",
                "identity_source": "authored",
            },
        ]
    )
    assert validate_target_catalog(catalog) == []

    with pytest.raises(ValueError, match="not target_N"):
        build_target_catalog(
            [
                {
                    "target_id": "target_0",
                    "display_name": "Fallback",
                    "identity_source": "generated_index",
                }
            ]
        )

    with pytest.raises(ValueError, match="not target_N"):
        build_target_catalog(
            [
                {
                    "target_id": "41d72ca4-43e2-48fb-a85c-c196366f2ca2",
                    "display_name": "Occurrence UUID",
                }
            ]
        )


def test_catalog_rejects_nonfinite_static_metadata() -> None:
    with pytest.raises(ValueError, match="must not contain NaN or infinity"):
        build_target_catalog(
            [
                {
                    "target_id": "walker.left",
                    "display_name": "Left walker",
                    "invariant_properties": {"nominal_rcs_m2": float("nan")},
                }
            ]
        )


def test_state_and_truth_order_do_not_define_identity() -> None:
    catalog = _catalog()
    state = build_target_state(
        catalog,
        scene_frame_id=7,
        reference_time_s=0.7,
        targets=[_active("walker_A", 1.0), _active("walker-B", 5.0)],
    )
    rows = target_rows_by_id(list(reversed(state["targets"])))
    assert rows["walker_A"]["position_m"] == [1.0, 0.0, 1.0]
    assert rows["walker-B"]["position_m"] == [5.0, 0.0, 1.0]

    truth = build_target_truth(
        catalog,
        targets=[
            _truth_active("walker-B", 5.0, 8.0),
            _truth_active("walker_A", 1.0, 3.0),
        ],
    )
    truth_rows = target_rows_by_id(truth["targets"])
    assert truth_rows["walker_A"]["pair_observables"][0]["monostatic_equivalent_range_m"] == 3.0
    assert truth_rows["walker-B"]["pair_observables"][0]["monostatic_equivalent_range_m"] == 8.0


def test_inactive_state_cannot_teleport_to_origin() -> None:
    catalog = _catalog()
    broken = build_target_state(
        catalog,
        scene_frame_id=0,
        reference_time_s=0.0,
        targets=[_inactive("walker_A"), _inactive("walker-B")],
    )
    broken["targets"][0]["position_m"] = [0.0, 0.0, 0.0]
    errors = validate_target_state(catalog, broken)
    assert any("must be null" in error for error in errors)


def test_lifecycle_allows_reappearance_from_inactive_but_not_removed() -> None:
    catalog = _catalog()
    states = [
        build_target_state(
            catalog,
            scene_frame_id=0,
            reference_time_s=0.0,
            targets=[_inactive("walker_A"), _inactive("walker-B")],
        ),
        build_target_state(
            catalog,
            scene_frame_id=1,
            reference_time_s=0.1,
            targets=[_active("walker_A", 1.0), _inactive("walker-B")],
        ),
        build_target_state(
            catalog,
            scene_frame_id=2,
            reference_time_s=0.2,
            targets=[_inactive("walker_A"), _active("walker-B", 2.0)],
        ),
        build_target_state(
            catalog,
            scene_frame_id=3,
            reference_time_s=0.3,
            targets=[_active("walker_A", 3.0), _inactive("walker-B", "removed")],
        ),
    ]
    assert validate_target_lifecycle_sequence(catalog, states) == []

    resurrection = deepcopy(states[-1])
    resurrection["scene_frame_id"] = 4
    resurrection["reference_time_s"] = 0.4
    resurrection["targets"][1] = _active("walker-B", 4.0)
    errors = validate_target_lifecycle_sequence(catalog, [*states, resurrection])
    assert any("reappears after removed" in error for error in errors)


def test_unknown_and_duplicate_state_ids_fail() -> None:
    catalog = _catalog()
    state = {
        "contract_version": 1,
        "scene_frame_id": 0,
        "reference_time_s": 0.0,
        "catalog_revision": catalog["catalog_revision"],
        "targets": [_active("walker_A", 0.0), _active("walker_A", 1.0), _active("C", 2.0)],
    }
    errors = validate_target_state(catalog, state)
    assert any("duplicates" in error for error in errors)
    assert any("absent from target_catalog" in error for error in errors)


def test_state_and_truth_must_cover_every_catalog_target() -> None:
    catalog = _catalog()
    state = {
        "contract_version": 1,
        "scene_frame_id": 0,
        "reference_time_s": 0.0,
        "catalog_revision": catalog["catalog_revision"],
        "targets": [_active("walker_A", 0.0)],
    }
    assert any(
        "contain every catalog target" in error for error in validate_target_state(catalog, state)
    )

    with pytest.raises(ValueError, match="contain every catalog target"):
        build_target_truth(catalog, targets=[_truth_active("walker_A", 0.0, 1.0)])
