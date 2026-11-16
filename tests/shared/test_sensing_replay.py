"""Canonical replay contract tests for native sensing schema-v4 frames."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    canonical_sensing_pair_id,
)
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.replay import (
    PIN_POLICY_ALLOW_STALE,
    PIN_POLICY_EXACT,
    ReplaySceneContext,
    _normalized_json,
    build_replay_event,
    build_replay_index,
    compare_replay_events,
)
from shared.sensing.targets import (
    build_target_catalog,
    build_target_state,
    build_target_truth,
)

CLOCK_ID = "5d6ac341-f653-4b6d-a856-f4c4efdd86fa"
TX_ID = "replay-tx"
RX_ID = "replay-rx"
PAIR_ID = canonical_sensing_pair_id(TX_ID, RX_ID)
ROOT_SOURCE_PRODUCT_ID = "a" * 64


def _uuid(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"orchav-replay-test:{label}"))


def _target_contracts(*, position_x_m: float) -> tuple[dict, dict, dict]:
    catalog = build_target_catalog(
        [
            {
                "display_name": "Walker Alpha",
                "identity_source": "test-fixture",
                "target_id": "walker-alpha",
            }
        ]
    )
    dynamic = {
        "invalid_reason": None,
        "lifecycle": "active",
        "orientation_rad": [0.0, 0.0, 0.0],
        "position_m": [position_x_m, 0.0, 0.0],
        "target_id": "walker-alpha",
        "valid": True,
        "velocity_m_s": [1.0, 0.0, 0.0],
    }
    state = build_target_state(
        catalog,
        scene_frame_id=101,
        reference_time_s=0.001,
        targets=[dynamic],
    )
    truth_row = {
        **dynamic,
        "pair_observables": [
            {
                "bistatic_bisector_velocity_m_s": 1.0,
                "bistatic_path_length_m": 20.0,
                "monostatic_equivalent_range_m": 10.0,
                "monostatic_equivalent_velocity_m_s": 1.0,
                "pair_id": PAIR_ID,
                "path_closing_rate_m_s": 2.0,
                "undefined_reason": None,
            }
        ],
    }
    truth = build_target_truth(catalog, targets=[truth_row])
    return catalog, state, truth


def _frame(
    *,
    run: str = "run-a",
    acquisition: str = "acquisition-a",
    recipe_threshold: float = 3.0,
    source_scene_ids: tuple[int, ...] = (100, 101, 103, 104),
    display_scene_frame_id: int = 101,
    storage_frame_id: int = 10,
    detections: list[dict] | None = None,
    truth_position_x_m: float | None = None,
    root_source_product_id: str = ROOT_SOURCE_PRODUCT_ID,
) -> dict:
    recipe = {
        "cfar_threshold": recipe_threshold,
        "window_doppler": "hann",
        "window_range": "hann",
    }
    samples = [
        {
            "phase_coherent": True,
            "sample_index": index,
            "sample_time_s": index * 0.001,
            "source": {
                "sample_offset": 0,
                "scene_frame_id": scene_frame_id,
                "source_generation_id": CLOCK_ID,
            },
            "valid": True,
        }
        for index, scene_frame_id in enumerate(source_scene_ids)
    ]
    temporal_truth = None
    target_contracts = None
    if truth_position_x_m is not None:
        target_contracts = _target_contracts(position_x_m=truth_position_x_m)
        temporal_truth = {
            "policy": "exact_result_epoch",
            "reference_time_s": 0.0015,
            "scene_frame_id": display_scene_frame_id,
            "source": "canonical_target_truth",
        }
    temporal = build_sensing_temporal_contract(
        clock_id=CLOCK_ID,
        acquisition_id=_uuid(acquisition),
        acquisition_config_id=canonical_sensing_hash(
            {"prf_hz": 1000.0, "sample_count": len(samples)}
        ),
        pair_id=PAIR_ID,
        tx_sensor_id=TX_ID,
        rx_sensor_id=RX_ID,
        samples=samples,
        processing_run_id=_uuid(run),
        processing_recipe_id=canonical_sensing_hash(recipe),
        reference_time_s=0.0015,
        reference_time_policy="cpi_center",
        display_scene_frame_id=display_scene_frame_id,
        display_policy="nearest_at_or_before_reference",
        display_reason="Explicit scene binding for the replay fixture.",
        truth=temporal_truth,
    )
    config = {
        "acquisition": {
            "cpi_hop_samples": 4,
            "cpi_length_samples": 4,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "explicit_test_phase_contract",
            "rt_solve_stride_scene_steps": 4,
            "sample_period_s": 0.001,
            "sample_rate_hz": 1000.0,
            "sample_source": "coherent_one_solve",
        },
        "carrier_freq_hz": 3.5e9,
        "contract_version": 1,
        "fft_size_doppler": 4,
        "fft_size_range": 8,
        "processing": {
            "evidence_status": "native_test_evidence",
            "fft_size_doppler": 4,
            "mode": "coherent_cpi",
            "product_semantics": "native_test_product",
            "result_stride_samples": 4,
            "strategy": "coherent_cpi_fft",
        },
        "processing_mode": "coherent_cpi",
        "processing_recipe": recipe,
        "prf_hz": 1000.0,
        "range_mode": "monostatic",
        "sampling_frequency": 20.0e6,
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "rx_array_processing": "siso",
            "rx_device_count": 1,
            "rx_elements": 1,
            "tx_device_count": 1,
            "tx_elements": 1,
        },
    }
    sensing = {
        "config": deepcopy(config),
        "config_resolved": config,
        "detections": (
            deepcopy(detections)
            if detections is not None
            else [
                {
                    "power_linear": 4.0,
                    "range_m": 10.0,
                    "track_id": 7,
                    "velocity_m_s": 1.0,
                }
            ]
        ),
        "doppler_axis_hz": np.arange(4, dtype=np.float64),
        "processing_strategy": "coherent_cpi_fft",
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
        "range_axis_m": np.arange(8, dtype=np.float64),
        "range_doppler_by_pair": np.full((1, 4, 8), recipe_threshold + 0.0j, dtype=np.complex64),
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
    }
    if target_contracts is not None:
        sensing["target_catalog"], sensing["target_state"], sensing["truth"] = target_contracts
    axes = (
        catalog_axis("pair", size=1, catalog_key="pair_axis"),
        dense_axis(
            "doppler_frequency",
            size=4,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
        ),
        dense_axis("range", size=8, unit="m", coordinate_key="range_axis_m"),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler_complex",
        axes=axes,
        representation="complex_voltage",
        unit="1",
        source_product_ids=(root_source_product_id,),
    )
    materialize_product_specs(sensing)
    return {
        "frame_index": storage_frame_id,
        "rx_orientations": np.array([[0.4, 0.2, -0.1]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": sensing,
        "timestamp_s": 0.0015,
        "tx_orientations": np.array([[0.1, -0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }


def _event(
    frame: dict,
    *,
    bound_scene_frame_id: int = 101,
    current_scene_frame_id: int = 101,
    bound_scene_time_s: float = 0.001,
    current_scene_time_s: float = 0.001,
    pinned: bool = False,
):
    return build_replay_event(
        frame,
        ReplaySceneContext(
            storage_frame_id=frame["frame_index"],
            bound_scene_frame_id=bound_scene_frame_id,
            current_scene_frame_id=current_scene_frame_id,
            bound_scene_time_s=bound_scene_time_s,
            current_scene_time_s=current_scene_time_s,
            pinned=pinned,
        ),
    )


def test_seek_and_cache_insertion_order_do_not_change_index_selection() -> None:
    first = _event(_frame(run="run-first", storage_frame_id=10))
    second_frame = _frame(
        run="run-second",
        acquisition="acquisition-second",
        source_scene_ids=(103, 104, 105, 106),
        display_scene_frame_id=104,
        storage_frame_id=20,
    )
    second = _event(
        second_frame,
        bound_scene_frame_id=104,
        current_scene_frame_id=104,
        bound_scene_time_s=0.004,
        current_scene_time_s=0.004,
    )

    forward = build_replay_index([first, second])
    reverse = build_replay_index([second, first])

    assert forward.events == reverse.events
    assert (
        forward.select(current_scene_frame_id=101, pin_policy=PIN_POLICY_EXACT).manifest_id
        == reverse.select(current_scene_frame_id=101, pin_policy=PIN_POLICY_EXACT).manifest_id
        == first.manifest_id
    )
    assert (
        forward.select(current_scene_frame_id=104).manifest_id
        == reverse.select(current_scene_frame_id=104).manifest_id
        == second.manifest_id
    )


def test_duplicate_events_are_idempotent_and_conflicts_fail_closed() -> None:
    frame = _frame()
    event = _event(frame)
    assert build_replay_index([event, event]).events == (event,)

    changed_frame = deepcopy(frame)
    changed_frame["sensing"]["detections"].append(
        {"power_linear": 1.0, "range_m": 15.0, "velocity_m_s": -1.0}
    )
    conflicting = _event(changed_frame)

    with pytest.raises(ValueError, match="conflicting replay events"):
        build_replay_index([event, conflicting])


def test_stale_result_requires_explicit_pinned_selection_policy() -> None:
    event = _event(
        _frame(),
        current_scene_frame_id=104,
        current_scene_time_s=0.004,
        pinned=True,
    )
    index = build_replay_index([event])

    assert event.scene_binding.state == "stale"
    assert event.scene_binding.stale is True
    assert event.scene_binding.fresh is False
    assert event.scene_binding.stale_age_s == pytest.approx(0.0025)
    assert index.find(current_scene_frame_id=104, pin_policy=PIN_POLICY_EXACT) is None
    assert index.select(current_scene_frame_id=104, pin_policy=PIN_POLICY_ALLOW_STALE) is event
    scene_check = next(
        check for check in event.preflight.checks if check.check_id == "scene_alignment"
    )
    assert scene_check.status == "warning"
    assert event.preflight.disposition == "ready_with_warnings"


def test_truth_layer_changes_never_change_estimate_references() -> None:
    first = _event(_frame(truth_position_x_m=10.0))
    changed_truth = _event(_frame(truth_position_x_m=12.0))

    assert first.truth
    assert changed_truth.truth
    assert first.truth != changed_truth.truth
    assert first.estimates == changed_truth.estimates
    assert first.estimate_reference_id == changed_truth.estimate_reference_id
    assert first.manifest_id != changed_truth.manifest_id
    assert all(reference.layer == "truth" for reference in first.truth)


def test_manifest_is_deterministic_under_detection_input_order() -> None:
    detections = [
        {"power_linear": 9.0, "range_m": 20.0, "track_id": 2, "velocity_m_s": -2.0},
        {"power_linear": 4.0, "range_m": 10.0, "track_id": 1, "velocity_m_s": 1.0},
    ]
    forward = _event(_frame(detections=detections))
    reverse = _event(_frame(detections=list(reversed(detections))))

    assert forward.evidence == reverse.evidence
    assert forward.manifest_id == reverse.manifest_id
    assert forward.to_json() == reverse.to_json()
    assert forward.to_json() == forward.manifest_json()


def test_scientific_manifest_id_ignores_storage_scene_location_and_pin_view_state() -> None:
    exact_frame = _frame(storage_frame_id=10, display_scene_frame_id=101)
    exact = _event(exact_frame)

    relocated_frame = deepcopy(exact_frame)
    relocated_frame["frame_index"] = 9001
    relocated_frame["sensing"]["temporal"]["display_binding"]["scene_frame_id"] = 7001
    relocated = _event(
        relocated_frame,
        bound_scene_frame_id=7001,
        current_scene_frame_id=8001,
        bound_scene_time_s=0.001,
        current_scene_time_s=9.0,
        pinned=True,
    )

    assert exact.manifest_id == relocated.manifest_id
    assert exact.scientific_manifest_payload() == relocated.scientific_manifest_payload()
    assert exact.view_manifest_id != relocated.view_manifest_id
    assert exact.manifest_payload() != relocated.manifest_payload()
    assert exact.to_manifest()["scene_binding"] != relocated.to_manifest()["scene_binding"]
    assert relocated.to_manifest()["view_manifest_id"] == relocated.view_manifest_id


def test_index_keeps_conflicting_view_context_fail_closed_with_stable_scientific_id() -> None:
    frame = _frame()
    first = _event(
        frame,
        current_scene_frame_id=104,
        current_scene_time_s=0.004,
        pinned=True,
    )
    contradictory_view = _event(
        frame,
        current_scene_frame_id=104,
        current_scene_time_s=0.005,
        pinned=True,
    )

    assert first.manifest_id == contradictory_view.manifest_id
    assert first.view_manifest_id != contradictory_view.view_manifest_id
    with pytest.raises(ValueError, match="conflicting replay events"):
        build_replay_index([first, contradictory_view])


def test_optional_nonfinite_detection_evidence_canonicalizes_to_json_null() -> None:
    real_shaped_detection = {
        "aoa_az_deg": float("nan"),
        "aoa_el_deg": np.float32(np.inf),
        "cluster_id": 4,
        "range_m": 10.0,
        "snr_db": 3.0,
        "uncertainty": [np.float64(-np.inf), 0.25],
        "velocity_m_s": 1.0,
    }
    null_equivalent = {
        **real_shaped_detection,
        "aoa_az_deg": None,
        "aoa_el_deg": None,
        "uncertainty": [None, 0.25],
    }

    event = _event(_frame(detections=[real_shaped_detection]))
    null_event = _event(_frame(detections=[null_equivalent]))

    assert event.detections == null_event.detections
    assert event.manifest_id == null_event.manifest_id
    assert json.loads(event.to_json())["manifest_id"] == event.manifest_id
    assert "NaN" not in event.to_json()
    assert "Infinity" not in event.to_json()

    assert _normalized_json(
        np.array([np.nan, np.inf, -np.inf, 0.25]),
        path="optional_evidence_array",
        nonfinite_to_none=True,
    ) == [None, None, None, 0.25]


def test_nonfinite_physical_pose_axis_and_time_remain_fail_closed() -> None:
    frame = _frame()
    frame["tx_positions"][0, 0] = np.nan

    with pytest.raises(ValueError, match=r"finite (?:\[frame_pair, xyz\] array|xyz vector)"):
        _event(frame)

    frame = _frame()
    frame["sensing"]["doppler_axis_hz"][0] = np.inf
    with pytest.raises(ValueError, match="finite|coordinate_sha256"):
        _event(frame)

    frame = _frame()
    frame["sensing"]["temporal"]["result"]["reference_time_s"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _event(frame)

    frame = _frame(
        detections=[
            {
                "detection_id": np.float64(np.nan),
                "range_m": 10.0,
                "velocity_m_s": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="detection_id must not be non-finite"):
        _event(frame)


def test_event_manifest_contains_exact_bound_pair_pose_and_root_product_ancestry() -> None:
    event = _event(_frame())

    assert event.pair_pose.pair_id == event.key.pair_id == PAIR_ID
    assert event.pair_pose.tx_sensor_id == TX_ID
    assert event.pair_pose.rx_sensor_id == RX_ID
    assert event.pair_pose.tx_position_m == (1.0, 2.0, 3.0)
    assert event.pair_pose.tx_orientation_yaw_pitch_roll_rad == (0.1, -0.2, 0.3)
    assert event.pair_pose.rx_position_m == (4.0, 5.0, 6.0)
    assert event.pair_pose.rx_orientation_yaw_pitch_roll_rad == (0.4, 0.2, -0.1)
    assert event.root_source_product_ids == (ROOT_SOURCE_PRODUCT_ID,)
    assert event.to_manifest()["pair_pose"] == event.pair_pose.to_manifest()
    assert event.to_manifest()["root_source_product_ids"] == [ROOT_SOURCE_PRODUCT_ID]


def test_builder_cross_checks_recipe_snapshot_hash_and_is_deeply_immutable() -> None:
    frame = _frame()
    frame["sensing"]["config_resolved"]["processing_recipe"]["cfar_threshold"] = 99.0
    frame["sensing"]["config"]["processing_recipe"]["cfar_threshold"] = 99.0

    with pytest.raises(ValueError, match="processing_recipe_id must match"):
        _event(frame)

    event = _event(_frame())
    with pytest.raises(FrozenInstanceError):
        event.key.reference_time_s = 12.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        event.recipe.parameters["cfar_threshold"] = 99.0  # type: ignore[index]


def test_ab_comparison_is_deterministic_and_reports_parameter_result_product_diff() -> None:
    left = _event(_frame(run="run-a", recipe_threshold=3.0, storage_frame_id=10))
    right = _event(_frame(run="run-b", recipe_threshold=8.0, storage_frame_id=11))

    comparison = compare_replay_events(left, right)
    reversed_comparison = compare_replay_events(right, left)

    assert comparison.to_json() == reversed_comparison.to_json()
    assert comparison.parameter_diff
    assert comparison.result_diff
    assert comparison.product_diff
    assert any(diff.status == "changed" for diff in comparison.product_diff)


def test_ab_comparison_rejects_same_acquisition_id_with_different_source_samples() -> None:
    left = _event(_frame(run="run-a", recipe_threshold=3.0))
    right_frame = _frame(
        run="run-b",
        recipe_threshold=8.0,
        source_scene_ids=(200, 201, 202, 203),
        display_scene_frame_id=201,
        storage_frame_id=11,
    )
    right = _event(
        right_frame,
        bound_scene_frame_id=201,
        current_scene_frame_id=201,
        bound_scene_time_s=0.001,
        current_scene_time_s=0.001,
    )

    with pytest.raises(ValueError, match="same ordered source samples"):
        compare_replay_events(left, right)


def test_ab_comparison_rejects_different_root_source_product_identity() -> None:
    left = _event(_frame(run="run-a", recipe_threshold=3.0))
    right = _event(
        _frame(
            run="run-b",
            recipe_threshold=8.0,
            storage_frame_id=11,
            root_source_product_id="b" * 64,
        )
    )

    with pytest.raises(ValueError, match="same root source-product identities"):
        compare_replay_events(left, right)
