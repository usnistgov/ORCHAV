"""Exact scene lookup preserves selection while avoiding unrelated index rows."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.extensions.sensing import canonical_sensing_pair_id
from shared.frames.json_codec import dumps_frame_json
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.fixtures.sensing_v3 import (
    build_visualizer_sensing_payload,
    materialize_visualizer_product_specs,
)
from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from tests.visualizer.unit.test_sensing_selection_reuse import _Provider, _scene
from visualizer.src.sensing.dashboard_data import DashboardSession
from visualizer.src.services.sensing_service import SensingService


def _payload(scene, *, kind="range_doppler", run=1, pair="a", value=1.0):
    products = {"detections": []}
    if kind == "range_doppler":
        products.update(
            range_doppler_power_by_pair=np.full((1, 2, 3), value),
            doppler_axis_hz=np.array([-1.0, 1.0]),
            range_axis_m=np.arange(3.0),
        )
    elif kind == "range_profile":
        products.update(range_profile_by_pair=np.full((1, 3), value), range_axis_m=np.arange(3.0))
    sensing = build_visualizer_sensing_payload(
        products, scene_frame_id=scene, sample_time_s=float(scene)
    )
    sensing["temporal"]["result"]["processing_run_id"] = f"00000000-0000-4000-8000-{run:012d}"
    acquisition = sensing["temporal"]["acquisition"]
    tx, rx = f"tx-{pair}", f"rx-{pair}"
    acquisition.update(tx_sensor_id=tx, rx_sensor_id=rx, pair_id=canonical_sensing_pair_id(tx, rx))
    materialize_visualizer_product_specs(sensing)
    return sensing


def _service():
    definitions = {
        10: _payload(5, kind="range_profile", run=4),
        20: _payload(5, run=1),
        30: _payload(5, run=2, pair="b"),
        31: _payload(5, run=2, pair="b"),  # Storage-ID tie-breaking also remains intact.
        40: _payload(6, run=3),
        50: _payload(7, kind="metadata"),
    }
    provider = _Provider(
        {
            storage: replace(build_standard_mpc_frame(frame_idx=storage), sensing=sensing)
            for storage, sensing in definitions.items()
        }
    )
    return (
        SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider))),
        provider,
    )


def _select(service, scene, names, *, pinned=False, current=None, storage=None, **kwargs):
    return service.resolve_display_sensing_frame(
        current if current is not None else _scene(scene),
        scene_frame_id=scene,
        storage_frame_id=scene if storage is None else storage,
        product_names=names,
        pin_freshest=pinned,
        load_scene_frame=_scene,
        **kwargs,
    )


def _assert_same_selection(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual.storage_frame_id == expected.storage_frame_id
    assert actual.bound_scene_frame_id == expected.bound_scene_frame_id
    assert dumps_frame_json(actual.display_frame) == dumps_frame_json(expected.display_frame)
    assert actual.replay_reason == expected.replay_reason
    assert (actual.replay_event.to_manifest() if actual.replay_event else None) == (
        expected.replay_event.to_manifest() if expected.replay_event else None
    )


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("scene", [5, 6, 7, 8])
@pytest.mark.parametrize("names", [("range_doppler",), ("range_profile",), ("config_resolved",)])
def test_exact_and_held_selection_match_full_scan(monkeypatch, pinned, scene, names):
    service, _ = _service()
    reference, _ = _service()
    monkeypatch.setattr(reference, "_exact_scene_result_rows", lambda rows, _scene: rows)
    _assert_same_selection(
        _select(service, scene, names, pinned=pinned),
        _select(reference, scene, names, pinned=pinned),
    )
    assert service.dashboard_exact_result_id(
        scene, product_names=names, pin_freshest=pinned, scene_time_s=float(scene)
    ) == reference.dashboard_exact_result_id(
        scene, product_names=names, pin_freshest=pinned, scene_time_s=float(scene)
    )


def test_grouping_preserves_typed_ids_missing_bindings_and_all_duplicate_rows():
    service, _ = _service()

    class Rows(tuple):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    rows = Rows(
        (storage, {"sensing": {"temporal": {"display_binding": {"scene_frame_id": bound}}}})
        for storage, bound in ((1, 5), ("1", "5"), (2, 5), (3, None), (4, False))
    )
    first = service._exact_scene_result_rows(rows, 5)
    assert [row[0] for row in first] == [1, 2]
    assert [row[0] for row in service._exact_scene_result_rows(rows, "5")] == ["1"]
    assert service._exact_scene_result_rows(rows, 99) == ()
    assert service._exact_scene_result_rows(rows, 5) is first
    assert rows.iterations == 1
    replacement = Rows((rows[1],))
    assert service._exact_scene_result_rows(replacement, 5) == ()
    assert service._exact_scene_result_rows(replacement, "5") == (rows[1],)


def test_current_frame_precedes_same_storage_row_and_remains_mutable(monkeypatch):
    service, _ = _service()
    reference, _ = _service()
    monkeypatch.setattr(reference, "_exact_scene_result_rows", lambda rows, _scene: rows)
    current = {**_scene(5), "sensing": _payload(5, kind="range_profile", value=99.0)}
    kwargs = dict(current=current, storage=20, candidate_storage_ids=(20,))
    actual = _select(service, 5, ("range_profile",), **kwargs)
    _assert_same_selection(actual, _select(reference, 5, ("range_profile",), **kwargs))
    assert actual.storage_frame_id == 20
    np.testing.assert_array_equal(actual.product_frame["sensing"]["range_profile_by_pair"], 99.0)
    current["sensing"]["temporal"]["acquisition"]["acquisition_id"] = "invalid edit"
    with pytest.raises(ValueError, match="acquisition_id"):
        _select(service, 5, ("range_profile",), **kwargs)


def test_exact_hint_keeps_cold_io_boundary_and_checks_only_bound_rows(monkeypatch):
    service, provider = _service()
    assert service.dashboard_exact_result_id(5) is None
    assert provider.requests == []
    rows = service.dashboard_result_rows()
    visited = []
    eligible = service._eligible_dashboard_result

    def count(frame, *args, **kwargs):
        visited.append(frame)
        return eligible(frame, *args, **kwargs)

    monkeypatch.setattr(service, "_eligible_dashboard_result", count)

    def unexpected(*_args, **_kwargs):
        pytest.fail("A cached hint must not perform source I/O")

    monkeypatch.setattr(provider, "load_frame_projection", unexpected)
    assert service.dashboard_exact_result_id(5, product_names=("range_doppler",)) == 31
    assert len(visited) == 4 < len(rows)
    visited.clear()
    assert service.dashboard_exact_result_id(100) is None
    assert visited == []
    provider.generation = "changed"
    assert service.dashboard_exact_result_id(5) is None
    assert visited == []


@pytest.mark.parametrize("change", ["generation", "frame_set", "inventory", "provider"])
def test_revised_source_rebuilds_exact_lookup(change):
    service, provider = _service()
    assert _select(service, 5, ("range_doppler",)).storage_frame_id == 31
    previous_rows = service._scene_result_index_rows
    if change == "provider":
        _, provider = _service()
        service.visualizer.frame_source.provider = provider
    provider.frames[31] = replace(provider.frames[31], sensing=_payload(8, run=2, pair="b"))
    if change in {"generation", "frame_set"}:
        setattr(provider, change, "revision-b")
    elif change == "inventory":
        provider.frames[60] = replace(build_standard_mpc_frame(frame_idx=60), sensing=_payload(9))
    assert service.dashboard_exact_result_id(5) is None
    assert _select(service, 5, ("range_doppler",)).storage_frame_id == 30
    assert _select(service, 8, ("range_doppler",)).storage_frame_id == 31
    assert service._scene_result_index_rows is not previous_rows
    assert [
        row[0] for row in service._exact_scene_result_rows(service.dashboard_result_rows(), 5)
    ] == [10, 20, 30]


def test_custom_loader_and_held_policy_do_not_use_exact_lookup(monkeypatch):
    service, _ = _service()

    def unexpected(*_args):
        pytest.fail("Custom and held selections must retain their ordinary candidate scan")

    monkeypatch.setattr(service, "_exact_scene_result_rows", unexpected)
    product = {"sensing": _payload(5)}
    assert (
        _select(
            service,
            5,
            ("range_doppler",),
            candidate_storage_ids=(20,),
            load_candidate=lambda _: product,
        )
        is not None
    )
    assert _select(service, 8, ("range_doppler",), pinned=True).bound_scene_frame_id == 6


def test_request_reuse_shares_exact_lookup_and_preserves_held_behavior(monkeypatch):
    session = DashboardSession(_held_provider())
    try:
        first = session._cached_request((0, "analysis", "auto", False, "auto"))
        visited = []
        eligible = session.service._eligible_dashboard_result

        def count(frame, *args, **kwargs):
            visited.append(frame)
            return eligible(frame, *args, **kwargs)

        monkeypatch.setattr(session.service, "_eligible_dashboard_result", count)
        assert session._reuse_result_request((3, "analysis", "auto", False, "auto")) is None
        assert len(visited) == 1
        visited.clear()
        exact = session._reuse_result_request((0, "analysis", "auto", False, "auto"))
        assert exact[1].replay_event.to_manifest() == first.replay_event.to_manifest()
        assert len(visited) == 1
        visited.clear()
        held = session._reuse_result_request((2, "analysis", "auto", True, "auto"))
        assert held[1].replay_event.scene_binding.stale
        assert len(visited) == len(session.service.dashboard_result_rows())
    finally:
        session.close()
