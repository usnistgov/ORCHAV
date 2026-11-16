"""Tracking reuses an exact common result without making standalone visits dense."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from tests.visualizer.unit.test_sensing_tracking_presentation import with_decisions
from visualizer.src.sensing import dashboard_data
from visualizer.src.sensing.dashboard_data import DashboardSession, load_dashboard_request
from visualizer.src.sensing.tracking_presentation import tracking_snapshot
from visualizer.src.services.sensing_service import SensingService


def _tracking_provider():
    provider = _held_provider()
    frame = provider.frames[1]
    sensing = deepcopy(frame.sensing)
    tracks = with_decisions()
    sensing["detections"] = tracks["detections"]
    sensing["detection_filter_metadata"] = tracks["detection_filter_metadata"]
    sensing["detection_filter_metadata"]["tracker"]["decisions"]["update_time_s"] = 0.0
    provider.frames[1] = replace(frame, sensing=sensing)
    return provider


@pytest.mark.parametrize("pinned", (False, True))
@pytest.mark.parametrize("variant", ("auto", "primary"))
def test_tracking_reuses_common_owner_with_exact_states_and_provenance(
    monkeypatch, pinned, variant
):
    provider = _tracking_provider()
    session = DashboardSession(provider)
    reference = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    try:
        current = session._cached_request((0, "analysis", variant, pinned, "auto"))
        expected = load_dashboard_request(reference, 0, "tracking", variant, pinned, "auto")
        original_size = session._request_cache_bytes

        def unexpected(*args, **kwargs):
            pytest.fail("Tracking reread an already-owned exact result")

        monkeypatch.setattr(dashboard_data, "load_dashboard_request", unexpected)
        monkeypatch.setattr(provider, "load_frame_projection", unexpected)
        actual = session._cached_request((0, "tracking", variant, pinned, "auto"))
        assert actual is current
        assert len(session._request_cache) == 1
        assert session._request_cache_bytes == original_size
        assert actual.replay_event.to_manifest() == expected.replay_event.to_manifest()
        assert (
            actual.display_frame["_sensing_display_provenance"]
            == expected.display_frame["_sensing_display_provenance"]
        )
        actual_tracks = tracking_snapshot(actual.display_frame["sensing"])
        assert actual_tracks == tracking_snapshot(expected.display_frame["sensing"])
        assert [state["status"] for state in actual_tracks.states] == ["observed", "coasting"]
    finally:
        session.close()


def test_standalone_tracking_does_not_load_or_prepare_dense_products(monkeypatch):
    provider = _tracking_provider()
    session = DashboardSession(provider)
    requested = []
    original = provider.load_frame_projection

    def record(step, request):
        requested.append(request.sensing_products or ())
        return original(step, request)

    monkeypatch.setattr(provider, "load_frame_projection", record)
    try:
        request = session._cached_request((0, "tracking", "primary", True, "auto"))
        assert request.rp_result is None and request.rd_result is None
        assert request.prepared_angular is None
        assert list(session._request_cache) == [(0, "tracking", "primary", True, "auto")]
        assert requested
        assert not any(
            {"range_profile", "range_doppler", "range_doppler_angle"}.intersection(names)
            for names in requested
        )
        assert [
            state["status"] for state in tracking_snapshot(request.display_frame["sensing"]).states
        ] == ["observed", "coasting"]
    finally:
        session.close()


def test_auto_tracking_does_not_inherit_a_raw_map_fallback():
    from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products

    provider = _held_provider()
    for storage in (1, 4, 7):
        sensing = deepcopy(provider.frames[storage].sensing)
        for product in sensing["products"].values():
            if product["kind"] == "range_doppler":
                product["variant"] = "raw"
        _rematerialize_products(sensing)
        provider.frames[storage] = replace(provider.frames[storage], sensing=sensing)
    session = DashboardSession(provider)
    try:
        current = session._cached_request((0, "analysis", "auto", True, "auto"))
        assert current.variant == "raw"
        for step in (0, 2):
            tracking = session._cached_request((step, "tracking", "auto", True, "auto"))
            assert tracking.variant == "primary"
            assert tracking.rd_result is None
    finally:
        session.close()


@pytest.mark.parametrize(
    "view,products",
    [("live", ("range_doppler",)), ("tracking", ()), ("tracking", ("range_doppler",))],
)
@pytest.mark.parametrize("pinned", [False, True])
def test_detached_held_requests_share_current_owner_and_exact_replay(
    monkeypatch, view, products, pinned
):
    provider = _tracking_provider()
    session = DashboardSession(provider)
    reference = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    try:
        first = session._cached_request((0, "analysis", "primary", pinned, "auto"))
        retained = session._request_cache_bytes
        for step, hold in ((0, not pinned), (2, True), (0, pinned)):
            expected = load_dashboard_request(
                reference, step, view, "primary", hold, "auto", products
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    dashboard_data,
                    "load_dashboard_request",
                    lambda *a, **k: pytest.fail("Reread resident result"),
                )
                actual = session._cached_request((step, view, "primary", hold, "auto", products))
            assert actual.rd_result is first.rd_result
            assert actual.replay_event.to_manifest() == expected.replay_event.to_manifest()
            assert (
                actual.display_frame["_sensing_display_provenance"]
                == expected.display_frame["_sensing_display_provenance"]
            )
            assert actual.state_frame["timestamp_s"] == expected.state_frame["timestamp_s"]
        assert len(session._request_cache) == 1
        assert session._request_cache_bytes == retained
        assert session._cached_request((2, view, "primary", False, "auto", products)) is None
    finally:
        session.close()


def test_warm_tracking_ignores_valid_unprocessed_acquisitions(monkeypatch):
    from shared.extensions.sensing import assert_sensing_contract
    from tests.shared.test_sensing_contract import _incomplete_sensing

    provider = _tracking_provider()
    incomplete = _incomplete_sensing()
    assert_sensing_contract({"sensing": incomplete}, require_schema_version=True)
    provider.frames[0] = replace(provider.frames[0], sensing=incomplete)
    session = DashboardSession(provider)
    try:
        current = session._cached_request((3, "analysis", "primary", False, "auto"))
        tracking = session._cached_request((2, "tracking", "primary", True, "auto"))
        assert tracking.replay_event.scene_binding.storage_frame_id == 1
        assert tracking.replay_event.scene_binding.bound_scene_frame_id == 0
        assert tracking.replay_event is not current.replay_event
        previous = session._cached_request((1, "tracking", "primary", True, "auto"))
        assert previous.display_data_token is tracking.display_data_token
        assert session._cached_request((3, "analysis", "primary", False, "auto")) is current
        assert (
            session.service.dashboard_exact_result_id(
                2, pin_freshest=True, scene_time_s=0.5, product_names=()
            )
            is None
        )
        assert (
            session.service.dashboard_exact_result_id(3, scene_time_s=0.75, product_names=()) == 4
        )
        assert len(session.service.dashboard_result_rows()) >= 4
    finally:
        session.close()


@pytest.mark.parametrize("epoch", [None, True, float("nan"), float("inf"), -float("inf")])
def test_cached_selection_skips_non_numeric_result_epochs(epoch):
    provider = _tracking_provider()
    session = DashboardSession(provider)
    try:
        session._cached_request((0, "analysis", "primary", True, "auto"))
        # Malformed complete data is rejected upstream. This boundary also guards
        # incomplete cached metadata; do not turn missing time into numeric zero.
        rows = session.service.dashboard_result_rows()
        invalid = deepcopy(rows[0][1])
        invalid["sensing"]["temporal"]["result"]["reference_time_s"] = epoch
        session.service._sensing_result_index = ((99, invalid), *rows)
        assert (
            session.service.dashboard_exact_result_id(0, pin_freshest=True, scene_time_s=0.0) == 1
        )
    finally:
        session.close()
