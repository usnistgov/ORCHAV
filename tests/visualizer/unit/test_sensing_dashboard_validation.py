"""Owned dashboard products validate once; external and changed data stay checked."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from visualizer.src.sensing.dashboard_data import load_dashboard_request
from visualizer.src.services.sensing_service import SensingService


def test_fresh_owned_products_do_not_repeat_caller_validation(monkeypatch):
    provider = _held_provider()
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("The freshly validated private selection must not be revalidated")

    monkeypatch.setattr(service, "dashboard_products", unexpected)
    request = load_dashboard_request(service, 0, "analysis", "auto", False, "auto")
    assert request.rd_result is not None
    assert request.display_data_token.sensing is request.display_frame["sensing"]
    assert request.display_data_token.require_complete is False
    request.validate_display_frame()


def test_unowned_selection_keeps_public_validation(monkeypatch):
    provider = _held_provider()
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    resolve = service.resolve_display_sensing_frame

    def external(*args, **kwargs):
        selection = resolve(*args, **kwargs)
        selection.display_frame["sensing"]["temporal"]["acquisition"]["acquisition_id"] = "bad"
        return replace(selection, _replay_identity=None)

    monkeypatch.setattr(service, "resolve_display_sensing_frame", external)
    with pytest.raises(ValueError, match="acquisition_id"):
        load_dashboard_request(service, 0, "analysis", "auto", False, "auto")


def test_invalid_source_is_rejected_before_validation_can_be_reused():
    provider = _held_provider()
    provider.frames[1].sensing["temporal"]["acquisition"]["acquisition_id"] = "bad"
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    with pytest.raises(ValueError, match="acquisition_id"):
        load_dashboard_request(service, 0, "analysis", "auto", False, "auto")
