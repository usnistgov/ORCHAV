"""Prepared products keep causal selection and the current scene on a held read."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from shared.sensing.products import add_product_spec, materialize_product_specs
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_exact_selection import _service
from tests.visualizer.unit.test_sensing_panel_fallback import _typed_dashboard_payload
from tests.visualizer.unit.test_sensing_prepared_cache import recording as _recording
from visualizer.src.sensing.dashboard_data import DashboardSession, load_dashboard_request
from visualizer.src.sensing.prepared_results import load_prepared
from visualizer.src.sensing.startup_cache import prepare_startup_cache

recording = _recording


@pytest.fixture
def future_bound_recording(tmp_path, monkeypatch):
    """The result bound to scene N becomes available after that scene's epoch."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    payloads = []
    with FrameSetWriter.create_new(tmp_path / "frames", compression=None) as writer:
        for step in range(2):
            sensing = _typed_dashboard_payload(scene_frame_id=step, sample_time_s=step + 0.25)
            descriptor = sensing["products"]["range_doppler_power_by_pair"]
            sensing["raw_rd"] = (step + 2) * sensing["range_doppler_power_by_pair"]
            add_product_spec(
                sensing,
                data_key="raw_rd",
                kind="range_doppler",
                variant="raw",
                axes=descriptor["axes"],
                representation="power_linear",
                unit="1",
                combine=descriptor["combine"],
                processing={"slow_time_mean_removal_applied": False},
            )
            materialize_product_specs(sensing)
            payloads.append(sensing)
            writer.append(
                replace(
                    build_standard_mpc_frame(frame_idx=step),
                    timestamp_s=float(step),
                    sensing=sensing,
                )
            )
        writer.finalize()
    provider = Hdf5Provider(tmp_path)
    try:
        prepare_startup_cache(provider, Event(), include_scene=False, retain_session=False)
    finally:
        provider.close()
    return tmp_path, payloads


@pytest.mark.parametrize("variant", ["primary", "raw"])
def test_reopened_prepared_held_result_uses_no_numeric_source_reads(
    future_bound_recording, monkeypatch, variant
):
    root, payloads = future_bound_recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    original = provider.load_frame_projection

    def metadata_only(step, request):
        projection = original(step, request)
        sensing = projection.frame.sensing or {}
        dense = [
            key for key in sensing.get("products", {}) if isinstance(sensing.get(key), np.ndarray)
        ]
        assert not dense, f"Prepared held request reread numeric products: {dense}"
        return projection

    monkeypatch.setattr(provider, "load_frame_projection", metadata_only)
    try:
        session.service.dashboard_result_rows()
        assert session.service.recording_cache._prepared_root is not None
        assert not session._request_cache
        held = load_dashboard_request(session.service, 1, "analysis", variant, True, "auto")
        assert held.state_step == 1
        assert held.display_step == 0
        assert held.display_from_cache
        assert held.replay_event.scene_binding.storage_frame_id == 0
        assert held.replay_event.scene_binding.current_scene_frame_id == 1
        assert held.replay_event.scene_binding.bound_scene_frame_id == 0
        assert held.replay_event.scene_binding.pinned
        assert held.variant == variant
        expected = payloads[0]["raw_rd" if variant == "raw" else "range_doppler_power_by_pair"][0]
        np.testing.assert_array_equal(held.rd_result.power_linear, expected)
        # Before the first reference epoch, a prepared future result is still ineligible.
        assert load_dashboard_request(session.service, 0, "analysis", variant, True, "auto") is None
        # Explicit exact display continues to select scene 1's own result.
        exact = load_dashboard_request(session.service, 1, "analysis", variant, False, "auto")
        assert exact.display_step == exact.state_step == 1
        assert exact.replay_event.scene_binding.storage_frame_id == 1
        assert not exact.display_from_cache
    finally:
        session.close()
        provider.close()


@pytest.mark.parametrize("damage", ["occurrence", "variant"])
def test_prepared_held_occurrence_mismatch_falls_back_to_selected_source(
    future_bound_recording, monkeypatch, damage
):
    from visualizer.src.sensing import prepared_results

    root, payloads = future_bound_recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    original = prepared_results.load_prepared
    attempted = []

    def mismatched(cache, kind, key):
        value = original(cache, kind, key)
        if kind == "plots" and value is not None:
            attempted.append(key)
            arrays = {name: np.full_like(array, -999) for name, array in value["arrays"].items()}
            if damage == "occurrence":
                return {**value, "identity": ("different-clock-run-pair",), "arrays": arrays}
            rd = replace(value["rd"], metadata={**value["rd"].metadata, "variant": "raw"})
            return {**value, "rd": rd, "arrays": arrays}
        return value

    monkeypatch.setattr(prepared_results, "load_prepared", mismatched)
    try:
        held = load_dashboard_request(session.service, 1, "analysis", "primary", True, "auto")
        assert attempted == [(0, "primary")]
        assert held.display_step == 0
        assert held.replay_event.scene_binding.storage_frame_id == 0
        np.testing.assert_array_equal(
            held.rd_result.power_linear, payloads[0]["range_doppler_power_by_pair"][0]
        )
    finally:
        session.close()
        provider.close()


def test_held_cache_hint_preserves_cross_pair_ties_and_exact_cursor_contract():
    service, _provider = _service()
    service.dashboard_result_rows()
    assert service.dashboard_result_cache_hint(
        5, pin_freshest=True, scene_time_s=5, product_names=("range_doppler",)
    ) == (31, 5)
    assert service.dashboard_result_cache_hint(5, pin_freshest=True, scene_time_s=6) == (40, 6)
    assert service.dashboard_exact_result_id(5, pin_freshest=True, scene_time_s=6) is None
    assert service.dashboard_result_cache_hint(5, pin_freshest=True, scene_time_s=4.9) is None
    assert (
        service.dashboard_result_cache_hint(5, pin_freshest=True, scene_time_s=float("nan")) is None
    )


def _forbid_numeric_source_reads(provider, monkeypatch):
    original = provider.load_frame_projection

    def metadata_only(step, request):
        projection = original(step, request)
        sensing = projection.frame.sensing or {}
        loaded = [name for name in sensing.get("products", {}) if name in sensing]
        assert not loaded, f"Prepared plot reread source products: {loaded}"
        return projection

    monkeypatch.setattr(provider, "load_frame_projection", metadata_only)


def _seed_angular_only(root, step=0, variant="primary"):
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        narrow = load_dashboard_request(session.service, step, "angular", variant, False, "auto")
        assert narrow.rp_result is None
        assert narrow.rd_result is not None
        cached = load_prepared(
            session.service.recording_cache.for_frame(step), "plots", (step, variant)
        )
        assert cached["rp"] is None
        return cached
    finally:
        session.close()
        provider.close()


def test_angular_first_cache_is_completed_by_prepare_and_reopened_without_signal_reads(
    recording, monkeypatch
):
    root, payloads = recording
    _seed_angular_only(root)
    provider = Hdf5Provider(root)
    try:
        assert prepare_startup_cache(provider, Event(), include_scene=False, retain_session=False)
    finally:
        provider.close()
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    _forbid_numeric_source_reads(provider, monkeypatch)
    try:
        request = load_dashboard_request(session.service, 0, "analysis", "primary", False, "auto")
        assert request.rp_result is not None
        np.testing.assert_array_equal(
            request.rd_result.power_linear, payloads[0]["range_doppler_power_by_pair"][0]
        )
        cached = load_prepared(
            session.service.recording_cache.for_frame(0),
            "plots",
            (0, "primary"),
            accept=lambda value: isinstance(value, dict) and value.get("rp") is not None,
        )
        assert session.service.recording_cache._prepared_root is not None
        assert cached["rp"] is not None
        assert "range_profile_by_pair" in cached["arrays"]
    finally:
        session.close()
        provider.close()


@pytest.mark.parametrize("resident_owner", [False, True])
def test_held_extension_repairs_original_bound_key_and_reopens_without_signal_reads(
    recording, monkeypatch, resident_owner
):
    root, payloads = recording
    _seed_angular_only(root, step=1)
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        if resident_owner:
            session._cached_request((2, "angular", "primary", True, "auto"))
            request = session._cached_request((2, "analysis", "primary", True, "auto"))
        else:
            request = load_dashboard_request(
                session.service, 2, "analysis", "primary", True, "auto"
            )
        assert request.state_step == 2 and request.display_step == 1
        assert request.rp_result is not None
        scoped = session.service.recording_cache.for_frame(1)
        assert load_prepared(scoped, "plots", (1, "primary"))["rp"] is not None
        assert load_prepared(scoped, "plots", (2, "primary")) is None
    finally:
        session.close()
        provider.close()
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    _forbid_numeric_source_reads(provider, monkeypatch)
    try:
        request = load_dashboard_request(session.service, 2, "analysis", "primary", True, "auto")
        assert request.rp_result is not None
        np.testing.assert_array_equal(
            request.rd_result.power_linear, payloads[1]["range_doppler_power_by_pair"][0]
        )
    finally:
        session.close()
        provider.close()


def test_extending_primary_does_not_merge_or_replace_raw_owner(recording):
    root, payloads = recording
    raw_before = _seed_angular_only(root, variant="raw")
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        request = load_dashboard_request(session.service, 0, "analysis", "primary", False, "auto")
        assert request.variant == "primary"
        assert "raw_rd" not in request.display_frame["sensing"]
        np.testing.assert_array_equal(
            request.rd_result.power_linear, payloads[0]["range_doppler_power_by_pair"][0]
        )
        raw_after = load_prepared(session.service.recording_cache.for_frame(0), "plots", (0, "raw"))
        assert raw_after["rp"] is None
        np.testing.assert_array_equal(raw_after["rd"].power_linear, raw_before["rd"].power_linear)
    finally:
        session.close()
        provider.close()


def test_unavailable_range_profile_remains_a_complete_raw_rd_hit(recording, monkeypatch):
    root, _ = recording
    _seed_angular_only(root, variant="raw")
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    _forbid_numeric_source_reads(provider, monkeypatch)
    try:
        request = load_dashboard_request(session.service, 0, "analysis", "raw", False, "auto")
        assert request.rp_result is None
        assert request.rd_result is not None
        assert request.variant == "raw"
    finally:
        session.close()
        provider.close()


def test_failed_staging_extension_preserves_ordinary_narrow_artifact(recording, monkeypatch):
    from visualizer.src.sensing import prepared_results

    root, _ = recording
    narrow = _seed_angular_only(root)
    ordinary_before = {
        path: path.read_bytes()
        for path in (root / "cache" / "orchav" / "sensing").iterdir()
        if path.is_file()
    }
    provider = Hdf5Provider(root)
    provider._sensing_cache_build_root = root / "building"
    session = DashboardSession(provider)
    original_save = prepared_results.save_prepared
    attempted = []

    def fail_union(cache, kind, key, value):
        if kind == "plots" and cache._building and value["rp"] is not None:
            attempted.append(key)
            return False
        return original_save(cache, kind, key, value)

    monkeypatch.setattr(prepared_results, "save_prepared", fail_union)
    try:
        with pytest.raises(OSError, match="Could not save the sensing figure model"):
            load_dashboard_request(session.service, 0, "analysis", "primary", False, "auto")
        assert attempted == [(0, "primary")]
    finally:
        session.close()
        provider.close()
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        session.service.dashboard_result_rows()
        unchanged = load_prepared(
            session.service.recording_cache.for_frame(0), "plots", (0, "primary")
        )
        assert unchanged["rp"] is None
        np.testing.assert_array_equal(unchanged["rd"].power_linear, narrow["rd"].power_linear)
        assert all(path.read_bytes() == content for path, content in ordinary_before.items())
        assert not tuple((root / "cache" / "orchav" / "sensing" / "recordings").glob("*.ready"))
    finally:
        session.close()
        provider.close()
