"""Selected offline Motion preparation is the GUI's durable component path."""

import json
from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_motion_overview import _micro_frames, _rehash
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing import startup_cache
from visualizer.src.sensing.cache_policy import SensingCachePolicy
from visualizer.src.sensing.dashboard_data import result_context
from visualizer.src.sensing.motion_overview import load_motion_overview
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.services.sensing_service import SensingService


@pytest.fixture
def motion_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    frames = _micro_frames([0.1, 0.116, 0.5])
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for index, frame in enumerate(frames):
        writer.append(
            replace(
                build_standard_mpc_frame(frame_idx=index),
                sensing=frame["sensing"],
                timestamp_s=float(index),
            )
        )
    writer.finalize()
    return tmp_path, frames


@pytest.mark.parametrize("mode", ["all-ranges", "saved-stft"])
def test_selected_motion_cold_warm_and_gui_reuse_without_unrelated_work(
    motion_recording, monkeypatch, mode
):
    provider = Hdf5Provider(motion_recording[0])
    reads = []
    original = SensingService.dashboard_spectral_products

    def observe(service, storage, variant, families):
        reads.append(families)
        return original(service, storage, variant, families)

    def unrelated(*_args, **_kwargs):
        pytest.fail("Motion-only preparation started unrelated work")

    monkeypatch.setattr(SensingService, "dashboard_spectral_products", observe)
    for name in (
        "DashboardSession",
        "load_recording_history",
        "load_run_summary",
        "prepare_scene_recording",
    ):
        monkeypatch.setattr(startup_cache, name, unrelated)
    expected = {"range-profile", "saved-stft" if mode == "saved-stft" else "summed-rd"}
    try:
        cold = startup_cache.prepare_motion_cache(provider, Event(), motion_mode=mode)
        assert cold["disk_ready"] and cold["coverage_complete"] and not cold["scene"]
        assert set().union(*reads) == expected
        assert {row["family"] for row in cold["motion"]} == expected
        cache = RecordingCache.for_provider(provider)
        assert not (cache.root / "recordings").exists()
        monkeypatch.setattr(SensingService, "dashboard_spectral_products", unrelated)
        assert startup_cache.prepare_motion_cache(provider, Event(), motion_mode=mode) == cold
        service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
        service.dashboard_result_rows()
        gui = load_motion_overview(
            service,
            result_context(motion_recording[1][0]["sensing"]),
            "primary",
            frozenset(expected),
            Event(),
        )
        assert set(gui.families) == expected
        assert all(f.coverage.state == "complete" for f in gui.families.values())
        assert all(f.coverage.loaded_results == 3 for f in gui.families.values())
    finally:
        provider.close()


def test_motion_reuses_full_bundle_without_replacing_pointer(recording, monkeypatch):
    provider = Hdf5Provider(recording[0])
    try:
        startup_cache.prepare_startup_cache(provider, Event())
        cache = RecordingCache.for_provider(provider)
        pointer = cache.root / "recordings" / f"{cache.signature[:20]}.ready"
        before = pointer.read_bytes()
        marker = (cache._prepared_root / "complete.json").read_bytes()
        monkeypatch.setattr(
            SensingService,
            "dashboard_spectral_products",
            lambda *_: pytest.fail("Motion ignored the full prepared component"),
        )
        assert startup_cache.prepare_motion_cache(provider, Event())["disk_ready"]
        assert pointer.read_bytes() == before
        assert (cache._prepared_root / "complete.json").read_bytes() == marker
    finally:
        provider.close()


def test_failed_component_write_cannot_report_motion_ready(motion_recording, monkeypatch):
    provider = Hdf5Provider(motion_recording[0])
    monkeypatch.setattr(RecordingCache, "save_arrays", lambda *_args, **_kwargs: False)
    try:
        with pytest.raises(OSError, match="durably cached"):
            startup_cache.prepare_motion_cache(provider, Event(), motion_mode="saved-stft")
        cache = RecordingCache.for_provider(provider)
        assert not (cache.root / "recordings").exists()
    finally:
        provider.close()


def test_motion_cancel_does_not_publish_false_bundle(motion_recording):
    provider = Hdf5Provider(motion_recording[0])
    cancelled = Event()

    def progress(stage, *_):
        if stage.startswith("Loading saved-stft"):
            cancelled.set()

    try:
        signature = RecordingCache.for_provider(provider).signature
        with pytest.raises(CancelledError):
            startup_cache.prepare_motion_cache(
                provider, cancelled, progress, motion_mode="saved-stft"
            )
        cache = RecordingCache.for_provider(provider)
        assert cache.signature == signature
        assert not (cache.root / "recordings").exists()
    finally:
        provider.close()


def test_source_change_during_motion_preparation_never_reports_ready(motion_recording):
    provider = Hdf5Provider(motion_recording[0])
    manifest = provider.recording_paths()[0]
    changed = False

    def progress(stage, *_):
        nonlocal changed
        if not changed and stage.startswith("Loading summed-rd"):
            manifest.write_bytes(manifest.read_bytes() + b" ")
            changed = True

    try:
        with pytest.raises((OSError, RuntimeError)):
            startup_cache.prepare_motion_cache(provider, Event(), progress)
        assert changed
        assert not (RecordingCache.for_provider(provider).root / "recordings").exists()
    finally:
        provider.close()


def test_motion_limit_is_durable_but_never_reported_as_complete(motion_recording, monkeypatch):
    provider = Hdf5Provider(motion_recording[0])
    policy = replace(SensingCachePolicy(), overview_bytes=1)
    try:
        result = startup_cache.prepare_motion_cache(provider, Event(), cache_policy=policy)
        assert result["disk_ready"] and not result["coverage_complete"]
        assert any(row["state"] == "limited" for row in result["motion"])
        monkeypatch.setattr(
            SensingService,
            "dashboard_spectral_products",
            lambda *_: pytest.fail("Warm limited component reread source"),
        )
        assert startup_cache.prepare_motion_cache(provider, Event(), cache_policy=policy) == result
    finally:
        provider.close()


def test_motion_optional_scene_is_explicit_and_independently_cached(recording):
    provider = Hdf5Provider(recording[0])
    try:
        result = startup_cache.prepare_motion_cache(provider, Event(), include_scene=True)
        assert result["disk_ready"] and result["scene"]
        cache = RecordingCache.for_provider(provider)
        assert cache.load("scene", startup_cache.SCENE_CACHE_KEY) is not None
        assert not (cache.root / "recordings").exists()
    finally:
        provider.close()


def test_full_plan_includes_stft_variant_independent_of_rd():
    frame = {
        "sensing": {
            "temporal": {
                "clock": {"clock_id": "clock"},
                "result": {"processing_run_id": "run", "reference_time_s": 0.5},
                "acquisition": {"pair_id": "pair"},
                "display_binding": {"scene_frame_id": 0},
            },
            "products": {
                "rd": {"kind": "range_doppler", "variant": "primary"},
                "stft": {"kind": "micro_doppler", "variant": "raw"},
            },
        }
    }
    session = SimpleNamespace(
        service=SimpleNamespace(
            dashboard_result_rows=lambda: ((0, frame),),
            _eligible_dashboard_result=SensingService._eligible_dashboard_result,
            _selection_key=lambda *_: 0,
        ),
        provider=SimpleNamespace(has_frame=lambda _: True),
    )
    _, contexts, scenes = startup_cache._selection_plan(session)
    assert contexts[("clock", "run", "pair")] == {"primary", "raw"}
    assert scenes[0][1] == {"primary"}


def test_full_prepare_keeps_raw_stft_without_requesting_raw_maps(tmp_path, monkeypatch):
    """A spectral-only variant is prepared without inventing selected map products."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for index, frame in enumerate(_micro_frames([0.1, 0.116])):
        sensing = frame["sensing"]
        for key, product in list(sensing["products"].items()):
            if (product["kind"] == "micro_doppler" and product["variant"] == "primary") or (
                product["kind"] in {"range_profile", "range_doppler"}
                and product["variant"] == "raw"
            ):
                del sensing["products"][key]
                del sensing[key]
        _rehash(sensing)
        writer.append(
            replace(
                build_standard_mpc_frame(frame_idx=index),
                sensing=sensing,
                timestamp_s=float(index),
            )
        )
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    requests = []
    original = startup_cache.DashboardSession._cached_request

    def observe(session, key, **kwargs):
        requests.append(key)
        return original(session, key, **kwargs)

    monkeypatch.setattr(startup_cache.DashboardSession, "_cached_request", observe)
    try:
        assert startup_cache.prepare_startup_cache(
            provider, Event(), include_scene=False, retain_session=True
        )
        cache = RecordingCache.for_provider(provider)
        marker = json.loads((cache._prepared_root / "complete.json").read_text())
        motion = marker["capabilities"]["motion"]
        assert any(
            row["variant"] == "raw"
            and row["family"] == "saved-stft"
            and row["state"] == "complete"
            and row["loaded_results"] == 2
            for row in motion
        )
        assert all(
            row["state"] == "unavailable"
            for row in motion
            if row["variant"] == "raw" and row["family"] != "saved-stft"
        )
        assert {model["variant"] for model in marker["models"]} == {"primary"}
        assert requests and all(key[2] == "primary" for key in requests)
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()
