"""Bundle reuse certifies current verified artifacts and declared coverage."""

import json
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from shared.frames.providers import Hdf5Provider
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing import prepared_results, startup_cache
from visualizer.src.sensing.cache_policy import SensingCachePolicy
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.services.sensing_service import SensingService


@pytest.mark.parametrize("damage", ["protocol", "corrupt", "missing"])
def test_changed_component_cannot_pass_complete_and_repairs_atomically(
    recording, monkeypatch, damage
):
    provider = Hdf5Provider(recording[0])
    try:
        assert startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        original = RecordingCache.for_provider(provider)
        old_bundle = original._prepared_root
        assert startup_cache._complete(old_bundle, original.signature, require_scene=False)
        if damage == "protocol":
            monkeypatch.setattr(
                prepared_results, "PREPARATION_VERSION", prepared_results.PREPARATION_VERSION + 1
            )
        else:
            path = (
                old_bundle
                / original.for_frame(0)
                ._array_path("plots", (prepared_results.PREPARATION_VERSION, 0, "primary"))
                .name
            )
            if damage == "corrupt":
                path.write_bytes(b"broken NPZ")
            else:
                path.unlink()
        assert not startup_cache._complete(old_bundle, original.signature, require_scene=False)
        assert startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        updated = RecordingCache.for_provider(provider)
        assert updated._prepared_root != old_bundle
        assert old_bundle.is_dir()
        assert startup_cache._complete(
            updated._prepared_root, updated.signature, require_scene=False
        )
        assert (
            prepared_results.load_prepared(updated.for_frame(0), "plots", (0, "primary"))
            is not None
        )
        marker = json.loads((updated._prepared_root / "complete.json").read_text())
        assert marker["capabilities"]["summary"][0]["acquisitions"] == 2
        assert {row["family"] for row in marker["capabilities"]["motion"]} == {
            "saved-stft",
            "summed-rd",
            "range-profile",
        }
    finally:
        provider.close()


def test_failed_component_repair_preserves_ready_pointer(recording, monkeypatch):
    provider = Hdf5Provider(recording[0])
    try:
        startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        cache = RecordingCache.for_provider(provider)
        pointer = cache.root / "recordings" / f"{cache.signature[:20]}.ready"
        original_pointer = pointer.read_bytes()
        monkeypatch.setattr(
            prepared_results, "PREPARATION_VERSION", prepared_results.PREPARATION_VERSION + 1
        )
        original_save = RecordingCache.save_arrays

        def reject_models(owner, kind, *args, **kwargs):
            return (
                False
                if owner._building and kind == "plots"
                else original_save(owner, kind, *args, **kwargs)
            )

        monkeypatch.setattr(RecordingCache, "save_arrays", reject_models)
        with pytest.raises(OSError):
            startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        assert pointer.read_bytes() == original_pointer
        assert cache._prepared_root.is_dir()
        assert len(list((cache.root / "recordings").glob("bundle_*"))) == 1
    finally:
        provider.close()


def test_coverage_policy_changes_invalidate_but_request_residency_does_not(recording):
    provider = Hdf5Provider(recording[0])
    try:
        policy = SensingCachePolicy.for_preparation()
        startup_cache.prepare_startup_cache(
            provider, Event(), include_scene=False, cache_policy=policy
        )
        cache = RecordingCache.for_provider(provider)
        assert startup_cache._complete(
            cache._prepared_root,
            cache.signature,
            require_scene=False,
            cache_policy=replace(policy, request_entries=1, request_bytes=128),
        )
        assert not startup_cache._complete(
            cache._prepared_root,
            cache.signature,
            require_scene=False,
            cache_policy=replace(policy, overview_bytes=2 * policy.overview_bytes),
        )
    finally:
        provider.close()


def test_preparation_plan_ignores_null_warmup_cohort():
    def frame(run, scene, epoch):
        return {
            "sensing": {
                "temporal": {
                    "clock": {"clock_id": "clock"},
                    "result": {"processing_run_id": run, "reference_time_s": epoch},
                    "acquisition": {"pair_id": "pair"},
                    "display_binding": {"scene_frame_id": scene},
                },
                "products": {},
            }
        }

    rows = ((0, frame(None, None, None)), (1, frame("run", 0, 0.5)))
    session = SimpleNamespace(
        service=SimpleNamespace(
            dashboard_result_rows=lambda: rows,
            _eligible_dashboard_result=SensingService._eligible_dashboard_result,
            _selection_key=lambda _frame, storage: storage,
        ),
        provider=SimpleNamespace(has_frame=lambda _: True),
    )
    original, contexts, scenes = startup_cache._selection_plan(session)
    assert original is rows
    assert contexts == {("clock", "run", "pair"): {"primary"}}
    assert set(scenes) == {0}


@pytest.mark.parametrize("field", ["overview_bytes", "history_bytes", "history_results"])
def test_budget_extension_reuses_models_summary_and_complete_spectra_without_replay(
    recording, monkeypatch, field
):
    provider = Hdf5Provider(recording[0])
    try:
        policy = SensingCachePolicy.for_preparation()
        startup_cache.prepare_startup_cache(
            provider, Event(), include_scene=False, cache_policy=policy
        )
        cache = RecordingCache.for_provider(provider)
        previous = cache._prepared_root

        def unexpected(*_args, **_kwargs):
            pytest.fail("Budget-only extension recomputed unchanged figures/replay/spectra")

        monkeypatch.setattr(SensingService, "resolve_display_sensing_frame", unexpected)
        monkeypatch.setattr(SensingService, "dashboard_spectral_products", unexpected)
        monkeypatch.setattr(startup_cache.DashboardSession, "_cached_request", unexpected)
        assert startup_cache.prepare_startup_cache(
            provider,
            Event(),
            include_scene=False,
            retain_session=False,
            cache_policy=replace(policy, **{field: 2 * getattr(policy, field)}),
        )
        updated = RecordingCache.for_provider(provider)
        assert updated._prepared_root != previous
        assert (
            prepared_results.load_prepared(updated.for_frame(0), "plots", (0, "primary"))
            is not None
        )
        marker = json.loads((updated._prepared_root / "complete.json").read_text())
        assert marker["capabilities"]["summary"][0]["acquisitions"] == 2
        assert len(marker["models"]) == 4
    finally:
        provider.close()
