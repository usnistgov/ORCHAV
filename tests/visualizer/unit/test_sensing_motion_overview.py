"""Full-span exact spectral histories remain independent of rich result limits."""

from concurrent.futures import CancelledError, Future
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _request
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing.dashboard_data import DashboardSession, history_cells, result_context
from visualizer.src.sensing.motion_overview import load_motion_overview
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.services.sensing_service import SensingService


class MemoryRecording:
    """Tiny authored arrays; real product validation/extraction on every read."""

    def __init__(self, frames, cache=None):
        self.frames = frames
        self.recording_cache = cache
        self.reads = []
        self.reader = SensingService(SimpleNamespace())

    def dashboard_result_rows(self):
        return tuple(enumerate(self.frames))

    def dashboard_load_row(self, storage, products):
        assert products == ("config_resolved",)
        return {
            "sensing": {
                key: value
                for key, value in self.frames[storage]["sensing"].items()
                if not isinstance(value, np.ndarray)
            }
        }

    def dashboard_spectral_products(self, storage, variant, families):
        self.reads.append((storage, families))
        frame = self.frames[storage]
        rp, rd = self.reader.dashboard_products(frame, variant)
        return frame, rp, rd


def _micro_frames(times):
    template = native_cfr(recording="maps", profile=True)
    frames = []
    for index, time in enumerate(times):
        sensing = deepcopy(template)
        delta = time - float(sensing["micro_doppler_time_s"][0])
        sensing["temporal"]["acquisition"]["acquisition_id"] = str(
            uuid5(NAMESPACE_URL, f"overview-acquisition-{index}")
        )
        acquisition = sensing["temporal"]["acquisition"]
        for key in ("start_time_s", "end_time_s", "default_reference_time_s"):
            acquisition[key] += delta
        for sample in acquisition["samples"]:
            sample["sample_time_s"] += delta
            sample["source"]["scene_frame_id"] = index
        sensing["temporal"]["result"]["reference_time_s"] += delta
        sensing["temporal"]["display_binding"]["scene_frame_id"] = index
        keys = {
            axis["coordinate_key"]
            for descriptor in sensing["products"].values()
            for axis in descriptor["axes"]
            if axis["name"] == "slow_time"
        }
        for key in keys:
            sensing[key] = np.asarray(sensing[key]) + delta
        _rehash(sensing)
        frames.append({"sensing": sensing})
    return frames


def _rehash(sensing):
    for descriptor in sensing["products"].values():
        if descriptor["temporal_support"]["type"] == "acquisition_sample":
            descriptor["temporal_support"]["type"] = "last_valid_acquisition_sample"
    _rematerialize_products(sensing)


def test_sparse_stft_capability_survives_current_to_auto_without_signal_reads(
    tmp_path, monkeypatch, qtbot
):
    """A selected RD-only row must not hide a recording's sparse STFT publication."""
    from visualizer.src.sensing import dashboard_data
    from visualizer.src.sensing.motion_overview import MotionOverview

    frames = _micro_frames([0.0, 0.5, 1.0])
    for key in [
        key for key, d in frames[1]["sensing"]["products"].items() if d["kind"] == "micro_doppler"
    ]:
        del frames[1]["sensing"]["products"][key]
        del frames[1]["sensing"][key]
    for key in ("micro_doppler_frequency_hz", "micro_doppler_time_s"):
        del frames[1]["sensing"][key]
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step, frame in enumerate(frames):
        writer.append(
            replace(
                build_standard_mpc_frame(frame_idx=step),
                sensing=frame["sensing"],
                timestamp_s=float(step),
            )
        )
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    session = DashboardSession(provider)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    try:
        current = session._load_request((1, "analysis", "primary", False, "auto"))
        assert "micro_doppler" not in current.available_product_kinds
        assert session.history is None
        # The resident Auto owner has replaced the previously selected STFT owner.
        session.overview = MotionOverview(
            result_context(current.display_frame["sensing"]), "primary", {}, 1000000
        )
        scans = []
        completed = dashboard_data._completed_summary_rows
        monkeypatch.setattr(
            dashboard_data,
            "_completed_summary_rows",
            lambda rows: (scans.append(rows), completed(rows))[1],
        )
        monkeypatch.setattr(
            provider,
            "load_frame_projection",
            lambda *a, **k: pytest.fail("capability loaded source data"),
        )
        monkeypatch.setattr(
            session.service,
            "dashboard_result_rows",
            lambda: pytest.fail("capability started an index scan"),
        )
        current = session.with_recording_data(current)
        assert current.recording_stft_variants == frozenset({"primary", "raw"})
        window._set_view_mode("analysis")
        window.update_from_request(current)
        window._set_view_mode("micro-doppler")
        window.update_from_request(current)
        item = window._micro_mode_box.model().item(window._micro_mode_box.findData("saved-stft"))
        assert item.isEnabled()
        for _ in range(5):
            assert (
                session.with_recording_data(current).recording_stft_variants
                == current.recording_stft_variants
            )
        assert len(scans) == 1
        window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("saved-stft"))
        window.update_from_request(current)
        for variant in ("primary", "raw"):
            assert (
                window._variant_box.model().item(window._variant_box.findData(variant)).isEnabled()
            )
    finally:
        session.close()
        provider.close()


def test_recording_stft_capability_retires_with_source_cohort_and_authored_variants(
    monkeypatch, qtbot
):
    provider = SimpleNamespace(_generation_epoch=0)
    session = DashboardSession(provider)
    frame = _micro_frames([0.0])[0]
    # Only the alternate STFT was authored in this cohort.
    for key in [
        key
        for key, d in frame["sensing"]["products"].items()
        if d["kind"] == "micro_doppler" and d["variant"] == "primary"
    ]:
        del frame["sensing"]["products"][key]
        del frame["sensing"][key]
    rows = ((0, frame),)
    monkeypatch.setattr(session.service, "cached_dashboard_result_rows", lambda: rows)
    request = replace(_request(), display_frame=frame, source_identity=session.source_identity)
    window = SensingWindow()
    qtbot.addWidget(window)
    try:
        assert session.with_recording_data(request).recording_stft_variants == frozenset({"raw"})
        assert session.with_recording_data(
            replace(request, variant="raw")
        ).recording_stft_variants == frozenset({"raw"})
        other = deepcopy(frame)
        other["sensing"]["temporal"]["result"]["processing_run_id"] = "another-run"
        other_request = session.with_recording_data(replace(request, display_frame=other))
        assert other_request.recording_stft_variants == frozenset()
        # Invalidate by replacing the ready immutable index, retaining no STFT.
        rows = ((0, _frame()),)
        absent = session.with_recording_data(_request())
        assert absent.recording_stft_variants == frozenset()
        window._update_recorded_capabilities(absent.display_frame["sensing"], absent)
        assert (
            not window._micro_mode_box.model()
            .item(window._micro_mode_box.findData("saved-stft"))
            .isEnabled()
        )
        provider._generation_epoch = 1
        assert session.with_recording_data(request).recording_stft_variants is None
        assert session._capability_rows is None
    finally:
        session.close()


def test_real_packed_recording_more_than_4096_results_keeps_exact_full_span(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step in range(4097):
        writer.append(
            replace(
                build_standard_mpc_frame(frame_idx=step),
                sensing=_frame(step)["sensing"],
                timestamp_s=step * 0.25,
            )
        )
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    try:
        context = result_context(_frame()["sensing"])
        overview = load_motion_overview(service, context, "primary", {"summed-rd"}, Event())
        family = overview.families["summed-rd"]
        assert family.coverage.state == "complete"
        assert family.coverage.available_results == family.coverage.loaded_results == 4097
        np.testing.assert_array_equal(family.times_s, np.arange(4097) * 0.25)
        expected = np.arange(1.0, 13.0).reshape(4, 3).sum(axis=-1)
        np.testing.assert_array_equal(family.power, np.tile(expected, (4097, 1)))
        assert not family.power.flags.writeable
        assert overview.retained_bytes <= overview.byte_limit
        monkeypatch.setattr(
            service,
            "dashboard_spectral_products",
            lambda *_: pytest.fail("warm component reread source arrays"),
        )
        warm = load_motion_overview(service, context, "primary", {"summed-rd"}, Event())
        np.testing.assert_array_equal(warm.families["summed-rd"].power, family.power)
    finally:
        provider.close()


def test_small_budget_is_visible_and_increased_budget_recovers_full_coverage():
    source = MemoryRecording([_frame(i) for i in range(40)])
    context = result_context(source.frames[0]["sensing"])
    small = load_motion_overview(
        source, context, "primary", {"summed-rd"}, Event(), byte_limit=4096
    )
    item = small.families["summed-rd"]
    assert item.coverage.state == "limited"
    assert 0 < item.coverage.loaded_results < item.coverage.available_results == 40
    assert small.retained_bytes <= 4096
    assert "--sensing-overview-cache-mb" in item.coverage.reason
    full = load_motion_overview(
        source, context, "primary", {"summed-rd"}, Event(), byte_limit=65536
    )
    assert full.families["summed-rd"].coverage.loaded_results == 40
    assert full.families["summed-rd"].coverage.state == "complete"


def test_sparse_stft_selects_only_authored_rows_and_preserves_gap():
    frames = _micro_frames([0.1, 0.116, 0.5])
    # The middle result has RD/profile evidence but no saved STFT block.
    for key in [
        key for key, d in frames[1]["sensing"]["products"].items() if d["kind"] == "micro_doppler"
    ]:
        del frames[1]["sensing"]["products"][key]
        del frames[1]["sensing"][key]
    source = MemoryRecording(frames)
    overview = load_motion_overview(
        source, result_context(frames[0]["sensing"]), "primary", {"saved-stft"}, Event()
    )
    item = overview.families["saved-stft"]
    assert source.reads == [(0, frozenset({"saved-stft"})), (2, frozenset({"saved-stft"}))]
    np.testing.assert_array_equal(item.times_s, [0.1, 0.5])
    np.testing.assert_array_equal(
        item.power[0], frames[0]["sensing"]["micro_doppler_by_pair"][0, 0]
    )
    _, cells = history_cells(item.times_s, item.widths_s, item.power)
    assert np.isnan(cells[1]).all()


@pytest.mark.parametrize("change", ["power", "grid", "recipe"])
def test_stft_rejects_conflicting_epochs_grids_and_recipes(change):
    frames = _micro_frames([0.1, 0.1 if change == "power" else 0.116])
    sensing = frames[1]["sensing"]
    if change == "power":
        sensing["micro_doppler_by_pair"] *= 2
    elif change == "grid":
        sensing["micro_doppler_frequency_hz"] += 1
    else:
        sensing["products"]["micro_doppler_by_pair"]["processing"]["recipe"]["window"] = "boxcar"
    _rehash(sensing)
    result = load_motion_overview(
        MemoryRecording(frames),
        result_context(frames[0]["sensing"]),
        "primary",
        {"saved-stft"},
        Event(),
    )
    assert result.families["saved-stft"].coverage.state == "error"
    assert not result.families["saved-stft"].power.size


@pytest.mark.parametrize("change", ["occurrence", "noise-policy", "transform", "unverified"])
def test_derived_rd_compares_verified_policy_but_preserves_occurrence_recipe(change):
    from shared.extensions.sensing import canonical_processing_recipe_id

    frames = [_frame(i) for i in range(2)]
    for index, frame in enumerate(frames):
        sensing = frame["sensing"]
        noise = {
            "domain": "unwindowed_frequency_observation",
            "generator": "PCG64/SeedSequence(seed,realization_index)",
            "observation_sha256": str(index) * 64,
            "realization_index": 64 * index,
            "seed": None,
            "variance": 0.0,
        }
        recipe = {
            "tool": "shared.sensing_postprocessor",
            "input_kind": "range_doppler_power_by_pair",
            "inherited_numeric_model": {
                "range_response_processing": {"noise": noise, "frequency_window": "hann"}
            },
            "effective_overrides": {"doppler_taper": "hann"},
        }
        if index and change == "noise-policy":
            noise["variance"] = 1.0
        if index and change == "transform":
            recipe["effective_overrides"]["doppler_taper"] = "boxcar"
        recipe_id = canonical_processing_recipe_id(recipe)
        sensing["config_resolved"]["processing_recipe"] = recipe
        sensing["config"] = deepcopy(sensing["config_resolved"])
        sensing["temporal"]["result"]["processing_recipe_id"] = recipe_id
        sensing["products"]["range_doppler_power_by_pair"]["processing"] = {
            "tool": recipe["tool"],
            "input_kind": recipe["input_kind"],
            "recipe_id": "f" * 64 if index and change == "unverified" else recipe_id,
        }
        _rehash(sensing)
    source = MemoryRecording(frames)
    overview = load_motion_overview(
        source, result_context(frames[0]["sensing"]), "primary", {"summed-rd"}, Event()
    )
    item = overview.families["summed-rd"]
    if change != "occurrence":
        assert item.coverage.state == "error"
        assert not item.power.size
        if change != "unverified":
            assert "processing recipes differ" in item.coverage.reason
        return
    assert item.coverage.state == "complete", item.coverage.reason
    assert item.coverage.loaded_results == 2
    expected = np.stack(
        [f["sensing"]["range_doppler_power_by_pair"][0].sum(axis=-1) for f in frames]
    )
    np.testing.assert_array_equal(item.power, expected)
    assert (
        item.recipe == frames[0]["sensing"]["products"]["range_doppler_power_by_pair"]["processing"]
    )
    assert item.sources == tuple(
        (index, frame["sensing"]["temporal"]["acquisition"]["acquisition_id"])
        for index, frame in enumerate(frames)
    )


def test_identical_stft_epoch_is_one_column_with_honest_result_coverage():
    source = MemoryRecording(_micro_frames([0.1, 0.1]))
    result = load_motion_overview(
        source, result_context(source.frames[0]["sensing"]), "primary", {"saved-stft"}, Event()
    )
    item = result.families["saved-stft"]
    assert item.coverage.state == "complete"
    assert item.coverage.loaded_results == 2
    assert item.times_s.size == len(item.sources) == 1


def test_distinct_rd_acquisitions_at_one_epoch_are_not_silently_combined():
    source = MemoryRecording(_micro_frames([0.1, 0.1]))
    result = load_motion_overview(
        source, result_context(source.frames[0]["sensing"]), "primary", {"summed-rd"}, Event()
    )
    item = result.families["summed-rd"]
    assert item.coverage.state == "error"
    assert "same spectral epoch" in item.coverage.reason


def test_cancelled_scan_never_returns_partial_success():
    cancelled = Event()
    source = MemoryRecording([_frame(i) for i in range(5)])

    def progress(_stage, done, _total):
        if done == 1:
            cancelled.set()

    with pytest.raises(CancelledError):
        load_motion_overview(
            source,
            result_context(source.frames[0]["sensing"]),
            "primary",
            {"summed-rd"},
            cancelled,
            progress=progress,
        )


def test_complete_prepared_families_reuse_without_sibling_or_budget_identity(tmp_path, monkeypatch):
    identity = tmp_path / "source-identity"
    identity.write_text("immutable recording")
    cache = RecordingCache((identity,), root=tmp_path / "cache")
    source = MemoryRecording(_micro_frames([0.1, 0.116, 0.132]), cache)
    context = result_context(source.frames[0]["sensing"])
    all_families = {"saved-stft", "summed-rd", "range-profile"}
    prepared = load_motion_overview(source, context, "primary", all_families, Event())
    assert all(
        item.coverage.state in {"complete", "unavailable"} for item in prepared.families.values()
    )
    monkeypatch.setattr(
        source,
        "dashboard_spectral_products",
        lambda *_: pytest.fail("Complete component reread because sibling or budget changed"),
    )
    for families, budget in (
        ({"summed-rd", "range-profile"}, 64 * 1024**2),
        ({"saved-stft", "range-profile"}, 128 * 1024**2),
        ({"saved-stft"}, 32768),
    ):
        loaded = load_motion_overview(
            source, context, "primary", families, Event(), byte_limit=budget
        )
        assert loaded.retained_bytes <= budget
        for name in families:
            np.testing.assert_array_equal(
                loaded.families[name].power, prepared.families[name].power
            )
            assert loaded.families[name].coverage.state == prepared.families[name].coverage.state


def test_limited_cache_does_not_satisfy_increased_budget(tmp_path):
    identity = tmp_path / "source-identity"
    identity.write_text("immutable recording")
    source = MemoryRecording(
        [_frame(i) for i in range(40)], RecordingCache((identity,), root=tmp_path / "cache")
    )
    context = result_context(source.frames[0]["sensing"])
    limited = load_motion_overview(
        source, context, "primary", {"summed-rd"}, Event(), byte_limit=4096
    )
    assert limited.families["summed-rd"].coverage.state == "limited"
    complete = load_motion_overview(
        source, context, "primary", {"summed-rd"}, Event(), byte_limit=65536
    )
    assert complete.families["summed-rd"].coverage.state == "complete"
    assert complete.families["summed-rd"].coverage.loaded_results == 40
    small = load_motion_overview(
        source, context, "primary", {"summed-rd"}, Event(), byte_limit=4096
    )
    assert small.retained_bytes <= 4096
    assert small.families["summed-rd"].coverage.state == "limited"


def test_packed_compact_index_recovers_recipe_without_warm_signal_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step, frame in enumerate(_micro_frames([0.0, 0.5, 1.0])):
        writer.append(
            replace(
                build_standard_mpc_frame(frame_idx=step),
                sensing=frame["sensing"],
                timestamp_s=float(step),
            )
        )
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    families = {"saved-stft", "summed-rd", "range-profile"}
    try:
        rows = service.dashboard_result_rows()
        assert all(
            "processing" not in descriptor
            for _, frame in rows
            for descriptor in frame["sensing"]["products"].values()
        )
        context = result_context(rows[0][1]["sensing"])
        cold = load_motion_overview(service, context, "primary", families, Event())
        assert all(item.coverage.state == "complete" for item in cold.families.values())
        assert cold.families["saved-stft"].recipe
        monkeypatch.setattr(
            service,
            "dashboard_spectral_products",
            lambda *_: pytest.fail("valid warm component reread source signal arrays"),
        )
        original = service.dashboard_load_row
        calls = []

        def metadata_only(storage, products):
            assert products == ("config_resolved",)
            calls.append(storage)
            frame = original(storage, products)
            assert not any(
                isinstance(frame["sensing"].get(key), np.ndarray)
                for key in frame["sensing"]["products"]
            )
            return frame

        monkeypatch.setattr(service, "dashboard_load_row", metadata_only)
        warm = load_motion_overview(service, context, "primary", families, Event())
        assert len(calls) == len(families)
        for family in families:
            np.testing.assert_array_equal(warm.families[family].power, cold.families[family].power)
            assert warm.families[family].recipe == cold.families[family].recipe
    finally:
        provider.close()


@pytest.mark.parametrize("family", ["summed-rd", "saved-stft"])
@pytest.mark.parametrize("corruption", ["negative", "empty", "recipe", "variant"])
def test_semantically_invalid_spectral_cache_is_rebuilt(tmp_path, family, corruption):
    from dataclasses import asdict

    from visualizer.src.sensing.component_cache import save_component
    from visualizer.src.sensing.motion_overview import SPECTRAL_PROTOCOL, spectral_cache_key

    identity = tmp_path / "source-identity"
    identity.write_text("immutable recording")
    cache = RecordingCache((identity,), root=tmp_path / "cache")
    source = MemoryRecording(
        _micro_frames([0.0, 0.5, 1.0]) if family == "saved-stft" else [_frame(i) for i in range(3)],
        cache,
    )
    context = result_context(source.frames[0]["sensing"])
    expected = load_motion_overview(source, context, "primary", {family}, Event()).families[family]
    arrays = {name: getattr(expected, name) for name in ("power", "times_s", "widths_s", "axis")}
    if corruption == "negative":
        arrays["power"] = -np.ones_like(expected.power)
    sources = expected.sources
    if corruption == "empty":
        arrays = {
            "power": np.empty((0, 0)),
            "times_s": np.array([]),
            "widths_s": np.array([]),
            "axis": np.array([]),
        }
        sources = ()
    assert save_component(
        cache,
        "motion-overview",
        spectral_cache_key(context, "primary", family),
        {
            "variant": "raw" if corruption == "variant" else expected.variant,
            "recipe": {"wrong_recipe": True} if corruption == "recipe" else expected.recipe,
            "sources": sources,
            "available_variants": expected.available_variants,
            "coverage": asdict(expected.coverage),
        },
        arrays,
        protocol=SPECTRAL_PROTOCOL,
    )
    source.reads.clear()
    rebuilt = load_motion_overview(source, context, "primary", {family}, Event()).families[family]
    assert len(source.reads) == 3
    np.testing.assert_array_equal(rebuilt.power, expected.power)
    assert rebuilt.recipe == expected.recipe
    assert rebuilt.variant == expected.variant


@pytest.mark.parametrize(
    "view,micro,allowed",
    [
        ("analysis", "auto", {"range_profile", "range_doppler"}),
        ("angular", "auto", {"range_doppler", "range_doppler_angle"}),
        ("processing-inputs", "auto", {"delay_domain_slow_time"}),
        ("processing-frequency", "auto", set()),
        ("micro-doppler", "auto", {"range_doppler"}),
        ("micro-doppler", "saved-stft", set()),
        ("summary", "auto", set()),
        ("tracking", "auto", set()),
    ],
)
def test_cold_dashboard_requests_only_visible_product_kinds(
    tmp_path, monkeypatch, view, micro, allowed
):
    from tests.visualizer.unit.test_sensing_prepared_cache import recording

    root, _ = recording.__wrapped__(tmp_path, monkeypatch)
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    seen = []
    original = provider.load_frame_projection

    def projection(step, request):
        result = original(step, request)
        sensing = result.frame.sensing or {}
        seen.extend(
            (descriptor["kind"], descriptor["variant"])
            for key, descriptor in sensing.get("products", {}).items()
            if isinstance(sensing.get(key), np.ndarray)
        )
        return result

    monkeypatch.setattr(provider, "load_frame_projection", projection)
    try:
        assert session._load_request((0, view, "primary", False, micro)) is not None
        assert {kind for kind, _ in seen} == allowed
        assert all(variant == "primary" for _, variant in seen)
    finally:
        session.close()
        provider.close()


def test_latest_demand_and_source_guard_reject_old_future(monkeypatch):
    session = DashboardSession(SimpleNamespace())
    future = Future()
    monkeypatch.setattr(session._executor, "submit", lambda *a, **kw: future)
    sensing = _frame()["sensing"]
    try:
        session.request_overview(sensing, "primary", frozenset({"summed-rd"}))
        session.request_overview(sensing, "primary", frozenset({"saved-stft"}))
        future.set_result(SimpleNamespace())
        assert not session.poll_overview()
        assert session.overview is None
    finally:
        session.close()


def test_initial_binding_changes_keep_same_overview_job_and_reuse_result(monkeypatch):
    """Moving from representative warm-up to a bound result does not rescan history."""
    session = DashboardSession(SimpleNamespace())
    future = Future()
    submissions = []

    def submit(*args, **kwargs):
        submissions.append((args, kwargs))
        return future

    monkeypatch.setattr(session._executor, "submit", submit)
    families = frozenset({"summed-rd", "range-profile"})
    source = MemoryRecording([_frame(i) for i in range(3)])
    context = result_context(source.frames[0]["sensing"])
    try:
        for frame in source.frames:
            session.request_overview(frame["sensing"], "primary", families)
        assert len(submissions) == 1
        assert session.overview_pending is future
        assert not session._overview_cancelled.is_set()

        overview = load_motion_overview(source, context, "primary", families, Event())
        future.set_result(overview)
        assert session.poll_overview()
        for frame in reversed(source.frames):
            session.request_overview(frame["sensing"], "primary", families)
        assert len(submissions) == 1
        assert session.overview is overview
        assert session.overview_pending is None
    finally:
        session.close()


def test_source_replacement_rejects_completed_spectral_worker(monkeypatch):
    provider = SimpleNamespace(_generation_epoch=0)
    session = DashboardSession(provider)
    future = Future()
    monkeypatch.setattr(session._executor, "submit", lambda *a, **kw: future)
    try:
        session.request_overview(_frame()["sensing"], "primary", frozenset({"summed-rd"}))
        provider._generation_epoch = 1
        future.set_result(SimpleNamespace())
        assert not session.poll_overview()
        assert session.overview is None
    finally:
        session.close()


@pytest.mark.parametrize("failure", [OSError, RuntimeError, MemoryError])
def test_failed_overview_worker_becomes_visible_status_without_retry(monkeypatch, qtbot, failure):
    session = DashboardSession(SimpleNamespace())
    future = Future()
    submissions = []

    def submit(*args, **kwargs):
        submissions.append((args, kwargs))
        return future

    monkeypatch.setattr(session._executor, "submit", submit)
    sensing = _frame()["sensing"]
    families = frozenset({"summed-rd", "range-profile"})
    try:
        session.request_overview(sensing, "primary", families)
        future.set_exception(failure("reader failed"))
        assert session.poll_overview()
        assert session.overview_pending is None
        assert all(item.coverage.state == "error" for item in session.overview.families.values())
        assert session.overview.retained_bytes == 0
        session.request_overview(sensing, "primary", families)
        assert len(submissions) == 1

        window = SensingWindow()
        qtbot.addWidget(window)
        window.show()
        window._set_view_mode("micro-doppler")
        window.update_from_request(replace(_request(), motion_overview=session.overview))
        assert window.retention_notice.isVisible()
        assert "reader failed" in window.retention_notice.text()
    finally:
        session.close()


@pytest.mark.parametrize("retirement", ["source", "demand", "close"])
def test_late_failed_overview_worker_does_not_replace_new_scope(monkeypatch, retirement):
    provider = SimpleNamespace(_generation_epoch=0)
    session = DashboardSession(provider)
    future = Future()
    monkeypatch.setattr(session._executor, "submit", lambda *a, **kw: future)
    try:
        session.request_overview(_frame()["sensing"], "primary", frozenset({"summed-rd"}))
        if retirement == "source":
            provider._generation_epoch = 1
        elif retirement == "demand":
            session.request_overview(_frame()["sensing"], "primary", frozenset({"saved-stft"}))
        else:
            session.close()
        future.set_exception(OSError("old source read failed"))
        assert not session.poll_overview()
        assert session.overview is None
        assert session.overview_pending is None
    finally:
        session.close()


def test_compact_plot_uses_full_series_and_exposes_budget_notice(qtbot):
    source = MemoryRecording([_frame(i) for i in range(40)])
    overview = load_motion_overview(
        source,
        result_context(source.frames[0]["sensing"]),
        "primary",
        {"summed-rd"},
        Event(),
        byte_limit=4096,
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("micro-doppler")
    request = replace(_request(), motion_overview=overview)
    window.update_from_request(request)
    item = overview.families["summed-rd"]
    np.testing.assert_allclose(window.collapsed_doppler_image.image, 10 * np.log10(item.power))
    assert window.retention_notice.isVisible()
    assert "--sensing-overview-cache-mb" in window.retention_notice.text()
    assert "saved results" in window.collapsed_doppler_plot.getPlotItem().titleLabel.text
    assert not window.wants_recording_history()
