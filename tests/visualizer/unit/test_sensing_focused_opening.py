"""Focused opening preserves real recording plots and independent navigation."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.frames.providers import Hdf5Provider
from tests.visualizer.unit.test_sensing_playback import _playback
from tests.visualizer.unit.test_sensing_prepared_cache import _loaded_products
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from tests.visualizer.unit.test_sensing_startup_cache_ui import figure_policy as figure_policy
from tests.visualizer.unit.test_startup_workflow import _startup_workspace_selection
from visualizer.src.app.startup_workflow import restore_startup_workspace
from visualizer.src.panels.sensing_panel import SensingPanel
from visualizer.src.sensing import preparation_controller, startup_cache_ui
from visualizer.src.sensing.preparation import PreparedSensing
from visualizer.src.sensing.runtime import state_value
from visualizer.src.sensing.scene_recording import provider_token
from visualizer.src.services.sensing_service import SensingService
from visualizer.src.state import create_initial_state


def _policy(app, policy, scene=None):
    startup_cache_ui.configure_sensing_figure_policy(
        SimpleNamespace(sensing_figures=policy, sensing_3d=scene, benchmark=0, render_frames=False),
        app,
    )


@pytest.fixture
def panel_factory(qtbot, recording, monkeypatch):
    root, _ = recording
    owned = []

    def create(*, show_3d=False, automated=False):
        old_plan, _, updates, _, _, _ = _playback(qtbot)
        viz = old_plan.visualizer
        old_plan.close()
        provider = Hdf5Provider(root)
        viz.frame_loader.provider = provider
        viz.frame_source = provider
        viz.renderer = None
        viz.ready = False
        viz._cli_driven_frame_run = automated
        viz.app_state = create_initial_state()
        viz.app_state = replace(
            viz.app_state,
            step=0,
            extension_state={"sensing": {"show_detections": show_3d}},
        )
        viz.set_state = lambda **changes: setattr(
            viz, "app_state", replace(viz.app_state, **changes)
        )
        viz.schedule_update = lambda: updates.append("scheduled")
        viz.extension_services = {"sensing": SensingService(viz)}
        panel = SensingPanel(viz)
        # Drive individual real dashboard requests explicitly; keep unrelated
        # scene/renderer refresh out of these component tests.
        monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
        container = panel.create_panel()
        qtbot.addWidget(container)
        viz.ui_manager.panels["sensing"] = panel
        owned.append((panel, provider))
        return panel, provider, updates

    yield create
    for panel, provider in reversed(owned):
        panel.cleanup()
        provider.close()


@pytest.mark.parametrize("policy", ("motion", "summary", "on-demand", "off"))
def test_hidden_scene_uses_metadata_navigation_without_full_preparation(
    figure_policy, panel_factory, monkeypatch, policy
):
    _policy(figure_policy, policy)
    monkeypatch.setattr(
        preparation_controller, "prepare_sensing", lambda *a: pytest.fail("Eager 3D preparation")
    )
    panel, provider, _ = panel_factory(show_3d=policy in {"motion", "summary"})
    future = panel.playback._future
    assert future.result(timeout=5) == ((0, 1), {0: 0.0, 1: 1.0})
    assert panel.preparation is None
    assert not state_value(panel.parent.app_state, "show_detections")
    assert panel._dashboard_session is None
    assert panel.playback._prepared_future is None
    panel.bind_recording_source()
    assert panel.playback._future is future  # Repeated scenario hookup is idempotent.
    assert panel.playback.checkbox.isChecked()
    assert panel.parent.sensing_playback is panel.playback


def test_minimal_off_off_does_no_automatic_sensing_work(figure_policy, panel_factory, monkeypatch):
    _policy(figure_policy, "off", "off")
    from visualizer.src.sensing import playback

    loads = []
    original = playback.load_sensing_playback_plan
    monkeypatch.setattr(
        playback, "load_sensing_playback_plan", lambda *a: (loads.append(True), original(*a))[1]
    )
    panel, _, _ = panel_factory(show_3d=True)
    assert panel.preparation is None
    assert panel.playback._future is None and panel.playback.steps is None
    assert not panel.widgets["sensing_playback"].isChecked()
    assert not panel.widgets["show_sensing_detections"].isChecked()
    assert not panel.widgets["load_sensing_figures"].isChecked()
    panel.bind_recording_source()
    assert not loads
    panel.widgets["sensing_playback"].setChecked(True)
    assert panel.playback._future.result(timeout=5)[0] == (0, 1)
    assert loads == [True]


@pytest.mark.parametrize("policy,view", (("motion", "micro-doppler"), ("summary", "summary")))
def test_focused_initial_view_survives_reset_and_loads_only_its_products(
    figure_policy, panel_factory, recording, monkeypatch, qtbot, policy, view
):
    _policy(figure_policy, policy)
    panel, provider, _ = panel_factory()
    products = _loaded_products(provider, monkeypatch)
    panel.playback._future.result(timeout=5)
    panel.parent.ready = True
    panel.playback._poll()
    panel._container.show()
    panel._activate_dashboard()
    panel._refresh_dashboard_async(provider, 0, force=True)
    window, session = panel._sensing_window, panel._dashboard_session
    assert window.request_view_mode() == view
    assert window._initial_view_selected
    qtbot.waitUntil(lambda: window._last_request is not None, timeout=5000)
    assert window.request_view_mode() == view
    assert window._selected_micro_doppler_mode() == "auto"
    if policy == "motion":
        qtbot.waitUntil(lambda: session.overview is not None, timeout=5000)
        assert session.history is None  # Spectra have an independent retention owner.
        assert set(session.overview.families) == {"summed-rd", "range-profile"}
        # This fixture has valid per-result maps but no sampled acquisition
        # clock, so a recording-wide spectral timeline must stay unavailable.
        for family in session.overview.families.values():
            assert family.coverage.available_results == 2
            assert family.coverage.state == "unavailable"
            assert "no sampled acquisition clock" in family.coverage.reason
        np.testing.assert_array_equal(
            window._last_request.rd_result.power_linear,
            recording[1][0]["range_doppler_power_by_pair"][0],
        )
        assert products
    else:
        qtbot.waitUntil(lambda: session.run_summary.data is not None, timeout=5000)
        assert session.history is None
        assert not products  # Run summary uses only metadata and poses.
    assert len(session._request_cache) == 1
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("diagnostics"))
    assert window.request_view_mode() == "diagnostics"
    panel._retire_dashboard_session()
    panel._refresh_dashboard_async(provider, 0, force=True)
    assert window.request_view_mode() == "diagnostics"


@pytest.mark.parametrize("policy", ("motion", "summary"))
def test_strategy_edits_apply_next_source_and_preserve_off_preference(
    figure_policy, panel_factory, policy
):
    _policy(figure_policy, policy)
    panel, _, _ = panel_factory()
    future = panel.playback._future
    strategy = panel.widgets["sensing_figure_strategy"]
    strategy.setCurrentIndex(strategy.findData("prepare"))
    panel.bind_recording_source()
    assert panel.recording_figure_policy == policy and panel.playback._future is future
    panel.widgets["load_sensing_figures"].setChecked(False)
    strategy.setCurrentIndex(strategy.findData(policy))
    assert startup_cache_ui.sensing_figure_policy() == "off"
    _policy(figure_policy, "off")
    assert startup_cache_ui.preferred_sensing_figure_policy() == policy
    _policy(figure_policy, None)
    assert startup_cache_ui.sensing_figure_policy() == policy
    strategy.setCurrentIndex(strategy.findData("on-demand"))
    panel.retire_recording_source()
    panel.bind_recording_source()
    assert panel.recording_figure_policy == "on-demand"
    assert panel.playback._future is not future
    assert panel.preparation is None


@pytest.mark.parametrize("outcome", ("ready", "replace", "close"))
def test_explicit_scene_preparation_never_replaces_active_navigation(
    figure_policy, panel_factory, monkeypatch, outcome
):
    _policy(figure_policy, "motion")
    panel, provider, _ = panel_factory()
    panel.playback._future.result(timeout=5)
    panel.parent.ready = True
    panel.playback._poll()
    original_steps, original_times = panel.playback.steps, dict(panel.playback.timestamps)
    panel.parent.animation_running = True
    panel.parent.animation_step = 1
    panel.parent.animation_controller = SimpleNamespace(
        toggle_animation=lambda: pytest.fail("Scene preparation interrupted playback")
    )
    future = Future()
    monkeypatch.setattr(
        preparation_controller,
        "ThreadPoolExecutor",
        lambda **kw: SimpleNamespace(submit=lambda *a: future, shutdown=lambda **kw: None),
    )
    panel.widgets["show_sensing_detections"].setChecked(True)
    job = panel.preparation
    assert job is not None and job.playback_plan is None
    assert panel.parent.extension_services["sensing"].preparation_pending
    panel.start_scene_preparation()
    assert panel.preparation is job
    scene = SimpleNamespace(token=provider_token(provider))
    job._scene_ready(scene)  # Completion must not publish another navigation plan.
    prepared = PreparedSensing(scene, None, None)
    if outcome == "replace":
        panel.parent.frame_loader.provider = None
        panel.parent.frame_source = None
    elif outcome == "close":
        panel._closed = True
    future.set_result(prepared)
    job.poll()
    assert panel.parent.animation_running and panel.parent.animation_step == 1
    assert panel.playback.steps is original_steps
    assert panel.playback.timestamps == original_times
    service = panel.parent.extension_services["sensing"]
    assert getattr(service, "prepared_sensing", None) is (prepared if outcome == "ready" else None)
    if outcome != "ready":
        assert job.cancelled.is_set()
    panel._closed = False  # Let fixture cleanup retire all real Qt owners.
    panel.parent.animation_running = False


@pytest.mark.parametrize("override", (None, False, True))
def test_workspace_override_is_applied_before_render_without_mutating_snapshot(override):
    selection = _startup_workspace_selection("session.json", "scenario", frame=0, camera=None)
    selection.payload["app_state"] = {
        "extension_state": {"sensing": {"show_detections": True, "show_tracking_uncertainty": True}}
    }
    original = selection.payload["app_state"]["extension_state"]["sensing"].copy()
    calls = []
    panel = SimpleNamespace(start_scene_preparation=lambda: calls.append("prepare"))

    def restore(snapshot, **kwargs):
        state = snapshot.payload["app_state"]["extension_state"]["sensing"]
        assert state["show_detections"] is (override is not False)
        assert state["show_tracking_uncertainty"] is (override is not False)
        assert calls == ([] if override is False else ["prepare"])
        calls.append("restore")
        return True

    viz = SimpleNamespace(
        session_service=SimpleNamespace(load_session=restore),
        ui_manager=SimpleNamespace(panels={"sensing": panel}),
    )
    assert (
        restore_startup_workspace(
            viz, selection, SimpleNamespace(note=lambda _: None), sensing_scene_enabled=override
        )
        == selection.path
    )
    assert selection.payload["app_state"]["extension_state"]["sensing"] == original


@pytest.mark.parametrize(
    "policy,override,restored,prepares,visible",
    (
        ("on-demand", None, True, True, True),
        ("on-demand", "off", True, False, False),
        ("on-demand", "on", False, True, True),
        ("motion", None, True, False, False),
        ("motion", "on", False, True, True),
        ("summary", "on", False, True, True),
        ("prepare", None, False, True, False),
        ("prepare", "off", True, False, False),
        ("off", "on", False, True, True),
    ),
)
def test_explicit_scene_intent_wins_without_changing_figure_policy(
    figure_policy, panel_factory, monkeypatch, policy, override, restored, prepares, visible
):
    _policy(figure_policy, policy, override)
    monkeypatch.setattr(
        preparation_controller,
        "ThreadPoolExecutor",
        lambda **kw: SimpleNamespace(submit=lambda *a: Future(), shutdown=lambda **kw: None),
    )
    panel, _, _ = panel_factory(show_3d=restored)
    assert (panel.preparation is not None) is prepares
    assert state_value(panel.parent.app_state, "show_detections") is visible
    assert panel.widgets["show_sensing_detections"].isChecked() is visible
    assert panel.recording_figure_policy == policy
    if prepares:
        assert panel.playback._prepared_future is panel.preparation.playback_plan
    else:
        assert panel.playback._prepared_future is None


def test_programmatic_scene_enable_gates_reads_before_native_frame_sync(
    figure_policy, panel_factory, monkeypatch
):
    from visualizer.src.sensing import runtime

    _policy(figure_policy, "motion")
    panel, _, _ = panel_factory()
    panel.playback._future.result(timeout=5)
    future = Future()
    monkeypatch.setattr(
        preparation_controller,
        "ThreadPoolExecutor",
        lambda **kw: SimpleNamespace(submit=lambda *a: future, shutdown=lambda **kw: None),
    )
    monkeypatch.setattr(runtime, "refresh_tracking_uncertainty", lambda *a, **kw: None)
    monkeypatch.setattr(
        runtime, "_sync_native_frame", lambda *a: pytest.fail("Synchronous unprepared scene read")
    )
    runtime.update_state(panel.parent, show_detections=True)
    service = panel.parent.extension_services["sensing"]
    monkeypatch.setattr(
        service.path_context, "prepare", lambda *a: pytest.fail("Unprepared packet metadata read")
    )
    packet = object()
    assert runtime._prepare_render_packet(panel.parent, service, {}, 0, packet) is packet
    assert runtime._sync_frame(panel.parent, service, {}, 0)
    job = panel.preparation
    assert job is not None and job.playback_plan is None
    assert service.preparation_pending
    assert runtime._sync_frame(panel.parent, service, {}, 1)
    assert panel.preparation is job
    panel.retire_recording_source()
    panel.start_scene_preparation()
    assert panel.preparation is None  # Final retirement cannot restart a reader.


def test_real_deferred_preparation_completes_without_navigation_publication(
    figure_policy, panel_factory
):
    _policy(figure_policy, "motion")
    panel, provider, _ = panel_factory()
    panel.playback._future.result(timeout=5)
    panel.parent.ready = True
    panel.playback._poll()
    steps = panel.playback.steps
    panel.widgets["show_sensing_detections"].setChecked(True)
    job = panel.preparation
    prepared = job.future.result(timeout=5)
    job.poll()
    assert len(prepared.scene.records) == 2
    assert prepared.scene.matches_provider(provider)
    assert job.playback_plan is None
    assert panel.playback.steps is steps
    assert panel.parent.extension_services["sensing"].prepared_sensing is prepared
    assert not panel.parent.extension_services["sensing"].preparation_pending


@pytest.mark.parametrize("scene", (None, "off"))
def test_automated_frame_runs_do_not_adopt_focused_navigation(
    figure_policy, panel_factory, monkeypatch, scene
):
    startup_cache_ui.configure_sensing_figure_policy(
        SimpleNamespace(
            sensing_figures="motion", sensing_3d=scene, benchmark=1, render_frames=False
        ),
        figure_policy,
    )
    monkeypatch.setattr(
        preparation_controller,
        "ThreadPoolExecutor",
        lambda **kw: SimpleNamespace(submit=lambda *a: Future(), shutdown=lambda **kw: None),
    )
    panel, _, _ = panel_factory(automated=True)
    assert not panel._initial_figure_view_pending
    assert panel.playback._future is None and panel.playback.steps is None
    assert not panel.playback.checkbox.isChecked()
    assert (panel.preparation is not None) is (scene is None)
