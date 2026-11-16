"""First-open sensing preparation runs before the main window, without blocking Qt."""

from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog

from shared.frames.providers import Hdf5Provider
from visualizer.src.app.startup_workflow import build_argument_parser
from visualizer.src.panels.sensing_panel import SensingPanel
from visualizer.src.sensing import startup_cache_ui


def _args():
    return SimpleNamespace(
        scenario="scenario.yaml",
        author=False,
        render_frames=False,
        benchmark=0,
        data_mode=None,
        sensing_frames_only=True,
        sensing_figures="prepare",
    )


@pytest.fixture(autouse=True)
def figure_policy(qtbot):
    app = QApplication.instance()
    properties = (
        startup_cache_ui.SENSING_FIGURE_POLICY_PROPERTY,
        startup_cache_ui.SENSING_FIGURES_ENABLED_PROPERTY,
        startup_cache_ui._INTERACTIVE_PROPERTY,
        startup_cache_ui.SENSING_3D_OVERRIDE_PROPERTY,
        startup_cache_ui.SENSING_CACHE_OPTIONS_PROPERTY,
        startup_cache_ui.SENSING_MOTION_MODE_PROPERTY,
    )
    previous = {key: app.property(key) for key in properties}
    setting = QSettings().value(startup_cache_ui._POLICY_SETTING)
    startup_cache_ui.configure_sensing_figure_policy(_args(), app)
    yield app
    for key, value in previous.items():
        app.setProperty(key, value)
    if setting is None:
        QSettings().remove(startup_cache_ui._POLICY_SETTING)
    else:
        QSettings().setValue(startup_cache_ui._POLICY_SETTING, setting)


def _scenario():
    return SimpleNamespace(data_mode="files", sensing={"enabled": True})


def _provider():
    # UI tests replace preparation; no actual recording is opened here.
    return object.__new__(Hdf5Provider)


def test_startup_progress_processes_events_before_opening(qtbot, monkeypatch):
    app = QApplication.instance()
    provider = _provider()
    seen = []

    def prepare(selected_provider, cancelled, progress, *, retain_session, cache_policy):
        assert selected_provider is provider
        assert retain_session
        assert cache_policy.request_bytes == 2048 * 1024**2
        progress("Preparing figures", 1, 2)
        assert not cancelled.wait(0.2)
        return True

    monkeypatch.setattr(startup_cache_ui, "prepare_startup_cache", prepare)
    QTimer.singleShot(20, lambda: seen.append(True))
    reporter = SimpleNamespace(note=lambda message: seen.append(message))
    assert startup_cache_ui.prepare_scenario_sensing_cache(_scenario(), provider, app, reporter)
    assert True in seen
    assert "Sensing cache prepared" in seen


def test_cancel_during_startup_aborts_launch(qtbot, monkeypatch):
    app = QApplication.instance()

    def prepare(_provider, cancelled, _progress, *, retain_session, cache_policy):
        assert cancelled.wait(5)
        raise CancelledError()

    monkeypatch.setattr(startup_cache_ui, "prepare_startup_cache", prepare)

    def cancel():
        dialog = next(w for w in app.topLevelWidgets() if isinstance(w, QProgressDialog))
        dialog.canceled.emit()

    QTimer.singleShot(20, cancel)
    assert not startup_cache_ui.prepare_scenario_sensing_cache(
        _scenario(), _provider(), app, SimpleNamespace(note=lambda _: None)
    )


def test_prepare_figures_with_explicit_scene_off_excludes_scene_preparation(
    figure_policy, monkeypatch
):
    args = _args()
    args.sensing_3d = "off"
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    calls = []

    def prepare(provider, cancelled, progress, *, retain_session, include_scene, cache_policy):
        calls.append((retain_session, include_scene))
        return True

    monkeypatch.setattr(startup_cache_ui, "prepare_startup_cache", prepare)
    assert startup_cache_ui.prepare_scenario_sensing_cache(
        _scenario(), _provider(), figure_policy, SimpleNamespace(note=lambda _: None)
    )
    assert calls == [(True, False)]


def test_cancel_wins_over_a_concurrent_source_read_failure(qtbot, monkeypatch):
    app = QApplication.instance()

    def prepare(_provider, cancelled, _progress, *, retain_session, cache_policy):
        assert cancelled.wait(5)
        raise OSError("Source read failed after cancellation")

    monkeypatch.setattr(startup_cache_ui, "prepare_startup_cache", prepare)
    monkeypatch.setattr(
        startup_cache_ui.QMessageBox,
        "warning",
        lambda *_args: pytest.fail("Cancelled opening must not warn and fall back to opening"),
    )

    def cancel():
        dialog = next(
            w for w in app.topLevelWidgets() if isinstance(w, QProgressDialog) and w.isVisible()
        )
        dialog.canceled.emit()

    QTimer.singleShot(20, cancel)
    assert not startup_cache_ui.prepare_scenario_sensing_cache(
        _scenario(), _provider(), app, SimpleNamespace(note=lambda _: None)
    )


def test_cli_defaults_to_demand_and_exposes_loading_and_scene_policies():
    parser = build_argument_parser()
    assert parser.parse_args([]).sensing_figures is None  # Saved preference, initially on-demand.
    for policy in ("on-demand", "prepare", "motion", "summary", "off"):
        assert parser.parse_args(["--sensing-figures", policy]).sensing_figures == policy
    assert parser.parse_args([]).sensing_3d is None
    for policy in ("on", "off"):
        assert parser.parse_args(["--sensing-3d", policy]).sensing_3d == policy


def test_cache_policy_is_captured_per_opening_with_explicit_overrides(figure_policy):
    args = _args()
    args.sensing_overview_cache_mb = 128
    args.sensing_history_results = 12000
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    prepared = startup_cache_ui.sensing_cache_policy(figure_policy)
    startup_cache_ui.set_sensing_figure_strategy("on-demand", figure_policy)
    ordinary = startup_cache_ui.sensing_cache_policy(figure_policy)
    assert prepared.request_bytes == 2048 * 1024**2
    assert ordinary.request_bytes == 512 * 1024**2
    assert ordinary.overview_bytes == prepared.overview_bytes == 128 * 1024**2
    assert ordinary.history_results == prepared.history_results == 12000

    args.sensing_result_cache_mb = 768
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    assert startup_cache_ui.sensing_cache_policy(figure_policy).request_bytes == 768 * 1024**2
    startup_cache_ui.set_sensing_figure_strategy("on-demand", figure_policy)
    assert startup_cache_ui.sensing_cache_policy(figure_policy).request_bytes == 768 * 1024**2
    assert prepared.request_bytes == 2048 * 1024**2  # Existing source policy stays frozen.


def test_explicit_saved_stft_opens_the_requested_existing_motion_view(figure_policy):
    args = _args()
    args.sensing_figures = "motion"
    args.sensing_motion = "saved-stft"
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    presets = []
    panel = SensingPanel(SimpleNamespace(renderer=None))
    panel.recording_figure_policy = "motion"
    panel.recording_motion_mode = startup_cache_ui.initial_sensing_motion_mode(figure_policy)
    panel._initial_figure_view_pending = True
    panel._sensing_window = SimpleNamespace(apply_display_preset=presets.append)
    panel._apply_initial_figure_view()
    assert presets == [{"view_mode": "micro-doppler", "micro_doppler_mode": "saved-stft"}]
    args.sensing_motion = "all-ranges"
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    assert startup_cache_ui.initial_sensing_motion_mode(figure_policy) == "auto"


@pytest.mark.parametrize("policy", ("on-demand", "motion", "summary", "off"))
def test_non_preparing_policies_do_not_open_a_progress_job(figure_policy, monkeypatch, policy):
    args = _args()
    args.sensing_figures = policy
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    monkeypatch.setattr(
        startup_cache_ui,
        "_wait_for_cache",
        lambda *_a, **_kw: pytest.fail("Prepared hidden figures"),
    )
    assert startup_cache_ui.prepare_scenario_sensing_cache(
        _scenario(), _provider(), figure_policy, SimpleNamespace(note=lambda _: None)
    )


@pytest.mark.parametrize("run_kind", ("benchmark", "render_frames"))
def test_automated_runs_skip_preparation_at_the_scenario_hook(figure_policy, monkeypatch, run_kind):
    args = _args()
    setattr(args, run_kind, 1)
    startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
    assert startup_cache_ui.preferred_sensing_figure_policy() == "prepare"
    monkeypatch.setattr(
        startup_cache_ui, "_wait_for_cache", lambda *_a, **_kw: pytest.fail("Prepared benchmark")
    )
    assert startup_cache_ui.prepare_scenario_sensing_cache(
        _scenario(), _provider(), figure_policy, SimpleNamespace(note=lambda _: None)
    )


def test_off_restored_panel_cannot_start_dashboard_or_pinned_figure_reads(
    figure_policy, qtbot, monkeypatch
):
    figure_policy.setProperty(startup_cache_ui.SENSING_FIGURES_ENABLED_PROPERTY, False)
    parent = SimpleNamespace(renderer=None)
    panel = SensingPanel(parent)
    container = panel.create_panel()
    qtbot.addWidget(container)
    try:
        container.show()
        panel.results_section.expand()  # A restored workspace may request its prior tab.
        panel._activate_dashboard()
        panel._refresh_dashboard_async(object(), 64, force=True)
        assert panel.ensure_live_figure_dashboard() is None
        assert panel._sensing_window is None
        assert panel._dashboard_session is None
        assert not panel._dashboard_is_visible()
        assert not panel.widgets["load_sensing_figures"].isChecked()
        assert panel.results_section.isHidden()
        # Figure policy does not disable either of the separate scene controls.
        assert panel.widgets["show_sensing_detections"].isEnabled()
        assert panel.widgets["sensing_playback"].isEnabled()
        assert parent.sensing_playback is panel.playback
    finally:
        panel.cleanup()


def test_figure_toggle_cancels_readers_and_reuses_the_existing_dashboard(
    figure_policy, qtbot, monkeypatch
):
    from visualizer.src.sensing.dashboard_data import DashboardSession

    panel = SensingPanel(SimpleNamespace(renderer=None))
    monkeypatch.setattr(panel, "_refresh_live_plots", lambda *a, **kw: None)
    container = panel.create_panel()
    qtbot.addWidget(container)
    try:
        container.show()
        panel._activate_dashboard()
        window = panel._sensing_window
        monkeypatch.setattr(window, "has_visible_live_figures", lambda: True)
        provider = SimpleNamespace(_generation_epoch=0)
        session = panel._dashboard_session = DashboardSession(provider)
        preparation, playback = panel.preparation, panel.playback
        panel.widgets["load_sensing_figures"].setChecked(False)
        assert session._cancelled.is_set()
        assert panel._dashboard_session is None
        assert panel._dashboard_cache is session
        assert panel.ensure_live_figure_dashboard() is None
        assert not panel._dashboard_is_visible()
        panel._refresh_dashboard_async(provider, 64, force=True)
        assert panel._dashboard_session is None
        assert panel.preparation is preparation and not preparation.cancelled.is_set()
        assert panel.playback is playback
        panel.widgets["load_sensing_figures"].setChecked(True)
        assert startup_cache_ui.sensing_figure_policy() == "prepare"
        assert panel.widgets["sensing_figure_strategy"].currentData() == "prepare"
        assert panel._sensing_window is window
        assert panel._dashboard_cache is session
        assert panel._dashboard_is_visible()
    finally:
        panel.cleanup()


def test_strategy_change_is_saved_without_starting_mid_playback_preparation(
    figure_policy, qtbot, monkeypatch
):
    monkeypatch.setattr(
        startup_cache_ui, "_wait_for_cache", lambda *a, **k: pytest.fail("Mid-playback preparation")
    )
    panel = SensingPanel(SimpleNamespace(renderer=None))
    qtbot.addWidget(panel.create_panel())
    try:
        panel.widgets["load_sensing_figures"].setChecked(False)
        strategy = panel.widgets["sensing_figure_strategy"]
        strategy.setCurrentIndex(strategy.findData("on-demand"))
        strategy.setCurrentIndex(strategy.findData("prepare"))
        assert startup_cache_ui.sensing_figure_policy() == "off"
        assert startup_cache_ui.preferred_sensing_figure_policy() == "prepare"
        # Restored sessions use the saved strategy; the enable switch is session-only.
        args = _args()
        args.sensing_figures = None
        startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
        assert startup_cache_ui.sensing_figure_policy() == "prepare"
        args.sensing_figures = "off"
        startup_cache_ui.configure_sensing_figure_policy(args, figure_policy)
        assert startup_cache_ui.sensing_figure_policy() == "off"
        assert startup_cache_ui.preferred_sensing_figure_policy() == "prepare"
    finally:
        panel.cleanup()
