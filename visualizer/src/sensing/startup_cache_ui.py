"""Explicit figure-loading policy and responsive preparation for local recordings."""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from threading import Event
from types import SimpleNamespace

from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from shared.frames.providers import Hdf5Provider

from .cache_policy import SensingCachePolicy, policy_from_options
from .preparation_status import prepared_coverage_messages
from .startup_cache import discard_prepared_dashboard, prepare_startup_cache

SENSING_FIGURE_POLICY_PROPERTY = "sensing_figure_policy"
SENSING_FIGURES_ENABLED_PROPERTY = "sensing_figures_enabled"
SENSING_3D_OVERRIDE_PROPERTY = "sensing_3d_override"
SENSING_CACHE_OPTIONS_PROPERTY = "sensing_cache_options"
SENSING_MOTION_MODE_PROPERTY = "sensing_motion_mode"
_INTERACTIVE_PROPERTY = "sensing_figures_interactive"
_POLICY_SETTING = "visualizer/sensing_figure_policy"
_STRATEGIES = frozenset({"on-demand", "prepare", "motion", "summary"})


def preferred_sensing_figure_policy(app=None) -> str:
    """The user's loading strategy survives temporary 3D-only operation."""
    app = app or QApplication.instance()
    policy = app.property(SENSING_FIGURE_POLICY_PROPERTY) if app else None
    return policy if policy in _STRATEGIES else "on-demand"


def sensing_figure_policy(app=None) -> str:
    """Return the process-wide figure policy, independent of 3D sensing."""
    app = app or QApplication.instance()
    if app and (
        app.property(SENSING_FIGURES_ENABLED_PROPERTY) is False
        or app.property(SENSING_FIGURE_POLICY_PROPERTY) == "off"
    ):
        return "off"
    return preferred_sensing_figure_policy(app)


def set_sensing_figure_strategy(policy: str, app=None) -> None:
    """Save a strategy; preparation runs only at the next scenario opening."""
    if policy not in _STRATEGIES:
        raise ValueError(f"Unknown sensing loading strategy: {policy}")
    app = app or QApplication.instance()
    if app is not None:
        app.setProperty(SENSING_FIGURE_POLICY_PROPERTY, policy)
    QSettings().setValue(_POLICY_SETTING, policy)


def configure_sensing_figure_policy(args, app) -> None:
    """Apply launch intent before any scenario or restored dashboard can load."""
    preferred = QSettings().value(_POLICY_SETTING, "on-demand")
    policy = getattr(args, "sensing_figures", None) or preferred
    strategy = preferred if policy == "off" else policy
    app.setProperty(
        SENSING_FIGURE_POLICY_PROPERTY,
        strategy if strategy in _STRATEGIES else "on-demand",
    )
    app.setProperty(SENSING_FIGURES_ENABLED_PROPERTY, policy != "off")
    app.setProperty(_INTERACTIVE_PROPERTY, not (args.benchmark or args.render_frames))
    scene = getattr(args, "sensing_3d", None)
    app.setProperty(SENSING_3D_OVERRIDE_PROPERTY, None if scene is None else scene == "on")
    app.setProperty(
        SENSING_MOTION_MODE_PROPERTY,
        "saved-stft" if getattr(args, "sensing_motion", None) == "saved-stft" else "auto",
    )
    app.setProperty(
        SENSING_CACHE_OPTIONS_PROPERTY,
        {
            name: getattr(args, name, None)
            for name in (
                "sensing_overview_cache_mb",
                "sensing_history_cache_mb",
                "sensing_history_results",
                "sensing_result_cache_mb",
                "sensing_result_cache_entries",
            )
        },
    )


def sensing_cache_policy(app=None) -> SensingCachePolicy:
    """Capture effective limits for one opening, preserving explicit overrides."""
    app = app or QApplication.instance()
    options = app.property(SENSING_CACHE_OPTIONS_PROPERTY) if app else None
    return policy_from_options(
        SimpleNamespace(**(options if isinstance(options, dict) else {})),
        prepare=sensing_figure_policy(app) == "prepare",
    )


def initial_sensing_motion_mode(app=None) -> str:
    """Keep authored STFT distinct from the all-range saved-RD opening view."""
    app = app or QApplication.instance()
    mode = app.property(SENSING_MOTION_MODE_PROPERTY) if app else None
    return "saved-stft" if mode == "saved-stft" else "auto"


def sensing_3d_override(app=None) -> bool | None:
    """Return an explicit launch override without changing saved scene settings."""
    app = app or QApplication.instance()
    value = app.property(SENSING_3D_OVERRIDE_PROPERTY) if app else None
    return value if isinstance(value, bool) else None


def focused_sensing_figure_policy(app=None) -> str | None:
    """Return focused interactive opening intent, excluding disabled figures."""
    app = app or QApplication.instance()
    if app is not None and app.property(_INTERACTIVE_PROPERTY) is False:
        return None
    policy = sensing_figure_policy(app)
    return policy if policy in {"motion", "summary"} else None


def prepare_scenario_sensing_cache(scenario, provider, app, reporter) -> bool:
    """Prepare before replacing an already-open workspace or restoring a session."""
    if sensing_figure_policy(app) != "prepare" or (
        app is not None and app.property(_INTERACTIVE_PROPERTY) is False
    ):
        return True
    sensing = getattr(scenario, "sensing", None)
    if (
        not isinstance(provider, Hdf5Provider)
        or not isinstance(sensing, dict)
        or not sensing.get("enabled")
        or app is None
    ):
        return True
    ready = _wait_for_cache(
        app,
        reporter,
        lambda cancelled, progress: prepare_startup_cache(
            provider,
            cancelled,
            progress,
            retain_session=True,
            cache_policy=sensing_cache_policy(app),
            **({"include_scene": False} if sensing_3d_override(app) is False else {}),
        ),
        cancel_label="Cancel opening",
    )
    if not ready:
        discard_prepared_dashboard(provider)
        return False
    session = getattr(provider, "_prepared_sensing_dashboard", None)
    summary = getattr(session, "preparation_summary", None)
    if summary is not None:
        coverage = prepared_coverage_messages(summary.get("disk_components", {}))
        message = (
            ("Saved-figure disk cache ready. " if summary.get("disk_ready") else "")
            + f"Saved result views in RAM: {summary['retained_requests']}/"
            f"{summary['requested_requests']}; "
            f"{summary['request_bytes'] / 1024**2:.0f} MiB retained "
            f"within {summary['request_byte_limit'] / 1024**2:.0f} MiB "
            f"and {summary['request_entry_limit']:,} cache entries."
        )
        if summary.get("bundle_bytes") is not None:
            message += f" Prepared disk bundle: {summary['bundle_bytes'] / 1024**2:.0f} MiB."
        if coverage:
            message += "\n\n" + "\n".join(coverage)
        reporter.note(message)
        actions = []
        if not summary["requests_complete"]:
            actions.append(
                "Remaining results will load from the prepared cache during playback. "
                "Increase --sensing-result-cache-mb or --sensing-result-cache-entries "
                "at the next launch to keep more individual results in RAM. "
                "These settings do not change recording-wide plot coverage."
            )
        if coverage or actions:
            QMessageBox.information(
                None,
                "Prepared sensing figures: coverage and RAM",
                "\n\n".join([message, *actions]),
            )
    return True


def _wait_for_cache(app, reporter, prepare, *, cancel_label):
    cancelled = Event()
    updates = SimpleQueue()

    previous_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    dialog = QProgressDialog(
        "Preparing sensing figures for playback.",
        cancel_label,
        0,
        0,
    )
    dialog.setWindowTitle("Preparing sensing playback")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.canceled.connect(cancelled.set)
    loop = QEventLoop()
    timer = QTimer()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-startup")
    future = executor.submit(prepare, cancelled, lambda *update: updates.put(update))

    def poll():
        latest = None
        while True:
            try:
                latest = updates.get_nowait()
            except Empty:
                break
        if latest is not None:
            stage, done, total = latest
            dialog.setLabelText(f"Preparing sensing playback\n{stage}")
            if total > 0:
                dialog.setRange(0, total)
                dialog.setValue(min(done, total))
            else:
                dialog.setRange(0, 0)
        if future.done():
            loop.quit()

    timer.timeout.connect(poll)
    try:
        reporter.note("Preparing saved sensing figures before opening the visualizer")
        dialog.show()
        timer.start(50)
        loop.exec()
        timer.stop()
        try:
            built = future.result()
        except CancelledError:
            return False
        except Exception as exc:  # Cache is disposable; show why normal reads will be used.
            if cancelled.is_set():
                return False
            reporter.note(f"Sensing cache preparation failed: {exc}")
            QMessageBox.warning(
                None,
                "Sensing cache unavailable",
                f"Could not prepare the sensing cache:\n{exc}\n\n"
                "The visualizer will open using the original recording.",
            )
            return True
        reporter.note("Sensing cache prepared" if built else "Reusing prepared sensing cache")
        return not cancelled.is_set()
    finally:
        timer.stop()
        dialog.close()
        executor.shutdown(wait=True, cancel_futures=True)
        app.setQuitOnLastWindowClosed(previous_quit)
