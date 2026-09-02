from types import SimpleNamespace

from visualizer.src.app import lifecycle


def _recording_method(events, name):
    return lambda *args: events.append((name, *args))


def test_shutdown_visualizer_releases_resources_in_dependency_order(monkeypatch):
    events = []
    monkeypatch.setattr(
        lifecycle.RecentFilesHandler,
        "save_recent_files",
        _recording_method(events, "recent_files"),
    )
    animation_service = SimpleNamespace(
        stop=_recording_method(events, "animation_service"),
        reset_preloading_state=_recording_method(events, "preloader"),
        set_frame_loader=_recording_method(events, "animation_frame_loader"),
    )
    viz = SimpleNamespace(
        ready=True,
        _benchmark_timer=SimpleNamespace(stop=_recording_method(events, "benchmark_timer")),
        update_timer=SimpleNamespace(stop=_recording_method(events, "update_timer")),
        animation_controller=SimpleNamespace(
            shutdown=_recording_method(events, "animation_controller"),
            stop_live_playback=_recording_method(events, "live_playback"),
        ),
        animation_service=animation_service,
        _vm_warmer=SimpleNamespace(stop=_recording_method(events, "warmer")),
        live_preview_service=SimpleNamespace(stop=_recording_method(events, "live_preview")),
        _mpc_explorer_session=SimpleNamespace(shutdown=_recording_method(events, "mpc_explorer")),
        config_file="recent.json",
        recent_files=["scenario.yaml"],
        session_service=SimpleNamespace(auto_save_on_exit=_recording_method(events, "session")),
        ui_manager=SimpleNamespace(
            cleanup=_recording_method(events, "ui_panels"),
            panels={
                "trajectory": SimpleNamespace(
                    cleanup=_recording_method(events, "trajectory_preview")
                )
            },
        ),
        trajectory_load_coordinator=SimpleNamespace(
            shutdown=_recording_method(events, "trajectory")
        ),
        frame_loader=SimpleNamespace(invalidate=_recording_method(events, "frame_loader")),
        frame_source=SimpleNamespace(close=_recording_method(events, "frame_source")),
        mpc_core=SimpleNamespace(set_frame_source=_recording_method(events, "mpc_frame_source")),
        target_asset_cache=SimpleNamespace(close=_recording_method(events, "target_asset_cache")),
        renderer=SimpleNamespace(close=_recording_method(events, "renderer")),
    )

    assert lifecycle.shutdown_visualizer(viz, persist_state=True) is True

    assert events == [
        ("benchmark_timer",),
        ("update_timer",),
        ("animation_controller",),
        ("live_playback",),
        ("animation_service",),
        ("preloader",),
        ("warmer",),
        ("live_preview",),
        ("mpc_explorer",),
        ("recent_files", "recent.json", ["scenario.yaml"]),
        ("session",),
        ("ui_panels",),
        ("trajectory",),
        ("frame_loader",),
        ("animation_frame_loader", None),
        ("frame_source",),
        ("mpc_frame_source", None),
        ("target_asset_cache",),
        ("renderer",),
    ]
    assert viz.frame_loader is None
    assert viz.frame_source is None
    assert viz._mpc_explorer_session is None
    assert viz.ready is False
    assert viz._shutdown_failures == ()


def test_shutdown_visualizer_is_idempotent():
    events = []
    viz = SimpleNamespace(
        renderer=SimpleNamespace(close=_recording_method(events, "renderer")),
    )

    assert lifecycle.shutdown_visualizer(viz) is True
    assert lifecycle.shutdown_visualizer(viz, persist_state=True) is True

    assert events == [("renderer",)]
    assert viz._shutdown_complete is True


def test_shutdown_visualizer_continues_after_independent_failures():
    events = []

    def fail(name):
        def _failure():
            events.append((name,))
            raise RuntimeError(name)

        return _failure

    viz = SimpleNamespace(
        update_timer=SimpleNamespace(stop=fail("update_timer")),
        live_preview_service=SimpleNamespace(stop=fail("live_preview")),
        trajectory_load_coordinator=SimpleNamespace(
            shutdown=_recording_method(events, "trajectory")
        ),
        frame_source=SimpleNamespace(close=_recording_method(events, "frame_source")),
        renderer=SimpleNamespace(close=fail("renderer")),
    )

    assert lifecycle.shutdown_visualizer(viz) is False

    assert events == [
        ("update_timer",),
        ("live_preview",),
        ("trajectory",),
        ("frame_source",),
        ("renderer",),
    ]
    assert viz._shutdown_failures == (
        "timer:update_timer",
        "live_preview",
        "renderer",
    )
    assert viz._shutdown_complete is True
