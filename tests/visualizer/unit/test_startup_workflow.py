import json
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.app import startup_ui, startup_workflow
from visualizer.src.app.window_manager import WindowLayout
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.session_service import (
    WorkspaceSnapshot,
    WorkspaceSnapshotSummary,
)


def _startup_workspace_selection(
    path: str,
    scenario_root: str,
    *,
    frame: int,
    camera: dict | None,
    scenario_name: str | None = None,
) -> WorkspaceSnapshot:
    """Build one parsed startup snapshot without touching the filesystem."""
    snapshot_path = Path(path)
    root = Path(scenario_root)
    return WorkspaceSnapshot(
        summary=WorkspaceSnapshotSummary(
            path=snapshot_path,
            scenario_root=root,
            scenario_name=scenario_name or root.name,
            created_at=datetime(2026, 7, 18, 12, 0, 0),
            frame=frame,
            is_autosave=True,
        ),
        payload={
            "version": "6.0",
            "created_at": "2026-07-18T12:00:00",
            "snapshot_kind": "autosave",
            "scenario_path": str(root),
            "animation": {"current_frame": frame},
            "camera": camera,
        },
    )


def test_parse_cli_args_rejects_benchmark_state_without_benchmark(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--benchmark-state-json", str(state_path)])


@pytest.mark.parametrize(
    "argv",
    [
        ["--scenario", "demo", "--benchmark", "-1"],
        ["--scenario", "demo", "--benchmark", "1", "--benchmark-warmup", "-1"],
        ["--benchmark-output", "results.json"],
    ],
)
def test_parse_cli_args_rejects_invalid_benchmark_counts_and_orphan_output(argv):
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(argv)


def test_parse_cli_args_accepts_layout_profile():
    args = startup_workflow.parse_cli_args(["--layout-profile", "capture-renderer"])

    assert args.layout_profile == "capture-renderer"

    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--layout-profile", "obs" + "-full"])


def test_parse_cli_args_accepts_data_mode_and_grpc_port_overrides():
    args = startup_workflow.parse_cli_args(
        [
            "--scenario",
            "scenarios/example",
            "--data-mode",
            "live_grpc",
            "--grpc-port",
            "50052",
        ]
    )

    assert args.data_mode == "live_grpc"
    assert args.grpc_port == 50052


@pytest.mark.parametrize(
    "argv",
    [
        ["--data-mode", "live_grpc"],
        ["--grpc-port", "50052"],
        ["--scenario", "demo", "--grpc-port", "0"],
        ["--scenario", "demo", "--grpc-port", "65536"],
        ["--scenario", "demo", "--data-mode", "streaming"],
        ["--scenario", "demo", "--data-mode", "files", "--grpc-port", "50052"],
        ["--scenario", "demo", "--author", "--data-mode", "files"],
    ],
)
def test_parse_cli_args_rejects_invalid_data_source_overrides(argv):
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(argv)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "embedded"),
        (["--renderer", "open3d"], "detached"),
        (["--layout-profile", "capture-workspace"], "detached"),
        (["--scenario", "demo", "--benchmark", "1"], "detached"),
        (["--scenario", "demo", "--render-frames", "frames"], "detached"),
        (
            ["--scenario", "demo", "--benchmark", "1", "--viewport-mode", "embedded"],
            "embedded",
        ),
    ],
)
def test_launch_mode_matrix_resolves_before_qt(argv, expected):
    args = startup_workflow.parse_cli_args(argv)

    assert args.resolved_viewport_mode == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["--renderer", "open3d", "--viewport-mode", "embedded"],
        ["--layout-profile", "capture-renderer", "--viewport-mode", "embedded"],
        ["--scenario", "demo", "--render-frames", "frames", "--viewport-mode", "embedded"],
        ["--benchmark", "1"],
    ],
)
def test_launch_mode_matrix_rejects_invalid_combinations(argv):
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(argv)


def test_parse_cli_args_accepts_no_resume_and_rejects_retired_no_session():
    args = startup_workflow.parse_cli_args(["--no-resume"])

    assert args.no_resume is True

    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(["--no-session"])


def test_workspace_resume_is_disabled_for_clean_and_cli_driven_runs():
    assert startup_workflow.workspace_resume_enabled(startup_workflow.parse_cli_args([])) is True
    assert (
        startup_workflow.workspace_resume_enabled(startup_workflow.parse_cli_args(["--no-resume"]))
        is False
    )
    assert (
        startup_workflow.workspace_resume_enabled(
            startup_workflow.parse_cli_args(["--scenario", "scenarios/example", "--benchmark", "1"])
        )
        is False
    )


def test_pygfx_present_method_cli_sets_launch_environment(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_PRESENT_METHOD", "screen")
    args = startup_workflow.parse_cli_args(
        ["--renderer", "pygfx", "--pygfx-present-method", "bitmap"]
    )

    notes = startup_workflow.apply_launch_environment(args)

    assert startup_workflow.os.environ["ORCHAV_PYGFX_PRESENT_METHOD"] == "bitmap"
    assert notes == ["ORCHAV_PYGFX_PRESENT_METHOD=bitmap"]


def test_pygfx_adapter_name_cli_sets_launch_environment(monkeypatch):
    monkeypatch.setenv("PYGFX_WGPU_ADAPTER_NAME", "NVIDIA")
    args = startup_workflow.parse_cli_args(
        ["--renderer", "pygfx", "--pygfx-adapter-name", "Intel(R) Graphics"]
    )

    notes = startup_workflow.apply_launch_environment(args)

    assert startup_workflow.os.environ["PYGFX_WGPU_ADAPTER_NAME"] == "Intel(R) Graphics"
    assert notes == ["PYGFX_WGPU_ADAPTER_NAME=Intel(R) Graphics"]


def test_pygfx_adapter_name_cli_rejects_non_pygfx_renderer():
    with pytest.raises(SystemExit):
        startup_workflow.parse_cli_args(
            ["--renderer", "open3d", "--pygfx-adapter-name", "Intel(R) Graphics"]
        )


def test_pygfx_adapter_name_is_not_selected_when_option_is_omitted(monkeypatch):
    monkeypatch.setenv("PYGFX_WGPU_ADAPTER_NAME", "existing adapter choice")
    args = startup_workflow.parse_cli_args([])

    notes = startup_workflow.apply_launch_environment(args)

    assert startup_workflow.os.environ["PYGFX_WGPU_ADAPTER_NAME"] == "existing adapter choice"
    assert notes == []


def test_apply_launch_environment_sets_texture_and_performance_env(monkeypatch):
    for name in (
        "ORCHAV_ENABLE_TEXTURES",
        "ORCHAV_DISABLE_TEXTURES",
        "VIZ_CANON_CACHE_MB",
        "ORCHAV_PYGFX_MPC_LINE_CACHE_MB",
    ):
        monkeypatch.delenv(name, raising=False)

    args = startup_workflow.parse_cli_args(
        ["--enable-textures", "--max-performance", "--renderer", "pygfx"]
    )

    notes = startup_workflow.apply_launch_environment(args)

    assert notes == [
        "max-performance: VIZ_CANON_CACHE_MB=4096",
        "max-performance: ORCHAV_PYGFX_MPC_LINE_CACHE_MB=1024",
    ]
    assert startup_workflow.os.environ["ORCHAV_ENABLE_TEXTURES"] == "1"
    assert "ORCHAV_DISABLE_TEXTURES" not in startup_workflow.os.environ
    assert startup_workflow.os.environ["VIZ_CANON_CACHE_MB"] == "4096"
    assert startup_workflow.os.environ["ORCHAV_PYGFX_MPC_LINE_CACHE_MB"] == "1024"


def test_apply_launch_environment_disable_textures_clears_enable_env(monkeypatch):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)

    args = startup_workflow.parse_cli_args(["--disable-textures"])

    startup_workflow.apply_launch_environment(args)

    assert "ORCHAV_ENABLE_TEXTURES" not in startup_workflow.os.environ
    assert startup_workflow.os.environ["ORCHAV_DISABLE_TEXTURES"] == "1"


def test_blocking_pygfx_benchmark_uses_manual_canvas_scheduler(monkeypatch):
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)
    args = startup_workflow.parse_cli_args(
        [
            "--renderer",
            "pygfx",
            "--benchmark",
            "10",
            "--scenario",
            "scenarios/example",
            "--benchmark-present-mode",
            "blocking",
        ]
    )

    notes = startup_workflow.apply_launch_environment(args)

    assert startup_workflow.os.environ["ORCHAV_PYGFX_CANVAS_SCHEDULER"] == "manual"
    assert notes == ["blocking benchmark: ORCHAV_PYGFX_CANVAS_SCHEDULER=manual"]


def test_blocking_benchmark_preserves_explicit_canvas_scheduler(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "fastest")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)
    args = startup_workflow.parse_cli_args(
        [
            "--renderer",
            "pygfx",
            "--benchmark",
            "10",
            "--scenario",
            "scenarios/example",
            "--benchmark-present-mode",
            "blocking",
        ]
    )

    notes = startup_workflow.apply_launch_environment(args)

    assert startup_workflow.os.environ["ORCHAV_PYGFX_CANVAS_SCHEDULER"] == "fastest"
    assert notes == []


def test_configure_cli_driven_frame_run_disables_background_work():
    calls = []
    viz = SimpleNamespace(
        _cli_driven_frame_run=False,
        use_preload_mode=True,
        set_background_update_enabled=lambda enabled: calls.append(("background", enabled)),
        cancel_startup_preload=lambda: calls.append("cancel_preload"),
        cancel_scheduled_update=lambda: calls.append("cancel_update"),
    )
    reporter = SimpleNamespace(note=lambda message: calls.append(("note", message)))

    startup_workflow.configure_cli_driven_frame_run(viz, reporter, enabled=True)

    assert viz._cli_driven_frame_run is True
    assert viz.use_preload_mode is False
    assert calls == [
        ("background", False),
        "cancel_preload",
        "cancel_update",
        ("note", "Background updates and startup preload disabled for CLI-driven frame stepping"),
    ]


def test_close_renderer_and_quit_uses_cooperative_shutdown():
    calls = []
    viz = SimpleNamespace(renderer=SimpleNamespace(close=lambda: calls.append("renderer")))
    app = SimpleNamespace(quit=lambda: calls.append("qt"))

    startup_workflow._close_renderer_and_quit(viz, app)

    assert calls == ["renderer", "qt"]


def test_close_renderer_and_quit_always_quits_if_renderer_close_fails():
    calls = []

    def fail_close():
        calls.append("renderer")
        raise ValueError("close failed")

    viz = SimpleNamespace(renderer=SimpleNamespace(close=fail_close))
    app = SimpleNamespace(quit=lambda: calls.append("qt"))

    startup_workflow._close_renderer_and_quit(viz, app)

    assert calls == ["renderer", "qt"]
    assert viz._shutdown_failures == ("renderer",)


def test_loading_updater_aborts_after_nested_close_event(monkeypatch):
    viz = SimpleNamespace(
        _shutdown_started=False,
        _loading_label=SimpleNamespace(setText=Mock()),
    )

    def process_events():
        viz._shutdown_started = True

    monkeypatch.setattr(
        startup_ui,
        "QApplication",
        SimpleNamespace(processEvents=process_events),
    )
    update_loading = startup_ui.make_loading_updater(viz)

    with pytest.raises(RuntimeError, match="cancelled during shutdown"):
        update_loading("Creating services...")


def test_complete_cli_startup_stops_after_shutdown_event_pump():
    args = startup_workflow.parse_cli_args(["--no-resume"])
    viz = SimpleNamespace(
        _shutdown_started=False,
        _deferred_init=Mock(),
    )

    def process_events():
        viz._shutdown_started = True

    with pytest.raises(RuntimeError, match="cancelled during shutdown"):
        startup_workflow._complete_cli_startup(
            args=args,
            app=SimpleNamespace(processEvents=process_events),
            reporter=SimpleNamespace(note=Mock()),
            viz=viz,
        )


def test_batch_render_rejects_empty_plan_before_creating_output(tmp_path):
    output_dir = tmp_path / "frames"
    args = SimpleNamespace(render_frames=str(output_dir))
    viz = SimpleNamespace(
        total_animation_steps=0,
        frame_source=SimpleNamespace(list_frames=lambda: []),
    )

    with pytest.raises(RuntimeError, match="at least one available frame"):
        startup_workflow.start_batch_render_mode(
            args=args,
            app=SimpleNamespace(),
            reporter=SimpleNamespace(note=lambda _message: None),
            viz=viz,
        )

    assert not output_dir.exists()


def test_benchmark_mode_rejects_an_empty_frame_plan():
    args = startup_workflow.parse_cli_args(["--scenario", "scenarios/example", "--benchmark", "1"])
    viz = SimpleNamespace(
        total_animation_steps=0,
        frame_source=SimpleNamespace(list_frames=lambda: []),
        pipeline=SimpleNamespace(benchmark_recorder=None),
        renderer=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="at least one available frame"):
        startup_workflow.start_benchmark_mode(
            args=args,
            app=SimpleNamespace(),
            reporter=SimpleNamespace(),
            viz=viz,
            benchmark_state_overrides={},
        )


class _FakeTimerSignal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class _FakeTimer:
    instances = []
    scheduled = []

    def __init__(self):
        self.timeout = _FakeTimerSignal()
        self.stopped = False
        self.started = []
        self.single_shot = False
        self.instances.append(self)

    def setSingleShot(self, enabled):  # noqa: N802 - Qt-compatible test double
        self.single_shot = bool(enabled)

    def start(self, delay):
        self.started.append(delay)

    def stop(self):
        self.stopped = True

    @classmethod
    def singleShot(cls, delay, callback):  # noqa: N802 - Qt-compatible test double
        cls.scheduled.append((delay, callback))

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.scheduled = []


def test_batch_timer_exception_becomes_nonzero_cli_failure(tmp_path, monkeypatch):
    _FakeTimer.reset()
    monkeypatch.setattr(startup_workflow, "QTimer", _FakeTimer)
    output_dir = tmp_path / "frames"
    app = SimpleNamespace(processEvents=Mock(), exit=Mock(), quit=Mock())
    viz = SimpleNamespace(
        total_animation_steps=1,
        frame_source=SimpleNamespace(list_frames=lambda: [0]),
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(screenshot_export=True),
            export_screenshot=Mock(return_value=True),
        ),
        update_frame=Mock(side_effect=RuntimeError("update exploded")),
        force_update_next_frame=False,
    )

    startup_workflow.start_batch_render_mode(
        args=SimpleNamespace(render_frames=str(output_dir)),
        app=app,
        reporter=SimpleNamespace(note=Mock()),
        viz=viz,
    )
    timer = _FakeTimer.instances[0]
    assert timer.single_shot is True
    timer.timeout.callback()

    assert isinstance(viz._cli_run_failure, RuntimeError)
    assert "update exploded" in str(viz._cli_run_failure)
    app.exit.assert_called_once_with(1)
    assert timer.stopped is True


def test_benchmark_finalize_exception_becomes_nonzero_cli_failure(monkeypatch):
    class _Recorder:
        is_done = True

        def __init__(self, **_kwargs):
            pass

        def set_metadata(self, *_args):
            pass

        def finalize(self):
            raise OSError("cannot write benchmark")

    _FakeTimer.reset()
    monkeypatch.setattr(startup_workflow, "QTimer", _FakeTimer)
    monkeypatch.setattr(startup_workflow, "BenchmarkRecorder", _Recorder)
    app = SimpleNamespace(exit=Mock(), quit=Mock())
    viz = SimpleNamespace(
        total_animation_steps=1,
        frame_source=SimpleNamespace(list_frames=lambda: [0]),
        renderer=SimpleNamespace(capabilities=RendererCapabilities()),
        pipeline=SimpleNamespace(benchmark_recorder=None),
        scene_boot_duration_ms=None,
        get_startup_timing_profile=lambda: {},
        force_update_next_frame=False,
        _viewport_mode="detached",
        _window_layout=None,
    )
    args = SimpleNamespace(
        benchmark=1,
        benchmark_warmup=0,
        benchmark_output=None,
        renderer="pygfx",
        scenario="demo",
        benchmark_present_mode="blocking",
        max_performance=False,
        benchmark_state_json=None,
        benchmark_previsit_all_frames=False,
    )

    startup_workflow.start_benchmark_mode(
        args=args,
        app=app,
        reporter=SimpleNamespace(note=Mock()),
        viz=viz,
        benchmark_state_overrides={},
    )
    _FakeTimer.instances[0].timeout.callback()

    assert "cannot write benchmark" in str(viz._cli_run_failure)
    app.exit.assert_called_once_with(1)


def test_detached_viewport_metadata_uses_layout_logical_and_physical_sizes():
    layout = WindowLayout(
        qt_x=0,
        qt_y=0,
        qt_width=500,
        qt_height=720,
        renderer_x=508,
        renderer_y=0,
        renderer_logical_width=1280,
        renderer_logical_height=720,
        renderer_physical_width=1920,
        renderer_physical_height=1080,
        device_pixel_ratio=1.5,
    )
    viz = SimpleNamespace(
        _viewport_mode="detached",
        _window_layout=layout,
        renderer=SimpleNamespace(),
    )

    metadata = startup_workflow._viewport_metadata(viz)

    assert metadata == {
        "viewport_mode": "detached",
        "viewport_logical_width": 1280,
        "viewport_logical_height": 720,
        "viewport_device_pixel_ratio": 1.5,
        "viewport_physical_width": 1920,
        "viewport_physical_height": 1080,
    }


def test_restore_startup_workspace_uses_preselected_file_and_skips_camera():
    calls = []

    def _restore(path, **kwargs):
        calls.append((path, kwargs))
        return True

    selection = _startup_workspace_selection(
        "workspace.json",
        "scenarios/example",
        frame=17,
        camera={"format": "orbit-v1"},
        scenario_name="example_scene",
    )
    viz = SimpleNamespace(session_service=SimpleNamespace(load_session=_restore))
    notes = []
    reporter = SimpleNamespace(note=lambda message: notes.append(message))

    restored = startup_workflow.restore_startup_workspace(
        viz,
        selection,
        reporter=reporter,
    )

    assert restored == selection.path
    assert calls == [(selection, {"skip_camera": True})]
    assert notes == ["Resumed workspace: Example Scene, frame 17"]


def _write_startup_workspace(path, scenario_root, *, frame=0, camera=None):
    path.write_text(
        json.dumps(
            {
                "version": "6.0",
                "created_at": "2026-07-18T12:00:00",
                "snapshot_kind": "autosave",
                "scenario_path": str(scenario_root),
                "animation": {"current_frame": frame},
                "camera": camera,
            }
        ),
        encoding="utf-8",
    )


def test_select_startup_workspace_matches_folder_and_yaml_identity(tmp_path):
    scenario_root = tmp_path / "example"
    scenario_root.mkdir()
    scenario_yaml = scenario_root / "scenario.yaml"
    scenario_yaml.write_text("name: example", encoding="utf-8")
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    workspace = session_dir / "example_autosave_123.json"
    camera = {"format": "orbit-v1", "eye": [1, 2, 3]}
    _write_startup_workspace(workspace, scenario_yaml, frame=17, camera=camera)

    folder_selection = startup_workflow.select_startup_workspace(
        str(scenario_root), session_dir=session_dir
    )
    yaml_selection = startup_workflow.select_startup_workspace(
        str(scenario_yaml), session_dir=session_dir
    )

    assert folder_selection == yaml_selection
    assert folder_selection is not None
    assert folder_selection.path == workspace.resolve()
    assert folder_selection.scenario_root == scenario_root.resolve()
    assert folder_selection.frame == 17
    assert folder_selection.camera == camera


def test_select_startup_workspace_skips_newer_missing_scenario(tmp_path):
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    (valid_root / "scenario.yaml").write_text("name: valid", encoding="utf-8")
    missing_root = tmp_path / "missing"
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    valid = session_dir / "valid.json"
    stale = session_dir / "stale.json"
    _write_startup_workspace(valid, valid_root, frame=5)
    _write_startup_workspace(stale, missing_root, frame=99)
    os.utime(valid, (100, 100))
    os.utime(stale, (200, 200))

    selection = startup_workflow.select_startup_workspace(None, session_dir=session_dir)

    assert selection is not None
    assert selection.path == valid.resolve()
    assert selection.frame == 5


def test_select_startup_workspace_rejects_invalid_utf8(tmp_path):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("name: scenario", encoding="utf-8")
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    workspace = session_dir / "broken.json"
    workspace.write_bytes(b"\xff\xfe")

    assert startup_workflow.select_startup_workspace(None, session_dir=session_dir) is None


def test_select_startup_workspace_decodes_once_and_isolates_camera(tmp_path, monkeypatch):
    from visualizer.src.services import session_service

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("name: scenario", encoding="utf-8")
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    workspace = session_dir / "scenario.json"
    camera = {"format": "orbit-v1", "eye": [1, 2, 3]}
    _write_startup_workspace(workspace, scenario_root, frame=7, camera=camera)

    decode_count = 0
    decode_json = session_service.json.load

    def _count_decode(handle):
        nonlocal decode_count
        decode_count += 1
        return decode_json(handle)

    monkeypatch.setattr(session_service.json, "load", _count_decode)
    selection = startup_workflow.select_startup_workspace(None, session_dir=session_dir)

    assert selection is not None
    assert decode_count == 1
    pending_camera = selection.camera
    assert pending_camera == camera
    pending_camera["eye"][0] = 99
    assert selection.payload["camera"] == camera


def test_exec_with_deferred_cli_startup_enters_qt_before_heavy_work(monkeypatch):
    calls = []
    scheduled = []
    app = SimpleNamespace(
        exec=lambda: (
            calls.append("exec-enter"),
            scheduled.pop()(),
            calls.append("exec-return"),
            7,
        )[-1],
        quit=lambda: calls.append("quit"),
    )
    monkeypatch.setattr(
        startup_workflow,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, callback: scheduled.append(callback)),
    )
    monkeypatch.setattr(
        startup_workflow,
        "_complete_cli_startup",
        lambda **_kwargs: calls.append("startup"),
    )

    exit_code = startup_workflow._exec_with_deferred_cli_startup(
        args=SimpleNamespace(),
        app=app,
        reporter=SimpleNamespace(),
        viz=SimpleNamespace(),
    )

    assert exit_code == 7
    assert calls == ["exec-enter", "startup", "exec-return"]


def test_exec_with_deferred_cli_startup_reraises_callback_failure(monkeypatch):
    scheduled = []
    calls = []

    def fail_startup(**_kwargs):
        raise ValueError("startup failed")

    def exec_loop():
        scheduled.pop()()
        return 0

    def fail_renderer_close():
        calls.append("renderer")
        raise ValueError("secondary close failure")

    app = SimpleNamespace(exec=exec_loop, quit=lambda: calls.append("quit"))
    viz = SimpleNamespace(renderer=SimpleNamespace(close=fail_renderer_close))
    monkeypatch.setattr(
        startup_workflow,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, callback: scheduled.append(callback)),
    )
    monkeypatch.setattr(startup_workflow, "_complete_cli_startup", fail_startup)

    with pytest.raises(ValueError, match="startup failed"):
        startup_workflow._exec_with_deferred_cli_startup(
            args=SimpleNamespace(),
            app=app,
            reporter=SimpleNamespace(),
            viz=viz,
        )

    assert calls == ["renderer", "quit"]


def test_run_visualizer_cli_exits_one_for_handled_scenario_failure(monkeypatch):
    calls = []
    app = SimpleNamespace(processEvents=lambda: calls.append("process-events"))
    viz = SimpleNamespace(show=lambda: calls.append("show"))
    reporter = SimpleNamespace(note=lambda message: calls.append(("note", message)))
    monkeypatch.setattr(startup_workflow, "QApplication", lambda _argv: app)
    monkeypatch.setattr(startup_workflow, "apply_application_identity", lambda _app: None)

    def reject_startup(**_kwargs):
        raise startup_workflow._CliScenarioOpenError("live generator is busy")

    monkeypatch.setattr(
        startup_workflow,
        "_exec_with_deferred_cli_startup",
        reject_startup,
    )

    with pytest.raises(SystemExit) as exc_info:
        startup_workflow.run_visualizer_cli(
            lambda **_kwargs: viz,
            lambda **_kwargs: reporter,
            ["--scenario", "scenarios/example", "--no-resume"],
        )

    assert exc_info.value.code == 1
    assert calls[:2] == [
        ("note", "Bootstrapping Qt application"),
        ("note", "Using pygfx renderer"),
    ]
    assert "show" in calls
    assert "process-events" in calls


def test_complete_cli_startup_resumes_one_selected_workspace(monkeypatch):
    calls = []
    pending_camera = {"format": "orbit-v1"}
    selection = _startup_workspace_selection(
        "workspace.json",
        "scenarios/example",
        frame=17,
        camera=pending_camera,
    )
    args = startup_workflow.parse_cli_args(["--scenario", "scenarios/example"])
    viz = SimpleNamespace(
        _deferred_init=lambda **kwargs: calls.append(("deferred", kwargs)),
        current_scenario_path=None,
        cancel_scheduled_update=lambda: calls.append("cancel-update"),
        session_service=SimpleNamespace(
            load_session=lambda *args, **kwargs: calls.append(("load", args, kwargs)) or True
        ),
    )

    def open_scenario(*open_args, **kwargs):
        calls.append(("scenario", open_args, kwargs))
        viz.current_scenario_path = selection.scenario_root
        return SimpleNamespace(succeeded=True, scenario_root=selection.scenario_root)

    viz.open_scenario = open_scenario
    app = SimpleNamespace(processEvents=lambda: calls.append("process-events"))
    reporter = SimpleNamespace(
        note=lambda message: calls.append(("note", message)),
        task=lambda description: nullcontext(),
    )
    monkeypatch.setattr(
        startup_workflow,
        "select_startup_workspace",
        lambda _scenario: selection,
    )
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda _viz, _reporter, enabled: calls.append(("configure-cli", enabled)),
    )

    startup_workflow._complete_cli_startup(
        args=args,
        app=app,
        reporter=reporter,
        viz=viz,
    )

    assert ("deferred", {"pending_camera": pending_camera}) in calls
    assert ("configure-cli", False) in calls
    assert (
        "scenario",
        ("scenarios/example",),
        {"pending_camera": pending_camera, "autorun_initial_frame": False},
    ) in calls
    assert [call for call in calls if isinstance(call, tuple) and call[0] == "scenario"] == [
        (
            "scenario",
            ("scenarios/example",),
            {"pending_camera": pending_camera, "autorun_initial_frame": False},
        )
    ]
    assert ("load", (selection,), {"skip_camera": True}) in calls
    assert ("note", "Resumed workspace: Example, frame 17") in calls


def test_complete_cli_startup_rejects_failed_explicit_scenario_without_retry(monkeypatch):
    calls = []
    selection = _startup_workspace_selection(
        "workspace.json",
        "scenarios/example",
        frame=17,
        camera=None,
    )
    args = startup_workflow.parse_cli_args(["--scenario", "scenarios/example"])
    load_session = Mock(return_value=True)
    viz = SimpleNamespace(
        current_scenario_path=None,
        _deferred_init=lambda **_kwargs: None,
        open_scenario=lambda *open_args, **kwargs: (
            calls.append(("scenario", open_args, kwargs)),
            SimpleNamespace(succeeded=False, scenario_root=None, message="load failed"),
        )[-1],
        session_service=SimpleNamespace(load_session=load_session),
        schedule_update=Mock(),
    )
    reporter = SimpleNamespace(
        note=lambda message: calls.append(("note", message)),
        task=lambda _description: nullcontext(),
    )
    monkeypatch.setattr(startup_workflow, "select_startup_workspace", lambda _path: selection)
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(startup_workflow._CliScenarioOpenError, match="load failed"):
        startup_workflow._complete_cli_startup(
            args=args,
            app=SimpleNamespace(processEvents=lambda: None),
            reporter=reporter,
            viz=viz,
        )

    assert len([call for call in calls if call[0] == "scenario"]) == 1
    load_session.assert_not_called()
    viz.schedule_update.assert_not_called()
    assert viz._explicit_cli_scenario_startup is False
    assert ("note", "load failed") in calls


def test_complete_cli_startup_keeps_failed_automatic_workspace_nonfatal(monkeypatch):
    calls = []
    selection = _startup_workspace_selection(
        "workspace.json",
        "scenarios/example",
        frame=17,
        camera=None,
    )
    args = startup_workflow.parse_cli_args([])
    load_session = Mock(return_value=True)
    viz = SimpleNamespace(
        current_scenario_path=None,
        _deferred_init=lambda **_kwargs: None,
        open_scenario=lambda *open_args, **kwargs: (
            calls.append(("scenario", open_args, kwargs)),
            SimpleNamespace(succeeded=False, scenario_root=None, message="load failed"),
        )[-1],
        session_service=SimpleNamespace(load_session=load_session),
        schedule_update=Mock(),
    )
    reporter = SimpleNamespace(
        note=lambda message: calls.append(("note", message)),
        task=lambda _description: nullcontext(),
    )
    monkeypatch.setattr(startup_workflow, "select_startup_workspace", lambda _path: selection)
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda *_args, **_kwargs: None,
    )

    startup_workflow._complete_cli_startup(
        args=args,
        app=SimpleNamespace(processEvents=lambda: None),
        reporter=reporter,
        viz=viz,
    )

    assert len([call for call in calls if call[0] == "scenario"]) == 1
    load_session.assert_not_called()
    viz.schedule_update.assert_not_called()
    assert (
        "note",
        "Workspace resume skipped because its scenario did not open",
    ) in calls


def test_complete_cli_startup_without_workspace_runs_initial_frame(monkeypatch):
    calls = []
    args = startup_workflow.parse_cli_args(["--scenario", "scenarios/example"])
    viz = SimpleNamespace(
        _deferred_init=lambda **kwargs: calls.append(("deferred", kwargs)),
        open_scenario=lambda *args, **kwargs: (
            calls.append(("scenario", args, kwargs)),
            SimpleNamespace(succeeded=True, scenario_root=Path("scenarios/example")),
        )[-1],
    )
    app = SimpleNamespace(processEvents=lambda: None)
    reporter = SimpleNamespace(note=lambda _message: None, task=lambda _description: nullcontext())
    monkeypatch.setattr(startup_workflow, "select_startup_workspace", lambda _scenario: None)
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda *_args, **_kwargs: None,
    )

    startup_workflow._complete_cli_startup(args=args, app=app, reporter=reporter, viz=viz)

    assert calls == [
        ("deferred", {"pending_camera": None}),
        (
            "scenario",
            ("scenarios/example",),
            {"pending_camera": None, "autorun_initial_frame": True},
        ),
    ]


def test_complete_cli_startup_forwards_data_source_overrides(monkeypatch):
    calls = []
    args = startup_workflow.parse_cli_args(
        [
            "--scenario",
            "scenarios/example",
            "--data-mode",
            "remote_hdf5",
            "--grpc-port",
            "50053",
            "--no-resume",
        ]
    )
    viz = SimpleNamespace(
        _deferred_init=lambda **_kwargs: None,
        open_scenario=lambda *open_args, **kwargs: (
            calls.append((open_args, kwargs)),
            SimpleNamespace(succeeded=True, scenario_root=Path("scenarios/example")),
        )[-1],
    )
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda *_args, **_kwargs: None,
    )

    startup_workflow._complete_cli_startup(
        args=args,
        app=SimpleNamespace(processEvents=lambda: None),
        reporter=SimpleNamespace(note=lambda _message: None, task=lambda _label: nullcontext()),
        viz=viz,
    )

    assert calls == [
        (
            ("scenarios/example",),
            {
                "pending_camera": None,
                "autorun_initial_frame": True,
                "data_mode_override": "remote_hdf5",
                "grpc_port_override": 50053,
            },
        )
    ]


def test_complete_cli_startup_without_scenario_opens_selected_workspace_once(monkeypatch):
    calls = []
    selection = _startup_workspace_selection(
        "workspace.json",
        "scenarios/last-opened",
        frame=8,
        camera=None,
    )
    args = startup_workflow.parse_cli_args([])
    viz = SimpleNamespace(
        _deferred_init=lambda **kwargs: calls.append(("deferred", kwargs)),
        current_scenario_path=None,
        session_service=SimpleNamespace(
            load_session=lambda *args, **kwargs: calls.append(("load", args, kwargs)) or True
        ),
    )

    def open_scenario(*open_args, **kwargs):
        calls.append(("scenario", open_args, kwargs))
        viz.current_scenario_path = selection.scenario_root
        return SimpleNamespace(succeeded=True, scenario_root=selection.scenario_root)

    viz.open_scenario = open_scenario
    app = SimpleNamespace(processEvents=lambda: None)
    reporter = SimpleNamespace(note=lambda _message: None, task=lambda _description: nullcontext())
    monkeypatch.setattr(startup_workflow, "select_startup_workspace", lambda _scenario: selection)
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda *_args, **_kwargs: None,
    )

    startup_workflow._complete_cli_startup(args=args, app=app, reporter=reporter, viz=viz)

    assert [call for call in calls if call[0] == "scenario"] == [
        (
            "scenario",
            (str(selection.scenario_root),),
            {"pending_camera": None, "autorun_initial_frame": False},
        )
    ]
    assert ("load", (selection,), {"skip_camera": False}) in calls


def test_complete_cli_startup_preserves_benchmark_scenario_semantics(monkeypatch):
    calls = []
    args = startup_workflow.parse_cli_args(["--scenario", "scenarios/example", "--benchmark", "3"])
    viz = SimpleNamespace(
        _deferred_init=lambda **kwargs: calls.append(("deferred", kwargs)),
        open_scenario=lambda *args, **kwargs: (
            calls.append(("scenario", args, kwargs)),
            SimpleNamespace(succeeded=True, scenario_root=Path("scenarios/example")),
        )[-1],
        cancel_scheduled_update=lambda: calls.append("cancel-update"),
    )
    app = SimpleNamespace(processEvents=lambda: calls.append("process-events"))
    reporter = SimpleNamespace(
        note=lambda message: calls.append(("note", message)),
        task=lambda _description: nullcontext(),
    )
    monkeypatch.setattr(
        startup_workflow,
        "select_startup_workspace",
        lambda _scenario: pytest.fail("benchmark startup must not scan workspaces"),
    )
    monkeypatch.setattr(
        startup_workflow,
        "configure_cli_driven_frame_run",
        lambda _viz, _reporter, enabled: calls.append(("configure-cli", enabled)),
    )
    monkeypatch.setattr(
        startup_workflow,
        "start_benchmark_mode",
        lambda **kwargs: calls.append(("benchmark", kwargs["benchmark_state_overrides"])),
    )

    startup_workflow._complete_cli_startup(args=args, app=app, reporter=reporter, viz=viz)

    assert ("deferred", {"pending_camera": None}) in calls
    assert ("configure-cli", True) in calls
    assert (
        "scenario",
        ("scenarios/example",),
        {"pending_camera": None, "autorun_initial_frame": False},
    ) in calls
    assert "cancel-update" in calls
    assert ("benchmark", {}) in calls
