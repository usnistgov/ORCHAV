from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from visualizer.src.app import scenario_workflow
from visualizer.src.app.scenario_workflow import ScenarioOpenStatus
from visualizer.src.services.scenario_loader_service import ScenarioLoadResult


class _FakeProgressReporter:
    def __init__(self):
        self.updates: list[tuple[int, str]] = []
        self.close_count = 0

    def update(self, step: int, message: str) -> None:
        self.updates.append((step, message))

    def close(self) -> None:
        self.close_count += 1


class _FakeProgress:
    @contextmanager
    def task(self, _description: str):
        yield


_DEFAULT_PREFLIGHT = object()


def _make_open_scenario_viz(
    load_result,
    calls: list,
    *,
    preflight_result=_DEFAULT_PREFLIGHT,
):
    class _Loader:
        def preflight_scenario(self, scenario_path, **overrides):
            call = ("preflight_scenario", scenario_path)
            calls.append((*call, overrides) if overrides else call)
            if isinstance(preflight_result, BaseException):
                raise preflight_result
            return preflight_result

        def load_scenario(
            self,
            scenario_path,
            *,
            cleanup_scene_first=True,
            preflight=None,
        ):
            calls.append(("load_scenario", scenario_path, cleanup_scene_first))
            assert preflight is preflight_result
            if isinstance(load_result, BaseException):
                raise load_result
            if callable(load_result):
                return load_result()
            return load_result

    class _UIManager:
        panels = {}

        def ensure_panel(self, key):
            calls.append(("ensure_panel", key))

        def set_panel_visible(self, key, visible):
            calls.append(("panel_visible", key, visible))

    viz = SimpleNamespace(
        cancel_startup_preload=lambda: calls.append("cancel_preload"),
        reset_preloading_state=lambda: calls.append("reset_preload"),
        scene_service=SimpleNamespace(
            cleanup_previous_scene=lambda: calls.append("scene_cleanup"),
            render_scene=lambda: calls.append("render_scene"),
        ),
        live_preview_service=SimpleNamespace(reset=lambda: calls.append("live_preview_reset")),
        scenario_config=object(),
        _scene_only_mode=False,
        _scene_boot_logged=True,
        _scene_boot_start=None,
        frame_loader=SimpleNamespace(invalidate=lambda: calls.append("teardown_loader")),
        _set_frame_data_available=lambda available: calls.append(("frame_data", available)),
        status_progress_bar=None,
        ui_manager=_UIManager(),
        reset_startup_timing_profile=lambda: calls.append("reset_timing"),
        record_startup_stage_timing=lambda name, _value: calls.append(("timing", name)),
        scenario_loader_service=_Loader(),
        visual_profile_service=SimpleNamespace(
            load_scenario_rules=lambda rules: calls.append(("visual_profiles", rules))
        ),
        sensing_service=None,
        set_state=lambda **changes: calls.append(("set_state", sorted(changes))),
        progress=_FakeProgress(),
        coverage_service=SimpleNamespace(
            reset_runtime_state=lambda _viz: calls.append("coverage_reset"),
            load_coverage_map=lambda root, _viz: calls.append(("coverage", root)) or True,
        ),
        node_service=SimpleNamespace(
            discover_available_tx_rx=lambda: calls.append("discover_txrx"),
        ),
        target_service=SimpleNamespace(load_target_models=lambda: calls.append("load_targets")),
        camera_controller=SimpleNamespace(
            update_target_focus_dropdown=lambda: calls.append("target_dropdown")
        ),
        cache_service=SimpleNamespace(
            invalidate=lambda *args, **kwargs: calls.append("invalidate")
        ),
        animation_step=0,
        get_animation_step_index=lambda _step: 0,
        ready=False,
        request_startup_preload=lambda: calls.append("request_preload"),
        force_update_next_frame=False,
        schedule_update=lambda: calls.append("schedule_update"),
        cancel_scheduled_update=lambda: calls.append("cancel_update"),
        frame_source=None,
        ui_controller=SimpleNamespace(
            configure_trajectory_checkboxes=lambda enabled: calls.append(
                ("trajectory_checkboxes", enabled)
            ),
            populate_material_filters=lambda: calls.append("material_filters"),
            add_recent_file=lambda path: calls.append(("recent_file", path)),
        ),
        trajectory_load_coordinator=SimpleNamespace(
            supports_source=lambda source: source is not None,
            reset=lambda: calls.append("cleanup_trajectory"),
        ),
        _set_status_message=lambda message, *args: calls.append(("status", message, args)),
        _apply_view_defaults=lambda defaults, **kwargs: calls.append(
            ("view_defaults", defaults, kwargs)
        ),
        current_scenario_path=None,
        current_scenario_policy=None,
        current_project_root=None,
        current_base_dir=None,
        scenario=None,
        mpc_core=SimpleNamespace(
            set_frame_source=lambda source: calls.append(("mpc_frame_source", source))
        ),
    )

    return viz


def test_panel_enabled_reads_bool_dict_and_defaults():
    viz = SimpleNamespace(
        scenario_config=SimpleNamespace(
            visualizer_cfg={
                "panels": {
                    "sensing": {"enabled": False},
                    "mpc": True,
                }
            }
        )
    )

    assert scenario_workflow.panel_enabled(viz, "sensing", default=True) is False
    assert scenario_workflow.panel_enabled(viz, "mpc", default=False) is True
    assert scenario_workflow.panel_enabled(viz, "coverage", default=True) is True


def test_start_scenario_statistics_binds_source_before_preload_completion():
    calls = []
    frame_source = object()
    statistics_panel = SimpleNamespace(
        set_statistics_source=lambda source: calls.append(source) or True
    )
    viz = SimpleNamespace(
        pipeline=SimpleNamespace(benchmark_recorder=None),
        scenario_config=SimpleNamespace(
            visualizer_cfg={"panels": {"statistics": {"enabled": True}}}
        ),
        ui_manager=SimpleNamespace(panels={"statistics": statistics_panel}),
    )

    assert scenario_workflow._start_scenario_statistics(viz, frame_source) is True
    assert calls == [frame_source]


def test_start_scenario_statistics_skips_benchmark_and_disabled_panel():
    calls = []
    statistics_panel = SimpleNamespace(
        set_statistics_source=lambda source: calls.append(source) or True
    )
    viz = SimpleNamespace(
        pipeline=SimpleNamespace(benchmark_recorder=object()),
        scenario_config=SimpleNamespace(visualizer_cfg={}),
        ui_manager=SimpleNamespace(panels={"statistics": statistics_panel}),
    )

    assert scenario_workflow._start_scenario_statistics(viz, object()) is False
    viz.pipeline.benchmark_recorder = None
    viz.scenario_config.visualizer_cfg = {"panels": {"statistics": {"enabled": False}}}
    assert scenario_workflow._start_scenario_statistics(viz, object()) is False
    assert calls == []


def test_cleanup_previous_scene_resets_app_state():
    calls = []

    class ProgressBar:
        def __init__(self):
            self.value = None
            self.visible = True

        def setValue(self, value):
            self.value = value

        def setVisible(self, visible):
            self.visible = visible

    progress_bar = ProgressBar()
    ui_manager = SimpleNamespace(
        set_panel_visible=lambda panel, visible: calls.append(("panel", panel, visible))
    )
    viz = SimpleNamespace(
        cancel_startup_preload=lambda: calls.append("cancel_preload"),
        reset_preloading_state=lambda: calls.append("reset_preload"),
        scene_service=SimpleNamespace(cleanup_previous_scene=lambda: calls.append("scene_cleanup")),
        live_preview_service=SimpleNamespace(reset=lambda: calls.append("live_preview_reset")),
        scenario_config=object(),
        _scene_only_mode=True,
        frame_loader=SimpleNamespace(invalidate=lambda: calls.append("teardown_loader")),
        trajectory_load_coordinator=SimpleNamespace(
            reset=lambda: calls.append("cleanup_trajectory")
        ),
        _set_frame_data_available=lambda available: calls.append(("frame_data", available)),
        status_progress_bar=progress_bar,
        ui_manager=ui_manager,
        coverage_service=SimpleNamespace(
            reset_runtime_state=lambda _viz: calls.append("coverage_reset")
        ),
        _mpc_presented_source_epoch=4,
        _mpc_explorer_session=SimpleNamespace(
            on_scenario_teardown=lambda: calls.append("mpc_explorer_teardown")
        ),
    )

    scenario_workflow.cleanup_previous_scene(viz)

    assert viz.scenario_config is None
    assert viz._scene_only_mode is False
    assert viz._scene_boot_logged is False
    assert viz._scene_boot_start is None
    assert viz._mpc_presented_source_epoch == 5
    assert progress_bar.value == 0
    assert progress_bar.visible is False
    assert calls == [
        "cancel_preload",
        "reset_preload",
        "coverage_reset",
        "mpc_explorer_teardown",
        "cleanup_trajectory",
        "scene_cleanup",
        "live_preview_reset",
        "teardown_loader",
        ("frame_data", False),
    ]


def test_open_scenario_reports_frame_backed_progress_order():
    calls: list = []
    progress = _FakeProgressReporter()
    frame_source = SimpleNamespace(
        list_frames=lambda: [0],
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg={},
    )
    viz = _make_open_scenario_viz(ScenarioLoadResult(scenario, frame_source, object(), True), calls)

    outcome = scenario_workflow.open_scenario(
        viz,
        "scenario.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert progress.updates == [
        (1, "Validating scenario..."),
        (2, "Cleaning up previous scene..."),
        (3, "Loading scenario configuration..."),
        (4, "Loading coverage data..."),
        (5, "Discovering TX/RX metadata..."),
        (6, "Loading target meshes..."),
        (7, "Rendering initial scene..."),
        (8, "Populating material filters..."),
    ]
    assert progress.close_count == 1
    assert outcome.status is ScenarioOpenStatus.SUCCEEDED
    assert outcome.succeeded is True
    assert outcome.frame_source_ready is True
    assert outcome.scenario_root == Path("scenario-root").resolve()
    assert ("status", "Scenario loaded successfully", (5000,)) in calls
    assert viz.current_scenario_path == str(Path("scenario-root").resolve())
    assert calls.count(("mpc_frame_source", None)) == 1


def test_open_scenario_passes_runtime_data_source_overrides_to_preflight():
    calls: list = []
    progress = _FakeProgressReporter()
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg={},
    )
    viz = _make_open_scenario_viz(
        ScenarioLoadResult(scenario, None, object(), False),
        calls,
    )

    outcome = scenario_workflow.open_scenario(
        viz,
        "scenario.yaml",
        data_mode_override="live_grpc",
        grpc_port_override=50052,
        _progress_factory=lambda _viz: progress,
    )

    assert outcome.succeeded is True
    assert (
        "preflight_scenario",
        "scenario.yaml",
        {
            "data_mode_override": "live_grpc",
            "grpc_port_override": 50052,
        },
    ) in calls


def test_open_scenario_records_canonical_root_only_after_success(tmp_path):
    calls: list = []
    progress = _FakeProgressReporter()
    scenario_root = tmp_path / "scenario-one"
    scenario_root.mkdir()
    frame_source = SimpleNamespace(
        list_frames=lambda: [0],
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root=scenario_root,
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg=None,
    )
    viz = _make_open_scenario_viz(ScenarioLoadResult(scenario, frame_source, object(), True), calls)

    scenario_workflow.open_scenario(
        viz,
        "scenario-one/scenario.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert [call for call in calls if isinstance(call, tuple) and call[0] == "recent_file"] == [
        ("recent_file", str(scenario_root.resolve()))
    ]


def test_open_scenario_installs_visual_profiles_before_target_and_scene_materials():
    calls: list = []
    progress = _FakeProgressReporter()
    rules = [{"match": {"name": "Pedestrian", "type": "target"}, "preset": "Skin"}]
    frame_source = SimpleNamespace(
        list_frames=lambda: [0],
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={"visual_profiles": rules},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg=None,
    )
    viz = _make_open_scenario_viz(ScenarioLoadResult(scenario, frame_source, object(), True), calls)

    scenario_workflow.open_scenario(
        viz,
        "profiled.yaml",
        _progress_factory=lambda _viz: progress,
    )

    profile_index = calls.index(("visual_profiles", rules))
    assert profile_index < calls.index("load_targets")
    assert profile_index < calls.index("render_scene")


def test_open_scenario_selects_sparse_initial_frame_before_first_render_update():
    calls: list = []
    progress = _FakeProgressReporter()
    frame_queries = 0

    def list_frames():
        nonlocal frame_queries
        frame_queries += 1
        return [42, 17, 90]

    frame_source = SimpleNamespace(
        list_frames=list_frames,
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg=None,
    )
    viz = _make_open_scenario_viz(ScenarioLoadResult(scenario, frame_source, object(), True), calls)

    scenario_workflow.open_scenario(
        viz,
        "sparse.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert frame_queries == 1
    assert viz.animation_step == 17
    assert ("set_state", ["step"]) in calls


def test_configure_visual_profiles_clears_rules_for_next_scenario():
    calls: list[list[dict]] = []
    viz = SimpleNamespace(
        visual_profile_service=SimpleNamespace(
            load_scenario_rules=lambda rules: calls.append(rules)
        )
    )

    scenario_workflow.configure_visual_profiles(
        viz,
        SimpleNamespace(view_defaults={"visual_profiles": [{"preset": "Skin"}]}),
    )
    scenario_workflow.configure_visual_profiles(
        viz,
        SimpleNamespace(view_defaults={}),
    )

    assert calls == [[{"preset": "Skin"}], []]


def test_open_scenario_reports_scene_only_progress_order():
    calls: list = []
    progress = _FakeProgressReporter()
    frame_source = SimpleNamespace(supports_reconstruction_type_color=False)
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg={},
    )
    viz = _make_open_scenario_viz(
        ScenarioLoadResult(scenario, frame_source, object(), False), calls
    )

    outcome = scenario_workflow.open_scenario(
        viz,
        "scene-only.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert progress.updates == [
        (1, "Validating scenario..."),
        (2, "Cleaning up previous scene..."),
        (3, "Loading scenario configuration..."),
        (4, "Loading coverage data..."),
        (5, "Rendering scene (no frame data)..."),
    ]
    assert progress.close_count == 1
    assert outcome.status is ScenarioOpenStatus.SUCCEEDED
    assert outcome.frame_source_ready is False
    assert viz._scene_only_mode is True
    assert ("load_scenario", "scene-only.yaml", False) in calls
    assert ("coverage", "scenario-root") in calls
    assert any(
        call[0] == "status" and call[1].startswith("Scene loaded")
        for call in calls
        if isinstance(call, tuple)
    )


def test_open_scenario_closes_progress_when_scenario_load_raises():
    calls: list = []
    progress = _FakeProgressReporter()
    viz = _make_open_scenario_viz(RuntimeError("load failed"), calls)

    with pytest.raises(RuntimeError, match="load failed"):
        scenario_workflow.open_scenario(
            viz,
            "bad-scenario.yaml",
            _progress_factory=lambda _viz: progress,
        )

    assert progress.updates == [
        (1, "Validating scenario..."),
        (2, "Cleaning up previous scene..."),
        (3, "Loading scenario configuration..."),
    ]
    assert progress.close_count == 1
    assert viz.scenario is None
    assert viz.frame_source is None
    assert viz.current_scenario_policy is None
    assert viz.ready is False
    assert not any(isinstance(call, tuple) and call[0] == "recent_file" for call in calls)


def test_open_scenario_clears_loading_guard_when_progress_creation_fails():
    calls: list = []
    viz = _make_open_scenario_viz(AssertionError("load must not run"), calls)

    def fail_progress(_viz):
        raise RuntimeError("progress construction failed")

    with pytest.raises(RuntimeError, match="progress construction failed"):
        scenario_workflow.open_scenario(
            viz,
            "scenario.yaml",
            _progress_factory=fail_progress,
        )

    assert viz._scenario_load_in_progress is False
    assert not any(isinstance(call, tuple) and call[0] == "preflight_scenario" for call in calls)


def test_open_scenario_progress_close_failure_does_not_wedge_loading_guard():
    calls: list = []
    viz = _make_open_scenario_viz(
        AssertionError("commit must not run"),
        calls,
        preflight_result=None,
    )

    class _FailingCloseReporter(_FakeProgressReporter):
        def close(self):
            raise RuntimeError("dialog was deleted")

    outcome = scenario_workflow.open_scenario(
        viz,
        "invalid.yaml",
        _progress_factory=lambda _viz: _FailingCloseReporter(),
    )

    assert outcome.status is ScenarioOpenStatus.FAILED
    assert viz._scenario_load_in_progress is False


def test_open_scenario_closes_progress_when_scenario_load_returns_none():
    calls: list = []
    progress = _FakeProgressReporter()
    viz = _make_open_scenario_viz(None, calls)

    outcome = scenario_workflow.open_scenario(
        viz,
        "missing-scenario.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert progress.updates == [
        (1, "Validating scenario..."),
        (2, "Cleaning up previous scene..."),
        (3, "Loading scenario configuration..."),
    ]
    assert progress.close_count == 1
    assert outcome.status is ScenarioOpenStatus.FAILED
    assert outcome.succeeded is False
    assert ("status", "Failed to load scenario: missing-scenario.yaml", (5000,)) in calls
    assert viz.current_scenario_path is None
    assert viz.scenario is None
    assert viz.frame_source is None
    assert viz.current_scenario_policy is None
    assert viz.ready is False
    assert not any(isinstance(call, tuple) and call[0] == "recent_file" for call in calls)


def test_open_scenario_preflight_failure_preserves_active_scenario():
    calls: list = []
    progress = _FakeProgressReporter()
    viz = _make_open_scenario_viz(
        AssertionError("commit must not run"),
        calls,
        preflight_result=None,
    )
    active_scenario = object()
    active_config = object()
    active_source = object()
    active_policy = object()
    viz.scenario = active_scenario
    viz.scenario_config = active_config
    viz.frame_source = active_source
    viz.current_scenario_path = "active-scenario"
    viz.current_scenario_policy = active_policy
    viz.ready = True

    outcome = scenario_workflow.open_scenario(
        viz,
        "invalid-scenario.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert outcome.status is ScenarioOpenStatus.FAILED
    assert progress.updates == [(1, "Validating scenario...")]
    assert progress.close_count == 1
    assert viz.scenario is active_scenario
    assert viz.scenario_config is active_config
    assert viz.frame_source is active_source
    assert viz.current_scenario_path == "active-scenario"
    assert viz.current_scenario_policy is active_policy
    assert viz.ready is True
    assert "scene_cleanup" not in calls
    assert not any(isinstance(call, tuple) and call[0] == "load_scenario" for call in calls)


def test_open_scenario_source_preparation_failure_preserves_active_scenario():
    calls: list = []
    progress = _FakeProgressReporter()
    viz = _make_open_scenario_viz(AssertionError("commit must not run"), calls)
    active_scenario = object()
    active_config = object()
    active_source = object()
    active_policy = object()
    viz.scenario = active_scenario
    viz.scenario_config = active_config
    viz.frame_source = active_source
    viz.current_scenario_path = "active-scenario"
    viz.current_scenario_policy = active_policy
    viz.ready = True
    viz.vis_initialized = True
    viz.scenario_loader_service.prepare_frame_source = lambda _preflight: None
    viz.scenario_loader_service.last_frame_source_preparation_error = (
        "Failed to prepare scenario data source: live generator is busy"
    )

    outcome = scenario_workflow.open_scenario(
        viz,
        "unavailable-remote.yaml",
        _progress_factory=lambda _viz: progress,
    )

    assert outcome.status is ScenarioOpenStatus.FAILED
    assert outcome.message == "Failed to prepare scenario data source: live generator is busy"
    assert progress.updates == [(1, "Validating scenario...")]
    assert progress.close_count == 1
    assert viz.scenario is active_scenario
    assert viz.scenario_config is active_config
    assert viz.frame_source is active_source
    assert viz.current_scenario_path == "active-scenario"
    assert viz.current_scenario_policy is active_policy
    assert viz.ready is True
    assert "scene_cleanup" not in calls
    assert not any(isinstance(call, tuple) and call[0] == "load_scenario" for call in calls)


def test_open_scenario_can_succeed_after_partial_load_failure():
    calls: list = []
    attempts = 0
    frame_source = SimpleNamespace(
        list_frames=lambda: [3],
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg=None,
    )

    def next_result():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            viz.scenario = object()
            viz.frame_source = object()
            viz.current_scenario_policy = object()
            return None
        return ScenarioLoadResult(scenario, frame_source, object(), True)

    viz = _make_open_scenario_viz(next_result, calls)

    scenario_workflow.open_scenario(
        viz,
        "retry.yaml",
        _progress_factory=lambda _viz: _FakeProgressReporter(),
    )
    assert viz.scenario is None
    assert viz.frame_source is None
    assert viz.current_scenario_policy is None

    scenario_workflow.open_scenario(
        viz,
        "retry.yaml",
        _progress_factory=lambda _viz: _FakeProgressReporter(),
    )

    assert attempts == 2
    assert viz.scenario_config is scenario
    assert viz.frame_source is frame_source
    assert viz.current_scenario_path == str(Path("scenario-root").resolve())
    assert viz.ready is True


def test_open_scenario_rolls_back_when_scene_synchronization_fails():
    calls: list = []
    progress = _FakeProgressReporter()
    frame_source = SimpleNamespace(
        list_frames=lambda: [0],
        supports_reconstruction_type_color=False,
    )
    scenario = SimpleNamespace(
        root="scenario-root",
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg=None,
    )
    viz = _make_open_scenario_viz(
        ScenarioLoadResult(scenario, frame_source, object(), True),
        calls,
    )

    def fail_scene_sync():
        raise RuntimeError("scene synchronization failed")

    viz.scene_service.render_scene = fail_scene_sync

    with pytest.raises(RuntimeError, match="scene synchronization failed"):
        scenario_workflow.open_scenario(
            viz,
            "sync-failure.yaml",
            _progress_factory=lambda _viz: progress,
        )

    assert progress.close_count == 1
    assert viz.current_scenario_path is None
    assert viz.scenario is None
    assert viz.frame_source is None
    assert viz.ready is False
    assert not any(isinstance(call, tuple) and call[0] == "recent_file" for call in calls)
    assert ("status", "Scenario loaded successfully", (5000,)) not in calls
