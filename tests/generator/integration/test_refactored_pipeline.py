import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from generator.core.configuration import SimulationConfig
from generator.core.pipeline import perform_offline_pipeline
from generator.io.storage.summary_publication import SummaryPublicationError


@contextmanager
def _capture_logger(caplog, name: str) -> Iterator[None]:
    """Capture one logger independently of process-wide logging configuration."""
    target = logging.getLogger(name)
    original_level = target.level
    original_propagate = target.propagate
    original_disabled = target.disabled
    handler_was_present = caplog.handler in target.handlers
    if not handler_was_present:
        target.addHandler(caplog.handler)
    target.setLevel(logging.INFO)
    target.propagate = False
    target.disabled = False
    try:
        yield
    finally:
        target.disabled = original_disabled
        target.propagate = original_propagate
        target.setLevel(original_level)
        if not handler_was_present:
            target.removeHandler(caplog.handler)


@pytest.fixture
def mock_simulation_config():
    cfg = MagicMock(spec=SimulationConfig)
    cfg.output_mode = "file"
    cfg.num_steps = 2
    cfg.start_step = 0
    cfg.duration = 1.0
    cfg.grpc_config = {}
    cfg.coverage = None
    cfg.cir_time_steps = 1
    cfg.get_quality_profile.return_value = {}
    return cfg


@pytest.fixture(autouse=True)
def mock_coverage_publication(monkeypatch):
    """Keep orchestration tests focused on service ordering and failure flow."""
    publication_cls = MagicMock(name="CoveragePublication")
    monkeypatch.setattr(
        "generator.core.pipeline.offline_pipeline.CoveragePublication",
        publication_cls,
    )
    return publication_cls


def _scenario_config(*, raytracing_enabled: bool) -> MagicMock:
    """Return a pipeline config mock with private/export-optional features off."""
    cfg = MagicMock()
    cfg.raytracing = {"enabled": raytracing_enabled}
    # MagicMock creates truthy attributes on demand, so keep sensing explicit.
    cfg.sensing = {}
    cfg.generator_summary = {}
    cfg.coverage_cfg = {}
    return cfg


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.SensingPipelineState")
@patch("generator.core.pipeline.offline_pipeline.SummaryPublication")
@patch("generator.figures.summary.maybe_generate_generator_summary")
@pytest.mark.parametrize(
    (
        "start_step",
        "num_steps",
        "duration",
        "expected_steps",
        "expected_frame_dt_s",
    ),
    (
        (0, 2, 1.0, [0, 1], 1.0),
        (3, 5, 4.0, [3, 4], 1.0),
    ),
    ids=("complete-range", "fresh-partial-range"),
)
def test_perform_offline_pipeline_orchestration(
    mock_summary,
    mock_summary_publication_cls,
    mock_sensing_state_cls,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
    caplog,
    start_step,
    num_steps,
    duration,
    expected_steps,
    expected_frame_dt_s,
    mock_coverage_publication,
):
    mock_simulation_config.start_step = start_step
    mock_simulation_config.num_steps = num_steps
    mock_simulation_config.duration = duration

    # Setup mocks
    mock_scene_service = mock_scene_cls.return_value
    mock_actor_state_service = mock_actor_state_cls.return_value
    mock_rt_service = mock_rt_cls.return_value
    mock_cov_service = mock_cov_cls.return_value
    mock_output = mock_hdf5_frame_output_cls.return_value
    mock_sensing_state = mock_sensing_state_cls.from_config.return_value
    mock_summary_publication = mock_summary_publication_cls.return_value
    mock_summary_publication.active = True
    mock_summary_publication.staging_directory = Path("summary-staging")
    mock_coverage = mock_coverage_publication.return_value

    # Set __name__ for PipelineContext.get_service lookup
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"

    # Setup returns
    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_service.prepare_actor_state.return_value = (MagicMock(), MagicMock())
    mock_rt_service.compute_step.return_value = {"data": "test"}
    mock_cov_service.compute_coverage.return_value = None
    lifecycle = MagicMock()

    def record_static_summary(*_args, **_kwargs):
        lifecycle.static_summary()
        return []

    mock_summary.side_effect = record_static_summary
    mock_sensing_state.generate_summary.side_effect = (
        lambda *_args, **_kwargs: lifecycle.sensing_summary()
    )
    mock_summary_publication.finalize.side_effect = lambda: lifecycle.publish_summary()
    mock_coverage.finalize.side_effect = lambda _manifest: lifecycle.publish_coverage()

    def finalize_output():
        lifecycle.finalize_frames()
        return "test output"

    mock_output.finalize.side_effect = finalize_output

    # Setup input
    tx_configs = [MagicMock()]
    rx_configs = [MagicMock()]
    target_configs = []

    # Scenario config with RT enabled
    scenario_cfg = _scenario_config(raytracing_enabled=True)

    # Execute
    with _capture_logger(caplog, "orchav.generator.core.pipeline.offline_pipeline"):
        perform_offline_pipeline(
            tx_configs,
            rx_configs,
            target_configs,
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    # Verification

    # 1. Services Instantiated
    mock_scene_cls.assert_called_with(mock_simulation_config)
    mock_actor_state_cls.assert_called_with(mock_simulation_config)
    mock_rt_cls.assert_called_with(mock_simulation_config)

    # 2. Actor-state step normalization
    mock_actor_state_service.normalize_scene_steps.assert_called_with(
        tx_configs, rx_configs, target_configs
    )

    # 3. Scene Built
    mock_scene_service.build_scene.assert_called_with(tx_configs, rx_configs, target_configs)
    mock_output.begin.assert_called_once_with()

    # 4. Actor state prepared
    mock_actor_state_service.prepare_actor_state.assert_called()

    # 5. RayTracing Prepared
    mock_rt_service.prepare_simulation.assert_called()

    # 6. Only the configured absolute frame range is computed.
    assert mock_rt_service.compute_step.call_count == len(expected_steps)
    mock_rt_service.compute_step.assert_has_calls([call(step) for step in expected_steps])

    # 7. Output Saved
    assert mock_output.save_frame_data.call_count == len(expected_steps)
    assert [
        output_call.args[0] for output_call in mock_output.save_frame_data.call_args_list
    ] == expected_steps
    mock_output.finalize.assert_called_once()
    mock_output.abort.assert_called_once()
    assert lifecycle.mock_calls == [
        call.static_summary(),
        call.finalize_frames(),
        call.publish_coverage(),
        call.sensing_summary(),
        call.publish_summary(),
    ]
    mock_sensing_state.generate_summary.assert_called_once_with(
        scenario_cfg,
        output_root=Path("summary-staging"),
        strict=True,
    )

    # 8. Coverage Computed
    coverage_call = mock_cov_service.compute_coverage.call_args
    assert coverage_call.args == (mock_scene_service, scenario_cfg)
    assert coverage_call.kwargs.keys() == {"publication"}
    assert coverage_call.kwargs["publication"] is mock_coverage
    mock_coverage.finalize.assert_called_once_with(mock_output.published_manifest)

    # 9. A partial frame set retains the complete scenario's per-step cadence.
    sensing_kwargs = mock_sensing_state_cls.from_config.call_args.kwargs
    assert sensing_kwargs["first_step"] == start_step
    assert sensing_kwargs["frame_dt_s"] == expected_frame_dt_s

    if start_step > 0:
        assert (
            f"Computing fresh partial frame set: {len(expected_steps)} steps "
            f"({expected_steps[0]} through {expected_steps[-1]})"
        ) in caplog.text
        assert "resum" not in caplog.text.lower()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.get_material_tuning_adapter")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_perform_offline_pipeline_material_tuning_failure_raises(
    mock_summary,
    mock_material_adapter_factory,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_service = mock_scene_cls.return_value
    mock_actor_state_service = mock_actor_state_cls.return_value
    mock_rt_service = mock_rt_cls.return_value
    mock_cov_service = mock_cov_cls.return_value

    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"

    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_service.prepare_actor_state.return_value = (MagicMock(), MagicMock())
    mock_cov_service.compute_coverage.return_value = None

    adapter = MagicMock()
    adapter.build_config.return_value = SimpleNamespace(enabled=True)
    adapter.run.side_effect = RuntimeError("adapter failed")
    mock_material_adapter_factory.return_value = adapter

    scenario_cfg = _scenario_config(raytracing_enabled=False)
    scenario_cfg.coverage_cfg = {}

    with pytest.raises(RuntimeError, match="adapter failed"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    adapter.run.assert_called_once_with(
        mock_rt_service.simulation_objects,
        adapter.build_config.return_value,
    )


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.get_material_tuning_adapter")
def test_coverage_and_material_tuning_are_rejected_before_output(
    mock_material_adapter_factory,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"
    mock_material_adapter_factory.return_value.build_config.return_value = SimpleNamespace(
        enabled=True
    )
    scenario_cfg = _scenario_config(raytracing_enabled=True)
    scenario_cfg.coverage_cfg = {"enabled": True}

    with pytest.raises(ValueError, match="cannot be enabled together"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    mock_hdf5_frame_output_cls.assert_not_called()
    mock_scene_cls.return_value.build_scene.assert_not_called()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_coverage_plus_rt_failure_aborts_without_finalizing_outputs(
    mock_summary,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
    mock_coverage_publication,
):
    mock_scene_service = mock_scene_cls.return_value
    mock_actor_state_service = mock_actor_state_cls.return_value
    mock_rt_service = mock_rt_cls.return_value
    mock_cov_service = mock_cov_cls.return_value
    mock_output = mock_hdf5_frame_output_cls.return_value

    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"

    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_service.prepare_actor_state.return_value = (MagicMock(), MagicMock())
    mock_cov_service.compute_coverage.return_value = "private-coverage.h5"
    mock_rt_service.compute_step.side_effect = [{"paths": MagicMock()}, None]

    tx_configs = [MagicMock()]
    rx_configs = [MagicMock()]
    target_configs = []
    scenario_cfg = _scenario_config(raytracing_enabled=True)
    scenario_cfg.coverage_cfg = {"enabled": True}

    with pytest.raises(RuntimeError, match="Failed to compute frame 1"):
        perform_offline_pipeline(
            tx_configs,
            rx_configs,
            target_configs,
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    assert mock_output.save_frame_data.call_count == 1
    mock_output.finalize.assert_not_called()
    mock_output.abort.assert_called_once()
    publication = mock_coverage_publication.return_value
    publication.finalize.assert_not_called()
    publication.abort.assert_called_once()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_output_lock_failure_stops_before_scene_construction(
    mock_summary,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"
    mock_output = mock_hdf5_frame_output_cls.return_value
    mock_output.begin.side_effect = RuntimeError("destination already locked")

    scenario_cfg = _scenario_config(raytracing_enabled=True)
    with pytest.raises(RuntimeError, match="already locked"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    mock_scene_cls.return_value.build_scene.assert_not_called()
    mock_output.abort.assert_called_once_with()


@patch("generator.core.pipeline.offline_pipeline.PipelineContext")
def test_summary_only_generation_rejects_reader_selected_frames_before_context(
    mock_pipeline_context,
    mock_simulation_config,
    tmp_path,
):
    selected_frames = tmp_path / "derived_frames"
    scenario_cfg = SimpleNamespace(
        root=tmp_path,
        project_root=tmp_path,
        frames_dir=selected_frames,
        frames_directory="derived_frames",
        raytracing={"enabled": False},
        coverage_cfg={},
        generator_summary={
            "enabled": True,
            "create": ["scene2d"],
            "output": {"dir": tmp_path / "summary", "dirs": {}},
        },
        scene_xml=None,
        actors=None,
        sensing={},
    )

    with pytest.raises(ValueError, match="data.files.directory is a read-only selection"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    mock_pipeline_context.assert_not_called()
    assert not (tmp_path / "summary").exists()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.SummaryPublication")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_rt_disabled_run_does_not_publish_an_empty_sensing_summary(
    mock_summary,
    mock_summary_publication_cls,
    _mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"
    mock_scene_cls.return_value.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_cls.return_value.prepare_actor_state.return_value = (
        MagicMock(),
        MagicMock(),
    )
    publication = mock_summary_publication_cls.return_value
    publication.active = True
    publication.sensing_requested = True
    publication.staging_directory = Path("summary-staging")
    scenario_cfg = _scenario_config(raytracing_enabled=False)
    scenario_cfg.sensing = {"enabled": True}
    scenario_cfg.generator_summary = {"enabled": True, "create": ["sensing"]}

    with pytest.raises(SummaryPublicationError, match="Summary generation failed"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    publication.fail.assert_called()
    publication.finalize.assert_not_called()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_coverage_output_failure_aborts_the_pipeline(
    mock_summary,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_service = mock_scene_cls.return_value
    mock_actor_state_service = mock_actor_state_cls.return_value
    mock_coverage_service = mock_cov_cls.return_value
    mock_output = mock_hdf5_frame_output_cls.return_value

    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"

    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_service.prepare_actor_state.return_value = (MagicMock(), MagicMock())
    mock_coverage_service.compute_coverage.side_effect = PermissionError(
        "coverage destination is read-only"
    )

    scenario_cfg = _scenario_config(raytracing_enabled=False)
    scenario_cfg.coverage_cfg = {"enabled": True}

    with pytest.raises(PermissionError, match="read-only"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=scenario_cfg,
        )

    mock_output.finalize.assert_not_called()
    mock_output.abort.assert_called_once()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.SummaryPublication")
@patch("generator.figures.summary.maybe_generate_generator_summary")
def test_summary_failure_keeps_successfully_committed_frames(
    mock_summary,
    mock_summary_publication_cls,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"

    mock_scene_service = mock_scene_cls.return_value
    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_cls.return_value.prepare_actor_state.return_value = (
        MagicMock(),
        MagicMock(),
    )
    mock_rt_cls.return_value.compute_step.return_value = {"data": "test"}
    mock_cov_cls.return_value.compute_coverage.return_value = None
    mock_output = mock_hdf5_frame_output_cls.return_value
    mock_output.finalize.return_value = "committed frames"
    publication = mock_summary_publication_cls.return_value
    publication.active = True
    publication.staging_directory = Path("summary-staging")
    mock_summary.side_effect = OSError("plot backend failed")

    with pytest.raises(SummaryPublicationError, match="Frames committed; summary failed"):
        perform_offline_pipeline(
            [MagicMock()],
            [MagicMock()],
            [],
            mock_simulation_config,
            scenario_configuration=_scenario_config(raytracing_enabled=True),
            show_progress=False,
        )

    assert mock_output.save_frame_data.call_count == 2
    mock_output.finalize.assert_called_once_with()
    assert publication.fail.call_count >= 1
    publication.finalize.assert_not_called()


@patch("generator.core.pipeline.offline_pipeline.SceneService")
@patch("generator.core.pipeline.offline_pipeline.ActorStateService")
@patch("generator.core.pipeline.offline_pipeline.RayTracingService")
@patch("generator.core.pipeline.offline_pipeline.CoverageService")
@patch("generator.core.pipeline.offline_pipeline.HDF5FrameOutputStrategy")
@patch("generator.core.pipeline.offline_pipeline.SensingPipelineState")
def test_injected_frame_writer_never_mutates_canonical_coverage(
    mock_sensing_state_cls,
    mock_hdf5_frame_output_cls,
    mock_cov_cls,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
    mock_coverage_publication,
    tmp_path,
):
    mock_simulation_config.num_steps = 1
    mock_simulation_config.duration = 0.0
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"
    mock_cov_cls.__name__ = "CoverageService"
    mock_scene_cls.return_value.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_cls.return_value.prepare_actor_state.return_value = (
        MagicMock(),
        MagicMock(),
    )
    mock_rt_cls.return_value.compute_step.return_value = {"data": "derived"}
    mock_cov_cls.return_value.compute_coverage.return_value = None
    mock_hdf5_frame_output_cls.return_value.finalize.return_value = "derived frames"
    mock_sensing_state_cls.from_config.return_value = MagicMock()

    canonical_coverage = tmp_path / "coverage" / "coverage_maps.h5"
    canonical_coverage.parent.mkdir()
    canonical_coverage.write_bytes(b"canonical coverage")
    scenario_cfg = _scenario_config(raytracing_enabled=True)
    scenario_cfg.root = tmp_path
    injected_writer = MagicMock()

    perform_offline_pipeline(
        [MagicMock()],
        [MagicMock()],
        [],
        mock_simulation_config,
        scenario_configuration=scenario_cfg,
        frame_set_writer=injected_writer,
        show_progress=False,
    )

    mock_hdf5_frame_output_cls.assert_called_once_with(
        mock_simulation_config,
        scenario_cfg,
        frame_set_writer=injected_writer,
    )
    coverage_call = mock_cov_cls.return_value.compute_coverage.call_args
    assert coverage_call.kwargs == {"publication": None}
    mock_coverage_publication.assert_not_called()
    assert canonical_coverage.read_bytes() == b"canonical coverage"
