"""Qt-thread bridge tests for Scenario Builder generation jobs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.domain import AuthoringScenario
from visualizer.src.authoring.generation import (
    GenerationPaths,
    GenerationResult,
    GenerationState,
)
from visualizer.src.authoring.generation_controller import QtGenerationController
from visualizer.src.authoring.undo import QtUndoStackAdapter


class _FakeJob:
    def __init__(
        self,
        scenario_yaml,
        launched_revision,
        *,
        revision_provider,
        on_progress,
        on_log,
        on_finished,
    ):
        self.scenario_yaml = Path(scenario_yaml)
        self.launched_revision = launched_revision
        self.revision_provider = revision_provider
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_finished = on_finished
        self.state = GenerationState.PENDING
        self.cancelled = False
        self.waited = False

    def start(self):
        self.state = GenerationState.RUNNING
        return self

    def cancel(self):
        if self.state is not GenerationState.RUNNING:
            return False
        self.cancelled = True
        return True

    def wait(self):
        self.waited = True
        self.state = GenerationState.CANCELLED
        return SimpleNamespace(state=self.state)


def test_generation_controller_saves_first_marshals_callbacks_and_guards_conflicts(
    qapp,
) -> None:
    document = ScenarioDocument.new(undo_stack=QtUndoStackAdapter())
    scenario_path = Path(__file__).parent / "authoring-generation" / "scenario.yaml"
    workspace = SimpleNamespace(
        set_generation_running=Mock(),
        append_generation_log=Mock(),
        set_generation_progress=Mock(),
        set_generation_result=Mock(),
    )

    def save() -> bool:
        document.mark_saved(scenario_path)
        return True

    controller = QtGenerationController(
        save_callback=Mock(side_effect=save),
        workspace_provider=lambda: workspace,
        job_factory=_FakeJob,
    )

    assert controller.start(document) is True
    controller._save_callback.assert_called_once_with()
    assert controller.running is True
    assert controller.start(document) is False

    progress = SimpleNamespace(total_steps=30, completed_steps=7, elapsed_s=1.5)
    controller.job.on_progress(progress)
    controller.job.on_log("stderr", "diagnostic")
    qapp.processEvents()
    workspace.set_generation_progress.assert_called_once_with(progress)
    workspace.append_generation_log.assert_any_call("[stderr] diagnostic")

    paths = GenerationPaths(
        scenario_yaml=scenario_path,
        snapshot_yaml=scenario_path.with_name("snapshot.yaml"),
        final_frames=scenario_path.parent / "frames",
    )
    result = GenerationResult(
        state=GenerationState.SUCCEEDED,
        stale=False,
        launched_revision=0,
        current_revision=0,
        returncode=0,
        paths=paths,
        progress=None,
        events=(),
        stdout_log=(),
        stderr_log=(),
    )
    controller.job.state = GenerationState.SUCCEEDED
    controller.job.on_finished(result)
    qapp.processEvents()
    assert controller.running is False
    assert controller.preview_path == scenario_path.parent
    workspace.set_generation_result.assert_called_once_with(result)

    document.add_default_actor("tx")
    assert controller.last_result.stale is True
    assert controller.last_result.current_revision == 1
    assert workspace.set_generation_result.call_count == 2
    assert any(
        "older draft revision" in call.args[0]
        for call in workspace.append_generation_log.call_args_list
    )


def test_generation_controller_cancel_forwards_to_active_job(qapp) -> None:
    document = ScenarioDocument.new(undo_stack=QtUndoStackAdapter())
    document.mark_saved(Path(__file__).parent / "authoring-generation" / "scenario.yaml")
    workspace = SimpleNamespace(
        set_generation_running=Mock(),
        append_generation_log=Mock(),
    )
    save_callback = Mock(return_value=True)
    controller = QtGenerationController(
        save_callback=save_callback,
        workspace_provider=lambda: workspace,
        job_factory=_FakeJob,
    )
    assert controller.start(document) is True
    save_callback.assert_called_once_with()
    assert controller.cancel() is True
    assert controller.job.cancelled is True
    controller.shutdown()
    assert controller.job.waited is True


def test_generation_controller_revalidates_clean_owned_file_before_snapshot(
    qapp, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("tampered: true\n", encoding="utf-8")
    document = ScenarioDocument.loaded(AuthoringScenario(), scenario_path)
    assert document.dirty is False
    constructed_jobs: list[_FakeJob] = []

    class _SnapshotReadingJob(_FakeJob):
        snapshot_text = ""

        def start(self):
            self.snapshot_text = self.scenario_yaml.read_text(encoding="utf-8")
            return super().start()

    def save() -> bool:
        scenario_path.write_text(
            "visualizer:\n  scenario_builder:\n    document_version: 1\n",
            encoding="utf-8",
        )
        return True

    def job_factory(*args, **kwargs):
        job = _SnapshotReadingJob(*args, **kwargs)
        constructed_jobs.append(job)
        return job

    controller = QtGenerationController(
        save_callback=Mock(side_effect=save),
        workspace_provider=lambda: None,
        job_factory=job_factory,
    )

    assert controller.start(document) is True
    controller._save_callback.assert_called_once_with()
    assert len(constructed_jobs) == 1
    assert "scenario_builder" in constructed_jobs[0].snapshot_text
    assert "tampered" not in constructed_jobs[0].snapshot_text
    assert constructed_jobs[0].launched_revision == document.revision


def test_generation_controller_does_not_construct_job_when_validated_save_fails(
    qapp, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    document = ScenarioDocument.loaded(AuthoringScenario(), scenario_path)
    job_factory = Mock()
    controller = QtGenerationController(
        save_callback=Mock(return_value=False),
        workspace_provider=lambda: None,
        job_factory=job_factory,
    )

    assert controller.start(document) is False
    controller._save_callback.assert_called_once_with()
    job_factory.assert_not_called()


def test_generation_controller_replays_stale_result_progress_and_logs_once(
    qapp, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    document = ScenarioDocument.new(undo_stack=QtUndoStackAdapter())
    active_workspace = SimpleNamespace(
        document=document,
        reset_generation_state=Mock(),
        set_generation_running=Mock(),
        append_generation_log=Mock(),
        set_generation_progress=Mock(),
        set_generation_result=Mock(),
    )

    def save() -> bool:
        document.mark_saved(scenario_path)
        return True

    controller = QtGenerationController(
        save_callback=save,
        workspace_provider=lambda: active_workspace,
        job_factory=_FakeJob,
    )
    assert controller.start(document) is True
    progress = SimpleNamespace(total_steps=4, completed_steps=3, elapsed_s=1.25)
    controller.job.on_progress(progress)
    controller.job.on_log("stdout", "generated step")
    controller.job.on_log("stderr", "diagnostic")
    qapp.processEvents()

    paths = GenerationPaths(
        scenario_yaml=scenario_path,
        snapshot_yaml=scenario_path.with_name("snapshot.yaml"),
        final_frames=scenario_path.parent / "frames",
    )
    result = GenerationResult(
        state=GenerationState.SUCCEEDED,
        stale=False,
        launched_revision=document.revision,
        current_revision=document.revision,
        returncode=0,
        paths=paths,
        progress=progress,
        events=(),
        stdout_log=("generated step",),
        stderr_log=("diagnostic",),
    )
    controller.job.state = GenerationState.SUCCEEDED
    controller.job.on_finished(result)
    qapp.processEvents()
    document.add_default_actor("tx")
    assert controller.last_result is not None
    assert controller.last_result.stale is True

    prior_active_log_calls = active_workspace.append_generation_log.call_count
    controller.bind_workspace(active_workspace)
    controller.bind_workspace(active_workspace)
    assert active_workspace.append_generation_log.call_count == prior_active_log_calls

    resumed_workspace = SimpleNamespace(
        document=document,
        reset_generation_state=Mock(),
        set_generation_running=Mock(),
        append_generation_log=Mock(),
        set_generation_progress=Mock(),
        set_generation_result=Mock(),
    )
    controller.bind_workspace(resumed_workspace)

    resumed_workspace.set_generation_progress.assert_called_once_with(progress)
    resumed_workspace.set_generation_result.assert_called_once_with(controller.last_result)
    replayed = [call.args[0] for call in resumed_workspace.append_generation_log.call_args_list]
    assert replayed == [
        f"Starting generation from saved document revision {result.launched_revision}.",
        "[stdout] generated step",
        "[stderr] diagnostic",
        "Generated result now corresponds to an older draft revision: "
        f"it was launched from revision {result.launched_revision}, while "
        f"the draft is revision {document.revision}.",
    ]
    assert [entry.stream_name for entry in controller.log_entries] == [
        "status",
        "stdout",
        "stderr",
        "status",
    ]

    controller.bind_workspace(resumed_workspace)
    assert resumed_workspace.append_generation_log.call_count == len(replayed)


def test_generation_controller_does_not_replay_result_into_unrelated_document(
    qapp, tmp_path: Path
) -> None:
    generated_document = ScenarioDocument.new(undo_stack=QtUndoStackAdapter())
    scenario_path = tmp_path / "scenario.yaml"

    def save() -> bool:
        generated_document.mark_saved(scenario_path)
        return True

    controller = QtGenerationController(
        save_callback=save,
        workspace_provider=lambda: None,
        job_factory=_FakeJob,
    )
    assert controller.start(generated_document) is True
    unrelated = SimpleNamespace(
        document=ScenarioDocument.new(),
        reset_generation_state=Mock(),
        set_generation_running=Mock(),
        append_generation_log=Mock(),
        set_generation_progress=Mock(),
        set_generation_result=Mock(),
    )

    controller.bind_workspace(unrelated)

    unrelated.reset_generation_state.assert_called_once_with()
    unrelated.set_generation_running.assert_not_called()
    unrelated.set_generation_result.assert_not_called()
    unrelated.append_generation_log.assert_not_called()
