"""Recorded inspection and editable acquisition variants keep separate ownership."""

import threading

import pytest
import yaml

from shared.frames.frame_set_writer import FrameSetWriter
from tests.visualizer.authoring.test_persistence import _external_mapping
from tests.visualizer.unit.test_scenario_authoring_workspace import _FixedCompiler, _workspace
from tests.visualizer.unit.test_trajectory_load_service import _MemoryFileSource
from visualizer.src.authoring.compiler import CompilationResult
from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.persistence import load_for_authoring
from visualizer.src.authoring.recording import RecordingPreview
from visualizer.src.authoring.recording_workflow import (
    RecordingVariantJob,
    acquisition_scenario_path,
)
from visualizer.src.authoring.viewport_port import PreviewProvenance, TrajectoryDisplayMode


def _recorded_document(root):
    mapping = _external_mapping()
    mapping["actors"]["tx"][0]["name"] = "tx_0"
    mapping["actors"]["rx"][0]["name"] = "rx_0"
    mapping["actors"]["targets"] = [
        {
            "name": "pedestrian",
            "asset": {"source": "catalog", "id": "cube"},
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
        }
    ]
    steps = list(range(7, 199, 3))
    mapping["timeline"] = {"steps": len(steps), "duration_s": 3}
    source = _MemoryFileSource(steps)
    writer = FrameSetWriter.create_new(root / "frames", compression=None)
    for step in steps:
        writer.append(source.load_frame(step))
    writer.finalize()
    path = root / "scenario.yaml"
    path.write_text("# Saved frames own target poses.\n" + yaml.safe_dump(mapping))
    return load_for_authoring(path).document, steps


def test_recorded_paths_reach_both_views_in_frame_order_and_survive_scrubbing(tmp_path, qtbot):
    document, steps = _recorded_document(tmp_path)
    recording = RecordingPreview(document.scenario, tmp_path, None)
    result = CompilationResult({}, "", (), {}, recording=recording, fully_validated=False)
    workspace = _workspace(document, _FixedCompiler(result))
    qtbot.addWidget(workspace)
    workspace.refresh_now()
    qtbot.waitUntil(lambda: recording.trajectory_loader.is_complete, timeout=5000)
    assert workspace.recording_banner.isHidden() is False
    assert "Generation limited" not in workspace.preview_status_label.text()
    assert "Recording ready" in workspace.preview_status_label.text()
    for actor in document.actors:
        document.select(actor.id)
        for step in (0, len(steps) // 2, len(steps) - 1):
            workspace.timeline_slider.setValue(step)
            overlay = next(
                a for a in workspace.viewport.port.snapshots[-1].actors if a.actor_id == actor.id
            )
            expected = recording.trajectory_positions[actor.id]
            assert len(expected) == len(steps)
            assert tuple(p[0] for p in expected) == tuple(float(s) for s in steps)
            assert overlay.positions == workspace.trajectory_preview.committed == expected
            assert overlay.current_position == expected[step]
            assert overlay.trajectory_display is TrajectoryDisplayMode.PATH
            assert overlay.preview_provenance is PreviewProvenance.RECORDED_FRAMES
    assert len(recording._frames) <= 3
    original = recording.trajectory_loader
    document.select(document.actors[0].id)
    assert recording.trajectory_loader is original
    workspace.bind_document(ScenarioDocument.new())
    before = workspace.viewport.port.snapshots[-1]
    workspace._recorded_trajectories_changed(recording)
    assert workspace.viewport.port.snapshots[-1] is before
    assert original.is_shutdown
    workspace.close_workspace()


def test_retired_recording_cancels_scan_and_closes_only_its_owned_source(
    tmp_path, qtbot, monkeypatch
):
    import visualizer.src.authoring.recording as module

    document, _ = _recorded_document(tmp_path)
    recording = RecordingPreview(document.scenario, tmp_path, None)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    source = _MemoryFileSource([0, 1], entered=entered, release=release)
    source.close = closed.set
    monkeypatch.setattr(module, "FileSource", lambda *args: source)
    updates = []
    recording.start_trajectories(lambda: updates.append(True))
    assert entered.wait(1)
    recording.close()
    release.set()
    qtbot.waitUntil(closed.is_set, timeout=3000)
    assert updates == []
    assert recording.trajectory_positions == {}


def _acquisition(root):
    root.mkdir()
    mapping = _external_mapping()
    (root / "scene.xml").write_text('<scene version="3.0.0"><shape type="cube"/></scene>')
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    mapping["sensing"] = {"enabled": False}
    path = root / "scenario.yaml"
    path.write_text(yaml.safe_dump(mapping))
    (root / "frames").mkdir()
    (root / "frames" / "saved-output.bin").write_bytes(b"original recording")
    return path


def test_variant_job_copies_inputs_without_touching_or_copying_recorded_outputs(tmp_path, qtbot):
    source = _acquisition(tmp_path / "acquisition")
    before = source.read_bytes()
    target = tmp_path / "variant"
    job = RecordingVariantJob()
    errors, completed = [], []
    job.failed.connect(errors.append)
    job.succeeded.connect(completed.append)
    job.start(source, target)
    qtbot.waitUntil(lambda: bool(completed or errors), timeout=10000)
    assert not errors
    assert not completed[0].read_only
    assert completed[0].path == target / "scenario.yaml"
    assert source.read_bytes() == before
    assert (source.parent / "frames" / "saved-output.bin").read_bytes() == b"original recording"
    assert not (target / "frames").exists()
    assert load_for_authoring(target).editable
    assert yaml.safe_load((target / "scenario.yaml").read_text())["sensing"] == {"enabled": False}


@pytest.mark.parametrize("inside", [True, False])
def test_variant_job_refuses_original_or_nonempty_destinations(tmp_path, qtbot, inside):
    source = _acquisition(tmp_path / "acquisition")
    destination = source.parent / "variant" if inside else tmp_path / "occupied"
    if not inside:
        destination.mkdir()
        (destination / "keep.txt").write_text("keep")
    job = RecordingVariantJob()
    errors = []
    job.failed.connect(errors.append)
    job.start(source, destination)
    qtbot.waitUntil(lambda: bool(errors), timeout=3000)
    assert "directory" in errors[0]
    assert not (destination / "scenario.yaml").exists()


def test_acquisition_discovery_uses_recording_directory_without_rewriting_profile(tmp_path):
    source = _acquisition(tmp_path / "acquisition")
    mapping = _external_mapping()
    mapping["data"] = {"files": {"directory": "acquisition/frames"}}
    path = tmp_path / "scenario.yaml"
    path.write_text("# Saved frames own target poses.\n" + yaml.safe_dump(mapping))
    before = path.read_bytes()
    document = load_for_authoring(path).document
    assert acquisition_scenario_path(document) == source
    assert path.read_bytes() == before
    source.unlink()
    assert acquisition_scenario_path(document) is None
