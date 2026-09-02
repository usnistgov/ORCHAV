"""Safety and typed-interaction checks for the display authoring smoke tool."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QDialog

from scripts import capture_scenario_builder_workflow as capture_script
from visualizer.src.authoring.interaction import InteractionSession
from visualizer.src.authoring.viewport_port import (
    ActorOverlaySnapshot,
    HitResult,
    OverlaySnapshot,
)


def test_capture_output_directory_must_not_overwrite_evidence(tmp_path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "prior.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        capture_script._prepare_output_directory(occupied)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_capture_output_directory_has_isolated_capture_root(
    tmp_path,
) -> None:
    output, captures = capture_script._prepare_output_directory(tmp_path / "evidence")

    assert output == (tmp_path / "evidence").resolve()
    assert captures == output / "captures"
    assert captures.is_dir()
    assert not (output / "scenario").exists()


def test_capture_work_directory_cleanup_requires_marker_ownership(tmp_path) -> None:
    repo_root = tmp_path / "checkout"
    work_root, scenario, token = capture_script._create_owned_work_scenario(repo_root)

    assert work_root.parent == (repo_root / "tmp").resolve()
    assert scenario == work_root / "scenario"
    capture_script._remove_owned_work_directory(
        work_root,
        token,
        repo_root=repo_root,
    )
    assert not work_root.exists()

    unowned = repo_root / "tmp" / "unowned"
    unowned.mkdir()
    with pytest.raises(RuntimeError, match="unowned"):
        capture_script._remove_owned_work_directory(
            unowned,
            token,
            repo_root=repo_root,
        )
    assert unowned.is_dir()


def test_capture_copies_relocatable_scenario_evidence(tmp_path) -> None:
    scenario = tmp_path / "work" / "scenario"
    frames = scenario / "frames"
    frames.mkdir(parents=True)
    (scenario / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    (frames / "mpc_frames_00000.h5").write_bytes(b"frame evidence")
    output = tmp_path / "external"
    output.mkdir()

    evidence = capture_script._copy_relocatable_scenario_evidence(scenario, output)

    assert evidence == {
        "scenario_yaml": "scenario/scenario.yaml",
        "frames_directory": "scenario/frames",
    }
    assert (output / Path(evidence["scenario_yaml"])).is_file()
    assert (output / Path(evidence["frames_directory"]) / "mpc_frames_00000.h5").is_file()


def test_capture_uses_direct_persistence_and_journals_required_stages() -> None:
    source = inspect.getsource(capture_script)

    assert "save_document(" in source
    assert "workspace_mode_controller._save_document" not in source
    for stage in ("save", "generation", "preview", "resume", "read_only"):
        assert f'"{stage}"' in source


def test_capture_rejects_and_records_unexpected_modal(qapp) -> None:
    class _UnexpectedTestDialog(QDialog):
        def text(self) -> str:
            return "blocking detail"

    captured = []
    guard = capture_script._UnexpectedModalGuard(captured.append)
    previous = capture_script._ACTIVE_MODAL_GUARD
    dialog = _UnexpectedTestDialog()
    dialog.setWindowTitle("Unexpected Test Dialog")
    qapp.installEventFilter(guard)
    capture_script._ACTIVE_MODAL_GUARD = guard
    try:
        dialog.show()
        with pytest.raises(RuntimeError, match="Unexpected modal dialog"):
            capture_script._pump_events(qapp)
    finally:
        capture_script._ACTIVE_MODAL_GUARD = previous
        qapp.removeEventFilter(guard)
        dialog.close()
        qapp.processEvents()

    assert captured == [
        {
            "title": "Unexpected Test Dialog",
            "text": "blocking detail",
        }
    ]


def test_native_event_is_dispatched_only_through_the_active_authoring_router() -> None:
    router = SimpleNamespace(
        active=True,
        session=InteractionSession.AUTHORING,
        _route_event=Mock(),
    )
    workspace = SimpleNamespace(
        viewport=SimpleNamespace(port=SimpleNamespace(router=router)),
        _on_viewport_input=Mock(),
    )
    event = capture_script._native_pointer_event(
        "pointer_down",
        (10.0, 20.0),
        button=1,
        buttons=(1,),
    )

    capture_script._route_native_event(workspace, event)

    router._route_event.assert_called_once_with(event)
    workspace._on_viewport_input.assert_not_called()


def test_capture_interactions_do_not_inject_workspace_typed_inputs_directly() -> None:
    source = inspect.getsource(capture_script)

    assert "_on_viewport_input" not in source


def test_native_event_rejects_an_inactive_or_non_authoring_router() -> None:
    workspace = SimpleNamespace(
        viewport=SimpleNamespace(
            port=SimpleNamespace(
                router=SimpleNamespace(
                    active=False,
                    session=InteractionSession.AUTHORING,
                )
            )
        )
    )

    with pytest.raises(RuntimeError, match="not active"):
        capture_script._route_native_event(
            workspace,
            capture_script._native_pointer_event("pointer_move", (0.0, 0.0)),
        )


def test_world_projection_uses_the_active_renderer_camera_matrices() -> None:
    runtime = SimpleNamespace(
        camera_matrices=Mock(
            return_value=(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        ),
        logical_size=(200.0, 100.0),
    )

    assert capture_script._project_world_to_screen(
        SimpleNamespace(runtime=runtime),
        (0.0, 0.0, 0.0),
    ) == pytest.approx((100.0, 50.0))


def test_work_plane_routing_requires_and_uses_the_visible_plane_height() -> None:
    runtime = SimpleNamespace(
        camera_matrices=Mock(
            return_value=(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        ),
        logical_size=(200.0, 100.0),
    )
    router = SimpleNamespace(
        active=True,
        session=InteractionSession.AUTHORING,
        runtime=runtime,
        resolve_hit=Mock(return_value=HitResult((0.0, 0.0, 1.0))),
        _route_event=Mock(),
    )
    plane = SimpleNamespace(value=Mock(return_value=1.0))
    workspace = SimpleNamespace(
        viewport=SimpleNamespace(port=SimpleNamespace(router=router)),
        work_plane_z_spin=plane,
    )

    capture_script._route_work_plane_pointer(
        workspace,
        "pointer_down",
        (0.0, 0.0, 1.0),
        button=1,
        buttons=(1,),
    )

    router._route_event.assert_called_once()
    plane.value.return_value = 0.0
    with pytest.raises(AssertionError, match="visible work plane"):
        capture_script._route_work_plane_pointer(
            workspace,
            "pointer_down",
            (0.0, 0.0, 1.0),
            button=1,
            buttons=(1,),
        )


def test_workspace_inventory_records_one_router_registration_set() -> None:
    port = SimpleNamespace(
        router=SimpleNamespace(handler_count=9),
        renderer_objects=Mock(return_value={"one": object(), "two": object()}),
    )
    workspace = SimpleNamespace(
        document=SimpleNamespace(
            revision=4,
            dirty=True,
            actors=(object(), object()),
            groups=(object(),),
            selected_subject=None,
        ),
        viewport=SimpleNamespace(port=port),
        compilation=SimpleNamespace(issues=()),
    )

    inventory = capture_script._workspace_inventory(workspace)

    assert inventory["renderer_object_count"] == 2
    assert inventory["handler_count"] == 9
    assert inventory["actor_count"] == 2
    assert inventory["group_count"] == 1


def test_capture_overlay_selection_uses_the_workspace_controls() -> None:
    combos = [Mock() for _ in range(5)]
    for combo in combos:
        combo.findData.return_value = 2
    workspace = SimpleNamespace(
        trajectory_visibility_combo=combos[0],
        frame_samples_visibility_combo=combos[1],
        control_rig_visibility_combo=combos[2],
        orientation_axes_combo=combos[3],
        look_at_rays_combo=combos[4],
    )

    capture_script._select_overlay_options(workspace)

    for combo in combos:
        combo.setCurrentIndex.assert_called_once_with(2)


def test_actor_overlay_reads_the_real_port_snapshot_boundary() -> None:
    wanted = uuid4()
    overlay = ActorOverlaySnapshot(
        actor_id=wanted,
        role="rx",
        name="RX1",
        positions=((0.0, 0.0, 0.0),),
    )
    snapshot = OverlaySnapshot(
        document_id=uuid4(),
        revision=3,
        actors=(overlay,),
    )
    port = SimpleNamespace(current_snapshot=Mock(return_value=snapshot))
    workspace = SimpleNamespace(viewport=SimpleNamespace(port=port))

    assert capture_script._actor_overlay(workspace, wanted) is overlay
    port.current_snapshot.assert_called_once_with()


def test_generated_preview_readiness_requires_frame_submission_and_packet() -> None:
    renderer = SimpleNamespace(last_frame_packet=object())
    visualizer = SimpleNamespace(
        vis_initialized=True,
        animation_step=4,
        renderer=renderer,
        get_available_animation_steps=Mock(return_value=[3, 4]),
        update_frame=Mock(return_value=True),
    )

    assert capture_script._try_render_generated_frame(visualizer) == 4
    visualizer.update_frame.assert_called_once_with(4)

    renderer.last_frame_packet = None
    assert capture_script._try_render_generated_frame(visualizer) is None


def test_hdf5_evidence_requires_promoted_frame_chunks(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    first = frames / "mpc_frames_0000.h5"
    first.write_bytes(b"\x89HDF\r\n\x1a\npayload")

    evidence = capture_script._hdf5_frame_evidence(frames)

    assert evidence["chunk_count"] == 1
    assert evidence["chunks"][0]["path"] == first.name
    assert evidence["chunks"][0]["byte_size"] == len(first.read_bytes())

    first.write_bytes(b"not hdf5")
    with pytest.raises(AssertionError, match="not HDF5"):
        capture_script._hdf5_frame_evidence(frames)


def test_capture_image_evidence_is_nonblank_and_states_must_differ(tmp_path) -> None:
    first_path = tmp_path / "first.png"
    first_image = QImage(16, 16, QImage.Format.Format_RGBA8888)
    first_image.fill(QColor(20, 30, 40))
    assert first_image.save(str(first_path))

    second_path = tmp_path / "second.png"
    second_image = QImage(16, 16, QImage.Format.Format_RGBA8888)
    second_image.fill(QColor(20, 30, 40))
    second_image.setPixelColor(8, 8, QColor(200, 50, 10))
    assert second_image.save(str(second_path))

    first = capture_script._image_evidence(first_path)
    second = capture_script._image_evidence(second_path)
    capture_script._assert_capture_images_differ(
        {
            "label": "first",
            "viewport_image_evidence": first,
        },
        {
            "label": "second",
            "viewport_image_evidence": second,
        },
    )

    assert first["width"] == 16
    assert first["height"] == 16
    assert first["sha256"] != second["sha256"]


def test_capture_image_evidence_rejects_a_black_image(tmp_path) -> None:
    image_path = tmp_path / "black.png"
    image = QImage(8, 8, QImage.Format.Format_RGBA8888)
    image.fill(QColor(0, 0, 0))
    assert image.save(str(image_path))

    with pytest.raises(AssertionError, match="visually blank"):
        capture_script._image_evidence(image_path)


def test_exact_structured_problem_assertion_checks_code_path_and_message() -> None:
    issue = SimpleNamespace(
        code="actors.rx.required",
        path="actors.rx",
        message="At least one RX is required.",
    )
    compilation = SimpleNamespace(issues=(issue,))

    capture_script._assert_problem(
        compilation,
        code="actors.rx.required",
        path="actors.rx",
        message="At least one RX is required.",
    )

    with pytest.raises(AssertionError, match="missing exact validation problem"):
        capture_script._assert_problem(
            compilation,
            code="actors.rx.required",
            path="actors.rx",
            message="different",
        )
