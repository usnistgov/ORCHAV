"""Runnable evidence tests for the animated-target workflow."""

from __future__ import annotations

import ast
import builtins
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path

import pytest

import shared.animated_target_workflow as workflow_module
from shared.animated_target import (
    animated_target_revision_sha256,
    canonical_json_bytes,
    load_animated_target,
)
from shared.animated_target_workflow import (
    ANIMATED_TARGET_WORKFLOW_VERSION,
    AnimatedTargetWorkflowError,
    default_synthetic_binding_context,
    default_synthetic_binding_times,
    publish_animated_target_workflow,
    publish_synthetic_animated_target_workflow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_asset_redistribution(manifest_path: Path, redistribution: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["license"]["redistribution"] = redistribution
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_synthetic_workflow_publishes_exact_pulse_to_geometry_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic_workflow"
    result = publish_synthetic_animated_target_workflow(output)

    assert result.output_dir == output.resolve()
    assert result.asset_id == "orchav:synthetic:asymmetric_motion_workflow_v1"
    assert len(result.asset_revision_sha256) == 64
    assert len(result.binding_id) == 64
    assert len(result.workflow_id) == 64
    assert _sha256(output / "artifact_manifest.json") == result.artifact_manifest_sha256
    assert {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    } == {
        "alignment_and_geometry.png",
        "artifact_manifest.json",
        "asset/animated_target.json",
        "asset/poses/a_asymmetric_settle.json",
        "asset/poses/m_left_reach.json",
        "asset/poses/z_neutral_stance.json",
        "binding.json",
        "binding_request.json",
        "binding_samples.csv",
    }

    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert artifact_manifest["workflow_manifest_version"] == ANIMATED_TARGET_WORKFLOW_VERSION
    assert artifact_manifest["workflow_id"] == result.workflow_id
    assert artifact_manifest["binding_id"] == result.binding_id
    assert artifact_manifest["binding_context"] == default_synthetic_binding_context().to_dict()
    assert artifact_manifest["source_policy"] == "included_orchav_authored_synthetic"
    assert artifact_manifest["output_governance"] == {
        "source_redistribution": "allowed",
        "derived_artifact_distribution": ("manifest_allows_redistribution_pending_release_review"),
        "release_review_required": True,
    }
    assert artifact_manifest["runtime"]["matplotlib_backend"] == "Agg"
    assert "runtime versions" in artifact_manifest["runtime"]["png_portability"]
    for artifact in artifact_manifest["artifacts"]:
        artifact_path = output / artifact["path"]
        assert artifact_path.stat().st_size == artifact["size_bytes"]
        assert _sha256(artifact_path) == artifact["sha256"]
        if artifact["path"].endswith(".png"):
            assert artifact["portability"] == "runtime_qualified"
        else:
            assert artifact["portability"] == "portable_exact"

    binding = json.loads((output / "binding.json").read_text())
    binding_request = json.loads((output / "binding_request.json").read_text())
    expected_times = default_synthetic_binding_times()
    assert binding_request == {
        "contract": "orchav.animated_target_binding_request",
        "contract_version": 1,
        "context": default_synthetic_binding_context().to_dict(),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "scenario_time_s": sample.scenario_time_s,
                "motion_time_s": sample.motion_time_s,
                "pulse_time_s": sample.pulse_time_s,
            }
            for sample in expected_times
        ],
    }
    assert binding["binding_id"] == result.binding_id
    assert binding["context"] == default_synthetic_binding_context().to_dict()
    assert len(binding["samples"]) == len(expected_times) == 9
    for expected, sample in zip(expected_times, binding["samples"]):
        assert sample["sample_id"] == expected.sample_id
        assert sample["scenario_time_s"] == expected.scenario_time_s
        assert sample["motion_time_s"] == expected.motion_time_s
        assert sample["pulse_time_s"] == expected.pulse_time_s
        assert sample["validity"] == {"valid": True, "reason": None}
        assert len(sample["root"]["resolved_affine_4x4"]) == 4
        assert all(len(row) == 4 for row in sample["root"]["resolved_affine_4x4"])
        assert len(sample["root"]["resolved_affine_sha256"]) == 64
    assert binding["samples"][3]["pose_sample_id"] == "pose:neutral_stance"
    assert binding["samples"][4]["pose_sample_id"] == "pose:left_reach"
    assert binding["samples"][8]["pose_sample_id"] == "pose:asymmetric_settle"

    with (output / "binding_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert rows[0]["target_id"] == "target:synthetic_asymmetric_subject"
    assert rows[0]["scenario_clock_id"] == "clock:scenario_reference"
    assert rows[0]["motion_clock_id"] == "clock:asset_motion"
    assert rows[0]["pulse_clock_id"] == "clock:sensing_pulse"
    assert rows[0]["retiming_policy"] == "affine_scenario_to_motion"
    assert rows[4]["pose_sample_id"] == "pose:left_reach"
    assert rows[4]["valid"] == "true"
    assert (
        rows[4]["resolved_root_affine_sha256"]
        == binding["samples"][4]["root"]["resolved_affine_sha256"]
    )
    assert (output / "alignment_and_geometry.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "alignment_and_geometry.png").stat().st_size > 10_000

    asset = load_animated_target(output / "asset/animated_target.json")
    rebound = asset.bind(default_synthetic_binding_context(), expected_times)
    assert rebound.binding_id == result.binding_id


def test_synthetic_portable_json_csv_and_asset_bytes_are_deterministic(tmp_path: Path) -> None:
    left = publish_synthetic_animated_target_workflow(tmp_path / "left")
    right = publish_synthetic_animated_target_workflow(tmp_path / "right")

    assert left.asset_revision_sha256 == right.asset_revision_sha256
    assert left.binding_id == right.binding_id
    for relative_path in (
        "asset/animated_target.json",
        "asset/poses/a_asymmetric_settle.json",
        "asset/poses/m_left_reach.json",
        "asset/poses/z_neutral_stance.json",
        "binding.json",
        "binding_request.json",
        "binding_samples.csv",
    ):
        assert (left.output_dir / relative_path).read_bytes() == (
            right.output_dir / relative_path
        ).read_bytes()


def test_existing_manifest_workflow_does_not_copy_user_owned_source(tmp_path: Path) -> None:
    source = publish_synthetic_animated_target_workflow(tmp_path / "source")
    selected_times = (
        default_synthetic_binding_times()[0],
        default_synthetic_binding_times()[4],
        default_synthetic_binding_times()[-1],
    )
    output = tmp_path / "existing_workflow"

    result = publish_animated_target_workflow(
        source.output_dir / "asset/animated_target.json",
        default_synthetic_binding_context(),
        selected_times,
        output,
    )

    assert result.asset_revision_sha256 == source.asset_revision_sha256
    assert not (output / "asset").exists()
    assert {path.name for path in output.iterdir()} == {
        "alignment_and_geometry.png",
        "artifact_manifest.json",
        "binding.json",
        "binding_request.json",
        "binding_samples.csv",
    }
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["source_policy"] == "validated_existing_input_not_copied"
    assert [
        sample["sample_id"]
        for sample in json.loads((output / "binding.json").read_text())["samples"]
    ] == [sample.sample_id for sample in selected_times]


def test_existing_user_supplied_output_is_explicitly_local_only(tmp_path: Path) -> None:
    source = publish_synthetic_animated_target_workflow(tmp_path / "source")
    manifest_path = source.output_dir / "asset/animated_target.json"
    _set_asset_redistribution(manifest_path, "user_supplied_only")

    result = publish_animated_target_workflow(
        manifest_path,
        default_synthetic_binding_context(),
        default_synthetic_binding_times()[:1],
        tmp_path / "local_only",
    )

    evidence = json.loads((result.output_dir / "artifact_manifest.json").read_text())
    assert evidence["output_governance"] == {
        "source_redistribution": "user_supplied_only",
        "derived_artifact_distribution": "local_user_controlled_only_not_redistributable",
        "release_review_required": True,
    }


def test_existing_prohibited_asset_cannot_publish_derived_evidence(tmp_path: Path) -> None:
    source = publish_synthetic_animated_target_workflow(tmp_path / "source")
    manifest_path = source.output_dir / "asset/animated_target.json"
    _set_asset_redistribution(manifest_path, "prohibited")
    output = tmp_path / "prohibited"

    with pytest.raises(AnimatedTargetWorkflowError, match="prohibits publication"):
        publish_animated_target_workflow(
            manifest_path,
            default_synthetic_binding_context(),
            default_synthetic_binding_times()[:1],
            output,
        )

    assert not output.exists()


def test_publication_never_overwrites_and_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")

    with pytest.raises(AnimatedTargetWorkflowError, match="already exists"):
        publish_synthetic_animated_target_workflow(existing)
    assert sentinel.read_text(encoding="utf-8") == "owned by caller"
    assert set(existing.iterdir()) == {sentinel}

    failed = tmp_path / "failed"

    def fail_figure(*_args, **_kwargs) -> None:
        raise RuntimeError("injected figure failure")

    monkeypatch.setattr(workflow_module, "_write_figure", fail_figure)
    with pytest.raises(AnimatedTargetWorkflowError, match="injected figure failure"):
        publish_synthetic_animated_target_workflow(failed)

    assert not failed.exists()
    assert not (tmp_path / ".failed.publish.lock").exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_explicit_empty_binding_times_are_not_replaced_by_defaults(tmp_path: Path) -> None:
    output = tmp_path / "empty_binding"

    with pytest.raises(AnimatedTargetWorkflowError, match="binding times must not be empty"):
        publish_synthetic_animated_target_workflow(output, binding_times=())

    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="renameat2 is the POSIX publication path")
def test_posix_publication_fails_closed_without_atomic_noreplace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "evidence.txt").write_text("staged", encoding="utf-8")

    class RuntimeWithoutRenameAt2:
        pass

    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: RuntimeWithoutRenameAt2())

    def unsafe_rename(*_args, **_kwargs) -> None:
        raise AssertionError("overwrite-capable os.rename fallback must not run")

    monkeypatch.setattr(workflow_module.os, "rename", unsafe_rename)

    with pytest.raises(AnimatedTargetWorkflowError, match="refusing an unsafe fallback"):
        workflow_module._promote_directory_noreplace(staging, destination)

    assert staging.is_dir()
    assert not destination.exists()


def test_matplotlib_is_lazy_with_canonical_install_guidance_and_no_restricted_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = Path(workflow_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", 1)[0])
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "matplotlib" not in top_level_imports
    assert imported_roots.isdisjoint(
        {"smplx", "chumpy", "torch", "tensorflow", "sionna", "mitsuba", "kimodo"}
    )

    real_import = builtins.__import__

    def missing_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            error = ModuleNotFoundError("No module named 'matplotlib'")
            error.name = "matplotlib"
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_matplotlib)
    output = tmp_path / "missing_plotting"
    with pytest.raises(
        AnimatedTargetWorkflowError,
        match=r"python -m pip install orchav",
    ):
        publish_synthetic_animated_target_workflow(output)
    assert not output.exists()


class _MockExclusiveRename:
    """Record a mocked libc invocation; this never renames on the host."""

    def __init__(self, result=0):
        self.result = result
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        self.calls.append(arguments)
        return self.result


def _mock_publication_os(monkeypatch, *, system, libc, error_number=0):
    """Isolate platform dispatch without mutating the host's global os.name."""
    import ctypes
    from types import SimpleNamespace

    import shared.animated_target_workflow as workflow

    def reject_plain_rename(*args, **kwargs):
        pytest.fail("exclusive publication must not fall back to ordinary rename")

    monkeypatch.setattr(
        workflow,
        "os",
        SimpleNamespace(
            name="posix",
            fsencode=os.fsencode,
            strerror=os.strerror,
            rename=reject_plain_rename,
        ),
    )
    monkeypatch.setattr(workflow.platform, "system", lambda: system)
    cdll_calls = []

    def load_libc(name, *, use_errno):
        cdll_calls.append((name, use_errno))
        return libc

    monkeypatch.setattr(ctypes, "CDLL", load_libc)
    monkeypatch.setattr(ctypes, "get_errno", lambda: error_number)
    return workflow, cdll_calls


def test_darwin_publication_dispatch_uses_apple_exclusive_rename(monkeypatch):
    """Mocked ABI check only; native macOS publication is not exercised."""
    import ctypes
    from types import SimpleNamespace

    exclusive = _MockExclusiveRename()
    workflow, loads = _mock_publication_os(
        monkeypatch, system="Darwin", libc=SimpleNamespace(renamex_np=exclusive)
    )
    source, destination = Path(".motion.staging-123"), Path("motion")
    workflow._promote_directory_noreplace(source, destination)
    assert loads == [(None, True)]
    assert exclusive.calls == [(os.fsencode(source), os.fsencode(destination), 0x00000004)]
    assert exclusive.argtypes == (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    assert exclusive.restype is ctypes.c_int


def test_darwin_existing_destination_is_not_replaced(monkeypatch, tmp_path):
    """Mock the kernel collision while preserving a real destination sentinel."""
    import errno
    from types import SimpleNamespace

    source, destination = tmp_path / "staging", tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    (source / "candidate.txt").write_text("candidate")
    sentinel = destination / "keep.txt"
    sentinel.write_text("existing output")
    exclusive = _MockExclusiveRename(-1)
    workflow, _ = _mock_publication_os(
        monkeypatch,
        system="Darwin",
        libc=SimpleNamespace(renamex_np=exclusive),
        error_number=errno.EEXIST,
    )
    with pytest.raises(AnimatedTargetWorkflowError, match="already exists"):
        workflow._promote_directory_noreplace(source, destination)
    assert sentinel.read_text() == "existing output"
    assert (source / "candidate.txt").read_text() == "candidate"
    assert len(exclusive.calls) == 1


@pytest.mark.parametrize("error_name", ("ENOSYS", "EINVAL", "ENOTSUP", "EOPNOTSUPP"))
def test_darwin_unsupported_exclusive_rename_has_no_fallback(monkeypatch, error_name):
    import errno
    from types import SimpleNamespace

    exclusive = _MockExclusiveRename(-1)
    workflow, _ = _mock_publication_os(
        monkeypatch,
        system="Darwin",
        libc=SimpleNamespace(renamex_np=exclusive),
        error_number=getattr(errno, error_name, errno.EINVAL),
    )
    with pytest.raises(AnimatedTargetWorkflowError, match="refusing an unsafe fallback"):
        workflow._promote_directory_noreplace(Path("staging"), Path("published"))
    assert len(exclusive.calls) == 1


def test_darwin_missing_exclusive_rename_symbol_has_no_fallback(monkeypatch):
    from types import SimpleNamespace

    workflow, _ = _mock_publication_os(monkeypatch, system="Darwin", libc=SimpleNamespace())
    with pytest.raises(AnimatedTargetWorkflowError, match="unavailable"):
        workflow._promote_directory_noreplace(Path("staging"), Path("published"))


@pytest.mark.parametrize("error_name", ("EACCES", "ENOENT", "EXDEV"))
def test_darwin_publication_preserves_actionable_os_errors(monkeypatch, error_name):
    import errno
    from types import SimpleNamespace

    error_number = getattr(errno, error_name)
    workflow, _ = _mock_publication_os(
        monkeypatch,
        system="Darwin",
        libc=SimpleNamespace(renamex_np=_MockExclusiveRename(-1)),
        error_number=error_number,
    )
    with pytest.raises(OSError) as failure:
        workflow._promote_directory_noreplace(Path("staging"), Path("published"))
    assert failure.value.errno == error_number
    assert failure.value.filename == "published"


def test_linux_publication_keeps_renameat2_abi(monkeypatch):
    import ctypes
    from types import SimpleNamespace

    exclusive = _MockExclusiveRename()
    workflow, _ = _mock_publication_os(
        monkeypatch, system="Linux", libc=SimpleNamespace(renameat2=exclusive)
    )
    source, destination = Path("staging"), Path("published")
    workflow._promote_directory_noreplace(source, destination)
    assert exclusive.calls == [(-100, os.fsencode(source), -100, os.fsencode(destination), 1)]
    assert exclusive.argtypes == (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    assert exclusive.restype is ctypes.c_int


def _windows_publication_error(code: int) -> OSError:
    error = OSError("injected Windows publication error")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_windows_publication_retries_transient_locks_without_replacement(
    monkeypatch, tmp_path, code
):
    source, destination = tmp_path / "staging", tmp_path / "published"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    original_rename = workflow_module.os.rename
    attempts, delays = [], []

    def locked_then_available(src, dst):
        attempts.append((src, dst))
        if len(attempts) < 3:
            raise _windows_publication_error(code)
        original_rename(src, dst)

    monkeypatch.setattr(workflow_module.os, "rename", locked_then_available)
    monkeypatch.setattr(workflow_module.time, "sleep", delays.append)
    workflow_module._promote_windows_directory_noreplace(source, destination)
    assert attempts == [(source, destination)] * 3
    assert delays == [0.05, 0.1]
    assert (destination / "payload").read_bytes() == b"complete"
    assert not source.exists()


def test_windows_publication_retry_is_bounded_and_preserves_os_error(monkeypatch, tmp_path):
    error = _windows_publication_error(5)
    attempts, delays = [], []

    def always_locked(src, dst):
        attempts.append((src, dst))
        raise error

    monkeypatch.setattr(workflow_module.os, "rename", always_locked)
    monkeypatch.setattr(workflow_module.time, "sleep", delays.append)
    with pytest.raises(OSError) as failure:
        workflow_module._promote_windows_directory_noreplace(
            tmp_path / "staging", tmp_path / "published"
        )
    assert failure.value is error
    assert len(attempts) == 6
    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8]


@pytest.mark.parametrize("code", [2, 17, 87])
def test_windows_publication_does_not_retry_other_os_errors(monkeypatch, tmp_path, code):
    error = _windows_publication_error(code)
    delays = []

    def reject_rename(src, dst):
        raise error

    monkeypatch.setattr(workflow_module.os, "rename", reject_rename)
    monkeypatch.setattr(workflow_module.time, "sleep", delays.append)
    with pytest.raises(OSError) as failure:
        workflow_module._promote_windows_directory_noreplace(
            tmp_path / "staging", tmp_path / "published"
        )
    assert failure.value is error
    assert delays == []


def test_windows_retry_refuses_a_competing_destination_immediately(monkeypatch, tmp_path):
    source, destination = tmp_path / "staging", tmp_path / "published"
    source.mkdir()
    attempts, delays = [], []

    def competing_publisher(src, dst):
        attempts.append((src, dst))
        destination.mkdir()
        (destination / "sentinel").write_bytes(b"preserve")
        raise _windows_publication_error(5)

    monkeypatch.setattr(workflow_module.os, "rename", competing_publisher)
    monkeypatch.setattr(workflow_module.time, "sleep", delays.append)
    with pytest.raises(AnimatedTargetWorkflowError, match="already exists"):
        workflow_module._promote_windows_directory_noreplace(source, destination)
    assert (destination / "sentinel").read_bytes() == b"preserve"
    assert source.is_dir()
    assert len(attempts) == 1
    assert delays == []
