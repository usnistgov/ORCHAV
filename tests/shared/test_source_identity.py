"""Tests for exact-source Python subprocess binding."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import shared.source_identity as source_identity_module
from shared.source_identity import (
    SourceIdentity,
    SourceIdentityError,
    loaded_source_identity,
    resolve_source_git_sha,
    source_bound_module_command,
)


def _write_probe(root: Path, marker: str) -> None:
    package = root / "probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        f"print({marker!r})\n",
        encoding="utf-8",
    )


def test_loaded_identity_uses_imported_tree_not_current_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)

    identity = loaded_source_identity("shared")

    assert identity.source_root.samefile(expected_root)
    assert identity.version == "0.1.0"
    assert identity.git_sha == resolve_source_git_sha(expected_root)


def test_loaded_identity_is_cached_once_per_process(monkeypatch) -> None:
    source_identity_module._loaded_source_identity_for_process.cache_clear()
    git_calls: list[Path] = []

    def fake_git_sha(source_root: Path) -> str:
        git_calls.append(Path(source_root))
        return f"sha-{len(git_calls)}"

    monkeypatch.setattr(source_identity_module, "resolve_source_git_sha", fake_git_sha)
    monkeypatch.setattr(source_identity_module.os, "getpid", lambda: 1_000_001)
    try:
        first = loaded_source_identity("shared")
        second = loaded_source_identity("shared")

        assert first is second
        assert first.git_sha == "sha-1"
        assert len(git_calls) == 1

        monkeypatch.setattr(source_identity_module.os, "getpid", lambda: 1_000_002)
        child_process_identity = loaded_source_identity("shared")

        assert child_process_identity.git_sha == "sha-2"
        assert len(git_calls) == 2
    finally:
        source_identity_module._loaded_source_identity_for_process.cache_clear()


def test_git_sha_does_not_climb_from_a_nested_non_checkout_root(
    monkeypatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repository_root)

    assert resolve_source_git_sha(repository_root / "shared") is None


def test_git_sha_matches_head_of_the_exact_source_root() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    if not (repository_root / ".git").exists():
        pytest.skip("source export has no Git metadata")
    expected = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert resolve_source_git_sha(repository_root) == expected


def test_source_identity_protocol_rejects_coerced_schema_version(tmp_path: Path) -> None:
    with pytest.raises(
        SourceIdentityError,
        match="source_identity.schema_version must be an integer",
    ):
        SourceIdentity.from_mapping(
            {
                "schema_version": "1",
                "source_root": str(tmp_path),
                "version": "0.1.0",
                "git_sha": None,
            }
        )


def test_bound_command_defeats_cwd_and_pythonpath_module_shadowing(tmp_path: Path) -> None:
    selected_root = tmp_path / "selected"
    pythonpath_root = tmp_path / "pythonpath"
    child_cwd = tmp_path / "scenario"
    _write_probe(selected_root, "selected source")
    _write_probe(pythonpath_root, "wrong PYTHONPATH source")
    _write_probe(child_cwd, "wrong scenario source")

    identity = SourceIdentity(
        source_root=selected_root,
        version="test",
        git_sha=None,
    )
    command = source_bound_module_command(
        "probe",
        identity=identity,
        python_executable=sys.executable,
        required_packages=("probe",),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(pythonpath_root)

    result = subprocess.run(
        command,
        cwd=child_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "selected source"
    assert command[0] == os.path.abspath(os.path.expanduser(sys.executable))
    assert command[1:3] == ("-I", "-u")


def test_bound_command_preserves_a_symlinked_venv_interpreter(tmp_path: Path) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    interpreter_target = tmp_path / "base-python"
    interpreter_target.touch()
    venv_interpreter = tmp_path / "venv" / "bin" / "python"
    venv_interpreter.parent.mkdir(parents=True)
    try:
        venv_interpreter.symlink_to(interpreter_target)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit symlink creation: {exc}")

    command = source_bound_module_command(
        "probe",
        identity=SourceIdentity(
            source_root=selected_root,
            version="test",
            git_sha=None,
        ),
        python_executable=venv_interpreter,
        required_packages=(),
    )

    assert command[0] == os.path.abspath(str(venv_interpreter))
    assert Path(command[0]).is_symlink()
    assert Path(command[0]).resolve() == interpreter_target.resolve()


def test_bound_command_fails_closed_when_module_is_outside_selected_root(
    tmp_path: Path,
) -> None:
    selected_root = tmp_path / "selected"
    pythonpath_root = tmp_path / "pythonpath"
    selected_root.mkdir()
    _write_probe(pythonpath_root, "wrong source")
    identity = SourceIdentity(
        source_root=selected_root,
        version="test",
        git_sha=None,
    )
    command = source_bound_module_command(
        "probe",
        identity=identity,
        python_executable=sys.executable,
        required_packages=(),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(pythonpath_root)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source binding failed" in result.stderr
