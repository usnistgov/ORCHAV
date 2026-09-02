"""Identify the loaded ORCHAV source tree and bind Python child processes to it.

Development environments can contain more than one editable ORCHAV checkout.
Using a console script or an ordinary ``python -m`` child from a scenario
directory can therefore import a different checkout from the parent process.
This module provides the small, pure-standard-library boundary used to prevent
that split-source execution.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SOURCE_IDENTITY_SCHEMA_VERSION = 1
EXPECTED_SOURCE_IDENTITY_ENV = "ORCHAV_EXPECTED_SOURCE_IDENTITY"

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ORCHAV_PACKAGES = ("generator", "shared", "visualizer")

# ``-I`` removes the scenario directory and environment-provided Python paths.
# The bootstrap then adds exactly the source tree selected by the parent,
# verifies every ORCHAV top-level package, and only then executes the requested
# module. Arguments remaining in ``sys.argv`` are preserved for that module.
_ISOLATED_MODULE_BOOTSTRAP = """\
import importlib.util
import json
import pathlib
import runpy
import sys

root = pathlib.Path(sys.argv.pop(1)).resolve()
module_name = sys.argv.pop(1)
required_packages = json.loads(sys.argv.pop(1))
sys.path.insert(0, str(root))

for candidate in [*required_packages, module_name]:
    spec = importlib.util.find_spec(candidate)
    origin = pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or not origin.is_relative_to(root):
        raise SystemExit(
            f"ORCHAV source binding failed for {candidate!r}: "
            f"expected a module under {root}, got {origin}"
        )

runpy.run_module(module_name, run_name="__main__", alter_sys=True)
"""


class SourceIdentityError(ValueError):
    """Raised when one process contains or reports inconsistent source trees."""


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Canonical identity shared by ORCHAV processes from one source tree."""

    source_root: Path
    version: str | None
    git_sha: str | None
    schema_version: int = SOURCE_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = Path(self.source_root).expanduser()
        if not root.is_absolute():
            raise SourceIdentityError("source_root must be absolute")
        object.__setattr__(self, "source_root", root.resolve())
        if self.schema_version != SOURCE_IDENTITY_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"unsupported source identity schema_version={self.schema_version}"
            )
        for label, value in (("version", self.version), ("git_sha", self.git_sha)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SourceIdentityError(f"{label} must be a non-empty string or null")

    @classmethod
    def from_mapping(cls, raw: Any) -> SourceIdentity:
        """Parse one strict source-identity protocol mapping."""
        if not isinstance(raw, Mapping):
            raise SourceIdentityError("source_identity must be a JSON object")
        root = raw.get("source_root")
        if not isinstance(root, str) or not root.strip():
            raise SourceIdentityError("source_identity.source_root must be a non-empty string")
        schema_version = raw.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise SourceIdentityError("source_identity.schema_version must be an integer")
        version = raw.get("version")
        git_sha = raw.get("git_sha")
        return cls(
            source_root=Path(root),
            version=version,
            git_sha=git_sha,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-ready process-handshake representation."""
        return {
            "schema_version": self.schema_version,
            "source_root": str(self.source_root),
            "version": self.version,
            "git_sha": self.git_sha,
        }

    def matches(self, other: SourceIdentity) -> bool:
        """Return whether ``other`` identifies this exact loaded source tree."""
        try:
            same_root = self.source_root.samefile(other.source_root)
        except OSError:
            same_root = os.path.normcase(str(self.source_root)) == os.path.normcase(
                str(other.source_root)
            )
        return (
            same_root
            and self.version == other.version
            and self.git_sha == other.git_sha
            and self.schema_version == other.schema_version
        )


def _package_source_root(package_name: str) -> Path:
    if not _PACKAGE_NAME_RE.fullmatch(package_name):
        raise SourceIdentityError(f"invalid top-level package name: {package_name!r}")
    module = importlib.import_module(package_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise SourceIdentityError(f"{package_name!r} has no filesystem source")
    package_dir = Path(module_file).resolve().parent
    if package_dir.name != package_name:
        raise SourceIdentityError(
            f"{package_name!r} resolved outside its expected package directory: {module_file}"
        )
    return package_dir.parent


def _source_version(source_root: Path) -> str | None:
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project")
        except (OSError, tomllib.TOMLDecodeError):
            project = None
        if isinstance(project, Mapping):
            version = project.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
    try:
        return importlib.metadata.version("orchav")
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_source_git_sha(source_root: str | Path) -> str | None:
    """Return ``HEAD`` only when ``source_root`` itself owns the Git checkout."""
    root = Path(source_root).expanduser().resolve()
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return None
    try:
        reported_root = Path(lines[0]).resolve()
    except OSError:
        return None
    try:
        same_root = reported_root.samefile(root)
    except OSError:
        same_root = os.path.normcase(str(reported_root)) == os.path.normcase(str(root))
    sha = lines[1].strip()
    return sha if same_root and sha else None


@lru_cache(maxsize=None)
def _loaded_source_identity_for_process(
    anchor_package: str,
    process_id: int,
) -> SourceIdentity:
    """Resolve one identity once per anchor and operating-system process."""
    # Include the PID in the cache key so a forked child does not reuse the
    # parent's cached Git/process identity.
    del process_id
    anchor_root = _package_source_root(anchor_package)
    shared_root = Path(__file__).resolve().parent.parent
    try:
        same_root = anchor_root.samefile(shared_root)
    except OSError:
        same_root = os.path.normcase(str(anchor_root)) == os.path.normcase(str(shared_root))
    if not same_root:
        raise SourceIdentityError(
            "ORCHAV packages were imported from different source trees: "
            f"{anchor_package}={anchor_root}, shared={shared_root}"
        )
    return SourceIdentity(
        source_root=anchor_root,
        version=_source_version(anchor_root),
        git_sha=resolve_source_git_sha(anchor_root),
    )


def loaded_source_identity(anchor_package: str) -> SourceIdentity:
    """Identify and cache the tree containing ``anchor_package`` and this helper."""
    return _loaded_source_identity_for_process(anchor_package, os.getpid())


def source_bound_module_command(
    module_name: str,
    *arguments: str | Path,
    identity: SourceIdentity | None = None,
    anchor_package: str = "shared",
    python_executable: str | Path | None = None,
    required_packages: Sequence[str] = _ORCHAV_PACKAGES,
) -> tuple[str, ...]:
    """Construct, but do not execute, a child bound to one source tree.

    The ``python -I`` bootstrap inserts exactly ``identity.source_root``,
    verifies every required package resolves beneath it, and then runs
    ``module_name`` with the supplied arguments.
    """
    if not _MODULE_NAME_RE.fullmatch(module_name):
        raise SourceIdentityError(f"invalid Python module name: {module_name!r}")
    package_names = tuple(str(name) for name in required_packages)
    if any(not _MODULE_NAME_RE.fullmatch(name) for name in package_names):
        raise SourceIdentityError("required_packages contains an invalid module name")
    selected_identity = identity or loaded_source_identity(anchor_package)
    # Do not call Path.resolve() here. POSIX virtual-environment interpreters
    # are commonly symlinks; resolving one to the base interpreter can discard
    # the venv selection that the parent process deliberately supplied.
    executable = os.path.abspath(os.path.expanduser(os.fspath(python_executable or sys.executable)))
    return (
        executable,
        "-I",
        "-u",
        "-c",
        _ISOLATED_MODULE_BOOTSTRAP,
        str(selected_identity.source_root),
        module_name,
        json.dumps(package_names, separators=(",", ":")),
        *(str(argument) for argument in arguments),
    )


__all__ = [
    "EXPECTED_SOURCE_IDENTITY_ENV",
    "SOURCE_IDENTITY_SCHEMA_VERSION",
    "SourceIdentity",
    "SourceIdentityError",
    "loaded_source_identity",
    "resolve_source_git_sha",
    "source_bound_module_command",
]
