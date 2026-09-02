"""Import-light launch policy for the Open3D/Filament renderer.

Without this, on a Linux host that has both NVIDIA and Mesa EGL ICDs
registered (the common case for workstations and remote render servers),
Open3D's ``OffscreenRenderer`` can fall through to Mesa's software rasterizer
(``llvmpipe``) instead of hitting the NVIDIA GPU. With ``DISPLAY`` unset,
``eglGetDisplay`` falls back to a surfaceless or default device, which on
multi-vendor hosts may still land on Mesa.

Headless callers can tell ``libglvnd`` to consider only the NVIDIA ICD by
setting ``__EGL_VENDOR_LIBRARY_FILENAMES`` before any ``import open3d``.
GUI callers leave graphics selection to the active desktop or display server.

This module degrades gracefully:

- On non-NVIDIA hosts (no NVIDIA ICD JSON file found), nothing is set.
- GUI processes proceed on the graphics stack selected by their environment.
- On non-Linux platforms the ICD search finds nothing and the function
  is a no-op.
- User can opt out unconditionally with ``ORCHAV_NO_GPU_PREFLIGHT=1``.

The module is intentionally import-light so that platform checks can run before
any Open3D or Qt import.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Sequence

# Internal flag used to make repeated headless ``apply()`` calls idempotent.
_APPLIED_FLAG = "ORCHAV_GPU_PREFLIGHT_APPLIED"
_OPT_OUT_FLAG = "ORCHAV_NO_GPU_PREFLIGHT"
_ICD_ENV = "__EGL_VENDOR_LIBRARY_FILENAMES"

# Common NVIDIA ICD JSON locations, in priority order.
_NVIDIA_ICD_CANDIDATES: tuple[str, ...] = (
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "/etc/glvnd/egl_vendor.d/10_nvidia.json",
)
_NVIDIA_ICD_GLOBS: tuple[str, ...] = (
    "/usr/share/glvnd/egl_vendor.d/*nvidia*.json",
    "/etc/glvnd/egl_vendor.d/*nvidia*.json",
)


def _find_nvidia_icd() -> str | None:
    """Return the path to the NVIDIA EGL ICD JSON, or ``None``."""
    for candidate in _NVIDIA_ICD_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    for pattern in _NVIDIA_ICD_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _uses_open3d_renderer(argv: Sequence[str]) -> bool:
    """Return True if an Open3D/Filament renderer selector appears in argv."""
    for i, arg in enumerate(argv):
        if arg == "--renderer" and i + 1 < len(argv):
            return argv[i + 1] == "open3d"
        if arg.startswith("--renderer="):
            return arg.split("=", 1)[1] == "open3d"
    return False


def _requests_cli_help(argv: Sequence[str]) -> bool:
    """Return True when the argv is asking for CLI help only."""
    return any(arg in {"-h", "--help"} for arg in argv)


def reject_unsupported_macos_open3d(
    argv: Sequence[str] | None = None,
    *,
    platform_name: str | None = None,
) -> None:
    """Reject the unsupported Open3D desktop renderer on macOS.

    This import-light check runs from ``visualizer.__main__`` before Qt or a
    renderer is imported. Open3D remains an installed runtime dependency
    because ORCHAV also uses its geometry utilities outside visualization.
    """
    argv = list(argv) if argv is not None else list(sys.argv)
    effective_platform = sys.platform if platform_name is None else platform_name
    if (
        effective_platform == "darwin"
        and not _requests_cli_help(argv)
        and _uses_open3d_renderer(argv)
    ):
        raise SystemExit(
            "The Open3D/Filament renderer is not supported on macOS in ORCHAV v0.1. "
            "Use --renderer pygfx (the default)."
        )


def apply(*, wants_gui: bool = False, argv: Sequence[str] | None = None) -> None:
    """Steer headless Open3D EGL initialization to NVIDIA when possible.

    Must be called before any ``import open3d`` so that ``libEGL.so`` picks
    up the ``__EGL_VENDOR_LIBRARY_FILENAMES`` setting at load time.

    Args:
        wants_gui: If True, leave graphics selection to the GUI environment.
            If False, this is a headless path, so the ICD filter environment
            variable may be set.
        argv: Command line whose renderer selection is checked. Defaults to
            ``sys.argv``.

    Side effects (all best-effort, guarded):
        - For headless Open3D paths, sets
          ``__EGL_VENDOR_LIBRARY_FILENAMES`` to the NVIDIA ICD path.
        - Sets ``ORCHAV_GPU_PREFLIGHT_APPLIED=1`` after applying that filter
          so later invocations skip.
    """
    if os.environ.get(_APPLIED_FLAG):
        return
    if os.environ.get(_OPT_OUT_FLAG):
        return

    # Gate the entire preflight on Open3D/Filament. Pygfx also interacts with
    # libEGL via wgpu's khronos-egl device probe, and filtering libglvnd's ICD
    # list to NVIDIA only can starve it of expected extensions, causing
    # wgpu-native to panic with a null unwrap at startup. The preflight's
    # benefit is specific to Open3D's Filament backend.
    argv = list(argv) if argv is not None else list(sys.argv)
    if _requests_cli_help(argv):
        return
    if not _uses_open3d_renderer(argv):
        return

    if wants_gui:
        # The GUI environment owns graphics-device selection.
        return

    nvidia_icd = _find_nvidia_icd()
    if nvidia_icd is None:
        # Not an NVIDIA host (AMD, Intel, software, macOS, Windows, etc.).
        # Leave EGL alone — libglvnd's default selection is correct.
        return

    os.environ[_APPLIED_FLAG] = "1"

    # Restrict libglvnd to only load the NVIDIA ICD so Open3D's headless
    # OffscreenRenderer picks the hardware device instead of Mesa llvmpipe.
    os.environ.setdefault(_ICD_ENV, nvidia_icd)
