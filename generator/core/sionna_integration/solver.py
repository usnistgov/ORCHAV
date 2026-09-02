"""PathSolver and Dr.Jit compatibility helpers.

Sionna RT changes PathSolver keyword support and differentiation behavior across
versions.  Core solver callers should ask this module for
capabilities instead of scattering version checks throughout the codebase.
"""

from __future__ import annotations

import inspect
import re

import drjit as dr

try:
    import sionna
except ImportError:  # pragma: no cover - optional dependency at runtime
    sionna = None

from sionna.rt import PathSolver

SIONNA_VERSION: str = getattr(sionna, "__version__", "0") if sionna else "0"


def _version_to_tuple(value: str, max_parts: int = 3) -> tuple[int, ...]:
    """Convert a version string into a fixed-width comparable integer tuple."""
    digits = [int(part) for part in re.split(r"[^\d]+", value) if part]
    if not digits:
        return (0,) * max_parts
    digits = digits[:max_parts]
    if len(digits) < max_parts:
        digits += [0] * (max_parts - len(digits))
    return tuple(digits)


def version_greater_equal(value: str, threshold: str) -> bool:
    """Return True if ``value`` >= ``threshold`` for simple dotted versions."""
    return _version_to_tuple(value) >= _version_to_tuple(threshold)


def _path_solver_signature():
    """Return the PathSolver call signature when introspection is available."""
    try:
        return inspect.signature(PathSolver.__call__)
    except (TypeError, ValueError):
        return None


def path_solver_supports_arg(arg_name: str) -> bool:
    """Return True if ``PathSolver.__call__`` accepts the given argument."""
    sig = _path_solver_signature()
    if sig is None:
        # If this Sionna build hides the signature, keep the caller permissive
        # and let the actual PathSolver call decide at runtime.
        return True
    return arg_name in sig.parameters


# These flags are computed once at import so hot frame-solving paths do not
# introspect the PathSolver signature repeatedly.
PATH_SOLVER_SUPPORTS_DIFFRACTION = path_solver_supports_arg("diffraction")
PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION = path_solver_supports_arg("edge_diffraction")
PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION = path_solver_supports_arg("diffraction_lit_region")

# Starting with Sionna RT 1.2, the shooting-and-bouncing candidate generator
# performs local-memory writes tracked by Dr.Jit. These writes are not
# differentiable in reverse mode, so analytic gradients through the path solver
# become invalid and we must fall back to numerical estimates.
PATH_SOLVER_AUTODIFF_COMPATIBLE = not (sionna and version_greater_equal(SIONNA_VERSION, "1.2"))


def set_drjit_seed(seed: int) -> bool:
    """Attempt to seed Dr.Jit RNG across supported API spellings."""
    for attr in ("seed", "set_seed"):
        setter = getattr(dr, attr, None)
        if callable(setter):
            try:
                setter(int(seed))
                return True
            except (ValueError, TypeError, RuntimeError):
                continue
    return False
