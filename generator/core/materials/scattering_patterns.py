"""Scattering-pattern registration for material overrides.

Sionna material overrides may name scattering patterns as strings. This module
registers the ORCHAV-supported documented names lazily, only when an override
needs registry lookup, so importing ``generator.core.materials`` does not
require Sionna/Mitsuba at module import time.
"""

from __future__ import annotations

from importlib import import_module

from shared.logging import get_logger

logger = get_logger(__name__)

# Polynomial constants approximate the Gaussian RER normalizer used by the
# custom pattern below. Keep them together so the scattering formula is readable.
_GAUSSIAN_RER_PADE_NUMERATOR = (16.0 / 9.0, 0.536, 0.399)
_GAUSSIAN_RER_PADE_DENOMINATOR = (1.0, 0.965, 0.457, 0.200)
_GAUSSIAN_RER_NORMALIZER_FLOOR = 1e-12
_ISOTROPIC_EPS_DEFAULT = 1e-2
_ISOTROPIC_EPS_MIN = 1e-6
_SUPPORTED_PATTERNS_REGISTERED = False
_PRIVATE_EXTENSIONS_LOADED = False


def register_supported_scattering_patterns() -> None:
    """Register scattering patterns documented for material overrides."""
    global _SUPPORTED_PATTERNS_REGISTERED
    if _SUPPORTED_PATTERNS_REGISTERED:
        return

    import drjit as dr
    import mitsuba as mi
    from sionna.rt.radio_materials.scattering_pattern import (
        ScatteringPattern,
        register_scattering_pattern,
        scattering_pattern_registry,
    )

    mi_float = getattr(mi, "Float", float)

    # Keep these classes local so importing generator.core.materials does not
    # require Sionna/Mitsuba until a string scattering-pattern override is used.
    class GaussianRERPattern(ScatteringPattern):
        """Gaussian RER lobe with tunable ``alpha_g``."""

        def __init__(self, alpha_g: mi.Float = 0.0):
            super().__init__()
            self.alpha_g = alpha_g

        @property
        def alpha_g(self):
            return self._alpha_g

        @alpha_g.setter
        def alpha_g(self, value):
            self._alpha_g = mi_float(value)

        def __call__(self, ki_local: mi.Vector3f, ko_local: mi.Vector3f) -> mi.Float:
            kr_local = mi.reflect(-ki_local)
            cos_theta_s = dr.clamp(ko_local[2], 0.0, 1.0)
            cos_psi = dr.clamp(dr.dot(kr_local, ko_local), -1.0, 1.0)

            kernel = dr.sqrt(cos_theta_s) * dr.exp(self.alpha_g * (cos_psi - 1.0))
            sqrt_cos_theta_i = dr.sqrt(dr.abs(-ki_local.z))
            alpha = self.alpha_g

            num = (
                _GAUSSIAN_RER_PADE_NUMERATOR[0]
                + _GAUSSIAN_RER_PADE_NUMERATOR[1] * alpha
                + _GAUSSIAN_RER_PADE_NUMERATOR[2] * alpha * alpha
            )
            den = (
                _GAUSSIAN_RER_PADE_DENOMINATOR[0]
                + _GAUSSIAN_RER_PADE_DENOMINATOR[1] * alpha
                + _GAUSSIAN_RER_PADE_DENOMINATOR[2] * alpha * alpha
                + _GAUSSIAN_RER_PADE_DENOMINATOR[3] * alpha * alpha * alpha
            )
            normalizer = dr.maximum(
                (dr.pi * num * dr.rcp(den)) * sqrt_cos_theta_i,
                _GAUSSIAN_RER_NORMALIZER_FLOOR,
            )
            return kernel * dr.rcp(normalizer)

    class IsotropicPattern(ScatteringPattern):
        """Regularized isotropic-RCS pattern."""

        def __init__(self, eps: mi.Float = _ISOTROPIC_EPS_DEFAULT):
            super().__init__()
            self.eps = mi_float(max(float(eps), _ISOTROPIC_EPS_MIN))

        def __call__(self, ki_local: mi.Vector3f, ko_local: mi.Vector3f) -> mi.Float:
            del ko_local
            cos_theta_i = dr.clamp(dr.abs(-ki_local.z), 0.0, 1.0)
            return dr.rcp(dr.sqrt(cos_theta_i * cos_theta_i + self.eps * self.eps))

    class ERIsotropicPattern(ScatteringPattern):
        """Constant hemispherical ER-like isotropic pattern."""

        def __call__(self, ki_local: mi.Vector3f, ko_local: mi.Vector3f) -> mi.Float:
            del ki_local, ko_local
            return dr.rcp(2.0 * dr.pi)

    def register_if_missing(name: str, pattern_cls: type) -> None:
        """Register a pattern only when Sionna has not already provided it."""
        try:
            factory = scattering_pattern_registry.get(name)
        except Exception as exc:
            logger.debug(
                "Could not read Sionna scattering-pattern registry entry '%s': %s", name, exc
            )
            factory = None
        if factory is None:
            register_scattering_pattern(name, pattern_cls)

    register_if_missing("g-rer", GaussianRERPattern)
    register_if_missing("iso", IsotropicPattern)
    register_if_missing("isotropic", IsotropicPattern)
    register_if_missing("er-isotropic", ERIsotropicPattern)

    _SUPPORTED_PATTERNS_REGISTERED = True


def register_extra_scattering_patterns() -> None:
    """Register documented patterns and optional local extension patterns."""
    register_supported_scattering_patterns()
    _register_private_scattering_pattern_extensions()


def _register_private_scattering_pattern_extensions() -> None:
    """Load optional deployment-local scattering-pattern extensions when present."""
    global _PRIVATE_EXTENSIONS_LOADED
    if _PRIVATE_EXTENSIONS_LOADED:
        return
    _PRIVATE_EXTENSIONS_LOADED = True
    try:
        module = import_module("generator.core.materials._private_scattering_pattern_extensions")
    except ImportError as exc:
        logger.debug("No private scattering-pattern extensions loaded: %s", exc)
        return

    register = getattr(module, "register", None)
    if callable(register):
        register()
