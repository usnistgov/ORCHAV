"""Route native target and background channels through one acquisition boundary."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from shared.scenarios.model import NativeRcsSolverModel

from ..exceptions import ConfigurationError


def has_native_targets(simulation: Any) -> bool:
    """Return whether the scene contains an explicitly selected native model."""
    return any(
        getattr(manager.config, "representation", "mesh_material") != "mesh_material"
        for manager in simulation.target_managers
    )


def compute_native_paths(
    simulation: Any, background_kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Solve against the same posed scene, with independent native ray budgets.

    The native proxy remains in both solves and absorbs rather than scattering
    as an ordinary material. ``target_only`` and ``background_only`` are named
    diagnostics; the default detector input is the combined channel.
    """
    config = NativeRcsSolverModel.model_validate(simulation.settings.get("rcs") or {})
    import mitsuba as mi

    if not (mi.variant() or "").endswith("_polarized"):
        raise ConfigurationError(
            "Native RCS requires a polarized Mitsuba variant; initialize "
            "llvm_ad_mono_polarized or cuda_ad_mono_polarized before scene construction"
        )
    if background_kwargs.get("synthetic_array", True) is not True:
        raise ConfigurationError("Native RCS integration currently requires synthetic_array=true")
    if len(simulation.tx_list) != 1 or len(simulation.rx_list) != 1:
        raise ConfigurationError("Native RCS integration currently requires one TX/RX pair")
    if simulation.rcs_solver is None and config.channel != "background_only":
        try:
            from sionna.rt.rcs import RCSSolver
        except ImportError as exc:
            raise ConfigurationError("Native RCS solving requires Sionna RT 2.2") from exc
        simulation.rcs_solver = RCSSolver()

    kwargs = config.model_dump(exclude={"channel"})
    kwargs["synthetic_array"] = True
    target_paths = None
    background_paths = None
    if config.channel != "background_only":
        target_paths = simulation.rcs_solver(scene=simulation.scene, **kwargs)
    if config.channel != "target_only":
        background_paths = simulation.path_solver(**background_kwargs)
    if target_paths is None:
        paths = background_paths
    elif background_paths is None:
        paths = target_paths
    else:
        paths = target_paths.concat(background_paths)

    # RCSSolver indexes the insertion-ordered scene.sensing_targets collection.
    # It is a separate namespace from scene mesh IDs and the full actor catalog.
    frame_targets = {
        manager.config.name: index for index, manager in enumerate(simulation.target_managers)
    }
    native_names = list(simulation.scene.sensing_targets)
    if any(name not in frame_targets for name in native_names):
        raise ConfigurationError("Scene contains a native sensing target outside the actor catalog")
    versions = {}
    for package in ("sionna-rt", "mitsuba", "drjit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    return paths, {
        "schema_version": 1,
        "channel": config.channel,
        "solver": config.model_dump(),
        "background_solver": {
            key: value for key, value in background_kwargs.items() if key != "scene"
        },
        "target_catalog_names": native_names,
        "target_catalog_frame_indices": [frame_targets[name] for name in native_names],
        "engine_versions": versions,
        "delay_convention": "absolute",
        "coefficient_normalization": "none",
        "target_motion_model": "rigid_translation",
    }
