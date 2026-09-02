"""Typed contracts for optional visualizer runtime features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .extension_loader import ensure_external_extensions_loaded

PanelFactory = Callable[[Any], Any]
ServiceFactory = Callable[[Any], Any]
FeaturePredicate = Callable[[Any], bool]
FrameSyncHook = Callable[[Any, Any, dict[str, Any], int], bool]
StatusChipsHook = Callable[[Any, Any, Any], Sequence[str]]


@dataclass(frozen=True, slots=True)
class RuntimeFeatureExtension:
    """One optional panel/service feature synchronized with rendered frames."""

    key: str
    panel_title: str
    tab_label: str
    panel_factory: PanelFactory
    service_factory: ServiceFactory
    enabled: FeaturePredicate
    sync_frame: FrameSyncHook
    status_chips: StatusChipsHook | None = None
    start_open: bool = False
    lazy: bool = True
    frame_data: bool = True


_RUNTIME_EXTENSIONS: dict[str, RuntimeFeatureExtension] = {}


def register_runtime_extension(extension: RuntimeFeatureExtension) -> None:
    """Register or replace one runtime feature by its stable key."""
    key = extension.key.strip()
    if not key:
        raise ValueError("Runtime extension key cannot be empty")
    if key != extension.key:
        extension = replace(extension, key=key)
    _RUNTIME_EXTENSIONS[key] = extension


def registered_runtime_extensions() -> tuple[RuntimeFeatureExtension, ...]:
    """Return installed runtime features in deterministic key order."""
    ensure_external_extensions_loaded()
    return tuple(_RUNTIME_EXTENSIONS[key] for key in sorted(_RUNTIME_EXTENSIONS))


def configure_runtime_extensions(viz: Any) -> tuple[str, ...]:
    """Create services for enabled extensions and retire disabled services."""
    services = getattr(viz, "extension_services", None)
    if not isinstance(services, dict):
        services = {}
        viz.extension_services = services

    enabled_keys: list[str] = []
    for extension in registered_runtime_extensions():
        if not bool(extension.enabled(viz)):
            services.pop(extension.key, None)
            continue
        enabled_keys.append(extension.key)
        if extension.key not in services:
            services[extension.key] = extension.service_factory(viz)
    return tuple(enabled_keys)


def clear_runtime_extension_services(viz: Any) -> None:
    """Drop scenario-scoped extension services and their transient state."""
    services = getattr(viz, "extension_services", None)
    if isinstance(services, dict):
        services.clear()
    state = getattr(viz, "app_state", None)
    if not getattr(state, "extension_state", None):
        return
    set_state = getattr(viz, "set_state", None)
    if callable(set_state):
        set_state(extension_state={})


def sync_runtime_extensions(viz: Any, raw_frame: dict[str, Any], step: int) -> bool:
    """Synchronize all active runtime features for one rendered frame."""
    services = getattr(viz, "extension_services", {})
    succeeded = True
    for extension in registered_runtime_extensions():
        service = services.get(extension.key)
        if service is None:
            continue
        succeeded = bool(extension.sync_frame(viz, service, raw_frame, step)) and succeeded
    return succeeded


def runtime_status_chips(renderer: Any, packet: Any = None) -> tuple[str, ...]:
    """Collect status-chip labels from active runtime features."""
    viz = getattr(renderer, "visualizer", None)
    services = getattr(viz, "extension_services", {}) if viz is not None else {}
    chips: list[str] = []
    for extension in registered_runtime_extensions():
        service = services.get(extension.key)
        if service is None or extension.status_chips is None:
            continue
        chips.extend(str(label) for label in extension.status_chips(viz, service, packet))
    return tuple(chips)


def extension_state_value(
    state: Any,
    extension_key: str,
    name: str,
    default: Any = None,
) -> Any:
    """Read one value from an extension's transient AppState namespace."""
    namespaces = getattr(state, "extension_state", {})
    namespace = namespaces.get(extension_key, {}) if isinstance(namespaces, dict) else {}
    return namespace.get(name, default) if isinstance(namespace, dict) else default


def set_extension_state(viz: Any, extension_key: str, **changes: Any) -> None:
    """Replace one extension namespace without mutating an AppState snapshot."""
    namespaces = dict(getattr(viz.app_state, "extension_state", {}) or {})
    namespace = dict(namespaces.get(extension_key, {}) or {})
    namespace.update(changes)
    namespaces[extension_key] = namespace
    viz.set_state(extension_state=namespaces)
