"""Public-core checks for optional visualizer extension discovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from visualizer.src import extension_loader
from visualizer.src import extensions as runtime_extensions
from visualizer.src.beamforming import extensions as beamforming_extensions
from visualizer.src.io import frame_source_extensions


def _missing_external_package(_name: str):
    raise ModuleNotFoundError(
        "No external visualizer extension package is installed",
        name="visualizer_extensions",
    )


def test_public_core_loads_with_no_external_extension_package(monkeypatch) -> None:
    """A public installation exposes empty registries without import failures."""
    monkeypatch.setattr(extension_loader, "_loaded", False)
    monkeypatch.setattr(extension_loader, "import_module", _missing_external_package)
    monkeypatch.setattr(frame_source_extensions, "_REGISTRY", {})
    monkeypatch.setattr(beamforming_extensions, "_REGISTRY", {})
    monkeypatch.setattr(runtime_extensions, "_RUNTIME_EXTENSIONS", {})

    assert frame_source_extensions.registered_frame_source_modes() == []
    assert beamforming_extensions.registered_beamforming_modes() == []
    assert runtime_extensions.registered_runtime_extensions() == ()
    assert extension_loader._loaded is True


def test_extension_dependency_import_failure_is_not_hidden(monkeypatch) -> None:
    """A broken installed extension remains visible and can be retried."""
    calls = 0

    def _missing_dependency(_name: str):
        nonlocal calls
        calls += 1
        raise ModuleNotFoundError(
            "The extension dependency is missing",
            name="extension_dependency",
        )

    monkeypatch.setattr(extension_loader, "_loaded", False)
    monkeypatch.setattr(extension_loader, "import_module", _missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="extension dependency"):
        extension_loader.ensure_external_extensions_loaded()
    with pytest.raises(ModuleNotFoundError, match="extension dependency"):
        extension_loader.ensure_external_extensions_loaded()

    assert calls == 2
    assert extension_loader._loaded is False


def test_clearing_extension_services_also_resets_transient_state() -> None:
    """A later scenario cannot resurrect private controls from an earlier one."""
    state_updates: list[dict] = []
    services = {"example": object()}
    viz = SimpleNamespace(
        extension_services=services,
        app_state=SimpleNamespace(extension_state={"example": {"enabled": True}}),
        set_state=lambda **changes: state_updates.append(changes),
    )

    runtime_extensions.clear_runtime_extension_services(viz)

    assert services == {}
    assert state_updates == [{"extension_state": {}}]
