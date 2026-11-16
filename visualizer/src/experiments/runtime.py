"""Pygfx-only registration, with no work until the experiment window is opened."""

from __future__ import annotations

from typing import Any

from ..extensions import RuntimeFeatureExtension, register_runtime_extension


def enabled(viz: Any) -> bool:
    """Require the explicitly supported renderer; never activate on Open3D."""
    return getattr(getattr(viz, "renderer", None), "renderer_id", None) == "pygfx"


def _service(viz: Any) -> Any:
    from .service import ExperimentService

    return ExperimentService(viz)


def register() -> None:
    """Register analysis services; the shared Path tools group owns the launcher."""
    register_runtime_extension(
        RuntimeFeatureExtension(
            key="experiments",
            panel_title="Path analysis",
            tab_label="Paths",
            panel_factory=None,
            service_factory=_service,
            enabled=enabled,
            sync_frame=lambda _viz, _service, _raw, _step: True,
            present_frame=lambda _viz, service, step, vm, packet: service.present(step, vm, packet),
            retire_service=lambda _viz, service: service.close(),
        )
    )
