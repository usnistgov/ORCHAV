"""Publish complete target object snapshots through the common renderer API."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..model import RenderObjectState, Transform, VisualEntity
from ..renderers.protocol import renderer_capabilities
from ..types.render_payloads import MaterialPayload
from .entity_render_service import EntityRenderService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer


class TargetRenderSync:
    """Sync target render state through renderer-neutral renderer APIs."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        entity_render_service: EntityRenderService,
    ) -> None:
        """Bind target render synchronization to the visualizer renderer."""
        self.visualizer = visualizer
        self._entity_render_service = entity_render_service
        self._benchmark_metrics_enabled = False
        self._benchmark_metrics: dict[str, float] = {}

    def reset_benchmark_metrics(self, *, enabled: bool | None = None) -> None:
        """Start a fresh target-handoff metrics bucket for the current frame."""
        self._benchmark_metrics_enabled = self._benchmark_active() if enabled is None else enabled
        self._benchmark_metrics = {}

    def get_benchmark_metrics(self) -> dict[str, float]:
        """Return benchmark-scoped target renderer handoff metrics."""
        return dict(self._benchmark_metrics)

    def benchmark_metrics_enabled(self) -> bool:
        """Return whether fine-grained target handoff metrics are active."""
        return bool(self._benchmark_metrics_enabled)

    def record_benchmark_metric(self, name: str, value: float = 1.0) -> None:
        """Accumulate one numeric benchmark metric when metrics are enabled."""
        if not self._benchmark_metrics_enabled:
            return
        self._benchmark_metrics[name] = self._benchmark_metrics.get(name, 0.0) + float(value)

    def _benchmark_active(self) -> bool:
        """Return True when the visualizer is currently running benchmark mode."""
        pipeline = getattr(self.visualizer, "pipeline", None)
        return getattr(pipeline, "benchmark_recorder", None) is not None

    def _time_call(
        self,
        func: Any,
        *args: Any,
        elapsed_metric: str,
        count_metric: str,
        **kwargs: Any,
    ) -> Any:
        """Call a renderer function and record elapsed time when benchmarking."""
        if not self._benchmark_metrics_enabled:
            return func(*args, **kwargs)
        self.record_benchmark_metric(count_metric)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self.record_benchmark_metric(
                elapsed_metric,
                (time.perf_counter() - start) * 1000.0,
            )

    def _record_success(self, metric: str, result: Any) -> None:
        """Record a success count for truthy renderer call results."""
        if bool(result):
            self.record_benchmark_metric(metric)

    def sync_render_handle(
        self,
        geometry: RenderObjectState,
        *,
        effective_visible: bool | None = None,
        snapshot_material: MaterialPayload | None = None,
    ) -> bool:
        """Ensure a neutral handle using a final renderer visibility snapshot."""
        if not isinstance(geometry, RenderObjectState):
            return False
        entity_id = geometry.id.rsplit("::", 1)[0] if "::" in geometry.id else geometry.id
        self.record_benchmark_metric("target_handoff_geometry_sync_count")
        render_object = geometry.to_render_object(effective_visible=effective_visible)
        if snapshot_material is not None:
            render_object = replace(render_object, material=snapshot_material)
        result = self._time_call(
            self._entity_render_service.sync_entity,
            VisualEntity(
                entity_id=entity_id,
                category="target",
                render_object=render_object,
                metadata=geometry.metadata,
            ),
            elapsed_metric="target_handoff_sync_entity_ms",
            count_metric="target_handoff_sync_entity_count",
        )
        self._record_success("target_handoff_sync_entity_success_count", result)
        return bool(result)

    def sync_target_label(
        self,
        *,
        geometry_name: str,
        label: Any,
        visible: bool,
        anchor_position: Any | None = None,
        offset: Any | None = None,
    ) -> bool:
        """Sync target label state through the common object contract."""
        label_start = time.perf_counter()
        self.record_benchmark_metric("target_label_sync_count")
        if not isinstance(label, RenderObjectState) or label.id != geometry_name:
            self.record_benchmark_metric("target_label_invalid_state_count")
            return False
        if anchor_position is not None:
            anchor = np.asarray(anchor_position, dtype=float).reshape(-1)
            delta = (
                np.zeros(3, dtype=float)
                if offset is None
                else np.asarray(offset, dtype=float).reshape(-1)
            )
            if anchor.size < 3 or delta.size < 3:
                self.record_benchmark_metric("target_label_invalid_transform_count")
                return False
            label.metadata["layout_anchor"] = tuple(float(value) for value in anchor[:3])
            label.metadata["layout_offset"] = tuple(float(value) for value in delta[:3])
            label.world_transform = Transform.from_translation(anchor[:3] + delta[:3])
        effective_visible = bool(label.visible and visible)
        self.record_benchmark_metric("target_label_ensure_count")
        self.record_benchmark_metric("target_label_ensure_call_count")
        result = self.sync_render_handle(
            label,
            effective_visible=effective_visible,
        )
        self._record_success("target_label_ensure_success_count", result)
        self.record_benchmark_metric(
            "target_label_sync_ms",
            (time.perf_counter() - label_start) * 1000.0,
        )
        return result

    def sync_mesh_geometry(
        self,
        mesh: RenderObjectState,
        mesh_name: str,
        *,
        visible: bool = True,
        snapshot_material: MaterialPayload | None = None,
    ) -> bool:
        """Sync one target mesh through its stable declarative object ID."""
        if not isinstance(mesh, RenderObjectState) or mesh.id != mesh_name:
            self.record_benchmark_metric("target_handoff_invalid_object_id_count")
            return False
        effective_mesh_visible = bool(mesh.visible and visible)
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer_capabilities(renderer).mesh_vertex_stream_updates:
            render_object = mesh.to_render_object(effective_visible=effective_mesh_visible)
            if snapshot_material is not None:
                render_object = replace(render_object, material=snapshot_material)
            self.record_benchmark_metric("target_vertex_stream_attempt_count")
            result = self._time_call(
                renderer.update_mesh_vertex_stream,
                render_object,
                elapsed_metric="target_vertex_stream_ms",
                count_metric="target_vertex_stream_call_count",
            )
            if bool(result):
                self.record_benchmark_metric("target_vertex_stream_success_count")
                return True
            self.record_benchmark_metric("target_vertex_stream_fallback_count")
        return self.sync_render_handle(
            mesh,
            effective_visible=effective_mesh_visible,
            snapshot_material=snapshot_material,
        )

    def sync_outline_geometry(
        self,
        outline: RenderObjectState,
        outline_name: str,
        *,
        visible: bool = True,
    ) -> bool:
        """Sync one target outline through its stable declarative object ID."""
        if not isinstance(outline, RenderObjectState) or outline.id != outline_name:
            self.record_benchmark_metric("target_handoff_invalid_object_id_count")
            return False
        return self.sync_render_handle(
            outline,
            effective_visible=bool(outline.visible and visible),
        )
