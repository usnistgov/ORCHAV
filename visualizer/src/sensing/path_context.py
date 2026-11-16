"""Explicit historical RT geometry for verified analytical-Doppler acquisitions.

Empty paths never imply analytical acquisition. Eligibility comes from a
validated stored result, its complete source membership, and current-frame
provenance. Historical geometry is an independent labelled overlay; it never
enters sensing, path statistics, or the stored current frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ..model import RenderObject, Transform
from ..services.sensing_overlay_sync import SensingOverlaySync
from ..state import MpcVisibility
from ..types.render_payloads import LineSetPayload, MaterialPayload, TextLabelPayload


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def analytical_source_record(sensing: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reduce validated result metadata to a bounded, explicit solve reference."""
    config = sensing.get("config_resolved") or {}
    processing = config.get("processing") or {}
    policy = config.get("acquisition") or {}
    if (
        processing.get("mode") != "coherent_cpi"
        or policy.get("sample_source") != "coherent_one_solve"
    ):
        return None
    temporal = sensing.get("temporal") or {}
    samples = (temporal.get("acquisition") or {}).get("samples") or []
    result = temporal.get("result") or {}
    binding = temporal.get("display_binding") or {}
    stride = _integer(policy.get("rt_solve_stride_scene_steps"))
    if not samples or stride is None or stride < 1:
        return None
    sources = [sample.get("source") or {} for sample in samples]
    source = _integer(sources[0].get("scene_frame_id"))
    solve_id = sources[0].get("solve_id")
    if (
        source is None
        or not solve_id
        or binding.get("scene_frame_id") != source
        or result.get("reference_time_policy") != "solve_start"
        or any(
            item.get("scene_frame_id") != source or item.get("solve_id") != solve_id
            for item in sources
        )
        or any(
            sample.get("phase_coherent") is not True or sample.get("valid") is not True
            for sample in samples
        )
    ):
        return None
    reference = result.get("reference_time_s")
    period = policy.get("sample_period_s")
    if (
        isinstance(reference, bool)
        or not isinstance(reference, (int, float))
        or not math.isfinite(reference)
        or isinstance(period, bool)
        or not isinstance(period, (int, float))
        or not math.isfinite(period)
        or period <= 0
    ):
        return None
    return {
        "source_scene_id": source,
        "stop_scene_id": source + stride,
        "reference_time_s": float(reference),
        "duration_s": len(samples) * float(period),
        "solve_id": solve_id,
    }


class AnalyticalPathContext:
    """Own one filtered geometry snapshot and its explicit historical label."""

    def __init__(self) -> None:
        self._sync = SensingOverlaySync()
        self._key: tuple | None = None
        self._line: RenderObject | None = None
        self._label_position = np.zeros(3)
        self._desired: tuple[RenderObject, ...] = ()
        self.status = ""

    @property
    def historical_paths_visible(self) -> bool:
        """Return whether the prepared context includes historical path geometry."""
        return bool(self._desired)

    def prepare(self, viz: Any, service: Any, raw: dict, step: int, packet: Any) -> Any:
        """Prepare a labelled context layer; return the current presentation packet."""
        self._desired = ()
        self.status = ""
        namespace = getattr(viz.app_state, "extension_state", {}).get("sensing", {})
        if not namespace.get("show_rt_context", True):
            self._key = None
            self._line = None
            return packet
        visible = viz.app_state.mpc_visibility
        source = _integer((raw.get("_source") or {}).get("source_rt_frame_idx"))
        if (
            not visible.enabled
            or not (visible.paths or visible.bounce_points)
            or (source is not None and source == step)
        ):
            # A current solve needs no historical replacement. Avoid indexing
            # sensing metadata just to draw ordinary or hidden current paths.
            return packet
        record, generation = service.analytical_path_source(raw, step)
        if record is None:
            return packet
        source = record["source_scene_id"]
        timestamp = service._scene_time_s(raw)
        if timestamp is None:
            return packet
        age = timestamp - record["reference_time_s"]
        if age < -1e-9 or age >= record["duration_s"] - 1e-9:
            return packet
        if source == step:
            self.status = f"Analytical-Doppler · fresh RT geometry: scene {source}"
            return packet
        self.status = (
            f"Analytical-Doppler · historical RT: scene {source} · age {age * 1000:.1f} ms"
        )
        # Even a CPI-end row can contain the earlier solve's paths. Do not
        # display them in the ordinary fresh-path layer when context is active.
        packet = replace(packet, mpc_visibility=MpcVisibility(enabled=False))
        if not visible.enabled or not visible.paths:
            return packet
        materials = getattr(viz, "mpc_allowed_materials", None)
        material_key = (
            None
            if materials is None
            else (str(getattr(viz, "mpc_material_filter_scope", "segment")), *sorted(materials))
        )
        # Metadata/replay consumers do not need the GUI frame pipeline.
        from ..pipeline.frame_pipeline import build_vm_cache_key

        key = (generation, source, build_vm_cache_key(source, viz.app_state, material_key))
        if key != self._key:
            self._key = key
            self._line = None
            loaded = viz.pipeline.load_context_frame(source)
            if loaded is not None:
                source_raw, source_packet = loaded
                source_meta = source_raw.get("_source") or {}
                source_time = service._scene_time_s(source_raw)
                if (
                    source_meta.get("source_rt_frame_idx") == source
                    and source_time is not None
                    and math.isclose(source_time, record["reference_time_s"], abs_tol=1e-9)
                    and len(source_packet.mpc_lines)
                ):
                    payload = LineSetPayload(source_packet.mpc_points, source_packet.mpc_lines)
                    self._line = RenderObject(
                        id="sensing:historical_rt_paths",
                        payload=payload,
                        material=MaterialPayload(
                            base_color=(0.55, 0.75, 0.95, 0.65), line_width=2.0
                        ),
                        metadata={"type": "sensing_context", "source_scene_id": source},
                    )
                    self._label_position = np.array(source_packet.mpc_points[0], copy=True)
                    self._label_position[2] += 1.0
        if self._line is None:
            self.status += " · source paths unavailable"
            return packet
        label = RenderObject(
            id="sensing:historical_rt_label",
            payload=TextLabelPayload(
                text=f"Historical RT geometry\nScene {source} · +{age * 1000:.1f} ms",
                font_size=0.3,
                screen_space=True,
            ),
            material=MaterialPayload(base_color=(0.75, 0.85, 0.95, 1.0)),
            transform=Transform.from_translation(self._label_position),
            metadata={"type": "sensing_context"},
        )
        self._desired = (self._line, label)
        return packet

    def sync(self, renderer: Any) -> bool:
        """Apply only context-owned objects through the common renderer boundary."""
        return self._sync.sync(renderer, self._desired)
