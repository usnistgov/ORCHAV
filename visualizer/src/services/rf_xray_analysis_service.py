"""Renderer-neutral RF X-Ray material analysis.

The service turns existing scene material metadata and current-frame canonical
MPC arrays into a compact overlay snapshot. Renderers decide how to present the
snapshot; the analysis deliberately stays at material/family granularity until
frame data carries stable per-bounce object IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from shared.frames.manifest import FRAMES_MANIFEST_FILENAME

from ..materials.catalog import (
    is_known_material_type,
    material_preset,
    normalize_material_type_name,
)
from ..metrics.mpc_canon import CanonicalStepData
from ..pipeline.core import FrameRenderPacket
from ..services.object_identity import (
    make_scene_entry_geometry_name,
    make_target_entry_geometry_name,
)
from ..state import (
    DEFAULT_RF_XRAY_OPACITY,
    DEFAULT_RF_XRAY_PROPERTY,
    RF_XRAY_PROPERTY_OPTIONS,
    normalize_rf_xray_opacity,
    normalize_rf_xray_property,
)
from ..utils.colors import ensure_continuous_lut

try:
    from generator.core.materials.defaults import DEFAULT_SCATTERING_COEFFICIENTS
except ImportError:  # pragma: no cover - visualizer-only packaging fallback
    DEFAULT_SCATTERING_COEFFICIENTS = {
        "marble": 0.08,
        "glass": 0.05,
        "metal": 0.10,
        "wood": 0.25,
        "plywood": 0.20,
        "chipboard": 0.25,
        "plasterboard": 0.35,
        "concrete": 0.40,
        "ceiling_board": 0.50,
    }

RFXRAY_MODE_MATERIAL_MAP = "material_map"
RFXRAY_MODE_MPC_USAGE = "mpc_usage"
RFXRAY_MODE_MATERIAL_PROPERTIES = "material_properties"
RFXRAY_MODES = frozenset(
    {RFXRAY_MODE_MATERIAL_MAP, RFXRAY_MODE_MPC_USAGE, RFXRAY_MODE_MATERIAL_PROPERTIES}
)
RFXRAY_PROPERTY_LABELS = {
    "relative_permittivity": "Relative permittivity",
    "conductivity": "Conductivity",
    "scattering_coefficient": "Scattering coefficient",
    "xpd_coefficient": "XPD coefficient",
    "thickness": "Thickness",
}

_NO_MATERIAL = "no-material"
_MISSING_COLOR = (1.0, 0.18, 0.08, 0.84)
_UNUSED_COLOR = (0.18, 0.20, 0.24, 0.28)
_BOUNCE_FALLBACK_COLOR = (1.0, 1.0, 1.0, 0.95)


@dataclass(frozen=True, slots=True)
class RFXRayMaterialUsage:
    """Aggregated material contribution for one RF X-Ray material key."""

    material_key: str
    display_name: str
    family: str
    bounce_count: int = 0
    path_count: int = 0
    weight: float = 0.0
    normalized_score: float = 0.0
    property_value: float | None = None
    unknown_material: bool = False
    color: tuple[float, float, float, float] = _UNUSED_COLOR


@dataclass(frozen=True, slots=True)
class RFXRayLegendEntry:
    """Renderer-neutral swatch entry for categorical RF X-Ray overlays."""

    material_key: str
    display_name: str
    color: tuple[float, float, float, float]
    value: float | None = None
    missing_data: bool = False


@dataclass(frozen=True, slots=True)
class RFXRayTopPath:
    """One strongest path selected for optional RF X-Ray line highlighting."""

    path_id: int
    weight: float
    path_loss_db: float | None
    material_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RFXRayAnalysisSnapshot:
    """Renderer-neutral RF X-Ray overlay payload."""

    enabled: bool
    mode: str
    signature: tuple[Any, ...]
    overlay_opacity: float = DEFAULT_RF_XRAY_OPACITY
    material_colors: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    geometry_colors: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    usage: tuple[RFXRayMaterialUsage, ...] = ()
    bounce_points: np.ndarray | None = None
    bounce_colors: np.ndarray | None = None
    top_path_points: np.ndarray | None = None
    top_path_lines: np.ndarray | None = None
    top_path_colors: np.ndarray | None = None
    top_paths: tuple[RFXRayTopPath, ...] = ()
    legend_entries: tuple[RFXRayLegendEntry, ...] = ()
    scalar_property: str | None = None
    scalar_property_label: str | None = None
    scalar_range: tuple[float, float] | None = None
    missing_material_keys: tuple[str, ...] = ()
    summary: str = ""


@dataclass(slots=True)
class _SceneMaterialInfo:
    """Material metadata attached to one rendered scene or target geometry."""

    material_key: str
    display_name: str
    geometry_names: set[str] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)
    unknown_material: bool = False


@dataclass(slots=True)
class _UsageAccumulator:
    """Mutable aggregation state before freezing a material usage row."""

    material_key: str
    display_name: str
    family: str
    unknown_material: bool
    bounce_count: int = 0
    weight: float = 0.0
    path_ids: set[int] = field(default_factory=set)


class RFXRayAnalysisService:
    """Build RF X-Ray material overlay snapshots from current visualizer state."""

    def __init__(self, visualizer: Any) -> None:
        """Store the owning visualizer without taking a renderer dependency."""
        self.visualizer = visualizer

    def build_snapshot(self, packet: FrameRenderPacket | None) -> RFXRayAnalysisSnapshot:
        """Return the current RF X-Ray analysis snapshot.

        The disabled snapshot is cheap and carries enough signature information
        for renderers to clear stale overlay state.
        """
        state = getattr(self.visualizer, "app_state", None)
        mode = self._normalize_mode(getattr(state, "rf_xray_mode", RFXRAY_MODE_MATERIAL_MAP))
        selected_property = normalize_rf_xray_property(
            getattr(state, "rf_xray_property", DEFAULT_RF_XRAY_PROPERTY)
        )
        enabled = bool(getattr(state, "show_rf_xray", False))
        opacity = normalize_rf_xray_opacity(
            getattr(state, "rf_xray_opacity", DEFAULT_RF_XRAY_OPACITY)
        )
        scene_materials = self._collect_scene_materials()
        rf_properties, scattering_preset = self._collect_scenario_material_properties()
        signature = self._signature(
            packet,
            state,
            scene_materials,
            rf_properties,
            scattering_preset,
            enabled,
            mode,
            selected_property,
            opacity,
        )

        if not enabled or packet is None:
            return RFXRayAnalysisSnapshot(
                enabled=False,
                mode=mode,
                signature=signature,
                overlay_opacity=opacity,
                summary="RF X-Ray off",
            )

        if mode == RFXRAY_MODE_MPC_USAGE:
            return self._build_mpc_usage_snapshot(
                packet,
                state,
                scene_materials,
                signature,
                opacity,
            )
        if mode == RFXRAY_MODE_MATERIAL_PROPERTIES:
            return self._build_material_properties_snapshot(
                scene_materials,
                rf_properties,
                scattering_preset,
                selected_property,
                signature,
                opacity,
            )
        return self._build_material_map_snapshot(scene_materials, signature, opacity)

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        """Return a supported RF X-Ray mode name."""
        value = str(mode or "").strip().lower()
        return value if value in RFXRAY_MODES else RFXRAY_MODE_MATERIAL_MAP

    def _signature(
        self,
        packet: FrameRenderPacket | None,
        state: Any,
        scene_materials: Mapping[str, _SceneMaterialInfo],
        rf_properties: Mapping[str, Mapping[str, float]],
        scattering_preset: str,
        enabled: bool,
        mode: str,
        selected_property: str,
        opacity: float,
    ) -> tuple[Any, ...]:
        """Build an O(1)-style freshness signature for overlay snapshots."""
        scene_key = tuple(
            sorted(
                (
                    key,
                    tuple(sorted(info.geometry_names)),
                    bool(info.unknown_material),
                    self._props_token(info.properties),
                )
                for key, info in scene_materials.items()
            )
        )
        rf_props_key = tuple(
            sorted((key, tuple(sorted(props.items()))) for key, props in rf_properties.items())
        )
        return (
            bool(enabled),
            mode,
            selected_property,
            round(float(opacity), 4),
            bool(getattr(state, "rf_xray_show_top_paths", False)),
            int(getattr(state, "rf_xray_max_top_paths", 12)),
            scattering_preset,
            rf_props_key,
            None if packet is None else packet.mpc_line_revision,
            self._array_token(None if packet is None else packet.segment_mask),
            self._canonical_usage_token(None if packet is None else packet.canonical_data),
            scene_key,
        )

    @staticmethod
    def _array_token(array: np.ndarray | None) -> tuple[Any, ...] | None:
        """Return a small identity token for a frame-packet array."""
        if array is None:
            return None
        return (id(array), tuple(array.shape), array.dtype.str, int(array.nbytes))

    @classmethod
    def _canonical_usage_token(cls, canon: CanonicalStepData | None) -> tuple[Any, ...] | None:
        """Return an O(1) token for canonical fields used by MPC usage coloring."""
        if canon is None:
            return None
        return (
            id(canon),
            cls._array_token(canon.segment_material_ids),
            cls._array_token(canon.segment_path_id),
            cls._array_token(canon.path_losses),
            cls._array_token(canon.lines),
        )

    @staticmethod
    def _props_token(props: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return a small token for material-map color-relevant properties."""
        color = props.get("color")
        try:
            rgb = tuple(round(float(color[i]), 6) for i in range(3))
        except (TypeError, ValueError, IndexError):
            rgb = ()
        return (rgb, str(props.get("material_type", "")))

    def _build_material_map_snapshot(
        self,
        scene_materials: Mapping[str, _SceneMaterialInfo],
        signature: tuple[Any, ...],
        opacity: float,
    ) -> RFXRayAnalysisSnapshot:
        """Color scene geometry by assigned material type and catalog availability."""
        material_colors: dict[str, tuple[float, float, float, float]] = {}
        geometry_colors: dict[str, tuple[float, float, float, float]] = {}
        usage_rows: list[RFXRayMaterialUsage] = []
        legend_entries: list[RFXRayLegendEntry] = []

        for key, info in sorted(scene_materials.items()):
            color = (
                _MISSING_COLOR
                if info.unknown_material
                else self._material_map_color(key, info.properties)
            )
            color = self._with_overlay_opacity(color, opacity)
            material_colors[key] = color
            for geometry_name in info.geometry_names:
                geometry_colors[geometry_name] = color
            usage_rows.append(
                RFXRayMaterialUsage(
                    material_key=key,
                    display_name=info.display_name,
                    family=self._family_for_key(key),
                    unknown_material=info.unknown_material,
                    normalized_score=0.0 if info.unknown_material else 1.0,
                    color=color,
                )
            )
            legend_entries.append(
                RFXRayLegendEntry(
                    material_key=key,
                    display_name=info.display_name,
                    color=color,
                    missing_data=info.unknown_material,
                )
            )

        summary = f"RF X-Ray material map: {len(scene_materials)} scene materials"
        missing_count = sum(1 for row in usage_rows if row.unknown_material)
        if missing_count:
            summary += f", {missing_count} unknown"

        return RFXRayAnalysisSnapshot(
            enabled=True,
            mode=RFXRAY_MODE_MATERIAL_MAP,
            signature=signature,
            overlay_opacity=opacity,
            material_colors=material_colors,
            geometry_colors=geometry_colors,
            usage=tuple(usage_rows),
            legend_entries=tuple(legend_entries),
            summary=summary,
        )

    def _build_mpc_usage_snapshot(
        self,
        packet: FrameRenderPacket,
        state: Any,
        scene_materials: Mapping[str, _SceneMaterialInfo],
        signature: tuple[Any, ...],
        opacity: float,
    ) -> RFXRayAnalysisSnapshot:
        """Color scene geometry by current-frame MPC material contribution."""
        canon = packet.canonical_data
        if canon is None:
            return RFXRayAnalysisSnapshot(
                enabled=True,
                mode=RFXRAY_MODE_MPC_USAGE,
                signature=signature,
                overlay_opacity=opacity,
                summary="RF X-Ray MPC usage: no MPC material data",
            )

        accumulators = self._aggregate_usage(canon, packet.segment_mask)
        max_weight = max((acc.weight for acc in accumulators.values()), default=0.0)

        material_colors: dict[str, tuple[float, float, float, float]] = {}
        geometry_colors: dict[str, tuple[float, float, float, float]] = {}
        usage_rows: list[RFXRayMaterialUsage] = []

        usage_row_keys: set[str] = set()
        all_material_keys = set(scene_materials) | set(accumulators)
        for key in sorted(all_material_keys, key=lambda item: (item not in scene_materials, item)):
            info = scene_materials.get(key)
            usage_key = key
            accumulator = accumulators.get(key)
            if accumulator is None and info is not None:
                usage_key, accumulator = self._scene_usage_accumulator(key, accumulators)
            missing = (
                bool(info.unknown_material)
                if info is not None
                else not is_known_material_type(usage_key)
            )
            weight = float(accumulator.weight) if accumulator is not None else 0.0
            score = weight / max_weight if max_weight > 0.0 else 0.0
            color = _MISSING_COLOR if missing and accumulator is None else self._usage_color(score)
            color = self._with_overlay_opacity(color, opacity)
            material_colors[key] = color
            if accumulator is not None and usage_key != key:
                material_colors[usage_key] = color
            if info is not None:
                for geometry_name in info.geometry_names:
                    geometry_colors[geometry_name] = color
            if accumulator is not None and usage_key not in usage_row_keys:
                usage_row_keys.add(usage_key)
                usage_rows.append(
                    RFXRayMaterialUsage(
                        material_key=usage_key,
                        display_name=accumulator.display_name,
                        family=accumulator.family,
                        bounce_count=accumulator.bounce_count,
                        path_count=len(accumulator.path_ids),
                        weight=weight,
                        normalized_score=score,
                        unknown_material=missing,
                        color=color,
                    )
                )

        top_paths = self._select_top_paths(
            canon,
            packet.segment_mask,
            int(getattr(state, "rf_xray_max_top_paths", 12)),
        )
        top_path_points = top_path_lines = top_path_colors = None
        if bool(getattr(state, "rf_xray_show_top_paths", False)) and top_paths:
            top_path_points, top_path_lines, top_path_colors = self._build_top_path_lines(
                canon,
                top_paths,
                material_colors,
            )
        else:
            top_paths = ()

        ordered = tuple(sorted(usage_rows, key=lambda row: (-row.weight, row.material_key)))
        summary = f"RF X-Ray MPC usage: {sum(row.bounce_count for row in ordered)} material bounces"
        if ordered:
            summary += f", top material {ordered[0].display_name}"

        return RFXRayAnalysisSnapshot(
            enabled=True,
            mode=RFXRAY_MODE_MPC_USAGE,
            signature=signature,
            overlay_opacity=opacity,
            material_colors=material_colors,
            geometry_colors=geometry_colors,
            usage=ordered,
            top_path_points=top_path_points,
            top_path_lines=top_path_lines,
            top_path_colors=top_path_colors,
            top_paths=top_paths,
            summary=summary,
        )

    def _build_material_properties_snapshot(
        self,
        scene_materials: Mapping[str, _SceneMaterialInfo],
        rf_properties: Mapping[str, Mapping[str, float]],
        scattering_preset: str,
        selected_property: str,
        signature: tuple[Any, ...],
        opacity: float,
    ) -> RFXRayAnalysisSnapshot:
        """Color scene geometry by configured scalar RF material properties."""
        selected_property = normalize_rf_xray_property(selected_property)
        values: dict[str, float] = {}
        for key in scene_materials:
            value = self._material_property_value(
                key,
                selected_property,
                rf_properties,
                scattering_preset,
            )
            if value is not None:
                values[key] = value

        value_min = min(values.values()) if values else 0.0
        value_max = max(values.values()) if values else 1.0

        material_colors: dict[str, tuple[float, float, float, float]] = {}
        geometry_colors: dict[str, tuple[float, float, float, float]] = {}
        usage_rows: list[RFXRayMaterialUsage] = []
        missing_keys: list[str] = []

        for key, info in sorted(scene_materials.items()):
            value = values.get(key)
            missing = value is None
            if missing:
                missing_keys.append(key)
                score = 0.0
                color = _UNUSED_COLOR
            else:
                score = self._normalize_scalar_value(value, value_min, value_max)
                color = self._usage_color(score)
            color = self._with_overlay_opacity(color, opacity)
            material_colors[key] = color
            for geometry_name in info.geometry_names:
                geometry_colors[geometry_name] = color
            usage_rows.append(
                RFXRayMaterialUsage(
                    material_key=key,
                    display_name=info.display_name,
                    family=self._family_for_key(key),
                    weight=float(value) if value is not None else 0.0,
                    normalized_score=score,
                    property_value=value,
                    unknown_material=info.unknown_material,
                    color=color,
                )
            )

        label = RFXRAY_PROPERTY_LABELS.get(selected_property, selected_property)
        summary = (
            f"RF X-Ray material properties: {label} for "
            f"{len(values)}/{len(scene_materials)} scene materials"
        )
        if missing_keys:
            missing_names = ", ".join(self._display_name_for_key(key) for key in missing_keys[:4])
            if len(missing_keys) > 4:
                missing_names += f", +{len(missing_keys) - 4} more"
            summary += f"; missing data: {missing_names}"

        return RFXRayAnalysisSnapshot(
            enabled=True,
            mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
            signature=signature,
            overlay_opacity=opacity,
            material_colors=material_colors,
            geometry_colors=geometry_colors,
            usage=tuple(usage_rows),
            scalar_property=selected_property,
            scalar_property_label=label,
            scalar_range=(float(value_min), float(value_max)) if values else None,
            missing_material_keys=tuple(missing_keys),
            summary=summary,
        )

    def _collect_scene_materials(self) -> dict[str, _SceneMaterialInfo]:
        """Collect rendered scene/target geometries by normalized material key."""
        materials: dict[str, _SceneMaterialInfo] = {}
        for entry, geometry_name, props in self._iter_rendered_material_entries():
            key = self._entry_material_key(entry, props)
            info = materials.setdefault(
                key,
                _SceneMaterialInfo(
                    material_key=key,
                    display_name=self._display_name_for_key(key),
                    unknown_material=not is_known_material_type(key),
                ),
            )
            info.geometry_names.add(geometry_name)
            if props:
                info.properties.update(props)
            if not is_known_material_type(key):
                info.unknown_material = True
        return materials

    def _collect_scenario_material_properties(self) -> tuple[dict[str, dict[str, float]], str]:
        """Return normalized effective RF material properties for RF X-Ray."""
        scenario_config = getattr(self.visualizer, "scenario_config", None)
        raytracing = self._mapping_get(scenario_config, "raytracing", None)

        properties: dict[str, dict[str, float]] = {}
        self._merge_material_properties(properties, self._collect_manifest_material_properties())
        self._merge_material_properties(
            properties,
            self._collect_frame_source_material_properties(),
        )
        self._merge_material_properties(properties, self._collect_live_material_properties())

        scene_materials = self._mapping_get(raytracing, "scene_materials", None)
        preset = (
            str(
                self._mapping_get(scene_materials, "scattering_coefficient_preset", "none")
                or "none"
            )
            .strip()
            .lower()
        )
        if preset != "itu":
            preset = "none"

        materials = self._mapping_get(raytracing, "materials", None)
        if not isinstance(materials, Mapping):
            return properties, preset

        for raw_key, raw_values in materials.items():
            key = self._normalize_material_key(raw_key)
            if key == _NO_MATERIAL:
                continue
            values = self._material_override_to_scalars(raw_values)
            if values:
                properties.setdefault(key, {}).update(values)
        return properties, preset

    @staticmethod
    def _merge_material_properties(
        target: dict[str, dict[str, float]],
        source: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Merge normalized material property mappings in-place."""
        for key, values in source.items():
            if values:
                target.setdefault(key, {}).update(values)

    def _collect_manifest_material_properties(self) -> dict[str, dict[str, float]]:
        """Return RF material properties stored once in the frame manifest."""
        manifest = self._load_frames_manifest()
        if not manifest:
            return {}
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            return {}
        return self._material_property_block_to_scalars(provenance.get("material_properties"))

    def _collect_frame_source_material_properties(self) -> dict[str, dict[str, float]]:
        """Return RF material properties exposed by the active frame source metadata."""
        frame_source = getattr(self.visualizer, "frame_source", None)
        metadata = getattr(frame_source, "metadata", None)
        if callable(metadata):
            try:
                metadata = metadata()
            except (AttributeError, TypeError, ValueError):
                metadata = None
        if not isinstance(metadata, Mapping):
            return {}

        raw_block = metadata.get("material_properties")
        if raw_block is None:
            raw_json = metadata.get("material_properties_json")
            if isinstance(raw_json, str) and raw_json.strip():
                try:
                    raw_block = json.loads(raw_json)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw_block = None
        return self._material_property_block_to_scalars(raw_block)

    def _load_frames_manifest(self) -> dict[str, Any]:
        """Load the generated frame-set manifest, if this scenario has one."""
        path = self._frames_manifest_path()
        if path is None:
            return {}
        try:
            stat = path.stat()
        except OSError:
            return {}
        cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        if getattr(self, "_frames_manifest_cache_key", None) == cache_key:
            return dict(getattr(self, "_frames_manifest_cache", {}))
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        self._frames_manifest_cache_key = cache_key
        self._frames_manifest_cache = dict(payload)
        return dict(payload)

    def _frames_manifest_path(self) -> Path | None:
        """Resolve candidate frame-set manifest locations from loaded context."""
        candidates: list[Path] = []

        frame_source = getattr(self.visualizer, "frame_source", None)
        source_root = getattr(frame_source, "root", None)
        source_directory = getattr(frame_source, "directory", None)
        if source_root is not None and source_directory is not None:
            selected_directory = Path(str(source_directory or "frames"))
            if not selected_directory.is_absolute():
                selected_directory = Path(source_root) / selected_directory
            candidates.append(selected_directory / FRAMES_MANIFEST_FILENAME)

        scenario = getattr(self.visualizer, "scenario", None)
        scenario_frames_dir = getattr(scenario, "frames_dir", None)
        if scenario_frames_dir is not None:
            candidates.append(Path(scenario_frames_dir) / FRAMES_MANIFEST_FILENAME)

        scenario_config = getattr(self.visualizer, "scenario_config", None)
        frames_dir = self._mapping_get(scenario_config, "frames_dir", None)
        if frames_dir is not None:
            candidates.append(Path(frames_dir) / FRAMES_MANIFEST_FILENAME)
        else:
            root = self._mapping_get(scenario_config, "root", None)
            data_spec = self._mapping_get(scenario_config, "data_spec", None)
            files_spec = self._mapping_get(data_spec, "files", None)
            directory = self._mapping_get(files_spec, "directory", "frames")
            if root is not None:
                selected_directory = Path(str(directory or "frames"))
                if not selected_directory.is_absolute():
                    selected_directory = Path(root) / selected_directory
                candidates.append(selected_directory / FRAMES_MANIFEST_FILENAME)

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
        return None

    @classmethod
    def _material_property_block_to_scalars(cls, raw_block: Any) -> dict[str, dict[str, float]]:
        """Normalize a manifest/live material property block to analysis keys."""
        if not isinstance(raw_block, Mapping):
            return {}
        raw_properties = raw_block.get("properties", raw_block)
        if not isinstance(raw_properties, Mapping):
            return {}

        properties: dict[str, dict[str, float]] = {}
        for raw_key, raw_values in raw_properties.items():
            key = cls._normalize_material_key(raw_key)
            if key == _NO_MATERIAL:
                continue
            values = cls._material_override_to_scalars(raw_values)
            if values:
                properties.setdefault(key, {}).update(values)
        return properties

    def _collect_live_material_properties(self) -> dict[str, dict[str, float]]:
        """Return resolved RF properties from an attached live Sionna scene."""
        for scene in self._iter_live_sionna_scenes():
            materials = getattr(scene, "radio_materials", None)
            if isinstance(materials, Mapping):
                return self._radio_materials_to_scalars(materials)
        return {}

    def _iter_live_sionna_scenes(self) -> Iterable[Any]:
        """Yield likely live Sionna scene holders without importing Sionna."""
        seen: set[int] = set()
        candidates = [
            getattr(self.visualizer, name, None)
            for name in (
                "sionna_scene",
                "rt_scene",
                "radio_scene",
                "raytracing_scene",
                "scene",
            )
        ]
        for service_name in ("scene_service", "generator_scene_service"):
            service = getattr(self.visualizer, service_name, None)
            if service is not None:
                candidates.extend(
                    getattr(service, name, None)
                    for name in ("sionna_scene", "rt_scene", "scene", "radio_scene")
                )
        for candidate in candidates:
            if candidate is None:
                continue
            ident = id(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            if hasattr(candidate, "radio_materials"):
                yield candidate

    @classmethod
    def _radio_materials_to_scalars(
        cls,
        materials: Mapping[Any, Any],
    ) -> dict[str, dict[str, float]]:
        """Normalize live Sionna ``radio_materials`` values to analysis keys."""
        properties: dict[str, dict[str, float]] = {}
        for raw_key, material in materials.items():
            scalars = cls._material_to_scalars(material)
            if not scalars:
                continue
            for name in cls._material_name_candidates(raw_key, material):
                key = cls._normalize_material_key(name)
                if key != _NO_MATERIAL:
                    properties.setdefault(key, {}).update(scalars)
        return properties

    @staticmethod
    def _material_name_candidates(raw_key: Any, material: Any) -> tuple[str, ...]:
        """Return dictionary and object names for one Sionna material."""
        names: list[str] = []
        for value in (raw_key, getattr(material, "name", None)):
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in names:
                names.append(text)
        return tuple(names)

    @classmethod
    def _material_to_scalars(cls, material: Any) -> dict[str, float]:
        """Extract supported scalar RF fields from one live material object."""
        scalars: dict[str, float] = {}
        for field_name in RF_XRAY_PROPERTY_OPTIONS:
            try:
                value = getattr(material, field_name)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
            scalar = cls._coerce_scalar(value)
            if scalar is not None:
                scalars[field_name] = scalar
        return scalars

    @staticmethod
    def _coerce_scalar(value: Any) -> float | None:
        """Coerce Python, NumPy, tensor, or DrJit-like scalar values."""
        if value is None or callable(value):
            return None
        for attr_name in ("numpy", "item"):
            method = getattr(value, attr_name, None)
            if callable(method):
                try:
                    value = method()
                except (TypeError, ValueError, RuntimeError):
                    return None
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return None
            if array.size != 1:
                return None
            try:
                scalar = float(array.reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                return None
        return scalar if np.isfinite(scalar) else None

    @staticmethod
    def _mapping_get(source: Any, key: str, default: Any = None) -> Any:
        """Read a key from a mapping or an attribute from a model-like object."""
        if source is None:
            return default
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    @classmethod
    def _material_override_to_scalars(cls, raw_values: Any) -> dict[str, float]:
        """Extract supported scalar RF properties from one material override."""
        if raw_values is None:
            return {}
        if isinstance(raw_values, Mapping):
            source = raw_values
        elif hasattr(raw_values, "model_dump"):
            try:
                source = raw_values.model_dump(exclude_none=True)
            except (AttributeError, TypeError, ValueError):
                source = {}
        else:
            source = {
                field: getattr(raw_values, field)
                for field in RF_XRAY_PROPERTY_OPTIONS
                if hasattr(raw_values, field)
            }

        scalars: dict[str, float] = {}
        for field_name in RF_XRAY_PROPERTY_OPTIONS:
            value = cls._mapping_get(source, field_name, None)
            if value is None:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                scalars[field_name] = scalar
        return scalars

    @staticmethod
    def _normalize_scalar_value(value: float, value_min: float, value_max: float) -> float:
        """Normalize one scalar value to a colorbar score."""
        if not np.isfinite(value):
            return 0.0
        span = float(value_max) - float(value_min)
        if abs(span) <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (float(value) - float(value_min)) / span))

    @classmethod
    def _material_property_value(
        cls,
        material_key: str,
        selected_property: str,
        rf_properties: Mapping[str, Mapping[str, float]],
        scattering_preset: str,
    ) -> float | None:
        """Return the configured scalar value for a material, if available."""
        value = rf_properties.get(material_key, {}).get(selected_property)
        if value is not None:
            return float(value)
        if selected_property != "scattering_coefficient" or scattering_preset != "itu":
            return None
        family = cls._family_for_scattering_defaults(material_key)
        if family is None:
            return None
        preset_value = DEFAULT_SCATTERING_COEFFICIENTS.get(family)
        return float(preset_value) if preset_value is not None else None

    @staticmethod
    def _family_for_scattering_defaults(material_key: str) -> str | None:
        """Resolve a normalized material key to an ITU scattering default family."""
        key = str(material_key or "").strip().lower()
        if not key:
            return None
        if key in DEFAULT_SCATTERING_COEFFICIENTS:
            return key
        for prefix in ("mat-itu_", "mat_itu_", "mat-", "itu_", "itu-"):
            if key.startswith(prefix):
                candidate = key[len(prefix) :]
                if candidate in DEFAULT_SCATTERING_COEFFICIENTS:
                    return candidate
        return None

    def _iter_rendered_material_entries(
        self,
    ) -> Iterable[tuple[dict[str, Any], str, dict[str, Any]]]:
        """Yield scene and target entries with their current renderer geometry name."""
        scene_service = getattr(self.visualizer, "scene_service", None)
        merged_seen: set[str] = set()

        for entry in getattr(self.visualizer, "mesh_entries", []) or []:
            if not isinstance(entry, dict):
                continue
            mesh = entry.get("mesh")
            group_name = None
            if mesh is not None and scene_service is not None:
                group_name = getattr(scene_service, "_mesh_id_to_group", {}).get(id(mesh))
            if group_name:
                if group_name in merged_seen:
                    continue
                merged_seen.add(group_name)
                merged_info = getattr(scene_service, "_merged_meshes", {}).get(group_name, {})
                yield entry, group_name, dict(merged_info.get("pbr_props") or {})
                continue
            yield entry, entry.get("geometry_name") or make_scene_entry_geometry_name(entry), (
                self._entry_effective_props(entry)
            )

        for entry in getattr(self.visualizer, "target_entries", []) or []:
            if not isinstance(entry, dict):
                continue
            yield entry, entry.get("geometry_name") or make_target_entry_geometry_name(entry), (
                self._entry_effective_props(entry)
            )

    def _entry_effective_props(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return effective material properties for one scene/target entry."""
        props = dict(entry.get("pbr_properties") or {})
        key = self._entry_material_key(entry, props)
        if not props and is_known_material_type(key):
            props = material_preset(key)
        pbr_service = getattr(self.visualizer, "material_pbr_service", None)
        if pbr_service is not None and key:
            try:
                props = dict(pbr_service.get_effective_properties(key, props))
            except (AttributeError, TypeError, ValueError):
                pass
        return props

    @classmethod
    def _entry_material_key(cls, entry: dict[str, Any], props: Mapping[str, Any]) -> str:
        """Return the normalized material key for scene and target entries."""
        for value in (
            props.get("material_type"),
            entry.get("material_type"),
            entry.get("material_id"),
            entry.get("material"),
        ):
            key = cls._normalize_material_key(value)
            if key != _NO_MATERIAL:
                return key
        return "default"

    @classmethod
    def _material_key_from_id(cls, canon: CanonicalStepData, material_id: int) -> str:
        """Resolve a canonical material ID to the normalized analysis key."""
        if material_id == 0:
            return _NO_MATERIAL
        for mapping in (
            canon.material_id_to_bare,
            canon.material_id_to_itu,
            canon.material_id_to_name,
        ):
            if mapping:
                key = cls._normalize_material_key(mapping.get(int(material_id), ""))
                if key != _NO_MATERIAL:
                    return key
        return f"material-{int(material_id)}"

    @classmethod
    def _scene_usage_accumulator(
        cls,
        scene_key: str,
        accumulators: Mapping[str, _UsageAccumulator],
    ) -> tuple[str, _UsageAccumulator | None]:
        """Resolve an MPC usage accumulator for scene material aliases."""
        for candidate in cls._scene_usage_aliases(scene_key):
            accumulator = accumulators.get(candidate)
            if accumulator is not None:
                return candidate, accumulator
        return scene_key, None

    @classmethod
    def _scene_usage_aliases(cls, scene_key: str) -> tuple[str, ...]:
        """Return canonical MPC labels that may describe a rendered scene key."""
        key = cls._normalize_material_key(scene_key)
        aliases = [key]
        if key.startswith("ground_"):
            suffix = cls._normalize_material_key(key[len("ground_") :])
            if suffix != _NO_MATERIAL:
                aliases.append(suffix)
        else:
            aliases.append(cls._normalize_material_key(f"ground_{key}"))
        return tuple(dict.fromkeys(alias for alias in aliases if alias != _NO_MATERIAL))

    @staticmethod
    def _normalize_material_key(value: Any) -> str:
        """Normalize scene and canonical material names to one family key."""
        if value is None:
            return _NO_MATERIAL
        text = str(value).strip().lower()
        if not text or text == "none":
            return _NO_MATERIAL
        for prefix in ("mat-itu_", "mat_itu_", "mat-", "itu_", "itu-"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        return normalize_material_type_name(text, default=_NO_MATERIAL)

    @staticmethod
    def _display_name_for_key(key: str) -> str:
        """Format a normalized material key for panel/status display."""
        return "No material" if key == _NO_MATERIAL else key.replace("_", " ").title()

    @staticmethod
    def _family_for_key(key: str) -> str:
        """Collapse detailed material types into broad RF inspection families."""
        if key in {_NO_MATERIAL, "default"}:
            return key
        if "ground" in key or key.endswith("asphalt"):
            return "ground"
        if key in {"glass", "water"}:
            return "transparent"
        if key in {"metal"}:
            return "metal"
        if key in {"wood", "floorboard", "plywood", "chipboard"}:
            return "wood"
        if key in {"vegetation", "skin"}:
            return "organic"
        return "building"

    @staticmethod
    def _material_map_color(
        material_key: str,
        props: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        """Return a visible material-map color from configured material props."""
        color = props.get("color")
        if color is None and is_known_material_type(material_key):
            color = material_preset(material_key).get("color")
        try:
            rgb = tuple(float(color[i]) for i in range(3))
        except (TypeError, ValueError, IndexError):
            rgb = (0.72, 0.72, 0.72)
        # Lift dark catalog colors so the overlay reads as an inspection layer.
        lifted = tuple(min(1.0, 0.25 + channel * 0.8) for channel in rgb)
        return (lifted[0], lifted[1], lifted[2], 0.82)

    @staticmethod
    def _usage_color(score: float) -> tuple[float, float, float, float]:
        """Map a normalized usage score through the standard continuous LUT."""
        value = max(0.0, min(1.0, float(score)))
        if value <= 0.0:
            return _UNUSED_COLOR
        try:
            lut = np.asarray(ensure_continuous_lut(), dtype=np.float32)
            if lut.ndim == 2 and lut.shape[0] > 0 and lut.shape[1] >= 3:
                index = int(round(value * float(lut.shape[0] - 1)))
                rgb = np.clip(lut[index, :3], 0.0, 1.0)
                return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.86)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return (value, value, value, 0.86)

    @staticmethod
    def _with_overlay_opacity(
        color: tuple[float, float, float, float],
        opacity: float,
    ) -> tuple[float, float, float, float]:
        """Apply RF X-Ray opacity as absolute overlay alpha."""
        alpha = normalize_rf_xray_opacity(opacity)
        return (float(color[0]), float(color[1]), float(color[2]), alpha)

    def _aggregate_usage(
        self,
        canon: CanonicalStepData,
        segment_mask: np.ndarray | None,
    ) -> dict[str, _UsageAccumulator]:
        """Aggregate visible material-bearing segments by material key."""
        if canon.segment_material_ids is None or canon.lines.size == 0:
            return {}
        visible = self._visible_segment_mask(canon, segment_mask)
        if not np.any(visible):
            return {}

        segment_material_ids = canon.segment_material_ids.astype(np.int32, copy=False)
        segment_path_ids = self._segment_path_ids(canon)
        accumulators: dict[str, _UsageAccumulator] = {}

        for segment_index in np.flatnonzero(visible):
            material_id = int(segment_material_ids[segment_index])
            if material_id == 0:
                continue
            key = self._material_key_from_id(canon, material_id)
            path_id = (
                int(segment_path_ids[segment_index])
                if segment_path_ids is not None and segment_index < segment_path_ids.size
                else -1
            )
            weight = self._path_weight(canon, path_id)
            acc = accumulators.setdefault(
                key,
                _UsageAccumulator(
                    material_key=key,
                    display_name=self._display_name_for_key(key),
                    family=self._family_for_key(key),
                    unknown_material=not is_known_material_type(key),
                ),
            )
            acc.bounce_count += 1
            acc.weight += weight
            if path_id >= 0:
                acc.path_ids.add(path_id)
        return accumulators

    @staticmethod
    def _visible_segment_mask(
        canon: CanonicalStepData,
        segment_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Return a bool mask aligned with canonical segments."""
        n_segments = int(canon.lines.shape[0]) if canon.lines is not None else 0
        if segment_mask is None:
            return np.ones(n_segments, dtype=bool)
        mask = np.asarray(segment_mask, dtype=bool)
        if mask.shape != (n_segments,):
            out = np.zeros(n_segments, dtype=bool)
            count = min(n_segments, int(mask.size))
            if count:
                out[:count] = mask.reshape(-1)[:count]
            return out
        return mask

    @staticmethod
    def _segment_path_ids(canon: CanonicalStepData) -> np.ndarray | None:
        """Return segment-aligned path IDs when canonical data provides them."""
        if canon.segment_path_id is not None:
            return canon.segment_path_id.astype(np.int32, copy=False)
        if canon.path_id is not None and canon.lines.size:
            return canon.path_id[canon.lines[:, 0]].astype(np.int32, copy=False)
        return None

    @staticmethod
    def _path_weight(canon: CanonicalStepData, path_id: int) -> float:
        """Return linear received-power weight for a path ID."""
        if path_id < 0 or canon.path_losses is None or path_id >= len(canon.path_losses):
            return 1.0
        loss = float(canon.path_losses[path_id])
        if not np.isfinite(loss):
            return 0.0
        return float(10.0 ** (-loss / 10.0))

    def _select_top_paths(
        self,
        canon: CanonicalStepData,
        segment_mask: np.ndarray | None,
        max_paths: int,
    ) -> tuple[RFXRayTopPath, ...]:
        """Select the strongest visible material-bearing paths."""
        if max_paths <= 0:
            return ()
        visible = self._visible_segment_mask(canon, segment_mask)
        segment_path_ids = self._segment_path_ids(canon)
        if segment_path_ids is None or canon.segment_material_ids is None or not np.any(visible):
            return ()

        path_materials: dict[int, set[str]] = {}
        for segment_index in np.flatnonzero(visible):
            material_id = int(canon.segment_material_ids[segment_index])
            if material_id == 0:
                continue
            path_id = int(segment_path_ids[segment_index])
            if path_id < 0:
                continue
            path_materials.setdefault(path_id, set()).add(
                self._material_key_from_id(canon, material_id)
            )
        ranked = sorted(
            (
                RFXRayTopPath(
                    path_id=path_id,
                    weight=self._path_weight(canon, path_id),
                    path_loss_db=(
                        float(canon.path_losses[path_id])
                        if canon.path_losses is not None and path_id < len(canon.path_losses)
                        else None
                    ),
                    material_keys=tuple(sorted(materials)),
                )
                for path_id, materials in path_materials.items()
            ),
            key=lambda path: (-path.weight, path.path_id),
        )
        return tuple(ranked[:max_paths])

    def _build_top_path_lines(
        self,
        canon: CanonicalStepData,
        top_paths: tuple[RFXRayTopPath, ...],
        material_colors: Mapping[str, tuple[float, float, float, float]],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return disjoint line payload arrays for selected top paths."""
        segment_path_ids = self._segment_path_ids(canon)
        if segment_path_ids is None or canon.lines.size == 0:
            return None, None, None
        top_path_ids = {path.path_id for path in top_paths}
        segment_indices = np.flatnonzero(np.isin(segment_path_ids, list(top_path_ids)))
        if segment_indices.size == 0:
            return None, None, None

        points = np.empty((segment_indices.size * 2, 3), dtype=np.float32)
        lines = np.empty((segment_indices.size, 2), dtype=np.int32)
        colors = np.empty((segment_indices.size, 3), dtype=np.float32)
        for row, segment_index in enumerate(segment_indices):
            src, dst = canon.lines[segment_index]
            points[row * 2] = canon.points[src]
            points[row * 2 + 1] = canon.points[dst]
            lines[row] = (row * 2, row * 2 + 1)
            material_id = (
                int(canon.segment_material_ids[segment_index])
                if canon.segment_material_ids is not None
                else 0
            )
            key = self._material_key_from_id(canon, material_id)
            colors[row] = material_colors.get(key, _BOUNCE_FALLBACK_COLOR)[:3]
        points.setflags(write=False)
        lines.setflags(write=False)
        colors.setflags(write=False)
        return points, lines, colors
