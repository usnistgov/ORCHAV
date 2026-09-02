#!/usr/bin/env python3
"""Generator adapter for transactional HDF5 v2 frame-set publication.

The shared :class:`~shared.frames.frame_set_writer.FrameSetWriter` owns the
storage lifecycle. This adapter keeps generator-specific responsibilities:
normalizing raw ray-tracing results, capturing generator provenance and
material properties, and selecting the path-filter diagnostic inputs.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from shared.frames.frame_set_writer import (
    MAX_FRAMES_PER_CHUNK,
    MAX_UNCOMPRESSED_BYTES_PER_CHUNK,
    FrameSetWriter,
)
from shared.frames.manifest import FrameSetManifest
from shared.frames.types import StandardMPCFrame
from shared.logging import get_logger
from shared.scenarios.frame_paths import DEFAULT_FRAMES_DIRECTORY
from shared.scenarios.parsers import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from shared.source_identity import loaded_source_identity

from ..frames.conversion import standard_mpc_frame_from_raw

logger = get_logger(__name__)


class HDF5FrameOutputStrategy:
    """Adapt raw generator results to the shared HDF5-v2 frame-set writer."""

    def __init__(
        self,
        simulation_config,
        scenario_configuration=None,
        *,
        frame_set_writer: FrameSetWriter | None = None,
    ):
        if scenario_configuration is None:
            raise ValueError("scenario_configuration is required for HDF5 frame output")

        self.simulation_config = simulation_config
        self.scenario_configuration = scenario_configuration
        self.material_properties: dict[str, Any] | None = None
        self._published_manifest: FrameSetManifest | None = None

        configured_chunk_size = int(
            getattr(scenario_configuration, "chunk_size", DEFAULT_CHUNK_SIZE)
        )
        compression = getattr(
            scenario_configuration,
            "compression",
            DEFAULT_COMPRESSION,
        )
        if frame_set_writer is None:
            scenario_root = self._required_configuration_path(
                scenario_configuration,
                "root",
            )
            self.require_canonical_scenario_frames(
                scenario_configuration,
                scenario_root=scenario_root,
            )
            self._frame_set_writer = FrameSetWriter.for_scenario(
                scenario_root,
                chunk_size=configured_chunk_size,
                compression=compression,
            )
            # Generator runs apply stricter chunk bounds than the reusable
            # writer defaults. Set them before any frame is prepared.
            self._frame_set_writer.max_frames_per_chunk = MAX_FRAMES_PER_CHUNK
            self._frame_set_writer.max_uncompressed_bytes = MAX_UNCOMPRESSED_BYTES_PER_CHUNK
            self._frame_set_writer.chunk_size = min(
                configured_chunk_size,
                MAX_FRAMES_PER_CHUNK,
            )
        else:
            self._frame_set_writer = frame_set_writer

        self.path_filter_config = self._resolve_path_filter_config()
        self.diag_output_dir = self._frame_set_writer.staging_directory
        self.diag_bandwidth_hz = self._resolve_bandwidth_hz()

        logger.info(
            "Initialized packed HDF5 v2 output "
            "(configured frame cap: %d, effective frame cap: %d, "
            "byte cap: %d, compression: %s)",
            self.configured_chunk_size,
            self.chunk_size,
            self.max_uncompressed_bytes,
            self.compression,
        )
        if self.path_filter_config:
            logger.info("  Path filtering enabled: %s", self.path_filter_config)

    @staticmethod
    def _required_configuration_path(configuration: Any, name: str) -> Path:
        value = getattr(configuration, name, None)
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(
                f"HDF5 frame output requires scenario_configuration.{name} " "as a concrete path"
            )
        return Path(value)

    @classmethod
    def require_canonical_scenario_frames(
        cls,
        configuration: Any,
        *,
        scenario_root: Path,
    ) -> None:
        """Reject noncanonical configured frame locations for generator publication.

        ``data.files.directory`` selects imported or derived frame sets for
        visualizers and analysis tools. Normal generation has a narrower
        contract: it may replace only the fixed ``<scenario>/frames`` child. A
        producer that creates a separate result must inject a writer built with
        :meth:`FrameSetWriter.create_new`.
        """
        declared = getattr(configuration, "frames_directory", None)
        configured = cls._required_configuration_path(configuration, "frames_dir")
        expected = scenario_root / DEFAULT_FRAMES_DIRECTORY

        configured_identity = os.path.normcase(os.path.abspath(os.fspath(configured)))
        expected_identity = os.path.normcase(os.path.abspath(os.fspath(expected)))
        if declared != DEFAULT_FRAMES_DIRECTORY or configured_identity != expected_identity:
            raise ValueError(
                "Generator frame output is fixed at <scenario>/frames; "
                "data.files.directory is a read-only selection and cannot choose "
                f"a generation destination (declared={declared!r}, resolved={configured})"
            )

    @property
    def generated_chunks(self):
        """Return finalized chunk metadata retained for generator reporting."""
        return self._frame_set_writer.generated_chunks

    @property
    def chunk_counter(self) -> int:
        return int(self._frame_set_writer.chunk_counter)

    @property
    def configured_chunk_size(self) -> int:
        return int(self._frame_set_writer.configured_chunk_size)

    @property
    def chunk_size(self) -> int:
        return int(self._frame_set_writer.chunk_size)

    @property
    def max_uncompressed_bytes(self) -> int:
        return int(self._frame_set_writer.max_uncompressed_bytes)

    @property
    def compression(self) -> str | None:
        return cast(str | None, self._frame_set_writer.compression)

    @property
    def generation_id(self) -> str:
        """Return the run identity used by frames and related outputs."""
        return str(self._frame_set_writer.generation_id)

    @property
    def published_manifest(self) -> FrameSetManifest | None:
        """Return the manifest committed by :meth:`finalize`, if any."""
        return self._published_manifest

    def begin(self) -> None:
        """Reserve the managed destination before expensive generation work."""
        self._frame_set_writer.begin()

    def _resolve_path_filter_config(self) -> Any:
        rt_config = getattr(self.scenario_configuration, "raytracing", None)
        if not rt_config:
            return None
        path_filter = (
            rt_config.get("path_filter")
            if isinstance(rt_config, Mapping)
            else getattr(rt_config, "path_filter", None)
        )
        if not path_filter:
            return None
        if hasattr(path_filter, "model_dump"):
            return path_filter.model_dump()
        if isinstance(path_filter, Mapping):
            return dict(path_filter)
        return {
            "relative_threshold_db": getattr(path_filter, "relative_threshold_db", None),
            "max_path_loss_db": getattr(path_filter, "max_path_loss_db", None),
            "max_paths_per_pair": getattr(path_filter, "max_paths_per_pair", None),
            "log_filtering_stats": getattr(path_filter, "log_filtering_stats", True),
            "generate_diagnostic": getattr(path_filter, "generate_diagnostic", False),
        }

    def _resolve_bandwidth_hz(self) -> float | None:
        rt_config = getattr(self.scenario_configuration, "raytracing", None)
        if not rt_config:
            return None
        bandwidth = (
            rt_config.get("bandwidth_hz")
            if isinstance(rt_config, Mapping)
            else getattr(rt_config, "bandwidth_hz", None)
        )
        if bandwidth is None:
            return None
        try:
            return float(bandwidth)
        except (TypeError, ValueError):
            return None

    def save_standard_frame(self, frame: StandardMPCFrame) -> None:
        """Append an already normalized frame without generator conversion."""
        self._frame_set_writer.append(frame)

    def save_frame_data(self, frame_idx: int, data: dict[str, Any]) -> None:
        """Normalize and append one raw generator frame without retaining it."""
        try:
            # Normalization may emit a path-filter diagnostic. Acquire the lock
            # and create the owned staging directory before invoking it.
            self._frame_set_writer.begin()
            if self.material_properties is None:
                material_properties = data.get("material_properties")
                if isinstance(material_properties, dict):
                    self.material_properties = material_properties

            frame = standard_mpc_frame_from_raw(
                data,
                frame_idx,
                source_provider="generator_file",
                simulation_config=self.simulation_config,
                path_filter_config=self.path_filter_config,
                output_dir=self.diag_output_dir,
                bandwidth_hz=self.diag_bandwidth_hz,
            )
            self._frame_set_writer.append(frame)
        except BaseException:
            self._frame_set_writer.abort()
            raise

    def abort(self) -> None:
        """Discard the active generator run without changing live frames."""
        self._frame_set_writer.abort()

    @staticmethod
    def _resolve_git_sha() -> str | None:
        """Return the revision of the tree that supplied the loaded generator."""
        return cast(str | None, loaded_source_identity("generator").git_sha)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): HDF5FrameOutputStrategy._json_safe(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [HDF5FrameOutputStrategy._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return HDF5FrameOutputStrategy._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return HDF5FrameOutputStrategy._json_safe(value.item())
        return str(value)

    def _scene_provenance(self) -> tuple[Any, Any]:
        scene = getattr(self.scenario_configuration, "scene", None)
        if isinstance(scene, Mapping):
            return scene.get("id"), scene.get("source")
        return (
            getattr(self.scenario_configuration, "scene_id", None),
            getattr(self.scenario_configuration, "scene_source", None),
        )

    def _quality_provenance(self) -> tuple[Any, dict[str, Any]]:
        base_quality = None
        if hasattr(self.simulation_config, "get_quality_profile"):
            try:
                base_quality = self.simulation_config.get_quality_profile()
            except (AttributeError, TypeError, ValueError):
                base_quality = None
        effective_quality = dict(base_quality) if isinstance(base_quality, Mapping) else {}

        rt_config = getattr(self.scenario_configuration, "raytracing", None)
        if rt_config:
            quality = (
                rt_config.get("quality", {})
                if isinstance(rt_config, Mapping)
                else getattr(rt_config, "quality", {})
            )
            if isinstance(quality, Mapping):
                custom = quality.get("custom", {}) or {}
                if isinstance(custom, Mapping):
                    allowed = {
                        "max_depth",
                        "samples_per_src",
                        "max_num_paths_per_src",
                        "los",
                        "specular_reflection",
                        "diffuse_reflection",
                        "refraction",
                        "diffraction",
                        "seed",
                        "synthetic_array",
                    }
                    effective_quality.update(
                        {key: value for key, value in custom.items() if key in allowed}
                    )
        return base_quality, effective_quality

    def _provenance(self) -> dict[str, Any]:
        scene_id, scene_source = self._scene_provenance()
        base_quality, effective_quality = self._quality_provenance()
        return cast(
            dict[str, Any],
            self._json_safe(
                {
                    "scene_id": scene_id,
                    "scene_source": scene_source,
                    "output_mode": getattr(self.simulation_config, "output_mode", None),
                    "quality": getattr(self.simulation_config, "quality", None),
                    "quality_profile": effective_quality,
                    "quality_profile_base": base_quality,
                    "path_filter_config": self.path_filter_config,
                    "material_properties": self.material_properties,
                    "git_sha": self._resolve_git_sha(),
                    "python_version": sys.version.split()[0],
                    "platform": platform.platform(),
                }
            ),
        )

    def finalize(self) -> str:
        """Publish the run and retain the generator's existing result string."""
        try:
            provenance = self._provenance()
            manifest = self._frame_set_writer.finalize(provenance=provenance)
            self._published_manifest = manifest
        except BaseException:
            self._frame_set_writer.abort()
            raise
        total_files = 0 if manifest is None else len(manifest.chunks)
        total_frames = 0 if manifest is None else len(manifest.frame_ids)
        result = (
            f"HDF5 v2 Output (frame_cap={self.chunk_size}, "
            f"byte_cap={self.max_uncompressed_bytes}): "
            f"Saved {total_frames} frames across {total_files} file(s)."
        )
        if manifest is None:
            logger.info("%s Existing frame output was left unchanged.", result)
        else:
            logger.info(result)
        return result


__all__ = [
    "HDF5FrameOutputStrategy",
    "MAX_FRAMES_PER_CHUNK",
    "MAX_UNCOMPRESSED_BYTES_PER_CHUNK",
]
