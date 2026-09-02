#!/usr/bin/env python3
"""Serve pre-generated ``StandardMPCFrame`` values via gRPC.

The server opens one manifest-driven HDF5 frame set, captures its startup file
identity, and serves complete frames as protobuf messages. It stops serving if
the manifest or an advertised chunk changes; clients therefore never observe
a mixture of two frame-set generations. No GPU ray tracing is required.

Usage:
    python -m generator.io.grpc.file_server --frames-dir /path/to/scenario --port 50052
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import OrderedDict
from concurrent import futures
from pathlib import Path
from typing import Any, Optional

import grpc

from shared.frames.manifest import FRAMES_MANIFEST_FILENAME, FrameSetManifest, load_frame_manifest
from shared.frames.protobuf import standard_mpc_frame_to_proto
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame
from shared.grpc_transport import (
    DEFAULT_GRPC_BIND_HOST,
    DEFAULT_GRPC_SHUTDOWN_GRACE_S,
    GRPC_MAX_MESSAGE_BYTES,
    GRPC_MESSAGE_OPTIONS,
    bind_grpc_server,
    format_grpc_endpoint,
    is_loopback_grpc_host,
)
from shared.logging import configure_logging, get_logger

# Import the gRPC generated code
try:
    from shared.protos import visualizer_pb2 as _visualizer_pb2
    from shared.protos import visualizer_pb2_grpc as _visualizer_pb2_grpc

    visualizer_pb2: Any = _visualizer_pb2
    visualizer_pb2_grpc: Any = _visualizer_pb2_grpc
except ImportError as e:
    raise ImportError(
        f"Could not import gRPC generated code: {e}. "
        "Run 'python scripts/protobuf/generate_protobuf.py' first."
    )

logger = get_logger(__name__)


class FrameFileServicer(visualizer_pb2_grpc.FrameFileServiceServicer):
    """gRPC servicer that serves pre-generated HDF5 frames.

    Unlike ``GeneratorService`` in ``live_server``, this class never sees raw
    Sionna path objects. It loads complete canonical ``StandardMPCFrame`` objects
    from ``Hdf5Provider``, converts them to protobuf, and protects clients from
    partially changed frame directories with a startup snapshot signature.
    """

    def __init__(
        self,
        frames_dir: Path,
        frames_subdir: str = "frames",
        proto_cache_size: int = 64,
    ):
        """Initialize the frame file servicer.

        Args:
            frames_dir: Path to the scenario directory containing frames
            frames_subdir: Subdirectory containing HDF5 frame files
            proto_cache_size: Number of serialized protobuf frames to cache
        """
        self.frames_dir = Path(frames_dir)
        self.frames_subdir = frames_subdir
        self.start_time = time.time()
        self.proto_cache_size = max(0, int(proto_cache_size))
        self._manifest: FrameSetManifest
        self._snapshot_signatures: dict[str, tuple[int, int, int, int]] = {}
        self._snapshot_error = ""

        provider: Optional[Hdf5Provider] = None
        try:
            provider = Hdf5Provider(self.frames_dir, frames_subdir=frames_subdir)
            self.provider = provider
            (
                self._manifest,
                self._snapshot_signatures,
                self._frame_indices,
            ) = self._capture_stable_startup_state()
            self.frame_set_id = self._manifest.frame_set_id
            self._metadata_cache: Optional[dict] = None
            self._proto_cache: OrderedDict[int, Any] = OrderedDict()
            self._proto_cache_sizes: dict[int, int] = {}
            self._proto_cache_lock = threading.Lock()
            self._proto_cache_current_bytes = 0
            self._proto_cache_peak_bytes = 0
            self._proto_cache_hits = 0
            self._proto_cache_misses = 0
            self._proto_cache_evictions = 0
            logger.info(
                "FrameFileServer initialized with %d frames from %s (frame_set_id=%s)",
                len(self._frame_indices),
                self.frames_dir,
                self.frame_set_id[:12],
            )
        except Exception as e:
            if provider is not None:
                provider.close()
            logger.error("Failed to initialize Hdf5Provider: %s", e)
            raise

    @property
    def _frames_path(self) -> Path:
        return self.frames_dir / self.frames_subdir

    def _snapshot_paths(self, manifest: FrameSetManifest) -> list[Path]:
        """Return files that define the served HDF5 frame set."""
        return [
            self._frames_path / FRAMES_MANIFEST_FILENAME,
            *(self._frames_path / chunk.file for chunk in manifest.chunks),
        ]

    def _capture_snapshot_signatures(
        self,
        manifest: FrameSetManifest | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Capture filesystem identity signatures for frame-set files."""
        active_manifest = manifest or self._manifest
        signatures: dict[str, tuple[int, int, int, int]] = {}
        for path in self._snapshot_paths(active_manifest):
            stat = path.stat()
            rel = str(path.relative_to(self._frames_path))
            signatures[rel] = (
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        return signatures

    def _capture_stable_startup_state(
        self,
    ) -> tuple[FrameSetManifest, dict[str, tuple[int, int, int, int]], list[int]]:
        provider_frames = list(self.provider.list_frames())
        provider_frame_set_id = str(self.provider.info.frame_set_id)
        first_manifest = load_frame_manifest(self._frames_path)
        first_signatures = self._capture_snapshot_signatures(first_manifest)
        second_manifest = load_frame_manifest(self._frames_path)
        second_signatures = self._capture_snapshot_signatures(second_manifest)

        if first_manifest != second_manifest or first_signatures != second_signatures:
            raise RuntimeError("Frames directory changed while the remote server was starting")
        if provider_frame_set_id != first_manifest.frame_set_id:
            raise RuntimeError(
                "HDF5 provider opened a different frame set than the manifest on disk"
            )
        if tuple(provider_frames) != tuple(first_manifest.frame_ids):
            raise RuntimeError("HDF5 provider frame index does not match the manifest on disk")

        return first_manifest, first_signatures, provider_frames

    def _validate_snapshot(self) -> bool:
        """Reject service responses if HDF5 files changed after startup."""
        try:
            current = self._capture_snapshot_signatures()
        except OSError as e:
            self._snapshot_error = f"could not read frame-set file state: {e}"
            logger.warning(
                "Remote HDF5 snapshot changed after server startup: %s",
                self._snapshot_error,
            )
            return False
        if current == self._snapshot_signatures:
            self._snapshot_error = ""
            return True

        missing = sorted(set(self._snapshot_signatures) - set(current))
        added = sorted(set(current) - set(self._snapshot_signatures))
        changed = sorted(
            rel
            for rel in set(current).intersection(self._snapshot_signatures)
            if current[rel] != self._snapshot_signatures[rel]
        )
        details = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if added:
            details.append(f"added={added[:3]}")
        if changed:
            details.append(f"changed={changed[:3]}")
        self._snapshot_error = "; ".join(details) or "frame files changed"
        logger.warning(
            "Remote HDF5 snapshot changed after server startup: %s", self._snapshot_error
        )
        return False

    def _get_cached_proto(self, frame_idx: int) -> Optional[Any]:
        with self._proto_cache_lock:
            if self.proto_cache_size <= 0:
                self._proto_cache_misses += 1
                return None
            frame_pb = self._proto_cache.get(frame_idx)
            if frame_pb is None:
                self._proto_cache_misses += 1
                return None
            self._proto_cache.move_to_end(frame_idx)
            self._proto_cache_hits += 1
            return frame_pb

    def _store_cached_proto(self, frame_idx: int, frame_pb: Any) -> None:
        if self.proto_cache_size <= 0:
            return
        retained_bytes = int(frame_pb.ByteSize())
        with self._proto_cache_lock:
            previous_bytes = self._proto_cache_sizes.pop(frame_idx, 0)
            self._proto_cache_current_bytes -= previous_bytes

            self._proto_cache[frame_idx] = frame_pb
            self._proto_cache_sizes[frame_idx] = retained_bytes
            self._proto_cache.move_to_end(frame_idx)
            self._proto_cache_current_bytes += retained_bytes
            self._proto_cache_peak_bytes = max(
                self._proto_cache_peak_bytes,
                self._proto_cache_current_bytes,
            )

            while len(self._proto_cache) > self.proto_cache_size:
                evicted_idx, _evicted_proto = self._proto_cache.popitem(last=False)
                self._proto_cache_current_bytes -= self._proto_cache_sizes.pop(
                    evicted_idx,
                    0,
                )
                self._proto_cache_evictions += 1

    @property
    def proto_cache_stats(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of local protobuf-cache telemetry."""
        with self._proto_cache_lock:
            total = self._proto_cache_hits + self._proto_cache_misses
            return {
                "frame_set_id": self.frame_set_id,
                "cached_entries": len(self._proto_cache),
                "max_entries": self.proto_cache_size,
                "current_bytes": self._proto_cache_current_bytes,
                "peak_bytes": self._proto_cache_peak_bytes,
                "hits": self._proto_cache_hits,
                "misses": self._proto_cache_misses,
                "total": total,
                "hit_ratio": self._proto_cache_hits / total if total > 0 else 0.0,
                "evictions": self._proto_cache_evictions,
            }

    def _get_metadata(self) -> dict:
        """Get cached metadata from the first frame."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        if not self._frame_indices:
            self._metadata_cache = {
                "num_tx": 0,
                "num_rx": 0,
                "num_targets": 0,
            }
            return self._metadata_cache

        first_frame = self.provider.load_frame(self._frame_indices[0])
        metadata = {
            "num_tx": first_frame.num_tx,
            "num_rx": first_frame.num_rx,
            "num_targets": first_frame.num_targets,
        }
        if not self._validate_snapshot():
            raise RuntimeError(f"Snapshot changed while loading metadata: {self._snapshot_error}")
        self._metadata_cache = metadata

        return self._metadata_cache

    def _convert_frame_to_protobuf(self, frame: StandardMPCFrame, frame_idx: int) -> Any:
        """Encode one loaded canonical frame for remote playback."""
        try:
            if frame.frame_index != frame_idx:
                raise ValueError(
                    f"Loaded frame index {frame.frame_index} does not match "
                    f"requested frame {frame_idx}"
                )
            return standard_mpc_frame_to_proto(frame)
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error converting frame %d to protobuf: %s", frame_idx, e)
            raise

    def _bounded_frame_response(
        self,
        *,
        frame_idx: int,
        frame_pb: Any,
        load_time_ms: float,
    ) -> Any:
        """Build a response only when its complete wire message fits policy."""
        response = visualizer_pb2.PreGeneratedFrameResponse(
            success=True,
            message="Frame retrieved successfully",
            frame_idx=frame_idx,
            frame_data=frame_pb,
            load_time_ms=load_time_ms,
            frame_set_id=self.frame_set_id,
        )
        encoded_size = response.ByteSize()
        if encoded_size <= GRPC_MAX_MESSAGE_BYTES:
            return response

        logger.error(
            "Encoded frame %d requires %d bytes, exceeding the %d-byte transport limit",
            frame_idx,
            encoded_size,
            GRPC_MAX_MESSAGE_BYTES,
        )
        return visualizer_pb2.PreGeneratedFrameResponse(
            success=False,
            message=(
                f"Encoded frame {frame_idx} requires {encoded_size} bytes, exceeding "
                f"the {GRPC_MAX_MESSAGE_BYTES}-byte transport limit"
            ),
            frame_idx=frame_idx,
            frame_set_id=self.frame_set_id,
        )

    def GetFileServerMetadata(self, request, context):
        """Return metadata about the file server and available frames."""
        try:
            snapshot_valid = self._validate_snapshot()
            metadata = self._get_metadata() if snapshot_valid else {}
            uptime = time.time() - self.start_time
            first_frame = min(self._frame_indices) if self._frame_indices else -1
            last_frame = max(self._frame_indices) if self._frame_indices else -1
            provenance = self._manifest.provenance
            quality_profile = provenance.get("quality_profile") or {}
            quality_json = json.dumps(quality_profile, sort_keys=True) if quality_profile else ""
            material_properties = provenance.get("material_properties") or {}
            material_properties_json = (
                json.dumps(material_properties, sort_keys=True) if material_properties else ""
            )

            return visualizer_pb2.FileServerMetadataResponse(
                success=True,
                message=(
                    "Metadata retrieved successfully"
                    if snapshot_valid
                    else f"Snapshot changed after startup: {self._snapshot_error}"
                ),
                total_frames=len(self._frame_indices),
                num_tx=metadata.get("num_tx", 0),
                num_rx=metadata.get("num_rx", 0),
                num_targets=metadata.get("num_targets", 0),
                scene_name=self.frames_dir.name,
                source_directory=str(self.frames_dir),
                is_bulk_format=self.provider.is_bulk,
                server_uptime_seconds=uptime,
                frame_set_id=self.frame_set_id,
                manifest_schema_version=self._manifest.manifest_version,
                first_frame_idx=first_frame,
                last_frame_idx=last_frame,
                chunk_size=int(self._manifest.segmentation.get("effective_frame_limit") or 0),
                total_files=len(self._manifest.chunks),
                snapshot_valid=snapshot_valid,
                snapshot_error=self._snapshot_error,
                git_sha=str(provenance.get("git_sha") or ""),
                quality_profile_json=quality_json,
                material_properties_json=material_properties_json,
            )
        except Exception as e:  # noqa: BLE001 - gRPC handlers return structured errors.
            logger.error("Error getting metadata: %s", e)
            return visualizer_pb2.FileServerMetadataResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    def ListAvailableFrames(self, request, context):
        """Return the list of available frame indices."""
        try:
            if not self._validate_snapshot():
                return visualizer_pb2.ListFramesResponse(
                    success=False,
                    message=f"Snapshot changed after startup: {self._snapshot_error}",
                    frame_set_id=self.frame_set_id,
                )
            return visualizer_pb2.ListFramesResponse(
                success=True,
                message="Frames listed successfully",
                frame_indices=self._frame_indices,
                total_count=len(self._frame_indices),
                frame_set_id=self.frame_set_id,
            )
        except Exception as e:  # noqa: BLE001 - gRPC handlers return structured errors.
            logger.error("Error listing frames: %s", e)
            return visualizer_pb2.ListFramesResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    def GetPreGeneratedFrame(self, request, context):
        """Return a single pre-generated frame."""
        frame_idx = request.frame_idx
        start_time = time.time()

        try:
            if not self._validate_snapshot():
                return visualizer_pb2.PreGeneratedFrameResponse(
                    success=False,
                    message=f"Snapshot changed after startup: {self._snapshot_error}",
                    frame_idx=frame_idx,
                    frame_set_id=self.frame_set_id,
                )

            if not self.provider.has_frame(frame_idx):
                return visualizer_pb2.PreGeneratedFrameResponse(
                    success=False,
                    message=f"Frame {frame_idx} not found",
                    frame_idx=frame_idx,
                    frame_set_id=self.frame_set_id,
                )

            frame_pb = self._get_cached_proto(frame_idx)
            loaded_for_cache = frame_pb is None
            if frame_pb is None:
                frame = self.provider.load_frame(frame_idx)
                frame_pb = self._convert_frame_to_protobuf(frame, frame_idx)
                if not self._validate_snapshot():
                    return visualizer_pb2.PreGeneratedFrameResponse(
                        success=False,
                        message=(
                            "Snapshot changed while loading frame "
                            f"{frame_idx}: {self._snapshot_error}"
                        ),
                        frame_idx=frame_idx,
                        frame_set_id=self.frame_set_id,
                    )
            load_time_ms = (time.time() - start_time) * 1000

            response = self._bounded_frame_response(
                frame_idx=frame_idx,
                frame_pb=frame_pb,
                load_time_ms=load_time_ms,
            )
            if loaded_for_cache and response.success:
                self._store_cached_proto(frame_idx, frame_pb)
            return response

        except Exception as e:  # noqa: BLE001 - gRPC handlers return structured errors.
            logger.error("Error getting frame %d: %s", frame_idx, e)
            return visualizer_pb2.PreGeneratedFrameResponse(
                success=False,
                message=f"Error loading frame {frame_idx}: {str(e)}",
                frame_idx=frame_idx,
                frame_set_id=self.frame_set_id,
            )


def _create_server(
    frames_path: Path,
    *,
    bind_host: str,
    port: int,
    max_workers: int,
):
    """Construct and bind a frame server without starting its event loop."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=GRPC_MESSAGE_OPTIONS,
    )
    servicer = FrameFileServicer(frames_path)
    try:
        visualizer_pb2_grpc.add_FrameFileServiceServicer_to_server(servicer, server)
        _requested_endpoint, bound_port = bind_grpc_server(server, bind_host, port)
    except Exception:
        servicer.provider.close()
        raise
    endpoint = format_grpc_endpoint(bind_host, bound_port)
    return server, servicer, endpoint


def serve(
    frames_dir: str,
    port: int = 50052,
    max_workers: int = 4,
    *,
    bind_host: str = DEFAULT_GRPC_BIND_HOST,
) -> None:
    """Start the Frame File Server.

    Args:
        frames_dir: Path to the scenario directory containing frames
        port: Port to listen on
        max_workers: Maximum number of worker threads
        bind_host: Local interface to bind. Non-loopback hosts are for trusted
            networks and must be selected explicitly.
    """
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_path}")

    if not is_loopback_grpc_host(bind_host):
        logger.warning(
            "Frame File Server is binding to non-loopback host %s without "
            "transport authentication; use only on a trusted network",
            bind_host,
        )

    server, servicer, endpoint = _create_server(
        frames_path,
        bind_host=bind_host,
        port=port,
        max_workers=max_workers,
    )
    started = False

    try:
        server.start()
        started = True
        logger.info("Frame File Server started at %s", endpoint)
        logger.info("Serving %d frames from %s", len(servicer._frame_indices), frames_path)
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Frame File Server...")
    finally:
        if started:
            server.stop(grace=DEFAULT_GRPC_SHUTDOWN_GRACE_S).wait()
        servicer.provider.close()


def main():
    """Main entry point for the Frame File Server."""
    parser = argparse.ArgumentParser(description="Serve pre-generated HDF5 frames via gRPC")
    parser.add_argument(
        "--frames-dir",
        "-d",
        required=True,
        help="Path to scenario directory containing frames/",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=50052,
        help="Port to listen on (default: 50052)",
    )
    parser.add_argument(
        "--bind-host",
        default=DEFAULT_GRPC_BIND_HOST,
        help=(
            "Local bind host (default: 127.0.0.1). Select a non-loopback host "
            "only for an explicitly trusted network."
        ),
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help="Maximum worker threads (default: 4)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level="DEBUG" if args.debug else "INFO")

    serve(
        args.frames_dir,
        args.port,
        args.max_workers,
        bind_host=args.bind_host,
    )


if __name__ == "__main__":
    main()
