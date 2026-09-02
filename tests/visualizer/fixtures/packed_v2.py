"""Small packed-HDF5-v2 frame sets for authoring boundary tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from generator.io.storage.hdf5_frame_output import HDF5FrameOutputStrategy
from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
)
from shared.frames.manifest import (
    FrameChunkManifest,
    FrameSetManifest,
    load_frame_manifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from tests.visualizer.fixtures.semantic_mpc import (
    FrameVariant,
    build_standard_mpc_frame,
)


def write_identity_only_frame_set(
    frames_dir: Path,
    *,
    frame_ids: Sequence[int] = (0,),
    attribute_overrides: Mapping[str, Any | None] | None = None,
) -> FrameSetManifest:
    """Write a finalized manifest and chunk-identity shell.

    This intentionally is not a loadable scientific frame. Tests of visual
    payloads use the real packed writer; this fixture isolates the smaller
    manifest/chunk-identity contract and is intentionally rejected by the
    structural Builder promotion gate. ``None`` in ``attribute_overrides``
    omits that attribute.
    """

    ids = tuple(int(frame_id) for frame_id in frame_ids)
    if not ids:
        raise ValueError("frame_ids must not be empty")

    frames_dir.mkdir(parents=True, exist_ok=True)
    topology_id = "test-topology"
    sensing_layout_id = "test-sensing-layout"
    chunk_path = frames_dir / f"mpc_frames_{ids[0]:05d}-{ids[-1]:05d}.h5"
    attributes: dict[str, Any | None] = {
        "file_kind": "mpc_frames",
        "schema_version": MPC_HDF5_SCHEMA_VERSION,
        "storage_layout": MPC_HDF5_LAYOUT,
        "packed_frame_version": PACKED_MPC_FRAME_VERSION,
        "generation_id": "test-generation",
        "publication_state": "complete",
        "num_frames": len(ids),
        "start_frame": ids[0],
        "end_frame": ids[-1],
        "topology_id": topology_id,
        "sensing_layout_id": sensing_layout_id,
    }
    attributes.update(attribute_overrides or {})

    with h5py.File(chunk_path, "w") as handle:
        for name, value in attributes.items():
            if value is not None:
                handle.attrs[name] = value
        frames = handle.create_group("frames")
        frames.create_dataset("id", data=np.asarray(ids, dtype=np.int64))

    chunk = FrameChunkManifest(
        file=chunk_path.name,
        frame_ids=ids,
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=0,
        topology_id=topology_id,
        sensing_layout_id=sensing_layout_id,
    )
    manifest = manifest_from_chunks(
        generation_id="test-generation",
        frame_set_id="test-frame-set",
        chunks=(chunk,),
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={"fixture": "identity-only"},
        created_utc="2026-07-30T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return manifest


def write_real_frame_set(
    frames_dir: Path,
    *,
    variants: Sequence[FrameVariant] = ("baseline",),
    chunk_size: int = 1,
) -> FrameSetManifest:
    """Write a small, fully loadable packed-v2 frame set with the real writer."""

    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    strategy = HDF5FrameOutputStrategy(
        SimpleNamespace(
            output_mode="file",
            quality="custom",
            get_quality_profile=lambda: {"max_depth": 2},
        ),
        SimpleNamespace(
            root=frames_dir.parent,
            project_root=frames_dir.parent,
            frames_directory=frames_dir.name,
            frames_dir=frames_dir,
            chunk_size=chunk_size,
            compression=None,
            raytracing={},
            scene={"id": "test-scene", "source": "test"},
        ),
    )
    for frame_idx, variant in enumerate(variants):
        strategy.save_standard_frame(build_standard_mpc_frame(variant, frame_idx=frame_idx))
    strategy.finalize()
    return load_frame_manifest(frames_dir)


__all__ = ["write_identity_only_frame_set", "write_real_frame_set"]
