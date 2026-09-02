"""Shared frame contracts, readers, providers, and transport codecs.

``StandardMPCFrame`` is the in-memory contract exchanged between generator
output, HDF5 readers, gRPC serializers, and visualizer consumers. This package
owns the parts that all of those users must agree on: type definitions,
validation, file readers/providers, cache helpers, and protobuf conversion.

The generator owns Sionna frame production and adapts raw results into this
contract. This package owns the concrete HDF5 codec and ``FrameSetWriter``
publication lifecycle alongside the reader. Any producer that creates a
``StandardMPCFrame`` can publish the same schema without importing the
generator runtime.
"""

from .adapters import project_standard_mpc_frame
from .contracts import (
    MPC_FRAME_MANIFEST_VERSION,
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_ORDER,
    PATH_METRIC_VALIDITY_BITS,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
)
from .directory_ownership import (
    FrameDirectoryChangedError,
    FrameDirectoryLockError,
    FrameDirectorySafetyError,
    FrameDirectorySnapshot,
    ManagedFrameDirectoryLock,
    capture_frame_directory,
    revalidate_frame_directory,
)
from .frame_set_writer import FrameSetWriter
from .normalization import standard_mpc_frame_from_pair_data
from .packed import FrameProjection, ProjectedMPCFrame
from .types import FRAME_FORMAT_VERSION, PATH_METRIC_ARRAY_FIELDS, StandardMPCFrame

__all__ = [
    "FRAME_FORMAT_VERSION",
    "MPC_FRAME_MANIFEST_VERSION",
    "MPC_HDF5_LAYOUT",
    "MPC_HDF5_SCHEMA_VERSION",
    "PACKED_MPC_FRAME_VERSION",
    "PATH_METRIC_ORDER",
    "PATH_METRIC_ARRAY_FIELDS",
    "PATH_METRIC_VALIDITY_BITS",
    "FrameComponent",
    "FrameDirectoryChangedError",
    "FrameDirectoryLockError",
    "FrameDirectorySafetyError",
    "FrameDirectorySnapshot",
    "FrameProjection",
    "FrameReadRequest",
    "FrameSetWriter",
    "ProjectedMPCFrame",
    "PathMetric",
    "StandardMPCFrame",
    "ManagedFrameDirectoryLock",
    "capture_frame_directory",
    "revalidate_frame_directory",
    "project_standard_mpc_frame",
    "standard_mpc_frame_from_pair_data",
]
