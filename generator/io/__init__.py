"""Generator I/O boundary for frame production, storage, providers, and transports.

This package is the producer side of ORCHAV frame I/O. It converts live
generator state into shared contracts, writes generated data to disk, and serves
generated frames over gRPC. The reusable consumer contract lives in
``shared.frames`` so readers, validators, and the visualizer do not import
generator runtime code.

The package keeps four frame vocabularies separate:

``raw generator frame``
    A dictionary produced by the live ray-tracing path. It still contains
    Sionna RT path objects, TX/RX scene objects, target managers, and optional
    streaming snapshots.
``StandardMPCFrame``
    The validated in-memory contract shared by generator, visualizer, HDF5
    readers, and protobuf serializers.
``packed HDF5 v2 frame set``
    The durable on-disk representation of one or more ``StandardMPCFrame``
    objects. Frame, path, and bounce axes use explicit offsets instead of
    padding every frame to the largest path or bounce count in a chunk.
``protobuf frame``
    The transport representation sent over gRPC after a frame has already been
    normalized to ``StandardMPCFrame``.

Start in ``frames.conversion`` for raw-to-canonical normalization. File mode
continues through ``storage.hdf5_frame_output`` and
``shared.frames.packed_hdf5_writer``; streaming and remote-HDF5 mode continue
through ``grpc.live_server`` and ``grpc.file_server``.
"""

__all__: list[str] = []
