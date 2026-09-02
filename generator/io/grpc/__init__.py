"""gRPC transport adapters for generator frame delivery.

``live_server`` computes or receives raw generator frames and normalizes them
before protobuf serialization. ``file_server`` serves pre-generated HDF5
``StandardMPCFrame`` chunks over the remote-HDF5 service. ``live_scene_update``
owns bounded temporary XML staging, ``client`` is a small diagnostic client,
and ``cache`` stores raw live-frame dictionaries before normalization.
"""

__all__: list[str] = []
