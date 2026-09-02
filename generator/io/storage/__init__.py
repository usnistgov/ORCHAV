"""Persistent generator output backends.

``hdf5_frame_output`` is the generator adapter: it accepts raw ray-tracing
frames, normalizes them, supplies generator provenance and diagnostics, and
delegates chunking and transactional publication to
``shared.frames.FrameSetWriter``. The shared packed-HDF5 codec only accepts
validated ``StandardMPCFrame`` objects and serializes the schema used by the
shared file provider. Other complete-frame publishers can therefore use the
same lifecycle and schema without importing the generator runtime.
``coverage_writer`` is a separate HDF5 schema for coverage maps, not MPC frames.
"""

__all__: list[str] = []
