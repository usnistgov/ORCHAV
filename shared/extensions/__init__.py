"""Optional ``StandardMPCFrame`` extension contracts.

Extension modules define validation and normalization for optional frame
payloads that sit beside the core MPC path data. Keep shared extension schemas
here so HDF5 storage and frame consumers agree on the same payload shape.
Transport codecs may support a narrower payload and must reject unsupported
extensions explicitly.
"""
