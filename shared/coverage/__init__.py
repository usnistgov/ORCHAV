"""Shared coverage-map file contract and selective HDF5 reader.

``schema`` defines the compact HDF5 layout emitted by generator coverage output
and consumed by inspection/visualization code. Start there for schema version,
metric derivation, TX selector, and dB/unit conversion rules.

``hdf5`` maps those logical metric requests to bounded HDF5 hyperslabs.  It can
select height and transmitter axes without first materializing the full
canonical coverage tensor.
"""

from .hdf5 import CoverageHDF5Reader

__all__ = ["CoverageHDF5Reader"]
