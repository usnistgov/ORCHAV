"""Cross-application ORCHAV contracts and helpers.

The ``shared`` package is the neutral boundary used by the generator,
visualizer, validation CLI, and post-processing tools. It should contain data
contracts, schema loaders, readers, and small pure helpers that more than one
application needs.

Start with the concept packages directly:

- ``shared.frames`` for frame contracts, providers, HDF5, and protobuf.
- ``shared.scenarios`` for scenario models, YAML loading, and path policy.
- ``shared.coverage`` for coverage HDF5 schema constants.
- ``shared.geometry`` for scene geometry extraction and caches.
- ``shared.extensions`` for optional frame extension contracts.

``generator.io`` adapts live generator state and orchestrates generator-owned
output and services. ``shared.frames`` owns the producer-neutral frame
contract, concrete frame-set codec, and transactional publication lifecycle.
Readers and validators also live here so any producer or consumer can use the
format without importing generator runtime code.
"""

__all__: list[str] = []
