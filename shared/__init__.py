"""Cross-application ORCHAV contracts and helpers.

The ``shared`` package is the neutral boundary used by the generator,
visualizer, validation CLI, and post-processing tools. It should contain data
contracts, schema loaders, readers, reusable sensing processing, and helpers
that more than one application needs.

Start with the concept packages directly:

- ``shared.frames`` for frame contracts, providers, HDF5, and protobuf.
- ``shared.scenarios`` for scenario models, YAML loading, and path policy.
- ``shared.coverage`` for coverage HDF5 schema constants.
- ``shared.geometry`` for scene geometry extraction and caches.
- ``shared.extensions`` for optional frame extension contracts.
- ``shared.sensing`` for the public numerical sensing API and saved-result processing.
- ``shared.sensing.products`` for native schema-v4 product identity and lineage.
- ``shared.animated_target`` for source-neutral geometry and clock binding.
- ``shared.sensing.measurement`` for post-calibration ingress and comparison.

Within ``shared.sensing``, start with ``postprocessor`` for saved-result workflows
and ``signal_processing`` for the numerical implementation. The ``detection``,
``fusion``, ``volume``, and ``plotting`` subpackages own their respective processing
and presentation helpers. Numerical functions remain available directly from
``shared.sensing``.

``generator.io`` adapts live generator state and orchestrates generator-owned
output and services. ``shared.frames`` owns the producer-neutral frame
contract, concrete frame-set codec, and transactional publication lifecycle.
Readers and validators also live here so any producer or consumer can use the
format without importing generator runtime code.

The animated-target and measurement workflow modules publish contract evidence
without making their JSON resources another native frame-storage format.
"""

__all__: list[str] = []
