"""Execution engine behind the :mod:`generator` facade.

Start with ``pipeline`` for orchestration. ``configuration`` normalizes run
settings, ``scenario_actors`` prepares canonical actor poses, and ``services``
adapts those values to live Sionna objects. ``runtime`` carries the prepared
objects consumed by ``propagation``, streaming, and on-demand requests.

``scenario_entities`` contains Sionna-specific array and scene-object helpers.
Scenario authoring models and validation belong to :mod:`shared.scenarios`.
"""
