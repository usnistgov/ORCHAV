"""Renderer-neutral reconciliation used by both concrete backends.

``surface_reconciler`` owns retry-aware coverage and beamforming state;
backend modules supply only the operations required to realize each desired
surface snapshot.
"""
