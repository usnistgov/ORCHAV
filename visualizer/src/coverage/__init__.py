"""Coverage-map analysis and mesh-cache helpers for the visualizer.

``analysis`` owns renderer-neutral metric labels, threshold summaries,
threshold masks, and isoline geometry. ``cache`` owns mesh-array reuse and
NaN-aware smoothing for coverage surfaces built by services and the frame
pipeline.
"""
