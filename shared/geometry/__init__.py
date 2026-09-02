"""Shared scene-geometry extraction and bounds helpers.

``transforms`` defines the exact mesh-transform subset accepted by lightweight
geometry consumers. ``scene.load_scene_geometry`` reads validated Mitsuba XML
mesh references into neutral metadata, while ``cache`` stores those entries
and derives padded XY/XYZ bounds for generator summaries, coverage defaults,
and visualization tools.
"""
